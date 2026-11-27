# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import math
import re
from typing import Iterable

import torch

from .base import MmprojModel, ModelBase, gguf
from .lfm3 import LFM3VLVisionMixin
from .qwen import Qwen3_5TextModel, Qwen3Model
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("MarvinMuseForConditionalGeneration", "MarvinSparkForConditionalGeneration")
class MarvinQwen3TextModel(Qwen3Model):
    """Export the Qwen3 language tower shared by Marvin Muse and Spark."""

    model_arch = gguf.MODEL_ARCH.QWEN3

    def get_vocab_base_pre(self, tokenizer):
        # Spark's MNN-Pocket tokenizer retains Qwen2's split rules with a smaller vocabulary.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        pre = backend.pre_tokenizer if backend is not None else None
        pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                   r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        if pre is not None and json.loads(pre.__getstate__()) == {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        }:
            return "qwen2"
        return super().get_vocab_base_pre(tokenizer)


@ModelBase.register("MarvinVerseForConditionalGeneration")
class MarvinVerseTextModel(Qwen3_5TextModel):
    """Export the Qwen3.5 language tower from Marvin Verse."""

    model_arch = gguf.MODEL_ARCH.QWEN35
    supports_mtp_export = False
    no_mtp = True


class MarvinVisionModel(MmprojModel):
    """Export a native vision encoder followed by Marvin's text-width projector."""

    _encoder = "model.vision_tower.vision_tower.encoder."
    _patch_merger = "model.vision_tower.vision_tower.projector."
    _projector = "model.mm_projector."
    vision_type: str
    processor_type: str

    def get_vision_config(self):
        vision = self.global_config.get("mm_vision_config")
        if not isinstance(vision, dict) or vision.get("model_type") != self.vision_type:
            raise ValueError(f"Marvin requires native {self.vision_type} vision settings.")
        if (self.global_config.get("image_aspect_ratio") != "dynamic_resolution"
                or self.global_config.get("mm_patch_merge_type", "flat") != "flat"
                or self.global_config.get("mm_use_im_start_end") or self.global_config.get("mm_use_im_patch_token")):
            raise ValueError("Marvin requires dynamic resolution, flat patch merging and plain image placeholders.")
        if type(vision.get("patch_size")) is not int or vision["patch_size"] <= 0:
            raise ValueError("Marvin requires a positive vision patch size.")
        return dict(vision)

    @classmethod
    def filter_tensors(cls, item):
        if item[0].startswith((cls._encoder, cls._patch_merger, cls._projector)):
            return MmprojModel.filter_tensors(item)
        return None

    def prepare_tensors(self):
        projector = self.global_config.get("mm_projector_type", "linear")
        width = self.global_config.get("mm_hidden_size")
        if type(width) is not int or width <= 0 or width != self.vision_output_width():
            raise ValueError("Marvin mm_hidden_size must match the native vision output width.")
        if projector == "identity" and width == self.n_embd_text:
            layers = []
        elif projector == "linear":
            layers = [self._projector]
        elif (match := re.fullmatch(r"mlp([1-9][0-9]*)x_gelu", projector)) and int(match[1]) <= 64:
            layers = [f"{self._projector}{2 * i}." for i in range(int(match[1]))]
        else:
            raise ValueError("Unsupported Marvin projector or mismatched identity dimensions.")
        self.projector_tensors = {}
        expected = {}
        for i, prefix in enumerate(layers):
            for param, shape in (("weight", (self.n_embd_text, width)), ("bias", (self.n_embd_text,))):
                name = prefix + param
                expected[name] = shape
                self.projector_tensors[name] = f"mm.marvin.{i}.{param}"
            width = self.n_embd_text
        if self.projector_tensors.keys() != {key for key in self.model_tensors if key.startswith(self._projector)}:
            raise ValueError("Marvin projector tensors do not match the configured layers.")
        self.check_tensor_shapes(expected)
        self.projector_depth = len(layers)
        super().prepare_tensors()

    def check_tensor_shapes(self, expected):
        for name, shape in expected.items():
            if name not in self.model_tensors:
                raise ValueError(f"Missing native Marvin tensor: {name}")
            actual = tuple(self.model_tensors[name]().shape)
            if actual != shape:
                raise ValueError(f"Unexpected native Marvin tensor shape for {name}: {actual}, expected {shape}")

    def vision_output_width(self):
        return self.hparams_vision["out_hidden_size"]

    def set_gguf_parameters(self):
        processor = self.preprocessor_config
        vision = self.hparams_vision
        if (processor.get("image_processor_type") != self.processor_type
                or any(processor.get(key) is not True for key in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"))
                or processor.get("rescale_factor") != 1 / 255
                or processor.get("patch_size") != vision["patch_size"]):
            raise ValueError("Marvin requires matching native image preprocessing.")
        size = processor.get("size")
        minimum, maximum = (size.get(key) if isinstance(size, dict) else None for key in ("shortest_edge", "longest_edge"))
        if type(minimum) is not int or type(maximum) is not int or not 0 < minimum <= maximum:
            raise ValueError("Marvin requires a positive image pixel budget.")
        for key in ("image_mean", "image_std"):
            values = processor.get(key)
            if not isinstance(values, list) or len(values) != 3 or not all(
                type(v) in (int, float) and math.isfinite(v) and (key != "image_std" or v > 0) for v in values
            ):
                raise ValueError(f"Marvin requires valid RGB {key}.")
            processor[key] = [float(v) for v in values]
        resize_algo = {2: "bilinear", 3: "bicubic"}.get(processor.get("resample"))
        if resize_algo is None:
            raise ValueError("Marvin requires bilinear or bicubic image resizing.")
        if self.vision_type == "minicpmv4_7_vision":
            mode = processor.get("downsample_mode")
            scale = processor.get("scale_resolution")
            if mode not in ("4x", "16x") or type(scale) is not int or scale <= 0 or processor.get("slice_mode") is not False:
                raise ValueError("Marvin Verse requires native MiniCPM preprocessing without slicing.")
            align = 2 if mode == "4x" else 4
            vision["image_size"] = min(max(scale, math.isqrt(minimum - 1) + 1), math.isqrt(maximum))
            if vision["image_size"] < align * vision["patch_size"]:
                raise ValueError("Marvin Verse image budget is too small for an aligned patch block.")
            self.gguf_writer.add_vision_spatial_merge_size(align)
        elif (processor.get("merge_size") != vision["spatial_merge_size"]
                or processor.get("temporal_patch_size") != vision["temporal_patch_size"]):
            raise ValueError("Marvin requires matching native image patch geometry.")
        MmprojModel.set_gguf_parameters(self)
        self.gguf_writer.add_uint32("clip.vision.marvin_projector_layers", self.projector_depth)
        if self.vision_type != "lfm3_vl_vision":
            self.gguf_writer.add_vision_image_resize_algo(resize_algo)
            self.gguf_writer.add_vision_min_pixels(minimum)
            self.gguf_writer.add_vision_max_pixels(maximum)
            self.gguf_writer.add_vision_use_gelu(vision["hidden_act"] == "gelu_pytorch_tanh")

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        if name.startswith(self._projector):
            yield self.projector_tensors[name], data_torch
        else:
            yield from self.modify_vision_tensors(data_torch, name, bid)


@ModelBase.register("MarvinMuseForConditionalGeneration")
class MarvinMuseVisionModel(MarvinVisionModel, LFM3VLVisionMixin):
    """Reuse the LFM3-VL patchifier, encoder and merger."""

    vision_type = "lfm3_vl_vision"
    processor_type = "MarvinMuseNativeImageProcessor"

    def get_vision_config(self):
        vision = super().get_vision_config()
        merger = self.global_config.get("mm_vision_projector_config")
        if not isinstance(merger, dict) or merger.get("model_type") != "lfm3_vl":
            raise ValueError("Marvin Muse requires native LFM3-VL patch merger settings.")
        if (vision["spatial_merge_size"] != 2 or vision["temporal_patch_size"] != 2
                or vision["hidden_size"] != merger["vision_config"]["hidden_size"]
                or vision["spatial_merge_size"] != merger["vision_config"]["spatial_merge_size"]):
            raise ValueError("Marvin Muse has unsupported patch geometry or mismatched projector widths.")
        self.global_config.update(
            projector_hidden_act=merger["projector_hidden_act"],
            projector_bias=merger["projector_bias"],
            projector_layer_norm_eps=1e-6,
            downsample_factor=vision["spatial_merge_size"],
        )
        return {**vision, "image_size": 512}

    def vision_output_width(self):
        return self.global_config["mm_vision_projector_config"]["text_config"]["hidden_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._set_lfm3vl_gguf_parameters()

    def modify_vision_tensors(self, data_torch, name, bid):
        prefix, target = (self._encoder, "model.vision_tower.") if name.startswith(self._encoder) else (
            self._patch_merger, "model.multi_modal_projector.",
        )
        yield from self._lfm3vl_tensors(data_torch, target + name.removeprefix(prefix), bid)


@ModelBase.register("MarvinSparkForConditionalGeneration")
class MarvinSparkVisionModel(MarvinVisionModel, Qwen3VLVisionModel):
    """Reuse the Qwen3.5 vision tower through the Qwen3-VL tensor mapping."""

    vision_type = "qwen3_5_vision"
    processor_type = "MarvinSparkImageProcessorFast"

    def get_vision_config(self):
        vision = super().get_vision_config()
        if (vision["spatial_merge_size"] != 2 or vision["temporal_patch_size"] != 2 or vision["in_channels"] != 3
                or vision.get("deepstack_visual_indexes") or vision["hidden_act"] not in ("gelu", "gelu_pytorch_tanh")
                or self.global_config.get("mm_vision_projector_config") is not None):
            raise ValueError("Marvin Spark requires RGB Qwen3.5 vision with 2×2 patch merging and no DeepStack.")
        return vision

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.QWEN3VL)
        self.gguf_writer.add_vision_spatial_merge_size(self.hparams_vision["spatial_merge_size"])
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)

    def modify_vision_tensors(self, data_torch, name, bid):
        if not name.startswith(self._encoder):
            raise ValueError(f"Unexpected Marvin Spark vision tensor: {name}")
        yield from Qwen3VLVisionModel.modify_tensors(self, data_torch, "visual." + name.removeprefix(self._encoder), bid)


@ModelBase.register("MarvinVerseForConditionalGeneration")
class MarvinVerseVisionModel(MarvinVisionModel):
    """Export the MiniCPM-V 4.7 encoder including its internal ViT merger."""

    vision_type = "minicpmv4_7_vision"
    processor_type = "MarvinVerseImageProcessorFast"

    def get_vision_config(self):
        vision = super().get_vision_config()
        if (vision.get("window_kernel_size") not in (None, [2, 2]) or vision.get("num_channels", 3) != 3
                or not 0 <= vision["insert_layer_id"] < vision["num_hidden_layers"]
                or vision["hidden_act"] not in ("gelu", "gelu_pytorch_tanh")
                or self.global_config.get("mm_vision_projector_config") is not None):
            raise ValueError("Marvin Verse requires the native RGB MiniCPM 2×2 ViT merger.")
        return vision

    def vision_output_width(self):
        return self.hparams_vision["hidden_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.MINICPMV4_7)
        self.gguf_writer.add_vision_projector_scale_factor(2)
        self.gguf_writer.add_vision_wa_layer_indexes([self.hparams_vision["insert_layer_id"]])
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision["layer_norm_eps"])

    def modify_vision_tensors(self, data_torch, name, bid):
        if not name.startswith(self._encoder):
            raise ValueError(f"Unexpected Marvin Verse vision tensor: {name}")
        yield from MmprojModel.modify_tensors(self, data_torch, "model.vision_tower." + name.removeprefix(self._encoder), bid)
