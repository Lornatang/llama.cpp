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
import math
from typing import Iterable

import torch

from .base import ModelBase, gguf
from .lfm3 import LFM3VLVisionMixin
from .qwen import Qwen3_5TextModel, Qwen3Model


@ModelBase.register("MarvinMuseForConditionalGeneration", "MarvinSparkForConditionalGeneration")
class MarvinQwen3TextModel(Qwen3Model):
    """Export the Qwen3 language tower shared by Marvin Muse and Spark."""

    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("MarvinVerseForConditionalGeneration")
class MarvinVerseTextModel(Qwen3_5TextModel):
    """Export the Qwen3.5 language tower from Marvin Verse."""

    model_arch = gguf.MODEL_ARCH.QWEN35
    supports_mtp_export = False
    no_mtp = True



@ModelBase.register("MarvinMuseForConditionalGeneration")
class MarvinMuseVisionModel(LFM3VLVisionMixin):
    """Export the LFM3-VL tower and both of Muse's projectors."""

    _encoder = "model.vision_tower.vision_tower.encoder."
    _patch_merger = "model.vision_tower.vision_tower.projector."
    _projector = "model.mm_projector."

    def get_vision_config(self):
        vision = self.global_config.get("mm_vision_config")
        merger = self.global_config.get("mm_vision_projector_config")
        if (not isinstance(vision, dict) or vision.get("model_type") != "lfm3_vl_vision"
                or not isinstance(merger, dict) or merger.get("model_type") != "lfm3_vl"
                or any(key in self.global_config for key in ("vision_config", "lfm3_vl_config", "siglip2_vision_config"))):
            raise ValueError("Marvin Muse requires native LFM3-VL vision and patch merger settings.")
        if (vision["spatial_merge_size"] != 2 or vision["temporal_patch_size"] != 2
                or type(vision["patch_size"]) is not int or vision["patch_size"] <= 0
                or vision["hidden_size"] != merger["vision_config"]["hidden_size"]
                or vision["spatial_merge_size"] != merger["vision_config"]["spatial_merge_size"]
                or self.global_config["mm_hidden_size"] != merger["text_config"]["hidden_size"]):
            raise ValueError("Marvin Muse has unsupported patch geometry or mismatched projector widths.")
        if (self.global_config["image_aspect_ratio"] != "dynamic_resolution"
                or self.global_config.get("mm_patch_merge_type", "flat") != "flat"):
            raise ValueError("Marvin Muse requires dynamic resolution and flat patch merging.")
        if self.global_config["mm_projector_type"] not in ("linear", "mlp2x_gelu"):
            raise ValueError("Marvin Muse requires a linear or mlp2x_gelu projector for GGUF.")

        self.global_config.update(
            projector_hidden_act=merger["projector_hidden_act"],
            projector_bias=merger["projector_bias"],
            projector_layer_norm_eps=1e-6,
            downsample_factor=vision["spatial_merge_size"],
        )
        return {**vision, "image_size": 512}

    @classmethod
    def filter_tensors(cls, item):
        if item[0].startswith((cls._encoder, cls._patch_merger, cls._projector)):
            return super().filter_tensors(item)
        return None

    def prepare_tensors(self):
        vision = self.hparams_vision
        assert vision is not None
        merger = self.global_config["mm_vision_projector_config"]
        width = vision["hidden_size"]
        merged_width = merger["text_config"]["hidden_size"]
        hidden = merger["projector_hidden_size"]
        expected = {
            self._encoder + "patchifier.weight": (width, 6 * vision["patch_size"] ** 2),
            self._encoder + "embedding_norm.weight": (width,),
            self._patch_merger + "layer_norm.weight": (width,),
            self._patch_merger + "layer_norm.bias": (width,),
            self._patch_merger + "linear_1.weight": (hidden, width * 4),
            self._patch_merger + "linear_1.bias": (hidden,),
            self._patch_merger + "linear_2.weight": (merged_width, hidden),
            self._patch_merger + "linear_2.bias": (merged_width,),
        }
        outer_layers = (("0.", merged_width), ("2.", self.n_embd_text)) if self.global_config["mm_projector_type"] == "mlp2x_gelu" else (
            ("", merged_width),
        )
        for prefix, input_width in outer_layers:
            expected[self._projector + prefix + "weight"] = (self.n_embd_text, input_width)
            expected[self._projector + prefix + "bias"] = (self.n_embd_text,)

        missing = sorted(expected.keys() - self.model_tensors.keys())
        if missing:
            raise ValueError(f"Missing native Marvin Muse vision/projector tensors: {', '.join(missing)}")
        for key, shape in expected.items():
            actual = tuple(self.model_tensors[key]().shape)
            if actual != shape:
                raise ValueError(f"Unexpected native Marvin Muse tensor shape for {key}: {actual}, expected {shape}")
        super().prepare_tensors()

    def set_gguf_parameters(self):
        processor = self.preprocessor_config
        vision = self.hparams_vision
        assert vision is not None
        if (processor.get("image_processor_type") != "MarvinMuseNativeImageProcessor"
                or any(processor.get(key) is not True for key in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"))
                or processor.get("rescale_factor") != 1 / 255
                or processor.get("patch_size") != vision["patch_size"]
                or processor.get("temporal_patch_size") != vision["temporal_patch_size"]
                or processor.get("merge_size") != vision["spatial_merge_size"]):
            raise ValueError("Marvin Muse requires matching native LFM3-VL image preprocessing.")
        size = processor.get("size")
        min_pixels, max_pixels = (size.get(key) if isinstance(size, dict) else None for key in ("shortest_edge", "longest_edge"))
        if type(min_pixels) is not int or type(max_pixels) is not int or not 0 < min_pixels <= max_pixels:
            raise ValueError("Marvin Muse requires a positive image pixel budget.")
        for key in ("image_mean", "image_std"):
            values = processor.get(key)
            if not isinstance(values, list) or len(values) != 3 or not all(
                type(v) in (int, float) and math.isfinite(v) and (key != "image_std" or v > 0) for v in values
            ):
                raise ValueError(f"Marvin Muse requires valid RGB {key}.")
            processor[key] = [float(v) for v in values]
        super().set_gguf_parameters()
        self._set_lfm3vl_gguf_parameters("marvin_muse")

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        if name.startswith(self._encoder):
            suffix = name.removeprefix(self._encoder)
            yield from self._lfm3vl_tensors(data_torch, "model.vision_tower." + suffix, bid)
        elif name.startswith(self._patch_merger):
            suffix = name.removeprefix(self._patch_merger)
            yield from self._lfm3vl_tensors(data_torch, "model.multi_modal_projector." + suffix, bid)
        else:
            suffix = name.removeprefix(self._projector)
            layer, _, param = suffix.rpartition(".")
            layers = {"": 3} if self.global_config["mm_projector_type"] == "linear" else {"0": 3, "2": 4}
            yield f"mm.{layers[layer]}.{param}", data_torch
