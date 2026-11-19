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
import re
from typing import Any, Callable, Dict, Iterable, Tuple

import torch

from .base import MmprojModel, ModelBase, gguf, logger
from .qwen import Qwen3Model

_STAGES = ((0, 2), (2, 12), (4, 24), (7, 4), (10, 2))
_WIDTHS = (96, 192, 384, 768, 1536)


def _fuse_conv_bn(
    weight: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse a Conv2d and BatchNorm2d into a single conv weight/bias pair.

    Args:
        weight (torch.Tensor): Conv2d weight of shape ``[out_c, in_c, kH, kW]``.
        bn_weight (torch.Tensor): BatchNorm affine scale (gamma), shape ``[out_c]``.
        bn_bias (torch.Tensor): BatchNorm affine bias (beta), shape ``[out_c]``.
        running_mean (torch.Tensor): BatchNorm running mean, shape ``[out_c]``.
        running_var (torch.Tensor): BatchNorm running variance, shape ``[out_c]``.
        eps (float): BatchNorm epsilon used in the variance denominator. Defaults to ``1e-5``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Float32 convolution weights in the original layout and a bias of shape ``[out_c]``.
    """
    weight, bn_weight, bn_bias, running_mean, running_var = (
        t.to(torch.float32) for t in (weight, bn_weight, bn_bias, running_mean, running_var)
    )
    std = (running_var + eps).sqrt()
    scale = (bn_weight / std).reshape(-1, 1, 1, 1)
    fused_w = weight * scale
    fused_b = bn_bias - running_mean * bn_weight / std
    return fused_w, fused_b


@ModelBase.register("MarvinMuseForConditionalGeneration")
class MarvinMuseTextModel(Qwen3Model):
    """Export only the Qwen3 language tower from a Marvin Muse checkpoint.

    Attributes:
        model_arch (gguf.MODEL_ARCH): GGUF architecture tag, always Qwen3.
    """

    model_arch = gguf.MODEL_ARCH.QWEN3

    @classmethod
    def filter_tensors(cls, item: Tuple[str, Callable[[], torch.Tensor]]) -> Tuple[str, Callable[[], torch.Tensor]] | None:
        """Select the tensors handled by the Qwen3 text converter.

        Args:
            item (Tuple[str, Callable[[], torch.Tensor]]): Checkpoint pair ``(tensor_name, lazy_loader)``.

        Returns:
            Tuple[str, Callable[[], torch.Tensor]] | None: The pair with any ``language_model.`` prefix removed, or ``None`` for discarded tensors.
        """
        name, gen = item
        if "vision_tower" in name or "mm_projector" in name:
            return None
        if name.startswith("language_model."):
            name = name[len("language_model.") :]
        return super().filter_tensors((name, gen))


@ModelBase.register("MarvinMuseForConditionalGeneration")
class MarvinMuseVisionModel(MmprojModel):
    """Export the Marvin Muse vision tower and MLP projector as an mmproj GGUF.

    Attributes:
        block_count (int): Synthetic block count for the mmproj tensor map.
        tensor_map: GGUF tensor name map for ``MODEL_ARCH.MMPROJ``.
        _ffn_bn_buf (dict[str, dict[str, torch.Tensor]]): Per-block buffer for ConvFFN convolution and batch normalization tensors.
        _expected_shapes (dict[str, tuple[int, ...]]): PyTorch dimension order for each required GGUF tensor.
        _emitted_tensors (set[str]): Required tensor names already emitted by the converter.
    """

    _ffn_bn_buf: dict[str, dict[str, torch.Tensor]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Marvin Muse vision converter and validate the checkpoint configuration.

        Args:
            *args (Any): Positional arguments forwarded to ``MmprojModel``.
            **kwargs (Any): Keyword args forwarded to ``MmprojModel``.

        Raises:
            ValueError: If the model architecture or preprocessing configuration is unsupported.
        """
        super().__init__(*args, **kwargs)
        self._ffn_bn_buf = {}
        self._validate_config()
        self._expected_shapes = self._tensor_shapes()
        self._emitted_tensors: set[str] = set()

    def _tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Build the tensor shape requirements for the configured FastViT-HD widths.

        Returns:
            dict[str, tuple[int, ...]]: Required GGUF tensor names mapped to shapes in PyTorch dimension order.
        """
        shapes: dict[str, tuple[int, ...]] = {}

        def affine(name: str, shape: tuple[int, ...]) -> None:
            shapes[name + ".weight"] = shape
            shapes[name + ".bias"] = (shape[0],)

        c0 = self._widths[0]
        affine("v.fastvit.stem.0", (c0, 3, 3, 3))
        affine("v.fastvit.stem.1", (c0, 1, 3, 3))
        affine("v.fastvit.stem.2", (c0, c0, 1, 1))
        for (net, count), c in zip(_STAGES, self._widths, strict=True):
            for block in range(count):
                prefix = f"v.fastvit.net.{net}.blk.{block}"
                affine(f"{prefix}.ffn.dw", (c, 1, 7, 7))
                affine(f"{prefix}.ffn.fc1", (4 * c, c, 1, 1))
                affine(f"{prefix}.ffn.fc2", (c, 4 * c, 1, 1))
                if net in (0, 2, 4):
                    affine(f"{prefix}.mixer", (c, 1, 3, 3))
                    shapes[f"{prefix}.ls.weight"] = (c, 1, 1)
                else:
                    affine(f"{prefix}.norm", (c,))
                    shapes[f"{prefix}.qkv.weight"] = (3 * c, c)
                    affine(f"{prefix}.proj", (c, c))
                    shapes[f"{prefix}.ls1.weight"] = shapes[f"{prefix}.ls2.weight"] = (c, 1, 1)
        for net, c in zip((1, 3, 5, 8), self._widths[1:], strict=True):
            affine(f"v.fastvit.net.{net}.down.0", (c, 1, 7, 7))
            affine(f"v.fastvit.net.{net}.down.1", (c, c, 1, 1))
        for net, c in zip((6, 9), self._widths[3:], strict=True):
            affine(f"v.fastvit.net.{net}.pos", (c, 1, 7, 7))
        vision_dim = 2 * self._widths[-1]
        affine("v.fastvit.conv_exp", (vision_dim, 1, 3, 3))
        affine("v.fastvit.conv_exp.se.reduce", (vision_dim // 16, vision_dim, 1, 1))
        affine("v.fastvit.conv_exp.se.expand", (vision_dim, vision_dim // 16, 1, 1))
        affine("mm.0", (self.n_embd_text, vision_dim))
        affine("mm.2", (self.n_embd_text, self.n_embd_text))
        return shapes

    def prepare_tensors(self) -> None:
        """Prepare the vision and projector tensors and check that the export is complete.

        Raises:
            ValueError: If a mapped tensor has an invalid shape or a required tensor or batch normalization parameter is missing.
        """
        super().prepare_tensors()
        if self._ffn_bn_buf:
            needed = {"w", "bn_w", "bn_b", "mean", "var"}
            incomplete = [f"{key}: {', '.join(sorted(needed - set(parts)))}" for key, parts in sorted(self._ffn_bn_buf.items())]
            raise ValueError(f"marvin_muse: incomplete ConvFFN batch normalization tensors: {'; '.join(incomplete)}")
        missing = sorted(self._expected_shapes.keys() - self._emitted_tensors)
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"marvin_muse: missing {len(missing)} required vision/projector tensors: {preview}")

    def get_vision_config(self) -> Dict[str, Any] | None:
        """Fill the generic vision metadata required by ``MmprojModel``.

        Returns:
            Dict[str, Any] | None: A copy of the vision config with the converter's dimension metadata and defaults.
        """
        cfg = dict(self.global_config.get("vision_config") or {})
        # The generic CLIP block count satisfies the metadata schema. FastViT stages are loaded separately.
        cfg["n_layers"] = 1
        cfg["hidden_size"] = int(self.global_config.get("mm_hidden_size", 3072))
        cfg["intermediate_size"] = cfg["hidden_size"]
        cfg["num_attention_heads"] = max(1, cfg["hidden_size"] // 32)
        cfg.setdefault("image_size", 1024)
        cfg.setdefault("patch_size", 64)
        return cfg

    def _validate_config(self) -> None:
        """Validate the FastViT-HD graph settings and normalize the RGB statistics.

        Raises:
            ValueError: If the runtime does not support the model or preprocessing configuration.
        """
        cfg_vision = self.hparams_vision or {}
        for config, expected in (
            (self.global_config, {
                "mm_projector_type": "mlp2x_gelu", "mm_patch_merge_type": "flat", "image_aspect_ratio": "pad",
            }),
            (cfg_vision, {
                "model_type": "fastvithd", "image_size": 1024, "patch_size": 64, "num_channels": 3,
                "inference_mode": True, "layers": [count for _, count in _STAGES],
                "token_mixers": ["repmixer", "repmixer", "repmixer", "attention", "attention"],
                "down_patch_size": 7, "down_stride": 2, "downsamples": [True] * 5,
                "repmixer_kernel_size": 3, "mlp_ratios": [4] * 5, "cls_ratio": 2.0,
                "pos_embs_shapes": [None, None, None, [7, 7], [7, 7]],
            }),
            (self.preprocessor_config, {
                "size": {"shortest_edge": 1024}, "crop_size": {"height": 1024, "width": 1024},
                "do_resize": True, "do_rescale": True, "do_normalize": True, "do_pad": False,
                "resample": 3, "rescale_factor": 1 / 255,
            }),
        ):
            for key, value in expected.items():
                if config.get(key, value) != value:
                    raise ValueError(f"marvin_muse: unsupported {key}={config[key]!r}; expected {value!r}")

        self._widths = tuple(cfg_vision.get("embed_dims", _WIDTHS))
        if (len(self._widths) != len(_STAGES)
                or any(not isinstance(c, int) or c <= 0 for c in self._widths)
                or any(b % a for a, b in zip(self._widths, self._widths[1:]))
                or any(c % 32 for c in self._widths[3:])
                or cfg_vision["hidden_size"] != 2 * self._widths[-1]):
            raise ValueError(f"marvin_muse: incompatible embed_dims {self._widths} and mm_hidden_size {cfg_vision['hidden_size']}")

        # GGUF infers array element types from Python values. The loader expects float32.
        for key, default in (("image_mean", 0.0), ("image_std", 1.0)):
            values = self.preprocessor_config.get(key, [default] * 3)
            if not isinstance(values, (list, tuple)) or len(values) != 3 or not all(
                type(v) in (int, float) and math.isfinite(v) and (key != "image_std" or v > 0) for v in values
            ):
                raise ValueError(f"marvin_muse: invalid {key}={values!r}; expected three finite RGB values and positive std")
            self.preprocessor_config[key] = [float(v) for v in values]

    def set_gguf_parameters(self) -> None:
        """Write the generic vision metadata and Marvin Muse's projector settings."""
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MARVIN_MUSE)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-5)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        """Keep convolution kernels compatible with ggml backends.

        Args:
            name (str): Original checkpoint tensor name.
            new_name (str): Mapped GGUF tensor name.
            bid (int | None): Block index when the tensor belongs to a block; otherwise ``None``.
            n_dims (int): Number of dimensions of the tensor.

        Returns:
            gguf.GGMLQuantizationType | bool: F16 or F32 for convolution weights, F32 for layer scales, or the base-class policy.
        """
        if new_name.startswith("v.fastvit.") or new_name.startswith("mm."):
            if n_dims == 4:
                return gguf.GGMLQuantizationType.F32 if self.ftype == gguf.LlamaFileType.ALL_F32 else gguf.GGMLQuantizationType.F16
            if n_dims == 3:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: Tuple[str, Callable[[], torch.Tensor]]) -> Tuple[str, Callable[[], torch.Tensor]] | None:
        """Keep vision-tower and projector tensors only.

        Args:
            item (Tuple[str, Callable[[], torch.Tensor]]): Checkpoint pair ``(tensor_name, lazy_loader)``.

        Returns:
            Tuple[str, Callable[[], torch.Tensor]] | None: The original pair for a vision or projector tensor, otherwise ``None``.
        """
        name, gen = item
        if "vision_tower" in name or "mm_projector" in name:
            return (name, gen)
        return None

    def _emit(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Validate a mapped tensor and record it if required.

        Args:
            name (str): Destination GGUF tensor name.
            data (torch.Tensor): Mapped tensor in PyTorch dimension order.

        Yields:
            Tuple[str, torch.Tensor]: The destination name and contiguous tensor.

        Raises:
            ValueError: If the tensor name or shape is incompatible with the graph.
        """
        shape = self._expected_shapes.get(name)
        required = shape is not None
        if shape is None and name.endswith(".qkv.bias"):
            weight_shape = self._expected_shapes.get(name.removesuffix("bias") + "weight")
            if weight_shape is not None:
                shape = (weight_shape[0],)
        if shape is None or tuple(data.shape) != shape:
            raise ValueError(f"marvin_muse: unexpected tensor {name} with shape {tuple(data.shape)}; expected {shape}")
        if required:
            self._emitted_tensors.add(name)
        yield name, data.contiguous()

    def _map_projector(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Map ``mm_projector.{i}.{weight,bias}`` to ``mm.{i}.{weight,bias}``.

        Args:
            name (str): Checkpoint tensor name.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Mapped projector tensor when ``name`` matches the projector pattern.

        Raises:
            ValueError: If the matched tensor is incompatible with the projector graph.
        """
        m = re.match(r".*mm_projector\.(\d+)\.(weight|bias)$", name)
        if not m:
            return
        yield from self._emit(f"mm.{m.group(1)}.{m.group(2)}", data)

    def _map_stem(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Map stem ``patch_embed`` reparam convs to ``v.fastvit.stem.*``.

        Args:
            name (str): Checkpoint tensor name.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Mapped stem tensor when ``name`` matches ``patch_embed.{i}.reparam_conv.{weight|bias}``.

        Raises:
            ValueError: If the matched tensor is incompatible with the stem.
        """
        m = re.match(r".*patch_embed\.(\d+)\.reparam_conv\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.stem.{m.group(1)}.{m.group(2)}", data)

    def _map_conv_exp(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Map the final conv expansion and SE blocks.

        Args:
            name (str): Checkpoint tensor name under ``backbone.conv_exp``.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Mapped ``v.fastvit.conv_exp.*`` tensor when ``name`` matches a known convolution or SE parameter.

        Raises:
            ValueError: If the matched tensor is incompatible with the expansion or SE layers.
        """
        if ".backbone.conv_exp.reparam_conv.weight" in name:
            yield from self._emit("v.fastvit.conv_exp.weight", data)
        elif ".backbone.conv_exp.reparam_conv.bias" in name:
            yield from self._emit("v.fastvit.conv_exp.bias", data)
        elif ".backbone.conv_exp.se.reduce.weight" in name:
            yield from self._emit("v.fastvit.conv_exp.se.reduce.weight", data)
        elif ".backbone.conv_exp.se.reduce.bias" in name:
            yield from self._emit("v.fastvit.conv_exp.se.reduce.bias", data)
        elif ".backbone.conv_exp.se.expand.weight" in name:
            yield from self._emit("v.fastvit.conv_exp.se.expand.weight", data)
        elif ".backbone.conv_exp.se.expand.bias" in name:
            yield from self._emit("v.fastvit.conv_exp.se.expand.bias", data)

    def _map_network(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Map backbone ``network.*`` tensors into the GGUF vision layout.

        Buffers the ConvFFN depthwise convolution and batch normalization parameters until the full set is available for fusion.

        Args:
            name (str): Checkpoint tensor name under ``backbone.network``.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Mapped tensors, including the fused depthwise weight and bias for a complete ConvFFN block.

        Raises:
            ValueError: If a mapped tensor or a parameter needed for batch normalization fusion has the wrong shape.
        """
        # Downsample: network.{i}.proj.{0,1}.*
        m = re.match(r".*network\.(\d+)\.proj\.0\.lkb_reparam\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.down.0.{m.group(2)}", data)
            return
        m = re.match(r".*network\.(\d+)\.proj\.1\.reparam_conv\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.down.1.{m.group(2)}", data)
            return

        # Conditional positional encoding: network.{i}.reparam_conv.*
        m = re.match(r".*network\.(\d+)\.reparam_conv\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.pos.{m.group(2)}", data)
            return

        # Token mixer
        m = re.match(r".*network\.(\d+)\.(\d+)\.token_mixer\.reparam_conv\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.mixer.{m.group(3)}", data)
            return

        # Mixer layer scale
        m = re.match(r".*network\.(\d+)\.(\d+)\.layer_scale$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.ls.weight", data)
            return

        # Attention block: norms / scales / qkv / proj
        m = re.match(r".*network\.(\d+)\.(\d+)\.norm\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.norm.{m.group(3)}", data)
            return
        m = re.match(r".*network\.(\d+)\.(\d+)\.layer_scale_1$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.ls1.weight", data)
            return
        m = re.match(r".*network\.(\d+)\.(\d+)\.layer_scale_2$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.ls2.weight", data)
            return
        m = re.match(r".*network\.(\d+)\.(\d+)\.token_mixer\.qkv\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.qkv.{m.group(3)}", data)
            return
        m = re.match(r".*network\.(\d+)\.(\d+)\.token_mixer\.proj\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.proj.{m.group(3)}", data)
            return

        # ConvFFN: pass through fc1/fc2; fuse depthwise conv + BN
        m = re.match(r".*network\.(\d+)\.(\d+)\.convffn\.fc([12])\.(weight|bias)$", name)
        if m:
            yield from self._emit(f"v.fastvit.net.{m.group(1)}.blk.{m.group(2)}.ffn.fc{m.group(3)}.{m.group(4)}", data)
            return

        # The converter can encounter batch normalization parameters before or after the convolution weights.
        m = re.match(r".*network\.(\d+)\.(\d+)\.convffn\.conv\.(conv\.weight|bn\.(weight|bias|running_mean|running_var|num_batches_tracked))$", name)
        if m:
            ni, bi = m.group(1), m.group(2)
            part = m.group(3)
            if part == "bn.num_batches_tracked":
                return
            key = f"{ni}.{bi}"
            buf = self._ffn_bn_buf.setdefault(key, {})
            if part == "conv.weight":
                buf["w"] = data
            elif part == "bn.weight":
                buf["bn_w"] = data
            elif part == "bn.bias":
                buf["bn_b"] = data
            elif part == "bn.running_mean":
                buf["mean"] = data
            elif part == "bn.running_var":
                buf["var"] = data
            needed = {"w", "bn_w", "bn_b", "mean", "var"}
            if needed.issubset(buf):
                prefix = f"v.fastvit.net.{ni}.blk.{bi}.ffn.dw"
                for part in needed:
                    param = "weight" if part == "w" else "bias"
                    shape = self._expected_shapes.get(f"{prefix}.{param}")
                    if tuple(buf[part].shape) != shape:
                        raise ValueError(f"marvin_muse: unexpected shape {tuple(buf[part].shape)} for {prefix} {part}; expected {shape}")
                fw, fb = _fuse_conv_bn(buf["w"], buf["bn_w"], buf["bn_b"], buf["mean"], buf["var"])
                yield from self._emit(f"{prefix}.weight", fw)
                yield from self._emit(f"{prefix}.bias", fb)
                del self._ffn_bn_buf[key]
            return

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[Tuple[str, torch.Tensor]]:
        """Route each checkpoint tensor to the appropriate GGUF mapper.

        Args:
            data_torch (torch.Tensor): Source tensor from the checkpoint.
            name (str): Checkpoint tensor name.
            bid (int | None): Block index from the base converter. Unused for Marvin Muse vision export.

        Yields:
            Tuple[str, torch.Tensor]: Mapped pairs for retained vision and projector tensors.

        Raises:
            ValueError: If a mapped tensor is incompatible with the runtime graph.
        """
        del bid
        if "mm_projector" in name:
            yield from self._map_projector(name, data_torch)
            return
        if "vision_tower" not in name:
            return
        # Classifier head under backbone.head.* is unused at inference time.
        if ".backbone.head." in name:
            return
        if "patch_embed" in name:
            yield from self._map_stem(name, data_torch)
            return
        if ".backbone.conv_exp." in name:
            yield from self._map_conv_exp(name, data_torch)
            return
        if ".backbone.network." in name:
            yield from self._map_network(name, data_torch)
            return
        logger.warning("Skipping unmapped vision tensor: %s %s", name, tuple(data_torch.shape))
