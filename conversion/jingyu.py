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
import re
from typing import Any, Callable, Dict, Iterable, Tuple

import torch

from .base import MmprojModel, ModelBase, gguf, logger
from .qwen import Qwen3Model


def _fuse_conv_bn(
    weight: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor,
    running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-5,
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
        Tuple[torch.Tensor, torch.Tensor]: ``(fused_weight, fused_bias)`` where ``fused_weight`` keeps the conv layout and ``fused_bias`` has shape
        ``[out_c]``. Suitable for inference without a separate BN op.
    """
    std = (running_var + eps).sqrt()
    scale = (bn_weight / std).reshape(-1, 1, 1, 1)
    fused_w = weight * scale
    fused_b = bn_bias - running_mean * bn_weight / std
    return fused_w, fused_b


@ModelBase.register("JingyuForConditionalGeneration")
class JingyuTextModel(Qwen3Model):
    """Export only the Qwen3 language tower from a Jingyu checkpoint.

    Attributes:
        model_arch (gguf.MODEL_ARCH): GGUF architecture tag, always Qwen3.
    """

    model_arch = gguf.MODEL_ARCH.QWEN3

    @classmethod
    def filter_tensors(cls, item: Tuple[str, Callable[[], torch.Tensor]]) -> Tuple[str, Callable[[], torch.Tensor]] | None:
        """Keep language-model tensors; drop vision and projector weights.

        Also strips the ``language_model.`` prefix so names match the Qwen3 converter expectations.

        Args:
            item (Tuple[str, Callable[[], torch.Tensor]]): Checkpoint pair ``(tensor_name, lazy_loader)``.

        Returns:
            Tuple[str, Callable[[], torch.Tensor]] | None: The possibly renamed pair for export, or ``None`` when the tensor belongs to
            ``vision_tower`` / ``mm_projector`` and must be skipped.
        """
        name, gen = item
        if "vision_tower" in name or "mm_projector" in name:
            return None
        if name.startswith("language_model."):
            name = name[len("language_model.") :]
        return super().filter_tensors((name, gen))


@ModelBase.register("JingyuForConditionalGeneration")
class JingyuVisionModel(MmprojModel):
    """Export the Jingyu vision tower and MLP projector as an mmproj GGUF.

    Attributes:
        block_count (int): Synthetic block count for the mmproj tensor map.
        tensor_map: GGUF tensor name map for ``MODEL_ARCH.MMPROJ``.
        _ffn_bn_buf (dict[str, dict[str, torch.Tensor]]): Per-block buffer that accumulates ConvFFN depthwise conv and BatchNorm pieces until they can
        be fused.
    """

    _ffn_bn_buf: dict[str, dict[str, torch.Tensor]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize vision export state and an empty BN fusion buffer.

        Args:
            *args (Any): Positional args forwarded to ``MmprojModel``.
            **kwargs (Any): Keyword args forwarded to ``MmprojModel``.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        self.block_count = 1
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)
        self._ffn_bn_buf = {}

    def get_vision_config(self) -> Dict[str, Any] | None:
        """Build a vision hyperparameter dict for MmprojModel helpers.

        Fills synthetic keys (layer counts, head count, sizes) required by the base loader when the Hugging Face config omits them.

        Returns:
            Dict[str, Any] | None: Vision config dictionary. Never ``None`` for a valid Jingyu checkpoint; the ``| None`` matches the base-class
            signature.
        """
        cfg = dict(self.global_config.get("vision_config") or {})
        # Synthetic keys so MmprojModel.__init__ / find_vparam succeed.
        cfg.setdefault("n_layers", 1)
        cfg.setdefault("num_hidden_layers", 1)
        cfg["hidden_size"] = int(self.global_config.get("mm_hidden_size", cfg.get("hidden_size", 3072)))
        cfg.setdefault("intermediate_size", int(self.global_config.get("mm_hidden_size", 3072)))
        cfg.setdefault("num_attention_heads", max(1, cfg["hidden_size"] // 32))
        cfg.setdefault("image_size", 1024)
        cfg.setdefault("patch_size", 64)
        return cfg

    def set_gguf_parameters(self) -> None:
        """Write Jingyu mmproj metadata expected by the C++ clip loader.

        Writes projector type, image/patch sizes, embedding dims, normalization stats, and architecture constants under the ``fastvit.*`` key
        namespace.

        Returns:
            None
        """
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_vision_projection_dim(self.n_embd_text)
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.JINGYU)

        image_size = int((self.hparams_vision or {}).get("image_size", 1024))
        patch_size = int((self.hparams_vision or {}).get("patch_size", 64))
        vision_dim = int(self.global_config.get("mm_hidden_size", 3072))

        self.gguf_writer.add_vision_image_size(image_size)
        self.gguf_writer.add_vision_patch_size(patch_size)
        self.gguf_writer.add_vision_embedding_length(vision_dim)
        self.gguf_writer.add_vision_feed_forward_length(vision_dim)
        self.gguf_writer.add_vision_block_count(1)
        self.gguf_writer.add_vision_head_count(max(1, vision_dim // 32))
        self.gguf_writer.add_vision_use_gelu(True)
        # Required by clip_model_loader::load_hparams; unused by ChannelNorm.
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-5)

        mean = self.preprocessor_config.get("image_mean", [0.0, 0.0, 0.0])
        std = self.preprocessor_config.get("image_std", [1.0, 1.0, 1.0])
        self.gguf_writer.add_vision_image_mean(mean)
        self.gguf_writer.add_vision_image_std(std)

        # Architecture constants consumed by the C++ Jingyu vision graph.
        self.gguf_writer.add_uint32("fastvit.image_size", image_size)
        self.gguf_writer.add_uint32("fastvit.patch_size", patch_size)
        self.gguf_writer.add_uint32("fastvit.embed_dim", vision_dim)
        for i, d in enumerate((self.hparams_vision or {}).get("embed_dims", [96, 192, 384, 768, 1536])):
            self.gguf_writer.add_uint32(f"fastvit.embed_dims.{i}", int(d))
        for i, n in enumerate((self.hparams_vision or {}).get("layers", [2, 12, 24, 4, 2])):
            self.gguf_writer.add_uint32(f"fastvit.layers.{i}", int(n))

        logger.info(
            "Jingyu mmproj: projector=jingyu image=%dx%d patch=%d vision_dim=%d text_dim=%d",
            image_size, image_size, patch_size, vision_dim, self.n_embd_text,
        )

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        """Force high precision for vision and projector tensors.

        Args:
            name (str): Original checkpoint tensor name.
            new_name (str): Mapped GGUF tensor name.
            bid (int | None): Block index when the tensor belongs to a block; otherwise ``None``.
            n_dims (int): Number of dimensions of the tensor.

        Returns:
            gguf.GGMLQuantizationType | bool: ``F16`` or ``F32`` for tensors whose ``new_name`` starts with ``v.fastvit.`` or ``mm.``; otherwise the
            base-class result (a quantization type or ``False`` to keep the default policy).
        """
        if new_name.startswith("v.fastvit.") or new_name.startswith("mm."):
            if self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                return gguf.GGMLQuantizationType.F16
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: Tuple[str, Callable[[], torch.Tensor]]) -> Tuple[str, Callable[[], torch.Tensor]] | None:
        """Keep vision-tower and projector tensors only.

        Args:
            item (Tuple[str, Callable[[], torch.Tensor]]): Checkpoint pair ``(tensor_name, lazy_loader)``.

        Returns:
            Tuple[str, Callable[[], torch.Tensor]] | None: The original pair when ``tensor_name`` contains ``vision_tower`` or ``mm_projector``;
            otherwise ``None``.
        """
        name, gen = item
        if "vision_tower" in name or "mm_projector" in name:
            return (name, gen)
        return None

    def _emit(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield a contiguous float32 tensor under the given GGUF name.

        Args:
            name (str): Destination GGUF tensor name.
            data (torch.Tensor): Source tensor of any dtype/layout.

        Yields:
            Tuple[str, torch.Tensor]: ``(name, float32_contiguous_tensor)``.
        """
        yield name, data.contiguous().to(torch.float32)

    def _map_projector(self, name: str, data: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
        """Map ``mm_projector.{i}.{weight,bias}`` to ``mm.{i}.{weight,bias}``.

        Args:
            name (str): Checkpoint tensor name.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Mapped projector tensor when ``name`` matches the projector pattern; otherwise yields nothing.
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
            Tuple[str, torch.Tensor]: Mapped stem tensor when ``name`` matches ``patch_embed.{i}.reparam_conv.{weight|bias}``; otherwise yields
            nothing.
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
            Tuple[str, torch.Tensor]: Mapped ``v.fastvit.conv_exp.*`` tensor when ``name`` matches a known conv_exp / SE path; otherwise yields
            nothing.
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

        Handles downsample, positional conv, mixer, attention, and ConvFFN paths. ConvFFN depthwise conv+BN pieces are buffered and fused when a
        complete set is available.

        Args:
            name (str): Checkpoint tensor name under ``backbone.network``.
            data (torch.Tensor): Source tensor.

        Yields:
            Tuple[str, torch.Tensor]: Zero or more mapped tensors. For ConvFFN BN pieces, yields nothing until fusion completes, then yields the fused
            depthwise ``weight`` and ``bias``.
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

        m = re.match(r".*network\.(\d+)\.(\d+)\.convffn\.conv\.(conv\.weight|bn\.(weight|bias|running_mean|running_var|num_batches_tracked))$", name)
        if m:
            ni, bi = m.group(1), m.group(2)
            key = f"{ni}.{bi}"
            buf = self._ffn_bn_buf.setdefault(key, {})
            part = m.group(3)
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
            # Ignore num_batches_tracked; it is not needed for inference.
            needed = {"w", "bn_w", "bn_b", "mean", "var"}
            if needed.issubset(buf):
                fw, fb = _fuse_conv_bn(buf["w"], buf["bn_w"], buf["bn_b"], buf["mean"], buf["var"])
                yield from self._emit(f"v.fastvit.net.{ni}.blk.{bi}.ffn.dw.weight", fw)
                yield from self._emit(f"v.fastvit.net.{ni}.blk.{bi}.ffn.dw.bias", fb)
                del self._ffn_bn_buf[key]
            return

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[Tuple[str, torch.Tensor]]:
        """Route each checkpoint tensor to the appropriate GGUF mapper.

        Args:
            data_torch (torch.Tensor): Source tensor from the checkpoint.
            name (str): Checkpoint tensor name.
            bid (int | None): Block index from the base converter. Unused for Jingyu vision export.

        Yields:
            Tuple[str, torch.Tensor]: Mapped ``(gguf_name, tensor)`` pairs for projector / vision tensors that are kept. Yields nothing for language
            tensors, unused classifier heads, or unmatched names (unmatched vision names are logged as warnings).
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
