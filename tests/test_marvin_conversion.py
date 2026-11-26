# Copyright Larry. All Rights Reserved.
# SPDX-License-Identifier: MIT
# ==============================================================================
"""Exercise Marvin GGUF conversion without downloading model weights.

The synthetic checkpoint has one attention block and one spatial-convolution block. Set ``MARVIN_MUSE_MMPROJ_TEST`` to the path to ``test-mtmd-impl`` to also
load the export and encode images on the CPU.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from conversion import get_model_class
from conversion.base import gguf
from conversion.marvin import MarvinMuseVisionModel

class TestMarvinTextConversion(unittest.TestCase):
    """Check architecture dispatch and keep vision weights out of text exports."""

    def test_dispatch(self):
        for variant, arch in (("Muse", gguf.MODEL_ARCH.QWEN3), ("Spark", gguf.MODEL_ARCH.QWEN3), ("Verse", gguf.MODEL_ARCH.QWEN35)):
            with self.subTest(variant=variant):
                model = get_model_class(f"Marvin{variant}ForConditionalGeneration")
                self.assertEqual(model.model_arch, arch)

        self.assertIs(get_model_class("MarvinMuseForConditionalGeneration", mmproj=True), MarvinMuseVisionModel)

    def test_filters_multimodal_weights(self):
        for variant in ("Muse", "Spark", "Verse"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                architecture = f"Marvin{variant}ForConditionalGeneration"
                config = {"model_type": f"marvin_{variant.lower()}", "architectures": [architecture], "num_hidden_layers": 1}
                (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
                save_file(
                    {
                        "language_model.model.embed_tokens.weight": np.ones((32, 32), dtype=np.float32),
                        "model.vision_tower.vision_tower.encoder.patchifier.weight": np.zeros((32, 32), dtype=np.float32),
                        "model.mm_projector.0.weight": np.zeros((32, 32), dtype=np.float32),
                    },
                    str(directory / "model.safetensors"),
                )
                model = get_model_class(architecture)(directory, gguf.LlamaFileType.MOSTLY_F16, directory / "model.gguf")
                self.assertEqual(model.block_count, 1)
                self.assertEqual(list(model.model_tensors), ["model.embed_tokens.weight"])


class TestMarvinMuseConversion(unittest.TestCase):
    """Check conversion and optionally run the exported projector on the CPU."""

    def make_checkpoint(self, directory, *, projector="mlp2x_gelu", override=None, tensor_override=None):
        vision = {
            "model_type": "lfm3_vl_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "conv_kernel_size": 3,
            "in_channels": 3,
            "rope_parameters": {"rope_theta": 1e6},
            "norm_eps": 1e-5,
        }
        config = {
            "architectures": ["MarvinMuseForConditionalGeneration"],
            "model_type": "marvin_muse",
            "hidden_size": 32,
            "mm_hidden_size": 64,
            "mm_projector_type": projector,
            "image_aspect_ratio": "dynamic_resolution",
            "mm_vision_config": vision,
            "mm_vision_projector_config": {
                "model_type": "lfm3_vl",
                "vision_config": vision,
                "text_config": {"hidden_size": 64},
                "projector_hidden_size": 64,
                "projector_hidden_act": "gelu",
                "projector_bias": True,
            },
            **(override or {}),
        }
        processor = {
            "image_processor_type": "MarvinMuseNativeImageProcessor",
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_convert_rgb": True,
            "rescale_factor": 1 / 255,
            "image_mean": [0, 0, 0],
            "image_std": [1, 1, 1],
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "size": {"shortest_edge": 1024, "longest_edge": 16384},
            "resample": 3,
        }
        Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")
        Path(directory, "preprocessor_config.json").write_text(json.dumps(processor), encoding="utf-8")

        encoder = "model.vision_tower.vision_tower.encoder."
        merger = "model.vision_tower.vision_tower.projector."
        shapes = {
            encoder + "patchifier.weight": (32, 1536),
            encoder + "embedding_norm.weight": (32,),
            encoder + "layers.0.self_attn.q_proj.weight": (32, 32),
            encoder + "layers.0.self_attn.k_proj.weight": (16, 32),
            encoder + "layers.0.self_attn.v_proj.weight": (16, 32),
            encoder + "layers.0.self_attn.out_proj.weight": (32, 32),
            encoder + "layers.0.self_attn.q_layernorm.weight": (8,),
            encoder + "layers.0.self_attn.k_layernorm.weight": (8,),
            encoder + "layers.1.conv.in_proj.weight": (96, 32),
            encoder + "layers.1.conv.spatial_conv.weight": (32, 1, 3, 3),
            encoder + "layers.1.conv.out_proj.weight": (32, 32),
            merger + "layer_norm.weight": (32,),
            merger + "layer_norm.bias": (32,),
            merger + "linear_1.weight": (64, 128),
            merger + "linear_1.bias": (64,),
            merger + "linear_2.weight": (64, 64),
            merger + "linear_2.bias": (64,),
        }
        for layer in range(2):
            prefix = encoder + f"layers.{layer}."
            shapes.update({
                prefix + "operator_norm.weight": (32,),
                prefix + "ffn_norm.weight": (32,),
                prefix + "feed_forward.w1.weight": (64, 32),
                prefix + "feed_forward.w2.weight": (32, 64),
                prefix + "feed_forward.w3.weight": (64, 32),
            })
        outer = {"model.mm_projector.": 64} if projector == "linear" else {"model.mm_projector.0.": 64, "model.mm_projector.2.": 32}
        for prefix, width in outer.items():
            shapes[prefix + "weight"] = (32, width)
            shapes[prefix + "bias"] = (32,)
        for name, shape in (tensor_override or {}).items():
            if shape is None:
                del shapes[name]
            else:
                shapes[name] = shape
        rng = np.random.default_rng(128)
        tensors = {key: rng.normal(0, 0.05, shape).astype(np.float32) for key, shape in shapes.items()}
        save_file(tensors, str(Path(directory, "model.safetensors")))

    def convert(self, directory, outtype="f16"):
        output = Path(directory, f"model-{outtype}.gguf")
        file_type = {
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "f32": gguf.LlamaFileType.ALL_F32,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        }[outtype]
        model = MarvinMuseVisionModel(directory, file_type, output)
        model.write()
        return output

    def test_formats_and_runtime(self):
        for projector, outtype, matrix_type in (
            ("mlp2x_gelu", "f16", gguf.GGMLQuantizationType.F16),
            ("mlp2x_gelu", "q8_0", gguf.GGMLQuantizationType.Q8_0),
            ("linear", "f32", gguf.GGMLQuantizationType.F32),
        ):
            with self.subTest(outtype=outtype), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.make_checkpoint(directory, projector=projector)
                output = self.convert(directory, outtype)
                reader = gguf.GGUFReader(output)
                self.assertEqual(reader.fields["clip.vision.projector_type"].contents(), "marvin_muse")
                self.assertEqual(reader.fields["clip.vision.spatial_merge_size"].contents(), 2)
                tensors = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
                for name in ("mm.0.weight", "mm.2.weight", "mm.3.weight", "v.blk.0.attn_q.weight"):
                    self.assertEqual(tensors[name], matrix_type)
                self.assertEqual("mm.4.weight" in tensors, projector == "mlp2x_gelu")
                self.assertEqual(tensors["v.patch_embd.weight"], gguf.GGMLQuantizationType.F16)
                self.assertEqual(tensors["v.patch_embd.weight.1"], gguf.GGMLQuantizationType.F16)
                self.assertEqual(tensors["v.blk.1.conv2d_spatial.weight"], gguf.GGMLQuantizationType.F32)
                for key in ("image_mean", "image_std"):
                    self.assertEqual(reader.fields[f"clip.vision.{key}"].types, [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32])
                if runtime := os.environ.get("MARVIN_MUSE_MMPROJ_TEST"):
                    run = subprocess.run([runtime, "--marvin-muse-mmproj", str(output)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr[-4000:])

    def test_rejects_incompatible_config_and_tensors(self):
        for override, tensor_override, error in (
            ({"vision_config": {"model_type": "fastvithd"}}, {}, "requires native LFM3-VL"),
            ({"image_aspect_ratio": "pad"}, {}, "requires dynamic resolution"),
            ({}, {"model.mm_projector.2.bias": None}, "Missing native Marvin Muse"),
            ({}, {"model.mm_projector.2.weight": (32, 33)}, "Unexpected native Marvin Muse"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.make_checkpoint(directory, override=override, tensor_override=tensor_override)
                with self.assertRaisesRegex(ValueError, error):
                    self.convert(directory)
                self.assertFalse(Path(directory, "model-f16.gguf").exists())


if __name__ == "__main__":
    unittest.main()
