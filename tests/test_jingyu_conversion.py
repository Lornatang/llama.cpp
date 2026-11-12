"""Regression tests for Jingyu mmproj conversion using synthetic weights."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
import torch.nn.functional as F

from conversion.jingyu import JingyuVisionModel, _fuse_conv_bn
import gguf


ROOT = Path(__file__).resolve().parents[1]


def fixture_tensors():
    tensors = {}
    prefix = "vision_tower.vision_model.backbone."

    def put(name, shape, value=0.0):
        tensors[prefix + name] = np.full(shape, value, dtype=np.float32)

    def conv(name, in_ch, out_ch, kernel, depthwise=False):
        put(name + ".weight", (out_ch, 1 if depthwise else in_ch, kernel, kernel), 0.001)
        put(name + ".bias", (out_ch,))

    def ffn(net, block, channels):
        name = f"network.{net}.{block}.convffn"
        put(name + ".conv.conv.weight", (channels, 1, 7, 7), 0.001)
        for part, value in (("weight", 1), ("bias", 0), ("running_mean", 0), ("running_var", 1)):
            put(name + ".conv.bn." + part, (channels,), value)
        put(name + ".fc1.weight", (channels * 4, channels, 1, 1), 0.001)
        put(name + ".fc1.bias", (channels * 4,))
        put(name + ".fc2.weight", (channels, channels * 4, 1, 1), 0.001)
        put(name + ".fc2.bias", (channels,))

    conv("patch_embed.0.reparam_conv", 3, 4, 3)
    conv("patch_embed.1.reparam_conv", 4, 4, 3, True)
    conv("patch_embed.2.reparam_conv", 4, 4, 1)
    for net, channels, count in ((0, 4, 2), (2, 8, 12), (4, 16, 24)):
        for block in range(count):
            conv(f"network.{net}.{block}.token_mixer.reparam_conv", channels, channels, 3, True)
            put(f"network.{net}.{block}.layer_scale", (channels, 1, 1), 1)
            ffn(net, block, channels)
    for net, in_ch, out_ch in ((1, 4, 8), (3, 8, 16), (5, 16, 32), (8, 32, 32)):
        conv(f"network.{net}.proj.0.lkb_reparam", in_ch, in_ch, 7, True)
        conv(f"network.{net}.proj.1.reparam_conv", in_ch, out_ch, 1)
    for net in (6, 9):
        conv(f"network.{net}.reparam_conv", 32, 32, 7, True)
    for net, count in ((7, 4), (10, 2)):
        for block in range(count):
            name = f"network.{net}.{block}"
            put(name + ".norm.weight", (32,), 1)
            put(name + ".norm.bias", (32,))
            put(name + ".layer_scale_1", (32, 1, 1), 1)
            put(name + ".layer_scale_2", (32, 1, 1), 1)
            put(name + ".token_mixer.qkv.weight", (96, 32), 0.001)
            put(name + ".token_mixer.proj.weight", (32, 32), 0.001)
            put(name + ".token_mixer.proj.bias", (32,))
            ffn(net, block, 32)
    conv("conv_exp.reparam_conv", 32, 64, 3, True)
    conv("conv_exp.se.reduce", 64, 16, 1)
    conv("conv_exp.se.expand", 16, 64, 1)
    for index, in_ch in ((0, 64), (2, 32)):
        tensors[f"mm_projector.{index}.weight"] = np.full((32, in_ch), 0.001, dtype=np.float32)
        tensors[f"mm_projector.{index}.bias"] = np.zeros((32,), dtype=np.float32)
    return tensors


class TestJingyuConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = fixture_tensors()

    def make_checkpoint(self, directory, *, size=1024, tensors=None):
        config = {
            "architectures": ["JingyuForConditionalGeneration"],
            "model_type": "jingyu", "hidden_size": 32, "mm_hidden_size": 64, "num_hidden_layers": 1,
            "vision_config": {"layers": [2, 12, 24, 4, 2], "embed_dims": [96, 192, 384, 768, 1536],
                              "image_size": size, "patch_size": 64},
        }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "preprocessor_config.json").write_text(json.dumps({"image_mean": [0, 0, 0], "image_std": [1, 1, 1]}), encoding="utf-8")
        save_file(self.tensors if tensors is None else tensors, str(directory / "model.safetensors"))

    def convert(self, directory, outtype="f16"):
        output = directory / f"model-{outtype}.gguf"
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, str(ROOT / "convert_hf_to_gguf.py"), str(directory), "--mmproj",
                                 "--outtype", outtype, "--outfile", str(output)],
                                cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result, output

    def test_formats_and_supported_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_checkpoint(directory, size=512)
            for outtype, matrix_type in (("f16", gguf.GGMLQuantizationType.F16),
                                         ("bf16", gguf.GGMLQuantizationType.BF16),
                                         ("q8_0", gguf.GGMLQuantizationType.Q8_0),
                                         ("f32", gguf.GGMLQuantizationType.F32)):
                with self.subTest(outtype=outtype):
                    result, output = self.convert(directory, outtype)
                    self.assertEqual(result.returncode, 0, result.stdout[-4000:])
                    reader = gguf.GGUFReader(output)
                    tensors = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
                    self.assertEqual(tensors["mm.0.weight"], matrix_type)
                    self.assertEqual(tensors["v.fastvit.net.7.blk.0.qkv.weight"], matrix_type)
                    self.assertEqual(tensors["v.fastvit.stem.0.weight"],
                                     gguf.GGMLQuantizationType.F32 if outtype == "f32" else gguf.GGMLQuantizationType.F16)
                    self.assertEqual(tensors["v.fastvit.net.0.blk.0.ls.weight"], gguf.GGMLQuantizationType.F32)
                    del reader

    def test_incomplete_bn_and_missing_backbone_fail(self):
        missing_bn = dict(self.tensors)
        del missing_bn["vision_tower.vision_model.backbone.network.0.0.convffn.conv.bn.running_var"]
        missing_bias = dict(self.tensors)
        del missing_bias["mm_projector.0.bias"]
        only_projector = {k: v for k, v in self.tensors.items() if k.startswith("mm_projector.")}
        for tensors, error in ((missing_bn, "incomplete ConvFFN batch normalization"),
                               (missing_bias, "required vision/projector tensors"),
                               (only_projector, "required vision/projector tensors")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.make_checkpoint(directory, tensors=tensors)
                result, output = self.convert(directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stdout)
                self.assertFalse(output.exists())

    def test_invalid_resolution_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_checkpoint(directory, size=513)
            result, output = self.convert(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("image_size must be a positive multiple of 64", result.stdout)
            self.assertFalse(output.exists())

    def test_bn_fusion_and_counter_order(self):
        torch.manual_seed(4)
        x = torch.randn(1, 4, 9, 9)
        weight = torch.randn(4, 1, 3, 3)
        gamma, beta, mean = torch.randn(4), torch.randn(4), torch.randn(4)
        var = torch.rand(4) + 0.5
        fused_w, fused_b = _fuse_conv_bn(weight, gamma, beta, mean, var)
        expected = F.batch_norm(F.conv2d(x, weight, groups=4), mean, var, gamma, beta, training=False)
        actual = F.conv2d(x, fused_w, fused_b, groups=4)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

        model = object.__new__(JingyuVisionModel)
        model._ffn_bn_buf = {}
        model._emitted_tensors = set()
        prefix = "vision_tower.vision_model.backbone.network.0.0.convffn.conv."
        parts = (("conv.weight", weight), ("bn.weight", gamma), ("bn.bias", beta),
                 ("bn.running_mean", mean), ("bn.running_var", var), ("bn.num_batches_tracked", torch.tensor(1)))
        for name, data in parts:
            list(model._map_network(prefix + name, data))
        self.assertFalse(model._ffn_bn_buf)
        self.assertIn("v.fastvit.net.0.blk.0.ffn.dw.weight", model._emitted_tensors)


if __name__ == "__main__":
    unittest.main()
