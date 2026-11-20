// Copyright Larry. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct mtmd_graph;
struct mtmd_image_f32;
struct mtmd_model;
struct mtmd_subcontext;

/// Kernel weights and optional bias for a convolution.
struct fastvit_conv2d {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

/// Weights for a RepMixer block and its convolutional feed-forward network.
struct fastvit_repmixer_block {
    fastvit_conv2d mixer;
    ggml_tensor *  ls = nullptr;
    fastvit_conv2d ffn_dw;
    fastvit_conv2d ffn_fc1;
    fastvit_conv2d ffn_fc2;
};

/// Weights for an attention block with separate layer scales for attention and ConvFFN.
struct fastvit_attn_block {
    ggml_tensor *  norm_w = nullptr;
    ggml_tensor *  norm_b = nullptr;
    ggml_tensor *  qkv_w  = nullptr;
    ggml_tensor *  qkv_b  = nullptr;
    ggml_tensor *  proj_w = nullptr;
    ggml_tensor *  proj_b = nullptr;
    ggml_tensor *  ls1    = nullptr;
    ggml_tensor *  ls2    = nullptr;
    fastvit_conv2d ffn_dw;
    fastvit_conv2d ffn_fc1;
    fastvit_conv2d ffn_fc2;
};

/// FastViT-HD weights grouped by stem, stages, transitions, and channel expansion.
struct fastvit_weights {
    fastvit_conv2d stem[3];
    std::vector<fastvit_repmixer_block> repmixer[3];
    std::vector<fastvit_attn_block> attention[2];
    /// Strided depthwise and pointwise convolutions for each stage transition.
    fastvit_conv2d down[4][2];

    /// Positional convolutions applied before the two attention stages.
    fastvit_conv2d pos[2];
    fastvit_conv2d conv_exp;
    fastvit_conv2d se_reduce;
    fastvit_conv2d se_expand;
};

/// Load and validate FastViT-HD and the multimodal projector before graph construction.
///
/// @param model       Destination model whose vision metadata has already been loaded.
/// @param get_tensor  Looks up tensors by GGUF name. Pass false for the second argument to allow a missing tensor.
/// @throws std::runtime_error If a required tensor is missing or the weights are incompatible.
void load_fastvithd(mtmd_model & model, const std::function<ggml_tensor *(const std::string &, bool)> & get_tensor);

std::unique_ptr<mtmd_graph> make_fastvithd_graph(mtmd_subcontext * ctx, const mtmd_image_f32 & img);
