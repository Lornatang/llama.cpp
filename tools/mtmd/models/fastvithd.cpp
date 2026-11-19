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
//
//  fastvithd.cpp
//  llama.cpp / mtmd
//
//  Loads Marvin Muse FastViT-HD weights and builds the ggml graph from RGB input
//  to projected image tokens.
//

#include "models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

void load_fastvithd(mtmd_model & model, const std::function<ggml_tensor *(const std::string &, bool)> & get_tensor) {
    auto & weights = model.fastvit;
    auto check_shape = [](const ggml_tensor * t, const std::array<int64_t, 4> & shape) {
        if (t && !std::equal(shape.begin(), shape.end(), t->ne)) {
            throw std::runtime_error(mtmd_format("marvin_muse: unexpected shape for %s; expected [%lld, %lld, %lld, %lld]",
                t->name, (long long) shape[0], (long long) shape[1], (long long) shape[2], (long long) shape[3]));
        }
    };
    auto get = [&](const std::string & name, const std::array<int64_t, 4> & shape, bool required = true) {
        auto * t = get_tensor(name, required);
        check_shape(t, shape);
        return t;
    };
    auto load_conv = [&](const std::string & name, int kernel, int64_t in, int64_t out = 0) {
        auto * w = get_tensor(name + ".weight", true);
        const int64_t channels = out > 0 ? out : w->ne[3];
        check_shape(w, {kernel, kernel, in, channels});
        if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) {
            throw std::runtime_error("marvin_muse: convolution requires F16 or F32 weights: " + name);
        }
        return fastvit_conv2d{w, get(name + ".bias", {channels, 1, 1, 1}, false)};
    };
    auto load_scale = [&](const std::string & name, int64_t channels) {
        auto * t = get_tensor(name + ".weight", true);
        if (ggml_nelements(t) != channels) {
            throw std::runtime_error("marvin_muse: incompatible channel count for " + name + ".weight");
        }
        return t;
    };

    weights.stem[0] = load_conv("v.fastvit.stem.0", 3, 3);
    int64_t c = weights.stem[0].w->ne[3];
    weights.stem[1] = load_conv("v.fastvit.stem.1", 3, 1, c);
    weights.stem[2] = load_conv("v.fastvit.stem.2", 1, c, c);

    // These indices refer to modules in FastViT-HD's original network sequence.
    const int stage_net[] = {0, 2, 4, 7, 10};
    const int stage_blocks[] = {2, 12, 24, 4, 2};
    const int down_net[] = {1, 3, 5, 8};
    const int pos_net[] = {6, 9};
    for (int i = 0; i < 5; ++i) {
        const auto prefix = mtmd_format("v.fastvit.net.%d.blk.", stage_net[i]);
        auto load_ffn = [&](auto & block, const std::string & base) {
            block.ffn_dw  = load_conv(base + ".ffn.dw", 7, 1, c);
            block.ffn_fc1 = load_conv(base + ".ffn.fc1", 1, c, 4 * c);
            block.ffn_fc2 = load_conv(base + ".ffn.fc2", 1, 4 * c, c);
        };
        if (i < 3) {
            auto & blocks = weights.repmixer[i];
            blocks.resize(stage_blocks[i]);
            for (int j = 0; j < stage_blocks[i]; ++j) {
                auto & block = blocks[j];
                const auto base = prefix + std::to_string(j);
                block.mixer = load_conv(base + ".mixer", 3, 1, c);
                block.ls = load_scale(base + ".ls", c);
                load_ffn(block, base);
            }
        } else {
            if (c % 32 != 0) {
                throw std::runtime_error("marvin_muse: attention requires channels divisible by 32");
            }
            weights.pos[i - 3] = load_conv(mtmd_format("v.fastvit.net.%d.pos", pos_net[i - 3]), 7, 1, c);
            auto & blocks = weights.attention[i - 3];
            blocks.resize(stage_blocks[i]);
            for (int j = 0; j < stage_blocks[i]; ++j) {
                auto & block = blocks[j];
                const auto base = prefix + std::to_string(j);
                block.norm_w = get(base + ".norm.weight", {c, 1, 1, 1});
                block.norm_b = get(base + ".norm.bias",   {c, 1, 1, 1});
                block.qkv_w  = get(base + ".qkv.weight",  {c, 3 * c, 1, 1});
                block.qkv_b  = get(base + ".qkv.bias",    {3 * c, 1, 1, 1}, false);
                block.proj_w = get(base + ".proj.weight", {c, c, 1, 1});
                block.proj_b = get(base + ".proj.bias",   {c, 1, 1, 1}, false);
                block.ls1 = load_scale(base + ".ls1", c);
                block.ls2 = load_scale(base + ".ls2", c);
                load_ffn(block, base);
            }
        }
        if (i < 4) {
            const auto base = mtmd_format("v.fastvit.net.%d.down.", down_net[i]);
            weights.down[i][0] = load_conv(base + "0", 7, 1);
            const int64_t depthwise_channels = weights.down[i][0].w->ne[3];
            if (depthwise_channels % c != 0) {
                throw std::runtime_error("marvin_muse: incompatible depthwise channel count for " + base + "0.weight");
            }
            weights.down[i][1] = load_conv(base + "1", 1, depthwise_channels);
            c = weights.down[i][1].w->ne[3];
        }
    }

    const int64_t vision_dim = 2 * c;
    weights.conv_exp = load_conv("v.fastvit.conv_exp", 3, 1, vision_dim);
    weights.se_reduce = load_conv("v.fastvit.conv_exp.se.reduce", 1, vision_dim);
    weights.se_expand = load_conv("v.fastvit.conv_exp.se.expand", 1, weights.se_reduce.w->ne[3], vision_dim);
    if (get_tensor("mm.4.weight", false) || get_tensor("mm.4.bias", false)) {
        throw std::runtime_error("marvin_muse: expected a two-layer multimodal projector");
    }
    const int64_t text_dim = model.hparams.projection_dim;
    model.mm_0_w = get("mm.0.weight", {vision_dim, text_dim, 1, 1});
    model.mm_0_b = get("mm.0.bias",   {text_dim, 1, 1, 1}, false);
    model.mm_2_w = get("mm.2.weight", {text_dim, text_dim, 1, 1});
    model.mm_2_b = get("mm.2.bias",   {text_dim, 1, 1, 1}, false);
}

/// Apply the exact GELU used by the PyTorch checkpoint.
///
/// @param x  Input activations.
/// @return A tensor with the same shape as x.
ggml_tensor * mtmd_graph_fastvithd::gelu_act(ggml_tensor * x) {
    return ggml_gelu_erf(ctx0, x);
}

/// Build a convolution with an optional bias.
///
/// @param c          Kernel and optional per-channel bias.
/// @param x          Input tensor in ggml dimension order [width, height, channels, batch].
/// @param stride     Stride on both spatial axes.
/// @param pad        Zero padding on each side of both spatial axes.
/// @param depthwise  Whether to apply the kernels independently to each input channel.
/// @return Convolved activations in the same dimension order as x.
ggml_tensor * mtmd_graph_fastvithd::conv2d(const fastvit_conv2d & c,
                                           ggml_tensor *          x,
                                           int                    stride,
                                           int                    pad,
                                           bool                   depthwise) {
    GGML_ASSERT(c.w);
    ggml_tensor * cur;
    if (depthwise) {
        // ggml depthwise convolution pairs each input channel with one kernel.
        // Repeat each input channel when a FastViT convolution has multiple kernels per group.
        const int64_t Cin  = x->ne[2];
        const int64_t Cout = c.w->ne[3];
        ggml_tensor * xin  = x;
        if (Cout != Cin) {
            GGML_ASSERT(Cout % Cin == 0);
            const int64_t r = Cout / Cin;
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t B = x->ne[3];
            xin             = ggml_reshape_4d(ctx0, x, W, H, 1, Cin * B);
            xin             = ggml_repeat_4d(ctx0, xin, W, H, r, Cin * B);
            xin             = ggml_reshape_4d(ctx0, xin, W, H, Cout, B);
        }
        // Use direct convolution where supported to avoid an intermediate im2col buffer.
        const bool direct_type = (c.w->type == GGML_TYPE_F16 || c.w->type == GGML_TYPE_F32) && xin->type == GGML_TYPE_F32;
        const bool direct_layout = ggml_is_contiguous(xin) || ggml_is_contiguous_channels(xin);
        ggml_tensor * direct = direct_type && direct_layout
            ? ggml_conv_2d_dw_direct(ctx0, c.w, xin, stride, stride, pad, pad, 1, 1)
            : nullptr;
        cur = direct && ggml_backend_supports_op(backend, direct)
            ? direct
            : ggml_conv_2d_dw(ctx0, c.w, xin, stride, stride, pad, pad, 1, 1);
    } else {
        cur = ggml_conv_2d_direct(ctx0, c.w, x, stride, stride, pad, pad, 1, 1);
    }
    if (c.b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, c.b, 1, 1, cur->ne[2], 1));
    }
    return cur;
}

/// Apply one layer-scale value per channel across space and batch.
///
/// @param x   Image activations in [width, height, channels, batch] order.
/// @param ls  One scale per channel, or nullptr to leave x unchanged.
/// @return Scaled activations.
ggml_tensor * mtmd_graph_fastvithd::layer_scale(ggml_tensor * x, ggml_tensor * ls) {
    if (!ls) {
        return x;
    }
    const int64_t C     = x->ne[2];
    ggml_tensor * scale = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ls), 1, 1, C, 1);
    if (scale->type != x->type) {
        scale = ggml_cast(ctx0, scale, x->type);
    }
    return ggml_mul(ctx0, x, scale);
}

/// Build the convolutional feed-forward network shared by both block types.
///
/// @param x    Input activations.
/// @param dw   Depthwise convolution with batch normalization folded into its weights and bias.
/// @param fc1  Pointwise convolution that expands the channel dimension.
/// @param fc2  Pointwise convolution that restores the original channel count after GELU.
/// @return Feed-forward activations before the layer scale and residual addition.
ggml_tensor * mtmd_graph_fastvithd::convffn(ggml_tensor *          x,
                                            const fastvit_conv2d & dw,
                                            const fastvit_conv2d & fc1,
                                            const fastvit_conv2d & fc2) {
    ggml_tensor * cur = conv2d(dw, x, 1, 3, true);
    cur               = conv2d(fc1, cur, 1, 0, false);
    cur               = gelu_act(cur);
    cur               = conv2d(fc2, cur, 1, 0, false);
    return cur;
}

/// Build a RepMixer block.
///
/// @param x    Input activations.
/// @param blk  Fused mixer and feed-forward weights.
/// @return Mixer output plus the scaled feed-forward output.
ggml_tensor * mtmd_graph_fastvithd::repmixer_block(ggml_tensor * x, const fastvit_repmixer_block & blk) {
    // The reparameterized mixer weights already contain its residual and layer scale.
    ggml_tensor * cur = conv2d(blk.mixer, x, 1, 1, true);
    ggml_tensor * ffn = convffn(cur, blk.ffn_dw, blk.ffn_fc1, blk.ffn_fc2);
    ffn               = layer_scale(ffn, blk.ls);
    return ggml_add(ctx0, cur, ffn);
}

/// Normalize across channels independently at each spatial location.
///
/// @param x    Input activations in [width, height, channels, batch] order.
/// @param w    Optional scale per channel.
/// @param b    Optional bias per channel.
/// @param eps  Epsilon added to the variance.
/// @return Normalized activations in the same dimension order as x.
ggml_tensor * mtmd_graph_fastvithd::channel_norm(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];
    const int64_t B = x->ne[3];

    // ggml_norm normalizes the first dimension.
    ggml_tensor * cur = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 2, 0, 3));
    cur               = ggml_reshape_3d(ctx0, cur, C, W * H, B);
    cur               = ggml_norm(ctx0, cur, eps);
    if (w) {
        cur = ggml_mul(ctx0, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx0, cur, b);
    }
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_4d(ctx0, cur, C, W, H, B);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    return cur;
}

/// Build multi-head self-attention over the spatial positions of one image.
///
/// @param x    Normalized activations with batch size one and 32 channels per attention head.
/// @param blk  Packed QKV and output projection weights.
/// @return Projected attention output on the input spatial grid.
ggml_tensor * mtmd_graph_fastvithd::mhsa(ggml_tensor * x, const fastvit_attn_block & blk) {
    const int64_t W        = x->ne[0];
    const int64_t H        = x->ne[1];
    const int64_t C        = x->ne[2];
    const int64_t B        = x->ne[3];
    const int64_t N        = W * H;
    const int     head_dim = 32;
    GGML_ASSERT(B == 1);
    GGML_ASSERT(C % head_dim == 0);
    const int   n_head = (int) (C / head_dim);
    const float scale  = 1.0f / sqrtf((float) head_dim);

    // [W,H,C,B] -> [C,N]
    ggml_tensor * cur = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 2, 0, 3));
    cur               = ggml_reshape_2d(ctx0, cur, C, N);

    ggml_tensor * qkv = build_mm(blk.qkv_w, cur);
    if (blk.qkv_b) {
        qkv = ggml_add(ctx0, qkv, blk.qkv_b);
    }
    // qkv: [3C, N] -> three [C, N] views
    ggml_tensor * q = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], C * qkv->nb[0]);
    ggml_tensor * v = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], 2 * C * qkv->nb[0]);
    q               = ggml_cont(ctx0, q);
    k               = ggml_cont(ctx0, k);
    v               = ggml_cont(ctx0, v);

    // build_attn expects [head_dim, n_heads, n_tokens].
    q = ggml_reshape_3d(ctx0, q, head_dim, n_head, N);
    k = ggml_reshape_3d(ctx0, k, head_dim, n_head, N);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_head, N);

    ggml_tensor * attn = build_attn(nullptr, nullptr, q, k, v, nullptr, scale, -1);
    attn = build_mm(blk.proj_w, attn);
    if (blk.proj_b) {
        attn = ggml_add(ctx0, attn, blk.proj_b);
    }

    attn = ggml_cont(ctx0, attn);
    attn = ggml_reshape_4d(ctx0, attn, C, W, H, B);
    attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 2, 0, 1, 3));
    return attn;
}

/// Build an attention block with separate attention and ConvFFN residual branches.
///
/// @param x    Input activations.
/// @param blk  Normalization, attention, feed-forward and layer-scale weights.
/// @return Activations after both residual additions.
ggml_tensor * mtmd_graph_fastvithd::attn_block(ggml_tensor * x, const fastvit_attn_block & blk) {
    ggml_tensor * h = channel_norm(x, blk.norm_w, blk.norm_b);
    h               = mhsa(h, blk);
    h               = layer_scale(h, blk.ls1);
    x               = ggml_add(ctx0, x, h);

    ggml_tensor * ffn = convffn(x, blk.ffn_dw, blk.ffn_fc1, blk.ffn_fc2);
    ffn               = layer_scale(ffn, blk.ls2);
    return ggml_add(ctx0, x, ffn);
}

/// Reduce spatial resolution between FastViT-HD stages.
///
/// @param x    Input activations.
/// @param lkb  Strided large-kernel depthwise convolution.
/// @param pw   Pointwise convolution that mixes the downsampled channels.
/// @return Activations on the reduced spatial grid.
ggml_tensor * mtmd_graph_fastvithd::downsample(ggml_tensor * x, const fastvit_conv2d & lkb, const fastvit_conv2d & pw) {
    ggml_tensor * cur = conv2d(lkb, x, 2, 3, true);
    cur               = gelu_act(cur);
    cur               = conv2d(pw, cur, 1, 0, false);
    cur               = gelu_act(cur);
    return cur;
}

/// Apply squeeze-and-excitation to the expanded feature map.
///
/// @param x  Expanded activations in [width, height, channels, batch] order.
/// @return Activations reweighted by one gate per channel and image.
ggml_tensor * mtmd_graph_fastvithd::se_block(ggml_tensor * x) {
    // The 1024-pixel checkpoint pools a 16x16 grid. Global pooling also handles other saved grid sizes.
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];

    ggml_tensor * cur = ggml_pool_2d(ctx0, x, GGML_OP_POOL_AVG, W, H, W, H, 0, 0);
    cur               = conv2d(model.fastvit.se_reduce, cur, 1, 0, false);
    cur               = ggml_relu(ctx0, cur);
    cur               = conv2d(model.fastvit.se_expand, cur, 1, 0, false);
    cur               = ggml_sigmoid(ctx0, cur);
    return ggml_mul(ctx0, x, cur);
}

/// Build the FastViT-HD encoder and the two-layer projector.
///
/// @return Graph producing projected image tokens in [text width, tokens, batch] order.
ggml_cgraph * mtmd_graph_fastvithd::build() {
    ggml_tensor * cur = build_inp_raw();

    cur = gelu_act(conv2d(model.fastvit.stem[0], cur, 2, 1, false));
    cur = gelu_act(conv2d(model.fastvit.stem[1], cur, 2, 1, true));
    cur = gelu_act(conv2d(model.fastvit.stem[2], cur, 1, 0, false));
    cb(cur, "fv_stem", -1);

    for (int i = 0; i < 3; ++i) {
        for (const auto & blk : model.fastvit.repmixer[i]) {
            cur = repmixer_block(cur, blk);
        }
        cur = downsample(cur, model.fastvit.down[i][0], model.fastvit.down[i][1]);
    }

    for (int i = 0; i < 2; ++i) {
        cur = conv2d(model.fastvit.pos[i], cur, 1, 3, true);
        for (const auto & blk : model.fastvit.attention[i]) {
            cur = attn_block(cur, blk);
        }
        if (i == 0) {
            cur = downsample(cur, model.fastvit.down[3][0], model.fastvit.down[3][1]);
        }
    }
    cb(cur, "fv_backbone", -1);

    cur = conv2d(model.fastvit.conv_exp, cur, 1, 1, true);
    cur = se_block(cur);
    cur = gelu_act(cur);
    cb(cur, "fv_conv_exp", -1);

    // Flatten the spatial grid in row-major order before projecting the tokens.
    const int64_t W = cur->ne[0];
    const int64_t H = cur->ne[1];
    const int64_t C = cur->ne[2];
    const int64_t B = cur->ne[3];
    GGML_ASSERT(W * H == n_patches);

    ggml_tensor * emb = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));  // [C,W,H,B]
    emb               = ggml_reshape_3d(ctx0, emb, C, W * H, B);

    emb = build_ffn(emb, model.mm_0_w, model.mm_0_b, nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);
    cb(emb, "fv_projector", -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
