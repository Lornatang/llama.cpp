#include "models.h"

#include <cmath>

// Jingyu FastViT-HD graph (inference-mode fused MobileOne / RepMixer / MHSA).
// Layout throughout: ggml NHWC-like vision tensors are [W, H, C, B].

ggml_tensor * clip_graph_fastvithd::gelu_act(ggml_tensor * x) {
    return ggml_gelu(ctx0, x);
}

ggml_tensor * clip_graph_fastvithd::conv2d(
        const fastvit_conv2d & c,
        ggml_tensor * x,
        int stride,
        int pad,
        bool depthwise) {
    GGML_ASSERT(c.w);
    ggml_tensor * cur;
    if (depthwise) {
        // FastViT LKB / conv_exp use groups=Cin with Cout = r*Cin (r>=1).
        // ggml_conv_2d_dw is 1:1 per channel, so repeat each input channel r times.
        const int64_t Cin  = x->ne[2];
        const int64_t Cout = c.w->ne[3];
        ggml_tensor * xin = x;
        if (Cout != Cin) {
            GGML_ASSERT(Cout % Cin == 0);
            const int64_t r = Cout / Cin;
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t B = x->ne[3];
            xin = ggml_reshape_4d(ctx0, x, W, H, 1, Cin * B);
            xin = ggml_repeat_4d(ctx0, xin, W, H, r, Cin * B);
            xin = ggml_reshape_4d(ctx0, xin, W, H, Cout, B);
        }
        cur = ggml_conv_2d_dw(ctx0, c.w, xin, stride, stride, pad, pad, 1, 1);
    } else {
        cur = ggml_conv_2d_direct(ctx0, c.w, x, stride, stride, pad, pad, 1, 1);
    }
    if (c.b) {
        // Bias is stored as [OC]; broadcast over [W,H,C,B].
        cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, c.b, 1, 1, cur->ne[2], 1));
    }
    return cur;
}

ggml_tensor * clip_graph_fastvithd::layer_scale(ggml_tensor * x, ggml_tensor * ls) {
    if (!ls) {
        return x;
    }
    // Stored as PyTorch [C,1,1] -> ggml ne roughly [1,1,C]; reshape to broadcast on C.
    const int64_t C = x->ne[2];
    ggml_tensor * scale = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ls), 1, 1, C, 1);
    if (scale->type != x->type) {
        scale = ggml_cast(ctx0, scale, x->type);
    }
    return ggml_mul(ctx0, x, scale);
}

ggml_tensor * clip_graph_fastvithd::convffn(
        ggml_tensor * x,
        const fastvit_conv2d & dw,
        const fastvit_conv2d & fc1,
        const fastvit_conv2d & fc2) {
    // DW 7x7 pad3 (BN fused) -> pw fc1 -> GELU -> pw fc2
    ggml_tensor * cur = conv2d(dw, x, 1, 3, true);
    cur = conv2d(fc1, cur, 1, 0, false);
    cur = gelu_act(cur);
    cur = conv2d(fc2, cur, 1, 0, false);
    return cur;
}

ggml_tensor * clip_graph_fastvithd::repmixer_block(ggml_tensor * x, const fastvit_repmixer_block & blk) {
    // token_mixer is fused DW conv (already includes residual/scale in reparam weights)
    ggml_tensor * cur = conv2d(blk.mixer, x, 1, 1, true);
    ggml_tensor * ffn = convffn(cur, blk.ffn_dw, blk.ffn_fc1, blk.ffn_fc2);
    ffn = layer_scale(ffn, blk.ls);
    return ggml_add(ctx0, cur, ffn);
}

ggml_tensor * clip_graph_fastvithd::channel_norm(
        ggml_tensor * x,
        ggml_tensor * w,
        ggml_tensor * b,
        float eps) {
    // LayerNorm over channel dim for each spatial location.
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];
    const int64_t B = x->ne[3];

    // ggml_permute(ax0..): places old dim i at position axi.
    // [W,H,C,B] -> [C,W,H,B] via permute(1, 2, 0, 3).
    ggml_tensor * cur = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 2, 0, 3));
    cur = ggml_reshape_3d(ctx0, cur, C, W * H, B);
    cur = ggml_norm(ctx0, cur, eps);
    if (w) {
        cur = ggml_mul(ctx0, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx0, cur, b);
    }
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_4d(ctx0, cur, C, W, H, B);
    // [C,W,H,B] -> [W,H,C,B]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    return cur;
}

ggml_tensor * clip_graph_fastvithd::mhsa(ggml_tensor * x, const fastvit_attn_block & blk) {
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];
    const int64_t B = x->ne[3];
    const int64_t N = W * H;
    const int head_dim = 32;
    GGML_ASSERT(B == 1);
    GGML_ASSERT(C % head_dim == 0);
    const int n_head = (int) (C / head_dim);
    const float scale = 1.0f / sqrtf((float) head_dim);

    // [W,H,C,B] -> [C,N]
    ggml_tensor * cur = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 2, 0, 3));
    cur = ggml_reshape_2d(ctx0, cur, C, N);

    ggml_tensor * qkv = build_mm(blk.qkv_w, cur);
    if (blk.qkv_b) {
        qkv = ggml_add(ctx0, qkv, blk.qkv_b);
    }
    // qkv: [3C, N] -> three [C, N] views
    ggml_tensor * q = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], C * qkv->nb[0]);
    ggml_tensor * v = ggml_view_2d(ctx0, qkv, C, N, qkv->nb[1], 2 * C * qkv->nb[0]);
    q = ggml_cont(ctx0, q);
    k = ggml_cont(ctx0, k);
    v = ggml_cont(ctx0, v);

    // Match clip_graph::build_attn / llava layout: [d_head, n_head, n_pos]
    q = ggml_reshape_3d(ctx0, q, head_dim, n_head, N);
    k = ggml_reshape_3d(ctx0, k, head_dim, n_head, N);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_head, N);

    ggml_tensor * attn = build_attn(nullptr, nullptr, q, k, v, nullptr, scale, -1);
    // attn: [C, N]

    attn = build_mm(blk.proj_w, attn);
    if (blk.proj_b) {
        attn = ggml_add(ctx0, attn, blk.proj_b);
    }

    attn = ggml_cont(ctx0, attn);
    attn = ggml_reshape_4d(ctx0, attn, C, W, H, B);
    // [C,W,H,B] -> [W,H,C,B]
    attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 2, 0, 1, 3));
    return attn;
}

ggml_tensor * clip_graph_fastvithd::attn_block(ggml_tensor * x, const fastvit_attn_block & blk) {
    ggml_tensor * h = channel_norm(x, blk.norm_w, blk.norm_b);
    h = mhsa(h, blk);
    h = layer_scale(h, blk.ls1);
    x = ggml_add(ctx0, x, h);

    ggml_tensor * ffn = convffn(x, blk.ffn_dw, blk.ffn_fc1, blk.ffn_fc2);
    ffn = layer_scale(ffn, blk.ls2);
    return ggml_add(ctx0, x, ffn);
}

ggml_tensor * clip_graph_fastvithd::downsample(
        ggml_tensor * x,
        const fastvit_conv2d & lkb,
        const fastvit_conv2d & pw) {
    // ReparamLargeKernelConv: DW 7x7 s2 p3 + GELU, then MobileOne 1x1 + GELU
    ggml_tensor * cur = conv2d(lkb, x, 2, 3, true);
    cur = gelu_act(cur);
    cur = conv2d(pw, cur, 1, 0, false);
    cur = gelu_act(cur);
    return cur;
}

ggml_tensor * clip_graph_fastvithd::se_block(ggml_tensor * x) {
    // Hard-coded avg pool 16x16 as in HF SEBlock (matches 16x16 final map).
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    GGML_ASSERT(W == 16 && H == 16);

    ggml_tensor * cur = ggml_pool_2d(ctx0, x, GGML_OP_POOL_AVG, 16, 16, 16, 16, 0, 0);
    cur = conv2d(model.fv_se_reduce, cur, 1, 0, false);
    cur = ggml_relu(ctx0, cur);
    cur = conv2d(model.fv_se_expand, cur, 1, 0, false);
    cur = ggml_sigmoid(ctx0, cur);
    // cur: [1,1,C,B] * x
    return ggml_mul(ctx0, x, cur);
}

ggml_cgraph * clip_graph_fastvithd::build() {
    ggml_tensor * cur = build_inp_raw(); // [W,H,3,B] = 1024x1024

    // convolutional stem: 3x3 s2, DW 3x3 s2, 1x1
    cur = gelu_act(conv2d(model.fv_stem[0], cur, 2, 1, false));
    cur = gelu_act(conv2d(model.fv_stem[1], cur, 2, 1, true));
    cur = gelu_act(conv2d(model.fv_stem[2], cur, 1, 0, false));
    cb(cur, "fv_stem", -1);

    for (const auto & blk : model.fv_stage0) {
        cur = repmixer_block(cur, blk);
    }
    cur = downsample(cur, model.fv_down[0][0], model.fv_down[0][1]);

    for (const auto & blk : model.fv_stage1) {
        cur = repmixer_block(cur, blk);
    }
    cur = downsample(cur, model.fv_down[1][0], model.fv_down[1][1]);

    for (const auto & blk : model.fv_stage2) {
        cur = repmixer_block(cur, blk);
    }
    cur = downsample(cur, model.fv_down[2][0], model.fv_down[2][1]);

    // RepCPE (fused)
    cur = conv2d(model.fv_pos[0], cur, 1, 3, true);

    for (const auto & blk : model.fv_stage3) {
        cur = attn_block(cur, blk);
    }
    cur = downsample(cur, model.fv_down[3][0], model.fv_down[3][1]);

    cur = conv2d(model.fv_pos[1], cur, 1, 3, true);

    for (const auto & blk : model.fv_stage4) {
        cur = attn_block(cur, blk);
    }
    cb(cur, "fv_backbone", -1);

    // conv_exp: DW 3x3 + SE + GELU, expands to 3072
    cur = conv2d(model.fv_conv_exp, cur, 1, 1, true);
    cur = se_block(cur);
    cur = gelu_act(cur);
    cb(cur, "fv_conv_exp", -1);

    // [W,H,C,B] -> [C, N, B] then MLP projector (LLaVA mlp2x_gelu)
    const int64_t W = cur->ne[0];
    const int64_t H = cur->ne[1];
    const int64_t C = cur->ne[2];
    const int64_t B = cur->ne[3];
    GGML_ASSERT(W * H == n_patches);

    ggml_tensor * emb = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3)); // [C,W,H,B]
    emb = ggml_cont(ctx0, emb);
    emb = ggml_reshape_3d(ctx0, emb, C, W * H, B);

    // mm.0 -> GELU -> mm.2
    GGML_ASSERT(model.mm_0_w && model.mm_2_w);
    emb = build_mm(model.mm_0_w, emb);
    if (model.mm_0_b) {
        emb = ggml_add(ctx0, emb, model.mm_0_b);
    }
    emb = gelu_act(emb);
    emb = build_mm(model.mm_2_w, emb);
    if (model.mm_2_b) {
        emb = ggml_add(ctx0, emb, model.mm_2_b);
    }
    cb(emb, "fv_projector", -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
