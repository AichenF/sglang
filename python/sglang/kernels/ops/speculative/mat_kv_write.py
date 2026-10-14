"""The DFlash draft-KV materialization's k-RMSNorm + RoPE (+ v scale) AND the KV-cache write of every draft
layer in ONE launch (SGLANG_DFLASH_MAT_FUSED_KV_WRITE).

Replaces _fused_norm_rope_kernel_stacked (-> stacked k_out / v_out) + set_kv_buffer_prefix_valid_multi_layer (byte copy of
every committed row into each layer's cache) on the materialize-in-verify side chain, whose end gates the verify graph.
Same arithmetic as _fused_norm_rope_kernel_stacked (statement for statement), stores straight into the cache rows ->
bitwise identical to the two-kernel path for a full commit (every block row; the MAT path commits all rows).
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_kv_write_kernel(
    kv_ptr,  # [total_ctx, n_layers, kv_size * 2]
    k_norm_weight_ptr,  # [n_layers, head_dim]
    eps_ptr,  # [n_layers]
    v_scale_ptr,  # [n_layers] fp32 (read only when HAS_VSCALE)
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    loc_ptr,  # [total_ctx] int64 cache rows
    dst_k_table,  # [n_layers] int64 cache base addresses
    dst_v_table,
    dst_row_stride,  # elements per cache row
    kv_stride_ctx,
    kv_stride_layer,
    k_norm_weight_stride_layer,
    cos_sin_stride_pos,
    total_ctx,
    n_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    HAS_VSCALE: tl.constexpr = False,
):
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    layer_id = tl.program_id(2)
    if ctx_id >= total_ctx or layer_id >= n_layers:
        return

    position = tl.load(positions_ptr + ctx_id)
    eps = tl.load(eps_ptr + layer_id).to(tl.float32)
    kv_base = kv_ptr + ctx_id * kv_stride_ctx + layer_id * kv_stride_layer
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    loc = tl.load(loc_ptr + ctx_id)
    dst_k = tl.load(dst_k_table + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
    dst_v = tl.load(dst_v_table + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
    k_write = dst_k + loc * dst_row_stride + head_id * head_dim
    v_write = dst_v + loc * dst_row_stride + head_id * head_dim

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0).to(tl.float32)

    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(k_base + half_rotary_dim + offs, mask=mask_half, other=0.0).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + half_rotary_dim + offs,
        mask=mask_half,
        other=1.0,
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    if HAS_VSCALE:
        v_scale = tl.load(v_scale_ptr + layer_id)
        tl.store(v_write + offs, (v_raw.to(tl.float32) * v_scale).to(v_raw.dtype), mask=mask_hd)
    else:
        tl.store(v_write + offs, v_raw, mask=mask_hd)
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half)
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def norm_rope_kv_write(
    kv: torch.Tensor,  # [total_ctx, n_layers, kv_size*2]
    k_norm_weight: torch.Tensor,
    eps: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,  # [total_ctx] int64
    loc: torch.Tensor,  # [total_ctx] int64
    k_table: torch.Tensor,
    v_table: torch.Tensor,
    row_stride: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    total_ctx, n_layers, kv_dim = kv.shape
    if total_ctx == 0:
        return
    kv_size = num_kv_heads * head_dim
    assert kv_dim == kv_size * 2 and kv.stride(2) == 1
    assert k_norm_weight.shape == (n_layers, head_dim) and eps.shape == (n_layers,)
    assert positions.dtype == torch.int64 and loc.dtype == torch.int64 and loc.numel() == total_ctx
    _norm_rope_kv_write_kernel[(total_ctx, num_kv_heads, n_layers)](
        kv,
        k_norm_weight,
        eps,
        v_scale if v_scale is not None else eps,
        cos_sin_cache,
        positions,
        loc,
        k_table,
        v_table,
        row_stride,
        kv.stride(0),
        kv.stride(1),
        k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        total_ctx,
        n_layers,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        rotary_dim // 2,
        triton.next_power_of_2(head_dim),
        HAS_VSCALE=v_scale is not None,
    )
