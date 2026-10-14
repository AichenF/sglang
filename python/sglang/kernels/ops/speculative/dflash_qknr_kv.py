"""DFlash draft attention prologue in one kernel.

Replaces, per draft layer, the chain  fused_qknorm (q, k per-head RMSNorm)  ->  rotary (NeoX, full or
partial rotary_dim)  ->  ``v * v_scale``  ->  store_kvcache(k, v)  with a single launch on the fused
qkv_proj output:

* q heads: RMSNorm -> RoPE, written back in place (the attention kernel reads q from the qkv rows);
* k heads: RMSNorm -> RoPE, written to the paged KV cache (and back in place);
* v heads: optional value scale, written to the paged KV cache.

Numerics follow the unfused path's rounding points (normalized values rounded to the activation dtype
before the rotation, the rotation and the value scale in fp32 and rounded once). Draft-only: the draft's
outputs only steer acceptance, the target verify stays bitwise.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_qknr_kv_kernel(
    QKV,
    QW,
    KW,
    CS,
    POS,
    LOC,
    KC,
    VC,
    row_stride,
    cs_stride,
    kc_stride,
    vc_stride,
    eps,
    v_scale,
    NHQ: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    HAS_VSCALE: tl.constexpr,
    SKIP_INDEX: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = ROT // 2
    d_ar = tl.arange(0, D)
    half_ar = tl.arange(0, HALF)
    pos = tl.load(POS + t).to(tl.int64)
    base = QKV + t * row_stride
    if h < NHQ + NKV:
        is_q = h < NHQ
        if is_q:
            col0 = h * D
            w_ptr = QW
        else:
            col0 = NHQ * D + (h - NHQ) * D
            w_ptr = KW
        row = base + col0
        x = tl.load(row + d_ar).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
        xn = (x * inv * tl.load(w_ptr + d_ar).to(tl.float32)).to(QKV.dtype.element_ty)
        x1 = (tl.load(row + half_ar).to(tl.float32) * inv * tl.load(w_ptr + half_ar).to(tl.float32))
        x2 = (
            tl.load(row + HALF + half_ar).to(tl.float32)
            * inv
            * tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        )
        x1 = x1.to(QKV.dtype.element_ty).to(tl.float32)
        x2 = x2.to(QKV.dtype.element_ty).to(tl.float32)
        c = tl.load(CS + pos * cs_stride + half_ar).to(tl.float32)
        s = tl.load(CS + pos * cs_stride + HALF + half_ar).to(tl.float32)
        o1 = (x1 * c - x2 * s).to(QKV.dtype.element_ty)
        o2 = (x2 * c + x1 * s).to(QKV.dtype.element_ty)
        pass_mask = d_ar >= ROT
        if is_q:
            tl.store(row + half_ar, o1)
            tl.store(row + HALF + half_ar, o2)
            tl.store(row + d_ar, xn, mask=pass_mask)
        else:
            tl.store(row + half_ar, o1)
            tl.store(row + HALF + half_ar, o2)
            tl.store(row + d_ar, xn, mask=pass_mask)
            slot = tl.load(LOC + t).to(tl.int64)
            if slot != SKIP_INDEX:
                kdst = KC + slot * kc_stride + (h - NHQ) * D
                tl.store(kdst + half_ar, o1)
                tl.store(kdst + HALF + half_ar, o2)
                tl.store(kdst + d_ar, xn, mask=pass_mask)
    else:
        j = h - NHQ - NKV
        vrow = base + (NHQ + NKV) * D + j * D
        v = tl.load(vrow + d_ar)
        if HAS_VSCALE:
            v = (v.to(tl.float32) * v_scale).to(QKV.dtype.element_ty)
        slot = tl.load(LOC + t).to(tl.int64)
        if slot != SKIP_INDEX:
            tl.store(VC + slot * vc_stride + j * D + d_ar, v)


def dflash_qk_norm_rope_kv_store(
    qkv: torch.Tensor,  # [T, (NHQ + 2 * NKV) * D], rows contiguous
    positions: torch.Tensor,  # [T]
    q_weight: torch.Tensor,  # [D]
    k_weight: torch.Tensor,  # [D]
    cos_sin_cache: torch.Tensor,  # [max_pos, rot] (cos | sin halves)
    out_loc: torch.Tensor,  # [T] cache slots
    k_cache: torch.Tensor,  # [slots, NKV, D]
    v_cache: torch.Tensor,  # [slots, NKV, D]
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    v_scale: Optional[float],
    skip_index: int = 0,
) -> None:
    T = int(qkv.shape[0])
    if T == 0:
        return
    grid = (T, num_q_heads + 2 * num_kv_heads)
    _dflash_qknr_kv_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        out_loc,
        k_cache,
        v_cache,
        qkv.stride(0),
        cos_sin_cache.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        float(eps),
        float(v_scale) if v_scale is not None else 1.0,
        NHQ=num_q_heads,
        NKV=num_kv_heads,
        D=head_dim,
        ROT=rotary_dim,
        HAS_VSCALE=v_scale is not None,
        SKIP_INDEX=skip_index,
        num_warps=1,
    )


def can_use_dflash_qknr_kv(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_loc: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    if not qkv.is_cuda or qkv.dtype not in (torch.bfloat16, torch.float16) or qkv.dim() != 2:
        return False
    T = int(qkv.shape[0])
    if qkv.stride(1) != 1 or qkv.shape[1] != (num_q_heads + 2 * num_kv_heads) * head_dim:
        return False
    if head_dim & (head_dim - 1) or rotary_dim <= 0 or rotary_dim > head_dim:
        return False
    if (rotary_dim // 2) & (rotary_dim // 2 - 1) or rotary_dim % 2:
        return False
    if out_loc is None or out_loc.dim() != 1 or int(out_loc.shape[0]) != T:
        return False
    if positions.dim() != 1 or int(positions.shape[0]) != T:
        return False
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] != rotary_dim:
        return False
    if cos_sin_cache.stride(1) != 1 or cos_sin_cache.device != qkv.device:
        return False
    for cache in (k_cache, v_cache):
        if cache.dtype != qkv.dtype or cache.dim() != 3:
            return False
        if tuple(cache.shape[1:]) != (num_kv_heads, head_dim):
            return False
        if cache.stride(2) != 1 or cache.stride(1) != head_dim:
            return False
    return True
