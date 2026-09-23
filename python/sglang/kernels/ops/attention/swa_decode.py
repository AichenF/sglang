"""Sliding-window attention for short verify blocks with a single KV head.

Semantics match ``flash_attn_with_kvcache(..., causal=True,
window_size=(W-1, 0), sinks=...)`` on the same page table and cache_seqlens:
query t of a request at position p attends keys in [p-(W-1), p]. The optional
per-head sink is an extra logit in the softmax denominator that adds no value.

One program per (request, head group) reads the window once through the page
table; every query head in the group reuses that K/V tile.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

# CUDA path, same semantics as the Triton kernel below; other shapes fall
# through. Query heads per program are chosen so W * HP == 16, one mma tile.
_MMA_HP = {8: 2, 6: 2, 4: 4, 2: 8, 1: 16}
_MMA_WINDOWS = (128, 129)


@cache_once
def _mma_module():
    return load_jit(
        "swa_decode_mma",
        cuda_files=["attention/swa_decode_mma.cuh"],
        cuda_wrappers=[("swa_decode_mma", "swa_decode_mma")],
        extra_cuda_cflags=["--use_fast_math"],
    )


# Query heads per program. Smaller groups mean more programs; the gain stops
# once the grid covers the machine.
_HEADS_PER_PROGRAM = 2
_DIM_CHUNK = 64
# Verify blocks are 6-8 tokens; the cap bounds the key tile.
_MAX_VERIFY_TOKENS = 32


@triton.jit
def _swa_decode_attn_kernel(
    Q, K, V, PT, SEQLENS, SINKS, OUT,
    stride_qm, stride_qh, stride_ks, stride_vs, stride_ptb, stride_om, stride_oh,
    sm_scale,
    W: tl.constexpr,          # query rows per request (verify tokens)
    HP: tl.constexpr,         # query heads per program
    WINDOW: tl.constexpr,     # sliding window length
    DQK: tl.constexpr, DV: tl.constexpr,
    DC: tl.constexpr,         # head-dim chunk for the QK dot
    BLOCK_R: tl.constexpr,    # next pow2 >= W * HP
    BLOCK_N: tl.constexpr,    # next pow2 >= WINDOW - 1 + W
    HAS_SINK: tl.constexpr,
):
    b = tl.program_id(0)
    hg = tl.program_id(1)

    seqlen = tl.load(SEQLENS + b)
    # Earliest key the first query row can see.
    base_pos = seqlen - W - (WINDOW - 1)

    rows = tl.arange(0, BLOCK_R)
    rvalid = rows < W * HP
    t = rows // HP
    h = hg * HP + rows % HP
    m_idx = b * W + t

    j = tl.arange(0, BLOCK_N)
    kpos = base_pos + j
    kvalid = (j < WINDOW - 1 + W) & (kpos >= 0)
    slot = tl.load(PT + b * stride_ptb + kpos, mask=kvalid, other=0)

    s = tl.zeros((BLOCK_R, BLOCK_N), dtype=tl.float32)
    for c in tl.static_range(DQK // DC):
        d = c * DC + tl.arange(0, DC)
        q = tl.load(Q + m_idx[:, None] * stride_qm + h[:, None] * stride_qh + d[None, :],
                    mask=rvalid[:, None], other=0.0)
        k = tl.load(K + slot[:, None] * stride_ks + d[None, :],
                    mask=kvalid[:, None], other=0.0)
        s += tl.dot(q, tl.trans(k))
    s = s * sm_scale

    # Query t sees keys t .. t + WINDOW - 1.
    visible = (j[None, :] >= t[:, None]) & (j[None, :] <= t[:, None] + (WINDOW - 1))
    s = tl.where(visible & kvalid[None, :], s, float("-inf"))

    m = tl.max(s, 1)
    if HAS_SINK:
        sink = tl.load(SINKS + h, mask=rvalid, other=float("-inf")).to(tl.float32)
        m = tl.maximum(m, sink)
    p = tl.exp(s - m[:, None])
    l = tl.sum(p, 1)
    if HAS_SINK:
        l += tl.exp(sink - m)

    dv = tl.arange(0, DV)
    v = tl.load(V + slot[:, None] * stride_vs + dv[None, :],
                mask=kvalid[:, None], other=0.0)
    o = tl.dot(p.to(v.dtype), v) / l[:, None]
    tl.store(OUT + m_idx[:, None] * stride_om + h[:, None] * stride_oh + dv[None, :],
             o.to(OUT.dtype.element_ty), mask=rvalid[:, None])


def can_use_swa_decode(
    *,
    q: torch.Tensor,
    forward_mode,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    v_head_dim: int,
    window_size: tuple,
    causal: bool,
    page_size: int,
    softcap: float,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    kwargs: dict,
) -> bool:
    """Shape gate. Anything outside this contract falls back to FA3."""
    # Verify blocks only: the kernel assumes every query row sits at the tail of
    # its request, which is false for a prefill chunk.
    if not forward_mode.is_target_verify():
        return False
    if max_seqlen_q is None or not 1 <= max_seqlen_q <= _MAX_VERIFY_TOKENS:
        return False
    if not causal or softcap not in (0, 0.0):
        return False
    if num_kv_heads != 1 or num_q_heads % _HEADS_PER_PROGRAM != 0:
        return False
    if page_size != 1:
        return False
    if window_size[0] <= 0 or window_size[1] != 0:
        return False
    if head_dim % _DIM_CHUNK != 0 or v_head_dim & (v_head_dim - 1):
        return False
    if q.dtype != torch.bfloat16:
        return False
    # Row and head strides are explicit; only the head dim must be contiguous.
    if q.stride(-1) != 1:
        return False
    if any(k in kwargs for k in ("q_descale", "k_descale", "v_descale")):
        return False
    # Uniform rows per request (the verify graph shape); ragged batches fall back.
    return q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q


def swa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    window_tokens: int,
    window_size: int,
    sm_scale: float,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    heads_per_program: int = _HEADS_PER_PROGRAM,
) -> torch.Tensor:
    """q: [M, HQ, DQK]; k_cache: [slots, ..., DQK]; v_cache: [slots, ..., DV];
    page_table: [bs, pages] (page_size 1); cache_seqlens: [bs] int32."""
    m, hq, dqk = q.shape
    bs = cache_seqlens.shape[0]
    dv = v_cache.shape[-1]
    # Write through a [M, HQ, DV] view, return the caller's buffer unchanged.
    if out is None:
        out = torch.empty((m, hq, dv), device=q.device, dtype=q.dtype)
    out_view = out.view(m, hq, dv)

    k2 = k_cache.view(k_cache.shape[0], -1)
    v2 = v_cache.view(v_cache.shape[0], -1)
    n_keys = window_size - 1 + window_tokens

    hp = _MMA_HP.get(window_tokens, 0)
    if (
        window_size in _MMA_WINDOWS
        and hp
        and hq % hp == 0
        # The CUDA query load is a 16-byte copy, so each row and head must
        # start on a 16-byte boundary.
        and q.stride(0) % 8 == 0
        and q.stride(1) % 8 == 0
        and q.data_ptr() % 16 == 0
    ):
        _mma_module().swa_decode_mma(
            q, k2, v2, page_table, cache_seqlens,
            sinks if sinks is not None else q, out_view,
            q.stride(0), q.stride(1),
            sm_scale, window_tokens, window_size, hp, 1 if sinks is not None else 0,
        )
        return out

    grid = (bs, hq // heads_per_program)
    _swa_decode_attn_kernel[grid](
        q, k2, v2, page_table, cache_seqlens,
        sinks if sinks is not None else q,
        out_view,
        q.stride(0), q.stride(1), k2.stride(0), v2.stride(0), page_table.stride(0),
        out_view.stride(0), out_view.stride(1),
        sm_scale,
        W=window_tokens, HP=heads_per_program, WINDOW=window_size, DQK=dqk, DV=dv,
        DC=_DIM_CHUNK,
        # tl.dot needs at least 16 rows.
        BLOCK_R=max(16, triton.next_power_of_2(window_tokens * heads_per_program)),
        BLOCK_N=triton.next_power_of_2(n_keys),
        HAS_SINK=sinks is not None,
        num_warps=4, num_stages=1,
    )
    return out
