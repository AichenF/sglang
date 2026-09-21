"""Bidirectional (non-causal) paged attention for the DFlash draft propose block.

Why this exists
---------------
The DFlash draft is 5 dense layers with 128 query heads / 8 KV heads, so at TP8
each rank runs 16 query heads against a **single** KV head.  Its layers are
``AttentionType.ENCODER_ONLY`` -- bidirectional -- and, because the checkpoint's
``sliding_window`` never survives into ``layer_types`` (HF's ``Qwen3Config``
rewrites it to five ``full_attention`` entries), SGLang gives them
``sliding_window_size = -1``, i.e. ``window_size = (-1, -1)``: full-context
bidirectional attention over ~1k keys with only ``block_size = 8`` query rows
per request.

Measured on sm_90 in a CUDA graph at bs4, ctx 1024 (see
a standalone launch-floor probe):

    empty Triton kernel, grid 128        0.82 us   <- launch floor
    gather + read K,V only, no math      3.61 us   <- page-table gather floor
    FA3 (prepare + fwd + combine)       14.16 us
    HBM roofline for the same bytes      0.44 us

FA3's cost is ~9 us of fixed overhead plus work: the same call at ctx 64 still
costs 9.07 us and only reaches 18.99 us at ctx 8192.  On five draft layers that
overhead is pure loss.  This kernel keeps the flash-decoding structure that the
shape actually wants -- split the key range across programs so the machine is
covered, and let every query head of a request share one KV tile, which is the
GQA sharing FA3 cannot exploit at 16:1 -- and drops the fixed cost.

Semantics match ``flash_attn_with_kvcache(..., causal=False,
window_size=(-1, -1))`` on the same page table and ``cache_seqlens``: every
query row of request ``b`` attends keys ``[0, cache_seqlens[b])`` through
``page_table[b]``.  The windowed case is kept behind ``WINDOWED`` constexprs so
a corrected checkpoint (real sliding_attention layers) stays one flag away; it
is compiled out entirely for the full-context path that ships today.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1.0e30)

# Tuned on sm_90 at the target shapes (bs4/bs8, 16 q heads, 1 kv head,
# head_dim 128, 8 query rows per request, ctx 1024).
_HEADS_PER_PROGRAM = 8
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3
# This part has 132 SMs; the sweep optimum is whatever NSPLIT puts the grid at ~128
# programs (bs4 -> 16, bs8 -> 8), and it falls off on both sides.
_TARGET_GRID = 128
_MAX_NSPLIT = 64

# The draft block is 6-8 tokens; the cap rejects anything that is not one.
_MAX_BLOCK_TOKENS = 32


@triton.jit
def _bidir_decode_kernel(
    Q, K, V, PT, SEQLENS, OUT, PART_O, PART_LSE,
    stride_qm, stride_qh, stride_ks, stride_vs, stride_ptb,
    stride_om, stride_oh,
    stride_pob, stride_poh, stride_pos,
    sm_scale,
    W: tl.constexpr,          # query rows per request (draft block size)
    HP: tl.constexpr,         # query heads per program
    WIN_L: tl.constexpr,      # window left; -1 = unlimited
    WIN_R: tl.constexpr,      # window right; -1 = unlimited
    D: tl.constexpr, DV: tl.constexpr,
    BLOCK_R: tl.constexpr,    # next pow2 >= W * HP, min 16 for tl.dot
    BLOCK_N: tl.constexpr,    # keys per iteration
    NSPLIT: tl.constexpr,     # kv splits; 1 = single-kernel path
    WINDOWED: tl.constexpr,   # compile the mask out entirely when full-context
    HQN: tl.constexpr,        # total query heads (for the PART_LSE index)
):
    b = tl.program_id(0)
    hg = tl.program_id(1)
    sp = tl.program_id(2)

    seqlen = tl.load(SEQLENS + b)

    rows = tl.arange(0, BLOCK_R)
    rvalid = rows < W * HP
    t = rows // HP                      # draft-block token index
    h = hg * HP + rows % HP             # query head
    m_idx = b * W + t                   # row in the flattened [M, HQ, D] q

    d = tl.arange(0, D)
    q = tl.load(
        Q + m_idx[:, None] * stride_qm + h[:, None] * stride_qh + d[None, :],
        mask=rvalid[:, None], other=0.0,
    )

    if WINDOWED:
        # Absolute position of each query row: the draft block sits at the tail.
        qpos = seqlen - W + t

    nblk = tl.cdiv(seqlen, BLOCK_N)

    m_i = tl.full((BLOCK_R,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, DV), tl.float32)
    dv = tl.arange(0, DV)

    # Round-robin block assignment across the splits. A contiguous-chunk split
    # (lo = sp * chunk) leaves late programs empty whenever NSPLIT does not
    # divide nblk -- at seqlen 1032 / BLOCK_N 32 / NSPLIT 32 only 17 of the 32
    # programs get any work. Strided assignment balances to within one block
    # for every seqlen. Measured 8.75 -> 8.39 us at bs4.
    for bi in tl.range(sp, nblk, NSPLIT):
        start = bi * BLOCK_N
        j = start + tl.arange(0, BLOCK_N)
        kvalid = j < seqlen
        slot = tl.load(PT + b * stride_ptb + j, mask=kvalid, other=0)
        k = tl.load(K + slot[:, None] * stride_ks + d[None, :],
                    mask=kvalid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        keep = kvalid[None, :]
        if WINDOWED:
            if WIN_L >= 0:
                keep = keep & (j[None, :] >= qpos[:, None] - WIN_L)
            if WIN_R >= 0:
                keep = keep & (j[None, :] <= qpos[:, None] + WIN_R)
        # Masked entries must sit strictly below the -1e30 running max, so a
        # fully masked tile contributes exactly zero rather than ones.
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * stride_vs + dv[None, :],
                    mask=kvalid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if NSPLIT == 1:
        # A padded CUDA-graph slot has cache_seqlens == 0, so the loop never
        # runs and l_i stays 0; emit zeros rather than 0/0 = NaN, which would
        # otherwise flow into o_proj and the all-reduce.
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(
            OUT + m_idx[:, None] * stride_om + h[:, None] * stride_oh + dv[None, :],
            o.to(OUT.dtype.element_ty), mask=rvalid[:, None],
        )
    else:
        # Store the *normalised* partial output plus its log-sum-exp; the
        # combine is then a plain lse-weighted average and needs no separate
        # running max / denominator.
        #   o_s = acc_s / l_s ,  lse_s = m_s + log l_s
        #   o   = sum_s w_s o_s / sum_s w_s ,  w_s = exp(lse_s - max_s lse)
        po = PART_O + m_idx[:, None] * stride_pob + h[:, None] * stride_poh \
            + sp * stride_pos + dv[None, :]
        # A split that received no blocks has l_i == 0; give it lse = -inf so
        # it drops out of the combine instead of producing a NaN.
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        tl.store(po, acc / l_safe[:, None], mask=rvalid[:, None])
        lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
        tl.store(PART_LSE + m_idx * (HQN * NSPLIT) + h * NSPLIT + sp, lse,
                 mask=rvalid)


@triton.jit
def _bidir_combine_kernel(
    PART_O, PART_LSE, OUT,
    stride_pob, stride_poh, stride_pos, stride_om, stride_oh,
    NSPLIT: tl.constexpr, DV: tl.constexpr, HQN: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    lse = tl.load(PART_LSE + m * (HQN * NSPLIT) + h * NSPLIT + sp)
    mx = tl.max(lse)
    # All splits empty (cache_seqlens == 0 on a padded slot): mx is -inf and
    # exp(-inf - -inf) is NaN. Pin the frame and let the zero denominator
    # guard below produce a zero row instead.
    mx = tl.where(mx > float("-inf"), mx, 0.0)
    w = tl.exp(lse - mx)
    dv = tl.arange(0, DV)
    po = tl.load(PART_O + m * stride_pob + h * stride_poh
                 + sp[:, None] * stride_pos + dv[None, :])
    den = tl.sum(w)
    o = tl.sum(po * w[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(OUT + m * stride_om + h * stride_oh + dv,
             o.to(OUT.dtype.element_ty))


def _pick_nsplit(bs: int, head_groups: int) -> int:
    """Split the key range just enough to cover the machine, rounded down to a
    power of two (the combine kernel indexes splits with tl.arange)."""
    want = max(1, _TARGET_GRID // max(1, bs * head_groups))
    ns = 1
    while ns * 2 <= min(want, _MAX_NSPLIT):
        ns *= 2
    return ns


def can_use_bidir_decode(
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
    """Shape gate for the DFlash draft propose block.

    Deliberately narrow: it must select the five draft layers and nothing else.
    The target's own non-SWA layers are causal and carry asymmetric head dims
    (192 / 128), so ``not causal`` plus ``head_dim == v_head_dim`` already
    separates them; the rest is the contract the kernel implements.
    """
    # The draft propose runs as TARGET_VERIFY (see DFlashWorkerV2), with a
    # uniform block of query rows per request.
    if not forward_mode.is_target_verify():
        return False
    if causal:
        return False
    if max_seqlen_q is None or not 1 <= max_seqlen_q <= _MAX_BLOCK_TOKENS:
        return False
    if softcap not in (0, 0.0):
        return False
    if num_kv_heads != 1 or num_q_heads % _HEADS_PER_PROGRAM != 0:
        return False
    if page_size != 1:
        return False
    if head_dim != v_head_dim or head_dim & (head_dim - 1):
        return False
    if q.dtype != torch.bfloat16:
        return False
    # Sinks are not built for this draft (is_nemotron_35_draft_config is False),
    # and FP8 KV descaling is a different contract.
    if any(k in kwargs for k in ("q_descale", "k_descale", "v_descale", "sinks",
                                 "score_mod", "rel_bias")):
        return False
    # Bidirectional window must be symmetric or absent; anything else is not
    # what this kernel computes.
    if window_size[0] != window_size[1]:
        return False
    # Uniform rows per request (the draft-block graph shape); ragged falls back.
    return q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q


def bidir_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    block_tokens: int,
    sm_scale: float,
    window: tuple = (-1, -1),
    out: Optional[torch.Tensor] = None,
    heads_per_program: int = _HEADS_PER_PROGRAM,
    block_n: int = _BLOCK_N,
    nsplit: Optional[int] = None,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> torch.Tensor:
    """q: [M, HQ, D]; k_cache/v_cache: [slots, ..., D]; page_table: [bs, pages]
    (page_size 1); cache_seqlens: [bs] int32.  Non-causal; ``window`` is
    (left, right) with -1 meaning unlimited on that side."""
    # The kernel loads the head dim as one contiguous vector; row and head
    # strides are explicit, so a row-strided q (the QKV slice) is fine.
    if q.stride(-1) != 1:
        q = q.contiguous()
    m, hq, d = q.shape
    bs = cache_seqlens.shape[0]
    dv = v_cache.shape[-1]
    if out is None:
        out = torch.empty((m, hq, dv), device=q.device, dtype=q.dtype)
    out_view = out.view(m, hq, dv)

    k2 = k_cache.view(k_cache.shape[0], -1)
    v2 = v_cache.view(v_cache.shape[0], -1)
    windowed = window[0] >= 0 or window[1] >= 0
    head_groups = hq // heads_per_program
    if nsplit is None:
        nsplit = _pick_nsplit(bs, head_groups)

    if nsplit == 1:
        part_o = part_lse = q            # unused dummy pointers
        spo = sph = sps = 0
    else:
        # Allocated per call on purpose. Under CUDA-graph capture the caching
        # allocator serves this from the capturing graph private pool and bakes
        # the pointer into the replay; a module-level cache would hand a buffer
        # owned by one graph pool to another graph and alias freed memory.
        part_o = torch.empty((m, hq, nsplit, dv), device=q.device, dtype=torch.float32)
        part_lse = torch.empty((m, hq, nsplit), device=q.device, dtype=torch.float32)
        spo, sph, sps = part_o.stride(0), part_o.stride(1), part_o.stride(2)

    _bidir_decode_kernel[(bs, head_groups, nsplit)](
        q, k2, v2, page_table, cache_seqlens, out_view, part_o, part_lse,
        q.stride(0), q.stride(1), k2.stride(0), v2.stride(0), page_table.stride(0),
        out_view.stride(0), out_view.stride(1),
        spo, sph, sps,
        sm_scale,
        W=block_tokens, HP=heads_per_program,
        WIN_L=window[0], WIN_R=window[1],
        D=d, DV=dv,
        BLOCK_R=max(16, triton.next_power_of_2(block_tokens * heads_per_program)),
        BLOCK_N=block_n, NSPLIT=nsplit, WINDOWED=windowed, HQN=hq,
        num_warps=num_warps, num_stages=num_stages,
    )
    if nsplit > 1:
        _bidir_combine_kernel[(m, hq)](
            part_o, part_lse, out_view,
            part_o.stride(0), part_o.stride(1), part_o.stride(2),
            out_view.stride(0), out_view.stride(1),
            NSPLIT=nsplit, DV=dv, HQN=hq, num_warps=4,
        )
    return out
