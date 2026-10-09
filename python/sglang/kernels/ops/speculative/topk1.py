from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def draft_topk1_argmax_only(next_token_logits: torch.Tensor):
    """Select top-k=1 without mutating caller-owned positions."""
    # The upstream finalizer always advances positions. Draft-extend has already
    # consumed its positions, so absorb that side effect in a discarded buffer.
    scratch_positions = torch.zeros(
        next_token_logits.shape[0],
        dtype=torch.long,
        device=next_token_logits.device,
    )
    return draft_topk1_postprocess(next_token_logits, scratch_positions)


# ---------------------------------------------------------------------------
# DFLASH draft: vocab-shard (max, argmax) for the tensor-parallel greedy select.
#
# The DFLASH draft sampler needs the shard-local maximum *value* as well as its
# index, because the ranks all-gather (max, id) pairs and pick the winner. That
# is one reduction more than `draft_topk1_postprocess` returns, so it gets its
# own pair of kernels rather than being bolted onto that one.
#
# The decomposition is the same and it is the whole point: `torch.max(logits,
# dim=-1)` gives one CTA per row, and the draft only has `block_size - 1` rows,
# so the reduction runs on 6 CTAs of 148 and reaches 37 GB/s. Splitting the
# vocab dimension across CTAs is what the EAGLE top-1 path already does.
# ---------------------------------------------------------------------------

_DFLASH_SHARD_BLOCK = 8192


@triton.jit
def _dflash_shard_max_partial_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once rows * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _dflash_shard_max_finalize_kernel(
    partial_vals,
    partial_indices,
    out_max,
    out_arg,
    org_vocab_start,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    # tie_break_left in both stages: the winning split is the lowest-numbered
    # one holding the maximum, and within it the lowest offset. That composes to
    # the globally first maximal index, which is what torch.max returns and what
    # _DflashDraftSampler's cross-rank select documents as its invariant.
    best = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + best).to(tl.int64)
    value = tl.load(partial_vals + row * num_splits + best)
    tl.store(out_max + row, value.to(out_max.dtype.element_ty))
    tl.store(out_arg + row, index + org_vocab_start)


def dflash_shard_max_argmax(
    logits: torch.Tensor,
    out_max: torch.Tensor,
    out_arg: torch.Tensor,
    partial_vals: torch.Tensor,
    partial_indices: torch.Tensor,
    org_vocab_start: int = 0,
    block: int = _DFLASH_SHARD_BLOCK,
) -> None:
    """Shard-local ``(max, argmax)`` over the vocab dimension, split across CTAs.

    Drop-in for ``torch.max(logits, dim=-1, out=(out_max, out_arg))`` followed by
    ``out_arg.add_(org_vocab_start)``, for contiguous-row 2D ``logits``.

    All five tensors are caller-owned and written in place, so the call is safe
    to capture in a CUDA graph: no allocation happens here and every address is
    fixed at capture time.

    ``partial_vals`` / ``partial_indices`` must each hold at least
    ``logits.shape[0] * cdiv(logits.shape[1], block)`` elements, fp32 and int32
    respectively.

    Bit-identical to ``torch.max`` for finite inputs: bf16 widens to fp32
    exactly, max is order-independent, and both reduction stages break ties to
    the left. NaN is the one divergence -- ``torch.max`` propagates it, this
    reduction ignores it, the same choice ``draft_topk1_postprocess`` makes.
    """
    assert logits.ndim == 2
    assert logits.stride(1) == 1
    n, vocab_size = logits.shape
    assert out_max.shape[0] == n and out_arg.shape[0] == n
    assert out_arg.dtype == torch.long
    if n == 0:
        return

    num_splits = triton.cdiv(vocab_size, block)
    assert partial_vals.numel() >= n * num_splits
    assert partial_indices.numel() >= n * num_splits

    _dflash_shard_max_partial_kernel[(n, num_splits)](
        logits,
        partial_vals,
        partial_indices,
        logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _dflash_shard_max_finalize_kernel[(n,)](
        partial_vals,
        partial_indices,
        out_max,
        out_arg,
        org_vocab_start,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
