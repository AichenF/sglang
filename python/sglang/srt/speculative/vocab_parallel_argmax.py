"""Vocab-parallel greedy argmax over a TP-sharded LM head.

Under greedy verification/sampling no rank needs the full ``[T, vocab]`` logits:
each rank reduces its own ``[T, vocab/tp]`` shard to one ``(max, first index)``
pair per row, the ``T * tp`` pairs (16 B each) are exchanged with the multimem
symmetric-memory all-gather (NCCL fallback), and every rank selects the same
winner. This replaces ``all-gather(logits) -> fp32 copy -> torch.argmax`` and is
bit-exact with ``torch.argmax(full_logits, dim=-1)``:

* ranks own contiguous ascending vocab shards, so "lowest global index among
  equal maxima" == "lowest rank, then lowest local index";
* torch's ArgMax ordering is reproduced exactly: NaN beats everything (first
  NaN wins), -0.0 == +0.0 (first index wins), ties -> first index.

Every launch here is CUDA-graph capturable (static buffers, no allocation).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as symm_ag

logger = logging.getLogger(__name__)

_INT32_MIN = -2147483648
_INT32_MAX = 2147483647
# One pair per row: [ordered_key(int32), global_index(int32), 0, 0] = 16 bytes,
# i.e. exactly one 128-bit multimem.st chunk (8 bf16) of the symm-mem gather.
_PAIR_WIDTH = 4


@triton.jit
def _row_argmax_pairs_kernel(
    logits_ptr,
    row_stride,
    n_cols,
    vocab_start,
    pairs_ptr,
    PAIR_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: (ordered key of the max, first global argmax index)."""
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * row_stride
    best_key = tl.full((), -2147483648, tl.int32)
    best_idx = tl.full((), 2147483647, tl.int32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        v = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        # -0.0 == +0.0 for torch: canonicalize so the tie goes to the first index.
        v = tl.where(v == 0.0, 0.0, v)
        bits = v.to(tl.int32, bitcast=True)
        # Monotonic float -> int32 order mapping (negative floats flip the low bits).
        key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        # NaN is the maximum for torch.argmax.
        key = tl.where(v != v, 2147483647, key)
        # Padded lanes lose against every real value, including -inf.
        key = tl.where(mask, key, -2147483648)
        blk_max = tl.max(key, axis=0)
        blk_idx = tl.min(tl.where(key == blk_max, cols, 2147483647), axis=0)
        better = (blk_max > best_key) | ((blk_max == best_key) & (blk_idx < best_idx))
        best_key = tl.where(better, blk_max, best_key)
        best_idx = tl.where(better, blk_idx, best_idx)
    out = pairs_ptr + row.to(tl.int64) * PAIR_WIDTH
    tl.store(out, best_key)
    tl.store(out + 1, best_idx + vocab_start)


@triton.jit
def _select_pairs_kernel(
    gathered_ptr,
    out_ptr,
    n_rows,
    row_stride,
    rank_stride,
    WORLD: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Per row, pick the (key, index) pair with the largest key; ties -> lowest index."""
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rmask = rows < n_rows
    best_key = tl.full((BLOCK_ROWS,), -2147483648, tl.int32)
    best_idx = tl.full((BLOCK_ROWS,), 2147483647, tl.int32)
    for r in tl.static_range(WORLD):
        p = gathered_ptr + rows.to(tl.int64) * row_stride + r * rank_stride
        key = tl.load(p, mask=rmask, other=-2147483648)
        idx = tl.load(p + 1, mask=rmask, other=2147483647)
        better = (key > best_key) | ((key == best_key) & (idx < best_idx))
        best_key = tl.where(better, key, best_key)
        best_idx = tl.where(better, idx, best_idx)
    tl.store(out_ptr + rows, best_idx.to(tl.int64), mask=rmask)


class VocabParallelGreedyHead:
    """Greedy argmax over TP-sharded logits; every rank gets the identical result.

    ``__call__(logits_shard)`` with ``logits_shard`` = this rank's ``[T, local_vocab]``
    (bf16/fp16/fp32, row-major) returns ``out[:T]`` (int64 global token ids).
    All buffers are static so the launches can be captured into a CUDA graph.
    """

    def __init__(
        self,
        *,
        tp_group,
        max_tokens: int,
        local_vocab: int,
        vocab_start: int,
        device,
        out: Optional[torch.Tensor] = None,
        name: str = "head",
    ):
        self.tp_group = tp_group
        self.tp_size = int(tp_group.world_size) if tp_group is not None else 1
        self.rank = int(tp_group.rank_in_group) if tp_group is not None else 0
        self.max_tokens = int(max_tokens)
        self.local_vocab = int(local_vocab)
        self.vocab_start = int(vocab_start)
        self.device = device
        self.name = name
        self.pairs = torch.zeros(
            (self.max_tokens, _PAIR_WIDTH), dtype=torch.int32, device=device
        )
        if out is None:
            out = torch.zeros((self.max_tokens,), dtype=torch.int64, device=device)
        assert out.numel() >= self.max_tokens and out.dtype == torch.int64
        self.out = out
        self._state = None
        self._gathered_nccl = None
        if self.tp_size > 1:
            self._state = self._build_symm_state()
            if self._state is None:
                self._gathered_nccl = torch.zeros(
                    (self.tp_size, self.max_tokens, _PAIR_WIDTH),
                    dtype=torch.int32,
                    device=device,
                )
        self.mode = (
            "tp1"
            if self.tp_size == 1
            else ("multimem" if self._state is not None else "nccl")
        )

    def _build_symm_state(self):
        try:
            state = symm_ag.create_state(
                group=self.tp_group.device_group,
                rank_in_group=self.tp_group.rank_in_group,
                max_tokens=self.max_tokens,
                # 8 bf16 (16 B) per rank per row.
                hidden_size=symm_ag._NUMEL_PER_THREAD * self.tp_size,
                device=torch.device(self.device),
            )
            if state.symm_mem_hdl.multicast_ptr == 0:
                logger.warning(
                    "VocabParallelGreedyHead[%s]: no multicast; using NCCL all-gather",
                    self.name,
                )
                return None
            return state
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "VocabParallelGreedyHead[%s]: symm-mem unavailable (%s); using NCCL",
                self.name,
                e,
            )
            return None

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        n = int(logits.shape[0])
        assert logits.ndim == 2 and n <= self.max_tokens, (
            logits.shape,
            self.max_tokens,
        )
        assert int(logits.shape[1]) == self.local_vocab, (
            logits.shape,
            self.local_vocab,
        )
        assert logits.stride(1) == 1
        if n == 0:
            return self.out[:0]
        pairs = self.pairs[:n]
        _row_argmax_pairs_kernel[(n,)](
            logits,
            logits.stride(0),
            self.local_vocab,
            self.vocab_start,
            pairs,
            PAIR_WIDTH=_PAIR_WIDTH,
            BLOCK=4096,
            num_warps=8,
        )
        out = self.out[:n]
        if self.tp_size == 1:
            out.copy_(pairs[:, 1])
            return out
        if self._state is not None:
            gathered = symm_ag.all_gather_inner(
                self._state,
                pairs.view(torch.bfloat16),  # [n, 8] bf16 == [n, 16 B]
                tp_hidden_dim=symm_ag._NUMEL_PER_THREAD * self.tp_size,
                # Consecutive calls are separated by the model's all-reduces, which
                # are cross-rank syncs, so the entry barrier can be skipped.
                skip_entry_sync=True,
                safe=False,
            )  # [n, 8 * tp] bf16 view: row = rank-major 16-B chunks
            g = gathered.view(torch.int32)  # [n, 4 * tp]
            row_stride, rank_stride = _PAIR_WIDTH * self.tp_size, _PAIR_WIDTH
        else:
            # all_gather_into_tensor wants a contiguous [tp * n, 4] output.
            g = self._gathered_nccl.view(-1)[: self.tp_size * n * _PAIR_WIDTH].view(
                self.tp_size, n, _PAIR_WIDTH
            )
            self.tp_group.all_gather_into_tensor(g, pairs)
            row_stride, rank_stride = _PAIR_WIDTH, n * _PAIR_WIDTH
        block_rows = 64
        _select_pairs_kernel[(triton.cdiv(n, block_rows),)](
            g,
            out,
            n,
            row_stride,
            rank_stride,
            WORLD=self.tp_size,
            BLOCK_ROWS=block_rows,
            num_warps=2,
        )
        return out


def reference_argmax_from_shard(logits_shard: torch.Tensor, tp_group) -> torch.Tensor:
    """Slow reference: NCCL all-gather of the shard + torch.argmax (test/self-check)."""
    if tp_group is None or tp_group.world_size == 1:
        return torch.argmax(logits_shard, dim=-1)
    full = tp_group.all_gather(logits_shard.contiguous(), dim=-1)
    return torch.argmax(full, dim=-1)
