"""Aux hidden states captured for Eagle3/DFlash draft models."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStateList(list):
    """Retain aux snapshots independently of storage reused by later layers."""

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Copy borrowed storage, or adopt a value whose ownership is transferred.

        Views and reusable communication buffers are borrowed even when they
        are different tensor objects from the source.
        """
        self.append(hidden if owned else hidden.clone())


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.

    ``for_batch`` writes into a decode graph runner's buffer when the batch
    carries one, so every graph size aliases it instead of pinning its own.
    """

    def __init__(
        self, num_captures: int, out: Optional[torch.Tensor] = None, defer_stable: bool = False
    ) -> None:
        self._num_captures = int(num_captures)
        self._buffer = out
        self._feature_size: Optional[int] = None
        self._idx = 0
        # capture_stable() records (slot start, source) instead of copying; finalize() hands the list to
        # flush_aux_pending (one launch for every deferred capture, possibly on a side stream)
        self._defer = bool(defer_stable)
        self._pending: List = []

    @classmethod
    def for_batch(
        cls, forward_batch: ForwardBatch, num_captures: int
    ) -> AuxHiddenStatePacker:
        return cls(num_captures, out=forward_batch.aux_hidden_states_buffer)

    def _slot(self, hidden: torch.Tensor) -> int:
        feature_size = int(hidden.shape[-1])
        if self._feature_size is None:
            self._feature_size = feature_size
            shape = (*hidden.shape[:-1], feature_size * self._num_captures)
            if self._buffer is None:
                self._buffer = hidden.new_empty(shape)
            elif self._buffer.shape != shape or self._buffer.dtype != hidden.dtype:
                raise ValueError(
                    f"aux hidden output buffer is {tuple(self._buffer.shape)} "
                    f"{self._buffer.dtype}, captures need {shape} {hidden.dtype}"
                )
        return self._idx * self._feature_size

    def append(self, hidden: torch.Tensor) -> None:
        start = self._slot(hidden)
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Write directly to the final packed buffer, without an intermediate copy."""
        self.append(hidden)

    def capture_stable(self, hidden: torch.Tensor) -> None:
        """Like capture(), for a source whose storage is NOT written again before the packed buffer is consumed
        (fixed norm-next residual buffers, the final updated residual): deferred to flush_aux_pending when enabled."""
        if (
            not self._defer
            or hidden.dim() != 2
            or not hidden.is_contiguous()
            or hidden.dtype != torch.bfloat16
            or len(self._pending) >= _AUX_PACK_MAX
        ):
            self.append(hidden)
            return
        start = self._slot(hidden)
        if (
            self._buffer.dim() != 2
            or hidden.shape[0] < self._buffer.shape[0]
            or self._buffer.stride(-1) != 1
        ):
            self.append_at(hidden, start)
            return
        self._pending.append((start, hidden))
        self._idx += 1

    def append_at(self, hidden: torch.Tensor, start: int) -> None:
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        if self._feature_size is None or self._idx != self._num_captures:
            raise RuntimeError(
                f"captured {self._idx} of {self._num_captures} aux hidden states"
            )
        if self._pending:
            self._buffer._aux_pending = self._pending
            self._pending = []
        return self._buffer


# Both collectors accept borrowed values through capture; storage belongs to aux.
AuxHiddenStateAccumulator = Union[AuxHiddenStateList, AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)


# ---- deferred aux captures, packed in one launch ----
_AUX_PACK_MAX = 5


def aux_has_pending(buf) -> bool:
    return bool(getattr(buf, "_aux_pending", None))


def flush_aux_pending(buf, keepalive: Optional[list] = None) -> None:
    """Copy the captures deferred by AuxHiddenStatePacker.capture_stable into the packed buffer (one Triton launch on
    the current stream). ``keepalive``: receives the sources (the caller must keep them alive until the copy is ordered
    before any reuse, e.g. until a side stream is joined)."""
    pend = getattr(buf, "_aux_pending", None)
    if not pend:
        return
    buf._aux_pending = None
    from sglang.kernels.ops.speculative.aux_pack import aux_pack

    aux_pack(buf, pend)
    if keepalive is not None:
        keepalive.extend(src for _, src in pend)
