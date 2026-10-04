"""Fused TP all-reduce + residual add + RMSNorm for the DFLASH draft layers.

The draft block is tiny (bs x block_size rows, e.g. 32 x 6144 bf16 = 384 KB per
rank), so every o_proj / down_proj all-reduce is latency-bound: the stock path
runs the custom 2-shot all-reduce (~8 us) and then flashinfer's fused_add_rmsnorm
(~6 us, partly PDL-overlapped). FlashInfer's TRT-LLM one-shot Lamport
``allreduce_fusion`` (AR + residual + RMSNorm in one kernel) replaces both.

Draft-only numerics: the fused kernel accumulates the 8 partials in fp32 in a
different order and rounds the residual/norm once; under greedy verification the
emitted tokens are decided by the target model, so the output is unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DFlashFusedARNorm:
    """One flashinfer TRT-LLM one-shot all-reduce + add + RMSNorm per call."""

    def __init__(
        self,
        *,
        hidden_size: int,
        max_token_num: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.max_token_num = int(max_token_num)
        # Stay in the one-shot Lamport regime the kernel is designed for
        # (flashinfer OneShotMaxToken = 128); larger draft blocks fall back to
        # the stock AR + norm path per forward.
        self.max_fused_tokens = min(self.max_token_num, 128)
        # Measured on H200 TP8 (profiler trace): the one-shot Lamport fusion costs
        # 16.4 us per 32x6144 bf16 call (it pushes world_size copies over NVLink)
        # vs 7.9 us 2shot AR + ~3.4 us PDL-overlapped add_rmsnorm stock. Use the
        # two-shot fusion (reduce-scatter + norm + all-gather) except for blocks
        # too small for it (flashinfer asserts token_num > world_size).
        self.world_size = 1
        self.oneshot_max_tokens = int(
            os.environ.get("SGLANG_DFLASH_FUSED_AR_ONESHOT_MAX_TOKENS", "0") or 0
        )
        self.dtype = dtype
        self.device = device
        self.enabled = False
        self._comm = None
        self._workspace = None
        self._pattern = None
        # OPT-IN (default off): measured on H200 TP8 with 32x6144 bf16 blocks the
        # flashinfer TRT-LLM fusion kernels are far slower than the stock custom
        # 2shot all-reduce + fused_add_rmsnorm (one-shot lamport 16.4 us/call,
        # two-shot 63 us/call vs ~11 us stock; E2E +0.09 / +0.63 ms per step).
        if os.environ.get("SGLANG_DFLASH_FUSED_AR_NORM", "0") not in ("1", "true", "True"):
            logger.info(
                "DFLASH fused AR+RMSNorm not enabled (opt in with SGLANG_DFLASH_FUSED_AR_NORM=1)"
            )
            return
        try:
            from sglang.srt.distributed import get_tp_group
            from sglang.srt.layers import flashinfer_comm_fusion as fcf

            tp_group = get_tp_group()
            world_size = int(tp_group.world_size)
            self.world_size = world_size
            if self.oneshot_max_tokens <= 0:
                self.oneshot_max_tokens = world_size
            if world_size <= 1:
                return
            comm = fcf._flashinfer_comm
            if comm is None or fcf.is_flashinfer_allreduce_unavailable():
                logger.info(
                    "DFLASH fused AR+RMSNorm unavailable (flashinfer.comm missing)."
                )
                return
            manager = fcf.FlashInferWorkspaceManager()
            manager.initialize(
                world_size=world_size,
                rank=int(tp_group.rank_in_group),
                max_token_num=self.max_token_num,
                hidden_dim=self.hidden_size,
                backend="trtllm",
                group=tp_group.cpu_group,
                use_fp32_lamport=False,
                dtype=dtype,
                use_oneshot=True,
                device_group=tp_group.device_group,
                cpu_group=tp_group.cpu_group,
            )
            if not manager.initialized or manager.workspace is None:
                logger.info("DFLASH fused AR+RMSNorm: workspace init failed; disabled.")
                return
            self._manager = manager
            self._comm = comm
            self._workspace = manager.workspace
            self._pattern = comm.AllReduceFusionPattern.kARResidualRMSNorm
            self.enabled = True
            if int(tp_group.rank_in_group) == 0:
                logger.info(
                    "DFLASH fused AR+RMSNorm enabled (trtllm, one-shot up to %d tokens else two-shot, max_token_num=%d, hidden=%d).",
                    self.oneshot_max_tokens,
                    self.max_token_num,
                    self.hidden_size,
                )
        except Exception as e:  # pragma: no cover - defensive: never break the draft
            logger.warning("DFLASH fused AR+RMSNorm init failed, disabled: %s", e)
            self.enabled = False

    def can_fuse(self, x: torch.Tensor) -> bool:
        return (
            self.enabled
            and x.ndim == 2
            and x.dtype == self.dtype
            and int(x.shape[1]) == self.hidden_size
            and 0 < int(x.shape[0]) <= self.max_fused_tokens
        )

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """norm_out, residual_out = RMSNorm(allreduce(partial) + residual) * weight."""
        if not partial.is_contiguous():
            partial = partial.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(partial)
        self._comm.allreduce_fusion(
            input=partial,
            workspace=self._workspace,
            pattern=self._pattern,
            launch_with_pdl=True,
            trigger_completion_at_end=False,
            residual_out=residual_out,
            norm_out=norm_out,
            residual_in=residual,
            rms_gamma=weight,
            rms_eps=float(eps),
            # two-shot unless the block is too small for it (see __init__).
            use_oneshot=bool(partial.shape[0] <= self.oneshot_max_tokens),
            fp32_acc=True,
        )
        return norm_out, residual_out
