"""Fused W4A8 MoE layer for Hopper TP decode (one persistent kernel per MoE layer).

Replaces, for small decode / target-verify batches (<= 64 tokens per rank), the sequence
  post_attention_layernorm(hidden, residual) -> gate (build-selected BF16/FP32 router) -> biased top-8 -> Humming fc1 -> SiLU*up ->
  fp8 requant -> Humming fc2 -> top-k combine -> tensor_model_parallel_all_reduce
of ``MiMoV2DecoderLayer`` with a single launch of ``fused_moe_wgmma.cu``.  The expert weights are the Humming MoE
layer's transformed tensors (``w13_weight`` / ``w13_weight_scale`` / ``w13_weight_scale_2``, ``w2_*``) used in place --
the kernel dequantizes Humming's fused-e8m0 layout in registers -- so there is no second copy of the weights and the
prefill / large-batch path keeps running Humming unchanged.

Enable with ``SGLANG_MIMO_FUSED_MOE=1`` (TP=8, EP=1, ``--moe-runner-backend humming`` with the W4A8 activation config,
no DP attention).

Stage-boundary integration (``layers/layer_boundary``): the decoder layer hands the kernel the attention's o_proj partial
BEFORE ``attn_boundary.finish`` records its declared attention-TP sum, plus the residual the attention prepare wrote into
``forward_batch.residual_stream`` (``fused_moe_stream_residual``). The kernel completes that sum, the residual add and the
post-attention norm itself; ``finish_fused_moe`` then writes its post-attention residual into the stream and records the
complete (already all-reduced) MoE output as the FFN's contribution with a plain residual add, so the next layer's
``attn_boundary.prepare`` runs only the add + input norm. With the norm-next handoff the next layer skips that prepare too:
``take_norm_next_handoff`` writes the kernel's residual_new into the stream (and into the DFLASH aux capture) and returns
the pre-quantized qkv_proj input.  Every TP rank is a separate process: the all-reduce buffers are allocated with cudaMalloc and
exchanged through CUDA IPC handles once, at the first forward (a collective over the TP group; all ranks take the
identical decision, so no rank can wait for a peer that fell back).  The kernel takes its all-reduce epoch tag from a
device counter and resets its own work queues, so it is CUDA-graph replay safe.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from typing import Any, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary import PLAIN_ADD, BatchVariant, SumGroup
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.moe.mimo_fused_moe import fused_moe_wgmma as F
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_shared: Optional[_SharedRankState] = None

# Diagnostics only (off unless set): per-layer phase stamps. Needs the kernel built with
# FMOE_NVCC_FLAGS=-DFMOE_PHASE_STAMPS. Creating <dir>/dump.<n> makes every rank write
# <dir>/stamps_rank<r>_<n>.pt = {layer_id: int64 [MAXM, grid * NSTAMP]} (last replay per layer and token count M, row M-1).
_STAMP_DIR = os.environ.get("SGLANG_MIMO_FUSED_MOE_STAMP_DIR", "")
_STAMP_GRID = 132
_stamp_layers: List[Any] = []
_stamp_thread: Optional[threading.Thread] = None


def _stamp_watcher(rank: int) -> None:
    n = 0
    while True:
        if os.path.exists(os.path.join(_STAMP_DIR, f"dump.{n}")):
            try:
                out = {lid: t.cpu() for lid, t in _stamp_layers}
                torch.save(out, os.path.join(_STAMP_DIR, f"stamps_rank{rank}_{n}.pt"))
            except Exception as e:  # diagnostics must never kill the server
                logger.warning(f"MiMo fused MoE stamp dump failed: {e}")
            n += 1
        time.sleep(0.2)


# Diagnostics only (off unless set): capture the M = 32 input-TP kernel inputs (this rank's o_proj partial, the residual)
# and the kernel's published top-8 ids of every layer into a per-layer ring of the last _CAP_RING replays. CUDA-graph
# safe: the ring slot comes from a device counter. Creating <dir>/dump.<n> makes every rank write
# <dir>/capture_rank<r>_<n>.pt = {layer_id: dict(partial [R,32,DIM] bf16, residual [R,32,DIM] bf16, topk [R,32,8] int64,
# count int64 [1])}; ring slot s holds replay number s + R*k, so order the slots by count when count > R.
_CAP_DIR = os.environ.get("SGLANG_MIMO_FUSED_MOE_CAPTURE_DIR", "")
_CAP_RING = int(os.environ.get("SGLANG_MIMO_FUSED_MOE_CAPTURE_RING", "16"))
_cap_layers: List[Any] = []
_cap_thread: Optional[threading.Thread] = None


def _cap_watcher(rank: int) -> None:
    n = 0
    while True:
        if os.path.exists(os.path.join(_CAP_DIR, f"dump.{n}")):
            try:
                out = {lid: {k: t.cpu() for k, t in bufs.items()} for lid, bufs in _cap_layers}
                torch.save(out, os.path.join(_CAP_DIR, f"capture_rank{rank}_{n}.pt"))
            except Exception as e:  # diagnostics must never kill the server
                logger.warning(f"MiMo fused MoE capture dump failed: {e}")
            n += 1
        time.sleep(0.2)


def mimo_fused_moe_enabled() -> bool:
    return bool(envs.SGLANG_MIMO_FUSED_MOE.get())


def _gather_objects(group, obj) -> List[Any]:
    """all_gather_object over the TP group's CPU group (rank order), gloo-safe under inference mode."""
    world = group.world_size
    all_data = [[None] for _ in range(world)]
    all_data[group.rank_in_group][0] = obj
    for i, rank in enumerate(
        group.ranks
    ):  # global ranks of the group, in group-rank order
        dist.broadcast_object_list(
            all_data[i], src=rank, group=group.cpu_group, device="cpu"
        )
    return [d[0] for d in all_data]


class _SharedRankState:
    """Per-process kernel state shared by every MoE layer: staging buffers, work queue, the CUDA-IPC all-reduce buffers."""

    def __init__(self, device: torch.device):
        group = get_tp_group()
        self.group = group
        self.rank = group.rank_in_group
        self.ndev = group.world_size
        self.device = device
        F.ext()  # JIT build (torch cpp_extension: file-locked, concurrent ranks wait for the builder)
        self.state = F.RankState(device, alloc_comm=False)
        lib = CudaRTLibrary()
        self._lib = lib
        self._ptrs = []
        all_ptrs = []
        for nbytes in (F.RankState.RS_BYTES, F.RankState.AG_BYTES):
            ptr = lib.cudaMalloc(nbytes)
            lib.cudaMemset(ptr, 0, nbytes)  # stale tags could alias a live epoch
            self._ptrs.append(ptr)
            handle = lib.cudaIpcGetMemHandle(ptr)
            handles = _gather_objects(group, bytes(handle))
            ptrs = []
            for r, h in enumerate(handles):
                if r == self.rank:
                    ptrs.append(ptr.value)
                else:
                    peer = lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h))
                    ptrs.append(peer.value)
            all_ptrs.append(ptrs)
        self.state.link_ptrs(all_ptrs[0], all_ptrs[1])
        # kernels built with -DFMOE_INPUT_TP=1 absorb the input TP all-reduce into their prologue over one more IPC
        # buffer per rank (reduce-scatter / all-gather LL messages + this rank's router-partial and top-8 regions).
        self.input_tp = (
            bool(getattr(F.ext(), "INPUT_TP", 0))
            and self.ndev == 8
            and os.environ.get("SGLANG_MIMO_FUSED_MOE_INPUT_TP", "1") == "1"
        )
        if self.input_tp:
            nbytes = int(F.ext().PRO_BYTES)
            ptr = lib.cudaMalloc(nbytes)
            lib.cudaMemset(
                ptr, 0, nbytes
            )  # epoch tags start at 1: the buffers must not hold a stale live tag
            self._ptrs.append(ptr)
            handle = lib.cudaIpcGetMemHandle(ptr)
            handles = _gather_objects(group, bytes(handle))
            ptrs = []
            for r, h in enumerate(handles):
                if r == self.rank:
                    ptrs.append(ptr.value)
                else:
                    peer = lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h))
                    ptrs.append(peer.value)
            self.state.link_pro_ptrs(ptrs, ptr.value)
        if os.environ.get("SGLANG_MIMO_FUSED_MOE_DG_PDL", "0") == "1":
            # Follow-on: launch DeepGEMM with programmatic dependent launch so the qkv GEMM that now directly follows the
            # fused MoE kernel (FMOE_LATE_TRIGGER primary) hides its launch gap. Global deep_gemm setting; opt-in.
            try:
                import deep_gemm

                deep_gemm.set_pdl(True)
                log_info_on_rank0(logger, "MiMo fused MoE: deep_gemm.set_pdl(True)")
            except Exception as e:  # never fatal
                logger.warning(f"MiMo fused MoE: deep_gemm.set_pdl failed: {e}")
        # SGLANG_MIMO_FUSED_MOE_L2_SETASIDE_MB = N raises the L2 persisting set-aside (cudaLimitPersistingL2CacheSize, default
        # 11.25 MB on H200, max 37.5) so the kernel's L2::evict_last prefetch of the next layer's o_proj weight (25.2 MB) is fully protected
        # until the o_proj GEMM reads it ~30 us later. Device-wide, set once per process (= per rank).
        mb = os.environ.get("SGLANG_MIMO_FUSED_MOE_L2_SETASIDE_MB", "")
        if mb:
            try:
                cudart = None
                for so in ("libcudart.so.12", "libcudart.so"):
                    try:
                        cudart = ctypes.CDLL(so)
                        break
                    except OSError:
                        continue
                if cudart is None:
                    raise OSError("libcudart not found")
                cudart.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
                cudart.cudaDeviceGetLimit.argtypes = [
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.c_int,
                ]
                rc = cudart.cudaDeviceSetLimit(
                    6, ctypes.c_size_t(int(float(mb) * 2**20))
                )  # cudaLimitPersistingL2CacheSize = 6
                cur = ctypes.c_size_t(0)
                cudart.cudaDeviceGetLimit(ctypes.byref(cur), 6)
                log_info_on_rank0(
                    logger,
                    f"MiMo fused MoE: L2 persisting set-aside request {mb} MB -> rc {rc}, now {cur.value / 2**20:.2f} MB",
                )
            except Exception as e:  # a cache-policy knob must never kill the server
                logger.warning(
                    f"MiMo fused MoE: cudaDeviceSetLimit(PersistingL2CacheSize) failed: {e}"
                )
        torch.cuda.synchronize(device)


def _shared_state(device: torch.device) -> _SharedRankState:
    global _shared
    if _shared is None:
        _shared = _SharedRankState(device)
    return _shared


def fused_moe_stream_residual(forward_batch) -> Optional[torch.Tensor]:
    """The residual the attention stage's prepare wrote into this forward's residual stream (borrowed, not modified by
    the kernel), or None when the stream is not in that written state."""
    stream = getattr(forward_batch, "residual_stream", None)
    if stream is None or stream.pending is not None:
        return None
    return stream.residual


def finish_fused_moe(out: torch.Tensor, residual_out: torch.Tensor, forward_batch) -> torch.Tensor:
    """Hand the fused kernel's results to the next stage boundary: ``residual_out`` (attention output + residual) becomes
    the written residual and ``out`` (the complete, all-reduced MoE output) the FFN contribution with a plain add. Returns
    the layer output to pass on (``out`` itself, so a norm-next handoff attached to it reaches the next layer)."""
    stream = residual_batch.stream_of(forward_batch)
    if stream.pending is not None:
        raise RuntimeError("fused MoE: the residual stream still holds an unconsumed contribution")
    stream.write(residual_out)
    return stream.record(out, PLAIN_ADD)


def take_norm_next_handoff(handoff, hidden_states, forward_batch, capture_gathered=None):
    """Consume the previous fused layer's norm-next handoff in place of ``attn_boundary.prepare``: the kernel already wrote
    residual_new = out + residual_out (bitwise flashinfer fused_add_rmsnorm's residual) and this layer's normalized fp8
    qkv_proj input. Returns that input as the (x_fp8, x_scale) pair."""
    stream = residual_batch.stream_of(forward_batch)
    stream.check(hidden_states)  # the pending contribution is the fused output carrying this handoff
    stream.write(handoff.residual_new)
    if capture_gathered is not None:
        # the residual the skipped prepare would have written and captured (aux hidden state for DFLASH); the buffer is
        # rewritten every step, so the accumulator copies it.
        capture_gathered.capture(handoff.residual_new, owned=False)
    return handoff.x_fp8, handoff.x_scale


class NormNextHandoff:
    """Attached to the fused MoE's returned hidden_states (``hidden_states._fmoe_norm_next``) when the kernel's TP tail already
    ran the NEXT decoder layer's input_layernorm + fp8 per-128 activation quant (bitwise flashinfer fused_add_rmsnorm + SGLang
    per_token_group_quant in the deep_gemm layout). Layer ``layer_id`` then skips attn_boundary.prepare (the stream's residual =
    residual_new, see take_norm_next_handoff) and calls qkv_proj((x_fp8, x_scale)); the DFLASH aux capture takes residual_new."""

    __slots__ = ("layer_id", "residual_new", "x_fp8", "x_scale")

    def __init__(
        self,
        layer_id: int,
        residual_new: torch.Tensor,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ):
        self.layer_id = layer_id
        self.residual_new = residual_new
        self.x_fp8 = x_fp8
        self.x_scale = x_scale


def _norm_next_check(decoder_layer) -> Optional[str]:
    """None when the fused kernel of ``decoder_layer`` may produce the NEXT layer's qkv_proj input (residual_new, x_fp8, scales), else why not.
    The row stage replicates flashinfer's FusedAddRMSNormKernel (bf16, weight_bias 0, eps) and the deep_gemm-layout per-token-group-128 fp8
    quant, so the next layer must run exactly that pair: a plain RMSNorm (no variance override / HF cast) and a block-quant [128, 128]
    Fp8LinearMethod qkv_proj dispatched to DeepGEMM."""
    nxt = getattr(decoder_layer, "_fmoe_next_layer", None)
    if nxt is None:
        return "no next layer"
    if os.environ.get("SGLANG_MIMO_FUSED_MOE_NORM_NEXT", "1") != "1":
        return "disabled by SGLANG_MIMO_FUSED_MOE_NORM_NEXT=0"
    if not getattr(F.ext(), "NORM_NEXT", 0):
        return "kernel built without FMOE_NORM_NEXT / FMOE_INPUT_TP"
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    norm = getattr(nxt, "input_layernorm", None)
    if (
        type(norm) is not RMSNorm
        or getattr(norm, "variance_size_override", None) is not None
        or getattr(norm, "cast_x_before_out_mul", False)
        or getattr(norm, "fp32_residual", False)
        or getattr(norm, "override_orig_dtype", None) is not None
    ):
        return f"next input_layernorm {type(norm).__name__} is not a plain RMSNorm"
    if (
        norm.weight.dtype != torch.bfloat16
        or norm.weight.numel() != F.DIM
        or not norm.weight.data.is_contiguous()
    ):
        return f"next input_layernorm weight {norm.weight.dtype} {norm.weight.numel()}"
    if (
        abs(
            float(norm.variance_epsilon) - float(decoder_layer.config.layernorm_epsilon)
        )
        > 0
    ):
        return f"next input_layernorm eps {norm.variance_epsilon}"
    try:
        target = norm.dispatch_forward()  # static dispatch target on this platform (forced backends / OOT overrides included)
    except Exception as e:  # pragma: no cover - defensive
        return f"next input_layernorm dispatch_forward failed: {e}"
    if target != norm.forward_cuda:
        return f"next input_layernorm dispatches to {getattr(target, '__name__', target)} (need forward_cuda -> flashinfer fused_add_rmsnorm)"
    qkv = getattr(getattr(nxt, "self_attn", None), "qkv_proj", None)
    qm = getattr(qkv, "quant_method", None)
    if (
        not isinstance(qm, Fp8LinearMethod)
        or not qm.block_quant
        or getattr(qm, "use_mxfp8", False)
        or getattr(qm, "block_fp8_as_mxfp8", False)
        or qm.use_marlin
    ):
        return f"next qkv_proj quant method {type(qm).__name__} (need block-quant Fp8LinearMethod)"
    if (
        list(qm.weight_block_size) != [128, 128]
        or qm.w8a8_block_fp8_linear is not deepgemm_w8a8_block_fp8_linear_with_fallback
    ):
        return f"next qkv_proj block {qm.weight_block_size} / linear {getattr(qm.w8a8_block_fp8_linear, '__name__', None)} (need [128,128] + DeepGEMM)"
    if (
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        or not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
    ):
        return "DeepGEMM disabled or UE8M0 scales"
    w = getattr(qkv, "weight", None)
    if (
        w is None
        or w.dtype != torch.float8_e4m3fn
        or w.shape[1] != F.DIM
        or w.shape[0] % 64 != 0
    ):
        return (
            f"next qkv_proj weight {None if w is None else (w.dtype, tuple(w.shape))}"
        )
    if getattr(qkv, "skip_bias_add", False) or getattr(qkv, "gather_output", False):
        return "next qkv_proj skip_bias_add / gather_output"
    return None


class MiMoFusedMoE:
    """The fused path of one MiMoV2DecoderLayer. Construct through ``try_create`` (collective)."""

    def __init__(self, decoder_layer, shared: _SharedRankState):
        self.shared = shared
        config = decoder_layer.config
        experts = decoder_layer.mlp.experts
        self.w = F.humming_layer_weights(experts)
        self.norm_w = decoder_layer.post_attention_layernorm.weight.data
        self.router_w = decoder_layer.mlp.gate.weight.data.contiguous()
        self.bias = decoder_layer.mlp.gate.e_score_correction_bias.data.contiguous()
        self.eps = float(config.layernorm_epsilon)
        self.max_tokens = min(F.MAXM, int(envs.SGLANG_MIMO_FUSED_MOE_MAX_TOKENS.get()))
        self.ffn_plan = decoder_layer.ffn_boundary.plan
        # SGLANG_MIMO_FUSED_MOE_PREFETCH = comma list of {qkv, o}: the kernel (built with -DFMOE_TAIL_PREFETCH=1..3) L2-prefetches the
        # NEXT layer's qkv_proj weight (+ block scales) and/or o_proj weight in its TP tail. Cache hint only: no dependency, no bit change.
        # The weights are their final (post process_weights_after_loading) tensors here: try_create runs at the first forward.
        # SGLANG_MIMO_FUSED_MOE_PREFETCH_HINT = comma list of {qkv, o}: those ranges are prefetched with an L2::evict_last hint (the o_proj
        # weight is read ~30 us later, after the qkv weight stream and attention have passed through L2).
        self.pf = None
        pf_mode = os.environ.get("SGLANG_MIMO_FUSED_MOE_PREFETCH", "o")
        pf_hint = os.environ.get("SGLANG_MIMO_FUSED_MOE_PREFETCH_HINT", "o")
        nxt = getattr(decoder_layer, "_mimo_next_layer", None)
        if pf_mode and nxt is not None and getattr(F.ext(), "TAIL_PREFETCH", 0):
            attn = getattr(nxt, "self_attn", None)
            ranges: List[Any] = []
            hints: List[bool] = []
            if attn is not None:
                if "qkv" in pf_mode:
                    ranges += [
                        getattr(attn.qkv_proj, "weight", None),
                        getattr(attn.qkv_proj, "weight_scale_inv", None),
                    ]
                    hints += ["qkv" in pf_hint] * 2
                if "o" in pf_mode.replace("qkv", ""):
                    ranges.append(getattr(attn.o_proj, "weight", None))
                    hints.append("o" in pf_hint.replace("qkv", ""))
            desc = F.pf_desc(ranges, hints)
            if int(desc[1]) > 0:
                self.pf = desc
                if shared.rank == 0:
                    logger.info(
                        "MiMo fused MoE layer %d: tail L2 prefetch of layer %d attention weights: %s bytes, evict_last mask %d",
                        decoder_layer.layer_id,
                        nxt.layer_id,
                        [int(desc[2 * k + 1]) for k in range(3)],
                        int(desc[6]),
                    )
        self.dbg = None
        # the next layer's input_layernorm + activation quant inside the kernel (M = 32 input-TP path only)
        self.norm_next = None
        reason = (
            _norm_next_check(decoder_layer) if shared.input_tp else "no input-TP kernel"
        )
        if reason is None:
            nxt = decoder_layer._fmoe_next_layer
            bufs = F.norm_next_buffers(self.router_w.device, 32)
            self.norm_next = dict(
                norm_w=nxt.input_layernorm.weight.data,
                eps=float(nxt.input_layernorm.variance_epsilon),
                **bufs,
            )
            self.norm_next_handoff = NormNextHandoff(
                nxt.layer_id, bufs["res_new"], bufs["xq"], bufs["xs"]
            )
        else:
            log_info_on_rank0(
                logger,
                f"MiMo fused MoE layer {decoder_layer.layer_id}: norm-next off ({reason})",
            )
        if _STAMP_DIR:
            global _stamp_thread
            # one buffer per token count: every captured batch-size graph keeps its own last-replay stamps
            self.dbg = torch.zeros(
                F.MAXM,
                _STAMP_GRID * F.NSTAMP,
                dtype=torch.int64,
                device=self.router_w.device,
            )
            _stamp_layers.append((decoder_layer.layer_id, self.dbg))
            if _stamp_thread is None:
                _stamp_thread = threading.Thread(
                    target=_stamp_watcher, args=(shared.rank,), daemon=True
                )
                _stamp_thread.start()
        self.cap = None
        if _CAP_DIR and shared.input_tp:
            global _cap_thread
            dev = self.router_w.device
            self.cap = dict(
                partial=torch.zeros(_CAP_RING, 32, F.DIM, dtype=torch.bfloat16, device=dev),
                residual=torch.zeros(_CAP_RING, 32, F.DIM, dtype=torch.bfloat16, device=dev),
                topk=torch.full((_CAP_RING, 32, F.TOPK), -1, dtype=torch.int64, device=dev),
                count=torch.zeros(1, dtype=torch.int64, device=dev),
            )
            _cap_layers.append((decoder_layer.layer_id, self.cap))
            if _cap_thread is None:
                _cap_thread = threading.Thread(
                    target=_cap_watcher, args=(shared.rank,), daemon=True
                )
                _cap_thread.start()

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _check(decoder_layer) -> Optional[str]:
        """None when this layer can run the fused kernel, otherwise the reason."""
        cfg = decoder_layer.config
        par = get_parallel()
        if par.tp_size < 2 or F.N_FC2_TILES % par.tp_size != 0 or par.tp_size > 8:
            return f"tp_size={par.tp_size} (need 2..8 dividing 48)"
        if (
            par.moe_ep_size != 1
            or par.attn_dp_size != 1
            or par.attn_tp_size != par.tp_size
        ):
            return f"ep={par.moe_ep_size} attn_dp={par.attn_dp_size} attn_tp={par.attn_tp_size} (need plain TP)"
        if getattr(cfg, "hidden_size", None) != F.DIM:
            return f"hidden_size={getattr(cfg, 'hidden_size', None)}"
        if getattr(cfg, "moe_intermediate_size", 0) // par.tp_size != F.INTER:
            return f"moe_intermediate_size/tp={getattr(cfg, 'moe_intermediate_size', 0)}/{par.tp_size}"
        if (
            getattr(cfg, "n_routed_experts", None) != F.NEXP
            or getattr(cfg, "num_experts_per_tok", None) != F.TOPK
        ):
            return f"experts={getattr(cfg, 'n_routed_experts', None)} top_k={getattr(cfg, 'num_experts_per_tok', None)}"
        if getattr(cfg, "n_group", 1) != 1 or getattr(cfg, "topk_group", 1) != 1:
            return "grouped top-k not supported"
        if (
            getattr(cfg, "scoring_func", "sigmoid") != "sigmoid"
            or getattr(cfg, "topk_method", None) != "noaux_tc"
        ):
            return f"scoring_func={getattr(cfg, 'scoring_func', None)} topk_method={getattr(cfg, 'topk_method', None)}"
        if not getattr(cfg, "norm_topk_prob", False):
            return "norm_topk_prob=False"
        if getattr(cfg, "n_shared_experts", None) not in (None, 0):
            return "shared experts not supported"
        if getattr(cfg, "routed_scaling_factor", None) not in (None, 1, 1.0):
            return f"routed_scaling_factor={cfg.routed_scaling_factor}"
        if getattr(cfg, "hidden_act", "silu") != "silu":
            return f"hidden_act={cfg.hidden_act}"
        # stage boundaries: the attention output owes exactly its attention-TP sum (no transform), and the FFN writes a plain
        # residual add with no fused finalize -- the contract finish_fused_moe reproduces with a complete output.
        attn_boundary = getattr(decoder_layer, "attn_boundary", None)
        ffn_boundary = getattr(decoder_layer, "ffn_boundary", None)
        if attn_boundary is None or ffn_boundary is None:
            return "no stage boundaries"
        attn_path = attn_boundary.plan.paths.get(BatchVariant.ORDINARY)
        if (
            attn_path is None
            or attn_path.output.group is not SumGroup.ATTN_TP
            or not attn_path.output.always_partial
            or attn_path.output.transform is not None
            or not attn_path.output.update.is_plain_add
        ):
            return "attention output is not a declared attention-TP partial sum with a plain add"
        ffn_decl = ffn_boundary.declaration
        if (
            not ffn_decl.update.is_plain_add
            or ffn_decl.output_transform is not None
            or ffn_boundary.plan.fusions is not None
            or ffn_boundary.norm is not decoder_layer.post_attention_layernorm
        ):
            return "FFN boundary is not a plain residual add + post_attention_layernorm"
        mlp = decoder_layer.mlp
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        if (
            gate is None
            or experts is None
            or getattr(gate, "e_score_correction_bias", None) is None
        ):
            return "no gate / experts / correction bias"
        router_bf16 = bool(getattr(F.ext(), "ROUTER_BF16", 0))
        router_dtype = torch.bfloat16 if router_bf16 else torch.float32
        if gate.weight.dtype != router_dtype or tuple(gate.weight.shape) != (
            F.NEXP,
            F.DIM,
        ):
            return f"gate weight {gate.weight.dtype} {tuple(gate.weight.shape)}"
        if (
            gate.e_score_correction_bias.dtype != torch.float32
            or gate.e_score_correction_bias.numel() != F.NEXP
        ):
            return f"correction bias {gate.e_score_correction_bias.dtype}"
        nw = decoder_layer.post_attention_layernorm.weight
        if nw.dtype != torch.bfloat16 or nw.numel() != F.DIM:
            return f"post_attention_layernorm weight {nw.dtype} {nw.numel()}"
        metas = getattr(experts, "humming_metas", None)
        if not isinstance(metas, dict) or "w13" not in metas or "w2" not in metas:
            return "experts are not a Humming MoE layer (need --moe-runner-backend humming)"
        try:
            from humming import dtypes
            from humming.config import MmaType
        except ImportError as e:
            return f"humming import failed: {e}"
        for name in ("w13", "w2"):
            m = metas[name]
            if (
                m.mma_type != MmaType.WGMMA
                or not m.use_fused_e8m0_scale
                or m.b_dtype != dtypes.float4e2m1
                or m.bs_dtype != dtypes.float8e8m0
            ):
                return f"{name}: mma={m.mma_type} fused_e8m0={m.use_fused_e8m0_scale} b={m.b_dtype} bs={m.bs_dtype} (need sm90 wgmma e2m1/e8m0 fused-e8m0)"
            if m.a_dtype != dtypes.float8e4m3 or m.input_scale_group_size != 128:
                return f"{name}: a_dtype={m.a_dtype} group={m.input_scale_group_size} (need W4A8: SGLANG_HUMMING_INPUT_QUANT_CONFIG fp8 g128)"
            if getattr(m, "use_packed_k_layout", False) or getattr(
                m, "use_native_dequant", False
            ):
                return f"{name}: unexpected weight layout flags"
            if getattr(m, "weight_scale_group_size", 32) != 32:
                return f"{name}: weight_scale_group_size={m.weight_scale_group_size}"
        want = {
            "w13_weight": (torch.int32, (F.NEXP, F.DIM // 32, 4 * 2 * F.INTER)),
            "w13_weight_scale": (
                torch.float8_e8m0fnu,
                (F.NEXP, F.DIM // 32, 2 * F.INTER),
            ),
            "w2_weight": (torch.int32, (F.NEXP, F.INTER // 32, 4 * F.DIM)),
            "w2_weight_scale": (torch.float8_e8m0fnu, (F.NEXP, F.INTER // 32, F.DIM)),
        }
        for name, (dt, shape) in want.items():
            t = getattr(experts, name, None)
            if (
                t is None
                or t.dtype != dt
                or tuple(t.shape) != shape
                or not t.is_contiguous()
            ):
                return f"{name}: {None if t is None else (t.dtype, tuple(t.shape))} (want {dt} {shape})"
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            t = getattr(experts, name, None)
            if t is None or t.dtype != torch.float32 or t.numel() != F.NEXP:
                return f"{name}: {None if t is None else (t.dtype, tuple(t.shape))} (want fp32 [{F.NEXP}])"
        if torch.cuda.get_device_capability(gate.weight.device) != (9, 0):
            return "needs sm90"
        return None

    @classmethod
    def try_create(cls, decoder_layer):
        """Collective over the TP group: returns a MiMoFusedMoE when EVERY rank can run it, else None (and logs why)."""
        reason = cls._check(decoder_layer)
        group = get_tp_group()
        reasons = _gather_objects(group, reason)
        bad = [(r, why) for r, why in enumerate(reasons) if why is not None]
        if bad:
            log_info_on_rank0(
                logger,
                f"MiMo fused MoE disabled for layer {decoder_layer.layer_id}: rank {bad[0][0]}: {bad[0][1]}",
            )
            return None
        shared = _shared_state(decoder_layer.mlp.gate.weight.device)
        self = cls(decoder_layer, shared)
        log_info_on_rank0(
            logger,
            f"MiMo fused MoE enabled for layer {decoder_layer.layer_id} (TP{shared.ndev}, <= {self.max_tokens} tokens, router={self.router_w.dtype}, "
            f"input_tp={shared.input_tp}, norm_next={self.norm_next is not None})",
        )
        return self

    # ------------------------------------------------------------------ forward
    def applicable(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch,
    ) -> bool:
        if (
            residual is None
            or hidden_states.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
        ):
            return False
        m = hidden_states.shape[0]
        if m < 1 or m > self.max_tokens or hidden_states.shape[-1] != F.DIM:
            return False
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return False
        # ordinary rows only: no sequence-parallel, input-scattered or context-parallel batch layout
        if self.ffn_plan.variant_for(forward_batch) is not BatchVariant.ORDINARY:
            return False
        if getattr(forward_batch, "can_run_tbo", False):
            return False
        if torch.compiler.is_compiling():
            return False
        return True

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        """hidden_states: this rank's o_proj partial (what prepare_mlp would all-reduce); returns (moe_out, hidden + residual)."""
        if (
            self.shared.input_tp
            and hidden_states.shape[0] == 32
            and self.router_w.dtype == torch.bfloat16
        ):
            # no separate all-reduce kernel -- the fused kernel takes the un-reduced partial and reduces it in its prologue
            # (bitwise the CustomAllReduceV2 result; routing and outputs are bitwise those of the all-reduce + kernel path).
            partial = (
                hidden_states
                if hidden_states.is_contiguous()
                else hidden_states.contiguous()
            )
            if not residual.is_contiguous():
                residual = residual.contiguous()
            out = torch.empty_like(partial)
            residual_out = torch.empty_like(residual)
            if self.cap is not None:
                # inputs before the launch (the kernel may stage through them); top-8 ids after it
                slot = self.cap["count"] % _CAP_RING
                self.cap["partial"].index_copy_(0, slot, partial.unsqueeze(0))
                self.cap["residual"].index_copy_(0, slot, residual.unsqueeze(0))
            inp = dict(
                partial=partial,
                residual=residual,
                norm_w=self.norm_w,
                router_w=self.router_w,
                router_w_bf16=self.router_w,
                bias=self.bias,
            )
            F.run_fused_full_tp(
                self.shared.state,
                inp,
                self.w,
                out,
                residual_out,
                self.shared.rank,
                self.shared.ndev,
                eps=self.eps,
                dbg=None if self.dbg is None else self.dbg[31],
                norm_next=self.norm_next,
                pf=self.pf,
            )
            if self.cap is not None:
                ids, _ = self.shared.state.kernel_topk(32)
                self.cap["topk"].index_copy_(0, slot, ids.unsqueeze(0))
                self.cap["count"] += 1
            if self.norm_next is not None:
                # the kernel also produced the next layer's residual_new / x_fp8 / scales (fixed per-layer buffers, CUDA-graph safe);
                # MiMoV2DecoderLayer.forward of layer_id + 1 consumes them and skips its input_layernorm + activation quant.
                out._fmoe_norm_next = self.norm_next_handoff
            return out, residual_out
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        out = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(residual)
        inp = dict(
            hidden=hidden_states,
            residual=residual,
            norm_w=self.norm_w,
            router_w=self.router_w,
            bias=self.bias,
        )
        if self.router_w.dtype == torch.bfloat16:
            # Reuse SGLang's already rounded, loaded parameter. Never cast
            # router weights inside a forward or a captured CUDA graph.
            inp["router_w_bf16"] = self.router_w
        F.run_fused_full(
            self.shared.state,
            inp,
            self.w,
            out,
            residual_out,
            self.shared.rank,
            self.shared.ndev,
            eps=self.eps,
            dbg=None if self.dbg is None else self.dbg[hidden_states.shape[0] - 1],
        )
        return out, residual_out
