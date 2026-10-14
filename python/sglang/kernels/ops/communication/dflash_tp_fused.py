"""TP8 decode fusions for the DFlash draft (csrc/distributed/dflash_tp_fused.cuh).

``DFlashTPState`` owns this rank's CUDA-IPC receive buffer and the peers' mapped addresses (one collective at
creation over the TP group's CPU group; every rank must create it at the same point) and the per-token call
counters. ``arnq`` = reduce-scatter + residual add (residual [M, H] replicated, updated in place) + RMSNorm + per-token FP8
quant (or bf16) + all-gather of a row-parallel projection's partial output, in one kernel (two NVLink hops).
"""

import functools
from typing import List, Optional

import torch

H = 6144
R = 8
SL = H // R
MAXT = 32
BUF_BYTES = 2 * MAXT * R * SL * 2 + 2 * MAXT * R * 8 + 2 * MAXT * R * (SL * 2 + 16) + 2 * MAXT * R * 16 + 2 * MAXT * H * 2


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    return load_jit(
        "dflash_tp_fused",
        *args,
        cuda_files=["distributed/dflash_tp_fused.cuh"],
        cuda_wrappers=[
            ("arnq", f"DFlashTPFused<{args}>::arnq"),
            ("silu_mul_quant", f"DFlashTPFused<{args}>::silu_mul_quant"),
            ("rmsnorm_quant", f"DFlashTPFused<{args}>::rmsnorm_quant"),
            ("argmax", f"DFlashTPFused<{args}>::argmax"),
            ("embed", f"DFlashTPFused<{args}>::embed"),
            ("ag_norm", f"DFlashTPFused<{args}>::ag_norm"),
            ("embed_nq", f"DFlashTPFused<{args}>::embed_nq"),
        ],
        extra_cuda_cflags=["-O3"] + (["-DDFLASH_TP_STAMP_EDGES=1"] if _stamp_edges() else []),
    )


def _stamp_edges() -> bool:
    import os

    return os.environ.get("SGLANG_MIMO_DENSE_TP_STAMPS", "0") == "1"  # diagnostics


@functools.lru_cache(maxsize=None)
def _pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return is_arch_support_pdl()


def _gather_objects(cpu_group, ranks: List[int], my_index: int, obj):
    import torch.distributed as dist

    data = [[None] for _ in ranks]
    data[my_index][0] = obj
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(data[i], src=r, group=cpu_group, device="cpu")
    return [d[0] for d in data]


class DFlashTPState:
    """Per-process IPC receive buffer + call counters. ``group``: an object with ``cpu_group``, ``ranks``,
    ``rank_in_group`` and ``world_size`` (SGLang GroupCoordinator) -- or pass them explicitly."""

    def __init__(self, device: torch.device, cpu_group, ranks: List[int], rank: int):
        from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        assert len(ranks) == R, f"dflash_tp_fused needs TP{R}, got {len(ranks)}"
        self.rank = rank
        self.device = device
        lib = CudaRTLibrary()
        self._lib = lib
        ptr = lib.cudaMalloc(BUF_BYTES)
        lib.cudaMemset(ptr, 0, BUF_BYTES)
        self._ptr = ptr
        handle = lib.cudaIpcGetMemHandle(ptr)
        handles = _gather_objects(cpu_group, ranks, rank, bytes(handle))
        addrs = []
        for r, h in enumerate(handles):
            if r == rank:
                addrs.append(ptr.value)
            else:
                addrs.append(lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h)).value)
        self.peers = torch.tensor(addrs, dtype=torch.int64)  # host tensor
        self.calls = torch.zeros(MAXT, dtype=torch.int32, device=device)
        self.acalls = torch.zeros(MAXT, dtype=torch.int32, device=device)
        self.ecalls = torch.zeros(MAXT, dtype=torch.int32, device=device)
        self.stamps = None  # diagnostics: int64 cuda tensor [grid * 16] (indexed blockIdx.x * 16 + k)
        torch.cuda.synchronize(device)

    def arnq(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        out_scale: Optional[torch.Tensor],
    ):
        """out e4m3 [M, H] (+ out_scale [M] fp32) -> mode 0; out bf16 [M, H] -> mode 1 (out_scale ignored)."""
        mode = 0 if out.dtype == torch.float8_e4m3fn else 1
        if out_scale is None:
            out_scale = self.calls  # unused placeholder (mode 1)
        _module(_pdl()).arnq(
            partial, residual, gamma, out.view(torch.uint8) if mode == 0 else out, out_scale, self.calls,
            self.peers, float(eps), self.rank, mode, 0 if self.stamps is None else self.stamps.data_ptr(), 0,
        )


    def arnq_g128(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out_q: torch.Tensor,
        out_scale_t: torch.Tensor,
    ):
        """arnq with per-token-group-128 fp8 output in the DeepGEMM pre-quantized layout: out_q [>= M, H] e4m3,
        out_scale_t [H/128, ld >= M] fp32 (out_scale_t.t() is the column-major [ld, H/128] scale view)."""
        _module(_pdl()).arnq(
            partial, residual, gamma, out_q.view(torch.uint8), out_scale_t, self.calls,
            self.peers, float(eps), self.rank, 2, 0 if self.stamps is None else self.stamps.data_ptr(), 0,
        )

    def argmax(self, logits: torch.Tensor, out: torch.Tensor, vocab_start: int, group: int = 0) -> None:
        """Vocab-parallel greedy argmax of this rank's logits shard [n <= 32, Vs] -> out int64 [n] (global ids, bit-exact
        torch.argmax of the full row), local argmax + IPC exchange + select in one kernel. group > 1: logits hold blocks
        of `group` rows and only rows 1..group-1 of each block are reduced (DFlash draft head, no slice copy)."""
        # the trailing (0, 0): no candidate-buffer write
        _module(_pdl()).argmax(logits, out, self.acalls, self.peers, int(vocab_start), self.rank, int(group), 0, 0)


    def ag_norm(self, x_slice: torch.Tensor, gamma: torch.Tensor, eps: float, out: torch.Tensor) -> None:
        """All-gather of a column-parallel output slice [M, 768] + RMSNorm -> out [M, 6144] bf16 (one NVLink hop;
        shares the all-gather plane and call counters with arnq)."""
        _module(_pdl()).ag_norm(x_slice, gamma, out, self.calls, self.peers, float(eps), self.rank)

    def embed(
        self, ids: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, ranges: torch.Tensor, local: bool = False
    ) -> None:
        """Vocab-parallel embedding [M <= 32] without the all-reduce: the owner of each row pushes it to every peer.
        ranges: host int64 [2 * R] = every rank's [start, end) org vocab range.
        local: weight is the full replicated table and ranges gives this rank [0, vocab): a local gather."""
        _module(_pdl()).embed(ids, weight, out, self.ecalls, self.peers, ranges, self.rank, 1 if local else 0)


    def embed_nq(
        self,
        ids: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        ranges: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out_q: torch.Tensor,
        out_scale_t: torch.Tensor,
        local: bool = False,
    ) -> None:
        """embed() + the first layer's input RMSNorm + group-128 fp8 quant (DeepGEMM pre-quantized layout): out [M, H]
        bf16 embedding rows, out_q [>= M, H] e4m3, out_scale_t [H/128, ld >= M] fp32 (bit-exact sgl rmsnorm + group quant)."""
        _module(_pdl()).embed_nq(
            ids, weight, out, out_q.view(torch.uint8), out_scale_t, gamma, float(eps), self.ecalls, self.peers, ranges, self.rank,
            1 if local else 0,
        )


_STATE: Optional[DFlashTPState] = None
_STATE_FAILED = False


def get_state(group=None) -> Optional[DFlashTPState]:
    """Process-wide state over the TP group (collective on the first call: every rank must call it at the same
    point, outside CUDA-graph capture). None when unavailable (not TP8, capture, IPC failure)."""
    global _STATE, _STATE_FAILED
    if _STATE is not None or _STATE_FAILED:
        return _STATE
    try:
        if group is None:
            from sglang.srt.distributed import get_tp_group

            group = get_tp_group()
        if group.world_size != R or torch.cuda.is_current_stream_capturing():
            _STATE_FAILED = True
            return None
        dev = torch.device("cuda", torch.cuda.current_device())
        _STATE = DFlashTPState(dev, group.cpu_group, list(group.ranks), group.rank_in_group)
    except Exception:
        _STATE_FAILED = True
        _STATE = None
    return _STATE


def silu_mul_quant(gate_up: torch.Tensor, out: torch.Tensor, scale: torch.Tensor):
    _module(_pdl()).silu_mul_quant(gate_up, out.view(torch.uint8), scale)


def rmsnorm_quant(x: torch.Tensor, gamma: torch.Tensor, eps: float, rank: int, out: torch.Tensor, scale: torch.Tensor):
    _module(_pdl()).rmsnorm_quant(x, gamma, out.view(torch.uint8), scale, float(eps), rank)


@functools.lru_cache(maxsize=None)
def _module_act(use_pdl: bool):
    from sglang.kernels.jit.utils import get_activation_cuda_cflags, load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    # same flags as the JIT activation kernel it replaces (bit-exact SiLU)
    return load_jit(
        "silu_mul_group_quant",
        *args,
        cuda_files=["elementwise/silu_mul_group_quant.cuh"],
        cuda_wrappers=[("run", f"SiluMulGroupQuant<{args}>::run")],
        extra_cuda_cflags=["-O3", *get_activation_cuda_cflags()],
    )


def silu_mul_group_quant(gate_up: torch.Tensor, out_q: torch.Tensor, out_scale_t: torch.Tensor) -> None:
    """SiluAndMul + per-token-group-128 fp8 quant (DeepGEMM pre-quantized layout) in one kernel, bit-exact with
    silu_and_mul + sglang_per_token_group_quant_fp8(column_major_scales=True, scale_tma_aligned=True).
    gate_up [M, 2I] bf16 -> out_q [>= M, I] e4m3, out_scale_t [I/128, ld >= M] fp32 (pass out_scale_t.t() to the GEMM)."""
    _module_act(_pdl()).run(gate_up, out_q.view(torch.uint8), out_scale_t)
