"""Vocab-parallel DFlash sampling verify (csrc/distributed/sample_verify.cuh), TP8, <= 64 verify rows.

One kernel on each rank's bf16 logits shard [n, Vs] does temperature + softmax + top-p + the chain rejection sampling
of tree_speculative_sampling_target_only + the bonus draw, exchanging only per-row statistics over CUDA IPC (no
all-gather of the logits, no host sync, no rank-0 broadcast), and writes `predict` [n] int64 in the greedy-verify
layout (accepted rows -> their candidate, first rejected row -> the bonus, later rows -> -1) so the greedy accept
kernel turns it into accept_len / bonus / commit lengths unchanged. Greedy requests get the vocab-parallel argmax.

``SampleVerifyState`` owns this rank's receive buffer and the peers' mapped addresses (one collective at creation over
the TP group's CPU group; every rank must create it at the same point), the per-row call counters and the static
per-request parameter buffers that a captured graph reads (``stage``).
"""

import functools
from typing import List, Optional, Sequence, Tuple

import torch

R = 8
MAXN = 64
MAXG = 16
NB = 256
NKEY = 65536
_R1_SLOT = (3 + NB // 2) * 16
_R2_SLOT = (NB // 2) * 16
BUF_BYTES = 2 * MAXN * R * _R1_SLOT + 2 * MAXN * R * _R2_SLOT + 2 * MAXN * 16
DBG = 32


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    return load_jit(
        "dflash_sample_verify",
        *args,
        cuda_files=["distributed/sample_verify.cuh"],
        cuda_wrappers=[("run", f"SampleVerify<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


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


class SampleVerifyState:
    def __init__(
        self,
        device: torch.device,
        cpu_group,
        ranks: List[int],
        rank: int,
        *,
        local_vocab: int,
        vocab_start: int,
        vocab: int,
        seed: int,
    ):
        from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        assert len(ranks) == R, f"sample_verify needs TP{R}, got {len(ranks)}"
        self.rank = int(rank)
        self.device = device
        self.local_vocab = int(local_vocab)
        self.vocab_start = int(vocab_start)
        self.vocab = int(vocab)
        self.seed = int(seed) & ((1 << 63) - 1)
        lib = CudaRTLibrary()
        self._lib = lib
        ptr = lib.cudaMalloc(BUF_BYTES)
        lib.cudaMemset(ptr, 0, BUF_BYTES)
        self._ptr = ptr
        handle = lib.cudaIpcGetMemHandle(ptr)
        handles = _gather_objects(cpu_group, ranks, self.rank, bytes(handle))
        addrs = []
        for r, h in enumerate(handles):
            if r == self.rank:
                addrs.append(ptr.value)
            else:
                addrs.append(lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h)).value)
        self.peers = torch.tensor(addrs, dtype=torch.int64)  # host tensor
        self.calls = torch.zeros(MAXN, dtype=torch.int32, device=device)
        self.flags = torch.zeros(MAXN, dtype=torch.int64, device=device)
        # static per-request parameters read by the kernel (a captured graph reads these addresses)
        self.temp = torch.ones(MAXN, dtype=torch.float32, device=device)
        self.top_p = torch.ones(MAXN, dtype=torch.float32, device=device)
        self.mode = torch.zeros(MAXN, dtype=torch.int32, device=device)
        self.out = torch.zeros(MAXN, dtype=torch.int64, device=device)
        self._staged_key = None
        torch.cuda.synchronize(device)

    # ---- per-request parameters ----
    def stage(self, params: Sequence[Tuple[int, float, float]]) -> None:
        """params[b] = (mode, temperature, top_p), mode 0 greedy / 1 sampled. Requests beyond len(params) (graph
        padding) are greedy. Copies only when the parameters changed (one small H2D copy on the current stream)."""
        key = tuple(params)
        if key == self._staged_key:
            return
        nb = len(params)
        assert nb <= MAXN
        mode = torch.zeros(MAXN, dtype=torch.int32)
        temp = torch.ones(MAXN, dtype=torch.float32)
        top_p = torch.ones(MAXN, dtype=torch.float32)
        if nb:
            mode[:nb] = torch.tensor([int(p[0]) for p in params], dtype=torch.int32)
            temp[:nb] = torch.tensor([float(p[1]) for p in params], dtype=torch.float32)
            top_p[:nb] = torch.tensor([float(p[2]) for p in params], dtype=torch.float32)
        self.mode.copy_(mode.pin_memory(), non_blocking=True)
        self.temp.copy_(temp.pin_memory(), non_blocking=True)
        self.top_p.copy_(top_p.pin_memory(), non_blocking=True)
        self._staged_key = key

    def stage_all_greedy(self, nb: int) -> None:
        self.stage([(0, 1.0, 1.0)] * nb)

    # ---- the kernel ----
    def run(
        self,
        logits: torch.Tensor,
        cand: torch.Tensor,
        group: int,
        out: Optional[torch.Tensor] = None,
        coins: Optional[torch.Tensor] = None,
        dbg: Optional[torch.Tensor] = None,
        stamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """logits: this rank's [n, Vs] bf16 shard; cand: [n] int64 verify input ids (block-major). Returns predict
        [n] int64 (written into ``out`` or the state's static buffer). coins (tests): fp32 [n/G * G + n/G]."""
        n = int(logits.shape[0])
        assert logits.ndim == 2 and int(logits.shape[1]) == self.local_vocab and logits.dtype == torch.bfloat16
        assert 1 <= n <= MAXN and n % group == 0 and 2 <= group <= MAXG
        assert cand.dtype == torch.int64 and cand.is_contiguous() and cand.numel() >= n
        if out is None:
            out = self.out
        _module(_pdl()).run(
            logits,
            cand,
            self.temp,
            self.top_p,
            self.mode,
            out,
            self.calls,
            self.flags,
            self.peers,
            self.vocab_start,
            self.vocab,
            self.rank,
            int(group),
            self.seed,
            0 if coins is None else coins.data_ptr(),
            0 if dbg is None else dbg.data_ptr(),
            0 if stamps is None else stamps.data_ptr(),
        )
        return out[:n]


def smem_bytes(local_vocab: int) -> int:
    """Dynamic shared memory of one CTA (csrc ``smem_bytes``): the shard row, the key count table, the histograms."""
    return ((local_vocab * 2 + 15) // 16) * 16 + NKEY * 2 + (2 * R * NB + NB) * 4


def unsupported_reason(local_vocab: int, device: Optional[torch.device] = None) -> Optional[str]:
    """None when the kernel can run on this device for a logits shard of ``local_vocab`` columns, else the reason."""
    if local_vocab % 8 != 0:
        return f"local vocab {local_vocab} not a multiple of 8"
    if local_vocab > NKEY // 2:  # the bonus scan reuses the count table for the shard's masses
        return f"local vocab {local_vocab} > {NKEY // 2}"
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    limit = min(220 * 1024, int(getattr(props, "shared_memory_per_block_optin", 0)))
    if smem_bytes(local_vocab) > limit:
        return f"needs {smem_bytes(local_vocab)} B of shared memory per CTA (device limit {limit} B)"
    return None


def create_state(group, *, local_vocab: int, vocab_start: int, vocab: int, seed: int) -> Optional[SampleVerifyState]:
    """Collective over the TP group (every rank, same point, outside graph capture). None if not TP8."""
    if group.world_size != R or torch.cuda.is_current_stream_capturing():
        return None
    dev = torch.device("cuda", torch.cuda.current_device())
    return SampleVerifyState(
        dev,
        group.cpu_group,
        list(group.ranks),
        group.rank_in_group,
        local_vocab=local_vocab,
        vocab_start=vocab_start,
        vocab=vocab,
        seed=seed,
    )
