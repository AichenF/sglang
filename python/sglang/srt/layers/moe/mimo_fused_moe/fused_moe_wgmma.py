"""Host side for fused_moe_wgmma.cu: the JIT build, per-rank state (staging buffers, flags, P2P all-reduce
buffers, prologue scratch) and the launchers (`run_fused_full` = complete path from bf16 hidden/residual,
`run_fused_full_tp` = exact-M32 TP8 entry with the input all-reduce in the prologue, `run_fused` = legacy
pre-routed path). The kernel consumes the SGLang Humming MoE layer's weight tensors in place. Reference
implementations and test helpers live in benchmark/kernels/mimo_fused_moe/.

Kernel weight format (= layer.w13_weight / w13_weight_scale / w13_weight_scale_2 and w2_* after SGLang's prepare_humming_moe_layer):
  w:  int32 [E][K/32][4R]  Humming weight_repack_nk order (see fused_moe_wgmma.cu, "Weight layout")
  s:  uint8 [E][K/32][R]   exponent offsets 1..12, rows permuted within 64-blocks
  s2: fp32 [E]             per-expert factor (weight = e2m1(code) * 2^(o-6) * s2 * 2^6)
"""

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
DIM, INTER, A_GROUP, TOPK = 6144, 256, 128, 8
NEXP = 384  # routed experts (kernel compile-time max; runtime n_exp <= 384)
MAXU = NEXP  # union slots
N_FC2_TILES = DIM // 128
N_FC2_PARTS = 3  # max parts per fc2 tile (3 = mixed 3/2 mode over the whole grid); sizes rs_buf / dbg
N_FC2_CTAS = N_FC2_TILES * N_FC2_PARTS
MSG_PER_TOK = 43
MAXM = 64
NSTAMP = 28
RUNITS, RMSG = (
    96,
    22,
)  # router units (16 k-groups x 6 row-groups), LL messages per (unit, token)
EPS = 1e-5

_ext = None


def ext():
    """JIT-build (once per flag set) and return the extension. wgmma_rs_fp8.cuh is committed source; after editing
    gen_wgmma_header.py, regenerate it with `python gen_wgmma_header.py wgmma_rs_fp8.cuh`."""
    global _ext
    if _ext is None:
        extra = (
            os.environ.get("FMOE_NVCC_FLAGS", "").split()
        )  # e.g. "-DFMOE_PHASE_STAMPS" (separate build cache per flag set)
        name = "fused_moe_wgmma" + (
            "_" + "".join(c if c.isalnum() else "_" for c in "".join(extra))
            if extra
            else ""
        )
        _ext = load(
            name=name,
            sources=[os.path.join(HERE, "fused_moe_wgmma.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_90a,code=sm_90a",
                "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            ]
            + extra,
            verbose=False,
        )
    return _ext


# The M = 32 input-TP fused MoE launch (grid == 132), attn_block (128 CTAs) and fa_block (132 CTAs) size their grids for one
# CTA per SM of a 132-SM sm_90 part and spin-wait across CTAs, so every CTA must be co-resident. Other SM counts take the
# unfused paths (gated in layer.py / attn_block.py / fa_block.py).
GRID_SMS = 132
_SM_COUNT = {}


def device_sm_count(device) -> int:
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    n = _SM_COUNT.get(idx)
    if n is None:
        n = _SM_COUNT[idx] = int(torch.cuda.get_device_properties(idx).multi_processor_count)
    return n


def full_grid_ok(device) -> bool:
    """True when the persistent kernels' 132-CTA grids are co-resident on ``device`` (exactly 132 SMs)."""
    return device_sm_count(device) == GRID_SMS


# ---------------------------------------------------------------- weights
def humming_layer_weights(layer):
    """The kernel's weight dict from an SGLang FusedMoE layer after prepare_humming_moe_layer (tensors used in place)."""
    return dict(
        fc1_w=layer.w13_weight.data,
        fc1_s=layer.w13_weight_scale.data.view(torch.uint8),
        fc1_s2=layer.w13_weight_scale_2.data.reshape(-1),
        fc2_w=layer.w2_weight.data,
        fc2_s=layer.w2_weight_scale.data.view(torch.uint8),
        fc2_s2=layer.w2_weight_scale_2.data.reshape(-1),
    )


# ---------------------------------------------------------------- per-rank state
class RankState:
    """Device-local intermediates + all-reduce buffers + prologue scratch for one rank (max size, allocated once;
    the union can reach 384 slots, so the staging buffers are always sized for MAXU)."""

    RS_BYTES = (
        N_FC2_TILES * MAXM * MSG_PER_TOK * N_FC2_PARTS * 16
    )  # reduce-scatter buffer: uint4 messages, [rank x part] sources
    AG_BYTES = N_FC2_TILES * MAXM * MSG_PER_TOK * 16  # all-gather buffer

    def __init__(self, device, max_union=MAXU, alloc_comm=True):
        """alloc_comm=False: the all-reduce buffers are allocated elsewhere (e.g. cudaMalloc + CUDA IPC across processes) and
        registered with link_ptrs()."""
        max_union = MAXU
        self.device = device
        with torch.cuda.device(device):
            self.h_buf = torch.zeros(
                max_union * 2 * MAXM * 128, dtype=torch.uint8, device=device
            )
            self.cs_buf = torch.zeros(
                max_union * MAXM * 2, dtype=torch.float32, device=device
            )
            # Original half-ready flags, then four independent token-chunk
            # flags per half. Epoch tags permit heavy/light Graph transitions.
            self.h_flags = torch.zeros(max_union * 10, dtype=torch.int32, device=device)
            if alloc_comm:
                self.rs_buf = torch.zeros(
                    self.RS_BYTES // 4, dtype=torch.int32, device=device
                )
                self.ag_buf = torch.zeros(
                    self.AG_BYTES // 4, dtype=torch.int32, device=device
                )
            self.work = torch.zeros(
                4 + N_FC2_TILES, dtype=torch.int32, device=device
            )  # fc1 front/back queues, completion counter, epoch (self-resetting), per-tile fc2 expert counters
            self.part_flags = torch.zeros(
                max_union * 2, dtype=torch.int32, device=device
            )
            # prologue scratch: normed bf16 / fp8 / scales of the tokens, per-token flags, router LL partials, top-8 LL lists
            self.xn_buf = torch.zeros(MAXM * DIM, dtype=torch.bfloat16, device=device)
            self.xq_buf = torch.zeros(MAXM * DIM, dtype=torch.uint8, device=device)
            self.xs_buf = torch.zeros(
                MAXM * (DIM // A_GROUP), dtype=torch.float32, device=device
            )
            self.xflags = torch.zeros(MAXM, dtype=torch.int32, device=device)
            self.rpart = torch.zeros(
                RUNITS * MAXM * RMSG * 4, dtype=torch.int32, device=device
            )
            self.topk = torch.zeros(
                MAXM * (TOPK // 2) * 4, dtype=torch.int32, device=device
            )
            self.ssq = torch.zeros(
                16 * RMSG * 4, dtype=torch.int32, device=device
            )  # per-k-slice sum-of-squares LL partials
            self.union_out = torch.zeros(1 + MAXU, dtype=torch.int32, device=device)
        self.rs_ptrs = None
        self.ag_ptrs = None
        # FMOE_INPUT_TP: the ranks' IPC prologue buffers (PRO_BYTES each; reduce-scatter / all-gather messages, and this rank's
        # rpart / topk regions). link_pro_ptrs() after a cudaMalloc + IPC exchange like the all-reduce buffers.
        self.pro_ptrs = None
        self.pro_local = 0
        # FMOE_TP_PTR_PARAMS: CPU copies of the three pointer tables, passed BY VALUE into the kernel parameter by run_fused_full_tp
        self.rs_cpu = None
        self.ag_cpu = None
        self.pro_cpu = None

    def link_pro_ptrs(self, pro_ptrs, local_ptr):
        """pro_ptrs: device addresses (ints, valid in this process) of every rank's prologue buffer, rank order; local_ptr = this rank's.
        The input-TP kernel publishes the top-8 lists into the buffer's TOPK region (written by every owner rank), so self.topk becomes a
        zero-copy view of that region (CUDA array interface) and kernel_topk() needs no change."""
        self.pro_ptrs = torch.tensor(
            list(pro_ptrs), dtype=torch.int64, device=self.device
        )
        self.pro_cpu = torch.tensor([int(p) for p in pro_ptrs], dtype=torch.int64)
        self.pro_local = int(local_ptr)
        n = MAXM * (TOPK // 2) * 4
        base = self.pro_local + ext().PRO_BYTES - n * 4

        class _View:
            __cuda_array_interface__ = dict(
                shape=(n,), typestr="<i4", data=(base, False), version=2, strides=None
            )

        self.topk = torch.as_tensor(_View(), device=self.device)
        assert self.topk.data_ptr() == base, (
            "expected a zero-copy view of the TOPK region"
        )

    def kernel_topk(self, M):
        """Decode the kernel's published top-8 lists: (ids [M][8] int64, weights [M][8] fp32) in rank order."""
        m = self.topk[: M * (TOPK // 2) * 4].reshape(M, TOPK // 2, 4)
        ids = (
            torch.stack([m[:, :, 0] & 0xFFFF, (m[:, :, 0] >> 16) & 0xFFFF], dim=-1)
            .reshape(M, TOPK)
            .long()
        )
        w = (
            torch.stack([m[:, :, 1], m[:, :, 2]], dim=-1)
            .reshape(M, TOPK)
            .view(torch.float32)
        )
        return ids, w

    def kernel_union(self):
        U = int(self.union_out[0].item())
        return self.union_out[1 : 1 + U].long()

    def link_ptrs(self, rs_ptrs, ag_ptrs):
        """rs_ptrs / ag_ptrs: device addresses (ints, valid in this process) of every rank's rs / ag buffer, rank order."""
        self.rs_ptrs = torch.tensor(
            list(rs_ptrs), dtype=torch.int64, device=self.device
        )
        self.ag_ptrs = torch.tensor(
            list(ag_ptrs), dtype=torch.int64, device=self.device
        )
        self.rs_cpu = torch.tensor([int(p) for p in rs_ptrs], dtype=torch.int64)
        self.ag_cpu = torch.tensor([int(p) for p in ag_ptrs], dtype=torch.int64)


_EMPTY = {}


def default_parts(M):
    """fc2 CTAs per output tile (measured on 8xH200): 2 at M<=24 (one fc1 item per fc2 CTA); 1 at M=25..40 (fc1
    2-round tail-bound, 138.5 us at M=32); 3 = mixed 3/2 parts over the whole grid at M=41..48 (163.6 vs 168.4 us); 2 above
    (shared queue, 209 vs 235 us at M=64)."""
    env = os.environ.get("FMOE_PARTS")
    return (
        int(env)
        if env
        else (2 if M <= 24 else (1 if M <= 40 else (3 if M <= 48 else 2)))
    )


def default_n_fc1(device, parts=1):
    if (
        parts == 3
    ):  # mixed mode: every CTA does fc2, the kernel sizes the grid to the SM count
        return 0
    return (
        torch.cuda.get_device_properties(device).multi_processor_count
        - N_FC2_TILES * parts
    )


def run_fused(
    state,
    x_fp8,
    x_scale,
    w,
    union_experts,
    gate_w,
    out,
    my_rank,
    ndev,
    epoch,
    n_fc1=None,
    dbg=None,
    mode=0,
    reserve=-1,
    parts=None,
):
    """Legacy pre-routed path. w = dict(fc1_w, fc1_s, fc1_s2, fc2_w, fc2_s, fc2_s2) (prep_fc1/prep_fc2 or humming_layer_weights)."""
    if parts is None:
        parts = default_parts(x_fp8.shape[0])
    if n_fc1 is None:
        n_fc1 = default_n_fc1(state.device, parts)
    if dbg is None:
        dev = str(state.device)
        if dev not in _EMPTY:
            _EMPTY[dev] = torch.empty(0, dtype=torch.int64, device=state.device)
        dbg = _EMPTY[dev]
    ext().fused_moe(
        x_fp8.view(torch.uint8),
        x_scale,
        w["fc1_w"],
        w["fc1_s"],
        w["fc1_s2"],
        w["fc2_w"],
        w["fc2_s"],
        w["fc2_s2"],
        union_experts,
        gate_w,
        state.h_buf,
        state.cs_buf,
        state.h_flags,
        state.rs_ptrs,
        state.ag_ptrs,
        my_rank,
        ndev,
        epoch,
        out,
        n_fc1,
        dbg,
        mode,
        state.work,
        reserve,
        parts,
        state.part_flags,
    )


def run_fused_full(
    state,
    inp,
    w,
    out,
    residual_out,
    my_rank,
    ndev,
    n_fc1=None,
    dbg=None,
    mode=0,
    reserve=-1,
    parts=None,
    eps=EPS,
):
    """Complete path: inp = dict(hidden, residual, norm_w, router_w, bias), w = weight dict (humming_layer_weights) -> bf16 out [M][DIM],
    bf16 residual_out [M][DIM]. The routing is recomputed on device (every rank identically); the union size can be read back with
    state.kernel_union()."""
    M = inp["hidden"].shape[0]
    if parts is None:
        parts = default_parts(M)
    if n_fc1 is None:
        n_fc1 = default_n_fc1(state.device, parts)
    if dbg is None:
        dev = str(state.device)
        if dev not in _EMPTY:
            _EMPTY[dev] = torch.empty(0, dtype=torch.int64, device=state.device)
        dbg = _EMPTY[dev]
    router_w = inp["router_w"]
    if getattr(ext(), "ROUTER_BF16", 0):
        # FMOE_ROUTER_BF16 build: the kernel takes the fp32 checkpoint weight rounded ONCE to bf16 (RNE, as SGLang's bf16 MoEGate
        # parameter load does). Graph-replaying callers must pass the rounded copy as inp["router_w_bf16"] (a cast here would be
        # captured into the graph); the fallback cast is for eager use only.
        router_w = inp.get("router_w_bf16")
        if router_w is None:
            router_w = (
                inp["router_w"]
                if inp["router_w"].dtype == torch.bfloat16
                else inp["router_w"].to(torch.bfloat16)
            )
    ext().fused_moe_full(
        inp["hidden"],
        inp["residual"],
        inp["norm_w"],
        router_w,
        inp["bias"],
        eps,
        w["fc1_w"],
        w["fc1_s"],
        w["fc1_s2"],
        w["fc2_w"],
        w["fc2_s"],
        w["fc2_s2"],
        state.xn_buf,
        state.xq_buf,
        state.xs_buf,
        state.xflags,
        state.rpart,
        state.topk,
        state.ssq,
        state.union_out,
        state.h_buf,
        state.cs_buf,
        state.h_flags,
        state.rs_ptrs,
        state.ag_ptrs,
        my_rank,
        ndev,
        out,
        residual_out,
        n_fc1,
        dbg,
        mode,
        state.work,
        reserve,
        parts,
        state.part_flags,
    )


_PF_NONE = {}
_PF_TEST = {}


def pf_desc(tensors, evict_last=()):
    """Describe up to 3 contiguous CUDA tensors as the kernel's L2-prefetch ranges: CPU int64 [7] = (data_ptr, nbytes) x 3 (zeros
    for unused slots) + a bitmask of the ranges prefetched with an L2::evict_last hint (`evict_last`: per-tensor booleans, same order as
    `tensors`). Non-contiguous / misaligned tensors are skipped (the prefetch is a cache hint, never a dependency)."""
    d = [0] * 7
    k = 0
    for i, t in enumerate(tensors):
        if t is None or not t.is_cuda or not t.is_contiguous():
            continue
        ptr, nbytes = t.data_ptr(), t.numel() * t.element_size()
        if ptr % 16 or nbytes < 16 or k >= 3:
            continue
        d[2 * k], d[2 * k + 1] = ptr, nbytes - nbytes % 16
        if i < len(evict_last) and evict_last[i]:
            d[6] |= 1 << k
        k += 1
    return torch.tensor(d, dtype=torch.int64)


def _pf_default(state):
    """No ranges from the caller: empty (no prefetch) -- or, in a bench build with -DFMOE_PF_TEST_KB=N, one N-KB dummy device range so the
    kernel-side cost of issuing the prefetch can be timed by harnesses that know nothing about it (input_tp_bench)."""
    dev = str(state.device)
    kb = int(getattr(ext(), "PF_TEST_KB", 0))
    if kb > 0:
        if dev not in _PF_TEST:
            buf = torch.zeros(kb * 1024, dtype=torch.uint8, device=state.device)
            _PF_TEST[dev] = (buf, pf_desc([buf]))
        return _PF_TEST[dev][1]
    if dev not in _PF_NONE:
        _PF_NONE[dev] = torch.empty(0, dtype=torch.int64)
    return _PF_NONE[dev]


def norm_next_buffers(device, M=32):
    """FMOE_NORM_NEXT output buffers for one layer: residual_new bf16 [M][DIM], x_fp8 e4m3 [M][DIM] and the deep_gemm TMA-aligned
    column-major fp32 scales [M][48] (= SGLang's create_per_token_group_quant_fp8_output_scale(column_major_scales=True,
    scale_tma_aligned=True): empty((48, ceil4(M))).transpose(0, 1)[:M], stride (1, ceil4(M)))."""
    aligned = (M + 3) // 4 * 4
    return dict(
        res_new=torch.empty(M, DIM, dtype=torch.bfloat16, device=device),
        xq=torch.empty(M, DIM, dtype=torch.float8_e4m3fn, device=device),
        xs=torch.empty(
            DIM // A_GROUP, aligned, dtype=torch.float32, device=device
        ).transpose(0, 1)[:M],
    )


def run_fused_full_tp(
    state,
    inp,
    w,
    out,
    residual_out,
    my_rank,
    ndev,
    n_fc1=None,
    dbg=None,
    mode=0,
    reserve=-1,
    parts=None,
    eps=EPS,
    norm_next=None,
    pf=None,
):
    """FMOE_INPUT_TP entry (build with FMOE_NVCC_FLAGS=-DFMOE_INPUT_TP=1): inp = dict(partial, residual, norm_w, router_w[_bf16], bias) where
    `partial` is this rank's UN-reduced o_proj output (bf16 [32][DIM]); the input all-reduce runs inside the kernel's prologue over the
    ranks' IPC prologue buffers (state.link_pro_ptrs). Exact M = 32, TP8 only; outputs are bitwise those of CustomAllReduceV2 + run_fused_full.
    norm_next (FMOE_NORM_NEXT): dict(norm_w = the NEXT layer's input_layernorm weight bf16 [DIM], eps, res_new, xq, xs (norm_next_buffers))
    -> the kernel's TP tail also produces residual_new = out + residual_out, x_fp8 and the column-major scales of the next layer's
    qkv_proj input, bitwise flashinfer fused_add_rmsnorm + SGLang per_token_group_quant (deep_gemm layout).
    pf: pf_desc(...) of the NEXT layer's weights to L2-prefetch in the TP tail (kernel built with -DFMOE_TAIL_PREFETCH=1..3); None = none."""
    M = inp["partial"].shape[0]
    assert M == 32 and ndev == 8 and state.pro_ptrs is not None and state.pro_local, (
        "input-TP entry: M == 32, TP8, linked prologue buffers"
    )
    assert (
        state.pro_cpu is not None
        and state.rs_cpu is not None
        and state.ag_cpu is not None
    ), "link_ptrs / link_pro_ptrs before the TP entry"
    if pf is None:
        pf = _pf_default(state)
    if parts is None:
        parts = default_parts(M)
    if n_fc1 is None:
        n_fc1 = default_n_fc1(state.device, parts)
    dev = str(state.device)
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.int64, device=state.device)
    if dbg is None:
        dbg = _EMPTY[dev]
    router_w = inp.get("router_w_bf16")
    if router_w is None:
        router_w = (
            inp["router_w"]
            if inp["router_w"].dtype == torch.bfloat16
            else inp["router_w"].to(torch.bfloat16)
        )  # eager use only
    if norm_next is not None:
        assert getattr(ext(), "NORM_NEXT", 0), (
            "this build has FMOE_NORM_NEXT=0 (or FMOE_INPUT_TP=0)"
        )
        nn_args = (
            norm_next["norm_w"],
            norm_next["res_new"],
            norm_next["xq"],
            norm_next["xs"],
            float(norm_next.get("eps", eps)),
        )
    else:
        nn_args = (_EMPTY[dev], _EMPTY[dev], _EMPTY[dev], _EMPTY[dev], eps)
    ext().fused_moe_full_tp(
        inp["partial"],
        inp["residual"],
        inp["norm_w"],
        router_w,
        inp["bias"],
        eps,
        w["fc1_w"],
        w["fc1_s"],
        w["fc1_s2"],
        w["fc2_w"],
        w["fc2_s"],
        w["fc2_s2"],
        state.xn_buf,
        state.xq_buf,
        state.xs_buf,
        state.xflags,
        state.pro_ptrs,
        state.pro_local,
        state.union_out,
        state.h_buf,
        state.cs_buf,
        state.h_flags,
        state.rs_ptrs,
        state.ag_ptrs,
        my_rank,
        ndev,
        out,
        residual_out,
        n_fc1,
        dbg,
        mode,
        state.work,
        reserve,
        parts,
        state.part_flags,
        state.pro_cpu,
        state.rs_cpu,
        state.ag_cpu,
        *nn_args,
        pf,
    )
