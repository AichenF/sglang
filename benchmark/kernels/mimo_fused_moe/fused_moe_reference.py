"""Reference and test helpers for the fused MoE kernel (python/sglang/srt/layers/moe/mimo_fused_moe/fused_moe_wgmma.cu),
used by check_fused_moe.py: a bit-exact torch re-implementation of Humming's fused-e8m0 W4A8 weight transform (so the
effective fp32 weights are available without a humming install), synthetic inputs, fp32 references of the prologue
(RMSNorm, routing, union order) and of fc1 / fc2, single-process multi-GPU linking of the per-rank states, and the
phase-stamp breakdown of a -DFMOE_PHASE_STAMPS build.

Raw input format (checkpoint / SGLang FusedMoE parameters before Humming's transform):
  w_packed: [E][R][K/2] uint8 -- e2m1 codes, low nibble = even k, high nibble = odd k
  w_scale:  [E][R][K/32] uint8 -- e8m0 (scale = 2^(byte-127))
"""

import ctypes

import torch

from sglang.srt.layers.moe.mimo_fused_moe.fused_moe_wgmma import (
    A_GROUP,
    DIM,
    EPS,
    INTER,
    N_FC2_CTAS,
    NEXP,
    NSTAMP,
    TOPK,
)

W_GROUP = 32  # e8m0 weight-scale group size
E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------- weight preprocessing (Humming fused-e8m0 layout)
# Bit-exact torch re-implementation of humming.transform.transform_humming_tensors for b=e2m1, bs=e8m0 g32, a=e4m3 on
# sm90 (mma WGMMA, use_fused_e8m0_scale, interleave mode 2, tensor-type weight_scale_2), validated word-for-word against
# humming's own ops. The kernel reads SGLang's Humming MoE layer tensors directly; these functions
# exist so the tests/benchmarks need no humming install and so the effective (post-transform) fp32 weights are available.
SIGN_SRC = [0, 4, 1, 5, 2, 6, 3, 7]  # nibble i carries the sign of element SIGN_SRC[i]
MAX_RANGE = 11  # octaves kept below the per-expert max e8m0 (fp8 activation)


def unpack_codes(packed):
    """[..., K/2] uint8 -> [..., K] uint8 e2m1 codes (low nibble first)."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def hl_fold(codes, scale):
    """codes u8 [E][N][K] (e2m1), scale u8 [E][N][K/32] e8m0 -> (codes' u8, o u8 [E][N][K/32] in 1..12, s2 f32 [E]).
    humming process_fused_e8m0_scale + process_mxfp4_w4a8: per-expert clamp of the e8m0 range to 11 octaves below the max;
    groups below the range get their scale raised and their codes requantized (value * 2^-delta rounded to the e2m1 grid,
    ties away from zero, sign kept; -0 codes become +0 first)."""
    E = codes.shape[0]
    dev = codes.device
    sc = scale.to(torch.int32)
    smax = sc.reshape(E, -1).amax(-1)
    smin = sc.reshape(E, -1).amin(-1)
    smin_new = smax - torch.clamp(smax - smin, max=MAX_RANGE)  # [E]
    sc_c = torch.maximum(sc, smin_new[:, None, None])
    delta = sc_c - sc  # [E][N][G] >= 0
    o = (sc_c - smin_new[:, None, None] + 1).to(torch.uint8)
    codes = torch.where(codes == 8, torch.zeros_like(codes), codes)
    d = delta.repeat_interleave(W_GROUP, dim=-1)
    grid = E2M1_LUT.to(dev)
    mag = grid[(codes & 7).long()] * torch.pow(2.0, -d.float())
    mids = (grid[:-1] + grid[1:]) / 2
    idx = torch.searchsorted(mids, mag.contiguous(), right=True).to(
        torch.uint8
    )  # v >= midpoint -> upper neighbour
    codes = torch.where(d > 0, ((codes >> 3) << 3) | idx, codes)
    s2 = torch.pow(2.0, smin_new.float() - 127.0) / 2
    return codes, o, s2


def hl_pack_words(codes):
    """codes u8 [E][N][K] -> int32 [E][K/32][4N] in humming weight_repack_nk order (4-bit B, 8-bit A, fused e8m0, interleave 2)."""
    E, N, K = codes.shape
    dev = codes.device
    G = K // 32
    nb = torch.arange(N // 64, device=dev)
    half = torch.arange(2, device=dev)
    tid = torch.arange(32, device=dev)
    q = torch.arange(4, device=dev)
    i = torch.arange(8, device=dev)
    row = (
        nb[:, None, None, None] * 64
        + half[None, :, None, None] * 32
        + (q >> 1)[None, None, None, :] * 16
        + (q & 1)[None, None, None, :] * 8
        + (tid // 4)[None, None, :, None]
    )  # [nb][half][tid][q]
    kk = (
        (tid % 4)[:, None] * 4 + (i % 4)[None, :] + (i // 4)[None, :] * 16
    )  # [tid][i] within a k32 slice
    k = torch.arange(G, device=dev)[:, None, None] * 32 + kk[None, :, :]  # [G][tid][i]
    rows = row[None, :, :, :, :, None].expand(G, N // 64, 2, 32, 4, 8)
    ks = k[:, None, None, :, None, :].expand(G, N // 64, 2, 32, 4, 8)
    el = codes[:, rows, ks]  # [E][G][nb][half][tid][q][i]
    nib = (el & 7).long() | (((el >> 3) & 1).long()[..., SIGN_SRC] << 3)
    word = (nib << (4 * i).long()).sum(-1)  # int64 [E][G][nb][half][tid][q]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.reshape(E, G, 4 * N).contiguous()


def hl_pack_scales(o):
    """o u8 [E][N][G] -> u8 [E][G][N] (humming transform_humming_weight_scale: transpose, then within each 64-row block
    position p holds row (p % 8) * 8 + p // 8)."""
    E, N, G = o.shape
    p = torch.arange(64, device=o.device)
    src = (p % 8) * 8 + p // 8
    return (
        o.transpose(-1, -2)
        .reshape(E, G, N // 64, 64)[..., src]
        .reshape(E, G, N)
        .contiguous()
    )


def hl_effective(codes, o, s2):
    """fp32 weights [E][N][K] exactly as the kernel (and humming) see them."""
    grid = E2M1_LUT.to(codes.device)
    mag = grid[(codes & 7).long()]
    sign = torch.where((codes & 8) > 0, -1.0, 1.0)
    sc = torch.pow(2.0, o.float() - 6.0).repeat_interleave(W_GROUP, dim=-1)
    return mag * sign * sc * (s2 * 64.0)[:, None, None]


def hl_transform(packed, scale, chunk=32):
    """packed u8 [E][N][K/2], scale u8 [E][N][K/32] -> (w int32 [E][K/32][4N], s u8 [E][K/32][N], s2 f32 [E], eff f32 [E][N][K]),
    processed `chunk` experts at a time (the gather intermediates are ~1 GB per 32 experts of 512 x 6144)."""
    E = packed.shape[0]
    ws, ss, s2s, effs = [], [], [], []
    for e0 in range(0, E, chunk):
        codes = unpack_codes(packed[e0 : e0 + chunk])
        codes, o, s2 = hl_fold(codes, scale[e0 : e0 + chunk])
        ws.append(hl_pack_words(codes))
        ss.append(hl_pack_scales(o))
        s2s.append(s2)
        effs.append(hl_effective(codes, o, s2))
    return torch.cat(ws), torch.cat(ss), torch.cat(s2s), torch.cat(effs)


def prep_fc1(gate_packed, gate_scale, up_packed, up_scale, chunk=32):
    """Humming-layout w13 (gate rows 0..255 | up rows 256..511, SGLang's w13 order) for E experts.
    -> dict(fc1_w int32 [E][192][2048], fc1_s u8 [E][192][512], fc1_s2 f32 [E]), wg_eff, wu_eff (fp32 [E][256][6144])."""
    w, s, s2, eff = hl_transform(
        torch.cat([gate_packed, up_packed], dim=1),
        torch.cat([gate_scale, up_scale], dim=1),
        chunk,
    )
    return (
        dict(fc1_w=w, fc1_s=s, fc1_s2=s2),
        eff[:, :INTER].contiguous(),
        eff[:, INTER:].contiguous(),
    )


def prep_fc2(down_packed, down_scale, chunk=32):
    """Humming-layout w2 for E experts. -> dict(fc2_w int32 [E][8][24576], fc2_s u8 [E][8][6144], fc2_s2 f32 [E]), wd_eff fp32 [E][6144][256]."""
    w, s, s2, eff = hl_transform(down_packed, down_scale, chunk)
    return dict(fc2_w=w, fc2_s=s, fc2_s2=s2), eff


# ---------------------------------------------------------------- test data
def tile_weights(w, E):
    """Tile a weight dict of n experts to E experts (tests: distinct bytes per expert do not matter for coverage)."""
    n = w["fc1_s2"].shape[0]
    rep = (E + n - 1) // n
    return {
        k: (
            v.repeat(rep, *([1] * (v.dim() - 1)))[:E].contiguous()
            if rep > 1
            else v.contiguous()
        )
        for k, v in w.items()
    }


# ---------------------------------------------------------------- test data
def make_bounded_weights(E, R, K, device, gen=None):
    packed = torch.randint(
        0, 256, (E, R, K // 2), dtype=torch.uint8, device=device, generator=gen
    )
    scale = torch.randint(
        124, 131, (E, R, K // W_GROUP), dtype=torch.uint8, device=device, generator=gen
    )
    return packed, scale


def quantize_x(x):
    """x [M][DIM] fp32 -> (x_fp8 [M][DIM] e4m3, x_scale [M][48] fp32) per-token per-128 dynamic."""
    M = x.shape[0]
    g = x.reshape(M, DIM // A_GROUP, A_GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    # tensor/tensor division = IEEE fp32 division like the kernel (torch's tensor/python-scalar path multiplies by the
    # reciprocal instead, which is 1 ulp off and flips exact fp8 rounding ties)
    scale = torch.where(
        amax > 0, amax / torch.full_like(amax, 448.0), torch.ones_like(amax)
    )
    xq = (g / scale).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, DIM)
    return xq, scale.reshape(M, DIM // A_GROUP).contiguous()


def make_routing(M, U, device, gen=None, topk=TOPK):
    """gate_w [M][U]: each token routes to min(topk, U) distinct union slots with positive weights."""
    k = min(topk, U)
    gate_w = torch.zeros(M, U, device=device)
    for t in range(M):
        idx = torch.randperm(U, device=device, generator=gen)[:k]
        w = torch.rand(k, device=device, generator=gen) + 0.1
        gate_w[t, idx] = w / w.sum()
    return gate_w


# Hot-set sizes used by the tests for skewed routing (real routing concentrates on popular experts). With the
# N(0, 0.1) correction bias the routing is already concentrated (U = 47/72/74/87 at M=24/32/48/64 for these H;
# uniform routing over 384 experts gives U ~ 85-125, not the birthday estimate); the bench calibrates H and the
# bias std against the oracle to reproduce the baseline benchmark's union sizes exactly.
HOT_SET = {24: 80, 32: 105, 48: 112, 64: 130}


def make_full_inputs(
    M, device, gen=None, n_exp=NEXP, hot=None, x_std=1.0, bias_std=0.1
):
    """Bounded test data for the complete path: bf16 hidden/residual ~N(0, x_std), bf16 norm weight ~1 +- 0.1,
    fp32 router weight ~N(0, 0.02), fp32 correction bias ~N(0, bias_std) (+3 on a random hot set of `hot` experts;
    the bench lowers bias_std when it needs a less concentrated routing to reach the baseline's union size)."""
    hidden = (torch.randn(M, DIM, device=device, generator=gen) * x_std).to(
        torch.bfloat16
    )
    residual = (torch.randn(M, DIM, device=device, generator=gen) * x_std).to(
        torch.bfloat16
    )
    norm_w = (1.0 + 0.1 * torch.randn(DIM, device=device, generator=gen)).to(
        torch.bfloat16
    )
    router_w = torch.randn(n_exp, DIM, device=device, generator=gen) * 0.02
    bias = torch.randn(n_exp, device=device, generator=gen) * bias_std
    if hot:
        hot_idx = torch.randperm(n_exp, device=device, generator=gen)[:hot]
        bias[hot_idx] += 3.0
    return dict(
        hidden=hidden,
        residual=residual,
        norm_w=norm_w,
        router_w=router_w.contiguous(),
        bias=bias.contiguous(),
    )


# ---------------------------------------------------------------- single-process multi-GPU
libcudart = None
_peer_enabled = set()


def enable_peer_access(ndev):
    """Idempotent: cudaDeviceEnablePeerAccess returns cudaErrorPeerAccessAlreadyEnabled (704) on a
    repeat call, and that error is sticky (torch trips on it) until cudaGetLastError() clears it."""
    global libcudart
    if libcudart is None:
        libcudart = ctypes.CDLL("libcudart.so")
    for i in range(ndev):
        torch.cuda.set_device(i)
        for j in range(ndev):
            if i == j or (i, j) in _peer_enabled:
                continue
            can = ctypes.c_int()
            libcudart.cudaDeviceCanAccessPeer(ctypes.byref(can), i, j)
            if can.value:
                err = libcudart.cudaDeviceEnablePeerAccess(j, 0)
                if err not in (0, 704):
                    raise RuntimeError(
                        f"cudaDeviceEnablePeerAccess({i}->{j}) failed: {err}"
                    )
                libcudart.cudaGetLastError()
                _peer_enabled.add((i, j))


def link_states(states):
    """Single-process multi-GPU: every rank's state gets the peers' buffer addresses (P2P access must be enabled)."""
    rs = [s.rs_buf.data_ptr() for s in states]
    ag = [s.ag_buf.data_ptr() for s in states]
    for s in states:
        s.link_ptrs(rs, ag)


PROFILE_KEYS = [
    "stage1_max",
    "slices_landed_max",
    "rstd_max",
    "w_landed_max",
    "mma_done_max",
    "router_max",
    "topk_gather_max",
    "topk_max",
    "tables_med",
    "tables_max",
    "fc1_first_tile_med",
    "fc1_end_med",
    "fc1_end_max",
    "fc2_compute_med",
    "fc2_compute_max",
    "push_max",
    "reduce_max",
    "pull_med",
    "pull_max",
]


def m_pad(M):
    """Kernel's padded token count (wgmma N): multiple of 8, and 56 -> 64 (the NT=56 instantiation is not built)."""
    p = (M + 7) & ~7
    return 64 if p == 56 else p


def profile_round(ranks_launch, ndev, n_fc1, grid_device0=None):
    """ranks_launch(dbg_list) launches one round with per-rank dbg tensors; returns a breakdown (us)
    per rank relative to the earliest CTA start on that rank. Stamps: 0 start, 6 stage-1 done, 7 router done,
    8 top-8 done, 9 routing tables built, 10 first fc1 tile landed, 1 fc1 items done, 2 fc2 compute, 3 push, 4 reduce, 5 pull."""
    dbgs = []
    for r in range(ndev):
        dbgs.append(
            torch.zeros(
                (n_fc1 + N_FC2_CTAS) * NSTAMP, dtype=torch.int64, device=f"cuda:{r}"
            )
        )  # sized for the max grid
    ranks_launch(dbgs)
    for r in range(ndev):
        torch.cuda.synchronize(r)
    res = {}
    starts = []
    for r in range(ndev):
        d = dbgs[r].cpu().reshape(-1, NSTAMP).double()
        d = d[d[:, 0] > 0]  # only launched CTAs
        fc2 = d[: d.shape[0] - n_fc1]
        t0 = d[:, 0].min().item()
        starts.append(t0)

        def us(col, rows):
            v = rows[:, col]
            v = v[v > 0]
            return (v - t0) / 1e3 if v.numel() else torch.zeros(1)

        res[r] = dict(
            stage1_max=us(6, d).max().item(),
            slices_landed_max=us(13, d).max().item(),
            rstd_max=us(14, d).max().item(),
            w_landed_max=us(12, d).max().item(),
            mma_done_max=us(15, d).max().item(),
            router_max=us(7, d).max().item(),
            topk_gather_max=us(11, d).max().item(),
            topk_max=us(8, d).max().item(),
            tables_med=us(9, d).median().item(),
            tables_max=us(9, d).max().item(),
            fc1_first_tile_med=us(10, d).median().item(),
            fc1_end_med=us(1, d).median().item(),
            fc1_end_max=us(1, d).max().item(),
            fc2_compute_med=us(2, fc2).median().item(),
            fc2_compute_max=us(2, fc2).max().item(),
            push_max=us(3, fc2).max().item(),
            reduce_max=us(4, fc2).max().item(),
            pull_med=us(5, fc2).median().item(),
            pull_max=us(5, fc2).max().item(),
        )
    smin = min(starts)
    for r in range(ndev):
        res[r]["start_skew"] = (
            starts[r] - smin
        ) / 1e3  # %globaltimer is node-wide: this rank's launch skew
    return res


# ---------------------------------------------------------------- references
def unswizzle_h(h_buf, U, M_pad):
    """kernel h_buf bytes -> [U][2][M_pad][128] e4m3 tensor in natural order."""
    h = h_buf[: U * 2 * M_pad * 128].reshape(
        U, 2, M_pad, 8, 16
    )  # [.., tok, chunk16 (swizzled position), 16 B]
    tok = torch.arange(M_pad, device=h.device)
    q = torch.arange(8, device=h.device)
    # natural chunk q of token tok is stored at position q ^ (tok % 8)
    pos = q[None, :] ^ (tok[:, None] & 7)  # [M_pad][8]
    idx = pos[None, None, :, :, None].expand(U, 2, M_pad, 8, 16)
    nat = torch.gather(h, 3, idx)
    return nat.reshape(U, 2, M_pad, 128).view(torch.float8_e4m3fn)


def reference_fc1(x_fp8, x_scale, wg_eff, wu_eff, union_experts):
    """fp32 h [U][M][INTER] = SiLU(x Wg^T) * (x Wu^T) using the kernel's effective weights."""
    M = x_fp8.shape[0]
    x = x_fp8.float() * x_scale.repeat_interleave(A_GROUP, dim=1)
    hs = []
    for e in union_experts.tolist():
        g = x @ wg_eff[e].t()
        u = x @ wu_eff[e].t()
        hs.append(torch.nn.functional.silu(g) * u)
    return torch.stack(hs)


def reference_fc2_from_kernel_h(state, U, M, wd_eff, union_experts, gate_w=None):
    """out [M][DIM] from the kernel's OWN fp8 h + combined scales (isolates fc2+combine)."""
    M_pad = m_pad(M)
    h = unswizzle_h(state.h_buf, U, M_pad).float()[:, :, :M]  # [U][2][M][128]
    cs = state.cs_buf[: U * M_pad * 2].reshape(U, M_pad, 2)[:, :M]  # [U][M][2]
    out = torch.zeros(M, DIM, device=h.device)
    for u, e in enumerate(union_experts.tolist()):
        for b in range(2):
            hb = h[u, b] * cs[u, :, b : b + 1]
            out += hb @ wd_eff[e][:, b * 128 : (b + 1) * 128].t()
    return out


def reference_full(x_fp8, x_scale, wg_eff, wu_eff, wd_eff, union_experts, gate_w):
    """fp32 end-to-end reference (with a kernel-mimicking fp8 requant of h)."""
    h = reference_fc1(x_fp8, x_scale, wg_eff, wu_eff, union_experts)  # [U][M][256]
    U, M, _ = h.shape
    hg = h.reshape(U, M, 2, 128)
    amax = hg.abs().amax(dim=-1, keepdim=True)
    hs = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    hq = (hg / hs).to(torch.float8_e4m3fn).float() * hs
    hq = hq.reshape(U, M, 256)
    out = torch.zeros(M, DIM, device=h.device)
    for u, e in enumerate(union_experts.tolist()):
        out += gate_w[:, u : u + 1] * (hq[u] @ wd_eff[e].t())
    return out


# ---------------------------------------------------------------- prologue references (SGLang / flashinfer semantics)
def rmsnorm_reference(hidden, residual, norm_w, eps=EPS):
    """flashinfer FusedAddRMSNormKernel: x = f32(h) + f32(r) (unrounded) feeds variance and output;
    residual_out = bf16(x); normed = bf16((x * rsqrt(mean(x^2) + eps)) * w). Returns (residual_out bf16, normed bf16)."""
    x = hidden.float() + residual.float()
    residual_out = x.to(torch.bfloat16)
    rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    normed = ((x * rstd) * norm_w.float()).to(torch.bfloat16)
    return residual_out, normed


def routing_reference(normed_bf16, router_w, bias, topk=TOPK):
    """SGLang biased_grouped_topk_impl with n_group = topk_group = 1: logits = normed.f32 @ W^T (fp32),
    scores = sigmoid, ids = topk(scores + bias), weights = scores[ids] / (sum + 1e-20). Returns (logits, ids [M][k] sorted
    by biased score desc, weights [M][k])."""
    logits = normed_bf16.float() @ router_w.t()
    scores = torch.sigmoid(logits)
    key = scores + bias.unsqueeze(0)
    _, ids = torch.topk(key, topk, dim=-1, sorted=True)
    w = scores.gather(1, ids)
    w = w / (w.sum(dim=-1, keepdim=True, dtype=torch.float32) + 1e-20)
    return logits, ids, w


def union_reference(ids):
    """Kernel's deterministic union order: first appearance scanning (token, rank) token-major."""
    seen, uni = set(), []
    for e in ids.reshape(-1).tolist():
        if e not in seen:
            seen.add(e)
            uni.append(e)
    return uni


def gate_w_from_routing(ids, w, union):
    """Dense [M][U] routing-weight matrix (0 when a token is not routed to a slot) from top-8 lists + union order."""
    M = ids.shape[0]
    slot = {e: u for u, e in enumerate(union)}
    gate_w = torch.zeros(M, len(union), device=ids.device)
    for t in range(M):
        for k in range(ids.shape[1]):
            gate_w[t, slot[int(ids[t, k])]] = w[t, k]
    return gate_w
