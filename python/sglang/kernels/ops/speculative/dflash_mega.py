"""DFlash draft megakernel (csrc/distributed/dflash_mega.cuh), goal11-draft.

All 5 draft decoder layers (TP8, M = 32 rows = 4 requests x 8 block tokens) plus the layer-0 input norm/quant and the
final norm in ONE persistent kernel (15 clusters x 8 CTAs). Shares the IPC receive buffers and per-token call counters
of ``dflash_tp_fused.DFlashTPState`` (the in-kernel all-reduces are tp_arnq's protocol).

``DFlashMegaState(tp_state, layers, final_norm, cos_sin_cache, ...)`` holds the static part (weights, tensor maps,
scratch, work plan); ``run(hidden, out, positions, seqlens, page_table, loc)`` launches the kernel.
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import Dict, List, Optional

import torch

R = 8
H = 6144
MT = 32
NREQ = 4
NL = 5
NCL = 15
CS = 8
NATT = 8
MAXT_PH = 5
QKV_TILES = 36
O_TILES = 96
GU_TILES = 64
DN_TILES = 96
NSLOT_ATT = 6
NSLOT_FULL = 10

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "jit", "csrc", "distributed", "dflash_mega.cuh",
)


@functools.lru_cache(maxsize=None)
def _src_hash() -> str:
    with open(_SRC, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool, extra: str = ""):
    from sglang.kernels.jit.utils import cuda_stubs_dir, load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    flags = ["-O3"] + [f for f in (extra or os.environ.get("DM_NVCC_FLAGS", "")).split() if f]
    return load_jit(
        "dflash_mega_" + _src_hash(),
        *args,
        cuda_files=["distributed/dflash_mega.cuh"],
        cuda_wrappers=[
            ("params_bytes", f"DFlashMega<{args}>::params_bytes"),
            ("build_maps", f"DFlashMega<{args}>::build_maps"),
            ("fill", f"DFlashMega<{args}>::fill"),
            ("run", f"DFlashMega<{args}>::run"),
            ("max_clusters", f"DFlashMega<{args}>::max_clusters"),
        ],
        extra_cuda_cflags=flags,
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def _use_pdl() -> bool:
    return os.environ.get("SGLANG_DFLASH_MEGA_PDL", "0") == "1"


def _assign(n_tiles: int, caps: List[int], order: Optional[List[int]] = None) -> List[List[int]]:
    """Distribute tile ids (in `order`) over buckets with capacities `caps` (round-robin over buckets with room)."""
    order = list(range(n_tiles)) if order is None else order
    assert sum(caps) >= n_tiles, (caps, n_tiles)
    out: List[List[int]] = [[] for _ in caps]
    i = 0
    while i < len(order):
        progressed = False
        for b, cap in enumerate(caps):
            if i < len(order) and len(out[b]) < cap:
                out[b].append(order[i])
                i += 1
                progressed = True
        assert progressed
    return out


def default_plan() -> Dict[str, object]:
    """Work plan: per GEMM phase, the 64-row weight tiles of every cluster (and k-group for the 4-way split-K GEMMs).
    Attention clusters (0..7) carry <= 4 tiles per phase (their GEMM region also holds the attention buffers)."""
    plan = {}
    # qkv: k / v tiles first so the clusters with few tiles finish them early
    qcaps = [2] * NATT + [3, 3, 3, 3, 3, 3, 2]
    qorder = list(range(32, 36)) + list(range(32))
    qkv = _assign(QKV_TILES, qcaps[::-1], qorder)[::-1]
    plan["qkv"] = [[q, []] for q in qkv]
    # o_proj / down: 2 k-groups per cluster
    # <= 4 tiles (2 slabs) per k-group everywhere: the phase lasts as long as its slowest group (slab boundaries ~1 us)
    gcaps = [int(x) for x in os.environ.get("SGLANG_DFLASH_MEGA_OCAPS", "").split(",") if x] or ([4] * (2 * NCL))
    for name, n in (("o", O_TILES), ("dn", DN_TILES)):
        groups = _assign(n, gcaps)
        plan[name] = [[groups[2 * c], groups[2 * c + 1]] for c in range(NCL)]
    gucaps = [4] * NATT + [5, 5, 5, 5, 4, 4, 4]
    gu = _assign(GU_TILES, gucaps)
    plan["gu"] = [[g, []] for g in gu]
    plan["nslot"] = [NSLOT_ATT] * NATT + [NSLOT_FULL] * (NCL - NATT)
    return plan


def plan_tensor(plan: Dict[str, object]) -> torch.Tensor:
    tile = torch.full((4, NCL, 2, MAXT_PH), -1, dtype=torch.int16)
    n = torch.zeros((4, NCL, 2), dtype=torch.int16)
    for ph, name in enumerate(("qkv", "o", "gu", "dn")):
        seen = []
        for c in range(NCL):
            for g in range(2):
                ts = plan[name][c][g]
                assert len(ts) <= MAXT_PH
                n[ph, c, g] = len(ts)
                for i, t in enumerate(ts):
                    tile[ph, c, g, i] = t
                seen += ts
        total = {"qkv": QKV_TILES, "o": O_TILES, "gu": GU_TILES, "dn": DN_TILES}[name]
        assert sorted(seen) == list(range(total)), f"plan {name}: tiles not a partition"
    nslot = torch.tensor(plan["nslot"], dtype=torch.int16)
    return torch.cat([tile.flatten(), n.flatten(), nslot])


class DFlashMegaState:
    """Static state of the draft megakernel for one rank.

    layers: list of NL dicts with keys
      w_qkv, w_o, w_gu, w_dn  -- fp8 e4m3 [N, K] contiguous (the Fp8LinearMethod weight.t())
      s_qkv, s_o, s_gu, s_dn  -- fp32 per-channel scales [N]
      ln_in, ln_post          -- bf16 [H]
      q_norm, k_norm          -- bf16 [128]
      sinks                   -- [16] bf16/fp32 or None
      k_cache, v_cache        -- bf16 [slots, 1, 128]
    """

    def __init__(
        self,
        tp_state,
        layers: List[dict],
        final_norm: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        qk_eps: float,
        v_scale: float,
        sm_scale: float,
        plan: Optional[Dict[str, object]] = None,
        stamps: bool = False,
        nl: int = NL,
        look: Optional[int] = None,
    ):
        assert len(layers) == NL
        self.tp = tp_state
        self.device = final_norm.device
        dev = self.device
        self.mod = _module(_use_pdl())
        self.layers = layers  # keep the tensors alive
        self.final_norm = final_norm
        self.cos_sin_cache = cos_sin_cache
        self.maps = torch.zeros((NL * 4 + 2) * 128, dtype=torch.uint8, device=dev)
        self.xq = torch.zeros(2 * MT * H, dtype=torch.uint8, device=dev)
        self.xs = torch.zeros(2 * MT, dtype=torch.float32, device=dev)
        self.qkv = torch.zeros(2, MT, 2304, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(MT, 2048, dtype=torch.bfloat16, device=dev)
        self.act = torch.zeros(MT, 2048, dtype=torch.bfloat16, device=dev)
        self.amax = torch.zeros(2 * 2 * NL * MT, dtype=torch.int32, device=dev)
        self.cnt = torch.zeros(64, dtype=torch.int32, device=dev)
        self.stamps = torch.zeros(120 * 128, dtype=torch.int64, device=dev) if stamps else None
        wptr = []
        for L in layers:
            for k in ("w_qkv", "w_o", "w_gu", "w_dn"):
                w = L[k]
                assert w.dtype == torch.float8_e4m3fn and w.is_contiguous(), k
                wptr.append(w.data_ptr())
        exp = {"w_qkv": (2304, H), "w_o": (H, 2048), "w_gu": (4096, H), "w_dn": (H, 2048)}
        for L in layers:
            for k, shp in exp.items():
                assert tuple(L[k].shape) == shp, (k, tuple(L[k].shape))
            kc, vc = L["k_cache"], L["v_cache"]
            assert kc.dtype == torch.bfloat16 and vc.dtype == torch.bfloat16
            assert kc.stride(0) == vc.stride(0) and kc.stride(-1) == 1 and vc.stride(-1) == 1
        self.kv_stride = int(layers[0]["k_cache"].stride(0))
        for L in layers:
            assert int(L["k_cache"].stride(0)) == self.kv_stride and int(L["v_cache"].stride(0)) == self.kv_stride
            assert L["k_cache"][0].numel() == 128 and L["v_cache"][0].numel() == 128
        self.mod.build_maps(self.maps, torch.tensor(wptr, dtype=torch.int64), self.xq)
        ptrs = []
        for L in layers:
            for k in ("s_qkv", "s_o", "s_gu", "s_dn"):
                s = L[k]
                assert s.dtype == torch.float32 and s.is_contiguous()
                ptrs.append(s.data_ptr())
            for k in ("ln_in", "ln_post", "q_norm", "k_norm"):
                assert L[k].dtype == torch.bfloat16 and L[k].is_contiguous()
                ptrs.append(L[k].data_ptr())
            sk = L.get("sinks")
            ptrs.append(0 if sk is None else sk.data_ptr())
            ptrs.append(L["k_cache"].data_ptr())
            ptrs.append(L["v_cache"].data_ptr())
        peers = tp_state.peers.tolist()
        ptrs += [
            self.maps.data_ptr(), final_norm.data_ptr(), cos_sin_cache.data_ptr(), self.xq.data_ptr(),
            self.xs.data_ptr(), self.qkv.data_ptr(), self.att.data_ptr(), self.act.data_ptr(),
            self.amax.data_ptr(), self.cnt.data_ptr(), tp_state.calls.data_ptr(),
            0 if self.stamps is None else self.stamps.data_ptr(),
        ] + peers
        sinks = [L.get("sinks") for L in layers]
        sink_f32 = int(sinks[0] is not None and sinks[0].dtype == torch.float32)
        assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64 and cos_sin_cache.stride(1) == 1
        assert cos_sin_cache.dtype in (torch.float32, torch.bfloat16)
        ivals = torch.tensor(
            [tp_state.rank, cos_sin_cache.stride(0), int(cos_sin_cache.dtype == torch.float32), sink_f32, self.kv_stride, nl,
             int(os.environ.get("SGLANG_DFLASH_MEGA_LOOK", "0")) if look is None else int(look)],
            dtype=torch.int64,
        )
        fvals = torch.tensor([eps, qk_eps, v_scale, sm_scale], dtype=torch.float32)
        self.plan = plan or default_plan()
        self.blob = torch.zeros(int(self.mod.params_bytes()), dtype=torch.uint8)
        self.mod.fill(self.blob, torch.tensor(ptrs, dtype=torch.int64), ivals, fvals, plan_tensor(self.plan))
        torch.cuda.synchronize(dev)

    def max_clusters(self) -> int:
        return int(self.mod.max_clusters(self.cnt))

    def run(self, hidden, out, positions, seqlens, page_table, loc) -> None:
        self.mod.run(self.blob, hidden, out, positions, seqlens, page_table, loc)
