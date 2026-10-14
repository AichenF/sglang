"""Fused tail of the MiMo-V2 full-attention (FA) layers at exact M = 32 (4 x 8 verify tokens), TP8 (fa_block.cu):
RoPE + new-K/V cache store + split-KV verify attention + split combine + o_proj + the push of the o_proj partial into the fused MoE
kernel's entry reduce-scatter buffers -- one kernel instead of mimo_rope_kv + fa_verify + fa_verify_combine + nvjet o_proj. The fused MoE
launch that follows must run with ``PREPUSHED_MODE`` (it then skips its own entry push), exactly like blk's fused SWA block."""

from __future__ import annotations

import hashlib
import logging
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
_ext = None

PREPUSHED_MODE = 1 << 30   # fused_moe_wgmma.cu Params.rs_prepushed (FMOE_BLK_PREPUSH builds)


def ext():
    global _ext
    if _ext is None:
        extra = os.environ.get("FAB_NVCC_FLAGS", "").split()
        # the extension name carries a hash of the source (+ flags): torch's JIT rebuild is mtime-based and different source trees may
        # share one TORCH_EXTENSIONS_DIR (an older-mtime source would otherwise silently load another version's binary)
        with open(os.path.join(HERE, "fa_block.cu"), "rb") as f:
            src_hash = hashlib.sha1(f.read()).hexdigest()[:12]
        name = "mimo_fa_block_" + src_hash + ("_" + "".join(ch if ch.isalnum() else "_" for ch in "".join(extra)) if extra else "")
        _ext = load(
            name=name,
            sources=[os.path.join(HERE, "fa_block.cu")],
            # --use_fast_math: the attention / combine math must match fa_verify_decode (JIT-built with it) bit for bit
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"] + extra,
            verbose=False,
        )
    return _ext


class Scratch:
    """Per-process scratch shared by every FA layer (flags are epoch-tagged with the fused MoE launch counter: no reset)."""

    def __init__(self, device):
        e = ext()
        self.part_o = torch.empty(4 * 128 * e.NSPLIT * 128, dtype=torch.float32, device=device)
        self.part_lse = torch.empty(4 * 128 * e.NSPLIT, dtype=torch.float32, device=device)
        self.attn_out = torch.zeros(32, 2048, dtype=torch.bfloat16, device=device)
        self.amap = e.make_amap(self.attn_out)
        self.flags_att = torch.zeros(e.NCTA * 32, dtype=torch.int32, device=device)
        self.flags_cmb = torch.zeros(e.NOP * 32, dtype=torch.int32, device=device)


_EMPTY_I64 = {}


def _empty_i64(device):
    return _EMPTY_I64.setdefault(str(device), torch.empty(0, dtype=torch.int64, device=device))


def run(qkv, cos_sin, positions, loc, k_cache, v_cache, page_table, seqlens, scratch: Scratch, partial, wmap, peers_cpu, work, my_rank, sm_scale,
        mode=0, pdl=True, dbg=None):
    """qkv bf16 [32][3392] (un-rotated) -> partial bf16 [32][6144] (rows of the tokens this rank owns; every row with MODE_FULL_LOCAL), the
    RS messages of the other rows pushed into the peers' fused MoE entry buffers (epoch = work[2] + 1), the 8 new K/V rows per request stored
    to the cache at loc, scratch.attn_out = the combined attention output."""
    ext().fa_block(qkv, cos_sin, positions, loc, k_cache, v_cache, page_table, seqlens, scratch.part_o, scratch.part_lse, scratch.attn_out,
                   scratch.flags_att, scratch.flags_cmb, partial, wmap, scratch.amap, peers_cpu, work, int(my_rank), float(sm_scale), int(mode),
                   int(bool(pdl)), dbg if dbg is not None else _empty_i64(qkv.device))


# ---------------------------------------------------------------------------------------------------------------- SGLang integration
_SCRATCH = {}
_logged = set()


def _rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _log_once(key, msg, *args, info=False):
    """Log once per process: fallbacks as warnings, the 'enabled' / 'running' notices (info=True) at INFO level on rank 0 only."""
    if key in _logged:
        return
    _logged.add(key)
    if not info:
        logger.warning("MiMo fa2: " + msg, *args)
    elif _rank0():
        logger.info("MiMo fa2: " + msg, *args)


def enabled() -> bool:
    # default ON; SGLANG_MIMO_FA2=0 = off
    return os.environ.get("SGLANG_MIMO_FA2", "1") == "1"


# Diagnostics (off unless set): per-layer in-kernel globaltimer stamps [132][32]. Creating <dir>/dump.<n> (the fused MoE / blk stamp-dump
# trigger, same dir) makes every rank write <dir>/fa2_stamps_rank<r>_<n>.pt = {layer_id: int64 [132 * 32]} of the last replay.
_STAMP_DIR = os.environ.get("SGLANG_MIMO_FA2_STAMP_DIR", "")
_stamp_layers = []
_stamp_thread = None


def _stamp_watcher(rank):
    import time

    n = 0
    while True:
        if os.path.exists(os.path.join(_STAMP_DIR, f"dump.{n}")):
            try:
                torch.save({lid: t.cpu() for lid, t in _stamp_layers}, os.path.join(_STAMP_DIR, f"fa2_stamps_rank{rank}_{n}.pt"))
            except Exception as e:  # diagnostics must never kill the server
                logger.warning(f"MiMo fa2 stamp dump failed: {e}")
            n += 1
        time.sleep(0.2)


def _scratch(device) -> Scratch:
    key = str(device)
    if key not in _SCRATCH:
        _SCRATCH[key] = Scratch(device)
    return _SCRATCH[key]


class FaBlock:
    """The fused tail of one full-attention MiMoV2DecoderLayer followed by the fused MoE (created lazily, outside CUDA-graph capture)."""

    kind = "fa2"

    def __init__(self, decoder_layer, shared, push=True):
        attn = decoder_layer.self_attn
        w = attn.o_proj.weight
        self.weight = w
        self.wmap = ext().make_wmap(w.data)
        self.shared = shared          # the fused MoE's per-rank state: prologue IPC buffers, work counters (epoch), rank
        self.push = push              # False: dense-MLP layer (layer 0) -- every partial row local, no RS push, the dense TP path reduces
        self.sm_scale = float(attn.attn.scaling)
        self.pdl = os.environ.get("SGLANG_MIMO_FA2_PDL", "1") == "1"
        self.scratch = _scratch(w.device)
        self.dbg = None
        if _STAMP_DIR:
            global _stamp_thread
            self.dbg = torch.zeros(ext().NCTA * 32, dtype=torch.int64, device=w.device)
            _stamp_layers.append((decoder_layer.layer_id, self.dbg))
            if _stamp_thread is None:
                import threading

                _stamp_thread = threading.Thread(target=_stamp_watcher, args=(shared.rank,), daemon=True)
                _stamp_thread.start()

    @staticmethod
    def ready(decoder_layer, forward_batch, m):
        """The FaBlock to run this forward's full attention with (the MoE then runs prepushed), or None. Deterministic in the shapes /
        config, so every TP rank takes the same decision."""
        if not enabled() or m != 32 or decoder_layer.is_swa_layer():
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        from sglang.srt.layers.moe.mimo_fused_moe import fused_moe_wgmma as F
        from sglang.srt.layers.moe.mimo_fused_moe import layer as FL

        if decoder_layer.is_layer_sparse:
            fmoe = getattr(decoder_layer, "_fused_moe", None)
            if fmoe is None or not fmoe.shared.input_tp or fmoe.router_w.dtype != torch.bfloat16:
                return None
            if not getattr(F.ext(), "BLK_PREPUSH", 0):
                return None
            residual = FL.fused_moe_stream_residual(forward_batch)
            if residual is None or residual.shape != (32, F.DIM) or not fmoe.applicable(residual, residual, forward_batch):
                return None
            shared, push = fmoe.shared, True
        else:
            # dense-MLP layer (layer 0): the partial stays local (MODE_FULL_LOCAL), the dense TP path reduces it; the fused MoE state only
            # supplies the epoch counter (it advances between any two FA-layer launches) and the input-TP peer table the launch takes
            shared = getattr(FL, "_shared", None)
            if os.environ.get("SGLANG_MIMO_FA2_DENSE", "1") != "1" or shared is None or not shared.input_tp:
                return None
            push = False
        if not F.full_grid_ok(shared.device):   # 132 co-resident CTAs spin-wait on each other
            return None
        attn = decoder_layer.self_attn
        if getattr(attn, "attention_sink_bias", None) is not None:
            _log_once("fb_sinks", "not used: full-attention layer with sinks")
            return None
        fb = getattr(decoder_layer, "_fa2", None)
        if fb is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            w = attn.o_proj.weight
            if w.dtype != torch.bfloat16 or tuple(w.shape) != (6144, 2048) or w.stride(1) != 1:
                _log_once("fb_w", "not used: o_proj weight %s %s", w.dtype, tuple(w.shape))
                return None
            fb = FaBlock(decoder_layer, shared, push)
            object.__setattr__(decoder_layer, "_fa2", fb)
            _log_once("created", "fused full-attention tail enabled (first: layer %d)", decoder_layer.layer_id, info=True)
        return fb

    def run(self, attn, qkv, positions, forward_batch):
        """qkv [32][3392] (un-rotated) -> this rank's o_proj partial (owned rows valid; the rest pushed to the MoE entry), tagged
        ``_blk_prepushed``; None when this forward's metadata does not fit (the caller then runs the unfused path)."""
        from sglang.srt.layers.attention.mimo_rope_kv import select_mimo_rope_kv
        from sglang.srt.model_executor.forward_context import get_attn_backend

        sel = select_mimo_rope_kv(attn, qkv, positions, forward_batch)
        if sel is None:
            _log_once("fb_sel", "fallback: select_mimo_rope_kv rejected qkv %s", tuple(qkv.shape))
            return None
        cos_sin, locations, keys, values = sel
        backend = get_attn_backend()
        md = backend.forward_metadata
        layer = attn.attn
        lab = getattr(backend, "local_attn_builder", None)
        if (
            (lab is not None and lab.applies(layer, md))
            or backend.page_size != 1
            or getattr(layer, "logit_cap", 0) not in (0, 0.0, None)
            or getattr(layer, "is_cross_attention", False)
            or (getattr(layer, "sliding_window_size", -1) or -1) > -1
            or getattr(backend, "fa_skip_kv_cache", False)
        ):
            _log_once("fb_backend", "fallback: backend (local %s, page %s)", lab is not None, backend.page_size)
            return None
        if any(t.shape[0] * t.stride(0) * t.element_size() >= (1 << 32) for t in (keys, values)):
            _log_once("fb_kv4g", "fallback: KV cache larger than 4 GB (the kernel uses 32-bit cache offsets)")
            return None
        page_table = md.page_table
        seqlens = md.cache_seqlens_int32
        if (
            page_table is None
            or page_table.dtype != torch.int32
            or page_table.dim() != 2
            or page_table.shape[0] != 4
            or page_table.stride(1) != 1
            or page_table.stride(0) % 4 != 0
            or page_table.data_ptr() % 16 != 0
            or seqlens is None
            or seqlens.dtype != torch.int32
            or seqlens.numel() != 4
            or not seqlens.is_contiguous()
            or md.max_seq_len_q != 8
        ):
            _log_once("fb_meta", "fallback: metadata (page_table %s, seqlens %s, max_seq_len_q %s)",
                      None if page_table is None else (page_table.dtype, tuple(page_table.shape), page_table.stride()),
                      None if seqlens is None else (seqlens.dtype, tuple(seqlens.shape)), md.max_seq_len_q)
            return None
        partial = torch.empty(32, 6144, dtype=torch.bfloat16, device=qkv.device)
        st = self.shared
        run(qkv, cos_sin, positions, locations, keys, values, page_table, seqlens, self.scratch, partial, self.wmap, st.state.pro_cpu,
            st.state.work, st.rank, self.sm_scale, 0 if self.push else ext().MODE_FULL_LOCAL, self.pdl, dbg=self.dbg)
        if self.push:
            partial._blk_prepushed = True
        _log_once("ran", "fused full-attention tail running (layer %d)", layer.layer_id, info=True)
        return partial
