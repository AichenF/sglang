"""goal11 "blk": fused MiMo-V2 SWA target-verify attention block (attn_block.cu) -- RoPE + KV store + sliding-window attention +
o_proj + the push of the o_proj partial into the fused MoE kernel's entry reduce-scatter buffers, for exact M = 32 (4 x 8 verify
tokens) at TP8. The fused MoE launch that follows must run with ``PREPUSHED_MODE`` (it then skips its own entry push)."""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None

# fused_moe_wgmma.cu's Params.rs_prepushed is taken from this bit of the launch's `mode` argument (FMOE_BLK_PREPUSH builds)
PREPUSHED_MODE = 1 << 30


def ext():
    global _ext
    if _ext is None:
        extra = os.environ.get("BLK_NVCC_FLAGS", "").split()
        # the extension name carries a hash of the source: torch's JIT build decides by file mtimes (ninja), and E2E runs of different
        # snapshots share one TORCH_EXTENSIONS_DIR -- an older-mtime source would otherwise silently load another version's binary
        import hashlib

        with open(os.path.join(HERE, "attn_block.cu"), "rb") as f:
            src_hash = hashlib.sha1(f.read()).hexdigest()[:12]
        name = "mimo_attn_block_" + src_hash + (
            "_" + "".join(c if c.isalnum() else "_" for c in "".join(extra)) if extra else ""
        )
        _ext = load(
            name=name,
            sources=[os.path.join(HERE, "attn_block.cu")],
            # --use_fast_math: the attention math must match swa_decode_mma (JIT-built with it: approximate 1/l, ftz exp2) bit for bit
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"] + extra,
            verbose=False,
        )
    return _ext


class Scratch:
    """Per-process scratch shared by every layer (each launch tags its flags with the fused MoE epoch, so no reset is needed)."""

    def __init__(self, device):
        self.attn_out = torch.zeros(32, 2048, dtype=torch.bfloat16, device=device)
        self.flags = torch.zeros(64 * 32, dtype=torch.int32, device=device)   # one u32 per 128-B line, up to 64 attention CTAs
        self.amap = ext().make_amap(self.attn_out)
        self.slots = torch.zeros(4 * 144, dtype=torch.int32, device=device)   # per-forward key-row slot cache (first SWA layer writes)
        self.slots_writer = None   # layer_id of the SWA layer that writes it
        # goal11 mega (qkv GEMM as phase 0): the qkv rows [32][3392] bf16 and one ready flag per CTA (epoch-tagged, no reset)
        self.qout = torch.zeros(32, 3392, dtype=torch.bfloat16, device=device)
        self.qflags = torch.zeros(128 * 32, dtype=torch.int32, device=device)
        # qkv-only launches (FA layers) release their flags with a fixed epoch (dummy_work): they get their own flag lines so a
        # SWA megakernel launch polling ``qflags`` for epoch work[2] + 1 can never match a stale qkv-only release after wraparound
        self.qflags_qkv = torch.zeros(128 * 32, dtype=torch.int32, device=device)
        self.qll = torch.zeros(32 * 128 * 6 * 4, dtype=torch.int32, device=device)   # LL messages {6 bf16, epoch} [32 tok][128 CTA][6]
        self.dummy_work = torch.zeros(8, dtype=torch.int32, device=device)   # qkv-only launches: epoch source (their flags are not polled)


def run(
    qkv,
    cos_sin,
    positions,
    loc,
    k_cache,
    v_cache,
    page_table,
    seqlens,
    sinks,
    scratch: Scratch,
    partial,
    wmap,
    peers_cpu,
    work,
    my_rank,
    sm_scale,
    window,
    mode=0,
    pdl=False,
    dbg=None,
    nkv=None,
):
    """partial: bf16 [32][6144] output (the rows of the tokens this rank owns, t % 8 == my_rank; every row with MODE_FULL_LOCAL)."""
    ext().attn_block_swa(
        qkv,
        cos_sin,
        positions,
        loc,
        k_cache,
        v_cache,
        page_table,
        seqlens,
        sinks if sinks is not None else _empty(qkv.device),
        scratch.attn_out,
        scratch.flags,
        partial,
        wmap,
        scratch.amap,
        peers_cpu,
        work,
        int(my_rank),
        float(sm_scale),
        int(window),
        int(mode),
        int(bool(pdl)),
        dbg if dbg is not None else _EMPTY_I64.setdefault(str(qkv.device), torch.empty(0, dtype=torch.int64, device=qkv.device)),
        scratch.slots,
        nkv[0] if nkv is not None else _empty_bf16(qkv.device),
        nkv[1] if nkv is not None else _empty_bf16(qkv.device),
    )


def run_mega(
    x,
    x_scale,
    qwmap,
    qsw,
    cos_sin,
    positions,
    loc,
    k_cache,
    v_cache,
    page_table,
    seqlens,
    sinks,
    scratch: Scratch,
    partial,
    wmap,
    peers_cpu,
    work,
    my_rank,
    sm_scale,
    window,
    mode=0,
    pdl=False,
    dbg=None,
    nkv=None,
    qpre=0,
    qout=None,
    xflags=None,
    xwork=None,
):
    """attention megakernel: x [32][6144] e4m3 + x_scale [32][48] fp32 (deep_gemm column-major, stride (1, ld)) -> qkv_proj (qwmap =
    ext().make_qwmap(weight), qsw = weight_scale_inv [27][48]) + the fused block of run(); the qkv rows land in scratch.qout."""
    if qout is None:
        qout = scratch.qout if not getattr(ext(), "MEGA_LL", 0) else _empty_bf16(x.device)
    ext().attn_block_mega(
        x,
        x_scale.t(),
        qwmap,
        qsw,
        qout,
        scratch.qflags,
        scratch.qll,
        int(qpre),
        cos_sin,
        positions,
        loc,
        k_cache,
        v_cache,
        page_table,
        seqlens,
        sinks if sinks is not None else _empty(x.device),
        scratch.attn_out,
        scratch.flags,
        partial,
        wmap,
        scratch.amap,
        peers_cpu,
        work,
        int(my_rank),
        float(sm_scale),
        int(window),
        int(mode),
        int(bool(pdl)),
        dbg if dbg is not None else _EMPTY_I64.setdefault(str(x.device), torch.empty(0, dtype=torch.int64, device=x.device)),
        scratch.slots,
        nkv[0] if nkv is not None else _empty_bf16(x.device),
        nkv[1] if nkv is not None else _empty_bf16(x.device),
        xflags if xflags is not None else _empty_i32(x.device),
        xwork if xwork is not None else _empty_i32(x.device),
    )


def run_qkv(x, x_scale, qwmap, qsw, out, scratch: Scratch, pdl=True, qpre=0, dbg=None, xflags=None, xwork=None):
    """FA layers: the megakernel's qkv GEMM alone -> out bf16 [32][3392] (x / x_scale = the norm-next (fp8, column-major scales) pair)."""
    ext().attn_block_qkv(
        x, x_scale.t(), qwmap, qsw, out, scratch.qflags_qkv, scratch.dummy_work, int(qpre), int(bool(pdl)),
        dbg if dbg is not None else _EMPTY_I64.setdefault(str(x.device), torch.empty(0, dtype=torch.int64, device=x.device)),
        xflags if xflags is not None else _empty_i32(x.device),
        xwork if xwork is not None else _empty_i32(x.device),
    )


def run_ext(attn, partial, wmap, peers_cpu, work, my_rank, mode=0, pdl=False, dbg=None):
    """FA layers: o_proj of `attn` (contiguous bf16 [32][2048], the fa3 output) + the fused MoE entry reduce-scatter push."""
    amap = ext().make_amap(attn)
    ext().attn_block_ext(
        attn, partial, wmap, amap, peers_cpu, work, int(my_rank), int(mode), int(bool(pdl)),
        dbg if dbg is not None else _EMPTY_I64.setdefault(str(attn.device), torch.empty(0, dtype=torch.int64, device=attn.device)),
    )


_EMPTY = {}
_EMPTY_I64 = {}
_EMPTY_BF16 = {}


_EMPTY_I32 = {}


def _empty_i32(device):
    return _EMPTY_I32.setdefault(str(device), torch.empty(0, dtype=torch.int32, device=device))


def x_ready_flags(xq):
    """goal11 mega (MEGA_XFLAG): the producing fused MoE's x-row flags + work counters attached to its norm-next x_fp8 buffer, or (None, None)."""
    if not mega_xflag_enabled():
        return None, None
    xf, xw = getattr(xq, "_nn_xflags", None), getattr(xq, "_nn_work", None)
    if xf is None or xw is None or not getattr(ext(), "MEGA_XFLAG", 0):
        return None, None
    return xf, xw


def _empty_bf16(device):
    return _EMPTY_BF16.setdefault(str(device), torch.empty(0, dtype=torch.bfloat16, device=device))


def _empty(device):
    key = str(device)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, dtype=torch.float32, device=device)
    return _EMPTY[key]


# ---------------------------------------------------------------------------------------------------------------- SGLang integration
import logging

logger = logging.getLogger(__name__)
_SCRATCH = {}
_logged = set()


# Diagnostics (off unless set): per-layer in-kernel globaltimer stamps [NCTA][32] (kernel built with them always; only the pointer is
# runtime). Creating <dir>/dump.<n> (the fused MoE stamp-dump trigger, same dir) makes every rank write <dir>/blk_stamps_rank<r>_<n>.pt =
# {layer_id: int64 [NCTA * 32]} of the last replay.
_STAMP_DIR = os.environ.get("SGLANG_MIMO_BLK_STAMP_DIR", "")
_stamp_layers = []
_stamp_thread = None


def _stamp_watcher(rank):
    import time

    n = 0
    while True:
        if os.path.exists(os.path.join(_STAMP_DIR, f"dump.{n}")):
            try:
                torch.save({lid: t.cpu() for lid, t in _stamp_layers}, os.path.join(_STAMP_DIR, f"blk_stamps_rank{rank}_{n}.pt"))
            except Exception as e:  # diagnostics must never kill the server
                logger.warning(f"MiMo blk stamp dump failed: {e}")
            n += 1
        time.sleep(0.2)


def _log_once(key, msg, *args):
    if key not in _logged:
        _logged.add(key)
        logger.warning("MiMo blk: " + msg, *args)


def enabled() -> bool:
    # default ON since the E2E validation of b36a7316bc on top of combo3 (g11b_c3b_on/off: -0.23 ms/step, parity 4/4); SGLANG_MIMO_BLK=0 = off
    return os.environ.get("SGLANG_MIMO_BLK", "1") == "1"


def mega_enabled() -> bool:
    # goal11 mega: the qkv_proj GEMM runs as phase 0 of the fused SWA block (attn_block_mega); SGLANG_MIMO_MEGA=0 = blk's separate GEMM.
    # Default ON since the E2E validation of 0d2aa1c6ea (g11m_m12: 10.573 / 10.568 vs knob off 11.164 / 11.060 ms, parity 4/4).
    return os.environ.get("SGLANG_MIMO_MEGA", "1") == "1"


def _scratch(device) -> Scratch:
    key = str(device)
    if key not in _SCRATCH:
        _SCRATCH[key] = Scratch(device)
    return _SCRATCH[key]


class BlkLayer:
    """The fused attention-block path of one SWA MiMoV2DecoderLayer (created lazily, outside CUDA-graph capture)."""

    def __init__(self, decoder_layer, fused_moe):
        attn = decoder_layer.self_attn
        w = attn.o_proj.weight
        self.weight = w
        self.wmap = ext().make_wmap(w.data)
        sinks = attn.attention_sink_bias
        self.sinks = None
        if sinks is not None:
            self.sinks = sinks.data if sinks.dtype == torch.float32 else sinks.data.float().contiguous()
            if sinks.dtype != torch.float32:
                _log_once("sinks", "attention_sink_bias is %s, using an fp32 copy", sinks.dtype)
        self.fmoe = fused_moe
        self.kind = "swa" if decoder_layer.is_swa_layer() else "fa"
        self.window = int(attn.attn.sliding_window_size) + 1
        self.sm_scale = float(attn.attn.scaling)
        self.pdl = os.environ.get("SGLANG_MIMO_BLK_PDL", "1") == "1"
        self.scratch = _scratch(w.device)
        self.slot_mode = 0
        if self.kind == "swa" and os.environ.get("SGLANG_MIMO_BLK_SLOTS", "1") == "1":
            if self.scratch.slots_writer is None:
                self.scratch.slots_writer = decoder_layer.layer_id   # created in layer order: the first SWA layer of every forward
            self.slot_mode = ext().MODE_SLOTS_WRITE if self.scratch.slots_writer == decoder_layer.layer_id else ext().MODE_SLOTS_READ
        # the next SWA layer (skipping FA layers): its K/V window rows are L2-prefetched at the end of this layer's attention
        self.next_swa_lid = None
        nxt = getattr(decoder_layer, "_mimo_next_layer", None)
        while nxt is not None and os.environ.get("SGLANG_MIMO_BLK_KV_PREFETCH", "1") == "1":
            if nxt.is_swa_layer():
                self.next_swa_lid = nxt.layer_id
                break
            nxt = getattr(nxt, "_mimo_next_layer", None)
        self.qwmap = None
        if self.kind == "swa" and mega_enabled():
            self._init_mega(attn)
        self.dbg = None
        if _STAMP_DIR:
            global _stamp_thread
            self.dbg = torch.zeros(128 * 32, dtype=torch.int64, device=w.device)
            _stamp_layers.append((decoder_layer.layer_id, self.dbg))
            if _stamp_thread is None:
                import threading

                _stamp_thread = threading.Thread(target=_stamp_watcher, args=(fused_moe.shared.rank,), daemon=True)
                _stamp_thread.start()

    def _init_mega(self, attn):
        qp = attn.qkv_proj
        w = getattr(qp, "weight", None)
        sc = getattr(qp, "weight_scale_inv", None)
        if (
            w is None
            or sc is None
            or w.dtype != torch.float8_e4m3fn
            or tuple(w.shape) != (3392, 6144)
            or w.stride(1) != 1
            or w.stride(0) % 16 != 0
            or w.data_ptr() % 16 != 0
            or sc.dtype != torch.float32
            or tuple(sc.shape) != (27, 48)
            or not sc.is_contiguous()
            or getattr(qp, "bias", None) is not None
            or list(getattr(getattr(qp, "quant_method", None), "weight_block_size", None) or []) != [128, 128]
        ):
            _log_once("mega_fb_w", "mega not used: qkv_proj weight %s %s scale %s %s", None if w is None else w.dtype,
                      None if w is None else tuple(w.shape), None if sc is None else sc.dtype, None if sc is None else tuple(sc.shape))
            return
        self.qweight = w
        self.qsw = sc
        self.qwmap = ext().make_qwmap(w.data)
        self.qpre = int(os.environ.get("SGLANG_MIMO_MEGA_QPRE", "16"))
        _log_once("mega_on", "attention megakernel enabled (qkv GEMM phase 0, qpre %d)", self.qpre)

    @staticmethod
    def ready(decoder_layer, forward_batch, m):
        """The BlkLayer to run this forward's attention with (the MoE kernel then runs prepushed), or None. Deterministic in the
        shapes / config, so every TP rank takes the same decision."""
        if not enabled() or m != 32:
            return None
        if not decoder_layer.is_swa_layer() and os.environ.get("SGLANG_MIMO_BLK_FA", "0") != "1":   # FA mode: no in-situ gain yet
            return None
        fmoe = getattr(decoder_layer, "_fused_moe", None)
        if fmoe is None or not fmoe.shared.input_tp or fmoe.router_w.dtype != torch.bfloat16:
            return None
        from sglang.srt.layers.moe.mimo_fused_moe import fused_moe_wgmma as F
        from sglang.srt.layers.moe.mimo_fused_moe.layer import fused_moe_stream_residual

        if not F.full_grid_ok(fmoe.router_w.device):   # 128 co-resident CTAs spin-wait on each other
            return None
        if not getattr(F.ext(), "BLK_PREPUSH", 0):
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        residual = fused_moe_stream_residual(forward_batch)
        if residual is None or residual.shape != (32, F.DIM) or not fmoe.applicable(residual, residual, forward_batch):
            _log_once("fb_res", "not used: residual %s", None if residual is None else tuple(residual.shape))
            return None
        blk = getattr(decoder_layer, "_blk", None)
        if blk is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            blk = BlkLayer(decoder_layer, fmoe)
            object.__setattr__(decoder_layer, "_blk", blk)
            _log_once("created_" + blk.kind, "fused attention block enabled (%s layers, first: layer %d, window %d)", blk.kind, decoder_layer.layer_id, blk.window)
        return blk

    def run(self, attn, qkv, positions, forward_batch):
        """rope + KV store + SWA attention + o_proj + reduce-scatter push. Returns this rank's partial (rows of its owned tokens valid)
        tagged ``_blk_prepushed``, or None when this forward's attention metadata does not fit (then the unfused path runs)."""
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
            or getattr(md, "swa_spec_metadata", None) is not None
            or backend.page_size != 1
            or self.window not in (128, 129)
            or getattr(layer, "logit_cap", 0) not in (0, 0.0, None)
            or getattr(layer, "is_cross_attention", False)
        ):
            _log_once("fb_backend", "fallback: backend (local %s, spec %s, page %s, window %s)", lab is not None,
                      getattr(md, "swa_spec_metadata", None) is not None, backend.page_size, self.window)
            return None
        if backend.use_sliding_window_kv_pool:
            page_table = md.swa_page_table
        else:
            page_table = md.page_table
        seqlens = md.cache_seqlens_int32
        if (
            page_table is None
            or page_table.dtype != torch.int32
            or page_table.dim() != 2
            or page_table.shape[0] != 4
            or page_table.stride(1) != 1
            or seqlens is None
            or seqlens.dtype != torch.int32
            or seqlens.numel() != 4
            or md.max_seq_len_q != 8
        ):
            _log_once("fb_meta", "fallback: metadata (page_table %s, seqlens %s, max_seq_len_q %s)",
                      None if page_table is None else (page_table.dtype, tuple(page_table.shape), page_table.stride()),
                      None if seqlens is None else (seqlens.dtype, tuple(seqlens.shape)), md.max_seq_len_q)
            return None
        partial = torch.empty(32, 6144, dtype=torch.bfloat16, device=qkv.device)
        st = self.fmoe.shared
        run(
            qkv,
            cos_sin,
            positions,
            locations,
            keys,
            values,
            page_table,
            seqlens,
            self.sinks,
            self.scratch,
            partial,
            self.wmap,
            st.state.pro_cpu,
            st.state.work,
            st.rank,
            self.sm_scale,
            self.window,
            self.slot_mode,
            self.pdl,
            dbg=self.dbg,
            nkv=self._next_kv(backend, keys, values),
        )
        partial._blk_prepushed = True
        _log_once("ran_swa", "fused SWA attention block running (layer %d)", layer.layer_id)
        return partial

    def run_mega(self, attn, x, positions, forward_batch):
        """qkv_proj (x = the norm-next (x_fp8, x_scale) pair) + rope + KV store + SWA attention + o_proj + reduce-scatter push in one
        kernel. Returns the prepushed partial, or None (the caller then runs qkv_proj + run())."""
        if self.qwmap is None or not isinstance(x, tuple) or len(x) != 2:
            return None
        xq, xs = x
        if (
            xq.dtype != torch.float8_e4m3fn
            or tuple(xq.shape) != (32, 6144)
            or xq.stride(1) != 1
            or xq.stride(0) % 16 != 0
            or xq.data_ptr() % 16 != 0
            or xs.dtype != torch.float32
            or tuple(xs.shape) != (32, 48)
            or xs.stride(0) != 1
            or xs.stride(1) != 32
            or xs.data_ptr() % 16 != 0
        ):
            _log_once("mega_fb_x", "mega fallback: x %s %s %s scale %s %s %s", xq.dtype, tuple(xq.shape), xq.stride(), xs.dtype,
                      tuple(xs.shape), xs.stride())
            return None
        from sglang.srt.layers.attention.mimo_rope_kv import select_mimo_rope_kv
        from sglang.srt.model_executor.forward_context import get_attn_backend

        sel = select_mimo_rope_kv(attn, self.scratch.qout, positions, forward_batch)
        if sel is None:
            _log_once("mega_fb_sel", "mega fallback: select_mimo_rope_kv rejected")
            return None
        cos_sin, locations, keys, values = sel
        backend = get_attn_backend()
        md = backend.forward_metadata
        layer = attn.attn
        lab = getattr(backend, "local_attn_builder", None)
        if (
            (lab is not None and lab.applies(layer, md))
            or getattr(md, "swa_spec_metadata", None) is not None
            or backend.page_size != 1
            or self.window not in (128, 129)
            or getattr(layer, "logit_cap", 0) not in (0, 0.0, None)
            or getattr(layer, "is_cross_attention", False)
        ):
            _log_once("mega_fb_backend", "mega fallback: backend")
            return None
        page_table = md.swa_page_table if backend.use_sliding_window_kv_pool else md.page_table
        seqlens = md.cache_seqlens_int32
        if (
            page_table is None
            or page_table.dtype != torch.int32
            or page_table.dim() != 2
            or page_table.shape[0] != 4
            or page_table.stride(1) != 1
            or seqlens is None
            or seqlens.dtype != torch.int32
            or seqlens.numel() != 4
            or md.max_seq_len_q != 8
        ):
            _log_once("mega_fb_meta", "mega fallback: metadata")
            return None
        partial = torch.empty(32, 6144, dtype=torch.bfloat16, device=xq.device)
        st = self.fmoe.shared
        xf, xw = x_ready_flags(xq)
        run_mega(
            xq,
            xs,
            self.qwmap,
            self.qsw,
            cos_sin,
            positions,
            locations,
            keys,
            values,
            page_table,
            seqlens,
            self.sinks,
            self.scratch,
            partial,
            self.wmap,
            st.state.pro_cpu,
            st.state.work,
            st.rank,
            self.sm_scale,
            self.window,
            self.slot_mode,
            self.pdl,
            dbg=self.dbg,
            nkv=self._next_kv(backend, keys, values),
            qpre=self.qpre,
            xflags=xf,
            xwork=xw,
        )
        partial._blk_prepushed = True
        _log_once("ran_mega", "attention megakernel running (layer %d)", layer.layer_id)
        return partial

    def _next_kv(self, backend, keys, values):
        if self.next_swa_lid is None:
            return None
        try:
            pool = backend.token_to_kv_pool
            nk, nv = pool.get_key_buffer(self.next_swa_lid), pool.get_value_buffer(self.next_swa_lid)
        except Exception:
            return None
        if nk.shape != keys.shape or nv.shape != values.shape or nk.stride() != keys.stride() or nv.stride() != values.stride():
            return None
        return nk, nv

    def run_fa(self, attn_output):
        """FA layers: o_proj of the fa3 output + the reduce-scatter push; None when the output layout does not fit."""
        if self.kind != "fa" or attn_output.dtype != torch.bfloat16 or attn_output.numel() != 32 * 2048:
            _log_once("fb_fa", "fallback (FA): attention output %s %s", attn_output.dtype, tuple(attn_output.shape))
            return None
        a = attn_output.reshape(32, 2048)
        if not a.is_contiguous() or a.data_ptr() % 16:
            _log_once("fb_fa2", "fallback (FA): attention output not contiguous / aligned")
            return None
        partial = torch.empty(32, 6144, dtype=torch.bfloat16, device=a.device)
        st = self.fmoe.shared
        run_ext(a, partial, self.wmap, st.state.pro_cpu, st.state.work, st.rank, 0, self.pdl, dbg=self.dbg)
        partial._blk_prepushed = True
        _log_once("ran_fa", "fused FA o_proj + push running")
        return partial


def mega_xflag_enabled() -> bool:
    # goal11 mega: start the qkv GEMM on the fused MoE's x-row flags (FMOE_NN_XFLAG / MEGA_XFLAG) instead of griddepcontrol.wait alone
    return mega_enabled() and os.environ.get("SGLANG_MIMO_MEGA_XFLAG", "0") == "1"


def mega_fa_enabled() -> bool:
    return mega_enabled() and os.environ.get("SGLANG_MIMO_MEGA_FA", "1") == "1"


class MegaQkv:
    """FA layers: qkv_proj through the megakernel's GEMM phase (MODE_QKV_ONLY) instead of DeepGEMM."""

    def __init__(self, attn):
        qp = attn.qkv_proj
        self.ok = False
        w = getattr(qp, "weight", None)
        sc = getattr(qp, "weight_scale_inv", None)
        if (
            w is None
            or sc is None
            or w.dtype != torch.float8_e4m3fn
            or tuple(w.shape) != (3392, 6144)
            or w.stride(1) != 1
            or w.stride(0) % 16 != 0
            or w.data_ptr() % 16 != 0
            or sc.dtype != torch.float32
            or tuple(sc.shape) != (27, 48)
            or not sc.is_contiguous()
            or getattr(qp, "bias", None) is not None
        ):
            return
        self.qweight = w
        self.qsw = sc
        self.qwmap = ext().make_qwmap(w.data)
        self.scratch = _scratch(w.device)
        self.qpre = int(os.environ.get("SGLANG_MIMO_MEGA_QPRE", "16"))
        self.ok = True

    @staticmethod
    def ready(attn, x, forward_batch):
        if not mega_fa_enabled() or not isinstance(x, tuple) or len(x) != 2:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        sw = getattr(attn.attn, "sliding_window_size", -1)
        if sw is not None and sw > -1:
            return None
        xq, xs = x
        if (
            xq.dtype != torch.float8_e4m3fn
            or tuple(xq.shape) != (32, 6144)
            or xq.stride(1) != 1
            or xq.stride(0) % 16 != 0
            or xq.data_ptr() % 16 != 0
            or xs.dtype != torch.float32
            or tuple(xs.shape) != (32, 48)
            or xs.stride(0) != 1
            or xs.stride(1) != 32
            or xs.data_ptr() % 16 != 0
        ):
            return None
        from sglang.srt.layers.moe.mimo_fused_moe import fused_moe_wgmma as F

        if not F.full_grid_ok(xq.device):
            return None
        mq = getattr(attn, "_mega_qkv", None)
        if mq is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            mq = MegaQkv(attn)
            object.__setattr__(attn, "_mega_qkv", mq)
            _log_once("mega_fa_" + str(mq.ok), "FA-layer qkv GEMM via the megakernel: %s", mq.ok)
        return mq if mq.ok else None

    def run(self, x):
        out = torch.empty(32, 3392, dtype=torch.bfloat16, device=x[0].device)
        xf, xw = x_ready_flags(x[0])
        run_qkv(x[0], x[1], self.qwmap, self.qsw, out, self.scratch, pdl=True, qpre=self.qpre, xflags=xf, xwork=xw)
        return out
