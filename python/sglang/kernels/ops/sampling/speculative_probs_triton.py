"""Exact FP32 preparation/metadata operations; no new probability reduction."""
import triton
import triton.language as tl


@triton.jit
def _prepare(X, TEMPERATURE, KS, PS, SCALED, TARGET, REPEATED_KS, REPEATED_PS,
             ROW_STRIDE: tl.constexpr, V: tl.constexpr, DRAFT: tl.constexpr,
             HAS_K: tl.constexpr, HAS_P: tl.constexpr, ZERO_TARGET: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    columns = chunk * BLOCK + tl.arange(0, BLOCK)
    request = row // DRAFT
    temperature = tl.load(TEMPERATURE + request)
    x = tl.load(X + row * ROW_STRIDE + columns, columns < V, other=0.0)
    # Use true RN division, not reciprocal multiplication or temperature
    # movement across top-k. Preserve the original FP32 intermediate.
    scaled = tl.div_rn(x, temperature)
    tl.store(SCALED + row * V + columns, scaled, columns < V)
    if ZERO_TARGET:
        tl.store(TARGET + row * V + columns, 0.0, columns < V)
    if chunk == 0:
        if HAS_K:
            tl.store(REPEATED_KS + row, tl.load(KS + request))
        if HAS_P:
            tl.store(REPEATED_PS + row, tl.load(PS + request))


@triton.jit
def _mask(VALUES, KS, K: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    keep = tl.minimum(tl.maximum(tl.load(KS + row).to(tl.int64), 1), V)
    # Only masked positions are written; original top-k values stay bitwise.
    tl.store(VALUES + row * K + columns, -float("inf"), (columns < K) & (columns >= keep))


def prepare_speculative_logits(x, temperature, ks, ps, scaled, target, repeated_ks,
                               repeated_ps, draft_token_num):
    # Unused pointers deliberately alias an existing tensor; constexpr branches
    # eliminate every access to them before lowering.
    _prepare[(x.shape[0], triton.cdiv(x.shape[1], 2048))](
        x, temperature, ks if repeated_ks is not None else temperature,
        ps if repeated_ps is not None else temperature, scaled,
        target if target is not None else scaled,
        repeated_ks if repeated_ks is not None else temperature,
        repeated_ps if repeated_ps is not None else temperature,
        x.stride(0), x.shape[1], draft_token_num,
        repeated_ks is not None, repeated_ps is not None, target is not None,
        BLOCK=2048, num_warps=4, enable_fp_fusion=False)


def mask_topk_logits(values, ks, vocab):
    _mask[(values.shape[0], triton.cdiv(values.shape[1], 256))](
        values, ks, values.shape[1], vocab, BLOCK=256, num_warps=4)
