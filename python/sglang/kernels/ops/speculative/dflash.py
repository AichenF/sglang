import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _dflash_accept_bonus_publish_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    future_idx_ptr,
    future_seq_lens_ptr,
    future_tokens_ptr,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    """goal11-glue: _dflash_accept_bonus_contig_kernel (contiguous rows) + the overlap scheduler's FutureMap relay in
    the same launch: new_seq_lens_buf[future_idx[row]] = new_seq_len (FutureMap.publish) and
    output_tokens_buf[future_idx[row]] = bonus (FutureMap.stash's bonus_tokens write)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)
    candidate_row_ptr = candidates_ptr + row * block_size
    target_row_ptr = target_top1_ptr + row * block_size
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)
    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)
    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row) + commit_len
    tl.store(accept_lens_out_ptr + row, accept_len)
    tl.store(commit_lens_out_ptr + row, commit_len)
    tl.store(bonus_ids_out_ptr + row, bonus_id)
    tl.store(new_seq_lens_out_ptr + row, new_seq_len)
    fidx = tl.load(future_idx_ptr + row).to(tl.int64)
    tl.store(future_seq_lens_ptr + fidx, new_seq_len.to(tl.int64))
    tl.store(future_tokens_ptr + fidx, bonus_id.to(tl.int64))
    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(out_tokens_ptr + row * block_size + cols, out_val, mask=row_mask)


def compute_dflash_accept_bonus_publish(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    future_idx: torch.Tensor,
    future_seq_lens: torch.Tensor,
    future_tokens: torch.Tensor,
) -> bool:
    """Returns False (nothing launched) when the layouts do not match the fused kernel's assumptions."""
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return False
    for t in (candidates, target_top1, out_tokens_out):
        if not _is_row_major_contiguous_2d(t) or t.shape != (batch_size, block_size):
            return False
    for t in (accept_lens_out, commit_lens_out, bonus_ids_out, new_seq_lens_out, prefix_lens, future_idx):
        if t.ndim != 1 or not t.is_contiguous() or t.shape[0] != batch_size:
            return False
    if future_seq_lens.dtype != torch.int64 or future_tokens.dtype != torch.int64:
        return False
    block = triton.next_power_of_2(block_size)
    _dflash_accept_bonus_publish_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        future_idx,
        future_seq_lens,
        future_tokens,
        block_size,
        BLOCK_SIZE=block,
        num_warps=_pick_num_warps(block),
    )
    return True


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return
    from sglang.kernels.ops.speculative import step_prep

    if step_prep.is_recording():  # goal11-post: deferred into the draft runner's fused replay prep
        step_prep.record(
            "prepare", bonus_tokens=bonus_tokens, prefix_lens=prefix_lens, req_pool_indices=req_pool_indices,
            req_to_token=req_to_token, block_ids_out=block_ids_out, positions_out=positions_out,
            cache_loc_out=cache_loc_out, mask_token_id=mask_token_id,
        )
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
    tokens_out=None,
    q_out=None,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = (
        tokens_out
        if tokens_out is not None
        else torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    )
    q_rows = (
        q_out
        if q_out is not None
        else torch.empty(
            (batch, slots, top_k), dtype=torch.float32, device=scores.device
        )
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _pack_candidates_kernel(vals_ptr, ids_ptr, out_ptr, id_offset, K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, K)
    vals = tl.load(vals_ptr + row * K + cols).to(tl.float32)
    ids = tl.load(ids_ptr + row * K + cols).to(tl.int64) + id_offset
    tl.store(out_ptr + row * 2 * K + cols, vals)
    tl.store(out_ptr + row * 2 * K + K + cols, ids.to(tl.float32))


@functools.lru_cache(maxsize=None)
def _selector_topk_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "selector_topk",
        cuda_files=["speculative/selector_topk.cuh"],
        cuda_wrappers=[("topk_pack", "SelectorTopK::topk_pack")],
    )


def selector_topk_pack(
    logits: torch.Tensor,
    k: int,
    n: int,
    id_offset: int,
    rows: int,
    group: int,
    gstride: int = 0,
    goff: int = 0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Local top-k of shard-logits rows + pack_candidates in one kernel -> [rows, 2k] fp32 [values | ids + id_offset].

    Output row r reads logits row (r // group) * gstride + goff + r % group (all block rows of the head output, the
    anchor row skipped), its first n columns. Same result as pack_candidates(*flashinfer.top_k(logits_rows[:, :n], k,
    sorted=True, deterministic=True, tie_break=TopKTieBreak.SMALL), id_offset): value descending, ties by smallest index."""
    if out is None:
        out = torch.empty((rows, 2 * k), dtype=torch.float32, device=logits.device)
    if rows:
        _selector_topk_module().topk_pack(logits, out, int(n), int(k), int(group), int(gstride), int(goff), int(id_offset))
    return out


def pack_candidates(vals: torch.Tensor, ids: torch.Tensor, id_offset: int) -> torch.Tensor:
    """[N, K] values and shard ids -> [N, 2K] fp32 rows [values | ids + id_offset]."""
    rows, k = vals.shape
    out = torch.empty((rows, 2 * k), dtype=torch.float32, device=vals.device)
    if rows:
        _pack_candidates_kernel[(rows,)](
            vals.contiguous(), ids.contiguous(), out, int(id_offset), K=k, num_warps=1
        )
    return out


@triton.jit
def _unpack_candidate_ids_kernel(
    gathered_ptr, sel_ptr, out_ptr, RANKS: tl.constexpr, K: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, K)
    sel = tl.load(sel_ptr + row * K + cols)
    src = row * RANKS * 2 * K + (sel // K) * 2 * K + K + sel % K
    tl.store(out_ptr + row * K + cols, tl.load(gathered_ptr + src).to(tl.int64))


def unpack_candidate_ids(
    gathered: torch.Tensor, sel: torch.Tensor, out: torch.Tensor = None
) -> torch.Tensor:
    """gathered [N, ranks, 2, K] rows [values | ids], sel [N, K] into the ranks * K values -> ids [N, K] int64."""
    rows, ranks, _, k = gathered.shape
    if out is None:
        out = torch.empty((rows, k), dtype=torch.int64, device=gathered.device)
    if rows:
        _unpack_candidate_ids_kernel[(rows,)](
            gathered, sel.contiguous(), out, RANKS=ranks, K=k, num_warps=1
        )
    return out


@triton.jit
def _bitonic_step(k, i, v, slot, AXIS: tl.constexpr, SIZE: tl.constexpr, LOG_S: tl.constexpr):
    # compare-exchange with slot ^ stride, the pair lies along AXIS
    pk = k ^ tl.xor_sum(k, AXIS, keep_dims=True)
    pi = i ^ tl.xor_sum(i, AXIS, keep_dims=True)
    pv = v ^ tl.xor_sum(v, AXIS, keep_dims=True)
    low = (slot & (1 << (LOG_S - AXIS))) == 0
    ka = tl.where(low, k, pk).to(tl.float32, bitcast=True)
    kb = tl.where(low, pk, k).to(tl.float32, bitcast=True)
    va = tl.where(low, v, pv) != 0
    vb = tl.where(low, pv, v) != 0
    swap = ((((ka != ka) & (kb == kb)) | (ka > kb)) & va) | (~vb)
    if SIZE < (1 << LOG_S):
        up = (slot & SIZE) != 0
    else:
        up = slot < 0
    x = swap == up
    return tl.where(x, pk, k), tl.where(x, pi, i), tl.where(x, pv, v)


@triton.jit
def _gathered_topk(g_ptr, rows, R: tl.constexpr, K: tl.constexpr, LOG_S: tl.constexpr):
    # torch.topk of two gathered rows with its tie order: values above the K-th, then
    # values equal to it, each in index order, then torch's 2**LOG_S-slot bitonic sort
    N: tl.constexpr = R * K
    S: tl.constexpr = 1 << LOG_S
    offs = tl.arange(0, N)[None, :]
    base = rows[:, None] * N * 2
    bits = tl.load(g_ptr + base + (offs // K) * 2 * K + offs % K).to(tl.uint32, bitcast=True)
    vals = bits.to(tl.float32, bitcast=True)
    flip = tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
    key = tl.where(vals != vals, 0xFFFFFFFF, bits ^ flip).to(tl.uint32)
    kth = tl.max(tl.where(offs == K - 1, tl.sort(key, dim=1, descending=True), 0), axis=1)
    above = key > kth[:, None]
    tie = key == kth[:, None]
    pos = tl.where(
        above,
        tl.cumsum(above.to(tl.int32), axis=1) - 1,
        tl.sum(above.to(tl.int32), axis=1)[:, None] + tl.cumsum(tie.to(tl.int32), axis=1) - 1,
    )
    pos = tl.where(above | tie, pos, K)
    at = pos[:, None, :] == tl.arange(0, K)[None, :, None]
    k = tl.sum(tl.where(at, bits.to(tl.int32, bitcast=True)[:, None, :], 0), axis=2)
    i = tl.sum(tl.where(at, offs[:, None, :], 0), axis=2)
    # slots K.. are empty (key 0, invalid)
    shape: tl.constexpr = [2] + [2] * LOG_S
    zero = tl.zeros((2, S - K), tl.int32)
    k = tl.reshape(tl.join(k, zero).permute(0, 2, 1), shape)
    i = tl.reshape(tl.join(i, zero).permute(0, 2, 1), shape)
    v = tl.reshape(tl.join(zero + 1, zero).permute(0, 2, 1), shape)
    slot = tl.reshape(tl.arange(0, 2 * S) % S, shape)
    for stage in tl.static_range(LOG_S):
        for sub in tl.static_range(stage + 1):
            k, i, v = _bitonic_step(k, i, v, slot, LOG_S - stage + sub, 2 << stage, LOG_S)
    first = tl.arange(0, S // K)[None, :, None] == 0
    k = tl.sum(tl.where(first, tl.reshape(k, [2, S // K, K]), 0), axis=1)
    i = tl.sum(tl.where(first, tl.reshape(i, [2, S // K, K]), 0), axis=1)
    ids = tl.load(g_ptr + rows[:, None] * N * 2 + (i // K) * 2 * K + K + i % K).to(tl.int64)
    return k, ids


@triton.jit
def _selector_lattice_kernel(
    g_ptr,
    anchor_ptr,
    anchor_stride,
    hidden_ptr,
    pred_ptr,
    succ_ptr,
    ids_ptr,
    scores_ptr,
    SLOTS: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    LOG_S: tl.constexpr,
    RANK: tl.constexpr,
    HROWS: tl.constexpr = 0,
    HOFF: tl.constexpr = 0,
):
    row = tl.program_id(0)
    cols = tl.arange(0, K)
    # this slot's candidates and the previous slot's (the anchor for slot 0)
    two = tl.arange(0, 2)
    k, ids = _gathered_topk(g_ptr, tl.maximum(row - two, 0), R, K, LOG_S)
    own = two[:, None] == 0
    vals = tl.sum(tl.where(own, k, 0), axis=0).to(tl.float32, bitcast=True)
    pred = tl.sum(tl.where(own, 0, ids), axis=0)
    ids = tl.sum(tl.where(own, ids, 0), axis=0)
    tl.store(ids_ptr + row * K + cols, ids)
    pred = tl.where(row % SLOTS == 0, tl.load(anchor_ptr + (row // SLOTS) * anchor_stride), pred)
    r = tl.arange(0, RANK)
    if HROWS == 0:
        hrow = row
    else:
        hrow = (row // SLOTS) * HROWS + HOFF + row % SLOTS  # projection of every block row: skip the anchor rows
    h = tl.load(hidden_ptr + hrow * RANK + r).to(tl.float32)
    a = tl.load(pred_ptr + pred[:, None] * RANK + r[None, :]).to(tl.float32)
    b = tl.load(succ_ptr + ids[:, None] * RANK + r[None, :])
    dots = tl.dot((a * h[None, :]).to(tl.bfloat16), tl.trans(b))
    scores = vals[None, :] + dots.to(tl.bfloat16).to(tl.float32)
    tl.store(scores_ptr + (row * K + cols[:, None]) * K + cols[None, :], scores)


def selector_lattice_triton(
    gathered, anchor_ids, hidden, predecessor_table, successor_table, ids_out, hidden_row_off=0
):
    """torch.topk of the gathered [bs * slots, ranks, 2, K] rows + build_lattice -> ids_out, scores [bs, slots, K, K].

    hidden: the selector projection [bs, slots, r], or [bs, slots + hidden_row_off, r] of every block row with the
    first hidden_row_off rows of each request skipped (ids_out [bs, slots, K] gives the slots)."""
    rows, ranks, _, k = gathered.shape
    bs, hrows, rank = hidden.shape
    slots = int(ids_out.shape[-2]) if hidden_row_off else hrows
    assert hrows == slots + hidden_row_off and hidden.is_contiguous()
    scores = torch.empty((bs, slots, k, k), dtype=torch.float32, device=hidden.device)
    _selector_lattice_kernel[(rows,)](
        gathered,
        anchor_ids,
        anchor_ids.stride(0),
        hidden,
        predecessor_table,
        successor_table,
        ids_out,
        scores,
        SLOTS=slots,
        R=ranks,
        K=k,
        LOG_S=5,
        RANK=rank,
        HROWS=hrows if hidden_row_off else 0,
        HOFF=hidden_row_off,
        num_warps=2,
    )
    return scores
