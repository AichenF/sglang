"""The FA backend's per-replay SWA write-location refill in one launch:
dst[i] = mapping[loc[i]] for i < n, dst[i] = 0 for n <= i < len(dst)
(replaces dst[n:].zero_() + mapping[loc] gather + dst[:n].copy_())."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_write_loc_kernel(dst_ptr, loc_ptr, map_ptr, map_len, n, total, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_dst = offs < total
    live = offs < n
    loc = tl.load(loc_ptr + offs, mask=live, other=0).to(tl.int64)
    loc = tl.where(loc < 0, loc + map_len, loc)  # torch negative-index semantics (the -1 sentinel row)
    val = tl.load(map_ptr + loc, mask=live, other=0)
    tl.store(dst_ptr + offs, tl.where(live, val, 0).to(dst_ptr.dtype.element_ty), mask=in_dst)


def fill_swa_write_loc(dst: torch.Tensor, loc: torch.Tensor, mapping: torch.Tensor) -> None:
    from sglang.kernels.ops.speculative import step_prep

    if step_prep.is_recording():  # fused replay prep (step_prep.py)
        step_prep.record("wloc", dst=dst, loc=loc, mapping=mapping)
        return
    loc = loc.contiguous()
    assert dst.dim() == 1 and loc.dim() == 1 and dst.is_contiguous()
    n = int(loc.shape[0])
    total = int(dst.shape[0])
    assert n <= total
    BLOCK = 1024
    _swa_write_loc_kernel[(triton.cdiv(total, BLOCK),)](dst, loc, mapping, int(mapping.shape[0]), n, total, BLOCK=BLOCK, num_warps=4)
