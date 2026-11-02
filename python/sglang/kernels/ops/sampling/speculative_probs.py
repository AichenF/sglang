"""Partial fusion of DFlash target-probability preparation.

Selection keeps the original torch.topk. On the exact validated runtime,
explicit small K50/128 can fuse mask/softmax/native top-p/scatter; other inputs
keep the original partial path. Temperature division stays before selection.
This helper never samples and never reads or modifies proposal probabilities.
"""
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def try_fused_speculative_verify_target_probs(
    *, next_token_logits: torch.Tensor, sampling_info,
    bs: int, draft_token_num: int, max_top_k: Optional[int],
    uniform_top_k_value: Optional[int], use_sparse_topk: bool,
    top_k_renorm: Callable, top_p_renorm: Callable,
) -> Optional[torch.Tensor]:
    """Return None outside the audited FP32 CUDA/block8/customer-vocab domain.

No device-to-host read is added. The max_top_k=None sparse case retains the
existing max().item() synchronization and, like the original, is not capturable.
Graph callers must supply the existing CPU-computed max_top_k metadata.
"""
    x = next_token_logits
    if (not isinstance(x, torch.Tensor) or x.device.type != "cuda"
        or torch.version.cuda is None or x.dtype != torch.float32
        or x.requires_grad
        or x.ndim != 2 or draft_token_num != 8 or not 1 <= bs <= 4
        or x.shape != (bs * 8, 152576) or x.stride(1) != 1
        or x.stride(0) < x.shape[1]):
        return None
    need_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    temperatures = getattr(sampling_info, "temperatures", None)
    top_ks = getattr(sampling_info, "top_ks", None)
    top_ps = getattr(sampling_info, "top_ps", None)

    def matches(tensor, shape, dtypes):
        return (isinstance(tensor, torch.Tensor) and tensor.shape == shape
                and tensor.dtype in dtypes and tensor.device == x.device
                and tensor.is_contiguous())

    if not matches(temperatures, (bs, 1), (torch.float32,)):
        return None
    if temperatures.requires_grad or (need_p and getattr(top_ps, "requires_grad", False)):
        return None
    if need_k and not matches(top_ks, (bs,), (torch.int32, torch.int64)):
        return None
    if need_p and not matches(top_ps, (bs,), (torch.float32,)):
        return None

    rows, vocab = x.shape
    sparse = False
    k = 0
    if use_sparse_topk and need_k:
        if max_top_k is None:
            # Repetition does not change the max. Match the original int64 cast
            # and clamp BEFORE taking it, including zero/out-of-range entries.
            k = int(top_ks.to(torch.int64).clamp(min=1, max=vocab).max().item())
        else:
            k = max(1, min(int(max_top_k), vocab))
        sparse = k < vocab

    # Lazy import allows unsupported/CPU inputs to use the unchanged fallback
    # without requiring Triton or importing CUDA implementation code.
    from sglang.kernels.ops.sampling.speculative_probs_triton import (
        mask_topk_logits, prepare_speculative_logits,
    )

    scaled = torch.empty((rows, vocab), device=x.device, dtype=x.dtype)
    target = torch.empty_like(scaled) if sparse else None
    repeated_ks = torch.empty((rows,), device=x.device, dtype=top_ks.dtype) if need_k else None
    repeated_ps = torch.empty((rows,), device=x.device, dtype=top_ps.dtype) if need_p else None
    prepare_speculative_logits(x, temperatures, top_ks, top_ps, scaled, target,
                               repeated_ks, repeated_ps, draft_token_num)
    if sparse:
        values, indices = torch.topk(scaled, k=k, dim=-1)
        if type(max_top_k) is int and max_top_k in (50, 128):
            from sglang.kernels.ops.sampling import speculative_probs_finish_cuda

            if speculative_probs_finish_cuda.can_use_finish(x.device.index):
                speculative_probs_finish_cuda.finish(
                    values, indices, repeated_ks, repeated_ps, target,
                    use_p=need_p,
                    skip_mask=uniform_top_k_value is not None and int(uniform_top_k_value) == k,
                    prezeroed=True,
                )
                return target.view(bs, draft_token_num, vocab).contiguous()
        if uniform_top_k_value is None or int(uniform_top_k_value) != k:
            mask_topk_logits(values, repeated_ks, vocab)
        probabilities = F.softmax(values, dim=-1)
        if need_p:
            probabilities = top_p_renorm(probabilities, repeated_ps)
        target.scatter_(1, indices, probabilities)
    else:
        target = F.softmax(scaled, dim=-1)
        if need_k:
            target = top_k_renorm(target, repeated_ks)
        if need_p:
            target = top_p_renorm(target, repeated_ps)
    return target.view(bs, draft_token_num, vocab).contiguous()
