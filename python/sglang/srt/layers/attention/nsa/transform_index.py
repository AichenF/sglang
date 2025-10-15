from typing import List, Optional

import torch
import triton
import triton.language as tl

# Try to import CUDA kernel
try:
    import transform_index_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA kernel not available. Run 'python setup_cuda_kernel.py install' to compile.")


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_ref(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_ref(**kwargs)


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    max_seqlen_k: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * max_seqlen_k
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    max_seqlen_k = page_table.shape[1]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        max_seqlen_k=max_seqlen_k,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # TODO(baizhou): can be implemented with another triton kernel
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table,
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_cuda(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    version: int = 1,
) -> torch.Tensor:
    """CUDA kernel implementation"""
    assert page_size == 1
    assert CUDA_AVAILABLE, "CUDA kernel not compiled!"
    if version == 1:
        return transform_index_cuda.transform_index_decode(page_table, topk_indices)
    else:
        return transform_index_cuda.transform_index_decode_v2(page_table, topk_indices)


def transform_index_page_table_prefill_cuda(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    """CUDA kernel implementation for prefill"""
    assert page_size == 1
    assert CUDA_AVAILABLE, "CUDA kernel not compiled!"
    
    # Create req_ids tensor: maps each token to its request id
    total_tokens = topk_indices.shape[0]
    req_ids = torch.empty(total_tokens, dtype=torch.int32, device=page_table.device)
    offset = 0
    for req_id, length in enumerate(extend_lens_cpu):
        req_ids[offset:offset + length] = req_id
        offset += length
    
    return transform_index_cuda.transform_index_prefill(page_table, topk_indices, req_ids)


if __name__ == "__main__":
    import time
    
    print("=" * 80)
    print("Benchmark: transform_index_page_table_decode")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # (bs, topk, max_seqlen, description)
        (8, 2048, 3000, "Small batch (decode typical)"),
        (32, 2048, 8192, "Medium batch"),
        (64, 2048, 16384, "Large batch"),
        (128, 2048, 32768, "XL batch (prefill typical)"),
    ]
    
    for bs, topk, max_seqlen, desc in configs:
        print(f"\n{desc}: bs={bs}, topk={topk}, max_seqlen={max_seqlen}")
        print("-" * 80)
        
        page_table = torch.randint(0, max_seqlen, (bs, max_seqlen), device="cuda", dtype=torch.int32)
        topk_indices = torch.full((bs, topk), -1, device="cuda", dtype=torch.int64)
        # Fill with random valid indices (simulate real sparse attention pattern)
        valid_count = min(1600, max_seqlen)
        topk_indices[:, :valid_count] = torch.randint(0, max_seqlen, (bs, valid_count), device="cuda", dtype=torch.int64)
        
        # Warmup
        for _ in range(20):
            _ = transform_index_page_table_decode_ref(page_table, topk_indices)
            _ = transform_index_page_table_decode_fast(page_table, topk_indices)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch gather (ref)
        num_iters = 1000
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            result_ref = transform_index_page_table_decode_ref(page_table, topk_indices)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        # Benchmark Triton kernel (fast)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            result_fast = transform_index_page_table_decode_fast(page_table, topk_indices)
        torch.cuda.synchronize()
        fast_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        # Verify correctness
        assert torch.all(result_ref == result_fast), "Results mismatch!"
        
        # Calculate throughput
        data_size_mb = (page_table.numel() + topk_indices.numel() + result_ref.numel()) * 4 / (1024**2)
        ref_bandwidth = data_size_mb / (ref_time / 1000)  # MB/s
        fast_bandwidth = data_size_mb / (fast_time / 1000)  # MB/s
        
        # Print results
        print(f"  PyTorch gather (ref):  {ref_time:>8.4f} ms  ({ref_bandwidth:>8.2f} MB/s)")
        print(f"  Triton kernel (fast):  {fast_time:>8.4f} ms  ({fast_bandwidth:>8.2f} MB/s)")
        speedup = ref_time / fast_time
        if speedup > 1:
            print(f"  → Triton is {speedup:.2f}x faster")
        else:
            print(f"  → PyTorch is {1/speedup:.2f}x faster (Triton is {speedup:.2f}x)")
    
    print("\n" + "=" * 80)
    print("Correctness test with edge cases")
    print("=" * 80)
    
    # Edge case test
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda", dtype=torch.int32)
    topk_indices = torch.full((bs, topk), -1, device="cuda", dtype=torch.int64)
    topk_indices[:, :1600] = torch.arange(1600, dtype=torch.int64).unsqueeze(0).repeat(bs, 1)
    
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    fast_result = transform_index_page_table_decode_fast(page_table, topk_indices)
    
    assert torch.all(ref_result == fast_result), "Correctness test failed!"
    assert torch.all(ref_result[topk_indices < 0] == -1), "Negative index handling failed!"
    print("✓ All correctness tests passed!")
    
    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    print("Based on the benchmark results above:")
    print("- If PyTorch is faster: use transform_index_page_table_decode_ref (current default)")
    print("- If Triton is faster: switch to transform_index_page_table_decode_fast")
    print("- Consider using NSA_FUSE_TOPK=True to avoid this transformation entirely")
    print("=" * 80)
