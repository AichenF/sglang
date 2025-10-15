#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// Decode kernel: each block handles one request, processes 2048 indices
template<int BLOCK_SIZE = 256, int VEC_SIZE = 4>
__global__ void transform_index_decode_kernel(
    const int32_t* __restrict__ page_table,      // [bs, max_seqlen_k]
    const int64_t* __restrict__ topk_indices,    // [bs, topk]
    int32_t* __restrict__ result,                // [bs, topk]
    const int32_t max_seqlen_k,
    const int32_t topk
) {
    const int req_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    
    // Pointers for this request
    const int32_t* page_row = page_table + req_id * max_seqlen_k;
    const int64_t* indices_row = topk_indices + req_id * topk;
    int32_t* result_row = result + req_id * topk;
    
    // Each thread processes multiple elements with vectorization
    const int elements_per_thread = (topk + total_threads - 1) / total_threads;
    const int start_idx = tid * elements_per_thread;
    const int end_idx = min(start_idx + elements_per_thread, topk);
    
    // Process elements in vectorized manner when possible
    #pragma unroll 4
    for (int i = start_idx; i < end_idx; i++) {
        int64_t idx = indices_row[i];
        result_row[i] = (idx >= 0 && idx < max_seqlen_k) ? page_row[idx] : -1;
    }
}

// Prefill kernel: processes ragged batch with different sequence lengths
template<int BLOCK_SIZE = 256>
__global__ void transform_index_prefill_kernel(
    const int32_t* __restrict__ page_table,      // [bs, max_seqlen_k]
    const int64_t* __restrict__ topk_indices,    // [total_tokens, topk]
    int32_t* __restrict__ result,                // [total_tokens, topk]
    const int32_t* __restrict__ cu_seqlens,      // [bs+1] cumulative sequence lengths
    const int32_t* __restrict__ req_ids,         // [total_tokens] which request each token belongs to
    const int32_t max_seqlen_k,
    const int32_t topk,
    const int32_t total_tokens
) {
    const int token_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    
    if (token_id >= total_tokens) return;
    
    // Get the request id for this token
    const int req_id = req_ids[token_id];
    
    // Pointers for this token
    const int32_t* page_row = page_table + req_id * max_seqlen_k;
    const int64_t* indices_row = topk_indices + token_id * topk;
    int32_t* result_row = result + token_id * topk;
    
    // Each thread processes multiple elements
    const int elements_per_thread = (topk + total_threads - 1) / total_threads;
    const int start_idx = tid * elements_per_thread;
    const int end_idx = min(start_idx + elements_per_thread, topk);
    
    #pragma unroll 4
    for (int i = start_idx; i < end_idx; i++) {
        int64_t idx = indices_row[i];
        result_row[i] = (idx >= 0 && idx < max_seqlen_k) ? page_row[idx] : -1;
    }
}

// Alternative decode kernel: higher parallelism, each block processes part of topk
template<int BLOCK_SIZE = 256, int CHUNK_SIZE = 256>
__global__ void transform_index_decode_kernel_v2(
    const int32_t* __restrict__ page_table,
    const int64_t* __restrict__ topk_indices,
    int32_t* __restrict__ result,
    const int32_t max_seqlen_k,
    const int32_t topk
) {
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int req_id = global_id / topk;
    const int local_idx = global_id % topk;
    
    if (local_idx >= topk) return;
    
    const int32_t* page_row = page_table + req_id * max_seqlen_k;
    int64_t idx = topk_indices[req_id * topk + local_idx];
    result[req_id * topk + local_idx] = (idx >= 0 && idx < max_seqlen_k) ? page_row[idx] : -1;
}

// Python interface
torch::Tensor transform_index_decode_cuda(
    torch::Tensor page_table,
    torch::Tensor topk_indices
) {
    const int bs = page_table.size(0);
    const int max_seqlen_k = page_table.size(1);
    const int topk = topk_indices.size(1);
    
    auto result = torch::empty_like(topk_indices, torch::TensorOptions().dtype(torch::kInt32));
    
    const int threads = 256;
    const int blocks = bs;
    
    transform_index_decode_kernel<256, 4><<<blocks, threads>>>(
        page_table.data_ptr<int32_t>(),
        topk_indices.data_ptr<int64_t>(),
        result.data_ptr<int32_t>(),
        max_seqlen_k,
        topk
    );
    
    return result;
}

torch::Tensor transform_index_decode_cuda_v2(
    torch::Tensor page_table,
    torch::Tensor topk_indices
) {
    const int bs = page_table.size(0);
    const int max_seqlen_k = page_table.size(1);
    const int topk = topk_indices.size(1);
    
    auto result = torch::empty_like(topk_indices, torch::TensorOptions().dtype(torch::kInt32));
    
    const int threads = 256;
    const int total_elements = bs * topk;
    const int blocks = (total_elements + threads - 1) / threads;
    
    transform_index_decode_kernel_v2<256, 256><<<blocks, threads>>>(
        page_table.data_ptr<int32_t>(),
        topk_indices.data_ptr<int64_t>(),
        result.data_ptr<int32_t>(),
        max_seqlen_k,
        topk
    );
    
    return result;
}

torch::Tensor transform_index_prefill_cuda(
    torch::Tensor page_table,
    torch::Tensor topk_indices,
    torch::Tensor req_ids
) {
    const int max_seqlen_k = page_table.size(1);
    const int total_tokens = topk_indices.size(0);
    const int topk = topk_indices.size(1);
    
    auto result = torch::empty_like(topk_indices, torch::TensorOptions().dtype(torch::kInt32));
    
    const int threads = 256;
    const int blocks = total_tokens;
    
    transform_index_prefill_kernel<256><<<blocks, threads>>>(
        page_table.data_ptr<int32_t>(),
        topk_indices.data_ptr<int64_t>(),
        result.data_ptr<int32_t>(),
        nullptr,  // cu_seqlens not used in this simple version
        req_ids.data_ptr<int32_t>(),
        max_seqlen_k,
        topk,
        total_tokens
    );
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transform_index_decode", &transform_index_decode_cuda, "Transform index decode (CUDA)");
    m.def("transform_index_decode_v2", &transform_index_decode_cuda_v2, "Transform index decode v2 (CUDA)");
    m.def("transform_index_prefill", &transform_index_prefill_cuda, "Transform index prefill (CUDA)");
}

