#pragma once

#include <distributed/dflash_mega.cuh>

namespace sglang {
namespace dflash_attention_test {

using namespace dflash_mega;
constexpr int SHARED_BYTES = MISC_BYTES + GREG_ATT + KV_BYTES;

// Exercise the production gather, cluster transport, combine and KV reuse
// without GEMMs, trained weights or TP IPC. The QKV producer has already
// completed.
__global__ void __launch_bounds__(320, 1)
    kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t storage[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rank = static_cast<int>(cluster_rank());
  const int c = static_cast<int>(cluster_id_x());
  Smem s{};
  s.kvfull = reinterpret_cast<uint64_t *>(storage);
  s.kvempty = s.kvfull + 1;
  s.combbar = s.kvfull + 2;
  s.combok = s.kvfull + 3;
  s.red = reinterpret_cast<float *>(storage + 832);
  s.slot = reinterpret_cast<int *>(storage + 1088);
  s.misc = reinterpret_cast<int *>(storage + 1728);
  s.csbar = reinterpret_cast<uint64_t *>(storage + 1752);
  s.pos = reinterpret_cast<long long *>(storage + 1792);
  s.greg = storage + MISC_BYTES;
  s.kv = s.greg + GREG_ATT;
  if (tid == 0) {
    mbar_init(s.kvfull, 33);
    mbar_init(s.kvempty, 1);
    mbar_init(s.combbar, 1);
    mbar_init(s.combok, CS);
    mbar_init(s.csbar, 1);
    s.misc[1] = p.seqlens[c >> 1];
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < BLK)
    s.pos[tid] = 0;
  __syncthreads();
  cluster_sync_all();
  if (warp == 9) {
    gather(p, s, c >> 1, c, rank, lane);
  } else if (warp < 8) {
    for (int layer = 0; layer < p.nl; ++layer)
      attention(p, s, layer, c, rank, tid, 0);
  }
  __syncthreads();
  cluster_sync_all();
}

} // namespace dflash_attention_test

struct DFlashAttentionTest {
  static void run(tvm::ffi::TensorView qkv, tvm::ffi::TensorView k,
                  tvm::ffi::TensorView v, tvm::ffi::TensorView page,
                  tvm::ffi::TensorView lengths, tvm::ffi::TensorView locations,
                  tvm::ffi::TensorView norm, tvm::ffi::TensorView rope,
                  tvm::ffi::TensorView sinks, tvm::ffi::TensorView counters,
                  tvm::ffi::TensorView amax, tvm::ffi::TensorView output,
                  int64_t layers, double value_scale) {
    using namespace host;
    using namespace dflash_mega;
    RuntimeCheck(qkv.numel() == MT * QKV_N && qkv.dtype().bits == 16);
    RuntimeCheck(k.ndim() == 3 && k.size(1) == 1 && k.size(2) == HD &&
                 k.stride(0) == HD);
    RuntimeCheck(v.numel() == k.numel() && v.stride(0) == HD);
    RuntimeCheck(page.ndim() == 2 && page.size(0) == NREQ &&
                 page.stride(1) == 1);
    RuntimeCheck(lengths.numel() == NREQ && locations.numel() == MT);
    RuntimeCheck(norm.numel() == HD && rope.numel() == 64);
    RuntimeCheck(sinks.numel() == 0 || sinks.numel() == NHQ);
    RuntimeCheck(counters.numel() == C_N && amax.numel() == 4 * NL * MT);
    RuntimeCheck(output.numel() == MT * NHQ * HD && layers > 0 && layers <= NL);
    Params p{};
    for (int layer = 0; layer < layers; ++layer) {
      p.L[layer].kc = static_cast<__nv_bfloat16 *>(k.data_ptr());
      p.L[layer].vc = static_cast<__nv_bfloat16 *>(v.data_ptr());
      p.L[layer].qn = p.L[layer].kn =
          static_cast<const __nv_bfloat16 *>(norm.data_ptr());
      p.L[layer].sinks = sinks.numel() ? sinks.data_ptr() : nullptr;
    }
    p.qkv = static_cast<__nv_bfloat16 *>(qkv.data_ptr());
    p.page_table = static_cast<const int *>(page.data_ptr());
    p.seqlens = static_cast<const int *>(lengths.data_ptr());
    p.loc = locations.data_ptr();
    p.loc64 = 1;
    p.pt_stride = page.stride(0);
    p.kv_stride = HD;
    p.cos_sin = rope.data_ptr();
    p.cs_stride = 0;
    p.cs_f32 = p.sink_f32 = 1;
    p.sm_scale = 0.08838834764831844f;
    p.v_scale = static_cast<float>(value_scale);
    p.qk_eps = 0.f;
    p.nl = static_cast<int>(layers);
    p.cnt = static_cast<unsigned *>(counters.data_ptr());
    p.amax = static_cast<unsigned *>(amax.data_ptr());
    p.att = static_cast<__nv_bfloat16 *>(output.data_ptr());
    auto kernel = dflash_attention_test::kernel;
    RuntimeDeviceCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        dflash_attention_test::SHARED_BYTES));
    LaunchKernel(dim3(NATT * CS), dim3(320), qkv.device(),
                 dflash_attention_test::SHARED_BYTES)
        .enable_cluster(dim3(CS, 1, 1))(kernel, p);
  }
};

} // namespace sglang
