// SiLU-and-mul + per-token-group-128 FP8 quant in ONE kernel, for the dense-MLP decode path
// (MiMo layer 0: gate_up [M, 2I] bf16 -> down_proj input).
//
// Bit-exact with the unfused pair it replaces when built with the same flags as the JIT activation kernel
// (--use_fast_math on sm90, see get_activation_cuda_cflags):
//   act_and_mul_kernel<bf16, kSiLU>:  a = bf16(x / (1 + expf(-x)) * up)        (fp32 math, one bf16 rounding)
//   per_token_group_quant_fp8 (column-major, TMA-aligned scales, as the DeepGEMM linear path calls it):
//     amax = max(max |a| over the 128-group, 1e-10); qs = 448 * rcp.approx.ftz(amax);
//     q = e4m3 satfinite(fmin(qs * a, 448)); scale = amax * (1 / 448)
// Output: q [>= M, I] e4m3 (row pitch I), scale_t [I / 128, ld >= M] fp32 (scale_t.t() is the [M, I/128]
// column-major view DeepGEMM takes).
//
// One CTA per token, I / 8 threads, 8 consecutive columns per thread (one 16-byte load of gate and of up),
// 16 threads per quant group.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace sglang {
namespace silu_mul_gq {

constexpr float FP8_MAX = 448.f;

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

template <int I, bool kUsePDL>
__global__ void __launch_bounds__(I / 8) silu_mul_group_quant_kernel(
    const __nv_bfloat16* __restrict__ in,
    uint8_t* __restrict__ out_q,
    float* __restrict__ out_scale,
    int64_t in_ld,
    int64_t sx_ld) {
  static_assert(I % 128 == 0 && I / 8 <= 1024);
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int e = tid * 8;
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const __nv_bfloat16* row = in + static_cast<int64_t>(t) * in_ld;
  const uint4 gv = *reinterpret_cast<const uint4*>(row + e);
  const uint4 uv = *reinterpret_cast<const uint4*>(row + I + e);
  const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
  const uint32_t uw[4] = {uv.x, uv.y, uv.z, uv.w};
  float a[8];
  float gmx = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float g0 = bf_lo(gw[j]), g1 = bf_hi(gw[j]);
    const float u0 = bf_lo(uw[j]), u1 = bf_hi(uw[j]);
    // the exact expression of apply_activation_f32<kSiLU> (compiled with the same fast-math flags)
    a[2 * j] = bf_round(g0 / (1.0f + expf(-g0)) * u0);
    a[2 * j + 1] = bf_round(g1 / (1.0f + expf(-g1)) * u1);
    gmx = fmaxf(gmx, fmaxf(fabsf(a[2 * j]), fabsf(a[2 * j + 1])));
  }
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) gmx = fmaxf(gmx, __shfl_xor_sync(0xffffffffu, gmx, o));  // 16 lanes = 128 cols
  gmx = fmaxf(gmx, 1e-10f);
  const float qs = mul_ftz(rcp_approx_ftz(gmx), FP8_MAX);
  uint32_t q[2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
    q[i] = static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, a[4 * i]), FP8_MAX), fminf(mul_ftz(qs, a[4 * i + 1]), FP8_MAX))) |
           (static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, a[4 * i + 2]), FP8_MAX),
                                              fminf(mul_ftz(qs, a[4 * i + 3]), FP8_MAX)))
            << 16);
  *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(t) * I + e) = make_uint2(q[0], q[1]);
  if ((tid & 15) == 0) out_scale[static_cast<int64_t>(e >> 7) * sx_ld + t] = mul_ftz(gmx, __uint_as_float(0x3b124925u));
}

}  // namespace silu_mul_gq

template <bool kUsePDL>
struct SiluMulGroupQuant {
  // in [M, 2I] bf16 (gate | up, row stride free), out_q [>= M, I] e4m3 (uint8 view, row pitch I),
  // out_scale_t [I / 128, ld >= M] fp32 (contiguous rows).
  static void run(tvm::ffi::TensorView in, tvm::ffi::TensorView out_q, tvm::ffi::TensorView out_scale_t) {
    using namespace host;
    using namespace silu_mul_gq;
    const int M = static_cast<int>(in.size(0));
    const int I = static_cast<int>(in.size(1) / 2);
    RuntimeCheck(M >= 1 && in.stride(1) == 1 && in.size(1) == 2 * I);
    RuntimeCheck(out_q.size(0) >= M && out_q.size(1) == I && (out_q.size(0) == 1 || out_q.stride(0) == I));
    RuntimeCheck(out_scale_t.ndim() == 2 && out_scale_t.size(0) == I / 128 && out_scale_t.stride(1) == 1 &&
                 out_scale_t.stride(0) >= M);
    const int64_t in_ld = in.stride(0);
    const int64_t sx_ld = out_scale_t.stride(0);
    auto* src = static_cast<const __nv_bfloat16*>(in.data_ptr());
    auto* q = static_cast<uint8_t*>(out_q.data_ptr());
    auto* s = static_cast<float*>(out_scale_t.data_ptr());
    if (I == 2048) {
      LaunchKernel(dim3(M), dim3(2048 / 8), in.device())
          .enable_pdl(kUsePDL)(silu_mul_group_quant_kernel<2048, kUsePDL>, src, q, s, in_ld, sx_ld);
    } else if (I == 1024) {
      LaunchKernel(dim3(M), dim3(1024 / 8), in.device())
          .enable_pdl(kUsePDL)(silu_mul_group_quant_kernel<1024, kUsePDL>, src, q, s, in_ld, sx_ld);
    } else {
      RuntimeCheck(false, "silu_mul_group_quant: intermediate size must be 1024 or 2048");
    }
  }
};

}  // namespace sglang
