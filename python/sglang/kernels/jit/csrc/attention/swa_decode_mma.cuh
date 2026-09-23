// Sliding-window verify-decode attention for sm_90: 1 KV head, asymmetric
// 192/128 head dims, page_size 1, per-head attention sink.
//
// The shape is bound by memory latency, not bandwidth or compute. Three choices
// follow from that.
//   * mma.m16n8k16 with explicit ldmatrix. M=16 is exactly W*HP, and N=8 pads a
//     135-key window to 144 rather than to a power of two. Do not switch back to
//     nvcuda::wmma: on a pointer from `extern __shared__` it lowers to generic
//     loads instead of ldmatrix, which was several times slower.
//   * Warp w owns key tiles {w, w+NW, ...} and gathers exactly those rows, so the
//     QK phase needs no barrier and K is committed one tile at a time; the mma on
//     one tile runs while the next is still in flight.
//   * V is a separate cp.async group waited on only after the softmax, so its
//     latency lands behind QK and the softmax rather than in front of them.
//
// Softmax is float32 and the sink joins the denominator only. A row with no
// visible key (a zero-length padded graph slot) yields an exact 0, not NaN.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <mutex>


namespace sglang {
namespace swa_mma_detail {


constexpr int DQK = 192;
constexpr int DV = 128;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ unsigned sm_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Four 8x8 b16 tiles; lanes 0-7 address tile 0, 8-15 tile 1, and so on.
__device__ __forceinline__ void ldm4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm_addr(p)));
}
__device__ __forceinline__ void ldm4t(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm_addr(p)));
}
__device__ __forceinline__ void mma16816(float* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo));
  return u;
}

// W  : query rows (verify tokens) per request
// HP : query heads per program; W*HP must be <= 16 (one m16 tile)
// NW : warps per CTA
template <int W, int HP, int WINDOW, int NW>
__global__ __launch_bounds__(NW * 32) void swa_mma_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const int* __restrict__ PT,
    const int* __restrict__ SEQLENS, const float* __restrict__ SINKS,
    __nv_bfloat16* __restrict__ OUT, int q_stride_m, int q_stride_h, int pt_stride,
    int hq, float sm_scale, int has_sink) {
  constexpr int NKEY = WINDOW - 1 + W;           // keys any row of this request can see
  constexpr int NPAD = (NKEY + 15) / 16 * 16;    // 135 -> 144, the PV k-dim must be x16
  constexpr int M = W * HP;
  static_assert(M <= 16, "one m16 tile only");
  constexpr int MPAD = 16;
  // +8 bf16 padding makes every 8-row ldmatrix phase hit 32 distinct banks.
  constexpr int KP = DQK + 8;    // 200 elem = 400 B; (400/4) % 32 == 4  -> conflict free
  constexpr int VP = DV + 8;     // 136 elem = 272 B; (272/4) % 32 == 4
  constexpr int PP = NPAD + 8;   // 152 elem = 304 B; (304/4) % 32 == 12
  constexpr int QKT = DQK / 16;  // 12 k-steps for QK
  constexpr int NT8 = NPAD / 8;  // 18 key tiles (n=8) for QK
  constexpr int NKT = NPAD / 16; //  9 k-steps for PV
  constexpr int DT8 = DV / 8;    // 16 value-dim tiles (n=8) for PV
  constexpr int NTW = (NT8 + NW - 1) / NW;  // QK n-tiles owned by one warp
  constexpr int DTW = (DT8 + NW - 1) / NW;  // PV n-tiles owned by one warp
  constexpr int KCH = DQK / 8;   // 24 x 16 B chunks per K row
  constexpr int VCH = DV / 8;    // 16 x 16 B chunks per V row

  extern __shared__ char smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + MPAD * KP;
  __nv_bfloat16* Vs = Ks + NPAD * KP;
  __nv_bfloat16* Ps = Vs + NPAD * VP;
  float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
  float* rsum = rmax + NW * MPAD;

  const int b = blockIdx.x, hg = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int seqlen = SEQLENS[b];
  const int base_pos = seqlen - W - (WINDOW - 1);

  // Warp w owns key tiles {w, w+NW, ...}; it gathers exactly those rows itself so
  // no warp reads a K row another warp fetched.
  constexpr int RPT = 8;              // key rows per n=8 tile
  constexpr int MYR = NTW * RPT;      // key rows owned by this warp
  int myslot[MYR];
#pragma unroll
  for (int i = 0; i < MYR; ++i) {
    const int ti = i / RPT, j = i % RPT;
    const int r = (warp + ti * NW) * RPT + j;
    const int p = base_pos + r;
    // Out-of-sequence rows still get gathered, branch-free, but from distinct
    // pages rather than all from page 0: their logits are forced to -inf (K) and
    // their probabilities to exact 0 (V), so any finite bytes will do, and a
    // padded graph slot must not point every CTA at one cache line at once.
    myslot[i] = (r < NKEY && p >= 0) ? PT[b * pt_stride + p] : r;
  }
  // Rows M..15 are never stored out; zero them so P stays finite.
  for (int i = tid; i < (MPAD - M) * KP; i += NW * 32) Qs[M * KP + i] = __float2bfloat16(0.f);
  for (int r = warp; r < M; r += NW) {
    const int t = r / HP, h = hg * HP + (r % HP);
    const __nv_bfloat16* src = &Q[(size_t)(b * W + t) * q_stride_m + (size_t)h * q_stride_h];
    if (lane < KCH) __pipeline_memcpy_async(&Qs[r * KP + lane * 8], src + lane * 8, 16);
  }
  // One 16 B cp.async per lane, lane index == chunk index: no division, and each
  // instruction is a fully coalesced 384 B (K) / 256 B (V) burst.
#pragma unroll
  for (int ti = 0; ti < NTW; ++ti) {
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
      const int r = (warp + ti * NW) * RPT + j;
      if (r < NPAD && lane < KCH)
        __pipeline_memcpy_async(&Ks[r * KP + lane * 8],
                                &K[(size_t)myslot[ti * RPT + j] * DQK] + lane * 8, 16);
    }
    __pipeline_commit();       // groups NTW-1 .. 0 for K (group 0 also carries Q)
  }
#pragma unroll
  for (int ti = 0; ti < NTW; ++ti) {
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
      const int r = (warp + ti * NW) * RPT + j;
      if (r < NPAD && lane < VCH)
        __pipeline_memcpy_async(&Vs[r * VP + lane * 8],
                                &V[(size_t)myslot[ti * RPT + j] * DV] + lane * 8, 16);
    }
  }
  __pipeline_commit();         // last group: V, not needed until after the softmax
  // Wait for Q and the first key tile only. The barrier is for Q, which every
  // warp reads; K needs none.
  __pipeline_wait_prior(NTW);
  __syncthreads();

  // ---- S = Q K^T ------------------------------------------------------------
  // Q fragments load once and are reused across every key tile this warp owns.
  const int rA = lane >> 2, rB = rA + 8, cA = (lane & 3) * 2;
  unsigned qa[QKT][4];
#pragma unroll
  for (int kk = 0; kk < QKT; ++kk)
    ldm4(qa[kk], &Qs[(lane & 15) * KP + kk * 16 + (lane >> 4) * 8]);

  float acc[NTW][4];
#pragma unroll
  for (int ti = 0; ti < NTW; ++ti) {
    const int nt = warp + ti * NW;
    if (ti > 0) __pipeline_wait_prior(NTW - ti);   // this tile's K has landed
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[ti][i] = 0.f;
    if (nt >= NT8) continue;
    const int n0 = nt * 8;
    unsigned kb[4];
#pragma unroll
    for (int kk = 0; kk < QKT; kk += 2) {
      // K is [key][dim] = [n][k], the non-transposed ldmatrix layout; one x4
      // feeds two k-steps.
      ldm4(kb, &Ks[(n0 + (lane & 7)) * KP + kk * 16 + (lane >> 3) * 8]);
      mma16816(acc[ti], qa[kk], kb);
      mma16816(acc[ti], qa[kk + 1], kb + 2);
    }
  }

  // ---- softmax --------------------------------------------------------------
  const float scale2 = sm_scale * LOG2E;
  float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
  for (int ti = 0; ti < NTW; ++ti) {
    const int nt = warp + ti * NW;
    if (nt >= NT8) continue;
    const int n0 = nt * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = (i < 2) ? rA : rB;
      const int j = n0 + cA + (i & 1);
      const int t = r / HP;
      const bool ok = (r < M) && (j >= t) && (j <= t + (WINDOW - 1)) && (j < NKEY) &&
                      (base_pos + j >= 0);
      acc[ti][i] = ok ? acc[ti][i] * scale2 : -INFINITY;
    }
    mx0 = fmaxf(mx0, fmaxf(acc[ti][0], acc[ti][1]));
    mx1 = fmaxf(mx1, fmaxf(acc[ti][2], acc[ti][3]));
  }
  // rows rA/rB are shared by the 4 lanes of a quad
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
  if ((lane & 3) == 0) { rmax[warp * MPAD + rA] = mx0; rmax[warp * MPAD + rB] = mx1; }
  __syncthreads();

  float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
  for (int w = 0; w < NW; ++w) {
    m0 = fmaxf(m0, rmax[w * MPAD + rA]);
    m1 = fmaxf(m1, rmax[w * MPAD + rB]);
  }
  float sk0 = 0.f, sk1 = 0.f;
  if (has_sink) {
    sk0 = SINKS[hg * HP + (rA % HP)] * LOG2E;
    sk1 = SINKS[hg * HP + (rB % HP)] * LOG2E;
    m0 = fmaxf(m0, sk0);
    m1 = fmaxf(m1, sk1);
  }
  // No visible key and no sink: keep exp2 finite so the row is an exact 0.
  if (!isfinite(m0)) m0 = 0.f;
  if (!isfinite(m1)) m1 = 0.f;

  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int ti = 0; ti < NTW; ++ti) {
    const int nt = warp + ti * NW;
    if (nt >= NT8) continue;
    const int n0 = nt * 8;
    const float p0 = exp2f(acc[ti][0] - m0), p1 = exp2f(acc[ti][1] - m0);
    const float p2 = exp2f(acc[ti][2] - m1), p3 = exp2f(acc[ti][3] - m1);
    s0 += p0 + p1;
    s1 += p2 + p3;
    *reinterpret_cast<unsigned*>(&Ps[rA * PP + n0 + cA]) = pack_bf16(p0, p1);
    *reinterpret_cast<unsigned*>(&Ps[rB * PP + n0 + cA]) = pack_bf16(p2, p3);
  }
  s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
  s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
  if ((lane & 3) == 0) { rsum[warp * MPAD + rA] = s0; rsum[warp * MPAD + rB] = s1; }
  __pipeline_wait_prior(0);   // V has had the whole QK + softmax to arrive
  __syncthreads();            // one barrier publishes Ps, rsum and every warp's V

  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int w = 0; w < NW; ++w) { l0 += rsum[w * MPAD + rA]; l1 += rsum[w * MPAD + rB]; }
  if (has_sink) { l0 += exp2f(sk0 - m0); l1 += exp2f(sk1 - m1); }

  // ---- O = P V --------------------------------------------------------------
  unsigned pa[NKT][4];
#pragma unroll
  for (int kk = 0; kk < NKT; ++kk)
    ldm4(pa[kk], &Ps[(lane & 15) * PP + kk * 16 + (lane >> 4) * 8]);

#pragma unroll
  for (int di = 0; di < DTW; ++di) {
    const int dt = warp + di * NW;
    if (dt >= DT8) continue;
    const int n0 = dt * 8;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned vb[4];
#pragma unroll
    for (int kk = 0; kk < NKT; kk += 2) {
      // V is [key][dim] = [k][n], so this one transposes; one x4 feeds two k-steps.
      ldm4t(vb, &Vs[(kk * 16 + lane) * VP + n0]);
      mma16816(o, pa[kk], vb);
      if (kk + 1 < NKT) mma16816(o, pa[kk + 1], vb + 2);
    }
    const float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
    const float inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
    if (rA < M) {
      const int t = rA / HP, h = hg * HP + (rA % HP);
      __nv_bfloat16* dst = &OUT[(size_t)(b * W + t) * hq * DV + (size_t)h * DV + n0 + cA];
      *reinterpret_cast<unsigned*>(dst) = pack_bf16(o[0] * inv0, o[1] * inv0);
    }
    if (rB < M) {
      const int t = rB / HP, h = hg * HP + (rB % HP);
      __nv_bfloat16* dst = &OUT[(size_t)(b * W + t) * hq * DV + (size_t)h * DV + n0 + cA];
      *reinterpret_cast<unsigned*>(dst) = pack_bf16(o[2] * inv1, o[3] * inv1);
    }
  }
}


template <int W, int HP, int WINDOW, int NW>
void launch_swa_mma(const void* q, const void* k, const void* v, const int* pt, const int* sl,
                    const float* sinks, void* out, int bs, int hq, int q_stride_m, int q_stride_h,
                    int pt_stride, float sm_scale, int has_sink, DLDevice device) {
  constexpr int NKEY = WINDOW - 1 + W;
  constexpr int NPAD = (NKEY + 15) / 16 * 16;
  constexpr int MPAD = 16;
  constexpr size_t kSmem =
      (size_t)(MPAD * (DQK + 8) + NPAD * (DQK + 8) + NPAD * (DV + 8) + MPAD * (NPAD + 8)) *
          sizeof(__nv_bfloat16) +
      (size_t)(2 * NW * MPAD) * sizeof(float);
  auto kern = swa_mma_kernel<W, HP, WINDOW, NW>;
  static std::once_flag once;
  std::call_once(once, [&] {
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem));
  });
  host::LaunchKernel(dim3(bs, hq / HP), dim3(NW * 32), device, kSmem)(
      kern, (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, pt, sl,
      sinks, (__nv_bfloat16*)out, q_stride_m, q_stride_h, pt_stride, hq, sm_scale, has_sink);
}

}  // namespace swa_mma_detail

// Sliding-window verify-decode attention. Shapes are validated by the Python gate;
// unsupported (window_tokens, heads_per_program) combinations raise so the caller
// can fall back.
inline void swa_decode_mma(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView cache_seqlens,
    tvm::ffi::TensorView sinks,
    tvm::ffi::TensorView out,
    int64_t q_stride_m,
    int64_t q_stride_h,
    double sm_scale,
    int64_t window_tokens,
    int64_t window_size,
    int64_t heads_per_program,
    int64_t has_sink) {
  using namespace swa_mma_detail;
  const int bs = (int)page_table.shape()[0];
  const int hq = (int)q.shape()[1];
  const int pt_stride = (int)page_table.shape()[1];
  const DLDevice device = q.device();
  const float s = (float)sm_scale;
  const void* qp = q.data_ptr();
  const void* kp = k.data_ptr();
  const void* vp = v.data_ptr();
  const int* ptp = (const int*)page_table.data_ptr();
  const int* slp = (const int*)cache_seqlens.data_ptr();
  const float* skp = has_sink ? (const float*)sinks.data_ptr() : nullptr;
  void* op = out.data_ptr();

#define SWA_MMA_CASE(w, hp, win)                                                          \
  if (window_tokens == w && heads_per_program == hp && window_size == win) {              \
    launch_swa_mma<w, hp, win, 8>(qp, kp, vp, ptp, slp, skp, op, bs, hq, (int)q_stride_m, \
                                  (int)q_stride_h, pt_stride, s, (int)has_sink, device);  \
    return;                                                                               \
  }
  SWA_MMA_CASE(8, 2, 129) SWA_MMA_CASE(8, 2, 128)
  SWA_MMA_CASE(8, 1, 129) SWA_MMA_CASE(8, 1, 128)
  SWA_MMA_CASE(6, 2, 129) SWA_MMA_CASE(6, 2, 128)
  SWA_MMA_CASE(4, 4, 129) SWA_MMA_CASE(4, 4, 128)
  SWA_MMA_CASE(2, 8, 129) SWA_MMA_CASE(2, 8, 128)
  SWA_MMA_CASE(1, 16, 129) SWA_MMA_CASE(1, 16, 128)
#undef SWA_MMA_CASE
  TVM_FFI_ICHECK(false) << "swa_decode_mma: unsupported shape";
}

}  // namespace sglang
