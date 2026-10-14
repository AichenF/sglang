// TP8 decode fusions for the DFlash draft (hidden 6144, M <= 32 tokens), goal11-dense.
//
// tp_arnq: the row-parallel projection's partial output (o_proj / down_proj, [M, H] bf16 on every rank) ->
//   reduce-scatter over the 8 ranks -> + residual (each rank keeps only its 768-column slice of the residual
//   stream) -> RMSNorm -> per-token FP8 quant (or bf16) -> all-gather, in ONE kernel. Replaces the custom
//   all-reduce (2-shot pull, 7-9 us), flashinfer fused_add_rmsnorm (5-6 us) and the per-token quant kernel
//   (2 us) of the unfused path. The per-token RMS and amax need the full row: ranks exchange their slice's
//   (sum x^2, max |x * gamma|) and every rank derives the same rstd / scale (rank-order reductions).
//
// Transport: Lamport-style pushes into the peers' CUDA-IPC receive buffers (two phases, alternated per call
//   through a per-token call counter): every 32-bit payload word is made non-zero (+0.0 -> -0.0, numerically
//   identical) so "word != 0" means "arrived"; the consumer resets its slots to 0 after reading. No fences, no
//   flags; one NVLink hop per exchange. A rank can only start call k+1 after every peer pushed its call-k
//   all-gather, i.e. after every peer finished reading (and resetting) the call-k reduce-scatter slots, so the
//   per-phase buffers are never overwritten early.
//
// One CTA per token (grid = 32, fixed: the per-token call counters of all ranks advance in lockstep because M is
// identical on every rank and a token's counter only advances when the token is active).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <climits>
#include <cstdint>

namespace sglang {
namespace dflash_tp {

constexpr int R = 8;               // TP ranks
constexpr int H = 6144;            // hidden
constexpr int SL = H / R;          // 768 columns owned per rank
constexpr int MAXT = 32;           // tokens (grid)
constexpr int THREADS = 256;
constexpr int ACT = SL / 4;        // 192 threads own 4 columns each in the owner phases
constexpr float FP8_MAX = 448.f;

// receive buffer layout (bytes, per rank)
constexpr int64_t RS_SLOT = SL * 2;                          // 1536: one source's bf16 slice of one token
constexpr int64_t RS_BYTES = 2LL * MAXT * R * RS_SLOT;       // 786432
constexpr int64_t ST_SLOT = 8;                               // (ssq, amax)
constexpr int64_t ST_BYTES = 2LL * MAXT * R * ST_SLOT;       // 4096
constexpr int64_t AG_SLOT = SL * 2 + 16;                     // bf16 slice (+ 16-byte trailer used by ag_norm)
constexpr int64_t AG_BYTES = 2LL * MAXT * R * AG_SLOT;       // 786432
constexpr int64_t RS_OFF = 0;
constexpr int64_t ST_OFF = RS_OFF + RS_BYTES;
constexpr int64_t AG_OFF = ST_OFF + ST_BYTES;
constexpr int64_t AX_SLOT = 16;                               // vocab-parallel argmax pair {key, tag, idx, tag}
constexpr int64_t AX_BYTES = 2LL * MAXT * R * AX_SLOT;       // 8192
constexpr int64_t AX_OFF = AG_OFF + AG_BYTES;
constexpr int64_t EM_SLOT = H * 2;                            // one embedding row (bf16), single source (the owner)
constexpr int64_t EM_BYTES = 2LL * MAXT * EM_SLOT;           // 786432
constexpr int64_t EM_OFF = AX_OFF + AX_BYTES;
constexpr int64_t BUF_BYTES = EM_OFF + EM_BYTES;

struct Params {
  const __nv_bfloat16* partial;   // [M, H] this rank's unreduced projection output (row stride H)
  __nv_bfloat16* residual;        // [M, H] residual (in/out, replicated on every rank)
  const __nv_bfloat16* gamma;     // [H]
  uint8_t* out_q;                 // mode 0: [M, H] e4m3
  float* out_scale;               // mode 0: [M]
  __nv_bfloat16* out_bf16;        // mode 1: [M, H]
  int* calls;                     // [MAXT] per-token call counters (local)
  uint8_t* peer[R];               // every rank's receive buffer (this rank's own at [rank])
  float eps;
  int rank;
  int M;
  unsigned long long* stamps;     // optional [gridDim.x][16] globaltimer (diagnostics; index blockIdx.x * 16 + k)
  int sx_ld;                      // mode 2: column-major group-scale row stride (>= M)
  int pre;                        // 1: the partial was already pushed into the RS slots by the producing GEMM's
                                  //    epilogue (skinny_gemm rs mode), own slice included; `partial` is unused
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v));
  return v;
}
#define ARNQ_STAMP(k) do { if (p.stamps && threadIdx.x == 0) p.stamps[blockIdx.x * 16 + (k)] = gtime(); } while (0)

__device__ __forceinline__ uint32_t nz(uint32_t w) { return w == 0u ? 0x80000000u : w; }  // +0.0 -> -0.0 (high half)
__device__ __forceinline__ uint32_t nz8(uint32_t w) { return w == 0u ? 0x80000000u : w; } // 4 x e4m3: top byte -> -0
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st_v2(void* p, uint2 v) {
  asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ uint4 ld_vol_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ld_vol_v2(const void* p) {
  uint2 v;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 poll_v4(const void* p) {
  uint4 v;
  do {
    v = ld_vol_v4(p);
  } while (v.x == 0u || v.y == 0u || v.z == 0u || v.w == 0u);
  return v;
}
__device__ __forceinline__ uint2 poll_v2(const void* p) {
  uint2 v;
  do {
    v = ld_vol_v2(p);
  } while (v.x == 0u || v.y == 0u);
  return v;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

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

// kMode 0: per-token fp8 (+ scale [M]); 1: bf16; 2: per-token-group-128 fp8 in the DeepGEMM pre-quantized layout.
// The all-reduce result h = bf16(sum of the partials in rank order) is bitwise the custom all-reduce's; every rank then
// gets the full h row and computes x = f32(h) + f32(residual) (unrounded), residual_out = bf16(x) and flashinfer
// FusedAddRMSNormKernel's exact arithmetic (64 virtual threads, ordered fma chains over columns 512 v + 8 t' + e, warp
// butterflies, w0 + w1, div.rn, rsqrt) for y = bf16((x * rstd) * w); mode 2 then runs SGLang's group quant bit for bit
// (built with --use_fast_math: qs = 448 * rcp.approx.ftz(amax), e4m3 satfinite of fmin(qs * y, 448), scale = amax / 448).
template <int kMode, bool kUsePDL>
__global__ void __launch_bounds__(THREADS) tp_arnq_kernel(const __grid_constant__ Params p) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ __align__(16) float s_v[H];                 // x = h + r (fp32), the norm's input
  __shared__ __align__(16) __nv_bfloat16 s_h[H];         // the gathered all-reduce result row
  __shared__ float s_w2[2];
  __shared__ float s_mx[THREADS / 32];
  const bool active = t < p.M;
  const int rank = p.rank;
  uint8_t* const mine = p.peer[rank];
#ifdef DFLASH_TP_STAMP_EDGES
  if (p.stamps && tid == 0) p.stamps[blockIdx.x * 16 + 9] = gtime();  // goal11-post diag: CTA resident (before the PDL wait)
#endif

  device::PDLWaitPrimary<kUsePDL>();
  // let the next kernel (a weight-streaming GEMM) get resident and prefetch now: this kernel is NVLink-bound and
  // leaves HBM idle; its own griddepcontrol.wait still covers every write below
  device::PDLTriggerSecondary<kUsePDL>();
  if (!active) return;
  ARNQ_STAMP(0);
  const int call = p.calls[t];
  const int phase = call & 1;
  __nv_bfloat16* const rrow = p.residual + static_cast<int64_t>(t) * H;  // full residual row (replicated, in place)
  const __nv_bfloat16* prow = p.partial + static_cast<int64_t>(t) * H;
  const int col = tid * 4;  // within the owned slice (tid < ACT)
  // goal11-post: the phase-5 residual chunks and phase-7 gamma chunks of this thread are issued now (they do not
  // depend on the exchange); their L2/HBM latency hides behind the two NVLink phases. In situ (layer-0 arnq after
  // fa_block, residual and gamma cold) the norm + quant tail was 5.5 us of mostly dependent load latency.
  uint4 rpre[3], gpre[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int e = c * THREADS * 8 + tid * 8;
    rpre[c] = *reinterpret_cast<const uint4*>(rrow + e);
    gpre[c] = __ldg(reinterpret_cast<const uint4*>(p.gamma + e));
  }

  // ---- 1. reduce-scatter push: peer q gets columns [q*SL, (q+1)*SL) of our partial ----
  if (!p.pre) {
    constexpr int NV = (R - 1) * (SL / 8);  // 672 16-byte vectors
    constexpr int PER = (NV + THREADS - 1) / THREADS;
    uint4 d[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      if (k < NV) {
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int q = di + (di >= rank);
        d[j] = *reinterpret_cast<const uint4*>(prow + q * SL + v * 8);
      }
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      if (k < NV) {
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int q = di + (di >= rank);
        uint4 w = d[j];
        w.x = nz(w.x); w.y = nz(w.y); w.z = nz(w.z); w.w = nz(w.w);
        st_v4(p.peer[q] + RS_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + rank) * RS_SLOT + v * 16, w);
      }
    }
  }
  ARNQ_STAMP(1);

  // ---- 2. owner reduce in rank order -> h (bf16) for the owned slice ----
  uint2 hb = make_uint2(0u, 0u);
  if (tid < ACT) {
    uint2 w[R];
    uint8_t* slot0 = mine + RS_OFF + (static_cast<int64_t>(phase) * MAXT + t) * R * RS_SLOT + col * 2;
#pragma unroll
    for (int s = 0; s < R; ++s) w[s] = (s == rank && !p.pre) ? make_uint2(1u, 1u) : ld_vol_v2(slot0 + s * RS_SLOT);
    bool done;
    do {
      done = true;
#pragma unroll
      for (int s = 0; s < R; ++s) {
        if (w[s].x == 0u || w[s].y == 0u) {
          w[s] = ld_vol_v2(slot0 + s * RS_SLOT);
          done = false;
        }
      }
    } while (!done);
    if (p.stamps && tid == 0) p.stamps[blockIdx.x * 16 + 8] = gtime();
#pragma unroll
    for (int s = 0; s < R; ++s)
      if (s != rank || p.pre) st_v2(slot0 + s * RS_SLOT, make_uint2(0u, 0u));
    if (!p.pre) w[rank] = *reinterpret_cast<const uint2*>(prow + rank * SL + col);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < R; ++s) {
      // the -0.0 arrival marker of a +0.0 pair adds like +0.0
      acc[0] += bf_lo(w[s].x); acc[1] += bf_hi(w[s].x); acc[2] += bf_lo(w[s].y); acc[3] += bf_hi(w[s].y);
    }
    hb = make_uint2(pack_bf2(acc[0], acc[1]), pack_bf2(acc[2], acc[3]));
    *reinterpret_cast<uint2*>(s_h + rank * SL + col) = hb;
  }
  // ---- 3. all-gather push of the owned h slice ----
  {
    const uint32_t o0 = __shfl_down_sync(0xffffffffu, hb.x, 1);
    const uint32_t o1 = __shfl_down_sync(0xffffffffu, hb.y, 1);
    if (tid < ACT && (tid & 1) == 0) {
      const uint4 d = make_uint4(nz(hb.x), nz(hb.y), nz(o0), nz(o1));
      const int v = tid >> 1;
#pragma unroll 1
      for (int di = 0; di < R - 1; ++di) {
        const int q = di + (di >= rank);
        st_v4(p.peer[q] + AG_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + rank) * AG_SLOT + v * 16, d);
      }
    }
  }
  ARNQ_STAMP(2);
  // ---- 4. receive the other h slices (concurrent polls) ----
  {
    constexpr int NV = (R - 1) * (SL / 8);  // 672
    constexpr int PER = (NV + THREADS - 1) / THREADS;
    uint4 d[PER];
    uint8_t* sl[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      sl[j] = nullptr;
      d[j] = make_uint4(1u, 1u, 1u, 1u);
      if (k < NV) {
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int s = di + (di >= rank);
        sl[j] = mine + AG_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + s) * AG_SLOT + v * 16;
        d[j] = ld_vol_v4(sl[j]);
      }
    }
    bool done;
    do {
      done = true;
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        if (d[j].x == 0u || d[j].y == 0u || d[j].z == 0u || d[j].w == 0u) {
          d[j] = ld_vol_v4(sl[j]);
          done = false;
        }
      }
    } while (!done);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      if (k < NV) {
        st_v4(sl[j], make_uint4(0u, 0u, 0u, 0u));
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int s = di + (di >= rank);
        uint4 w = d[j];
        w.x = w.x == 0x80000000u ? 0u : w.x;  // undo the arrival marker of a +0.0 pair
        w.y = w.y == 0x80000000u ? 0u : w.y;
        w.z = w.z == 0x80000000u ? 0u : w.z;
        w.w = w.w == 0x80000000u ? 0u : w.w;
        *reinterpret_cast<uint4*>(s_h + s * SL + v * 8) = w;
      }
    }
  }
  __syncthreads();
  ARNQ_STAMP(3);
  // ---- 5. x = f32(h) + f32(r) (unrounded) -> s_v; residual_out = bf16(x) ----
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int e = c * THREADS * 8 + tid * 8;
    const uint4 hv = *reinterpret_cast<const uint4*>(s_h + e);
    const uint4 rv = rpre[c];
    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
    const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
    float v[8];
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = bf_lo(hw[j]) + bf_lo(rw[j]);
      v[2 * j + 1] = bf_hi(hw[j]) + bf_hi(rw[j]);
      o[j] = pack_bf2(v[2 * j], v[2 * j + 1]);
    }
    *reinterpret_cast<float4*>(s_v + e) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(s_v + e + 4) = make_float4(v[4], v[5], v[6], v[7]);
    *reinterpret_cast<uint4*>(rrow + e) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  __syncthreads();
  // ---- 6. sum of squares in flashinfer's order (64 virtual threads, 2 warps) ----
  if (tid < 64) {
    float acc = 0.f;
#pragma unroll
    for (int v = 0; v < 12; ++v) {
      const float4* vp = reinterpret_cast<const float4*>(s_v + 512 * v + 8 * tid);
      const float4 a = vp[0], b = vp[1];
      acc = fmaf(a.x, a.x, acc); acc = fmaf(a.y, a.y, acc); acc = fmaf(a.z, a.z, acc); acc = fmaf(a.w, a.w, acc);
      acc = fmaf(b.x, b.x, acc); acc = fmaf(b.y, b.y, acc); acc = fmaf(b.z, b.z, acc); acc = fmaf(b.w, b.w, acc);
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) s_w2[warp] = acc;
  }
  __syncthreads();
  const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(__fadd_rn(s_w2[0], s_w2[1]), static_cast<float>(H)), p.eps));

  // ---- 7. y = bf16((x * rstd) * w) and the output form; 3 chunks of 8 consecutive columns per thread ----
  float tmax = 0.f;
  uint32_t yb[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int e = c * THREADS * 8 + tid * 8;
    const float4 a = *reinterpret_cast<const float4*>(s_v + e);
    const float4 b = *reinterpret_cast<const float4*>(s_v + e + 4);
    const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const uint4 gv = gpre[c];
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float n0 = __fmul_rn(__fmul_rn(xv[2 * j], rstd), __fadd_rn(bf_lo(gw[j]), 0.0f));
      const float n1 = __fmul_rn(__fmul_rn(xv[2 * j + 1], rstd), __fadd_rn(bf_hi(gw[j]), 0.0f));
      yb[c][j] = pack_bf2(n0, n1);
      tmax = fmaxf(tmax, fmaxf(fabsf(bf_lo(yb[c][j])), fabsf(bf_hi(yb[c][j]))));
    }
  }
  if constexpr (kMode == 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int e = c * THREADS * 8 + tid * 8;
      *reinterpret_cast<uint4*>(p.out_bf16 + static_cast<int64_t>(t) * H + e) =
          make_uint4(yb[c][0], yb[c][1], yb[c][2], yb[c][3]);
    }
  } else if constexpr (kMode == 2) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int e = c * THREADS * 8 + tid * 8;
      float gmx = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) gmx = fmaxf(gmx, fmaxf(fabsf(bf_lo(yb[c][j])), fabsf(bf_hi(yb[c][j]))));
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) gmx = fmaxf(gmx, __shfl_xor_sync(0xffffffffu, gmx, o));  // 16 threads = 128 cols
      gmx = fmaxf(gmx, 1e-10f);
      const float qs = mul_ftz(rcp_approx_ftz(gmx), FP8_MAX);
      uint32_t q[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        q[i] = static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, bf_lo(yb[c][2 * i])), FP8_MAX),
                                                 fminf(mul_ftz(qs, bf_hi(yb[c][2 * i])), FP8_MAX))) |
               (static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, bf_lo(yb[c][2 * i + 1])), FP8_MAX),
                                                  fminf(mul_ftz(qs, bf_hi(yb[c][2 * i + 1])), FP8_MAX)))
                << 16);
      *reinterpret_cast<uint2*>(p.out_q + static_cast<int64_t>(t) * H + e) = make_uint2(q[0], q[1]);
      if ((tid & 15) == 0) p.out_scale[static_cast<int64_t>(e >> 7) * p.sx_ld + t] = mul_ftz(gmx, __uint_as_float(0x3b124925u));
    }
  } else {
    // per-token scale over the whole row (draft: its own quant numerics)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    if (lane == 0) s_mx[warp] = tmax;
    __syncthreads();
    float am = 0.f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) am = fmaxf(am, s_mx[w]);
    const float scale = fmaxf(am, 1e-10f) / FP8_MAX;
    const float inv = 1.f / scale;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int e = c * THREADS * 8 + tid * 8;
      uint32_t q[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        q[i] = static_cast<uint32_t>(pack_e4m3x2(fminf(fmaxf(bf_lo(yb[c][2 * i]) * inv, -FP8_MAX), FP8_MAX),
                                                 fminf(fmaxf(bf_hi(yb[c][2 * i]) * inv, -FP8_MAX), FP8_MAX))) |
               (static_cast<uint32_t>(pack_e4m3x2(fminf(fmaxf(bf_lo(yb[c][2 * i + 1]) * inv, -FP8_MAX), FP8_MAX),
                                                  fminf(fmaxf(bf_hi(yb[c][2 * i + 1]) * inv, -FP8_MAX), FP8_MAX)))
                << 16);
      *reinterpret_cast<uint2*>(p.out_q + static_cast<int64_t>(t) * H + e) = make_uint2(q[0], q[1]);
    }
    if (tid == 0) p.out_scale[t] = scale;
  }
  __syncthreads();
  ARNQ_STAMP(5);
  if (tid == 0) p.calls[t] = call + 1;
}

// ---- column-parallel output all-gather + RMSNorm (draft `fc` -> hidden_norm): in [M, SL] (this rank's columns) ->
// out [M, H] bf16 normalized on every rank. Uses the all-gather plane and call counters of tp_arnq (same phases). ----
template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS) ag_norm_kernel(const __grid_constant__ Params p) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float s_red[THREADS / 32];
  __shared__ float s_stat[R];
  const int rank = p.rank;
  uint8_t* const mine = p.peer[rank];
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  if (t >= p.M) return;
  const int call = p.calls[t];
  const int phase = call & 1;
  __nv_bfloat16* const orow = p.out_bf16 + static_cast<int64_t>(t) * H;
  const int col = tid * 4;
  float ssq = 0.f;
  uint2 xb = make_uint2(0u, 0u);
  if (tid < ACT) {
    xb = *reinterpret_cast<const uint2*>(p.partial + static_cast<int64_t>(t) * SL + col);
    const float x0 = bf_lo(xb.x), x1 = bf_hi(xb.x), x2 = bf_lo(xb.y), x3 = bf_hi(xb.y);
    ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  if (lane == 0) s_red[warp] = ssq;
  {
    const uint32_t o0 = __shfl_down_sync(0xffffffffu, xb.x, 1);
    const uint32_t o1 = __shfl_down_sync(0xffffffffu, xb.y, 1);
    if (tid < ACT && (tid & 1) == 0) {
      const uint4 d = make_uint4(nz(xb.x), nz(xb.y), nz(o0), nz(o1));
      const int v = tid >> 1;
#pragma unroll 1
      for (int di = 0; di < R - 1; ++di) {
        const int q = di + (di >= rank);
        st_v4(p.peer[q] + AG_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + rank) * AG_SLOT + v * 16, d);
      }
      *reinterpret_cast<uint4*>(orow + rank * SL + col) = make_uint4(xb.x, xb.y, o0, o1);
    }
  }
  __syncthreads();
  if (tid < R - 1) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) a += s_red[w];
    const int q = tid + (tid >= rank);
    st_v4(p.peer[q] + AG_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + rank) * AG_SLOT + SL * 2,
          make_uint4(nz(__float_as_uint(a)), nz(__float_as_uint(a)), nz(__float_as_uint(a)), nz(__float_as_uint(a))));
    if (tid == 0) s_stat[rank] = a;
  }
  {
    constexpr int NV = (R - 1) * (SL / 8 + 1);
    constexpr int PER = (NV + THREADS - 1) / THREADS;
    uint4 d[PER];
    uint8_t* sl[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      sl[j] = nullptr;
      d[j] = make_uint4(1u, 1u, 1u, 1u);
      if (k < NV) {
        const int di = k / (SL / 8 + 1);
        const int v = k - di * (SL / 8 + 1);
        const int s = di + (di >= rank);
        sl[j] = mine + AG_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + s) * AG_SLOT + v * 16;
        d[j] = ld_vol_v4(sl[j]);
      }
    }
    bool done;
    do {
      done = true;
#pragma unroll
      for (int j = 0; j < PER; ++j)
        if (d[j].x == 0u || d[j].y == 0u || d[j].z == 0u || d[j].w == 0u) {
          d[j] = ld_vol_v4(sl[j]);
          done = false;
        }
    } while (!done);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * THREADS + tid;
      if (k < NV) {
        st_v4(sl[j], make_uint4(0u, 0u, 0u, 0u));
        const int di = k / (SL / 8 + 1);
        const int v = k - di * (SL / 8 + 1);
        const int s = di + (di >= rank);
        if (v < SL / 8) {
          uint4 w = d[j];
          w.x = w.x == 0x80000000u ? 0u : w.x;
          w.y = w.y == 0x80000000u ? 0u : w.y;
          w.z = w.z == 0x80000000u ? 0u : w.z;
          w.w = w.w == 0x80000000u ? 0u : w.w;
          *reinterpret_cast<uint4*>(orow + s * SL + v * 8) = w;
        } else {
          s_stat[s] = __uint_as_float(d[j].x);
        }
      }
    }
  }
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int s = 0; s < R; ++s) tot += s_stat[s];
  const float rstd = rsqrtf(tot / static_cast<float>(H) + p.eps);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int e = c * THREADS * 8 + tid * 8;
    const uint4 xv = *reinterpret_cast<const uint4*>(orow + e);
    const uint4 gv = *reinterpret_cast<const uint4*>(p.gamma + e);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      w[j] = pack_bf2(bf_lo(xw[j]) * rstd * bf_lo(gw[j]), bf_hi(xw[j]) * rstd * bf_hi(gw[j]));
    // each thread rewrites only the elements it read (in place)
    *reinterpret_cast<uint4*>(orow + e) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  __syncthreads();
  if (tid == 0) p.calls[t] = call + 1;
}

// ---- act_and_mul + per-token FP8 quant: in [M, 2I] (gate | up) bf16 -> out [M, I] e4m3 + scale [M] ----
template <int I, bool kUsePDL>
__global__ void __launch_bounds__(THREADS) silu_mul_quant_kernel(
    const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ out, float* __restrict__ scale, int64_t in_ld) {
  static_assert(I % (THREADS * 8) == 0);
  constexpr int PER = I / THREADS;  // elements per thread (8 for I = 2048)
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_red[THREADS / 32];
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const __nv_bfloat16* row = in + t * in_ld;
  float y[PER];
  float amx = 0.f;
#pragma unroll
  for (int c = 0; c < PER; c += 8) {
    const int e = tid * PER + c;
    const uint4 gv = *reinterpret_cast<const uint4*>(row + e);
    const uint4 uv = *reinterpret_cast<const uint4*>(row + I + e);
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    const uint32_t uw[4] = {uv.x, uv.y, uv.z, uv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = bf_lo(gw[j]), g1 = bf_hi(gw[j]);
      const float u0 = bf_lo(uw[j]), u1 = bf_hi(uw[j]);
      // match SiluAndMul: silu in fp32, product rounded to bf16 (the unfused kernel writes bf16)
      const float a0 = __bfloat162float(__float2bfloat16_rn(g0 / (1.f + __expf(-g0)) * u0));
      const float a1 = __bfloat162float(__float2bfloat16_rn(g1 / (1.f + __expf(-g1)) * u1));
      y[c + 2 * j] = a0;
      y[c + 2 * j + 1] = a1;
      amx = fmaxf(amx, fmaxf(fabsf(a0), fabsf(a1)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amx = fmaxf(amx, __shfl_xor_sync(0xffffffffu, amx, o));
  if ((tid & 31) == 0) s_red[tid >> 5] = amx;
  __syncthreads();
  float m = 0.f;
#pragma unroll
  for (int w = 0; w < THREADS / 32; ++w) m = fmaxf(m, s_red[w]);
  const float sc = fmaxf(m, 1e-10f) / FP8_MAX;
  const float inv = 1.f / sc;
#pragma unroll
  for (int c = 0; c < PER; c += 8) {
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __nv_fp8_e4m3 q(fminf(fmaxf(y[c + j] * inv, -FP8_MAX), FP8_MAX));
      w[j >> 2] |= static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(&q)) << (8 * (j & 3));
    }
    *reinterpret_cast<uint2*>(out + static_cast<int64_t>(t) * I + tid * PER + c) = make_uint2(w[0], w[1]);
  }
  if (tid == 0) scale[t] = sc;
}

// ---- RMSNorm + per-token FP8 quant of a replicated [M, H] row (first draft layer: no residual add) ----
template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS) rmsnorm_quant_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma, float eps, int rank,
    uint8_t* __restrict__ out, float* __restrict__ scale) {
  constexpr int PER = H / THREADS;  // 24
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_red[2][THREADS / 32];
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const __nv_bfloat16* row = x + static_cast<int64_t>(t) * H;
  float v[PER], gm[PER];
  float ssq = 0.f;
#pragma unroll
  for (int c = 0; c < PER; c += 8) {
    const int e = (c / 8) * THREADS * 8 + tid * 8;  // 3 chunks of 8, strided by THREADS * 8
    const uint4 xv = *reinterpret_cast<const uint4*>(row + e);
    const uint4 gv = *reinterpret_cast<const uint4*>(gamma + e);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[c + 2 * j] = bf_lo(xw[j]);
      v[c + 2 * j + 1] = bf_hi(xw[j]);
      gm[c + 2 * j] = bf_lo(gw[j]);
      gm[c + 2 * j + 1] = bf_hi(gw[j]);
    }
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) ssq += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  if ((tid & 31) == 0) s_red[0][tid >> 5] = ssq;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < THREADS / 32; ++w) tot += s_red[0][w];
  const float rstd = rsqrtf(tot / static_cast<float>(H) + eps);
  float amx = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    // unfused: RMSNorm writes bf16, the quant reads it
    v[i] = __bfloat162float(__float2bfloat16_rn(v[i] * rstd * gm[i]));
    amx = fmaxf(amx, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amx = fmaxf(amx, __shfl_xor_sync(0xffffffffu, amx, o));
  if ((tid & 31) == 0) s_red[1][tid >> 5] = amx;
  __syncthreads();
  float m = 0.f;
#pragma unroll
  for (int w = 0; w < THREADS / 32; ++w) m = fmaxf(m, s_red[1][w]);
  const float sc = fmaxf(m, 1e-10f) / FP8_MAX;
  const float inv = 1.f / sc;
#pragma unroll
  for (int c = 0; c < PER; c += 8) {
    const int e = (c / 8) * THREADS * 8 + tid * 8;
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __nv_fp8_e4m3 q(fminf(fmaxf(v[c + j] * inv, -FP8_MAX), FP8_MAX));
      w[j >> 2] |= static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(&q)) << (8 * (j & 3));
    }
    *reinterpret_cast<uint2*>(out + static_cast<int64_t>(t) * H + e) = make_uint2(w[0], w[1]);
  }
  if (tid == 0) scale[t] = sc;
  device::PDLTriggerSecondary<kUsePDL>();
}


// ---- vocab-parallel greedy argmax: local (key, index) per row -> exchange over IPC -> select, one kernel ----
// Bit-exact with torch.argmax over the full logits (VocabParallelGreedyHead semantics): monotonic float->int32 key,
// NaN is the maximum, -0.0 == +0.0, ties -> lowest global index (ranks own ascending vocab shards). Messages are
// tagged {key, tag, idx, tag} (8-byte halves are single-copy atomic), two phases alternated by per-row counters.
struct ArgmaxParams {
  const void* logits;   // [n, Vs] row stride ld (bf16 or fp32)
  int64_t ld;
  int Vs;
  int vocab_start;
  int64_t* out;         // [n]
  int* calls;           // [MAXT] per-row counters (local)
  uint8_t* peer[R];
  int rank;
  int n;
  int group;            // > 1: output row r reads logits row (r / (group-1)) * group + 1 + r % (group-1) (skip block pos 0)
  int64_t* cand;        // goal12-glue (group > 1, optional): verify candidates [blocks, group]: cand[b, 1 + j] = out of
  const int64_t* cand_src;  //   row (b, j), cand[b, 0] = cand_src[b * group] (the block's bonus id)
};

__device__ __forceinline__ int ordered_key(float v) {
  v = (v == 0.f) ? 0.f : v;
  const int bits = __float_as_int(v);
  int key = bits >= 0 ? bits : (bits ^ 0x7FFFFFFF);
  return (v != v) ? 0x7FFFFFFF : key;
}

template <typename T, bool kUsePDL>
__global__ void __launch_bounds__(THREADS) argmax_ag_kernel(const __grid_constant__ ArgmaxParams p) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ int s_key[THREADS / 32], s_idx[THREADS / 32];
  __shared__ int s_pk[R], s_pi[R];
  device::PDLWaitPrimary<kUsePDL>();
  if (t >= p.n) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int src = p.group > 1 ? (t / (p.group - 1)) * p.group + 1 + t % (p.group - 1) : t;
  const T* row = static_cast<const T*>(p.logits) + static_cast<int64_t>(src) * p.ld;
  int bk = INT_MIN, bi = INT_MAX;
  if constexpr (sizeof(T) == 2) {
    if ((p.Vs & 7) == 0 && (p.ld & 7) == 0 && (reinterpret_cast<uintptr_t>(row) & 15) == 0) {
      // 16-byte vectors (8 bf16); per thread the columns still ascend, so strict '>' keeps the first index.
      // goal11-glue: the first AX_UNR vectors of every thread are loaded up front (one memory latency instead of a
      // dependent load -> compare chain per vector; Vs 19072 = 9.3 vectors per thread); the tail loop covers the rest.
      constexpr int AX_UNR = 10;
      const int nv = p.Vs >> 3;
      uint4 wv[AX_UNR];
#pragma unroll
      for (int u = 0; u < AX_UNR; ++u) {
        const int c8 = tid + u * THREADS;
        wv[u] = c8 < nv ? __ldcg(reinterpret_cast<const uint4*>(row + c8 * 8)) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < AX_UNR; ++u) {
        const int c8 = tid + u * THREADS;
        if (c8 < nv) {
          const uint32_t ws[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int k0 = ordered_key(bf_lo(ws[j])), k1 = ordered_key(bf_hi(ws[j]));
            if (k0 > bk) { bk = k0; bi = c8 * 8 + 2 * j; }
            if (k1 > bk) { bk = k1; bi = c8 * 8 + 2 * j + 1; }
          }
        }
      }
      for (int c8 = tid + AX_UNR * THREADS; c8 < nv; c8 += THREADS) {
        const uint4 w = *reinterpret_cast<const uint4*>(row + c8 * 8);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int k0 = ordered_key(bf_lo(ws[j])), k1 = ordered_key(bf_hi(ws[j]));
          if (k0 > bk) { bk = k0; bi = c8 * 8 + 2 * j; }
          if (k1 > bk) { bk = k1; bi = c8 * 8 + 2 * j + 1; }
        }
      }
    } else {
      for (int c = tid; c < p.Vs; c += THREADS) {
        const int k = ordered_key(__bfloat162float(row[c]));
        if (k > bk) { bk = k; bi = c; }
      }
    }
  } else {
    for (int c = tid; c < p.Vs; c += THREADS) {
      const int k = ordered_key(row[c]);
      if (k > bk) { bk = k; bi = c; }  // ascending c per thread: first index wins ties
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const int ok = __shfl_xor_sync(0xffffffffu, bk, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
  }
  if ((tid & 31) == 0) { s_key[tid >> 5] = bk; s_idx[tid >> 5] = bi; }
  __syncthreads();
  const int call = p.calls[t];
  const int phase = call & 1;
  const uint32_t tag = static_cast<uint32_t>(call + 1) | 0x40000000u;
  if (tid < R) {
    int k = s_key[0], i = s_idx[0];
#pragma unroll
    for (int w = 1; w < THREADS / 32; ++w)
      if (s_key[w] > k || (s_key[w] == k && s_idx[w] < i)) { k = s_key[w]; i = s_idx[w]; }
    i += p.vocab_start;
    const int s = tid;
    if (s == p.rank) {
      s_pk[s] = k; s_pi[s] = i;
    } else {
      // push ours to peer s, then collect peer s's
      st_v4(p.peer[s] + AX_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + p.rank) * AX_SLOT,
            make_uint4(static_cast<uint32_t>(k), tag, static_cast<uint32_t>(i), tag));
      const uint8_t* slot = p.peer[p.rank] + AX_OFF + ((static_cast<int64_t>(phase) * MAXT + t) * R + s) * AX_SLOT;
      uint4 d;
      do {
        d = ld_vol_v4(slot);
      } while (d.y != tag || d.w != tag);
      s_pk[s] = static_cast<int>(d.x); s_pi[s] = static_cast<int>(d.z);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int k = s_pk[0], i = s_pi[0];
#pragma unroll
    for (int s = 1; s < R; ++s)
      if (s_pk[s] > k || (s_pk[s] == k && s_pi[s] < i)) { k = s_pk[s]; i = s_pi[s]; }
    p.out[t] = static_cast<int64_t>(i);
    p.calls[t] = call + 1;
    if (p.cand != nullptr) {
      const int b = t / (p.group - 1), jj = t % (p.group - 1);
      p.cand[static_cast<int64_t>(b) * p.group + 1 + jj] = static_cast<int64_t>(i);
      if (jj == 0) p.cand[static_cast<int64_t>(b) * p.group] = p.cand_src[static_cast<int64_t>(b) * p.group];
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}


// ---- vocab-parallel embedding: the owner rank of each token's row pushes it to every peer (replaces the zero-padded
// local gather + all-reduce; the all-reduce sum of one row and seven zero rows is that row exactly) ----
struct EmbedParams {
  const void* ids;            // [M] int32 / int64
  int ids64;
  const __nv_bfloat16* weight;  // [rows, H] this rank's shard
  __nv_bfloat16* out;         // [M, H]
  int* calls;                 // [MAXT] own counters
  uint8_t* peer[R];
  int64_t start[R];           // org vocab range [start, end) of each rank's shard
  int64_t end[R];
  int rank;
  int M;
  int local;                  // goal12-glue: weight is the FULL (replicated) table, start/end[rank] = [0, vocab): every
                              // rank gathers its rows locally (no pushes, no receive polling, counters untouched)
  int early;                  // goal12-glue: trigger the PDL secondary at entry (its pre-wait weight streaming overlaps
                              // this kernel; the secondary's griddepcontrol.wait still orders every read of our output)
};

template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS) embed_ag_kernel(const __grid_constant__ EmbedParams p) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  device::PDLWaitPrimary<kUsePDL>();
  if (p.early) device::PDLTriggerSecondary<kUsePDL>();
  if (t >= p.M) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t id = p.ids64 ? static_cast<const int64_t*>(p.ids)[t] : static_cast<const int32_t*>(p.ids)[t];
  int owner = -1;
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (id >= p.start[r] && id < p.end[r]) owner = r;
  const int call = p.calls[t];
  const int phase = call & 1;
  __nv_bfloat16* orow = p.out + static_cast<int64_t>(t) * H;
  constexpr int NV = H / 8;  // 768 16-byte vectors
  if (owner == p.rank) {
    const __nv_bfloat16* src = p.weight + (id - p.start[p.rank]) * H;
    // goal11-glue: all of this thread's row vectors are loaded up front (one HBM latency, not one per vector)
    constexpr int PER = NV / THREADS;  // 3
    uint4 dv[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) dv[j] = *reinterpret_cast<const uint4*>(src + (j * THREADS + tid) * 8);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int v = j * THREADS + tid;
      const uint4 d = dv[j];
      const uint4 m = make_uint4(nz(d.x), nz(d.y), nz(d.z), nz(d.w));
      if (!p.local) {
#pragma unroll 1
        for (int di = 0; di < R - 1; ++di) {
          const int q = di + (di >= p.rank);
          st_v4(p.peer[q] + EM_OFF + (static_cast<int64_t>(phase) * MAXT + t) * EM_SLOT + v * 16, m);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) *reinterpret_cast<uint4*>(orow + (j * THREADS + tid) * 8) = dv[j];
  } else if (owner >= 0) {
    uint8_t* slot = p.peer[p.rank] + EM_OFF + (static_cast<int64_t>(phase) * MAXT + t) * EM_SLOT;
    constexpr int PER = NV / THREADS;  // 3
    uint4 d[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) d[j] = ld_vol_v4(slot + (j * THREADS + tid) * 16);
    bool done;
    do {
      done = true;
#pragma unroll
      for (int j = 0; j < PER; ++j)
        if (d[j].x == 0u || d[j].y == 0u || d[j].z == 0u || d[j].w == 0u) {
          d[j] = ld_vol_v4(slot + (j * THREADS + tid) * 16);
          done = false;
        }
    } while (!done);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int v = j * THREADS + tid;
      st_v4(slot + v * 16, make_uint4(0u, 0u, 0u, 0u));
      // undo the arrival marker of an all-zero word (+0.0 pair)
      uint4 w = d[j];
      w.x = w.x == 0x80000000u ? 0u : w.x;
      w.y = w.y == 0x80000000u ? 0u : w.y;
      w.z = w.z == 0x80000000u ? 0u : w.z;
      w.w = w.w == 0x80000000u ? 0u : w.w;
      *reinterpret_cast<uint4*>(orow + v * 8) = w;
    }
  } else {
    // out-of-range id: the unfused path masks it to a zero row on every rank
    for (int v = tid; v < NV; v += THREADS) *reinterpret_cast<uint4*>(orow + v * 8) = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();
  device::PDLTriggerSecondary<kUsePDL>();
  if (tid == 0 && owner >= 0 && !p.local) p.calls[t] = call + 1;
}

// ---- embedding (owner push, as embed_ag_kernel) + the first decoder layer's input RMSNorm + group-128 fp8 quant in one
// kernel: out = the embedding row (bf16, the residual stream), out_q / out_scale = the layer's qkv_proj input in the
// DeepGEMM pre-quantized layout. Norm = flashinfer/sgl_kernel rmsnorm's exact arithmetic (same reduction as tp_arnq). ----
struct EmbedNQParams {
  EmbedParams e;
  const __nv_bfloat16* gamma;  // [H]
  uint8_t* out_q;              // [>= M, H] e4m3
  float* out_scale;            // [H / 128, sx_ld]
  int sx_ld;
  float eps;
};

template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS) embed_nq_kernel(const __grid_constant__ EmbedNQParams q) {
  const EmbedParams& p = q.e;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ __align__(16) float s_v[H];
  __shared__ float s_w2[2];
  device::PDLWaitPrimary<kUsePDL>();
  if (p.early) device::PDLTriggerSecondary<kUsePDL>();
  if (t >= p.M) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t id = p.ids64 ? static_cast<const int64_t*>(p.ids)[t] : static_cast<const int32_t*>(p.ids)[t];
  int owner = -1;
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (id >= p.start[r] && id < p.end[r]) owner = r;
  const int call = p.calls[t];
  const int phase = call & 1;
  __nv_bfloat16* orow = p.out + static_cast<int64_t>(t) * H;
  constexpr int NV = H / 8;          // 768 16-byte vectors
  constexpr int PER = NV / THREADS;  // 3
  // goal11-post: the norm's gamma chunks issued up front (a dependent cold load at the kernel's tail otherwise)
  uint4 gpre[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) gpre[c] = __ldg(reinterpret_cast<const uint4*>(q.gamma + c * THREADS * 8 + tid * 8));
  uint4 d[PER];
  if (owner == p.rank) {
    const __nv_bfloat16* src = p.weight + (id - p.start[p.rank]) * H;
#pragma unroll
    for (int j = 0; j < PER; ++j) d[j] = *reinterpret_cast<const uint4*>(src + (j * THREADS + tid) * 8);
#pragma unroll
    if (!p.local) {
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const uint4 m = make_uint4(nz(d[j].x), nz(d[j].y), nz(d[j].z), nz(d[j].w));
#pragma unroll 1
        for (int di = 0; di < R - 1; ++di) {
          const int qq = di + (di >= p.rank);
          st_v4(p.peer[qq] + EM_OFF + (static_cast<int64_t>(phase) * MAXT + t) * EM_SLOT + (j * THREADS + tid) * 16, m);
        }
      }
    }
  } else if (owner >= 0) {
    uint8_t* slot = p.peer[p.rank] + EM_OFF + (static_cast<int64_t>(phase) * MAXT + t) * EM_SLOT;
#pragma unroll
    for (int j = 0; j < PER; ++j) d[j] = ld_vol_v4(slot + (j * THREADS + tid) * 16);
    bool done;
    do {
      done = true;
#pragma unroll
      for (int j = 0; j < PER; ++j)
        if (d[j].x == 0u || d[j].y == 0u || d[j].z == 0u || d[j].w == 0u) {
          d[j] = ld_vol_v4(slot + (j * THREADS + tid) * 16);
          done = false;
        }
    } while (!done);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      st_v4(slot + (j * THREADS + tid) * 16, make_uint4(0u, 0u, 0u, 0u));
      d[j].x = d[j].x == 0x80000000u ? 0u : d[j].x;
      d[j].y = d[j].y == 0x80000000u ? 0u : d[j].y;
      d[j].z = d[j].z == 0x80000000u ? 0u : d[j].z;
      d[j].w = d[j].w == 0x80000000u ? 0u : d[j].w;
    }
  } else {
#pragma unroll
    for (int j = 0; j < PER; ++j) d[j] = make_uint4(0u, 0u, 0u, 0u);
  }
  // the row: bf16 out (residual) + fp32 copy for the norm
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int e = (j * THREADS + tid) * 8;
    *reinterpret_cast<uint4*>(orow + e) = d[j];
    const uint32_t w[4] = {d[j].x, d[j].y, d[j].z, d[j].w};
    *reinterpret_cast<float4*>(s_v + e) = make_float4(bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1]));
    *reinterpret_cast<float4*>(s_v + e + 4) = make_float4(bf_lo(w[2]), bf_hi(w[2]), bf_lo(w[3]), bf_hi(w[3]));
  }
  __syncthreads();
  if (tid < 64) {
    float acc = 0.f;
#pragma unroll
    for (int v = 0; v < 12; ++v) {
      const float4* vp = reinterpret_cast<const float4*>(s_v + 512 * v + 8 * tid);
      const float4 a = vp[0], b = vp[1];
      acc = fmaf(a.x, a.x, acc); acc = fmaf(a.y, a.y, acc); acc = fmaf(a.z, a.z, acc); acc = fmaf(a.w, a.w, acc);
      acc = fmaf(b.x, b.x, acc); acc = fmaf(b.y, b.y, acc); acc = fmaf(b.z, b.z, acc); acc = fmaf(b.w, b.w, acc);
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) s_w2[warp] = acc;
  }
  __syncthreads();
  const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(__fadd_rn(s_w2[0], s_w2[1]), static_cast<float>(H)), q.eps));
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int e = c * THREADS * 8 + tid * 8;
    const float4 a = *reinterpret_cast<const float4*>(s_v + e);
    const float4 b = *reinterpret_cast<const float4*>(s_v + e + 4);
    const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const uint4 gv = gpre[c];
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    uint32_t yb[4];
    float gmx = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float n0 = __fmul_rn(__fmul_rn(xv[2 * j], rstd), __fadd_rn(bf_lo(gw[j]), 0.0f));
      const float n1 = __fmul_rn(__fmul_rn(xv[2 * j + 1], rstd), __fadd_rn(bf_hi(gw[j]), 0.0f));
      yb[j] = pack_bf2(n0, n1);
      gmx = fmaxf(gmx, fmaxf(fabsf(bf_lo(yb[j])), fabsf(bf_hi(yb[j]))));
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) gmx = fmaxf(gmx, __shfl_xor_sync(0xffffffffu, gmx, o));
    gmx = fmaxf(gmx, 1e-10f);
    const float qs = mul_ftz(rcp_approx_ftz(gmx), FP8_MAX);
    uint32_t qv[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      qv[i] = static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, bf_lo(yb[2 * i])), FP8_MAX),
                                                fminf(mul_ftz(qs, bf_hi(yb[2 * i])), FP8_MAX))) |
              (static_cast<uint32_t>(pack_e4m3x2(fminf(mul_ftz(qs, bf_lo(yb[2 * i + 1])), FP8_MAX),
                                                 fminf(mul_ftz(qs, bf_hi(yb[2 * i + 1])), FP8_MAX)))
               << 16);
    *reinterpret_cast<uint2*>(q.out_q + static_cast<int64_t>(t) * H + e) = make_uint2(qv[0], qv[1]);
    if ((tid & 15) == 0) q.out_scale[static_cast<int64_t>(e >> 7) * q.sx_ld + t] = mul_ftz(gmx, __uint_as_float(0x3b124925u));
  }
  __syncthreads();
  device::PDLTriggerSecondary<kUsePDL>();
  if (tid == 0 && owner >= 0 && !p.local) p.calls[t] = call + 1;
}

}  // namespace dflash_tp

template <bool kUsePDL>
struct DFlashTPFused {
  // mode 0: out_q [M, H] e4m3 + out_scale [M]; mode 1: out_bf16 [M, H]. peers: int64 [8] receive-buffer addresses.
  static void arnq(
      tvm::ffi::TensorView partial,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView out_scale,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView peers,
      double eps,
      int64_t rank,
      int64_t mode,
      int64_t stamps,
      int64_t pre) {
    using namespace host;
    using namespace dflash_tp;
    const int M = static_cast<int>(partial.size(0));
    // (a single row's stride is irrelevant: the kernel uses the dense row pitch)
    RuntimeCheck(M >= 1 && M <= MAXT && partial.size(1) == H && (M == 1 || partial.stride(0) == H) && partial.stride(1) == 1);
    RuntimeCheck(residual.size(0) >= M && residual.size(1) == H && (residual.size(0) == 1 || residual.stride(0) == H));
    RuntimeCheck(gamma.numel() == H && peers.numel() == R && calls.numel() >= MAXT);
    RuntimeCheck(out.size(0) >= M && out.size(1) == H && (out.size(0) == 1 || out.stride(0) == H));
    Params prm{};
    prm.partial = static_cast<const __nv_bfloat16*>(partial.data_ptr());
    prm.residual = static_cast<__nv_bfloat16*>(residual.data_ptr());
    prm.gamma = static_cast<const __nv_bfloat16*>(gamma.data_ptr());
    prm.calls = static_cast<int*>(calls.data_ptr());
    // peers is a host (CPU) int64 tensor
    const int64_t* pp = static_cast<const int64_t*>(peers.data_ptr());
    for (int i = 0; i < R; ++i) prm.peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
    prm.eps = static_cast<float>(eps);
    prm.rank = static_cast<int>(rank);
    prm.M = M;
    prm.stamps = reinterpret_cast<unsigned long long*>(stamps);
    prm.pre = pre != 0 ? 1 : 0;
    if (mode == 2) {
      // out_scale: [H/128, sx_ld] fp32 (column-major group scales)
      RuntimeCheck(out.dtype().bits == 8 && out_scale.ndim() == 2 && out_scale.size(0) == H / 128 && out_scale.stride(1) == 1);
      prm.sx_ld = static_cast<int>(out_scale.stride(0));
      RuntimeCheck(prm.sx_ld >= M);
      prm.out_q = static_cast<uint8_t*>(out.data_ptr());
      prm.out_scale = static_cast<float*>(out_scale.data_ptr());
      LaunchKernel(dim3(MAXT), dim3(THREADS), partial.device()).enable_pdl(kUsePDL)(tp_arnq_kernel<2, kUsePDL>, prm);
    } else if (mode == 0) {
      RuntimeCheck(out.dtype().bits == 8 && out_scale.numel() >= M);
      prm.out_q = static_cast<uint8_t*>(out.data_ptr());
      prm.out_scale = static_cast<float*>(out_scale.data_ptr());
      LaunchKernel(dim3(MAXT), dim3(THREADS), partial.device()).enable_pdl(kUsePDL)(tp_arnq_kernel<0, kUsePDL>, prm);
    } else {
      RuntimeCheck(out.dtype().bits == 16);
      prm.out_bf16 = static_cast<__nv_bfloat16*>(out.data_ptr());
      LaunchKernel(dim3(MAXT), dim3(THREADS), partial.device()).enable_pdl(kUsePDL)(tp_arnq_kernel<1, kUsePDL>, prm);
    }
  }


  // logits [n, Vs] bf16/fp32 (row stride free), out int64 [>= n], calls int32 [MAXT] (its own counters).
  static void argmax(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView peers,
      int64_t vocab_start,
      int64_t rank,
      int64_t group,
      int64_t cand,
      int64_t cand_src) {
    using namespace host;
    using namespace dflash_tp;
    const int rows = static_cast<int>(logits.size(0));
    const int n = group > 1 ? rows / static_cast<int>(group) * static_cast<int>(group - 1) : rows;
    RuntimeCheck(group <= 1 || rows % group == 0);
    RuntimeCheck(n >= 1 && n <= MAXT && logits.stride(1) == 1 && out.numel() >= n && calls.numel() >= MAXT);
    ArgmaxParams prm{};
    prm.logits = logits.data_ptr();
    prm.ld = logits.stride(0);
    prm.Vs = static_cast<int>(logits.size(1));
    prm.vocab_start = static_cast<int>(vocab_start);
    prm.out = static_cast<int64_t*>(out.data_ptr());
    prm.calls = static_cast<int*>(calls.data_ptr());
    const int64_t* pp = static_cast<const int64_t*>(peers.data_ptr());
    for (int i = 0; i < R; ++i) prm.peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
    prm.rank = static_cast<int>(rank);
    prm.n = n;
    prm.group = static_cast<int>(group);
    RuntimeCheck(cand == 0 || (group > 1 && cand_src != 0));
    prm.cand = reinterpret_cast<int64_t*>(cand);
    prm.cand_src = reinterpret_cast<const int64_t*>(cand_src);
    if (logits.dtype().bits == 16) {
      LaunchKernel(dim3(MAXT), dim3(THREADS), logits.device()).enable_pdl(kUsePDL)(argmax_ag_kernel<__nv_bfloat16, kUsePDL>, prm);
    } else {
      RuntimeCheck(logits.dtype().bits == 32);
      LaunchKernel(dim3(MAXT), dim3(THREADS), logits.device()).enable_pdl(kUsePDL)(argmax_ag_kernel<float, kUsePDL>, prm);
    }
  }


  // ids [M] int32/int64, weight [rows, H] bf16 shard, out [M, H] bf16, ranges int64 [2 * R] (host): starts then ends.
  static void embed(
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView peers,
      tvm::ffi::TensorView ranges,
      int64_t rank,
      int64_t local) {
    using namespace host;
    using namespace dflash_tp;
    const int M = static_cast<int>(ids.size(0));
    RuntimeCheck(M >= 1 && M <= MAXT && weight.size(1) == H && weight.stride(0) == H && out.size(1) == H);
    RuntimeCheck(out.size(0) == 1 || out.stride(0) == H);
    EmbedParams prm{};
    prm.local = (local & 1) != 0 ? 1 : 0;
    prm.early = (local & 2) != 0 ? 1 : 0;
    prm.ids = ids.data_ptr();
    prm.ids64 = ids.dtype().bits == 64 ? 1 : 0;
    prm.weight = static_cast<const __nv_bfloat16*>(weight.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.calls = static_cast<int*>(calls.data_ptr());
    const int64_t* pp = static_cast<const int64_t*>(peers.data_ptr());
    const int64_t* rg = static_cast<const int64_t*>(ranges.data_ptr());
    for (int i = 0; i < R; ++i) {
      prm.peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
      prm.start[i] = rg[i];
      prm.end[i] = rg[R + i];
    }
    prm.rank = static_cast<int>(rank);
    prm.M = M;
    LaunchKernel(dim3(MAXT), dim3(THREADS), ids.device()).enable_pdl(kUsePDL)(embed_ag_kernel<kUsePDL>, prm);
  }



  // embed + RMSNorm + group-128 quant: ids [M], weight shard, out [M, H] bf16 (embedding rows), out_q [>= M, H] e4m3,
  // out_scale [H / 128, ld >= M] fp32, gamma [H]; shares the embed plane + counters (calls = ecalls)
  static void embed_nq(
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_scale,
      tvm::ffi::TensorView gamma,
      double eps,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView peers,
      tvm::ffi::TensorView ranges,
      int64_t rank,
      int64_t local) {
    using namespace host;
    using namespace dflash_tp;
    const int M = static_cast<int>(ids.size(0));
    RuntimeCheck(M >= 1 && M <= MAXT && weight.size(1) == H && weight.stride(0) == H && out.size(1) == H);
    RuntimeCheck(out.size(0) == 1 || out.stride(0) == H);
    RuntimeCheck(out_q.size(0) >= M && out_q.size(1) == H && (out_q.size(0) == 1 || out_q.stride(0) == H));
    RuntimeCheck(out_scale.ndim() == 2 && out_scale.size(0) == H / 128 && out_scale.stride(1) == 1 && out_scale.stride(0) >= M);
    RuntimeCheck(gamma.numel() == H);
    EmbedNQParams prm{};
    prm.e.local = (local & 1) != 0 ? 1 : 0;
    prm.e.early = (local & 2) != 0 ? 1 : 0;
    prm.e.ids = ids.data_ptr();
    prm.e.ids64 = ids.dtype().bits == 64 ? 1 : 0;
    prm.e.weight = static_cast<const __nv_bfloat16*>(weight.data_ptr());
    prm.e.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.e.calls = static_cast<int*>(calls.data_ptr());
    const int64_t* pp = static_cast<const int64_t*>(peers.data_ptr());
    const int64_t* rg = static_cast<const int64_t*>(ranges.data_ptr());
    for (int i = 0; i < R; ++i) {
      prm.e.peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
      prm.e.start[i] = rg[i];
      prm.e.end[i] = rg[R + i];
    }
    prm.e.rank = static_cast<int>(rank);
    prm.e.M = M;
    prm.gamma = static_cast<const __nv_bfloat16*>(gamma.data_ptr());
    prm.out_q = static_cast<uint8_t*>(out_q.data_ptr());
    prm.out_scale = static_cast<float*>(out_scale.data_ptr());
    prm.sx_ld = static_cast<int>(out_scale.stride(0));
    prm.eps = static_cast<float>(eps);
    LaunchKernel(dim3(MAXT), dim3(THREADS), ids.device()).enable_pdl(kUsePDL)(embed_nq_kernel<kUsePDL>, prm);
  }

  // in [M, SL] bf16 (this rank's columns of a column-parallel output), gamma [H], out [M, H] bf16 (normalized)
  static void ag_norm(
      tvm::ffi::TensorView in,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView peers,
      double eps,
      int64_t rank) {
    using namespace host;
    using namespace dflash_tp;
    const int M = static_cast<int>(in.size(0));
    RuntimeCheck(M >= 1 && M <= MAXT && in.size(1) == SL && (M == 1 || in.stride(0) == SL) && in.stride(1) == 1);
    RuntimeCheck(out.size(1) == H && (out.size(0) == 1 || out.stride(0) == H) && gamma.numel() == H);
    Params prm{};
    prm.partial = static_cast<const __nv_bfloat16*>(in.data_ptr());
    prm.gamma = static_cast<const __nv_bfloat16*>(gamma.data_ptr());
    prm.out_bf16 = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.calls = static_cast<int*>(calls.data_ptr());
    const int64_t* pp = static_cast<const int64_t*>(peers.data_ptr());
    for (int i = 0; i < R; ++i) prm.peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
    prm.eps = static_cast<float>(eps);
    prm.rank = static_cast<int>(rank);
    prm.M = M;
    LaunchKernel(dim3(MAXT), dim3(THREADS), in.device()).enable_pdl(kUsePDL)(ag_norm_kernel<kUsePDL>, prm);
  }

  static void silu_mul_quant(tvm::ffi::TensorView in, tvm::ffi::TensorView out, tvm::ffi::TensorView scale) {
    using namespace host;
    using namespace dflash_tp;
    constexpr int I = 2048;
    const int M = static_cast<int>(in.size(0));
    RuntimeCheck(in.size(1) == 2 * I && in.stride(1) == 1 && out.size(1) == I && out.stride(0) == I);
    LaunchKernel(dim3(M), dim3(THREADS), in.device())
        .enable_pdl(kUsePDL)(silu_mul_quant_kernel<I, kUsePDL>,
                             static_cast<const __nv_bfloat16*>(in.data_ptr()),
                             static_cast<uint8_t*>(out.data_ptr()),
                             static_cast<float*>(scale.data_ptr()),
                             in.stride(0));
  }

  static void rmsnorm_quant(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView scale,
      double eps,
      int64_t rank) {
    using namespace host;
    using namespace dflash_tp;
    const int M = static_cast<int>(x.size(0));
    RuntimeCheck(x.size(1) == H && (M == 1 || x.stride(0) == H) && (M == 1 || out.stride(0) == H));
    LaunchKernel(dim3(M), dim3(THREADS), x.device())
        .enable_pdl(kUsePDL)(rmsnorm_quant_kernel<kUsePDL>,
                             static_cast<const __nv_bfloat16*>(x.data_ptr()),
                             static_cast<const __nv_bfloat16*>(gamma.data_ptr()),
                             static_cast<float>(eps),
                             static_cast<int>(rank),
                             static_cast<uint8_t*>(out.data_ptr()),
                             static_cast<float*>(scale.data_ptr()));
  }
};

}  // namespace sglang
