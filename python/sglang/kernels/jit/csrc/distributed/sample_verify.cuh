// Vocab-parallel DFlash verify for sampled requests (temperature + top-p + rejection sampling), TP8, <= 64 rows.
//
// Replaces, for a chain verify block of G rows per request (row (b, j) = target logits after block position j):
//   all-gather of the bf16 logits shard -> fp32 copy -> logits / T -> softmax -> top-p renorm (flashinfer pivot
//   search) -> dense draft_probs -> tree_speculative_sampling_target_only (accept candidate j+1 at row j iff
//   coin[b, j] < p_target(candidate), then sample the bonus from relu(p_target - p_rejected) at the first rejected
//   row) -> rank-0 broadcasts of accept_len / bonus
// with one kernel on each rank's own [n, Vs] logits shard. One CTA per row; the shard row stays in shared memory.
//
// Exact top-p set from per-rank statistics: q_i is a non-decreasing function of the bf16 logit x_i, so the kept set
// {q >= v*} (v* = largest value with sum_{q >= v*} q >= top_p, ties included; flashinfer's pivot search converges to
// it) is {key(x_i) >= key*} for a 16-bit monotonic key of the bf16 logit. key* is found with two radix rounds of
// 256 bins: round 1 exchanges (local max, local sum exp, mass histogram over the high key byte, the candidate's
// logit if owned, the local first argmax); round 2 the mass histogram over the low byte inside the crossing bin.
// Masses are exp(x/T - m_rank) per rank and rescaled by exp(m_rank - M) after the exchange (online-softmax merge).
// Every element with the same bf16 value has the same mass, so one pass over the row builds a count table over all
// 65536 keys (packed 16-bit counters, native 32-bit shared atomics) and both radix rounds take their masses as
// count x exp(x(key)/T - m) per distinct key (no float atomics, deterministic, no second pass over the row).
// Every rank reduces the same 8 messages in rank order, so the accept decisions are identical on all ranks without a
// broadcast. Round 3: the rank whose shard holds the sampled bonus token (prefix of per-rank kept mass in vocab
// order) scans its row and pushes the token id to every rank.
//
// Candidate-selector drafts (sel_ids != nullptr): a sampled request's draft token at row j was drawn from a sparse q
// over the K candidate ids of that position. Row j accepts iff coin * q(x) < p(x) (q(x) in (0, 1]) and the bonus at the
// first rejected row is drawn from max(p - q, 0); only the K candidates differ from p, so each owner sends their bf16
// logits in round 1 and every rank subtracts min(p_c, q_c) from the owner's kept mass. Without sel_ids the draft is
// deterministic (q = one-hot) and the kernel runs the path above unchanged.
//
// Output: predict [n] int64 in the greedy-verify layout (predict[b, j] = candidate j+1 for accepted rows, the bonus
// at the first rejected row, -1 after it), consumed by the unchanged greedy accept kernel. Greedy requests (mode 0)
// get the vocab-parallel argmax (torch.argmax order: NaN max, -0 == +0, lowest index on ties) after round 1 only.
// skip_greedy: when no request of the batch samples, every CTA returns at once (no output, no counter update); the
// worker captures the kernel next to the in-graph greedy argmax, which then serves the batch.
//
// Uniforms: Philox4x32-10(seed, (call, request, k)) with the per-row call counter (identical on every rank), or explicit
// coin buffers (level coins [b, j] at b * coin_ld + j, final coin [b]), e.g. the torch.rand draws of the reference path. Transport as in dflash_tp_fused: pushes into the peers' CUDA-IPC receive buffers,
// every 8-byte half {payload, tag} with tag = (call + 1) | 2^30, two phases alternated by the per-row call counter.
// The buffer addresses come from a device table, so a captured graph does not fix them (the worker allocates the
// receive buffers after the KV cache and graph capture; until then it stages every batch greedy).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <climits>
#include <cstdint>

namespace sglang {
namespace dflash_sv {

constexpr int R = 8;          // TP ranks
constexpr int MAXN = 64;      // rows (requests x block)
constexpr int MAXG = 16;      // block size
constexpr int THREADS = 512;
constexpr int NW = THREADS / 32;
constexpr int NB = 256;       // radix bins (one key byte)
constexpr int NKEY = 65536;   // bf16 keys; count table = NKEY / 2 words, word w = 16-bit counts of keys w and w + 32768
                              // (a negative and a positive value: adjacent keys never share a word -> fewer atomic conflicts)
constexpr int HDR_V = 3;      // 16-B header vectors of a round-1 message
constexpr int HIST_V = NB / 2;  // 16-B vectors of a histogram {h, tag, h, tag}
constexpr int R1_V = HDR_V + HIST_V;
constexpr int SEL_K = 16;     // selector candidates per position
constexpr int SEL_V = SEL_K / 4;  // 16-B vectors carrying the K owned bf16 logits (two per 32-bit payload)
constexpr int64_t R1_SLOT = (R1_V + SEL_V) * 16;
constexpr int64_t R2_SLOT = HIST_V * 16;
constexpr int64_t R3_SLOT = 16;
constexpr int64_t R1_OFF = 0;
constexpr int64_t R1_BYTES = 2LL * MAXN * R * R1_SLOT;
constexpr int64_t R2_OFF = R1_OFF + R1_BYTES;
constexpr int64_t R2_BYTES = 2LL * MAXN * R * R2_SLOT;
constexpr int64_t R3_OFF = R2_OFF + R2_BYTES;
constexpr int64_t R3_BYTES = 2LL * MAXN * R3_SLOT;
constexpr int64_t BUF_BYTES = R3_OFF + R3_BYTES;
constexpr int DBG = 32;       // floats per row in the optional debug output

struct SVParams {
  const __nv_bfloat16* logits;  // [n, Vs] this rank's shard, row stride ld (elements)
  int64_t ld;
  int Vs;
  int vocab_start;
  int vocab;                    // full vocabulary (degenerate-sample fallback id vocab - 1)
  const int64_t* cand;          // [n] verify input ids, block-major: cand[b*G] = bonus, cand[b*G + 1 + j] = draft j
  const float* temp;            // [>= n/G] temperature (mode 1)
  const float* top_p;           // [>= n/G] top-p (mode 1; >= 1 -> no top-p)
  const int* mode;              // [>= n/G] 0 greedy argmax, 1 temperature/top-p sampling
  int64_t* out;                 // [n] predict
  int* calls;                   // [MAXN] per-row call counters
  unsigned long long* flags;    // [MAXN] per-row accept flags {tag << 32 | accept} (this GPU)
  const unsigned long long* peers;  // [R] device table: every rank's receive buffer (this rank's own at [rank])
  int rank;
  int n;
  int G;
  unsigned long long seed;
  const float* coins;           // optional level coins, row b at b * coin_ld (j < G - 1); nullptr -> Philox
  int coin_ld;
  const float* coins_final;     // final coin of request b (with coins)
  float* dbg;                   // optional [n, DBG]
  unsigned long long* stamps;   // optional [MAXN, 16] %globaltimer per phase (diagnostics)
  int skip_greedy;              // 1: return at once when every request is greedy (mode 0)
  const int64_t* sel_ids;       // optional [n/G, G - 1, SEL_K] candidate ids of the selector's sparse q (contiguous)
  const float* sel_q;           // [n/G, G - 1, SEL_K] q over sel_ids; read for sampled requests only
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v));
  return v;
}
#define SV_STAMP(k) do { if (p.stamps != nullptr && threadIdx.x == 0) p.stamps[blockIdx.x * 16 + (k)] = gtime(); } while (0)

__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld_vol_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_vol_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_vol_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 poll_tag(const void* p, uint32_t tag) {
  uint4 d;
  do {
    d = ld_vol_v4(p);
  } while (d.y != tag || d.w != tag);
  return d;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf16_bits_to_f(uint32_t raw) { return __uint_as_float(raw << 16); }

// torch.argmax order on fp32 values: NaN is the maximum, -0.0 == +0.0
__device__ __forceinline__ int ordered_key(float v) {
  v = (v == 0.f) ? 0.f : v;
  const int bits = __float_as_int(v);
  int key = bits >= 0 ? bits : (bits ^ 0x7FFFFFFF);
  return (v != v) ? 0x7FFFFFFF : key;
}
__device__ __forceinline__ float key_to_float(int key) {
  return __int_as_float(key >= 0 ? key : (key ^ 0x7FFFFFFF));
}
// monotonic 16-bit key of a bf16 bit pattern (-0.0 -> +0.0)
__device__ __forceinline__ uint32_t key16(uint32_t raw) {
  raw &= 0xFFFFu;
  raw = (raw == 0x8000u) ? 0u : raw;
  return (raw & 0x8000u) ? ((~raw) & 0xFFFFu) : (raw | 0x8000u);
}
// inverse of key16 (as a float value)
__device__ __forceinline__ float key16_value(uint32_t key) {
  const uint32_t raw = (key & 0x8000u) ? (key & 0x7FFFu) : ((~key) & 0xFFFFu);
  return __uint_as_float(raw << 16);
}

__device__ __forceinline__ uint4 philox4x32(uint32_t k0, uint32_t k1, uint4 c) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
    const uint32_t lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
    c = make_uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}
// x / T exactly as torch's logits / temperature (IEEE division); T == 1 is the identity
__device__ __forceinline__ float scaled(float x, float T, bool t1) { return t1 ? x : __fdiv_rn(x, T); }

__device__ __forceinline__ float u01(uint32_t x) { return static_cast<float>(x >> 8) * (1.0f / 16777216.0f); }

// coin k of request b (k < G: level coins, k == MAXG: the final-sampling coin)
__device__ __forceinline__ float coin(const SVParams& p, int call, int b, int k) {
  if (p.coins != nullptr) return k < MAXG ? p.coins[b * p.coin_ld + k] : p.coins_final[b];
  const uint4 r = philox4x32(static_cast<uint32_t>(p.seed), static_cast<uint32_t>(p.seed >> 32),
                             make_uint4(static_cast<uint32_t>(call), static_cast<uint32_t>(b),
                                        static_cast<uint32_t>(k >> 2), 0x5A3C96E1u));
  const int c = k & 3;
  return u01(c == 0 ? r.x : c == 1 ? r.y : c == 2 ? r.z : r.w);
}

// block-wide sum in a fixed order (deterministic); every thread gets the result
__device__ __forceinline__ float block_sum(float v, float* s_red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int w = 0; w < NW; ++w) t += s_red[w];
  __syncthreads();
  return t;
}

// warp 0: first bin from the top (k = 255 .. 0) where base + sum_{bins >= k} H >= P.
// Returns k (0 if never reached) and the mass strictly above k (in *above) and up to and including k (in *upto).
__device__ __forceinline__ void top_scan(const float* H, float base, float P, int* s_k, float* s_above, float* s_upto) {
  const int lane = threadIdx.x & 31;
  float v[8];
  float ls = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = H[NB - 1 - (lane * 8 + i)];
    ls += v[i];
  }
  float incl = ls;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  const unsigned hit = __ballot_sync(0xffffffffu, base + incl >= P);
  const int f = hit ? (__ffs(hit) - 1) : 31;
  if (lane == f) {
    float cum = base + incl - ls;
    int k = 0;
    float above = cum, upto = cum;
    bool found = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (!found) {
        above = cum;
        cum += v[i];
        if (cum >= P) {
          found = true;
          k = NB - 1 - (lane * 8 + i);
          upto = cum;
        }
      }
    }
    if (!found) {  // rounding (lane sum vs sequential sum) or never reached: the lane's lowest bin (lane 31 -> bin 0)
      k = NB - 1 - (lane * 8 + 7);
      above = cum - v[7];
      upto = cum;
    }
    *s_k = k;
    *s_above = above;
    *s_upto = upto;
  }
}

template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS) sample_verify_kernel(const __grid_constant__ SVParams p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ float s_red[NW];
  __shared__ int s_wk[NW], s_wi[NW];
  __shared__ uint32_t s_hdr[R][6];   // akey, aidx, m, Z, xc, -
  __shared__ float s_scale[R], s_skr[R], s_w[R];
  __shared__ int s_i[8];
  __shared__ float s_f[8];
  __shared__ int s_acc[MAXG];
  __shared__ float s_tsum[THREADS];
  __shared__ int s_lastv[THREADS];
  __shared__ uint32_t s_selw[R][SEL_K / 2];  // received candidate logits, two bf16 per word, by source rank
  __shared__ uint32_t s_selraw[SEL_K];       // this rank's owned candidate logits (0 if not owned)
  __shared__ uint32_t s_selmask;             // bit k: this rank owns candidate k
  __shared__ int s_seli[SEL_K];              // candidate ids (-1: out of vocabulary or a repeated id)
  __shared__ float s_selq[SEL_K];            // q, non-probabilities as 0
  __shared__ float s_sele[SEL_K];            // kept mass relative to M (0 outside the top-p set)
  __shared__ int s_selown[SEL_K];            // owner rank of a kept candidate, else -1

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  device::PDLWaitPrimary<kUsePDL>();
  if (t >= p.n) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  if (p.skip_greedy) {  // all-greedy batch: the in-graph argmax serves it (same staged modes on every rank)
    int any = 0;
    for (int i = 0; i < p.n / p.G; ++i) any |= p.mode[i];
    if (any == 0) {
      device::PDLTriggerSecondary<kUsePDL>();
      return;
    }
  }
  SV_STAMP(0);
  const int G = p.G;
  const int b = t / G, j = t - b * G;
  const bool samp = p.mode[b] != 0;
  const bool sel = samp && p.sel_ids != nullptr && j < G - 1;
  const float T = p.temp[b];
  const bool t1 = (T == 1.0f);
  const float topp = p.top_p[b];
  const int call = p.calls[t];
  const int phase = call & 1;
  const uint32_t tag = static_cast<uint32_t>(call + 1) | 0x40000000u;
  const int Vs = p.Vs;
  const int rowbytes = ((Vs * 2 + 15) / 16) * 16;
  uint16_t* srow = reinterpret_cast<uint16_t*>(smem);
  uint32_t* cnt = reinterpret_cast<uint32_t*>(smem + rowbytes);  // [NKEY / 2] packed 16-bit key counts
  float* rh = reinterpret_cast<float*>(cnt + NKEY / 2);          // [R][NB] round-1 histograms by source rank
  float* rh2 = rh + R * NB;                                       // [R][NB] round-2 histograms
  float* H = rh2 + R * NB;                                        // [NB]
  __shared__ uint8_t* s_peer[R];
  if (tid < R) s_peer[tid] = reinterpret_cast<uint8_t*>(p.peers[tid]);  // read after the argmax reduce barrier

  // ---- 1) row -> smem (sampling), local first argmax over the shard ----
  const uint4* row4 = reinterpret_cast<const uint4*>(p.logits + static_cast<int64_t>(t) * p.ld);
  const int nv = Vs >> 3;
  int bk = INT_MIN, bi = INT_MAX;
  {
    constexpr int UNR = 6;
    uint4 wv[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int c8 = tid + u * THREADS;
      wv[u] = c8 < nv ? __ldcg(row4 + c8) : make_uint4(0u, 0u, 0u, 0u);
    }
    if (samp) {
      for (int i = tid; i < NKEY / 8; i += THREADS) reinterpret_cast<uint4*>(cnt)[i] = make_uint4(0u, 0u, 0u, 0u);
      __syncthreads();
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int c8 = tid + u * THREADS;
      if (c8 < nv) {
        const uint32_t ws[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
        if (samp) {
          reinterpret_cast<uint4*>(srow)[c8] = wv[u];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t ka = key16(ws[q]), kb = key16(ws[q] >> 16);
            atomicAdd(cnt + (ka & 0x7FFFu), 1u << ((ka >> 15) * 16u));
            atomicAdd(cnt + (kb & 0x7FFFu), 1u << ((kb >> 15) * 16u));
          }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k0 = ordered_key(bf_lo(ws[q])), k1 = ordered_key(bf_hi(ws[q]));
          if (k0 > bk) { bk = k0; bi = c8 * 8 + 2 * q; }
          if (k1 > bk) { bk = k1; bi = c8 * 8 + 2 * q + 1; }
        }
      }
    }
    for (int c8 = tid + UNR * THREADS; c8 < nv; c8 += THREADS) {
      const uint4 w = __ldcg(row4 + c8);
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
      if (samp) {
        reinterpret_cast<uint4*>(srow)[c8] = w;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t ka = key16(ws[q]), kb = key16(ws[q] >> 16);
          atomicAdd(cnt + (ka & 0x7FFFu), 1u << ((ka >> 15) * 16u));
          atomicAdd(cnt + (kb & 0x7FFFu), 1u << ((kb >> 15) * 16u));
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int k0 = ordered_key(bf_lo(ws[q])), k1 = ordered_key(bf_hi(ws[q]));
        if (k0 > bk) { bk = k0; bi = c8 * 8 + 2 * q; }
        if (k1 > bk) { bk = k1; bi = c8 * 8 + 2 * q + 1; }
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const int ok = __shfl_xor_sync(0xffffffffu, bk, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
  }
  if (lane == 0) { s_wk[warp] = bk; s_wi[warp] = bi; }
  __syncthreads();
  uint8_t* const mybuf = s_peer[p.rank];
  {
    int k = s_wk[0], i = s_wi[0];
#pragma unroll
    for (int w = 1; w < NW; ++w)
      if (s_wk[w] > k || (s_wk[w] == k && s_wi[w] < i)) { k = s_wk[w]; i = s_wi[w]; }
    bk = k;
    bi = i;
  }
  const uint32_t akey = static_cast<uint32_t>(bk), aidx = static_cast<uint32_t>(bi + p.vocab_start);
  SV_STAMP(1);

  // ---- 2) sampling rows: local mass statistics (masses exp(x/T - m_loc)) ----
  float m_loc = 0.f, z_loc = 0.f;
  uint32_t xc_info = 0u;
  int cand_id = -1;
  if (samp) {
    m_loc = scaled(key_to_float(bk), T, t1);
    // masses per high-byte bin from the count table: iteration k, thread tid reads words 4 (tid + 512 k) .. + 3, i.e. warp
    // w covers 128 consecutive low keys (half of bin (w + 16 k) / 2) and the same half of bin + 128 in the high counts;
    // per bin the two warps' partials are added in a fixed order (deterministic)
    static_assert(THREADS == 512 && NKEY / 2 == THREADS * 4 * 16, "key ownership");
    float* bpart = rh2;  // [NB][2] scratch (round-2 buffers are not used yet)
    const uint4* cnt4 = reinterpret_cast<const uint4*>(cnt);
#pragma unroll 2
    for (int k = 0; k < 16; ++k) {
      const uint4 w4 = cnt4[tid + THREADS * k];
      float mlo = 0.f, mhi = 0.f;
      if (__any_sync(0xffffffffu, (w4.x | w4.y | w4.z | w4.w) != 0u)) {
        const uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (ws[q] == 0u) continue;
          const uint32_t key = 4u * static_cast<uint32_t>(tid + THREADS * k) + q;  // low key; high key = key + 32768
          const uint32_t c0 = ws[q] & 0xFFFFu, c1 = ws[q] >> 16;
          if (c0) mlo += static_cast<float>(c0) * expf(scaled(key16_value(key), T, t1) - m_loc);
          if (c1) mhi += static_cast<float>(c1) * expf(scaled(key16_value(key + 32768u), T, t1) - m_loc);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          mlo += __shfl_xor_sync(0xffffffffu, mlo, o);
          mhi += __shfl_xor_sync(0xffffffffu, mhi, o);
        }
      }
      if (lane == 0) {
        const int bin = (warp + 16 * k) >> 1;
        bpart[bin * 2 + (warp & 1)] = mlo;
        bpart[(bin + 128) * 2 + (warp & 1)] = mhi;
      }
    }
    __syncthreads();
    float zl = 0.f;
    if (tid < NB) {
      const float h = bpart[tid * 2] + bpart[tid * 2 + 1];
      H[tid] = h;
      zl = h;
    }
    z_loc = block_sum(zl, s_red);  // (syncs: H complete)
    if (j < G - 1) {
      const int64_t c = p.cand[static_cast<int64_t>(b) * G + j + 1];
      cand_id = (c >= 0 && c < p.vocab) ? static_cast<int>(c) : -1;
      const int64_t loc = c - p.vocab_start;
      if (loc >= 0 && loc < Vs) xc_info = 0x10000u | static_cast<uint32_t>(srow[loc]);
    }
    if (sel && tid < 32) {
      const int64_t base = (static_cast<int64_t>(b) * (G - 1) + j) * SEL_K;
      bool own = false;
      if (tid < SEL_K) {
        const int64_t c = p.sel_ids[base + tid];
        int id = (c >= 0 && c < p.vocab) ? static_cast<int>(c) : -1;
        for (int k = 0; k < tid; ++k)  // a repeated id keeps its first q (dense q holds one value per id)
          if (id >= 0 && p.sel_ids[base + k] == c) id = -1;
        const float q = p.sel_q[base + tid];
        s_seli[tid] = id;
        s_selq[tid] = (q >= 0.f && q <= 1.f) ? q : 0.f;
        const int64_t loc = c - p.vocab_start;
        own = id >= 0 && loc >= 0 && loc < Vs;
        s_selraw[tid] = own ? static_cast<uint32_t>(srow[loc]) : 0u;
      }
      const unsigned m = __ballot_sync(0xffffffffu, own);
      if (tid == 0) s_selmask = m;
    }
    __syncthreads();
    SV_STAMP(2);
  }

  // ---- 3) round 1: push {argmax, (m, Z), candidate logit, hist} to every rank; collect all 8 ----
  {
    const int nvec = samp ? (sel ? R1_V + SEL_V : R1_V) : 1;
    for (int idx = tid; idx < R * nvec; idx += THREADS) {
      const int d = idx / nvec, v = idx - d * nvec;
      uint4 val;
      if (v == 0) val = make_uint4(akey, tag, aidx, tag);
      else if (v == 1) val = make_uint4(__float_as_uint(m_loc), tag, __float_as_uint(z_loc), tag);
      else if (v == 2) val = make_uint4(xc_info, tag, sel ? s_selmask : 0u, tag);
      else if (v < R1_V)
        val = make_uint4(__float_as_uint(H[2 * (v - HDR_V)]), tag, __float_as_uint(H[2 * (v - HDR_V) + 1]), tag);
      else {
        const int k = 4 * (v - R1_V);
        val = make_uint4(s_selraw[k] | (s_selraw[k + 1] << 16), tag, s_selraw[k + 2] | (s_selraw[k + 3] << 16), tag);
      }
      st_v4(s_peer[d] + R1_OFF + ((static_cast<int64_t>(phase) * MAXN + t) * R + p.rank) * R1_SLOT + v * 16, val);
    }
    SV_STAMP(3);
    for (int idx = tid; idx < R * nvec; idx += THREADS) {
      const int s = idx / nvec, v = idx - s * nvec;
      const uint4 d = poll_tag(mybuf + R1_OFF + ((static_cast<int64_t>(phase) * MAXN + t) * R + s) * R1_SLOT + v * 16, tag);
      if (v == 0) { s_hdr[s][0] = d.x; s_hdr[s][1] = d.z; }
      else if (v == 1) { s_hdr[s][2] = d.x; s_hdr[s][3] = d.z; }
      else if (v == 2) { s_hdr[s][4] = d.x; s_hdr[s][5] = d.z; }
      else if (v < R1_V) { rh[s * NB + 2 * (v - HDR_V)] = __uint_as_float(d.x); rh[s * NB + 2 * (v - HDR_V) + 1] = __uint_as_float(d.z); }
      else { s_selw[s][2 * (v - R1_V)] = d.x; s_selw[s][2 * (v - R1_V) + 1] = d.z; }
    }
    __syncthreads();
    SV_STAMP(4);
  }

  if (!samp) {
    if (tid == 0) {
      int k = static_cast<int>(s_hdr[0][0]), i = static_cast<int>(s_hdr[0][1]);
#pragma unroll
      for (int s = 1; s < R; ++s) {
        const int ks = static_cast<int>(s_hdr[s][0]), is = static_cast<int>(s_hdr[s][1]);
        if (ks > k || (ks == k && is < i)) { k = ks; i = is; }
      }
      p.out[t] = static_cast<int64_t>(i);
      p.calls[t] = call + 1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // ---- 4) merge round 1: M, Z, global high-byte histogram ----
  float M = __uint_as_float(s_hdr[0][2]);
#pragma unroll
  for (int s = 1; s < R; ++s) M = fmaxf(M, __uint_as_float(s_hdr[s][2]));
  if (tid < R) s_scale[tid] = expf(__uint_as_float(s_hdr[tid][2]) - M);
  __syncthreads();
  float Z = 0.f;
#pragma unroll
  for (int s = 0; s < R; ++s) Z += __uint_as_float(s_hdr[s][3]) * s_scale[s];
  for (int k = tid; k < NB; k += THREADS) {
    float h = 0.f;
#pragma unroll
    for (int s = 0; s < R; ++s) h += rh[s * NB + k] * s_scale[s];
    H[k] = h;
  }
  __syncthreads();
  const bool use_top_p = topp < 1.0f;
  const float P = topp * Z;
  uint32_t keystar = 0u;
  float SK = Z;
  if (use_top_p) {
    if (warp == 0) top_scan(H, 0.f, P, &s_i[0], &s_f[0], &s_f[1]);
    __syncthreads();
    SV_STAMP(5);
    const int d1 = s_i[0];
    const float A1 = s_f[0];
    // ---- 5) round 2: low-byte histogram inside the crossing bin (count x mass per key) ----
    for (int k = tid; k < NB; k += THREADS) {
      const uint32_t key = (static_cast<uint32_t>(d1) << 8) | static_cast<uint32_t>(k);
      const uint32_t c = (cnt[key & 0x7FFFu] >> ((key >> 15) * 16u)) & 0xFFFFu;
      H[k] = c ? static_cast<float>(c) * expf(scaled(key16_value(key), T, t1) - m_loc) : 0.f;
    }
    __syncthreads();
    for (int idx = tid; idx < R * HIST_V; idx += THREADS) {
      const int d = idx / HIST_V, v = idx - d * HIST_V;
      st_v4(s_peer[d] + R2_OFF + ((static_cast<int64_t>(phase) * MAXN + t) * R + p.rank) * R2_SLOT + v * 16,
            make_uint4(__float_as_uint(H[2 * v]), tag, __float_as_uint(H[2 * v + 1]), tag));
    }
    for (int idx = tid; idx < R * HIST_V; idx += THREADS) {
      const int s = idx / HIST_V, v = idx - s * HIST_V;
      const uint4 d = poll_tag(mybuf + R2_OFF + ((static_cast<int64_t>(phase) * MAXN + t) * R + s) * R2_SLOT + v * 16, tag);
      rh2[s * NB + 2 * v] = __uint_as_float(d.x);
      rh2[s * NB + 2 * v + 1] = __uint_as_float(d.z);
    }
    __syncthreads();
    SV_STAMP(6);
    for (int k = tid; k < NB; k += THREADS) {
      float h = 0.f;
#pragma unroll
      for (int s = 0; s < R; ++s) h += rh2[s * NB + k] * s_scale[s];
      H[k] = h;
    }
    __syncthreads();
    if (warp == 0) top_scan(H, A1, P, &s_i[1], &s_f[2], &s_f[3]);
    // per-rank kept mass: bins above d1 of round 1 + bins >= d2 of round 2 (warp s sums rank s)
    __syncthreads();
    const int d2 = s_i[1];
    if (warp < R) {
      float a = 0.f;
      for (int k = lane; k < NB; k += 32) {
        if (k > d1) a += rh[warp * NB + k];
        if (k >= d2) a += rh2[warp * NB + k];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) s_skr[warp] = a * s_scale[warp];
    }
    keystar = (static_cast<uint32_t>(d1) << 8) | static_cast<uint32_t>(d2);
    SK = s_f[3];
  } else {
    if (tid < R) s_skr[tid] = __uint_as_float(s_hdr[tid][3]) * s_scale[tid];
  }
  if (sel && tid < SEL_K) {
    int own = -1;
#pragma unroll
    for (int s = R - 1; s >= 0; --s)
      if ((s_hdr[s][5] >> tid) & 1u) own = s;
    float e = 0.f;
    if (own >= 0 && s_seli[tid] >= 0) {
      const uint32_t raw = (s_selw[own][tid >> 1] >> ((tid & 1) * 16)) & 0xFFFFu;
      e = expf(scaled(bf16_bits_to_f(raw), T, t1) - M);
      if (!(key16(raw) >= keystar && e > 0.f)) { e = 0.f; own = -1; }
    } else {
      own = -1;
    }
    s_sele[tid] = e;
    s_selown[tid] = own;
  }
  __syncthreads();
  SV_STAMP(7);

  // ---- 6) accept test of this row's candidate; per-request accept length ----
  int owner = -1;
  float e_c = 0.f;
  bool in_k = false;
  if (j < G - 1) {
#pragma unroll
    for (int s = R - 1; s >= 0; --s)
      if (s_hdr[s][4] & 0x10000u) owner = s;
    if (owner >= 0) {
      const uint32_t raw = s_hdr[owner][4] & 0xFFFFu;
      e_c = expf(scaled(bf16_bits_to_f(raw), T, t1) - M);
      in_k = key16(raw) >= keystar && e_c > 0.f;
    }
  }
  const float cj = (j < G - 1) ? coin(p, call, b, j) : 1.f;
  float tp = in_k ? __fdiv_rn(e_c, SK) : 0.f;
  bool acc = (j < G - 1) && tp > 0.f && cj < tp;
  if (sel) {  // the drafted token's q; an id outside the candidates has q = 0 and is rejected
    const int64_t x = p.cand[static_cast<int64_t>(b) * G + j + 1];
    float qx = 0.f, ex = 0.f;
    for (int k = 0; k < SEL_K; ++k)
      if (s_seli[k] >= 0 && s_seli[k] == x) { qx = s_selq[k]; ex = s_sele[k]; break; }
    tp = ex > 0.f ? __fdiv_rn(ex, SK) : 0.f;
    acc = qx > 0.f && qx <= 1.f && cj * qx < tp;
  }
  if (tid == 0) st_vol_u64(p.flags + t, (static_cast<unsigned long long>(tag) << 32) | (acc ? 1ull : 0ull));
  if (tid < G) {
    unsigned long long f;
    do {
      f = ld_vol_u64(p.flags + b * G + tid);
    } while (static_cast<uint32_t>(f >> 32) != tag);
    s_acc[tid] = static_cast<int>(f & 1ull);
  }
  __syncthreads();
  SV_STAMP(8);
  int a = G - 1;
  for (int i = 0; i < G; ++i)
    if (!s_acc[i]) { a = i; break; }

  if (p.dbg != nullptr && tid == 0) {
    float* dd = p.dbg + static_cast<int64_t>(t) * DBG;
    dd[0] = M; dd[1] = Z; dd[2] = SK; dd[3] = static_cast<float>(keystar); dd[4] = tp; dd[5] = acc ? 1.f : 0.f;
    dd[6] = static_cast<float>(a); dd[7] = cj; dd[8] = e_c; dd[9] = static_cast<float>(owner);
    for (int s = 0; s < R; ++s) {
      dd[16 + s] = __uint_as_float(s_hdr[s][2]);  // m_r
      dd[24 + s] = __uint_as_float(s_hdr[s][3]);  // Z_r (relative to m_r)
    }
  }
  if (j != a) {
    if (tid == 0) {
      p.out[t] = j < a ? p.cand[static_cast<int64_t>(b) * G + j + 1] : -1;
      p.calls[t] = call + 1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // ---- 7) bonus: sample from the kept mass minus the rejected candidate; round 3 announces the token ----
  const float rejmass = (j < G - 1 && in_k) ? e_c : 0.f;
  if (tid == 0) {
    float sub[R];
#pragma unroll
    for (int s = 0; s < R; ++s) sub[s] = (!sel && s == owner) ? rejmass : 0.f;
    if (sel) {  // residual max(p - q, 0): the owner's kept mass loses min(p_c, q_c) per candidate (mass units)
      const int64_t x = p.cand[static_cast<int64_t>(b) * G + j + 1];
      for (int k = 0; k < SEL_K; ++k)  // the rejected token had p < q, so it leaves no residual
        if (s_selown[k] >= 0) sub[s_selown[k]] += s_seli[k] == x ? s_sele[k] : fminf(s_sele[k], s_selq[k] * SK);
    }
    float total = 0.f;
#pragma unroll
    for (int s = 0; s < R; ++s) {
      const float w = fmaxf(s_skr[s] - sub[s], 0.f);
      s_w[s] = w;
      total += w;
    }
    const float u = coin(p, call, b, MAXG) * total;
    int rs = -1;
    float pre = 0.f, excl = 0.f;
    for (int s = 0; s < R; ++s) {
      if (rs < 0 && pre + s_w[s] > u) { rs = s; excl = pre; }
      pre += s_w[s];
    }
    if (rs < 0) {  // rounding: the last rank with mass
      for (int s = R - 1; s >= 0; --s)
        if (s_w[s] > 0.f) { rs = s; break; }
      excl = 0.f;
      for (int s = 0; s < rs; ++s) excl += s_w[s];
    }
    s_i[2] = total > 0.f ? rs : -1;
    s_f[4] = u - excl;
    if (p.dbg != nullptr) {
      float* dd = p.dbg + static_cast<int64_t>(t) * DBG;
      dd[10] = total; dd[11] = u; dd[12] = static_cast<float>(rs);
    }
  }
  __syncthreads();
  const int rs = s_i[2];
  if (rs < 0) {  // no mass left (reference: sampled id d - 1)
    if (tid == 0) {
      p.out[t] = static_cast<int64_t>(p.vocab - 1);
      p.calls[t] = call + 1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  SV_STAMP(9);
  if (rs == p.rank) {
    const float ul = s_f[4];
    const float sc = s_scale[p.rank];
    const int rej_loc = (j < G - 1 && owner == p.rank) ? static_cast<int>(p.cand[static_cast<int64_t>(b) * G + j + 1] - p.vocab_start) : -1;
    // thread-contiguous chunks of C (a multiple of 8: 16-B row loads, 2 x 16-B mass stores per 8 elements)
    const int C = (((Vs + THREADS - 1) / THREADS) + 7) & ~7;
    const int i0 = min(tid * C, Vs), i1 = min(i0 + C, Vs);
    float* wbuf = reinterpret_cast<float*>(cnt);  // [Vs] kept masses (the count table is no longer needed)
    float ts = 0.f;
    int lastv = -1;
    if (sel) {
      for (int v = i0; v < i1; v += 8) {
        const uint4 r4 = *reinterpret_cast<const uint4*>(srow + v);
        const uint32_t rw[4] = {r4.x, r4.y, r4.z, r4.w};
        float w[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t raw = (q & 1) ? (rw[q >> 1] >> 16) : (rw[q >> 1] & 0xFFFFu);
          w[q] = key16(raw) >= keystar ? expf(scaled(bf16_bits_to_f(raw), T, t1) - m_loc) * sc : 0.f;
        }
        *reinterpret_cast<float4*>(wbuf + v) = make_float4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<float4*>(wbuf + v + 4) = make_float4(w[4], w[5], w[6], w[7]);
      }
      __syncthreads();
      if (tid < SEL_K && s_selown[tid] == p.rank) {
        const int loc = s_seli[tid] - p.vocab_start;
        const bool rejected = s_seli[tid] == p.cand[static_cast<int64_t>(b) * G + j + 1];
        wbuf[loc] = rejected ? 0.f : fmaxf(wbuf[loc] - s_selq[tid] * SK, 0.f);
      }
      __syncthreads();
      for (int v = i0; v < i1; ++v)
        if (wbuf[v] > 0.f) { ts += wbuf[v]; lastv = v; }
    }
    for (int v = i0; v < i1 && !sel; v += 8) {
      const uint4 r4 = *reinterpret_cast<const uint4*>(srow + v);
      const uint32_t rw[4] = {r4.x, r4.y, r4.z, r4.w};
      float w[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t raw = (q & 1) ? (rw[q >> 1] >> 16) : (rw[q >> 1] & 0xFFFFu);
        w[q] = 0.f;
        if (key16(raw) >= keystar && v + q != rej_loc) w[q] = expf(scaled(bf16_bits_to_f(raw), T, t1) - m_loc) * sc;
      }
      *reinterpret_cast<float4*>(wbuf + v) = make_float4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<float4*>(wbuf + v + 4) = make_float4(w[4], w[5], w[6], w[7]);
#pragma unroll
      for (int q = 0; q < 8; ++q)
        if (w[q] > 0.f) { ts += w[q]; lastv = v + q; }
    }
    // exclusive block scan of the per-thread sums (thread order == vocab order): warp scans + warp totals
    float incl = ts;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    if (lane == 31) s_red[warp] = incl;
    if (tid == 0) { s_i[3] = INT_MAX; s_i[4] = -1; }
    __syncthreads();
    float wbase = 0.f;
    for (int w = 0; w < warp; ++w) wbase += s_red[w];
    const float ex = wbase + incl - ts;
    s_tsum[tid] = ex;
    s_lastv[tid] = lastv;
    if (ex + ts > ul && lastv >= 0) atomicMin(&s_i[3], tid);
    if (lastv >= 0) atomicMax(&s_i[4], lastv);
    __syncthreads();
    if (warp == 0) {
      const int tt = s_i[3];
      int pick = s_i[4];  // rounding (u beyond the shard's mass): last valid index of the shard
      if (tt != INT_MAX) {
        // warp-cooperative inverse CDF inside thread tt's chunk (C <= 64 elements: two per lane)
        const int j0 = min(tt * C, Vs), j1 = min(j0 + C, Vs);
        float cum = s_tsum[tt];
        pick = s_lastv[tt];
        bool found = false;
        for (int base = j0; base < j1 && !found; base += 32) {
          const int i = base + lane;
          const float w = i < j1 ? wbuf[i] : 0.f;
          float sc_ = w;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, sc_, o);
            if (lane >= o) sc_ += y;
          }
          const unsigned hit = __ballot_sync(0xffffffffu, w > 0.f && cum + sc_ > ul);
          if (hit) {
            pick = base + __ffs(hit) - 1;
            found = true;
          }
          cum += __shfl_sync(0xffffffffu, sc_, 31);
        }
      }
      if (lane == 0) s_i[5] = pick;
    }
    __syncthreads();
    SV_STAMP(10);
    if (tid < R) {
      const int pick = s_i[5] < 0 ? (Vs - 1) : s_i[5];
      const uint32_t tok = static_cast<uint32_t>(pick + p.vocab_start);
      st_v4(s_peer[tid] + R3_OFF + (static_cast<int64_t>(phase) * MAXN + t) * R3_SLOT, make_uint4(tok, tag, 0u, tag));
    }
  }
  if (tid == 0) {
    const uint4 d = poll_tag(mybuf + R3_OFF + (static_cast<int64_t>(phase) * MAXN + t) * R3_SLOT, tag);
    p.out[t] = static_cast<int64_t>(d.x);
    p.calls[t] = call + 1;
    if (p.dbg != nullptr) p.dbg[static_cast<int64_t>(t) * DBG + 13] = static_cast<float>(d.x);
  }
  SV_STAMP(11);
  device::PDLTriggerSecondary<kUsePDL>();
}

inline int smem_bytes(int Vs) {
  return ((Vs * 2 + 15) / 16) * 16 + NKEY * 2 + (2 * R * NB + NB) * 4;
}

}  // namespace dflash_sv

template <bool kUsePDL>
struct SampleVerify {
  // logits [n, Vs] bf16 shard (row stride free, 16-B aligned rows), cand [n] int64, temp/top_p [>= n/G] fp32,
  // mode [>= n/G] int32, out [>= n] int64, calls [MAXN] int32, flags [MAXN] int64, peers [8] int64 (device table).
  static void run(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView cand,
      tvm::ffi::TensorView temp,
      tvm::ffi::TensorView top_p,
      tvm::ffi::TensorView mode,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView calls,
      tvm::ffi::TensorView flags,
      tvm::ffi::TensorView peers,
      int64_t vocab_start,
      int64_t vocab,
      int64_t rank,
      int64_t group,
      int64_t seed,
      int64_t coins,
      int64_t coin_ld,
      int64_t coins_final,
      int64_t dbg,
      int64_t stamps,
      int64_t skip_greedy,
      int64_t sel_ids,
      int64_t sel_q) {
    using namespace host;
    using namespace dflash_sv;
    const int n = static_cast<int>(logits.size(0));
    const int Vs = static_cast<int>(logits.size(1));
    RuntimeCheck(n >= 1 && n <= MAXN && group >= 2 && group <= MAXG && n % group == 0);
    RuntimeCheck(logits.dtype().bits == 16 && logits.stride(1) == 1 && (Vs & 7) == 0 && (logits.stride(0) & 7) == 0);
    RuntimeCheck((reinterpret_cast<uintptr_t>(logits.data_ptr()) & 15) == 0);
    RuntimeCheck(cand.numel() >= n && out.numel() >= n && calls.numel() >= MAXN && flags.numel() >= MAXN);
    RuntimeCheck(temp.numel() >= n / group && top_p.numel() >= n / group && mode.numel() >= n / group);
    RuntimeCheck(peers.numel() == R);
    SVParams prm{};
    prm.logits = static_cast<const __nv_bfloat16*>(logits.data_ptr());
    prm.ld = logits.stride(0);
    prm.Vs = Vs;
    prm.vocab_start = static_cast<int>(vocab_start);
    prm.vocab = static_cast<int>(vocab);
    prm.cand = static_cast<const int64_t*>(cand.data_ptr());
    prm.temp = static_cast<const float*>(temp.data_ptr());
    prm.top_p = static_cast<const float*>(top_p.data_ptr());
    prm.mode = static_cast<const int*>(mode.data_ptr());
    prm.out = static_cast<int64_t*>(out.data_ptr());
    prm.calls = static_cast<int*>(calls.data_ptr());
    prm.flags = static_cast<unsigned long long*>(flags.data_ptr());
    prm.peers = static_cast<const unsigned long long*>(peers.data_ptr());
    prm.rank = static_cast<int>(rank);
    prm.n = n;
    prm.G = static_cast<int>(group);
    prm.seed = static_cast<unsigned long long>(seed);
    prm.coins = reinterpret_cast<const float*>(coins);
    prm.coin_ld = static_cast<int>(coin_ld);
    prm.coins_final = reinterpret_cast<const float*>(coins_final);
    RuntimeCheck((coins == 0) == (coins_final == 0));
    prm.dbg = reinterpret_cast<float*>(dbg);
    prm.stamps = reinterpret_cast<unsigned long long*>(stamps);
    prm.skip_greedy = skip_greedy != 0 ? 1 : 0;
    prm.sel_ids = reinterpret_cast<const int64_t*>(sel_ids);
    prm.sel_q = reinterpret_cast<const float*>(sel_q);
    RuntimeCheck((sel_ids == 0) == (sel_q == 0));
    const int smem = smem_bytes(Vs);
    RuntimeCheck(smem <= 220 * 1024 && Vs <= NKEY / 2);  // the final scan reuses the count table for Vs floats
    auto kern = sample_verify_kernel<kUsePDL>;
    static int attr_smem = 0;
    if (smem > attr_smem) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      attr_smem = smem;
    }
    LaunchKernel(dim3(static_cast<unsigned>(n)), dim3(THREADS), logits.device(), static_cast<std::size_t>(smem))
        .enable_pdl(kUsePDL)(kern, prm);
  }
};

}  // namespace sglang
