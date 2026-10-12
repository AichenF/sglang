// goal11 "blk": MiMo-V2 SWA target-verify attention block in ONE kernel (exact M = 32 = 4 requests x 8 verify tokens, TP8, sm_90a):
//   RoPE (NeoX, 64 of 192 dims) + KV-cache store of the 8 new tokens + sliding-window attention (window 128/129, 1 KV head, per-head sink)
//   + o_proj (bf16, K = 16 heads x 128 = 2048 per rank, N = 6144) + the TP reduce-scatter PUSH of the o_proj partial straight into the
//   fused MoE kernel's entry receive buffers (IN_RS region of every peer's prologue IPC buffer, owner-reduce layout, {6 bf16, epoch}
//   LL messages). The following fused MoE launch skips its own entry push (prepushed mode bit) and finds the messages landed.
//
// v2 grid: 128 CTAs = 64 clusters of 2 (1 CTA per SM, 8 math warps + 1 TMA producer warp). Cluster p owns o_proj columns [96p, 96p + 96);
// its CTA h (cluster rank) multiplies the K half [1024h, 1024h + 1024) (= heads 8h..8h+7), so each SM ingests 192 KB of weight + only
// 64 KB of the attention output; the two fp32 partials meet over DSMEM and CTA h finalizes columns [96p + 48h, +48).
// CTAs 0..31 first run the attention of (request c / 8, heads 2 (c % 8), +1): the swa_decode_mma kernel's code (8 warps, mma.m16n8k16,
// same reduction order -> bitwise its output) with the RoPE + KV store of rope_kv fused in front (bitwise rope_kv: bf16 fma/mul).
// Smem: a 13-stage ring, stage = {weight k-tile [96 rows][64] SW128 12 KB | attention k-tile [32 tok][64] SW128 4 KB}. The producer warp
// fills the weight halves from kernel start (L2-resident: the MoE tail prefetched them), waits for the 32 attention-ready flags (epoch,
// no reset), then TMA-loads the attention k-tiles (all 13 at once: the per-SM ingress needs depth) and the last 3 weight k-tiles.
// Attention CTAs keep stages 6..12 (their attention scratch) out until the attention is done, and issue stages 0..5 only after their
// own gathers landed (an SM's loads queue behind its outstanding TMA bytes).
// Epilogue: fp32 -> bf16 RN, 256 threads build the 32 x 8 messages; a warp writes 32 consecutive messages = whole 32-B sectors
// (the LL tearing rule: every 16-B store carries the epoch tag, and no 32-B sector is shared by two store instructions).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace blk {

constexpr int DIM = 6144, NH = 16, DQK = 192, DV = 128, KO = NH * DV, QKVW = NH * DQK + DQK + DV;   // 2048, 3392
constexpr int VT = 8, NREQ = 4, M = VT * NREQ;   // verify tokens per request, requests, rows
#ifndef BLK_KS_MAXX
#define BLK_KS_MAXX 1                         // key-split attention: 1 = exchange the row maxima first (P bitwise swa_mma's, one more DSMEM round
#endif                                        //    trip); 0 = split-softmax combine (local maxima, rescaled in the single exchange)
#ifndef BLK_KSPLIT
#define BLK_KSPLIT 1                          // 1: every (request, head pair) attention runs on a cluster PAIR, each CTA half of the key window
#endif                                        //    (64 attention CTAs; row maxima and PV partials meet over DSMEM). 0: one CTA per pair (bitwise swa_mma)
constexpr int HP = 2, NHG = NH / HP, NATT = NREQ * NHG * (BLK_KSPLIT ? 2 : 1);   // heads per attention CTA, head groups, attention CTAs
constexpr int NW = 8, NMATH = NW * 32, NTHREADS = NMATH + 32;   // 8 math warps + 1 producer warp
constexpr int CL = 2, PCOLS = 96, COLS = PCOLS / CL, NCTA = CL * (DIM / PCOLS);   // cluster pairs: 96 columns per pair, 48 finalized per CTA
constexpr int KT = 64, NKT = KO / KT / CL;    // 16 k-tiles of 64 per CTA (its K half)
constexpr int WT_BYTES = PCOLS * KT * 2;      // 12288: one SW128 weight k-tile (96 rows x 128 B)
constexpr int BT_BYTES = M * KT * 2;          // 4096: one SW128 attention k-tile (32 tokens x 128 B)
constexpr int ST_BYTES = WT_BYTES + BT_BYTES; // 16384 per stage
constexpr int NS = 13;                        // ring stages
constexpr int NS0_ATT = BLK_KSPLIT ? 8 : 6;
constexpr int FLAG_STRIDE = 32;               // attention-ready flags one per 128-B line (u32 index c * FLAG_STRIDE)
#ifndef NACC
#define NACC 4                                // independent wgmma accumulator chains per warpgroup (N = 32: one chain is latency-bound)
#endif
#ifndef BLK_WD
#define BLK_WD 3                              // weight k-tiles in flight per SM (Little: deeper queues only add latency to everyone's loads)
#endif                    // stages an attention CTA fills during its attention (6..12 hold the attention scratch)
constexpr int OFF_ATT = NS0_ATT * ST_BYTES;   // 98304
constexpr int OFF_X = NS * ST_BYTES;          // 212992: [48 rows][32 tok] fp32: the partner's partial of my final rows (DSMEM st.async)
constexpr int OFF_BAR = OFF_X + COLS * M * 4;    // 219136
constexpr int OFF_MISC = OFF_BAR + 512;
constexpr int SMEM_USED = OFF_MISC + 64;
constexpr int SMEM_ALLOC = SMEM_USED + 1024;  // + alignment slack
static_assert(SMEM_ALLOC <= 232448, "smem");
// fused MoE entry receive buffer (fused_moe_wgmma.cu, FMOE_TP_OWNER_REDUCE): IN_RS at message 0 of every rank's prologue buffer,
// [src 8][owner-local token 4][slice 8][128], message m of slice k = elements 768 k + 6 m .. + 5 of the token's partial row.
constexpr int TP_NDEV = 8, TP_KS = DIM / TP_NDEV, TP_RS_ROW = TP_KS / 6, TP_TOK_PER_RANK = M / TP_NDEV;
static_assert(TP_KS == 768 && TP_RS_ROW == 128 && COLS % 12 == 0 && TP_KS % COLS == 0 && NKT <= 2 * NS && NKT > NS, "geometry");
__device__ __forceinline__ size_t tp_rs_idx(int src, int t, int k, int m) {
    return ((size_t)((src * TP_TOK_PER_RANK + t / TP_NDEV) * TP_NDEV + k) * TP_RS_ROW + m);
}

struct Params {
    CUtensorMap tm_w;                  // o_proj weight [6144][2048] bf16: box {64, 96}, SWIZZLE_128B
    CUtensorMap tm_a;                  // attention output [32][2048] bf16: box {64, 32}, SWIZZLE_128B
    const __nv_bfloat16* qkv;          // [32][3392]
    const __nv_bfloat16* cos_sin;      // [max_pos][64]
    const void* positions;             // [32] int32 / int64
    const void* loc;                   // [32] KV write slots int32 / int64
    __nv_bfloat16* kc;                 // key cache rows (DQK elements, stride ks)
    __nv_bfloat16* vc;                 // value cache rows (DV elements, stride vs)
    long long ks, vs, kv_rows;
    const int* pt;                     // [NREQ][pt_stride] page table (page size 1)
    const int* seqlens;                // [NREQ]
    const float* sinks;                // [NH] or null
    __nv_bfloat16* attn_out;           // [32][2048] scratch
    unsigned* flags;                   // [NATT] attention-ready (epoch)
    __nv_bfloat16* partial;            // [32][6144] local o_proj partial (owned tokens; all rows with MODE_FULL_LOCAL)
    uint4* peer[TP_NDEV];              // prologue IPC buffers (IN_RS at message 0)
    const int* work;                   // fused MoE work counters: epoch = work[2] + 1
    unsigned long long* dbg;           // optional [NCTA][32] globaltimer stamps (diagnostics)
    const __nv_bfloat16* nkc;          // next SWA layer's key / value cache (same row strides) or null: the window rows are L2-prefetched
    const __nv_bfloat16* nvc;          // (evict_last) at the end of this layer's attention -- the next layer gathers the same slots
    int* slot_cache;                   // [NREQ][NPAD_MAX] key-row cache slots of this forward (same for every SWA layer): written by the first
                                       // SWA layer (MODE_SLOTS_WRITE), read by the others (MODE_SLOTS_READ: one load instead of seqlen -> pt)
    int pt_stride, my_rank, mode;
    float sm_scale;
};
constexpr int MODE_POS64 = 1, MODE_LOC64 = 2, MODE_FULL_LOCAL = 4, MODE_NO_PUSH = 8, MODE_SINK = 16, MODE_DBG_NOW = 32, MODE_DBG_NOB = 64, MODE_DBG_ATTONLY = 128, MODE_EXT_ATTN = 256, MODE_DBG_NOPOLL = 512, MODE_DBG_GARB = 1024,
              MODE_SLOTS_WRITE = 2048, MODE_SLOTS_READ = 4096, MODE_DBG_GNOMMA = 8192, MODE_DBG_GNOX = 16384, MODE_DBG_GNOW = 32768, MODE_QKV_ONLY = 65536;

constexpr float LOG2E = 1.4426950408889634f;
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned ph) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" :: "r"(sa(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 " cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], policy;\n}"
                 :: "r"(sa(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d_nohint(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(sa(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void cp16(void* dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa(dst)), "l"(src) : "memory"); }
__device__ __forceinline__ void cp_arrive_noinc(uint64_t* b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
constexpr int NDBG = 32;
#define BSTAMP(k) do { if (P.dbg) P.dbg[blockIdx.x * NDBG + (k)] = gtimer(); } while (0)
#define CSTAMP(k) do { if (P.dbg && (P.mode & MODE_DBG_ATTONLY)) { long long ck_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(ck_) :: "memory"); P.dbg[blockIdx.x * NDBG + (k)] = ck_; } } while (0)
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// exp2 / reciprocal as --use_fast_math emits them (ex2.approx.ftz / rcp.approx.ftz): the same instructions whether or not the including
// translation unit is built with fast math (goal11 layer: layer_fused.cu includes this file next to the fused MoE, which must not be)
__device__ __forceinline__ float fast_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float fast_rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// swa_decode_mma's fragment helpers (verbatim)
__device__ __forceinline__ void ldm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa(p)));
}
__device__ __forceinline__ void ldm4t(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa(p)));
}
__device__ __forceinline__ void mma16816(float* d, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) { unsigned u; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo)); return u; }
// wgmma (SS, both operands K-major SW128 in smem)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wgmma_m64n32k16(float* d, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n"
                 " wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da), "l"(db), "r"(scale_d));
}
// RS form: A (64 x 16, m16n8k16 A-fragment layout per warp: rows 16 (warp % 4) + lane / 4 (+8), k 2 (lane % 4) (+1, +8)) in registers
__device__ __forceinline__ void wgmma_rs_m64n32k16(float* d, const uint32_t* a, uint64_t db, int scale_d) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %21, 0;\n"
                 " wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ long long ld_idx(const void* p, int i, bool i64) {
    return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}
// rope_kv's bf16 rotation: lo = fma(x, c, -(y * s)), hi = fma(y, c, x * s) with one RN per bf16 op (mul.rn / fma.rn .bf16)
__device__ __forceinline__ void rope_pair(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 c, __nv_bfloat16 s, __nv_bfloat16& lo, __nv_bfloat16& hi) {
    lo = __hfma(x, c, __hneg(__hmul(y, s)));
    hi = __hfma(y, c, __hmul(x, s));
}

// ------------------------------------------------------------------------------------------------------------------ attention
// swa_mma_kernel<8, 2, WINDOW, 8> with the KV for key rows NKEY-8 .. NKEY-1 (the 8 new tokens) taken from qkv after RoPE (the
// unfused path reads the same bf16 values back from the cache rope_kv wrote) and rows outside the sequence zero-filled instead of
// gathered from arbitrary pages (their logits are -inf and their probabilities exact 0 either way).
template <int WINDOW>
__device__ __forceinline__ void attention(const Params& P, uint8_t* smem, int b, int hg, uint64_t* att_in) {
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, NPAD = (NKEY + 15) / 16 * 16, MR = W * HP, MPAD = 16;
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NPAD + 8;
    constexpr int QKT = DQK / 16, NT8 = NPAD / 8, NKT2 = NPAD / 16, DT8 = DV / 8;
    constexpr int NTW = (NT8 + NW - 1) / NW, DTW = (DT8 + NW - 1) / NW;
    constexpr int KCH = DQK / 8, VCH = DV / 8;
    constexpr int NOLD = NKEY - W;   // key rows 0 .. NOLD-1 come from the cache
    static_assert(MR == 16, "one m16 tile");
    static_assert((MPAD * KP + NPAD * KP + NPAD * VP + MPAD * PP) * 2 + 2 * NW * MPAD * 4 + NPAD * 4 <= OFF_X - OFF_ATT, "attention scratch fits");
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NPAD * KP;
    __nv_bfloat16* Ps = Vs + NPAD * VP;
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int* s_slot = reinterpret_cast<int*>(rsum + NW * MPAD);   // [NPAD] cache slot of key row r (-1: zero / new row)
    if (tid == 0) BSTAMP(20);
    const bool pos64 = P.mode & MODE_POS64;
    // ---- independent loads first: seqlen, positions (Q-row token / new-key token of this thread), sinks, cached slots
    const int qr = tid >> 4, qq = tid & 15;                 // Q: row qr (token b*8 + qr/2, head 2 hg + qr%2), pairs 2qq, 2qq+1, dims 64 + 8qq ..
    const int qtok = b * W + qr / HP, qh = hg * HP + (qr % HP);
    const int ktok = b * W + warp;                          // new key / value row NOLD + warp (lane: pair lane, dims 64 + 8 lane / V 8 (lane-16))
    const int seqlen = P.seqlens[b];
    const long long qpos = ld_idx(P.positions, qtok, pos64), kpos = ld_idx(P.positions, ktok, pos64);
    const long long kslot = ld_idx(P.loc, ktok, P.mode & MODE_LOC64);
    float sk0 = 0.f, sk1 = 0.f;
    if (P.mode & MODE_SINK) { sk0 = P.sinks[hg * HP + ((lane >> 2) % HP)] * LOG2E; sk1 = P.sinks[hg * HP + (((lane >> 2) + 8) % HP)] * LOG2E; }
    const bool slots_read = P.mode & MODE_SLOTS_READ;
    int cslot = 0;
    if (slots_read && tid < NPAD) cslot = P.slot_cache[b * NPAD + tid];
    const int base_pos = seqlen - W - (WINDOW - 1);
    if (tid < NPAD) {
        int v;
        if (slots_read) v = cslot;
        else {
            const int p = base_pos + tid;
            v = (tid < NOLD && p >= 0) ? P.pt[b * P.pt_stride + p] : ((tid >= NKEY && (P.mode & MODE_DBG_GARB)) ? tid : -1);
            if ((P.mode & MODE_SLOTS_WRITE) && hg == 0) P.slot_cache[b * NPAD + tid] = v;
        }
        s_slot[tid] = v;
    }
    // ---- qkv (previous kernel): raw Q / new K, V loads issued now, consumed after the gather issue loop (in flight meanwhile)
    griddep_wait();
    if (tid == 0) *reinterpret_cast<unsigned*>(smem + OFF_MISC) = (unsigned)(*reinterpret_cast<const volatile int*>(P.work + 2)) + 1u;
    const __nv_bfloat16* qsrc = P.qkv + (size_t)qtok * QKVW + (size_t)qh * DQK;
    const __nv_bfloat16* ksrc = P.qkv + (size_t)ktok * QKVW + NH * DQK;
    const __nv_bfloat16* qcs = P.cos_sin + (size_t)qpos * 64;
    const __nv_bfloat16* kcs = P.cos_sin + (size_t)kpos * 64;
    __nv_bfloat16 qx[2], qy[2], qc[2], qs[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) { const int i = 2 * qq + e; qx[e] = qsrc[i]; qy[e] = qsrc[i + 32]; qc[e] = qcs[i]; qs[e] = qcs[32 + i]; }
    const uint4 qrest = *reinterpret_cast<const uint4*>(qsrc + 64 + 8 * qq);
    const __nv_bfloat16 kx = ksrc[lane], ky = ksrc[lane + 32], kc_ = kcs[lane], ks_ = kcs[32 + lane];
    const uint4 krest = *reinterpret_cast<const uint4*>(lane < 16 ? ksrc + 64 + 8 * lane : ksrc + DQK + 8 * (lane - 16));
    named_bar_sync(1, NMATH);
    if (tid == 0) BSTAMP(22);
    {
        // fixed 16-B chunk per thread (c = tid % 40: K chunk 0..23 | V chunk 0..15), rows gq, gq + 6, ... (gq = tid / 40 < 6): no per-chunk
        // division / dependent slot lookup in the issue loop (the q-strided loop was instruction-latency bound: 2.4 vs 1.4 us, ub/gath.cu)
        constexpr int CPR = KCH + VCH, NG = NMATH / CPR, RPG = (NPAD + NG - 1) / NG;   // 40 chunks per row, 6 row groups, 24 rows each
        const int c = tid % CPR, gq = tid / CPR;
        if (gq < NG) {
            const bool isk = c < KCH;
            const __nv_bfloat16* gbase = isk ? P.kc + c * 8 : P.vc + (c - KCH) * 8;
            const long long gstr = isk ? P.ks : P.vs;
            __nv_bfloat16* sbase = isk ? Ks + c * 8 : Vs + (c - KCH) * 8;
            const int sstr = isk ? KP : VP;
            int sl[RPG];
#pragma unroll
            for (int i = 0; i < RPG; ++i) sl[i] = (gq + NG * i < NPAD) ? s_slot[gq + NG * i] : -2;
#pragma unroll
            for (int i = 0; i < RPG; ++i) {
                const int r = gq + NG * i;
                if (sl[i] >= 0) cp16(sbase + r * sstr, gbase + (size_t)sl[i] * gstr);
                else if (sl[i] == -1 && (r < NOLD || r >= NKEY)) *reinterpret_cast<uint4*>(sbase + r * sstr) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    cp_commit();
    for (int i = tid; i < (MPAD - MR) * KP; i += NMATH) Qs[MR * KP + i] = __float2bfloat16(0.f);
    if (tid == 0) BSTAMP(27);   // gathers issued
    // ---- RoPE of Q / the new key rows (rope_kv's bf16 ops), new V rows, KV-cache store
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int i = 2 * qq + e;
        __nv_bfloat16 lo, hi;
        rope_pair(qx[e], qy[e], qc[e], qs[e], lo, hi);
        Qs[qr * KP + i] = lo; Qs[qr * KP + i + 32] = hi;
    }
    *reinterpret_cast<uint4*>(&Qs[qr * KP + 64 + 8 * qq]) = qrest;
    {
        const int r = NOLD + warp;
        __nv_bfloat16 lo, hi;
        rope_pair(kx, ky, kc_, ks_, lo, hi);
        Ks[r * KP + lane] = lo; Ks[r * KP + lane + 32] = hi;
        if (lane < 16) *reinterpret_cast<uint4*>(&Ks[r * KP + 64 + 8 * lane]) = krest;
        else *reinterpret_cast<uint4*>(&Vs[r * VP + 8 * (lane - 16)]) = krest;
        if (hg == 0 && kslot > 0 && kslot < P.kv_rows) {   // KV-cache store (rope_kv semantics: slot 0 is the padding slot, never written)
            __nv_bfloat16* kd = P.kc + (size_t)kslot * P.ks;
            kd[lane] = lo; kd[lane + 32] = hi;
            if (lane < 16) *reinterpret_cast<uint4*>(kd + 64 + 8 * lane) = krest;
            else *reinterpret_cast<uint4*>(P.vc + (size_t)kslot * P.vs + 8 * (lane - 16)) = krest;
        }
    }
    if (tid == 0) BSTAMP(24);   // Q / new K, V written, KV stored (issue)
    cp_wait_all();
    named_bar_sync(1, NMATH);
    if (tid == 0) { BSTAMP(6); mbar_arrive(att_in); }   // attention inputs in smem -> the producer may start this SM's weight TMAs

    // ---- S = Q K^T (verbatim swa_mma)
    if (tid == 0) CSTAMP(8);
    const int rA = lane >> 2, rB = rA + 8, cA = (lane & 3) * 2;
    unsigned qa[QKT][4];
#pragma unroll
    for (int kk = 0; kk < QKT; ++kk) ldm4(qa[kk], &Qs[(lane & 15) * KP + kk * 16 + (lane >> 4) * 8]);
    if (tid == 0) CSTAMP(9);
    // kk outer / n-tile inner: the NTW chains of 12 dependent mma (24 clk latency each) interleave instead of running back to back;
    // every output keeps its k order -> bitwise swa_mma
    float acc[NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][i] = 0.f;
    // explicit software pipeline: the fragment loads of k-step kk + 2 are issued before the mma of step kk (the asm statements are
    // volatile, so the compiler keeps program order: written naively every mma waited for the ldmatrix right in front of it)
    unsigned kb[2][NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt < NT8) ldm4(kb[0][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (lane >> 3) * 8]);
    }
#pragma unroll
    for (int kk = 0; kk < QKT; kk += 2) {
        const int cur = (kk >> 1) & 1;
        if (kk + 2 < QKT) {
#pragma unroll
            for (int ti = 0; ti < NTW; ++ti) {
                const int nt = warp + ti * NW;
                if (nt < NT8) ldm4(kb[cur ^ 1][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (kk + 2) * 16 + (lane >> 3) * 8]);
            }
        }
#pragma unroll
        for (int ti = 0; ti < NTW; ++ti) {
            const int nt = warp + ti * NW;
            if (nt < NT8) { mma16816(acc[ti], qa[kk], kb[cur][ti]); mma16816(acc[ti], qa[kk + 1], kb[cur][ti] + 2); }
        }
    }
    if (tid == 0) { CSTAMP(10); BSTAMP(25); }   // QK done (warp 0)
    // ---- softmax (verbatim)
    const float scale2 = P.sm_scale * LOG2E;
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
            const bool ok = (r < MR) && (j >= t) && (j <= t + (WINDOW - 1)) && (j < NKEY) && (base_pos + j >= 0);
            acc[ti][i] = ok ? acc[ti][i] * scale2 : -INFINITY;
        }
        mx0 = fmaxf(mx0, fmaxf(acc[ti][0], acc[ti][1]));
        mx1 = fmaxf(mx1, fmaxf(acc[ti][2], acc[ti][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    if ((lane & 3) == 0) { rmax[warp * MPAD + rA] = mx0; rmax[warp * MPAD + rB] = mx1; }
    if (tid == 0) CSTAMP(11);
    named_bar_sync(1, NMATH);
    if (tid == 0) CSTAMP(12);
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) { m0 = fmaxf(m0, rmax[w * MPAD + rA]); m1 = fmaxf(m1, rmax[w * MPAD + rB]); }
    const bool has_sink = P.mode & MODE_SINK;
    if (has_sink) {   // sk0 / sk1 loaded at entry (rA = lane / 4, rB = rA + 8)
        m0 = fmaxf(m0, sk0);
        m1 = fmaxf(m1, sk1);
    }
    if (!isfinite(m0)) m0 = 0.f;
    if (!isfinite(m1)) m1 = 0.f;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt >= NT8) continue;
        const int n0 = nt * 8;
        const float p0 = fast_ex2(acc[ti][0] - m0), p1 = fast_ex2(acc[ti][1] - m0);
        const float p2 = fast_ex2(acc[ti][2] - m1), p3 = fast_ex2(acc[ti][3] - m1);
        s0 += p0 + p1;
        s1 += p2 + p3;
        *reinterpret_cast<unsigned*>(&Ps[rA * PP + n0 + cA]) = pack_bf16(p0, p1);
        *reinterpret_cast<unsigned*>(&Ps[rB * PP + n0 + cA]) = pack_bf16(p2, p3);
    }
    s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
    s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
    if ((lane & 3) == 0) { rsum[warp * MPAD + rA] = s0; rsum[warp * MPAD + rB] = s1; }
    if (tid == 0) CSTAMP(13);
    named_bar_sync(1, NMATH);
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) { l0 += rsum[w * MPAD + rA]; l1 += rsum[w * MPAD + rB]; }
    if (has_sink) { l0 += fast_ex2(sk0 - m0); l1 += fast_ex2(sk1 - m1); }
    if (tid == 0) { CSTAMP(14); BSTAMP(26); }   // softmax done
    // ---- O = P V (verbatim), output row (token b*8 + t, head h) -> attn_out[tok][h * 128 + d]
    unsigned pa[NKT2][4];
#pragma unroll
    for (int kk = 0; kk < NKT2; ++kk) ldm4(pa[kk], &Ps[(lane & 15) * PP + kk * 16 + (lane >> 4) * 8]);
    float o[DTW][4];
#pragma unroll
    for (int di = 0; di < DTW; ++di)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[di][i] = 0.f;
    unsigned vb[2][DTW][4];
#pragma unroll
    for (int di = 0; di < DTW; ++di) {
        const int dt = warp + di * NW;
        if (dt < DT8) ldm4t(vb[0][di], &Vs[lane * VP + dt * 8]);
    }
#pragma unroll
    for (int kk = 0; kk < NKT2; kk += 2) {   // kk outer, value tile inner, next step's V fragments loaded ahead (per-output k order unchanged)
        const int cur = (kk >> 1) & 1;
        if (kk + 2 < NKT2) {
#pragma unroll
            for (int di = 0; di < DTW; ++di) {
                const int dt = warp + di * NW;
                if (dt < DT8) ldm4t(vb[cur ^ 1][di], &Vs[((kk + 2) * 16 + lane) * VP + dt * 8]);
            }
        }
#pragma unroll
        for (int di = 0; di < DTW; ++di) {
            const int dt = warp + di * NW;
            if (dt < DT8) {
                mma16816(o[di], pa[kk], vb[cur][di]);
                if (kk + 1 < NKT2) mma16816(o[di], pa[kk + 1], vb[cur][di] + 2);
            }
        }
    }
    if (tid == 0) CSTAMP(15);
    const float inv0 = (l0 > 0.f) ? fast_rcp(l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? fast_rcp(l1) : 0.f;
#pragma unroll
    for (int di = 0; di < DTW; ++di) {
        const int dt = warp + di * NW;
        if (dt >= DT8) continue;
        const int n0 = dt * 8;
        if (rA < MR) {
            const int t = rA / HP, h = hg * HP + (rA % HP);
            *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + n0 + cA]) = pack_bf16(o[di][0] * inv0, o[di][1] * inv0);
        }
        if (rB < MR) {
            const int t = rB / HP, h = hg * HP + (rB % HP);
            *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + n0 + cA]) = pack_bf16(o[di][2] * inv1, o[di][3] * inv1);
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ kernel
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ float4 ld_dsmem_f4(uint32_t a) {
    float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void st_async_f32(uint32_t addr, float a, uint32_t bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(addr), "r"(__float_as_uint(a)), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_async_v2(uint32_t addr, float a, float b, uint32_t bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" :: "r"(addr), "f"(a), "f"(b), "r"(bar) : "memory");
}
__device__ __forceinline__ unsigned cluster_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }


// slot list of the key-split attention scratch (same layout arithmetic as attention_ks)
__device__ __forceinline__ const int* attention_ks_slots(const uint8_t* smem) {
    constexpr int MPAD = 16, NRMAX = 80, KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    const __nv_bfloat16* Qs = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_ATT);
    const __nv_bfloat16* Ps = Qs + MPAD * KP + NRMAX * KP + (NRMAX + 16) * VP;
    const float* rmax = reinterpret_cast<const float*>(Ps + MPAD * PP);
    return reinterpret_cast<const int*>(rmax + 2 * NW * MPAD);
}
// ---- key-split attention (BLK_KSPLIT): CTA kh of a cluster pair takes key rows [0, 64) (kh 0) or [64, 144) (kh 1) of the window; the row
// maxima are exchanged first (so P = exp2(s - m) is bitwise swa_mma's), then each half's PV partial for the partner's 64 output dims and its
// row sums go over DSMEM; CTA kh finalizes dims [64 kh, 64 kh + 64). Only the fp32 order of (half 0 + half 1) differs from swa_mma.
template <int WINDOW>
__device__ __forceinline__ void attention_ks(const Params& P, uint8_t* smem, int b, int hg, int kh, uint64_t* att_in, uint64_t* xm_full, uint64_t* xo_full) {
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, NPAD = (NKEY + 15) / 16 * 16, MR = W * HP, MPAD = 16;
    constexpr int R0H = 64, NRMAX = NPAD - R0H;   // half 0: 64 rows, half 1: 80 rows
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    constexpr int QKT = DQK / 16, DT8 = DV / 8, NTW = 2;
    constexpr int KCH = DQK / 8, VCH = DV / 8;
    constexpr int NOLD = NKEY - W;
    static_assert(MR == 16 && NRMAX == 80 && NOLD >= R0H && DT8 == 2 * NW, "key-split geometry");
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NRMAX * KP;
    __nv_bfloat16* Ps = Vs + (NRMAX + 16) * VP;     // + 16 rows: the x4.trans V fragment of the last (odd) k-step reads rows 80..95
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    int* s_slot = reinterpret_cast<int*>(rsum + NW * MPAD);
    float* xm = reinterpret_cast<float*>(s_slot + NRMAX);   // [16] the partner's row maxima
    float* xl = xm + MPAD;                                   // [16] the partner's row sums
    float* xo = xl + MPAD;                                   // [16][64] the partner's PV partial of my output dims
    static_assert((MPAD * KP + NRMAX * KP + (NRMAX + 16) * VP + MPAD * PP) * 2 + (2 * NW * MPAD + NRMAX + 2 * MPAD + MPAD * 64) * 4 <= OFF_X - OFF_ATT,
                  "key-split attention scratch fits");
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int R0 = kh ? R0H : 0, NR = kh ? NRMAX : R0H, NT8 = NR / 8, NK16 = NR / 16;
    const uint32_t pxm = mapa(sa(xm), (uint32_t)(kh ^ 1)), pxl = mapa(sa(xl), (uint32_t)(kh ^ 1)), pxo = mapa(sa(xo), (uint32_t)(kh ^ 1));
    const uint32_t pbm = mapa(sa(xm_full), (uint32_t)(kh ^ 1)), pbo = mapa(sa(xo_full), (uint32_t)(kh ^ 1));
    if (tid == 0) BSTAMP(20);
    const bool pos64 = P.mode & MODE_POS64;
    const int qr = tid >> 4, qq = tid & 15;
    const int qtok = b * W + qr / HP, qh = hg * HP + (qr % HP);
    const int ktok = b * W + warp;
    const int seqlen = P.seqlens[b];
    const long long qpos = ld_idx(P.positions, qtok, pos64), kpos = ld_idx(P.positions, ktok, pos64);
    const long long kslot = ld_idx(P.loc, ktok, P.mode & MODE_LOC64);
    float sk0 = 0.f, sk1 = 0.f;
    if (P.mode & MODE_SINK) { sk0 = P.sinks[hg * HP + ((lane >> 2) % HP)] * LOG2E; sk1 = P.sinks[hg * HP + (((lane >> 2) + 8) % HP)] * LOG2E; }
    const bool slots_read = P.mode & MODE_SLOTS_READ;
    int cslot = 0;
    if (slots_read && tid < NR) cslot = P.slot_cache[b * NPAD + R0 + tid];
    const int base_pos = seqlen - W - (WINDOW - 1);
    if (tid < NR) {
        const int r = R0 + tid;
        int v;
        if (slots_read) v = cslot;
        else {
            const int p = base_pos + r;
            v = (r < NOLD && p >= 0) ? P.pt[b * P.pt_stride + p] : -1;
            if ((P.mode & MODE_SLOTS_WRITE) && hg == 0) P.slot_cache[b * NPAD + r] = v;
        }
        s_slot[tid] = v;
    }
    griddep_wait();
    if (tid == 0) *reinterpret_cast<unsigned*>(smem + OFF_MISC) = (unsigned)(*reinterpret_cast<const volatile int*>(P.work + 2)) + 1u;
    const __nv_bfloat16* qsrc = P.qkv + (size_t)qtok * QKVW + (size_t)qh * DQK;
    const __nv_bfloat16* ksrc = P.qkv + (size_t)ktok * QKVW + NH * DQK;
    const __nv_bfloat16* qcs = P.cos_sin + (size_t)qpos * 64;
    const __nv_bfloat16* kcs = P.cos_sin + (size_t)kpos * 64;
    __nv_bfloat16 qx[2], qy[2], qc[2], qs[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) { const int i = 2 * qq + e; qx[e] = qsrc[i]; qy[e] = qsrc[i + 32]; qc[e] = qcs[i]; qs[e] = qcs[32 + i]; }
    const uint4 qrest = *reinterpret_cast<const uint4*>(qsrc + 64 + 8 * qq);
    __nv_bfloat16 kx = __float2bfloat16(0.f), ky = kx, kc_ = kx, ks_ = kx;
    uint4 krest = make_uint4(0u, 0u, 0u, 0u);
    if (kh) {
        kx = ksrc[lane]; ky = ksrc[lane + 32]; kc_ = kcs[lane]; ks_ = kcs[32 + lane];
        krest = *reinterpret_cast<const uint4*>(lane < 16 ? ksrc + 64 + 8 * lane : ksrc + DQK + 8 * (lane - 16));
    }
    named_bar_sync(1, NMATH);
    if (tid == 0) BSTAMP(22);
    {
        constexpr int CPR = KCH + VCH, NG = NMATH / CPR, RPG = (NRMAX + NG - 1) / NG;   // 40 chunks per row, 6 row groups, <= 14 rows each
        const int c = tid % CPR, gq = tid / CPR;
        if (gq < NG) {
            const bool isk = c < KCH;
            const __nv_bfloat16* gbase = isk ? P.kc + c * 8 : P.vc + (c - KCH) * 8;
            const long long gstr = isk ? P.ks : P.vs;
            __nv_bfloat16* sbase = isk ? Ks + c * 8 : Vs + (c - KCH) * 8;
            const int sstr = isk ? KP : VP;
            int sl[RPG];
#pragma unroll
            for (int i = 0; i < RPG; ++i) sl[i] = (gq + NG * i < NR) ? s_slot[gq + NG * i] : -2;
#pragma unroll
            for (int i = 0; i < RPG; ++i) {
                const int lr = gq + NG * i, r = R0 + lr;
                if (sl[i] >= 0) cp16(sbase + lr * sstr, gbase + (size_t)sl[i] * gstr);
                else if (sl[i] == -1 && (r < NOLD || r >= NKEY)) *reinterpret_cast<uint4*>(sbase + lr * sstr) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    cp_commit();
    if (tid == 0) BSTAMP(27);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int i = 2 * qq + e;
        __nv_bfloat16 lo, hi;
        rope_pair(qx[e], qy[e], qc[e], qs[e], lo, hi);
        Qs[qr * KP + i] = lo; Qs[qr * KP + i + 32] = hi;
    }
    *reinterpret_cast<uint4*>(&Qs[qr * KP + 64 + 8 * qq]) = qrest;
    if (kh) {
        const int lr = NOLD - R0 + warp;
        __nv_bfloat16 lo, hi;
        rope_pair(kx, ky, kc_, ks_, lo, hi);
        Ks[lr * KP + lane] = lo; Ks[lr * KP + lane + 32] = hi;
        if (lane < 16) *reinterpret_cast<uint4*>(&Ks[lr * KP + 64 + 8 * lane]) = krest;
        else *reinterpret_cast<uint4*>(&Vs[lr * VP + 8 * (lane - 16)]) = krest;
        if (hg == 0 && kslot > 0 && kslot < P.kv_rows) {
            __nv_bfloat16* kd = P.kc + (size_t)kslot * P.ks;
            kd[lane] = lo; kd[lane + 32] = hi;
            if (lane < 16) *reinterpret_cast<uint4*>(kd + 64 + 8 * lane) = krest;
            else *reinterpret_cast<uint4*>(P.vc + (size_t)kslot * P.vs + 8 * (lane - 16)) = krest;
        }
    }
    if (tid == 0) BSTAMP(24);
    cp_wait_all();
    named_bar_sync(1, NMATH);
    if (tid == 0) { BSTAMP(6); mbar_arrive(att_in); CSTAMP(8); }
    // ---- S = Q K^T over this half's key tiles (warp w: tiles w, w + 8)
    const int rA = lane >> 2, rB = rA + 8, cA = (lane & 3) * 2;
    unsigned qa[QKT][4];
#pragma unroll
    for (int kk = 0; kk < QKT; ++kk) ldm4(qa[kk], &Qs[(lane & 15) * KP + kk * 16 + (lane >> 4) * 8]);
    float acc[NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][i] = 0.f;
    unsigned kb[2][NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt < NT8) ldm4(kb[0][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (lane >> 3) * 8]);
    }
#pragma unroll
    for (int kk = 0; kk < QKT; kk += 2) {
        const int cur = (kk >> 1) & 1;
        if (kk + 2 < QKT) {
#pragma unroll
            for (int ti = 0; ti < NTW; ++ti) {
                const int nt = warp + ti * NW;
                if (nt < NT8) ldm4(kb[cur ^ 1][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (kk + 2) * 16 + (lane >> 3) * 8]);
            }
        }
#pragma unroll
        for (int ti = 0; ti < NTW; ++ti) {
            const int nt = warp + ti * NW;
            if (nt < NT8) { mma16816(acc[ti], qa[kk], kb[cur][ti]); mma16816(acc[ti], qa[kk + 1], kb[cur][ti] + 2); }
        }
    }
    if (tid == 0) { BSTAMP(25); CSTAMP(9); }
    // ---- softmax: local maxima -> exchange -> global m (bitwise swa_mma's m: max is order-free)
    const float scale2 = P.sm_scale * LOG2E;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt >= NT8) continue;
        const int n0 = nt * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (i < 2) ? rA : rB;
            const int j = R0 + n0 + cA + (i & 1);
            const int t = r / HP;
            const bool ok = (r < MR) && (j >= t) && (j <= t + (WINDOW - 1)) && (j < NKEY) && (base_pos + j >= 0);
            acc[ti][i] = ok ? acc[ti][i] * scale2 : -INFINITY;
        }
        mx0 = fmaxf(mx0, fmaxf(acc[ti][0], acc[ti][1]));
        mx1 = fmaxf(mx1, fmaxf(acc[ti][2], acc[ti][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    if ((lane & 3) == 0) { rmax[warp * MPAD + rA] = mx0; rmax[warp * MPAD + rB] = mx1; }
    named_bar_sync(1, NMATH);
    if (tid == 0) CSTAMP(10);
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) { m0 = fmaxf(m0, rmax[w * MPAD + rA]); m1 = fmaxf(m1, rmax[w * MPAD + rB]); }
#if BLK_KS_MAXX
    if (warp == 0 && (lane & 3) == 0) { st_async_f32(pxm + rA * 4, m0, pbm); st_async_f32(pxm + rB * 4, m1, pbm); }
    mbar_wait(xm_full, 0);
#endif
    if (tid == 0) CSTAMP(11);
#if BLK_KS_MAXX
    m0 = fmaxf(m0, xm[rA]); m1 = fmaxf(m1, xm[rB]);
#endif
    const bool has_sink = P.mode & MODE_SINK;
    if (has_sink) { m0 = fmaxf(m0, sk0); m1 = fmaxf(m1, sk1); }   // the sink only lowers / keeps exp2 terms: harmless in a local max
    if (!isfinite(m0)) m0 = 0.f;
    if (!isfinite(m1)) m1 = 0.f;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt >= NT8) continue;
        const int n0 = nt * 8;
        const float p0 = fast_ex2(acc[ti][0] - m0), p1 = fast_ex2(acc[ti][1] - m0);
        const float p2 = fast_ex2(acc[ti][2] - m1), p3 = fast_ex2(acc[ti][3] - m1);
        s0 += p0 + p1;
        s1 += p2 + p3;
        *reinterpret_cast<unsigned*>(&Ps[rA * PP + n0 + cA]) = pack_bf16(p0, p1);
        *reinterpret_cast<unsigned*>(&Ps[rB * PP + n0 + cA]) = pack_bf16(p2, p3);
    }
    s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
    s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
    if ((lane & 3) == 0) { rsum[warp * MPAD + rA] = s0; rsum[warp * MPAD + rB] = s1; }
    named_bar_sync(1, NMATH);
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) { l0 += rsum[w * MPAD + rA]; l1 += rsum[w * MPAD + rB]; }
    if (tid == 0) { BSTAMP(26); CSTAMP(12); }
    // ---- O_h = P_h V_h (warp w: value tiles w, w + 8)
    unsigned pa[5][4];
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) if (kk < NK16) ldm4(pa[kk], &Ps[(lane & 15) * PP + kk * 16 + (lane >> 4) * 8]);
    float o[2][4];
#pragma unroll
    for (int di = 0; di < 2; ++di)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[di][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 5; kk += 2) {
        if (kk >= NK16) break;
        unsigned vb[2][4];
#pragma unroll
        for (int di = 0; di < 2; ++di) ldm4t(vb[di], &Vs[(kk * 16 + lane) * VP + (warp + di * NW) * 8]);
#pragma unroll
        for (int di = 0; di < 2; ++di) {
            mma16816(o[di], pa[kk], vb[di]);
            if (kk + 1 < NK16) mma16816(o[di], pa[kk + 1], vb[di] + 2);
        }
    }
    if (tid == 0) CSTAMP(13);
    // ---- exchange: send the partner's dims (tile 1 - kh) + my row sums; keep tile kh
    const int si = kh ^ 1, ki = kh;
    st_async_v2(pxo + (rA * 64 + 8 * warp + cA) * 4, o[si][0], o[si][1], pbo);
    st_async_v2(pxo + (rB * 64 + 8 * warp + cA) * 4, o[si][2], o[si][3], pbo);
    if (warp == 0 && (lane & 3) == 0) {
        st_async_f32(pxl + rA * 4, l0, pbo); st_async_f32(pxl + rB * 4, l1, pbo);
#if !BLK_KS_MAXX
        st_async_f32(pxm + rA * 4, m0, pbo); st_async_f32(pxm + rB * 4, m1, pbo);
#endif
    }
    if (tid == 0) CSTAMP(14);
    mbar_wait(xo_full, 0);
    if (tid == 0) CSTAMP(15);
#if BLK_KS_MAXX
    // fixed order (half 0 + half 1) for l so both CTAs of the pair use the same row sum
    float lt0 = kh ? xl[rA] + l0 : l0 + xl[rA], lt1 = kh ? xl[rB] + l1 : l1 + xl[rB];
    if (has_sink) { lt0 += fast_ex2(sk0 - m0); lt1 += fast_ex2(sk1 - m1); }
    const float c0s = 1.f, c1s = 1.f, p0s = 1.f, p1s = 1.f;
#else
    // split-softmax combine: m = max(m_mine, m_partner) (both already include the sink); rescale both halves by exp2(m_h - m), in the
    // fixed order (half 0 + half 1) so both CTAs of the pair use the same row sum
    const float pm0 = xm[rA], pm1 = xm[rB];
    const float M0 = fmaxf(m0, pm0), M1 = fmaxf(m1, pm1);
    const float c0s = fast_ex2(m0 - M0), c1s = fast_ex2(m1 - M1), p0s = fast_ex2(pm0 - M0), p1s = fast_ex2(pm1 - M1);
    float lt0 = kh ? xl[rA] * p0s + l0 * c0s : l0 * c0s + xl[rA] * p0s;
    float lt1 = kh ? xl[rB] * p1s + l1 * c1s : l1 * c1s + xl[rB] * p1s;
    if (has_sink) { lt0 += fast_ex2(sk0 - M0); lt1 += fast_ex2(sk1 - M1); }
#endif
    const float inv0 = (lt0 > 0.f) ? fast_rcp(lt0) : 0.f;
    const float inv1 = (lt1 > 0.f) ? fast_rcp(lt1) : 0.f;
    const float2 pa0 = *reinterpret_cast<const float2*>(xo + rA * 64 + 8 * warp + cA);
    const float2 pa1 = *reinterpret_cast<const float2*>(xo + rB * 64 + 8 * warp + cA);
    const int dim = 64 * kh + 8 * warp + cA;
    {
        const int t = rA / HP, h = hg * HP + (rA % HP);
        *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + dim]) =
            pack_bf16((o[ki][0] * c0s + pa0.x * p0s) * inv0, (o[ki][1] * c0s + pa0.y * p0s) * inv0);
    }
    {
        const int t = rB / HP, h = hg * HP + (rB % HP);
        *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + dim]) =
            pack_bf16((o[ki][2] * c1s + pa1.x * p1s) * inv1, (o[ki][3] * c1s + pa1.y * p1s) * inv1);
    }
}

template <int WINDOW>
__global__ void __launch_bounds__(NTHREADS, 1) attn_block_kernel(const __grid_constant__ Params P) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
    uint64_t* wfull = reinterpret_cast<uint64_t*>(smem + OFF_BAR);   // [NS] weight half of a stage landed
    uint64_t* bfull = wfull + NS;                                     // [NS] attention half landed
    uint64_t* sempty = bfull + NS;                                    // [NS] stage consumed (8 math warps)
    uint64_t* att_in = sempty + NS;                                   // attention inputs in smem (attention CTAs)
    uint64_t* att_done = att_in + 1;                                  // attention scratch dead (attention CTAs)
    uint64_t* xfull = att_done + 1;                                   // the partner's 6 KB partial landed (st.async complete_tx)
    uint64_t* xm_full = xfull + 1;                                    // key-split attention: the partner's row maxima landed
    uint64_t* xo_full = xm_full + 1;                                  // key-split attention: the partner's PV partial + row sums landed
    uint64_t* bgo = xo_full + 1;                                      // all attention outputs visible (math warps -> producer lane 1)
    unsigned* s_epoch = reinterpret_cast<unsigned*>(smem + OFF_MISC);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c = blockIdx.x;
    const int h = (int)cluster_rank(), p = c / CL;                    // K half, column pair
    const int kbase = h * (KO / CL), rbase = p * PCOLS;               // first k element / weight row of this CTA
    const bool ext_attn = P.mode & MODE_EXT_ATTN;   // FA layers: the attention output comes from the previous kernel (tm_a over it)
    const bool att = c < NATT && !ext_attn;
    if ((P.mode & MODE_DBG_ATTONLY) && !att) return;
    if (tid == 0) {
        BSTAMP(0);
        for (int i = 0; i < NS; ++i) { mbar_init(&wfull[i], 1); mbar_init(&bfull[i], 1); mbar_init(&sempty[i], NW); }
        mbar_init(att_in, 1); mbar_init(att_done, 1); mbar_init(xfull, 1); mbar_init(xm_full, 1); mbar_init(xo_full, 1); mbar_init(bgo, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect(xfull, COLS * M * 4);   // armed before the partner can see the barrier (cluster sync below)
        if (BLK_KSPLIT && att) {
            if (BLK_KS_MAXX) mbar_expect(xm_full, 16 * 4);
            mbar_expect(xo_full, 16 * 64 * 4 + 16 * 4 + (BLK_KS_MAXX ? 0 : 16 * 4));
        }
    }
    cluster_sync_relaxed();   // barrier inits visible cluster-wide (the partner st.asyncs into my xfull later)
    float d[NACC][16];   // NACC independent wgmma accumulator chains (k16 step j of every k-tile -> chain j % NACC); summed at the end
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 16; ++i) d[a][i] = 0.f;
    if (warp == NW && (P.mode & MODE_DBG_ATTONLY)) return;
    if (warp == NW) {
        // ================= producer warp: all TMA issue + the attention-ready flag poll
        auto issue_w = [&](int kt) {
            const int s = kt % NS;
            mbar_expect(&wfull[s], WT_BYTES);
            tma_load_2d(smem + s * ST_BYTES, &P.tm_w, kbase + kt * KT, rbase, &wfull[s]);
        };
        if (lane == 0) {
            // weight issuer: BLK_WD k-tiles in flight; attention CTAs start after their gathers landed and keep stages
            // NS0_ATT.. (their attention scratch) out until the attention is done
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_w)) : "memory");
            if (att) mbar_wait(att_in, 0);
            for (int kt = 0; kt < NKT && !(P.mode & MODE_DBG_NOW); ++kt) {
                const int s = kt % NS;
                if (att && kt == NS0_ATT) mbar_wait(att_done, 0);
                if (kt >= NS) mbar_wait(&sempty[s], ((kt / NS) - 1) & 1);
                if (!ext_attn && kt >= BLK_WD) mbar_wait(&wfull[(kt - BLK_WD) % NS], ((kt - BLK_WD) / NS) & 1);   // FA mode: no latency-critical loads to protect
                issue_w(kt);
            }
        } else if (lane == 1) {
            // lane 1: the attention k-tiles. FA mode: all of them after griddepcontrol.wait (the fa3 output is the previous kernel's).
            // SWA mode: the math warps poll the attention flags (one flag per thread: per-lane sequential polling costs one round trip per
            // flag, ub/flag.cu) and issue k-tiles 0..NS-1 themselves; this lane issues NS.. as stages free up, after bgo.
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_a)) : "memory");
            int kt0 = NKT;
            if (ext_attn) {
                griddep_wait();
                BSTAMP(2);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                kt0 = 0;
            } else if (!(P.mode & MODE_DBG_NOB)) {
                mbar_wait(bgo, 0);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                kt0 = NS;
            }
            for (int kt = kt0; kt < NKT && !(P.mode & MODE_DBG_NOB); ++kt) {
                const int s = kt % NS;
                if (kt >= NS) mbar_wait(&sempty[s], ((kt / NS) - 1) & 1);
                mbar_expect(&bfull[s], BT_BYTES);
                tma_load_2d_nohint(smem + s * ST_BYTES + WT_BYTES, &P.tm_a, kbase + kt * KT, 0, &bfull[s]);
            }
        }
        __syncwarp();
    } else {
        // ================= math warps
        if (att) {
#if BLK_KSPLIT
            attention_ks<WINDOW>(P, smem, (c / CL) / NHG, (c / CL) % NHG, h, att_in, xm_full, xo_full);   // contains griddepcontrol.wait
#else
            attention<WINDOW>(P, smem, c / NHG, c % NHG, att_in);   // contains griddepcontrol.wait
#endif
            if (P.mode & MODE_DBG_ATTONLY) { if (tid == 0) BSTAMP(1); return; }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic-proxy attention scratch accesses -> the TMA writes over it
            named_bar_sync(1, NMATH);
            if (tid == 0) {
                BSTAMP(1);
                st_release_gpu(&P.flags[c * FLAG_STRIDE], s_epoch[0]);   // one release (cumulative over the CTA's stores via the barrier above), BEFORE
                BSTAMP(30);                                 // this SM's remaining weight TMAs are triggered
            }
#if BLK_KSPLIT
            // next SWA layer: same slots -> L2-prefetch (evict_last) its K / V window rows. AFTER the release (a fence waits for outstanding
            // prefetches too) and before att_done (the slot list lives in the attention scratch the late weight TMAs overwrite)
            if (P.nkc && ((c / CL) % NHG) == 0) {
                constexpr int R0H = 64;
                const int* s_slot = attention_ks_slots(smem);
                const int NR = h ? (144 - R0H) : R0H;
                for (int i = tid; i < NR * 5; i += NMATH) {
                    const int lr = i / 5, part = i - lr * 5;
                    const int sl = s_slot[lr];
                    if (sl < 0) continue;
                    const char* a = part < 3 ? reinterpret_cast<const char*>(P.nkc + (size_t)sl * P.ks) + part * 128
                                             : reinterpret_cast<const char*>(P.nvc + (size_t)sl * P.vs) + (part - 3) * 128;
                    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
                }
            }
            named_bar_sync(1, NMATH);
#endif
            if (tid == 0) mbar_arrive(att_done);
        }
        else {
            griddep_wait();
            if (tid == 0) s_epoch[0] = (unsigned)(*reinterpret_cast<const volatile int*>(P.work + 2)) + 1u;
            if (!ext_attn) named_bar_sync(1, NMATH);   // epoch published to the pollers below
        }
        if (!ext_attn) {
            // one flag per polling thread (flag f on its own 128-B line), light backoff; then k-tiles 0..NS-1 from here, NS.. from lane 1
            const unsigned epoch = s_epoch[0];
            if (tid == 0) BSTAMP(31);
            if (tid < NATT && !(P.mode & MODE_DBG_NOPOLL)) {
                // ACQUIRE by the observing thread itself (a relaxed poll + a fence in another thread after the barrier is not an acquire
                // of this flag: the graph-vs-eager stress test caught stale attention k-tiles in ~1 of 10 launches)
                const unsigned* fl = P.flags + tid * FLAG_STRIDE;
                for (;;) {
                    if (ld_acquire_gpu(fl) == epoch) break;
                    __nanosleep(32);
                }
            }
            named_bar_sync(1, NMATH);
            if (tid == 0) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                BSTAMP(2);   // all attention flags seen
                asm volatile("fence.proxy.async.global;" ::: "memory");   // generic-proxy attention stores (released) -> async-proxy TMA reads
                for (int kt = 0; kt < NS && !(P.mode & MODE_DBG_NOB); ++kt) {
                    mbar_expect(&bfull[kt], BT_BYTES);
                    tma_load_2d_nohint(smem + kt * ST_BYTES + WT_BYTES, &P.tm_a, kbase + kt * KT, 0, &bfull[kt]);
                }
                mbar_arrive(bgo);
            }
        }
        // D[96 (+32) weight rows][32 tokens] += W[rows][k] * A[tok][k]: warpgroup g takes rows 64g .. 64g + 63
        const int g = warp >> 2;
        // fully unrolled (stage / phase / descriptor offsets are compile-time): with ~100 address instructions per k-tile per warp the
        // HGMMA issue serialized behind them (6.5 K vs 3.3 K cycles for the same 128 m64n32k16 in ub/wgl2.cu)
        const uint32_t sbase = sa(smem);
        const uint64_t dA0 = desc_sw128(sbase + g * 64 * 128), dB0 = desc_sw128(sbase + WT_BYTES);
        const bool dbg_now = P.mode & MODE_DBG_NOW, dbg_nob = P.mode & MODE_DBG_NOB;
#pragma unroll
        for (int kt = 0; kt < NKT; ++kt) {
            constexpr int dummy = 0; (void)dummy;
            const int s = kt % NS;
            const unsigned ph = (kt / NS) & 1;
            if (!dbg_now) mbar_wait(&wfull[s], ph);
            if (!dbg_nob) mbar_wait(&bfull[s], ph);
            if (kt == 0 && tid == 0) { BSTAMP(3); if (P.dbg && !att) { long long ck; asm volatile("mov.u64 %0, %%clock64;" : "=l"(ck)); P.dbg[blockIdx.x * NDBG + 24] = ck; } }
            wg_fence();
#pragma unroll
            for (int j = 0; j < KT / 16; ++j)
                wgmma_m64n32k16(d[j % NACC], dA0 + (uint64_t)((s * ST_BYTES + 32 * j) >> 4), dB0 + (uint64_t)((s * ST_BYTES + 32 * j) >> 4),
                                (kt * (KT / 16) + j) >= NACC);
            wg_commit();
            wg_wait<1>();
            if (kt >= 1 && lane == 0) mbar_arrive(&sempty[(kt - 1) % NS]);
        }
        wg_wait<0>();
#pragma unroll
        for (int a = 1; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 16; ++i) d[0][i] += d[a][i];
        if (tid == 0) { BSTAMP(4); if (P.dbg && !att) { long long ck; asm volatile("mov.u64 %0, %%clock64;" : "=l"(ck)); P.dbg[blockIdx.x * NDBG + 25] = ck; } }   // all MMAs of this CTA done
        // pair exchange: the rows the PARTNER finalizes go to its OFF_X by st.async (complete_tx on its xfull); my own rows stay in registers
        const int row = 64 * g + 16 * (warp & 3) + (lane >> 2);   // rows row, row + 8 of d[0][4i + 0/1], d[0][4i + 2/3]; tokens 8i + 2 (lane & 3) (+1)
        const int prow0 = COLS * (h ^ 1);
        const uint32_t rx = mapa(sa(smem + OFF_X), (uint32_t)(h ^ 1)), rbar = mapa(sa(xfull), (uint32_t)(h ^ 1));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = row + 8 * hh;
            if (r >= prow0 && r < prow0 + COLS) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int tok = 8 * i + 2 * (lane & 3);
                    st_async_v2(rx + ((r - prow0) * M + tok) * 4, d[0][4 * i + 2 * hh], d[0][4 * i + 2 * hh + 1], rbar);
                }
            }
        }
        if (tid == 0) BSTAMP(16);   // st.async issued
        mbar_wait(xfull, 0);   // the partner's 48 x 32 partial of my rows
        if (tid == 0) BSTAMP(17);
        // outs overwrite stage 0 (k-tile 13's weight rows 0..23, read by warpgroup 0): the OTHER warpgroup may still be in its MMA
        // loop (warp 0 issues the broadcast TMAs, so warpgroup 0 can trail warpgroup 1 by several k-tiles). Without this barrier the
        // graph-vs-eager stress test saw one corrupted o_proj column (pair-local column 1) in ~1 of 20 launches, attention pairs only.
        named_bar_sync(1, NMATH);
        __nv_bfloat16* outs = reinterpret_cast<__nv_bfloat16*>(smem);   // [32 tok][48] bf16 over stage 0 (all MMAs and TMAs done)
        const float* xs = reinterpret_cast<const float*>(smem + OFF_X);
        const int orow0 = COLS * h;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = row + 8 * hh;
            if (r >= orow0 && r < orow0 + COLS) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int tok = 8 * i + 2 * (lane & 3);
                    const float2 pv = *reinterpret_cast<const float2*>(xs + (r - orow0) * M + tok);
                    outs[tok * COLS + r - orow0] = __float2bfloat16_rn(d[0][4 * i + 2 * hh] + pv.x);
                    outs[(tok + 1) * COLS + r - orow0] = __float2bfloat16_rn(d[0][4 * i + 2 * hh + 1] + pv.y);
                }
            }
        }
        if (tid == 0) BSTAMP(18);
        named_bar_sync(1, NMATH);   // outs complete
        if (tid == 0) BSTAMP(19);
        // message (token tok, 6-column group g6): thread tok * 8 + g6. A warp = 4 tokens x 8 groups; lanes 2j, 2j+1 = messages m, m+1 (m even).
        const unsigned epoch = s_epoch[0];
        const int c0 = rbase + COLS * h;
        const int tok = tid >> 3, g6 = tid & 7;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(outs + tok * COLS + 6 * g6);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2];
        const int owner = tok % TP_NDEV;
        if (owner == P.my_rank || (P.mode & MODE_FULL_LOCAL)) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(P.partial + (size_t)tok * DIM + c0 + 6 * g6);
            dst[0] = w0; dst[1] = w1; dst[2] = w2;
        }
        if (owner != P.my_rank && !(P.mode & MODE_NO_PUSH)) {
            const int k = c0 / TP_KS, m = (c0 % TP_KS) / 6 + g6;
            st_ll(P.peer[owner] + tp_rs_idx(P.my_rank, tok, k, m), make_uint4(w0, w1, w2, epoch));
        }
    }
    if (tid == 0) BSTAMP(5);   // messages issued
    griddep_launch();
    cluster_sync_relaxed();   // no CTA exits while its partner's st.async into it may be in flight
}

// ================================================================================================================== mega (goal11 "mega")
// attention MEGAKERNEL: the target qkv_proj fp8 GEMM (M = 32, N = 3392, K = 6144; DeepGEMM 128x128 weight block scales x per-token
// per-128 activation scales = the fused MoE's norm-next output) as phase 0 of the fused SWA block, in the SAME grid (128 CTAs = 64
// cluster pairs). Phase 0: cluster pair p owns qkv rows [53 p, 53 p + 53) (64 x 53 = 3392), its CTA h the K half [3072 h, +3072):
// 24 units of {53 x 128 fp8 weight box, 32 x 128 x tile, 32 x scales}, swap-AB m64n32k32 wgmma (rows 53..63 of the A tile are
// don't-care), warpgroup w takes units w, w + 2, .. with DeepGEMM's per-unit promotion (acc += partial * sw[row block][kb] * sx[tok]);
// warpgroup 1 hands its sum to warpgroup 0 through smem, the pair exchanges halves over DSMEM, CTA h finalizes rows [27 h, ..) in the
// fixed order (K half 0 + K half 1), writes them as bf16 into the qkv buffer (= what the unfused GEMM writes, up to the summation order)
// and releases one flag (epoch). The 64 attention CTAs issued their K/V window gathers before griddepcontrol.wait (they depend only on
// earlier kernels), poll the flags of the row groups they read and run attention_ks from the qkv loads on; phase 2 (o_proj + RS push) is
// attn_block_kernel's.
constexpr int QN = QKVW, QK = DIM, QBK = 128, QKB = QK / QBK, QNB = (QN + 127) / 128;   // 3392, 6144, 48 k-blocks, 27 scale row blocks
constexpr int QG = 53, QKH = QKB / CL, QFIN0 = 27;   // rows per pair, k-blocks per CTA, rows finalized by cluster rank 0 (rank 1: 26)
#ifndef MEGA_SPLIT
#define MEGA_SPLIT 1                           // 0: 64 pairs x 53 rows; 1: pairs 0..49 x 60 rows + pairs 50..63 x 28 rows (the pairs that land last
#endif                                         //    -- on the SMs of the fused MoE's tail CTAs -- get the small share; the early ones preload, qpre)
constexpr int QSPLIT_P = 50, QG_E = 60, QG_L = 28;   // in situ CTAs 100..127 (pairs 50..63) land on the MoE tail SMs at its end
static_assert(QSPLIT_P * QG_E + (64 - QSPLIT_P) * QG_L == QN, "mega split");
__device__ __forceinline__ int q_rows(int p) { return MEGA_SPLIT ? (p < QSPLIT_P ? QG_E : QG_L) : QG; }
__device__ __forceinline__ int q_row0(int p) { return MEGA_SPLIT ? (p < QSPLIT_P ? QG_E * p : QG_E * QSPLIT_P + QG_L * (p - QSPLIT_P)) : QG * p; }
__device__ __forceinline__ int q_pair(int r) { return MEGA_SPLIT ? (r < QG_E * QSPLIT_P ? r / QG_E : QSPLIT_P + (r - QG_E * QSPLIT_P) / QG_L) : r / QG; }
constexpr int G_W = 8192, G_X = M * QBK, G_SX = M * 4, G_ST = 12288;   // <= 60 x 128 B weight box (1024-aligned slot) + x tile
#ifndef MEGA_LL
#define MEGA_LL 0                              // 1: qkv rows handed to the attention CTAs as LL messages {6 bf16, epoch} (no release fence, no flag
#endif                                         //    round trip: the consumers' tag polls ARE the data loads); 0: bf16 rows + per-CTA release flags
#ifndef MEGA_LATE_GATHER
#define MEGA_LATE_GATHER 1                     // 1: attention CTAs compute their window slots in the prologue but issue the K / V gathers after their
#endif                                         //    GEMM loop (L2-resident rows, landing during the handoff) -> their GEMM ring can use 16 stages
#ifndef MEGA_KVPF_END
#define MEGA_KVPF_END 0                        // 1: the next SWA layer's K / V L2 prefetch at the kernel end; 0: right after the attention release
#endif
#ifndef MEGA_KTREADY
#define MEGA_KTREADY 1                         // 1: each o_proj attention k-tile is TMA'd as soon as ITS 4 producer CTAs released (thread kt of warp kt % 8
#endif                                         //    polls them), instead of after all 64 attention flags + one thread issuing 13 TMAs
#ifndef MEGA_PAIRS
#define MEGA_PAIRS 1                           // 1: each warpgroup consumes unit pairs (one fence / commit / wait per 2 k-blocks); 0: single units
#endif                                         //    alternating between the warpgroups (the e4c5053e41 consumer)
#ifndef MEGA_PSTAGE
#define MEGA_PSTAGE 1                          // 1: GEMM ring of PAIR stages [W k-block a | W b | x a | x b] (24 KB), each side loaded by ONE 3D TMA
#endif                                         //    (box {128, rows, 2 k-blocks}): half the TMA issues (in situ the x producer's ~190 ns per TMA bounded the
                                               //    loop) and one full barrier per unit pair; early pairs load 64-row W boxes (4 rows of the neighbour pair)
#ifndef MEGA_PROD_LOOP
#define MEGA_PROD_LOOP 1                       // 1: the GEMM producer loops stay loops (#pragma unroll 1): unrolled, every unit's TMA issue is fresh code,
#endif                                         //    and the kernel's code is cold every launch (the MoE in between evicts it) -> ~350 cycles per unit
#if MEGA_PROD_LOOP
#define MEGA_PROD_UNROLL _Pragma("unroll 1")
#else
#define MEGA_PROD_UNROLL
#endif
#ifndef MEGA_XFLAG
#define MEGA_XFLAG 0                           // 1: with P.xflags (the fused MoE's FMOE_NN_XFLAG row flags) the x producer starts on the 32 x-row flags of the
#endif                                         //    producing MoE launch (or griddepcontrol.wait, whichever is first) instead of griddepcontrol.wait alone.
                                               //    MEASURED NO GAIN (8-GPU lab, MoE stamps build): the x producer passed its wait at the same +1.0 us after the
                                               //    last row CTA as with griddepcontrol.wait (flag release + poll ~= PDL completion latency); per layer +0.0..+1.4 us
                                               //    (a 1-lane 8 x v4 poll of the flag line during the MoE tail cost 1.4 us). The MoE side (FMOE_NN_XFLAG row-CTA
                                               //    release) is not in the tree: claude_tmp/goal11/mega/xflag_moe.patch.
#ifndef MEGA_XFLAG_NS
#define MEGA_XFLAG_NS 128                      // x-flag poll back-off (ns)
#endif
#ifndef MEGA_MID_GATHER
#define MEGA_MID_GATHER 0                      // 1 (pair stages + late gathers): attention CTAs issue their K / V window gathers inside the last pair-loop
#endif                                         //    iteration -- the second lap only uses pair stages 0..3, the attention scratch = pair stages 4..7 is dead
                                               //    once pairs 4..7 are finished -- instead of after the loop (~0.8 us of cp.async issue on the critical path)
#ifndef MEGA_GWARP
#define MEGA_GWARP 0                           // 1 (pair stages + late gathers): a 12th warp issues the attention CTAs' K / V window gathers (cp.async) as soon
#endif                                         //    as pair stages 4..7 are dead (first lap consumed) -- off the math warps (~1 us of their issue) and
                                               //    under the GEMM's second lap. MEASURED +7.5 us (1-GPU lab): one warp's cp.async memory-level
                                               //    parallelism cannot carry 82 KB of gathers (bulk copies: ~130 issue cycles each x 160). N > 1 = N gather
                                               //    warps; 8-GPU lab (L2-warm K/V): 1 warp attention inputs +6 us, 2 warps +1.5 us, 3 warps +0.3 us (the
                                               //    attention CTAs' qkv rows 0.3-0.8 us earlier, layer +0.4 us). Off.
#ifndef MEGA_OPF
#define MEGA_OPF 0                             // 1: once the GEMM loops are done, the o_proj producer L2-prefetches the weight k-tiles that only get an smem stage late
#endif                                         //    (others: 13..15 after stages free up; attention CTAs: 8..15 after attention) -- without the MoE tail's o_proj
                                               //    prefetch those reloads were HBM round trips at the end of the o_proj. MEASURED +1.3 us per layer
                                               //    (8-GPU lab): the 8 MB of prefetches at the GEMM end delayed the qkv row stores / flags. Off.
#ifndef MEGA_WPF
#define MEGA_WPF 0                             // 1 (pair stages): right after its wait (MoE done), the W producer L2-prefetches the qkv weight pairs that were not
#endif                                         //    preloaded (they get a ring stage only once the first lap is consumed: an HBM round trip at the loop end)
                                               //    2: issue those prefetches before the wait (under the MoE tail). MEASURED (1): +0.8..1.0 us per layer
                                               //    (8-GPU lab): x tiles landed 0.5 us later behind the prefetch traffic. Off.
#ifndef MEGA_LOOPDBG
#define MEGA_LOOPDBG 0                         // 1: extra GEMM-loop stamps (warpgroup 0 thread 0: slots 15 / 18 / 19 / 20 = after the first issue / pair-loop
#endif                                         //    iterations q = 2, 4 / after the last wait; W / x producer done issuing: 23 / 24)
#ifndef MEGA_ATT_EARLY_W
#define MEGA_ATT_EARLY_W 0                     // 1: attention CTAs issue their o_proj weight k-tiles 0..7 right after their GEMM loop (not after att_in):
#endif                                         //    without the o_proj tail prefetch they stream from HBM and need the lead time
#ifndef MEGA_KCH
#define MEGA_KCH 1                             // independent accumulator chains per k-block (1 = DeepGEMM's single chain)
#endif
#ifndef MEGA_ROT
#define MEGA_ROT 5                             // k-block order of pair p starts at (p * MEGA_ROT) % 24 (x-tile hot spot); 0 = all in order
#endif
constexpr int GNS_ATT = 10, GNS_OTH = 16;      // GEMM ring stages: attention CTAs below OFF_ATT, the others below OFF_X
#ifndef MEGA_GNS_ATT
#define MEGA_GNS_ATT (MEGA_LATE_GATHER ? 16 : (MEGA_PAIRS ? 8 : 10))   // GEMM ring depth of the attention CTAs (early gathers: K/V window above OFF_ATT)
#endif
#ifndef MEGA_GNS_OTH
#define MEGA_GNS_OTH 16                        // GEMM ring depth of the other CTAs (<= GNS_OTH)
#endif
static_assert((MEGA_LATE_GATHER || MEGA_GNS_ATT <= GNS_ATT) && MEGA_GNS_OTH <= GNS_OTH && MEGA_GNS_ATT % 2 == 0 && MEGA_GNS_OTH % 2 == 0, "mega ring depth");
// every ring stage must always be consumed by the SAME warpgroup (it then waits for lap k only after it consumed lap k - 1 of that stage):
// a warpgroup two laps ahead of a stage whose data has not landed yet would see the parity of lap k as 'complete' (mbarrier parity aliasing)
// -> E2E GPU hang g11m_m7_eg_r1 (pairs + a 10-stage ring: stage s alternated between the warpgroups from lap to lap)
static_assert(MEGA_PAIRS ? (MEGA_GNS_ATT % 4 == 0 && MEGA_GNS_OTH % 4 == 0) : (MEGA_GNS_ATT % 2 == 0 && MEGA_GNS_OTH % 2 == 0), "ring stage -> warpgroup");
static_assert(!MEGA_PSTAGE || MEGA_PAIRS, "pair stages need the unit-pair consumer");
static_assert(!MEGA_GWARP || (MEGA_PSTAGE && MEGA_LATE_GATHER && !MEGA_MID_GATHER && MEGA_GNS_ATT == 16 && OFF_ATT >= 4 * 2 * G_ST && QKH / 2 == 12),
              "gather warp: 12 pairs over 8 pair stages, the attention scratch within pair stages 4..7 (+ above the ring)");
static_assert(!MEGA_MID_GATHER || (MEGA_PSTAGE && MEGA_LATE_GATHER && MEGA_GNS_ATT == 16 && OFF_ATT >= 4 * 2 * G_ST && QKH / 2 == 12),
              "mid-loop gathers: 12 pairs over 8 pair stages, the attention scratch within pair stages 4..7 (+ above the ring)");
static_assert(!MEGA_PSTAGE || (2 * G_W + 2 * G_X == 2 * G_ST && G_W == 64 * 128), "pair stage = 2 x (64-row W slot) + 2 x tile");
constexpr int OFF_RED_ATT = GNS_ATT * G_ST, OFF_RED_OTH = GNS_OTH * G_ST;   // [64 rows][32 tok] fp32 warpgroup-1 partial (8 KB)
constexpr int LL_MSG = 6;                      // LL slots per (token, CTA): ceil(27 / 6) = 5 messages + 1 pad (a CTA's slots = whole 32-B sectors)
constexpr int OFF_FIN = 0;                     // [32 tok][30] bf16: this CTA's finalized qkv rows (staging for the LL messages; dead GEMM ring)
constexpr int OFF_QS = 4096;                   // [8 tok][QS_LD] bf16: an attention CTA's qkv slice (unpacked LL messages; dead GEMM ring)
constexpr int QS_LD = 704;                     // Q head 2 hg (0..191) | Q head 2 hg + 1 (192..383) | new K (384..575) | new V (576..703)
constexpr int MG_NTHREADS = NTHREADS + 64 + 32 * MEGA_GWARP;   // + warp 9: GEMM weight TMA producer, warp 10: x TMA producer, [warp 11: gathers] (a TMA issue costs ~100 cycles
                                               //   of the issuing thread: one thread issuing W + x + scales per unit could not keep up)
constexpr int MG_OFF_BAR = OFF_X + COLS * M * 4;
constexpr int MG_OFF_MISC = MG_OFF_BAR + 1024;
constexpr int MG_OFF_QSW = MG_OFF_MISC + 64;   // [2][24] fp32: the weight block scales of this CTA's (<= 2) scale row blocks x its 24 k-blocks
constexpr int MG_OFF_SX = MG_OFF_QSW + 256;    // [24][32] fp32: the x scales of this CTA's K half (one bulk copy after griddepcontrol.wait)
constexpr int MG_OFF_XG = MG_OFF_SX + QKB / 2 * M * 4;   // [2 warpgroups][<= 30 rows][32 tok] fp32: the partner's partials of my finalized rows
constexpr int MG_SMEM_ALLOC = MG_OFF_XG + 2 * 30 * M * 4 + 1024;
static_assert(MG_SMEM_ALLOC <= 232448 && OFF_RED_ATT + 8192 <= OFF_ATT && OFF_RED_OTH + 8192 <= OFF_X && QG * 64 == QN && QG_E * QBK <= G_W && G_W + G_X <= G_ST, "mega smem");
static_assert(!(MEGA_SPLIT && MEGA_LL), "MEGA_LL assumes the uniform 53-row split");
static_assert(QFIN0 * 2 >= QG && COLS >= QFIN0, "mega exchange fits OFF_X");

struct MegaParams : Params {
    CUtensorMap tm_qw;                 // qkv weight [3392][6144] e4m3: box {128 K, 53 (MEGA_SPLIT: 60) rows}, SWIZZLE_128B
    CUtensorMap tm_qw2;                // MEGA_SPLIT: box {128 K, 32 rows} (the late pairs)
    CUtensorMap tm_qx;                 // x [32][6144] e4m3: box {128, 32}, SWIZZLE_128B
    const float* qsx;                  // x scales, column-major: k-group g, token t at qsx[g * qsx_ld + t]
    const float* qsw;                  // weight block scales [27][48] fp32
    __nv_bfloat16* qout;               // [32][3392] bf16 qkv (written by phase 0, read by the attention CTAs after the flags)
    unsigned* qflags;                  // [128] qkv-rows-ready flags (epoch), one per 128-B line (MEGA_LL 0)
    uint4* qll;                        // [32 tok][128 CTA][LL_MSG] LL messages {6 bf16, epoch} of the finalized qkv rows (MEGA_LL 1)
    int qsx_ld, qpre;                  // qpre: weight units TMA-issued before griddepcontrol.wait (<= ring stages)
    const unsigned* xflags;            // MEGA_XFLAG: [32] x-row-ready flags of the producing fused MoE (epoch = its work[2] + 1), nullptr = off
    const int* xwork;                  // that MoE's work counters (work[2] = launches completed)
};

__device__ __forceinline__ uint64_t desc_sw128_q(uint32_t addr) {   // fp8 K-major SW128 (128-B rows, 8-row atoms: SBO 1024)
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t desc_lo(uint32_t lo, uint32_t hi) { uint64_t d; asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "r"(lo), "r"(hi)); return d; }
__device__ __forceinline__ void wgmma_m64n32k32_e4m3(float* d, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n"
                 " wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void tma_load_2d_last(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n"
                 " cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], policy;\n}"
                 :: "r"(sa(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d_pol(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint64_t pol) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(sa(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(sa(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_3d_pol(void* dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar, uint64_t pol) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :: "r"(sa(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(sa(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ bool mbar_test(uint64_t* b, unsigned ph) {
    unsigned r;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(r) : "r"(sa(b)), "r"(ph) : "memory");
    return r != 0;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void st_relaxed_v4(uint4* p, uint4 v) {
    asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_v4(const uint4* p) {
    uint4 v; asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v;
}
// the qkv rows read by the attention CTA (b, hg, kh): Q of heads 2 hg, 2 hg + 1 (+ the new K / V rows for kh 1) -> row groups
__device__ __forceinline__ void mega_groups(int hg, int kh, int& qa, int& qb, int& ka, int& kb) {
    qa = q_pair(2 * DQK * hg); qb = q_pair(2 * DQK * hg + 2 * DQK - 1);
    ka = kh ? q_pair(NH * DQK) : 1; kb = kh ? q_pair(QN - 1) : 0;   // (ka > kb: none)
}

// MEGA_LL consumer: poll the LL messages of the qkv rows this attention CTA reads (tokens 8 b .. 8 b + 7; Q of heads 2 hg, 2 hg + 1; kh 1:
// also the new K / V rows) and unpack them into qs [8][QS_LD]. One message = 6 consecutive finalized rows of one producer CTA.
__device__ __forceinline__ void mega_ll_gather(const MegaParams& P, uint8_t* smem, int b, int hg, int kh, unsigned epoch) {
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_QS);
    int qa, qb, ka, kb;
    mega_groups(hg, kh, qa, qb, ka, kb);
    if (ka <= qb) ka = qb + 1;                     // hg 7: the K/V groups start inside the Q range
    const int nq = qb - qa + 1, nk = ka <= kb ? kb - ka + 1 : 0, ng = nq + nk;
    const int per_tok = ng * 2 * 5;                // groups x K halves x 5 messages
    const int nmsg = VT * per_tok;                 // <= 8 x 16 x 10 = 1280 -> <= 5 per thread
    constexpr int MPT = 5;
    // one sentinel poll per producer CTA first (2 ng threads, backoff): 64 x 256 threads spinning on the whole message set flood L2 while
    // the last producers still stream their GEMM operands; then every message is loaded once (stragglers re-polled)
    if (threadIdx.x < 2 * ng) {
        const int gi = threadIdx.x >> 1, grp = gi < nq ? qa + gi : ka + (gi - nq);
        const uint4* sp = P.qll + ((size_t)(8 * b + VT - 1) * NCTA + 2 * grp + (threadIdx.x & 1)) * LL_MSG + 4;
        while (ld_relaxed_v4(sp).w != epoch) __nanosleep(64);
    }
    named_bar_sync(1, NMATH);
    if (threadIdx.x == 0) BSTAMP(21);
    uint4 v[MPT];
    const uint4* src[MPT];
    // all of this thread's message loads in flight at once (a per-message poll loop paid one L2 round trip per message)
#pragma unroll
    for (int k = 0; k < MPT; ++k) {
        const int idx = threadIdx.x + k * NMATH;
        src[k] = nullptr;
        if (idx < nmsg) {
            const int t = idx / per_tok, r = idx - t * per_tok;
            const int gi = r / 10, hj = r - gi * 10, hh = hj / 5, j = hj - hh * 5;
            const int grp = gi < nq ? qa + gi : ka + (gi - nq);
            src[k] = P.qll + ((size_t)(8 * b + t) * NCTA + 2 * grp + hh) * LL_MSG + j;
            v[k] = ld_relaxed_v4(src[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < MPT; ++k)
        if (src[k]) while (v[k].w != epoch) { __nanosleep(20); v[k] = ld_relaxed_v4(src[k]); }
    if (threadIdx.x == 0) BSTAMP(22);
#pragma unroll
    for (int k = 0; k < MPT; ++k) {
        const int idx = threadIdx.x + k * NMATH;
        if (idx >= nmsg) break;
        const int t = idx / per_tok, r = idx - t * per_tok;
        const int gi = r / 10, hj = r - gi * 10, hh = hj / 5, j = hj - hh * 5;
        const int grp = gi < nq ? qa + gi : ka + (gi - nq);
        const uint32_t wv[3] = {v[k].x, v[k].y, v[k].z};
        const int lr0 = (hh ? QFIN0 : 0) + 6 * j, lr_end = hh ? QG : QFIN0;
#pragma unroll
        for (int e = 0; e < 6; ++e) {
            const int lr = lr0 + e;
            if (lr >= lr_end) break;
            const int col = grp * QG + lr;
            int lc = -1;
            if (col >= 2 * DQK * hg && col < 2 * DQK * (hg + 1)) lc = col - 2 * DQK * hg;
            else if (kh && col >= NH * DQK) lc = 2 * DQK + (col - NH * DQK);
            if (lc >= 0) {
                const uint32_t w = wv[e >> 1];
                qs[t * QS_LD + lc] = __ushort_as_bfloat16((unsigned short)((e & 1) ? (w >> 16) : (w & 0xffff)));
            }
        }
    }
    named_bar_sync(1, NMATH);
}

// K / V window gathers of a key-split attention CTA (attention_ks's slot + gather code), issued in the prologue (cp.async, not waited)
template <int WINDOW>
__device__ __forceinline__ void mega_gathers(const MegaParams& P, uint8_t* smem, int b, int hg, int kh) {
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, NPAD = (NKEY + 15) / 16 * 16, MPAD = 16;
    constexpr int R0H = 64, NRMAX = NPAD - R0H;
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    constexpr int KCH = DQK / 8, VCH = DV / 8;
    constexpr int NOLD = NKEY - W;
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NRMAX * KP;
    __nv_bfloat16* Ps = Vs + (NRMAX + 16) * VP;
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    int* s_slot = reinterpret_cast<int*>(rsum + NW * MPAD);
    const int tid = threadIdx.x;
    const int R0 = kh ? R0H : 0, NR = kh ? NRMAX : R0H;
    const int seqlen = P.seqlens[b];
    const bool slots_read = P.mode & MODE_SLOTS_READ;
    const int base_pos = seqlen - W - (WINDOW - 1);
    if (tid < NR) {
        const int r = R0 + tid;
        int v;
        if (slots_read) v = P.slot_cache[b * NPAD + r];
        else {
            const int p = base_pos + r;
            v = (r < NOLD && p >= 0) ? P.pt[b * P.pt_stride + p] : -1;
            if ((P.mode & MODE_SLOTS_WRITE) && hg == 0) P.slot_cache[b * NPAD + r] = v;
        }
        s_slot[tid] = v;
    }
    named_bar_sync(1, NMATH);
#if MEGA_LATE_GATHER
}
// the K / V window cp.async gathers of a key-split attention CTA (slots from mega_gathers), issued once its GEMM ring is dead
template <int WINDOW>
__device__ __forceinline__ void mega_gather_issue(const MegaParams& P, uint8_t* smem, int kh) {
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, NPAD = (NKEY + 15) / 16 * 16, MPAD = 16;
    constexpr int R0H = 64, NRMAX = NPAD - R0H;
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    constexpr int KCH = DQK / 8, VCH = DV / 8;
    constexpr int NOLD = NKEY - W;
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NRMAX * KP;
    __nv_bfloat16* Ps = Vs + (NRMAX + 16) * VP;
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    const int* s_slot = reinterpret_cast<const int*>(rsum + NW * MPAD);
    const int tid = threadIdx.x;
    const int R0 = kh ? R0H : 0, NR = kh ? NRMAX : R0H;
#endif
    {
        constexpr int CPR = KCH + VCH, NG = NMATH / CPR, RPG = (NRMAX + NG - 1) / NG;
        const int c = tid % CPR, gq = tid / CPR;
        if (gq < NG) {
            const bool isk = c < KCH;
            const __nv_bfloat16* gbase = isk ? P.kc + c * 8 : P.vc + (c - KCH) * 8;
            const long long gstr = isk ? P.ks : P.vs;
            __nv_bfloat16* sbase = isk ? Ks + c * 8 : Vs + (c - KCH) * 8;
            const int sstr = isk ? KP : VP;
            int sl[RPG];
#pragma unroll
            for (int i = 0; i < RPG; ++i) sl[i] = (gq + NG * i < NR) ? s_slot[gq + NG * i] : -2;
#pragma unroll
            for (int i = 0; i < RPG; ++i) {
                const int lr = gq + NG * i, r = R0 + lr;
                if (sl[i] >= 0) cp16(sbase + lr * sstr, gbase + (size_t)sl[i] * gstr);
                else if (sl[i] == -1 && (r < NOLD || r >= NKEY)) *reinterpret_cast<uint4*>(sbase + lr * sstr) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    cp_commit();
}

#if MEGA_GWARP
// one warp issues the K / V window gathers (the same rows / layout as mega_gather_issue) and signals gath_full
template <int WINDOW>
__device__ __forceinline__ void mega_gather_warp(const MegaParams& P, uint8_t* smem, int kh, int gt, uint64_t* gath_full) {
    constexpr int NGT = 32 * MEGA_GWARP;   // gather threads
    const int lane = gt & 31;
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, MPAD = 16;
    constexpr int NPAD = (NKEY + 15) / 16 * 16, R0H = 64, NRMAX = NPAD - R0H;
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    constexpr int KCH = DQK / 8, VCH = DV / 8, CPR = KCH + VCH;
    constexpr int NOLD = NKEY - W;
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NRMAX * KP;
    __nv_bfloat16* Ps = Vs + (NRMAX + 16) * VP;
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    const int* s_slot = reinterpret_cast<const int*>(rsum + NW * MPAD);
    const int R0 = kh ? R0H : 0, NR = kh ? NRMAX : R0H;
    // zero rows first (padding keys outside the window), then the plain arrive (release: orders the warp's zero stores)
    for (int i = gt; i < NR * CPR; i += NGT) {
        const int lr = i / CPR, c = i - lr * CPR, r = R0 + lr;
        if (s_slot[lr] == -1 && (r < NOLD || r >= NKEY)) {
            __nv_bfloat16* d = c < KCH ? Ks + lr * KP + c * 8 : Vs + lr * VP + (c - KCH) * 8;
            *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(gath_full);
#pragma unroll 4
    for (int i = gt; i < NR * CPR; i += NGT) {
        const int lr = i / CPR, c = i - lr * CPR;
        const int sl = s_slot[lr];
        if (sl >= 0) {
            if (c < KCH) cp16(Ks + lr * KP + c * 8, P.kc + (size_t)sl * P.ks + c * 8);
            else cp16(Vs + lr * VP + (c - KCH) * 8, P.vc + (size_t)sl * P.vs + (c - KCH) * 8);
        }
    }
    cp_arrive_noinc(gath_full);
}
#endif

__device__ __forceinline__ __nv_bfloat16 ldcg_bf(const __nv_bfloat16* p) {
    const unsigned short u = __ldcg(reinterpret_cast<const unsigned short*>(p));
    return __ushort_as_bfloat16(u);
}
// attention_ks from the qkv loads on (the gathers are in flight since mega_gathers); qkv rows written by phase 0 of THIS grid ->
// L2-coherent (.cg) loads after the row-group flags were acquired
template <int WINDOW>
__device__ __forceinline__ void mega_attention(const MegaParams& P, uint8_t* smem, int b, int hg, int kh, uint64_t* att_in, uint64_t* xm_full, uint64_t* xo_full,
                                               uint64_t* gath_full) {
    constexpr int W = VT;
    constexpr int NKEY = WINDOW - 1 + W, NPAD = (NKEY + 15) / 16 * 16, MR = W * HP, MPAD = 16;
    constexpr int R0H = 64, NRMAX = NPAD - R0H;
    constexpr int KP = DQK + 8, VP = DV + 8, PP = NRMAX + 8;
    constexpr int QKT = DQK / 16, DT8 = DV / 8, NTW = 2;
    constexpr int NOLD = NKEY - W;
    static_assert(MR == 16 && NRMAX == 80 && NOLD >= R0H && DT8 == 2 * NW, "key-split geometry");
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_ATT);
    __nv_bfloat16* Ks = Qs + MPAD * KP;
    __nv_bfloat16* Vs = Ks + NRMAX * KP;
    __nv_bfloat16* Ps = Vs + (NRMAX + 16) * VP;
    float* rmax = reinterpret_cast<float*>(Ps + MPAD * PP);
    float* rsum = rmax + NW * MPAD;
    int* s_slot = reinterpret_cast<int*>(rsum + NW * MPAD);
    float* xm = reinterpret_cast<float*>(s_slot + NRMAX);
    float* xl = xm + MPAD;
    float* xo = xl + MPAD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int R0 = kh ? R0H : 0, NT8 = (kh ? NRMAX : R0H) / 8, NK16 = (kh ? NRMAX : R0H) / 16;
    const uint32_t pxm = mapa(sa(xm), (uint32_t)(kh ^ 1)), pxl = mapa(sa(xl), (uint32_t)(kh ^ 1)), pxo = mapa(sa(xo), (uint32_t)(kh ^ 1));
    const uint32_t pbm = mapa(sa(xm_full), (uint32_t)(kh ^ 1)), pbo = mapa(sa(xo_full), (uint32_t)(kh ^ 1));
    const bool pos64 = P.mode & MODE_POS64;
    const int qr = tid >> 4, qq = tid & 15;
    const int qtok = b * W + qr / HP, qh = hg * HP + (qr % HP);
    const int ktok = b * W + warp;
    const int seqlen = P.seqlens[b];
    const long long qpos = ld_idx(P.positions, qtok, pos64), kpos = ld_idx(P.positions, ktok, pos64);
    const long long kslot = ld_idx(P.loc, ktok, P.mode & MODE_LOC64);
    float sk0 = 0.f, sk1 = 0.f;
    if (P.mode & MODE_SINK) { sk0 = P.sinks[hg * HP + ((lane >> 2) % HP)] * LOG2E; sk1 = P.sinks[hg * HP + (((lane >> 2) + 8) % HP)] * LOG2E; }
    const int base_pos = seqlen - W - (WINDOW - 1);
    const __nv_bfloat16* qcs = P.cos_sin + (size_t)qpos * 64;
    const __nv_bfloat16* kcs = P.cos_sin + (size_t)kpos * 64;
    __nv_bfloat16 qx[2], qy[2], qc[2], qsn[2];
    __nv_bfloat16 kx = __float2bfloat16(0.f), ky = kx, kc_ = kx, ks_ = kx;
    uint4 qrest, krest = make_uint4(0u, 0u, 0u, 0u);
#if MEGA_LL
    {
        const __nv_bfloat16* qs = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_QS);
        const __nv_bfloat16* qsrc = qs + (qr / HP) * QS_LD + (qr % HP) * DQK;
        const __nv_bfloat16* ksrc = qs + warp * QS_LD + 2 * DQK;
#pragma unroll
        for (int e = 0; e < 2; ++e) { const int i = 2 * qq + e; qx[e] = qsrc[i]; qy[e] = qsrc[i + 32]; qc[e] = qcs[i]; qsn[e] = qcs[32 + i]; }
        qrest = *reinterpret_cast<const uint4*>(qsrc + 64 + 8 * qq);
        if (kh) {
            kx = ksrc[lane]; ky = ksrc[lane + 32]; kc_ = kcs[lane]; ks_ = kcs[32 + lane];
            krest = *reinterpret_cast<const uint4*>(lane < 16 ? ksrc + 64 + 8 * lane : ksrc + DQK + 8 * (lane - 16));
        }
    }
#else
    {
        const __nv_bfloat16* qsrc = P.qout + (size_t)qtok * QKVW + (size_t)qh * DQK;
        const __nv_bfloat16* ksrc = P.qout + (size_t)ktok * QKVW + NH * DQK;
#pragma unroll
        for (int e = 0; e < 2; ++e) { const int i = 2 * qq + e; qx[e] = ldcg_bf(qsrc + i); qy[e] = ldcg_bf(qsrc + i + 32); qc[e] = qcs[i]; qsn[e] = qcs[32 + i]; }
        qrest = __ldcg(reinterpret_cast<const uint4*>(qsrc + 64 + 8 * qq));
        if (kh) {
            kx = ldcg_bf(ksrc + lane); ky = ldcg_bf(ksrc + lane + 32); kc_ = kcs[lane]; ks_ = kcs[32 + lane];
            krest = __ldcg(reinterpret_cast<const uint4*>(lane < 16 ? ksrc + 64 + 8 * lane : ksrc + DQK + 8 * (lane - 16)));
        }
    }
#endif
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int i = 2 * qq + e;
        __nv_bfloat16 lo, hi;
        rope_pair(qx[e], qy[e], qc[e], qsn[e], lo, hi);
        Qs[qr * KP + i] = lo; Qs[qr * KP + i + 32] = hi;
    }
    *reinterpret_cast<uint4*>(&Qs[qr * KP + 64 + 8 * qq]) = qrest;
    if (kh) {
        const int lr = NOLD - R0 + warp;
        __nv_bfloat16 lo, hi;
        rope_pair(kx, ky, kc_, ks_, lo, hi);
        Ks[lr * KP + lane] = lo; Ks[lr * KP + lane + 32] = hi;
        if (lane < 16) *reinterpret_cast<uint4*>(&Ks[lr * KP + 64 + 8 * lane]) = krest;
        else *reinterpret_cast<uint4*>(&Vs[lr * VP + 8 * (lane - 16)]) = krest;
        if (hg == 0 && kslot > 0 && kslot < P.kv_rows) {
            __nv_bfloat16* kd = P.kc + (size_t)kslot * P.ks;
            kd[lane] = lo; kd[lane + 32] = hi;
            if (lane < 16) *reinterpret_cast<uint4*>(kd + 64 + 8 * lane) = krest;
            else *reinterpret_cast<uint4*>(P.vc + (size_t)kslot * P.vs + 8 * (lane - 16)) = krest;
        }
    }
#if MEGA_GWARP
    mbar_wait(gath_full, 0);   // the gather warp's K / V window rows (cp.async + its zero rows)
#endif
    cp_wait_all();
#if MEGA_LL
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // qs (generic reads) -> the o_proj weight TMAs into that region after att_in
#endif
    named_bar_sync(1, NMATH);
    if (tid == 0) { BSTAMP(6); mbar_arrive(att_in); }
    // ---- S = Q K^T over this half's key tiles (verbatim attention_ks)
    const int rA = lane >> 2, rB = rA + 8, cA = (lane & 3) * 2;
    unsigned qa[QKT][4];
#pragma unroll
    for (int kk = 0; kk < QKT; ++kk) ldm4(qa[kk], &Qs[(lane & 15) * KP + kk * 16 + (lane >> 4) * 8]);
    float acc[NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][i] = 0.f;
    unsigned kb[2][NTW][4];
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt < NT8) ldm4(kb[0][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (lane >> 3) * 8]);
    }
#pragma unroll
    for (int kk = 0; kk < QKT; kk += 2) {
        const int cur = (kk >> 1) & 1;
        if (kk + 2 < QKT) {
#pragma unroll
            for (int ti = 0; ti < NTW; ++ti) {
                const int nt = warp + ti * NW;
                if (nt < NT8) ldm4(kb[cur ^ 1][ti], &Ks[(nt * 8 + (lane & 7)) * KP + (kk + 2) * 16 + (lane >> 3) * 8]);
            }
        }
#pragma unroll
        for (int ti = 0; ti < NTW; ++ti) {
            const int nt = warp + ti * NW;
            if (nt < NT8) { mma16816(acc[ti], qa[kk], kb[cur][ti]); mma16816(acc[ti], qa[kk + 1], kb[cur][ti] + 2); }
        }
    }
    if (tid == 0) BSTAMP(25);
    const float scale2 = P.sm_scale * LOG2E;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt >= NT8) continue;
        const int n0 = nt * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (i < 2) ? rA : rB;
            const int j = R0 + n0 + cA + (i & 1);
            const int t = r / HP;
            const bool ok = (r < MR) && (j >= t) && (j <= t + (WINDOW - 1)) && (j < NKEY) && (base_pos + j >= 0);
            acc[ti][i] = ok ? acc[ti][i] * scale2 : -INFINITY;
        }
        mx0 = fmaxf(mx0, fmaxf(acc[ti][0], acc[ti][1]));
        mx1 = fmaxf(mx1, fmaxf(acc[ti][2], acc[ti][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    if ((lane & 3) == 0) { rmax[warp * MPAD + rA] = mx0; rmax[warp * MPAD + rB] = mx1; }
    named_bar_sync(1, NMATH);
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) { m0 = fmaxf(m0, rmax[w * MPAD + rA]); m1 = fmaxf(m1, rmax[w * MPAD + rB]); }
    if (warp == 0 && (lane & 3) == 0) { st_async_f32(pxm + rA * 4, m0, pbm); st_async_f32(pxm + rB * 4, m1, pbm); }
    mbar_wait(xm_full, 0);
    m0 = fmaxf(m0, xm[rA]); m1 = fmaxf(m1, xm[rB]);
    const bool has_sink = P.mode & MODE_SINK;
    if (has_sink) { m0 = fmaxf(m0, sk0); m1 = fmaxf(m1, sk1); }
    if (!isfinite(m0)) m0 = 0.f;
    if (!isfinite(m1)) m1 = 0.f;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int ti = 0; ti < NTW; ++ti) {
        const int nt = warp + ti * NW;
        if (nt >= NT8) continue;
        const int n0 = nt * 8;
        const float p0 = fast_ex2(acc[ti][0] - m0), p1 = fast_ex2(acc[ti][1] - m0);
        const float p2 = fast_ex2(acc[ti][2] - m1), p3 = fast_ex2(acc[ti][3] - m1);
        s0 += p0 + p1;
        s1 += p2 + p3;
        *reinterpret_cast<unsigned*>(&Ps[rA * PP + n0 + cA]) = pack_bf16(p0, p1);
        *reinterpret_cast<unsigned*>(&Ps[rB * PP + n0 + cA]) = pack_bf16(p2, p3);
    }
    s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
    s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
    if ((lane & 3) == 0) { rsum[warp * MPAD + rA] = s0; rsum[warp * MPAD + rB] = s1; }
    named_bar_sync(1, NMATH);
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) { l0 += rsum[w * MPAD + rA]; l1 += rsum[w * MPAD + rB]; }
    if (tid == 0) BSTAMP(26);
    unsigned pa[5][4];
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) if (kk < NK16) ldm4(pa[kk], &Ps[(lane & 15) * PP + kk * 16 + (lane >> 4) * 8]);
    float o[2][4];
#pragma unroll
    for (int di = 0; di < 2; ++di)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[di][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 5; kk += 2) {
        if (kk >= NK16) break;
        unsigned vb[2][4];
#pragma unroll
        for (int di = 0; di < 2; ++di) ldm4t(vb[di], &Vs[(kk * 16 + lane) * VP + (warp + di * NW) * 8]);
#pragma unroll
        for (int di = 0; di < 2; ++di) {
            mma16816(o[di], pa[kk], vb[di]);
            if (kk + 1 < NK16) mma16816(o[di], pa[kk + 1], vb[di] + 2);
        }
    }
    const int si = kh ^ 1, ki = kh;
    st_async_v2(pxo + (rA * 64 + 8 * warp + cA) * 4, o[si][0], o[si][1], pbo);
    st_async_v2(pxo + (rB * 64 + 8 * warp + cA) * 4, o[si][2], o[si][3], pbo);
    if (warp == 0 && (lane & 3) == 0) { st_async_f32(pxl + rA * 4, l0, pbo); st_async_f32(pxl + rB * 4, l1, pbo); }
    mbar_wait(xo_full, 0);
    float lt0 = kh ? xl[rA] + l0 : l0 + xl[rA], lt1 = kh ? xl[rB] + l1 : l1 + xl[rB];
    if (has_sink) { lt0 += fast_ex2(sk0 - m0); lt1 += fast_ex2(sk1 - m1); }
    const float inv0 = (lt0 > 0.f) ? fast_rcp(lt0) : 0.f;
    const float inv1 = (lt1 > 0.f) ? fast_rcp(lt1) : 0.f;
    const float2 pa0 = *reinterpret_cast<const float2*>(xo + rA * 64 + 8 * warp + cA);
    const float2 pa1 = *reinterpret_cast<const float2*>(xo + rB * 64 + 8 * warp + cA);
    const int dim = 64 * kh + 8 * warp + cA;
    {
        const int t = rA / HP, h = hg * HP + (rA % HP);
        *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + dim]) =
            pack_bf16((o[ki][0] + pa0.x) * inv0, (o[ki][1] + pa0.y) * inv0);
    }
    {
        const int t = rB / HP, h = hg * HP + (rB % HP);
        *reinterpret_cast<unsigned*>(&P.attn_out[(size_t)(b * W + t) * KO + (size_t)h * DV + dim]) =
            pack_bf16((o[ki][2] + pa1.x) * inv1, (o[ki][3] + pa1.y) * inv1);
    }
}

// goal11 layer: the megakernel body as a device function. FUSED = phase 1 of layer_fused.cu's one-kernel SWA layer (attention block +
// fused MoE): run by warps 0..10 of CTAs 0..127 of the 640-thread layer kernel (warps 11..19 and CTAs 128..131 mirror its two cluster
// syncs). FUSED differences: the own rank's o_proj rows are pushed as LL messages into this rank's IN_RS like the peers' (no kernel
// boundary orders P.partial any more), the math warps of CTAs 0..LAYER_TOKC-1 arrive on LAYER_PUSH_BAR once their pushes are issued (the
// MoE token reduce runs on the layer kernel's idle warps 12..19 of those CTAs), and no early griddepcontrol.launch_dependents.
constexpr int LAYER_TOKC = 32, LAYER_PUSH_BAR = 9;
__device__ __forceinline__ void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
// ROLE: 0 = every role (standalone kernel); FUSED layer kernel: 1 = the 8 math warps, 2 = the 3 producer warps (separate inlined copies:
// the layer kernel's setmaxnreg phases give them different register budgets, which ptxas assigns per disjoint branch)
template <int WINDOW, bool FUSED, int ROLE = 0>
__device__ __forceinline__ void mega_body(const MegaParams& P, uint8_t* smem) {
    constexpr bool R_MATH = ROLE != 2, R_PROD = ROLE != 1;
    uint64_t* wfull = reinterpret_cast<uint64_t*>(smem + MG_OFF_BAR);
    uint64_t* bfull = wfull + NS;
    uint64_t* sempty = bfull + NS;
    uint64_t* att_in = sempty + NS;
    uint64_t* att_done = att_in + 1;
    uint64_t* xfull = att_done + 1;
    uint64_t* xm_full = xfull + 1;
    uint64_t* xo_full = xm_full + 1;
    uint64_t* bgo = xo_full + 1;
    uint64_t* gfull = bgo + 1;                 // [GNS_OTH] GEMM stage landed (W + x + sx)
    uint64_t* gempty = gfull + GNS_OTH;        // [GNS_OTH] GEMM stage consumed (the 4 warps of the consuming warpgroup)
    uint64_t* gdone = gempty + GNS_OTH;        // this CTA's GEMM done (8 math warps): the ring region is free for the o_proj weights
    uint64_t* xg_full = gdone + 1;             // the partner's K-half partial of my finalized rows landed (st.async)
    uint64_t* ep_bar = xg_full + 1;            // s_epoch written (o_proj producer lane 0, right after its griddepcontrol.wait)
    uint64_t* sx_full = ep_bar + 1;            // the K half's x scales landed (x producer)
    uint64_t* qready = sx_full + 1;            // this CTA's qkv rows stored (8 math warps) -> the W producer releases the CTA's flag
    uint64_t* kt_seen = qready + 1;            // MEGA_KTREADY: the producers of o_proj k-tiles NS..NKT-1 seen (math lanes) -> producer lane 1
    uint64_t* xgo = kt_seen + 1;               // x may be read (x producer: row flags seen or griddepcontrol.wait returned) -> W producer's rest
    uint64_t* slot_ready = xgo + 1;            // MEGA_GWARP: the K / V window slots are in smem (math tid 0 after mega_gathers' barrier)
    uint64_t* gath_full = slot_ready + 1;      // MEGA_GWARP: the gather warp's K / V rows landed (32 lanes' cp.async arrivals + 1 for its zero rows)
    unsigned* s_epoch = reinterpret_cast<unsigned*>(smem + MG_OFF_MISC);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c = blockIdx.x;
    const int h = (int)cluster_rank(), p = c / CL;
    const int kbase = h * (KO / CL), rbase = p * PCOLS;
    const bool qonly = P.mode & MODE_QKV_ONLY;   // FA layers: the qkv GEMM alone (rows into qout), no attention / o_proj
    const bool att = c < NATT && !qonly;
    const int gns = att ? MEGA_GNS_ATT : MEGA_GNS_OTH;   // ring stage / lap parity tracked incrementally (no runtime division)
    const int qrow0 = q_row0(p), qg = q_rows(p), qmid = (qg + 1) / 2;
    const int fin0 = h ? qmid : 0, fin1 = h ? qg : qmid;   // my qkv rows [qrow0, qrow0 + qg), the rows I finalize [fin0, fin1)
    if (R_MATH && tid == 0) {
        BSTAMP(0);
        for (int i = 0; i < NS; ++i) { mbar_init(&wfull[i], 1); mbar_init(&bfull[i], 1); mbar_init(&sempty[i], NW); }
        for (int i = 0; i < GNS_OTH; ++i) { mbar_init(&gfull[i], 2); mbar_init(&gempty[i], 4); }   // gfull: W producer + x producer
        mbar_init(gdone, NW); mbar_init(xg_full, 1); mbar_init(ep_bar, 1); mbar_init(sx_full, 1); mbar_init(qready, NW); mbar_init(kt_seen, NKT - NS); mbar_init(xgo, 1); mbar_init(slot_ready, 1); mbar_init(gath_full, 33 * (MEGA_GWARP ? MEGA_GWARP : 1));
        mbar_init(att_in, 1); mbar_init(att_done, 1); mbar_init(xfull, 1); mbar_init(xm_full, 1); mbar_init(xo_full, 1); mbar_init(bgo, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect(xfull, COLS * M * 4);
        mbar_expect(xg_full, 2 * (fin1 - fin0) * M * 4);   // both of the partner's warpgroups send
        if (att) { mbar_expect(xm_full, 16 * 4); mbar_expect(xo_full, 16 * 64 * 4 + 16 * 4); }
    }
    cluster_sync_relaxed();
    float d[NACC][16];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < 16; ++i) d[a][i] = 0.f;
    if (R_PROD && warp == NW) {
        auto issue_w = [&](int kt) {
            const int s = kt % NS;
            mbar_expect(&wfull[s], WT_BYTES);
            tma_load_2d(smem + s * ST_BYTES, &P.tm_w, kbase + kt * KT, rbase, &wfull[s]);
        };
        if (lane == 0) {
            // the epoch (fused MoE launch counter + 1) of every flag of this launch: a dependent load after griddepcontrol.wait, off the
            // GEMM producers' issue paths
            griddep_wait();
            s_epoch[0] = (unsigned)(*reinterpret_cast<const volatile int*>(P.work + 2)) + 1u;
            mbar_arrive(ep_bar);
            // o_proj weights after this CTA's GEMM (its ring region); attention CTAs after their attention inputs are in smem
            if (!qonly) asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_w)) : "memory");
#if MEGA_OPF && !MEGA_LL
            if (!qonly) {
                mbar_wait(gdone, 0);   // GEMM loops done (attention CTAs arrive it too with MEGA_OPF)
#pragma unroll 1
                for (int kt = att ? NS0_ATT : NS; kt < NKT; ++kt)
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 :: "l"(reinterpret_cast<uint64_t>(&P.tm_w)), "r"(kbase + kt * KT), "r"(rbase) : "memory");
            }
#endif
            // MEGA_ATT_EARLY_W: attention CTAs too take stages 0..7 right after their GEMM (only stages 8..12 hold their attention scratch)
            if (qonly) {} else if (att && (MEGA_LL || !MEGA_ATT_EARLY_W)) mbar_wait(att_in, 0); else mbar_wait(gdone, 0);
            for (int kt = 0; kt < NKT && !qonly; ++kt) {
                const int s = kt % NS;
                if (att && kt == NS0_ATT) mbar_wait(att_done, 0);
                if (kt >= NS) mbar_wait(&sempty[s], ((kt / NS) - 1) & 1);
                if (kt >= BLK_WD) mbar_wait(&wfull[(kt - BLK_WD) % NS], ((kt - BLK_WD) / NS) & 1);
                issue_w(kt);
            }
        } else if (lane == 1 && !qonly) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_a)) : "memory");
            mbar_wait(bgo, 0);
            asm volatile("fence.proxy.async.global;" ::: "memory");
            for (int kt = NS; kt < NKT; ++kt) {
                const int s = kt % NS;
                mbar_wait(&sempty[s], ((kt / NS) - 1) & 1);
#if MEGA_KTREADY
                if (kt == NS) { mbar_wait(kt_seen, 0); asm volatile("fence.proxy.async.global;" ::: "memory"); }   // producers of k-tiles NS.. seen
#endif
                mbar_expect(&bfull[s], BT_BYTES);
                tma_load_2d_nohint(smem + s * ST_BYTES + WT_BYTES, &P.tm_a, kbase + kt * KT, 0, &bfull[s]);
            }
        }
        __syncwarp();
    } else if (R_PROD && warp == NW + 1) {
        if (lane == 0) {
            // GEMM weight producer: independent of the previous kernel (qpre units before griddepcontrol.wait)
            const CUtensorMap* wmap_q = (MEGA_SPLIT && p >= QSPLIT_P) ? &P.tm_qw2 : &P.tm_qw;
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(wmap_q)) : "memory");
            uint64_t pol;
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
            const int pre = P.qpre < gns ? P.qpre : gns;
            const int kb0 = h * QKH, rot = MEGA_PSTAGE ? ((p * MEGA_ROT) % (QKH / 2)) * 2 : (p * MEGA_ROT) % QKH;
            const bool now_ = P.mode & MODE_DBG_GNOW;
            int s = 0, lp = 0;
#if MEGA_PSTAGE
            // pair j = k-blocks (kj, kj + 1) of the K half (rot even: never wraps inside a pair) -> pair stage j % (gns / 2), one 3D box
            const int nps = gns / 2, prep = pre / 2;
            const unsigned wbytes = (MEGA_SPLIT && p >= QSPLIT_P) ? 2u * 32u * QBK : 2u * 64u * QBK;
            MEGA_PROD_UNROLL
            auto wprefetch = [&]() {
#pragma unroll 1
                for (int jj = prep < nps ? nps : prep; jj < QKH / 2; ++jj) {
                    int kq = 2 * jj + rot; if (kq >= QKH) kq -= QKH;
                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];"
                                 :: "l"(reinterpret_cast<uint64_t>(wmap_q)), "r"(0), "r"(qrow0), "r"(kb0 + kq) : "memory");
                }
            };
            if (MEGA_WPF == 2 && !now_) wprefetch();
            for (int j = 0; j < QKH / 2; ++j) {
                int kj = 2 * j + rot; if (kj >= QKH) kj -= QKH;
                if (j == prep) {
                    if (MEGA_XFLAG) mbar_wait(xgo, 0); else griddep_wait();
                    if (MEGA_WPF == 1 && !now_) wprefetch();
                }
                if (j >= nps) mbar_wait(&gempty[s], lp ^ 1);
                mbar_expect(&gfull[s], now_ ? 0u : wbytes);
                if (!now_) tma_load_3d_pol(smem + s * 2 * G_ST, wmap_q, 0, qrow0, kb0 + kj, &gfull[s], pol);
#if MEGA_LOOPDBG
                if (j == 7) BSTAMP(26);
                if (j == 0) BSTAMP(21);
#endif
                if (++s == nps) { s = 0; lp ^= 1; }
            }
#if MEGA_LOOPDBG
            BSTAMP(23);
#endif
            if (prep >= QKH / 2) { if (MEGA_XFLAG) mbar_wait(xgo, 0); else griddep_wait(); }
#else
            for (int i = 0; i < QKH; ++i) {
                int ki = i + rot; if (ki >= QKH) ki -= QKH;
                if (i == pre) { if (MEGA_XFLAG) mbar_wait(xgo, 0); else griddep_wait(); }
                if (i >= gns) mbar_wait(&gempty[s], lp ^ 1);
                mbar_expect(&gfull[s], now_ ? 0u : (unsigned)(qg * QBK));
                if (!now_) tma_load_2d_pol(smem + s * G_ST, wmap_q, (kb0 + ki) * QBK, qrow0, &gfull[s], pol);
#if MEGA_LOOPDBG
                if (i == 15) BSTAMP(26);
                if (i == 0) BSTAMP(21);
#endif
                if (++s == gns) { s = 0; lp ^= 1; }
            }
#if MEGA_LOOPDBG
            BSTAMP(23);
#endif
            if (pre >= QKH) { if (MEGA_XFLAG) mbar_wait(xgo, 0); else griddep_wait(); }
#endif
#if !MEGA_LL
            // the CTA's qkv-ready flag: released by THIS thread (no outstanding memory operations of its own -- the math threads that stored
            // the rows also issued cp.async gathers / the CTA has o_proj weight TMAs in flight, which a release fence there would wait for);
            // cumulative over the math warps' stores through the qready mbarrier (arrive = release.cta, wait = acquire.cta)
            mbar_wait(ep_bar, 0);
            mbar_wait(qready, 0);
            st_release_gpu(&P.qflags[c * FLAG_STRIDE], s_epoch[0]);
            if (P.dbg) P.dbg[blockIdx.x * NDBG + 9] = gtimer();
#endif
        }
        __syncwarp();
    } else if (R_PROD && warp == NW + 2) {
#if MEGA_XFLAG
        if (P.xflags) {
            // fast path: the producing MoE's 32 row CTAs released x row t (+ its scales) with flag t = that launch's epoch = work[2] + 1 as read
            // here while it still runs; once it has finished (work[2] already bumped) the flags never reach want and griddepcontrol.wait
            // (o_proj producer, ep_bar) has returned or is about to -> slow path. Flags never exceed the running launch's epoch.
            // Lane t polls flag t: ONE coalesced 128-B request per poll and CTA (128 CTAs poll one L2 line while the MoE tail runs).
            const unsigned want = ld_acquire_gpu(reinterpret_cast<const unsigned*>(P.xwork + 2)) + 1u;
            for (;;) {
                const unsigned v = ld_relaxed_gpu(P.xflags + lane);
                if (__all_sync(0xffffffffu, (int)(v - want) >= 0)) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); break; }
                if (mbar_test(ep_bar, 0)) break;
                __nanosleep(MEGA_XFLAG_NS);
            }
            __syncwarp();
        } else if (lane == 0) {
            griddep_wait();
        }
#endif
        if (lane == 0) {
            // x producer: the K half's x scales (one 3-KB bulk copy), then one 32 x 128 x tile per unit, after griddepcontrol.wait
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_qx)) : "memory");
            uint64_t pol;
            asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
            const int kb0 = h * QKH, rot = MEGA_PSTAGE ? ((p * MEGA_ROT) % (QKH / 2)) * 2 : (p * MEGA_ROT) % QKH;
            const bool nox = P.mode & MODE_DBG_GNOX;
#if MEGA_XFLAG
            asm volatile("fence.proxy.async.global;" ::: "memory");   // generic-proxy writes of the MoE -> this thread's TMA / bulk reads
            mbar_arrive(xgo);
#else
            griddep_wait();
#endif
            if (P.dbg) P.dbg[blockIdx.x * NDBG + 7] = gtimer();
            mbar_expect(sx_full, QKH * M * 4);
            bulk_g2s(smem + MG_OFF_SX, P.qsx + (size_t)kb0 * P.qsx_ld, QKH * M * 4, sx_full);
            int s = 0, lp = 0;
#if MEGA_PSTAGE
            const int nps = gns / 2;
            MEGA_PROD_UNROLL
            for (int j = 0; j < QKH / 2; ++j) {
                int kj = 2 * j + rot; if (kj >= QKH) kj -= QKH;
                if (j >= nps) mbar_wait(&gempty[s], lp ^ 1);
                mbar_expect(&gfull[s], nox ? 0u : (unsigned)(2 * G_X));
                if (!nox) tma_load_3d_pol(smem + s * 2 * G_ST + 2 * G_W, &P.tm_qx, 0, 0, kb0 + kj, &gfull[s], pol);
#if MEGA_LOOPDBG
                if (j == 7) BSTAMP(25);
                if (j == 0) BSTAMP(22);
#endif
                if (++s == nps) { s = 0; lp ^= 1; }
            }
            if (false)
#endif
            for (int i = 0; i < QKH; ++i) {
                int ki = i + rot; if (ki >= QKH) ki -= QKH;
                if (i >= gns) mbar_wait(&gempty[s], lp ^ 1);
                mbar_expect(&gfull[s], nox ? 0u : (unsigned)G_X);
                if (!nox) tma_load_2d_pol(smem + s * G_ST + G_W, &P.tm_qx, (kb0 + ki) * QBK, 0, &gfull[s], pol);
#if MEGA_LOOPDBG
                if (i == 15) BSTAMP(25);
                if (i == 0) BSTAMP(22);
#endif
                if (++s == gns) { s = 0; lp ^= 1; }
            }
#if MEGA_LOOPDBG
            BSTAMP(24);
#endif
        }
        __syncwarp();
#if MEGA_GWARP
    } else if (warp >= NW + 3) {
        if (att) {
            // K / V window gathers of this key-split attention CTA into its scratch (pair stages 4..7 of the GEMM ring + above): wait for the
            // slots (math warps) and for pairs 4..7 consumed (their stages never get a second lap), then cp.async; lane 0's plain arrive after
            // the zero rows + 32 lanes' cp.async arrivals complete gath_full
            mbar_wait(slot_ready, 0);
#pragma unroll 1
            for (int ps = 4; ps < 8; ++ps) mbar_wait(&gempty[ps], 0);
            mega_gather_warp<WINDOW>(P, smem, h, (warp - NW - 3) * 32 + lane, gath_full);
        }
        __syncwarp();
#endif
    } else if (R_MATH) {
        // ================= math warps
        const int pb = (c / CL) / NHG, phg = (c / CL) % NHG;   // attention: request, head group (key half = h)
        float* s_qsw = reinterpret_cast<float*>(smem + MG_OFF_QSW);
        const int sbA = qrow0 >> 7;
        if (tid < 2 * QKH) s_qsw[tid] = P.qsw[min(sbA + tid / QKH, QNB - 1) * QKB + h * QKH + tid % QKH];
        if (att) mega_gathers<WINDOW>(P, smem, pb, phg, h);   // contains a math-warp barrier
        else named_bar_sync(1, NMATH);
#if MEGA_GWARP
        if (att && tid == 0) mbar_arrive(slot_ready);   // the slots (written before mega_gathers' barrier) -> the gather warp
#endif
        // ---- phase 0: this CTA's K half of rows [qrow0, qrow0 + 53): warpgroup w = units w, w + 2, ..
        {
            const int g = warp >> 2, ctid = tid & 127;
            const int frow = (ctid >> 5) * 16 + ((ctid & 31) >> 2), fcol = (ctid & 3) * 2;
            const float* sw0p = s_qsw + ((min(qrow0 + frow, QN - 1) >> 7) - sbA) * QKH;      // scale rows of my 2 fragment rows
            const float* sw1p = s_qsw + ((min(qrow0 + frow + 8, QN - 1) >> 7) - sbA) * QKH;
            float acc[16];
#pragma unroll
            for (int r = 0; r < 16; ++r) acc[r] = 0.f;
            const int rot = MEGA_PSTAGE ? ((p * MEGA_ROT) % (QKH / 2)) * 2 : (p * MEGA_ROT) % QKH;
            const float* sxt = reinterpret_cast<const float*>(smem + MG_OFF_SX);
            const bool nomma = P.mode & MODE_DBG_GNOMMA;
            // software pipeline over this warpgroup's 12 units (i = g, g + 2, ..): the wgmma group of unit i is in flight while unit i - 2 is
            // promoted (acc += partial * sw * sx in unit order: the per-thread summation order is unchanged)
            // software pipeline over this warpgroup's unit PAIRS (warpgroup g: units 4 q + 2 g, 4 q + 2 g + 1, q = 0..5): one fence / commit /
            // wait per two k-blocks (the per-iteration warpgroup sync + barrier overhead was ~4x the m64n32k32 time); the group of pair q is in
            // flight while pair q - 1 is promoted (acc += partial * sw * sx in unit order)
            long long cw = 0, c0 = clock64();
            const uint32_t sbase = sa(smem) & 0x3FFFF;
            int cs = 2 * g, cp = 0;   // stage / lap parity of the next unit to issue (2g < gns)
            int fs = 2 * g;           // stage of the next unit to finish
            auto adv = [&](int& st, int& par, int d) { st += d; if (st >= gns) { st -= gns; par ^= 1; } };
#if MEGA_PSTAGE
            // pair stages: warpgroup g's pairs j = g, g + 2, .. live in pair stages j % (gns / 2) (always the same warpgroup: gns / 2 even)
            const int nps = gns / 2;
            const uint32_t wsp = (MEGA_SPLIT && p >= QSPLIT_P) ? 32u * QBK : 64u * QBK;   // W tile b offset (box rows x 128 B)
            int ps_ = g, pp_ = 0;   // pair stage / lap parity of the next pair to issue
            int pf_ = g;            // pair stage of the next pair to finish
            auto issue = [&](int i, float* t0, float* t1) {
                (void)i;
                const int s0 = ps_, p0 = pp_;
                ps_ += 2; if (ps_ >= nps) { ps_ -= nps; pp_ ^= 1; }
                const long long ca = clock64();
                mbar_wait(&gfull[s0], p0);
                cw += clock64() - ca;
                const uint32_t lo0 = ((sbase + s0 * 2 * G_ST) >> 4) | (1u << 16), lo1 = lo0 + (wsp >> 4);
                constexpr uint32_t HI = (1024 >> 4) | (1u << 30), XA = (2 * G_W) >> 4;
                const uint32_t XB = (2 * G_W + G_X - wsp) >> 4;
                wg_fence();
                if (!nomma) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(t0, desc_lo(lo0 + 2 * k, HI), desc_lo(lo0 + XA + 2 * k, HI), k > 0);
#pragma unroll
                    for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(t1, desc_lo(lo1 + 2 * k, HI), desc_lo(lo1 + XB + 2 * k, HI), k > 0);
                }
                wg_commit();
            };
#else
            auto issue = [&](int i, float* t0, float* t1) {
                (void)i;
                const int s0 = cs, p0 = cp;
                int s1 = cs, p1 = cp; adv(s1, p1, 1);
                adv(cs, cp, 4);
                const long long ca = clock64();
                mbar_wait(&gfull[s0], p0);
                mbar_wait(&gfull[s1], p1);
                cw += clock64() - ca;
                // descriptor = {(addr >> 4) & 0x3FFF | 1 << 16, SBO 64 | swizzle 128B}: the k32 steps only add 2 to the low word (no carry)
                const uint32_t lo0 = ((sbase + s0 * G_ST) >> 4) | (1u << 16), lo1 = ((sbase + s1 * G_ST) >> 4) | (1u << 16);
                constexpr uint32_t HI = (1024 >> 4) | (1u << 30), XO = G_W >> 4;
                wg_fence();
                if (!nomma) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(t0, desc_lo(lo0 + 2 * k, HI), desc_lo(lo0 + XO + 2 * k, HI), k > 0);
#pragma unroll
                    for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(t1, desc_lo(lo1 + 2 * k, HI), desc_lo(lo1 + XO + 2 * k, HI), k > 0);
                }
                wg_commit();
            };
#endif
            auto finish1 = [&](int i, const float* t) {
                int ki = i + rot; if (ki >= QKH) ki -= QKH;
                const float sw0 = sw0p[ki], sw1 = sw1p[ki];
                const float* sx = sxt + ki * M;
                float sc[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) sc[q] = sx[(q >> 1) * 8 + fcol + (q & 1)];
#pragma unroll
                for (int r = 0; r < 16; ++r) acc[r] += t[r] * ((r & 2) ? sw1 : sw0) * sc[(r >> 2) * 2 + (r & 1)];
            };
            auto finish = [&](int i, const float* t0, const float* t1) {
                __syncwarp();
#if MEGA_PSTAGE
                if (lane == 0) mbar_arrive(&gempty[pf_]);
                pf_ += 2; if (pf_ >= nps) pf_ -= nps;
#else
                if (lane == 0) { const int f1 = fs + 1 == gns ? 0 : fs + 1; mbar_arrive(&gempty[fs]); mbar_arrive(&gempty[f1]); }
                fs += 4; if (fs >= gns) fs -= gns;
#endif
                finish1(i, t0);
                finish1(i + 1, t1);
            };
#if MEGA_PAIRS
            float tA0[16], tA1[16], tB0[16], tB1[16];
            mbar_wait(sx_full, 0);
            if (tid == 0) BSTAMP(8);
            constexpr int NPAIR = QKH / 4;   // 6 pairs per warpgroup
#pragma unroll 1
            for (int q = 0; q < NPAIR; q += 2) {
                const int iA = 4 * q + 2 * g, iB = iA + 4;
#if MEGA_LOOPDBG
                if (tid == 0 && q == 2) BSTAMP(18);
                if (tid == 0 && q == 4) BSTAMP(19);
#endif
                issue(iA, tA0, tA1);
#if MEGA_LOOPDBG
                if (tid == 0 && q == 0) BSTAMP(15);
#endif
                if (q > 0) { wg_wait<1>(); finish(iA - 4, tB0, tB1); }
#if MEGA_MID_GATHER
                if (att && q == NPAIR - 2) { named_bar_sync(1, NMATH); mega_gather_issue<WINDOW>(P, smem, h); }   // pairs 0..7 finished by both
#endif
                issue(iB, tB0, tB1);
                wg_wait<1>();
                finish(iA, tA0, tA1);
            }
            wg_wait<0>();
#if MEGA_LOOPDBG
            if (tid == 0) BSTAMP(20);
#endif
            finish(4 * (NPAIR - 1) + 2 * g, tB0, tB1);
#else
            // single units, alternating warpgroups (units g, g + 2, ..): group of unit i in flight while unit i - 2 is promoted (as e4c5053e41)
            int us_ = g, up_ = 0;   // stage / parity of the next unit to issue
            auto issue1 = [&](float* t) {
                const int s0 = us_, p0 = up_;
                us_ += 2; if (us_ >= gns) { us_ -= gns; up_ ^= 1; }
                const long long ca = clock64();
                mbar_wait(&gfull[s0], p0);
                cw += clock64() - ca;
                const uint32_t lo0 = ((sbase + s0 * G_ST) >> 4) | (1u << 16);
                constexpr uint32_t HI = (1024 >> 4) | (1u << 30), XO = G_W >> 4;
                wg_fence();
                if (!nomma) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(t, desc_lo(lo0 + 2 * k, HI), desc_lo(lo0 + XO + 2 * k, HI), k > 0);
                }
                wg_commit();
            };
            int fs1 = g;
            auto finish_1 = [&](int i, const float* t) {
                __syncwarp();
                if (lane == 0) mbar_arrive(&gempty[fs1]);
                fs1 += 2; if (fs1 >= gns) fs1 -= gns;
                finish1(i, t);
            };
            float tA[16], tB[16];
            mbar_wait(sx_full, 0);
            if (tid == 0) BSTAMP(8);
#pragma unroll 1
            for (int j = 0; j < QKH / 2; j += 2) {
                const int iA = g + 2 * j;
                issue1(tA);
                if (j > 0) { wg_wait<1>(); finish_1(iA - 2, tB); }
                issue1(tB);
                wg_wait<1>();
                finish_1(iA, tA);
            }
            wg_wait<0>();
            finish_1(g + QKH - 2, tB);
#endif
            if (P.dbg && (tid & 127) == 0) { P.dbg[blockIdx.x * NDBG + 27 + g] = cw; P.dbg[blockIdx.x * NDBG + 29] = clock64() - c0; }
#if MEGA_LATE_GATHER && !MEGA_MID_GATHER && !MEGA_GWARP
            if (att) { named_bar_sync(1, NMATH); mega_gather_issue<WINDOW>(P, smem, h); }   // every unit consumed: the ring is dead
#endif
            // warpgroup 1 -> smem -> warpgroup 0 (fixed order: wg 0 + wg 1)
            // the warpgroup-1 partial [64 rows][32 tok] fp32 lives in the attention-tile halves of o_proj stages 0 / 1 (rows 0..31 / 32..63): the
            // o_proj WEIGHT TMAs may start right after the loops (gdone), the attention tiles land there only after every attention flag
            auto red_at = [&](int row) { return reinterpret_cast<float*>(smem + (row >> 5) * ST_BYTES + WT_BYTES) + (row & 31) * M; };
#if !MEGA_LL
            named_bar_sync(1, NMATH);   // both warpgroups' last wgmma retired: the ring (incl. its stage-0/1 parts) is dead
            __syncwarp();
            if ((!att || MEGA_ATT_EARLY_W || MEGA_OPF) && lane == 0) mbar_arrive(gdone);
#endif
            // both warpgroups send their partial of the PARTNER's finalized rows straight over DSMEM (slot g); warpgroup 1 hands its partial of
            // MY finalized rows to warpgroup 0 through smem; warpgroup 0 finalizes in the fixed order (K half 0: wg0 + wg1) + (K half 1: wg0 + wg1)
            const int pf0 = h ? 0 : qmid, pf1 = h ? qmid : qg;   // the partner's finalized rows
            {
                const uint32_t rx = mapa(sa(smem + MG_OFF_XG), (uint32_t)(h ^ 1)), rbar = mapa(sa(xg_full), (uint32_t)(h ^ 1));
#pragma unroll
                for (int r = 0; r < 16; r += 2) {
                    const int row = frow + ((r >> 1) & 1) * 8, tok = (r >> 2) * 8 + fcol;
                    if (row >= pf0 && row < pf1) st_async_v2(rx + ((g * 30 + row - pf0) * M + tok) * 4, acc[r], acc[r + 1], rbar);
                    else if (g == 1 && row >= fin0 && row < fin1) *reinterpret_cast<float2*>(red_at(row) + tok) = make_float2(acc[r], acc[r + 1]);
                }
            }
            if (tid == 0) BSTAMP(12);
            named_bar_sync(1, NMATH);
            if (tid == 0) BSTAMP(11);
            if (g == 0) {
                mbar_wait(xg_full, 0);
                if (tid == 0) BSTAMP(13);
                const float* xs0 = reinterpret_cast<const float*>(smem + MG_OFF_XG);
                const float* xs1 = xs0 + 30 * M;
                __nv_bfloat16* fin = reinterpret_cast<__nv_bfloat16*>(smem + OFF_FIN);
#pragma unroll
                for (int r = 0; r < 16; r += 2) {
                    const int row = frow + ((r >> 1) & 1) * 8, tok = (r >> 2) * 8 + fcol;
                    if (row >= fin0 && row < fin1) {
                        const float2 o1 = *reinterpret_cast<const float2*>(red_at(row) + tok);
                        const float2 q0 = *reinterpret_cast<const float2*>(xs0 + (row - fin0) * M + tok);
                        const float2 q1 = *reinterpret_cast<const float2*>(xs1 + (row - fin0) * M + tok);
                        const float m0 = acc[r] + o1.x, m1 = acc[r + 1] + o1.y, p0 = q0.x + q1.x, p1 = q0.y + q1.y;   // mine / partner's K half
                        const float v0 = h ? p0 + m0 : m0 + p0, v1 = h ? p1 + m1 : m1 + p1;                             // K half 0 + K half 1
                        const __nv_bfloat16 b0 = __float2bfloat16_rn(v0), b1 = __float2bfloat16_rn(v1);
                        if (MEGA_LL) {   // LL staging (stage 0's weight half: only MEGA_LL delays the o_proj weight TMAs past this point)
                            fin[tok * 30 + (row - fin0)] = b0;
                            fin[(tok + 1) * 30 + (row - fin0)] = b1;
                        } else {         // staging for the coalesced row stores below
                            __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem + 2 * ST_BYTES + WT_BYTES);
                            stg[tok * 32 + (row - fin0)] = b0;
                            stg[(tok + 1) * 32 + (row - fin0)] = b1;
                        }
                    }
                }
            }
        }
#if !MEGA_LL
        // qkv rows: token t's (fin1 - fin0) finalized columns are contiguous in qout -> warp w writes tokens w, w + 8, .. (one 2-B store per lane)
        named_bar_sync(1, NMATH);
        {
            const __nv_bfloat16* stg = reinterpret_cast<const __nv_bfloat16*>(smem + 2 * ST_BYTES + WT_BYTES);
            const int nfin = fin1 - fin0;
            for (int t = warp; t < M; t += NW)
                if (lane < nfin) P.qout[(size_t)t * QN + qrow0 + fin0 + lane] = stg[t * 32 + lane];
        }
#endif
        // the warpgroup-1 partial / staging (generic writes / reads) sit in attention-tile slots that the TMA (async proxy) overwrites later
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (tid == 0) BSTAMP(14);
#if !MEGA_LL
        __syncwarp();
        if (lane == 0) mbar_arrive(qready);
#endif
        mbar_wait(ep_bar, 0);   // also orders the math warps after the o_proj producer's griddepcontrol.wait
        named_bar_sync(1, NMATH);
#if MEGA_LL
        {
            // LL messages: thread (t, j) of the first 192 writes {rows 6 j .. 6 j + 5 of token t, epoch}; lanes 2k / 2k + 1 = messages 2k', 2k' + 1
            // of one token -> every 32-B sector of the buffer is written by ONE store instruction (the LL tearing rule)
            if (tid < M * LL_MSG) {
                const int t = tid / LL_MSG, j = tid - t * LL_MSG;
                if (j < 5) {
                    const uint32_t* f = reinterpret_cast<const uint32_t*>(smem + OFF_FIN) + (t * 30 + 6 * j) / 2;
                    st_relaxed_v4(P.qll + ((size_t)t * NCTA + c) * LL_MSG + j, make_uint4(f[0], f[1], f[2], s_epoch[0]));
                }
            }
            if (tid == 0) BSTAMP(9);
        }
#endif
#if MEGA_LL
        // LL: the staging tile lives in stage 0's weight half -> the o_proj weight TMAs of the non-attention CTAs only now
        named_bar_sync(1, NMATH);
        __syncwarp();
        if (!att && lane == 0) mbar_arrive(gdone);
#endif
        if (att) {
            // the row groups this attention CTA reads: both CTAs (K halves finalize different rows) of each group
#if MEGA_LL
            mega_ll_gather(P, smem, pb, phg, h, s_epoch[0]);
#else
            int qa, qb, ka, kb;
            mega_groups(phg, h, qa, qb, ka, kb);
            const int nq = 2 * (qb - qa + 1), nk = ka <= kb ? 2 * (kb - ka + 1) : 0;
            if (tid < nq + nk) {
                const int f = tid < nq ? 2 * qa + tid : 2 * ka + (tid - nq);
                const unsigned* fl = P.qflags + f * FLAG_STRIDE;
                const unsigned epoch = s_epoch[0];
                for (;;) {
                    if (ld_acquire_gpu(fl) == epoch) break;
                    __nanosleep(32);
                }
            }
            named_bar_sync(1, NMATH);
#endif
            if (tid == 0) BSTAMP(10);
            mega_attention<WINDOW>(P, smem, pb, phg, h, att_in, xm_full, xo_full, gath_full);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            named_bar_sync(1, NMATH);
            if (tid == 0) {
                BSTAMP(1);
                st_release_gpu(&P.flags[c * FLAG_STRIDE], s_epoch[0]);
                BSTAMP(30);
            }
            // next SWA layer's K / V window rows: the L2 prefetch moves to the kernel end (it delayed this CTA's attention-flag poll by
            // ~0.5 us); the slot list is copied out of the attention scratch (the late o_proj weight TMAs overwrite it after att_done)
#if MEGA_KVPF_END
            if (P.nkc && phg == 0 && tid < 80) reinterpret_cast<int*>(smem + MG_OFF_XG)[tid] = attention_ks_slots(smem)[tid];
#else
            if (P.nkc && phg == 0) {   // right after the release (as attn_block_kernel)
                const int* s_slot = attention_ks_slots(smem);
                const int NR = h ? (144 - 64) : 64;
                for (int i = tid; i < NR * 5; i += NMATH) {
                    const int lr = i / 5, part = i - lr * 5;
                    const int sl = s_slot[lr];
                    if (sl < 0) continue;
                    const char* a = part < 3 ? reinterpret_cast<const char*>(P.nkc + (size_t)sl * P.ks) + part * 128
                                             : reinterpret_cast<const char*>(P.nvc + (size_t)sl * P.vs) + (part - 3) * 128;
                    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
                }
            }
#endif
            named_bar_sync(1, NMATH);
            if (tid == 0) mbar_arrive(att_done);
        }
        if (!qonly) {
        {
            const unsigned epoch = s_epoch[0];
            if (tid == 0) BSTAMP(31);
#if MEGA_KTREADY
            // k-tile kt (K elements [1024 h + 64 kt, +64) = head 8 h + kt / 2, dims 64 (kt % 2) ..) comes from the attention CTAs
            // (request b, head pair 4 h + kt / 4, key half kt % 2), b = 0..3: lane 0 of warp kt % 8 (kt < 8) / lane 1 (kt >= 8) polls them
            // (acquire by the issuing thread), then issues that k-tile's TMA; k-tiles NS.. are lane 1 of the producer warp's (after bgo)
            // lanes 4 j .. 4 j + 3 of warp w: k-tile w + 8 j, one producer flag per lane (concurrent polls), then lane 4 j issues the TMA
            // (k-tiles 13..15: the stage is still busy -> lane 4 j arrives kt_seen and the producer warp's lane 1 issues them later)
            if (lane < 8) {
                const int j = lane >> 2, kt = warp + NW * j, b = lane & 3;
                if (kt < NKT) {
                    const unsigned* fl = P.flags + (2 * (b * NHG + 4 * h + kt / 4) + (kt & 1)) * FLAG_STRIDE;
                    while (ld_acquire_gpu(fl) != epoch) __nanosleep(32);
                    __syncwarp(0xFu << (4 * j));
                    if (b == 0) {
                        if (kt < NS) {
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            mbar_expect(&bfull[kt], BT_BYTES);
                            tma_load_2d_nohint(smem + kt * ST_BYTES + WT_BYTES, &P.tm_a, kbase + kt * KT, 0, &bfull[kt]);
                        } else {
                            mbar_arrive(kt_seen);
                        }
                    }
                }
            }
            if (tid == 0) { BSTAMP(2); mbar_arrive(bgo); }
#else
            if (tid < NATT) {
                const unsigned* fl = P.flags + tid * FLAG_STRIDE;
                for (;;) {
                    if (ld_acquire_gpu(fl) == epoch) break;
                    __nanosleep(32);
                }
            }
            named_bar_sync(1, NMATH);
            if (tid == 0) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                BSTAMP(2);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                for (int kt = 0; kt < NS; ++kt) {
                    mbar_expect(&bfull[kt], BT_BYTES);
                    tma_load_2d_nohint(smem + kt * ST_BYTES + WT_BYTES, &P.tm_a, kbase + kt * KT, 0, &bfull[kt]);
                }
                mbar_arrive(bgo);
            }
#endif
        }
        const int g = warp >> 2;
        const uint32_t sbase = sa(smem);
        const uint64_t dA0 = desc_sw128(sbase + g * 64 * 128), dB0 = desc_sw128(sbase + WT_BYTES);
#pragma unroll
        for (int kt = 0; kt < NKT; ++kt) {
            const int s = kt % NS;
            const unsigned ph = (kt / NS) & 1;
            mbar_wait(&wfull[s], ph);
            mbar_wait(&bfull[s], ph);
            if (kt == 0 && tid == 0) BSTAMP(3);
            wg_fence();
#pragma unroll
            for (int j = 0; j < KT / 16; ++j)
                wgmma_m64n32k16(d[j % NACC], dA0 + (uint64_t)((s * ST_BYTES + 32 * j) >> 4), dB0 + (uint64_t)((s * ST_BYTES + 32 * j) >> 4),
                                (kt * (KT / 16) + j) >= NACC);
            wg_commit();
            wg_wait<1>();
            if (kt >= 1 && lane == 0) mbar_arrive(&sempty[(kt - 1) % NS]);
        }
        wg_wait<0>();
#pragma unroll
        for (int a = 1; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 16; ++i) d[0][i] += d[a][i];
        if (tid == 0) BSTAMP(4);
        const int row = 64 * g + 16 * (warp & 3) + (lane >> 2);
        const int prow0 = COLS * (h ^ 1);
        const uint32_t rx = mapa(sa(smem + OFF_X), (uint32_t)(h ^ 1)), rbar = mapa(sa(xfull), (uint32_t)(h ^ 1));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = row + 8 * hh;
            if (r >= prow0 && r < prow0 + COLS) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int tok = 8 * i + 2 * (lane & 3);
                    st_async_v2(rx + ((r - prow0) * M + tok) * 4, d[0][4 * i + 2 * hh], d[0][4 * i + 2 * hh + 1], rbar);
                }
            }
        }
        if (tid == 0) BSTAMP(16);
        mbar_wait(xfull, 0);
        if (tid == 0) BSTAMP(17);
        named_bar_sync(1, NMATH);   // the epilogue tile overwrites stage 0 (see attn_block_kernel: warpgroup 0 may trail)
        __nv_bfloat16* outs = reinterpret_cast<__nv_bfloat16*>(smem);
        const float* xs = reinterpret_cast<const float*>(smem + OFF_X);
        const int orow0 = COLS * h;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = row + 8 * hh;
            if (r >= orow0 && r < orow0 + COLS) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int tok = 8 * i + 2 * (lane & 3);
                    const float2 pv = *reinterpret_cast<const float2*>(xs + (r - orow0) * M + tok);
                    outs[tok * COLS + r - orow0] = __float2bfloat16_rn(d[0][4 * i + 2 * hh] + pv.x);
                    outs[(tok + 1) * COLS + r - orow0] = __float2bfloat16_rn(d[0][4 * i + 2 * hh + 1] + pv.y);
                }
            }
        }
        named_bar_sync(1, NMATH);
        const unsigned epoch = s_epoch[0];
        const int c0 = rbase + COLS * h;
        const int tok = tid >> 3, g6 = tid & 7;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(outs + tok * COLS + 6 * g6);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2];
        const int owner = tok % TP_NDEV;
        if (!FUSED && (owner == P.my_rank || (P.mode & MODE_FULL_LOCAL))) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(P.partial + (size_t)tok * DIM + c0 + 6 * g6);
            dst[0] = w0; dst[1] = w1; dst[2] = w2;
        }
        if ((FUSED || owner != P.my_rank) && !(P.mode & MODE_NO_PUSH)) {
            const int k = c0 / TP_KS, m = (c0 % TP_KS) / 6 + g6;
            st_ll(P.peer[owner] + tp_rs_idx(P.my_rank, tok, k, m), make_uint4(w0, w1, w2, epoch));
        }
        if constexpr (FUSED) { if (c < LAYER_TOKC) named_bar_arrive(LAYER_PUSH_BAR, 2 * NMATH); }   // -> the layer kernel's token-reduce warps
        }   // !qonly
    }
    if (R_MATH && MEGA_KVPF_END && att && P.nkc && ((c / CL) % NHG) == 0 && tid < NMATH) {
        const int* sl_ = reinterpret_cast<const int*>(smem + MG_OFF_XG);
        const int NR = h ? (144 - 64) : 64;
        for (int i = tid; i < NR * 5; i += NMATH) {
            const int lr = i / 5, part = i - lr * 5;
            const int sl = sl_[lr];
            if (sl < 0) continue;
            const char* a = part < 3 ? reinterpret_cast<const char*>(P.nkc + (size_t)sl * P.ks) + part * 128
                                     : reinterpret_cast<const char*>(P.nvc + (size_t)sl * P.vs) + (part - 3) * 128;
            asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
        }
    }
    if (R_MATH && tid == 0) BSTAMP(5);
    if constexpr (!FUSED) griddep_launch();
    // FUSED: the layer kernel's MoE phase reuses this shared memory (generic and TMA writes): order this phase's generic smem writes
    // before later async-proxy writes, and invalidate the mbarrier objects once both CTAs of the pair are past the end sync
    if constexpr (FUSED && R_MATH) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    cluster_sync_relaxed();
    if constexpr (FUSED && R_MATH) {
        constexpr int MG_NBAR = 3 * NS + 6 + 2 * GNS_OTH + 9;   // wfull, bfull, sempty | att_in .. bgo | gfull, gempty | gdone .. gath_full
        static_assert(MG_NBAR * 8 <= MG_OFF_MISC - MG_OFF_BAR, "mega barrier region");
        if (tid < MG_NBAR) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(sa(wfull + tid)) : "memory");
    }
}

template <int WINDOW>
__global__ void __launch_bounds__(MG_NTHREADS, 1) attn_block_mega_kernel(const __grid_constant__ MegaParams P) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
    mega_body<WINDOW, false>(P, smem);
}

// ------------------------------------------------------------------------------------------------------------------ host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_tiled_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled not available");
        fn = reinterpret_cast<EncodeTiledFn>(p);
    }
    return fn;
}
// o_proj weight [6144][2048] bf16 row-major -> 128-byte CUtensorMap (CPU uint8 tensor), box {64 k, 48 rows}, SWIZZLE_128B
static torch::Tensor make_map(const void* base, int64_t rows, int64_t row_stride_elems, int box_rows) {
    auto out = torch::zeros({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
    const cuuint64_t dims[2] = {(cuuint64_t)KO, (cuuint64_t)rows};
    const cuuint64_t strides[1] = {(cuuint64_t)row_stride_elems * 2};
    const cuuint32_t box[2] = {KT, (cuuint32_t)box_rows}, estr[2] = {1, 1};
    CUresult r = encode_tiled_fn()(reinterpret_cast<CUtensorMap*>(out.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return out;
}
// attention output [32][2048] bf16 (contiguous rows) -> CPU u8[128] map, box {64, 32}
torch::Tensor make_amap(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && a.size(0) == M && a.numel() == (int64_t)M * KO && a.is_contiguous() &&
                reinterpret_cast<uint64_t>(a.data_ptr()) % 16 == 0, "attention output must be a contiguous bf16 [32][2048]");
    return make_map(a.data_ptr(), M, KO, M);
}
torch::Tensor make_wmap(torch::Tensor w) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.dim() == 2 && w.size(0) == DIM && w.size(1) == KO && w.stride(1) == 1 &&
                w.stride(0) % 8 == 0 && reinterpret_cast<uint64_t>(w.data_ptr()) % 16 == 0, "o_proj weight must be bf16 [6144][2048], 16-B aligned rows");
    return make_map(w.data_ptr(), DIM, w.stride(0), PCOLS);
}

template <int WINDOW>
static void launch_t(const Params& P, cudaStream_t stream, bool pdl) {
    static bool configured[16] = {false};
    int dev = 0; cudaGetDevice(&dev);
    if (!configured[dev]) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_block_kernel<WINDOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC) == cudaSuccess, "smem attribute");
        configured[dev] = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCTA); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = SMEM_ALLOC; cfg.stream = stream;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 2 : 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, attn_block_kernel<WINDOW>, P) == cudaSuccess, "attn_block launch failed");
}

// qkv [32][3392] bf16 | cos_sin [P][64] bf16 | positions [32] i32/i64 | loc [32] i32/i64 | kc [rows][..192] / vc [rows][..128] bf16 |
// pt [4][S] i32 | seqlens [4] i32 | sinks [16] f32 or empty | attn_out [32][2048] bf16 | flags [32] i32 | partial [32][6144] bf16 |
// wmap CPU u8[128] (make_wmap) | peers CPU i64[8] (prologue IPC buffers) | work i32 (fused MoE counters) | mode bits MODE_*
void attn_block_swa(torch::Tensor qkv, torch::Tensor cos_sin, torch::Tensor positions, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                    torch::Tensor pt, torch::Tensor seqlens, torch::Tensor sinks, torch::Tensor attn_out, torch::Tensor flags, torch::Tensor partial,
                    torch::Tensor wmap, torch::Tensor amap, torch::Tensor peers, torch::Tensor work, int64_t my_rank, double sm_scale, int64_t window, int64_t mode,
                    int64_t pdl, torch::Tensor dbg, torch::Tensor slot_cache, torch::Tensor nkc, torch::Tensor nvc) {
    TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16 && qkv.size(0) == M && qkv.size(1) == QKVW && qkv.is_contiguous(), "qkv [32][3392] bf16");
    TORCH_CHECK(cos_sin.scalar_type() == torch::kBFloat16 && cos_sin.size(1) == 64 && cos_sin.is_contiguous(), "cos_sin [P][64] bf16");
    TORCH_CHECK(positions.numel() == M && loc.numel() == M && positions.is_contiguous() && loc.is_contiguous(), "positions / loc [32]");
    TORCH_CHECK(kc.scalar_type() == torch::kBFloat16 && vc.scalar_type() == torch::kBFloat16 && kc.size(0) == vc.size(0), "kv cache bf16");
    TORCH_CHECK(kc.stride(0) % 8 == 0 && vc.stride(0) % 8 == 0 && kc.numel() / kc.size(0) >= DQK && vc.numel() / vc.size(0) >= DV, "kv cache rows");
    TORCH_CHECK(pt.scalar_type() == torch::kInt32 && pt.dim() == 2 && pt.size(0) == NREQ && pt.stride(1) == 1, "page table [4][S] int32");
    TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 && seqlens.numel() == NREQ, "seqlens [4] int32");
    TORCH_CHECK(attn_out.numel() >= M * KO && flags.numel() >= NATT * FLAG_STRIDE && partial.numel() >= (int64_t)M * DIM && partial.scalar_type() == torch::kBFloat16, "scratch");
    TORCH_CHECK(wmap.device().is_cpu() && wmap.numel() == sizeof(CUtensorMap), "wmap");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == torch::kInt64 && peers.numel() == TP_NDEV, "peers CPU int64[8]");
    TORCH_CHECK(work.scalar_type() == torch::kInt32 && work.numel() >= 4, "work");
    Params P{};
    std::memcpy(&P.tm_w, wmap.data_ptr(), sizeof(CUtensorMap));
    TORCH_CHECK(amap.device().is_cpu() && amap.numel() == sizeof(CUtensorMap), "amap");
    std::memcpy(&P.tm_a, amap.data_ptr(), sizeof(CUtensorMap));
    P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
    P.cos_sin = reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr());
    P.positions = positions.data_ptr(); P.loc = loc.data_ptr();
    P.kc = reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()); P.vc = reinterpret_cast<__nv_bfloat16*>(vc.data_ptr());
    P.ks = kc.stride(0); P.vs = vc.stride(0); P.kv_rows = kc.size(0);
    P.pt = pt.data_ptr<int>(); P.pt_stride = (int)pt.stride(0);
    P.seqlens = seqlens.data_ptr<int>();
    P.sinks = sinks.numel() ? sinks.data_ptr<float>() : nullptr;
    P.attn_out = reinterpret_cast<__nv_bfloat16*>(attn_out.data_ptr());
    P.flags = reinterpret_cast<unsigned*>(flags.data_ptr());
    P.partial = reinterpret_cast<__nv_bfloat16*>(partial.data_ptr());
    for (int d = 0; d < TP_NDEV; ++d) P.peer[d] = reinterpret_cast<uint4*>(peers.data_ptr<int64_t>()[d]);
    P.work = work.data_ptr<int>();
    P.dbg = dbg.numel() >= (int64_t)NCTA * NDBG ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
    P.slot_cache = slot_cache.numel() >= NREQ * 144 ? slot_cache.data_ptr<int>() : nullptr;
    if (nkc.numel() > 0) {
        TORCH_CHECK(nkc.scalar_type() == torch::kBFloat16 && nvc.scalar_type() == torch::kBFloat16 && nkc.stride(0) == kc.stride(0) && nvc.stride(0) == vc.stride(0) &&
                    nkc.size(0) == kc.size(0), "next-layer kv cache must match this layer's row strides");
        P.nkc = reinterpret_cast<const __nv_bfloat16*>(nkc.data_ptr()); P.nvc = reinterpret_cast<const __nv_bfloat16*>(nvc.data_ptr());
    }
    TORCH_CHECK(!((mode & (MODE_SLOTS_WRITE | MODE_SLOTS_READ)) && !P.slot_cache), "slot cache mode without a [4][144] int32 slot_cache");
    P.my_rank = (int)my_rank;
    int md = (int)mode & ~(MODE_POS64 | MODE_LOC64 | MODE_SINK);
    if (positions.scalar_type() == torch::kInt64) md |= MODE_POS64; else TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions int32/int64");
    if (loc.scalar_type() == torch::kInt64) md |= MODE_LOC64; else TORCH_CHECK(loc.scalar_type() == torch::kInt32, "loc int32/int64");
    if (P.sinks) { TORCH_CHECK(sinks.scalar_type() == torch::kFloat32 && sinks.numel() >= NH, "sinks f32 [16]"); md |= MODE_SINK; }
    P.mode = md;
    P.sm_scale = (float)sm_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (window == 129) launch_t<129>(P, stream, pdl != 0);
    else if (window == 128) launch_t<128>(P, stream, pdl != 0);
    else TORCH_CHECK(false, "window must be 128 or 129");
}

// FA layers: o_proj of an attention output computed by the previous kernel (fa3 combine) + the reduce-scatter push. attn: contiguous
// bf16 [32][2048]; amap = make_amap(attn) (the caller rebuilds it whenever attn moves; CUDA-graph replays keep the captured address).
void attn_block_ext(torch::Tensor attn, torch::Tensor partial, torch::Tensor wmap, torch::Tensor amap, torch::Tensor peers, torch::Tensor work,
                    int64_t my_rank, int64_t mode, int64_t pdl, torch::Tensor dbg) {
    TORCH_CHECK(attn.scalar_type() == torch::kBFloat16 && attn.numel() == (int64_t)M * KO && attn.is_contiguous(), "attn [32][2048] bf16");
    TORCH_CHECK(partial.numel() >= (int64_t)M * DIM && partial.scalar_type() == torch::kBFloat16, "partial");
    TORCH_CHECK(wmap.device().is_cpu() && wmap.numel() == sizeof(CUtensorMap) && amap.device().is_cpu() && amap.numel() == sizeof(CUtensorMap), "maps");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == torch::kInt64 && peers.numel() == TP_NDEV, "peers CPU int64[8]");
    TORCH_CHECK(work.scalar_type() == torch::kInt32 && work.numel() >= 4, "work");
    Params P{};
    std::memcpy(&P.tm_w, wmap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_a, amap.data_ptr(), sizeof(CUtensorMap));
    P.partial = reinterpret_cast<__nv_bfloat16*>(partial.data_ptr());
    for (int d = 0; d < TP_NDEV; ++d) P.peer[d] = reinterpret_cast<uint4*>(peers.data_ptr<int64_t>()[d]);
    P.work = work.data_ptr<int>();
    P.dbg = dbg.numel() >= (int64_t)NCTA * NDBG ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
    P.my_rank = (int)my_rank;
    P.mode = ((int)mode & ~(MODE_POS64 | MODE_LOC64 | MODE_SINK)) | MODE_EXT_ATTN;
    launch_t<129>(P, at::cuda::getCurrentCUDAStream(), pdl != 0);
}

// ------------------------------------------------------------------------------------------------------------------ mega host
// fp8 (1-byte) row-major [rows][cols] -> CPU u8[128] tensor map, box {128 cols, box_rows}, SWIZZLE_128B
static torch::Tensor make_map_u8(const void* base, int64_t rows, int64_t cols, int64_t row_stride_bytes, int box_rows) {
    auto out = torch::zeros({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
    const cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
    const cuuint64_t strides[1] = {(cuuint64_t)row_stride_bytes};
    const cuuint32_t box[2] = {(cuuint32_t)QBK, (cuuint32_t)box_rows}, estr[2] = {1, 1};
    CUresult r = encode_tiled_fn()(reinterpret_cast<CUtensorMap*>(out.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (u8) failed: ", (int)r);
    return out;
}
// MEGA_PSTAGE: fp8 row-major [rows][cols] viewed as 3D {128 (k in block), rows, cols / 128 (k-block, stride 128 B)} -> box {128, box_rows, 2}:
// two consecutive k-blocks land as two [box_rows][128] SWIZZLE_128B tiles back to back
static torch::Tensor make_map_u8_3d(const void* base, int64_t rows, int64_t cols, int64_t row_stride_bytes, int box_rows) {
    auto out = torch::zeros({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
    const cuuint64_t dims[3] = {(cuuint64_t)QBK, (cuuint64_t)rows, (cuuint64_t)(cols / QBK)};
    const cuuint64_t strides[2] = {(cuuint64_t)row_stride_bytes, (cuuint64_t)QBK};
    const cuuint32_t box[3] = {(cuuint32_t)QBK, (cuuint32_t)box_rows, 2u}, estr[3] = {1, 1, 1};
    CUresult r = encode_tiled_fn()(reinterpret_cast<CUtensorMap*>(out.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (u8 3d) failed: ", (int)r);
    return out;
}
static torch::Tensor make_qxmap(const torch::Tensor& x) {
    return MEGA_PSTAGE ? make_map_u8_3d(x.data_ptr(), M, QK, x.stride(0), M) : make_map_u8(x.data_ptr(), M, QK, x.stride(0), M);
}
// qkv weight fp8 [3392][6144] (contiguous rows) -> CPU u8[256] = two maps: box {128, 53 or 60 (MEGA_SPLIT)}, box {128, 32}
// (MEGA_PSTAGE: 3D maps, box {128, 64, 2} / {128, 32, 2})
torch::Tensor make_qwmap(torch::Tensor w) {
    TORCH_CHECK(w.is_cuda() && w.element_size() == 1 && w.dim() == 2 && w.size(0) == QN && w.size(1) == QK && w.stride(1) == 1 && w.stride(0) % 16 == 0 &&
                reinterpret_cast<uint64_t>(w.data_ptr()) % 16 == 0, "qkv weight must be fp8 [3392][6144], 16-B aligned rows");
    if (MEGA_PSTAGE) return torch::cat({make_map_u8_3d(w.data_ptr(), QN, QK, w.stride(0), 64), make_map_u8_3d(w.data_ptr(), QN, QK, w.stride(0), 32)});
    return torch::cat({make_map_u8(w.data_ptr(), QN, QK, w.stride(0), MEGA_SPLIT ? QG_E : QG), make_map_u8(w.data_ptr(), QN, QK, w.stride(0), QG_L)});
}

// MEGA_XFLAG: xflags = the producing fused MoE's x-row flags (int32 [>= 32], device) + its work counters; empty = griddepcontrol.wait only
static void set_xflags(MegaParams& P, const torch::Tensor& xflags, const torch::Tensor& xwork) {
    P.xflags = nullptr; P.xwork = nullptr;
    if (xflags.numel() == 0) return;
    TORCH_CHECK(MEGA_XFLAG, "xflags given but this build has MEGA_XFLAG=0");
    TORCH_CHECK(xflags.is_cuda() && xflags.scalar_type() == torch::kInt32 && xflags.numel() >= M && xflags.is_contiguous() &&
                reinterpret_cast<uint64_t>(xflags.data_ptr()) % 16 == 0, "xflags must be a 16-B aligned contiguous device int32 [>= 32]");
    TORCH_CHECK(xwork.is_cuda() && xwork.scalar_type() == torch::kInt32 && xwork.numel() >= 4, "xwork (fused MoE work counters)");
    P.xflags = reinterpret_cast<const unsigned*>(xflags.data_ptr<int>());
    P.xwork = xwork.data_ptr<int>();
}

#ifdef BLK_LAYER_CAPTURE
// goal11 layer (layer_fused.cu): while g_mega_capture is set, attn_block_mega only builds its params (no launch); the layer kernel runs them
struct MegaCapture { MegaParams P; int window; bool pdl; bool valid; };
static MegaCapture* g_mega_capture = nullptr;
#endif
template <int WINDOW>
static void launch_mega_t(const MegaParams& P, cudaStream_t stream, bool pdl) {
#ifdef BLK_LAYER_CAPTURE
    if (g_mega_capture) { g_mega_capture->P = P; g_mega_capture->window = WINDOW; g_mega_capture->pdl = pdl; g_mega_capture->valid = true; return; }
#endif
    static bool configured[16] = {false};
    int dev = 0; cudaGetDevice(&dev);
    if (!configured[dev]) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_block_mega_kernel<WINDOW>, cudaFuncAttributeMaxDynamicSharedMemorySize, MG_SMEM_ALLOC) == cudaSuccess, "smem attribute");
        configured[dev] = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCTA); cfg.blockDim = dim3(MG_NTHREADS); cfg.dynamicSmemBytes = MG_SMEM_ALLOC; cfg.stream = stream;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 2 : 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, attn_block_mega_kernel<WINDOW>, P) == cudaSuccess, "attn_block_mega launch failed");
}

// x [32][6144] fp8 (row stride % 16 B == 0) | x_scale_t [48][>= 32] fp32 rows (= the deep_gemm column-major [32, 48] scale's transpose
// view, row stride % 4 == 0) | qwmap = make_qwmap(qkv weight) | qsw [27][48] fp32 | qout bf16 [32][3392] (the qkv rows, scratch) |
// qflags int32 >= 128 * 32 | qpre: weight units issued before griddepcontrol.wait | the rest = attn_block_swa's.
void attn_block_mega(torch::Tensor x, torch::Tensor x_scale_t, torch::Tensor qwmap, torch::Tensor qsw, torch::Tensor qout, torch::Tensor qflags,
                     torch::Tensor qll, int64_t qpre,
                     torch::Tensor cos_sin, torch::Tensor positions, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                     torch::Tensor pt, torch::Tensor seqlens, torch::Tensor sinks, torch::Tensor attn_out, torch::Tensor flags, torch::Tensor partial,
                     torch::Tensor wmap, torch::Tensor amap, torch::Tensor peers, torch::Tensor work, int64_t my_rank, double sm_scale, int64_t window, int64_t mode,
                     int64_t pdl, torch::Tensor dbg, torch::Tensor slot_cache, torch::Tensor nkc, torch::Tensor nvc, torch::Tensor xflags, torch::Tensor xwork) {
    TORCH_CHECK(x.is_cuda() && x.element_size() == 1 && x.dim() == 2 && x.size(0) == M && x.size(1) == QK && x.stride(1) == 1 && x.stride(0) % 16 == 0 &&
                reinterpret_cast<uint64_t>(x.data_ptr()) % 16 == 0, "x must be fp8 [32][6144]");
    TORCH_CHECK(x_scale_t.scalar_type() == torch::kFloat32 && x_scale_t.dim() == 2 && x_scale_t.size(0) == QKB && x_scale_t.size(1) >= M &&
                x_scale_t.stride(1) == 1 && x_scale_t.stride(0) == M && reinterpret_cast<uint64_t>(x_scale_t.data_ptr()) % 16 == 0, "x_scale_t [48][32] fp32 rows");
    TORCH_CHECK(qsw.scalar_type() == torch::kFloat32 && qsw.numel() == QNB * QKB && qsw.is_contiguous(), "qkv weight scale [27][48] fp32");
    TORCH_CHECK(qout.numel() == 0 || (qout.scalar_type() == torch::kBFloat16 && qout.numel() == (int64_t)M * QN && qout.is_contiguous()), "qout bf16 [32][3392] or empty");
    TORCH_CHECK(!MEGA_LL || (qll.is_cuda() && qll.numel() * qll.element_size() >= (int64_t)M * NCTA * LL_MSG * 16 && reinterpret_cast<uint64_t>(qll.data_ptr()) % 16 == 0), "qll >= 32 x 128 x 6 x 16 B");
    TORCH_CHECK(qflags.numel() >= (int64_t)NCTA * FLAG_STRIDE && qflags.element_size() == 4, "qflags");
    TORCH_CHECK(qwmap.device().is_cpu() && qwmap.numel() == 2 * sizeof(CUtensorMap), "qwmap (make_qwmap)");
    TORCH_CHECK(cos_sin.scalar_type() == torch::kBFloat16 && cos_sin.size(1) == 64 && cos_sin.is_contiguous(), "cos_sin [P][64] bf16");
    TORCH_CHECK(positions.numel() == M && loc.numel() == M && positions.is_contiguous() && loc.is_contiguous(), "positions / loc [32]");
    TORCH_CHECK(kc.scalar_type() == torch::kBFloat16 && vc.scalar_type() == torch::kBFloat16 && kc.size(0) == vc.size(0), "kv cache bf16");
    TORCH_CHECK(kc.stride(0) % 8 == 0 && vc.stride(0) % 8 == 0 && kc.numel() / kc.size(0) >= DQK && vc.numel() / vc.size(0) >= DV, "kv cache rows");
    TORCH_CHECK(pt.scalar_type() == torch::kInt32 && pt.dim() == 2 && pt.size(0) == NREQ && pt.stride(1) == 1, "page table [4][S] int32");
    TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 && seqlens.numel() == NREQ, "seqlens [4] int32");
    TORCH_CHECK(attn_out.numel() >= M * KO && flags.numel() >= NATT * FLAG_STRIDE && partial.numel() >= (int64_t)M * DIM && partial.scalar_type() == torch::kBFloat16, "scratch");
    TORCH_CHECK(wmap.device().is_cpu() && wmap.numel() == sizeof(CUtensorMap) && amap.device().is_cpu() && amap.numel() == sizeof(CUtensorMap), "maps");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == torch::kInt64 && peers.numel() == TP_NDEV, "peers CPU int64[8]");
    TORCH_CHECK(work.scalar_type() == torch::kInt32 && work.numel() >= 4, "work");
    TORCH_CHECK(BLK_KSPLIT == 1, "mega needs the key-split attention build");
    MegaParams P{};
    std::memcpy(&P.tm_w, wmap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_a, amap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_qw, qwmap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_qw2, reinterpret_cast<const uint8_t*>(qwmap.data_ptr()) + sizeof(CUtensorMap), sizeof(CUtensorMap));
    {
        torch::Tensor xm = make_qxmap(x);
        std::memcpy(&P.tm_qx, xm.data_ptr(), sizeof(CUtensorMap));
    }
    P.qsx = x_scale_t.data_ptr<float>(); P.qsx_ld = (int)x_scale_t.stride(0);
    P.qsw = qsw.data_ptr<float>();
    P.qout = qout.numel() ? reinterpret_cast<__nv_bfloat16*>(qout.data_ptr()) : nullptr;
    TORCH_CHECK(MEGA_LL || P.qout, "MEGA_LL 0 needs the qout buffer");
    P.qll = reinterpret_cast<uint4*>(qll.data_ptr());
    P.qflags = reinterpret_cast<unsigned*>(qflags.data_ptr());
    P.qpre = (int)std::max<int64_t>(0, std::min<int64_t>(qpre, GNS_OTH));
    set_xflags(P, xflags, xwork);
    P.qkv = P.qout;
    P.cos_sin = reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr());
    P.positions = positions.data_ptr(); P.loc = loc.data_ptr();
    P.kc = reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()); P.vc = reinterpret_cast<__nv_bfloat16*>(vc.data_ptr());
    P.ks = kc.stride(0); P.vs = vc.stride(0); P.kv_rows = kc.size(0);
    P.pt = pt.data_ptr<int>(); P.pt_stride = (int)pt.stride(0);
    P.seqlens = seqlens.data_ptr<int>();
    P.sinks = sinks.numel() ? sinks.data_ptr<float>() : nullptr;
    P.attn_out = reinterpret_cast<__nv_bfloat16*>(attn_out.data_ptr());
    P.flags = reinterpret_cast<unsigned*>(flags.data_ptr());
    P.partial = reinterpret_cast<__nv_bfloat16*>(partial.data_ptr());
    for (int d = 0; d < TP_NDEV; ++d) P.peer[d] = reinterpret_cast<uint4*>(peers.data_ptr<int64_t>()[d]);
    P.work = work.data_ptr<int>();
    P.dbg = dbg.numel() >= (int64_t)NCTA * NDBG ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
    P.slot_cache = slot_cache.numel() >= NREQ * 144 ? slot_cache.data_ptr<int>() : nullptr;
    if (nkc.numel() > 0) {
        TORCH_CHECK(nkc.scalar_type() == torch::kBFloat16 && nvc.scalar_type() == torch::kBFloat16 && nkc.stride(0) == kc.stride(0) && nvc.stride(0) == vc.stride(0) &&
                    nkc.size(0) == kc.size(0), "next-layer kv cache must match this layer's row strides");
        P.nkc = reinterpret_cast<const __nv_bfloat16*>(nkc.data_ptr()); P.nvc = reinterpret_cast<const __nv_bfloat16*>(nvc.data_ptr());
    }
    TORCH_CHECK(!((mode & (MODE_SLOTS_WRITE | MODE_SLOTS_READ)) && !P.slot_cache), "slot cache mode without a [4][144] int32 slot_cache");
    P.my_rank = (int)my_rank;
    int md = (int)mode & ~(MODE_POS64 | MODE_LOC64 | MODE_SINK | MODE_EXT_ATTN);
    if (positions.scalar_type() == torch::kInt64) md |= MODE_POS64; else TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions int32/int64");
    if (loc.scalar_type() == torch::kInt64) md |= MODE_LOC64; else TORCH_CHECK(loc.scalar_type() == torch::kInt32, "loc int32/int64");
    if (P.sinks) { TORCH_CHECK(sinks.scalar_type() == torch::kFloat32 && sinks.numel() >= NH, "sinks f32 [16]"); md |= MODE_SINK; }
    P.mode = md;
    P.sm_scale = (float)sm_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (window == 129) launch_mega_t<129>(P, stream, pdl != 0);
    else if (window == 128) launch_mega_t<128>(P, stream, pdl != 0);
    else TORCH_CHECK(false, "window must be 128 or 129");
}

// FA layers: the megakernel's qkv GEMM alone (MODE_QKV_ONLY): x [32][6144] fp8 + x_scale_t [48][32] -> qout bf16 [32][3392]
void attn_block_qkv(torch::Tensor x, torch::Tensor x_scale_t, torch::Tensor qwmap, torch::Tensor qsw, torch::Tensor qout, torch::Tensor qflags,
                    torch::Tensor work, int64_t qpre, int64_t pdl, torch::Tensor dbg, torch::Tensor xflags, torch::Tensor xwork) {
    TORCH_CHECK(x.is_cuda() && x.element_size() == 1 && x.dim() == 2 && x.size(0) == M && x.size(1) == QK && x.stride(1) == 1 && x.stride(0) % 16 == 0 &&
                reinterpret_cast<uint64_t>(x.data_ptr()) % 16 == 0, "x must be fp8 [32][6144]");
    TORCH_CHECK(x_scale_t.scalar_type() == torch::kFloat32 && x_scale_t.dim() == 2 && x_scale_t.size(0) == QKB && x_scale_t.size(1) >= M &&
                x_scale_t.stride(1) == 1 && x_scale_t.stride(0) == M && reinterpret_cast<uint64_t>(x_scale_t.data_ptr()) % 16 == 0, "x_scale_t [48][32] fp32 rows");
    TORCH_CHECK(qsw.scalar_type() == torch::kFloat32 && qsw.numel() == QNB * QKB && qsw.is_contiguous(), "qkv weight scale [27][48] fp32");
    TORCH_CHECK(qout.scalar_type() == torch::kBFloat16 && qout.numel() == (int64_t)M * QN && qout.is_contiguous(), "qout bf16 [32][3392]");
    TORCH_CHECK(qflags.numel() >= (int64_t)NCTA * FLAG_STRIDE && qflags.element_size() == 4, "qflags");
    TORCH_CHECK(qwmap.device().is_cpu() && qwmap.numel() == 2 * sizeof(CUtensorMap), "qwmap (make_qwmap)");
    TORCH_CHECK(work.scalar_type() == torch::kInt32 && work.numel() >= 4, "work");
    TORCH_CHECK(!MEGA_LL, "qkv-only mode needs the flag handoff build");
    MegaParams P{};
    std::memcpy(&P.tm_qw, qwmap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_qw2, reinterpret_cast<const uint8_t*>(qwmap.data_ptr()) + sizeof(CUtensorMap), sizeof(CUtensorMap));
    {
        torch::Tensor xm = make_qxmap(x);
        std::memcpy(&P.tm_qx, xm.data_ptr(), sizeof(CUtensorMap));
    }
    P.qsx = x_scale_t.data_ptr<float>(); P.qsx_ld = (int)x_scale_t.stride(0);
    P.qsw = qsw.data_ptr<float>();
    P.qout = reinterpret_cast<__nv_bfloat16*>(qout.data_ptr());
    P.qkv = P.qout;
    P.qflags = reinterpret_cast<unsigned*>(qflags.data_ptr());
    P.qpre = (int)std::max<int64_t>(0, std::min<int64_t>(qpre, GNS_OTH));
    set_xflags(P, xflags, xwork);
    P.work = work.data_ptr<int>();
    P.dbg = dbg.numel() >= (int64_t)NCTA * NDBG ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
    P.mode = MODE_QKV_ONLY;
    launch_mega_t<129>(P, at::cuda::getCurrentCUDAStream(), pdl != 0);
}

}  // namespace blk

#ifndef BLK_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_block_swa", &blk::attn_block_swa);
    m.def("make_wmap", &blk::make_wmap);
    m.def("make_amap", &blk::make_amap);
    m.def("attn_block_ext", &blk::attn_block_ext);
    m.def("attn_block_mega", &blk::attn_block_mega);
    m.def("attn_block_qkv", &blk::attn_block_qkv);
    m.def("make_qwmap", &blk::make_qwmap);
    m.attr("NCTA") = blk::NCTA;
    m.attr("NATT") = blk::NATT;
    m.attr("SMEM_ALLOC") = blk::SMEM_ALLOC;
    m.attr("MODE_FULL_LOCAL") = blk::MODE_FULL_LOCAL;
    m.attr("MODE_NO_PUSH") = blk::MODE_NO_PUSH;
    m.attr("MODE_DBG_NOW") = blk::MODE_DBG_NOW;
    m.attr("MODE_DBG_NOB") = blk::MODE_DBG_NOB;
    m.attr("MODE_SLOTS_WRITE") = blk::MODE_SLOTS_WRITE;
    m.attr("MODE_SLOTS_READ") = blk::MODE_SLOTS_READ;
    m.attr("MODE_DBG_GNOMMA") = blk::MODE_DBG_GNOMMA;
    m.attr("MODE_DBG_GNOX") = blk::MODE_DBG_GNOX;
    m.attr("MODE_DBG_GNOW") = blk::MODE_DBG_GNOW;
    m.attr("MEGA_LL") = MEGA_LL;
    m.attr("MEGA_SPLIT") = MEGA_SPLIT;
    m.attr("MEGA_XFLAG") = MEGA_XFLAG;
    m.attr("MEGA_PSTAGE") = MEGA_PSTAGE;
}
#endif  // BLK_NO_PYBIND
