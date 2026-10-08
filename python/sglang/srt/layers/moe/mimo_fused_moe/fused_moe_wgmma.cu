// fused_moe_wgmma.cu -- ONE persistent Hopper (sm90a) kernel for the complete
// W4A8 MoE layer (TP8 slice): residual add + RMSNorm -> fp8 quant + fp32 router -> top-8 ->
// union work list -> fc1 (gate+up) + SiLU-gate + fp8 requant -> fc2 (down) + top-k weighted
// combine -> device-side TP8 all-reduce -> bf16 output, all in a single launch.
//
// Compute engine: raw-PTX wgmma.mma_async (m64nNk32, f32 += e4m3 * e4m3) in RS
// mode -- the MXFP4 weight is dequantized in registers (prmt LUT with the e8m0
// group exponent folded straight into the fp8 exponent field, a per-row fp32
// residual scale applied in the epilogue) and used as the A operand; the fp8
// activations (tokens) are the B operand in 128B-swizzled shared memory.
//
// Threads: 544 = 4 consumer warpgroups (warps 0-15) + 1 producer warp (warp 16).
// The per-warpgroup wgmma issue cost (~35 ns per instruction, nearly independent of N,
// measured in wgmma_ubench.cu) is the fc1 bottleneck, so the fc1 work of a CTA is spread
// over 4 warpgroups, each issuing only 4 wgmmas + 4 slices of dequant per k-tile.
//
// Prologue, 4 stages chained by epoch-tagged flags / LL messages,
// no host involvement, every rank recomputes identical routing from identical hidden states:
//   stage 1 (CTA t < M):        residual add + RMSNorm of token t (flashinfer fused_add_rmsnorm
//                               arithmetic) -> residual_out bf16, normed bf16 (xn_buf), fp8 e4m3 +
//                               per-128 scales (xq_buf/xs_buf); release xflags[t].
//   stage 2 (96 router CTAs):   unit = (k-group of 384, 64 expert rows): router weight x bf16 normed
//                               tokens on the tensor cores (wgmma m64nNk16 bf16, fp32 accumulate; 4
//                               warpgroups split K and are summed in a fixed order) -> 22 LL messages
//                               per token. FMOE_ROUTER_BF16 (default): the weight is the checkpoint's
//                               fp32 weight rounded once to bf16 by the host (SGLang's bf16 MoEGate),
//                               one wgmma per k16 step; else the fp32 weight is split exactly into 3
//                               bf16 planes in registers (hi+mid+lo == fp32), 3 wgmmas per step. The
//                               tokens are normalized in smem with the rstd the stage-1 CTAs publish
//                               (FMOE_RSTD_DIRECT, one LL message per token; else recomputed here).
//   stage 3 (CTA t < M):        gathers the 16 k-group partials of token t (fixed order), sigmoid,
//                               + correction bias, top-8 (two-level warp redux/ballot, ties -> lower
//                               expert id), weights = sigmoid / (sum + 1e-20) -> 4 LL messages.
//   stage 4 (every CTA):        gathers all M top-8 lists, builds the union work list (order of
//                               first appearance in (token, rank) order) with smem atomics:
//                               s_union[u] (expert), s_mask[u] (uint64 routed-token mask),
//                               s_slot[t][k] / s_tkw[t][k] (slot + weight per top-8 entry).
// CTA roles (grid = n_fc1 + 48*parts): every CTA first works the fc1 item queue (items = (union slot,
//   128-row half of INTER), grabbed dynamically with an atomic counter; the fc2 CTAs leave the
//   last `reserve` items to the others so they can start fc2 early), then blockIdx < 48*parts run fc2.
//   fc1 phase:
//     Producer: streams the pre-permuted weight tiles (cp.async.bulk, 5-stage mbarrier ring) and
//     gathers ONLY the tokens routed to that expert (cp.async 16B, zero-filled padding) as N=8
//     token chunks, so tensor work scales with routed tokens, not with M.
//     Consumer WG w: matrix (w&1: gate/up) x row half (w>>1) of the 128-row half; per k-tile it issues
//     4 wgmmas m64 x n(8C) x k32 into a scratch that is promoted into the fp32 accumulator with
//     the per-token per-k128 activation scale.
//     Epilogue: SiLU(g)*u, per-token amax over the 128 columns, fp8 requant, h written
//     pre-swizzled to a global staging buffer + combined scale (h_scale * routing weight), then a
//     per-(slot,half) epoch flag is released.
//   fc2 phase (blockIdx < 48*parts): `parts` CTAs per fixed 128-row output tile of DIM (experts split
//     by parity, their partials are separate sources of the all-reduce), each looping over its experts
//     (dense N = M_pad tokens), accumulating locally (no atomics).
//     Warpgroup pairs split the experts by parity (WG w: rows (w&1)*64, experts u%2 == w>>1) and
//     are summed in smem at the end. The producer waits on each expert's h flags and bulk-copies
//     h + scales + the expert's weight tile. Then a two-phase LL all-reduce (reduce-scatter to the
//     owning rank, all-gather), 16-byte epoch-tagged messages (3 floats + tag), direct P2P NVLink
//     stores, batched in-kernel polling; the pull phase writes the bf16 output directly.
//
// FMOE_INPUT_TP (exact M32 / TP8, entry fused_moe_full_tp): the INPUT TP all-reduce is part of the prologue. The kernel takes
// this rank's un-reduced o_proj partial; every CTA pushes its K-slices to the owning ranks as LL messages at kernel entry (reduce-
// scatter: data carries the epoch, no input-ready flag, no peer reads); token CTA t reduces (token t, slice r) in rank order
// (fp32 acc from rank 0, += ranks 1..7, one bf16 RN = CustomAllReduceV2's 2shot_pull, bitwise), pushes its two stage-1 sum-of-squares
// k-group partials and then the reduced slice to every rank (all-gather); stage 1 runs on the gathered row (rstd = the 16 k-group
// partials summed in stage 1's order). Router (FMOE_TP_ROUTER_TOKENS): rank r runs all 96 units over the full K for its 4 owned
// tokens (t % 8 == r, wgmma N = 8) so the partial logits stay local; the owner CTAs run stage 3 for those tokens and push the top-8
// lists to every rank; stage 4 / FC1 / FC2 / TP tail are unchanged. Three cross-rank hops (RS, AG, top-8) replace the AR kernel.
#include <cuda_runtime.h>
#include <type_traits>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>   // CUtensorMap (TMA tensor copies of the weight tiles; encoded through the driver entry point, no libcuda link)
#include <unordered_map>
#include "wgmma_rs_fp8.cuh"

namespace fmoe {

constexpr int DIM = 6144, INTER = 256, KT = 128, NKT1 = DIM / KT;   // 48 k-tiles for fc1
constexpr int KSPLIT = 1;                 // 1: fc1 item = (union slot, 128-row half), 48 k-tiles. 2: k halves as separate items (the k-half-1 CTA
                                          // publishes an fp32 partial the k-half-0 CTA adds before SiLU) -- measured no gain, kept selectable.
constexpr int NKT_ITEM = NKT1 / KSPLIT;   // k-tiles per fc1 item
static_assert(NKT_ITEM % 2 == 0, "tile pairs");
constexpr int MAXM = 64;
constexpr int NEXP = 384;                // routed experts (compile-time max; runtime n_exp <= NEXP)
constexpr int TOPK = 8;
constexpr int MAXU = NEXP;               // union slots (<= min(8*M, n_exp))
constexpr int N_FC2_TILES = DIM / 128;   // 48 output row tiles of 128
constexpr int N_FC2_PARTS_MAX = 3;       // max CTAs per fc2 tile (experts split round-robin, partials are separate all-reduce sources)
constexpr int N_FC2_CTAS = N_FC2_TILES * N_FC2_PARTS_MAX;   // 144 (max; the mixed mode uses the whole grid, <= 132)
// Warp specialization (CUTLASS sm90 style): 4 consumer warpgroups (warps 0-15) + 1 producer warpgroup (warps 16-19,
// only warp 16 works after the prologue). 640 threads compile at 96 registers; after the prologue the producer
// warpgroup releases registers (setmaxnreg.dec 24) and the consumers grow to 112 (24*128 + 112*512 <= 96*640).
// Four consumer warps per SM sub-partition: each warp's per-tile chain (dequant -> wait -> promote -> 4 wgmma issues)
// is half as long as with 2 warpgroups, and the other three warps hide its stalls (the 2-WG version ran at IPC ~0.25).
constexpr int NCONS = 512;               // consumer threads (4 warpgroups)
constexpr int NTHREADS = NCONS + 128;    // + producer warpgroup
constexpr int PROD_WARP = NCONS / 32;    // warp 16
// setmaxnreg split (REG_PRODUCER*128 + REG_CONSUMER*512 == 96*640): the consumers need 112 only at NT=64 (D32+S32+A16);
// below that 104 is spill-free and the producer warp (fc1 gather + fc2 flag polling) gets 64 instead of 32.
template <int NT> struct RegSplit { static constexpr int consumer = (NT >= 64) ? 112 : 104; static constexpr int producer = (NT >= 64) ? 32 : 64; };
constexpr int NSTAGE1 = 6;
#ifndef FMOE_MIXED_WIDTH
#define FMOE_MIXED_WIDTH 0               // 1: hot experts (> 8 tokens) of a two-wave load stream once as N16 tasks on the last CTAs
#endif
#ifndef FMOE_ROUTER_BF16
#define FMOE_ROUTER_BF16 1               // 1: router GEMM on a bf16 copy of the router weight (the fp32 checkpoint weight rounded ONCE, RNE,
#endif                                   //    = SGLang mimo_v2 MoEGate after commit 3b7633d: bf16 parameter, bf16 x bf16 -> fp32 GEMM); one
                                         //    wgmma pass per k16 step and half the W tile bytes (48 KB / unit, 4.7 MB / layer) instead of the
                                         //    exact 3-plane fp32 emulation. LOSSY vs the fp32 weight: top-8 sets can differ from the
                                         //    fp32-router reference near ranking ties (measured per case by real_bench --router-ref bf16).
                                         // 0: fp32 weight, exact fp32 logits via 3 bf16 planes (hi + mid + lo == fp32), 3 wgmmas per step.
#ifndef FMOE_FC2_PREFILL
#define FMOE_FC2_PREFILL 0               // 1 (exact M32, split producer): the three FC2 producer warps fill the FC2 ring on its OWN mbarrier objects
#endif                                   //    (OFF_BAR2, initialised at kernel start) as soon as the 16 consumer warps have published this CTA's last
                                         //    h (named barrier FC2_START_BAR, bar.arrive by the consumers right after fc1_consumer), i.e. during the
                                         //    consumers' FC1->FC2 join / barrier re-init / plan chain instead of after it; the joint FC2 plan is built
                                         //    by the idle warp 19 during FC1 and published to the producers by the producer warpgroup's named barrier 3
                                         //    at the FC1 sentinel. The consumer branch keeps its structure (join, dead re-init of the FC1 barriers,
                                         //    tid-0 plan rebuild) so ptxas does not re-allocate the N8/FC2 consumer bodies (register re-allocation); only its
                                         //    join barrier becomes consumer-only (id 5, 512) and fc2_consumer takes the FC2 barrier objects.
                                         //    the handoff bubble (fc1_end -> first FC2 stage landed, 3.9-4.0 us on the critical tail-B CTAs)
                                         //    is the join chain ~1.5 + first issue ~0.7 + cold-ring TMA ~1.5; this hides the chain under the fill.
                                         //    MEASURED FLAT and left OFF: the bubble on the critical tail-B CTAs does
                                         //    shrink 3.9-4.0 -> ~2 us, and putting it back costs +2.0 us, but the
                                         //    10-case is -0.20 and the 103-case -0.18 with -0.21 baseline drift (51 faster /
                                         //    30 neutral / 22 slower: joint U81-108 -0.4..-0.5, U67-80 +0.3, all-light +1.2, low-U one-wave +0.9..+1.4)
                                         //    -- the same signature as the FMOE_FC2_EARLY experiment in six builds. Delaying only the early leavers
                                         //    (knob below) did not recover the losing cases. The off build is byte-identical to HEAD.
#ifndef FMOE_FC2_PREFILL_NOTAIL_DELAY_NS
#define FMOE_FC2_PREFILL_NOTAIL_DELAY_NS 0      // > 0: only CTAs that leave FC1 late (a second FC1 task; > 52 us after CTA start) keep the early
#endif                                          //    FC2 start; early leavers of a two-wave load (A owners, tail-less helpers) idle this long first,
                                                //    i.e. keep the old join-chain start time (their earlier FC2 streams moved into the
                                                //    wave-2 FC1 window and bought nothing -- A entered 2.85 us earlier, ended at the
                                                //    same time, full-waits +3.9 us -- while all-light / U76-79 cases lost 0.4-1.5 us).
#ifndef FMOE_FC2_TAIL_PREFETCH
#define FMOE_FC2_TAIL_PREFETCH 0                // N > 0 (with FMOE_FC2_PREFILL): a tail CTA's FC1 weight lane, on taking its SECOND FC1 item (~44 us,
#endif                                          //    wave-2 HBM has ~0.8 TB/s of headroom), L2-prefetches the weight boxes of the first N experts it
                                                //    will stream in FC2 ~25 us later (the joint plan is already built by warp 19), so the cold FC2 ring
                                                //    (first stage landed 2.1 us after issue:) would fill from L2. Measured (N=6): 10-case
                                                //    +2.3 us vs PREFILL alone (every two-wave case +1.8..+5.1, one-wave +0.75 from the +1080-instruction
                                                //    re-allocation) -- the FMOE_FC2_L2_PREFETCH failure mode (+5.5 us) again: any extra HBM
                                                //    demand inside the wave-2 window is paid by the tail. OFF.
#ifndef FMOE_SETUP_HOIST
#define FMOE_SETUP_HOIST 1              // 1: task-table consumers compute their item and run the per-item setup before waiting for tile 0
#endif
#ifndef FMOE_FC2_ACQUIRE_EARLY
#define FMOE_FC2_ACQUIRE_EARLY 1         // 1: the FC2 producer lane acquires/prefetches tensor maps 2/3 during the prologue's message wait
#endif
#ifndef FMOE_FC1_ACQUIRE_EARLY
#define FMOE_FC1_ACQUIRE_EARLY 1         // 1: the two FC1 TMA lanes acquire maps 0/1 in the same window; only a prefetch stays before the first TMA
#endif
#ifndef FMOE_ONEWAVE_HELPERS
#define FMOE_ONEWAVE_HELPERS 1           // 1: one-wave (U <= 66) token-task loads use the 36 FC2 helper CTAs with the balanced 2U/5 split
#endif
#ifndef FMOE_LIGHT_PLAN
#define FMOE_LIGHT_PLAN 1                // 1: all-light legacy loads (Tmax <= 8) use the joint FC2 plan with K36/K12 tail offsets
#endif
#ifndef FMOE_BATCH_FENCE
#define FMOE_BATCH_FENCE 8               // N > 0: token-task FC2 producer polls N experts' ready words per L2 round trip + ONE fence pair
#endif                                   // (2/4/8 all compile spill-free for M32 once the token and legacy walks are separate loops)
#ifndef FMOE_LEGACY_ACQUIRE
#define FMOE_LEGACY_ACQUIRE 1            // 1: the legacy (all-light / pre-routed) FC2 poll uses fence.acquire.gpu instead of fence.acq_rel.gpu
#endif
#ifndef FMOE_LATE_SPLIT
#define FMOE_LATE_SPLIT 1                // 1: 108 < U <= 132 token-task loads: tail tickets owners > single helpers > dual helpers, FC2 planned with start offsets, helpers take the warm experts
#endif
#ifndef FMOE_GP
#define FMOE_GP 1                        // 1 (goal10, real DFLASH routing U 110-200): GENERAL PLAN for exact-M32 token-task loads with n_tasks > 216 (no joint
#endif                                   //    plan): up to FMOE_GP_ROUNDS FC1 rounds (round 0: task b -> CTA b; partial rounds: tasks dealt to CTAs in the
                                         //    m32_gp_pos order, which cycles over the 24 single-tile trios and 12 dual-tile pairs so every output tile's
                                         //    FC2 crew carries a near-equal FC1 load); FC2 helpers (CTAs 96..131) for ANY U; per tile the three streams
                                         //    (helper | owner B | owner A, in that slot order = the order in which they leave FC1) get expert ranges from a
                                         //    makespan-balanced partition of this crew's FC1 end offsets (m32_gp_fc2_split). Replaces FMOE_LATE_SPLIT
                                         //    (216 < n_tasks <= 264) and the legacy dynamic queue (n_tasks > 264: 36 CTAs idled through FC2 and the FC2
                                         //    CTAs won the third FC1 round -- 150-170 us at U 140-190).
#ifndef FMOE_FC1_DESYNC
#define FMOE_FC1_DESYNC 0                // 1 (exact-M32 N8 token-task FC1): the two "up" warpgroups run each tile's phases in the order wait -> promote ->
#endif                                   //    dequant -> issue while the "gate" warpgroups keep dequant -> wait -> promote -> issue, so the INT-pipe-bound
                                         //    dequant of one pair overlaps the wgmma wait / FP32 promote / issue of the other on every SM sub-partition
                                         //    (DIAG: 4 warps dequanting together take 560 cycles of the 1190-cycle tile; the INT pipe idles during the rest)
#ifndef FMOE_FC1_DESYNC_NS
#define FMOE_FC1_DESYNC_NS 0             // N > 0 (exact-M32 N8 token-task FC1): the two "up" warpgroups wait N ns before their first tile of every item, so
#endif                                   //    they trail the "gate" warpgroups by ~half a tile period on every SM sub-partition (same code, no second wgmma loop):
                                         //    the INT-pipe-bound dequant of one pair then overlaps the wgmma wait / promote / issue of the other. The ring
                                         //    (stage released by the last warp) keeps the offset while the consumer is the bottleneck.
#ifndef FMOE_GP2
#define FMOE_GP2 1                       // 1 (with FMOE_GP): uniform 2.75-stream FC2 helper layout for the GP regime. Helper CTA 96+h (h 0..35) streams
#endif                                   //    tile h with its WG pair 0 and with the SECOND half of pair 1, and tile 36 + h%12 with the FIRST half of pair 1
                                         //    (pair 1 flushes its accumulator into an arena slot between the two segments); tiles 36..47 receive quarter
                                         //    shares from three helpers (owner B merges 3 partials). Every tile has 2 + 0.75 streams: no structural
                                         //    pair/trio offset, and one extra FC1 task shifts a 4-tile group's makespan by TU/11 instead of TU/3.
#ifndef FMOE_FC1_PUB_OFFLOAD
#define FMOE_FC1_PUB_OFFLOAD 1           // 1 (exact-M32 N8 token-task FC1): the idle producer warp 18 (a) stages every union slot's per-expert factors
#endif                                   //    fc1_s2 * 64 / fc2_s2 * 64 in smem at FC1 start (the item epilogue read them with two dependent global loads)
                                         //    and (b) publishes each item's h-ready chunk bit (ld.acquire + atom.cas round trips, ~2 us per item on
                                         //    thread 0 of the consumers, which the next item's setup barrier waited for); the consumers hand the item
                                         //    over through an smem queue (release.cta / acquire.cta) and start the next item immediately.
                                         //    Not with FMOE_FC2_PREFILL (its producer barrier 3 at the FC1 sentinel needs warp 18, which would still wait for
                                         //    the consumers' sentinel: deadlock) -- FC1_PUB_OFFLOAD below is the effective switch.
constexpr bool FC1_PUB_OFFLOAD = FMOE_FC1_PUB_OFFLOAD && !FMOE_FC2_PREFILL;
#ifndef FMOE_GP2_BW
#define FMOE_GP2_BW 4                    // FMOE_GP2 helper h-copy walk: ready words polled per round trip (8 spilled in the 64-register producer warp)
#endif
#ifndef FMOE_GP_MIN_TASKS
#define FMOE_GP_MIN_TASKS 216            // FMOE_GP runs every exact-M32 token-task load with more tasks than this (216: the joint plan keeps 132 < n <= 216)
#endif
#ifndef FMOE_GP_ROUNDS
#define FMOE_GP_ROUNDS 3                 // token-task loads admitted up to 132 * ROUNDS tasks (U ~ 195 at ~2 tasks per expert); beyond: legacy queue
#endif
#ifndef FMOE_GP_TASK_UNITS
#define FMOE_GP_TASK_UNITS 44            // one FC1 N8 task (48 tiles, ~32 us with 132 CTAs streaming) in FC2 expert-stage units (~0.72 us)
#endif
#ifndef FMOE_GP_EARLY_PCT
#define FMOE_GP_EARLY_PCT 60             // FC2 stages streamed while other CTAs still run FC1 are slower (HBM saturated by FC1): a CTA that leaves FC1
                                         //    early is charged this percentage of the window x (fraction of CTAs still in FC1) as a start offset
#endif
#ifndef FMOE_GP_MERGE_UNITS
#define FMOE_GP_MERGE_UNITS 1            // the helper ends this many stages before owner B (B merges the helper's partial before its TP push)
#endif
#ifndef FMOE_PLAN_SECOND_TASK
#define FMOE_PLAN_SECOND_TASK 36         // joint FC2 plan: a second (tail) FC1 task in A-expert units (earlier schedule: 32)
#endif
#ifndef FMOE_PLAN_MERGE_UNITS
#define FMOE_PLAN_MERGE_UNITS 1          // joint FC2 plan: B's helper-partial merge in expert units, B and H end that much before A (earlier schedule: 0)
#endif
#ifndef FMOE_FC2_RING7
#define FMOE_FC2_RING7 0                 // 1: exact-M32 FC2 ring has 7 stages (out_s shrunk to its 32 token rows) instead of 6
#endif
#ifndef FMOE_H_EVICT_LAST
#define FMOE_H_EVICT_LAST 0              // 1: FC2 h/cs bulk copies carry an L2::evict_last hint
#endif
#ifndef FMOE_LATE_TAIL_UNITS
#define FMOE_LATE_TAIL_UNITS 42          // U>108 late-split FC2 plan: start offset of a tail-carrying CTA in A-expert units
#endif
#ifndef FMOE_LIGHT_K36_UNITS
#define FMOE_LIGHT_K36_UNITS 27          // all-light joint FC2 plan: start offset of a K36-tail CTA in A-expert units (earlier schedule: 27)
#endif
#ifndef FMOE_LIGHT_K12_UNITS
#define FMOE_LIGHT_K12_UNITS 10          // all-light joint FC2 plan: start offset per K12 round (earlier schedule: 10)
#endif
#ifndef FMOE_MERGE_PREFETCH
#define FMOE_MERGE_PREFETCH 1            // 1: B's producer lane bulk-copies the helper partial into the ring slot after its last stage while the
#endif                                   //    consumers still stream; the epilogue adds it from smem
#ifndef FMOE_PROLOGUE_TIGHT
#define FMOE_PROLOGUE_TIGHT 1            // 1: stage 3 reads the router bias from smem (prefetched in the wait window) and drops the
#endif                                   //    level-2 ballot; stage 4 uses a scanned first-bit prefix instead of a per-candidate popc loop
#ifndef FMOE_FC2_SPLIT_PRODUCER
#define FMOE_FC2_SPLIT_PRODUCER 1        // 1: exact-M32 FC2 streams from three producer warps (poll + h/cs | weight TMA | offset TMA); the
#endif                                   //    single lane spent ~1300 of ~1440 cycles per stage issuing
#ifndef FMOE_CS_PERM
#define FMOE_CS_PERM 1                   // 1: exact-M32 prescaled FC2 scales live in the upper half of cs_buf in a per-lane permuted layout and are
#endif                                   //    copied into an 8-entry smem ring outside the stage: the retire arrives first, then reads float2 pairs
#ifndef FMOE_FC2_EARLY_RELEASE
#define FMOE_FC2_EARLY_RELEASE 0         // 1: exact-M32 CS_PERM FC2 pair waits for its k-block-1 group right after issuing it and releases the stage there
                                         //    (the ring slot frees one dequant earlier; the next expert's k-block-0 dequant no longer overlaps that group).
#endif
#ifndef FMOE_FC2_L2_PREFETCH
#define FMOE_FC2_L2_PREFETCH 0           // N > 0: exact-M32 FC2 weight/offset producer warps L2-prefetch the boxes of the expert N stages ahead
#endif                                   //    into L2 (cp.async.bulk.prefetch.tensor) while they wait for a free ring slot
#ifndef FMOE_FC1_L2_PREFETCH_KT
#define FMOE_FC1_L2_PREFETCH_KT 0        // N > 0: during the prologue's message wait (HBM idle), CTA b bulk-prefetches k-tiles 0..N-1 of experts
#endif                                   //    b, b+132, b+264 (32 KB + 2 KB each, contiguous) into L2 so the first FC1 tiles of every expert hit L2
#ifndef FMOE_FC1_XS_STAGE
#define FMOE_FC1_XS_STAGE 1              // 1: exact-M32 N8 FC1 gather lanes cp.async each k-tile's 8 activation scales into the stage (x area + 1 KB);
#endif                                   //    the consumer's per-item setup drops its 384 dependent xs_buf global reads (~1 us per item)
#ifndef FMOE_FC2_LOCKSTEP
#define FMOE_FC2_LOCKSTEP 0              // 1: exact-M32 FC2 consumers = 4 warpgroups (row half, k-block) on every stage in lockstep, FC1's tile-loop
#endif                                   //    shape; dual helpers run two passes. +4.6 us -- all warps of a sub-partition then sit in the SAME phase
                                         //    (dequant / wgmma issue / retire) and serialize on its pipes; the pair scheme's phase diversity is worth more
#ifndef FMOE_PLAN_B_UNITS
#define FMOE_PLAN_B_UNITS (FMOE_MERGE_PREFETCH ? 0 : FMOE_PLAN_MERGE_UNITS)   // joint FC2 plan: B ends this many expert units before A
#endif
#ifndef FMOE_MASK_OR32
#define FMOE_MASK_OR32 1                 // 1: exact-M32 union build sets the routed-token mask bits with a native 32-bit smem atomicOr (the 64-bit
#endif                                   //    atomicOr is a CAS loop; under hot-expert contention the slot loop cost ~1.1 us of the prologue)
#ifndef FMOE_XFLAG_ACQUIRE
#define FMOE_XFLAG_ACQUIRE 1             // 1: stage 4 polls the x_fp8 flags with ld.acquire.gpu instead of ld.relaxed + fence.acq_rel.gpu (the MEMBAR.GPU
#endif                                   //    drained this CTA's outstanding stores: ~0.35 us on every CTA's routing path)
#ifndef FMOE_RANK_TOPK
#define FMOE_RANK_TOPK 1                 // 1: exact-M32 stage 3 selects the top-8 by rank (every key counts the (key, expert) pairs above it: independent,
#endif                                   //    pipelined compares) instead of 8 serial redux/ballot rounds per level (~1050 + ~1350 cycles of collective
                                         //    latency per token); same (key desc, expert asc) order, bit-identical messages
#ifndef FMOE_EARLY_FIRST_TILE
#define FMOE_EARLY_FIRST_TILE 2          // N > 0: exact-M32 token-task loads resolve this CTA's first task inside the (parallel) task-table loop and issue
#endif                                   //    the weight TMAs + x gathers of its tiles 0..N-1 at fc1_producer entry, before the item / token-list handoff
                                         //    chain (~1.4 us of dependent single-warp work before the first TMA in a probe; the consumer idled
                                         //    for tile 0). N = 6 (the whole ring) made every CTA's tile 0 land 3-4 us after issue: 132 x 13.5 MB requested
                                         //    within 0.2 us are served interleaved, so tile 0 completes near the END of the burst; the producer's natural
                                         //    ~0.2 us stagger per tile is what makes tile 0 land in ~1.2 us. fc1_producer skips those tiles of its first item.
#ifndef FMOE_EARLY_PRE_SPLIT
#define FMOE_EARLY_PRE_SPLIT 0           // 1: the early tiles are issued in the kernel body right after the task table, BEFORE the role split (delays the
#endif                                   //    producer warpgroup's setmaxnreg.dec by the issue time, ~0.5 us, but the first TMA leaves ~1 us earlier);
                                         //    0: at fc1_producer entry, after setmaxnreg
#ifndef FMOE_RSTD_DIRECT
#define FMOE_RSTD_DIRECT 1               // 1: every token CTA publishes its stage-1 rstd as ONE LL message (P.ssq[t]) the moment it is known (~1.6 us) and
#endif                                   //    the router CTAs just gather those M values while their k-slices land; 0: the router CTAs recompute rstd
                                         //    from their own slices (24 partials/token -> 16-k-group LL exchange: published 2.9, gathered 3.4 us)
#ifndef FMOE_ROUTER_TMA
#define FMOE_ROUTER_TMA 0                // 1: a router CTA fetches its unit with four async copies from ONE thread -- 3-D TMA boxes (SWIZZLE_128B) for the
#endif                                   //    hidden / residual k-slices and the bf16 weight tile ([k64 seg 6][row 64][128 B]) + a bulk copy of the norm-w slice,
                                         //    on two prologue mbarriers -- instead of ~6200 16-B cp.asyncs from all 640 threads (1.6 us of LSU issue that also
                                         //    queued the rstd poll behind it: message out at 1.6 us, gathered at 2.8). The maps ride in the __grid_constant__
                                         //    kernel parameter (no tensormap acquire fence). Needs FMOE_ROUTER_BF16. Default 0: measured (in probes,
                                         //    router publish, 4-case median) 5.77 / 5.36 / 5.64 / 5.35 us for W_MODE 0 / 1 / 2 / 0+DELAY 400 against 5.20 with
                                         //    the cp.async prologue -- the rstd poll round trip stretches to ~1.5 us whenever the dense TMA W burst is in the
                                         //    L2/HBM queues, and W issued after the poll lands late (4.3); the LSU-limited cp.async trickle interleaves best.
#ifndef FMOE_ROUTER_POLL_FIRST
#define FMOE_ROUTER_POLL_FIRST 0         // 1 (cp.async router prologue): warp 0 issues none of the ~6200 prologue cp.asyncs and polls the rstd messages from
#endif                                   //    kernel start, instead of queueing its first poll behind the CTA's own copies (gathered 2.8 us for a 1.6-us message).
                                         //    Probe: gathered 2.99 (the polls queue behind the other 19 warps' copies just the same), publish 5.10 vs 5.20 -> off
#ifndef FMOE_TASK_TABLE_FUSED
#define FMOE_TASK_TABLE_FUSED 1          // 1: the token-task table's cross-warp scan (warp 0 serial pass + barrier, ~0.3 us on the tables -> tile0 chain) is
#endif                                   //    folded into the table-writing pass (every thread sums the 8 warp totals itself); identical tables
#ifndef FMOE_EARLY_FENCE_HOIST
#define FMOE_EARLY_FENCE_HOIST 1         // 1: the fence.proxy.async that precedes the early first-tile TMAs is executed by every thread right after the
#endif                                   //    union tables' barrier (their ring-region writes are the ones it orders) instead of by the issuing lane after
                                         //    the task table -- off the single-warp chain task table -> first TMA (0.4 us between the map prefetch and the issue)
#ifndef FMOE_ROUTER_TMA_W_MODE
#define FMOE_ROUTER_TMA_W_MODE 0         // when to fire the 48-KB weight box (4.7 MB from HBM over the 96 units): 0 = with the slices at kernel start,
#endif                                   //    1 = once the slices landed (~1.4 us), 2 = once the rstd messages are gathered (~2.6 us)
#ifndef FMOE_ROUTER_TMA_DELAY_NS
#define FMOE_ROUTER_TMA_DELAY_NS 400     // > 0: a router CTA waits this long (globaltimer) before issuing anything, yielding the first memory-system
#endif                                   //    window to the 32 token CTAs' stage-1 loads (fired at t = 0 the 10-MB router burst delayed their rstd 1.6 -> 2.2 us)
#ifndef FMOE_TASK_REGIME_PLAN
#define FMOE_TASK_REGIME_PLAN 1          // 1: the FC2 helper / joint two-round plan is selected by the task-count regime (n_tasks) instead of the
#endif                                   //    U thresholds that only proxy it. (a) U <= 66 token-task loads whose N16 count exceeds 132
                                         //    (hot experts: 2*(U + experts>16) > 132) stream two N8 waves but had NO FC2 helpers and the LATE_SPLIT
                                         //    tail order, so CTAs 0..25 carried a tail AND streamed U/2 experts alone while 36 CTAs idled 50 us
                                         //    (103-case U=64-66: 93-96 us vs 75.8 at U=62 / 83 at U=68; head122 stamps). Now: joint plan + tickets.
                                         //    (b) U <= 108 loads with n_tasks > 216 (no joint plan) take the U > 108 late-split FC2 plan instead of
                                         //    the "everyone leaves FC1 together" balanced split (owners carry tails there too).
#ifndef FMOE_LATE_FREE_B_LOW
#define FMOE_LATE_FREE_B_LOW 1           // 1: late-split FC2 plan: a tail-free late owner (busy < 96: CTAs busy..95, dual tiles) streams the low,
#endif                                   //    wave-1-complete experts and its tail-carrying A the cold ones
#ifndef FMOE_JOINT_TAIL_SINGLE_FIRST
#define FMOE_JOINT_TAIL_SINGLE_FIRST 0   // 1: joint plan tail tickets go to the SINGLE-tile late owners (tiles 12..35) before the dual-tile ones
#endif                                   //    (0..11, then 36..47) so that with <= 24 tail tasks no dual tile carries a tail.
                                         //    net noise on the 16 joint cases with tail <= 34 (U69/66/64 tail 18-20: -1.9/-1.0/-0.6; U65/64/68 tail
                                         //    24-30: +0.6/+0.6/+1.0, reproducible). Stamps: a dual tile's makespan does not move -- while its B is
                                         //    away on the tail, A+H stream at 0.78 us/expert instead of 1.2 (fewer concurrent streams), so the tail
                                         //    is nearly free there and the tail-free plan just gives B 27 experts at the slow rate. Kept off.
#ifndef FMOE_ONEWAVE_KSPLIT
#define FMOE_ONEWAVE_KSPLIT 0            // 1: one-wave N16 loads with n_tasks < 132 split every task's K range: the idle CTAs n_tasks..131 take the
#endif                                   //    last 24 (2*n_tasks <= 132) or 12 (n_tasks <= 3*idle) k-tiles of up to three owners' tasks and publish fp32
                                         //    partials (part_buf / part_flags[256..]) the owner adds before SiLU. U26 had 66
                                         //    CTAs with no FC1 task at all while the 66 owners ran 29 us; pieces are whole 12-tile ring periods.
                                         //    Measured: U26 -10.2, U30 -5.4, U35 -5.0, U40-42 -0.4..-1.2 us,
                                         //    the 24 other one-wave cases neutral, BUT the 75 two-wave cases +0.15 (code placement: none of this
                                         //    runs there; stable across two layouts) -> net -0.12 on the mean, below the commit bar. Kept OFF;
                                         //    turn on for serving mixes dominated by U <= 40 layers (n_tasks <= 99).
#ifndef FMOE_INPUT_TP
#define FMOE_INPUT_TP 1                  // 1 (default since it won: E2E 10.906 -> 10.351 ms, bitwise parity): builds the exact-M32 TP8 entry
                                         //    `fused_moe_full_tp` whose input is this rank's UN-REDUCED o_proj
#endif                                   //    partial: the input all-reduce is absorbed into the prologue as LL pushes (reduce-scatter of K-slices to
                                         //    the owning rank, all-gather of the reduced bf16 slices + sum-of-squares partials, all epoch-tagged and
                                         //    data-carrying: no input-ready flags, no peer reads), and the router runs TP-sharded (rank r: its two
                                         //    k-groups of 384 for all 384 experts, 12 units, 590 KB of bf16 weight instead of 4.7 MB), partial logits
                                         //    pushed to the token owner (t % 8) in the unchanged per-unit LL format, top-8 lists pushed to every rank.
                                         //    Bitwise: reduced values = stock CustomAllReduceV2 (fp32 acc from rank 0, += ranks 1..7, one bf16 RN),
                                         //    rstd = stage-1's 16 k-group partial order, router tiles / sums / stage 3 / 4 unchanged. 0: byte-identical
                                         //    to the eaad992 kernel (the extra instantiation and entry point are not compiled).
#ifndef FMOE_LATE_TRIGGER
#define FMOE_LATE_TRIGGER 1              // 1 = each FC2 CTA issues `griddepcontrol.launch_dependents` when its FC2 compute is done (FC1-only CTAs
#endif                                   //    count by exiting), so the next PDL-launched kernel of the graph -- in SGLang the next layer's flashinfer
                                         //    fused_add_rmsnorm (griddepcontrol.wait at its start) -- is scheduled a few us before this grid ends and hides
                                         //    its launch gap. Independent of how THIS kernel is launched. Trigger at kernel start measured +0.13 ms/step
                                         //    (the dependent's CTAs co-reside with the running MoE CTAs for ~90 us); late trigger measured -0.07..-0.11 ms
                                         //    (E2E G/F/F2 on 332 vs 10.860/10.876 baselines, parity 4/4). 0: never (implicit at exit).
#ifndef FMOE_NORM_NEXT
#define FMOE_NORM_NEXT 1                 // 1 = the INPUT_TP instantiation can run the NEXT decoder layer's input_layernorm + fp8 per-128 activation
#endif                                   //    quant on the reduced output row inside the TP tail (runtime P.norm_next, see norm_next_row): residual_new,
                                         //    x_fp8 and the deep_gemm column-major scales come out of the MoE kernel bitwise equal to flashinfer's
                                         //    fused_add_rmsnorm + SGLang's per_token_group_quant, so layer L+1 skips both kernels. 0: not compiled.
#ifndef FMOE_NN_PREPOLL
#define FMOE_NN_PREPOLL 1                // 1 = a row CTA first waits (one thread, nanosleep backoff) for the last all-gather message of a LOCAL tile
#endif                                   //    of its token before its 512 threads poll all 1056 messages (the row CTAs are free ~20 us before the tail:
                                         //    16K threads spinning on 540 KB of L2 lines the NVLink ingress is writing is not free). 0: poll from entry.
#ifndef FMOE_NN_POLL_NS
#define FMOE_NN_POLL_NS 0                // > 0 = nanosleep between the row stage's poll rounds
#endif
#ifndef FMOE_NN_DRY
#define FMOE_NN_DRY 0                    // 1 = the row CTA runs the norm + quant code once on garbage right at entry (idle window) so the
#endif                                   //    instruction / constant fetches are warm when the real pass runs (cold-cache probe)
constexpr int GATHER_LANES = 32;        // producer lanes issuing the x gather (each ends its tile with one cp.async.mbarrier.arrive.noinc)
template <int EXACT_M> constexpr int FC1_GATHER_THREADS = EXACT_M == 48 ? 64 : GATHER_LANES;
template <int EXACT_M> constexpr bool FC2_PRESCALE = EXACT_M == 32 || EXACT_M == 64;
// Weight layout = Humming's fused-e8m0 W4A8 layout (vllm-project/humming: weight_repack_nk + transform_humming_weight_scale for
// b=e2m1, bs=e8m0 g32, a=e4m3, mma=wgmma, interleave mode 2), i.e. exactly the tensors SGLang's Humming MoE layer holds after
// process_weights_after_loading -- the fused kernel reads them in place, no second copy of the 67 GB/rank of expert weights.
//   w  int32 [E][K/32][4N]: per k32 slice, 64-row blocks of 1 KB = [half 32 rows][32 lanes][4 words]; lane l of half h holds
//        rows h*32 + 16*(q>>1) + 8*(q&1) + l/4 (word q), 8 nibbles each: 0-3 = k (l%4)*4+0..3, 4-7 = k 16+(l%4)*4+0..3
//        (magnitudes in natural order; the sign of element j sits in nibble 2j (j<4) / 2(j-4)+1 (j>=4), bit 3).
//   s   uint8 [E][K/32][N]: exponent offsets o in 1..12, rows permuted within each 64-block (row r stored at 8*(r%8) + r/8).
//   s2  fp32 [E]: per-expert factor; weight = e2m1(code) * 2^(o-6) * s2 * 2^6 (the 2^6 is Humming's e2m1->e4m3 exponent-bias
//        offset, which its own kernel applies in the epilogue). Groups more than 11 octaves below the expert's max were
//        requantized by Humming's transform (process_mxfp4_w4a8), so the kernel needs no flush logic.
// The tiles are fetched with TMA tensor copies (one 8/16-KB box per matrix per k-tile / expert, one box for the offsets)
// into stages laid out [k32 slice][64-row block][1 KB] + [slice][128 B offsets]. A thread's two fragment words (rows
// g, g+8 of its wgmma warp) are 8 B of a 16-B lane chunk; to read them conflict-free (LDS.64, 16 lanes x 16-B stride would
// hit every bank twice) the warps of a pair split the chunks' word pairs by lane octet: lane l of warp wq takes words
// 2*sel, 2*sel+1 with sel = ((l/8) + (wq&1)) & 1, i.e. it holds weight rows 32*(wq>>1) + 16*sel + l/4 (+8). The wgmma row
// index is arbitrary as long as the epilogue maps it back (row_local0 below); the offsets of those rows are the u16 at
// 8*(l/4) + 4*(wq>>1) + 2*sel of the block's 64-B offset run.
constexpr int HL_BLOCK_BYTES = 1024;                              // one 64-row x k32 block
constexpr int FC1_N = 2 * INTER;                                  // w13 rows per expert (gate rows 0..255 | up rows 256..511)
constexpr int FC1_W_SLICE_BYTES = FC1_N * 32 / 2;                 // 8192: one k32 slice of w13 (8 blocks)
constexpr int FC1_W_EXPERT_BYTES = NKT1 * 4 * FC1_W_SLICE_BYTES;  // 1572864
constexpr int FC1_S_SLICE_BYTES = FC1_N;                          // 512 offset bytes per k32 slice
constexpr int FC1_S_EXPERT_BYTES = NKT1 * 4 * FC1_S_SLICE_BYTES;  // 98304
constexpr int FC1_TILE_W_BYTES = 2 * 4 * 2 * HL_BLOCK_BYTES;      // stage: [k32 slice 4][gate|up][row block rh 2][1 KB] = 16384 (one 5D TMA box)
constexpr int FC1_TILE_S_BYTES = 2 * 4 * 128;                     // [k32 slice 4][gate|up][128 B = rh 2 x 64 offsets] = 1024 (one 4D box)
constexpr int FC1_TILE_BYTES = FC1_TILE_W_BYTES + FC1_TILE_S_BYTES;   // 17408 (one 128-row half x k128 of gate+up)
constexpr int FC1_X_BYTES = 8 * 1024;                             // up to 8 chunks of 8 tokens x 128 B
constexpr int FC1_STAGE_BYTES = FC1_TILE_BYTES + FC1_X_BYTES;     // 25600 (multiple of 1024)
constexpr int FC2_W_SLICE_BYTES = DIM * 32 / 2;                   // 98304: one k32 slice of w2 (96 blocks)
constexpr int FC2_W_EXPERT_BYTES = (INTER / 32) * FC2_W_SLICE_BYTES;   // 786432
constexpr int FC2_S_SLICE_BYTES = DIM;                            // 6144 offset bytes per k32 slice
constexpr int FC2_S_EXPERT_BYTES = (INTER / 32) * FC2_S_SLICE_BYTES;   // 49152
constexpr int FC2_TILE_W_BYTES = 8 * 2 * HL_BLOCK_BYTES;          // stage: [k32 slice 8][row block rh 2][1 KB] = 16384
constexpr int FC2_TILE_S_BYTES = 8 * 128;                         // [k32 slice 8][128 B]
constexpr int FC2_W_BYTES = FC2_TILE_W_BYTES + FC2_TILE_S_BYTES;  // 17408 (one 128-row output tile x k256)
constexpr int HST_STRIDE = 132;
constexpr int HST_BYTES = MAXM * HST_STRIDE * 4;   // 33792
constexpr int XS_BYTES = NKT1 * MAXM * 4;          // 12288: xp_s[kt][tig 4][j 8][2] = x_scale[tok(pos = 8j + 2tig + e)][kt] (a lane's 8 chunks are contiguous)
constexpr int MSG_PER_TOK = 43;                    // 128 rows / 3 floats per 16-B message (last: 2 valid)
constexpr int NSTAMP = 28;                         // dbg globaltimer stamps per CTA (16..27: DIAG accumulators)

// Router prologue geometry: unit = (k-group kg of RKS columns, row-group rg of 64 expert rows).
constexpr int RKG = 16;                            // k-groups
constexpr int RKS = DIM / RKG;                     // 384 k per unit = 24 k16 steps = 6 per warpgroup
constexpr int RROWS = 64;                          // expert rows per unit (one wgmma m64)
constexpr int NRG_MAX = NEXP / RROWS;              // 6 row-groups
constexpr int RUNITS_MAX = RKG * NRG_MAX;          // 96
constexpr int RMSG = (RROWS + 2) / 3;              // 22 LL messages (3 fp32 + tag) per (unit, token)
constexpr int W_ELEM = FMOE_ROUTER_BF16 ? 2 : 4;   // router weight element bytes in HBM and in the W tile (bf16 / fp32)
constexpr int W_STRIDE = RKS * W_ELEM + (FMOE_ROUTER_BF16 ? 16 : 32);   // padded rows: 1568 B (fp32, float2 loads) / 784 B (bf16, 196 words:
                                                                          // row g of a fragment lands on banks 4g + tig): conflict-free fragment loads
constexpr int RWG = 4;                             // router GEMM: the first 4 warpgroups split the 24 k16 steps (6 each)
static_assert(RKS % 64 == 0 && (RKS / 16) % RWG == 0 && RWG * 128 <= NTHREADS, "router k-slice must be whole k64 atoms, the WGs split the k16 steps evenly");
// FMOE_INPUT_TP: per-rank IPC "prologue" buffer (uint4 = one 16-B LL message; every region is 32-B aligned and every bulk row an even
// number of messages so a warp's 32 consecutive stores cover whole 32-B sectors):
//   IN_RS [8 src][32 tok][128]: rank src's un-reduced K-slice of token tok destined to THIS rank ({6 bf16, tag}: elements 6m..6m+5 of the 768)
//   XAG   [8 src][32 tok][130]: rank src's REDUCED slice of token tok (msgs 0..127 as above) + msg 128 = {p[2 src], tag, p[2 src+1], tag}
//         (stage-1 sum-of-squares k-group partials, double-tagged: a lone 16-B store is not single-copy atomic); msg 129 pad
//   RPART [96 units][MAXM][22]: the unchanged stage-2 -> stage-3 partial-logit messages, now written by the 8 ranks' router CTAs for the
//         tokens this rank owns (t % 8 == rank); TOPK [MAXM][4]: the unchanged stage-3 -> stage-4 lists, written by every owner rank.
constexpr int TP_NDEV = 8, TP_M = 32;
constexpr int TP_KS = DIM / TP_NDEV;               // 768: this rank's K-slice = k-groups 2r, 2r+1
static_assert(TP_KS == 2 * RKS && TP_KS % 6 == 0, "a K-slice is exactly two router k-groups and a whole number of 6-element messages");
constexpr int TP_RS_ROW = TP_KS / 6;               // 128 messages per (src, tok) slice
constexpr int TP_AG_ROW = TP_RS_ROW + 2;           // 130: + p message + pad (even)
constexpr int TP_PMSG = TP_RS_ROW;                 // index of the p message in an XAG row
constexpr int PRO_IN_RS = 0;
constexpr int PRO_XAG = PRO_IN_RS + TP_NDEV * TP_M * TP_RS_ROW;          // 32768
constexpr int PRO_RPART = PRO_XAG + TP_NDEV * TP_M * TP_AG_ROW;          // 66048
constexpr int TP_TOK_PER_RANK = TP_M / TP_NDEV;    // 4 owned tokens per rank (token t -> owner t % 8, owner-local row t / 8 in RPART)
constexpr int PRO_RP = PRO_RPART + RUNITS_MAX * MAXM * RMSG;             // 201216: [4 owned tokens][16 kg] {p, tag, p, tag} -- the router units' LOCAL
                                                                         //         exchange of stage-1 sum-of-squares k-group partials (FMOE_TP_ROUTER_RS)
constexpr int PRO_TOPK = PRO_RP + TP_TOK_PER_RANK * RKG;                 // 201280 (TOPK stays the LAST region: the host maps it as PRO_BYTES - 4 KB)
constexpr int PRO_U4 = PRO_TOPK + MAXM * (TOPK / 2);                     // 201536 messages = 3224576 B
static_assert(PRO_XAG % 2 == 0 && PRO_RPART % 2 == 0 && PRO_RP % 2 == 0 && PRO_TOPK % 2 == 0 && TP_RS_ROW % 2 == 0 && TP_AG_ROW % 2 == 0, "32-B sector alignment");
constexpr int TP_NUNITS = 2 * NRG_MAX;             // 12 router units per rank (needs n_exp == NEXP)
#ifndef FMOE_TP_W_DELAY_NS
#define FMOE_TP_W_DELAY_NS 2000                    // > 0: a router CTA issues its 48-KB weight tile this long after kernel start (96 CTAs = 4.7 MB, an L2-bandwidth
#endif                                             //    burst even when L2-resident): at t = 0 it delayed the reduce-scatter landing 1.5 -> 1.9 us (stamps3), at
                                                   //    1.5 us it hit the landing window itself (2.15, stamps5); the routers need the tile only at ~4.7 us (warm)
#ifndef FMOE_TP_OWNER_REDUCE
#define FMOE_TP_OWNER_REDUCE 1                     // 1: rank s pushes token t's FULL partial row to owner(t) = t % 8 and the owner's 32 CTAs
#endif                                             //    reduce the (token, slice) pairs of its 4 tokens, so its routers read the reduced slices + p after a
                                                   //    LOCAL hop; the peers get the rows for their stage 1 (off the critical path). 0: rank s reduces slice
                                                   //    s of every token and the owner's routers wait for the cross-rank all-gather (v4).
#ifndef FMOE_TP_S3_DRY
#define FMOE_TP_S3_DRY 0                           // 1: the owner CTA runs a dry stage-3 pass (no gather, no publish) while it waits for the
#endif                                             //    partials -- probes whether the +1.1 us stage-3 slowdown of the TP kernel is a cold-code effect
#ifndef FMOE_TP_ACQ_EARLY
#define FMOE_TP_ACQ_EARLY 2                        // 2: the producer warps' tensormap acquire fences run at the start of the TP prologue instead of
#endif                                             //    after the router phase (probe: do they land on a stage-3 barrier as the floating ~1.1 us stall?)
#ifndef FMOE_TP_S3_FENCE
#define FMOE_TP_S3_FENCE 0                         // 1: every owner-CTA thread executes fence.acq_rel.sys before stage 3 -- if the floating ~1.1 us
#endif                                             //    stall inside stage 3 (stamps11) is the completion of the CTA's own ~900 remote AG stores, the fence
                                                   //    absorbs it in the idle wait for the partials instead of on stage 3's barriers
#ifndef FMOE_TP_S4_BACKOFF_NS
#define FMOE_TP_S4_BACKOFF_NS 0                    // N > 0: stage 4's top-8 / xflags polls back off N ns between misses (probe: do the 128 x 128
#endif                                             //    spinning threads' L2 hot-spot slow the 4 owner CTAs' stage 3?)
#ifndef FMOE_TP_FC1_PREFETCH_KT
#define FMOE_TP_FC1_PREFETCH_KT 0                  // N > 0: after publishing, router CTA u0 L2-prefetches k-tiles 0..N-1 of experts u0 + 96 j
#endif                                             //    (13 MB at N = 1) in the HBM-idle window before the first FC1 tiles are issued
// the TP output tail (last FC2 tile -> kernel end, ~6 us in situ) is NVLink-latency-bound with HBM idle, and the next layer's
// qkv_proj deep_gemm (20.8 MB fp8/rank, starts ~5.5 us after this kernel ends) and o_proj nvjet (25.2 MB bf16, ~28 us) stream those
// weights cold from HBM at ~2 TB/s. The host passes up to 3 global ranges (P.pf_*: the NEXT layer's qkv_proj weight / weight_scale_inv /
// o_proj weight, fixed addresses -> CUDA-graph safe); the 96 FC2 CTAs each cp.async.bulk.prefetch.L2 a contiguous 1/96 share
// (fire-and-forget; may still be in flight at kernel exit; no bit changes). Default 0 = no code, byte-identical.
#ifndef FMOE_TAIL_PREFETCH
#define FMOE_TAIL_PREFETCH 1                       // (default 1 = the measured E2E config) trigger point: 1 = right after this CTA's reduce-scatter push (own FC2 compute done; other CTAs may still
#endif                                             //    stream FC2 weights), 2 = after the 16-source poll of the owned tile (every rank's FC2 for that tile is done),
                                                   //    3 = after the pull (CTA exit; least lead time, zero overlap with this kernel's own traffic)
#ifndef FMOE_TAIL_PREFETCH_CHUNK
#define FMOE_TAIL_PREFETCH_CHUNK 65536             // bytes per cp.async.bulk.prefetch.L2 instruction (a CTA's share is issued as ceil(share / CHUNK) instructions)
#endif
#ifndef FMOE_TAIL_PREFETCH_HINT
#define FMOE_TAIL_PREFETCH_HINT 0                  // 1: L2::cache_hint evict_last on the prefetch (lines survive the qkv stream / attention until o_proj reads them)
#endif
#ifndef FMOE_PF_TEST_KB
#define FMOE_PF_TEST_KB 0                          // bench only: N > 0 makes the python wrapper pass an N-KB dummy range when the caller gives none (coupled bench
#endif                                             //    of the kernel-side cost; the harness never passes ranges)
#ifndef FMOE_TP_ROUTER_TOKENS
#define FMOE_TP_ROUTER_TOKENS 1                    // 1 (design B): the router is partitioned by TOKEN -- rank r runs all 96 units (full K, 4.7 MB weight,
#endif                                             //    loaded from kernel start) for its 4 owned tokens (wgmma N = 8) so the 16 k-group partials stay local
                                                   //    and only the top-8 lists cross ranks (3 cross-rank hops: RS, AG, top-8). 0 (design A): partitioned
                                                   //    by K -- 12 units (k-groups 2r, 2r+1) for all 32 tokens, partials pushed to the token owner (4 hops).
// ---- prologue chain + tail hops, all bitwise-neutral, TP instantiation only ----
#ifndef FMOE_TP_PTR_PARAMS
#define FMOE_TP_PTR_PARAMS 1                       // 1: the 24 peer-buffer pointers (prologue / reduce-scatter / all-gather tables) travel BY VALUE in the kernel
#endif                                             //    parameter (constant bank) instead of int64 tables in device memory: each P.*_bufs[d] load was a dependent
                                                   //    HBM miss on a cold-cache critical path (RS push at kernel entry, tail push, tail AG fan-out, tail pull)
#ifndef FMOE_TP_EPOCH_HINT
#define FMOE_TP_EPOCH_HINT 1                       // 1: the epoch counter line (work[]) is read / bumped with an L2::evict_last policy so it survives the inter-layer
#endif                                             //    traffic (the entry read of work[2] gates the reduce-scatter push: a dependent HBM miss when cold)
#ifndef FMOE_TP_UNION_PREFETCH_KT
#define FMOE_TP_UNION_PREFETCH_KT 2                // N > 0: stage 4 L2-prefetches k-tiles 0..N-1 (both row halves' weights + offsets, 34 KB per k-tile) of every
#endif                                             //    wave-1 union expert (slot < 66) the moment its slot is known, ~1.5 us before the first-tile TMAs are issued
                                                   //    (HBM is idle in that window; unlike an earlier variant, only the routed experts are fetched: ~5 MB, not 13-25 MB)
#ifndef FMOE_TP_S4_FOLD
#define FMOE_TP_S4_FOLD 1                          // 1: stage 4's first-occurrence bits come from one warp ballot per 32 candidates and the slot prefix is summed
#endif                                             //    inline from the 8 words (no smem atomicOr round, no warp-0 scan pass: two barriers less on the tables chain)
#ifndef FMOE_TP_WIDE_PUBLISH
#define FMOE_TP_WIDE_PUBLISH 1                     // 1: the two critical-path fan-outs use one lane per remote store -- the p message (8 lanes of warp 4 instead
#endif                                             //    of 8 dependent-issue stores from thread 128) and the owner's top-8 list (32 lanes instead of 4 x 8)
#ifndef FMOE_TP_TASK_NOINIT
#define FMOE_TP_TASK_NOINIT 1                      // 1: build_m32_token_tasks' misc init moves into stage 4 (published by its last barrier) and the token-task
#endif                                             //    guard misc[39] is recomputed per thread: one 640-thread barrier less on the tables -> task table chain
#ifndef FMOE_TP_EARLY_PRE_SPLIT
#define FMOE_TP_EARLY_PRE_SPLIT 0                  // 1: TP instantiation issues the first task's tiles 0..N-1 right after the task table, BEFORE the role split
#endif                                             //    (= FMOE_EARLY_PRE_SPLIT for the TP path only; re-tried because those tiles are now L2-resident)
#ifndef FMOE_TP_OWNER_PREFETCH
#define FMOE_TP_OWNER_PREFETCH 0                   // 1: the owner CTA L2-prefetches k-tiles 0..1 of its token's 8 winners right after the top-8 (8 lanes, 3 bulk
#endif                                             //    prefetches each) -- ~1.4 us before the lists have crossed NVLink and the union-slot prefetch fires.
                                                   //    Measured neutral -> off
#ifndef FMOE_TP_ROUTER_RS
#define FMOE_TP_ROUTER_RS 0                        // 1: every router unit reduces ITS k-group of the 4 owned tokens straight from the reduce-scatter messages
#endif                                             //    (8 sources in rank order, one bf16 RN: the token CTA's / stock AR's bits), computes that k-group's
                                                   //    sum-of-squares partial and exchanges the 16 partials per token through a local table (PRO_RP): the
                                                   //    token CTA -> router local hop (slice + p, ~1.1 us) leaves the routing chain; the token CTAs' all-gather
                                                   //    only feeds stage 1 (every rank) any more. Needs the router weight earlier (see FMOE_TP_W_DELAY_NS).

// smem map. fc1 CTAs: ring | hst (SiLU(g)*u staging) | xp_s | tok lists | barriers | routing tables.
// fc2 CTAs (after the fc1 phase): ring [0, OFF_OUTS) over the dead fc1 regions, out_s at OFF_OUTS.
constexpr int OFF_RING = 0;
constexpr int RING1_BYTES = NSTAGE1 * FC1_STAGE_BYTES;   // 153600
constexpr int OFF_HSTG = OFF_RING + RING1_BYTES;
static_assert(FC1_STAGE_BYTES % 1024 == 0 && FC1_TILE_BYTES % 1024 == 0, "swizzle atoms need 1024-B alignment");
constexpr int OFF_XS = OFF_HSTG + HST_BYTES;             // xp_s: per-item activation scales by chunk position (layout above)
constexpr int OFF_TOK = 204800;                          // s_tok[64], s_inv[64], p_tok[64], misc[64] ints (>= end of the prologue scratch)
static_assert(OFF_XS + XS_BYTES <= OFF_TOK && OFF_XS % 16 == 0, "fc1 regions fit before the token lists");
constexpr int OFF_OUTS = OFF_TOK - HST_BYTES;            // fc2 phase: ring [0, OFF_OUTS) (NT-dependent stage size), out_s after it
static_assert(OFF_OUTS % 1024 == 0, "fc2 ring end aligned");
constexpr int OFF_BAR = OFF_TOK + 1024;
constexpr int OFF_TAB = OFF_BAR + 128;                   // persistent routing tables (live through fc1 and fc2)
constexpr int TAB_UNION = 0;                             // int   s_union[MAXU]
constexpr int TAB_MASK = TAB_UNION + MAXU * 4;           // u64   s_mask[MAXU]
constexpr int TAB_SLOT = TAB_MASK + MAXU * 8;            // int   s_slot[MAXM][8]
constexpr int TAB_TKW = TAB_SLOT + MAXM * TOPK * 4;      // float s_tkw[MAXM][8]
constexpr int TAB_BYTES = TAB_TKW + MAXM * TOPK * 4;     // 8704
constexpr int OFF_BAR2 = OFF_TAB + TAB_BYTES;            // FMOE_FC2_PREFILL: the FC2 ring's own full[8] | empty[8] mbarriers (live from kernel start)
static_assert(OFF_BAR2 % 8 == 0, "fc2 barriers aligned");
constexpr int OFF_PUB = OFF_BAR2 + (FMOE_FC2_PREFILL ? 128 : 0);   // FC1_PUB_OFFLOAD: [0] items queued (int), [4] s2 table ready (epoch), [16..48) queue[8]
constexpr int SMEM_TOTAL = OFF_PUB + 64;
constexpr int OFF_S2TAB = OFF_XS + XS_BYTES;              // FC1_PUB_OFFLOAD: float2 [MAXU] (fc1_s2 * 64, fc2_s2 * 64) during FC1 (out_s in FC2)
static_assert(OFF_S2TAB + MAXU * 8 <= OFF_TOK, "s2 table fits between the FC1 scale table and the token lists");
constexpr int SMEM_ALLOC = SMEM_TOTAL + 1024;   // slack for manual 1024-B alignment
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert((OFF_TAB + TAB_MASK) % 8 == 0, "u64 masks aligned");
// Prologue scratch (the ring/hst regions are idle before fc1 starts): W fp32 tile, swizzled B tile,
// then (after the wgmmas) the per-WG partial sums / gathered partials / union scratch over the W region.
constexpr int SLICE_BYTES = MAXM * 128 * (RKS / 64);      // a token k-slice tile: [6 k64 columns][NT/8 atoms][1024 B] <= 49152
constexpr int OFF_PW = OFF_RING;                          // [64][W_STRIDE] = 100352 (fp32) / 50176 (bf16, FMOE_ROUTER_BF16)
constexpr int W_TILE_BYTES = FMOE_ROUTER_TMA ? (RKS / 64) * RROWS * 128 : RROWS * W_STRIDE;   // 49152 (TMA box [seg 6][row 64][128 B]) / padded rows
constexpr int OFF_PH = OFF_PW + W_TILE_BYTES;             // hidden k-slice (swizzled bf16), normalized IN PLACE into the wgmma B tile
static_assert(!FMOE_ROUTER_TMA || FMOE_ROUTER_BF16, "FMOE_ROUTER_TMA needs the bf16 router weight");
static_assert(!FMOE_ROUTER_TMA || (OFF_PW % 1024 == 0 && OFF_PH % 1024 == 0), "SWIZZLE_128B TMA destinations are 1024-B aligned");
constexpr int OFF_PR = OFF_PH + SLICE_BYTES;              // residual k-slice, same layout (149504)
constexpr int OFF_P1 = OFF_PR + SLICE_BYTES;              // 198656: stage-1 scratch ss[384] + p[16] (never overlaps in-flight cp.asyncs)
constexpr int SQ_STRIDE = 3 * RMSG;                       // 66 floats per k-group of gathered sum-of-squares partials
constexpr int OFF_PSQ = OFF_P1 + (DIM / 16 + RKG) * 4;    // 200256: sq[16][66] fp32
constexpr int OFF_PNW = OFF_TOK;                          // norm weight k-slice (768 B) over s_tok/s_inv/p_tok (unused before fc1)
constexpr int OFF_PS = OFF_RING;                          // after the wgmmas: psum[RWG][64][NT+8] (<= 55296) | s_part[96][64] (24576) ...
constexpr int OFF_PS_FIRST = OFF_PS + 32768;              // s_first[384] int, s_fbits[16]
constexpr int OFF_PS_TKID = OFF_PS + 36864;               // s_tkid[64][8] int
constexpr int SLICE_VT = (DIM / 16) / RKG;                // 24 "virtual threads" of 16 elements per k-slice (stage-1 reduction order)
// routing-table region doubles as router scratch before stage 4: vs[64][24] partial sums, s_rstd[64], s_pub[64]
constexpr int TABX_VS = 0, TABX_RSTD = MAXM * SLICE_VT * 4, TABX_PUB = TABX_RSTD + MAXM * 4;
constexpr int TABX_PBAR = TABX_PUB + MAXM * 4;            // FMOE_ROUTER_TMA: prologue mbarriers {slices, W}; invalidated before stage 4 reuses the region
static_assert(TABX_PBAR % 8 == 0 && TABX_PBAR + 16 <= TAB_BYTES, "prologue mbarriers fit in the table region");
constexpr int TABX_PEER = TABX_PUB;                       // FMOE_INPUT_TP: uint4* s_peer[8] = the ranks' prologue buffers, cached in smem at kernel entry (a P.pro_bufs[d]
                                                          //    load per remote store put an L2 round trip in front of the p push and the top-8 publish); s_pub is unused in TP
static_assert(TABX_PEER % 8 == 0 && TABX_PEER + 8 * 8 <= TAB_BYTES, "peer pointer cache fits the s_pub slot");
static_assert(OFF_PH % 1024 == 0 && OFF_PR % 1024 == 0, "slice tiles' swizzle atoms need 1024-B alignment");
static_assert(OFF_PSQ + RKG * SQ_STRIDE * 4 <= OFF_TOK, "prologue scratch inside the ring/hst/xs region");
// psum is written after every router WG's wgmma wait + the router named barrier, when the W tile AND the in-place-normalized B tile
// are dead (the next unit's slice/W loads are issued only after the following __syncthreads): it may cover [OFF_PW, OFF_PR).
// (fp32 W tile: 73728 <= 100352 fits over W alone; bf16 W tile: 50176 + 49152 = 99328 >= 73728 needs the B region too.)
static_assert(RWG * RROWS * (MAXM + 8) * 4 <= OFF_PH + SLICE_BYTES, "psum fits over the W + B tiles");
static_assert(NTHREADS >= DIM / 16 && NTHREADS >= NEXP && NTHREADS >= 256 + MAXM, "prologue thread mappings");
static_assert(RUNITS_MAX * RROWS * 4 + NEXP * 8 + 2 * (NTHREADS / 32) * TOPK * 4 + NEXP * 4 <= OFF_PS_FIRST, "stage-3 scratch (+ bias copy) fits");
static_assert(OFF_PS_TKID + MAXM * TOPK * 4 <= OFF_PH, "stage-4 scratch fits");
static_assert(OFF_PS_FIRST + (NEXP + 16 + 16) * sizeof(int) <= OFF_PS_TKID, "union prefixes fit after first-occurrence bits");
static_assert(TABX_PUB + MAXM * 4 <= TAB_BYTES, "router scratch fits in the table region");
static_assert(RKS * 2 <= 3 * 64 * 4, "norm weight slice fits over the token lists");

struct Params {
    // ---- prologue inputs (full path; unused when pre_routed) ----
    const __nv_bfloat16* hidden;     // [M][DIM]
    const __nv_bfloat16* residual;   // [M][DIM]
    const __nv_bfloat16* norm_w;     // [DIM]
    const float* router_w;           // [n_exp][DIM] fp32 (FMOE_ROUTER_BF16 == 0)
    const __nv_bfloat16* router_wb;  // [n_exp][DIM] bf16 (FMOE_ROUTER_BF16 == 1: the fp32 checkpoint weight rounded once, RNE, by the host)
    const float* bias;               // [n_exp] fp32 correction bias
    float eps;
    int n_exp;                       // routed experts (8..384)
    __nv_bfloat16* residual_out;     // [M][DIM] (= hidden + residual, bf16)
    __nv_bfloat16* xn_buf;           // [MAXM][DIM] normed bf16 (rank-local scratch)
    uint8_t* xq_buf;                 // [MAXM][DIM] e4m3 (rank-local scratch; the given x_fp8 when pre_routed)
    float* xs_buf;                   // [MAXM][48] per-token per-k128 scales
    unsigned* xflags;                // [MAXM] epoch-tagged: token normalized/quantized
    uint4* rpart;                    // [96 units][MAXM][22] LL router partials (3 fp32 + tag)
    uint4* topk;                     // [MAXM][4] LL top-8 lists ({id|id<<16, w, w, tag})
    uint4* ssq;                      // [16 kg][22] LL per-k-slice sum-of-squares partials (3 tokens + tag)
    int* union_out;                  // optional [1 + MAXU]: U, union experts (written by CTA 0)
    int pre_routed;                  // 1: routing + quantized x given by the host (legacy entry point)
    // ---- pre-routed inputs (legacy path) ----
    const int* union_experts;        // [U]
    const float* gate_w;             // [M][U] (0 when token not routed; <= 8 nonzeros per token)
    int U;
    // ---- weights ----
    const uint8_t* fc1_w;        // w13 int32 [E][192][2048] (Humming layout, see the constants)
    const uint8_t* fc1_s;        // w13 exponent offsets uint8 [E][192][512]
    const float* fc1_s2;         // w13 per-expert factor [E]
    const uint8_t* fc2_w;        // w2 int32 [E][8][24576]
    const uint8_t* fc2_s;        // w2 exponent offsets uint8 [E][8][6144]
    const float* fc2_s2;         // w2 per-expert factor [E]
    // TMA tensor maps over those tensors, in device memory (encoded on the host and cached per weight set):
    //   [0] w13: int32 {256 words, 2 rh, 2 hf, 2 matrices, 192*E slices}, box {256,2,1,2,4} (16 KB)
    //   [1] s13: uint8 {128,2 hf,2 matrices,192*E}, box {128,1,2,4} (1 KB)
    //   [2] w2: int32 {256,96,8*E}, box {256,2,8} (16 KB); [3] s2: uint8 {6144,8*E}, box {128,8}
    const CUtensorMap* tmaps;
    // FMOE_ROUTER_TMA: per-launch maps over the prologue inputs, carried IN the kernel parameter (__grid_constant__, so no
    // tensormap acquire fence): hidden / residual {64 bf16 = one 128-B k64 column, M tokens, 96 columns} box {64, NT, 6} and router_wb
    // {64 bf16, n_exp rows, 96 columns} box {64, 64, 6}, all SWIZZLE_128B: the boxes land directly in the kernel's swizzled
    // [k64 column][row][128 B] smem layouts (the wgmma B tile and the W fragment tile). Unused (zero) on the pre-routed path.
    alignas(64) CUtensorMap tm_h, tm_r, tm_w;
    int M, M_pad;
    uint8_t* h_buf;              // [MAXU][2 kblock][M_pad][128] e4m3, 128B-swizzled per 8-token atom
    float* cs_buf;               // [MAXU][M_pad][2] combined scale = h_scale * routing weight;
                                // M32/M64 with U<=MAXU/2 use the unused upper half for cs*(fc2_s2*64)
    unsigned* h_flags;           // [MAXU][2] epoch-tagged ready flags
    int m32_chunk_scratch;       // host checked: four additional chunk flags per half
    float* part_buf;             // [U][2 half][2 mat][64 pos][128 rows] fp32 partial of the k-half-1 item (KSPLIT=2 only)
    unsigned* part_flags;        // [U][2] epoch-tagged: k-half-1 partial published
    uint4* const* rs_bufs;       // [ndev] peer ptrs: [48 (src*TPR+lt)][M_pad][43] messages
    uint4* const* ag_bufs;       // [ndev] peer ptrs: [48 tiles][M_pad][43]
    int my_rank, ndev;
    float* out_f32;              // [M][DIM] (legacy fp32 output; used when out_bf16 == nullptr)
    __nv_bfloat16* out_bf16;     // [M][DIM]
    int n_fc1;
    int* work;                   // [4] device counters: fc1 front item queue, CTA completion (self-resetting), epoch, fc1 back (reserve) queue
    int reserve;                 // fc1 items reserved for the fc1-only CTAs (back region of the queue); < 0 = auto (device-side)
    int parts;                   // fc2 CTAs per output tile (1 or 2; experts split by parity, partials are separate all-reduce sources)
    unsigned long long* dbg;     // optional [grid][NSTAMP] stamps; requires FMOE_PHASE_STAMPS or FMOE_DIAG
    int mode;                    // diagnostics only: 1 = fc1 consumers skip compute, 2 = producer skips x gather, 4 = fc2 CTAs exit, 8 = fc2 consumers skip compute
    int fc2_helper_capacity;      // host checked: FC2 flags64..111, disjoint from FC1 tails
    int m32_tail_scratch;         // host-checked spare h storage and FC1 partial-flag capacity
    int ks_capacity;              // host checked: part_buf holds 132 one-wave K-split partials, part_flags reaches KS_FLAG_BASE + 132
    // ---- FMOE_INPUT_TP: input all-reduce absorbed into the prologue ----
    const __nv_bfloat16* partial;    // [M][DIM] this rank's UN-reduced o_proj partial (replaces `hidden`)
    uint4* const* pro_bufs;          // [ndev] peer ptrs: prologue LL buffers (PRO_IN_RS | PRO_XAG | PRO_RPART | PRO_TOPK regions, see the constants)
    int input_tp;                    // 1: TP entry (rpart / topk point into this rank's own pro buffer)
    // ---- FMOE_TP_PTR_PARAMS: the same three peer tables BY VALUE (constant bank; appended so every other field keeps its offset) ----
    uint4* pro_p[TP_NDEV];
    uint4* rs_p[TP_NDEV];
    uint4* ag_p[TP_NDEV];
    // ---- FMOE_NORM_NEXT: the next layer's input_layernorm + fp8 quant on the reduced output row (TP tail, M32) ----
    const __nv_bfloat16* norm_w_next;   // [DIM] next layer's input_layernorm weight
    __nv_bfloat16* res_new;             // [M][DIM] bf16 = out + residual_out (what fused_add_rmsnorm leaves in `residual`)
    uint8_t* xq_next;                   // [M][DIM] e4m3 (the next layer's qkv_proj activation)
    float* xs_next;                     // column-major fp32 scales: (token t, group g) at g * xs_next_stride + t (deep_gemm TMA-aligned layout)
    int xs_next_stride;
    float eps_next;
    int norm_next;                      // 1: the row stage replaces the pull stage (INPUT_TP entry only)
    // ---- FMOE_TAIL_PREFETCH: up to 3 global ranges L2-prefetched by the FC2 CTAs in the TP tail (16-B aligned, bytes % 16 == 0) ----
    const uint8_t* pf_ptr[3];
    unsigned pf_bytes[3];
    unsigned pf_hint;                // bit r: range r is prefetched with an L2::evict_last cache hint (must survive until a later kernel reads it)
};
constexpr int KS_FLAG_BASE = 256;              // part_flags index of one-wave K-split task t (legacy tails use [0,64), FC2 helpers [64,148))
constexpr int KS_PART_FLOATS = NCONS * 8;      // per task: [512 consumer threads][8 accumulator floats] (thread-private slots)

// Keep even runtime-disabled instrumentation out of production codegen.
// Diagnostic builds have separate extension-cache keys via FMOE_NVCC_FLAGS.
#if defined(FMOE_PHASE_STAMPS) || defined(FMOE_DIAG)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(P, k) do { if ((P).dbg) (P).dbg[blockIdx.x * NSTAMP + (k)] = gtimer(); } while (0)
#else
#define STAMP(P, k) do {} while (0)
#endif

// ---------------- PTX helpers ----------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long globaltimer_ns() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }   // production-build timer (delays)
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count)); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst_smem, const void* src_gmem, unsigned bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(smem_u32(dst_smem)), "l"(src_gmem), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// Same copy with an L2 eviction-priority hint (policy from createpolicy.*); the FC2 h/cs tiles are re-read by 47 CTAs
// while the evict_first weight stream passes through L2.
__device__ __forceinline__ void bulk_g2s_hint(void* dst_smem, const void* src_gmem, unsigned bytes, uint64_t* bar, uint64_t policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(smem_u32(dst_smem)), "l"(src_gmem), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
    uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
// A tensor map in global memory written through the generic proxy (host cudaMemcpy) must be acquired by the tensormap proxy
// before the first TMA use; without it a recycled allocator block can hit a stale TMA descriptor-cache entry left by another
// kernel (cudaErrorIllegalInstruction inside the SGLang server). Executed once per kernel by the TMA-issuing lanes.
__device__ __forceinline__ void tma_acquire_map(const CUtensorMap* map) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void tma_prefetch_map(const CUtensorMap* map) {   // descriptor-cache warm-up only (the fence was done earlier by this thread)
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
// L2-only prefetch of a contiguous global range (no smem, no barrier); bytes: multiple of 16.
__device__ __forceinline__ void bulk_prefetch_l2(const void* src_gmem, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src_gmem), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_prefetch_l2_hint(const void* src_gmem, unsigned bytes, uint64_t policy) {
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(src_gmem), "r"(bytes), "l"(policy) : "memory");
}
// CTA idx of n issues its contiguous 1/n share of every host-given range (P.pf_*) as L2-only bulk prefetches (one thread; nothing
// waits for them). share is rounded up to 16 B; the last CTA's share is clamped to the range.
__device__ __forceinline__ void tail_prefetch_share(const Params& P, int idx, int n) {
    uint64_t policy = 0ull;
    const unsigned hint = FMOE_TAIL_PREFETCH_HINT ? 7u : P.pf_hint;
    if (hint) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
#pragma unroll 1
    for (int r = 0; r < 3; ++r) {
        const uint8_t* base = P.pf_ptr[r];
        const unsigned bytes = P.pf_bytes[r];
        if (base == nullptr || bytes == 0u) continue;
        const bool hinted = (hint >> r) & 1u;
        const unsigned share = ((bytes + (unsigned)n - 1u) / (unsigned)n + 15u) & ~15u;
        unsigned off = share * (unsigned)idx;
        const unsigned end = min(off + share, bytes);
        for (; off < end; off += FMOE_TAIL_PREFETCH_CHUNK) {
            const unsigned len = min((unsigned)FMOE_TAIL_PREFETCH_CHUNK, end - off);
            if (hinted) bulk_prefetch_l2_hint(base + off, len, policy); else bulk_prefetch_l2(base + off, len);
        }
    }
}
// PDL: allow the dependent grid (the next kernel launched with programmatic stream serialization) to be scheduled.
__device__ __forceinline__ void griddep_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// L2-only prefetch of a tensor box (no smem, no barrier): warms L2 for a later tma_load of the same coordinates.
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int c0, int c1) {
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                 :: "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* map, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];"
                 :: "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst_smem, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("{ .reg .b64 policy;\n"
                 "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], policy;\n}"
                 :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst_smem, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar) {
    // Packed expert weights stream through L2; avoid displacing the repeatedly
    // consumed activation/flag working set. Both packed data and their 2D
    // offset scales stream; this hint does not change completion semantics.
    asm volatile("{ .reg .b64 policy;\n"
                 "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], policy;\n}"
                 :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar)) : "memory");
}
// No cache hint: the router's activation slices are re-read by the 6 row-group CTAs of a k-group within ~1 us.
__device__ __forceinline__ void tma_load_3d_nohint(void* dst_smem, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_inval(uint64_t* bar) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void tma_load_4d(void* dst_smem, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint64_t* bar) {
    asm volatile("{ .reg .b64 policy;\n"
                 "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], policy;\n}"
                 :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_5d(void* dst_smem, const CUtensorMap* map, int c0, int c1, int c2, int c3, int c4, uint64_t* bar) {
    asm volatile("{ .reg .b64 policy;\n"
                 "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], policy;\n}"
                 :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src, uint32_t src_size) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst_smem)), "l"(src), "r"(src_size) : "memory");
}
__device__ __forceinline__ void cp_async_4(void* dst_smem, const void* src, uint32_t src_size) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(smem_u32(dst_smem)), "l"(src), "r"(src_size) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint64_t* bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_group1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }   // all but the newest group
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
// FMOE_FC2_PREFILL start gate of the three FC2 producer warps: the 16 consumer warps bar.arrive after fc1_consumer (their h/cs/flag
// publish precedes it), the producer warps bar.sync. Named barrier ids in use: 0 join/drain, 1 router WGs / consumers, 2 FC1 producer
// item handoff, 3 producer warpgroup (plan publish), 4 this gate, 5 consumer-only FC1->FC2 join.
constexpr int FC2_START_BAR = 4, FC2_START_THREADS = NCONS + 3 * 32, CONS_JOIN_BAR = 5;
__device__ __forceinline__ unsigned gtimer32() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return (unsigned)t; }   // ns, low 32 bits (production build)
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ld_acquire_cta_s(const int* p) { int v; asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v; }
__device__ __forceinline__ void st_release_cta_s(int* p, int v) { asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v) : "memory"); }
__device__ __forceinline__ uint64_t ld_acquire_gpu_u64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
// Each busy expert owns eight reserved32-bit chunk-flag words. Use its
// first aligned64-bit word as epoch32 | ready8; fallback half flags are
// elsewhere. CAS both resets stale epochs and merges concurrent arrivals.
__device__ __forceinline__ void publish_m32_chunk(unsigned* flags, int u, int hf, int chunk, unsigned epoch) {
    auto* p = reinterpret_cast<uint64_t*>(flags + MAXU * 2 + u * 8);
    const uint64_t tag = uint64_t(epoch) << 32, bit = 1ull << (hf * 4 + chunk);
    uint64_t old = ld_acquire_gpu_u64(p);
    while (true) {
        const uint64_t desired = (unsigned(old >> 32) == epoch ? old : tag) | bit;
        uint64_t observed;
        asm volatile("atom.acq_rel.gpu.global.cas.b64 %0, [%1], %2, %3;"
                     : "=l"(observed) : "l"(p), "l"(old), "l"(desired) : "memory");
        if (observed == old) break;
        old = observed;
    }
}
// FMOE_FC1_PUB_OFFLOAD worker (producer warp 18, exact-M32 N8 token-task FC1): (a) the per-slot factor table, (b) the h-ready publishes
// of the consumers' items in queue order until the sentinel (-1). Queue entry: u << 8 | hf << 4 | chunk.
__device__ __forceinline__ void fc1_pub_worker(const Params& P, unsigned epoch, uint8_t* smem, const int* s_union, const int* s_misc, int lane) {
    int* pub = reinterpret_cast<int*>(smem + OFF_PUB);
    float2* s2tab = reinterpret_cast<float2*>(smem + OFF_S2TAB);
    const int U = s_misc[32];
    for (int u = lane; u < U; u += 32) {
        const int e = s_union[u];
        s2tab[u] = make_float2(__ldg(P.fc1_s2 + e) * 64.0f, __fmul_rn(__ldg(P.fc2_s2 + e), 64.0f));   // = the epilogue's own expressions
    }
    __syncwarp();
    if (lane == 0) {
        st_release_cta_s(pub + 1, (int)epoch);   // table ready (consumers acquire before their first epilogue)
#pragma unroll 1
        for (int k = 0;; ++k) {
            while (ld_acquire_cta_s(pub) <= k) __nanosleep(64);
            const int w = pub[4 + (k & 7)];
            if (w < 0) break;
            publish_m32_chunk(P.h_flags, w >> 8, (w >> 4) & 1, w & 15, epoch);
        }
    }
    __syncwarp();
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
// peer-table accessors. TP instantiation with FMOE_TP_PTR_PARAMS: the pointer comes out of the kernel parameter (LDC, no memory
// dependency); otherwise the device int64 table (a dependent global load, an HBM miss when the cache is cold).
template <bool TP> __device__ __forceinline__ uint4* tp_pro(const Params& P, int d) { if constexpr (TP && FMOE_TP_PTR_PARAMS) return P.pro_p[d]; else return P.pro_bufs[d]; }
template <bool TP> __device__ __forceinline__ uint4* tp_rs(const Params& P, int d) { if constexpr (TP && FMOE_TP_PTR_PARAMS) return P.rs_p[d]; else return P.rs_bufs[d]; }
template <bool TP> __device__ __forceinline__ uint4* tp_ag(const Params& P, int d) { if constexpr (TP && FMOE_TP_PTR_PARAMS) return P.ag_p[d]; else return P.ag_bufs[d]; }
// FMOE_TP_EPOCH_HINT: accesses to the work-counter line with an L2::evict_last policy (the line then outlives the ~50 MB of
// attention / GEMM traffic between two MoE layers; a plain access leaves it at normal priority and the next launch's entry read misses).
__device__ __forceinline__ int ld_evict_last_s32(const int* p) {
    int v;
    asm volatile("{ .reg .b64 pol;\n createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n ld.global.L2::cache_hint.s32 %0, [%1], pol; }" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ int atom_add_evict_last_s32(int* p, int v) {
    int old;
    asm volatile("{ .reg .b64 pol;\n createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n atom.global.add.L2::cache_hint.s32 %0, [%1], %2, pol; }" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void atom_exch_evict_last_s32(int* p, int v) {
    int old;
    asm volatile("{ .reg .b64 pol;\n createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n atom.global.exch.L2::cache_hint.b32 %0, [%1], %2, pol; }" : "=r"(old) : "l"(p), "r"(v) : "memory");
    (void)old;
}
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_ll(const uint4* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) {
    uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
// bf16 helpers: bf16 -> fp32 is a bit shift (exact); packs are round-to-nearest-even.
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
    __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);   // .x (low 16 bits) = lo
    return *reinterpret_cast<uint32_t*>(&r);
}
// Exact 3-way bf16 split of an fp32: x == hi + mid + lo (each term is a bf16 value; 8+8+8 >= 24 bits).
__device__ __forceinline__ void split3(float x, float& hi, float& mid, float& lo) {
    hi = __bfloat162float(__float2bfloat16_rn(x));
    const float r = x - hi;
    mid = __bfloat162float(__float2bfloat16_rn(r));
    lo = r - mid;
}
__device__ __forceinline__ int f2ordered(float f) { const int i = __float_as_int(f); return i >= 0 ? i : (i ^ 0x7FFFFFFF); }

// fc2 CTA -> (tile, part, nparts). PARTS 1/2: the first 48*PARTS CTAs, tile = b % 48, part = b / 48. PARTS 3 = "mixed":
// EVERY CTA of the grid owns an fc2 tile part -- the first n3 = grid - 96 tiles get 3 parts (CTAs 3t..3t+2), the rest 2
// (132 SMs -> 36 x 3 + 12 x 2), so fc2's per-CTA expert count drops to U/3 (U/2) and no CTA idles after fc1.
template <int PARTS> __device__ __forceinline__ void fc2_map(int b, int grid, int& tile, int& part, int& nparts) {
    if constexpr (PARTS < 3) { tile = b % N_FC2_TILES; part = b / N_FC2_TILES; nparts = PARTS; }
    else {
        const int n3 = grid - 2 * N_FC2_TILES;
        if (b < 3 * n3) { tile = b / 3; part = b - 3 * tile; nparts = 3; }
        else { const int c = b - 3 * n3; tile = n3 + (c >> 1); part = c & 1; nparts = 2; }
    }
}
template <int PARTS> __device__ __forceinline__ int fc2_nparts_of_tile(int tile, int grid) {
    if constexpr (PARTS < 3) return PARTS;
    return tile < grid - 2 * N_FC2_TILES ? 3 : 2;
}

// Three-warp FC2 producer (exact M32 only): warp 16 lane 0 polls readiness and copies h/cs, warp 17 lane 0 streams the weight
// boxes, warp 18 lane 0 the offset boxes; each arrives on the stage's full barrier with its own byte count (count 3).
template <int EXACT_M, int PARTS>
__host__ __device__ constexpr bool fc2_split_producer() {
    return EXACT_M == 32 && PARTS == 2 && FMOE_FC2_SPLIT_PRODUCER && FMOE_BATCH_FENCE && !FMOE_MIXED_WIDTH;
}

// Lockstep FC2 consumer (exact M32, needs the split producer and the scale ring): see fc2_consumer.
template <int EXACT_M, int PARTS>
__host__ __device__ constexpr bool fc2_lockstep() {
    return EXACT_M == 32 && PARTS == 2 && FMOE_FC2_LOCKSTEP && FMOE_CS_PERM && fc2_split_producer<EXACT_M, PARTS>();
}

// FC2 ring pre-fill during the consumers' FC1->FC2 join (FMOE_FC2_PREFILL, exact M32): separate FC2 barrier objects, producers gated
// on the consumers' h publish, joint plan built by warp 19. Needs the split producer (three warps) and the out-of-stage scale ring.
template <int EXACT_M, int PARTS>
__host__ __device__ constexpr bool fc2_prefill() {
    return EXACT_M == 32 && PARTS == 2 && FMOE_FC2_PREFILL && FMOE_CS_PERM && KSPLIT == 1 && fc2_split_producer<EXACT_M, PARTS>();
}

template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_asymmetric_cohorts(const Params& P, int U) {
    //132 first FC1 half-expert items, then at most84 tail items on the
    // late cohort. Explicit reserve and other workload sizes keep their
    // existing dynamic queue and round-robin expert decomposition.
    if constexpr (EXACT_M == 32 && PARTS == 2)
        return P.reserve < 0 && U > 66 && U <= 108;
    return false;
}

template <int EXACT_M, int PARTS>
__device__ __forceinline__ void fc2_expert_range(const Params& P, int U, int part, int parts,
                                                int& begin, int& end, int& stride) {
    begin = part; end = U; stride = parts;
    if (m32_asymmetric_cohorts<EXACT_M, PARTS>(P, U)) {
        begin = part == 0 ? 0 : 66;
        end = part == 0 ? 66 : U;
        stride = 1;
    }
}

// One-wave N16 token-task loads (U <= 66): every CTA leaves FC1 together, so the 36 helper CTAs would otherwise idle
// through the whole FC2 phase while 96 owners each stream U/2 experts. misc[58] is only defined when misc[39] is set.
template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_one_wave_tasks(const int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2 && KSPLIT == 1 && FMOE_ONEWAVE_HELPERS) return misc[39] != 0 && misc[58] != 0;
    return false;
}

// Two-wave (N8) token-task loads: the N16 count did not fit one wave. Below U = 66 this happens only with hot experts
// (2*(U + experts>16) > 132); such a load has 132 < n_tasks <= 216 and needs the same helper/joint plan as U > 66.
template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_two_wave_tasks(const int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2 && KSPLIT == 1 && FMOE_TASK_REGIME_PLAN) return misc[39] != 0 && misc[58] == 0;
    return false;
}

// FMOE_GP regime: exact-M32 token-task load, more than 216 tasks (so no joint plan). misc[39] (token tasks) and misc[38] (n_tasks)
// are final after build_m32_token_tasks' last barrier.
constexpr int M32_MAX_TASKS = FMOE_GP ? 132 * FMOE_GP_ROUNDS : 2 * 132;   // token-task admission (table holds 512 descriptors)
static_assert(M32_MAX_TASKS <= 512, "task table capacity");
__device__ __forceinline__ bool m32_gp_active(const int* misc) { return FMOE_GP && misc[39] != 0 && misc[38] > FMOE_GP_MIN_TASKS && misc[48] == 0; }
template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_fc2_helpers(const Params& P, int U, const int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2)
        return P.reserve < 0 && (U > 66 || m32_one_wave_tasks<EXACT_M, PARTS>(misc) || m32_two_wave_tasks<EXACT_M, PARTS>(misc)) &&
               (U <= 132 || m32_gp_active(misc)) &&   // FMOE_GP: helpers for any U of a token-task load (the partial arena holds U + 64 + 84 <= 384 slots)
               P.fc2_helper_capacity && !P.pre_routed && P.out_bf16 && P.mode == 0;
    return false;
}

template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_fc2_dual_helper(const Params& P, int U, const int* misc) {
    return m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc) &&
           (int)blockIdx.x >= 96 && (int)blockIdx.x < 108;
}

__device__ __forceinline__ void m32_joint_partition(int n, int a, int b, int h, bool dual, int& na, int& nb, int& nh);
__device__ __forceinline__ int m32_late_ticket(int b);
__device__ __forceinline__ void m32_gp_fc2_split(int U, int n_tasks, int tile, int& nh, int& nb, int& na);
// FMOE_GP2 per-CTA FC2 plan, built by the idle producer warp 19 during FC1 (m32_gp2_plan) and published by the FC1->FC2 join
// barrier: misc[GP2_MISC + 0/1] = slot range [begin, end) of an owner, or [0, stages) of a helper; misc[GP2_MISC + 2] = helper
// pair-1 first-segment stages nq (tile 36 + h%12), misc[GP2_MISC + 3] = that segment's first slot in tile 36 + h%12.
constexpr int GP2_MISC = 16;   // misc[16..19]: unused by the prologue / FC1 (misc[8..15] = the FC1 item ring)
template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_gp2_active(const Params& P, int U, const int* misc) {
    return FMOE_GP2 && m32_gp_active(misc) && m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc);
}

template <int EXACT_M, int PARTS>
__device__ __forceinline__ void fc2_helper_range(const Params& P, int U, const int* misc,
                                                int& tile, int& part, int& parts,
                                                int& begin, int& end, int& stride) {
    if (m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc)) {
        // Give early FC2 owners60 experts only when FC1 actually leaves
        // CTAs0..47 with one ticket. Above216 token tasks that head start
        // disappears, even when the expert union remains <=108.
        if ((int)blockIdx.x >= 96) tile = (int)blockIdx.x - 96;
        const bool dual_tile = tile < 12 || tile >= 36;
        if constexpr (EXACT_M == 32 && PARTS == 2) {
            if (m32_gp_active(misc)) {
                if constexpr (FMOE_GP2) {   // uniform 2.75-stream layout (m32_gp2_plan): owners get their slot range, helpers [0, stages)
                    stride = 1;
                    if ((int)blockIdx.x >= 96) { part = 1; parts = 2; }
                    begin = misc[GP2_MISC]; end = misc[GP2_MISC + 1];
                    return;
                }
                // FMOE_GP: slot order [helper 0..nh) | owner B [nh, nh+nb) | owner A [nh+nb, U) -- the order in which the crew leaves FC1
                // (the pos order deals extra FC1 tasks to A before B before H), so the earliest-free stream takes the earliest-ready slots.
                int nh, nb, na;
                m32_gp_fc2_split(U, misc[38], tile, nh, nb, na);
                stride = 1;
                if ((int)blockIdx.x >= 96) { part = 1; parts = 2; begin = 0; end = nh; }
                else if (part == 1) { begin = nh; end = nh + nb; }
                else { begin = nh + nb; end = U; }
                return;
            }
            // 108 < U <= 132 token-task loads have no joint plan: the tail wave (items 132..n_tasks-1) runs round-robin
            // on CTAs 0..n_tasks-133 (m32_task_item), so every owner is busy until ~71 us while the helpers above that
            // index are free at ~45 us. the earlier schedule's static split handed those helpers the LAST experts, whose FC1 tail
            // tasks finish at ~71 us as well: each idled ~25 us and still ended 8 us before the owners
            // (U=112). Plan with start offsets instead (a tail task = 26.5 us ~= 32 experts) and let the
            // helper stream the low, wave-1 (ready) experts; the owners take the cold ones after their own tail task.
            // U <= 108 loads with n_tasks > 216 have no joint plan either (m32_task_item hands out the same late-split
            // tickets); the balanced split below assumed every CTA leaves FC1 together while the owners carried the tail.
            if (FMOE_LATE_SPLIT && misc[39] && (U > 108 || (FMOE_TASK_REGIME_PLAN && misc[38] + misc[59] > 216)) && !misc[48]) {
                const int busy = misc[38] - 132;                 // tail tasks = second-round tickets 0..busy-1 (>= 86 here)
                const int t = tile % 36;                          // a dual pair (t, t + 36) shares its helper's range
                constexpr int TAIL = FMOE_LATE_TAIL_UNITS;       // a tail task in A-expert units (earlier schedule: 32)
                const int sh = m32_late_ticket(96 + t) < busy ? TAIL : 0;   // does helper CTA 96+t carry a tail?
                int na, nb, nh;
                m32_joint_partition(U, TAIL, TAIL, sh, dual_tile, na, nb, nh);   // CTAs 0..95 always carry a tail task here
                bool b_low = false;
                if (dual_tile && tile >= 36 && 84 + t >= busy) {   // this pair's second B is free early: re-split A/B only
                    int rest; m32_joint_partition(U - nh, TAIL, 0, 512, false, na, nb, rest); nb += rest;
                    // that B is free at ~42 us, so it must stream the LOW experts right after the helper's range
                    // (wave-1 complete) and leave the cold, tail-wave experts (ready at ~68 us) to the tail-carrying A. With
                    // the ranges the other way round (head638025d stamps, U=105-109 / busy=88-94) B streamed 12 warm
                    // experts, idled ~18 us until the tail wave landed, and ended at 97.5 us -- 3.5 us after every other CTA.
                    b_low = FMOE_LATE_FREE_B_LOW != 0;
                }
                stride = 1;
                if ((int)blockIdx.x >= 96) { part = 1; parts = 2; begin = 0; end = nh; }
                else if (part == 1) { begin = b_low ? nh : nh + na; end = b_low ? nh + nb : U; }
                else { begin = b_low ? nh + nb : nh; end = b_low ? U : nh + na; }
                return;
            }
        }
        // Above the old asymmetric FC1 domain, both owner cohorts may
        // finish FC1 together. Balance stages across two owners plus a
        // whole helper (single tile) or half helper (two interleaved tiles).
        // One-wave loads also leave FC1 together: same balanced split (2U/5 on dual tiles, U/3 on single tiles).
        const bool early_owners = U <= 108 && (!misc[39] || misc[38] <= 216) && !m32_one_wave_tasks<EXACT_M, PARTS>(misc);
        const int early = early_owners ? 60 : (dual_tile ? (2 * U + 4) / 5 : (U + 2) / 3);
        const int helper_experts = dual_tile ? (U - early + 2) / 3
                                             : (U - early + 1) / 2;
        const int cut = U - helper_experts;
        stride = 1;
        if ((int)blockIdx.x >= 96) {
            part = 1; parts = 2;
            begin = cut; end = U; stride = 1;
        } else if (part == 1) {
            begin = early; end = cut;
        } else {
            begin = 0; end = early;
        }
        if constexpr (EXACT_M == 32 && PARTS == 2) {
            if (misc[48]) { begin = 0; end = misc[52]; stride = 1; }
        }
    }
}

// Joint two-wave schedule: leave dual-output helpers at one FC1 task
// whenever the late owners and single-output helpers can cover the tail.
// The last H CTAs (H <= 36: within the helper set 96..131) run one N16 hot-expert task instead and take no tail;
// tickets stay contiguous: late owners 48..95, then single helpers 108..131-H, then dual helpers 96..min(108,132-H).
__device__ __forceinline__ int m32_joint_tail_ticket(int b, int H) {
    if (FMOE_JOINT_TAIL_SINGLE_FIRST && b < 96) {   // late owners: single tiles 12..35 (CTAs 60..83), then 0..11, then 36..47
        const int t = b - 48;
        return t < 12 ? t + 24 : (t < 36 ? t - 12 : t);
    }
    return b < 96 ? b - 48 : (b >= 108 ? b - 60 : b - 96 + 48 + max(24 - H, 0));
}
// Second-round ticket of the two-round token-task loads without a joint plan (n_tasks > 216, i.e. U > 108): the tail tasks
// 132.. go to the owners first, then the single helpers 108..131, and the dual helpers 96..107 LAST.
// the earlier schedule's round-robin (b + 132) handed them to CTAs 0..n_tasks-133, i.e. to the dual helpers before the single helpers:
// with U=112 (234 tasks) the six dual pairs whose helper carried a tail had all five CTAs start FC2 at ~70 us and ended
// 103-106 us while the single tiles ended 92-96 (validation-phases-latesplit stamps).
__device__ __forceinline__ int m32_late_ticket(int b) {
    return b < 96 ? b : (b >= 108 ? b - 12 : b + 24);
}
// ---- FMOE_GP: general plan for token-task loads with n_tasks > 216 ----
// Position of CTA b in the order the extra tasks of a partial FC1 round are dealt. Output tile t is served in FC2 by owner A (CTA t),
// owner B (48 + t) and helper 96 + t % 36; tiles 12..35 have a helper of their own (a "trio", 3 streams), tiles t < 12 share their
// helper with tile t + 36 (a "pair": 5 CTAs for 2 tiles = 2.5 streams per tile). With every CTA at 2 tasks a tile's makespan is
// T_trio = (U + 2TU + eH)/3, T_pair = (2U + 4TU + eH)/5 (pairs ~8 units behind, structural); one extra task raises a trio tile by
// TU/3 and both tiles of a pair by TU/5. The greedy order that keeps the maximum tile makespan lowest after any prefix (checked for
// U 117..200) is trios-A, pairs-A, trios-B, pairs-A', pairs-B, trios-H, pairs-B', pairs-H (a CTA takes at most one task per round).
// Within a crew A leaves FC1 last, then B, then H = the FC2 slot order (H takes the earliest-ready slots).
__device__ __forceinline__ int m32_gp_pos(int b) {
    if constexpr (FMOE_GP2) {   // group-round-robin order of the uniform layout (see m32_gp2_plan): pos = 12 * rank(member b / 12) + b % 12
        // member ranks: A of tiles g/12+g/24+g/36+g -> 1 2 3 0, B -> 5 6 7 4, H -> 8 9 10
        constexpr unsigned long long R = (1ull << 0) | (2ull << 4) | (3ull << 8) | (0ull << 12) | (5ull << 16) | (6ull << 20) | (7ull << 24) |
                                         (4ull << 28) | (8ull << 32) | (9ull << 36) | (10ull << 40);
        const int m = b / 12;
        return 12 * (int)((R >> (4 * m)) & 0xFull) + (b - 12 * m);
    }
    return b < 12 ? b + 24 : (b < 36 ? b - 12 : (b < 60 ? b + 24 : (b < 84 ? b - 24 : (b < 108 ? b + 24 : b - 24))));
}
__device__ __forceinline__ int m32_gp_ntasks_of(int b, int n_tasks) {
    const int p = m32_gp_pos(b);
    return 1 + (132 + p < n_tasks ? 1 : 0) + (264 + p < n_tasks ? 1 : 0);
}
// The task of CTA b at round n (>= n_tasks: none). Round 0 is the identity (task b: the early first tile of pair b/2 holds).
__device__ __forceinline__ int m32_gp_task(int b, int n, int n_tasks) {
    if (n == 0) return b;
    if (n >= FMOE_GP_ROUNDS) return n_tasks;
    const int t = 132 * n + m32_gp_pos(b);
    return t < n_tasks ? t : n_tasks;
}
// FC2 ranges of output tile `tile` (0..47): helper [0, nh), owner B [nh, nh + nb), owner A [nh + nb, U). The crew's FC1 end offsets
// (in expert-stage units) are (tasks - 1) * TU; a dual pair is planned jointly (five streams, 2U experts, the helper's range is the
// SAME for both tiles because it walks one expert index for its two tiles), then each tile's owners split the rest so they end together.
__device__ __forceinline__ int m32_gp_offset(int b, int n_tasks) {
    constexpr int TU = FMOE_GP_TASK_UNITS;
    const int n = m32_gp_ntasks_of(b, n_tasks);
    // CTAs with more tasks than this one keep HBM saturated with FC1 while this one already streams FC2: charge the window
    int busy = 0;
#pragma unroll
    for (int r = 1; r < FMOE_GP_ROUNDS; ++r) if (r >= n) busy += min(max(n_tasks - 132 * r, 0), 132);
    return (n - 1) * TU + (FMOE_GP_EARLY_PCT * TU * busy) / (100 * 132);
}
__device__ __forceinline__ void m32_gp_fc2_split(int U, int n_tasks, int tile, int& nh, int& nb, int& na) {
    const bool dual = tile < 12 || tile >= 36;
    const int h = 96 + tile % 36;
    const int eA = m32_gp_offset(tile, n_tasks), eB = m32_gp_offset(48 + tile, n_tasks);
    const int eH = m32_gp_offset(h, n_tasks) + FMOE_GP_MERGE_UNITS;
    if (!dual) {
        int lo = 0, hi = 2048;
#pragma unroll 1
        for (int k = 0; k < 11; ++k) {
            const int mid = (lo + hi) >> 1;
            const int cap = max(mid - eA, 0) + max(mid - eB, 0) + max(mid - eH, 0);
            if (cap >= U) hi = mid; else lo = mid + 1;
        }
        nh = min(U, max(lo - eH, 0));
        nb = min(U - nh, max(lo - eB, 0));
        na = U - nh - nb;
        return;
    }
    const int mate = tile < 12 ? tile + 36 : tile - 36;
    const int eA2 = m32_gp_offset(mate, n_tasks), eB2 = m32_gp_offset(48 + mate, n_tasks);
    int lo = 0, hi = 2048;
#pragma unroll 1
    for (int k = 0; k < 11; ++k) {
        const int mid = (lo + hi) >> 1;
        const int cap = max(mid - eA, 0) + max(mid - eB, 0) + max(mid - eA2, 0) + max(mid - eB2, 0) + max(mid - eH, 0);
        if (cap >= 2 * U) hi = mid; else lo = mid + 1;
    }
    const int ab = max(lo - eA, 0) + max(lo - eB, 0) + max(lo - eA2, 0) + max(lo - eB2, 0);
    const int nH = min(max(2 * U - ab, 0), max(lo - eH, 0));
    nh = min(nH >> 1, U);                                   // per tile, identical for both tiles of the pair
    const int s = U - nh;                                    // this tile's owners share s experts and should end together
    na = min(max((s + (eB - eA)) >> 1, 0), s);
    nb = s - na;
}
// The FC1 task of CTA blockIdx.x at round n on the task-table paths (must match fc1_producer's assignment exactly).
// One-wave N16 loads and the hot (N16) CTAs of a mixed load take task blockIdx.x only; N8 CTAs take blockIdx.x, then
// on the joint two-wave schedule one tail ticket per late CTA, otherwise round-robin waves of 132. Returns >= misc[38]
// (n_tasks) when the CTA has no further task.
template <bool ONE_WAVE>
__device__ __forceinline__ int m32_task_item(const Params& P, int n, const int* misc) {
    const int b = (int)blockIdx.x, n_tasks = misc[38];
    if constexpr (ONE_WAVE) return n == 0 ? b : n_tasks;
    const int H = misc[59];
    if (FMOE_GP && FMOE_GP_MIN_TASKS < 216 && n_tasks > FMOE_GP_MIN_TASKS && !misc[48]) return m32_gp_task(b, n, n_tasks);   // = m32_gp_active
    // misc[48] (joint plan) is set for every two-wave load with 132 < n_tasks <= 216 and helpers, including U <= 66
    if (P.reserve < 0 && (misc[32] > 66 || (FMOE_TASK_REGIME_PLAN && misc[48])) && misc[32] <= 108 && n_tasks + H <= 216) {
        if (n == 0) return b;
        if (n == 1 && b >= 48) return 132 + (misc[48] ? m32_joint_tail_ticket(b, H) : b - 48);
        return n_tasks;
    }
    if (FMOE_GP && n_tasks > FMOE_GP_MIN_TASKS && !misc[48]) return m32_gp_task(b, n, n_tasks);   // = m32_gp_active (misc[39] holds on this path)
    if (FMOE_LATE_SPLIT) return n == 0 ? b : (n == 1 ? 132 + m32_late_ticket(b) : n_tasks);
    return b + n * 132;
}
// One-wave K-split. n16 = misc[38] tasks sit on CTAs 0..n16-1 and the CTAs n16..131 had no FC1 work at all (a
// 29-us hole per idle CTA). Quarters (12 k-tiles = one even ring lap pair) offloaded per task: 2 when every task has its
// own helper (2*n16 <= 132: owner k-tiles 0..23, helper 24..47), 1 when three tasks per helper cover them (n16 <= 3*idle:
// owner 0..35, helper 36..47 of tasks j, j+idle, j+2*idle), else 0. Owner/helper roles are a pure function of (blockIdx,
// n, misc[38]) so fc1_producer and fc1_consumer<16> agree without a handoff. Helpers never wait on owners (no cycle).
__device__ __forceinline__ int m32_ks_quarters(const Params& P, int n16) {
    if (!FMOE_ONEWAVE_KSPLIT || !P.ks_capacity || n16 >= 132) return 0;
    if (2 * n16 <= 132) return 2;
    return n16 <= 3 * (132 - n16) ? 1 : 0;
}
// Returns the task index of CTA blockIdx.x's n-th piece (>= n16: none) and its k-tile range; helper = publishes a partial.
// Reads only smem (misc[61] = quarters, resolved once in build_m32_token_tasks; misc[38] = n16). The N16 consumer sits at its
// 104-register cap: this real multi-item loop only compiles spill-free together with the two-pass requant, the
// constant-bounded scale fill, the local k index and the thread-private partial slots below.
__device__ __forceinline__ int m32_ks_piece(const int* misc, int n, int& kt0, int& nkt, bool& helper) {
    const int b = (int)blockIdx.x, n16 = misc[38], q = misc[61];
    kt0 = 0; nkt = NKT1; helper = false;
    if (b < n16) { nkt = NKT1 - 12 * q; return n == 0 ? b : n16; }
    if (q == 0) return n16;
    const int idle = 132 - n16, t = (b - n16) + n * idle;
    helper = true; kt0 = NKT1 - 12 * q; nkt = 12 * q;
    return (n < 3 && t < n16) ? t : n16;   // q == 2: idle >= n16, so only n == 0 finds a task
}
template <int EXACT_M, int PARTS>
__device__ __forceinline__ int fc2_union_index(const int* misc, int i) {
    if constexpr (EXACT_M == 32 && PARTS == 2) {
        if (misc[48]) return i < misc[50] ? misc[49] + i : misc[51] + i - misc[50];
    }
    return i;
}

// Balanced makespan T (expert units) for owner A (from a), owner B (from b) and helper H (from h; half rate on dual
// tiles). B waits for H's partial and then merges it (~1 us) before its TP publish, so B and H are planned to end
// ~1 expert before A. The
// rounding slack goes to H, which may only end earlier.
__device__ __forceinline__ void m32_joint_partition(int n, int a, int b, int h, bool dual,
                                                  int& na, int& nb, int& nh) {
    // Biasing this split (SECOND_TASK 36 + dual helper -3 experts overshot A by 4-8 us;
    // B -1 / dual H -2 / single H +1 was ~1 us worse than unbiased on the two-wave cases) did not pay off.
    int lo = 0, hi = 512;
#pragma unroll 1
    for (int k = 0; k < 10; ++k) {
        const int mid = (lo + hi) / 2;
        const int capacity = max(mid - a, 0) + max(mid - b, 0) +
                             (max(mid - h, 0) >> int(dual));
        if (capacity >= n) hi = mid; else lo = mid + 1;
    }
    na = min(n, max(lo - a, 0));
    nb = min(n - na, max(lo - b, 0));
    nh = n - na - nb;
}

// Called by thread 0 of the consumers after the FC1 join (SOLO = false) and, under FMOE_FC2_PREFILL, by warp 19 lane 0 during FC1
// (SOLO = true; same inputs, so the later thread-0 rebuild rewrites identical values while the producers may already read them).
template <int EXACT_M, int PARTS, bool SOLO = false>
__device__ __forceinline__ void build_m32_joint_plan(int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2) {
        if ((SOLO || threadIdx.x == 0) && misc[48]) {
            const int b = blockIdx.x;
            int tile = b < 96 ? b % 48 : b - 96;
            const bool dual = tile < 12 || tile >= 36;
            if (tile >= 36) tile -= 36;  // shared plan for the dual helper's pair
            int start_b, start_h;
            if (misc[48] == 2) {
                // All-light legacy tails: roles 0..tails-1 (CTAs 48..) add a K36 tail + partial merge (~27 experts),
                // the other late CTAs run K12 tails in rounds of ~10 experts each (see fc1_producer's split_tail).
                const int tails = 2 * misc[32] - 132, helpers = 84 - tails;
                auto k12_rounds = [&](int role) { const int q0 = role - tails; return q0 < 0 ? 0 : (tails - 1 - q0) / helpers + 1; };
                // Offsets in A-expert units. the earlier schedule used 27 (K36 tail) / 10 per K12 round, i.e. B and H at A's rate; the
                // (L4) have A ending 7-8 us before B/H on the single tiles (A 79, H 85.6, B 87 waiting on H):
                // B/H stream ~1.2x slower per expert than the early A, so their offsets are scaled.
                start_b = tile < tails ? FMOE_LIGHT_K36_UNITS : FMOE_LIGHT_K12_UNITS * k12_rounds(tile);
                start_h = 48 + tile < tails ? FMOE_LIGHT_K36_UNITS : FMOE_LIGHT_K12_UNITS * k12_rounds(48 + tile);
            } else {
                const int tail = misc[38] - 132, H = misc[59];
                // A second N8 task = 26.5 us. In A-expert units that is ~36: while only the 60 early CTAs stream FC2
                // (44-70 us) an expert-tile costs 0.72 us; once all 132 stream it costs 0.95-1.0 us for every role
                // (per-CTA stamps, U=94: A 53 experts in 41.5 us, B 21 in 20 us, H 20 in 19.5 us).
                // the earlier schedule's 32 left B 4-9 us and the single helpers 3-7 us behind A. an earlier "36 overshoot" was
                // 36 plus a 3-expert dual-helper bias under mixed widths and a 4-lane producer (not a clean A/B).
                constexpr int SECOND_TASK = FMOE_PLAN_SECOND_TASK;
                // B waits for H's partial and merges it (~1.3 us) before its TP publish, and H must therefore end
                // before B: both are planned MERGE_UNITS experts shorter than A.
                constexpr int MERGE_UNITS = FMOE_PLAN_MERGE_UNITS;
                // With the merge prefetch B's compute may end with A: the partial lands in smem while B's
                // last stages stream and the epilogue adds it from there, so only H keeps the MERGE_UNITS head start.
                start_b = (tail > m32_joint_tail_ticket(48 + tile, H) ? SECOND_TASK : 0) + FMOE_PLAN_B_UNITS;   // this pair's tile-t B
                // A hot helper's single N16 task ends ~4.5 us (~5 experts) after the N8 first wave.
                start_h = (96 + tile >= 132 - H ? 5 : (tail > m32_joint_tail_ticket(96 + tile, H) ? SECOND_TASK : 0)) + MERGE_UNITS;
            }
            int wa, wb, wh, ca, cb, ch;
            const int warm = misc[53];
            m32_joint_partition(warm, 0, start_b, start_h, dual, wa, wb, wh);
            m32_joint_partition(misc[32] - warm, max(32, wa), max(32, start_b + wb),
                                max(32, start_h + (wh << int(dual))), dual, ca, cb, ch);
            const int role = b < 48 ? 0 : (b < 96 ? 1 : 2);
            misc[49] = role == 0 ? 0 : (role == 1 ? wa : wa + wb);
            misc[50] = role == 0 ? wa : (role == 1 ? wb : wh);
            misc[51] = warm + (role == 0 ? 0 : (role == 1 ? ca : ca + cb));
            misc[52] = misc[50] + (role == 0 ? ca : (role == 1 ? cb : ch));
        }
    }
}

// ---- FMOE_GP2: per-CTA FC2 plan of the uniform 2.75-stream layout ----
// Group g (0..11) = tiles {g, 12+g, 24+g, 36+g}: owners A (CTA t) / B (CTA 48+t) of the four tiles and the helpers h = g, 12+g, 24+g
// (CTA 96+h). Helper h streams tile h with its WG pair 0 and with pair 1 after a first segment of nq stages for tile 36+g (pair 1
// then flushes that accumulator to arena slot 36+h and restarts); tile 36+g's owner B merges the three quarter partials. All 11
// CTAs of a group are planned to end together: min T with every tile covered (m32_gp2_group). Slot order (= readiness order,
// the earliest-free stream takes the earliest-ready slots): tile h < 36: [helper (both pairs, stage order) | earlier owner |
// later owner]; tile 36+g: [Q(g) | Q(12+g) | Q(24+g) | earlier owner | later owner].
// Extra FC1 tasks of a partial round are dealt group-round-robin (pos = 12 * rank(member) + g; members m = b / 12: A of tiles
// g/12+g/24+g/36+g = 0..3, B = 4..7, H = 8..10; rank order A36, A, B36, B, H).
// The group's FC1 end offsets take three values (1, 2 or 3 tasks): e[k] = m32_gp_offset of a k-task CTA; the 11 members' task counts
// are packed 2 bits each (member m = 0..10: A of tiles g/12+g/24+g/36+g, B of the same, H of g/12+g/24+g). Few live registers: the
// plan runs in the 64-register producer warp 19.
struct M32Gp2Group { int e1, e2, e3; unsigned kbits; };
__device__ __forceinline__ int m32_gp2_e(const M32Gp2Group& G, int m) {
    const int k = (G.kbits >> (2 * m)) & 3;
    return k == 1 ? G.e1 : (k == 2 ? G.e2 : G.e3);
}
// Helper i's stages per pair, the stages it can give to tile 36+g, and its own tile's deficit at makespan T.
__device__ __forceinline__ void m32_gp2_helper_at(const M32Gp2Group& G, int U, int T, int i, int& n0, int& avail, int& need) {
    n0 = max(T - m32_gp2_e(G, 8 + i) - FMOE_GP_MERGE_UNITS, 0) >> 1;
    need = max(U - max(T - m32_gp2_e(G, i), 0) - max(T - m32_gp2_e(G, 4 + i), 0), 0);
    avail = min(n0, 2 * n0 - need);
}
__device__ __forceinline__ bool m32_gp2_feasible(const M32Gp2Group& G, int U, int T) {
    int give = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int n0, avail, need;
        m32_gp2_helper_at(G, U, T, i, n0, avail, need);
        if (avail < 0) return false;   // need > 2 n0
        give += avail;
    }
    return give >= U - max(T - m32_gp2_e(G, 3), 0) - max(T - m32_gp2_e(G, 7), 0);   // tile 36+g's deficit
}
// Builds this CTA's FC2 plan into misc[GP2_MISC..+3] (one thread; warp 19 during FC1). Pure function of (U, n_tasks, blockIdx).
__device__ __forceinline__ void m32_gp2_plan(int U, int n_tasks, int b, int* misc) {
    const int g = b >= 96 ? (b - 96) % 12 : (b % 48) % 12;
    M32Gp2Group G;
    {
        constexpr int TU = FMOE_GP_TASK_UNITS;
        const int r1 = min(max(n_tasks - 132, 0), 132), r2 = min(max(n_tasks - 264, 0), 132);   // tasks of rounds 1 / 2
        G.e1 = (FMOE_GP_EARLY_PCT * TU * (r1 + r2)) / (100 * 132);   // = m32_gp_offset of a 1 / 2 / 3-task CTA
        G.e2 = TU + (FMOE_GP_EARLY_PCT * TU * r2) / (100 * 132);
        G.e3 = 2 * TU;
        G.kbits = 0;
#pragma unroll 1
        for (int m = 0; m < 11; ++m) G.kbits |= (unsigned)m32_gp_ntasks_of(12 * m + g, n_tasks) << (2 * m);
    }
    int lo = 0, hi = 4096;
#pragma unroll 1
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (m32_gp2_feasible(G, U, mid)) hi = mid; else lo = mid + 1;
    }
    const int T = lo;
    // tile 36+g's deficit D, water-filled over the helpers' avail (nq_i = min(avail_i, x), smallest x), excess taken back from capped ones
    const int D = max(U - max(T - m32_gp2_e(G, 3), 0) - max(T - m32_gp2_e(G, 7), 0), 0);
    int x = 0;
    {
        int xl = 0, xh = 4096;
#pragma unroll 1
        while (xl < xh) {
            const int mid = (xl + xh) >> 1;
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) { int n0, avail, need; m32_gp2_helper_at(G, U, T, i, n0, avail, need); sum += min(avail, mid); }
            if (sum >= D) xh = mid; else xl = mid + 1;
        }
        x = xl;
    }
    int excess = -D, sumq = 0, my_n0 = 0, my_nq = 0, q0 = 0, my_base = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) { int n0, avail, need; m32_gp2_helper_at(G, U, T, i, n0, avail, need); excess += min(avail, x); }
    const int me_h = b >= 96 ? (b - 96) / 12 : -1, me_t = b < 96 ? (b % 48) / 12 : -1;
#pragma unroll
    for (int i = 2; i >= 0; --i) {   // highest index first gives a capped stage back (same rule as the analysis mirror)
        int n0, avail, need; m32_gp2_helper_at(G, U, T, i, n0, avail, need);
        int nq = min(avail, x);
        if (excess > 0 && nq == x && x > 0) { --nq; --excess; }
        if (2 * n0 - nq > U) n0 = (U + nq) >> 1;   // a helper never streams more than U slots of its own tile
        sumq += nq;
        if (i == me_h) { my_n0 = n0; my_nq = nq; }
        if (i == me_t) my_base = 2 * n0 - nq;     // tile i's slots taken by its helper
        if (i < me_h) q0 += nq;                   // quarter segments of tile 36+g in helper order
    }
    int begin = 0, end = 0;
    if (b >= 96) {
        end = 2 * my_n0;
    } else {
        const bool is_b = b >= 48;
        const int base = me_t < 3 ? my_base : sumq;
        const int ea = m32_gp2_e(G, me_t), eb = m32_gp2_e(G, 4 + me_t);
        const int s = max(U - base, 0);
        const int cA = max(T - ea, 0), cB = max(T - eb, 0);
        // the two owners share s and end together: na - nb = eB - eA (clamped to their capacities)
        int na = min(max((s + eb - ea) >> 1, max(s - cB, 0)), min(cA, s));
        na = min(max(na, 0), s);
        const int nb = s - na;
        const bool a_first = ea < eb;                      // the owner that leaves FC1 first takes the earlier-ready slots
        const bool me_first = is_b != a_first;
        begin = base + (me_first ? 0 : (a_first ? na : nb));
        end = begin + (is_b ? nb : na);
    }
    misc[GP2_MISC + 0] = begin;
    misc[GP2_MISC + 1] = end;
    misc[GP2_MISC + 2] = my_nq;
    misc[GP2_MISC + 3] = q0;
}
// Helper walk: stage l -> (weight tile, union slot). Stage pairs (2j, 2j+1): j < nq -> (h, j) for pair 0 and (36 + h%12, q0 + j) for
// pair 1; j >= nq -> both pairs on tile h in slot order (2j - nq + p). Tile-h slots [0, stages - nq) in stage order.
__device__ __forceinline__ void m32_gp2_stage(int h, int nq, int q0, int l, int& tile, int& slot) {
    const int p = l & 1, j = l >> 1;
    if (j < nq) { tile = p ? 36 + h % 12 : h; slot = p ? q0 + j : j; }
    else { tile = h; slot = 2 * j - nq + p; }
}

// ---------------- MXFP4 -> e4m3 register dequant (Humming layout) ----------------
// A word holds ONE row's 8 nibbles: magnitudes of k-elements 0..7 in nibbles 0..7 (0-3 = this lane's quad of the low k16 half of
// the slice, 4-7 = the same quad of the high half), the sign of element j pre-permuted (by Humming's repack) into nibble 2j
// (j<4) / 2(j-4)+1 (j>=4) bit 3, so (q << 4) & 0x80808080 / q & 0x80808080 drop the signs straight onto the output bytes.
// The LUT pools hold the e4m3 byte of each magnitude code with the group's exponent offset o (1..12) folded into the exponent
// field (byte = base + 8*o; identical to Humming's fused_dequant_single_for_mxfp4<Float8E4M3>).
__device__ __forceinline__ void make_pools(uint32_t o, uint32_t& lo, uint32_t& hi) {
    lo = 0x0C080000u + o * 0x08080800u;   // codes 0..3 -> {0, 0.5, 1, 1.5} * 2^(o-6) as e4m3 bytes
    hi = 0x1C181410u + o * 0x08080808u;   // codes 4..7 -> {2, 3, 4, 6} * 2^(o-6)
}
template <bool RAW_PRMT = false>
__device__ __forceinline__ void dequant_word(uint32_t q, uint32_t lo, uint32_t hi, uint32_t& w_lo, uint32_t& w_hi) {
    if constexpr (RAW_PRMT) {
        uint32_t mag_lo, mag_hi;
        // One mask for both selectors. The CUDA intrinsic independently clears
        // the selector high bits, producing two masks in the original SASS.
        // Temporaries prevent an output from overwriting a still-live input.
        asm volatile("{ .reg .b32 sel, sel_hi, ml, mh;\n"
                     "and.b32 sel, %2, 0x77777777;\n"
                     "shr.u32 sel_hi, sel, 16;\n"   //
                     "prmt.b32 ml, %3, %4, sel;\n"
                     "prmt.b32 mh, %3, %4, sel_hi;\n"
                     "mov.b32 %0, ml; mov.b32 %1, mh; }"
                     : "=r"(mag_lo), "=r"(mag_hi) : "r"(q), "r"(lo), "r"(hi));
        w_lo = ((q << 4) & 0x80808080u) | mag_lo;
        w_hi = (q & 0x80808080u) | mag_hi;
        return;
    }
    const uint32_t sel = q & 0x77777777u;
    w_lo = ((q << 4) & 0x80808080u) | __byte_perm(lo, hi, sel);    // k 0..3 of this lane's quad (A fragment reg 0 / 1)
    w_hi = (q & 0x80808080u) | __byte_perm(lo, hi, sel >> 16);     // k 16..19 (reg 2 / 3)
}
// 4 consecutive k32 slices of this warpgroup's 64-row block. frag: this thread's 8 B of slice 0 = the words of rows g and g+8
// (block + ((wq>>1)*32 + lane)*16 + (wq&1)*8), FSTRIDE bytes between slices; scl: the 2 offset bytes of rows g, g+8 of slice 0
// (the block's 64-B offset run + 8*g + 2*wq), SSTRIDE between slices. A[s] = {row g k0-3, row g+8 k0-3, row g k16-19,
// row g+8 k16-19} = the m64k32 e4m3 A fragment of slice s. The pools are computed with IMADs on purpose: a 12-entry smem LUT
// (2 LDS.64 per slice) added 32 KB of shared-memory traffic per tile -- twice the fragments themselves.
template <int FSTRIDE, int SSTRIDE, bool RAW_PRMT = false>
__device__ __forceinline__ void dequant_slices4(const uint8_t* frag, const uint8_t* scl, uint32_t A[4][4]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        // two byte loads instead of a u16 load + LOP3 + SHF: the two extracts were INT-pipe ops in a dequant that is INT-issue bound
        // (16 lanes per sub-partition)
        const uint32_t o_a = scl[s * SSTRIDE], o_b = scl[s * SSTRIDE + 1];
        uint32_t lo_a, hi_a, lo_b, hi_b;
        make_pools(o_a, lo_a, hi_a);
        make_pools(o_b, lo_b, hi_b);
        const uint2 q = *reinterpret_cast<const uint2*>(frag + s * FSTRIDE);
        dequant_word<RAW_PRMT>(q.x, lo_a, hi_a, A[s][0], A[s][2]);
        dequant_word<RAW_PRMT>(q.y, lo_b, hi_b, A[s][1], A[s][3]);
    }
}

#ifdef FMOE_DIAG
// diagnostics: the dequant ALU with the fragment words taken from registers (offset loads kept, no fragment loads)
template <int FSTRIDE, int SSTRIDE>
__device__ __forceinline__ void dequant_slices4_noload(const uint8_t* scl, int tid128, uint32_t A[4][4]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t o2 = *reinterpret_cast<const uint16_t*>(scl + s * SSTRIDE);
        uint32_t lo_a, hi_a, lo_b, hi_b;
        make_pools(o2 & 0xFFu, lo_a, hi_a);
        make_pools(o2 >> 8, lo_b, hi_b);
        const uint2 q = make_uint2(0x12345678u ^ (tid128 * 0x9E3779B1u) ^ (s * 0x01010101u), 0x9ABCDEF0u ^ (tid128 * 0x85EBCA6Bu));
        dequant_word(q.x, lo_a, hi_a, A[s][0], A[s][2]);
        dequant_word(q.y, lo_b, hi_b, A[s][1], A[s][3]);
    }
}
// diagnostics: only the loads of the dequant (no LUT ALU)
template <int FSTRIDE, int SSTRIDE>
__device__ __forceinline__ void dequant_slices4_loadonly(const uint8_t* frag, const uint8_t* scl, uint32_t A[4][4]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t o2 = *reinterpret_cast<const uint16_t*>(scl + s * SSTRIDE);
        const uint2 q = *reinterpret_cast<const uint2*>(frag + s * FSTRIDE);
        A[s][0] = q.x; A[s][1] = q.y; A[s][2] = q.x ^ o2; A[s][3] = q.y ^ o2;
    }
}
#endif

// Token list of union slot u from its routed-token mask (bit t set <=> token t routed to slot u),
// ascending; pads to 64 with 0. inv[t] = position of token t (or -1).
__device__ __forceinline__ int build_token_list(unsigned long long mask, int* tok, int* inv, int lane) {
    const unsigned ma = (unsigned)mask, mb = (unsigned)(mask >> 32);
    const bool a = (ma >> lane) & 1u, b = (mb >> lane) & 1u;
    const unsigned lt = (1u << lane) - 1u;
    const int na = __popc(ma);
    const int T = na + __popc(mb);
    if (inv) { inv[lane] = -1; inv[lane + 32] = -1; }
    __syncwarp();
    if (a) { const int pos = __popc(ma & lt); tok[pos] = lane; if (inv) inv[lane] = pos; }
    if (b) { const int pos = na + __popc(mb & lt); tok[pos] = lane + 32; if (inv) inv[lane + 32] = pos; }
    __syncwarp();
    for (int i = T + lane; i < 64; i += 32) tok[i] = 0;
    __syncwarp();
    return T;
}
// Routing weight of (token, union slot) from the per-token top-8 slot table (0 when not routed).
__device__ __forceinline__ float slot_weight(const int* s_slot, const float* s_tkw, int tok, int u) {
    float w = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) if (s_slot[tok * TOPK + k] == u) w = s_tkw[tok * TOPK + k];
    return w;
}

// Operand fences (CUTLASS warpgroup_fence_operand): the wgmma fence/commit/wait asms carry no register operands, so
// without these nvcc may schedule accumulator reads (promote) or A-fragment writes (next dequant) across them; ptxas then
// detects the hazard and serializes the whole pipeline (C7511/C7514 -- every wgmma followed by a wait).
template <int N> __device__ __forceinline__ void fence_operand(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
__device__ __forceinline__ void fence_operand_a(uint32_t (&a)[4][4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(a[i][0]), "+r"(a[i][1]), "+r"(a[i][2]), "+r"(a[i][3]) :: "memory");
}

// ======================= prologue stage 1: residual add + RMSNorm + fp8 quant of token t =======================
// flashinfer FusedAddRMSNormKernel arithmetic (what SGLang's fused_add_rmsnorm runs):
// x = f32(h) + f32(r) (unrounded) feeds both the variance and the output; residual_out = bf16(x);
// rstd = rsqrtf(sum_sq / DIM + eps); normed = bf16((x * rstd) * w). Then per-128 dynamic fp8 quant of the
// bf16 normed values (scale = amax / 448, or 1 for an all-zero group; q = RN-satfinite(v / scale)).
// 384 threads x 16 contiguous elements; a 128-group = 8 consecutive threads.
__device__ __forceinline__ void stage1_token(const Params& P, unsigned epoch, int t, uint8_t* smem) {
    const int tid = threadIdx.x;
    const int k0 = tid * 16;
    float* s_ss = reinterpret_cast<float*>(smem + OFF_P1);   // [384] per-thread partial sums of squares
    float* s_p = s_ss + DIM / 16;                             // [16] per-k-slice sums
    float v[16];
    float ss = 0.f;
    uint4 w0 = make_uint4(0, 0, 0, 0), w1 = w0;
    if (tid < DIM / 16) {
        const uint4* hp = reinterpret_cast<const uint4*>(P.hidden + (size_t)t * DIM + k0);
        const uint4* rp = reinterpret_cast<const uint4*>(P.residual + (size_t)t * DIM + k0);
        const uint4* wp = reinterpret_cast<const uint4*>(P.norm_w + k0);
        const uint4 h0 = hp[0], h1 = hp[1], r0 = rp[0], r1 = rp[1];
        w0 = wp[0]; w1 = wp[1];   // norm weight loads issued with the activation loads (one memory latency, not two)
        const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint32_t rw[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        uint32_t ow[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2 * i] = bf16lo(hw[i]) + bf16lo(rw[i]);
            v[2 * i + 1] = bf16hi(hw[i]) + bf16hi(rw[i]);
            ow[i] = pack_bf16x2_rn(v[2 * i], v[2 * i + 1]);
            ss = fmaf(v[2 * i], v[2 * i], ss);
            ss = fmaf(v[2 * i + 1], v[2 * i + 1], ss);
        }
        uint4* op = reinterpret_cast<uint4*>(P.residual_out + (size_t)t * DIM + k0);
        op[0] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        op[1] = make_uint4(ow[4], ow[5], ow[6], ow[7]);
    }
    // Reduction in a FIXED order shared with the router CTAs (which recompute the same sum from their k-slices):
    // 16 k-slices x 24 sixteen-element partials, sequential inside a slice, sequential over slices -> bit-identical rstd.
    if (tid < DIM / 16) s_ss[tid] = ss;
    __syncthreads();
    if (tid < RKG) { float p = 0.f; for (int j = 0; j < SLICE_VT; ++j) p += s_ss[tid * SLICE_VT + j]; s_p[tid] = p; }
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int kg = 0; kg < RKG; ++kg) tot += s_p[kg];
    const float rstd = rsqrtf(tot / (float)DIM + P.eps);
    // FMOE_RSTD_DIRECT: the router CTAs gather this value (bit-identical to what they used to recompute) instead of exchanging
    // sum-of-squares partials among themselves; published before the normalize/quant/store work below.
    // Message = {rstd, epoch, rstd, epoch}: a lone 16-B store is NOT single-copy atomic (PTX guarantees 8 B) -- with {rstd,0,0,epoch}
    // the eager launch's routers read {0, .., epoch} for 7-55 of 3072 (router, token) pairs (torn halves against the zeroed buffer;
    // the other LL protocols write whole 32-B sectors from one warp instruction and never showed it). Each 8-B half carries its tag.
    if (FMOE_RSTD_DIRECT && tid == 0) st_ll(P.ssq + t, make_uint4(__float_as_uint(rstd), epoch, __float_as_uint(rstd), epoch));
    if (tid < DIM / 16) {
        const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        uint32_t nw[8];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float n0 = __fmul_rn(__fmul_rn(v[2 * i], rstd), bf16lo(ww[i]));
            const float n1 = __fmul_rn(__fmul_rn(v[2 * i + 1], rstd), bf16hi(ww[i]));
            nw[i] = pack_bf16x2_rn(n0, n1);
            v[2 * i] = bf16lo(nw[i]); v[2 * i + 1] = bf16hi(nw[i]);   // the bf16-rounded normed values feed BOTH consumers
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
        uint4* np = reinterpret_cast<uint4*>(P.xn_buf + (size_t)t * DIM + k0);
        np[0] = make_uint4(nw[0], nw[1], nw[2], nw[3]);
        np[1] = make_uint4(nw[4], nw[5], nw[6], nw[7]);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        const float hs = amax > 0.f ? amax / 448.0f : 1.0f;
        uint32_t q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            q[i] = (uint32_t)pack_e4m3x2(v[4 * i] / hs, v[4 * i + 1] / hs) | ((uint32_t)pack_e4m3x2(v[4 * i + 2] / hs, v[4 * i + 3] / hs) << 16);
        *reinterpret_cast<uint4*>(P.xq_buf + (size_t)t * DIM + k0) = make_uint4(q[0], q[1], q[2], q[3]);
        if ((tid & 7) == 0) P.xs_buf[t * NKT1 + (k0 >> 7)] = hs;
    }
    // bar.sync orders every thread's stores before thread 0's cumulative st.release.gpu (no fence.sc needed);
    // the consumers (router cp.async, fc1 cp.async, plain loads) are all generic-proxy.
    __syncthreads();
    if (tid == 0) st_release_gpu(&P.xflags[t], epoch);
}

// ======================= FMOE_INPUT_TP: input all-reduce inside the prologue =======================
// Token CTA smem during the TP prologue (the ring region is idle; the stage-3 scratch at OFF_PS is written only after stage 1):
#ifndef FMOE_TAIL_POLL16
#define FMOE_TAIL_POLL16 1               // 1 = the output reduce-scatter owner polls its 16 sources in ONE round trip (a second thread polls
#endif                                   //    sources 8..15 and parks the decoded values in smem; the owner adds them in the original order -> bitwise identical)
constexpr int OFF_TP_HALF = OFF_RING;                      // [64 (tok, j)][8 sources][float4]: the partner half of the output reduce (dead FC2 ring)
constexpr int OFF_TP_ROW = OFF_RING;                       // [DIM] bf16: the all-gathered reduced row of this CTA's token
constexpr int OFF_TP_SLICE = OFF_TP_ROW + DIM * 2;         // [TP_KS] bf16: the reduced K-slice this CTA owns
static_assert(OFF_TP_SLICE + TP_KS * 2 <= OFF_P1 && OFF_TP_ROW % 16 == 0 && OFF_TP_SLICE % 16 == 0, "TP staging fits before the stage-1 scratch");
// (1) Every CTA pushes its share of the LOCAL partial: (dst rank s != me, token t) rows of 128 messages, 224 rows over the grid
// (CTAs 0..91 two rows, the rest one); a warp's 32 lanes write 32 consecutive messages of one row (16 whole sectors). The
// data carries the epoch, so the owner needs no input-ready flag and never reads peer memory.
// Message indices. IN_RS on the receiving rank: owner-reduce = [src 8][owner-local token 4][slice 8][128] (the owner gets whole rows of its
// 4 tokens), else [src 8][token 32][128] (slice src's owner gets slice `src` of every token). XAG on every rank: slice k of token t =
// [t][k][130] (owner-reduce) or [k][t][130]; message TP_PMSG of a row = {p[2k], tag, p[2k+1], tag}.
__device__ __forceinline__ size_t tp_rs_idx(int src, int t, int k, int m) {
    if constexpr (FMOE_TP_OWNER_REDUCE) return ((size_t)((src * TP_TOK_PER_RANK + t / TP_NDEV) * TP_NDEV + k) * TP_RS_ROW + m);
    else return ((size_t)(src * TP_M + t) * TP_RS_ROW + m);
}
__device__ __forceinline__ size_t tp_ag_idx(int k, int t, int m) {
    if constexpr (FMOE_TP_OWNER_REDUCE) return ((size_t)(t * TP_NDEV + k) * TP_AG_ROW + m);
    else return ((size_t)(k * TP_M + t) * TP_AG_ROW + m);
}
__device__ __forceinline__ void tp_rs_push(const Params& P, unsigned epoch, int grid) {
    const int tid = (int)threadIdx.x - 2 * TP_RS_ROW;   // warps 8..15: the mbarrier-init threads (warps 0..1) are not held up by the partial loads
    if (tid < 0 || tid >= 2 * TP_RS_ROW) return;
    const int row = (int)blockIdx.x + grid * (tid / TP_RS_ROW), m = tid % TP_RS_ROW;
    constexpr int NROWS = (TP_NDEV - 1) * TP_M;          // 224 rows of 128 messages: (dst, token) slices, or (token, slice) of the 28 non-owned tokens
    if (row >= NROWS) return;
    int dst, t, k;
    if constexpr (FMOE_TP_OWNER_REDUCE) {
        const int j = row >> 3, i = j / (TP_NDEV - 1), jj = j - i * (TP_NDEV - 1);
        dst = jj + (jj >= P.my_rank ? 1 : 0);             // owner rank (!= me) of token t = 8 i + dst
        t = TP_NDEV * i + dst; k = row & 7;
    } else {
        const int d7 = row / TP_M;
        dst = d7 + (d7 >= P.my_rank ? 1 : 0);             // destination = slice owner (skip myself)
        t = row - d7 * TP_M; k = dst;
    }
    const uint32_t* src = reinterpret_cast<const uint32_t*>(P.partial + (size_t)t * DIM + k * TP_KS) + 3 * m;   // elements 6m.. of slice k (4-B aligned)
    st_ll(tp_pro<true>(P, dst) + PRO_IN_RS + tp_rs_idx(P.my_rank, t, k, m), make_uint4(src[0], src[1], src[2], epoch));
}
// (2) Token CTA t = reduce-scatter owner of (token t, slice my_rank): thread m (< 128) gathers the 7 peer messages of its 6 elements,
// takes its own from the partial, and reduces in RANK order with an fp32 accumulator initialised from rank 0 and ONE bf16 RN at the
// end -- exactly reduce_vec<fp32 acc> of SGLang's CustomAllReduceV2 (2shot_pull), so the reduced values are bitwise the stock AR result.
// Then the two stage-1 sum-of-squares k-group partials of the slice (p[2r], p[2r+1]: 24 sixteen-element fmaf chains each, summed
// sequentially -- stage 1's exact order) and the all-gather push: the reduced slice (128 msgs) + {p0, tag, p1, tag} to every rank.
// Register budget (the whole kernel compiles at 96): a thread keeps 4 messages in flight, not 8 -- threads 0..127 poll ranks 0..3 and
// sum them (in order) into smem, threads 128..255 poll ranks 4..7 and continue that sum (in order) after one barrier: the same
// sequential fp32 chain, all 8 loads in flight at once. (8 in-flight messages + 8 destination pointers per thread spilled the whole
// kernel, incl. the FC1/FC2 consumers: 152 B stack, 480/1252 B spill st/ld, 373 STL/LDL sites -- bisected to this function.)
constexpr int OFF_TP_ACC = OFF_TP_SLICE + TP_KS * 2;                                   // [128][6] fp32 partial sums of ranks 0..3
static_assert(OFF_TP_ACC + TP_RS_ROW * 6 * 4 <= OFF_P1 && OFF_TP_ACC % 16 == 0, "TP partial-sum staging fits before the stage-1 scratch");
__device__ __forceinline__ uint4* tp_peer_buf(const Params& P, int d, unsigned dep) {   // dep == 0 at runtime, derived from a polled message: the
    return P.pro_bufs[d + (int)dep];                                                     // pointer load cannot be hoisted above the polls by ptxas
}
__device__ __forceinline__ void tp_reduce_token(const Params& P, unsigned epoch, int t, int k, uint8_t* smem) {   // this CTA reduces (token t, slice k)
    const int tid = threadIdx.x, r = P.my_rank, warp = tid >> 5, lane = tid & 31;
    uint32_t* s_slice = reinterpret_cast<uint32_t*>(smem + OFF_TP_SLICE);   // [384] words = 768 bf16
    float* s_ss = reinterpret_cast<float*>(smem + OFF_P1);                    // [48] sixteen-element partials of this slice
    float* s_acc = reinterpret_cast<float*>(smem + OFF_TP_ACC);              // [128][6]
    const uint4* rs_me = tp_pro<true>(P, r) + PRO_IN_RS;
    const int m = tid & (TP_RS_ROW - 1), g = (tid >> 7) & 1;                 // message index, source group (ranks 4g .. 4g+3)
    uint4 pm[4];
    uint4 msg = make_uint4(0u, 0u, 0u, 0u);
    // Sum-of-squares groups (16 consecutive slice elements, stage 1's virtual threads 48 k + g): group g = 12 (warp - 4) + lane, lane < 12,
    // of the four reducing warps 4..7, whose lanes hold elements 6 lane .. +5 of the warp's 192-element run (192 = 12 x 16: groups never
    // straddle warps, so the 16 values come from <= 4 lanes by warp shuffles -- no smem round trip, no CTA barrier).
    const int gl = warp - 4;                                                  // reducing warp index 0..3 (valid for tid 128..255)
    uint4 rr0 = make_uint4(0u, 0u, 0u, 0u), rr1 = rr0;                       // that group's residual, loaded before the wait
    if (gl >= 0 && gl < 4 && lane < 12) {
        const uint4* rp = reinterpret_cast<const uint4*>(P.residual + (size_t)t * DIM + k * TP_KS + 16 * (12 * gl + lane));
        rr0 = rp[0]; rr1 = rp[1];
    }
    if (tid < 2 * TP_RS_ROW) {
        const uint32_t* own = reinterpret_cast<const uint32_t*>(P.partial + (size_t)t * DIM + k * TP_KS) + 3 * m;
        const uint32_t o0 = own[0], o1 = own[1], o2 = own[2];
        unsigned pending = 0xfu;
        if ((r >> 2) == g) pending &= ~(1u << (r & 3));                       // my own contribution comes from the partial, not from a message
#pragma unroll
        for (int j = 0; j < 4; ++j) pm[j] = make_uint4(o0, o1, o2, epoch);
        do {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (pending & (1u << j)) pm[j] = ld_ll(rs_me + tp_rs_idx(4 * g + j, t, k, m));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((pending & (1u << j)) && pm[j].w == epoch) pending &= ~(1u << j);
        } while (pending);
    }
    if (tid < TP_RS_ROW) {   // ranks 0..3, in order
        float acc[6];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[3] = {pm[j].x, pm[j].y, pm[j].z};
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                const float x = bf16lo(w[q]), y = bf16hi(w[q]);
                if (j == 0) { acc[2 * q] = x; acc[2 * q + 1] = y; }
                else { acc[2 * q] = __fadd_rn(acc[2 * q], x); acc[2 * q + 1] = __fadd_rn(acc[2 * q + 1], y); }
            }
        }
#pragma unroll
        for (int q = 0; q < 6; ++q) s_acc[m * 6 + q] = acc[q];
    }
    __syncthreads();
    if (tid == 0) STAMP(P, 16);   // TP: reduce-scatter messages of this token landed (all 7 peers)
    (void)s_slice;
    if (gl >= 0 && gl < 4) {   // warps 4..7 (tid 128..255): += ranks 4..7 in order, one bf16 RN, then everything that depends on the reduced slice
        constexpr int TPB = 6;   // named barrier of the four reducing warps (ids 0..5 are taken by the FC1/FC2 roles)
        float acc[6];
#pragma unroll
        for (int q = 0; q < 6; ++q) acc[q] = s_acc[m * 6 + q];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[3] = {pm[j].x, pm[j].y, pm[j].z};
#pragma unroll
            for (int q = 0; q < 3; ++q) { acc[2 * q] = __fadd_rn(acc[2 * q], bf16lo(w[q])); acc[2 * q + 1] = __fadd_rn(acc[2 * q + 1], bf16hi(w[q])); }
        }
        msg = make_uint4(pack_bf16x2_rn(acc[0], acc[1]), pack_bf16x2_rn(acc[2], acc[3]), pack_bf16x2_rn(acc[4], acc[5]), epoch);
        if (tid == 2 * TP_RS_ROW - TP_RS_ROW) STAMP(P, 24);   // TP diag: reduced slice in registers (thread 128)
        // Priority order on the NVLink egress: (1) this row to the token's OWNER rank (design B: only its routers need it -- 1/7 of the
        // all-gather payload is router-critical), (2) the p message to every rank, (3) the row to the other 6 peers (their stage 1, off
        // the critical path). Design A: the local routers need every token's row -> local copy first.
        uint4* const* s_peer = reinterpret_cast<uint4* const*>(smem + OFF_TAB + TABX_PEER);   // the 8 prologue buffers, cached in smem at kernel entry
        const int own = FMOE_TP_ROUTER_TOKENS ? (t % TP_NDEV) : r;   // owner-reduce: == r (the local routers read this copy)
        st_ll(s_peer[own] + PRO_XAG + tp_ag_idx(k, t, m), msg);
        // sum of squares of group g = 12 gl + lane (lane < 12): its 16 elements are elements 16 lane .. +15 of this warp's run, i.e. offset
        // o = 16 lane - 6 s0 (0, 2 or 4) into the 24 elements held by lanes s0 .. s0 + 3, s0 = 16 lane / 6
        const int s0 = (16 * lane) / 6, o = 16 * lane - 6 * s0;
        uint32_t gw[12];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int src = min(s0 + k, 31);
            gw[3 * k] = __shfl_sync(0xffffffffu, msg.x, src);
            gw[3 * k + 1] = __shfl_sync(0xffffffffu, msg.y, src);
            gw[3 * k + 2] = __shfl_sync(0xffffffffu, msg.z, src);
        }
        const uint32_t rw[8] = {rr0.x, rr0.y, rr0.z, rr0.w, rr1.x, rr1.y, rr1.z, rr1.w};
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // word (o / 2 + i) of the gathered run: o / 2 in {0, 1, 2} -> compile-time selects, no local-memory indexing
            const uint32_t hw = o == 0 ? gw[i] : (o == 2 ? gw[i + 1] : gw[i + 2]);
            const float v0 = bf16lo(hw) + bf16lo(rw[i]), v1 = bf16hi(hw) + bf16hi(rw[i]);
            ss = fmaf(v0, v0, ss);
            ss = fmaf(v1, v1, ss);
        }
        if (lane < 12) s_ss[12 * gl + lane] = ss;
        named_bar_sync(TPB, 128);
        if (tid == 2 * TP_RS_ROW - TP_RS_ROW) STAMP(P, 25);   // TP diag: sum-of-squares groups in smem (after the 128-thread barrier)
        // The remote routers' critical input is the tiny p message: it leaves BEFORE this token's 7 x 2 KB of slice messages (which only feed
        // the peers' stage 1, off the critical path), so it is not queued behind ~0.5 MB of all-gather payload on the NVLink egress.
        if constexpr (FMOE_TP_WIDE_PUBLISH) {
            // warp 4 -- lane 0 sums k-group 2k, lane 1 k-group 2k+1 (each in stage 1's sequential 24-term order, loads first), the
            // sums are broadcast by shuffle and lanes 0..7 store the double-tagged p message to rank `lane` IN PARALLEL (8 remote stores
            // in one issue slot instead of 8 dependent-issue stores from one thread on the routers' rstd critical path).
            if (gl == 0) {
                float pj = 0.f;
                if (lane < 2) {
                    float pv[SLICE_VT];
#pragma unroll
                    for (int j = 0; j < SLICE_VT; ++j) pv[j] = s_ss[lane * SLICE_VT + j];
#pragma unroll
                    for (int j = 0; j < SLICE_VT; ++j) pj += pv[j];
                }
                const float p0 = __shfl_sync(0xffffffffu, pj, 0), p1 = __shfl_sync(0xffffffffu, pj, 1);
                const uint4 pmsg = make_uint4(__float_as_uint(p0), epoch, __float_as_uint(p1), epoch);
                if (lane < TP_NDEV) st_ll(s_peer[lane] + PRO_XAG + tp_ag_idx(k, t, TP_PMSG), pmsg);
                if (lane == 0) STAMP(P, 17);   // TP: p message of this token issued
            }
        } else
        if (tid == 2 * TP_RS_ROW - TP_RS_ROW) {   // thread 128: the two k-group sums in stage 1's sequential 24-term order (loads first), one double-tagged message per rank
            float pv[2 * SLICE_VT];
#pragma unroll
            for (int j = 0; j < 2 * SLICE_VT; ++j) pv[j] = s_ss[j];
            float p0 = 0.f, p1 = 0.f;
#pragma unroll
            for (int j = 0; j < SLICE_VT; ++j) p0 += pv[j];
#pragma unroll
            for (int j = 0; j < SLICE_VT; ++j) p1 += pv[SLICE_VT + j];
            const uint4 pmsg = make_uint4(__float_as_uint(p0), epoch, __float_as_uint(p1), epoch);
            st_ll(s_peer[own] + PRO_XAG + tp_ag_idx(k, t, TP_PMSG), pmsg);   // the local routers' rstd input first
#pragma unroll
            for (int d = 0; d < TP_NDEV; ++d) if (d != own) st_ll(s_peer[d] + PRO_XAG + tp_ag_idx(k, t, TP_PMSG), pmsg);
            STAMP(P, 17);   // TP: p message of this token issued
        }
        named_bar_sync(TPB, 128);
#pragma unroll
        for (int d = 0; d < TP_NDEV; ++d) {   // all-gather of the reduced slice to the remaining ranks (32 lanes = 32 consecutive messages per destination)
            if (d != own) st_ll(s_peer[d] + PRO_XAG + tp_ag_idx(k, t, m), msg);
        }
    }
}
// (3) Stage 1 of token t on the all-gathered row: the 8 x 128 slice messages + 8 p messages land in the local XAG buffer; the row is
// unpacked into smem, rstd = rsqrtf(sum_{kg = 0..15} p[kg] / DIM + eps) in k-group order (stage 1's exact sum), then stage 1's
// residual add / normalize / fp8 quant / stores verbatim (hidden read from smem instead of P.hidden).
__device__ __forceinline__ void stage1_token_tp(const Params& P, unsigned epoch, int t, uint8_t* smem) {
    const int tid = threadIdx.x;
    uint32_t* s_row = reinterpret_cast<uint32_t*>(smem + OFF_TP_ROW);       // [3072] words = 6144 bf16
    float* s_p = reinterpret_cast<float*>(smem + OFF_P1) + DIM / 16;        // [16] k-group partial sums (same slot as stage 1's s_p)
    const uint4* xag_me = tp_pro<true>(P, P.my_rank) + PRO_XAG;
    {
        constexpr int NH = TP_NDEV * TP_RS_ROW;   // 1024 slice messages of this token
        constexpr int P0 = 384;                    // threads 384..391 also poll the p message of rank tid - 384
        static_assert(NH - NTHREADS <= P0 && P0 + TP_NDEV <= NTHREADS, "message-to-thread mapping");
        const int g1 = tid + NTHREADS;
        unsigned pending = 1u | (g1 < NH ? 2u : 0u) | ((tid >= P0 && tid < P0 + TP_NDEV) ? 4u : 0u);
        uint4 m0, m1, mp;
        do {
            if (pending & 1u) m0 = ld_ll(xag_me + tp_ag_idx(tid >> 7, t, tid & 127));
            if (pending & 2u) m1 = ld_ll(xag_me + tp_ag_idx(g1 >> 7, t, g1 & 127));
            if (pending & 4u) mp = ld_ll(xag_me + tp_ag_idx(tid - P0, t, TP_PMSG));
            if ((pending & 1u) && m0.w == epoch) {   // slice src, message m -> row words src*384 + 3m ..
                uint32_t* d = s_row + (tid >> 7) * (TP_KS / 2) + 3 * (tid & 127);
                d[0] = m0.x; d[1] = m0.y; d[2] = m0.z;
                pending &= ~1u;
            }
            if ((pending & 2u) && m1.w == epoch) {
                uint32_t* d = s_row + (g1 >> 7) * (TP_KS / 2) + 3 * (g1 & 127);
                d[0] = m1.x; d[1] = m1.y; d[2] = m1.z;
                pending &= ~2u;
            }
            if ((pending & 4u) && mp.y == epoch && mp.w == epoch) {   // both 8-B halves carry the tag
                s_p[2 * (tid - P0)] = __uint_as_float(mp.x); s_p[2 * (tid - P0) + 1] = __uint_as_float(mp.z);
                pending &= ~4u;
            }
        } while (pending);
    }
    __syncthreads();
    if (tid == 0) STAMP(P, 18);   // TP: the all-gathered row (8 slices + 8 p messages) landed
    float pv[RKG];
#pragma unroll
    for (int kg = 0; kg < RKG; ++kg) pv[kg] = s_p[kg];   // loads first, then stage 1's sequential k-group sum
    float tot = 0.f;
#pragma unroll
    for (int kg = 0; kg < RKG; ++kg) tot += pv[kg];
    const float rstd = rsqrtf(tot / (float)DIM + P.eps);
    const int k0 = tid * 16;
    if (tid < DIM / 16) {
        const uint4* hp = reinterpret_cast<const uint4*>(s_row) + 2 * tid;
        const uint4* rp = reinterpret_cast<const uint4*>(P.residual + (size_t)t * DIM + k0);
        const uint4* wp = reinterpret_cast<const uint4*>(P.norm_w + k0);
        const uint4 h0 = hp[0], h1 = hp[1], r0 = rp[0], r1 = rp[1];
        const uint4 w0 = wp[0], w1 = wp[1];
        const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint32_t rw[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float v[16];
        uint32_t ow[8], nw[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2 * i] = bf16lo(hw[i]) + bf16lo(rw[i]);
            v[2 * i + 1] = bf16hi(hw[i]) + bf16hi(rw[i]);
            ow[i] = pack_bf16x2_rn(v[2 * i], v[2 * i + 1]);
        }
        uint4* op = reinterpret_cast<uint4*>(P.residual_out + (size_t)t * DIM + k0);
        op[0] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        op[1] = make_uint4(ow[4], ow[5], ow[6], ow[7]);
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float n0 = __fmul_rn(__fmul_rn(v[2 * i], rstd), bf16lo(ww[i]));
            const float n1 = __fmul_rn(__fmul_rn(v[2 * i + 1], rstd), bf16hi(ww[i]));
            nw[i] = pack_bf16x2_rn(n0, n1);
            v[2 * i] = bf16lo(nw[i]); v[2 * i + 1] = bf16hi(nw[i]);
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
        uint4* np = reinterpret_cast<uint4*>(P.xn_buf + (size_t)t * DIM + k0);
        np[0] = make_uint4(nw[0], nw[1], nw[2], nw[3]);
        np[1] = make_uint4(nw[4], nw[5], nw[6], nw[7]);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        const float hs = amax > 0.f ? amax / 448.0f : 1.0f;
        uint32_t q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            q[i] = (uint32_t)pack_e4m3x2(v[4 * i] / hs, v[4 * i + 1] / hs) | ((uint32_t)pack_e4m3x2(v[4 * i + 2] / hs, v[4 * i + 3] / hs) << 16);
        *reinterpret_cast<uint4*>(P.xq_buf + (size_t)t * DIM + k0) = make_uint4(q[0], q[1], q[2], q[3]);
        if ((tid & 7) == 0) P.xs_buf[t * NKT1 + (k0 >> 7)] = hs;
    }
    __syncthreads();
    if (tid == 0) st_release_gpu(&P.xflags[t], epoch);
}

// ======================= FMOE_NORM_NEXT: the next layer's input_layernorm + fp8 quant in the TP tail =======================
// Row CTA b = NN_ROW_CTA0 + t (an FC1-only / FC2-helper CTA, idle once its FC1 join / helper publish is done) polls the 48 x 22 all-gather
// messages of token t (the reduced bf16 MoE output that the pull stage used to copy into `out`) and computes for the NEXT decoder layer
// exactly what flashinfer's FusedAddRMSNormKernel (H = 6144 bf16: 2 rows x 64 threads, 12 vectors of 8 per thread at columns 512 v + 8 t',
// one sequential fma chain per thread -> warp butterfly xor 1..16 -> w0 + w1, div.rn by 6144, rsqrt.approx, y = bf16((h * rstd) * (w + 0)))
// followed by SGLang's per_token_group_quant_flat_kernel<bf16, e4m3, 128> (built with --use_fast_math: amax = fmax(max |bf16|, 1e-10);
// qs = mul.ftz(rcp.approx.ftz(amax), 448); q = e4m3_satfinite(fmin(mul.ftz(qs, y), 448)); scale = mul.ftz(amax, 1/448), column-major fp32)
// produce: residual_new, x_fp8 and the TMA-aligned scales -- BITWISE (numerics probe: 5184 random rows x 4 outputs, 0 mismatches; the
// flashinfer PTX shows 96 fma.rn.f32, div.rn.f32, rsqrt.approx.ftz.f32, add w+0). The CTA also writes `out` (the same bf16 words the pull
// stage wrote), so every gate / fallback that still reads `out` is unchanged.
// v3 -- the work after the last message is instruction-issue bound (v1: 870 + 538 SASS instructions on 2 / 12 warps = 1.3 + 1.4 us warm), so
// everything that does not need rstd runs WHILE the tiles land: warp w owns tiles 3w..3w+2 (66 messages, <= 3 per lane); when a tile's 22
// messages are in (warp vote, no atomics / fences) the warp writes that tile's fp32 h + r to smem, residual_new = bf16(h + r) and `out`.
// After the CTA barrier only the pure 96-fma chain (64 threads, fp32 from smem), the butterfly and the normalize / quant remain.
constexpr int NN_ROW_CTA0 = 96;                     // row CTAs: blockIdx 96 .. 96 + TP_M - 1 (token = blockIdx - 96)
constexpr int NN_BAR = 7;                           // named barrier of the 512 consumer threads of a row CTA (ids 0..6 are taken)
constexpr int NN_AG_BMSG = (128 + 5) / 6;           // 22 all-gather messages per (tile, token): 21 x 6 bf16 + 1 x 2
constexpr int NN_NMSG = N_FC2_TILES * NN_AG_BMSG;   // 1056 messages per token row
constexpr int NN_TPW = N_FC2_TILES / (NCONS / 32);  // 3 tiles per consumer warp
constexpr int NN_WMSG = NN_TPW * NN_AG_BMSG;        // 66 messages per warp, lane l: m = l, l + 32, l + 64 (< 66)
constexpr int OFF_NN_ROW = OFF_RING;                // [DIM] bf16: the reduced output row (dead FC1 / FC2 ring)
constexpr int OFF_NN_V = OFF_NN_ROW + DIM * 2;      // [DIM] fp32: h + r (the norm's input, unrounded)
constexpr int OFF_NN_W2 = OFF_NN_V + DIM * 4;       // [2] fp32: the two warp sums of squares
static_assert(OFF_NN_W2 + 16 <= OFF_OUTS && OFF_NN_ROW % 16 == 0 && OFF_NN_V % 16 == 0, "row-stage staging fits in the dead ring");
static_assert(NN_ROW_CTA0 + TP_M <= 132 && NN_TPW * (NCONS / 32) == N_FC2_TILES && NN_WMSG <= 96 && DIM / 16 <= NCONS && NN_ROW_CTA0 >= 2 * N_FC2_TILES,
              "row-stage thread / CTA mappings");
__device__ __forceinline__ float nn_mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float nn_rcp_approx_ftz(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
// Tile `tile` of token t has landed (its 22 messages unpacked into s_row by this warp): lane l takes elements tile*128 + 4l .. +3:
// v = f32(h) + f32(r) -> s_v (fp32), residual_new = bf16(v) and `out` = h.
__device__ __forceinline__ void nn_tile_landed(const Params& P, uint8_t* smem, int t, int tile, int lane, uint2 r) {
    const uint32_t* s_row = reinterpret_cast<const uint32_t*>(smem + OFF_NN_ROW);
    float* s_v = reinterpret_cast<float*>(smem + OFF_NN_V);
    const int e0 = tile * 128 + 4 * lane;
    const uint2 h = *reinterpret_cast<const uint2*>(s_row + e0 / 2);
    const float v0 = bf16lo(h.x) + bf16lo(r.x), v1 = bf16hi(h.x) + bf16hi(r.x), v2 = bf16lo(h.y) + bf16lo(r.y), v3 = bf16hi(h.y) + bf16hi(r.y);
    *reinterpret_cast<float4*>(s_v + e0) = make_float4(v0, v1, v2, v3);
    *reinterpret_cast<uint2*>(P.out_bf16 + (size_t)t * DIM + e0) = h;
    *reinterpret_cast<uint2*>(P.res_new + (size_t)t * DIM + e0) = make_uint2(pack_bf16x2_rn(v0, v1), pack_bf16x2_rn(v2, v3));
}
// Step B: 384 threads x 16 consecutive elements (a 128-group = 8 consecutive threads = the quant kernel's 8 lanes x 16): y = bf16((v * rstd) *
// (w + 0)), amax over the group, e4m3 quant, column-major scale. v (fp32) from smem.
__device__ __forceinline__ void nn_step_b(const Params& P, uint8_t* smem, int t, float ssum, const uint4& w0, const uint4& w1) {
    const int tid = threadIdx.x, k0 = tid * 16;
    const float* s_v = reinterpret_cast<const float*>(smem + OFF_NN_V);
    if (tid < DIM / 16) {
        const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(ssum, (float)DIM), P.eps_next));
        const float4* vp = reinterpret_cast<const float4*>(s_v + k0);
        float v[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float4 f = vp[i]; v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w; }
        const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // y = bf16((h * rstd) * (w + 0.0)); the quant kernel sees the bf16 values
            const float n0 = __fmul_rn(__fmul_rn(v[2 * i], rstd), __fadd_rn(bf16lo(ww[i]), 0.0f));
            const float n1 = __fmul_rn(__fmul_rn(v[2 * i + 1], rstd), __fadd_rn(bf16hi(ww[i]), 0.0f));
            const uint32_t nw = pack_bf16x2_rn(n0, n1);
            v[2 * i] = bf16lo(nw); v[2 * i + 1] = bf16hi(nw);
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, 1e-10f);
        const float qs = nn_mul_ftz(nn_rcp_approx_ftz(amax), 448.0f);   // 448 / amax under --use_fast_math = MUFU.RCP + FMUL.FTZ
        uint32_t q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            q[i] = (uint32_t)pack_e4m3x2(fminf(nn_mul_ftz(qs, v[4 * i]), 448.0f), fminf(nn_mul_ftz(qs, v[4 * i + 1]), 448.0f)) |
                   ((uint32_t)pack_e4m3x2(fminf(nn_mul_ftz(qs, v[4 * i + 2]), 448.0f), fminf(nn_mul_ftz(qs, v[4 * i + 3]), 448.0f)) << 16);
        *reinterpret_cast<uint4*>(P.xq_next + (size_t)t * DIM + k0) = make_uint4(q[0], q[1], q[2], q[3]);
        if ((tid & 7) == 0) P.xs_next[(k0 >> 7) * P.xs_next_stride + t] = nn_mul_ftz(amax, __uint_as_float(0x3b124925u));   // amax * (1/448)
    }
}
__device__ __forceinline__ void norm_next_row(const Params& P, unsigned epoch, uint8_t* smem, int t) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;   // the consumer threads 0 .. NCONS-1 (the producer warps have exited)
    uint32_t* s_row = reinterpret_cast<uint32_t*>(smem + OFF_NN_ROW);
    const float* s_v = reinterpret_cast<const float*>(smem + OFF_NN_V);
    float* s_w2 = reinterpret_cast<float*>(smem + OFF_NN_W2);
    if (tid == 0) STAMP(P, 16);   // row stage: entered
    // The PDL trigger counts CTAs that called launch_dependents or exited; a row CTA no longer exits after FC1, so it triggers here (the
    // FC2 CTAs trigger after their FC2 compute as before) -- otherwise a PDL-launched successor (deep_gemm with set_pdl) would only be
    // scheduled at grid end.
    if constexpr (FMOE_LATE_TRIGGER) griddep_launch_dependents();
    // Inputs issued ~10 us before the messages land: this warp's 3 tiles' residual_out elements (4 per lane per tile; written by this
    // kernel's stage 1 on other CTAs -> L2-coherent load) and this thread's 16 next-layer norm weights (step B mapping).
    uint2 rr[NN_TPW];
    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = w0;
#pragma unroll
    for (int i = 0; i < NN_TPW; ++i) rr[i] = __ldcg(reinterpret_cast<const uint2*>(P.residual_out + (size_t)t * DIM + (NN_TPW * warp + i) * 128 + 4 * lane));
    if (tid < DIM / 16) { const uint4* wp = reinterpret_cast<const uint4*>(P.norm_w_next + tid * 16); w0 = __ldg(wp); w1 = __ldg(wp + 1); }
    const uint4* ag_me = P.ag_bufs[P.my_rank];
    if constexpr (FMOE_NN_PREPOLL) {   // wait for the tail to start: the last message of this rank's last owned tile for token t (a local push)
        if (tid == 0) {
            const uint4* sentinel = ag_me + ((size_t)(P.my_rank * (N_FC2_TILES / TP_NDEV) + N_FC2_TILES / TP_NDEV - 1) * TP_M + t) * MSG_PER_TOK + (NN_AG_BMSG - 1);
            while (ld_ll(sentinel).w != epoch) __nanosleep(256);
        }
        named_bar_sync(NN_BAR, NCONS);
    }
    {   // poll this warp's 66 messages (tiles 3w .. 3w+2; lane l: m = l + 32 k < 66, tile = m / 22, j = m % 22), all in flight; when a tile
        // is complete (warp vote) the warp runs nn_tile_landed on it -- the loop is warp-uniform
        uint4 m[3];
        unsigned pending = 0u;   // bit k: message l + 32 k still to land
#pragma unroll
        for (int k = 0; k < 3; ++k) if (lane + 32 * k < NN_WMSG) pending |= 1u << k;
        unsigned tiles_todo = (1u << NN_TPW) - 1u;
        do {
#pragma unroll
            for (int k = 0; k < 3; ++k)
                if (pending & (1u << k)) {
                    const int mm = lane + 32 * k, tl = mm / NN_AG_BMSG, j = mm - tl * NN_AG_BMSG;
                    m[k] = ld_ll(ag_me + ((size_t)(NN_TPW * warp + tl) * TP_M + t) * MSG_PER_TOK + j);
                }
#pragma unroll
            for (int k = 0; k < 3; ++k)
                if ((pending & (1u << k)) && m[k].w == epoch) {
                    const int mm = lane + 32 * k, tl = mm / NN_AG_BMSG, j = mm - tl * NN_AG_BMSG;
                    uint32_t* d = s_row + (NN_TPW * warp + tl) * 64 + 3 * j;   // elements tile * 128 + 6 j .. (the last message of a tile carries 2)
                    d[0] = m[k].x;
                    if (j < NN_AG_BMSG - 1) { d[1] = m[k].y; d[2] = m[k].z; }
                    pending &= ~(1u << k);
                }
            // per tile: still pending in this lane? (bit k belongs to tile (l + 32 k) / 22)
            unsigned mine = 0u;
#pragma unroll
            for (int k = 0; k < 3; ++k) if (pending & (1u << k)) mine |= 1u << ((lane + 32 * k) / NN_AG_BMSG);
            const unsigned still = __reduce_or_sync(0xffffffffu, mine);   // tiles with a message still in flight somewhere in the warp
            const unsigned done = tiles_todo & ~still;
            if (done) {
                __syncwarp();   // the other lanes' s_row words of the completed tiles are visible
#pragma unroll
                for (int i = 0; i < NN_TPW; ++i) if (done & (1u << i)) nn_tile_landed(P, smem, t, NN_TPW * warp + i, lane, rr[i]);
                tiles_todo &= ~done;
            }
            if constexpr (FMOE_NN_POLL_NS > 0) { if (tiles_todo) __nanosleep(FMOE_NN_POLL_NS); }
        } while (tiles_todo);
    }
    named_bar_sync(NN_BAR, NCONS);   // every tile's v in smem, residual_new / out stored
    if (tid == 0) STAMP(P, 17);   // row stage: all 1056 messages landed, v computed
    // flashinfer's 64 virtual threads -- thread t' sums v^2 over columns 512 v + 8 t' + e (v 0..11, e 0..7) as ONE fma chain (vector.reduction<add>
    // with acc = 0, ordered), then the warp butterfly (xor 1, 2, 4, 8, 16) and the two warp sums are added (w0 + w1)
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
    named_bar_sync(NN_BAR, NCONS);
    if (tid == 0) STAMP(P, 18);   // row stage: sum of squares done
    nn_step_b(P, smem, t, __fadd_rn(s_w2[0], s_w2[1]), w0, w1);
    if (tid == 0) STAMP(P, 5);   // row stage done (the legacy pull's stamp)
}

// ======================= prologue stage 2: router unit (kg, rg) =======================
// W tile: 64 expert rows x 384 fp32 (rows >= n_exp clamped to a valid row; their scores are discarded).
// (The alternative -- 64 cp.async.bulk row copies through a prologue mbarrier -- measured SLOWER:
// w_landed 4.80 -> 5.00 us and the token slices landed 0.6 us later; 6144 in-flight 16-B cp.asyncs win here.)
__device__ __forceinline__ void issue_w_load(const Params& P, uint8_t* smem, int kg, int rg) {
    constexpr int CPR = RKS * W_ELEM / 16;   // 96 (fp32) / 48 (bf16) x 16-B chunks per row
    constexpr int T0 = FMOE_ROUTER_POLL_FIRST ? 32 : 0;   // warp 0 polls the rstd messages instead (FMOE_ROUTER_POLL_FIRST)
    if ((int)threadIdx.x >= T0) for (int c = threadIdx.x - T0; c < RROWS * CPR; c += NTHREADS - T0) {
        const int row = c / CPR, q = c - row * CPR;
        const int e = min(rg * RROWS + row, P.n_exp - 1);
        const uint8_t* src = FMOE_ROUTER_BF16 ? reinterpret_cast<const uint8_t*>(P.router_wb + (size_t)e * DIM + kg * RKS)
                                              : reinterpret_cast<const uint8_t*>(P.router_w + (size_t)e * DIM + kg * RKS);
        cp_async_16(smem + OFF_PW + row * W_STRIDE + q * 16, src + q * 16, 16u);
    }
    cp_async_commit();
}
// Same weight tile from the producer warpgroup alone (threads NCONS..NTHREADS-1: 24 x 16-B cp.asyncs each); the bf16 build only (input-TP path).
__device__ __forceinline__ void issue_w_load_pg(const Params& P, uint8_t* smem, int kg, int rg) {
    constexpr int CPR = RKS * W_ELEM / 16;   // 48 x 16-B chunks per row (bf16)
    for (int c = (int)threadIdx.x - NCONS; c < RROWS * CPR; c += NTHREADS - NCONS) {
        const int row = c / CPR, q = c - row * CPR;
        const int e = min(rg * RROWS + row, P.n_exp - 1);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(P.router_wb + (size_t)e * DIM + kg * RKS);
        cp_async_16(smem + OFF_PW + row * W_STRIDE + q * 16, src + q * 16, 16u);
    }
    cp_async_commit();
}
// FMOE_ROUTER_TMA: the whole unit from ONE thread -- the two activation boxes and the norm-w slice on pbar[0], the weight box on
// pbar[1]. The k coordinate of k-group kg is its first k64 column (kg * 6); rows >= M / n_exp lie outside the tensor and land as zeros
// (the old cp.async path zero-filled / clamped them the same way).
__device__ __forceinline__ void issue_unit_slices(const Params& P, uint8_t* smem, int kg, int NT, uint64_t* pbar) {
    constexpr int NCOL = RKS / 64;   // 6 k64 columns per unit
    mbar_arrive_expect_tx(&pbar[0], (unsigned)(2 * NT * 128 * NCOL + RKS * 2));
    tma_load_3d_nohint(smem + OFF_PH, &P.tm_h, 0, 0, kg * NCOL, &pbar[0]);
    tma_load_3d_nohint(smem + OFF_PR, &P.tm_r, 0, 0, kg * NCOL, &pbar[0]);
    bulk_g2s(smem + OFF_PNW, P.norm_w + (size_t)kg * RKS, RKS * 2, &pbar[0]);
}
// Issued only once the slices are in (~1 us): the 96 units' weight boxes are 4.7 MB from HBM, and fired at kernel start together with the
// slices they starved the token CTAs' own stage-1 loads (rstd known 1.6 -> 2.2 us median / 2.8 max, which the routers then waited for).
__device__ __forceinline__ void issue_unit_weight(const Params& P, uint8_t* smem, int kg, int rg, uint64_t* pbar) {
    mbar_arrive_expect_tx(&pbar[1], (unsigned)W_TILE_BYTES);
    tma_load_3d(smem + OFF_PW, &P.tm_w, 0, rg * RROWS, kg * (RKS / 64), &pbar[1]);   // evict_first: the router weight is read exactly once per launch
}
// hidden + residual k-slices of all tokens as swizzled bf16 tiles ([6 k64 columns][NT/8 atoms of 8 tokens x 128 B],
// the wgmma B-tile layout; tokens >= M zero-filled) + the norm weight k-slice. Issued at kernel start: the router
// path does not depend on stage 1 at all.
__device__ __forceinline__ void issue_slice_loads(const Params& P, uint8_t* smem, int kg, int NT, int M) {
    constexpr int CPT = (RKS / 64) * 8;   // 48 x 16-B chunks per token
    constexpr int T0 = FMOE_ROUTER_POLL_FIRST ? 32 : 0;   // warp 0 polls the rstd messages instead (FMOE_ROUTER_POLL_FIRST)
    if ((int)threadIdx.x >= T0) for (int i = threadIdx.x - T0; i < 2 * NT * CPT; i += NTHREADS - T0) {
        const int which = i / (NT * CPT), r0 = i - which * (NT * CPT);
        const int tok = r0 / CPT, r = r0 - tok * CPT, c = r >> 3, q = r & 7;
        const bool valid = tok < M;
        const __nv_bfloat16* base = which ? P.residual : P.hidden;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(base + (size_t)(valid ? tok : 0) * DIM + kg * RKS + c * 64) + q * 16;
        uint8_t* dst = smem + (which ? OFF_PR : OFF_PH) + c * (NT * 128) + (tok >> 3) * 1024 + (tok & 7) * 128 + ((q ^ (tok & 7)) << 4);
        cp_async_16(dst, src, valid ? 16u : 0u);
    }
    if ((int)threadIdx.x >= T0 && (int)threadIdx.x < T0 + CPT)
        cp_async_16(smem + OFF_PNW + (threadIdx.x - T0) * 16, reinterpret_cast<const uint8_t*>(P.norm_w + kg * RKS) + (threadIdx.x - T0) * 16, 16u);
    cp_async_commit();
}
// FMOE_INPUT_TP router unit: only the residual k-slice and the norm-weight slice come from memory (the reduced hidden slice
// arrives as LL messages from this rank's own token CTAs and is unpacked straight into the B tile by router_gather_tp).
__device__ __forceinline__ void issue_slice_loads_tp(const Params& P, uint8_t* smem, int kg, int NT, int M) {
    constexpr int CPT = (RKS / 64) * 8;   // 48 x 16-B chunks per token
    for (int i = threadIdx.x; i < NT * CPT; i += NTHREADS) {
        const int tok = i / CPT, r = i - tok * CPT, c = r >> 3, q = r & 7;
        const bool valid = tok < M;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(P.residual + (size_t)(valid ? tok : 0) * DIM + kg * RKS + c * 64) + q * 16;
        uint8_t* dst = smem + OFF_PR + c * (NT * 128) + (tok >> 3) * 1024 + (tok & 7) * 128 + ((q ^ (tok & 7)) << 4);
        cp_async_16(dst, src, valid ? 16u : 0u);
    }
    if ((int)threadIdx.x < CPT)
        cp_async_16(smem + OFF_PNW + threadIdx.x * 16, reinterpret_cast<const uint8_t*>(P.norm_w + kg * RKS) + threadIdx.x * 16, 16u);
    cp_async_commit();
}
// Per-token rstd from the k-slices: this unit's 24 x M sixteen-element partials (same element order and fmaf chain
// as stage 1's threads) -> sequential 24-sum per token -> published per k-group as LL messages (identical from every
// unit with this kg) -> gather the 16 k-groups -> sequential 16-sum -> rsqrtf: bit-identical to stage 1's rstd.
__device__ __forceinline__ void router_rstd(const Params& P, unsigned epoch, uint8_t* smem, int kg, int NT, int M, bool gather) {
    const int tid = threadIdx.x;
    float* vs = reinterpret_cast<float*>(smem + OFF_TAB + TABX_VS);       // [tok][24]
    float* s_rstd = reinterpret_cast<float*>(smem + OFF_TAB + TABX_RSTD);
    float* s_pub = reinterpret_cast<float*>(smem + OFF_TAB + TABX_PUB);
    float* sq = reinterpret_cast<float*>(smem + OFF_PSQ);                  // [16][SQ_STRIDE]
    const uint8_t* H = smem + OFF_PH;
    const uint8_t* R = smem + OFF_PR;
    if constexpr (FMOE_RSTD_DIRECT) {   // one LL message per token from its stage-1 CTA (the same bits this path used to recompute)
        if (!gather) return;
        // both 8-B halves must carry this epoch (see stage1_token: a 16-B load may observe the halves of the store separately)
        if (tid < M) { uint4 v; do { v = ld_ll(P.ssq + tid); } while (v.y != epoch || v.w != epoch); s_rstd[tid] = __uint_as_float(v.x); }
        __syncthreads();
        return;
    }
    for (int i = tid; i < M * SLICE_VT; i += NTHREADS) {
        const int tok = i / SLICE_VT, j = i - tok * SLICE_VT;
        const int c = j >> 2, q = 2 * (j & 3);   // k64 column and first 16-B chunk of this 32-B (16-element) group
        const int base = c * (NT * 128) + (tok >> 3) * 1024 + (tok & 7) * 128;
        const int o0 = base + ((q ^ (tok & 7)) << 4), o1 = base + (((q + 1) ^ (tok & 7)) << 4);
        const uint4 h0 = *reinterpret_cast<const uint4*>(H + o0), h1 = *reinterpret_cast<const uint4*>(H + o1);
        const uint4 r0 = *reinterpret_cast<const uint4*>(R + o0), r1 = *reinterpret_cast<const uint4*>(R + o1);
        const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint32_t rw[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float v0 = bf16lo(hw[k]) + bf16lo(rw[k]), v1 = bf16hi(hw[k]) + bf16hi(rw[k]);
            ss = fmaf(v0, v0, ss);
            ss = fmaf(v1, v1, ss);
        }
        vs[tok * SLICE_VT + j] = ss;
    }
    __syncthreads();
    if (tid < M) { float p = 0.f; for (int j = 0; j < SLICE_VT; ++j) p += vs[tid * SLICE_VT + j]; s_pub[tid] = p; }
    __syncthreads();
    if (tid < RMSG) {
        const int a = 3 * tid;
        st_ll(P.ssq + kg * RMSG + tid, make_uint4(__float_as_uint(a < M ? s_pub[a] : 0.f), __float_as_uint(a + 1 < M ? s_pub[a + 1] : 0.f),
                                                  __float_as_uint(a + 2 < M ? s_pub[a + 2] : 0.f), epoch));
    }
    if (!gather) return;   // later units of this CTA: rstd already known (tokens are the same)
    if (tid < RKG * RMSG) {   // 352 messages, one per thread, all in flight
        const int kq = tid / RMSG, m = tid - kq * RMSG;
        uint4 v;
        do { v = ld_ll(P.ssq + kq * RMSG + m); } while (v.w != epoch);
        float* d = sq + kq * SQ_STRIDE + 3 * m;
        d[0] = __uint_as_float(v.x); d[1] = __uint_as_float(v.y); d[2] = __uint_as_float(v.z);
    }
    __syncthreads();
    if (tid < M) {
        float tot = 0.f;
#pragma unroll
        for (int kq = 0; kq < RKG; ++kq) tot += sq[kq * SQ_STRIDE + tid];
        s_rstd[tid] = rsqrtf(tot / (float)DIM + P.eps);
    }
    __syncthreads();
}
// In-place RMSNorm of the hidden slice into the wgmma B tile: bf16((f32(h) + f32(r)) * rstd * w), the same
// operations (and therefore the same bits) as stage 1's normed values that feed the fp8 quant.
__device__ __forceinline__ void router_normalize(const Params& P, uint8_t* smem, int NT, int M) {
    constexpr int CPT = (RKS / 64) * 8;
    const float* s_rstd = reinterpret_cast<const float*>(smem + OFF_TAB + TABX_RSTD);
    uint8_t* H = smem + OFF_PH;
    const uint8_t* R = smem + OFF_PR;
    const uint8_t* NW = smem + OFF_PNW;
    for (int i = threadIdx.x; i < M * CPT; i += NTHREADS) {
        const int tok = i / CPT, r = i - tok * CPT, c = r >> 3, q = r & 7;
        const int off = c * (NT * 128) + (tok >> 3) * 1024 + (tok & 7) * 128 + ((q ^ (tok & 7)) << 4);
        const uint4 h = *reinterpret_cast<const uint4*>(H + off), rr = *reinterpret_cast<const uint4*>(R + off), w = *reinterpret_cast<const uint4*>(NW + r * 16);
        const float rstd = s_rstd[tok];
        const uint32_t hw[4] = {h.x, h.y, h.z, h.w}, rw[4] = {rr.x, rr.y, rr.z, rr.w}, ww[4] = {w.x, w.y, w.z, w.w};
        uint32_t nw[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float v0 = bf16lo(hw[k]) + bf16lo(rw[k]), v1 = bf16hi(hw[k]) + bf16hi(rw[k]);
            nw[k] = pack_bf16x2_rn(__fmul_rn(__fmul_rn(v0, rstd), bf16lo(ww[k])), __fmul_rn(__fmul_rn(v1, rstd), bf16hi(ww[k])));
        }
        *reinterpret_cast<uint4*>(H + off) = make_uint4(nw[0], nw[1], nw[2], nw[3]);
    }
}
// FMOE_INPUT_TP: the router unit's B tile from this rank's OWN token CTAs' all-gather messages (slice my_rank of every token sits in
// the local XAG buffer: no cross-rank hop for the router data) and the per-token rstd from the 8 ranks' p messages (all 16 k-group
// partials, summed in k-group order = stage 1's rstd bit for bit). The slice messages are unpacked straight into the swizzled
// [k64 column][token] B-tile layout (three 4-B smem stores per message; router_normalize then works in place as before).
__device__ __forceinline__ void router_gather_tp(const Params& P, unsigned epoch, uint8_t* smem, int NT, int M, int half) {
    const int tid = threadIdx.x, r = P.my_rank;
    const uint4* xag_me = tp_pro<true>(P, r) + PRO_XAG;
    uint8_t* H = smem + OFF_PH;
    float* sp = reinterpret_cast<float*>(smem + OFF_TAB + TABX_VS);         // [tok][16] k-group sum-of-squares partials
    float* s_rstd = reinterpret_cast<float*>(smem + OFF_TAB + TABX_RSTD);
    static_assert(TP_M * RKG * 4 <= TABX_RSTD - TABX_VS, "p table fits the vs scratch");
    // this unit's k-group = half `half` of the 768-slice: messages 64 half .. 64 half + 63 of every token (2048, one poll pass)
    constexpr int MPK = RKS / 6;                      // 64 messages per (token, k-group)
    constexpr int K = 4, NMSG = TP_M * MPK;
    static_assert(NMSG <= K * NTHREADS, "one pass");
    {
        uint4 m[K];
        unsigned pending = 0;
#pragma unroll
        for (int k = 0; k < K; ++k) if (tid + k * NTHREADS < NMSG) pending |= 1u << k;
        do {
#pragma unroll
            for (int k = 0; k < K; ++k)
                if (pending & (1u << k)) {
                    const int g = tid + k * NTHREADS, tok = g / MPK, i = g - tok * MPK;
                    m[k] = ld_ll(xag_me + ((size_t)(r * TP_M + tok) * TP_AG_ROW + half * MPK + i));
                }
#pragma unroll
            for (int k = 0; k < K; ++k)
                if ((pending & (1u << k)) && m[k].w == epoch) {
                    const int g = tid + k * NTHREADS, tok = g / MPK, i = g - tok * MPK;
                    const uint32_t w[3] = {m[k].x, m[k].y, m[k].z};
#pragma unroll
                    for (int j = 0; j < 3; ++j) {   // word 3 i + j = k-group elements 6 i + 2 j, +1 -> k64 column c, byte kb of the token's 128-B row
                        const int kk = 6 * i + 2 * j, c = kk >> 6, kb = (kk & 63) * 2, q = kb >> 4;
                        *reinterpret_cast<uint32_t*>(H + c * (NT * 128) + (tok >> 3) * 1024 + (tok & 7) * 128 + ((q ^ (tok & 7)) << 4) + (kb & 15)) = w[j];
                    }
                    pending &= ~(1u << k);
                }
        } while (pending);
    }
    if (tid < TP_NDEV * TP_M) {   // p messages: one per (src rank, token); both 8-B halves must carry the tag
        const int src = tid / TP_M, tok = tid - src * TP_M;
        uint4 v;
        do { v = ld_ll(xag_me + ((size_t)(src * TP_M + tok) * TP_AG_ROW + TP_PMSG)); } while (v.y != epoch || v.w != epoch);
        sp[tok * RKG + 2 * src] = __uint_as_float(v.x); sp[tok * RKG + 2 * src + 1] = __uint_as_float(v.z);
    }
    __syncthreads();
    if (tid < M) {
        float tot = 0.f;
#pragma unroll
        for (int kg = 0; kg < RKG; ++kg) tot += sp[tid * RKG + kg];
        s_rstd[tid] = rsqrtf(tot / (float)DIM + P.eps);
    }
    __syncthreads();
}
// FMOE_TP_ROUTER_TOKENS (design B) router unit (kg, rg) for this rank's 4 owned tokens t = r + 8 lt (rows lt of one 8-token atom):
// residual k-slices + norm-w slice from memory at kernel start; the reduced hidden slice of k-group kg comes from rank kg/2's all-gather
// messages (half kg&1 of its slice) in the local XAG buffer, the rstd from the 8 p messages of each token.
__device__ __forceinline__ void issue_slice_loads_tp4(const Params& P, uint8_t* smem, int kg) {
    constexpr int NT8 = 8, CPT = (RKS / 64) * 8;   // 48 x 16-B chunks per token
    for (int i = threadIdx.x; i < TP_TOK_PER_RANK * CPT; i += NTHREADS) {
        const int lt = i / CPT, rr = i - lt * CPT, c = rr >> 3, q = rr & 7;
        const int t = P.my_rank + TP_NDEV * lt;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(P.residual + (size_t)t * DIM + kg * RKS + c * 64) + q * 16;
        cp_async_16(smem + OFF_PR + c * (NT8 * 128) + lt * 128 + ((q ^ lt) << 4), src, 16u);
    }
    if ((int)threadIdx.x < CPT)
        cp_async_16(smem + OFF_PNW + threadIdx.x * 16, reinterpret_cast<const uint8_t*>(P.norm_w + kg * RKS) + threadIdx.x * 16, 16u);
    cp_async_commit();
}
__device__ __forceinline__ void router_gather_tp4(const Params& P, unsigned epoch, uint8_t* smem, int kg) {
    const int tid = threadIdx.x, r = P.my_rank;
    const uint4* xag_me = tp_pro<true>(P, r) + PRO_XAG;
    uint8_t* H = smem + OFF_PH;
    float* sp = reinterpret_cast<float*>(smem + OFF_TAB + TABX_VS);         // [4 tokens][16] k-group sum-of-squares partials
    float* s_rstd = reinterpret_cast<float*>(smem + OFF_TAB + TABX_RSTD);
    constexpr int NT8 = 8, MPK = RKS / 6;   // 64 slice messages per (token, k-group)
    constexpr int NH = TP_TOK_PER_RANK * MPK;   // 256
    const int src = kg >> 1, half = kg & 1;
    if (tid < NH) {   // one slice message per thread -> three 4-B words into the swizzled B tile (token row lt)
        const int lt = tid / MPK, i = tid - lt * MPK, t = r + TP_NDEV * lt;
        uint4 v;
        do { v = ld_ll(xag_me + tp_ag_idx(src, t, half * MPK + i)); } while (v.w != epoch);
        const uint32_t w[3] = {v.x, v.y, v.z};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int kk = 6 * i + 2 * j, c = kk >> 6, kb = (kk & 63) * 2, q = kb >> 4;
            *reinterpret_cast<uint32_t*>(H + c * (NT8 * 128) + lt * 128 + ((q ^ lt) << 4) + (kb & 15)) = w[j];
        }
    } else if (tid < NH + TP_NDEV * TP_TOK_PER_RANK) {   // p messages of the 4 tokens from the 8 slices (both 8-B halves tagged)
        const int j = tid - NH, s = j / TP_TOK_PER_RANK, lt = j - s * TP_TOK_PER_RANK, t = r + TP_NDEV * lt;
        uint4 v;
        do { v = ld_ll(xag_me + tp_ag_idx(s, t, TP_PMSG)); } while (v.y != epoch || v.w != epoch);
        sp[lt * RKG + 2 * s] = __uint_as_float(v.x); sp[lt * RKG + 2 * s + 1] = __uint_as_float(v.z);
    }
    __syncthreads();
    if (tid < TP_TOK_PER_RANK) {   // rstd = rsqrt(sum_{kg = 0..15} p[kg] / DIM + eps): loads first, then stage 1's sequential sum
        float pv[RKG];
#pragma unroll
        for (int k = 0; k < RKG; ++k) pv[k] = sp[tid * RKG + k];
        float tot = 0.f;
#pragma unroll
        for (int k = 0; k < RKG; ++k) tot += pv[k];
        s_rstd[tid] = rsqrtf(tot / (float)DIM + P.eps);
    }
    __syncthreads();
}
// FMOE_TP_ROUTER_RS: design-B router unit (kg, rg) reduces k-group kg of the 4 owned tokens itself from the reduce-scatter
// messages (this rank's IN_RS rows: 7 peers' slices + its own partial), in RANK order with an fp32 accumulator from rank 0 and one bf16
// RN -- bitwise the token CTA's (= stock 2shot_pull) values -- straight into the swizzled B tile; then the k-group's stage-1 sum-of-
// squares partial (24 sixteen-element fmaf chains over bf16(reduced) + residual, summed sequentially = tp_reduce_token's p[kg]) is
// published to the LOCAL table PRO_RP[token][kg] (all six row-group units of a k-group write the same bits) and the 16 partials of each
// token are gathered back for rstd. Thread map as tp_reduce_token: threads 0..255 poll ranks 0..3, 256..511 ranks 4..7 (4 messages in
// flight each; one CTA barrier joins the two halves of the sequential chain). The producer warpgroup (W tile) is not involved.
__device__ __forceinline__ void router_gather_rs_tp4(const Params& P, unsigned epoch, uint8_t* smem, int kg) {
    const int tid = threadIdx.x, r = P.my_rank;
    constexpr int NT8 = 8, MPK = RKS / 6;                                                   // 64 messages per (token, k-group)
    const int k = kg >> 1, half = kg & 1;
    uint8_t* H = smem + OFF_PH;
    const uint8_t* R = smem + OFF_PR;
    float* s_acc = reinterpret_cast<float*>(smem + OFF_P1);                                 // [256 pairs][6] partial sums of ranks 0..3
    float* sp = reinterpret_cast<float*>(smem + OFF_TAB + TABX_VS);                         // [4 tokens][16] k-group partials
    float* s_ss = sp + TP_TOK_PER_RANK * RKG;                                                // [4 tokens][24] sixteen-element partials
    float* s_rstd = reinterpret_cast<float*>(smem + OFF_TAB + TABX_RSTD);
    static_assert(OFF_P1 + TP_TOK_PER_RANK * MPK * 6 * 4 <= OFF_TOK, "router partial-sum staging fits before the token lists");
    static_assert(TABX_VS + (TP_TOK_PER_RANK * RKG + TP_TOK_PER_RANK * SLICE_VT) * 4 <= TABX_RSTD, "p table + ssq partials fit the vs scratch");
    const uint4* rs_me = tp_pro<true>(P, r) + PRO_IN_RS;
    uint4* rp = tp_pro<true>(P, r) + PRO_RP;
    const int pidx = tid & 255, lt = pidx / MPK, i = pidx - lt * MPK, m = half * MPK + i, g = (tid >> 8) & 1;   // (token, message) pair, source group
    const int t = r + TP_NDEV * lt;
    uint4 pm[4];
    if (tid < 2 * 256) {
        const uint32_t* own = reinterpret_cast<const uint32_t*>(P.partial + (size_t)t * DIM + k * TP_KS) + 3 * m;
        const uint32_t o0 = own[0], o1 = own[1], o2 = own[2];
        unsigned pending = 0xfu;
        if ((r >> 2) == g) pending &= ~(1u << (r & 3));                                    // my own contribution comes from the partial
#pragma unroll
        for (int j = 0; j < 4; ++j) pm[j] = make_uint4(o0, o1, o2, epoch);
        do {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (pending & (1u << j)) pm[j] = ld_ll(rs_me + tp_rs_idx(4 * g + j, t, k, m));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((pending & (1u << j)) && pm[j].w == epoch) pending &= ~(1u << j);
        } while (pending);
    }
    if (tid < 256) {   // ranks 0..3, in order
        float acc[6];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[3] = {pm[j].x, pm[j].y, pm[j].z};
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                const float x = bf16lo(w[q]), y = bf16hi(w[q]);
                if (j == 0) { acc[2 * q] = x; acc[2 * q + 1] = y; }
                else { acc[2 * q] = __fadd_rn(acc[2 * q], x); acc[2 * q + 1] = __fadd_rn(acc[2 * q + 1], y); }
            }
        }
#pragma unroll
        for (int q = 0; q < 6; ++q) s_acc[pidx * 6 + q] = acc[q];
    }
    __syncthreads();
    if (tid == 0) STAMP(P, 16);   // TP diag (router CTA): reduce-scatter messages of its k-group landed
    if (tid >= 256 && tid < 512) {   // += ranks 4..7 in order, one bf16 RN, three words into the swizzled B tile (row lt, elements 6 i .. 6 i + 5)
        float acc[6];
#pragma unroll
        for (int q = 0; q < 6; ++q) acc[q] = s_acc[pidx * 6 + q];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[3] = {pm[j].x, pm[j].y, pm[j].z};
#pragma unroll
            for (int q = 0; q < 3; ++q) { acc[2 * q] = __fadd_rn(acc[2 * q], bf16lo(w[q])); acc[2 * q + 1] = __fadd_rn(acc[2 * q + 1], bf16hi(w[q])); }
        }
        const uint32_t w[3] = {pack_bf16x2_rn(acc[0], acc[1]), pack_bf16x2_rn(acc[2], acc[3]), pack_bf16x2_rn(acc[4], acc[5])};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int kk = 6 * i + 2 * j, c = kk >> 6, kb = (kk & 63) * 2, q = kb >> 4;
            *reinterpret_cast<uint32_t*>(H + c * (NT8 * 128) + lt * 128 + ((q ^ lt) << 4) + (kb & 15)) = w[j];
        }
    }
    // The residual / norm-w slices were cp.async'd by threads 0..191 at kernel start; only the consumer threads drain them (the producer
    // warpgroup's weight-tile copies are still in flight and are waited for later by the kernel body).
    if (tid < NCONS) { cp_async_wait_all(); named_bar_sync(1, NCONS); }
    if (tid < TP_TOK_PER_RANK * SLICE_VT) {   // 96 threads: (token, sixteen-element group gq) = elements 16 gq .. +15 of the k-group, stage 1's fmaf order
        const int lt2 = tid / SLICE_VT, gq = tid - lt2 * SLICE_VT, kk = 16 * gq, c = kk >> 6, kb = (kk & 63) * 2, q = kb >> 4;
        const int off = c * (NT8 * 128) + lt2 * 128;
        const uint4 h0 = *reinterpret_cast<const uint4*>(H + off + ((q ^ lt2) << 4)), h1 = *reinterpret_cast<const uint4*>(H + off + (((q + 1) ^ lt2) << 4));
        const uint4 r0 = *reinterpret_cast<const uint4*>(R + off + ((q ^ lt2) << 4)), r1 = *reinterpret_cast<const uint4*>(R + off + (((q + 1) ^ lt2) << 4));
        const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w}, rw[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float v0 = bf16lo(hw[e]) + bf16lo(rw[e]), v1 = bf16hi(hw[e]) + bf16hi(rw[e]);
            ss = fmaf(v0, v0, ss);
            ss = fmaf(v1, v1, ss);
        }
        s_ss[lt2 * SLICE_VT + gq] = ss;
    }
    if (tid < NCONS) named_bar_sync(1, NCONS);
    if (tid < TP_TOK_PER_RANK) {   // p[kg] of token lt: the 24 partials summed sequentially (loads first) -> local table, double-tagged
        float pv[SLICE_VT];
#pragma unroll
        for (int j = 0; j < SLICE_VT; ++j) pv[j] = s_ss[tid * SLICE_VT + j];
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < SLICE_VT; ++j) p += pv[j];
        st_ll(rp + tid * RKG + kg, make_uint4(__float_as_uint(p), epoch, __float_as_uint(p), epoch));
        if (tid == 0) STAMP(P, 17);   // TP diag (router CTA): its p partial published
    }
    if (tid < TP_TOK_PER_RANK * RKG) {   // 64 threads gather the 16 k-group partials of the 4 tokens
        const int lt3 = tid / RKG, kq = tid - lt3 * RKG;
        uint4 v;
        do { v = ld_ll(rp + lt3 * RKG + kq); } while (v.y != epoch || v.w != epoch);
        sp[lt3 * RKG + kq] = __uint_as_float(v.x);
    }
    __syncthreads();
    if (tid < TP_TOK_PER_RANK) {   // rstd = rsqrt(sum_{kg = 0..15} p[kg] / DIM + eps): loads first, then stage 1's sequential sum
        float pv[RKG];
#pragma unroll
        for (int kq = 0; kq < RKG; ++kq) pv[kq] = sp[tid * RKG + kq];
        float tot = 0.f;
#pragma unroll
        for (int kq = 0; kq < RKG; ++kq) tot += pv[kq];
        s_rstd[tid] = rsqrtf(tot / (float)DIM + P.eps);
    }
    __syncthreads();
}
// Tensor-core router GEMM for one unit: logits[64 rows][NT tokens] over this unit's 384 k, fp32 accumulate.
// WG w handles k16 steps [6w, 6w+6); per step the fp32 weight fragment is split into 3 exact bf16 planes
// (hi, mid, lo) and 3 wgmmas accumulate into the same fp32 D. The 4 WG partials are summed in a fixed
// order and published as LL messages (unit, token, 22 x 3 rows).
template <int NT, int EXACT_M = 0, bool TP = false>
__device__ __forceinline__ void router_compute(const Params& P, unsigned epoch, uint8_t* smem, int unit, int M) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int NR = NT / 2;
    constexpr int PSTR = NT + 8;   // padded row stride: the fragment stores become 2-way instead of 8-way bank conflicted
    float* psum = reinterpret_cast<float*>(smem + OFF_PS);   // [RWG wg][64 rows][PSTR]
    constexpr int RSTEPS = RKS / 16 / RWG;   // k16 steps per warpgroup (6)
    if (warp < RWG * 4) {
        const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, tig = lane & 3;
        const uint32_t pb = smem_u32(smem + OFF_PH);   // normalized tokens (B tile), in place over the hidden slice
        const uint8_t* wrow = smem + OFF_PW + (wq * 16 + g) * W_STRIDE + tig * 2 * W_ELEM;   // (row g, k 2tig); +8 rows / +8 k below
        float D[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) D[i] = 0.f;
        constexpr int NPLANE = FMOE_ROUTER_BF16 ? 1 : 3;   // bf16 weight: the fragment word IS the A operand; fp32: 3 exact bf16 planes
        uint32_t A0[NPLANE][4], A1[NPLANE][4];
        auto load_split = [&](int s, uint32_t (&A)[NPLANE][4]) {
            const int kb = s * 16 * W_ELEM;   // byte offset of k16 step s
#pragma unroll
            for (int i = 0; i < 4; ++i) {   // a0: (g, k), a1: (g+8, k), a2: (g, k+8), a3: (g+8, k+8)
                const uint8_t* wp;
                if constexpr (FMOE_ROUTER_TMA) {
                    // TMA tile [k64 seg][row 64][128 B] with SWIZZLE_128B (16-B chunk index ^= row & 7 = g for rows g and g + 8): the 8 lanes g
                    // of one (s, i) hit 8 distinct chunks x 4 tig words = 32 distinct banks
                    const int row = wq * 16 + g + ((i & 1) ? 8 : 0), chunk = 2 * (s & 3) + (i >> 1);
                    wp = smem + OFF_PW + (s >> 2) * (RROWS * 128) + row * 128 + ((chunk ^ g) << 4) + tig * 4;
                    (void)wrow; (void)kb;
                } else wp = wrow + ((i & 1) ? 8 * W_STRIDE : 0) + kb + ((i >> 1) ? 8 * W_ELEM : 0);
                if constexpr (FMOE_ROUTER_BF16) {
                    A[0][i] = *reinterpret_cast<const uint32_t*>(wp);   // two bf16 (k 2tig, 2tig+1) of row g / g+8: no conversion
                } else {
                    const float2 w2 = *reinterpret_cast<const float2*>(wp);
                    if constexpr (EXACT_M == 32) {
                        // Same three RNE planes; reuse packed high/mid directly.
                        const uint32_t h = pack_bf16x2_rn(w2.x, w2.y);
                        const float r0 = w2.x - bf16lo(h), r1 = w2.y - bf16hi(h);
                        const uint32_t m = pack_bf16x2_rn(r0, r1);
                        A[0][i] = h; A[1 % NPLANE][i] = m;
                        A[2 % NPLANE][i] = pack_bf16x2_rn(r0 - bf16lo(m), r1 - bf16hi(m));
                    } else {
                        float h0, m0, l0, h1, m1, l1;
                        split3(w2.x, h0, m0, l0);
                        split3(w2.y, h1, m1, l1);
                        A[0][i] = pack_bf16x2_rn(h0, h1); A[1 % NPLANE][i] = pack_bf16x2_rn(m0, m1); A[2 % NPLANE][i] = pack_bf16x2_rn(l0, l1);
                    }
                }
            }
        };
        auto issue3 = [&](const uint32_t (&A)[NPLANE][4], int s, int first) {
            const uint64_t desc = wg::make_desc_sw128(pb + (s >> 2) * (NT * 128) + (s & 3) * 32);
            wg::MmaRSBF16<NT>::fma(D, A[0], desc, first ? 0 : 1);
#pragma unroll
            for (int p = 1; p < NPLANE; ++p) wg::MmaRSBF16<NT>::fma(D, A[p], desc, 1);
        };
        const int s0 = wg * RSTEPS;
        // double-buffered A: the next step's loads/splits overlap the in-flight group (statically unrolled pairs)
        static_assert(RSTEPS % 2 == 0, "step pairs");
        load_split(s0, A0); wg::fence(); issue3(A0, s0, 1); wg::commit();
#pragma unroll
        for (int j = 1; j < RSTEPS; j += 2) {
            load_split(s0 + j, A1); wg::fence(); issue3(A1, s0 + j, 0); wg::commit(); wg::wait<1>();
            if (j + 1 < RSTEPS) { load_split(s0 + j + 1, A0); wg::fence(); issue3(A0, s0 + j + 1, 0); wg::commit(); wg::wait<1>(); }
        }
        wg::wait<0>();
        named_bar_sync(1, RWG * 128);   // router WGs done: W and B tiles are dead, psum may overlay W
        if (tid == 0) STAMP(P, 15);   // router MMAs done
        const int row0 = wq * 16 + g;
#pragma unroll
        for (int j = 0; j < NT / 8; ++j) {
            *reinterpret_cast<float2*>(psum + (wg * RROWS + row0) * PSTR + 8 * j + 2 * tig) = make_float2(D[4 * j], D[4 * j + 1]);
            *reinterpret_cast<float2*>(psum + (wg * RROWS + row0 + 8) * PSTR + 8 * j + 2 * tig) = make_float2(D[4 * j + 2], D[4 * j + 3]);
        }
    }
    __syncthreads();
    // token fastest across threads: consecutive lanes read consecutive psum words (row-fastest indexing was a
    // 32-way bank conflict on every read -- it cost 3-10 us here, growing with M)
    // TP: the messages are staged in smem and then stored destination-contiguously -- owner rank d receives its 4 tokens as rows
    // 0..3 (owner-local row t / 8) of the unit, 88 consecutive messages = 1408 B, so a warp's 32 stores form one 512-B run to ONE
    // peer instead of 32 lone 16-B writes to 8 peers (the scattered version: publish 1.3 us, hop 2.3 us; stamps validation-tp-stamps).
    uint4* s_msg = reinterpret_cast<uint4*>(smem + OFF_PS + RWG * RROWS * PSTR * 4);   // [8 owners][4 rows][22] (11 KB, after psum)
    static_assert(OFF_PS + RWG * RROWS * (MAXM + 8) * 4 + TP_NDEV * TP_TOK_PER_RANK * RMSG * 16 <= OFF_PR, "staged rpart messages fit after psum");
    for (int i = tid; i < M * RMSG; i += NTHREADS) {
        const int m = i / M, tok = i - m * M;
        float v[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int row = 3 * m + j;
            v[j] = 0.f;
            if (row < RROWS) {
                const float* p = psum + row * PSTR + tok;
                float acc = p[0];
#pragma unroll
                for (int w = 1; w < RWG; ++w) acc += p[w * RROWS * PSTR];   // fixed order: identical on every rank
                v[j] = acc;
            }
        }
        if constexpr (TP)
            s_msg[((tok % TP_NDEV) * TP_TOK_PER_RANK + tok / TP_NDEV) * RMSG + m] = make_uint4(__float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), epoch);
        else
        st_ll(P.rpart + ((size_t)unit * MAXM + tok) * RMSG + m, make_uint4(__float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), epoch));
    }
    if constexpr (TP) {
        __syncthreads();
        constexpr int PER_DST = TP_TOK_PER_RANK * RMSG;   // 88 messages per owner rank
        for (int i = tid; i < TP_NDEV * PER_DST; i += NTHREADS) {
            const int d = i / PER_DST, k = i - d * PER_DST;   // k = owner-local row * 22 + m: contiguous in the owner's RPART region
            st_ll(tp_pro<true>(P, d) + PRO_RPART + (size_t)unit * MAXM * RMSG + k, s_msg[i]);
        }
    }
}

// ======================= prologue stage 3: k-group reduction + sigmoid/bias + top-8 of token t =======================
// The correction bias (n_exp fp32, an HBM miss on every launch) used to be loaded per thread right after the partial gather,
// i.e. one dependent global round trip (~0.8 us) on the routing critical path. The token CTAs now
// copy it to smem while they wait for the router partials (after the router work on a dual-role CTA: the router's psum
// overlays this region). The stage-3 scratch is dead until the gather starts, and the gather never touches this slot.
__device__ __forceinline__ void stage3_prefetch_bias(const Params& P, uint8_t* smem) {
    float* s_bias = reinterpret_cast<float*>(smem + OFF_PS) + RUNITS_MAX * RROWS + 2 * NEXP + 2 * (NTHREADS / 32) * TOPK;
    for (int i = threadIdx.x; i < NEXP; i += NTHREADS) s_bias[i] = i < P.n_exp ? __ldg(P.bias + i) : 0.f;
}
template <bool COMPACT_TOPK = false, bool TP = false, bool DRY = false>
__device__ __forceinline__ void stage3_token(const Params& P, unsigned epoch, uint8_t* smem, int t, int n_rg) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float* s_part = reinterpret_cast<float*>(smem + OFF_PS);      // [unit = rg*16 + kg][64]
    float* s_score = s_part + RUNITS_MAX * RROWS;                 // [384] sigmoid
    int* s_key = reinterpret_cast<int*>(s_score + NEXP);          // [384] ordered biased score
    int* s_cand = s_key + NEXP;                                   // [NW warps][8] expert
    int* s_ckey = s_cand + (NTHREADS / 32) * TOPK;                // [NW warps][8] key
    const float* s_bias = reinterpret_cast<const float*>(s_ckey + (NTHREADS / 32) * TOPK);   // [384] (stage3_prefetch_bias)
    const int n_units = RKG * n_rg;
    const int nm = n_units * RMSG;
    const int trow = TP ? t / TP_NDEV : t;   // TP: the owner rank's RPART rows hold its 4 tokens as rows 0..3
    if constexpr (!DRY) {   // gather this token's partial messages (all of a thread's messages in flight; retry the missing tags)
        constexpr int K = (RUNITS_MAX * RMSG + NTHREADS - 1) / NTHREADS;   // 4
        uint4 m[K];
        unsigned pending = 0;
#pragma unroll
        for (int k = 0; k < K; ++k) if (tid + k * NTHREADS < nm) pending |= 1u << k;
        do {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (pending & (1u << k)) {
                    const int i = tid + k * NTHREADS, unit = i / RMSG, mm = i - unit * RMSG;
                    m[k] = ld_ll(P.rpart + ((size_t)unit * MAXM + trow) * RMSG + mm);
                }
            }
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if ((pending & (1u << k)) && m[k].w == epoch) {
                    const int i = tid + k * NTHREADS, unit = i / RMSG, mm = i - unit * RMSG;
                    float* d = s_part + unit * RROWS + 3 * mm;
                    d[0] = __uint_as_float(m[k].x);
                    if (3 * mm + 1 < RROWS) { d[1] = __uint_as_float(m[k].y); d[2] = __uint_as_float(m[k].z); }
                    pending &= ~(1u << k);
                }
            }
        } while (pending);
    }
    __syncthreads();
    if (tid == 0) STAMP(P, 11);   // all router partials of this token gathered
    int register_key = INT_MIN;
    if (tid < NEXP) {
        float sc = 0.f;
        int key = INT_MIN;
        if (tid < P.n_exp) {
            const int rg = tid >> 6, row = tid & 63;
            float l = 0.f;
            if constexpr (TP) {   // same fixed-order sum, but the 16 smem loads are issued together instead of a load-add chain
                float pv[RKG];
#pragma unroll
                for (int kg = 0; kg < RKG; ++kg) pv[kg] = s_part[(rg * RKG + kg) * RROWS + row];
#pragma unroll
                for (int kg = 0; kg < RKG; ++kg) l += pv[kg];
            } else
#pragma unroll
            for (int kg = 0; kg < RKG; ++kg) l += s_part[(rg * RKG + kg) * RROWS + row];   // fixed order: identical on every rank
            sc = 1.0f / (1.0f + expf(-l));
            key = f2ordered(sc + (FMOE_PROLOGUE_TIGHT ? s_bias[tid] : P.bias[tid]));
        }
        s_score[tid] = sc;
        if constexpr (COMPACT_TOPK) register_key = key;
        else s_key[tid] = key;
    }
    if constexpr (TP) { if (tid == 0) STAMP(P, 20); }   // TP diag: sums + sigmoid + keys done
    // Exact M32 keeps each score thread's key in its own register. Scores
    // are published by the candidate barrier below before warp0 reads them.
    // Other entries retain their original shared-key remapping and barrier.
    if constexpr (!COMPACT_TOPK) __syncthreads();
    constexpr int NW = COMPACT_TOPK ? NEXP / 32 : NTHREADS / 32;
    constexpr int EPW = (NEXP + NW - 1) / NW;
    constexpr int CPL = NW * TOPK / 32;        // 3 (compact) or 5 candidates per lane
    static_assert(EPW <= 32 && (NW * TOPK) % 32 == 0 && NW * TOPK <= 256, "top-8 level-1/2 mapping");
    if constexpr (COMPACT_TOPK && FMOE_RANK_TOPK) {
        // Rank selection. Order = (key desc, expert asc), exactly the order the redux/ballot rounds below produce.
        // Level 1: lane e = warp*32+lane counts the warp's (key, lane) pairs above it -- 32 independent shuffles + compares
        // instead of 8 dependent redux+ballot rounds; the 8 lowest ranks are this warp's candidates, already in order.
        int* s_win_e = reinterpret_cast<int*>(const_cast<float*>(s_bias) + NEXP);   // [8] winners (expert), [8] winners (score)
        float* s_win_s = reinterpret_cast<float*>(s_win_e + TOPK);
        if (warp < NW) {
            const int key = register_key;
            int rank = 0;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const int kj = __shfl_sync(0xffffffffu, key, j);
                rank += (kj > key || (kj == key && j < lane)) ? 1 : 0;
            }
            if (rank < TOPK) { s_cand[warp * TOPK + rank] = warp * EPW + lane; s_ckey[warp * TOPK + rank] = key; }
        }
        __syncthreads();
        if constexpr (TP) { if (tid == 0) STAMP(P, 21); }   // TP diag: level-1 candidates
        // Level 2: 4 threads per candidate (NW*8 = 96 candidates x 4 = 384 threads, warps 0-11) rank it among all candidates by
        // (key desc, expert asc): each thread compares against a quarter (24, loaded first, then compared), two xor-shuffles sum
        // the quarters. The candidates are unique in (key, expert), so ranks 0..7 are a permutation of the winners.
        static_assert(NW * TOPK * 4 <= NTHREADS && (NW * TOPK) % 16 == 0, "level-2 thread mapping");
        if (tid < NW * TOPK * 4) {
            const int cand = tid >> 2, quarter = tid & 3;
            const int mk = s_ckey[cand], me = s_cand[cand];
            constexpr int Q = NW * TOPK / 4;   // 24 candidates per quarter
            int4 kc[Q / 4], ec[Q / 4];
#pragma unroll
            for (int i = 0; i < Q / 4; ++i) {
                kc[i] = *reinterpret_cast<const int4*>(s_ckey + quarter * Q + 4 * i);
                ec[i] = *reinterpret_cast<const int4*>(s_cand + quarter * Q + 4 * i);
            }
            int rank = 0;
#pragma unroll
            for (int i = 0; i < Q / 4; ++i) {
                rank += (kc[i].x > mk || (kc[i].x == mk && ec[i].x < me)) ? 1 : 0;
                rank += (kc[i].y > mk || (kc[i].y == mk && ec[i].y < me)) ? 1 : 0;
                rank += (kc[i].z > mk || (kc[i].z == mk && ec[i].z < me)) ? 1 : 0;
                rank += (kc[i].w > mk || (kc[i].w == mk && ec[i].w < me)) ? 1 : 0;
            }
            rank += __shfl_xor_sync(0xffffffffu, rank, 1);
            rank += __shfl_xor_sync(0xffffffffu, rank, 2);
            if (quarter == 0 && rank < TOPK) { s_win_e[rank] = me; s_win_s[rank] = s_score[me]; }
        }
        __syncthreads();
        if constexpr (TP) { if (tid == 0) STAMP(P, 22); }   // TP diag: level-2 winners
        if (warp == 0) {   // same sequential fp32 sum over ranks 0..7 and the same message layout as the round-based path
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) sum += s_win_s[k];
            const float den = sum + 1e-20f;
            if constexpr (TP && FMOE_TP_WIDE_PUBLISH) {
                // lane = (destination rank d = lane >> 2, message j = lane & 3): the 32 remote stores of the list leave in one issue
                // slot instead of 8 dependent-issue stores per lane (same messages, same order within a destination)
                const int j = lane & 3, d = lane >> 2;
                const int ia = s_win_e[2 * j], ib = s_win_e[2 * j + 1];
                const float wa = s_win_s[2 * j], wb = s_win_s[2 * j + 1];
                const uint4 mo = make_uint4((unsigned)ia | ((unsigned)ib << 16), __float_as_uint(wa / den), __float_as_uint(wb / den), epoch);
                uint4* const* s_peer = reinterpret_cast<uint4* const*>(smem + OFF_TAB + TABX_PEER);   // cached at kernel entry
                if (!DRY) st_ll(s_peer[d] + PRO_TOPK + t * (TOPK / 2) + j, mo);
                if (lane == 0) STAMP(P, 23);   // TP diag: top-8 list stores issued
                if constexpr (FMOE_TP_OWNER_PREFETCH) {   // this token's 8 experts are certainly in the union -> warm their first FC1 k-tiles now
                    if (!DRY) {
                        const int e = s_win_e[lane & 7];
                        const uint8_t* w = P.fc1_w + (size_t)e * FC1_W_EXPERT_BYTES + (size_t)(lane >> 3 & 1) * 4 * FC1_W_SLICE_BYTES;   // lanes 0..7: k-tile 0, 8..15: k-tile 1
                        if (lane < 16) { bulk_prefetch_l2(w, 2 * FC1_W_SLICE_BYTES); bulk_prefetch_l2(w + 2 * FC1_W_SLICE_BYTES, 2 * FC1_W_SLICE_BYTES); }
                        else if (lane < 24) bulk_prefetch_l2(P.fc1_s + (size_t)e * FC1_S_EXPERT_BYTES, 2 * 4 * FC1_S_SLICE_BYTES);   // offsets of k-tiles 0..1 (4 KB)
                    }
                }
                return;
            }
            if (lane < TOPK / 2 && !DRY) {
                const int ia = s_win_e[2 * lane], ib = s_win_e[2 * lane + 1];
                const float wa = s_win_s[2 * lane], wb = s_win_s[2 * lane + 1];
                if constexpr (TP) {   // the owner rank publishes the token's list to every rank's TOPK region (stage 4 gathers locally)
                    const uint4 mo = make_uint4((unsigned)ia | ((unsigned)ib << 16), __float_as_uint(wa / den), __float_as_uint(wb / den), epoch);
                    uint4* const* s_peer = reinterpret_cast<uint4* const*>(smem + OFF_TAB + TABX_PEER);   // cached at kernel entry
#pragma unroll
                    for (int d = 0; d < TP_NDEV; ++d) st_ll(s_peer[d] + PRO_TOPK + t * (TOPK / 2) + lane, mo);
                    if (lane == 0) STAMP(P, 23);   // TP diag: top-8 list stores issued
                } else
                st_ll(P.topk + t * (TOPK / 2) + lane, make_uint4((unsigned)ia | ((unsigned)ib << 16), __float_as_uint(wa / den), __float_as_uint(wb / den), epoch));
            }
        }
        return;
    }
    if (!COMPACT_TOPK || warp < NW) {   // warp-uniform; all32 lanes participate
        const int e = warp * EPW + lane;
        int key;
        if constexpr (COMPACT_TOPK) key = register_key;
        else key = (lane < EPW && e < NEXP) ? s_key[e] : INT_MIN;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int wmax = __reduce_max_sync(0xffffffffu, key);
            const int win = __ffs(__ballot_sync(0xffffffffu, key == wmax)) - 1;
            if (lane == 0) { s_cand[warp * TOPK + k] = warp * EPW + win; s_ckey[warp * TOPK + k] = wmax; }
            if (lane == win) key = INT_MIN;
        }
    }
    __syncthreads();
    if (warp == 0) {   // level 2: NW*8 candidates, CPL per lane, 8 rounds of redux.max + ballot (ties -> lower expert id)
        int kb[CPL + 1], eb[CPL + 1];
#pragma unroll
        for (int i = 0; i < CPL; ++i) { kb[i] = s_ckey[CPL * lane + i]; eb[i] = s_cand[CPL * lane + i]; }
        kb[CPL] = INT_MIN; eb[CPL] = INT_MAX;
        // sort the lane's CPL candidates descending by (key, -expert) (insertion sort, compile-time bounds)
#pragma unroll
        for (int i = 1; i < CPL; ++i)
#pragma unroll
            for (int j = i; j > 0; --j)
                if (kb[j - 1] < kb[j] || (kb[j - 1] == kb[j] && eb[j - 1] > eb[j])) { int tk = kb[j - 1]; kb[j - 1] = kb[j]; kb[j] = tk; int te = eb[j - 1]; eb[j - 1] = eb[j]; eb[j] = te; }
        // 8 rounds; the winner (warp-uniform) is recorded by lane k/2 as its (a, b) pair -> no runtime-indexed arrays
        int ia = 0, ib = 0;
        float wa = 0.f, wb = 0.f, sum = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int wmax = __reduce_max_sync(0xffffffffu, kb[0]);
            const int emin = __reduce_min_sync(0xffffffffu, kb[0] == wmax ? eb[0] : INT_MAX);
            // (key, expert) is unique across the candidates, so the winner lane needs no ballot
            const bool win = FMOE_PROLOGUE_TIGHT ? (kb[0] == wmax && eb[0] == emin)
                                                 : lane == __ffs(__ballot_sync(0xffffffffu, kb[0] == wmax && eb[0] == emin)) - 1;
            const float sc = s_score[emin];
            sum += sc;
            if (lane == (k >> 1)) { if (k & 1) { ib = emin; wb = sc; } else { ia = emin; wa = sc; } }
            if (win) {
#pragma unroll
                for (int i = 0; i < CPL; ++i) { kb[i] = kb[i + 1]; eb[i] = eb[i + 1]; }
                kb[CPL] = INT_MIN; eb[CPL] = INT_MAX;
            }
        }
        const float den = sum + 1e-20f;
        if constexpr (TP) {
            if (lane < TOPK / 2) {
                const uint4 mo = make_uint4((unsigned)ia | ((unsigned)ib << 16), __float_as_uint(wa / den), __float_as_uint(wb / den), epoch);
                uint4* const* s_peer = reinterpret_cast<uint4* const*>(smem + OFF_TAB + TABX_PEER);
#pragma unroll
                for (int d = 0; d < TP_NDEV; ++d) st_ll(s_peer[d] + PRO_TOPK + t * (TOPK / 2) + lane, mo);
            }
        } else
        if (lane < TOPK / 2)
            st_ll(P.topk + t * (TOPK / 2) + lane, make_uint4((unsigned)ia | ((unsigned)ib << 16), __float_as_uint(wa / den), __float_as_uint(wb / den), epoch));
    }
}

// FMOE_INPUT_TP: stage 3 of an owner token behind an ABI call (own register allocation). Tried because the identical stage-3 code ran
// 2.75 us in the TP kernel against 1.63 us in the full-K kernel (validation-tp-stamps7: level 1 alone 1.17 us). MEASURED WORSE
// -- the call's caller-saved state costs
// more than the allocation buys. Default off; kept as a knob.
#ifndef FMOE_TP_S3_NOINLINE
#define FMOE_TP_S3_NOINLINE 0
#endif
__device__ __noinline__ void stage3_token_tp_call(const Params& P, unsigned epoch, uint8_t* smem, int t, int n_rg) {
    stage3_token<true, true>(P, epoch, smem, t, n_rg);
}

// FMOE_TP_UNION_PREFETCH_KT: L2-prefetch k-tiles 0..N-1 of expert e's w13 (a k-tile = 4 k32 slices x 8 KB, both matrices and
// both row halves, contiguous) and the matching offset rows (2 KB per k-tile, contiguous). Issued by the thread that discovers the
// expert's union slot, ~1.5 us before fc1_early_weights fires the same bytes as TMA boxes from HBM.
// Every CTA builds the same union, so the work is STRIPED over the grid: CTA b prefetches k-tile b / 66 of the expert in slot b % 66
//.
__device__ __forceinline__ void tp_prefetch_fc1_first_tiles(const Params& P, int e, int slot) {
    constexpr int KT_N = FMOE_TP_UNION_PREFETCH_KT > 0 ? FMOE_TP_UNION_PREFETCH_KT : 1;
    const int b = (int)blockIdx.x;
    if (slot != b % 66) return;
#pragma unroll
    for (int kt = b / 66; kt < KT_N; kt += 2) {   // KT_N > 2: CTA b also takes k-tiles b/66 + 2, + 4, ... of its slot
        const uint8_t* w = P.fc1_w + (size_t)e * FC1_W_EXPERT_BYTES + (size_t)kt * 4 * FC1_W_SLICE_BYTES;   // k-tile kt: 4 k32 slices x 8 KB, contiguous
        bulk_prefetch_l2(w, 2 * FC1_W_SLICE_BYTES);
        bulk_prefetch_l2(w + 2 * FC1_W_SLICE_BYTES, 2 * FC1_W_SLICE_BYTES);
        bulk_prefetch_l2(P.fc1_s + (size_t)e * FC1_S_EXPERT_BYTES + (size_t)kt * 4 * FC1_S_SLICE_BYTES, 4 * FC1_S_SLICE_BYTES);   // 2 KB of offsets
    }
}
// ======================= prologue stage 4: gather all top-8 lists, build the union tables =======================
// Slot order = order of first appearance scanning (token, rank) candidates token-major; computed with
// order-independent smem atomics (min / or), so every CTA and every rank gets the identical list.
template <int EXACT_M, int PARTS, bool TP = false>
__device__ __forceinline__ void stage4_build_tables(const Params& P, unsigned epoch, uint8_t* smem, int* s_union, unsigned long long* s_mask,
                                                    int* s_slot, float* s_tkw, int* s_misc, int M) {
    int* s_first = reinterpret_cast<int*>(smem + OFF_PS_FIRST);         // [384] first candidate index of expert e
    unsigned* s_fbits = reinterpret_cast<unsigned*>(s_first + NEXP);    // [16] bit i: candidate i is a first occurrence
    int* s_prefix = reinterpret_cast<int*>(s_fbits + 16);                // [16] exclusive popcount prefix, prologue-only scratch
    int* s_tkid = reinterpret_cast<int*>(smem + OFF_PS_TKID);           // [64][8]
    const int tid = threadIdx.x;
    if constexpr (EXACT_M == 32 && PARTS == 2) {
        if (tid == 0) {
#ifndef FMOE_DIAG
            s_misc[35] = P.m32_tail_scratch && P.mode == 0 && P.reserve < 0 && P.out_bf16;
#else
            s_misc[35] = 0;
#endif
        }
    }
    for (int i = tid; i < NEXP; i += NTHREADS) { s_first[i] = 0x7FFFFFFF; s_mask[i] = 0ull; }
    if (tid < 16) s_fbits[tid] = 0u;
    if (tid < (TOPK / 2) * M) {   // gather top-8 lists (one message per thread, all in flight)
        const int t = tid / (TOPK / 2), j = tid - t * (TOPK / 2);
        uint4 m;
        if constexpr (TP && FMOE_TP_S4_BACKOFF_NS > 0) {   // 128 CTAs x 128 threads spin on the same 4 KB while the owners run stage 3: back off
            while (true) { m = ld_ll(P.topk + t * (TOPK / 2) + j); if (m.w == epoch) break; __nanosleep(FMOE_TP_S4_BACKOFF_NS); }
        } else
        do { m = ld_ll(P.topk + t * (TOPK / 2) + j); } while (m.w != epoch);
        s_tkid[t * TOPK + 2 * j] = (int)(m.x & 0xFFFFu); s_tkid[t * TOPK + 2 * j + 1] = (int)(m.x >> 16);
        s_tkw[t * TOPK + 2 * j] = __uint_as_float(m.y); s_tkw[t * TOPK + 2 * j + 1] = __uint_as_float(m.z);
    } else if (tid >= 256 && tid < 256 + M) {   // x_fp8 / scales of every token visible before fc1 gathers them
        // acquire per poll: the barrier below extends the ordering to every thread's later loads (fc1 gathers are generic-proxy cp.async)
        if constexpr (TP && FMOE_TP_S4_BACKOFF_NS > 0) { while (ld_acquire_gpu(&P.xflags[tid - 256]) != epoch) __nanosleep(FMOE_TP_S4_BACKOFF_NS); }
        else
        if (FMOE_XFLAG_ACQUIRE) { while (ld_acquire_gpu(&P.xflags[tid - 256]) != epoch) {} }
        else { while (ld_relaxed_gpu(&P.xflags[tid - 256]) != epoch) {} }
    }
    if (!FMOE_XFLAG_ACQUIRE) fence_acq_rel_gpu();
    fence_proxy_async_global();
    __syncthreads();
    const int ncand = TOPK * M;   // up to 512 candidates over NTHREADS threads
    for (int i = tid; i < ncand; i += NTHREADS) atomicMin(&s_first[s_tkid[i]], i);
    __syncthreads();
    // TP, FMOE_TP_S4_FOLD: 256 candidates = 8 warps -> the first-occurrence word of a warp is ONE ballot (no smem atomicOr
    // round + barrier), and every candidate thread sums its slot prefix from the 8 words inline (no warp-0 scan pass + barrier).
    // Same definition of slot / union / mask -> identical tables. The wave-1 experts' first FC1 k-tiles are L2-prefetched here.
    if constexpr (TP && FMOE_TP_S4_FOLD && EXACT_M == 32 && PARTS == 2) {
        static_assert(TOPK * EXACT_M == 256 && NTHREADS >= 256, "one candidate per thread, eight candidate warps");
        if (tid < 256) {
            const bool first = s_first[s_tkid[tid]] == tid;
            const unsigned bits = __ballot_sync(0xffffffffu, first);
            if ((tid & 31) == 0) s_fbits[tid >> 5] = bits;
        }
        __syncthreads();
        if (tid < 256) {
            const int i = tid, e = s_tkid[i], f = s_first[e];
            unsigned fb[8];
#pragma unroll
            for (int w = 0; w < 8; ++w) fb[w] = s_fbits[w];   // 8 broadcast smem reads
            const int fw = f >> 5;
            const unsigned fm = (1u << (f & 31)) - 1u;
            int slot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) slot += __popc(fb[w] & (w < fw ? 0xffffffffu : (w == fw ? fm : 0u)));
            s_slot[i] = slot;
            if (f == i) {
                s_union[slot] = e;
                if constexpr (FMOE_TP_UNION_PREFETCH_KT > 0) { if (slot < 66) tp_prefetch_fc1_first_tiles(P, e, slot); }   // wave-1 slots only, striped over CTAs
            }
            static_assert(FMOE_MASK_OR32, "the folded stage 4 uses the 32-bit mask atomics");
            const unsigned bit = 1u << (i / TOPK);
            const unsigned old = atomicOr(reinterpret_cast<unsigned*>(&s_mask[slot]), bit);
            if (__popc(old | bit) > 8) atomicExch(&s_misc[35], 0);
            if (tid == 0) { int U = 0;
#pragma unroll
                for (int w = 0; w < 8; ++w) U += __popc(fb[w]);
                s_misc[32] = U;
                if constexpr (FMOE_TP_TASK_NOINIT) { s_misc[48] = 0; s_misc[59] = 0; s_misc[60] = -1; }   // build_m32_token_tasks<TP> skips its init barrier
            }
        }
        __syncthreads();
        if (blockIdx.x == 0 && P.union_out) {
            const int U = s_misc[32];
            if (tid == 0) P.union_out[0] = U;
            for (int u = tid; u < U; u += NTHREADS) P.union_out[1 + u] = s_union[u];
        }
        return;
    }
    // First-occurrence bits (one candidate per thread), then a 16-word exclusive popcount prefix by warp 0 so that a slot is
    // two smem reads instead of a serial popc loop over up to 15 words per candidate (that loop cost ~0.5 us of the union
    // build on the M32 path:). The prefix scan also yields U.
    constexpr bool SCAN_PREFIX = EXACT_M == 64 || FMOE_PROLOGUE_TIGHT;
    if constexpr (EXACT_M == 64) {
    static_assert(NTHREADS >= MAXM * TOPK, "one candidate per thread");
    if (tid < 16 * 32) {
        const bool first = tid < ncand && s_first[s_tkid[tid]] == tid;
        const unsigned bits = __ballot_sync(0xffffffffu, first);
        if ((tid & 31) == 0) s_fbits[tid >> 5] = bits;
    }
    __syncthreads();
    } else {
        for (int i = tid; i < ncand; i += NTHREADS) if (s_first[s_tkid[i]] == i) atomicOr(&s_fbits[i >> 5], 1u << (i & 31));
        __syncthreads();
    }
    if constexpr (SCAN_PREFIX) {
    if (tid < 32) {
        const int count = tid < 16 ? __popc(s_fbits[tid]) : 0;
        int inclusive = count;
#pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int previous = __shfl_up_sync(0xffffffffu, inclusive, delta);
            if (tid >= delta) inclusive += previous;
        }
        if (tid < 16) s_prefix[tid] = inclusive - count;
        const int total = __shfl_sync(0xffffffffu, inclusive, 15);
        if (tid == 0) s_misc[32] = total;
    }
    __syncthreads();
    }
    for (int i = tid; i < ncand; i += NTHREADS) {
        const int e = s_tkid[i], f = s_first[e];
        int slot = __popc(s_fbits[f >> 5] & ((1u << (f & 31)) - 1u));
        if constexpr (SCAN_PREFIX) slot += s_prefix[f >> 5];
        else for (int w = 0; w < (f >> 5); ++w) slot += __popc(s_fbits[w]);
        s_slot[i] = slot;
        if (f == i) {
            s_union[slot] = e;
            if constexpr (TP && FMOE_TP_UNION_PREFETCH_KT > 0) { if (slot < 66) tp_prefetch_fc1_first_tiles(P, e, slot); }   // wave-1 slots only, striped over CTAs
        }
        if constexpr (EXACT_M == 32 && PARTS == 2 && FMOE_MASK_OR32) {
            // M <= 32: every token bit lives in the low word (little-endian) and the high word stays 0, so a native ATOMS.OR.32
            // replaces the 64-bit CAS loop; the popcount over the low word is the popcount of the whole mask.
            static_assert(EXACT_M <= 32, "token bits fit the low word");
            const unsigned bit = 1u << (i / TOPK);
            const unsigned old = atomicOr(reinterpret_cast<unsigned*>(&s_mask[slot]), bit);
            if (__popc(old | bit) > 8) atomicExch(&s_misc[35], 0);
        } else if constexpr (EXACT_M == 32 && PARTS == 2) {
            const unsigned long long bit = 1ull << (i / TOPK);
            const unsigned long long old = atomicOr(&s_mask[slot], bit);
            // Monotone masks detect the ninth routed token in any atomic order.
            // The existing final CTA join publishes the guard to both roles.
            if (__popcll(old | bit) > 8) atomicExch(&s_misc[35], 0);
        } else {
            atomicOr(&s_mask[slot], 1ull << (i / TOPK));
        }
    }
    if constexpr (!SCAN_PREFIX) {
        if (tid == 0) { int U = 0; for (int w = 0; w < 16; ++w) U += __popc(s_fbits[w]); s_misc[32] = U; }
    }
    __syncthreads();
    if (blockIdx.x == 0 && P.union_out) {
        const int U = s_misc[32];
        if (tid == 0) P.union_out[0] = U;
        for (int u = tid; u < U; u += NTHREADS) P.union_out[1 + u] = s_union[u];
    }
}
// M32 only: upper halves of slot/weight tables are dead (tokens32..63).
// Reuse1024 bytes each for256 prefixes and512 uint16 task descriptors.
// After LUT construction the prefix storage retains chunk counts through
// FC2; FC2's warp-scale scratch overwrites the old masks, not this storage.
// Live rows0..31 and all persistent routing/FC2 tables are unchanged.
template <int EXACT_M, int PARTS>
__device__ __forceinline__ bool m32_token_tasks(const Params& P, const int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2 && KSPLIT == 1) return misc[39] != 0;
    return false;
}
// ---- first FC1 tiles issued from the task build (FMOE_EARLY_FIRST_TILE) ----
// misc[60] = this CTA's first token task as (slot << 8) | (chunk << 1) | row half, or -1 (no task / legacy schedule). Both
// helpers run once, by the same warps that later own the roles (PROD_WARP: weight TMA lanes 0/1, PROD_WARP + 1: the 32 gather
// lanes), on stages 0..NSTAGE1-1 with the ring's fresh barriers; fc1_producer then skips period 0 of its first item.
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fc1_early_weights(const Params& P, uint8_t* smem, uint64_t* full, int e, int hf, bool pre_split = false) {
    const int lane = threadIdx.x & 31;
    if ((FMOE_EARLY_PRE_SPLIT || pre_split) && lane < 2) tma_prefetch_map(P.tmaps + lane);   // maps 0/1 were acquired by these lanes in the prologue wait window
    if constexpr (!FMOE_EARLY_FENCE_HOIST) fence_proxy_async_smem();   // the ring bytes were prologue scratch written through the generic proxy (else: hoisted)
    // Single-warp prologue code runs at ~6 cycles per dependent instruction: keep the issue straight-line and batched
    // (all expect_tx arrives first, one syncwarp, then the copies) instead of produce_tile's per-stage sequence.
    constexpr int N_EARLY = FMOE_EARLY_FIRST_TILE < NSTAGE1 ? FMOE_EARLY_FIRST_TILE : NSTAGE1;
    if (lane == 0) {
        STAMP(P, 14);   // phase build: first weight TMA issued
#pragma unroll
        for (int s = 0; s < N_EARLY; ++s) mbar_arrive_expect_tx(&full[s], FC1_TILE_BYTES);
    }
    __syncwarp();   // the expect_tx arrives precede every copy's complete_tx
    const int row0 = e * (NKT1 * 4);
    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < N_EARLY; ++s) tma_load_5d(smem + OFF_RING + s * FC1_STAGE_BYTES, P.tmaps + 0, 0, 0, hf, 0, row0 + s * 4, &full[s]);
    } else if (lane == 1) {
#pragma unroll
        for (int s = 0; s < N_EARLY; ++s) tma_load_4d(smem + OFF_RING + s * FC1_STAGE_BYTES + FC1_TILE_W_BYTES, P.tmaps + 1, 0, hf, 0, row0 + s * 4, &full[s]);
    }
}
// x rows (and, at WIDTH 8, the per-k-tile activation scales) of the first task's chunk into stages 0..NSTAGE1-1: fc1_producer's
// CACHE_GATHER path for tiles kt = 0..5. p_tok is filled here from the slot's mask (the producer rebuilds the same list later).
template <int WIDTH>
__device__ __forceinline__ void fc1_early_gather(const Params& P, uint8_t* smem, uint64_t* full, int* p_tok, unsigned long long mask, int first_token) {
    static_assert(WIDTH == 8 || WIDTH == 16, "exact-M32 FC1 widths");
    const int lane = threadIdx.x & 31;
    const int T = build_token_list(mask, p_tok, nullptr, lane);
    const int chunk_tokens = min(T - first_token, WIDTH);
    const int r0 = lane >> 3, r1 = r0 + 4, q = lane & 7;
    const bool gather_enabled = chunk_tokens > 0 && !(P.mode & 2);
    const bool valid0 = r0 < chunk_tokens, valid1 = r1 < chunk_tokens;
    constexpr bool XS_STAGE = FMOE_FC1_XS_STAGE && WIDTH == 8;   // matches fc1_consumer<8>'s stage-resident scales
    const uint8_t* src0 = nullptr; const uint8_t* src1 = nullptr; const uint8_t* src2 = nullptr; const uint8_t* src3 = nullptr;
    const float* xs0 = nullptr; const float* xs1 = nullptr;
    if (gather_enabled) {
        const int tok0 = valid0 ? p_tok[first_token + r0] : 0;
        const int tok1 = valid1 ? p_tok[first_token + r1] : 0;
        src0 = P.xq_buf + (size_t)tok0 * DIM + q * 16;
        src1 = P.xq_buf + (size_t)tok1 * DIM + q * 16;
        if constexpr (XS_STAGE) { xs0 = P.xs_buf + (size_t)tok0 * NKT1; xs1 = P.xs_buf + (size_t)tok1 * NKT1; }
        if constexpr (WIDTH == 16) {
            const int tok2 = r0 + 8 < chunk_tokens ? p_tok[first_token + r0 + 8] : 0;
            const int tok3 = r1 + 8 < chunk_tokens ? p_tok[first_token + r1 + 8] : 0;
            src2 = P.xq_buf + (size_t)tok2 * DIM + q * 16;
            src3 = P.xq_buf + (size_t)tok3 * DIM + q * 16;
        }
    }
    constexpr int N_EARLY = FMOE_EARLY_FIRST_TILE < NSTAGE1 ? FMOE_EARLY_FIRST_TILE : NSTAGE1;
#pragma unroll
    for (int kt = 0; kt < N_EARLY; ++kt) {
        uint8_t* xst = smem + OFF_RING + kt * FC1_STAGE_BYTES + FC1_TILE_BYTES;
        if (gather_enabled) {
            cp_async_16(xst + r0 * 128 + ((q ^ r0) << 4), src0 + kt * KT, valid0 ? 16u : 0u);
            cp_async_16(xst + r1 * 128 + ((q ^ r1) << 4), src1 + kt * KT, valid1 ? 16u : 0u);
            if constexpr (XS_STAGE) {
                if (q == 0) {
                    cp_async_4(xst + 1024 + r0 * 4, xs0 + kt, valid0 ? 4u : 0u);
                    cp_async_4(xst + 1024 + r1 * 4, xs1 + kt, valid1 ? 4u : 0u);
                }
            }
            if constexpr (WIDTH == 16) {
                cp_async_16(xst + 1024 + r0 * 128 + ((q ^ r0) << 4), src2 + kt * KT, r0 + 8 < chunk_tokens ? 16u : 0u);
                cp_async_16(xst + 1024 + r1 * 128 + ((q ^ r1) << 4), src3 + kt * KT, r1 + 8 < chunk_tokens ? 16u : 0u);
            }
        }
        cp_async_mbar_arrive_noinc(&full[kt]);
    }
}
template <int EXACT_M, int PARTS, bool TP = false>
__device__ __forceinline__ void build_m32_token_tasks(const Params& P, uint8_t* smem, int* misc) {
    if constexpr (EXACT_M == 32 && PARTS == 2 && KSPLIT == 1) {
        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        constexpr bool NOINIT = TP && FMOE_TP_TASK_NOINIT && !FMOE_ONEWAVE_KSPLIT;   // misc[48/59/60] were zeroed in stage 4; misc[39] recomputed per thread
        bool ok39 = true;
        if constexpr (NOINIT) {
            ok39 = P.m32_chunk_scratch && P.m32_tail_scratch && !P.pre_routed && P.mode == 0 && P.reserve < 0 && P.out_bf16 &&
                   (!misc[35] || (FMOE_GP && misc[32] > 108));   // uniform (misc[35] final since stage 4's last barrier); FMOE_GP: all-light loads above U 108 too
            if (tid == 0) misc[39] = ok39;   // published by the scan barrier below (read after it by the FC1 roles)
        } else
        if (tid == 0) {
            misc[48] = 0;
            misc[59] = 0;   // hot (N16) task count of the mixed two-wave schedule
            misc[60] = -1;  // first token task of this CTA (FMOE_EARLY_FIRST_TILE), resolved in the task-table loop below
            if constexpr (FMOE_ONEWAVE_KSPLIT) {
                misc[61] = 0;   // one-wave K-split quarters per task, resolved with misc[38] below
                misc[62] = 0;   // one-wave K-split: the consumer's piece counter
            }
            misc[39] = P.m32_chunk_scratch && P.m32_tail_scratch &&
                !P.pre_routed && P.mode == 0 && P.reserve < 0 && P.out_bf16 && (!misc[35] || (FMOE_GP && misc[32] > 108));   // FMOE_GP: all-light loads above U 108 too
        }
        if constexpr (!NOINIT) { __syncthreads(); ok39 = misc[39]; }
        if (!ok39) {
            if constexpr (NOINIT) __syncthreads();   // publish misc[39] = 0 before the FC1 roles read it (the init barrier used to)
            // All-light loads keep the legacy K36/K12 static tails, but their FC2 split (60 experts for the early
            // owners) left the A-owners 10+ us behind the helpers. Plan it like the token-task schedule instead
            // (misc[48] == 2: light offsets in build_m32_joint_plan; every expert is ready before B/H start).
            if (FMOE_LIGHT_PLAN && tid == 0) {
                const int U = misc[32];
                if (misc[35] && P.reserve < 0 && U > 66 && U <= 97 && !P.pre_routed &&
                    m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc)) { misc[48] = 2; misc[53] = U; }
            }
            if constexpr (fc2_prefill<EXACT_M, PARTS>()) __syncthreads();   // warp 19 reads misc[48]/[53] at FC1 start (the token-task path below ends in a barrier)
            return;
        }
        const int U = misc[32];
        const auto* masks = reinterpret_cast<const unsigned long long*>(smem + OFF_TAB + TAB_MASK);
        int* prefix = reinterpret_cast<int*>(smem + OFF_TAB + TAB_SLOT) + 32 * TOPK;
        auto* tasks = reinterpret_cast<uint16_t*>(smem + OFF_TAB + TAB_TKW + 32 * TOPK * 4);
#if FMOE_MIXED_WIDTH
        // Per expert: cold (<= 8 tokens, one N8 chunk) or hot (> 8 tokens). Three prefix fields, 10 bits each
        // (<= 256 token assignments): cold count | hot experts' N16 chunks | hot experts' N8 chunks.
        int fields = 0, inclusive = 0, tokens = 0;
        if (tid < 256) {
            tokens = tid < U ? __popcll(masks[tid]) : 0;
            const bool hot = tokens > 8;
            fields = hot ? ((((tokens + 15) / 16) << 10) | (((tokens + 7) / 8) << 20)) : (tokens > 0 ? 1 : 0);
            inclusive = fields;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, inclusive, d);
                if (lane >= d) inclusive += v;
            }
            prefix[tid] = inclusive - fields;
            if (lane == 31) misc[40 + warp] = inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            int total = lane < 8 ? misc[40 + lane] : 0;
            const int own = total;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, total, d);
                if (lane >= d) total += v;
            }
            if (lane < 8) misc[40 + lane] = total - own;
            if (lane == 7) {
                const int cold = total & 0x3ff, hot16 = (total >> 10) & 0x3ff, hot8 = (total >> 20) & 0x3ff;
                // Physical widths: one wave of N16 when 2*sum(ceil(T/16)) fits 132 CTAs
                // (unchanged); otherwise MIXED two-wave: cold experts stream N8 (2 tasks each), hot experts
                // stream their weights ONCE per half as N16 chunks on the last H = 2*hot16 CTAs (helpers),
                // instead of 2*ceil(T/8) N8 tasks. N8 task positions [0, 132-H) and [132, n_tasks), N16
                // positions [132-H, 132); misc[38] - 132 stays the N8 tail count. Hot CTAs stay within the
                // helper set (H <= 36) and the tail must fit the late N8 CTAs; otherwise all-N8 as before.
                const bool wide = 2 * (cold + hot16) <= 132;
                int H = 2 * hot16, n_tasks = 2 * cold + H;
                const bool helpers = m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc);
                bool mixed = FMOE_MIXED_WIDTH && !wide && H > 0 && H <= 36 && n_tasks + 2 * H <= 264;
                if (!mixed) { H = 0; n_tasks = wide ? 2 * (cold + hot16) : 2 * (cold + hot8); }
                misc[58] = wide;
                misc[59] = H;
                misc[38] = n_tasks;
                if constexpr (FMOE_ONEWAVE_KSPLIT) misc[61] = wide ? m32_ks_quarters(P, n_tasks) : 0;   // one-wave K-split quarters
                if (n_tasks > M32_MAX_TASKS) misc[39] = 0;
                misc[48] = !wide && n_tasks > 132 && n_tasks + H <= 216 && U <= 108 && helpers && !(FMOE_GP && n_tasks > FMOE_GP_MIN_TASKS);
            }
        }
        __syncthreads();
        if (!misc[39]) return;
        if (tid < U) {
            const int H = misc[59], wide = misc[58];
            const int base = prefix[tid] + misc[40 + warp];
            const int cold_p = base & 0x3ff, hot16_p = (base >> 10) & 0x3ff, hot8_p = (base >> 20) & 0x3ff;
            const bool hot = tokens > 8;
            int nchunks, first, hot_task = 0;
            if (wide) { first = cold_p + hot16_p; nchunks = hot ? (tokens + 15) / 16 : 1; }
            else if (H > 0 && hot) { first = hot16_p; nchunks = (tokens + 15) / 16; hot_task = 1; }
            else if (H > 0) { first = cold_p; nchunks = 1; }
            else { first = cold_p + hot8_p; nchunks = hot ? (tokens + 7) / 8 : 1; }
            prefix[tid] = nchunks | (hot_task << 8);   // FC2: chunk count (ready mask) and hot flag (ordering)
            const int wave1_pairs = (132 - H) / 2;
            if (misc[48] && !hot_task && first < wave1_pairs && first + nchunks >= wave1_pairs)
                misc[53] = tid + (first + nchunks == wave1_pairs);
            for (int c = 0; c < nchunks; ++c) {
                int pos = 2 * (first + c);
                if (hot_task) pos += 132 - H;
                else if (pos >= 132 - H) pos += H;
                tasks[pos] = uint16_t(((tid * 2) << 2) | c);
                tasks[pos + 1] = uint16_t(((tid * 2 + 1) << 2) | c);
            }
        }
#else
        // the earlier schedule's build, byte for byte (the three-field version above measured +0.5 us on every case when
        // compiled in with the mixed schedule disabled; paired, bitwise-identical output).
        int nchunks = 0, inclusive = 0;
        if (tid < 256) {
            const int tokens = tid < U ? __popcll(masks[tid]) : 0;
            // Prefix both widths together; <=256token assignments ensures
            // neither16-bit sum overflows into the other count.
            nchunks = (((tokens + 7) / 8) << 16) | ((tokens + 15) / 16);
            inclusive = nchunks;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, inclusive, d);
                if (lane >= d) inclusive += v;
            }
            prefix[tid] = inclusive - nchunks;
            if (lane == 31) misc[40 + warp] = inclusive;
        }
        __syncthreads();
#if FMOE_TASK_TABLE_FUSED
        // the cross-warp scan is redundant per thread (8 smem reads) instead of a serial warp-0 pass + barrier: the same
        // wide / n_tasks / joint values as the two-pass build below, misc words written by thread 0 only, one barrier less.
        int total = 0, wbase = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) { const int v = misc[40 + w]; total += v; wbase += w < warp ? v : 0; }
        const bool wide = 2 * (total & 0xffff) <= 132;
        const int n_tasks = 2 * (wide ? (total & 0xffff) : (total >> 16));
        const bool ok = n_tasks <= M32_MAX_TASKS;   // FMOE_GP: up to FMOE_GP_ROUNDS rounds (task table: 512 uint16 entries)
        // = m32_fc2_helpers(P, U, misc) with misc[39] = 1 and misc[58] = wide, which are not in smem yet (m32_one_wave_tasks / m32_two_wave_tasks)
        const bool helpers = P.reserve < 0 && (U > 66 || (FMOE_ONEWAVE_HELPERS && wide) || (FMOE_TASK_REGIME_PLAN && !wide)) && U <= 132 &&
                             P.fc2_helper_capacity && !P.pre_routed && P.out_bf16 && P.mode == 0;
        const bool joint = !wide && n_tasks > 132 && n_tasks <= 216 && U <= 108 && helpers && !(FMOE_GP && n_tasks > FMOE_GP_MIN_TASKS);
        if (tid == 0) {
            misc[58] = wide;
            misc[38] = n_tasks;
            if constexpr (FMOE_ONEWAVE_KSPLIT) misc[61] = wide ? m32_ks_quarters(P, n_tasks) : 0;   // one-wave K-split quarters
            if (!ok) misc[39] = 0;
            misc[48] = joint;
        }
        if (!ok) { __syncthreads(); return; }   // uniform (same inputs on every thread); the barrier publishes misc[39] = 0
        if (tid < U) {
            const int shift = wide ? 0 : 16;
            const int first = ((prefix[tid] + wbase) >> shift) & 0xffff;
            nchunks = (nchunks >> shift) & 0xffff;
            prefix[tid] = nchunks;
            if (joint && first < 66 && first + nchunks >= 66)
                misc[53] = tid + (first + nchunks == 66);
            if constexpr (FMOE_EARLY_FIRST_TILE > 0) {   // this CTA's first task: pair b/2 in this slot's chunk range (exactly one slot matches)
                const int b = (int)blockIdx.x, pair = b >> 1;
                if (first <= pair && pair < first + nchunks) misc[60] = (tid << 8) | ((pair - first) << 1) | (b & 1);
            }
            for (int c = 0; c < nchunks; ++c) {
                tasks[2 * (first + c)] = uint16_t(((tid * 2) << 2) | c);
                tasks[2 * (first + c) + 1] = uint16_t(((tid * 2 + 1) << 2) | c);
            }
        }
#else
        if (warp == 0) {
            int total = lane < 8 ? misc[40 + lane] : 0;
            // The ready mask removes the per-chunk polling cost. Keep the
            // all-light path, then admit by physical two-wave capacity.
            // !misc[35] already excludes all-light loads. Prefix both widths
            // before choosing the physical one-wave N16 path.
            const int own = total;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, total, d);
                if (lane >= d) total += v;
            }
            if (lane < 8) misc[40 + lane] = total - own;
            if (lane == 7) {
                const bool wide = 2 * (total & 0xffff) <= 132;
                misc[58] = wide;
                misc[38] = 2 * (wide ? (total & 0xffff) : (total >> 16));
                if constexpr (FMOE_ONEWAVE_KSPLIT) misc[61] = wide ? m32_ks_quarters(P, misc[38]) : 0;   // one-wave K-split quarters
                if (misc[38] > M32_MAX_TASKS) misc[39] = 0;
                misc[48] = !wide && misc[38] > 132 && misc[38] <= 216 && !(FMOE_GP && misc[38] > FMOE_GP_MIN_TASKS) &&
                           U <= 108 && m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc);
            }
        }
        __syncthreads();
        if (!misc[39]) return;
        if (tid < U) {
            const int shift = misc[58] ? 0 : 16;
            const int first = ((prefix[tid] + misc[40 + warp]) >> shift) & 0xffff;
            nchunks = (nchunks >> shift) & 0xffff;
            prefix[tid] = nchunks;
            if (misc[48] && first < 66 && first + nchunks >= 66)
                misc[53] = tid + (first + nchunks == 66);
            if constexpr (FMOE_EARLY_FIRST_TILE > 0) {   // this CTA's first task: pair b/2 in this slot's chunk range (exactly one slot matches)
                const int b = (int)blockIdx.x, pair = b >> 1;
                if (first <= pair && pair < first + nchunks) misc[60] = (tid << 8) | ((pair - first) << 1) | (b & 1);
            }
            for (int c = 0; c < nchunks; ++c) {
                tasks[2 * (first + c)] = uint16_t(((tid * 2) << 2) | c);
                tasks[2 * (first + c) + 1] = uint16_t(((tid * 2 + 1) << 2) | c);
            }
        }
#endif   // FMOE_TASK_TABLE_FUSED
#endif
        __syncthreads();
    }
}
// Legacy path: routing given by the host as union_experts[U] + gate_w[M][U] (<= 8 nonzeros per token).
__device__ __forceinline__ void build_tables_prerouted(const Params& P, int* s_union, unsigned long long* s_mask, int* s_slot, float* s_tkw, int* s_misc, int M) {
    const int tid = threadIdx.x, U = P.U;
    for (int u = tid; u < U; u += NTHREADS) {
        s_union[u] = P.union_experts[u];
        unsigned long long m = 0ull;
        for (int t = 0; t < M; ++t) if (P.gate_w[t * U + u] != 0.f) m |= 1ull << t;
        s_mask[u] = m;
    }
    if (tid < M) {
        int k = 0;
        for (int u = 0; u < U; ++u) {
            const float w = P.gate_w[tid * U + u];
            if (w != 0.f && k < TOPK) { s_slot[tid * TOPK + k] = u; s_tkw[tid * TOPK + k] = w; ++k; }
        }
        for (; k < TOPK; ++k) { s_slot[tid * TOPK + k] = -1; s_tkw[tid * TOPK + k] = 0.f; }
    }
    if (tid == 0) s_misc[32] = U;
}

// ======================= fc1 role =======================
// Consumer warpgroup rh (warps 4rh..4rh+3) owns rows [64rh, 64rh+64) of the item's 128-row half for BOTH matrices:
// per k32 slice it issues one gate and one up wgmma (m64 x n(8C) x k32) against the same token tile, so SiLU(g)*u is
// formed in registers. Producer = warp PROD_WARP (weights via cp.async.bulk, routed tokens via cp.async 16 B); the
// other warps of the producer warpgroup idle.
template <int N, bool ALIGNED_K128 = false>
__device__ __forceinline__ void issue4_n(float* S, const uint32_t A[4][4], uint32_t xaddr) {
    const uint64_t base = wg::make_desc_sw128(xaddr);
#pragma unroll
    for (int s4 = 0; s4 < 4; ++s4) {
        // The K128 base is128-byte aligned. Advancing K by32 changes only
        // start-address bits1:2 of the descriptor, with no carry into LBO/SBO.
        const uint64_t desc = ALIGNED_K128 ? (base | uint64_t(s4 * 2))
                                         : wg::make_desc_sw128(xaddr + s4 * 32);
        wg::MmaRS<N>::fma(S, A[s4], desc, s4 > 0 ? 1 : 0);
    }
}
template <int N>
__device__ __forceinline__ void issue4x2_n(float* Sg, float* Su, const uint32_t A[2][4][4], uint32_t xaddr) {
#pragma unroll
    for (int s4 = 0; s4 < 4; ++s4) {
        const uint64_t d = wg::make_desc_sw128(xaddr + s4 * 32);
        wg::MmaRS<N>::fma(Sg, A[0][s4], d, s4 > 0 ? 1 : 0);
        wg::MmaRS<N>::fma(Su, A[1][s4], d, s4 > 0 ? 1 : 0);
    }
}
// Only the guarded M32 static tail calls this permutation. Prioritize raw
// partials for CTAs96..107 before they begin their two-output FC2 streams.
// The five buckets are disjoint and cover [0,tails), including U67..97.
__device__ __forceinline__ int m32_fc1_helper_tail(int q, int tails) {
    const int critical = max(min(tails, 60) - 48, 0);
    const int first = min(tails, 12);
    const int last = max(min(tails, 48) - 36, 0);
    const int middle = max(min(tails, 36) - 12, 0);
    if (q < critical) return 48 + q;
    q -= critical;
    if (q < first) return q;
    q -= first;
    if (q < last) return 36 + q;
    q -= last;
    if (q < middle) return 12 + q;
    return 60 + q - middle;
}
template <int NT, int PARTS, int EXACT_M, int WIDTH_OVERRIDE = 0, bool TOKEN_ONLY = false, bool PRE_SPLIT = FMOE_EARLY_PRE_SPLIT != 0>
__device__ __forceinline__ void fc1_producer(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, int* p_tok, int* s_misc) {
    static_assert(!EXACT_M || (KSPLIT == 1 && NKT_ITEM % (2 * NSTAGE1) == 0),
                  "exact FC1 chunks span whole stage/parity periods");
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grid = EXACT_M ? 132 : (int)gridDim.x;
    const int* s_union = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_UNION);
    const unsigned long long* s_mask = reinterpret_cast<const unsigned long long*>(smem + OFF_TAB + TAB_MASK);
    const int n_items = 2 * KSPLIT * s_misc[32];   // (slot, row half, k half); the k-half-1 item of a pair comes first in queue order
    constexpr bool ONE_WAVE = EXACT_M == 32 && PARTS == 2 && WIDTH_OVERRIDE == 16;
    constexpr bool TASK_ONLY = ONE_WAVE || TOKEN_ONLY;
    static_assert(!TOKEN_ONLY || (EXACT_M == 32 && PARTS == 2 && WIDTH_OVERRIDE == 8));
    const bool asymmetric = !ONE_WAVE && m32_asymmetric_cohorts<EXACT_M, PARTS>(P, s_misc[32]);
    const bool token_tasks = TASK_ONLY || m32_token_tasks<EXACT_M, PARTS>(P, s_misc);
    const auto* task_table = reinterpret_cast<const uint16_t*>(smem + OFF_TAB + TAB_TKW + 32 * TOPK * 4);
    bool split_tail = false;
    if constexpr (EXACT_M == 32 && PARTS == 2 && !TASK_ONLY)
        split_tail = asymmetric && s_misc[32] <= 97 && !P.pre_routed && s_misc[35];
    const int n_tasks = token_tasks ? s_misc[38] : (split_tail ? 132 + 2 * (n_items - 132) : n_items);
    // "front" CTAs take exactly one full fc1 item (KSPLIT half-items) and then start fc2: PARTS 1/2 = the fc2 CTAs; PARTS 3 =
    // the CTAs of the 2-part tiles (they have the most fc2 work); everyone else drains the back region.
    int ftile, fpart, fnparts;
    fc2_map<PARTS>((int)blockIdx.x, grid, ftile, fpart, fnparts);
    const bool is_fc2 = PARTS < 3 ? ((int)blockIdx.x < N_FC2_TILES * PARTS) : (fnparts == 2);
    const int n_front_ctas = PARTS < 3 ? N_FC2_TILES * PARTS : 2 * (N_FC2_TILES - (grid - 2 * N_FC2_TILES));
    int* s_item_q = s_misc + 8;   // item indices handed from the producer to the consumers (ring of 8)
    {
        constexpr bool SPLIT_LOADS = EXACT_M != 0;
        constexpr int NGATHER = FC1_GATHER_THREADS<EXACT_M>;
        constexpr bool PREFILL = fc2_prefill<EXACT_M, PARTS>();
        if (warp != PROD_WARP && !(SPLIT_LOADS && warp <= PROD_WARP + NGATHER / 32)) {
            // FMOE_FC2_PREFILL: warps 18/19 have no FC1 role. Warp 19 lane 0 built the joint FC2 plan (misc[49..52]) right before this
            // call; the producer warpgroup's named barrier 3 publishes it to warps 16-18 once the FC1 sentinel is out (below).
            if constexpr (PREFILL) named_bar_sync(3, 128);
            return;
        }
        const bool weight_role = warp == PROD_WARP;
        const bool gather_role = !SPLIT_LOADS || warp > PROD_WARP;
        const int gather_tid = NGATHER == 32 ? lane : (warp - PROD_WARP - 1) * 32 + lane;
        // The per-thread tensormap acquire fence (~1 us) ran in the prologue's message-wait window when FMOE_FC1_ACQUIRE_EARLY
        // ; here only the cheap descriptor prefetch is repeated so the first TMA finds it cached. (Moving
        // the fence to the kernel start instead delayed stage 1's barrier by the fence: two-wave regressions.)
        if (weight_role && lane < 2) {
            if (FMOE_FC1_ACQUIRE_EARLY && !P.pre_routed) tma_prefetch_map(P.tmaps + lane); else tma_acquire_map(P.tmaps + lane);
        }
        if constexpr (TASK_ONLY && FMOE_EARLY_FIRST_TILE > 0 && !PRE_SPLIT) {
            // Tiles 0..N-1 of this CTA's first task (resolved in the task-table loop: s_misc[60]) go out right here, before the
            // item / token-list handoff chain below; period 0 of the first item is skipped. Issued after the role split on
            // purpose: every memory instruction issue crawls (0.2-0.8 us each) while the 132-CTA first-tile burst is in flight,
            // and setmaxnreg is a warpgroup collective -- issuing before it delayed the consumers' setup by ~3 us).
            const int ft = s_misc[60];
            if (ft >= 0) {
                const int u = ft >> 8, c = (ft >> 1) & 0x7f, hf = ft & 1;
                if (weight_role) fc1_early_weights(P, smem, full, s_union[u], hf);
                else if (gather_role) fc1_early_gather<WIDTH_OVERRIDE>(P, smem, full, p_tok, s_mask[u], c * WIDTH_OVERRIDE);
            }
        }
        // ---------------- producer warp: grabs items from the device-wide queue ----------------
        // Deterministic split of the item queue: the last `reserve` items form a back region drawn only by the
        // fc1-only CTAs through a second counter (work[3]); the front region [0, n_items - reserve) is drawn through
        // work[0] by the fc2 CTAs and, once the back region is exhausted, by the fc1-only CTAs too. With the prologue
        // all CTAs reach the queue within ~1 us, and the old single-counter quota ("fc2 CTAs stop when work[0] passes
        // the threshold") became a grab race that left fc1-only CTAs with a third item (+45 us of tail) on some rounds.
        // reserve (auto when P.reserve < 0): the front region holds exactly one item per fc2 CTA (front = min(n_fc2,
        // n_items)); the fc1-only CTAs take everything else in ceil((n_items - front) / n_fc1) rounds. Rationale (
        // profiles): an fc2 CTA holding a 2nd fc1 item starts fc2 one item late for ALL its experts, while the fc2 work
        // left after the fc1 tail is proportional to the item count of the LAST fc1-only round, which a larger front
        // region minimizes.
        int it = 0;
#ifdef FMOE_DIAG
        long long dp_wait = 0, dp_issue = 0;   // FC1 producer: cycles waiting for a free stage / issuing a tile (weight lane, gather lane)
#endif
        int reserve = P.reserve;
        if (reserve < 0) {
            const int n_fc2 = n_front_ctas;
            // measured: with parts=2 and more than two full items per fc2 CTA (M >= 48) a single shared queue
            // (front = everything) beats the one-item front by 15-35 us; otherwise one full item (KSPLIT half-items) per front CTA.
            reserve = (PARTS == 2 && n_items > 2 * KSPLIT * n_fc2) ? 0 : n_items - min(KSPLIT * n_fc2, n_items);
            // M32 two-part FC2 leaves36 back CTAs. Let all132 CTAs drain
            // the same FC1 queue instead of assigning96 back items to36.
            if constexpr (EXACT_M == 32 && PARTS == 2) reserve = 0;
        }
        const int n_front = max(n_items - reserve, 0);
        for (int n = 0;; ++n) {
            int item = n_tasks;
            if (weight_role && lane == 0) {
                if constexpr (ONE_WAVE) {
                    // The GPU prefix selected N16 only when every task fits
                    // the first132 CTAs. Still publish the usual sentinel.
                    if constexpr (FMOE_ONEWAVE_KSPLIT) { int k0, nk; bool hp; item = m32_ks_piece(s_misc, n, k0, nk, hp); }
                    else if (n == 0) item = (int)blockIdx.x;
                } else if (token_tasks) {
                    // Keep early FC2 owners at one FC1 task when84 late
                    // CTAs can finish all remaining tasks in one more round.
                    // Mixed widths: this (N8) CTA is below 132 - H; tail positions start at 132 either way.
                    if (FMOE_GP && FMOE_GP_MIN_TASKS < 216 && n_tasks > FMOE_GP_MIN_TASKS && !s_misc[48]) {
                        item = m32_gp_task((int)blockIdx.x, n, n_tasks);   // = m32_task_item (GP below the old 216 threshold)
                    } else if ((asymmetric || (FMOE_TASK_REGIME_PLAN && s_misc[48])) && n_tasks + s_misc[59] <= 216) {   // = m32_task_item
                        if (n == 0) item = (int)blockIdx.x;
                        else if (n == 1 && (int)blockIdx.x >= 48)
                            item = 132 + (s_misc[48] ? m32_joint_tail_ticket((int)blockIdx.x, s_misc[59])
                                                      : (int)blockIdx.x - 48);
                    } else if (FMOE_GP && n_tasks > FMOE_GP_MIN_TASKS && !s_misc[48]) {
                        item = m32_gp_task((int)blockIdx.x, n, n_tasks);   // FMOE_GP: must match fc1_consumer's m32_task_item
                    } else if (FMOE_LATE_SPLIT) {
                        // no joint plan (n_tasks > 216): owners, then single helpers, then dual helpers take the tail
                        if (n == 0) item = (int)blockIdx.x;
                        else if (n == 1) item = 132 + m32_late_ticket((int)blockIdx.x);
                    } else item = (int)blockIdx.x + n * 132;
                } else if (split_tail) {
                    // First132 full items stay fixed. Each late owner handles
                    // one K36 tail, each helper at most three K12 tails. No
                    // shared tail counter or owner/helper dependency cycle.
                    if (n == 0) item = (int)blockIdx.x;
                    else if ((int)blockIdx.x >= N_FC2_TILES) {
                        const int tails = n_items - 132, helpers = 84 - tails;
                        const int role = (int)blockIdx.x - N_FC2_TILES;
                        if (role < tails) {
                            if (n == 1) item = 132 + 2 * role;
                        } else {
                            const int q = role - tails + (n - 1) * helpers;
                            if (q < tails) item = 133 + 2 * m32_fc1_helper_tail(q, tails);
                        }
                    }
                } else if (asymmetric) {
                    // All132 first items are in-range. The first48 CTAs
                    // enter FC2 after one item; the other84 each request
                    // at most one tail ticket, preserving complete coverage.
                    if (n == 0) item = (int)blockIdx.x;
                    else if (n == 1 && (int)blockIdx.x >= N_FC2_TILES)
                        item = 132 + atomicAdd(&P.work[3], 1);
                } else {
                bool took = false;
                if (!is_fc2) {
                    const int r = atomicAdd(&P.work[3], 1);
                    if (r < n_items - n_front) { item = n_front + r; took = true; }   // back region, ascending
                }
                if (!took && *reinterpret_cast<volatile int*>(&P.work[0]) < n_front) {
                    const int f = atomicAdd(&P.work[0], 1);
                    if (f < n_front) item = f;
                }
                }
            }
            if constexpr (SPLIT_LOADS) {
                if (weight_role && lane == 0) s_misc[34] = item;
                // Also joins the previous item's gather before p_tok can be reused.
                named_bar_sync(2, 32 + NGATHER);
                item = s_misc[34];
            } else {
                item = __shfl_sync(0xffffffffu, item, 0);
            }
            if (item >= n_tasks) {   // no more work: publish the sentinel through the ring
                const int s = EXACT_M ? 0 : it % NSTAGE1;
                const int ph = EXACT_M ? 0 : (it / NSTAGE1) & 1;
                mbar_wait(&empty[s], ph ^ 1);
                if constexpr (PREFILL) named_bar_sync(3, 128);   // see the entry: joint plan (warp 19) visible to the FC2 producer warps
                if (weight_role && lane == 0) { s_item_q[n & 7] = -1; mbar_arrive(&full[s]); }
                if (gather_role) cp_async_mbar_arrive_noinc(&full[s]);
#ifdef FMOE_DIAG
                if (P.dbg && lane == 0 && (warp == PROD_WARP || warp == PROD_WARP + 1)) {   // FC1 producer counters: dbg[11] weight lane, dbg[12] gather lane
                    const unsigned w32 = (unsigned)min(dp_wait, 0xffffffffll), i32 = (unsigned)min(dp_issue, 0xffffffffll);
                    P.dbg[blockIdx.x * NSTAMP + (warp == PROD_WARP ? 11 : 12)] = (unsigned long long)w32 | ((unsigned long long)i32 << 32);
                }
#endif
                break;
            }
            if (weight_role && lane == 0) s_item_q[n & 7] = item;
            if constexpr (fc2_prefill<EXACT_M, PARTS>() && FMOE_FC2_TAIL_PREFETCH > 0 && !ONE_WAVE) {
                // a tail CTA (second FC1 item) already knows its FC2 expert sequence (warp 19 built the joint plan at FC1 start)
                // and will stream it ~25 us from now: warm L2 with its first FMOE_FC2_TAIL_PREFETCH weight boxes (same boxes, same order as
                // the weight lane's walk in fc2_producer) while wave-2 FC1 leaves HBM headroom, so the cold FC2 ring fills from L2.
                if (n == 1 && weight_role && lane == 0) {
                    const int U = s_misc[32];
                    if (is_fc2 || m32_fc2_helpers<EXACT_M, PARTS>(P, U, s_misc)) {
                        int eb, ee, es;
                        fc2_expert_range<EXACT_M, PARTS>(P, U, fpart, fnparts, eb, ee, es);
                        fc2_helper_range<EXACT_M, PARTS>(P, U, s_misc, ftile, fpart, fnparts, eb, ee, es);
                        const bool dual = m32_fc2_dual_helper<EXACT_M, PARTS>(P, U, s_misc);
                        int idx = eb;
                        for (int l = 0; l < FMOE_FC2_TAIL_PREFETCH && idx < ee; ++l) {
                            const int e = s_union[fc2_union_index<EXACT_M, PARTS>(s_misc, idx)];
                            tma_prefetch_3d(P.tmaps + 2, 0, (ftile + ((dual && (l & 1)) ? 36 : 0)) * 2, e * (INTER / 32));
                            if (!dual || (l & 1)) idx += es;
                        }
                    }
                }
            }
            int base_item = token_tasks ? task_table[item] >> 2 : item;
            constexpr int WIDTH = WIDTH_OVERRIDE ? WIDTH_OVERRIDE : (EXACT_M == 64 ? 16 : ((EXACT_M == 24 || EXACT_M == 32) ? 8 : (EXACT_M ? 16 : MAXM)));
            const int first_token = token_tasks ? (task_table[item] & 3) * WIDTH : 0;
            int kt0 = (KSPLIT == 2 ? 1 - (item & 1) : 0) * NKT_ITEM, nkt = NKT_ITEM;
            if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) { bool hp; m32_ks_piece(s_misc, n, kt0, nkt, hp); }   // owner: k-tiles from 0 (early tiles match)
            if constexpr (EXACT_M == 32 && PARTS == 2 && !TASK_ONLY) {
                if (split_tail && item >= 132) {
                    base_item = 132 + (item - 132) / 2;
                    const bool helper = (item & 1) != 0;
                    kt0 = helper ? 36 : 0;
                    nkt = helper ? 12 : 36;
                }
            }
            const int u = base_item / (2 * KSPLIT), hf = (base_item / KSPLIT) & 1, e = s_union[u];
            int T;
            if (weight_role) {
                T = build_token_list(s_mask[u], p_tok, nullptr, lane);
                if constexpr (SPLIT_LOADS) { if (lane == 0) s_misc[33] = T; }
            }
            if constexpr (SPLIT_LOADS) {
                named_bar_sync(2, 32 + NGATHER);
                T = s_misc[33];
            }
            const int token_end = token_tasks ? min(first_token + WIDTH, T) : max(T, 1);
            for (int token_base = first_token; token_base < token_end; token_base += WIDTH) {
            const int chunk_tokens = min(T - token_base, WIDTH);
            const int C = (chunk_tokens + 7) >> 3;
            constexpr bool CACHE_GATHER = EXACT_M == 32;
            const int r0 = lane >> 3, r1 = r0 + 4, q = lane & 7;
            const bool gather_enabled = chunk_tokens > 0 && !(P.mode & 2);
            const bool valid0 = r0 < chunk_tokens, valid1 = r1 < chunk_tokens;
            const uint8_t* src0 = nullptr;
            const uint8_t* src1 = nullptr;
            const uint8_t* src2 = nullptr;
            const uint8_t* src3 = nullptr;
            constexpr bool XS_STAGE = FMOE_FC1_XS_STAGE && EXACT_M == 32 && PARTS == 2 && WIDTH == 8;   // matches fc1_consumer<8>'s stage-resident scales
            const float* xs0 = nullptr;
            const float* xs1 = nullptr;
            if constexpr (CACHE_GATHER) {
                static_assert((WIDTH == 8 || WIDTH == 16) && NGATHER == 32);
                // The item handoff keeps p_tok stable throughout this chunk.
                // Reuse these two row bases for all 48 K128 tiles.
                if (gather_role && gather_enabled) {
                    const int tok0 = valid0 ? p_tok[token_base + r0] : 0;
                    const int tok1 = valid1 ? p_tok[token_base + r1] : 0;
                    src0 = P.xq_buf + (size_t)tok0 * DIM + q * 16;
                    src1 = P.xq_buf + (size_t)tok1 * DIM + q * 16;
                    if constexpr (XS_STAGE) { xs0 = P.xs_buf + (size_t)tok0 * NKT1; xs1 = P.xs_buf + (size_t)tok1 * NKT1; }
                    if constexpr (WIDTH == 16) {
                        const int tok2 = r0 + 8 < chunk_tokens ? p_tok[token_base + r0 + 8] : 0;
                        const int tok3 = r1 + 8 < chunk_tokens ? p_tok[token_base + r1 + 8] : 0;
                        src2 = P.xq_buf + (size_t)tok2 * DIM + q * 16;
                        src3 = P.xq_buf + (size_t)tok3 * DIM + q * 16;
                    }
                }
            }
            // A 5D map selects the same row half of gate and up without changing global packing.
            // Two TMA copies per k-tile: 16-KB [slice][mat][rh][1 KB] weights and 1-KB [slice][mat][128 B] offsets.
            // (Throttling the kernel-start burst -- tile 0 alone, then tiles 1-5 once it landed --
            // moved the median first tile only 0.2 us earlier and left fc1_end unchanged; not kept.)
            auto produce_tile = [&](int kt, int s, int ph) {
#ifdef FMOE_DIAG
                const long long tw0_ = clock64();
#endif
                mbar_wait(&empty[s], ph ^ 1);
#ifdef FMOE_DIAG
                const long long tw1_ = clock64(); dp_wait += tw1_ - tw0_;
#endif
                uint8_t* stage = smem + OFF_RING + s * FC1_STAGE_BYTES;
                if (weight_role) {
#ifdef FMOE_DIAG
                if (P.mode & 128) { if (lane == 0) mbar_arrive(&full[s]); } else   // diagnostics: no weight traffic (consumers compute on stale smem)
#endif
                {
                    if (n == 0 && kt == kt0 && tid == PROD_WARP * 32) STAMP(P, 14);   // phase build: first weight TMA issued
                    if (lane == 0) mbar_arrive_expect_tx(&full[s], FC1_TILE_BYTES);
                    __syncwarp();   // the expect_tx arrive precedes every copy's complete_tx
                    if (lane < 2) {
                        const int row = e * (NKT1 * 4) + kt * 4;   // k32 slice row of (expert, k-tile)
                        if (lane == 0) tma_load_5d(stage, P.tmaps + 0, 0, 0, hf, 0, row, &full[s]);
                        else tma_load_4d(stage + FC1_TILE_W_BYTES, P.tmaps + 1, 0, hf, 0, row, &full[s]);
                    }
                }
                }
                if (gather_role) {
                uint8_t* xst = stage + FC1_TILE_BYTES;
                if constexpr (CACHE_GATHER) {
                    if (gather_enabled) {
                        cp_async_16(xst + r0 * 128 + ((q ^ r0) << 4),
                                    src0 + kt * KT, valid0 ? 16u : 0u);
                        cp_async_16(xst + r1 * 128 + ((q ^ r1) << 4),
                                    src1 + kt * KT, valid1 ? 16u : 0u);
                        if constexpr (XS_STAGE) {   // this k-tile's 8 token scales [pos] behind the 1-KB x chunk (zero-filled past T)
                            if (q == 0) {
                                cp_async_4(xst + 1024 + r0 * 4, xs0 + kt, valid0 ? 4u : 0u);
                                cp_async_4(xst + 1024 + r1 * 4, xs1 + kt, valid1 ? 4u : 0u);
                            }
                        }
                        if constexpr (WIDTH == 16) {
                            cp_async_16(xst + 1024 + r0 * 128 + ((q ^ r0) << 4),
                                        src2 + kt * KT, r0 + 8 < chunk_tokens ? 16u : 0u);
                            cp_async_16(xst + 1024 + r1 * 128 + ((q ^ r1) << 4),
                                        src3 + kt * KT, r1 + 8 < chunk_tokens ? 16u : 0u);
                        }
                    }
                } else {
                const int n_gather = (P.mode & 2) ? 0 : C * 64;
                for (int i = gather_tid; i < n_gather; i += NGATHER) {
                    const int c = i >> 6, r = (i >> 3) & 7, q = i & 7;
                    const int pos = c * 8 + r;
                    const bool valid = pos < chunk_tokens;
                    const int tok = valid ? p_tok[token_base + pos] : 0;
                    const uint8_t* src = P.xq_buf + (size_t)tok * DIM + kt * KT + q * 16;
                    uint8_t* dst = xst + c * 1024 + r * 128 + ((q ^ r) << 4);
                    cp_async_16(dst, src, valid ? 16u : 0u);
                }
                }
                cp_async_mbar_arrive_noinc(&full[s]);
                }
#ifdef FMOE_DIAG
                dp_issue += clock64() - tw1_;
#endif
            };
            if constexpr (EXACT_M != 0) {
                // Tiles 0..N-1 of the first task were issued at entry (FMOE_EARLY_FIRST_TILE, misc[60]): skip them in period 0.
                const bool early = FMOE_EARLY_FIRST_TILE > 0 && TASK_ONLY && n == 0 && s_misc[60] >= 0;
#ifdef FMOE_PROBE
                if (early && ((s_misc[60] >> 8) != u || ((s_misc[60] >> 1) & 0x7f) != first_token / WIDTH || (s_misc[60] & 1) != hf)) __trap();
#endif
                // A full chunk resets the six-stage ring's index and parity.
                // Unroll one period only; the consumer's MMA loop is unchanged.
#pragma unroll 1
                for (int period = 0; period < nkt / NSTAGE1; ++period) {
#pragma unroll
                    for (int s = 0; s < NSTAGE1; ++s) {
                        if (early && period == 0 && s < FMOE_EARLY_FIRST_TILE) continue;   // issued at entry (fc1_early_weights / fc1_early_gather)
                        produce_tile(kt0 + period * NSTAGE1 + s, s, period & 1);
                    }
                }
            } else {
                for (int kt = kt0; kt < kt0 + NKT_ITEM; ++kt, ++it)
                    produce_tile(kt, it % NSTAGE1, (it / NSTAGE1) & 1);
            }
            }
        }
        return;
    }
}

template <int NT, int PARTS, bool CHUNKED = false, int EXACT_M = 0, bool TOKEN_ONLY = false>
__device__ __forceinline__ void fc1_consumer(const Params& P, unsigned epoch, uint8_t* smem, uint64_t* full, uint64_t* empty,
                                             int* s_tok, int* s_inv, int* s_misc, float* xp_s, float* hst, int M, int M_pad) {
    static_assert(!CHUNKED || KSPLIT == 1, "routed-token chunks require the unsplit-K path");
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // routing tables (built by the prologue; constant offsets from smem so they cost no live registers)
    const int* s_union = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_UNION);
    const unsigned long long* s_mask = reinterpret_cast<const unsigned long long*>(smem + OFF_TAB + TAB_MASK);
    const int* s_slot = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_SLOT);
    const float* s_tkw = reinterpret_cast<const float*>(smem + OFF_TAB + TAB_TKW);
    const int* s_item_q = s_misc + 8;
    constexpr bool ONE_WAVE = EXACT_M == 32 && PARTS == 2 && NT == 16;
    constexpr bool TASK_ONLY = ONE_WAVE || TOKEN_ONLY;
    static_assert(!TOKEN_ONLY || (EXACT_M == 32 && PARTS == 2 && NT == 8 && CHUNKED));
    const bool token_tasks = TASK_ONLY || m32_token_tasks<EXACT_M, PARTS>(P, s_misc);
    const auto* task_table = reinterpret_cast<const uint16_t*>(smem + OFF_TAB + TAB_TKW + 32 * TOPK * 4);
    // ---------------- consumer warpgroups (warps 0-15): WG = (row half, gate|up) ----------------
    // Per k-tile a WG issues 4 wgmmas m64 x nNT x k32 (one per k32 slice, all token chunks at once) into the scratch S
    // (scale_d = 0 on the first slice), then promotes S into the fp32 accumulator D with the per-token per-k128
    // activation scale (fp32 FMAs: keeps Hopper's reduced-precision fp8 accumulation confined to 128 k). The A fragments
    // are double-buffered: tile kt+1's dequant runs while tile kt's group is in flight.
    constexpr int NR = NT / 2;
    constexpr int NA = (NT >= 64) ? 1 : 2;   // A-fragment buffers: 2 (dequant overlaps the in-flight group); 1 at NT=64 where D+S+2A spill at 112 registers
    const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, tig = lane & 3;
    const int rh = wg >> 1, mat = wg & 1;
    const int tid128 = wq * 32 + lane;
    const int sel = ((g >> 1) + (wq & 1)) & 1;                        // which word pair of the lane chunk this warp takes (see "Weight layout")
    const int row_local0 = rh * 64 + 32 * (wq >> 1) + 16 * sel + g;   // weight row (within the 128-row half) of this thread's first accumulator row; +8 for the second
    int it = 0;
#ifdef FMOE_DIAG
    unsigned long long d_setup = 0, d_tiles = 0, d_epi = 0, d_items = 0, tA = 0, tB = 0, tC = 0;
    unsigned long long d_e[5] = {0, 0, 0, 0, 0}, tE = 0;   // epilogue / setup breakdown (DIAG): see FMOE_DIAG_EPI below
    long long c_wait = 0, c_dq = 0, c_mw = 0, c_pi = 0, c_is = 0;   // per-tile phase cycles (tid 0): full-wait, dequant, wgmma-wait, promote, fence+issue+commit
#define DIAG_T(v) do { if (tid == 0) (v) = gtimer(); } while (0)
#define DIAG_C(acc, t0) do { if (tid == 0) { const long long t1 = clock64(); (acc) += t1 - (t0); (t0) = t1; } } while (0)
#else
#define DIAG_T(v) do {} while (0)
#define DIAG_C(acc, t0) do {} while (0)
#endif
    for (int n = 0;; ++n) {
        DIAG_T(tA);
        if (n == 0 && tid == 0) STAMP(P, 19);   // phase build: consumer entered fc1_consumer (overwrites the TP router diag stamp)
        int item;
        // Measured: -1.45 us on both one-wave cases, +0.2..0.8 us on every two-wave
        // case, so the hoist is on for the N16 one-wave instantiation only; the N8 two-wave consumer stays as in the earlier schedule.
        if constexpr (ONE_WAVE || (TASK_ONLY && FMOE_SETUP_HOIST)) {
            // The task sequence of this CTA is a pure function of (blockIdx, n, misc): compute it here and run the
            // per-item setup (token list, activation scales) while tile 0 is still in flight, instead of after it
            // landed (that setup cost ~1.1 us per item on the critical path). The sentinel stage is still consumed
            // before leaving so that every mbarrier arrival has landed before the FC2 re-initialization.
            if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) {
                // Every piece is a whole even number of ring laps (12-tile multiples), so the ring is at stage 0 / phase 0 at each
                // item start: restart `it` per item and the mainloop's stage/phase arithmetic stays as cheap as the single-item one.
                it = 0;
                // The piece index lives in smem (s_misc[62], advanced by thread 0 after the setup barrier) so the loop carries no
                // register state of its own across the mainloop.
                int k0, nk; bool hp; item = m32_ks_piece(s_misc, reinterpret_cast<volatile int*>(s_misc)[62], k0, nk, hp);
            } else item = m32_task_item<ONE_WAVE>(P, n, s_misc);
            if (item >= s_misc[38]) {
                const int s = it % NSTAGE1, ph = (it / NSTAGE1) & 1;
                mbar_wait(&full[s], ph);
                if (n == 0 && tid == 0) STAMP(P, 10);
                if constexpr (TOKEN_ONLY && FC1_PUB_OFFLOAD) {   // sentinel for producer warp 18's publish loop
                    if (tid == 0) { int* pub_ = reinterpret_cast<int*>(smem + OFF_PUB); pub_[4 + (n & 7)] = -1; st_release_cta_s(pub_, n + 1); }
                }
                break;
            }
        } else {
            { const int s = it % NSTAGE1, ph = (it / NSTAGE1) & 1; mbar_wait(&full[s], ph); }   // tile 0 (or the sentinel)
            if (n == 0 && tid == 0) STAMP(P, 10);   // first fc1 tile landed (or no work)
            item = s_item_q[n & 7];
            if (item < 0) break;
        }
        int base_item = token_tasks ? task_table[item] >> 2 : item, split_piece = -1;
        const int first_token = token_tasks ? (task_table[item] & 3) * NT : 0;
        const int kh = (KSPLIT == 2 ? 1 - (item & 1) : 0);
        int kt0 = kh * NKT_ITEM, nkt = NKT_ITEM;
        // One-wave K-split: the consumer's k index stays LOCAL (0..nkt, kt0 keeps its compile-time 0) so tile_step/promote see
        // the same constants as a full task; only the scale-table fill adds the piece's true first k-tile (ks_kt0).
        int ks_kt0 = 0; bool ks_helper = false;
        if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) m32_ks_piece(s_misc, reinterpret_cast<volatile int*>(s_misc)[62], ks_kt0, nkt, ks_helper);
        if constexpr (EXACT_M == 32 && PARTS == 2 && !TASK_ONLY) {
            if (m32_asymmetric_cohorts<EXACT_M, PARTS>(P, s_misc[32]) && s_misc[32] <= 97 &&
                !P.pre_routed && s_misc[35] && item >= 132) {
                base_item = 132 + (item - 132) / 2;
                split_piece = item & 1;   // 0: K36 owner; 1: K12 helper
                kt0 = split_piece ? 36 : 0;
                nkt = split_piece ? 12 : 36;
            }
        }
        int u = base_item / (2 * KSPLIT), hf = (base_item / KSPLIT) & 1, e = s_union[u];
        if constexpr (EXACT_M == 32 && PARTS == 2) {
            // Consumer-only context, outside producer ring/misc words. Every
            // path joins all512 after reading it, before next-item overwrite.
            if (tid == 0) {
                s_misc[36] = base_item;
                if constexpr (!TASK_ONLY) s_misc[37] = split_piece;
                else if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) {
                    s_misc[37] = item | (ks_helper ? 0x8000 : 0) | (nkt << 16);   // epilogue re-reads (no live registers)
                }
            }
        }
#ifdef FMOE_DIAG
        if (tid == 0) { tE = gtimer(); d_e[4] += tE - tA; }   // setup part 1: item resolve (+ previous publish tail on tid 0)
#endif
        if (warp == 0) { const int T = build_token_list(s_mask[u], s_tok, s_inv, lane); if (lane == 0) s_misc[0] = T; }
        named_bar_sync(1, NCONS);
        if (n == 0 && tid == 0) STAMP(P, 20);   // phase build: first item's token list built (after the consumer barrier)
        const int T = s_misc[0];
        const int token_end = token_tasks ? min(first_token + NT, T) : max(T, 1);
        for (int token_base = first_token; token_base < token_end; token_base += (CHUNKED ? NT : MAXM)) {
        const int C = min((T - token_base + 7) >> 3, NT / 8);
        // Compact [kt][tig][NT/8][2] scale table: one contiguous run per MMA token lane,
        // only the physical FC1 width is staged instead of an unconditional 64-token table.
        constexpr bool XS_STAGE = FMOE_FC1_XS_STAGE && EXACT_M == 32 && PARTS == 2 && NT == 8;   // scales arrive in each stage (fc1_producer)
        if constexpr (XS_STAGE) {
        } else if constexpr (EXACT_M == 32 && PARTS == 2 && NT == 8) {
            static_assert(NT == 8 && NKT_ITEM * NT <= NCONS);
            // At most384 scale entries: one per consumer for K48/K36/K12.
            if (tid < nkt * NT) {
                const int i = kt0 * NT + tid, kt = i / NT, r = i % NT;
                const int pos = token_base + r;
                xp_s[i] = P.xs_buf[(pos < T ? s_tok[pos] : 0) * NKT1 + kt];
            }
        } else {
        // Constant trip count (2 predicated rounds at NT=16) even when the piece is shorter: a runtime bound made this a real
        // loop and pushed the N16 setup over the consumer's register cap.
        for (int i = kt0 * NT + tid; i < (kt0 + NKT_ITEM) * NT; i += NCONS) {
            if (i >= (kt0 + nkt) * NT) break;
            const int kt = i / NT, r = i - kt * NT, tg = r / (NT / 4);
            const int j = (r - tg * (NT / 4)) >> 1, e = r & 1;
            const int pos = token_base + 8 * j + 2 * tg + e;
            xp_s[i] = P.xs_buf[(pos < T ? s_tok[pos] : 0) * NKT1 + kt + ks_kt0];
        }
        }
        // the expert's weight factor for the epilogue (Humming's per-expert s2 x the e2m1->e4m3 exponent-bias offset 2^6): issued now
        float rs_e;
        if constexpr (!(EXACT_M == 32 && PARTS == 2)) rs_e = __ldg(P.fc1_s2 + e) * 64.0f;
        named_bar_sync(1, NCONS);
        // K-split piece counter: every thread read s_misc[62] before this barrier; the epilogue barriers order this write before the
        // next loop-top read.
        if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) { if (tid == 0) ++s_misc[62]; }
        DIAG_T(tB);
        if (n == 0 && token_base == first_token && tid == 0) STAMP(P, 15);   // phase build: first item's consumer setup done
        if constexpr (ONE_WAVE || (TASK_ONLY && FMOE_SETUP_HOIST)) {   // setup done: now wait for the item's tile 0 (one chunk per task, ring restarts at 0/0)
            if constexpr (FMOE_FC1_DESYNC_NS > 0 && TOKEN_ONLY) {
                if (mat == 1) { const unsigned long long t0_ = globaltimer_ns(); while (globaltimer_ns() - t0_ < FMOE_FC1_DESYNC_NS) {} }   // up WGs trail the gate WGs
            }
            const int s = it % NSTAGE1, ph = (it / NSTAGE1) & 1;
            mbar_wait(&full[s], ph);
            if (n == 0 && tid == 0) STAMP(P, 10);
        }

        float D[NR], S[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) { D[i] = 0.f; S[i] = 0.f; }
        uint32_t A[NA][4][4];   // [buffer][k32 slice][4 regs]

        // The k-tile loop always issues the full wgmma width N = NT: the shape and the accumulator set are compile-time
        // constants and there is exactly ONE wgmma loop per kernel instantiation. Anything else -- a runtime switch on C
        // inside the loop, or several N-specialized copies of the loop in the same function -- made ptxas give up on the
        // async pipeline (C7511/C7515: every wgmma followed by a wait). Chunks in [C, NT/8) multiply stale smem into
        // accumulator columns that are never read (positions >= T are not mapped by s_inv).
        {
            constexpr int N = NT;
            constexpr int CN = N / 8;
            auto promote = [&](int kt, float2 xin) {   // D += S * x_scale[tok][kt]; XS_STAGE (N8): xin = this lane's pair, read from the tile's stage
                const float* xr = xp_s + kt * NT + tig * (NT / 4);   // this lane's compact, contiguous (j, e) pairs
#pragma unroll
                for (int j = 0; j < CN; ++j) {
                    if constexpr (EXACT_M == 64) {
                        // The epilogue reads only the expert's live token panels.
                        if (j >= C) continue;
                    }
                    const float2 x = XS_STAGE ? xin : *reinterpret_cast<const float2*>(xr + 2 * j);
                    D[4 * j + 0] = fmaf(S[4 * j + 0], x.x, D[4 * j + 0]);
                    D[4 * j + 1] = fmaf(S[4 * j + 1], x.y, D[4 * j + 1]);
                    D[4 * j + 2] = fmaf(S[4 * j + 2], x.x, D[4 * j + 2]);
                    D[4 * j + 3] = fmaf(S[4 * j + 3], x.y, D[4 * j + 3]);
                }
            };
            // One k-tile. Called twice per loop iteration with a *static* A buffer reference so the
            // double-buffered fragments stay in registers (a runtime-indexed A[kt&1] goes to local memory).
            auto tile_step = [&](int kt, uint32_t (&Ac)[4][4], int ring_stage,
                                 int ring_phase, int previous_stage) {
                const int s = TOKEN_ONLY ? ring_stage : it % NSTAGE1;
                const int ph = TOKEN_ONLY ? ring_phase : (it / NSTAGE1) & 1;
#ifdef FMOE_DIAG
                long long tc = 0; if (tid == 0) tc = clock64();
#endif
                if (kt > kt0 || token_base > 0) mbar_wait(&full[s], ph);
                DIAG_C(c_wait, tc);
                if constexpr (!TOKEN_ONLY) {
                    if (P.mode & 1) { if (lane == 0) mbar_arrive(&empty[s]); ++it; return; }   // diagnostics: pipeline only
                }   // TOKEN_ONLY admission already requires P.mode == 0
                if constexpr (FMOE_FC1_DESYNC && TOKEN_ONLY && NA == 2) {
                    if (mat == 1) {   // "up" warpgroups: wait -> promote -> dequant -> issue (warp-uniform branch)
                        const uint8_t* stage_ = smem + OFF_RING + s * FC1_STAGE_BYTES;
                        const uint32_t xaddr_ = smem_u32(stage_ + FC1_TILE_BYTES);
                        const uint8_t* frag_ = stage_ + mat * (2 * HL_BLOCK_BYTES) + rh * HL_BLOCK_BYTES + ((wq >> 1) * 32 + lane) * 16 + sel * 8;
                        const uint8_t* scl_ = stage_ + FC1_TILE_W_BYTES + mat * 128 + rh * 64 + 8 * g + 4 * (wq >> 1) + 2 * sel;
                        wg::wait<0>();
                        fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[1]);
                        if (kt > kt0) {
                            const int ps = previous_stage;
                            float2 xv = make_float2(0.f, 0.f);
                            if constexpr (XS_STAGE) {
                                xv = *reinterpret_cast<const float2*>(smem + OFF_RING + ps * FC1_STAGE_BYTES + FC1_TILE_BYTES + 1024 + tig * 8);
                                asm volatile("" : "+f"(xv.x), "+f"(xv.y) :: "memory");
                            }
                            if (lane == 0) mbar_arrive(&empty[ps]);
                            promote(kt - 1, xv);
                        }
                        fence_operand(D);
                        dequant_slices4<4096, 256, (EXACT_M != 0)>(frag_, scl_, Ac);
                        fence_operand(S);
                        wg::fence();
                        issue4_n<N, (EXACT_M != 0)>(S, Ac, xaddr_);
                        wg::commit();
                        ++it;
                        return;
                    }
                }
                const uint8_t* stage = smem + OFF_RING + s * FC1_STAGE_BYTES;
                const uint32_t xaddr = smem_u32(stage + FC1_TILE_BYTES);
                // Combined gate/up TMA packs [slice][mat][rh] weights and [slice][mat] offsets.
                const uint8_t* frag = stage + mat * (2 * HL_BLOCK_BYTES) + rh * HL_BLOCK_BYTES + ((wq >> 1) * 32 + lane) * 16 + sel * 8;
                const uint8_t* scl = stage + FC1_TILE_W_BYTES + mat * 128 + rh * 64 + 8 * g + 4 * (wq >> 1) + 2 * sel;
#ifdef FMOE_DIAG   // diagnostics build: mode 16 = skip dequant, 32 = skip wgmma, 64 = skip promote
                const bool do_dq = !(P.mode & 16), do_mma = !(P.mode & 32), do_pr = !(P.mode & 64);
#else
                constexpr bool do_dq = true, do_mma = true, do_pr = true;
#endif
                auto dequant = [&]() {
#ifdef FMOE_DIAG   // mode 1024: fragment loads only; 2048: LUT ALU only (fragments from registers)
                    if (do_dq) {
                        if (P.mode & 1024) dequant_slices4_loadonly<4096, 256>(frag, scl, Ac);
                        else if (P.mode & 2048) dequant_slices4_noload<4096, 256>(scl, tid128, Ac);
                        else dequant_slices4<4096, 256, (EXACT_M != 0)>(frag, scl, Ac);
                    }
                    if (tid == 0) asm volatile("" :: "r"(Ac[0][0]), "r"(Ac[1][0]), "r"(Ac[2][0]), "r"(Ac[3][0]));   // dequant results materialized before the stamp
#else
                    if (do_dq) dequant_slices4<4096, 256, (EXACT_M != 0)>(frag, scl, Ac);
#endif
                };
                if constexpr (NA == 2) dequant();   // overlaps tile kt-1's group (other A buffer)
                DIAG_C(c_dq, tc);
                if (do_mma) wg::wait<0>();   // tile kt-1's group complete: S holds its products, its A buffer and stage are free
                fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[NA - 1]);
                DIAG_C(c_mw, tc);
                if (kt > kt0) {
                    const int ps = TOKEN_ONLY ? previous_stage : (it - 1) % NSTAGE1;
                    float2 xv = make_float2(0.f, 0.f);
                    if constexpr (XS_STAGE) {   // read the previous tile's scales before releasing its stage (the arrive's release orders the load)
                        xv = *reinterpret_cast<const float2*>(smem + OFF_RING + ps * FC1_STAGE_BYTES + FC1_TILE_BYTES + 1024 + tig * 8);
                        asm volatile("" : "+f"(xv.x), "+f"(xv.y) :: "memory");
                    }
                    if (lane == 0) mbar_arrive(&empty[ps]);
                    if (do_pr) promote(kt - 1, xv);
                }
                // Pin the completed promotion before S is reused. Otherwise
                // nvcc can clone S, defer the FMAs past the next MMA group and
                // copy its new accumulator before wait_group (C7515).
                if constexpr (EXACT_M == 24 || EXACT_M == 32 || EXACT_M == 64) fence_operand(D);
                if constexpr (NA == 1) dequant();   // single buffer: only after the group that read it completed
                DIAG_C(c_pi, tc);
                if (do_mma) {
                    fence_operand(S);
                    wg::fence();
                    static_assert(OFF_RING % 128 == 0 && FC1_STAGE_BYTES % 128 == 0 && FC1_TILE_BYTES % 128 == 0,
                                  "FC1 K128 activation bases must be128-byte aligned");
                    issue4_n<N, (EXACT_M != 0)>(S, Ac, xaddr);
                    wg::commit();
                } else {
#pragma unroll
                    for (int s4 = 0; s4 < 4; ++s4) asm volatile("" :: "r"(Ac[s4][0]), "r"(Ac[s4][1]), "r"(Ac[s4][2]), "r"(Ac[s4][3]));
                }
                DIAG_C(c_is, tc);
                ++it;
            };
            if constexpr (TOKEN_ONLY) {
                static_assert(NSTAGE1 == 6 && NKT1 % (2 * NSTAGE1) == 0 && NA == 2,
                              "N8 full-K tasks restart at stage0/phase0");
                // Keep the original two-body MMA loop. An explicit circular
                // cursor avoids per-tile division without sixfold code growth.
                int ring_stage = 0, ring_phase = 0;
#pragma unroll 1
                for (int kt = 0; kt < NKT1; kt += 2) {
                    tile_step(kt, A[0], ring_stage, ring_phase,
                              ring_stage == 0 ? NSTAGE1 - 1 : ring_stage - 1);
                    tile_step(kt + 1, A[1], ring_stage + 1, ring_phase, ring_stage);
                    const bool wrap = ring_stage == NSTAGE1 - 2;
                    ring_stage = wrap ? 0 : ring_stage + 2;
                    ring_phase ^= wrap;
                }
            } else {
                for (int kt = kt0; kt < kt0 + nkt; kt += 2) {
                    tile_step(kt, A[0], 0, 0, 0);
                    tile_step(kt + 1, A[NA - 1], 0, 0, 0);
                }
            }
            wg::wait<0>();
            fence_operand(S);
            float2 xl = make_float2(0.f, 0.f);
            if constexpr (XS_STAGE) xl = *reinterpret_cast<const float2*>(smem + OFF_RING + ((it - 1) % NSTAGE1) * FC1_STAGE_BYTES + FC1_TILE_BYTES + 1024 + tig * 8);
            promote(kt0 + nkt - 1, xl);
        }
        if (lane == 0 && (TOKEN_ONLY || !(P.mode & 1))) mbar_arrive(&empty[(it - 1) % NSTAGE1]);
        DIAG_T(tC);

        if constexpr (EXACT_M == 32 && PARTS == 2) {
            // Shorten setup metadata live ranges across the MMA mainloop.
            base_item = reinterpret_cast<volatile int*>(s_misc)[36];
            if constexpr (!TASK_ONLY) split_piece = reinterpret_cast<volatile int*>(s_misc)[37];
            u = base_item / 2; hf = base_item & 1; e = s_union[u];
            if constexpr (TOKEN_ONLY && FC1_PUB_OFFLOAD) {   // factor table staged by producer warp 18 (fc1_pub_worker)
                const int* pub_ = reinterpret_cast<const int*>(smem + OFF_PUB);
                while (ld_acquire_cta_s(pub_ + 1) != (int)epoch) {}
                rs_e = reinterpret_cast<const float2*>(smem + OFF_S2TAB)[u].x;
            } else
            rs_e = __ldg(P.fc1_s2 + e) * 64.0f;
            static_assert(KSPLIT == 1, "M32 light tail uses one guarded N8 panel");
            if constexpr (ONE_WAVE && FMOE_ONEWAVE_KSPLIT) {
                // One-wave K-split: a helper piece publishes its raw fp32 accumulator of task `item` and is done; the owner (k-tiles
                // from 0) adds it before SiLU. Owner and helper threads own identical accumulator fragments, so the partial is stored
                // in THREAD-PRIVATE contiguous slots ([task][consumer thread][8 floats]: one pointer, two float4 per thread) -- the
                // [pos][row] layout with 8 scattered 64-bit addresses per thread pushed the consumer over its register cap.
                const int ksw = reinterpret_cast<volatile int*>(s_misc)[37];
                const int ks_item = ksw & 0x7fff, ks_nkt = ksw >> 16;
                if (ks_nkt < NKT1) {
                    static_assert(NR == 8 && KS_PART_FLOATS == NCONS * 8, "one-wave K-split partial = 8 floats per consumer thread");
                    float4* pb = reinterpret_cast<float4*>(P.part_buf + (size_t)ks_item * KS_PART_FLOATS) + tid * 2;
                    if (ksw & 0x8000) {   // helper piece
                        pb[0] = make_float4(D[0], D[1], D[2], D[3]);
                        pb[1] = make_float4(D[4], D[5], D[6], D[7]);
                        named_bar_sync(1, NCONS);   // every thread's stores precede thread 0's release
                        if (tid == 0) st_release_gpu(&P.part_flags[KS_FLAG_BASE + ks_item], epoch);
                        continue;   // the owner alone performs SiLU / requant / h publication
                    }
                    if (tid == 0) while (ld_acquire_gpu(&P.part_flags[KS_FLAG_BASE + ks_item]) != epoch) {}
                    named_bar_sync(1, NCONS);   // partial visible to every consumer thread (acquire by tid 0 + barrier); L2 loads
                    const float4 a = __ldcg(pb), b = __ldcg(pb + 1);
                    D[0] += a.x; D[1] += a.y; D[2] += a.z; D[3] += a.w;
                    D[4] += b.x; D[5] += b.y; D[6] += b.z; D[7] += b.w;
                }
            }
            if (!TASK_ONLY && split_piece >= 0) {
                const int tail = base_item - 132;
                // [tail][mat][8pos][128row] lies beyond all active h rows;
                // U<=97 requires at most1302528B of the3145728B allocation.
                float* partial = reinterpret_cast<float*>(P.h_buf + (size_t)s_misc[32] * 2 * 32 * 128)
                               + tail * (2 * 8 * 128) + mat * (8 * 128);
                if (split_piece == 1) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int pos = 2 * tig + (i & 1);
                        if (pos < T) partial[pos * 128 + row_local0 + 8 * (i >> 1)] = D[i];
                    }
                    named_bar_sync(1, NCONS);
                    if (tid == 0) st_release_gpu(&P.part_flags[tail], epoch);
                    continue;   // owner alone performs SiLU/requant/h publication
                }
                if (tid == 0) while (ld_acquire_gpu(&P.part_flags[tail]) != epoch) {}
                named_bar_sync(1, NCONS);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int pos = 2 * tig + (i & 1), off = pos * 128 + row_local0 + 8 * (i >> 1);
                    if (pos < T) D[i] += __ldcg(partial + off);
                }
            }
        }

        // ---- K-split: the k-half-1 item publishes its raw fp32 partial [mat][pos][row]; the k-half-0 item adds it ----
        float* pb = P.part_buf + (((size_t)u * 2 + hf) * 2 + mat) * (64 * 128);
        if (KSPLIT == 2 && kh == 1) {
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                if (j < C) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) pb[(8 * j + 2 * tig + (i & 1)) * 128 + row_local0 + 8 * (i >> 1)] = D[4 * j + i];
                }
            }
            named_bar_sync(1, NCONS);   // every thread's stores precede thread 0's cumulative release
            if (tid == 0) st_release_gpu(&P.part_flags[u * 2 + hf], epoch);
#ifdef FMOE_DIAG
            if (tid == 0) { const unsigned long long tD = gtimer(); d_setup += tB - tA; d_tiles += tC - tB; d_epi += tD - tC; ++d_items; }
#endif
            continue;
        }
        if (KSPLIT == 2) {
            if (tid == 0) { while (ld_acquire_gpu(&P.part_flags[u * 2 + hf]) != epoch) {} }
            named_bar_sync(1, NCONS);   // partial visible to every consumer thread (acquire by tid 0 + barrier); L2 loads (no stale L1)
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                if (j < C) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) D[4 * j + i] += __ldcg(pb + (8 * j + 2 * tig + (i & 1)) * 128 + row_local0 + 8 * (i >> 1));
                }
            }
        }

#ifdef FMOE_DIAG
        if (tid == 0) { tE = gtimer(); d_e[0] += tE - tC; }   // epi part 0: metadata re-read + fc1_s2 load
#endif
        // ---- epilogue: the up WGs stage u * s_e in hst[pos][row]; the gate WGs then form h = SiLU(g * s_e) * u in place ----
        if (mat == 1) {
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                if (j < C) {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        hst[(8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1)] = D[4 * j + i] * rs_e;
                }
            }
        }
        named_bar_sync(1, NCONS);
        if (mat == 0) {
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                if (j < C) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float* hp = hst + (8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1);
                        *hp = silu_f(D[4 * j + i] * rs_e) * *hp;
                    }
                }
            }
        }
        named_bar_sync(1, NCONS);
#ifdef FMOE_DIAG
        if (tid == 0) { const unsigned long long t_ = gtimer(); d_e[1] += t_ - tE; tE = t_; }   // epi part 1: stage + SiLU (2 barriers)
#endif
        {   // requant: 8 threads per token (16 columns each), all M_pad tokens in one pass (unrouted -> zeros)
            const int tok = tid >> 3, q = tid & 7;
            const int full_pos = (tok < M) ? s_inv[tok] : -1;
            const int pos = full_pos >= token_base && full_pos < token_base + NT ? full_pos - token_base : -1;
            // Two passes over the smem staging (amax, then reload + pack): keeps 4 instead of 16 h values live through the shuffles
            //.
            const float4* sh = reinterpret_cast<const float4*>(hst + (pos >= 0 ? pos : 0) * HST_STRIDE + q * 16);
            float amax = 0.f;
            if (pos >= 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 a = sh[i];
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            const float hs = amax > 0.f ? amax * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / hs;
            uint32_t w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 a = pos >= 0 ? sh[i] : make_float4(0.f, 0.f, 0.f, 0.f);   // unrouted rows pack zeros
                w[i] = (uint32_t)pack_e4m3x2(a.x * inv, a.y * inv) | ((uint32_t)pack_e4m3x2(a.z * inv, a.w * inv) << 16);
            }
            if (tok < M_pad && (pos >= 0 || (token_base == 0 && (!token_tasks || full_pos < 0)))) {
                // Parallel token tasks must never zero another task's live
                // rows. Chunk0 exclusively clears genuinely unrouted rows.
                uint8_t* dst = P.h_buf + (((size_t)u * 2 + hf) * M_pad + tok) * 128 + ((q ^ (tok & 7)) << 4);
                *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
                if (q == 0) {
                    const size_t ci = ((size_t)u * M_pad + tok) * 2 + hf;
                    const float cs = (pos >= 0) ? hs * slot_weight(s_slot, s_tkw, tok, u) : 0.f;
                    P.cs_buf[ci] = cs;   // preserve the original externally inspected scale plane
                    if constexpr (EXACT_M == 32 && FMOE_CS_PERM) {
                        // Prescaled plane in the true upper half of cs_buf (room for all MAXU experts, so the FC2 retire has one
                        // path), permuted per expert as [hf][tig 4][j 4][e 2]: FC2 lane (tig) reads its 8 scales of k-block hf
                        // as two float4.
                        const float rs2 = (TOKEN_ONLY && FC1_PUB_OFFLOAD) ? reinterpret_cast<const float2*>(smem + OFF_S2TAB)[u].y
                                                                                : __fmul_rn(__ldg(P.fc2_s2 + e), 64.0f);
                        const int perm = hf * 32 + ((tok >> 1) & 3) * 8 + (tok >> 3) * 2 + (tok & 1);
                        P.cs_buf[(size_t)MAXU * M_pad * 2 + (size_t)u * 64 + perm] = __fmul_rn(cs, rs2);
                    } else if constexpr (FC2_PRESCALE<EXACT_M>) {
                        if (s_misc[32] <= MAXU / 2) {
                            const float rs2 = __fmul_rn(__ldg(P.fc2_s2 + e), 64.0f);
                            P.cs_buf[(size_t)(MAXU / 2) * M_pad * 2 + ci] = __fmul_rn(cs, rs2);
                        }
                    }
                }
            }
        }
        // Generic-proxy stores -> visible to the fc2 producer's cp.async.bulk (async proxy) after its acquire + proxy fence;
        // bar.sync orders every thread's stores before thread 0's cumulative st.release.gpu.
#ifdef FMOE_DIAG
        if (tid == 0) { const unsigned long long t_ = gtimer(); d_e[2] += t_ - tE; tE = t_; }   // epi part 2: requant + h/cs stores (tid 0's view)
#endif
        fence_proxy_async_global();
        named_bar_sync(1, NCONS);
#ifdef FMOE_DIAG
        if (tid == 0) { const unsigned long long t_ = gtimer(); d_e[3] += t_ - tE; tE = t_; }   // epi part 3: proxy fence + barrier
#endif
        if (tid == 0) {
            if constexpr (TOKEN_ONLY && FC1_PUB_OFFLOAD) {   // hand the publish to producer warp 18 (item n of this CTA)
                int* pub_ = reinterpret_cast<int*>(smem + OFF_PUB);
                pub_[4 + (n & 7)] = (u << 8) | (hf << 4) | (token_base / NT);
                st_release_cta_s(pub_, n + 1);
            } else
            if (token_tasks)
                publish_m32_chunk(P.h_flags, u, hf, token_base / NT, epoch);
            else if (!CHUNKED || token_base + NT >= T)
                st_release_gpu(&P.h_flags[u * 2 + hf], epoch);
        }
#ifdef FMOE_DIAG
        if (tid == 0) { const unsigned long long tD = gtimer(); d_setup += tB - tA; d_tiles += tC - tB; d_epi += tD - tC; ++d_items; }
#endif
        }   // routed-token chunks; all chunks finish before publishing the expert-half flag
    }
    if (tid == 0) STAMP(P, 1);   // fc1 CTA: all items done
#ifdef FMOE_DIAG
    if (tid == 0 && P.dbg) {   // epilogue/setup breakdown in the (DIAG-overwritten) prologue stamp slots 6..10
        P.dbg[blockIdx.x * NSTAMP + 6] = d_e[0]; P.dbg[blockIdx.x * NSTAMP + 7] = d_e[1]; P.dbg[blockIdx.x * NSTAMP + 8] = d_e[2];
        P.dbg[blockIdx.x * NSTAMP + 9] = d_e[3]; P.dbg[blockIdx.x * NSTAMP + 10] = d_e[4];
    }
    if (tid == 0 && P.dbg) { P.dbg[blockIdx.x * NSTAMP + 16] = d_setup; P.dbg[blockIdx.x * NSTAMP + 17] = d_tiles; P.dbg[blockIdx.x * NSTAMP + 18] = d_epi; P.dbg[blockIdx.x * NSTAMP + 19] = d_items;
                            P.dbg[blockIdx.x * NSTAMP + 20] = c_wait; P.dbg[blockIdx.x * NSTAMP + 21] = c_dq; P.dbg[blockIdx.x * NSTAMP + 22] = c_mw; P.dbg[blockIdx.x * NSTAMP + 23] = c_pi; P.dbg[blockIdx.x * NSTAMP + 24] = c_is; }
#endif
#undef DIAG_T
#undef DIAG_C
}

// ======================= fc2 role =======================
// fc2 ring geometry: stage = [h tile NT x 256 B (swizzled, 1024-aligned)][cs NT x 8 B (512 B slot)][weights 17920 B]
template <int NT, int EXACT_M = 0> struct Fc2Geom {
    static constexpr int H_OFF = 0, CS_OFF = NT * 256, W_OFF = NT * 256 + 512;
    static constexpr int STAGE = ((W_OFF + FC2_W_BYTES) + 1023) & ~1023;
    // Exact M32: out_s only needs its 32 token rows (16896 B, not MAXM's 33792), so the ring
    // extends to OFF_TOK - 16896 and holds a 7th 26-KB stage. With all 132 CTAs streaming FC2 the aggregate delivery
    // is ~3.5 TB/s at ~2.4 TB/s of HBM (h/cs hit L2): a closed loop of 132 x NST stages in flight, i.e. paced by
    // in-flight bytes / latency, so a deeper ring buys throughput.
    static constexpr int OUTS = (FMOE_FC2_RING7 && EXACT_M == 32 && NT == 32) ? OFF_TOK - 32 * HST_STRIDE * 4 : OFF_OUTS;
    static constexpr int NST = (OUTS / STAGE) < 7 ? (OUTS / STAGE) : 7;   // 4 (NT=64) .. 6 (NT<=32) stages, 7 for exact M32
    static_assert(NST >= 2 && NST <= 8, "fc2 ring depth");
    static_assert(NST * STAGE <= OUTS && OUTS % 16 == 0 && OUTS + NT * HST_STRIDE * 4 <= OFF_TOK, "fc2 ring and out_s fit below the token lists");
};

// Exact-M32 FC2 scale ring (FMOE_CS_PERM): entry l & 7 holds stage l's 64 prescaled, lane-permuted scales (256 B). It lives in
// the dead routing-mask table, is NOT covered by the stage's empty barrier (entry l & 7 is rewritten only at stage l + 8, after the
// retire of stage l + 2 released stage l + 2, i.e. long after retire(l) read it) but IS covered by the stage's full barrier
// (the producer's expect_tx includes its bytes), so the consumer may release the stage before reading the scales.
constexpr int FC2_CS_RING = 8;
__device__ __forceinline__ float* fc2_cs_ring(uint8_t* smem) { return reinterpret_cast<float*>(smem + OFF_TAB + TAB_MASK); }
static_assert(TAB_MASK + FC2_CS_RING * 64 * 4 <= TAB_SLOT + 32 * TOPK * 4, "FC2 scale ring fits before the M32 chunk counts");

template <int NT, int PARTS, int EXACT_M>
__device__ __forceinline__ void fc2_producer(const Params& P, unsigned epoch, uint8_t* smem, uint64_t* full, uint64_t* empty) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if constexpr (fc2_prefill<EXACT_M, PARTS>()) {
        // Gate on the consumers' h publish (bar.arrive after fc1_consumer): the FC1 ring, HST and xp_s are dead once all 16 consumer warps
        // have arrived, and the FC2 barrier objects (full/empty here = OFF_BAR2) are fresh, so the first-lap empty waits pass at once.
        // Not earlier: FC2 weight boxes issued ahead of the publish queued 132 x 85 KB of TMA under the epilogue's global chain and
        // delayed fc1_end by +1.7..2.0 us on every wave-1 CTA (probes).
        if (warp <= PROD_WARP + 2) {
            named_bar_sync(FC2_START_BAR, FC2_START_THREADS);
            if constexpr (FMOE_FC2_PREFILL_NOTAIL_DELAY_NS > 0) {
                // Early leavers (see the knob) idle for the old join-chain time. bar.sync blocks lazily (BAR.SYNC.DEFER_BLOCKING): the
                // timer is read only inside a branch on a shared load, which does wait for the barrier.
                const int* mi = reinterpret_cast<const int*>(smem + OFF_TOK) + 3 * 64;
                const bool one_wave = mi[39] && mi[58];
                const unsigned t0 = (unsigned)mi[63];   // CTA start, low 32 bits of %globaltimer (kernel body)
                unsigned now = 0;
                if (t0 != 0xFFFFFFFFu) now = gtimer32();
                if (!one_wave && now - t0 < 52000u) {
                    const unsigned until = now + FMOE_FC2_PREFILL_NOTAIL_DELAY_NS;
                    while ((int)(gtimer32() - until) < 0) __nanosleep(100);
                }
            }
        }
    }
    int tile, part, parts;   // rows [tile*128, +128); this CTA handles experts u % parts == part
    fc2_map<PARTS>((int)blockIdx.x, EXACT_M ? 132 : (int)gridDim.x, tile, part, parts);
    const int* s_union = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_UNION);
    const int* misc = reinterpret_cast<const int*>(smem + OFF_TOK) + 3 * 64;
    const int U = misc[32];
    int expert_begin, expert_end, expert_stride;
    fc2_expert_range<EXACT_M, PARTS>(P, U, part, parts, expert_begin, expert_end, expert_stride);
    fc2_helper_range<EXACT_M, PARTS>(P, U, misc, tile, part, parts, expert_begin, expert_end, expert_stride);
    const bool gp2 = m32_gp2_active<EXACT_M, PARTS>(P, U, misc);
    const bool gp2_helper = gp2 && (int)blockIdx.x >= 96;   // FMOE_GP2 helper: two tiles, m32_gp2_stage walk
    const bool dual_helper = !gp2 && m32_fc2_dual_helper<EXACT_M, PARTS>(P, U, misc);
    const bool token_tasks = m32_token_tasks<EXACT_M, PARTS>(P, misc);
    const int* chunk_counts = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_SLOT) + 32 * TOPK;
    static_assert(TAB_MASK + (NCONS / 32) * 32 * sizeof(float) <= TAB_SLOT + 32 * TOPK * sizeof(int),
                  "M32 chunk counts outlive FC2 warp-scale scratch");
    using G = Fc2Geom<NT, EXACT_M>;
    constexpr int H_OFF = G::H_OFF, CS_OFF = G::CS_OFF, W_OFF = G::W_OFF, STAGE = G::STAGE, NST = G::NST;
    // Exact-shape operators let weights run ahead independently of FC1 completion, as in the GLM seq4 producer/gather
    // split. Both producers wait for the same empty phase and arrive once with their own disjoint byte count. The
    // full barrier completes only after BOTH arrivals and BOTH transfers, preserving the consumer's existing contract.
    if constexpr (EXACT_M == 24) {
        if (warp == PROD_WARP + 1 && lane == 0) {
            tma_acquire_map(P.tmaps + 2); tma_acquire_map(P.tmaps + 3);
            for (int l = 0, u = expert_begin; u < expert_end; ++l, u += expert_stride) {
                const int s = l % NST, ph = (l / NST) & 1, e = s_union[u];
                mbar_wait(&empty[s], ph ^ 1);
                uint8_t* stage = smem + OFF_RING + s * STAGE;
                mbar_arrive_expect_tx(&full[s], FC2_W_BYTES);
                tma_load_3d(stage + W_OFF, P.tmaps + 2, 0, tile * 2, e * (INTER / 32), &full[s]);
                tma_load_2d(stage + W_OFF + FC2_TILE_W_BYTES, P.tmaps + 3, tile * 128, e * (INTER / 32), &full[s]);
            }
        }
    }
    constexpr bool SPLIT_PROD = fc2_split_producer<EXACT_M, PARTS>();
    constexpr bool LOCKSTEP = fc2_lockstep<EXACT_M, PARTS>();
    if constexpr (SPLIT_PROD) {
        // Weight / offset producer warps. The cycle probe of the single-lane producer showed ~640 cycles
        // per stage in the four copy issues, ~285 in the batched poll and ~265 of walk overhead: ~1300 of the ~1440-cycle
        // stage, while the consumers waited for data 17-24% of the time -- the lane paced FC2. These two warps walk the same
        // expert sequence, wait on the same empty phase and each arrive with their own byte count; they need no readiness
        // (the weights do not depend on FC1), so the weight boxes run ahead of the h copies up to the ring depth.
        // The role is a compile-time constant of each loop (a runtime role spilled to local memory and was reloaded once per
        // stage -- an L1 miss after every CCTL.IVALL of warp 16's poll).
        auto walk = [&](auto role_t) {
            constexpr int ROLE = decltype(role_t)::value;   // 1: weight boxes (map 2), 2: offset boxes (map 3)
            if (!FMOE_FC2_ACQUIRE_EARLY || P.pre_routed) tma_acquire_map(P.tmaps + 1 + ROLE);
            int L = 0, l = 0;
            if (gp2_helper) {   // FMOE_GP2 helper: pair 0 -> tile h, pair 1 -> tile 36+h%12 then tile h (m32_gp2_stage)
                const int h = (int)blockIdx.x - 96, nq = misc[GP2_MISC + 2], q0 = misc[GP2_MISC + 3];
#pragma unroll 1
                for (; l < expert_end; ++l) {
                    int wt, idx; m32_gp2_stage(h, nq, q0, l, wt, idx);
                    const int e = s_union[idx];
                    const int s = l % NST, ph = (l / NST) & 1;
                    mbar_wait(&empty[s], ph ^ 1);
                    uint8_t* stage = smem + OFF_RING + s * STAGE;
                    if constexpr (ROLE == 1) {
                        mbar_arrive_expect_tx(&full[s], FC2_TILE_W_BYTES);
                        tma_load_3d(stage + W_OFF, P.tmaps + 2, 0, wt * 2, e * (INTER / 32), &full[s]);
                    } else {
                        mbar_arrive_expect_tx(&full[s], FC2_TILE_S_BYTES);
                        tma_load_2d(stage + W_OFF + FC2_TILE_W_BYTES, P.tmaps + 3, wt * 128, e * (INTER / 32), &full[s]);
                    }
                }
                return;   // helpers merge nothing
            }
            // Lockstep consumers: a dual helper streams its first tile for every expert, then its second (two passes), and
            // every pass has an even stage count -- an odd count gets one more stage of its last expert whose scales warp 16
            // zeroes (the consumers' two-stage unroll then needs no conditional around a wgmma group, which ptxas serialized).
            // Otherwise the pair consumers take alternating tiles of the same expert (two adjacent stages).
            const int n_per = (expert_end - expert_begin + expert_stride - 1) / expert_stride;
            const int pad_end = expert_end + ((LOCKSTEP && (n_per & 1)) ? expert_stride : 0);
#pragma unroll 1
            for (int pass = 0; pass < ((LOCKSTEP && dual_helper) ? 2 : 1); ++pass)
            for (int idx0 = expert_begin; idx0 < pad_end;
                 ++l, idx0 += (!LOCKSTEP && dual_helper && (l & 1)) ? 0 : expert_stride) {
                // A lockstep padding stage repeats the walk's LAST expert (idx0 - stride). NOT min(idx0, end - stride): with the
                // one-wave owners' stride-2 walk that clamp replaced the final expert's weight tile (103-case matrix:
                // 4 one-wave cases failed the own-h reference).
                const int idx = idx0 >= expert_end ? idx0 - expert_stride : idx0;
                const int u = fc2_union_index<EXACT_M, PARTS>(misc, idx);
                const int e = s_union[u];
                const int s = l % NST, ph = (l / NST) & 1;
                if constexpr (FMOE_FC2_L2_PREFETCH > 0 && !LOCKSTEP) {
                    // The ring is almost always full (this lane waits for empty ~68% of each stage) while the consumers still wait
                    // for landed data ~10-25% of the time: the box issued NST stages ahead has not arrived when its stage comes up.
                    // Warm L2 with the boxes of the expert FMOE_FC2_L2_PREFETCH stages ahead (both tiles of a dual helper) before
                    // blocking on the slot, so the later smem TMA is an L2 hit. A miss on the prefetch costs nothing but HBM headroom.
                    if (!dual_helper || !(l & 1)) {
                        const int ip = idx + (dual_helper ? FMOE_FC2_L2_PREFETCH / 2 : FMOE_FC2_L2_PREFETCH) * expert_stride;
                        if (ip < expert_end) {
                            const int ep = s_union[fc2_union_index<EXACT_M, PARTS>(misc, ip)];
                            if constexpr (ROLE == 1) {
                                tma_prefetch_3d(P.tmaps + 2, 0, tile * 2, ep * (INTER / 32));
                                if (dual_helper) tma_prefetch_3d(P.tmaps + 2, 0, (tile + 36) * 2, ep * (INTER / 32));
                            } else {
                                tma_prefetch_2d(P.tmaps + 3, tile * 128, ep * (INTER / 32));
                                if (dual_helper) tma_prefetch_2d(P.tmaps + 3, (tile + 36) * 128, ep * (INTER / 32));
                            }
                        }
                    }
                }
                mbar_wait(&empty[s], ph ^ 1);
                uint8_t* stage = smem + OFF_RING + s * STAGE;
                const int weight_tile = tile + (LOCKSTEP ? pass * 36 : (dual_helper ? (l & 1) * 36 : 0));
                if constexpr (ROLE == 1) {
                    mbar_arrive_expect_tx(&full[s], FC2_TILE_W_BYTES);
                    tma_load_3d(stage + W_OFF, P.tmaps + 2, 0, weight_tile * 2, e * (INTER / 32), &full[s]);
                } else {
                    mbar_arrive_expect_tx(&full[s], FC2_TILE_S_BYTES);
                    tma_load_2d(stage + W_OFF + FC2_TILE_W_BYTES, P.tmaps + 3, weight_tile * 128, e * (INTER / 32), &full[s]);
                }
                L = l + 1;
            }
            // The B-owner's merge pseudo-stage (below) carries only warp 16's 16-KB copy: arrive once without bytes so the
            // count-3 barrier can complete.
            if (FMOE_MERGE_PREFETCH && (int)blockIdx.x < 96 && part == 1 && m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc)) {
                const int nmerge = (gp2 && tile >= 36) ? 3 : 1;   // FMOE_GP2: tiles 36..47 merge three quarter partials
#pragma unroll 1
                for (int k = 0; k < nmerge; ++k) {
                    const int s = (L + k) % NST, ph = ((L + k) / NST) & 1;
                    mbar_wait(&empty[s], ph ^ 1);
                    mbar_arrive(&full[s]);
                }
            }
        };
        if (lane == 0) {
            if (warp == PROD_WARP + 1) walk(std::integral_constant<int, 1>{});
            else if (warp == PROD_WARP + 2) walk(std::integral_constant<int, 2>{});
        }
    }
    {
        if (warp == PROD_WARP && lane == 0) {
            // Acquire + prefetch of maps 2/3 by this lane: immediately before the first use (earlier schedule), or already done
            // by this lane inside the prologue's message-wait window (FMOE_FC2_ACQUIRE_EARLY; the pre-routed legacy
            // entry has no prologue and always acquires here). With the split producer this lane issues no tensor copies.
            if (!SPLIT_PROD && (!FMOE_FC2_ACQUIRE_EARLY || P.pre_routed)) { tma_acquire_map(P.tmaps + 2); tma_acquire_map(P.tmaps + 3); }
            unsigned ready = 0; int win = 0;   // bit i: expert win+i confirmed ready (both halves)
            // Dual helpers give the two WG pairs separate output tiles.
            // Repeat each expert for two adjacent stages in the SAME ring.
            // Mixed-width loads: hot experts' N16 tasks end ~4-9 us after the cold first wave, so every role streams
            // its cold experts first and its hot experts last (a data-dependent, deterministic order; the consumer
            // only needs the stage count, and M32 token tasks always use the prescaled cs plane).
#if FMOE_BATCH_FENCE
            // Batched ready check for token-task loads. the earlier schedule paid one L2 poll round trip +
            // fence.acquire.gpu (= CCTL.IVALL) + fence.proxy.async per expert on this single lane, which paces FC2 once
            // all 132 CTAs stream (B-owners 1.0-1.08 us/expert vs A 0.85). Here the ready words of this expert and the
            // next BW-1 of this CTA's walk are read with BW back-to-back relaxed loads (ONE round trip) followed by ONE
            // fence pair; confirmed experts are remembered in `ready` (bit = idx - win; a dual helper's two stages of
            // one expert share the bit) and skip the chain when reached. Only a matching read acquires: the fence orders
            // the confirmed experts' chunk payloads before the bulk copies issued below, as the per-expert form did.
            // The token and legacy walks are separate loops on purpose: sharing one loop let ptxas hoist the legacy
            // path's 8 flag-pair addresses to the loop top (~45 instructions per expert on BOTH paths) and spill the
            // dual-helper shift amount -- one LDL per expert that misses L1 after every CCTL.IVALL.
            const uint64_t h_policy = FMOE_H_EVICT_LAST ? l2_policy_evict_last() : 0ull;
            auto issue = [&](int l, int u, int e, bool dummy) {
                const int s = l % NST, ph = (l / NST) & 1;
                mbar_wait(&empty[s], ph ^ 1);
                uint8_t* stage = smem + OFF_RING + s * STAGE;
                mbar_arrive_expect_tx(&full[s], ((EXACT_M == 24 || SPLIT_PROD) ? 0 : FC2_W_BYTES) + NT * 256 + (dummy ? 0 : NT * 8));
                const int scale_u = (EXACT_M == 32 && FMOE_CS_PERM) ? MAXU + u : u + ((FC2_PRESCALE<EXACT_M> && U <= MAXU / 2) ? MAXU / 2 : 0);
                uint8_t* cs_dst = (EXACT_M == 32 && FMOE_CS_PERM) ? reinterpret_cast<uint8_t*>(fc2_cs_ring(smem) + (l & (FC2_CS_RING - 1)) * 64) : stage + CS_OFF;
                if (dummy) {   // lockstep padding stage: real h (finite fp8) but zero scales, so it adds nothing
                    uint4* z = reinterpret_cast<uint4*>(cs_dst);
#pragma unroll
                    for (int i = 0; i < NT * 8 / 16; ++i) z[i] = make_uint4(0u, 0u, 0u, 0u);
                    bulk_g2s(stage + H_OFF, P.h_buf + (size_t)u * 2 * NT * 128, NT * 256, &full[s]);
                } else if (FMOE_H_EVICT_LAST) {
                    bulk_g2s_hint(stage + H_OFF, P.h_buf + (size_t)u * 2 * NT * 128, NT * 256, &full[s], h_policy);
                    bulk_g2s_hint(cs_dst, P.cs_buf + (size_t)scale_u * NT * 2, NT * 8, &full[s], h_policy);
                } else {
                    bulk_g2s(stage + H_OFF, P.h_buf + (size_t)u * 2 * NT * 128, NT * 256, &full[s]);
                    bulk_g2s(cs_dst, P.cs_buf + (size_t)scale_u * NT * 2, NT * 8, &full[s]);
                }
                // the tile's 128 rows = blocks tile*2, tile*2+1 of the expert's 8 k32 slices: one 16-KB TMA box ([slice][rh][1 KB]) + one 1-KB offset box
                if constexpr (EXACT_M != 24 && !SPLIT_PROD) {
                    const int weight_tile = tile + (dual_helper ? (l & 1) * 36 : 0);
                    tma_load_3d(stage + W_OFF, P.tmaps + 2, 0, weight_tile * 2, e * (INTER / 32), &full[s]);
                    tma_load_2d(stage + W_OFF + FC2_TILE_W_BYTES, P.tmaps + 3, weight_tile * 128, e * (INTER / 32), &full[s]);
                }
            };
            if (gp2_helper) {
                // FMOE_GP2 helper walk: stage l -> union slot (m32_gp2_stage); readiness batched over the next BW stages (consecutive
                // stage numbers -> a plain 32-bit window), one fence pair per round trip as in the token-task walk below. The expected
                // ready words are recomputed after the fence (fewer live registers in this 64-register warp).
                constexpr int BW = FMOE_GP2_BW;
                const int nq = misc[GP2_MISC + 2], q0 = misc[GP2_MISC + 3], L = expert_end;
                auto slot_of = [&](int l) { const int p = l & 1, j = l >> 1; return j < nq ? (p ? q0 + j : j) : 2 * j - nq + p; };
                unsigned rdy = 0u; int rwin = 0;
#pragma unroll 1
                for (int l = 0; l < L; ++l) {
                    const int u = slot_of(l);
                    if (l >= rwin + 32) { rwin = l; rdy = 0u; }
                    while (!((rdy >> (l - rwin)) & 1u)) {
                        uint64_t f[BW];
#pragma unroll
                        for (int k = 0; k < BW; ++k)
                            f[k] = ld_relaxed_gpu_u64(reinterpret_cast<const uint64_t*>(P.h_flags + MAXU * 2 + slot_of(min(l + k, L - 1)) * 8));
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        fence_proxy_async_global();
#pragma unroll
                        for (int k = 0; k < BW; ++k) {
                            const int lk = l + k;
                            if (lk < L && lk < rwin + 32) {
                                const int uk = slot_of(lk);
                                if (f[k] == ((uint64_t(epoch) << 32) | (((1u << chunk_counts[uk]) - 1u) * 0x11u))) rdy |= 1u << (lk - rwin);
                            }
                        }
                    }
                    issue(l, u, s_union[u], false);
                }
            } else
            if (token_tasks) {
                constexpr int BW = FMOE_BATCH_FENCE;   // experts per round trip (register-limited: the producer warp has 64)
#if FMOE_MIXED_WIDTH
                // Mixed widths: the hot experts' single N16 tasks end ~4.5 us after the cold N8 first
                // wave, so each role walks its range twice -- pass 0 streams the cold experts, pass 1 the hot ones (bit 8 of
                // the chunk count; a deterministic order, and M32 token tasks always use the prescaled cs plane so the
                // consumer needs only the stage count). A two-pass walk sat on the per-expert poll chain; here the
                // skipped expert costs two smem reads and the polls stay batched. The ready window restarts per pass.
                int l = 0;
#pragma unroll 1
                for (int pass = 0; pass < 2; ++pass) {
                    ready = 0; win = expert_begin;
                    for (int idx = expert_begin; idx < expert_end; idx += expert_stride) {
                        const int u = fc2_union_index<EXACT_M, PARTS>(misc, idx);
                        if (((chunk_counts[u] >> 8) & 1) != pass) continue;
                        const int e = s_union[u];
                        if (idx >= win + 32) { win = idx; ready = 0; }
                        while (!(ready & (1u << (idx - win)))) {
                            uint64_t f[BW]; unsigned lo[BW];
#pragma unroll
                            for (int k = 0; k < BW; ++k) {
                                const int ik = idx + k * expert_stride;
                                const int uk = k == 0 ? u : fc2_union_index<EXACT_M, PARTS>(misc, ik < expert_end ? ik : idx);
                                lo[k] = ((1u << (chunk_counts[uk] & 0xff)) - 1u) * 0x11u;
                                f[k] = ld_relaxed_gpu_u64(reinterpret_cast<const uint64_t*>(P.h_flags + MAXU * 2 + uk * 8));
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            fence_proxy_async_global();
#pragma unroll
                            for (int k = 0; k < BW; ++k) {
                                const int ik = idx + k * expert_stride;
                                if (ik < expert_end && ik < win + 32 && f[k] == ((uint64_t(epoch) << 32) | lo[k]))
                                    ready |= 1u << (ik - win);
                            }
                        }
                        issue(l, u, e, false); ++l;
                        if (dual_helper) { issue(l, u, e, false); ++l; }   // the pair's second tile, same expert (weight_tile + 36)
                    }
                }
#else
                int l = 0;
                const int n_per = (expert_end - expert_begin + expert_stride - 1) / expert_stride;
                const int pad_end = expert_end + ((LOCKSTEP && (n_per & 1)) ? expert_stride : 0);   // see the weight walk above
#pragma unroll 1
                for (int pass = 0; pass < ((LOCKSTEP && dual_helper) ? 2 : 1); ++pass)
                for (int idx0 = expert_begin; idx0 < pad_end;
                     ++l, idx0 += (!LOCKSTEP && dual_helper && (l & 1)) ? 0 : expert_stride) {
                    const bool dummy = idx0 >= expert_end;
                    const int idx = dummy ? idx0 - expert_stride : idx0;
                    const int u = fc2_union_index<EXACT_M, PARTS>(misc, idx);
                    const int e = s_union[u];
                    if (idx >= win + 32 || idx < win) { win = idx; ready = 0; }   // window restart (a lockstep dual helper's second pass walks back)
                    while (!(ready & (1u << (idx - win)))) {
                        uint64_t f[BW]; unsigned lo[BW];
#pragma unroll
                        for (int k = 0; k < BW; ++k) {
                            const int ik = idx + k * expert_stride;
                            const int uk = k == 0 ? u : fc2_union_index<EXACT_M, PARTS>(misc, ik < expert_end ? ik : idx);
                            lo[k] = ((1u << chunk_counts[uk]) - 1u) * 0x11u;
                            f[k] = ld_relaxed_gpu_u64(reinterpret_cast<const uint64_t*>(P.h_flags + MAXU * 2 + uk * 8));
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        fence_proxy_async_global();
#pragma unroll
                        for (int k = 0; k < BW; ++k) {
                            const int ik = idx + k * expert_stride;
                            if (ik < expert_end && ik < win + 32 && f[k] == ((uint64_t(epoch) << 32) | lo[k]))
                                ready |= 1u << (ik - win);
                        }
                    }
                    issue(l, u, e, dummy);
                }
#endif
            } else {
                int l = 0;
                const int n_per = (expert_end - expert_begin + expert_stride - 1) / expert_stride;
                const int pad_end = expert_end + ((LOCKSTEP && (n_per & 1)) ? expert_stride : 0);
#pragma unroll 1
                for (int pass = 0; pass < ((LOCKSTEP && dual_helper) ? 2 : 1); ++pass)
                for (int idx0 = expert_begin; idx0 < pad_end;
                     ++l, idx0 += (!LOCKSTEP && dual_helper && (l & 1)) ? 0 : expert_stride) {
                    const bool dummy = idx0 >= expert_end;
                    const int idx = dummy ? idx0 - expert_stride : idx0;
                    const int u = fc2_union_index<EXACT_M, PARTS>(misc, idx);
                    const int e = s_union[u];
                    while (true) {   // poll up to 8 experts' flag pairs per round trip: 16 relaxed loads, then ONE acquire
                        if (u >= win + 32 || u < win) { win = u; ready = 0; }   // fence + ONE proxy fence for the whole batch; window restart on a second pass
                        if (ready & (1u << (u - win))) break;
                        unsigned f[16];
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int ue = min(u + k, U - 1);  // codespell:ignore
                            f[2 * k] = ld_relaxed_gpu(&P.h_flags[ue * 2]);  // codespell:ignore
                            f[2 * k + 1] = ld_relaxed_gpu(&P.h_flags[ue * 2 + 1]);  // codespell:ignore
                        }
                        // acquire: h data of the confirmed experts is visible (generic proxy) ... and to the bulk-copy
                        // (async) proxy. This lane only reads here, so the acquire-only fence of the token path (one
                        // CCTL.IVALL) replaces the earlier schedule's fence.acq_rel.gpu (MEMBAR.ALL.GPU + CCTL.IVALL).
                        if (FMOE_LEGACY_ACQUIRE) asm volatile("fence.acquire.gpu;" ::: "memory"); else fence_acq_rel_gpu();
                        fence_proxy_async_global();
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            if (u + k < U && u + k < win + 32 && f[2 * k] == epoch && f[2 * k + 1] == epoch) ready |= 1u << (u + k - win);
                    }
                    issue(l, u, e, dummy);
                }
            }
            if constexpr (EXACT_M == 32 && PARTS == 2) {
                // B-owner merge prefetch: the helper partial used to be fetched by the consumers after
                // their last expert (flag wait + 16 KB of L2 reads, 1.1-1.6 us on every B's critical path
                // B - H = +1.1..1.6). This lane is idle once the last stage is issued, ~NST stages before the consumers
                // finish: it treats the partial as one more ring stage L -- waits for slot L % NST to free up, acquires
                // the helper's flag, and bulk-copies the 16 KB into that slot behind the same full barrier the consumers
                // already know how to wait on (parity of stage L). The epilogue then adds from smem in the same order as
                // before (out_s = D0 + D1, += partial): bitwise-identical to the flag-wait form.
                if (FMOE_MERGE_PREFETCH && (int)blockIdx.x < 96 && part == 1 && m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc)) {
                    static_assert(NT * 128 * 4 <= STAGE, "helper partial fits one ring stage");
                    int L = (expert_end - expert_begin + expert_stride - 1) / expert_stride;   // B is never a dual helper
                    if (LOCKSTEP) L += L & 1;   // lockstep padding stage
                    const int nmerge = (gp2 && tile >= 36) ? 3 : 1;   // FMOE_GP2: tiles 36..47 take three quarter partials (slots 36+h, h = g, 12+g, 24+g)
#pragma unroll 1
                    for (int k = 0; k < nmerge; ++k) {
                        const int slot = (gp2 && tile >= 36) ? tile + 12 * k : tile;
                        const int s = (L + k) % NST, ph = ((L + k) / NST) & 1;
                        mbar_wait(&empty[s], ph ^ 1);
                        while (ld_acquire_gpu(&P.part_flags[64 + slot]) != epoch) {}
                        fence_proxy_async_global();
                        const uint8_t* partial = P.h_buf + ((size_t)U + 64) * 2 * NT * 128 + (size_t)slot * NT * 128 * 4;
                        mbar_arrive_expect_tx(&full[s], NT * 128 * 4);
                        bulk_g2s(smem + OFF_RING + s * STAGE, partial, NT * 128 * 4, &full[s]);
                    }
                }
            }
#else
#if FMOE_MIXED_WIDTH
            // Without mixed widths this is the plain range walk: the two-pass form's extra per-expert index/flag reads
            // on this single lane cost ~2 us on every two-wave case -- the producer
            // lane, not the consumer pairs, paces FC2 once all 132 CTAs stream.
            int next_idx = expert_begin, pass = 0;
            auto next_expert = [&]() -> int {
                if constexpr (!FMOE_MIXED_WIDTH) {
                    if (next_idx >= expert_end) return -1;
                    const int cur = next_idx;
                    next_idx += expert_stride;
                    return cur;
                }
                while (true) {
                    if (next_idx >= expert_end) { if (pass) return -1; pass = 1; next_idx = expert_begin; continue; }
                    const int cur = next_idx;
                    next_idx += expert_stride;
                    const int hot = token_tasks ? ((chunk_counts[fc2_union_index<EXACT_M, PARTS>(misc, cur)] >> 8) & 1) : 0;
                    if (hot == pass) return cur;
                }
            };
            int idx = -1;
            for (int l = 0;; ++l) {
                if (!(dual_helper && (l & 1))) { idx = next_expert(); if (idx < 0) break; }
#else
            for (int l = 0, idx = expert_begin; idx < expert_end;
                 ++l, idx += (dual_helper && (l & 1)) ? 0 : expert_stride) {
#endif
                const int u = fc2_union_index<EXACT_M, PARTS>(misc, idx);
                const int e = s_union[u];
                while (true) {   // poll up to 8 experts' flag pairs per round trip: 16 relaxed loads, then ONE acquire
                    if (token_tasks) {
                        // NVFP4-style single arrival mask, extended with a
                        // full epoch so changed-input Graph replays cannot
                        // consume a previous launch's completed chunks.
                        const int nc = FMOE_MIXED_WIDTH ? (chunk_counts[u] & 0xff) : chunk_counts[u];
                        const uint64_t expected = (uint64_t(epoch) << 32) | (((1u << nc) - 1u) * 0x11u);
                        const auto* p = reinterpret_cast<const uint64_t*>(P.h_flags + MAXU * 2 + u * 8);
                        // Only the successful read acquires the published
                        // chunk payloads; failed polls never consume data.
                        while (ld_relaxed_gpu_u64(p) != expected) {}
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        fence_proxy_async_global();
                        break;
                    }
                    if (u >= win + 32) { win = u; ready = 0; }   // fence + ONE proxy fence for the whole batch
                    if (ready & (1u << (u - win))) break;
                    unsigned f[16];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int ue = min(u + k, U - 1);  // codespell:ignore
                        f[2 * k] = ld_relaxed_gpu(&P.h_flags[ue * 2]);  // codespell:ignore
                        f[2 * k + 1] = ld_relaxed_gpu(&P.h_flags[ue * 2 + 1]);  // codespell:ignore
                    }
                    fence_acq_rel_gpu();          // acquire: h data of the confirmed experts is visible (generic proxy)
                    fence_proxy_async_global();   // ... and to the bulk-copy (async) proxy
#pragma unroll
                    for (int k = 0; k < 8; ++k)
                        if (u + k < U && u + k < win + 32 && f[2 * k] == epoch && f[2 * k + 1] == epoch) ready |= 1u << (u + k - win);
                }
                const int s = l % NST, ph = (l / NST) & 1;
                mbar_wait(&empty[s], ph ^ 1);
                uint8_t* stage = smem + OFF_RING + s * STAGE;
                mbar_arrive_expect_tx(&full[s], (EXACT_M == 24 ? 0 : FC2_W_BYTES) + NT * 256 + NT * 8);
                bulk_g2s(stage + H_OFF, P.h_buf + (size_t)u * 2 * NT * 128, NT * 256, &full[s]);
                const int scale_u = (EXACT_M == 32 && FMOE_CS_PERM) ? MAXU + u : u + ((FC2_PRESCALE<EXACT_M> && U <= MAXU / 2) ? MAXU / 2 : 0);
                uint8_t* cs_dst = (EXACT_M == 32 && FMOE_CS_PERM) ? reinterpret_cast<uint8_t*>(fc2_cs_ring(smem) + (l & (FC2_CS_RING - 1)) * 64) : stage + CS_OFF;
                bulk_g2s(cs_dst, P.cs_buf + (size_t)scale_u * NT * 2, NT * 8, &full[s]);
                // the tile's 128 rows = blocks tile*2, tile*2+1 of the expert's 8 k32 slices: one 16-KB TMA box ([slice][rh][1 KB]) + one 1-KB offset box
                if constexpr (EXACT_M != 24) {
                    const int weight_tile = tile + (dual_helper ? (l & 1) * 36 : 0);
                    tma_load_3d(stage + W_OFF, P.tmaps + 2, 0, weight_tile * 2, e * (INTER / 32), &full[s]);
                    tma_load_2d(stage + W_OFF + FC2_TILE_W_BYTES, P.tmaps + 3, weight_tile * 128, e * (INTER / 32), &full[s]);
                }
            }
#endif
        }
        return;
    }
}

template <int NT, int PARTS, int EXACT_M, bool TP = false, bool NN = false>   // NN (FMOE_NORM_NEXT, INPUT_TP entry): P.norm_next may replace the pull stage by the row stage
__device__ __forceinline__ void fc2_consumer(const Params& P, unsigned epoch, uint8_t* smem, uint64_t* full, uint64_t* empty, float* out_s, int M) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grid = EXACT_M ? 132 : (int)gridDim.x;
    const int ndev = EXACT_M ? 8 : P.ndev;
    int tile, part, parts;   // rows [tile*128, +128); this CTA handles experts u % parts == part
    fc2_map<PARTS>((int)blockIdx.x, grid, tile, part, parts);
    const int* misc = reinterpret_cast<const int*>(smem + OFF_TOK) + 3 * 64;
    const int U = misc[32];
    const int* s_union = reinterpret_cast<const int*>(smem + OFF_TAB + TAB_UNION);
    int expert_begin, expert_end, expert_stride;
    fc2_expert_range<EXACT_M, PARTS>(P, U, part, parts, expert_begin, expert_end, expert_stride);
    fc2_helper_range<EXACT_M, PARTS>(P, U, misc, tile, part, parts, expert_begin, expert_end, expert_stride);
    const bool gp2 = m32_gp2_active<EXACT_M, PARTS>(P, U, misc);
    const bool dual_helper = !gp2 && m32_fc2_dual_helper<EXACT_M, PARTS>(P, U, misc);
    using G = Fc2Geom<NT, EXACT_M>;
    constexpr int H_OFF = G::H_OFF, CS_OFF = G::CS_OFF, W_OFF = G::W_OFF, STAGE = G::STAGE, NST = G::NST;
    // Consumer WG pair par = wg>>1 takes the CTA's local experts l = par, par+2, ...; within a pair WG rh = wg&1 owns
    // rows rh*64.. of the tile. Two experts are in flight per CTA. Per expert a WG runs: dequant k-block 0 (overlapping
    // the previous expert's k-block-1 group) -> wait -> retire the previous k-block -> issue k-block 0 -> dequant
    // k-block 1 -> wait -> retire k-block 0 -> issue k-block 1. S is promoted into D after every k128 block with
    // cs[tok][kb] * rs[row]; the stage is released (8 warps of the pair) after the k-block-1 scalars are in registers.
    const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, tig = lane & 3;
    const int rh = wg & 1, par = wg >> 1;
    const int sel = ((g >> 1) + (wq & 1)) & 1;                        // see "Weight layout" (conflict-free fragment loads)
    const int row_local0 = rh * 64 + 32 * (wq >> 1) + 16 * sel + g;   // weight row of this thread's first accumulator row; +8 for the second
    constexpr int NR = NT / 2;
    float D[NR], S[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) { D[i] = 0.f; S[i] = 0.f; }
    // M64 spends 64 registers on D/S alone; a single A fragment saves 16 registers at the cost of dequant/MMA overlap.
    constexpr int NA = EXACT_M == 64 ? 1 : 2;
    uint32_t A[NA][4][4];   // [k-block buffer][slice][regs]
    int L = (expert_end - expert_begin + expert_stride - 1) / expert_stride;
    if constexpr (fc2_lockstep<EXACT_M, PARTS>()) L += L & 1;   // lockstep: even stages per pass (producer pads with a zero-scale stage)
    L *= dual_helper ? 2 : 1;

    auto stage_of = [&](int l) { return smem + OFF_RING + (l % NST) * STAGE; };
    auto cs_of = [&](const uint8_t* st) { return reinterpret_cast<const float*>(st + CS_OFF); };
    // D += S * cs[tok][KB] * s2[e] for k-block KB of local expert l (after wait<0>). The scalars are read into registers
    // first; k-block 1 then releases the stage (the mbarrier arrive must precede the accumulator reads).
    auto retire = [&](auto kb_t, int l) {
        constexpr int KB = decltype(kb_t)::value;
        const uint8_t* st = stage_of(l);
        const float* cs_s = cs_of(st);
        // FC1 precomputes these products in a disjoint spare scale plane.
        // Large unions retain the original load and multiplication path.
        const bool pre_scaled = (EXACT_M == 32 && FMOE_CS_PERM) || (FC2_PRESCALE<EXACT_M> && U <= MAXU / 2);
        float rs = 1.f;
        if (!pre_scaled) rs = __ldg(P.fc2_s2 + s_union[fc2_union_index<EXACT_M, PARTS>(misc,
                                  expert_begin + (dual_helper ? l / 2 : l) * expert_stride)]) * 64.0f;
        if constexpr (EXACT_M == 32 && FMOE_CS_PERM) {
            // The scales sit in the ring outside the stage (fc2_cs_ring), so the stage is released as soon as its k-block-1
            // group completed (nothing else reads it) and the permuted scales are then streamed as float2 pairs -- no staging
            // copy, no warp barriers, two live scale registers (the float4 form kept 8 live across the arrive and spilled).
            static_assert(NT == 32, "permuted scale plane is the M32 layout");
            (void)cs_s;
            if constexpr (KB == 1 && !(FMOE_FC2_EARLY_RELEASE && NA == 2)) { if (lane == 0) mbar_arrive(&empty[l % NST]); }
            const float* cr = fc2_cs_ring(smem) + (l & (FC2_CS_RING - 1)) * 64 + KB * 32 + tig * 8;
            fence_operand(S);
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                const float2 cv = *reinterpret_cast<const float2*>(cr + 2 * j);
                D[4 * j + 0] = fmaf(S[4 * j + 0], cv.x, D[4 * j + 0]);
                D[4 * j + 1] = fmaf(S[4 * j + 1], cv.y, D[4 * j + 1]);
                D[4 * j + 2] = fmaf(S[4 * j + 2], cv.x, D[4 * j + 2]);
                D[4 * j + 3] = fmaf(S[4 * j + 3], cv.y, D[4 * j + 3]);
            }
        } else if constexpr (EXACT_M == 32) {
            // FC1 is finished in this CTA: masks/slots/weights are dead, only
            // the union remains live. A private row per warp retains scales
            // after the ring stage is released, without NT/4 live registers.
            static_assert(TAB_MASK + (NCONS / 32) * NT * sizeof(float) <= TAB_BYTES,
                          "warp-local scales fit in dead routing tables");
            float* warp_cs = reinterpret_cast<float*>(smem + OFF_TAB + TAB_MASK) + warp * NT;
            for (int t = lane; t < NT; t += 32) {
                const float cs = cs_s[t * 2 + KB];
                warp_cs[t] = pre_scaled ? cs : cs * rs;
            }
            __syncwarp();
            if constexpr (KB == 1) { if (lane == 0) mbar_arrive(&empty[l % NST]); }
            // Keep all accumulator reads after stage retirement, as on the
            // original path. No producer or other warp writes warp_cs.
            fence_operand(S);
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                const float2 cv = *reinterpret_cast<const float2*>(warp_cs + 8 * j + 2 * tig);
                D[4 * j + 0] = fmaf(S[4 * j + 0], cv.x, D[4 * j + 0]);
                D[4 * j + 1] = fmaf(S[4 * j + 1], cv.y, D[4 * j + 1]);
                D[4 * j + 2] = fmaf(S[4 * j + 2], cv.x, D[4 * j + 2]);
                D[4 * j + 3] = fmaf(S[4 * j + 3], cv.y, D[4 * j + 3]);
            }
            // Prevent early lanes in this warp from overwriting the scratch
            // for the next k-block before all current reads have completed.
            __syncwarp();
        } else {
        float cv[NT / 4];
#pragma unroll
        for (int j = 0; j < NT / 8; ++j) {
            const int t0 = 8 * j + 2 * tig;
            const float c0 = cs_s[t0 * 2 + KB], c1 = cs_s[(t0 + 1) * 2 + KB];
            cv[2 * j] = pre_scaled ? c0 : c0 * rs;
            cv[2 * j + 1] = pre_scaled ? c1 : c1 * rs;
        }
        if constexpr (KB == 1) { __syncwarp(); if (lane == 0) mbar_arrive(&empty[l % NST]); }
        // Pin the accumulator reads below AFTER the arrive: with cv final before the arrive, nvcc hoisted the FMAs above it, and
        // ptxas (which puts a WG.DP in front of the arrive) then saw accumulator reads inside the pipeline stage -> C7514, the
        // whole function serialized. The old code only escaped because its per-row factors were loaded after the arrive.
        fence_operand(S);
#pragma unroll
        for (int j = 0; j < NT / 8; ++j) {
            D[4 * j + 0] = fmaf(S[4 * j + 0], cv[2 * j], D[4 * j + 0]);
            D[4 * j + 1] = fmaf(S[4 * j + 1], cv[2 * j + 1], D[4 * j + 1]);
            D[4 * j + 2] = fmaf(S[4 * j + 2], cv[2 * j], D[4 * j + 2]);
            D[4 * j + 3] = fmaf(S[4 * j + 3], cv[2 * j + 1], D[4 * j + 3]);
        }
        }
    };
    auto issue_kb = [&](uint32_t (&Ac)[4][4], const uint8_t* stage, int kb) {
        fence_operand(S);
        wg::fence();
        const uint32_t haddr = smem_u32(stage + H_OFF) + kb * NT * 128;
        static_assert(OFF_RING % 128 == 0 && STAGE % 128 == 0 && H_OFF % 128 == 0,
                      "FC2 K128 activation bases must be128-byte aligned");
        issue4_n<NT, (EXACT_M != 0)>(S, Ac, haddr);
        wg::commit();
    };
    // one expert of this pair; RET: retire the previous expert's k-block 1 (compile-time so no runtime branch guards
    // accumulator reads -- C7518 otherwise)
#ifdef FMOE_DIAG
    long long c2_wait = 0, c2_dq = 0, c2_mw = 0, c2_rt = 0, c2_is = 0, c2_n = 0;   // per-expert phase cycles (tid 0 = pair 0): full-wait, dequant, wgmma-wait, retire, issue
    long long w_dq = 0, w_is = 0, w_t = 0;   // FMOE_DIAG_SKEW: this warp's own dequant / issue cycles (lane 0 of warps 0..3)
#define DIAG2(acc, t0) do { if (tid == 0) { const long long t1 = clock64(); (acc) += t1 - (t0); (t0) = t1; } } while (0)
#ifdef FMOE_DIAG_SKEW
#define DIAGW(acc) do { if (lane == 0 && warp < 4) { const long long t1 = clock64(); (acc) += t1 - w_t; w_t = t1; } } while (0)
#define DIAGW0() do { if (lane == 0 && warp < 4) w_t = clock64(); } while (0)
#else
#define DIAGW(acc) do {} while (0)
#define DIAGW0() do {} while (0)
#endif
#else
#define DIAGW(acc) do {} while (0)
#define DIAGW0() do {} while (0)
#define DIAG2(acc, t0) do {} while (0)
#endif
    auto expert = [&](auto ret_t, int l) {
        constexpr bool RET = decltype(ret_t)::value;
#ifdef FMOE_DIAG
        long long tc = 0; if (tid == 0) { tc = clock64(); ++c2_n; }
#endif
        mbar_wait(&full[l % NST], (l / NST) & 1);
        DIAG2(c2_wait, tc);
        const uint8_t* stage = stage_of(l);
        // this thread's fragment words / offset bytes: [k32 slice][rh] blocks (slice stride 2048), [slice] offset runs (stride 128)
        const uint8_t* frag = stage + W_OFF + rh * HL_BLOCK_BYTES + ((wq >> 1) * 32 + lane) * 16 + sel * 8;
        const uint8_t* scl = stage + W_OFF + FC2_TILE_W_BYTES + rh * 64 + 8 * g + 4 * (wq >> 1) + 2 * sel;
        if constexpr (NA == 1) {
            wg::wait<0>();
            fence_operand(S); fence_operand_a(A[0]);
            if constexpr (RET) retire(std::integral_constant<int, 1>{}, l - 2);
            dequant_slices4<2048, 128, (EXACT_M != 0)>(frag, scl, A[0]);
            issue_kb(A[0], stage, 0);
            wg::wait<0>();
            fence_operand(S); fence_operand_a(A[0]);
            retire(std::integral_constant<int, 0>{}, l);
            dequant_slices4<2048, 128, (EXACT_M != 0)>(frag + 4 * 2048, scl + 4 * 128, A[0]);
            issue_kb(A[0], stage, 1);
        } else {
        DIAGW0();
        dequant_slices4<2048, 128, (EXACT_M != 0)>(frag, scl, A[0]);   // overlaps the previous expert's k-block 1 (reads A[1])
        DIAG2(c2_dq, tc); DIAGW(w_dq);
        wg::wait<0>();
        fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[1]);
        DIAG2(c2_mw, tc);
        if constexpr (RET) retire(std::integral_constant<int, 1>{}, l - 2);
        DIAG2(c2_rt, tc); DIAGW0();
        issue_kb(A[0], stage, 0);
        DIAG2(c2_is, tc); DIAGW(w_is);
        dequant_slices4<2048, 128, (EXACT_M != 0)>(frag + 4 * 2048, scl + 4 * 128, A[1]);   // overlaps k-block 0 (reads A[0])
        DIAG2(c2_dq, tc); DIAGW(w_dq);
        wg::wait<0>();
        fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[1]);
        DIAG2(c2_mw, tc);
        retire(std::integral_constant<int, 0>{}, l);
        DIAG2(c2_rt, tc); DIAGW0();
        issue_kb(A[1], stage, 1);
        DIAG2(c2_is, tc); DIAGW(w_is);
        if constexpr (FMOE_FC2_EARLY_RELEASE && EXACT_M == 32 && FMOE_CS_PERM) {
            // release the stage as soon as its k-block-1 group completed (the scales live outside the stage): the slot goes back
            // to the producer one dequant earlier than the retire-time arrive; costs the overlap of the next k-block-0 dequant
            // with this group (~120 cycles of exposed wgmma latency, ubench: -2% per pair-stage under ring coupling)
            wg::wait<0>();
            fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[1]);
            if (lane == 0) mbar_arrive(&empty[l % NST]);
        }
        }
    };
    using BF = std::integral_constant<bool, false>; using BT = std::integral_constant<bool, true>;
    constexpr bool LOCKSTEP = fc2_lockstep<EXACT_M, PARTS>();
    if constexpr (LOCKSTEP) {
        // Lockstep consumers: WG (rh, kb) owns rows rh*64.. and k-block kb of EVERY stage, so all four
        // warpgroups run FC1's tile-loop shape -- per stage one 4-wgmma group, one wait, one retire (D += S * cs[kb]) -- instead
        // of two pairs alternating stages with two waits / two retires each. kb == par (wg >> 1), so the epilogue's pair sum
        // (out_s = D0 + D1) is the k-block sum. The stage is released after all 16 warps retired it (empty count 16).
        const int kb = par;
        const int n_pass = dual_helper ? 2 : 1;
        int n_per = (expert_end - expert_begin + expert_stride - 1) / expert_stride;
        n_per += n_per & 1;   // the producer pads an odd pass with one zero-scale stage of its last expert
        float* cs_ring = fc2_cs_ring(smem);
        auto retire_l = [&](int l) {   // after wait<0> of stage l's group: release the stage, D += S * cs[l][kb][this lane's tokens]
            if (lane == 0) mbar_arrive(&empty[l % NST]);
            const float* cr = cs_ring + (l & (FC2_CS_RING - 1)) * 64 + kb * 32 + tig * 8;
            fence_operand(S);
#pragma unroll
            for (int j = 0; j < NT / 8; ++j) {
                const float2 cv = *reinterpret_cast<const float2*>(cr + 2 * j);
                D[4 * j + 0] = fmaf(S[4 * j + 0], cv.x, D[4 * j + 0]);
                D[4 * j + 1] = fmaf(S[4 * j + 1], cv.y, D[4 * j + 1]);
                D[4 * j + 2] = fmaf(S[4 * j + 2], cv.x, D[4 * j + 2]);
                D[4 * j + 3] = fmaf(S[4 * j + 3], cv.y, D[4 * j + 3]);
            }
            fence_operand(D);   // pin the promotion before S is reused (C7515 otherwise, as in fc1_consumer)
        };
        auto stage_step = [&](int l, uint32_t (&Ac)[4][4], int l0) {
            mbar_wait(&full[l % NST], (l / NST) & 1);
            const uint8_t* stage = stage_of(l);
            const uint8_t* frag = stage + W_OFF + kb * (4 * 2048) + rh * HL_BLOCK_BYTES + ((wq >> 1) * 32 + lane) * 16 + sel * 8;
            const uint8_t* scl = stage + W_OFF + FC2_TILE_W_BYTES + kb * (4 * 128) + rh * 64 + 8 * g + 4 * (wq >> 1) + 2 * sel;
            dequant_slices4<2048, 128, (EXACT_M != 0)>(frag, scl, Ac);   // overlaps the previous stage's group (other A buffer)
            wg::wait<0>();
            fence_operand(S); fence_operand_a(A[0]); fence_operand_a(A[1]);
            if (l > l0) retire_l(l - 1);
            fence_operand(S);
            wg::fence();
            issue4_n<NT, (EXACT_M != 0)>(S, Ac, smem_u32(stage + H_OFF) + kb * NT * 128);
            wg::commit();
        };
        if (P.mode & 8) {   // diagnostics: fc2 pipeline only
            for (int l = 0; l < L; ++l) { mbar_wait(&full[l % NST], (l / NST) & 1); if (lane == 0) mbar_arrive(&empty[l % NST]); }
        } else {
            int l = 0;
#pragma unroll 1
            for (int pass = 0; pass < n_pass; ++pass) {
                const int l0 = l, l1 = l0 + n_per;
#pragma unroll 1
                for (; l < l1; l += 2) {   // even count: static A buffers by stage parity (a runtime-indexed A goes to local memory)
                    stage_step(l, A[0], l0);
                    stage_step(l + 1, A[1], l0);
                }
                if (l1 > l0) { wg::wait<0>(); fence_operand(S); retire_l(l1 - 1); }   // drain the pass's last stage
                if (dual_helper) {
                    // this pass's tile partial = D(kb 0) + D(kb 1), summed through out_s; then restart the accumulator for tile + 36
                    if (kb == 0) {
#pragma unroll
                        for (int j = 0; j < NT / 8; ++j)
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                out_s[(8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1)] = D[4 * j + i];
                    }
                    named_bar_sync(1, NCONS);
                    if (kb == 1) {
#pragma unroll
                        for (int j = 0; j < NT / 8; ++j)
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                out_s[(8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1)] += D[4 * j + i];
                    }
                    named_bar_sync(1, NCONS);
                    float* partial = reinterpret_cast<float*>(P.h_buf + ((size_t)U + 64) * 2 * NT * 128) + (size_t)(tile + pass * 36) * NT * 128;
                    for (int i = tid; i < NT * 128; i += NCONS) partial[i] = out_s[(i / 128) * HST_STRIDE + i % 128];
                    named_bar_sync(1, NCONS);
                    if (tid == 0) st_release_gpu(&P.part_flags[64 + tile + pass * 36], epoch);
#pragma unroll
                    for (int i = 0; i < NR; ++i) D[i] = 0.f;
                }
            }
        }
        if (dual_helper) {
            if (tid == 0) STAMP(P, 2);   // phase build: dual helper's partials published
            return;   // one pipeline, two partials, no extra TP source
        }
    } else {
    if (P.mode & 8) {   // diagnostics: fc2 pipeline only
        for (int l = par; l < L; l += 2) { mbar_wait(&full[l % NST], (l / NST) & 1); if (lane == 0) mbar_arrive(&empty[l % NST]); }
    } else {
        // This pair's stages l = par, par + 2, ... < L. FMOE_GP2 helper pair 1 first runs a peeled segment, stages 1..2nq-1 for tile
        // 36 + h%12, flushes that accumulator to arena slot 36 + h, and then joins the common walk at 2nq+1 for tile h. (Peeled rather
        // than a two-iteration segment loop: no extra loop-carried state across the common walk -- the loop form spilled.)
        int l0 = par;
        if (gp2 && (int)blockIdx.x >= 96 && par == 1) {
            const int l1 = 2 * misc[GP2_MISC + 2];
            if (l0 < l1) {
                expert(BF{}, l0);
                int l = l0 + 2;
                for (; l < l1; l += 2) expert(BT{}, l);
                wg::wait<0>(); fence_operand(S);
                retire(std::integral_constant<int, 1>{}, l - 2);   // drain: the segment's last expert's k-block 1
            }
            const int slot = 36 + ((int)blockIdx.x - 96);
            float* qpart = reinterpret_cast<float*>(P.h_buf + ((size_t)U + 64) * 2 * NT * 128) + (size_t)slot * NT * 128;
#pragma unroll
            for (int j = 0; j < NT / 8; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    qpart[(8 * j + 2 * tig + (i & 1)) * 128 + row_local0 + 8 * (i >> 1)] = D[4 * j + i];
            named_bar_sync(8, NCONS / 2);
            if (tid == NCONS / 2) st_release_gpu(&P.part_flags[64 + slot], epoch);
#pragma unroll
            for (int i = 0; i < NR; ++i) D[i] = 0.f;
            l0 = l1 + 1;
        }
        if (l0 < L) {
            expert(BF{}, l0);
            int l = l0 + 2;
            for (; l < L; l += 2) expert(BT{}, l);
            wg::wait<0>(); fence_operand(S);
            retire(std::integral_constant<int, 1>{}, l - 2);   // drain: the pair's last expert's k-block 1
        }
    }
    }

    if constexpr (EXACT_M == 32 && PARTS == 2 && !LOCKSTEP) {
        if (dual_helper) {
            // Each WG pair owns a different [NT,128] tile. All experts for
            // that tile are already in D; summing the pairs would be wrong.
            float* partial = reinterpret_cast<float*>(P.h_buf + ((size_t)U + 64) * 2 * NT * 128) +
                             (size_t)(tile + par * 36) * NT * 128;
#pragma unroll
            for (int j = 0; j < NT / 8; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    partial[(8 * j + 2 * tig + (i & 1)) * 128 + row_local0 + 8 * (i >> 1)] = D[4 * j + i];
            named_bar_sync(1, NCONS);
            if (tid == 0) {
                st_release_gpu(&P.part_flags[64 + tile], epoch);
                st_release_gpu(&P.part_flags[64 + tile + 36], epoch);
                STAMP(P, 2);   // phase build: dual helper's partials published
            }
            return;   // one pipeline, two partials, no extra TP source
        }
    }

    // ---- epilogue: the two pairs sum into out_s[tok][row] ----
    if (par == 0) {
#pragma unroll
        for (int j = 0; j < NT / 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out_s[(8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1)] = D[4 * j + i];
    }
    named_bar_sync(1, NCONS);
    if (par == 1) {
#pragma unroll
        for (int j = 0; j < NT / 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out_s[(8 * j + 2 * tig + (i & 1)) * HST_STRIDE + row_local0 + 8 * (i >> 1)] += D[4 * j + i];
    }
    named_bar_sync(1, NCONS);
    if constexpr (EXACT_M == 32 && PARTS == 2) {
        if (m32_fc2_helpers<EXACT_M, PARTS>(P, U, misc)) {
            static_assert(KSPLIT == 1 && NT == 32, "helper flags and scratch are M32/unsplit-K only");
            // Reserve64 slots after active h for concurrent FC1 raw partials.
            // The separate FC2 arena fits384 slots: U+64+96<=292 atU132.
            float* partial = reinterpret_cast<float*>(P.h_buf + ((size_t)U + 64) * 2 * NT * 128) +
                             (size_t)tile * NT * 128;
            if ((int)blockIdx.x >= 96) {
                for (int i = tid; i < NT * 128; i += NCONS)
                    partial[i] = out_s[(i / 128) * HST_STRIDE + i % 128];
                named_bar_sync(1, NCONS);
                if (tid == 0) { st_release_gpu(&P.part_flags[64 + tile], epoch); STAMP(P, 2); }   // phase build: helper partial published
                return;   // helper is not an additional TP all-reduce source
            }
            if (part == 1) {
                if (FMOE_MERGE_PREFETCH && FMOE_BATCH_FENCE) {
                    // The producer lane parked the helper partial(s) in ring slots L.. (see fc2_producer): wait for those
                    // pseudo-stages like any other and add from smem, in slot order (FMOE_GP2 tiles 36..47: three quarter partials).
                    const int nmerge = (gp2 && tile >= 36) ? 3 : 1;
#pragma unroll 1
                    for (int k = 0; k < nmerge; ++k) {
                        mbar_wait(&full[(L + k) % NST], ((L + k) / NST) & 1);
                        const float* ps = reinterpret_cast<const float*>(stage_of(L + k));
                        for (int i = tid; i < NT * 128; i += NCONS)
                            out_s[(i / 128) * HST_STRIDE + i % 128] += ps[i];   // disjoint elements per thread: no barrier between partials
                    }
                } else {
                    if (tid == 0) while (ld_acquire_gpu(&P.part_flags[64 + tile]) != epoch) {}
                    named_bar_sync(1, NCONS);
                    for (int i = tid; i < NT * 128; i += NCONS)
                        out_s[(i / 128) * HST_STRIDE + i % 128] += __ldcg(partial + i);
                }
                named_bar_sync(1, NCONS);
            }
        }
    }
    if (tid == 0) STAMP(P, 2);   // fc2 compute done, including any helper partial
#ifdef FMOE_DIAG
    if (tid == 0 && P.dbg) {   // FC2 pair-0 phase cycles (DIAG build only; overwrites the TP diag stamps 25..27)
        P.dbg[blockIdx.x * NSTAMP + 25] = (unsigned long long)(unsigned)c2_wait | ((unsigned long long)(unsigned)c2_dq << 32);
        P.dbg[blockIdx.x * NSTAMP + 26] = (unsigned long long)(unsigned)c2_mw | ((unsigned long long)(unsigned)c2_rt << 32);
        P.dbg[blockIdx.x * NSTAMP + 27] = (unsigned long long)(unsigned)c2_is | ((unsigned long long)(unsigned)c2_n << 32);
    }
#ifdef FMOE_DIAG_SKEW
    if (lane == 0 && warp < 4 && P.dbg) P.dbg[blockIdx.x * NSTAMP + 16 + warp] = (unsigned long long)(unsigned)w_dq | ((unsigned long long)(unsigned)w_is << 32);   // WG 0, per warp
#endif
#undef DIAG2
#undef DIAGW
#undef DIAGW0
#endif
    if constexpr (FMOE_LATE_TRIGGER) griddep_launch_dependents();   // late PDL trigger for the next layer's norm kernel

    // M32 must not round local TP partials to BF16: the captured layer2
    // counterexample exceeds the unchanged own-h tolerance. Keep24bits
    // until FP32 rank accumulation; round only the final output to BF16.

    // Exact two-part M32/M64: four24-bit-rounded FP32 partials + epoch per LL packet.
    // All16 sources accumulate in FP32; final output alone rounds to BF16.
    if constexpr (EXACT_M == 64 || (EXACT_M == 32 && PARTS == 2)) {
        static_assert(NT == EXACT_M && PARTS == 2 && (NT == 32 || NT == 64));
        if (P.out_bf16) {
            // All packet indices are nonnegative and <2048. Unsigned address
            // arithmetic avoids carrying signed high words across FC1/FC2.
            const unsigned wire_tid = threadIdx.x;
            constexpr int BMSG = 128 / 4, TPR = N_FC2_TILES / 8;
            constexpr int H = (N_FC2_TILES * 2) / TPR, TPS = NT / H;
            constexpr int NMSG = NT * BMSG;
            constexpr int AG_BMSG = EXACT_M == 32 ? (128 + 5) / 6 : BMSG;
            constexpr int AG_NMSG = NT * AG_BMSG;
            const int owner = tile / TPR, lt = tile % TPR;
            uint4* dst = tp_rs<TP>(P, owner) +
                ((size_t)((P.my_rank * N_FC2_PARTS_MAX + part) * TPR + lt) * NT) * MSG_PER_TOK;
            auto pack24 = [](float f) {
                const uint32_t u = __float_as_uint(f);
                uint32_t q = (u + 0x7fu + ((u >> 8) & 1u)) >> 8;
                if ((u & 0x7f800000u) == 0x7f800000u)
                    q = (u >> 8) | ((u & 0x7fffffu) ? 0x4000u : 0u);
                return q;
            };
            for (unsigned i = wire_tid; i < NMSG; i += NCONS) {
                const unsigned tok = i / BMSG, j = i - tok * BMSG;
                const float* p = out_s + tok * HST_STRIDE + 4 * j;
                const uint32_t q0 = pack24(p[0]), q1 = pack24(p[1]), q2 = pack24(p[2]), q3 = pack24(p[3]);
                const uint32_t x = q0 | (q1 << 24), y = (q1 >> 8) | (q2 << 16), z = (q2 >> 16) | (q3 << 8);
                st_ll(dst + tok * MSG_PER_TOK + j, make_uint4(x, y, z, epoch));
            }
            if (tid == 0) STAMP(P, 3);
            // trigger 1: this CTA's partial is pushed; thread NCONS-1 is idle until its pull poll (no reduce role) -> it issues the share.
            if constexpr (FMOE_TAIL_PREFETCH == 1) { if (tid == NCONS - 1) tail_prefetch_share(P, (int)blockIdx.x, N_FC2_TILES * PARTS); }
            {
                const int olt = blockIdx.x / H, slice = blockIdx.x % H;  // codespell:ignore
                const int rtile = P.my_rank * TPR + olt, t0 = slice * TPS;  // codespell:ignore
                const uint4* rs_me = tp_rs<TP>(P, P.my_rank);
                auto reduce_batch = [&](int base, int tok, int j,
                                        float& a0, float& a1, float& a2, float& a3) {
                    uint4 m[8];
                    unsigned pending = 0xffu;
                    do {
#pragma unroll
                        for (int d = 0; d < 8; ++d)
                            if (pending & (1u << d)) {
                                const int sr = base + d, slot = (sr / 2) * N_FC2_PARTS_MAX + sr % 2;
                                m[d] = ld_ll(rs_me +
                                    ((size_t)(slot * TPR + olt) * NT + tok) * MSG_PER_TOK + j);  // codespell:ignore
                            }
#pragma unroll
                        for (int d = 0; d < 8; ++d)
                            if ((pending & (1u << d)) && m[d].w == epoch) pending &= ~(1u << d);
                    } while (pending);
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        a0 += __uint_as_float(m[d].x << 8);
                        a1 += __uint_as_float(((m[d].x >> 16) | (m[d].y << 16)) & 0xffffff00u);
                        a2 += __uint_as_float(((m[d].y >> 8) | (m[d].z << 24)) & 0xffffff00u);
                        a3 += __uint_as_float(m[d].z & 0xffffff00u);
                    }
                };
                for (unsigned i = wire_tid; i < TPS * BMSG; i += NCONS) {
                    const unsigned tok = t0 + i / BMSG, j = i - (i / BMSG) * BMSG;
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    if constexpr (FMOE_TAIL_POLL16 && EXACT_M == 32) {
                        // the two batches of 8 sources were two dependent poll round trips on the critical path (the second
                        // batch's loads were issued only after the first landed). Thread i (< 64) polls sources 0..7 while the otherwise
                        // idle thread i + 64 polls sources 8..15 of the same (tok, j) concurrently and parks their DECODED values in
                        // the dead FC2 ring; after one named barrier thread i adds them in order: ((0 + m0) + ... + m7) + m8 + ... + m15,
                        // exactly the original sequence -> bitwise identical. One poll round trip instead of two.
                        static_assert(TPS * BMSG == 64 && NCONS >= 128, "two halves of 64 threads");
                        reduce_batch(0, tok, j, a0, a1, a2, a3);
                        const float4* hs = reinterpret_cast<const float4*>(smem + OFF_TP_HALF) + (size_t)i * 8;   // sources 8..15 of this (tok, j)
                        named_bar_sync(6, 128);
#pragma unroll
                        for (int d = 0; d < 8; ++d) { const float4 v = hs[d]; a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w; }
                    } else {
                        reduce_batch(0, tok, j, a0, a1, a2, a3);
                        reduce_batch(8, tok, j, a0, a1, a2, a3);
                    }
                    if constexpr (EXACT_M == 32) {
                        static_assert(NT == 32 && TPS == 2 && BMSG == 32);
                        // One complete reduce warp owns one token's128rows.
                        // Repack the already-rounded BF16 pairs: no new rounding.
                        // Every lane executes all shuffles, including lanes22..31.
                        const uint32_t lo = pack_bf16x2_rn(a0, a1), hi = pack_bf16x2_rn(a2, a3);
                        const unsigned s0 = ((3 * j) / 2) & 31u, s1 = (s0 + 1) & 31u;
                        const uint32_t al = __shfl_sync(0xffffffffu, lo, s0);
                        const uint32_t ah = __shfl_sync(0xffffffffu, hi, s0);
                        const uint32_t bl = __shfl_sync(0xffffffffu, lo, s1);
                        const uint32_t bh = __shfl_sync(0xffffffffu, hi, s1);
                        if (j < AG_BMSG) {
                            const uint32_t x = (j & 1) ? ah : al;
                            const uint32_t y = j == AG_BMSG - 1 ? 0u : ((j & 1) ? bl : ah);
                            const uint32_t z = j == AG_BMSG - 1 ? 0u : ((j & 1) ? bh : bl);
                            const uint4 mo = make_uint4(x, y, z, epoch);
                            for (int d = 0; d < 8; ++d)
                                st_ll(tp_ag<TP>(P, d) + ((size_t)rtile * NT + tok) * MSG_PER_TOK + j, mo);
                        }
                    } else {
                        const uint4 mo = make_uint4(pack_bf16x2_rn(a0, a1), pack_bf16x2_rn(a2, a3),
                                                   0u, epoch);
                        for (int d = 0; d < 8; ++d)
                            st_ll(tp_ag<TP>(P, d) + ((size_t)rtile * NT + tok) * MSG_PER_TOK + j, mo);
                    }
                }
                if constexpr (FMOE_TAIL_POLL16 && EXACT_M == 32) {   // partner half: poll + decode sources 8..15, hand over through smem
                    if (wire_tid >= 64 && wire_tid < 128) {
                        const unsigned i = wire_tid - 64, tok = t0 + i / BMSG, j = i - (i / BMSG) * BMSG;
                        uint4 m[8];
                        unsigned pending = 0xffu;
                        do {
#pragma unroll
                            for (int d = 0; d < 8; ++d)
                                if (pending & (1u << d)) {
                                    const int sr = 8 + d, slot = (sr / 2) * N_FC2_PARTS_MAX + sr % 2;
                                    m[d] = ld_ll(rs_me + ((size_t)(slot * TPR + olt) * NT + tok) * MSG_PER_TOK + j);  // codespell:ignore
                                }
#pragma unroll
                            for (int d = 0; d < 8; ++d)
                                if ((pending & (1u << d)) && m[d].w == epoch) pending &= ~(1u << d);
                        } while (pending);
                        float4* hs = reinterpret_cast<float4*>(smem + OFF_TP_HALF) + (size_t)i * 8;
#pragma unroll
                        for (int d = 0; d < 8; ++d)
                            hs[d] = make_float4(__uint_as_float(m[d].x << 8), __uint_as_float(((m[d].x >> 16) | (m[d].y << 16)) & 0xffffff00u),
                                                __uint_as_float(((m[d].y >> 8) | (m[d].z << 24)) & 0xffffff00u), __uint_as_float(m[d].z & 0xffffff00u));
                        named_bar_sync(6, 128);
                        // trigger 2: all 16 sources of this CTA's owned (tile, tokens) landed = every rank's FC2 for that tile is done;
                        // the partner half has nothing left but the pull -> its first thread issues the share.
                        if constexpr (FMOE_TAIL_PREFETCH == 2) { if (wire_tid == 64) tail_prefetch_share(P, (int)blockIdx.x, N_FC2_TILES * PARTS); }
                    }
                }
            }
            if (tid == 0) STAMP(P, 4);
            if (!NN || !P.norm_next) {   // FMOE_NORM_NEXT: with the row stage on, the row CTAs write `out` (and residual_new / x_fp8 / scales)
                constexpr int MAXMSG = (AG_NMSG + NCONS - 1) / NCONS;
                const uint4* ag_me = tp_ag<TP>(P, P.my_rank) + (size_t)tile * NT * MSG_PER_TOK;
                uint4 m[MAXMSG];
                unsigned pending = 0;
#pragma unroll
                for (int k = 0; k < MAXMSG; ++k)
                    if (wire_tid + k * NCONS < AG_NMSG && (((wire_tid + k * NCONS) / AG_BMSG) & 1) == part)
                        pending |= 1u << k;
                do {
#pragma unroll
                    for (int k = 0; k < MAXMSG; ++k)
                        if (pending & (1u << k)) {
                            const unsigned i = wire_tid + k * NCONS, tok = i / AG_BMSG, j = i - tok * AG_BMSG;
                            m[k] = ld_ll(ag_me + tok * MSG_PER_TOK + j);
                        }
#pragma unroll
                    for (int k = 0; k < MAXMSG; ++k)
                        if ((pending & (1u << k)) && m[k].w == epoch) {
                            const unsigned i = wire_tid + k * NCONS, tok = i / AG_BMSG, j = i - tok * AG_BMSG;
                            const size_t off = (size_t)tok * DIM + tile * 128 + (EXACT_M == 32 ? 6 : 4) * j;
                            uint32_t* o = reinterpret_cast<uint32_t*>(P.out_bf16 + off);
                            o[0] = m[k].x;
                            if constexpr (EXACT_M == 32) {
                                if (j < AG_BMSG - 1) { o[1] = m[k].y; o[2] = m[k].z; }
                            } else {
                                o[1] = m[k].y;
                            }
                            pending &= ~(1u << k);
                        }
                } while (pending);
                if (tid == 0) STAMP(P, 5);
            }
            // trigger 3: the CTA's output tile is written (CTA exit): no overlap at all with this kernel's own traffic.
            if constexpr (FMOE_TAIL_PREFETCH == 3) { if (tid == 0) tail_prefetch_share(P, (int)blockIdx.x, N_FC2_TILES * PARTS); }
            return;
        }
    }

    // Exact M32 BF16 output: six BF16 values + epoch per LL packet. Keep the
    // physical FP32-path pitches and all rank slots; only the active packet
    // count changes. Each local partial rounds once before rank-ordered FP32
    // accumulation. The legacy FP32-output path below keeps its precision.
    if constexpr (EXACT_M == 32 && PARTS == 1) {
        static_assert(NT == 32 && PARTS == 1);
        if (P.out_bf16) {
            constexpr int BMSG = (128 + 5) / 6, TPR = N_FC2_TILES / 8;
            constexpr int H = N_FC2_TILES / TPR, TPS = NT / H;
            constexpr int NMSG = NT * BMSG;
            const int owner = tile / TPR, lt = tile % TPR;
            uint4* dst = P.rs_bufs[owner] +
                ((size_t)((P.my_rank * N_FC2_PARTS_MAX) * TPR + lt) * NT) * MSG_PER_TOK;
            for (int i = tid; i < NMSG; i += NCONS) {
                const int tok = i / BMSG, j = i - tok * BMSG;
                const float* p = out_s + tok * HST_STRIDE + 6 * j;
                const uint32_t x = pack_bf16x2_rn(p[0], p[1]);
                // The final packet has only rows126/127, never read padding.
                const uint32_t y = j < BMSG - 1 ? pack_bf16x2_rn(p[2], p[3]) : 0u;
                const uint32_t z = j < BMSG - 1 ? pack_bf16x2_rn(p[4], p[5]) : 0u;
                st_ll(dst + tok * MSG_PER_TOK + j, make_uint4(x, y, z, epoch));
            }
            if (tid == 0) STAMP(P, 3);
            {
                const int olt = blockIdx.x / H, slice = blockIdx.x % H;  // codespell:ignore
                const int rtile = P.my_rank * TPR + olt, t0 = slice * TPS;  // codespell:ignore
                const uint4* rs_me = P.rs_bufs[P.my_rank];
                for (int i = tid; i < TPS * BMSG; i += NCONS) {
                    const int tok = t0 + i / BMSG, j = i - (i / BMSG) * BMSG;
                    uint4 m[8];
                    unsigned pending = 0xffu;
                    do {
#pragma unroll
                        for (int d = 0; d < 8; ++d)
                            if (pending & (1u << d))
                                m[d] = ld_ll(rs_me +
                                    ((size_t)((d * N_FC2_PARTS_MAX) * TPR + olt) * NT + tok) * MSG_PER_TOK + j);  // codespell:ignore
#pragma unroll
                        for (int d = 0; d < 8; ++d)
                            if ((pending & (1u << d)) && m[d].w == epoch) pending &= ~(1u << d);
                    } while (pending);
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        a0 += bf16lo(m[d].x); a1 += bf16hi(m[d].x);
                        a2 += bf16lo(m[d].y); a3 += bf16hi(m[d].y);
                        a4 += bf16lo(m[d].z); a5 += bf16hi(m[d].z);
                    }
                    const uint4 mo = make_uint4(pack_bf16x2_rn(a0, a1), pack_bf16x2_rn(a2, a3),
                                               pack_bf16x2_rn(a4, a5), epoch);
                    for (int d = 0; d < 8; ++d)
                        st_ll(P.ag_bufs[d] + ((size_t)rtile * NT + tok) * MSG_PER_TOK + j, mo);
                }
            }
            if (tid == 0) STAMP(P, 4);
            {
                constexpr int MAXMSG = (NMSG + NCONS - 1) / NCONS;
                const uint4* ag_me = P.ag_bufs[P.my_rank] + (size_t)tile * NT * MSG_PER_TOK;
                uint4 m[MAXMSG];
                unsigned pending = 0;
#pragma unroll
                for (int k = 0; k < MAXMSG; ++k)
                    if (tid + k * NCONS < NMSG) pending |= 1u << k;
                do {
#pragma unroll
                    for (int k = 0; k < MAXMSG; ++k)
                        if (pending & (1u << k)) {
                            const int i = tid + k * NCONS, tok = i / BMSG, j = i - tok * BMSG;
                            m[k] = ld_ll(ag_me + tok * MSG_PER_TOK + j);
                        }
#pragma unroll
                    for (int k = 0; k < MAXMSG; ++k)
                        if ((pending & (1u << k)) && m[k].w == epoch) {
                            const int i = tid + k * NCONS, tok = i / BMSG, j = i - tok * BMSG;
                            const size_t off = (size_t)tok * DIM + tile * 128 + 6 * j;
                            uint32_t* o = reinterpret_cast<uint32_t*>(P.out_bf16 + off);
                            o[0] = m[k].x;
                            if (j < BMSG - 1) { o[1] = m[k].y; o[2] = m[k].z; }
                            pending &= ~(1u << k);
                        }
                } while (pending);
            }
            if (tid == 0) STAMP(P, 5);
            return;
        }
    }

    // ---- two-phase LL all-reduce over the 128-row tile ----
    // Polling is batched (all sources / all of a thread's messages in flight at once) so the
    // cost is one memory round trip per batch, not one per message per source.
    const int TPR = N_FC2_TILES / ndev;   // tiles owned per rank
    const int owner = tile / TPR, lt = tile % TPR;
    const int nmsg = M * MSG_PER_TOK;
    {   // 1. push this rank's partial of the tile to the owning rank's reduce-scatter buffer
        uint4* dst = P.rs_bufs[owner] + ((size_t)((P.my_rank * N_FC2_PARTS_MAX + part) * TPR + lt) * NT) * MSG_PER_TOK;   // source slot = rank * 3 + part
        for (int i = tid; i < nmsg; i += NCONS) {
            const int tok = i / MSG_PER_TOK, j = i - tok * MSG_PER_TOK;
            const float* p = out_s + tok * HST_STRIDE + 3 * j;
            st_ll(dst + tok * MSG_PER_TOK + j, make_uint4(__float_as_uint(p[0]), __float_as_uint(p[1]), __float_as_uint(p[2]), epoch));
        }
    }
    if (tid == 0) STAMP(P, 3);
    {   // 2. reduce: the N_FC2_CTAS CTAs of this rank split the rank's TPR owned tiles into H token
        //    slices each; sources = ndev ranks x N_FC2_PARTS parts, polled 8 at a time.
        const int H = (PARTS < 3 ? N_FC2_TILES * PARTS : grid) / TPR;   // helper CTAs per owned tile (8 / 16 / 22 at ndev=8)
        const int olt = blockIdx.x / H, slice = blockIdx.x % H;  // codespell:ignore
        const int rtile = P.my_rank * TPR + olt;              // global tile being reduced  // codespell:ignore
        const int np = fc2_nparts_of_tile<PARTS>(rtile, grid);
        const int tps = (M + H - 1) / H;                    // tokens per slice
        const int t0 = slice * tps, t1 = min(M, t0 + tps);
        const int S = ndev * np;                            // sources: (rank, part) of the reduced tile
        const uint4* rs_me = P.rs_bufs[P.my_rank];
        const int nm = ((int)blockIdx.x < H * TPR) ? (t1 - t0) * MSG_PER_TOK : 0;
        // one batch = up to 8 sources polled together (straight-line code: keeps m[] in registers)
        auto reduce_batch = [&](int base, int tok, int j, float& ax, float& ay, float& az) {
            const int nb = min(8, S - base);
            uint4 m[8];
            unsigned pending = (1u << nb) - 1u;
            do {
#pragma unroll
                for (int d = 0; d < 8; ++d)
                    if (pending & (1u << d)) { const int sr = base + d, slot = (sr / np) * N_FC2_PARTS_MAX + (sr - (sr / np) * np); m[d] = ld_ll(rs_me + ((size_t)(slot * TPR + olt) * NT + tok) * MSG_PER_TOK + j); }  // codespell:ignore
#pragma unroll
                for (int d = 0; d < 8; ++d)
                    if ((pending & (1u << d)) && m[d].w == epoch) pending &= ~(1u << d);
            } while (pending);
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (d < nb) { ax += __uint_as_float(m[d].x); ay += __uint_as_float(m[d].y); az += __uint_as_float(m[d].z); }
        };
        for (int i = tid; i < nm; i += NCONS) {
            const int tok = t0 + i / MSG_PER_TOK, j = i - (i / MSG_PER_TOK) * MSG_PER_TOK;
            float ax = 0.f, ay = 0.f, az = 0.f;
            reduce_batch(0, tok, j, ax, ay, az);
            if (S > 8) reduce_batch(8, tok, j, ax, ay, az);
            if (S > 16) reduce_batch(16, tok, j, ax, ay, az);
            const uint4 mo = make_uint4(__float_as_uint(ax), __float_as_uint(ay), __float_as_uint(az), epoch);
            for (int d = 0; d < ndev; ++d) st_ll(P.ag_bufs[d] + ((size_t)rtile * NT + tok) * MSG_PER_TOK + j, mo);
        }
    }
    if (tid == 0) STAMP(P, 4);
    {   // 3. everyone: pull the reduced tile from the local all-gather buffer (all of a thread's
        //    messages in flight together; retry only the ones whose tag has not landed yet) and
        //    write the output directly in its final dtype (bf16, or fp32 for the legacy entry point).
        constexpr int POLL_M = EXACT_M ? EXACT_M : MAXM;
        constexpr int MAXMSG = (POLL_M * MSG_PER_TOK + NCONS - 1) / NCONS;
        const uint4* ag_me = P.ag_bufs[P.my_rank] + (size_t)tile * NT * MSG_PER_TOK;
        uint4 m[MAXMSG];
        unsigned pending = 0;
#pragma unroll
        for (int k = 0; k < MAXMSG; ++k) if (tid + k * NCONS < nmsg && (parts == 1 || (((tid + k * NCONS) / MSG_PER_TOK) % parts) == part)) pending |= 1u << k;
        do {
#pragma unroll
            for (int k = 0; k < MAXMSG; ++k) {
                if (pending & (1u << k)) {
                    const int i = tid + k * NCONS, tok = i / MSG_PER_TOK, j = i - tok * MSG_PER_TOK;
                    m[k] = ld_ll(ag_me + tok * MSG_PER_TOK + j);
                }
            }
#pragma unroll
            for (int k = 0; k < MAXMSG; ++k)
                if ((pending & (1u << k)) && m[k].w == epoch) {
                    const int i = tid + k * NCONS, tok = i / MSG_PER_TOK, j = i - tok * MSG_PER_TOK;
                    const size_t off = (size_t)tok * DIM + tile * 128 + 3 * j;
                    if (P.out_bf16) {
                        __nv_bfloat16* o = P.out_bf16 + off;
                        o[0] = __float2bfloat16_rn(__uint_as_float(m[k].x)); o[1] = __float2bfloat16_rn(__uint_as_float(m[k].y));
                        if (j < MSG_PER_TOK - 1) o[2] = __float2bfloat16_rn(__uint_as_float(m[k].z));
                    } else {
                        float* o = P.out_f32 + off;
                        o[0] = __uint_as_float(m[k].x); o[1] = __uint_as_float(m[k].y);
                        if (j < MSG_PER_TOK - 1) o[2] = __uint_as_float(m[k].z);
                    }
                    pending &= ~(1u << k);
                }
        } while (pending);
    }
    if (tid == 0) STAMP(P, 5);
}

template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }

template <int NT, int PARTS, int EXACT_M = 0, bool INPUT_TP = false>
__global__ void __launch_bounds__(NTHREADS, 1) fused_moe_kernel(const __grid_constant__ Params P) {   // grid_constant: the router TMA maps live in the parameter
    extern __shared__ uint8_t smem_raw[];
    const uint32_t raw_u32 = smem_u32(smem_raw);
    uint8_t* smem = smem_raw + (((raw_u32 + 1023u) & ~1023u) - raw_u32);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
    uint64_t* empty = full + 8;
    int* s_tok = reinterpret_cast<int*>(smem + OFF_TOK);
    int* s_inv = s_tok + 64;
    int* p_tok = s_inv + 64;
    int* s_misc = p_tok + 64;
    if constexpr (fc2_prefill<EXACT_M, PARTS>() && FMOE_FC2_PREFILL_NOTAIL_DELAY_NS > 0) { if (threadIdx.x == 0) s_misc[63] = (int)gtimer32(); }   // CTA start for the FC2 producers' early-leaver test
    if constexpr (INPUT_TP) {   // the reduce-scatter pushes leave before anything else (no smem needed)
        if (threadIdx.x == 0) STAMP(P, 27);   // TP diag: kernel entry, BEFORE the epoch read + RS push (stamp 0 follows the barrier init: use --t0 27 for absolute times)
        if constexpr (FMOE_TP_EPOCH_HINT) tp_rs_push(P, (unsigned)ld_evict_last_s32(&P.work[2]) + 1u, 132);
        else tp_rs_push(P, (unsigned)__ldg(&P.work[2]) + 1u, 132);
    }
    float* xp_s = reinterpret_cast<float*>(smem + OFF_XS);
    float* hst = reinterpret_cast<float*>(smem + OFF_HSTG);
    int* s_union = reinterpret_cast<int*>(smem + OFF_TAB + TAB_UNION);
    unsigned long long* s_mask = reinterpret_cast<unsigned long long*>(smem + OFF_TAB + TAB_MASK);
    int* s_slot = reinterpret_cast<int*>(smem + OFF_TAB + TAB_SLOT);
    float* s_tkw = reinterpret_cast<float*>(smem + OFF_TAB + TAB_TKW);
    const int tid = threadIdx.x, warp = tid >> 5;
    constexpr bool exact_shape = EXACT_M != 0;
    const int M = exact_shape ? EXACT_M : P.M;
    const int M_pad = exact_shape ? EXACT_M : P.M_pad;
    const bool is_fc2 = PARTS < 3 ? ((int)blockIdx.x < N_FC2_TILES * PARTS) : true;   // PARTS 3: every CTA owns an fc2 tile part

    // FMOE_FC2_PREFILL (exact M32): the FC2 ring has its own barrier objects, initialised here with FC1's (the consumer branch's
    // re-init of the FC1 objects after FC1 then touches dead barriers; kept for its code structure).
    constexpr bool PREFILL_FC2 = fc2_prefill<EXACT_M, PARTS>();
    uint64_t* full2 = PREFILL_FC2 ? reinterpret_cast<uint64_t*>(smem + OFF_BAR2) : full;
    uint64_t* empty2 = PREFILL_FC2 ? full2 + 8 : empty;
    if (tid < 8) mbar_init(&full[tid], 1u + FC1_GATHER_THREADS<EXACT_M>);  // one arrive from every activation-copy lane
    else if (tid < 16) mbar_init(&empty[tid - 8], (unsigned)(NCONS / 32));   // one arrive per consumer warp
    else if constexpr (PREFILL_FC2) {
        if (tid < 24) mbar_init(&full2[tid - 16], 3u);          // h/cs lane + weight lane + offset lane (split producer)
        else if (tid < 32) mbar_init(&empty2[tid - 24], fc2_lockstep<EXACT_M, PARTS>() ? 16u : 8u);   // one WG pair per fc2 stage
    }
    uint64_t* pbar = reinterpret_cast<uint64_t*>(smem + OFF_TAB + TABX_PBAR);   // FMOE_ROUTER_TMA prologue barriers: [0] slices + norm-w, [1] W tile
    if constexpr (FMOE_ROUTER_TMA) { if (tid == 0 && !P.pre_routed) { mbar_init(&pbar[0], 1u); mbar_init(&pbar[1], 1u); } }
    if constexpr (INPUT_TP) { if (tid >= 32 && tid < 32 + TP_NDEV) reinterpret_cast<uint4**>(smem + OFF_TAB + TABX_PEER)[tid - 32] = tp_pro<true>(P, tid - 32); }   // published by the barrier below
    if constexpr (FC1_PUB_OFFLOAD && EXACT_M == 32 && PARTS == 2) { if (tid == 40) { reinterpret_cast<int*>(smem + OFF_PUB)[0] = 0; reinterpret_cast<int*>(smem + OFF_PUB)[1] = 0; } }   // FC1 publish queue (scheduling slice)
    if (PREFILL_FC2 ? tid < 32 : tid == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    if (tid == 0) STAMP(P, 0);   // CTA start
    // Epoch tag for this launch: device counter (work[2]) + 1, bumped by the last CTA out, so graph
    // replays get a fresh tag with no host involvement (all ranks launch the same number of times).
    const unsigned epoch = (INPUT_TP && FMOE_TP_EPOCH_HINT) ? (unsigned)ld_evict_last_s32(&P.work[2]) + 1u : (unsigned)__ldg(&P.work[2]) + 1u;

    // ---------------- prologue: RMSNorm + quant -> router -> top-8 -> union tables ----------------
    if constexpr (INPUT_TP) {
        // FMOE_INPUT_TP: the input all-reduce is part of the prologue. Every CTA pushes its share of the local partial to the
        // slice owners (reduce-scatter); token CTA t reduces (token t, slice my_rank) in rank order, pushes the reduced slice + its two
        // sum-of-squares k-group partials to all ranks (all-gather) and runs stage 1 on the gathered row; the 12 router units of this
        // rank (k-groups 2r, 2r+1) take their B tile from the local all-gather messages and publish to the token owners (t % 8);
        // the owners run stage 3 for their 4 tokens and publish the lists to every rank; stage 4 is unchanged.
        static_assert(!INPUT_TP || (EXACT_M == TP_M && PARTS == 2 && FMOE_ROUTER_BF16 && !FMOE_ROUTER_TMA), "input-TP entry: exact M32, 96 FC2 CTAs, bf16 cp.async router");
        constexpr int grid = 132;
        const int b = (int)blockIdx.x;
        const int u0 = grid - 1 - b;                      // router units from the top of the grid
        constexpr bool ROUTER_TOKENS = FMOE_TP_ROUTER_TOKENS != 0;
        constexpr int NUNITS = ROUTER_TOKENS ? RUNITS_MAX : TP_NUNITS;   // B: 96 units (all k-groups) for this rank's 4 tokens; A: 12 units for all tokens
        const bool router = u0 < NUNITS;
        const int kg = ROUTER_TOKENS ? u0 % RKG : 2 * P.my_rank + (u0 & 1), rg = ROUTER_TOKENS ? u0 / RKG : u0 >> 1;
        const int unit = kg + RKG * rg;                   // unit numbering of the full-K prologue (stage 3's s_part index)
        (void)grid;   // the reduce-scatter pushes (tp_rs_push) were issued at kernel entry, before the barrier init
        if constexpr (FMOE_TP_ACQ_EARLY) if (FMOE_TP_ACQ_EARLY == 1 || !router) {   // tensormap acquire fences at the very start (the producer warps idle through
            // the reduce anyway): they can then not land on a stage-3 barrier. Value 2: router CTAs keep the late acquires (their consumer
            // warps' gather barrier waited for the fences: stamps13 partials gathered 7.57 vs 6.34)
            if (warp == PROD_WARP && (tid & 31) < 2) { if (FMOE_FC1_ACQUIRE_EARLY) tma_acquire_map(P.tmaps + (tid & 31)); }
            if constexpr (fc2_split_producer<EXACT_M, PARTS>()) {
                if (FMOE_FC2_ACQUIRE_EARLY && (warp == PROD_WARP + 1 || warp == PROD_WARP + 2) && (tid & 31) == 0) tma_acquire_map(P.tmaps + 1 + (warp - PROD_WARP));
            }
        }
        // Roles: CTA t < 32 reduces (token t, slice r) and all-gathers it; the 4 owner CTAs (t % 8 == r) then go straight to stage 3 (their
        // partials arrive ~1 us after the all-gather lands) while CTAs 32..35 run stage 1 of the owned tokens; the other token CTAs run their
        // own stage 1 (design B, validation-tp-stamps3: with stage 1 in front, the owner started gathering 1.3 us after the routers published).
        const bool owner = ROUTER_TOKENS && b < M && (b % TP_NDEV) == P.my_rank;
        const bool spare = ROUTER_TOKENS && b >= M && b < M + TP_TOK_PER_RANK;
        const int t1 = spare ? P.my_rank + TP_NDEV * (b - M) : b;   // token whose stage 1 this CTA runs
        if (router) {   // input-independent loads at kernel start
            if constexpr (ROUTER_TOKENS) issue_slice_loads_tp4(P, smem, kg); else issue_slice_loads_tp(P, smem, kg, NT, M);
            if constexpr (ROUTER_TOKENS && FMOE_TP_OWNER_REDUCE) {
                // Only the producer warpgroup waits out the delay and issues the weight tile (24 cp.asyncs per thread); the 16 consumer warps
                // go straight to the local slice / p gather, whose inputs now land at ~3 us. (With all 640 threads spinning through the
                // 2.8-us delay the routers' inputs waited behind it: stamps9 slice+rstd 4.74 for data that had landed at ~3.2.)
                if (warp >= PROD_WARP) {
                    if constexpr (FMOE_TP_W_DELAY_NS > 0) { const unsigned long long t0_ = globaltimer_ns(); while (globaltimer_ns() - t0_ < FMOE_TP_W_DELAY_NS) {} }
                    issue_w_load_pg(P, smem, kg, rg);
                }
            } else {
                if constexpr (FMOE_TP_W_DELAY_NS > 0) {   // let the reduce-scatter messages land before the 4.7 MB weight stream hits HBM / L2
                    const unsigned long long t0_ = globaltimer_ns();
                    while (globaltimer_ns() - t0_ < FMOE_TP_W_DELAY_NS) {}
                }
                issue_w_load(P, smem, kg, rg);
            }
        }
        if (FMOE_PROLOGUE_TIGHT && owner) stage3_prefetch_bias(P, smem);   // its smem slot is disjoint from the reduce staging; published by the barriers below
        if (b < M) {   // reduce-scatter: owner-reduce -> (owned token r + 8 (b / 8), slice b % 8); else (token b, slice r); then the all-gather push
            if constexpr (FMOE_TP_OWNER_REDUCE) tp_reduce_token(P, epoch, P.my_rank + TP_NDEV * (b >> 3), b & 7, smem);
            else tp_reduce_token(P, epoch, b, P.my_rank, smem);
        }
        if ((b < M && !owner) || spare) stage1_token_tp(P, epoch, t1, smem);   // stage 1 on the gathered row
        if (tid == 0) STAMP(P, 6);
        if (router) {
            if constexpr (ROUTER_TOKENS) {
                if constexpr (FMOE_TP_ROUTER_RS) router_gather_rs_tp4(P, epoch, smem, kg);   // reduce the k-group from the RS messages, exchange p locally
                else router_gather_tp4(P, epoch, smem, kg);     // the 4 owned tokens' k-group slice (local AG messages) -> B tile rows 0..3; p messages -> rstd
                if (tid == 0) STAMP(P, 14);                // router: slice + rstd known
                cp_async_wait_all();                       // residual slices, norm-w slice, weight tile landed
                __syncthreads();
                if (tid == 0) STAMP(P, 19);                // router: W / residual / norm-w tiles landed (cp.async drained)
                router_normalize(P, smem, 8, TP_TOK_PER_RANK);
                __syncthreads();                           // normalized B tile visible to every warpgroup
                if (tid == 0) STAMP(P, 12);
                router_compute<8, EXACT_M>(P, epoch, smem, unit, TP_TOK_PER_RANK);   // partial logits of rows 0..3 -> the LOCAL RPART rows (stage 3 reads row t / 8)
                if constexpr (FMOE_TP_FC1_PREFETCH_KT > 0) {   // HBM is idle from here until the first FC1 tiles: warm L2 with k-tiles 0..N-1 of experts u0 + 96 j
                    if (warp == PROD_WARP + 3 && (tid & 31) == 0)
                        for (int kt = 0; kt < FMOE_TP_FC1_PREFETCH_KT; ++kt)
                            for (int e = u0; e < P.n_exp; e += RUNITS_MAX) {
                                const uint8_t* w = P.fc1_w + (size_t)e * FC1_W_EXPERT_BYTES + (size_t)kt * 4 * FC1_W_SLICE_BYTES;
                                bulk_prefetch_l2(w, 2 * FC1_W_SLICE_BYTES);
                                bulk_prefetch_l2(w + 2 * FC1_W_SLICE_BYTES, 2 * FC1_W_SLICE_BYTES);
                                bulk_prefetch_l2(P.fc1_s + (size_t)e * FC1_S_EXPERT_BYTES + (size_t)kt * 4 * FC1_S_SLICE_BYTES, 4 * FC1_S_SLICE_BYTES);
                            }
                }
            } else {
                router_gather_tp(P, epoch, smem, NT, M, u0 & 1);   // this k-group's half of the reduced slice (local AG messages) -> B tile; p messages -> rstd
                if (tid == 0) STAMP(P, 14);                // router: slice + rstd known
                cp_async_wait_all();                       // residual slice, norm-w slice, weight tile landed
                __syncthreads();
                if (tid == 0) STAMP(P, 19);                // router: W / residual / norm-w tiles landed (cp.async drained)
                router_normalize(P, smem, NT, M);
                __syncthreads();                           // normalized B tile visible to every warpgroup
                if (tid == 0) STAMP(P, 12);
                router_compute<NT, EXACT_M, true>(P, epoch, smem, unit, M);   // partial logits -> the token owners' RPART buffers
            }
        }
        if (tid == 0) STAMP(P, 7);
        if (FMOE_TP_ACQ_EARLY == 0 || (FMOE_TP_ACQ_EARLY == 2 && router)) {
        if (warp == PROD_WARP && (tid & 31) < 2) {   // tensor-map acquire fences in the message-wait window (as the full-K prologue)
            if (FMOE_FC1_ACQUIRE_EARLY) tma_acquire_map(P.tmaps + (tid & 31));
            if constexpr (fc2_prefill<EXACT_M, PARTS>() && FMOE_FC2_TAIL_PREFETCH > 0) { if ((tid & 31) == 0) tma_acquire_map(P.tmaps + 2); }
            if (FMOE_FC2_ACQUIRE_EARLY && !fc2_split_producer<EXACT_M, PARTS>() && (tid & 31) == 0) { tma_acquire_map(P.tmaps + 2); tma_acquire_map(P.tmaps + 3); }
        }
        if constexpr (fc2_split_producer<EXACT_M, PARTS>()) {
            if (FMOE_FC2_ACQUIRE_EARLY && (warp == PROD_WARP + 1 || warp == PROD_WARP + 2) && (tid & 31) == 0) tma_acquire_map(P.tmaps + 1 + (warp - PROD_WARP));
        }
        }
        if constexpr (!ROUTER_TOKENS) { if (FMOE_PROLOGUE_TIGHT && b < M && (b % TP_NDEV) == P.my_rank) stage3_prefetch_bias(P, smem); }
        if (b < M && (b % TP_NDEV) == P.my_rank) {   // this rank's 4 tokens
            if constexpr (FMOE_TP_S3_FENCE) fence_acq_rel_sys();   // drain this thread's remote stores here (idle window), not inside stage 3
            __syncthreads();
            if (tid == 0) STAMP(P, 26);   // TP diag: owner CTA enters stage 3 (after the fence + barrier)
            if constexpr (FMOE_TP_S3_DRY) { stage3_token<true, true, true>(P, epoch, smem, b, NRG_MAX); __syncthreads(); }   // warm-up pass, no gather / publish
            if constexpr (FMOE_TP_S3_NOINLINE) stage3_token_tp_call(P, epoch, smem, b, NRG_MAX);
            else stage3_token<true, true>(P, epoch, smem, b, NRG_MAX);
        }
        if (tid == 0) STAMP(P, 8);
        __syncthreads();
        stage4_build_tables<EXACT_M, PARTS, true>(P, epoch, smem, s_union, s_mask, s_slot, s_tkw, s_misc, M);
    } else if (!P.pre_routed) {
        const int grid = exact_shape ? 132 : (int)gridDim.x, b = (int)blockIdx.x;
        const int n_rg = (P.n_exp + RROWS - 1) / RROWS, n_units = RKG * n_rg;
        const int u0 = grid - 1 - b;   // router units are taken from the top of the grid
        const bool router = u0 < n_units;
        // stage-1 tokens go round-robin to the NON-router CTAs (36 at grid 132: up to 2 tokens each, all off the critical
        // path) so no router unit waits behind a token; only tiny grids (< 96 + 1 CTAs) get dual-role CTAs.
        const int n_tok_cta = grid > n_units ? grid - n_units : grid;
        if (router) {   // the router path depends only on these loads (hidden/residual/norm-weight slices, then the weights)
            if constexpr (FMOE_ROUTER_TMA) {
                if (tid == 0) {
                    if constexpr (FMOE_ROUTER_TMA_DELAY_NS > 0) { const unsigned long long t0_ = globaltimer_ns(); while (globaltimer_ns() - t0_ < FMOE_ROUTER_TMA_DELAY_NS) {} }
                    issue_unit_slices(P, smem, u0 % RKG, NT, pbar);
                    if constexpr (FMOE_ROUTER_TMA_W_MODE == 0) issue_unit_weight(P, smem, u0 % RKG, u0 / RKG, pbar);   // else: in the unit loop
                }
            } else { issue_slice_loads(P, smem, u0 % RKG, NT, M); issue_w_load(P, smem, u0 % RKG, u0 / RKG); }
        }
        if (b < n_tok_cta) for (int t = b; t < M; t += n_tok_cta) stage1_token(P, epoch, t, smem);   // fp8 x / residual_out for fc1
        if (tid == 0) STAMP(P, 6);
        if (router) {
            bool have_rstd = false;
            unsigned pphase = 0;   // FMOE_ROUTER_TMA: parity of the prologue barriers for this unit
            for (int u = u0; u < n_units; u += grid) {
                if (u != u0) {
                    if constexpr (FMOE_ROUTER_TMA) {   // the tiles were read (psum overlay, fragments) and written (normalize) through the generic proxy
                        fence_proxy_async_smem();
                        __syncthreads();
                        if (tid == 0) {
                            issue_unit_slices(P, smem, u % RKG, NT, pbar);
                            if constexpr (FMOE_ROUTER_TMA_W_MODE == 0) issue_unit_weight(P, smem, u % RKG, u / RKG, pbar);
                        }
                    } else { __syncthreads(); issue_slice_loads(P, smem, u % RKG, NT, M); issue_w_load(P, smem, u % RKG, u / RKG); }
                }
                // FMOE_ROUTER_TMA_W_MODE 1 / 2: warp 1 lane 0 fires the weight box once the slices are in / once the rstd poll is over as well
                if constexpr (FMOE_ROUTER_TMA && FMOE_ROUTER_TMA_W_MODE == 1) { if (tid == 32) { mbar_wait(&pbar[0], pphase); issue_unit_weight(P, smem, u % RKG, u / RKG, pbar); } }
                // FMOE_RSTD_DIRECT: the M rstd messages are polled BEFORE the slice wait (they need no slice data; the token CTAs publish
                // at ~1.6 us), so the normalize can start as soon as both are in.
                if constexpr (FMOE_RSTD_DIRECT) router_rstd(P, epoch, smem, u % RKG, NT, M, !have_rstd);
                if constexpr (FMOE_ROUTER_TMA && FMOE_ROUTER_TMA_W_MODE == 2) { if (tid == 32) { mbar_wait(&pbar[0], pphase); issue_unit_weight(P, smem, u % RKG, u / RKG, pbar); } }
                if constexpr (FMOE_ROUTER_TMA) mbar_wait(&pbar[0], pphase);   // hidden / residual / norm-w slices landed (visible to every waiting thread)
                else { cp_async_wait_group1(); __syncthreads(); }   // slices landed (the weight tile may still be in flight)
                if constexpr (!FMOE_RSTD_DIRECT) router_rstd(P, epoch, smem, u % RKG, NT, M, !have_rstd);
                have_rstd = true;
                router_normalize(P, smem, NT, M);
                if constexpr (FMOE_ROUTER_TMA) { mbar_wait(&pbar[1], pphase); pphase ^= 1u; }   // weight tile landed
                else cp_async_wait_all();
                __syncthreads();   // normalized B tile visible to every warpgroup
                if (tid == 0 && u == u0) STAMP(P, 12);   // weight tile landed, B tile normalized
                router_compute<NT, EXACT_M>(P, epoch, smem, u, M);
            }
        }
        if constexpr (FMOE_ROUTER_TMA) { if (tid == 0) { mbar_inval(&pbar[0]); mbar_inval(&pbar[1]); } }   // stage 4 reuses the table region
        if (tid == 0) STAMP(P, 7);
        // Token CTAs now wait ~5 us for router partials and router CTAs ~4 us for the top-8 lists: the FC2 producer
        // lane can run its two tensor-map acquire fences (~1 us each, per thread) inside that window instead of on
        // every CTA's FC2 start. (Only this lane's own maps: the fence is per thread.)
        if (warp == PROD_WARP && (tid & 31) < 2) {
            if (FMOE_FC1_ACQUIRE_EARLY) tma_acquire_map(P.tmaps + (tid & 31));   // FC1 maps 0/1: one per issuing lane, needed first
            if constexpr (fc2_prefill<EXACT_M, PARTS>() && FMOE_FC2_TAIL_PREFETCH > 0) { if ((tid & 31) == 0) tma_acquire_map(P.tmaps + 2); }   // lane 0 also prefetches FC2 weight boxes
            if (FMOE_FC2_ACQUIRE_EARLY && !fc2_split_producer<EXACT_M, PARTS>() && (tid & 31) == 0) { tma_acquire_map(P.tmaps + 2); tma_acquire_map(P.tmaps + 3); }
        }
        if constexpr (fc2_split_producer<EXACT_M, PARTS>()) {   // the FC2 weight / offset producer warps acquire their own maps here
            if (FMOE_FC2_ACQUIRE_EARLY && (warp == PROD_WARP + 1 || warp == PROD_WARP + 2) && (tid & 31) == 0) tma_acquire_map(P.tmaps + 1 + (warp - PROD_WARP));
        }
        if (FMOE_PROLOGUE_TIGHT && b < M) stage3_prefetch_bias(P, smem);   // published by the __syncthreads below
        for (int t = b; t < M; t += grid) { __syncthreads(); stage3_token<(EXACT_M == 32 && PARTS == 2)>(P, epoch, smem, t, n_rg); }
        if (tid == 0) STAMP(P, 8);
        if constexpr (FMOE_FC1_L2_PREFETCH_KT > 0 && EXACT_M == 32) {
            // HBM is idle from the top-8 (~9 us) until the first FC1 TMA (~16 us) while the CTAs build the union tables. Which experts
            // are routed is unknown, but their first k-tiles are: warm L2 with k-tiles 0..KT-1 of every expert (KT x 34 KB x n_exp,
            // 25.5 MB at KT=2) so the 132-CTA first-tile burst (13.5 MB, ~1.2 us on the critical path) becomes L2 hits. (Issued right
            // after STAMP 7 instead -- during the router-weight loads of the slower CTAs -- it cost +1.8 us on every case.)
            if (warp == PROD_WARP + 3 && (tid & 31) == 0) {
                for (int kt = 0; kt < FMOE_FC1_L2_PREFETCH_KT; ++kt)
                    for (int e = (int)blockIdx.x; e < P.n_exp; e += 132) {
                        const uint8_t* w = P.fc1_w + (size_t)e * FC1_W_EXPERT_BYTES + (size_t)kt * 4 * FC1_W_SLICE_BYTES;
                        bulk_prefetch_l2(w, 2 * FC1_W_SLICE_BYTES);
                        bulk_prefetch_l2(w + 2 * FC1_W_SLICE_BYTES, 2 * FC1_W_SLICE_BYTES);
                        bulk_prefetch_l2(P.fc1_s + (size_t)e * FC1_S_EXPERT_BYTES + (size_t)kt * 4 * FC1_S_SLICE_BYTES, 4 * FC1_S_SLICE_BYTES);
                    }
            }
        }
        __syncthreads();
        stage4_build_tables<EXACT_M, PARTS>(P, epoch, smem, s_union, s_mask, s_slot, s_tkw, s_misc, M);
    } else {
        build_tables_prerouted(P, s_union, s_mask, s_slot, s_tkw, s_misc, M);
    }
    __syncthreads();
    // FMOE_EARLY_FENCE_HOIST: every thread orders its own generic-proxy writes to the ring region (router tiles, stage-3/4 scratch) ahead of the
    // async-proxy TMA writes of the early tiles here, instead of one thread fencing on the issue chain after the task table.
    if constexpr (FMOE_EARLY_FENCE_HOIST && FMOE_EARLY_FIRST_TILE > 0) fence_proxy_async_smem();
    if (tid == 0) STAMP(P, 9);   // routing tables ready: fc1 starts
    static_assert(!(INPUT_TP && FMOE_TP_TASK_NOINIT) || (FMOE_TASK_TABLE_FUSED && !FMOE_ONEWAVE_KSPLIT && !FMOE_MIXED_WIDTH), "TP task-table init moved into the folded stage 4");
    build_m32_token_tasks<EXACT_M, PARTS, INPUT_TP && FMOE_TP_S4_FOLD>(P, smem, s_misc);
    if (tid == 0) STAMP(P, 13);   // phase build: task table built (router-internal stamp 13 repurposed)
    constexpr bool PRE_SPLIT = FMOE_EARLY_PRE_SPLIT || (INPUT_TP && FMOE_TP_EARLY_PRE_SPLIT);
    if constexpr (FMOE_EARLY_FIRST_TILE > 0 && PRE_SPLIT && EXACT_M == 32 && PARTS == 2 && KSPLIT == 1) {
        const int ft = s_misc[60];   // this CTA's first token task (build_m32_token_tasks), -1 on the legacy schedule / no task
        if (ft >= 0) {
            const int u = ft >> 8, c = (ft >> 1) & 0x7f, hf = ft & 1;
            if (warp == PROD_WARP) fc1_early_weights(P, smem, full, s_union[u], hf, true);
            else if (warp == PROD_WARP + 1) {
                if (s_misc[58]) fc1_early_gather<16>(P, smem, full, p_tok, s_mask[u], c * 16);
                else fc1_early_gather<8>(P, smem, full, p_tok, s_mask[u], c * 8);
            }
        }
    }
    constexpr int FC1_WIDTH = EXACT_M == 64 ? 16 : ((EXACT_M == 24 || EXACT_M == 32) ? 8 : 16);
    // N16 instantiation: every CTA of a one-wave load, or the last misc[59] CTAs (hot experts) of a mixed two-wave load.
    const bool wide_m32_fc1 = m32_token_tasks<EXACT_M, PARTS>(P, s_misc) &&
                              (s_misc[58] || (int)blockIdx.x >= 132 - s_misc[59]);

    // Register redistribution (whole warpgroups): the producer warpgroup shrinks, the consumer warpgroups grow so the
    // fc1/fc2 wgmma pipelines (accumulators + double-buffered A fragments) fit without ptxas serializing the wgmmas.
    // The two roles stay in disjoint branches from here on (ptxas assigns the register budget per branch); the phase
    // boundaries are named barriers over all NTHREADS that both branches hit the same number of times.
    const bool use_fc2_helpers = m32_fc2_helpers<EXACT_M, PARTS>(P, s_misc[32], s_misc);
    const bool run_fc2 = (is_fc2 || use_fc2_helpers) && !(P.mode & 4);
    if (warp >= PROD_WARP) {
        // Exact M64 FC1 uses N16, so give its gather producer64 registers.
        // FC2 retains the original112/32 budget after the full-CTA join.
        setmaxnreg_dec<EXACT_M == 64 ? 64 : RegSplit<NT>::producer>();
        if (tid == PROD_WARP * 32) STAMP(P, 18);   // phase build: producer warps past the register split
        if constexpr (PREFILL_FC2) {   // idle warp 19 builds the joint FC2 plan during FC1; fc1_producer's barrier 3 publishes it
            if (warp == PROD_WARP + 3 && (tid & 31) == 0) build_m32_joint_plan<EXACT_M, PARTS, true>(s_misc);
        }
        if constexpr (EXACT_M == 32 && PARTS == 2 && FC1_PUB_OFFLOAD) {   // idle warp 18: FC1 factor table + h-ready publishes of the N8 token path
            if (warp == PROD_WARP + 2 && !wide_m32_fc1 && m32_token_tasks<EXACT_M, PARTS>(P, s_misc))
                fc1_pub_worker(P, epoch, smem, s_union, s_misc, tid & 31);
        }
        if constexpr (EXACT_M == 32 && PARTS == 2 && FMOE_GP && FMOE_GP2) {   // FMOE_GP2: idle warp 19 builds this CTA's FC2 plan during FC1
            if (warp == PROD_WARP + 3 && (tid & 31) == 0 && m32_gp2_active<EXACT_M, PARTS>(P, s_misc[32], s_misc))   // (published by the join)
                m32_gp2_plan(s_misc[32], s_misc[38], (int)blockIdx.x, s_misc);
        }
        if constexpr (EXACT_M == 32 && PARTS == 2) {
            if (wide_m32_fc1) fc1_producer<NT, PARTS, EXACT_M, 16, false, PRE_SPLIT>(P, smem, full, empty, p_tok, s_misc);
            else if (m32_token_tasks<EXACT_M, PARTS>(P, s_misc))
                fc1_producer<NT, PARTS, EXACT_M, 8, true, PRE_SPLIT>(P, smem, full, empty, p_tok, s_misc);
            else fc1_producer<NT, PARTS, EXACT_M, 8>(P, smem, full, empty, p_tok, s_misc);
        } else fc1_producer<NT, PARTS, EXACT_M>(P, smem, full, empty, p_tok, s_misc);
        // FMOE_FC2_PREFILL: no join on the producer side. The FC2 producer warps leave fc1_producer at its sentinel (joint plan
        // published by barrier 3), wait on FC2_START_BAR for the consumers' h publish inside fc2_producer and fill the FC2 ring's
        // own barriers while the consumers run their join / re-init / plan chain. Only the final drain barrier remains.
        if constexpr (!PREFILL_FC2) named_bar_sync(0, NTHREADS);
        if (run_fc2) {
            if constexpr (!PREFILL_FC2) {
                if constexpr (EXACT_M == 64) setmaxnreg_dec<32>();
                named_bar_sync(0, NTHREADS);   // fc2 barriers re-initialized by the consumer branch
            }
            fc2_producer<NT, PARTS, EXACT_M>(P, epoch, smem, full2, empty2);
            if constexpr (EXACT_M == 32 && PARTS == 2)
                named_bar_sync(0, NTHREADS);   // drain all roles before CTA completion
        }
    } else {
        setmaxnreg_inc<EXACT_M == 64 ? 104 : RegSplit<NT>::consumer>();
        if (tid == 0) STAMP(P, 17);   // phase build: consumer warps past the register split
        if constexpr (EXACT_M == 32 && PARTS == 2) {
            if (wide_m32_fc1) fc1_consumer<16, PARTS, true, EXACT_M>(P, epoch, smem, full, empty, s_tok, s_inv, s_misc, xp_s, hst, M, M_pad);
            else if (m32_token_tasks<EXACT_M, PARTS>(P, s_misc))
                fc1_consumer<8, PARTS, true, EXACT_M, true>(P, epoch, smem, full, empty, s_tok, s_inv, s_misc, xp_s, hst, M, M_pad);
            else fc1_consumer<8, PARTS, true, EXACT_M>(P, epoch, smem, full, empty, s_tok, s_inv, s_misc, xp_s, hst, M, M_pad);
        } else if constexpr (exact_shape) {
            fc1_consumer<FC1_WIDTH, PARTS, true, EXACT_M>(P, epoch, smem, full, empty, s_tok, s_inv, s_misc, xp_s, hst, M, M_pad);
        } else {
            fc1_consumer<NT, PARTS>(P, epoch, smem, full, empty, s_tok, s_inv, s_misc, xp_s, hst, M, M_pad);
        }
        // FMOE_FC2_PREFILL: each consumer warp releases the FC2 producers as soon as its own epilogue (h/cs/flag publish) is done --
        // the gate completes with the last warp -- then the consumers join among themselves (barrier 5) and keep the rest of this
        // chain (tid-0 plan rebuild with identical values, re-init of the now-dead FC1 barrier objects) for its code structure.
        if constexpr (PREFILL_FC2) { if (run_fc2) named_bar_arrive(FC2_START_BAR, FC2_START_THREADS); named_bar_sync(CONS_JOIN_BAR, NCONS); }
        else named_bar_sync(0, NTHREADS);
        if (run_fc2) {
            if constexpr (EXACT_M == 64) {
                static_assert(104 * NCONS + 64 * 128 == 96 * NTHREADS &&
                              112 * NCONS + 32 * 128 == 96 * NTHREADS,
                              "M64 phase budgets fit the same CTA pool");
                setmaxnreg_inc<112>();
            }
            build_m32_joint_plan<EXACT_M, PARTS>(s_misc);
            if constexpr (EXACT_M == 32 && PARTS == 2) {
                // All640 threads joined after FC1.
                // No async operation still uses these objects; invalidate before init.
                if (tid < 16) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                                          :: "r"(smem_u32(full + tid)) : "memory");
            }
            if (tid < 8) mbar_init(&full[tid], EXACT_M == 24 ? 2u : (fc2_split_producer<EXACT_M, PARTS>() ? 3u : 1u));
            else if (tid < 16) mbar_init(&empty[tid - 8], fc2_lockstep<EXACT_M, PARTS>() ? 16u : 8u);   // lockstep: every consumer warp; else one WG pair (8 warps) per fc2 stage
            if constexpr (EXACT_M == 32 && PARTS == 2) {
                if (tid < 16) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            } else {
                if (tid == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
            if constexpr (PREFILL_FC2) named_bar_sync(CONS_JOIN_BAR, NCONS); else named_bar_sync(0, NTHREADS);
            fc2_consumer<NT, PARTS, EXACT_M, INPUT_TP, INPUT_TP && FMOE_NORM_NEXT>(P, epoch, smem, full2, empty2, reinterpret_cast<float*>(smem + Fc2Geom<NT, EXACT_M>::OUTS), M);
            if constexpr (EXACT_M == 32 && PARTS == 2)
                named_bar_sync(0, NTHREADS);
        }
        if constexpr (INPUT_TP && FMOE_NORM_NEXT) {   // the row CTAs (FC1-only / helper CTAs, their producer warps have left) run the
            if (P.norm_next && (int)blockIdx.x >= NN_ROW_CTA0 && (int)blockIdx.x < NN_ROW_CTA0 + TP_M)   // next layer's norm + quant on the reduced row
                norm_next_row(P, epoch, smem, (int)blockIdx.x - NN_ROW_CTA0);
        }
        if (tid == 0) {   // last CTA out resets the queue counters for the next launch (stream order makes this safe)
            if constexpr (INPUT_TP && FMOE_TP_EPOCH_HINT) {   // keep the counter line at L2::evict_last priority for the next launch's entry read
                if (atom_add_evict_last_s32(&P.work[1], 1) == (int)gridDim.x - 1) {
                    atom_exch_evict_last_s32(&P.work[0], 0); atom_exch_evict_last_s32(&P.work[3], 0); atom_exch_evict_last_s32(&P.work[1], 0); atom_add_evict_last_s32(&P.work[2], 1);
                }
            } else
            if (atomicAdd(&P.work[1], 1) == (int)gridDim.x - 1) { atomicExch(&P.work[0], 0); atomicExch(&P.work[3], 0); atomicExch(&P.work[1], 0); atomicAdd(&P.work[2], 1); }
        }
    }
}

template <int NT, int PARTS, int EXACT_M = 0, bool INPUT_TP = false>
void launch(const Params& P, int grid, cudaStream_t stream) {
    static bool configured[16] = {false};
    int dev = 0; cudaGetDevice(&dev);
    if (!configured[dev]) {
        cudaFuncSetAttribute(fused_moe_kernel<NT, PARTS, EXACT_M, INPUT_TP>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC);
        configured[dev] = true;
    }
    fused_moe_kernel<NT, PARTS, EXACT_M, INPUT_TP><<<grid, NTHREADS, SMEM_ALLOC, stream>>>(P);
}
template <int NT>
void launch_parts(const Params& P, int grid, cudaStream_t stream) {
    if (P.parts == 3) launch<NT, 3>(P, grid, stream); else if (P.parts == 2) launch<NT, 2>(P, grid, stream); else launch<NT, 1>(P, grid, stream);
}

// Fields shared by both entry points (weights, staging, all-reduce, queue, diagnostics) + the launch.
// TMA tensor-map encoding through the driver entry point (no link against libcuda). Maps are cached per (base address, kind):
// the weights of a layer never move, and an eager launch must not pay 4 driver calls.
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_tiled_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled not available from the driver");
        fn = reinterpret_cast<EncodeTiledFn>(p);
    }
    return fn;
}
// rank-D tiled map: dims (elements, innermost first), strides (bytes, dims 1..rank-1), box (elements), no swizzle.
// L2 promotion 128 B, not 256 B: with the 256-B promotion the HBM stream of these boxes (2-KB pieces + 128-B offset rows)
// tops out at 3.95 / 4.25 TB/s (w13+s13 / w2+s2 pairs, 132 CTAs, K=6) against 4.35 / 4.65 with 128 B or none
//; the kernel's FC1 wave-1 rate (4.16) sat on the 256-B ceiling, not on HBM.
static void encode_map(CUtensorMap* m, const void* base, CUtensorMapDataType dt, uint32_t rank, const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                       CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    const cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    CUresult r = encode_tiled_fn()(m, dt, rank, const_cast<void*>(base), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
}

// Expert weights in Humming's layout (see the constants): w13 = (fc1_w, fc1_s, fc1_s2), w2 = (fc2_w, fc2_s, fc2_s2); any dtype for the
// packed words / offsets (Humming hands over int32 / float8_e8m0fnu views), fp32 factors. Returns the number of experts covered.
static int set_weights(Params& P, torch::Tensor fc1_w, torch::Tensor fc1_s, torch::Tensor fc1_s2, torch::Tensor fc2_w, torch::Tensor fc2_s, torch::Tensor fc2_s2) {
    TORCH_CHECK(fc1_w.is_cuda() && fc1_w.is_contiguous() && fc1_s.is_contiguous() && fc1_s2.is_contiguous() && fc2_w.is_contiguous() && fc2_s.is_contiguous() && fc2_s2.is_contiguous(),
                "expert weights must be contiguous CUDA tensors");
    TORCH_CHECK(fc1_s2.scalar_type() == torch::kFloat32 && fc2_s2.scalar_type() == torch::kFloat32, "per-expert factors must be fp32");
    const int64_t n1 = fc1_w.numel() * fc1_w.element_size() / FC1_W_EXPERT_BYTES, n1s = fc1_s.numel() * fc1_s.element_size() / FC1_S_EXPERT_BYTES;
    const int64_t n2 = fc2_w.numel() * fc2_w.element_size() / FC2_W_EXPERT_BYTES, n2s = fc2_s.numel() * fc2_s.element_size() / FC2_S_EXPERT_BYTES;
    const int64_t n = std::min(std::min(std::min(n1, n1s), std::min(n2, n2s)), std::min(fc1_s2.numel(), fc2_s2.numel()));
    TORCH_CHECK(n >= 1, "expert weights too small (need [E][K/32][4N] words, [E][K/32][N] offsets, [E] factors for w13 and w2)");
    P.fc1_w = static_cast<const uint8_t*>(fc1_w.data_ptr()); P.fc1_s = static_cast<const uint8_t*>(fc1_s.data_ptr()); P.fc1_s2 = fc1_s2.data_ptr<float>();
    P.fc2_w = static_cast<const uint8_t*>(fc2_w.data_ptr()); P.fc2_s = static_cast<const uint8_t*>(fc2_s.data_ptr()); P.fc2_s2 = fc2_s2.data_ptr<float>();
    TORCH_CHECK((reinterpret_cast<uint64_t>(P.fc1_w) | reinterpret_cast<uint64_t>(P.fc1_s) | reinterpret_cast<uint64_t>(P.fc2_w) | reinterpret_cast<uint64_t>(P.fc2_s)) % 16 == 0,
                "weight tensors must be 16-B aligned (TMA)");
    {   // the 4 maps of this weight set live in a 512-B device buffer, encoded once per weight set and cached by the four tensor
        // addresses + sizes (ALL of them: a key on two of the pointers left maps pointing at freed scale tensors when the allocator
        // recycled addresses between test cases -> NaNs); their outer dimension covers exactly the experts present (n1..n2s per tensor)
        static std::unordered_map<std::string, torch::Tensor> cache;
        const std::string key = std::to_string(reinterpret_cast<uint64_t>(P.fc1_w)) + "/" + std::to_string(reinterpret_cast<uint64_t>(P.fc1_s)) + "/" +
                                std::to_string(reinterpret_cast<uint64_t>(P.fc2_w)) + "/" + std::to_string(reinterpret_cast<uint64_t>(P.fc2_s)) + "/" +
                                std::to_string(n1) + "/" + std::to_string(n1s) + "/" + std::to_string(n2) + "/" + std::to_string(n2s);
        auto it = cache.find(key);
        if (it == cache.end()) {
            alignas(64) CUtensorMap maps[4];
            const cuuint64_t d_w13[5] = {256, 2, 2, 2, (cuuint64_t)n1 * NKT1 * 4};
            const cuuint64_t s_w13[4] = {HL_BLOCK_BYTES, 2 * HL_BLOCK_BYTES, 4 * HL_BLOCK_BYTES, FC1_W_SLICE_BYTES};
            const cuuint32_t b_w13[5] = {256, 2, 1, 2, 4};
            const cuuint64_t d_s13[4] = {128, 2, 2, (cuuint64_t)n1s * NKT1 * 4}, s_s13[3] = {128, 256, FC1_S_SLICE_BYTES};
            const cuuint32_t b_s13[4] = {128, 1, 2, 4};
            const cuuint64_t d_w2[3] = {256, DIM / 64, (cuuint64_t)n2 * (INTER / 32)}, s_w2[2] = {HL_BLOCK_BYTES, FC2_W_SLICE_BYTES};
            const cuuint32_t b_w2[3] = {256, 2, 8};
            const cuuint64_t d_s2[2] = {DIM, (cuuint64_t)n2s * (INTER / 32)}, s_s2[1] = {FC2_S_SLICE_BYTES};
            const cuuint32_t b_s2[2] = {128, 8};
            encode_map(&maps[0], P.fc1_w, CU_TENSOR_MAP_DATA_TYPE_INT32, 5, d_w13, s_w13, b_w13);
            encode_map(&maps[1], P.fc1_s, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, d_s13, s_s13, b_s13);
            encode_map(&maps[2], P.fc2_w, CU_TENSOR_MAP_DATA_TYPE_INT32, 3, d_w2, s_w2, b_w2);
            encode_map(&maps[3], P.fc2_s, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_s2, s_s2, b_s2);
            torch::Tensor buf = torch::empty({(int64_t)sizeof(maps)}, torch::TensorOptions().dtype(torch::kUInt8).device(fc1_w.device()));
            // Initialize descriptors on PyTorch's current stream, respecting allocator reuse.
            // Cache misses must be warmed before capture. Synchronize this one-time upload
            // so stack-backed maps stay alive and the cached buffer is ready on any stream.
            auto map_stream = at::cuda::getCurrentCUDAStream();
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            cudaError_t map_error = cudaStreamIsCapturing(map_stream, &capture_status);
            TORCH_CHECK(map_error == cudaSuccess, "TMA capture query failed: ", cudaGetErrorString(map_error));
            TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                        "Warm up each fused MoE weight set before CUDA Graph capture");
            map_error = cudaMemcpyAsync(buf.data_ptr(), maps, sizeof(maps), cudaMemcpyHostToDevice, map_stream);
            TORCH_CHECK(map_error == cudaSuccess, "TMA descriptor upload failed: ", cudaGetErrorString(map_error));
            map_error = cudaStreamSynchronize(map_stream);
            TORCH_CHECK(map_error == cudaSuccess, "TMA descriptor initialization failed: ", cudaGetErrorString(map_error));
            it = cache.emplace(key, buf).first;
        }
        P.tmaps = reinterpret_cast<const CUtensorMap*>(it->second.data_ptr());
    }
    return (int)n;
}

static void fill_common_and_launch(Params& P, int M,
                                   torch::Tensor h_buf, torch::Tensor cs_buf, torch::Tensor h_flags, torch::Tensor rs_ptrs, torch::Tensor ag_ptrs,
                                   int64_t my_rank, int64_t ndev, int64_t n_fc1, torch::Tensor dbg, int64_t mode, torch::Tensor work,
                                   int64_t reserve, int64_t parts, torch::Tensor part_buf, torch::Tensor part_flags) {
    TORCH_CHECK(M >= 1 && M <= MAXM, "M must be in [1, 64]");
    TORCH_CHECK(N_FC2_TILES % ndev == 0, "ndev must divide 48");
    int M_pad = (M + 7) & ~7;
    if (M_pad == 56) M_pad = 64;   // the NT=56 instantiation spills inside the fc1 loop at the 96-register cap; NT=64 does not
#ifdef FMOE_DIAG
    if (mode & 256) M_pad = 32;    // diagnostics: run the NT=32 instantiation on larger M (fc1 timing only; h of tokens >= 32 is dropped)
#endif
    P.M = M; P.M_pad = M_pad;
    if (FMOE_ROUTER_TMA && !P.pre_routed) {
        // Router prologue maps, encoded per launch into the kernel parameter: 3-D {64 bf16 (one 128-B k64 column), rows, 96 columns}
        // with strides {row pitch DIM*2, column pitch 128}; boxes {64, NT, 6} (activations) / {64, 64, 6} (weight); SWIZZLE_128B. The host
        // call costs a few us per eager launch and nothing per graph replay.
        TORCH_CHECK((reinterpret_cast<uint64_t>(P.hidden) | reinterpret_cast<uint64_t>(P.residual) | reinterpret_cast<uint64_t>(P.router_wb) |
                     reinterpret_cast<uint64_t>(P.norm_w)) % 16 == 0, "hidden/residual/router_w/norm_w must be 16-B aligned (router TMA)");
        const cuuint64_t d_act[3] = {64, (cuuint64_t)M, DIM / 64}, s_act[2] = {DIM * 2, 128};
        const cuuint32_t b_act[3] = {64, (cuuint32_t)M_pad, RKS / 64};
        encode_map(&P.tm_h, P.hidden, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d_act, s_act, b_act, CU_TENSOR_MAP_SWIZZLE_128B);
        encode_map(&P.tm_r, P.residual, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d_act, s_act, b_act, CU_TENSOR_MAP_SWIZZLE_128B);
        const cuuint64_t d_w[3] = {64, (cuuint64_t)P.n_exp, DIM / 64};
        const cuuint32_t b_w[3] = {64, RROWS, RKS / 64};
        encode_map(&P.tm_w, P.router_wb, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d_w, s_act, b_w, CU_TENSOR_MAP_SWIZZLE_128B);
    }
    P.h_buf = h_buf.data_ptr<uint8_t>(); P.cs_buf = cs_buf.data_ptr<float>(); P.h_flags = reinterpret_cast<unsigned*>(h_flags.data_ptr<int>());
    P.m32_chunk_scratch = h_flags.numel() >= (int64_t)MAXU * 10;
    P.part_buf = part_buf.data_ptr<float>(); P.part_flags = reinterpret_cast<unsigned*>(part_flags.data_ptr<int>());
    P.fc2_helper_capacity = part_flags.numel() >= 64 + N_FC2_TILES;
    P.m32_tail_scratch = h_buf.numel() >= (int64_t)MAXU * 2 * 32 * 128 && part_flags.numel() >= (int64_t)MAXU * 2;
    P.ks_capacity = part_buf.numel() >= (int64_t)132 * KS_PART_FLOATS && part_flags.numel() >= KS_FLAG_BASE + 132;
    if (KSPLIT == 2) TORCH_CHECK(part_buf.numel() >= (int64_t)MAXU * 2 * 2 * 64 * 128 && part_flags.numel() >= (int64_t)MAXU * 2, "K-split partial buffers too small");
    P.rs_bufs = reinterpret_cast<uint4* const*>(rs_ptrs.data_ptr<int64_t>()); P.ag_bufs = reinterpret_cast<uint4* const*>(ag_ptrs.data_ptr<int64_t>());
    P.my_rank = (int)my_rank; P.ndev = (int)ndev;
    P.n_fc1 = (int)n_fc1;
    P.dbg = dbg.numel() > 0 ? reinterpret_cast<unsigned long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    P.mode = (int)mode;
    P.work = work.data_ptr<int>();
    TORCH_CHECK(parts >= 1 && parts <= 3, "parts must be 1, 2 or 3 (3 = mixed 3/2 parts over the whole grid)");
    P.parts = (int)parts;
    P.reserve = (int)reserve;   // < 0: computed on device from the union size (see fc1_role); >= 0: explicit item count
    TORCH_CHECK(work.numel() >= 4, "work counters must have 4 ints (front item queue, completion, epoch, back item queue)");
    static int sms_cache[16] = {0};
    int dev = 0; cudaGetDevice(&dev);
    if (sms_cache[dev] == 0) cudaDeviceGetAttribute(&sms_cache[dev], cudaDevAttrMultiProcessorCount, dev);
    int grid;
    if (parts == 3) {   // mixed: the whole grid does fc2; n3 = grid - 96 tiles get 3 parts
        grid = sms_cache[dev];
        TORCH_CHECK(grid > 2 * N_FC2_TILES && grid <= 3 * N_FC2_TILES && grid >= N_FC2_TILES / (int)ndev, "mixed parts need 97..144 SMs");
    } else {
        TORCH_CHECK(n_fc1 >= 1, "need at least one fc1-only CTA");
        grid = (int)n_fc1 + N_FC2_TILES * (int)parts;
    }
    TORCH_CHECK(grid <= sms_cache[dev], "grid must be co-resident (1 CTA/SM): n_fc1 + 48*parts <= #SMs (CTAs spin-wait on each other)");
#if !defined(FMOE_PHASE_STAMPS) && !defined(FMOE_DIAG)
    TORCH_CHECK(!P.dbg, "phase stamps require a separate build with -DFMOE_PHASE_STAMPS or -DFMOE_DIAG");
#endif
    if (P.dbg) TORCH_CHECK(dbg.numel() >= (int64_t)grid * NSTAMP, "dbg too small");
    TORCH_CHECK(h_buf.numel() >= (int64_t)MAXU * 2 * M_pad * 128, "h_buf too small (sized for 384 union slots)");
    TORCH_CHECK(cs_buf.numel() >= (int64_t)MAXU * M_pad * 2 * ((M_pad == 32 && FMOE_CS_PERM) ? 2 : 1), "cs_buf too small (M32 needs the permuted prescaled plane in its upper half)");
    TORCH_CHECK(h_flags.numel() >= (int64_t)MAXU * 2, "h_flags too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (P.input_tp) {   // FMOE_INPUT_TP entry: one instantiation
#if FMOE_INPUT_TP
        TORCH_CHECK(M == TP_M && ndev == TP_NDEV && grid == 132 && parts == 1 && !P.pre_routed && P.out_bf16 && P.n_exp == NEXP,
                    "input-TP entry needs M == 32, TP8, a 132-CTA grid (n_fc1 84, parts 1), bf16 output and 384 experts");
        launch<32, 2, 32, true>(P, grid, stream);
#else
        TORCH_CHECK(false, "fused_moe_full_tp: this build has FMOE_INPUT_TP=0 (rebuild with FMOE_NVCC_FLAGS=-DFMOE_INPUT_TP=1)");
#endif
        return;
    }
    // The four production shapes use their own exact-M operator.  Besides fixing the FC2 decomposition, this makes M,
    // M_pad and the all-reduce polling window compile-time constants throughout the inlined prologue/FC1/FC2 path.
    if (M == M_pad && ndev == 8 && grid == 132) {
        if (M == 24 && parts == 2) { launch<24, 2, 24>(P, grid, stream); return; }
        if (M == 32 && parts == 1) {
            // The BF16 full operator uses96 FC2 CTAs on the same132-CTA
            // grid. Keep the legacy FP32-output operator's original split.
            if (P.out_bf16) launch<32, 2, 32>(P, grid, stream);
            else launch<32, 1, 32>(P, grid, stream);
            return;
        }
        if (M == 48 && parts == 3) { launch<48, 3, 48>(P, grid, stream); return; }
        if (M == 64 && parts == 2) { launch<64, 2, 64>(P, grid, stream); return; }
    }
    switch (M_pad) {
        case 8: launch_parts<8>(P, grid, stream); break;
        case 16: launch_parts<16>(P, grid, stream); break;
        case 24: launch_parts<24>(P, grid, stream); break;
        case 32: launch_parts<32>(P, grid, stream); break;
        case 40: launch_parts<40>(P, grid, stream); break;
        case 48: launch_parts<48>(P, grid, stream); break;
        case 64: launch_parts<64>(P, grid, stream); break;
        default: TORCH_CHECK(false, "unsupported M_pad");
    }
}

// Legacy entry point: routing (union_experts + gate_w) and the quantized activation given by the host; fp32 out.
void fused_moe(torch::Tensor x_fp8, torch::Tensor x_scale, torch::Tensor fc1_w, torch::Tensor fc1_s, torch::Tensor fc1_s2,
               torch::Tensor fc2_w, torch::Tensor fc2_s, torch::Tensor fc2_s2,
               torch::Tensor union_experts, torch::Tensor gate_w, torch::Tensor h_buf, torch::Tensor cs_buf, torch::Tensor h_flags,
               torch::Tensor rs_ptrs, torch::Tensor ag_ptrs, int64_t my_rank, int64_t ndev, int64_t epoch, torch::Tensor out, int64_t n_fc1,
               torch::Tensor dbg, int64_t mode, torch::Tensor work, int64_t reserve, int64_t parts, torch::Tensor part_buf, torch::Tensor part_flags) {
    (void)epoch;   // the tag comes from work[2]
    const int M = (int)x_fp8.size(0), U = (int)union_experts.size(0);
    TORCH_CHECK(U >= 1 && U <= MAXU, "U must be in [1, 384]");
    Params P{};
    P.pre_routed = 1;
    P.xq_buf = x_fp8.data_ptr<uint8_t>(); P.xs_buf = x_scale.data_ptr<float>();
    P.union_experts = union_experts.data_ptr<int>(); P.gate_w = gate_w.data_ptr<float>(); P.U = U;
    P.out_f32 = out.data_ptr<float>(); P.out_bf16 = nullptr;
    P.n_exp = NEXP;
    set_weights(P, fc1_w, fc1_s, fc1_s2, fc2_w, fc2_s, fc2_s2);
    fill_common_and_launch(P, M, h_buf, cs_buf, h_flags, rs_ptrs, ag_ptrs, my_rank, ndev, n_fc1, dbg, mode, work, reserve, parts, part_buf, part_flags);
}

// Complete path: bf16 hidden + residual -> RMSNorm -> router/top-8 -> MoE -> all-reduce -> bf16 out (+ bf16 residual_out).
void fused_moe_full(torch::Tensor hidden, torch::Tensor residual, torch::Tensor norm_w, torch::Tensor router_w, torch::Tensor bias, double eps,
                    torch::Tensor fc1_w, torch::Tensor fc1_s, torch::Tensor fc1_s2, torch::Tensor fc2_w, torch::Tensor fc2_s, torch::Tensor fc2_s2,
                    torch::Tensor xn_buf, torch::Tensor xq_buf, torch::Tensor xs_buf, torch::Tensor xflags, torch::Tensor rpart, torch::Tensor topk, torch::Tensor ssq, torch::Tensor union_out,
                    torch::Tensor h_buf, torch::Tensor cs_buf, torch::Tensor h_flags, torch::Tensor rs_ptrs, torch::Tensor ag_ptrs, int64_t my_rank, int64_t ndev,
                    torch::Tensor out, torch::Tensor residual_out, int64_t n_fc1, torch::Tensor dbg, int64_t mode, torch::Tensor work, int64_t reserve, int64_t parts,
                    torch::Tensor part_buf, torch::Tensor part_flags) {
    const int M = (int)hidden.size(0), n_exp = (int)router_w.size(0);
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && residual.scalar_type() == torch::kBFloat16 && norm_w.scalar_type() == torch::kBFloat16, "hidden/residual/norm_w must be bf16");
    TORCH_CHECK(router_w.scalar_type() == (FMOE_ROUTER_BF16 ? torch::kBFloat16 : torch::kFloat32) && bias.scalar_type() == torch::kFloat32,
                FMOE_ROUTER_BF16 ? "router_w must be bf16 (FMOE_ROUTER_BF16 build: round the fp32 weight once on the host), bias fp32"
                                 : "router_w/bias must be fp32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && residual_out.scalar_type() == torch::kBFloat16, "out/residual_out must be bf16");
    TORCH_CHECK(hidden.size(1) == DIM && router_w.size(1) == DIM && hidden.is_contiguous() && residual.is_contiguous() && router_w.is_contiguous(), "shapes/strides");
    TORCH_CHECK(n_exp >= TOPK && n_exp <= NEXP && bias.numel() >= n_exp, "n_exp must be in [8, 384]");
    TORCH_CHECK(xn_buf.numel() >= (int64_t)MAXM * DIM && xq_buf.numel() >= (int64_t)MAXM * DIM && xs_buf.numel() >= (int64_t)MAXM * NKT1 && xflags.numel() >= MAXM, "x scratch too small");
    TORCH_CHECK(rpart.numel() >= (int64_t)RUNITS_MAX * MAXM * RMSG * 4 && topk.numel() >= (int64_t)MAXM * (TOPK / 2) * 4 && union_out.numel() >= 1 + MAXU
                && ssq.numel() >= (int64_t)RKG * RMSG * 4, "routing scratch too small");
    Params P{};
    P.pre_routed = 0;
    P.hidden = reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()); P.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    P.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
    if (FMOE_ROUTER_BF16) P.router_wb = reinterpret_cast<const __nv_bfloat16*>(router_w.data_ptr());
    else P.router_w = router_w.data_ptr<float>();
    P.bias = bias.data_ptr<float>(); P.eps = (float)eps; P.n_exp = n_exp;
    P.residual_out = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    P.xn_buf = reinterpret_cast<__nv_bfloat16*>(xn_buf.data_ptr()); P.xq_buf = xq_buf.data_ptr<uint8_t>(); P.xs_buf = xs_buf.data_ptr<float>();
    P.xflags = reinterpret_cast<unsigned*>(xflags.data_ptr<int>());
    P.rpart = reinterpret_cast<uint4*>(rpart.data_ptr<int>()); P.topk = reinterpret_cast<uint4*>(topk.data_ptr<int>());
    P.ssq = reinterpret_cast<uint4*>(ssq.data_ptr<int>());
    P.union_out = union_out.data_ptr<int>();
    P.out_f32 = nullptr; P.out_bf16 = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int n_wexp = set_weights(P, fc1_w, fc1_s, fc1_s2, fc2_w, fc2_s, fc2_s2);
    TORCH_CHECK(n_wexp >= n_exp, "expert weights must cover n_exp experts");
    fill_common_and_launch(P, M, h_buf, cs_buf, h_flags, rs_ptrs, ag_ptrs, my_rank, ndev, n_fc1, dbg, mode, work, reserve, parts, part_buf, part_flags);
}

// FMOE_INPUT_TP entry: like fused_moe_full, but `partial` is this rank's UN-reduced o_proj output (bf16 [32][DIM]); the
// input all-reduce happens inside the prologue through the ranks' IPC prologue buffers (pro_ptrs = int64[ndev] device table of the
// peers' buffers, pro_local = this rank's, PRO_BYTES each, zero-initialised once). rpart / topk live inside the local buffer.
// FMOE_NORM_NEXT: norm_w_next (bf16 [DIM]; numel 0 = off), res_new (bf16 [32][DIM]), xq_next (e4m3 / uint8 [32][DIM]) and xs_next
// (fp32 [32][48] column-major view: stride (1, S >= 32) = SGLang's TMA-aligned deep_gemm scale layout) receive the NEXT layer's
// input_layernorm output (residual_new) and its fp8 per-128 activation quant; eps_next = that norm's epsilon.
void fused_moe_full_tp(torch::Tensor partial, torch::Tensor residual, torch::Tensor norm_w, torch::Tensor router_w, torch::Tensor bias, double eps,
                       torch::Tensor fc1_w, torch::Tensor fc1_s, torch::Tensor fc1_s2, torch::Tensor fc2_w, torch::Tensor fc2_s, torch::Tensor fc2_s2,
                       torch::Tensor xn_buf, torch::Tensor xq_buf, torch::Tensor xs_buf, torch::Tensor xflags, torch::Tensor pro_ptrs, int64_t pro_local,
                       torch::Tensor union_out, torch::Tensor h_buf, torch::Tensor cs_buf, torch::Tensor h_flags, torch::Tensor rs_ptrs, torch::Tensor ag_ptrs,
                       int64_t my_rank, int64_t ndev, torch::Tensor out, torch::Tensor residual_out, int64_t n_fc1, torch::Tensor dbg, int64_t mode,
                       torch::Tensor work, int64_t reserve, int64_t parts, torch::Tensor part_buf, torch::Tensor part_flags,
                       torch::Tensor pro_cpu, torch::Tensor rs_cpu, torch::Tensor ag_cpu,
                       torch::Tensor norm_w_next, torch::Tensor res_new, torch::Tensor xq_next, torch::Tensor xs_next, double eps_next,
                       torch::Tensor pf) {
    const int M = (int)partial.size(0), n_exp = (int)router_w.size(0);
    TORCH_CHECK(partial.scalar_type() == torch::kBFloat16 && residual.scalar_type() == torch::kBFloat16 && norm_w.scalar_type() == torch::kBFloat16, "partial/residual/norm_w must be bf16");
    // FMOE_TP_PTR_PARAMS: the same three peer tables as CPU int64[ndev] tensors -> by value into the kernel parameter (no device
    // indirection; host-side reads only, so graph capture is unaffected)
    for (const torch::Tensor* t : {&pro_cpu, &rs_cpu, &ag_cpu})
        TORCH_CHECK(t->device().is_cpu() && t->scalar_type() == torch::kInt64 && t->numel() >= ndev && t->is_contiguous(), "pro_cpu/rs_cpu/ag_cpu must be CPU int64[ndev] pointer tables");
    // pf = CPU int64 [6 or 7] = {ptr, bytes} x 3 (0 = unused) [+ evict_last hint bitmask] -> kernel-parameter constants (fixed
    // addresses, CUDA-graph safe); empty = no prefetch.
    TORCH_CHECK(pf.numel() == 0 || (pf.numel() >= 6 && pf.device().is_cpu() && pf.scalar_type() == torch::kInt64), "pf must be empty or a CPU int64 [6|7] tensor");
    TORCH_CHECK(FMOE_ROUTER_BF16 && router_w.scalar_type() == torch::kBFloat16 && bias.scalar_type() == torch::kFloat32, "router_w must be bf16 (FMOE_ROUTER_BF16 build), bias fp32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && residual_out.scalar_type() == torch::kBFloat16, "out/residual_out must be bf16");
    TORCH_CHECK(partial.size(1) == DIM && router_w.size(1) == DIM && partial.is_contiguous() && residual.is_contiguous() && router_w.is_contiguous(), "shapes/strides");
    TORCH_CHECK(n_exp == NEXP && bias.numel() >= n_exp, "input-TP entry needs n_exp == 384");
    TORCH_CHECK(xn_buf.numel() >= (int64_t)MAXM * DIM && xq_buf.numel() >= (int64_t)MAXM * DIM && xs_buf.numel() >= (int64_t)MAXM * NKT1 && xflags.numel() >= MAXM, "x scratch too small");
    TORCH_CHECK(pro_ptrs.is_cuda() && pro_ptrs.scalar_type() == torch::kInt64 && pro_ptrs.numel() >= ndev && pro_ptrs.is_contiguous() && pro_local != 0 && pro_local % 32 == 0,
                "pro_ptrs must be a device int64[ndev] table of 32-B aligned prologue buffers");
    TORCH_CHECK(union_out.numel() >= 1 + MAXU, "union_out too small");
    Params P{};
    P.pre_routed = 0; P.input_tp = 1;
    if (pf.numel() >= 6) {
        const int64_t* d = pf.data_ptr<int64_t>();
        for (int r = 0; r < 3; ++r) {
            const uint64_t ptr = (uint64_t)d[2 * r], bytes = (uint64_t)d[2 * r + 1];
            TORCH_CHECK(ptr % 16 == 0 && bytes < (1ull << 32), "pf range ", r, ": 16-B aligned pointer, < 4 GB");
            P.pf_ptr[r] = ptr ? reinterpret_cast<const uint8_t*>(ptr) : nullptr;
            P.pf_bytes[r] = ptr ? (unsigned)(bytes & ~15ull) : 0u;
        }
        P.pf_hint = pf.numel() >= 7 ? (unsigned)(d[6] & 7) : 0u;
    }
    P.partial = reinterpret_cast<const __nv_bfloat16*>(partial.data_ptr());
    P.pro_bufs = reinterpret_cast<uint4* const*>(pro_ptrs.data_ptr<int64_t>());
    TORCH_CHECK(ndev == TP_NDEV, "input-TP entry: TP8");
    for (int d = 0; d < TP_NDEV; ++d) {
        P.pro_p[d] = reinterpret_cast<uint4*>(pro_cpu.data_ptr<int64_t>()[d]);
        P.rs_p[d] = reinterpret_cast<uint4*>(rs_cpu.data_ptr<int64_t>()[d]);
        P.ag_p[d] = reinterpret_cast<uint4*>(ag_cpu.data_ptr<int64_t>()[d]);
        TORCH_CHECK(P.pro_p[d] && P.rs_p[d] && P.ag_p[d], "null peer pointer in the CPU tables");
    }
    TORCH_CHECK(reinterpret_cast<uint4*>(pro_cpu.data_ptr<int64_t>()[my_rank]) == reinterpret_cast<uint4*>(pro_local), "pro_cpu[my_rank] must be the local prologue buffer");
    P.rpart = reinterpret_cast<uint4*>(pro_local) + PRO_RPART; P.topk = reinterpret_cast<uint4*>(pro_local) + PRO_TOPK;
    P.hidden = nullptr; P.ssq = nullptr;
    P.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    P.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
    P.router_wb = reinterpret_cast<const __nv_bfloat16*>(router_w.data_ptr());
    P.bias = bias.data_ptr<float>(); P.eps = (float)eps; P.n_exp = n_exp;
    P.residual_out = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    P.xn_buf = reinterpret_cast<__nv_bfloat16*>(xn_buf.data_ptr()); P.xq_buf = xq_buf.data_ptr<uint8_t>(); P.xs_buf = xs_buf.data_ptr<float>();
    P.xflags = reinterpret_cast<unsigned*>(xflags.data_ptr<int>());
    P.union_out = union_out.data_ptr<int>();
    P.out_f32 = nullptr; P.out_bf16 = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    if (norm_w_next.numel() > 0) {
#if FMOE_NORM_NEXT
        TORCH_CHECK(norm_w_next.scalar_type() == torch::kBFloat16 && norm_w_next.numel() == DIM && norm_w_next.is_contiguous(), "norm_w_next must be a contiguous bf16 [DIM]");
        TORCH_CHECK(res_new.scalar_type() == torch::kBFloat16 && res_new.numel() >= (int64_t)TP_M * DIM && res_new.is_contiguous(), "res_new must be a contiguous bf16 [32][DIM]");
        TORCH_CHECK((xq_next.scalar_type() == torch::kUInt8 || xq_next.scalar_type() == torch::kFloat8_e4m3fn) && xq_next.numel() >= (int64_t)TP_M * DIM && xq_next.is_contiguous(),
                    "xq_next must be a contiguous e4m3 / uint8 [32][DIM]");
        TORCH_CHECK(xs_next.scalar_type() == torch::kFloat32 && xs_next.dim() == 2 && xs_next.size(0) == TP_M && xs_next.size(1) == NKT1 && xs_next.stride(0) == 1 &&
                    xs_next.stride(1) >= TP_M && xs_next.stride(1) % 4 == 0, "xs_next must be the column-major fp32 [32][48] scale view (stride (1, S), S >= 32, 16-B aligned)");
        TORCH_CHECK(reinterpret_cast<uint64_t>(res_new.data_ptr()) % 16 == 0 && reinterpret_cast<uint64_t>(xq_next.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uint64_t>(norm_w_next.data_ptr()) % 16 == 0 && reinterpret_cast<uint64_t>(residual_out.data_ptr()) % 16 == 0, "row-stage buffers must be 16-B aligned");
        P.norm_w_next = reinterpret_cast<const __nv_bfloat16*>(norm_w_next.data_ptr());
        P.res_new = reinterpret_cast<__nv_bfloat16*>(res_new.data_ptr());
        P.xq_next = reinterpret_cast<uint8_t*>(xq_next.data_ptr());
        P.xs_next = xs_next.data_ptr<float>(); P.xs_next_stride = (int)xs_next.stride(1);
        P.eps_next = (float)eps_next; P.norm_next = 1;
#else
        TORCH_CHECK(false, "fused_moe_full_tp: norm_w_next given but this build has FMOE_NORM_NEXT=0");
#endif
    }
    const int n_wexp = set_weights(P, fc1_w, fc1_s, fc1_s2, fc2_w, fc2_s, fc2_s2);
    TORCH_CHECK(n_wexp >= n_exp, "expert weights must cover n_exp experts");
    fill_common_and_launch(P, M, h_buf, cs_buf, h_flags, rs_ptrs, ag_ptrs, my_rank, ndev, n_fc1, dbg, mode, work, reserve, parts, part_buf, part_flags);
}

}  // namespace fmoe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_moe", &fmoe::fused_moe);
    m.def("fused_moe_full", &fmoe::fused_moe_full);
    m.def("fused_moe_full_tp", &fmoe::fused_moe_full_tp);
    m.attr("INPUT_TP") = (int)FMOE_INPUT_TP;
    m.attr("NORM_NEXT") = (int)(FMOE_INPUT_TP && FMOE_NORM_NEXT);
    m.attr("NN_ROW_CTA0") = fmoe::NN_ROW_CTA0;
    m.attr("TAIL_PREFETCH") = (int)FMOE_TAIL_PREFETCH;   // trigger mode (0 = built without the tail prefetch)
    m.attr("PF_TEST_KB") = (int)FMOE_PF_TEST_KB;
    m.attr("PRO_BYTES") = fmoe::PRO_U4 * 16;
    m.attr("N_FC2_TILES") = fmoe::N_FC2_TILES;
    m.attr("N_FC2_CTAS") = fmoe::N_FC2_CTAS;
    m.attr("MSG_PER_TOK") = fmoe::MSG_PER_TOK;
    m.attr("SMEM_ALLOC") = fmoe::SMEM_ALLOC;
    m.attr("NEXP") = fmoe::NEXP;
    m.attr("MAXU") = fmoe::MAXU;
    m.attr("RUNITS") = fmoe::RUNITS_MAX;
    m.attr("ROUTER_BF16") = (int)FMOE_ROUTER_BF16;
    m.attr("RMSG") = fmoe::RMSG;
    m.attr("NSTAMP") = fmoe::NSTAMP;
    m.attr("KSPLIT") = fmoe::KSPLIT;
    m.attr("FC1_W_EXPERT_BYTES") = fmoe::FC1_W_EXPERT_BYTES;
    m.attr("FC1_S_EXPERT_BYTES") = fmoe::FC1_S_EXPERT_BYTES;
    m.attr("FC2_W_EXPERT_BYTES") = fmoe::FC2_W_EXPERT_BYTES;
    m.attr("FC2_S_EXPERT_BYTES") = fmoe::FC2_S_EXPERT_BYTES;
}
