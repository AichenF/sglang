# Opt-in MiMo O-projection weight prefetch on Hopper

**The attention-first experiment did not show an incremental latency benefit
over RoPE/KV fusion alone.** Keep prefetch off by default; do not infer a
standalone gain from older MegaMoE-enabled combined runs. This branch preserves
the implementation for explicit workload-specific evaluation.

This change stacks on the MiMo RoPE/KV fusion PR. Enable both
`SGLANG_OPT_MIMO_ROPE_KV=1` and `SGLANG_OPT_MIMO_O_PREFETCH=1` to issue read-only
O-projection weight prefetches from the fused kernel. Both flags default off.
If the weight is unsupported, execution uses RoPE/KV fusion without prefetch;
if fusion is unsupported, execution uses the original attention path.

The weight must be contiguous BF16, shape 6144 by 2048, on the QKV device and
16-byte aligned. The SM90 fused kernel launches 32 CTAs; one physical thread
per CTA issues one `cp.async.bulk.prefetch.L2.global` for a disjoint 768 KiB
chunk. The 32 chunks cover the 24 MiB tensor. Prefetch is a cache hint, not a
weight transformation or synchronization dependency. Neither the weight nor
the attention calculation changes.

The shared GPU test additionally checks exact QKV/KV parity, repeated CUDA
Graph replay and unchanged O weights with prefetch. CPU tests check each weight
guard, both independent opt-ins and fallback to fusion without prefetch.

## Evaluation and merge order

Merge RoPE/KV fusion first, then this change, then MegaMoE. This PR's code diff
is against the fusion branch, but its benchmark must include all three arms:

1. Original `h200_m_fp4` at `203f188f3de4996dab558307ab9aa5a79c42ba65`.
2. RoPE/KV fusion only.
3. RoPE/KV fusion plus O-weight prefetch.

MegaMoE is absent/disabled in every arm. Report both the incremental effect
of prefetch (arm 2 versus arm 3) and the combined effect relative to the
original branch (arm 1 versus arm 3). Use the fixed native benchmark contract
in `mimo_rope_kv.md`, including all three repeats and token/text parity.

L2 residency depends on the workload. A result for this small target-verify
batch must not be presented as a universal throughput improvement.

## Original and incremental comparison, 2026-09-28

Measured code: original `203f188f3de4996dab558307ab9aa5a79c42ba65`, fusion
`021c5e3f7e753e3131cb99b62f512c1ceb85d338`, fusion plus prefetch
`ba625cc8f08813f6f0885c1e31888d39abe64268`. Subsequent commits only add docs.

| Arm | All native step samples (ms) | Median (ms) | Range (ms) |
| --- | --- | --- | --- |
| Original `h200_m_fp4` | 11.558042, 11.365562, 11.687017 | 11.558042 | 11.365562–11.687017 |
| RoPE/KV fusion | 11.271207, 11.274559, 11.353022 | 11.274559 | 11.271207–11.353022 |
| Fusion + O-weight prefetch | 11.271076, 11.313126, 11.339134 | 11.313126 | 11.271076–11.339134 |

Relative to the original branch, the combined median is 0.244916 ms lower
(**2.119% reduction**). Relative to fusion alone, prefetch is 0.038566 ms higher
(**0.342% increase**). Sample ranges overlap; three sequential repetitions do
not establish a statistically significant regression or improvement.
The result does **not** demonstrate incremental prefetch benefit.
Prefetch acceptance lengths were `[1.03125, 1.03125, 1.03125]`; use each
repetition's unrounded acceptance and throughput to compute step time.

All arms used the same eight H200s, TP8/EP1, B4/DFLASH8, 1024/64 lengths,
temperature zero, seed 42, cache zero, CUDA Graph, complete model revision,
image and Humming commit documented in `mimo_rope_kv.md`. MegaMoE was absent
and disabled. To reproduce prefetch, use that document's command with both
feature flags set to `1` and a fresh result directory.

Nine CPU tests and three GPU test methods passed (32 cases in total), including
exact fallback QKV/full-cache parity, padding, graph replay and unchanged O
weights. Native E2E verified 560 prefetch capture records (70 layers, eight
ranks), serving smoke, custom all-reduce and exact token/text parity with the
original on the four fixed 1024/64 inputs. Scoped isort/Ruff checks passed.
Repository-wide CI and model-wide accuracy were not evaluated. The original
branch's short text-prompt self-repeatability limitation documented in
`mimo_rope_kv.md` also applies to interpreting smoke results here.
