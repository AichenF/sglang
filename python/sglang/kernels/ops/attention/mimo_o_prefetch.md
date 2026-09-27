# Opt-in MiMo O-projection weight prefetch on Hopper

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
