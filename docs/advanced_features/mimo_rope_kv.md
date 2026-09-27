# Opt-in MiMo RoPE and KV-store fusion on Hopper

`SGLANG_OPT_MIMO_ROPE_KV=1` enables a narrow MiMo attention specialization.
It fuses in-place partial RoPE on Q/K with the BF16 KV-cache write. The default
is **off**. Unsupported shapes, devices, backends, cache layouts and forward
modes use the existing implementation. The two-batch-overlap path is unchanged.

The supported path is SM90 target verification with 32 token rows, 16 local Q
heads, one local KV head, head dimensions 192/128, and NeoX rotary dimension 64.
QKV and the cosine/sine cache are contiguous BF16. The attention backend is
FlashAttention with a nonquantized, contiguous NHD `MHATokenToKVPool`, directly
or through `SWAKVPool`. Sliding-window layers use the backend's mapped cache
locations. Cache location zero is padding and is never written.

This implementation preserves the BF16 fallback's operation order in the
validated CUDA 13 build: one BF16 multiply rounds before the BF16 fused multiply
add. It is not a float32-RoPE replacement. Bitwise unit tests compare against
the original fallback, including the entire cache and CUDA Graph replay.
Compiler/runtime changes must rerun these tests before enabling the feature.
Position and nonzero cache indices must satisfy the normal backend contract;
nonzero destinations within one launch must be distinct.

## Verification

From the repository root, in a supported SGLang environment:

```bash
python3 test/registered/unit/layers/attention/test_mimo_rope_kv_dispatch.py -v
python3 test/registered/kernels/ops/attention/test_mimo_rope_kv.py -v
```

The first command is CPU-only. The second requires SM90 and checks both int32
and int64 indices, sparse destinations, padding-only writes, multiple CTA head
groupings and repeated CUDA Graph replay. A successful graph-capture dispatch
logs `MIMO_ROPE_KV_CAPTURE` once per attention module.

## Benchmark contract and merge order

This PR is intended to merge **before** the O-projection weight-prefetch PR and
before MegaMoE. Compare it with the unmodified `h200_m_fp4` commit
`203f188f3de4996dab558307ab9aa5a79c42ba65`, without MegaMoE or external overlays.
Do not use the historical MegaMoE-enabled combined-optimization run as its
control.

Use identical model revision, image, Humming build, eight H200 GPUs, TP8/EP1,
DFLASH with eight draft tokens, batch four, input/output lengths 1024/64,
temperature zero, seed 42, cache-hit-rate zero and CUDA Graphs. Run three native
repetitions per arm and keep every sample. Step time is
`4000 * measured_acceptance_length / decode_output_tokens_per_second`, using
unrounded JSON values. It is not ITL or a profiler duration.

Check fixed-prompt token/text parity against the original branch, actual
specialized dispatch, custom-all-reduce availability and GPU telemetry before
interpreting performance. Three repetitions characterize this workload, not
statistical significance or model-wide accuracy.
