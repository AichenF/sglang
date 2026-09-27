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

## Original-branch comparison, 2026-09-28

Measured code: original `203f188f3de4996dab558307ab9aa5a79c42ba65`, fusion
`021c5e3f7e753e3131cb99b62f512c1ceb85d338`. Later documentation-only commits
do not change the measured implementation. Both arms exclude MegaMoE.

| Arm | Step samples (ms) | Median (ms) | Range (ms) |
| --- | --- | --- | --- |
| Original `h200_m_fp4` | 11.558042, 11.365562, 11.687017 | 11.558042 | 11.365562–11.687017 |
| RoPE/KV fusion | 11.271207, 11.274559, 11.353022 | 11.274559 | 11.271207–11.353022 |

The observed median step reduction is **2.453% (0.283482 ms)**. Acceptance
lengths were respectively `[1.0375, 1.03125, 1.05]` and
`[1.0375, 1.025, 1.03125]`; the step calculation uses each repetition's own
unrounded acceptance and throughput. These are three sequential repetitions
per arm, not a statistical-significance claim.

Seven CPU dispatch tests and two GPU test methods passed (20 parameterized
cases, including repeated CUDA Graph replay). The first GPU attempt exposed
an int32-input error in the test's int64-only fallback reference call; the
reference call was corrected and both diagnostic and clean reruns passed.
The fused implementation and zero-tolerance comparison were not relaxed.
The native run verified all 70 layers on all eight ranks (560 capture records),
five-request serving smoke, custom all-reduce, and exact token/text parity on
four fixed 1024-token inputs with 64 output tokens each. Pre/post measurement
telemetry had no active slowdown. Repository-wide CI was not run.

### Runtime and commands

Use Linux, eight NVIDIA H200 GPUs, driver 615.71.09, CUDA 13, Torch 2.13 cu130,
and Humming commit `32f8d3156d98b4d7cc5ea1a465925d7997502966`.
The measured image ID was
`sha256:360c23e1723c65b0c9d8d379938485365474d6fa8882911b56880c31c86a5cf6`.
Use a complete local copy of `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`, revision
`b754e6c86008bdb5cc901308dda5a38173ec7276`, including its `dflash` subdirectory.
Install the exact selected SGLang checkout in that environment. No external
adapter overlay is used. Set `MODEL_PATH` to the checkpoint directory.

Launch one arm at a time. Use `SGLANG_OPT_MIMO_ROPE_KV=0` for the original arm
and `1` for the fusion arm; keep all other options unchanged:

```bash
export SGLANG_MIMO_FUSED_MOE=0
export SGLANG_OPT_MIMO_ROPE_KV=1
export SGLANG_OPT_MIMO_O_PREFETCH=0
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","input_scale_group_size":128}'
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --context-length 65536 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$MODEL_PATH/dflash" \
  --speculative-num-draft-tokens 8 --tp-size 8 --ep-size 1 \
  --trust-remote-code --moe-runner-backend humming \
  --mem-fraction-static 0.75 --random-seed 42 \
  --host 127.0.0.1 --port 29999
```

In another shell, wait for readiness and then run three native repetitions.
Choose a fresh results directory for each arm:

```bash
mkdir -p results/rope
for repeat in 1 2 3; do
  python3 -m sglang.benchmark.one_batch_server \
    --model None --base-url http://127.0.0.1:29999 \
    --batch-size 4 --input-len 1024 --output-len 64 \
    --temperature 0 --seed 42 --cache-hit-rate 0 --show-report \
    --run-name "rope_r${repeat}" \
    --result-filename "results/rope/r${repeat}.jsonl" \
    --pydantic-result-filename "results/rope/r${repeat}.pydantic.json" \
    --no-append-to-github-summary
done
```
