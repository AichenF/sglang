# H200 mixed-FP4 decode with DFlash (TP8)

This branch runs mixed-FP4 MiMo-V2 checkpoints (MXFP4 routed experts through the Humming MoE backend) with DFlash
speculative decoding on one 8xH200 node, and adds fused decode kernels for the target-verify step and the DFlash
draft.

With DFlash and 8 draft tokens the measured average acceptance length is about 8 (7.97-8.00) on the benchmark
prompts with greedy sampling. The greedy outputs of the parity prompts are token-identical to the unfused
reference; the fused kernels change only summation order.

## Requirements

- 8x H200 (sm_90 with **132 SMs**; H100 SXM has the same SM count), tensor parallelism 8, expert parallelism 1,
  no DP attention.
- The fused MoE kernel and the attention megakernels size their grids for one CTA per SM and spin-wait across CTAs.
  On an sm_90 part with a different SM count (H20, H100 PCIe) they are disabled and the existing paths run.
- CUDA-graph decode: do not pass `--disable-cuda-graph`.
- Humming: `python/pyproject.toml` pins Humming to source revision `a74973b`. The measurements were taken with a
  Humming build at `32f8d31`, which has the same weight layout; the fused MoE kernel reads the Humming tensors in
  place and does not call the Humming kernels.

## Switches

### W4A8 MoE activations

The MXFP4 expert weights are W4 in both modes; Humming uses BF16 activations (W4A16) by default. The fused MoE
kernel needs W4A8 (dynamic group-128 FP8 E4M3 activations), selected with:

```bash
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","input_scale_group_size":128}'
```

### Master switch: `SGLANG_MIMO_FUSED_MOE` (default off)

`SGLANG_MIMO_FUSED_MOE=1` enables the fused target-model decode paths:

- the fused MoE kernel (post-attention RMSNorm + router + top-8 + experts + TP all-reduce in one launch) for every
  layer with routed experts, for decode / target-verify batches of up to `SGLANG_MIMO_FUSED_MOE_MAX_TOKENS` (64)
  tokens per rank;
- for batches of up to 32 tokens: the layer-0 dense MLP TP path and the fused embedding + layer-0 input norm /
  FP8 quantization;
- at exactly M = 32 tokens (batch 4 x 8 verify tokens):
  - the MoE kernel takes the attention o_proj partial directly, performs the TP reduction in its prologue, and
    computes the next layer's input RMSNorm + FP8 quantization in its tail;
  - the SWA attention megakernel (qkv GEMM, RoPE + KV store, sliding-window attention, o_proj and the push into
    the MoE input buffers in one kernel) for the sliding-window layers;
  - the fused full-attention tail (RoPE + KV store, split-KV attention, combine, o_proj, push) for the
    full-attention layers, whose qkv GEMM runs through the megakernel's GEMM phase.

Without the master switch none of these paths is used. The DFlash, attention-backend and step-scheduling changes
in the table below are independent of the master switch and on by default.

### Shape scope

The attention megakernels, the draft megakernel and the M = 32 fusions run only for the verify shape M = 32
(batch 4 x 8 draft tokens). Other batch sizes use the generic fused MoE kernel (with the master switch) or the
existing paths; prefill always uses the existing paths. Unsupported layouts fall back per layer and per forward,
with the same decision on every TP rank.

### Per-feature off switches

Each feature can be switched off on its own (all default on):

| Feature | Off switch |
|---|---|
| Fused MoE: attention partial reduced in the MoE prologue (M = 32) | `SGLANG_MIMO_FUSED_MOE_INPUT_TP=0` |
| Fused MoE: next layer's RMSNorm + FP8 quant in the MoE tail | `SGLANG_MIMO_FUSED_MOE_NORM_NEXT=0` |
| Fused MoE: PDL launch behind the attention kernel | `SGLANG_MIMO_MOE_PDL=0` |
| SWA attention block / megakernel | `SGLANG_MIMO_BLK=0` / `SGLANG_MIMO_MEGA=0` |
| Full-attention qkv GEMM through the megakernel | `SGLANG_MIMO_MEGA_FA=0` |
| Fused full-attention tail (layer 0 only: `SGLANG_MIMO_FA2_DENSE=0`) | `SGLANG_MIMO_FA2=0` |
| Full-attention split-KV verify kernel (replaces FA3) | `SGLANG_FA_VERIFY=0` |
| Fused RoPE + KV store | `SGLANG_OPT_MIMO_ROPE_KV=0` |
| Layer-0 dense MLP TP path / fused embedding + norm + quant | `SGLANG_MIMO_DENSE_TP=0` / `SGLANG_MIMO_EMBED_NQ=0` |
| DFlash draft TP fusions / draft megakernel | `SGLANG_DFLASH_TP_FUSED=0` / `SGLANG_DFLASH_MEGA=0` |
| Draft small-M FP8 GEMM / qk-norm + RoPE + KV store kernel | `SGLANG_DFLASH_TP_SKINNY=0` / `SGLANG_DFLASH_FUSED_QKNR_KV=0` |
| Vocab-parallel argmax for the target verify (draft-KV materialization in the verify graph requires it) | `SGLANG_DFLASH_VP_ARGMAX=0` |
| Vocab-parallel draft argmax / embedding | `SGLANG_DFLASH_TP_ARGMAX=0` / `SGLANG_DFLASH_TP_EMBED=0` |
| Local embedding replica (see Memory) | `SGLANG_G12_EMBED_LOCAL=0` |
| Draft-KV materialization inside the verify graph | `SGLANG_DFLASH_MAT_IN_VERIFY=0` |
| One-launch replay preparation per CUDA graph | `SGLANG_STEP_PREP_FUSED=0` |
| Other step-scheduling glue | `SGLANG_DFLASH_AUX_DEFER=0`, `SGLANG_DFLASH_FUSED_PUBLISH=0`, `SGLANG_DFLASH_TOKENS_IN_GRAPH=0`, `SGLANG_FA_SWA_WLOC_FUSED=0`, `SGLANG_DFLASH_MAT_FUSED_KV_WRITE=0`, `SGLANG_DFLASH_MAT_EARLY_FORK=0`, `SGLANG_DFLASH_EMBED_IN_GRAPH=0`, `SGLANG_MIMO_FINAL_CAPTURE_IN_NORM=0`, `SGLANG_FA_VERIFY_FUSED_SEQLENS=0` |

### Memory

The local embedding replica keeps a full copy of the target embedding table on every rank (+1.87 GB per rank),
allocated after the KV pools are sized, so the KV capacity is unchanged. While it is active, online weight updates
are rejected; set `SGLANG_G12_EMBED_LOCAL=0` to disable it.

## Measured configuration

Server (8xH200, one node):

```bash
export MODEL_PATH=/path/to/target/model
export DRAFT_MODEL_PATH="$MODEL_PATH/dflash"

SGLANG_MIMO_FUSED_MOE=1 \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","input_scale_group_size":128}' \
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --context-length 65536 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-model-quantization fp8 \
  --tp-size 8 \
  --ep-size 1 \
  --trust-remote-code \
  --moe-runner-backend humming \
  --mem-fraction-static 0.75 \
  --random-seed 42 \
  --host 127.0.0.1 \
  --port 29999
```

Client: batch 4 x 65536 input tokens, greedy, 512 output tokens:

```bash
until curl -fsS http://127.0.0.1:29999/health >/dev/null; do sleep 5; done
python3 -m sglang.benchmark.one_batch_server \
  --model None \
  --base-url http://127.0.0.1:29999 \
  --batch-size 4 \
  --input-len 65536 \
  --output-len 512 \
  --temperature 0 \
  --seed 42 \
  --show-report
```

Report the SGLang and Humming revisions, GPU model and count, draft-token count, batch size, input and output
lengths, the verify step latency and the average acceptance length.
