DFlash2 sampling component update, 2026-11-02

This branch publishes the optional sampling probability preparation and selector bonus-gather paths measured on H200. The persistent DFlash2 decoder is still under numerical investigation. These measurements do not certify the complete worker or an end-to-end speedup.

Runtime and activation
- Measured runtime: H200, PyTorch2.14.1+cu130, CUDA13.0, FlashInfer0.7.1. Native header/API identities are checked by the CUDA helpers. Unsupported requests fall back to the original implementation.
- SGLANG_DFLASH_VERIFY_PROBS_FUSION=1 enables the small-top-k preparation/finish path.
- SGLANG_DFLASH_AIR_PREPARE=1 enables the no-top-k/top-p AIR preparation path.
- SGLANG_DFLASH_SELECTOR_FUSED_BONUS=1 enables selector bonus gather for the supported domain.
- All three options default to0. Warm CUDA modules eagerly before graph capture. Use SGLANG_DFLASH_MEGA=0 and SGLANG_DFLASH_TP_FUSED=0 for the engineering sampling-only serving comparison.
- The measured domain is batch1/2/4, block8, vocabulary152576. Small-top-k finishing supports K50/128; AIR preparation supports no-top-k plus top-p.

Measured target-probability preparation
Protocol: T1, p.95, graph7x50, native default timing options, alternating arms; deterministic exact-byte and changed-graph controls run separately. The source4/source5 regressions relative to each other are retained.
B mode native_us source4_us source5_us native/source5
1 k50 108.021116 97.525759 97.755518 1.105013
1 k128 119.509764 106.345596 105.781755 1.129777
1 full_vocab 172.345600 163.977604 163.800316 1.052169
1 no_topk 153.098879 151.971197 146.076803 1.048071
2 k50 128.871040 114.535685 114.983683 1.120777
2 k128 137.186556 121.342716 121.062403 1.133189
2 full_vocab 198.182392 189.305592 190.399361 1.040877
2 no_topk 178.481922 175.182076 170.970249 1.043936
4 k50 168.000641 154.702082 154.633598 1.086443
4 k128 174.672642 160.187511 160.339203 1.089394
4 full_vocab 248.984966 231.941757 233.262711 1.067401
4 no_topk 224.327679 216.526718 213.813114 1.049176

Sampling evidence: job5520349, reportSHA 0fcdfc5bd8ce6b7613a84d8e11c7724cae099095b3fde7b87211c4bb2a0b20fa. Production-dispatch validation job5519844 also restored native/reference probability bytes, metadata, unchanged RNG and changed-graph writes.

Measured selector bonus dispatch
Graph B1:38.280640→32.161920us; B2:38.762560→33.200641us; B4:40.701122→34.570241us. Job5519410 retained156 strict controls, reportSHA 65b18c813a6fb7e9239fa2b5a8498e6de54ff9cc255d6ab67c4357f026d19f27. The combined sampling+bonus complete worker has not been accepted.

Measured coefficient projection experiment
Job5530895:50 strict-native-byte controls, graph7x50 and50 distinct outputs/arm. Stage0 current direct-split4:47.390718us; new shared-split4:23.276801us (2.036x component speedup). Native8.231680us and old shared-split8 21.503999us are retained controls. This helper is being integrated/tested independently; these component numbers do not establish a complete decoder gain. ReportSHA a4062c2d38ce259ca8c3863917ee6a6537f380d9a5a7e22f5ff5e11dcad505c4.

Current complete decoder limitation
The latest failed-numerics decoder diagnostic remains generic587.796783us vsMega1973.107147us, outputrelativeL2 .0744137 against unchanged .02 gate. Do not infer model acceptance from the component results or add isolated stage timings as a proven end-to-end saving.

Validation commands
PYTHONPATH=python python -m pytest test/registered/unit/spec/test_verify_probs_air.py test/registered/unit/spec/test_verify_probs_fusion.py test/registered/unit/spec/test_verify_probs_finish_cuda.py test/registered/unit/spec/test_dflash2_attention_config.py
On the allocated CUDA runtime: PYTHONPATH=python python -m pytest test/registered/kernels/ops/sampling/test_speculative_probs.py test/registered/kernels/ops/sampling/test_speculative_probs_finish_cuda.py

The machine-readable companion preserves all measured12 rows, negative source5/source4 comparisons, hashes and scope limitations. Raw campaign artifacts remain outside the source repository.

Conditional estimate (not a measured decoder result)
The isolated10-stage direct/shared sums are474.074879/233.769600us. If these costs translate additively into the failed1973.107147us decoder, with no overlap, occupancy, scheduling or other changes, predicted time is1732.801869us (12.179% less time,1.13868x). The new256-thread isolated helper still needs integration into the512-thread persistent kernel. This scenario does not predict parity with generic587.796783us or numerical correctness. The companion gains_estimate JSON includes an Amdahl sensitivity table and all12 measured sampling comparisons.

Shared4 source fragment
python/sglang/kernels/jit/csrc/distributed/dflash_coefficients_shared4.cuh is the verbatim helper measured by job5530895 (SHA 80e7c50eaeaa40f02530a5534516a6fda88f151f2e75ea8e4000f10a53089261). It expects the existing projection prefix definitions ConvP,Smem,projection_prefetch,bar_sync,red_release,H6144,GRID120,NCT256,PROJ_SHARED_BYTES49152 and8computewarps. It is an internal include fragment, not a standalone launch entry. No default runtime dispatch is changed to call it; complete512-thread persistent integration and original numerical gates run separately.
