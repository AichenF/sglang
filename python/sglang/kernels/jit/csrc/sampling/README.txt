Small-K DFlash finish, opt-in via SGLANG_DFLASH_VERIFY_PROBS_FUSION.
The native top-p clone preserves FlashInfer sampling.cuh SHA e6288bbecabc433be8782d4e5fd9c7f9e3ba3815ddcde90176a750455a5daeba.
Only kernel declaration/name and blockIdx.x -> 0 differ; inputs are row-local shared buffers.
Original Apache notice is in native_sampling_LICENSE.txt.
Native K50/K128 CUB algorithm and all CUDA arithmetic match H200-validated experimental v2.
Torch/FlashInfer versions and source hashes are admitted by speculative_probs_finish_cuda.py.
