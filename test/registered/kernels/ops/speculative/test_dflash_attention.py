"""Check persistent draft attention against generic attention without model weights.

Sign-valued queries and keys have exactly unit RMS; identity RoPE and unit norm
weights let the independent generic path consume the same normalized inputs.
The fixture keeps only the physical sliding-window KV rows, even for 64K logical
positions. The CUDA adapter calls production gather/attention directly.
"""

import functools
from pathlib import Path
import unittest

import torch

from sglang.kernels.jit.utils import KERNEL_PATH, cuda_stubs_dir, load_jit
from sglang.kernels.ops.attention.bidir_decode import bidir_decode_attention
from sglang.kernels.ops.speculative.dflash_mega import _src_hash
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@functools.lru_cache(maxsize=1)
def _module():
    """Compile the adapter that includes the current production native header.

    :return: CUDA test adapter.
    :rtype: object
    """
    return load_jit(
        "dflash_attention_regression_" + _src_hash(),
        cuda_files=[str(Path(__file__).with_name("dflash_attention_test.cuh"))],
        cuda_wrappers=[("run", "DFlashAttentionTest::run")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def _fixture(lengths, seed, with_sinks):
    """Construct independent logical pages and compact physical KV storage.

    :param lengths: Four logical cache lengths, including any padded zero rows.
    :type lengths: tuple[int, int, int, int]
    :param seed: Deterministic CPU random seed.
    :type seed: int
    :param with_sinks: Include nonuniform per-head softmax sink logits.
    :type with_sinks: bool
    :return: Native arguments, generic queries, and expected updated KV caches.
    :rtype: tuple
    """
    generator = torch.Generator().manual_seed(seed)
    starts = [max(0, length - 8 - 1023) // 32 * 32 for length in lengths]
    total = sum(length - start for start, length in zip(starts, lengths))
    slots = torch.randperm(total, generator=generator) + 1
    page = torch.zeros((4, max(max(lengths), 8)), dtype=torch.int32)
    locations = torch.zeros(32, dtype=torch.int64)
    cursor = 0
    for request, (start, length) in enumerate(zip(starts, lengths)):
        count = length - start
        page[request, start:length] = slots[cursor : cursor + count].to(torch.int32)
        if length:
            locations[8 * request : 8 * request + 8] = page[
                request, length - 8 : length
            ]
        cursor += count
    qkv = (2 * torch.randint(2, (32, 18, 128), generator=generator) - 1).to(
        torch.bfloat16
    )
    qkv[:, 17] = torch.randn((32, 128), generator=generator).to(torch.bfloat16)
    key = (2 * torch.randint(2, (total + 1, 1, 128), generator=generator) - 1).to(
        torch.bfloat16
    )
    value = torch.randn((total + 1, 1, 128), generator=generator).to(torch.bfloat16)
    key[0].zero_()
    value[0].zero_()
    expected_key, expected_value = key.clone(), value.clone()
    value_scale = 0.375
    valid = locations != 0
    expected_key[locations[valid], 0] = qkv[valid, 16]
    expected_value[locations[valid], 0] = (
        qkv[valid, 17].float() * value_scale
    ).bfloat16()
    sinks = torch.linspace(-2, 9, 16) if with_sinks else torch.empty(0)
    arguments = [
        qkv.flatten(1).cuda(),
        key.cuda(),
        value.cuda(),
        page.cuda(),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        locations.cuda(),
        torch.ones(128, dtype=torch.bfloat16, device="cuda"),
        torch.cat((torch.ones(32), torch.zeros(32))).cuda(),
        sinks.cuda(),
        torch.zeros(64, dtype=torch.int32, device="cuda"),
        torch.zeros(4 * 5 * 32, dtype=torch.int32, device="cuda"),
        torch.empty((32, 16, 128), dtype=torch.bfloat16, device="cuda"),
    ]
    return (
        arguments,
        qkv[:, :16].contiguous().cuda(),
        expected_key.cuda(),
        expected_value.cuda(),
        value_scale,
    )


@unittest.skipUnless(
    _HOPPER, "Persistent draft attention requires SM90 cluster instructions"
)
class TestDFlashAttention(unittest.TestCase):
    """Exercise production attention partitioning, gathers and shared KV reuse.

    .. code-block:: python

        # python test_dflash_attention.py
    """

    def test_reference_partition_and_replay(self):
        """Check short, window-edge, long and padded requests in eager and graph mode.

        :return: None after all independent reference and cache comparisons pass.
        :rtype: None
        """
        module = _module()
        patterns = ((8, 1032, 64008, 65544), (0, 0, 1031, 64001))
        for lengths in patterns:
            for with_sinks in (False, True):
                for layers in (1, 5):
                    with self.subTest(lengths=lengths, sinks=with_sinks, layers=layers):
                        args, query, expected_key, expected_value, scale = _fixture(
                            lengths, 17, with_sinks
                        )
                        reference = bidir_decode_attention(
                            query,
                            expected_key,
                            expected_value,
                            args[3],
                            args[4],
                            block_tokens=8,
                            sm_scale=128**-0.5,
                            window_size=(1023, 1023),
                            sinks=args[8] if with_sinks else None,
                            block_n=32,
                            nsplit=16,
                        )

                        def run():
                            """Reset producer state and invoke actual gather plus attention.

                            :return: None.
                            :rtype: None
                            """
                            args[9].zero_()
                            args[9][8:44].fill_(40)
                            args[10].zero_()
                            args[11].fill_(float("nan"))
                            module.run(*args, layers, scale)

                        for _ in range(2):
                            run()
                            torch.testing.assert_close(
                                args[11], reference, rtol=0, atol=0
                            )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        for _ in range(2):
                            graph.replay()
                            torch.testing.assert_close(
                                args[11], reference, rtol=0, atol=0
                            )
                        torch.testing.assert_close(
                            args[1], expected_key, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            args[2], expected_value, rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
