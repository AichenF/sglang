"""Compare paired production TMA stages with two original independent copies."""

import functools
import time
import unittest
from pathlib import Path

import torch

from sglang.kernels.jit.utils import KERNEL_PATH, cuda_stubs_dir, load_jit
from sglang.kernels.ops.speculative.dflash_mega import _src_hash
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
_PHASES = ((0, 2304, 6144), (1, 6144, 2048), (2, 4096, 6144), (3, 6144, 2048))


@functools.lru_cache(maxsize=1)
def _module():
    """Compile the test adapter against the current production weight issuer.

    :return: Production-header CUDA test adapter.
    :rtype: object
    """
    return load_jit(
        "dflash_weight_pair_stages_" + _src_hash(),
        cuda_files=[str(Path(__file__).with_name("dflash_weight_pairs_test.cuh"))],
        cuda_wrappers=[("run", "DFlashWeightPairsTest::run")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def exercise_phase(phase, rows, columns):
    """Check every paired stage under eager execution and changed-input graph replay.

    :param phase: Main-projection production phase index.
    :type phase: int
    :param rows: Actual projection weight row count.
    :type rows: int
    :param columns: Actual complete K dimension.
    :type columns: int
    :return: Observed fixture coverage without a performance claim.
    :rtype: dict

    .. code-block:: python

        coverage = exercise_phase(2, 4096, 6144)
    """
    index = torch.arange(rows * columns, device="cuda", dtype=torch.int32)
    pattern = ((index * 73244475) ^ (index >> 11) ^ (index >> 23)).to(torch.uint8)
    weights = pattern.reshape(rows, columns).clone()
    paired, reference = torch.empty_like(pattern), torch.empty_like(pattern)
    started = time.monotonic()
    module = _module()
    jit = time.monotonic() - started
    module.run(weights, paired, reference, phase)
    torch.cuda.synchronize()
    torch.testing.assert_close(paired, reference, rtol=0, atol=0)
    original_stages = paired.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.run(weights, paired, reference, phase)
    for mask in (37, 193):
        weights.copy_((pattern ^ mask).reshape(rows, columns))
        paired.fill_(29)
        reference.fill_(71)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(paired, reference, rtol=0, atol=0)
        torch.testing.assert_close(paired, original_stages ^ mask, rtol=0, atol=0)
    return dict(
        phase=phase,
        rows=rows,
        columns=columns,
        bytes=rows * columns,
        paired_stages=rows * columns // 16384,
        eager_calls=1,
        graph_replays=2,
        changed_input_and_output_poison=True,
        jit_or_cache_lookup_seconds=jit,
        GPU_event_time_microseconds=None,
    )


@unittest.skipUnless(_HOPPER, "Requires SM90 Hopper TMA instructions")
class TestDFlashWeightPairs(unittest.TestCase):
    """Cover every main projection without trained weights or TP communication."""

    def test_all_main_projections(self):
        """Require byte equality for every stage in all four projection shapes.

        :return: None after all four fixtures pass.
        :rtype: None
        """
        for phase, rows, columns in _PHASES:
            with self.subTest(phase=phase, rows=rows, columns=columns):
                exercise_phase(phase, rows, columns)


if __name__ == "__main__":
    unittest.main()
