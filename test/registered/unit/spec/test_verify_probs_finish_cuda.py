"""Fail-closed host admission and preserved validated native arithmetic."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]


def module():
    p=ROOT/'python/sglang/kernels/ops/sampling/speculative_probs_finish_cuda.py'
    spec=importlib.util.spec_from_file_location('finish_under_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class FinishAdmission(unittest.TestCase):
    def test_exact_runtime_and_each_mismatch(self):
        m=module()
        info=dict(m.RUNTIME_CONTRACT,device_name='NVIDIA H200',capability=(9,0))
        self.assertTrue(m._runtime_matches(info))
        for key in m.RUNTIME_CONTRACT:
            with self.subTest(key=key):
                self.assertFalse(m._runtime_matches(dict(info,**{key:'different'})))
        self.assertFalse(m._runtime_matches(dict(info,device_name='NVIDIA H100')))
        self.assertFalse(m._runtime_matches(dict(info,capability=(8,0))))
        self.assertFalse(m._runtime_matches({}))

    def test_cold_capture_declines_without_probe_or_compile(self):
        m=module()
        with patch.object(m,'_capture_in_progress',return_value=True), patch.object(m,'_probe_runtime',side_effect=AssertionError('capture must not read files')):
            self.assertFalse(m.can_use_finish(0))
            m._RUNTIME_SUPPORT[0]=True
            self.assertFalse(m.can_use_finish(0))
            m._MODULE=object()
            self.assertTrue(m.can_use_finish(0))
            self.assertFalse(m.can_use_finish(1))

    def test_eager_mismatch_cached_as_fallback(self):
        m=module()
        with patch.object(m,'_capture_in_progress',return_value=False), patch.object(m,'_probe_runtime',return_value=False) as probe:
            self.assertFalse(m.can_use_finish(0));self.assertFalse(m.can_use_finish(0))
            probe.assert_called_once_with(0)

    def test_kernel_and_clone_match_actual_gpu_validated_v2(self):
        base=ROOT/'python/sglang/kernels/jit/csrc/sampling'
        for file,digest in [('speculative_probs_finish.cuh','e127b0d7cacb4c9a8b92686827b68cb96f93051986edb182cc378f9ffbb4f4b4'),('native_topp_clone.cuh','e6a34b7945d32633ac401c89116123f936f6ff75da6e5facfc2d36e6750dd4df')]:
            self.assertEqual(hashlib.sha256((base/file).read_bytes()).hexdigest(),digest)

if __name__=='__main__':unittest.main()
