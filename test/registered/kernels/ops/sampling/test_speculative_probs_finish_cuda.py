"""Real public dispatch to the compiled production small-K finish (one H200)."""
import os
from unittest.mock import patch

import pytest
import torch

from test_speculative_probs import FLAG, cuda_required, exact, inputs, reference


@pytest.mark.parametrize('bs,k',[(1,50),(4,50),(1,128),(4,128)])
def test_public_finish_eager_and_changed_graph(bs,k,monkeypatch):
    from sglang.srt.speculative import dflash_utils as original
    from sglang.kernels.ops.sampling import speculative_probs_finish_cuda as finish
    x,info=inputs(bs,k)
    real=finish.finish
    calls=[]
    def observe(*args,**kwargs):
        result=real(*args,**kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(finish,'finish',observe)
    monkeypatch.setenv(FLAG,'1')
    kwargs=dict(next_token_logits=x,sampling_info=info,bs=bs,draft_token_num=8,
                max_top_k=k,uniform_top_k_value=k)
    expected=reference(x,info,bs,k,k)
    actual=original.build_speculative_verify_target_probs(**kwargs)
    exact(actual,expected)
    assert len(calls)==1, 'Public eligible branch did not invoke production finish'
    assert finish.can_use_finish(x.device.index), 'Validated runtime was rejected'
    for _ in range(3):original.build_speculative_verify_target_probs(**kwargs)
    calls.clear()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=original.build_speculative_verify_target_probs(**kwargs)
    assert len(calls)==1
    for iteration in range(4):
        x.copy_(x.roll(97,1));info.temperatures.fill_(.7+.1*iteration);info.top_ps.fill_(.7+.05*iteration)
        expected=reference(x,info,bs,k,k)
        graph.replay();exact(actual,expected)
        assert len(calls)==1, 'Graph replay re-entered Python'
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        actual=original.build_speculative_verify_target_probs(**kwargs)
    exact(actual,expected)
    names=[event.name for event in prof.events() if event.device_type==torch.autograd.DeviceType.CUDA]
    assert sum('smallk_finish::kernel' in name for name in names)==1, names
    assert not any('softmax_warp_forward' in name or 'TopPRenormProbKernel' in name for name in names), names


@pytest.mark.parametrize('reason',['runtime_mismatch','no_cpu_maximum','full_vocab','flag_off','cold_capture'])
def test_public_fallback_keeps_native_math(reason,monkeypatch):
    from sglang.srt.speculative import dflash_utils as original
    from sglang.kernels.ops.sampling import speculative_probs_finish_cuda as finish
    k=152576 if reason=='full_vocab' else 50
    x,info=inputs(1,k)
    maximum=None if reason=='no_cpu_maximum' else k
    kwargs=dict(next_token_logits=x,sampling_info=info,bs=1,draft_token_num=8,
                max_top_k=maximum,uniform_top_k_value=k)
    def forbidden(*a,**kw):raise AssertionError('Unsupported/cold branch invoked finish')
    # Source4 runner uses deterministic native option on BOTH sides for fullV,
    # where unchanged native atomic ordering is already known nondeterministic.
    expected=reference(x,info,1,maximum,k)
    monkeypatch.setenv(FLAG,'0' if reason=='flag_off' else '1')
    if reason=='runtime_mismatch':monkeypatch.setitem(finish._RUNTIME_SUPPORT,x.device.index,False)
    if reason=='cold_capture':
        # Warm unchanged partial path, then prove the real capture guard declines.
        with patch.object(finish,'can_use_finish',return_value=False):
            for _ in range(3):original.build_speculative_verify_target_probs(**kwargs)
        monkeypatch.setattr(finish,'_MODULE',None)
        monkeypatch.setattr(finish,'finish',forbidden)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=original.build_speculative_verify_target_probs(**kwargs)
        graph.replay()
    else:
        monkeypatch.setattr(finish,'finish',forbidden)
        actual=original.build_speculative_verify_target_probs(**kwargs)
    exact(actual,expected)
