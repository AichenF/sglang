"""Reject DFlash2 exports that would silently omit trained draft components."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashDraftModel,
    DFlashGroupedConv,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _checkpoint_model(*, selector=True, convolution=True):
    # The loader only needs registered parameters. Use the real v2 modules
    # without constructing distributed attention or allocating a KV pool.
    cls = DFlash2DraftModel if selector else DFlashDraftModel
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.projector_type = "linear"
    model.candidate_selector = (
        CandidateSelector(hidden_size=8, vocab_size=16, state_rank=2, top_k=4)
        if selector
        else None
    )
    layer = nn.Module()
    layer.attention_conv = DFlashGroupedConv(8, 4, 2, 4) if convolution else None
    layer.mlp_conv = DFlashGroupedConv(8, 4, 2, 4) if convolution else None
    model.layers = nn.ModuleList([layer])
    model.norm = nn.LayerNorm(8)
    return model


def _weights(model):
    return {
        name: torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) + 0.5
        for name, param in model.named_parameters()
    }


@pytest.mark.parametrize(
    "selector,convolution,prefix",
    ((True, True, ""), (True, True, "model."), (False, False, "")),
)
def test_checkpoint_loads_declared_components_exactly(prefix, selector, convolution):
    model = _checkpoint_model(selector=selector, convolution=convolution)
    expected = _weights(model)
    model.load_weights((prefix + name, weight) for name, weight in expected.items())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "unexpected",
    (
        "candidate_selector.predecessor_codebook",
        "model.layers.0.attention_conv.base_kernel",
        "layers.0.mlp_conv.kernel_projection.weight",
    ),
)
def test_checkpoint_rejects_v2_weights_without_the_declared_component(unexpected):
    model = _checkpoint_model(selector=False, convolution=False)
    with pytest.raises(ValueError, match="DFlash2 checkpoint parameter"):
        model.load_weights([(unexpected, torch.zeros(1))])


def test_rotary_cache_extras_remain_ignored():
    model = _checkpoint_model()
    weights = _weights(model)
    weights["layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(4)
    model.load_weights(weights.items())


def test_selector_config_cannot_instantiate_a_v1_draft():
    config = SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=1,
        dflash_config={"selector_rank": 2, "selector_top_k": 4},
    )
    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        DFlashDraftModel(config)


def test_v2_architecture_requires_a_selector_before_building_layers():
    config = SimpleNamespace(hidden_size=8, num_hidden_layers=1, dflash_config={})
    with pytest.raises(ValueError, match="selector_rank"):
        DFlash2DraftModel(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
