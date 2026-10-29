"""Output convolution -> rounded BF16 -> residual/RMSNorm component contract.

This test does not enable a model path or certify persistent DFlash2 inference.
Run with pytest on an allocated Hopper GPU. Missing CUDA is a failure, not a skip.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-small")

ROOT = Path(__file__).resolve().parents[5]
H, BLOCK, GROUP, EPS = 6144, 8, 128, 1e-5
# A BF16 ULP near 1 is 1/128. Different parallel FP32 variance reductions,
# plus the installed RMSNorm's numerator rounding, permit one such ULP.
# Residual (including convolution rounding) must agree bit-for-bit.
NORM_RTOL, NORM_ATOL = 0.008, 0.008


@pytest.fixture(scope="module", autouse=True)
def runtime():
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.test.test_utils import published_topology

    assert torch.cuda.is_available(), "FIXTURE: allocated CUDA GPU required"
    assert torch.cuda.get_device_capability()[0] == 9, "FIXTURE: Hopper required"
    disabled = {name: os.environ[name] for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")
                if os.environ.get(name, "").lower() not in ("", "0", "false")}
    assert not disabled, f"FIXTURE: compiled production oracle is disabled: {disabled}"
    with published_topology(cuda_graph_config=CudaGraphConfig()):
        yield


@pytest.fixture(scope="module")
def reference():
    from torch._dynamo.backends.registry import lookup_backend
    from sglang.srt.layers.layernorm import RMSNorm

    compile_count = 0
    official_inductor = lookup_backend("inductor")

    def observed_inductor(graph_module, example_inputs, **kwargs):
        nonlocal compile_count
        compiled = official_inductor(graph_module, example_inputs, **kwargs)
        compile_count += 1
        return compiled

    path = ROOT / "python/sglang/srt/models/dflash.py"
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_grouped_conv")
    namespace = {"torch": torch, "F": F, "get_compiler_backend": lambda: observed_inductor,
                 "_is_npu": False}
    # Execute the unchanged production AST, including its torch.compile decorator.
    # Raise instead of silently accepting an eager fallback on specialization limit.
    limit = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    options = {"suppress_errors": False, limit: 32}
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        options["fail_on_recompile_limit_hit"] = True
    with torch._dynamo.config.patch(options):
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        norm = RMSNorm(H, eps=EPS).to(device="cuda", dtype=torch.bfloat16).eval()

        def run(x, delta, base, residual, gamma):
            torch._dynamo.mark_static(x, 1)
            torch._dynamo.mark_static(delta, 1)
            torch._dynamo.mark_static(delta, 2)
            conv = namespace["_grouped_conv"](x, delta, base, BLOCK, H // GROUP, GROUP, base.shape[0])
            assert compile_count > 0, "FIXTURE: official Inductor backend was never invoked"
            conv_before_norm = conv.clone()
            with torch.no_grad():
                norm.weight.copy_(gamma)
                normalized, updated = norm(conv, residual.clone())
            return normalized, updated, conv_before_norm

        yield run


@pytest.fixture(scope="module")
def fused():
    path = ROOT / "python/sglang/kernels/ops/speculative/dflash_conv_norm.py"
    assert path.exists(), "FEATURE_ABSENT: fused DFlash output-conv/residual/RMSNorm is missing"
    spec = importlib.util.spec_from_file_location("dflash_conv_norm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grouped_conv_add_rmsnorm


def inputs(rows, taps=3, layout="projection", seed=431):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def dyadic(shape, divisor=32):
        return (torch.randint(-16, 17, shape, generator=generator, device="cuda").float() / divisor).bfloat16()

    x, base = dyadic((rows, H)), dyadic((taps, H), 64)
    if layout == "projection":
        delta = dyadic((rows, 2, taps, H // GROUP), 64)[:, 1]
    elif layout == "sliced":
        delta = dyadic((rows, 2 * taps, 3 * H // GROUP), 64)[:, ::2, ::3]
    elif layout == "broadcast":
        delta = dyadic((1, taps, H // GROUP), 64).expand(rows, -1, -1)
    else:
        raise AssertionError(layout)
    residual = dyadic((rows, H), 64)
    gamma = torch.linspace(0.375, 1.875, H, device="cuda").bfloat16()
    return x, delta, base, residual, gamma


def oracle(x, delta, base, residual, gamma):
    """Independent CPU float64 stencil, with explicit boundary rounding."""
    x, delta, base, residual, gamma = [v.detach().cpu().double() for v in (x, delta, base, residual, gamma)]
    conv = torch.zeros_like(x)
    for row in range(x.shape[0]):
        for tap in range(min(base.shape[0], row % BLOCK + 1)):
            conv[row] += (base[tap] + delta[row, tap].repeat_interleave(GROUP)) * x[row - tap]
    rounded_conv = conv.bfloat16().double()
    summed = rounded_conv + residual
    normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + EPS) * gamma
    return rounded_conv.bfloat16(), summed.bfloat16(), normalized.bfloat16()


def exact(actual, expected):
    torch.testing.assert_close(actual, expected.to(actual.device), rtol=0, atol=0)


def norm_close(actual, expected):
    torch.testing.assert_close(actual, expected.to(actual.device), rtol=NORM_RTOL, atol=NORM_ATOL)


@pytest.mark.parametrize("rows", [8, 32])
@pytest.mark.parametrize("layout", ["projection", "sliced", "broadcast"])
@torch.inference_mode()
def test_compiled_conv_real_rmsnorm_and_float64(reference, fused, rows, layout):
    x, delta, base, residual, gamma = inputs(rows, layout=layout)
    originals = [value.clone() for value in (x, delta, base, residual, gamma)]
    expected_norm, expected_residual, expected_conv = reference(x, delta, base, residual, gamma)
    oracle_conv, oracle_residual, oracle_norm = oracle(x, delta, base, residual, gamma)
    exact(expected_conv, oracle_conv)
    exact(expected_residual, oracle_residual)
    norm_close(expected_norm, oracle_norm)
    normalized, updated = fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    assert updated is residual, "Residual must be updated in place"
    assert normalized.data_ptr() not in (x.data_ptr(), residual.data_ptr())
    exact(updated, expected_residual)
    norm_close(normalized, expected_norm)
    norm_close(normalized, oracle_norm)
    for actual, old in zip((x, delta, base, gamma), (originals[0], originals[1], originals[2], originals[4])):
        exact(actual, old)
    # A zero residual exposes the convolution BF16 boundary directly.
    residual.zero_()
    _, updated = fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    exact(updated, expected_conv)


@pytest.mark.parametrize("boundary", ["sum", "gamma"])
@torch.inference_mode()
def test_rounding_boundary_witnesses(fused, boundary):
    """Exact stable witnesses prevent the broad RMS tolerance hiding early casts."""
    x = torch.ones((8, H), device="cuda", dtype=torch.bfloat16)
    delta = torch.zeros((8, 1, H // GROUP), device="cuda", dtype=torch.bfloat16)
    base = torch.ones((1, H), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    gamma = torch.empty(H, device="cuda", dtype=torch.bfloat16)
    if boundary == "sum":
        # FP32 sum 1 + 1/256 rounds down to BF16 1. Keep the unrounded sum
        # for normalization; reloading the stored residual produces 1.4921875.
        residual.fill_(1 / 256)
        gamma.fill_(1.5)
    else:
        # Variance is 2.5. For gamma=65/128, correct even-channel output is
        # 0.3203125; BF16-rounding the normalized value before gamma gives
        # 0.322265625. Both answers are far from their rounding boundaries.
        residual.zero_()
        residual[:, 1::2] = 1
        gamma.fill_(65 / 128)
    _, expected_residual, expected_norm = oracle(x, delta, base, residual, gamma)
    normalized, updated = fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    exact(updated, expected_residual)
    exact(normalized, expected_norm)


@pytest.mark.parametrize("rows", [8, 32])
@torch.inference_mode()
def test_graph_replay_changes_all_inputs(reference, fused, rows):
    values = inputs(rows)
    x, delta, base, residual, gamma = values
    for _ in range(3):
        fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        normalized, updated = fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    for iteration in range(4):
        replacement = inputs(rows, seed=900 + iteration)
        replacement[-1].mul_(0.75 + 0.125 * iteration)
        for destination, source in zip(values, replacement):
            destination.copy_(source)
        expected_norm, expected_residual, _ = reference(*replacement)
        graph.replay()
        exact(updated, expected_residual)
        norm_close(normalized, expected_norm)
        for actual, expected in zip((x, delta, base, gamma), (replacement[0], replacement[1], replacement[2], replacement[4])):
            exact(actual, expected)


@torch.inference_mode()
def test_no_cross_block_nonfinite_history(fused):
    x, delta, base, residual, gamma = inputs(32)
    x[6].fill_(float("inf"))
    x[7].fill_(float("nan"))
    expected_conv, expected_residual, expected_norm = oracle(x, delta, base, residual, gamma)
    normalized, updated = fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    assert torch.isfinite(normalized[BLOCK:]).all()
    exact(updated[BLOCK:], expected_residual[BLOCK:])
    norm_close(normalized[BLOCK:], expected_norm[BLOCK:])


@pytest.mark.parametrize("alias", ["hidden", "delta", "base", "gamma"])
def test_residual_overlap_rejected_before_launch(fused, alias):
    x, delta, base, residual, gamma = inputs(8)
    if alias == "hidden":
        x = residual.view_as(residual)
    elif alias == "delta":
        delta = residual.flatten()[:8 * 3 * (H // GROUP)].view(8, 3, H // GROUP)
    elif alias == "base":
        base = residual[:3]
    else:
        gamma = residual[0]
    before = residual.clone()
    with pytest.raises(ValueError, match="overlap|alias"):
        fused(x, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    exact(residual, before)


def test_offset_dlpack_storage_overlap_rejected(fused):
    _, delta, base, _, gamma = inputs(8)
    backing = torch.randn((9, H), device="cuda", dtype=torch.bfloat16)
    residual = backing[:8]
    hidden = torch.from_dlpack(backing[1:9])
    assert hidden.untyped_storage().data_ptr() != residual.untyped_storage().data_ptr()
    before = residual.clone()
    with pytest.raises(ValueError, match="overlap|alias"):
        fused(hidden, delta, base, residual, gamma, EPS, BLOCK, GROUP)
    exact(residual, before)


@pytest.mark.parametrize("invalid", ["cpu", "dtype", "width", "residual_stride", "gamma_shape", "epsilon"])
def test_invalid_contract_rejected(fused, invalid):
    x, delta, base, residual, gamma = inputs(8)
    eps = EPS
    if invalid == "cpu":
        x, delta, base, residual, gamma = [v.cpu() for v in (x, delta, base, residual, gamma)]
    elif invalid == "dtype":
        gamma = gamma.float()
    elif invalid == "width":
        x = x[:, :H // 2].contiguous()
    elif invalid == "residual_stride":
        residual = residual.t().contiguous().t()
    elif invalid == "gamma_shape":
        gamma = gamma[None]
    else:
        eps = float("nan")
    with pytest.raises((ValueError, TypeError)):
        fused(x, delta, base, residual, gamma, eps, BLOCK, GROUP)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
