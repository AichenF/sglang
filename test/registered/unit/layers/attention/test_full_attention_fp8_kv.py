"""CPU contracts for opt-in full-attention FP8 KV cache."""

from types import SimpleNamespace as NS

import pytest
import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fa_verify import can_use_fa_verify
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache import kv_cache_configurator as cfg
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def configure_pool(monkeypatch, *, enabled=True, draft=False, **options):
    monkeypatch.setenv("SGLANG_FULL_ATTENTION_KV_FP8", str(int(enabled)))
    monkeypatch.setenv("SGLANG_USE_HND_KVCACHE", str(int(options.get("hnd", False))))
    monkeypatch.setattr(
        cfg, "get_parallel", lambda: NS(attn_tp_size=1, attn_dcp_size=1)
    )
    monkeypatch.setattr(cfg, "get_spec", lambda: NS(speculative_algorithm=None))
    monkeypatch.setattr(
        cfg,
        "get_exec",
        lambda: NS(
            features=NS(enable_memory_saver=False), kernel=NS(attention_backend="fa3")
        ),
    )
    monkeypatch.setattr(
        cfg, "attention_backends", lambda: options.get("backends", ("fa3", "fa3"))
    )
    monkeypatch.setattr(
        cfg, "get_model", lambda: NS(quantization_param_path=options.get("scale_file"))
    )
    monkeypatch.setattr(
        cfg, "get_disagg", lambda: NS(disaggregation_mode=options.get("disagg", "null"))
    )

    def cpu_pool(**kwargs):
        kwargs.update(device="cpu", enable_alt_stream=False)
        return SWAKVPool(**kwargs)

    monkeypatch.setattr(cfg, "SWAKVPool", cpu_pool)
    config = NS(
        server_args=NS(quantization_param_path=options.get("scale_file")),
        is_draft_worker=draft,
        is_hybrid_swa_compress=True,
        kv_cache_dtype_str="auto",
        kv_cache_dtype=options.get("dtype", torch.bfloat16),
        layer_info=NS(swa_attention_layer_ids=[0], full_attention_layer_ids=[1]),
        draft_swa_full_capacity=False,
        pool_page_size=options.get("page_size", 1),
        post_capture_kv_active=False,
        device=options.get("device", "cuda"),
        model_config=NS(
            hf_text_config=NS(
                model_type=options.get("model_type", "mimo_v2"),
                swa_num_key_value_heads=1,
            ),
            head_dim=192,
            v_head_dim=128,
            swa_head_dim=128,
            swa_v_head_dim=128,
            get_num_kv_heads=lambda *args: 1,
        ),
    )
    return cfg.KVCacheConfigurator._build_hybrid_swa_kv_pool(
        config,
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=32,
        mha_pool_class=MHATokenToKVPool,
    )


@pytest.mark.parametrize(
    "enabled,draft,full_dtype",
    [
        (False, False, torch.bfloat16),
        (False, True, torch.bfloat16),
        (True, False, torch.float8_e4m3fn),
        (True, True, torch.bfloat16),
    ],
)
def test_only_enabled_target_full_pool_uses_fp8(
    monkeypatch, enabled, draft, full_dtype
):
    pool = configure_pool(monkeypatch, enabled=enabled, draft=draft)
    assert pool.get_key_buffer(1).dtype == full_dtype
    assert pool.get_value_buffer(1).dtype == full_dtype
    assert pool.get_key_buffer(0).dtype == torch.bfloat16
    assert pool.get_value_buffer(0).dtype == torch.bfloat16


def test_explicit_draft_dtype_is_not_overridden(monkeypatch):
    pool = configure_pool(monkeypatch, draft=True, dtype=torch.float8_e4m3fn)
    assert pool.get_key_buffer(0).dtype == torch.float8_e4m3fn
    assert pool.get_key_buffer(1).dtype == torch.float8_e4m3fn


def test_disabled_flag_preserves_other_models(monkeypatch):
    pool = configure_pool(monkeypatch, enabled=False, model_type="llama")
    assert pool.get_key_buffer(0).dtype == torch.bfloat16
    assert pool.get_key_buffer(1).dtype == torch.bfloat16


def test_mixed_pool_writer_preserves_values_and_swa_locations(monkeypatch):
    pool = configure_pool(monkeypatch)
    loc = KVWriteLoc(torch.tensor([3]), swa_loc=torch.tensor([2]))
    for layer_id, width, slot in [(1, 192, 3), (0, 128, 2)]:
        key = torch.full((1, 1, width), 1.1, dtype=torch.bfloat16)
        value = torch.full((1, 1, 128), -0.37, dtype=torch.bfloat16)
        dtype = torch.float8_e4m3fn if layer_id == 1 else torch.bfloat16
        k_scale, v_scale = (0.5, 0.25) if layer_id == 1 else (1.0, 1.0)
        expected_key = (key.float() / k_scale).to(dtype)
        expected_value = (value.float() / v_scale).to(dtype)
        pool.set_kv_buffer(NS(layer_id=layer_id), loc, key, value, k_scale, v_scale)
        torch.testing.assert_close(
            pool.get_key_buffer(layer_id)[slot].float(),
            expected_key[0].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            pool.get_value_buffer(layer_id)[slot].float(),
            expected_value[0].float(),
            rtol=0,
            atol=0,
        )
    assert pool.full_kv_pool.k_buffer[0].dtype == torch.uint8
    assert pool.swa_kv_pool.k_buffer[0].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"backends": ("fa3", "fa4")}, "fa3"),
        ({"backends": ("fa4", "fa3")}, "fa3"),
        ({"scale_file": "scales.json"}, "quantization_param_path"),
        ({"dtype": torch.float8_e4m3fn}, "BF16"),
        ({"model_type": "llama"}, "supported"),
        ({"page_size": 16}, "page"),
        ({"device": "cpu"}, "CUDA"),
        ({"hnd": True}, "NHD"),
        ({"disagg": "prefill"}, "disaggregation"),
    ],
)
def test_unsupported_configuration_fails_before_allocation(
    monkeypatch, options, reason
):
    with pytest.raises(ValueError, match=reason):
        configure_pool(monkeypatch, **options)


def test_unified_pool_cannot_silently_ignore_opt_in(monkeypatch):
    monkeypatch.setenv("SGLANG_FULL_ATTENTION_KV_FP8", "1")
    monkeypatch.setattr(cfg, "get_memory", lambda: NS(enable_unified_memory=True))
    monkeypatch.setattr(cfg, "reject_out_of_tree_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(cfg, "is_deepseek_v4", lambda config: False)
    bundle = NS(
        req_to_token_pool=None,
        token_to_kv_pool=None,
        token_to_kv_pool_allocator=None,
        unified_memory_pool=None,
    )
    config = NS(
        is_draft_worker=False,
        model_config=NS(hf_config=NS()),
        is_hybrid_swa=True,
        mambaish_config=None,
        _init_unified_swa_pools=lambda **kwargs: bundle,
    )
    sizes = NS(
        max_running_requests=1,
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=32,
        unified_memory_pool_bytes=4096,
        unified_total_bytes=4096,
    )
    with pytest.raises(ValueError, match="unified"):
        cfg.KVCacheConfigurator._init_pools(
            config, sizes=sizes, req_to_token_pool=None, token_to_kv_pool_allocator=None
        )


def backend_for(pool, selector="auto"):
    backend = object.__new__(FlashAttentionBackend)
    backend.token_to_kv_pool = pool
    backend.kv_cache_dtype = torch.bfloat16
    backend.kv_cache_dtype_str = selector
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    return backend


@pytest.mark.parametrize("selector", ["auto", "bfloat16"])
def test_query_and_descales_follow_physical_layer_dtype(monkeypatch, selector):
    backend = backend_for(configure_pool(monkeypatch), selector)
    query = torch.tensor([1.1, -0.37, 3.0], dtype=torch.bfloat16)
    ks, vs = torch.tensor([0.25]), torch.tensor([0.5])
    layer = NS(layer_id=1, head_dim=192, k_scale=ks, v_scale=vs)
    values = backend.prepare_paged_mha_query(
        query, query, query, layer, logical_batch_size=4, kv_head_num=1, is_prefill=True
    )
    for actual in values[:3]:
        assert actual.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            actual.float(), query.to(torch.float8_e4m3fn).float(), rtol=0, atol=0
        )
    torch.testing.assert_close(values[3], ks.expand(4, 1))
    torch.testing.assert_close(values[4], vs.expand(4, 1))
    ks.fill_(0.75)
    torch.testing.assert_close(values[3], torch.full((4, 1), 0.75))
    layer.layer_id, layer.k_scale, layer.v_scale = 0, None, None
    swa = backend.prepare_paged_mha_query(
        query, None, None, layer, logical_batch_size=4, kv_head_num=1, is_prefill=False
    )
    assert swa[0] is query
    assert swa[3:] == (None, None)


def test_scheduler_uses_full_pool_dtype(monkeypatch):
    backend = backend_for(configure_pool(monkeypatch))
    backend.use_mla = backend._disable_scheduler_metadata_precompute = False
    backend.num_attention_heads, backend.num_kv_heads, backend.head_dim = 16, 1, 192
    backend.page_size, backend.num_splits, backend.has_softcap = 1, 0, False
    backend._get_scheduler_metadata = lambda **kwargs: kwargs
    result = backend._compute_scheduler_metadata(4, 1024, None, None)
    assert result["qkv_dtype"] == torch.float8_e4m3fn


def test_fp8_query_falls_back_from_bf16_verify_kernel(monkeypatch):
    monkeypatch.setenv("SGLANG_FA_VERIFY", "1")
    backend = backend_for(configure_pool(monkeypatch))
    query = torch.ones((32, 16, 192), dtype=torch.bfloat16)
    layer = NS(layer_id=1, head_dim=192, k_scale=None, v_scale=None)
    fp8_query = backend.prepare_paged_mha_query(
        query, None, None, layer, logical_batch_size=4, kv_head_num=1, is_prefill=True
    )[0]
    options = dict(
        forward_mode=NS(is_target_verify=lambda: True),
        num_kv_heads=1,
        num_q_heads=16,
        head_dim=192,
        v_head_dim=128,
        window_size=(-1, -1),
        causal=True,
        page_size=1,
        softcap=0,
        cache_seqlens=torch.zeros(4),
        max_seqlen_q=8,
        kwargs={},
    )
    assert can_use_fa_verify(q=query, **options)
    assert not can_use_fa_verify(q=fp8_query, **options)


def test_auto_regular_pool_preserves_query_and_no_descales():
    backend = backend_for(object())
    query = torch.tensor([1.3, -0.7], dtype=torch.float32)
    layer = NS(
        layer_id=0,
        head_dim=192,
        k_scale=torch.tensor([0.5]),
        v_scale=torch.tensor([0.25]),
    )
    result = backend.prepare_paged_mha_query(
        query, None, None, layer, logical_batch_size=4, kv_head_num=1, is_prefill=False
    )
    assert result[0] is query
    assert result[3:] == (None, None)


def test_regular_fp8_selector_preserves_query_conversion():
    backend = backend_for(object(), "fp8_e4m3")
    backend.kv_cache_dtype = torch.float8_e4m3fn
    layer = NS(layer_id=0, head_dim=192, k_scale=None, v_scale=None)
    query = torch.ones(3, dtype=torch.bfloat16)
    result = backend.prepare_paged_mha_query(
        query, None, None, layer, logical_batch_size=4, kv_head_num=1, is_prefill=False
    )
    assert result[0].dtype == torch.float8_e4m3fn
    assert result[3:] == (None, None)
