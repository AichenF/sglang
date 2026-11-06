"""CPU contracts for the legacy DFlash noncausal SWA checkpoint schema.

Legacy ``swa_window_size`` is a left/right radius, not the native HF window
including the current token. Native sliding/mixed layers take precedence;
otherwise legacy use_swa=True wins over absent or all-full layer_types.
The mask checks below are CPU semantic oracles, not FlashAttention execution.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig, Qwen3Config

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.dflash import _get_dflash_layer_attention_params
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    get_dflash_legacy_attention_sliding_window_size,
    is_nemotron_35_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Config(SimpleNamespace):
    def get_text_config(self):
        return self


def _legacy_config(**overrides):
    values = {
        "num_hidden_layers": 5,
        "dflash_config": {"use_swa": True, "swa_window_size": 1024},
    }
    values.update(overrides)
    return _Config(**values)


def _qwen3_legacy_config(**overrides):
    # Relevant fields from the checkpoint; construction is local and loads no
    # weights. Real Qwen3 normalization must be covered, not only a test double.
    values = {
        "architectures": ["DFlashDraftModel"],
        "num_hidden_layers": 5,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "sliding_window": 1024,
        "dflash_config": {"use_swa": True, "swa_window_size": 1024},
    }
    values.update(overrides)
    return Qwen3Config(**values)


def _assert_legacy_layers(config, window, attn_type=AttentionType.ENCODER_ONLY):
    assert get_dflash_legacy_attention_sliding_window_size(config) == window
    assert get_dflash_attention_sliding_window_size(config) == window
    for layer in range(config.num_hidden_layers):
        assert _get_dflash_layer_attention_params(config, layer) == (
            window,
            attn_type,
        )


def test_real_qwen3_normalized_all_full_restores_legacy_swa():
    config = _qwen3_legacy_config()
    assert config.layer_types == ["full_attention"] * 5
    assert config.use_sliding_window is False
    assert config.sliding_window is None
    _assert_legacy_layers(config, 1024)


@pytest.mark.parametrize("layer_types", [None, ["full_attention"] * 5])
def test_legacy_true_wins_over_missing_or_all_full_schema(layer_types):
    config = _legacy_config(layer_types=layer_types)
    _assert_legacy_layers(config, 1024)


def test_explicit_all_full_conflict_is_disabled_by_legacy_false():
    config = _legacy_config(
        layer_types=["full_attention"] * 5,
        sliding_window=1024,
        dflash_config={"use_swa": False, "swa_window_size": 1024},
    )
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) is None
    for layer in range(5):
        assert _get_dflash_layer_attention_params(config, layer) == (
            -1,
            AttentionType.ENCODER_ONLY,
        )


@pytest.mark.parametrize("window", [1, 2, 1024])
def test_legacy_radius_is_not_decremented_or_limited_by_block_size(window):
    config = _legacy_config(
        block_size=8,
        dflash_config={"use_swa": True, "swa_window_size": window},
    )
    _assert_legacy_layers(config, window)


def test_nested_legacy_window_precedes_top_level_window():
    _assert_legacy_layers(_legacy_config(sliding_window=2048), 1024)


@pytest.mark.parametrize(
    "nested", [{"use_swa": True}, {"use_swa": True, "swa_window_size": None}]
)
def test_missing_nested_window_falls_back_to_top_level(nested):
    config = _legacy_config(dflash_config=nested, sliding_window=256)
    _assert_legacy_layers(config, 256)


def test_qwen_erased_top_level_window_fails_without_path_lookup():
    config = _qwen3_legacy_config(dflash_config={"use_swa": True})
    config._name_or_path = "/must-not-read-a-checkpoint/config.json"
    assert config.sliding_window is None
    forbidden = AssertionError("SWA resolution must not reload config from a path")
    with (
        patch("builtins.open", side_effect=forbidden),
        patch.object(Path, "open", side_effect=forbidden),
        patch.object(PretrainedConfig, "get_config_dict", side_effect=forbidden),
    ):
        for resolve in (
            get_dflash_legacy_attention_sliding_window_size,
            get_dflash_attention_sliding_window_size,
            lambda value: _get_dflash_layer_attention_params(value, 0),
        ):
            with pytest.raises(ValueError, match="window|sliding"):
                resolve(config)


@pytest.mark.parametrize("legacy", [{}, {"use_swa": False}, {"swa_window_size": 1024}])
def test_legacy_window_requires_explicit_true(legacy):
    config = _legacy_config(dflash_config=legacy, sliding_window=1024)
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) is None
    assert _get_dflash_layer_attention_params(config, 0) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1024", [], {}])
@pytest.mark.parametrize("source", ["nested", "top_level"])
def test_invalid_legacy_window_is_rejected(value, source):
    config = _legacy_config(dflash_config={"use_swa": True})
    if source == "nested":
        config.dflash_config["swa_window_size"] = value
        # An invalid nested value cannot be hidden by a valid fallback.
        config.sliding_window = 1024
    else:
        config.sliding_window = value
    with pytest.raises(ValueError, match="window|swa"):
        get_dflash_legacy_attention_sliding_window_size(config)


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_non_boolean_legacy_switch_is_rejected(value):
    config = _legacy_config(dflash_config={"use_swa": value, "swa_window_size": 1024})
    with pytest.raises(ValueError, match="use_swa|bool"):
        get_dflash_legacy_attention_sliding_window_size(config)


@pytest.mark.parametrize("is_causal", [False, True])
def test_explicit_causality_has_priority_over_legacy_default(is_causal):
    config = _legacy_config(layer_types=["full_attention"] * 5, is_causal=is_causal)
    expected = AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY
    _assert_legacy_layers(config, 1024, expected)


@pytest.mark.parametrize(
    "layer_types",
    [
        ["sliding_attention"] * 5,
        [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    ],
)
@pytest.mark.parametrize("legacy_window", [512, 0])
def test_native_sliding_and_mixed_keep_native_window_and_layer_types(
    layer_types, legacy_window
):
    config = _legacy_config(
        layer_types=layer_types,
        sliding_window=1024,
        dflash_config={"use_swa": True, "swa_window_size": legacy_window},
    )
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) == 1023
    for index, layer_type in enumerate(layer_types):
        expected = (
            (1023, AttentionType.DECODER)
            if layer_type == "sliding_attention"
            else (-1, AttentionType.ENCODER_ONLY)
        )
        assert _get_dflash_layer_attention_params(config, index) == expected


@pytest.mark.parametrize("is_causal", [False, True])
def test_native_mixed_explicit_causality_is_preserved(is_causal):
    config = _legacy_config(
        layer_types=["full_attention", "sliding_attention"],
        num_hidden_layers=2,
        sliding_window=64,
        is_causal=is_causal,
    )
    expected = AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert _get_dflash_layer_attention_params(config, 0) == (-1, expected)
    assert _get_dflash_layer_attention_params(config, 1) == (63, expected)


def test_native_missing_window_is_not_rescued_by_legacy_window():
    config = _legacy_config(layer_types=["sliding_attention"] * 5)
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    with pytest.raises(ValueError, match="sliding_window|window"):
        get_dflash_attention_sliding_window_size(config)


@pytest.mark.parametrize("layer_types", [[], ["unknown_attention"], "full_attention"])
def test_legacy_switch_does_not_hide_invalid_native_layer_types(layer_types):
    config = _legacy_config(layer_types=layer_types)
    with pytest.raises(ValueError, match="layer|entry|entries"):
        _get_dflash_layer_attention_params(config, 0)


def test_real_qwen3_native_mixed_schema_stays_native():
    config = _qwen3_legacy_config(
        use_sliding_window=True,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) == 1023
    assert _get_dflash_layer_attention_params(config, 0) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )
    assert _get_dflash_layer_attention_params(config, 1) == (
        1023,
        AttentionType.DECODER,
    )


def _nemotron_config(layer_types):
    return _legacy_config(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=2,
        has_embed_tokens=True,
        has_lm_head=False,
        quantization_config={"quant_algo": "W4A16_NVFP4"},
        eagle_aux_hidden_state_layer_ids=[1, 5],
        layer_types=layer_types,
        sliding_window=None,
        dflash_config={
            "use_swa": True,
            "swa_window_size": 1024,
            "target_layer_ids": [0, 4],
            "sample_from_anchor": False,
        },
    )


def test_nemotron_native_sliding_keeps_nested_fallback_and_hf_window():
    config = _nemotron_config(["sliding_attention"] * 2)
    assert is_nemotron_35_draft_config(config)
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) == 1023
    for layer in range(2):
        assert _get_dflash_layer_attention_params(config, layer) == (
            1023,
            AttentionType.DECODER,
        )


@pytest.mark.parametrize("layer_types", [None, ["full_attention"] * 2])
def test_nemotron_missing_or_all_full_schema_does_not_enable_legacy(layer_types):
    config = _nemotron_config(layer_types)
    assert is_nemotron_35_draft_config(config)
    assert get_dflash_legacy_attention_sliding_window_size(config) is None
    assert get_dflash_attention_sliding_window_size(config) is None
    for layer in range(2):
        assert _get_dflash_layer_attention_params(config, layer) == (
            -1,
            AttentionType.ENCODER_ONLY,
        )


def test_dict_and_nested_text_config_are_supported_without_mutation():
    configs = [
        {
            "num_hidden_layers": 5,
            "dflash_config": {"use_swa": True, "swa_window_size": 1024},
        },
        {
            "text_config": {"num_hidden_layers": 5, "sliding_window": 256},
            "sliding_window": 512,
            "dflash_config": {"use_swa": True},
        },
    ]
    before = copy.deepcopy(configs)
    for _ in range(2):
        for config, expected in zip(configs, [1024, 256]):
            assert get_dflash_legacy_attention_sliding_window_size(config) == expected
            assert get_dflash_attention_sliding_window_size(config) == expected
    assert configs == before


def test_repeated_resolution_does_not_mutate_qwen_or_native_configs():
    legacy = _qwen3_legacy_config()
    native = _legacy_config(layer_types=["sliding_attention"] * 5, sliding_window=64)
    legacy_before = copy.deepcopy(legacy.to_dict())
    native_before = copy.deepcopy(vars(native))
    for _ in range(2):
        _assert_legacy_layers(legacy, 1024)
        assert get_dflash_attention_sliding_window_size(native) == 63
        _assert_legacy_layers(legacy, 1024)
    assert legacy.to_dict() == legacy_before
    assert vars(native) == native_before


def _cpu_visible_keys(config, prefix_len, block_size, row):
    """Apply the documented FA radius/causality contract to integer positions."""
    window, attn_type = _get_dflash_layer_attention_params(config, 0)
    query = prefix_len + row
    causal = attn_type == AttentionType.DECODER
    return [
        key
        for key in range(prefix_len + block_size)
        if (window < 0 or abs(key - query) <= window) and (not causal or key <= query)
    ]


@pytest.mark.parametrize("row,first_key", [(0, 1024), (3, 1027), (7, 1031)])
def test_legacy_1024_block_row_mask_has_old_symmetric_radius(row, first_key):
    visible = _cpu_visible_keys(_qwen3_legacy_config(), 2048, 8, row)
    assert visible == list(range(first_key, 2056))
    # Every row sees the complete short draft block, including future rows.
    assert all(key in visible for key in range(2048, 2056))
    assert first_key - 1 not in visible


@pytest.mark.parametrize(
    "row,expected", [(0, [4, 5, 6]), (1, [5, 6, 7]), (2, [6, 7, 8]), (3, [7, 8])]
)
def test_legacy_radius_one_clips_both_sides_of_a_longer_block(row, expected):
    config = _legacy_config(dflash_config={"use_swa": True, "swa_window_size": 1})
    assert _cpu_visible_keys(config, 5, 4, row) == expected


def test_explicit_causal_legacy_mask_does_not_see_future_block_rows():
    config = _legacy_config(is_causal=True)
    assert _cpu_visible_keys(config, 2048, 8, 0) == list(range(1024, 2049))


@pytest.mark.parametrize("prefix_len", [0, 1023, 1024, 1025])
def test_legacy_first_row_mask_at_history_boundaries(prefix_len):
    visible = _cpu_visible_keys(_legacy_config(), prefix_len, 8, 0)
    first_key = 0 if prefix_len <= 1024 else 1
    assert visible == list(range(first_key, prefix_len + 8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
