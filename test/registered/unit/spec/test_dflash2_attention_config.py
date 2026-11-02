"""Actual HF local-config parsing plus unchanged production configuration ASTs.

No CUDA import/topology required. Only unrelated model-family dispatch is stubbed;
AutoConfig.from_pretrained and every DFlash attention decision execute real code.
"""
import ast
import copy
from enum import Enum
import json
from pathlib import Path
import tempfile
from typing import Any, Optional, Sequence, Tuple
import unittest

from transformers import AutoConfig, PretrainedConfig

ROOT = Path(__file__).resolve().parents[4]
UTILS = ROOT / 'python/sglang/srt/speculative/dflash_utils.py'
MODEL = ROOT / 'python/sglang/srt/models/dflash.py'
CONFIG = ROOT / 'python/sglang/srt/utils/hf_transformers/config.py'
CUSTOMER = dict(architectures=['DFlash2DraftModel'], model_type='qwen3', hidden_size=6144,
    intermediate_size=16384, num_hidden_layers=5, num_attention_heads=128,
    num_key_value_heads=8, head_dim=128, max_position_embeddings=1048576,
    sliding_window=1024, partial_rotary_factor=.5, block_size=8,
    dflash_config=dict(use_swa=True, swa_window_size=1024, conv_kernel_size=2,
                       conv_group_size=16, selector_rank=256, selector_top_k=16))


def production_namespace():
    ns = dict(Any=Any, Optional=Optional, Sequence=Sequence, Tuple=Tuple,
              Enum=Enum, AutoConfig=AutoConfig, PretrainedConfig=PretrainedConfig)
    wanted = {'_cfg_get', '_get_text_config', '_get_dflash_config',
              'is_nemotron_35_draft_config', 'get_dflash_layer_types',
              'get_dflash_attention_sliding_window_size', '_get_dflash_attention_type',
              '_get_dflash_layer_attention_params', '_normalize_dflash2_attention_config'}
    for path in (ROOT / 'python/sglang/srt/layers/radix_attention.py', UTILS, MODEL, CONFIG):
        body = [n for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and (n.name in wanted or n.name == 'AttentionType')]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    cls = next(n for n in ast.parse(CONFIG.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'HfModelConfigParser')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'parse')
    ns.update(_try_load_longcat_config=lambda *a, **k: None,
              _try_load_raw_mamba_config=lambda *a, **k: None,
              _is_deepseek_ocr_model=lambda cfg: False,
              _is_deepseek_ocr2_model=lambda cfg: False,
              DeepseekV41Config=type('UnrelatedDeepseekConfig', (), {}),
              _CONFIG_REGISTRY={}, _GEMMA4_MODEL_TYPES=(),
              get_hf_text_config=lambda config: config.get_text_config())
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(CONFIG), 'exec'), ns)
    return ns


class AttentionConfig(unittest.TestCase):
    def setUp(self):
        self.ns = production_namespace()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def parse(self, raw):
        path = Path(self.tmp.name)
        (path / 'config.json').write_text(json.dumps(raw))
        return self.ns['parse'](None, str(path), trust_remote_code=False, local_files_only=True)

    def params(self, cfg, index=0):
        return self.ns['_get_dflash_layer_attention_params'](cfg, index)

    def test_actual_hf_customer_config_keeps_swa_and_noncausal_default(self):
        cfg = self.parse(CUSTOMER)
        self.assertEqual(cfg.layer_types, ['sliding_attention'] * 5)
        self.assertEqual(self.params(cfg), (1023, self.ns['AttentionType'].ENCODER_ONLY))
        self.assertEqual(cfg.dflash_config, CUSTOMER['dflash_config'])

    def test_explicit_full_layers_override_nested_swa(self):
        raw = dict(CUSTOMER, layer_types=['full_attention'] * 5)
        cfg = self.parse(raw)
        self.assertEqual(cfg.layer_types, raw['layer_types'])
        self.assertEqual(self.params(cfg), (-1, self.ns['AttentionType'].ENCODER_ONLY))

    def test_explicit_mixed_layers_window_and_causality_are_preserved(self):
        for causal in (False, True):
            raw = dict(CUSTOMER, layer_types=['sliding_attention'] + ['full_attention'] * 4,
                       sliding_window=512, is_causal=causal)
            cfg = self.parse(raw)
            self.assertEqual(cfg.layer_types, raw['layer_types'])
            kind = self.ns['AttentionType'].DECODER if causal else self.ns['AttentionType'].ENCODER_ONLY
            self.assertEqual(self.params(cfg), (511, kind))
            self.assertEqual(self.params(cfg, 1), (-1, kind))

    def test_nested_window_used_when_top_level_window_absent(self):
        raw = copy.deepcopy(CUSTOMER); raw.pop('sliding_window')
        self.assertEqual(self.params(self.parse(raw))[0], 1023)

    def test_legacy_config_with_same_nested_fields_remains_unchanged(self):
        raw = dict(CUSTOMER, architectures=['DFlashDraftModel'])
        cfg = self.parse(raw)
        self.assertEqual(cfg.layer_types, ['full_attention'] * 5)
        self.assertEqual(self.params(cfg), (-1, self.ns['AttentionType'].ENCODER_ONLY))
        raw.update(layer_types=['sliding_attention'] * 5, use_sliding_window=True)
        self.assertEqual(self.params(self.parse(raw)), (1023, self.ns['AttentionType'].DECODER))

    def test_raw_v2_config_resolves_without_hf_generated_fields(self):
        self.assertEqual(self.ns['get_dflash_layer_types'](CUSTOMER), ['sliding_attention'] * 5)
        self.assertEqual(self.ns['get_dflash_attention_sliding_window_size'](CUSTOMER), 1023)

    def test_disabled_swa_does_not_invent_sliding_attention(self):
        raw = copy.deepcopy(CUSTOMER); raw['dflash_config']['use_swa'] = False
        self.assertEqual(self.params(self.parse(raw))[0], -1)

    def test_missing_or_nonpositive_swa_window_is_rejected(self):
        for value in (None, 0, -1):
            raw = copy.deepcopy(CUSTOMER); raw.pop('sliding_window')
            raw['dflash_config']['swa_window_size'] = value
            with self.assertRaises(ValueError): self.parse(raw)


if __name__ == '__main__': unittest.main(verbosity=2)
