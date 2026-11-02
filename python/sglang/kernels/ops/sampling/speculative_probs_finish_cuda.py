"""Pinned-runtime small-K finish for the existing opt-in DFlash fusion path.

Torch topk remains external. Unknown runtime or cold capture keeps the existing
partial/native path. JIT/runtime execution failures are never silently replaced.
No model configuration, proposal distribution, RNG or accept/reject logic here.
"""
import hashlib
import importlib.metadata
from pathlib import Path

import torch

RUNTIME_CONTRACT = {
    "torch": "2.14.1+cu130",
    "cuda": "13.0",
    "flashinfer": "0.7.1",
    "torch_softmax": "121b209980aad7648321c28d07a63b41c3633ea90bd8da1047a672a02f8e8c66",
    "fi_sampling": "e6288bbecabc433be8782d4e5fd9c7f9e3ba3815ddcde90176a750455a5daeba",
    "fi_api": "fca8a2ba3ef29ef0e6576f560af71257edf71473f9dcd1cd4d933fd3e20f4da1",
    "fi_renorm": "06cc6aa68dbf102333dbc4995473a58468cf765ac5cde0834d481219d6d79340",
}
_RUNTIME_SUPPORT = {}
_MODULE = None


def _runtime_matches(info):
    return (all(info.get(k) == value for k, value in RUNTIME_CONTRACT.items())
            and info.get("capability") == (9, 0)
            and "H200" in info.get("device_name", ""))


def _probe_runtime(device_index):
    # Cheap version rejection precedes optional imports and header IO.
    if torch.__version__ != RUNTIME_CONTRACT["torch"] or torch.version.cuda != RUNTIME_CONTRACT["cuda"]:
        return False
    try:
        if importlib.metadata.version("flashinfer-python") != RUNTIME_CONTRACT["flashinfer"]:
            return False
        import flashinfer.sampling as fi
        from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
        inc = Path(FLASHINFER_INCLUDE_DIR)
        paths = {
            "torch_softmax": Path(torch.__file__).parent / "include/ATen/native/cuda/PersistentSoftmax.cuh",
            "fi_sampling": inc / "flashinfer/sampling.cuh",
            "fi_api": Path(fi.__file__),
            "fi_renorm": inc.parent / "csrc/renorm.cu",
        }
        info = dict(RUNTIME_CONTRACT)
        info.update({key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()})
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError):
        return False
    props = torch.cuda.get_device_properties(device_index)
    info.update(device_name=props.name, capability=(props.major, props.minor))
    return _runtime_matches(info)


def _capture_in_progress(device_index):
    with torch.cuda.device(device_index):
        return torch.cuda.is_current_stream_capturing()


def can_use_finish(device_index):
    if _capture_in_progress(device_index):
        # No filesystem/version probing or native compilation during capture.
        return _RUNTIME_SUPPORT.get(device_index, False) and _MODULE is not None
    if device_index not in _RUNTIME_SUPPORT:
        _RUNTIME_SUPPORT[device_index] = _probe_runtime(device_index)
    return _RUNTIME_SUPPORT[device_index]


def _module():
    global _MODULE
    if _MODULE is None:
        from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
        from sglang.kernels.jit.utils import load_jit
        _MODULE = load_jit(
            "dflash_smallk_finish_production_v1",
            cuda_files=["sampling/speculative_probs_finish.cuh"],
            cuda_wrappers=[(f"run_{k}_{bits}", f"SmallKFinish<{k}, int{bits}_t>::run")
                           for k in (50, 128) for bits in (32, 64)],
            extra_include_paths=[str(FLASHINFER_INCLUDE_DIR)],
            extra_cuda_cflags=["--fmad=false", "--ftz=false", "--prec-div=true"],
        )
    return _MODULE


def finish(values, indices, ks, ps, target, *, use_p=True, skip_mask=False, prezeroed=True):
    """One launch. Caller owns zeroing when prezeroed=True (no host read).

    All IDs must be distinct valid vocabulary IDs from actual torch.topk;
    ks/ps retain original sampling parameter contracts. Writes target only at
    those IDs when prezeroed=True. Returns debug (softmax, renorm) small tensors.
    The debug stores are included in any timing of this prototype.
    """
    if ps is None and not use_p:
        ps = torch.empty((values.shape[0],), device=values.device, dtype=torch.float32)
    tensors = (values, indices, ks, ps, target)
    if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous()
               and not t.requires_grad and t.device == values.device for t in tensors):
        raise ValueError('contiguous same-device CUDA inference tensors required')
    if values.ndim != 2 or values.shape[1] not in (50, 128) or values.shape[0] not in (8,16,24,32):
        raise ValueError('expected M=8/16/24/32 and K=50/128')
    m, k = values.shape
    if (values.dtype != torch.float32 or indices.dtype != torch.int64 or indices.shape != values.shape
        or ks.dtype not in (torch.int32, torch.int64) or ks.shape != (m,)
        or ps.dtype != torch.float32 or ps.shape != (m,)
        or target.dtype != torch.float32 or target.shape != (m,152576)):
        raise ValueError('invalid finish shape/dtype')
    # Full-storage interval rejection is conservative and capture-safe.
    def span(t):
        s = t.untyped_storage()
        return s.data_ptr(), s.data_ptr() + s.nbytes()
    a, b = span(target)
    if any(a < d and c < b for c, d in map(span, tensors[:-1])):
        raise ValueError('target must not overlap inputs')
    soft = torch.empty_like(values)
    renorm = torch.empty_like(values)
    if not can_use_finish(values.device.index):
        raise RuntimeError('Small-K finish runtime unsupported or cold capture; caller must retain partial path')
    fn = getattr(_module(), f'run_{k}_{32 if ks.dtype == torch.int32 else 64}')
    fn(values, indices, ks, ps, target, soft, renorm, bool(use_p), bool(skip_mask), not prezeroed)
    return soft, renorm
