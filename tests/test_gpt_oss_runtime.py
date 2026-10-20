"""CPU regressions for compiled attention layouts and an undrained queue."""

from pathlib import Path
import importlib.metadata
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from gpt_oss_training import install_attention_layout_guards
from model_backend import (
    _is_packed_mxfp4, _require_packed_mxfp4_training_runtime)
from train_multy_CVaR import _coder_vllm_persistent_workers


def _rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin,
                          second * cos + first * sin), dim=-1)
    return rotate(q), rotate(k)


def _attention(query, key, value, *, mask, sinks):
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    scores = (query @ key.transpose(-2, -1)) / query.shape[-1] ** .5
    scores = scores.masked_fill(~mask, -torch.inf)
    sink = sinks[None, :, None, None].expand(*scores.shape[:-1], 1)
    weights = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
    return weights @ value


def test_packed_mxfp4_training_requires_modern_unsloth(monkeypatch):
    cfg = SimpleNamespace(coder_model_profile="gpt-oss-120b")
    versions = {"unsloth": "2026.9.12", "unsloth_zoo": "2026.9.8",
                "transformers": "5.17.0"}
    monkeypatch.delenv("UNSLOTH_MXFP4_NO_DEQUANTIZE", raising=False)
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    monkeypatch.setattr(
        importlib.metadata, "version", lambda package: versions[package])
    _require_packed_mxfp4_training_runtime(cfg)
    assert os.environ["UNSLOTH_MXFP4_KEEP_PACKED"] == "1"
    assert "UNSLOTH_MXFP4_NO_DEQUANTIZE" not in os.environ

    versions["unsloth"] = "2026.9.11"
    with pytest.raises(RuntimeError, match="packed-MXFP4"):
        _require_packed_mxfp4_training_runtime(cfg)


def test_packed_mxfp4_runtime_does_not_modify_qwen(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    monkeypatch.delenv("UNSLOTH_MXFP4_NO_DEQUANTIZE", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: pytest.fail(name))
    for name in ("qwen3-8b", "qwen3.8-27b"):
        _require_packed_mxfp4_training_runtime(
            SimpleNamespace(coder_model_profile=name))
    assert os.environ["UNSLOTH_MXFP4_KEEP_PACKED"] == "0"
    assert "UNSLOTH_MXFP4_NO_DEQUANTIZE" not in os.environ


def test_only_gpt_oss_training_uses_transient_coder_engines():
    assert _coder_vllm_persistent_workers(SimpleNamespace(
        coder_model_profile="gpt-oss-120b", no_train=False)) == 0
    assert _coder_vllm_persistent_workers(SimpleNamespace(
        coder_model_profile="gpt-oss-120b", no_train=True)) is None
    for profile in ("qwen3-8b", "qwen3.8-27b"):
        assert _coder_vllm_persistent_workers(SimpleNamespace(
            coder_model_profile=profile, no_train=False)) is None


@pytest.fixture
def packed_model(monkeypatch):
    class PackedExpert(torch.nn.Parameter):
        def __new__(cls):
            weight = torch.Tensor._make_subclass(
                cls, torch.zeros((2, 8, 2, 16), dtype=torch.uint8), False)
            weight.mxfp4_scales = torch.ones((2, 8, 2), dtype=torch.uint8)
            return weight

    # Avoid importing Unsloth and mutating Transformers during a CPU unit test.
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mxfp4_dequant", SimpleNamespace(
        is_mxfp4_expert_param=lambda weight: isinstance(weight, PackedExpert)))
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=2, quantization_config={
        "quant_method": "mxfp4", "dequantize": True})
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for layer in model.layers:
        layer.experts = torch.nn.Module()
        layer.experts.gate_up_proj = PackedExpert()
        layer.experts.down_proj = PackedExpert()
    return model


def test_packed_mxfp4_accepts_actual_packed_experts_with_dequantize_true(packed_model):
    assert _is_packed_mxfp4(packed_model)


@pytest.mark.parametrize("failure", ["dense", "missing_layer", "missing_scales"])
def test_packed_mxfp4_rejects_incomplete_or_dense_experts(packed_model, failure):
    # Metadata claiming packed storage must never excuse dense/missing tensors.
    packed_model.config.quantization_config["dequantize"] = False
    if failure == "dense":
        packed_model.layers[1].experts.down_proj = torch.nn.Parameter(
            torch.zeros((2, 64, 8)), requires_grad=False)
    elif failure == "missing_layer":
        del packed_model.layers[1].experts
    else:
        packed_model.layers[1].experts.down_proj.mxfp4_scales = None
    assert not _is_packed_mxfp4(packed_model)


@pytest.mark.parametrize("batch,length,window,hooked", [
    (1, 17, None, False), (2, 11, 4, True)])
def test_compiled_attention_normalizes_layout_without_changing_gradients(
        monkeypatch, batch, length, window, hooked):
    compiled_rope = torch.compile(_rotary, backend="aot_eager", fullgraph=True)
    compiled_attention = torch.compile(_attention, backend="aot_eager", fullgraph=True)
    seen = []

    def rotary_kernel(q, k, cos, sin):
        seen.append(("rotary", q.is_contiguous(), k.is_contiguous(),
                     cos.is_contiguous(), sin.is_contiguous()))
        return compiled_rope(q, k, cos, sin)

    def attention_kernel(q, k, v, **kwargs):
        seen.append(("attention", q.is_contiguous(), k.is_contiguous(), v.is_contiguous()))
        return compiled_attention(q, k, v, **kwargs)

    sink_module = SimpleNamespace(flex_attention=attention_kernel)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.flex_attention.attention_sink", sink_module)
    apply_rotary_pos_emb = rotary_kernel

    def forward_function(self, q, k, v, cos, sin, mask):
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return sink_module.flex_attention(q, k, v, mask=mask, sinks=self.sinks)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sinks = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))

        def forward(self, *args):
            return forward_function(self, *args)

    torch.manual_seed(12)
    model = Attention()
    if hooked:
        # Quantized replica placement installs this actual Accelerate wrapper:
        # forward becomes a partial and the bound method lives in _old_forward.
        from accelerate.hooks import (
            ModelHook, add_hook_to_module, remove_hook_from_module)
        add_hook_to_module(model, ModelHook())
    placed_forward = model.forward
    assert install_attention_layout_guards(model) == 1
    if hooked:
        assert model.forward is placed_forward  # Keep Accelerate's routing.
        remove_hook_from_module(model)
        add_hook_to_module(model, ModelHook())
        # Repeat installation after the same hook removal/redispatch lifecycle
        # used between generation and training. It must remain idempotent.
        flex_guard = sink_module.flex_attention
        assert install_attention_layout_guards(model) == 1
        assert sink_module.flex_attention is flex_guard
    # The original kernels remain the delegates, not an eager replacement.
    assert sink_module.flex_attention.__wrapped__ is attention_kernel
    raw = [torch.randn(batch, length, heads, 4, dtype=torch.float64)
           for heads in (4, 2, 2)]
    q, k, v = [x.transpose(1, 2).detach().requires_grad_() for x in raw]
    assert not v.is_contiguous()  # BLHD storage viewed as BHLD, as in the run
    positions = torch.arange(length)
    mask = positions[:, None] >= positions[None, :]
    if window is not None:
        mask &= positions[:, None] - positions[None, :] < window
    # GPT-OSS rotary tensors are [B, L, head_dim/2]. This reproduces the
    # transposed stride that broke the checkpointed compiled backward.
    cos = torch.randn(batch, 2, length, dtype=torch.float64).transpose(1, 2)
    sin = torch.randn(batch, 2, length, dtype=torch.float64).transpose(1, 2)
    assert not cos.is_contiguous() and not sin.is_contiguous()
    qr, kr = _rotary(q, k, cos, sin)
    expected = _attention(qr, kr, v, mask=mask, sinks=model.sinks)
    actual = checkpoint(
        lambda q, k, v, cos, sin: model(q, k, v, cos, sin, mask),
        q, k, v, cos, sin, use_reentrant=False)
    weights = torch.randn_like(expected)
    inputs = (q, k, v, model.sinks)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    rotary_seen = [entry for entry in seen if entry[0] == "rotary"]
    attention_seen = [entry for entry in seen if entry[0] == "attention"]
    assert len(rotary_seen) >= 2  # forward plus checkpoint recomputation
    assert attention_seen
    assert all(entry == ("rotary", True, True, True, True)
               for entry in rotary_seen)
    assert all(entry == ("attention", True, True, True)
               for entry in attention_seen)


def test_failed_trainer_exits_with_unconsumed_tensor_work():
    # A real feeder pipe with no consumer reproduces the atexit join from the
    # run. Without cancel_join_thread the subprocess never reaches process exit.
    script = """
import multiprocessing as mp
import time
from train_multy_CVaR import ProcessDistributedTrainer
trainer = object.__new__(ProcessDistributedTrainer)
trainer._work_queue = mp.get_context('spawn').Queue()
trainer._result_queue = None
trainer._command_queues = {}
trainer._work_queue.put(b'x' * 1024 * 1024)
time.sleep(.05)
trainer._close_queues()
print('queue closed', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "queue closed" in result.stdout
