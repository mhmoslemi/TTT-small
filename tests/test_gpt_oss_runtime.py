"""CPU regressions for compiled attention layouts and an undrained queue."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from gpt_oss_training import install_attention_layout_guards


def _rotary(q, k, cos, sin):
    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def _attention(query, key, value, *, mask, sinks):
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    scores = (query @ key.transpose(-2, -1)) / query.shape[-1] ** .5
    scores = scores.masked_fill(~mask, -torch.inf)
    sink = sinks[None, :, None, None].expand(*scores.shape[:-1], 1)
    weights = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
    return weights @ value


@pytest.mark.parametrize("batch,length,window,hooked", [
    (1, 17, None, False), (2, 11, 4, True)])
def test_compiled_attention_normalizes_layout_without_changing_gradients(
        monkeypatch, batch, length, window, hooked):
    compiled_rope = torch.compile(_rotary, backend="aot_eager", fullgraph=True)
    compiled_attention = torch.compile(_attention, backend="aot_eager", fullgraph=True)
    seen = []

    def rotary_kernel(q, k, *args):
        seen.append(("rotary", q.is_contiguous(), k.is_contiguous()))
        return compiled_rope(q, k, *args)

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
    angle = torch.randn(1, 1, length, 4, dtype=torch.float64)
    cos, sin = angle.cos(), angle.sin()
    qr, kr = _rotary(q, k, cos, sin)
    expected = _attention(qr, kr, v, mask=mask, sinks=model.sinks)
    actual = model(q, k, v, cos, sin, mask)
    weights = torch.randn_like(expected)
    inputs = (q, k, v, model.sinks)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    assert seen == [("rotary", True, True), ("attention", True, True, True)]


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
