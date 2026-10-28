import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import training.context_parallel as cp
import models.qwen_training_kernels as kernels


def test_module_function_interface_routes_native_forward_and_context(monkeypatch):
    import models.model_backend as model_backend

    # Reproduce modern Transformers' global kernel calls, not the legacy
    # self.causal_conv1d_fn interface that the previous tests covered alone.
    implementation = ModuleType('test_qwen_delta_implementation')
    implementation.nn = nn
    exec('''
def causal_conv1d_fn(x, weight, bias=None, **kwargs):
    return x * 3
def torch_chunk_gated_delta_rule(query, key, value, **kwargs):
    return query + 7, None
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(2, 2, 2, groups=2)
    def forward(self, hidden_states, **kwargs):
        x = causal_conv1d_fn(hidden_states, self.conv1d.weight.squeeze(1),
                            self.conv1d.bias, activation='silu', **kwargs)
        return torch_chunk_gated_delta_rule(x, x, x, **kwargs)[0]
''', implementation.__dict__)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    layer = implementation.Qwen3_5GatedDeltaNet()
    untouched = implementation.Qwen3_5GatedDeltaNet()
    model = nn.Sequential(nn.Embedding(3, 2), layer)
    model.get_input_embeddings = lambda: model[0]
    calls = []

    def conv(x, weight, bias=None, *, activation=None, seq_idx=None):
        calls.append(('conv', activation, x.shape[-1]))
        return x * 3

    def delta(q, k, v, *, cp_context=None):
        calls.append(('delta', cp_context))
        return q + 7, None

    x = torch.randn(1, 2, 64, requires_grad=True)
    expected = layer(x)
    assert kernels.install_training_kernels(model, (conv, delta)) == 1
    wrapped_forward = layer.forward
    assert kernels.install_training_kernels(model, (conv, delta)) == 1
    assert layer.forward == wrapped_forward  # no duplicate wrapper
    actual = layer(x, position_ids='layer metadata')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    assert calls == [('conv', 'silu', 64), ('delta', None)]
    calls.clear()
    torch.testing.assert_close(untouched(x), expected)
    assert calls == []  # other model instances keep the original kernels

    def head_exchange(*args, kernel, context, **kwargs):
        calls.append(('head-exchange', context.size))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(cp, 'head_parallel_delta', head_exchange)
    monkeypatch.setattr(cp.dist, 'get_rank', lambda *a: 0)
    monkeypatch.setattr(cp.dist, 'get_world_size', lambda *a: 2)
    monkeypatch.setattr(cp, 'gather', lambda value, *a: torch.cat((value, value), dim=2))
    monkeypatch.setattr(model_backend, 'set_gradient_checkpointing', lambda *a: None)
    with pytest.raises(RuntimeError, match='interrupted'):
        with cp.hybrid_context(model, None):
            torch.testing.assert_close(layer(x), expected)
            assert calls == [('conv', 'silu', 65), ('head-exchange', 2), ('delta', None)]
            raise RuntimeError('interrupted')
    assert layer.causal_conv1d_fn is conv
    assert layer.chunk_gated_delta_rule is delta
    assert layer.forward == wrapped_forward
    assert not hasattr(layer, '_ttt_context_parallel')


def test_legacy_interface_uses_native_kernels_without_replacing_forward():
    layer = type('Qwen3_5GatedDeltaNet', (nn.Module,), {})()
    layer.causal_conv1d_fn = layer.chunk_gated_delta_rule = lambda *a: None
    original = layer.forward
    conv, delta = lambda *a: 1, lambda *a: 2
    assert kernels.install_training_kernels(layer, (conv, delta)) == 1
    assert layer.forward == original
    assert layer.causal_conv1d_fn is conv
    assert layer.chunk_gated_delta_rule is delta


def test_broken_installed_extension_surfaces_original_error(monkeypatch):
    broken = ModuleType('causal_conv1d')

    def broken_import(name):
        if name == 'causal_conv1d_fn':
            raise ImportError('CUDA extension: undefined symbol')
        raise AttributeError(name)

    broken.__getattr__ = broken_import
    monkeypatch.setitem(sys.modules, 'causal_conv1d', broken)
    with pytest.raises(RuntimeError, match='CUDA extension: undefined symbol'):
        kernels.load_training_kernels()
