"""Exactness and retained-activation checks; no CUDA/model download required."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training_checkpointing import (
    checkpoint_group_size, checkpoint_layer_groups, grouped_qwen_hidden,
)


class CheckpointLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, hidden, *, scale):
        def compute(value):
            self.calls += 1
            return value + scale * self.linear(value).tanh()
        return checkpoint(compute, hidden, use_reentrant=False)


def test_nested_checkpoints_retain_fewer_boundaries_with_identical_gradients():
    torch.manual_seed(12)
    original = nn.ModuleList(CheckpointLayer() for _ in range(64))
    grouped = deepcopy(original)
    source = torch.randn(1, 32, 8)
    results = []
    for layers, size in ((original, 1), (grouped, 8)):
        hidden = source.clone().requires_grad_()
        saved_bytes = []

        def pack(tensor):
            if tensor.shape == hidden.shape:
                saved_bytes.append(tensor.numel() * tensor.element_size())
            return tensor

        kwargs = [dict(scale=(index + 1) / 6400) for index in range(64)]
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = checkpoint_layer_groups(layers, hidden, kwargs, size)
        retained = sum(saved_bytes)
        output.square().mean().backward()
        results.append((output.detach(), hidden.grad, retained))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    for first, second in zip(original.parameters(), grouped.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    assert results[1][2] == results[0][2] // 8
    # Nested groups add a bounded extra forward, not quadratic replay.
    assert max(layer.calls for layer in grouped) <= 3


def test_group_size_uses_available_memory():
    assert checkpoint_group_size(64, 100, 6400) == 1
    assert checkpoint_group_size(64, 100, 4000) == 2
    assert checkpoint_group_size(64, 100, 2400) == 4
    assert checkpoint_group_size(64, 100, 1600) == 8
    assert checkpoint_group_size(64, 100, 100) == 8


def test_unsupported_decoder_keeps_existing_forward():
    assert grouped_qwen_hidden(SimpleNamespace(), torch.ones(1, 2), None) is None


@pytest.mark.parametrize("padded", [False, True])
def test_real_qwen_grouped_forward_loss_and_gradients(padded, monkeypatch):
    # The ordinary dev environment can have older Transformers. Validation
    # uses an isolated current SDK; this test needs no pretrained weights.
    implementation = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
                    + ["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=128,
        rope_parameters=dict(rope_theta=10000.0, rope_type="default",
                             partial_rotary_factor=1.0, mrope_section=[1, 1, 2]))
    config._attn_implementation = "eager"
    model = implementation.Qwen3_5TextModel(config).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    grouped = deepcopy(model)
    # Exercise the production MLP/norm tile checkpoints nested inside layer
    # checkpoints, not only vanilla Transformers layers.
    import model_backend
    monkeypatch.setattr(model_backend, "_EXACT_MLP_PROJECTION_TILE_BYTES", 2048)
    monkeypatch.setattr(model_backend, "_EXACT_GATED_NORM_TILE_BYTES", 2048)
    profile = SimpleNamespace(coder_model_profile="qwen3.8-27b",
                              long_training_headroom_min_tokens=1)
    model_backend._install_exact_qwen38_mlp_chunking(grouped, profile)
    model_backend._install_exact_qwen38_gated_norm_chunking(grouped, profile)
    model_backend._install_exact_qwen38_mlp_chunking(model, profile)
    model_backend._install_exact_qwen38_gated_norm_chunking(model, profile)
    grouped._ttt_group_checkpoint_min_tokens = 1
    parameters_before = [(name, id(value)) for name, value in grouped.named_parameters()]
    ids = torch.randint(0, 48, (2 if padded else 1, 16))
    mask = torch.ones_like(ids) if padded else None
    if padded:
        mask[1, -4:] = 0
    expected = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    actual = grouped_qwen_hidden(grouped, ids, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Use a token-dependent weighted loss so every sequence position matters.
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    for (name, first), (_, second) in zip(model.named_parameters(), grouped.named_parameters()):
        assert (first.grad is None) == (second.grad is None), name
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-5, atol=1e-6, msg=name)
    assert parameters_before == [(name, id(value)) for name, value in grouped.named_parameters()]


@pytest.mark.parametrize("one_still_oversized", [False, True])
def test_saved_overflow_tries_all_replicas_and_commits_only_once(
        monkeypatch, tmp_path, one_still_oversized):
    import exact_overflow
    import fast_distributed
    import train_multy_CVaR as training

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10)
    model._ttt_group_checkpoint_min_tokens = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    keys = [(rank, 1) for rank in range(8)]
    examples = [dict(_overlap_key=key, _cooperative_overflow=True) for key in keys]
    metric_names = exact_overflow.METRICS

    def stats(trained):
        return dict(totals={name: float(trained) for name in metric_names},
                    trained_examples=trained, backward_batches=trained,
                    max_ratio=1., max_prefix_ratio=1., entropy_examples=0,
                    training_seconds=0., quarantined_examples=0,
                    deferred_examples=1 - trained,
                    deferred_keys=[keys[-1]] if not trained else [], token_budget=64)

    state = dict(step=2, remaining_keys=keys[:], deferred_keys=keys[:],
                 completed_keys=[(9, 1)], gradients={"weight": torch.full((1, 1), 2.)},
                 parameter_signature=fast_distributed.trainable_parameter_signature(model),
                 normal_stats=stats(1), overflow_stats=stats(0),
                 path=str(tmp_path / 'overflow.pt'))
    commands = {rank: [] for rank in range(1, 8)}
    trainer = SimpleNamespace(
        model=model, optimizer=optimizer, backend=SimpleNamespace(), tokenizer=None,
        _world_size=8, _closed=False, _workers_idle=True, _offloaded=False,
        _token_budgets=[64] * 8, _memory_fraction=.8, _process_label="test",
        _activation_offload_semaphore=None, _work_queue=object(), _exp_dir=tmp_path,
        _warm_overflow_min_tokens=10,
        _command_queues={rank: SimpleNamespace(put=commands[rank].append)
                         for rank in range(1, 8)})

    def queue(items, *, weighted_offload):
        assert not weighted_offload
        assert not hasattr(trainer, "_warm_overflow_min_tokens")
        assert {item["_overlap_key"] for item in items} == set(keys)
        assert all("_cooperative_overflow" not in item for item in items)
        return items

    trainer._queue_examples = queue
    trainer._collect_event = lambda event, ranks: {
        rank: dict(stats=stats(0 if one_still_oversized and rank == 7 else 1))
        for rank in ranks}
    trainer._abort_workers = lambda: pytest.fail("unexpected worker abort")

    def local(*args, **kwargs):
        assert all(commands[rank][0]["kind"] == "train_spo_rs" for rank in commands)
        for rank in commands:
            options = commands[rank][0]["step_cfg"]
            assert options["_exact_replica_force_activation_offload"] is False
            assert options["_exact_replica_overflow_min_tokens"] == 0
        return stats(1)

    def reduce(model, **kwargs):
        model.weight.grad = torch.full_like(model.weight, 28. if one_still_oversized else 36.)

    def fallback(_trainer, _examples, _cfg, _step, saved):
        assert one_still_oversized
        assert saved["remaining_keys"] == [keys[-1]]
        assert saved["replica_checkpoint_groups_attempted"] is True
        assert float(saved["gradients"]["weight"]) == 30.
        assert float(model.weight.detach()) == 10.  # no partial optimizer update
        return {"fallback": True}

    monkeypatch.setattr(fast_distributed, "local_spo_rs_update", local)
    monkeypatch.setattr(fast_distributed, "reduce_trainable_gradients", reduce)
    monkeypatch.setattr(fast_distributed, "broadcast_trainable_parameters", lambda *a, **k: None)
    monkeypatch.setattr(exact_overflow, "complete_warm_overflow", fallback)
    cfg = SimpleNamespace(backend="hf", num_training_gpus=8, training_layout="replicated",
                          advantage_mode="spo-rs", spo_rs_update_epochs=1, grad_clip=1e6)
    result = training.ProcessDistributedTrainer.recover_spo_rs_overflow(
        trainer, examples, cfg, 2, state)
    if one_still_oversized:
        assert result == {"fallback": True}
    else:
        assert state["remaining_keys"] == []
        assert float(state["gradients"]["weight"]) == 38.
        assert float(model.weight.detach()) == pytest.approx(9.62)
        assert result["fast_trained_examples_per_gpu"] == [1] * 8
        assert result["exact_overflow_replica_examples"] == 8
    assert trainer._workers_idle
