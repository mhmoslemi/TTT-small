from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import exact_overflow as overflow
import train_multy_CVaR as training


def example(index, length=12):
    return dict(_overlap_key=(0, index),
                prompt_ids=torch.ones(1, 3, dtype=torch.long),
                response_ids=torch.ones(1, length, dtype=torch.long),
                advantage=(-1.0 if index % 2 else 1.0),
                sample_weight=0.2, rank_entropy_gate=False,
                behavior_logprobs=torch.full((length,), -0.7))


def config():
    return SimpleNamespace(backend="hf", load_in_4bit=False,
                           training_layout="replicated", num_training_gpus=8,
                           advantage_mode="spo-rs", grad_clip=0.5,
                           logprob_chunk=8, rank_update_epochs=1)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 17])
def test_wave_uses_all_cards_without_overlapping_groups(count):
    items = [example(i, 10 + i) for i in range(count)]
    wave = overflow.plan_overflow_wave(items, 8)
    assert [card for _, group in wave for card in group] == list(range(8))
    assert len(wave) == min(4, count)
    assert all(len(group) >= 2 for _, group in wave)
    assert len({overflow.trajectory_key(item) for item, _ in wave}) == len(wave)
    assert wave[0][0] is items[-1]


def test_failed_group_is_widened_not_truncated():
    items = [example(i, 250_000) for i in range(4)]
    wave = overflow.plan_overflow_wave(items, 8, {(0, 3): 8})
    assert len(wave) == 1
    assert wave[0][0] is items[3]
    assert wave[0][1] == tuple(range(8))
    assert wave[0][0]["response_ids"].shape[1] == 250_000


def test_unsupported_backends_keep_existing_path():
    cfg = config()
    assert overflow.supports_warm_overflow(cfg)
    cfg.load_in_4bit = True
    assert not overflow.supports_warm_overflow(cfg)
    cfg.load_in_4bit = False
    cfg.backend = "unsloth"
    assert not overflow.supports_warm_overflow(cfg)


def test_parking_ram_is_checked_before_allocating(monkeypatch):
    import host_memory
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: {
        "total": 1024 * host_memory.GIB, "available": 100 * host_memory.GIB})
    with pytest.raises(RuntimeError, match="insufficient host RAM"):
        overflow.require_parking_capacity(model, 8)


class CountingAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


def prepare_state(tmp_path, model, items):
    model.weight.grad = torch.tensor([[0.2, -0.1]])
    normal = dict(
        totals={name: 0.1 for name in overflow.METRICS},
        max_ratio=1.0, max_prefix_ratio=1.0, entropy_examples=0,
        trained_examples=1, backward_batches=1,
        deferred_examples=len(items),
        deferred_keys=[overflow.trajectory_key(item) for item in items])
    training._save_exact_spo_rs_overflow(tmp_path, 0, model, [normal])
    model.zero_grad(set_to_none=True)
    return training._load_exact_spo_rs_overflow(tmp_path, 0)


class FakeTrainer:
    def __init__(self, model, cfg, tmp_path, compute):
        self.model = model
        self.backend = SimpleNamespace(model=model)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.optimizer = CountingAdam(model.parameters())
        self._closed = False
        self._workers_idle = True
        self._offloaded = False
        self.world_size = 8
        self.parks = self.restores = 0
        self.events = {}
        self.aborted = False
        self._exp_dir = tmp_path
        self._command_queues = {
            rank: SimpleNamespace(put=lambda command, rank=rank: self.command(
                rank, command, compute)) for rank in range(1, 8)}

    def command(self, rank, command, compute):
        if command["kind"] == "overflow_backward":
            self.events[rank] = dict(result=compute(
                self.backend, self.tokenizer, command["example"],
                SimpleNamespace(**command["step_cfg"]), command["gpu_ids"]))
        elif command["kind"] == "sync_update":
            self.events[rank] = {}
        else:
            raise AssertionError(command)

    def _collect_event(self, event, ranks):
        return {rank: self.events.pop(rank) for rank in ranks}

    _cpu_examples = staticmethod(lambda items: list(items))
    reclaim_host_memory = lambda self: None

    def offload_for_generation(self, **kwargs):
        self.parks += 1
        self._offloaded = True

    def restore_after_generation(self):
        if self._offloaded:
            self.restores += 1
            self._offloaded = False

    def _abort_workers(self):
        self.aborted = True


def test_parallel_tail_widens_only_failures_and_applies_same_adam_update(
        tmp_path, monkeypatch):
    import fast_distributed
    cfg = config()
    model = torch.nn.Linear(2, 1, bias=False)
    baseline = torch.nn.Linear(2, 1, bias=False)
    baseline.load_state_dict(model.state_dict())
    items = [example(i, 12 + i) for i in range(4)]
    state = prepare_state(tmp_path, model, items)
    calls = []
    grads = {i: torch.tensor([[0.05 * i, -0.12 * (i + 1)]]) for i in range(4)}

    def compute(backend, tokenizer, item, cfg, devices):
        index = overflow.trajectory_key(item)[1]
        calls.append((index, len(devices)))
        if index == 3 and len(devices) < 8:
            return {"completed": False}
        return dict(completed=True, gradient_pack=overflow.pack_cpu_gradients({"weight": grads[index]}),
                    stats=dict(totals={name: 0.1 for name in overflow.METRICS},
                               max_ratio=1.0, max_prefix_ratio=1.0,
                               entropy_examples=0))

    trainer = FakeTrainer(model, cfg, tmp_path, compute)
    monkeypatch.setattr(overflow, "compute_overflow_work", compute)
    monkeypatch.setattr(overflow, "require_parking_capacity", lambda *args: None)
    broadcasts = []
    monkeypatch.setattr(fast_distributed, "broadcast_trainable_parameters",
                        lambda *a, **k: broadcasts.append(trainer.optimizer.steps))
    expected_opt = CountingAdam(baseline.parameters())
    baseline.weight.grad = torch.tensor([[0.2, -0.1]]) + sum(grads.values())
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), cfg.grad_clip,
                                  error_if_nonfinite=True)
    expected_opt.step()
    result = overflow.complete_warm_overflow(trainer, items, cfg, 0, state)

    assert trainer.optimizer.steps == 1
    assert broadcasts == [1]
    assert trainer.parks == trainer.restores == 1
    assert not trainer.aborted
    assert sorted(calls) == [(0, 2), (1, 2), (2, 2), (3, 2), (3, 8)]
    assert state["remaining_keys"] == []
    assert len(state["completed_keys"]) == 4
    assert result["exact_overflow_inprocess"]
    torch.testing.assert_close(model.weight, baseline.weight)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(trainer.optimizer.state[model.weight][name],
                                   expected_opt.state[baseline.weight][name])
    # Successful gradients survive an interruption without replaying them.
    saved = training._load_exact_spo_rs_overflow(tmp_path, 0)
    assert saved["completed_keys"] == state["completed_keys"]
    assert saved["remaining_keys"] == []


def test_all_gpu_failure_is_durable_without_partial_step_or_restart(tmp_path, monkeypatch):
    cfg = config()
    model = torch.nn.Linear(2, 1, bias=False)
    initial = model.weight.detach().clone()
    items = [example(0)]
    state = prepare_state(tmp_path, model, items)
    compute = lambda *a, **k: {"completed": False}
    trainer = FakeTrainer(model, cfg, tmp_path, compute)
    monkeypatch.setattr(overflow, "compute_overflow_work", compute)
    monkeypatch.setattr(overflow, "require_parking_capacity", lambda *a: None)
    with pytest.raises(RuntimeError, match="refusing a restart loop"):
        overflow.complete_warm_overflow(trainer, items, cfg, 0, state)
    assert trainer.optimizer.steps == 0
    assert trainer.aborted
    assert state["remaining_keys"] == [(0, 0)]
    torch.testing.assert_close(model.weight, initial)


def test_overflow_backward_preserves_loss_weight_and_every_token(monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    model._ttt_strict_exact_long_training = True
    model.get_input_embeddings = lambda: model
    backend = SimpleNamespace(model=model, set_training_mode=model.train)
    cfg = config()
    item = example(0, 91)
    item["sample_weight"] = 1 / 512

    def score(model, examples, with_grad, chunk, **kwargs):
        n = examples[0]["response_ids"].shape[1]
        assert n == 91
        x = torch.stack((torch.linspace(-1, 1, n), torch.ones(n)), dim=1)
        return [torch.nn.functional.logsigmoid(model(x).flatten())]

    monkeypatch.setattr(overflow, "temporary_gpu_shards", lambda *a: nullcontext())
    monkeypatch.setattr(training, "compute_batched_token_logprobs", score)
    current = score(model, [item], True, 8)[0]
    epsilon, low, high, kl = training._clipped_policy_options(cfg)
    loss, _ = training.clipped_policy_loss(
        cfg, current, item["behavior_logprobs"], None, item["advantage"],
        clip_epsilon=epsilon, clip_epsilon_low=low,
        clip_epsilon_high=high, kl_coef=kl)
    (item["sample_weight"] * loss).backward()
    expected = model.weight.grad.detach().clone()
    result = overflow.compute_overflow_work(
        backend, SimpleNamespace(pad_token_id=0), item, cfg, (0, 1))
    assert result["completed"]
    torch.testing.assert_close(overflow.unpack_cpu_gradients(result["gradient_pack"])["weight"], expected)
    assert model.weight.grad is None


def test_gradient_ipc_uses_one_storage_per_dtype_not_per_parameter():
    values = {f"lora_{i}": torch.arange(6, dtype=torch.float32).view(2, 3)
              for i in range(1500)}
    values["bf16"] = torch.tensor([0.25, 0.5], dtype=torch.bfloat16)
    packed = overflow.pack_cpu_gradients(values)
    assert len(packed["buffers"]) == 2
    restored = overflow.unpack_cpu_gradients(packed)
    for name, expected in values.items():
        torch.testing.assert_close(restored[name], expected)
    assert len({value.untyped_storage().data_ptr() for value in restored.values()}) == 2


def test_overlap_retains_exact_original_examples_for_warm_tail():
    trainer = training.ProcessDistributedTrainer.__new__(training.ProcessDistributedTrainer)
    items = [example(0), example(1)]
    trainer._queue_examples = lambda examples, **kwargs: examples
    trainer._spo_rs_overlap = dict(sealed=False, queued_examples=0,
                                   supplied_old=0, scheduled_keys=set(),
                                   overflow_examples={})
    assert trainer.queue_spo_rs_overlap(items) == 2
    for item in items:
        saved = trainer._spo_rs_overlap["overflow_examples"][overflow.trajectory_key(item)]
        assert saved is item
        assert saved["sample_weight"] == 0.2


def test_resume_keeps_replicas_even_after_legacy_offload_exhaustion(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    import yaml
    import gpu_runtime

    saved = yaml.safe_load((Path(training.__file__).parent / "configs/erdos.yaml").read_text())
    saved.update(model_name="Qwen/Qwen3.8-27B", coder_model_name="Qwen/Qwen3.8-27B",
                 training_layout="replicated", backend="hf", load_in_4bit=False,
                 advantage_mode="spo-rs", spo_rs_adaptive_beta=False,
                 num_training_gpus=8, available_gpu_ids="0,1,2,3,4,5,6,7")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    training._write_exact_sharded_recovery(
        tmp_path, 1, "previous oversized trajectory",
        overflow_gradient_file=".exact_spo_rs_overflow_step0001.pt",
        replica_offload_exhausted=True, deferred_keys=[[0, 1]])
    monkeypatch.setattr(sys, "argv", [training.__file__, "--resume", str(tmp_path),
                                     "--fast", "--strategies"])
    monkeypatch.setenv("AVAILABLE_GPUS", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", lambda: {
        i: gpu_runtime.GPUMemory(i, "RTX PRO 6000 Blackwell", 95.6, 95.6)
        for i in range(8)})
    cfg, merged = training.load_config()
    assert cfg.training_layout == "replicated"
    assert cfg.fast
    assert cfg._exact_inprocess_recovery_step == 1
    assert "_exact_sharded_recovery_step" not in merged


@pytest.mark.parametrize("fail", [False, True])
def test_temporary_placement_unwinds_without_replacing_parameters(monkeypatch, fail):
    import accelerate
    import accelerate.hooks
    import accelerate.utils
    import fast_distributed
    import host_memory

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    identities = dict(model.named_parameters())
    optimizer = CountingAdam(model.parameters())
    backend = SimpleNamespace(model=model, _placement_model=model,
                              _trainer_is_offloaded=True)
    old_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], int):
            args = ("cpu", *args[1:])
        return old_to(tensor, *args, **kwargs)

    def dispatch(module, **kwargs):
        assert all(dict(module.named_parameters())[key] is parameter
                   for key, parameter in identities.items())
        module.hf_device_map = kwargs["device_map"]
        module.to = lambda *a, **k: None  # dispatch's instance-level wrapper

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (96 * 2**30, 96 * 2**30))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(fast_distributed, "set_long_rollout_memory_ceiling", lambda *a: 0.96)
    monkeypatch.setattr(host_memory, "reclaim_unused_host_memory", lambda *a, **k: None)
    monkeypatch.setattr(accelerate.utils, "get_balanced_memory", lambda *a, **k: k["max_memory"])
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *a, **k: {"0": 0, "1": 1})
    monkeypatch.setattr(accelerate, "dispatch_model", dispatch)
    monkeypatch.setattr(accelerate.hooks, "remove_hook_from_submodules", lambda *a: None)
    expected_context = pytest.raises(RuntimeError, match="injected") if fail else nullcontext()
    with expected_context:
        with overflow.temporary_gpu_shards(backend, (0, 1)):
            model(torch.ones(1, 2)).sum().backward()
            if fail:
                raise RuntimeError("injected")
    assert "to" not in model.__dict__
    assert "hf_device_map" not in model.__dict__
    assert backend._trainer_is_offloaded
    assert {id(p) for p in model.parameters()} == {
        id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        assert parameter is identities[name]
        assert parameter.grad is not None
        assert parameter.device.type == parameter.grad.device.type == "cpu"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_real_cuda_shards_preserve_gradients_and_parameter_identity():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64)

        def forward(self, hidden):
            return hidden + torch.tanh(self.proj(hidden))

    class Model(torch.nn.Module):
        _no_split_modules = ["Block"]

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 64)
            self.layers = torch.nn.ModuleList([Block() for _ in range(8)])
            self.head = torch.nn.Linear(64, 16)

        def forward(self, tokens):
            hidden = self.embed(tokens)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.head(hidden)

    torch.manual_seed(7)
    model = Model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    model(tokens).square().mean().backward()
    expected = {name: parameter.grad.clone()
                for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    parameters = dict(model.named_parameters())
    backend = SimpleNamespace(model=model, _placement_model=model,
                              _trainer_is_offloaded=True)
    with overflow.temporary_gpu_shards(backend, (0, 1)):
        assert {p.device.index for p in model.parameters()} == {0, 1}
        model(tokens.to(next(model.embed.parameters()).device)).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        assert parameter.device.type == "cpu"
        torch.testing.assert_close(parameter.grad, expected[name], rtol=5e-4, atol=2e-6)
