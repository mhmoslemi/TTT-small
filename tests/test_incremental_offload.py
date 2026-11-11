"""CPU checks for incremental transfers; no model download or GPU workload."""

import copy
import ast
from contextlib import nullcontext
import gc
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from runtime import host_memory, incremental_offload as incremental


def small_gate(monkeypatch, *, available=288):
    memory = dict(total=1024, available=available)
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    monkeypatch.setattr(incremental, "COPY_CHUNK_BYTES", 64)
    monkeypatch.setattr(incremental, "MIN_COPY_BYTES", 16)
    gate = host_memory.HostMemoryAdmission(multiprocessing.get_context("spawn"),
                                          hardware_profile="B200_full_ram")
    return gate, memory


def test_accumulates_real_copies_with_less_ram_than_one_tensor(monkeypatch):
    gate, memory = small_gate(monkeypatch)
    source = torch.arange(32, dtype=torch.float32)
    expected, original_pointer = source.clone(), source.data_ptr()
    copied, waits = [], []
    original_copy = torch.Tensor.copy_

    def copy_chunk(destination, data, **kwargs):
        size = destination.numel()
        assert size == gate.reserved_bytes
        assert size <= memory["available"] - 256  # Headroom stays untouched.
        result = original_copy(destination, data, **kwargs)
        memory["available"] -= size  # Simulate real CPU pages becoming resident.
        copied.append(size)
        return result

    def release_more_ram(timeout):
        waits.append(timeout)
        assert gate.reserved_bytes == 0
        assert source.data_ptr() == original_pointer  # Publish only a complete storage.
        memory["available"] += 32

    monkeypatch.setattr(torch.Tensor, "copy_", copy_chunk)
    monkeypatch.setattr(gate._condition, "wait", release_more_ram)
    incremental._copy_tensors_to_cpu([source], gate, stage="test", deadline=None)
    assert copied == [32, 32, 32, 32] and waits == [1, 1, 1]
    assert gate.reserved_bytes == 0 and memory["available"] == 256
    assert source.data_ptr() != original_pointer and torch.equal(source, expected)


def test_storage_aliases_offsets_gradients_and_next_adam_update_survive(monkeypatch):
    gate, _ = small_gate(monkeypatch, available=1024)
    model = torch.nn.Linear(4, 2)
    model.tied = model.weight
    model.register_buffer("view", model.weight.detach().t()[1:])
    model.register_buffer("bf16", torch.arange(9, dtype=torch.bfloat16), persistent=False)
    model.register_buffer("empty", torch.empty(0))
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    control = torch.optim.Adam(reference.parameters(), lr=.01)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    ids = [id(p) for p in model.parameters()]
    for net, adam in ((model, optimizer), (reference, control)):
        net(inputs).square().mean().backward()
        adam.step()
    values = {k: v.clone() for k, v in model.state_dict().items()}
    gradients = [p.grad.clone() for p in model.parameters()]
    optimizer_tensors = [v for state in optimizer.state.values() for v in state.values()
                         if torch.is_tensor(v)]
    incremental._copy_tensors_to_cpu(
        list(incremental._model_tensors(model)) + optimizer_tensors, gate, stage="test", deadline=None)
    assert [id(p) for p in model.parameters()] == ids
    assert model.tied is model.weight
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    assert model.view.storage_offset() == 1 and model.view.stride() == (1, 4)
    assert model.bf16.dtype == torch.bfloat16 and torch.equal(model.bf16, reference.bf16)
    for name, value in model.state_dict().items():
        assert torch.equal(value, values[name])
    for p, gradient in zip(model.parameters(), gradients):
        assert torch.equal(p.grad, gradient)
    for net, adam in ((model, optimizer), (reference, control)):
        adam.zero_grad()
        net(inputs).square().mean().backward()
        adam.step()
    for p, q in zip(model.parameters(), reference.parameters()):
        assert torch.equal(p, q)
        for name, value in optimizer.state[p].items():
            assert torch.equal(value, control.state[q][name])


@pytest.mark.parametrize("failure", [KeyboardInterrupt, host_memory.HostMemoryTimeout])
def test_interruption_keeps_completed_storages_and_intact_uncopied_values(monkeypatch, failure):
    gate, _ = small_gate(monkeypatch, available=1024)
    first, second = torch.arange(8.), torch.arange(32.)
    old_pointers = [t.data_ptr() for t in (first, second)]
    expected = [t.clone() for t in (first, second)]
    original_copy = torch.Tensor.copy_
    calls = []
    def fail_mid_storage(destination, data, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise failure()
        return original_copy(destination, data, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_mid_storage)
    with pytest.raises(failure):
        incremental._copy_tensors_to_cpu([first, second], gate, stage="test", deadline=None)
    assert first.data_ptr() != old_pointers[0]  # Its transfer completed.
    assert second.data_ptr() == old_pointers[1]  # Its partial transfer never published.
    assert all(torch.equal(t, e) for t, e in zip((first, second), expected))
    assert gate.reserved_bytes == gate._priority_waiters.value == 0


def test_chunks_share_one_deadline_instead_of_restarting_two_hours(monkeypatch):
    gate, memory = small_gate(monkeypatch)
    clock, waits = [0.], []
    monkeypatch.setattr(host_memory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_copy = torch.Tensor.copy_
    def copy_chunk(destination, data, **kwargs):
        memory["available"] -= destination.numel()
        return original_copy(destination, data, **kwargs)
    def wait(timeout):
        clock[0] += timeout
        waits.append(timeout)
        if clock[0] == 1:
            memory["available"] += 32
    monkeypatch.setattr(torch.Tensor, "copy_", copy_chunk)
    monkeypatch.setattr(gate._condition, "wait", wait)
    with pytest.raises(host_memory.HostMemoryTimeout):
        incremental._copy_tensors_to_cpu([torch.arange(32.)], gate, stage="test", deadline=3.)
    assert waits == [1, 1, 1] and gate.reserved_bytes == 0


def test_incremental_mode_is_limited_to_supported_plain_b200_tensors():
    model = torch.nn.Linear(2, 2)
    assert incremental.can_incrementally_offload(model, SimpleNamespace(hardware_profile="B200_full_ram"))
    assert incremental.can_incrementally_offload(model, SimpleNamespace(hardware_profile="B200"))
    assert not incremental.can_incrementally_offload(model, SimpleNamespace(hardware_profile="RTX6000"))
    assert not incremental.can_incrementally_offload(model, SimpleNamespace(hardware_profile="B200"), "disk")
    class PackedParameter(torch.nn.Parameter):
        pass
    model.weight = PackedParameter(model.weight)
    assert not incremental.can_incrementally_offload(model, SimpleNamespace(hardware_profile="B200_full_ram"))


def test_process_trainers_share_chunk_gate_deadline_and_acknowledge_after_copy(monkeypatch):
    from training.trainers import ProcessDistributedTrainer
    gate, _ = small_gate(monkeypatch, available=1024)
    events = []
    trainer = object.__new__(ProcessDistributedTrainer)
    trainer._offloaded, trainer._world_size = False, 2
    trainer.optimizer = SimpleNamespace(state={})
    trainer._command_queues = {1: SimpleNamespace(put=lambda cmd: events.append(cmd))}
    trainer.backend = SimpleNamespace(offload_for_generation=lambda **kw: events.append(("rank0", kw)))
    trainer._collect_event = lambda event, ranks: events.append((event, tuple(ranks)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    trainer.offload_for_generation(purge_host_cache=False, host_admission=gate, deadline=123.)
    assert events == [
        {"kind": "offload", "disk_directory": None, "incremental": True, "deadline": 123.},
        ("rank0", {"host_admission": gate, "deadline": 123.}), ("offloaded", (1,))]
    source = ast.parse((Path(__file__).resolve().parents[1] / "training/fast_distributed.py").read_text())
    branch = next(n for n in ast.walk(source) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "kind == 'offload'")
    worker_events = []
    scope = dict(kind="offload", command=events[0], offloaded=False, rank=1, torch=torch, gc=gc,
                 activation_offload_semaphore=SimpleNamespace(host_admission=gate),
                 backend=SimpleNamespace(offload_for_generation=lambda **kw: worker_events.append(kw)),
                 result_queue=SimpleNamespace(put=lambda event: worker_events.append(event)))
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), "real-offload-command", "exec"), scope)
    assert worker_events == [{"host_admission": gate, "deadline": 123.}, {"event": "offloaded", "rank": 1}]
