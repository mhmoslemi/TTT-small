"""CPU-only regressions for RAM admission and the GPU phase handoff."""

import ast
from contextlib import nullcontext
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

import runtime.host_memory as host_memory
from runtime.host_memory import GIB, HostMemoryAdmission, TokenWeightedActivationOffloadGate
from generation.gen_workers import PhasedVLLMGenerationPool
from training import exact_overflow


def test_disk_parking_preserves_weights_ties_buffers_and_adam_updates(tmp_path, monkeypatch):
    import copy
    import torch
    from runtime import disk_offload

    monkeypatch.setattr(disk_offload, "CHUNK_BYTES", 16)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(5, 3, bias=False).requires_grad_(False)
            self.adapter = torch.nn.Linear(5, 3, bias=False)
            self.tied = self.base.weight
            self.register_buffer("offset", torch.arange(3.))
            self.register_buffer("counter", torch.tensor(7), persistent=False)
            self.register_buffer("bf16", torch.arange(11, dtype=torch.bfloat16))

        def forward(self, inputs):
            return self.base(inputs) + self.adapter(inputs) + self.offset

    model = Model()
    baseline = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=.001)
    reference = torch.optim.Adam(baseline.adapter.parameters(), lr=.001)
    inputs = torch.arange(10.).reshape(2, 5)
    parameter_ids = [id(p) for p in model.parameters()]
    for _ in range(2):
        for network, adam in ((model, optimizer), (baseline, reference)):
            adam.zero_grad()
            network(inputs).square().mean().backward()
            adam.step()
        expected = copy.deepcopy(model.state_dict())
        parked = disk_offload.DiskWeightOffload(model, tmp_path)
        assert model.base.weight.numel() == model.offset.numel() == model.counter.numel() == 0
        assert model.adapter.weight.numel() == 15
        assert model.tied is model.base.weight
        assert [id(p) for p in model.parameters()] == parameter_ids
        assert next(iter(optimizer.state)) is model.adapter.weight
        parked.restore()
        parked.restore()  # idempotent after the first successful restore
        assert parked.file.closed and not parked.entries
        assert model.counter.item() == 7
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, expected[name]), name
            assert torch.equal(tensor, baseline.state_dict()[name]), name
        assert torch.equal(model(inputs), baseline(inputs))


def test_disk_parking_write_failure_restores_already_parked_weights(tmp_path, monkeypatch):
    import torch
    from runtime import disk_offload

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)).requires_grad_(False)
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    original = disk_offload.DiskWeightOffload._release_file_cache
    calls = []

    def fail_once(parked):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("disk write failed")
        original(parked)

    monkeypatch.setattr(disk_offload.DiskWeightOffload, "_release_file_cache", fail_once)
    with pytest.raises(OSError, match="disk write failed"):
        disk_offload.DiskWeightOffload(model, tmp_path)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name


def test_disk_parking_checks_capacity_before_changing_weights(tmp_path, monkeypatch):
    import torch
    from runtime import disk_offload

    model = torch.nn.Linear(2, 2).requires_grad_(False)
    expected = model.weight.clone()
    monkeypatch.setattr(disk_offload.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="insufficient disk space"):
        disk_offload.DiskWeightOffload(model, tmp_path)
    assert torch.equal(model.weight, expected)


def test_backend_disk_handoff_restores_before_training_and_keeps_parameter_objects(tmp_path, monkeypatch):
    import torch
    from models.model_backend import HFBackend

    backend = HFBackend(SimpleNamespace())
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    model[0].requires_grad_(False)
    backend.model = model
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    identities = [id(p) for p in model.parameters()]
    moves = []
    original_to = model.to

    def move(target):
        moves.append(target)
        if target != "cpu":
            for name, tensor in model.state_dict().items():
                assert torch.equal(tensor, expected[name]), name
        return original_to("cpu")  # Exercise placement order on the CPU test host.

    monkeypatch.setattr(model, "to", move)
    backend.offload_for_generation(disk_directory=tmp_path)
    assert model[0].weight.numel() == 0 and model[1].weight.numel() == 6
    backend.offload_for_generation(disk_directory=tmp_path)
    backend.restore_after_generation()
    backend.restore_after_generation()
    assert moves == ["cpu", "cuda:0"]
    assert backend._disk_weights is None and not backend._trainer_is_offloaded
    assert [id(p) for p in model.parameters()] == identities


def test_process_trainer_forwards_disk_handoff_to_all_ranks(tmp_path, monkeypatch):
    import torch
    from training.trainers import ProcessDistributedTrainer

    events = []
    trainer = object.__new__(ProcessDistributedTrainer)
    trainer._offloaded, trainer._world_size = False, 2
    trainer.optimizer = SimpleNamespace(state={})
    trainer._command_queues = {1: SimpleNamespace(put=lambda message: events.append(message))}
    trainer.backend = SimpleNamespace(offload_for_generation=lambda **options: events.append(options))
    trainer._collect_event = lambda name, ranks: events.append((name, tuple(ranks)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    trainer.offload_for_generation(purge_host_cache=False, disk_directory=str(tmp_path))
    assert events == [{"kind": "offload", "disk_directory": str(tmp_path)},
                      {"disk_directory": str(tmp_path)}, ("offloaded", (1,))]
    assert trainer._offloaded_ranks == {0, 1}


def test_disk_offload_estimate_keeps_adapter_gradients_and_optimizer():
    import torch
    from training.trainers import ProcessDistributedTrainer

    model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Linear(128, 2))
    model[0].requires_grad_(False)
    model(torch.ones(1, 128)).sum().backward()
    optimizer = torch.optim.Adam(model[1].parameters())
    optimizer.step()
    trainer = object.__new__(ProcessDistributedTrainer)
    trainer.model, trainer.optimizer, trainer._world_size = model, optimizer, 2
    adapter_bytes = sum(t.numel() * t.element_size() for p in model[1].parameters()
                        for t in (p, p.grad))
    assert host_memory.estimated_module_bytes(model, frozen_to_disk=True) == adapter_bytes
    expected = int(1.05 * (2 * adapter_bytes + host_memory.estimated_optimizer_bytes(optimizer)) + 2 * GIB)
    assert trainer.estimated_host_offload_bytes(frozen_to_disk=True) == expected
    assert trainer.estimated_host_offload_bytes() > expected


def test_run_specific_ram_reserve_admits_small_strategy_without_changing_defaults(monkeypatch):
    memory = dict(total=1696 * GIB, available=int(112.9 * GIB))
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: memory)
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    assert host_memory.host_memory_reserve_bytes(memory) == 128 * GIB
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"))
    with pytest.raises(host_memory.HostMemoryUnavailable):
        with gate.reservation(26 * GIB, wait_for_external=False):
            pass
    monkeypatch.setenv("TTT_HOST_MEMORY_RESERVE_GIB", "16")
    with gate.reservation(26 * GIB, wait_for_external=False):
        assert gate.reserved_bytes == 26 * GIB
    for invalid in ("0", "-1", "nan", "inf"):
        monkeypatch.setenv("TTT_HOST_MEMORY_RESERVE_GIB", invalid)
        with pytest.raises(ValueError, match="finite and positive"):
            host_memory.host_memory_reserve_bytes(memory)


def test_unsloth_offload_releases_storage_and_buffers_can_regrow(monkeypatch):
    import torch
    from models.model_backend import UnslothBackend, _ModelPlacementBackend

    buffers = [torch.ones(4096) for _ in range(3)]
    buffers[1].resize_(0)
    assert buffers[1].untyped_storage().nbytes() == 4096 * 4
    checkpointing = SimpleNamespace(
        CPU_BUFFERS=[buffers[0]], GPU_BUFFERS=(buffers[1],),
        GPU_BUFFERS_B=(buffers[2],), USE_DOUBLE_BUFFER=True,
        NEXT_BUFFER_SLOT=[1])
    monkeypatch.setitem(sys.modules, "unsloth_zoo.gradient_checkpointing", checkpointing)
    monkeypatch.setattr(host_memory, "reclaim_unused_host_memory", lambda *a, **kw: None)
    offloads = []

    def move_model(backend):
        assert all(b.untyped_storage().nbytes() == 0 for b in buffers)
        if not getattr(backend, "_trainer_is_offloaded", False):
            offloads.append(True)
        backend._trainer_is_offloaded = True

    monkeypatch.setattr(_ModelPlacementBackend, "offload_for_generation", move_model)
    backend = UnslothBackend(SimpleNamespace(coder_model_profile="gpt-oss-120b"))
    backend.offload_for_generation()
    backend.offload_for_generation()
    assert offloads == [True]
    assert checkpointing.GPU_BUFFERS[0] is buffers[1]
    assert checkpointing.USE_DOUBLE_BUFFER and checkpointing.NEXT_BUFFER_SLOT == [1]
    # This is how Unsloth reuses its scratch tensors on the next forward.
    for buffer in buffers:
        buffer.resize_(128).fill_(7)
        assert buffer.sum().item() == 896


def test_wake_oom_recovery_uses_live_budget_and_keeps_eight_workers(monkeypatch, capsys):
    import generation.gen_workers as gen_workers
    import runtime.gpu_runtime as gpu_runtime

    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", lambda: {
        gpu: SimpleNamespace(total_gib=95.6, free_gib=77.4) for gpu in range(8)})
    monkeypatch.setattr(gen_workers, "_resolve_vllm_enforce_eager", lambda eager: (eager, False))
    started = []

    def start(pool, ranks, **kwargs):
        started.extend(ranks)
        pool._sleep_capable_workers.update(ranks)
        # The 16-GiB shared-pool reserve is kept beyond the 18.2 GiB already
        # allocated when recovery begins: 77.4 GiB free - 16 GiB reserve.
        assert pool._worker_options["gpu_memory_utilization"] * 95.6 <= 61.4 + 1e-8

    monkeypatch.setattr(gen_workers.GenerationPool, "_start_workers", start)
    calls = []
    phase = gen_workers.PhasedVLLMGenerationPool(
        lambda: calls.append("offload"), lambda: calls.append("restore"),
        model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", backend="vllm",
        num_workers=8, gpu_ids=list(range(8)), vllm_tensor_parallel_size=1,
        vllm_co_resident_sleep=True, vllm_runtime_reserve_gib=16,
        max_seq_length=131072, vllm_sleep_level=1)

    def failed_wake():
        calls.append("wake")
        raise RuntimeError("CUDA out of memory")

    phase._pool = SimpleNamespace(
        wake_up=failed_wake, shutdown=lambda: calls.append("stop_failed_pool"))
    phase._persistent = True
    pool = phase._ensure_started()
    assert calls == ["offload", "wake", "stop_failed_pool"]
    assert phase.active and phase._persistent
    assert started == list(range(8)) and pool.num_workers == 8
    assert pool.tensor_parallel_size == 1 and pool.max_seq_length == 131072
    assert "level-1 vLLM wake failed" in capsys.readouterr().out
    for q in (pool.result_queue, pool._ready_queue, pool.control_queue):
        q.close()


def test_live_ram_admission_accounts_for_not_yet_allocated_reservations(monkeypatch):
    monkeypatch.setattr(host_memory, "host_memory_info",
                        lambda: {"total": 100 * GIB, "available": 60 * GIB})
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"))
    started, admitted = threading.Event(), threading.Event()

    def other_evaluator():
        started.set()
        with gate.reservation(25 * GIB):
            admitted.set()

    with gate.reservation(30 * GIB):
        thread = threading.Thread(target=other_evaluator, daemon=True)
        thread.start()
        assert started.wait(1)
        assert not admitted.wait(.05)
        assert gate.available_bytes() == 20 * GIB
    assert admitted.wait(2)
    thread.join(timeout=2)
    assert gate.reserved_bytes == 0
    with pytest.raises(ValueError):
        with gate.reservation(GIB):
            raise ValueError("evaluation failed")
    assert gate.reserved_bytes == 0


def test_activation_offload_shares_live_ram_gate_and_never_under_reserves(monkeypatch):
    monkeypatch.setattr(host_memory, "host_memory_info",
                        lambda: {"total": 1000 * GIB, "available": 800 * GIB})
    gate = TokenWeightedActivationOffloadGate(
        multiprocessing.get_context("spawn"), 500 * GIB, 131072)
    with gate.host_admission.reservation(100 * GIB):
        with gate.reservation(32000):
            assert gate.host_admission.reserved_bytes == 484 * GIB
        assert gate.host_admission.reserved_bytes == 100 * GIB
    assert gate.host_admission.reserved_bytes == 0
    too_small = TokenWeightedActivationOffloadGate(
        multiprocessing.get_context("spawn"), 100 * GIB, 131072)
    assert not too_small.can_admit(32000)
    with pytest.raises(RuntimeError, match="budget"):
        too_small.reservation(32000)


def test_waiting_model_transfer_has_priority_over_new_evaluations(monkeypatch):
    memory = dict(total=100 * GIB, available=50 * GIB)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200_full_ram")
    waiting, admitted, finish = threading.Event(), threading.Event(), threading.Event()
    order = []
    original_wait = gate._condition.wait

    def wait(timeout):
        if threading.current_thread().name == "model-transfer":
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(gate._condition, "wait", wait)

    def transfer():
        with gate.reservation(60 * GIB, priority=True, stage="model transfer"):
            order.append("model")
            admitted.set()
            finish.wait(2)

    def evaluate():
        with gate.reservation(5 * GIB):
            order.append("evaluation")

    model_thread = threading.Thread(target=transfer, name="model-transfer", daemon=True)
    evaluator = threading.Thread(target=evaluate, daemon=True)
    try:
        model_thread.start()
        assert waiting.wait(2)
        evaluator.start()
        evaluator.join(.05)
        assert not order  # A small evaluation could fit, but must yield.
        with gate._condition:
            memory["available"] = 100 * GIB
            gate._condition.notify_all()
        assert admitted.wait(2)
        evaluator.join(2)
        assert order == ["model", "evaluation"]
    finally:
        finish.set()
        gate.stop_waiting()
        model_thread.join(2)
        evaluator.join(2)
    assert gate.reserved_bytes == gate._priority_waiters.value == 0


def test_interrupted_ram_wait_releases_priority(monkeypatch):
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=200*GIB, available=0))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200_full_ram")
    def interrupt(timeout):
        raise KeyboardInterrupt
    monkeypatch.setattr(gate._condition, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with gate.reservation(32*GIB, priority=True):
            pytest.fail("allocation must not start")
    assert gate._priority_waiters.value == gate.reserved_bytes == 0


def test_main_full_ram_offload_waits_and_holds_reservation_during_all_copies(monkeypatch):
    memory = dict(total=200 * GIB, available=16 * GIB)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200_full_ram")
    waits, copies = [], []
    def wait(timeout):
        waits.append(timeout)
        memory["available"] = 64 * GIB
    monkeypatch.setattr(gate._condition, "wait", wait)
    def offload(**kwargs):
        assert gate.reserved_bytes == 32 * GIB
        copies.append(kwargs)
    trainer = SimpleNamespace(is_offloaded=False, world_size=2, reclaim_host_memory=lambda: [],
                              estimated_host_offload_bytes=lambda **kw: 32 * GIB,
                              offload_for_generation=offload)
    source = ast.parse((Path(__file__).resolve().parents[1] / "train_multy_CVaR.py").read_text())
    function = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef)
                    and n.name == "_offload_trainer_for_generation")
    wrapper = ast.parse("def build():\n    trainer_offloaded = False\n    return None").body[0]
    wrapper.body[1:] = [function, ast.Return(value=ast.Name(id=function.name, ctx=ast.Load()))]
    scope = dict(cfg=SimpleNamespace(hardware_profile="B200_full_ram"),
                 parallel_trainer=trainer, gen_pool=None, strategy_pool=None,
                 pool_options={"host_memory_admission": gate}, runtime_training_active={"value": True},
                 setting_log_only=lambda *a, **kw: None, HostMemoryUnavailable=host_memory.HostMemoryUnavailable)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 "real-trainer-handoff", "exec"), scope)
    scope["build"]()()
    assert waits == [1] and copies == [{"purge_host_cache": False, "disk_directory": None}]
    assert gate.reserved_bytes == 0


def test_full_ram_model_initialization_holds_claim_until_loader_returns(monkeypatch):
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=200*GIB, available=100*GIB))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200_full_ram")
    monkeypatch.setattr(host_memory, "HostMemoryAdmission", lambda *a, **kw: gate)
    def load():
        assert gate.reserved_bytes == 8 * GIB
        return "model"
    assert host_memory.retry_host_allocation(load, hardware_profile="B200_full_ram",
                                             stage="trainer loading", required_bytes=8*GIB) == "model"
    assert gate.reserved_bytes == 0


def test_full_ram_coder_startup_is_also_admitted(monkeypatch):
    from generation import gen_workers as workers
    memory = dict(total=200*GIB, available=0)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200_full_ram")
    waits = []
    def wait(timeout):
        waits.append(timeout)
        memory["available"] = 100 * GIB
    monkeypatch.setattr(gate._condition, "wait", wait)
    def create(**kwargs):
        assert gate.reserved_bytes == int(8 * GIB * 1.05)
        return "pool"
    monkeypatch.setattr(workers, "GenerationPool", create)
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: None,
        hardware_profile="B200_full_ram", model_name="Qwen/Qwen3.8-27B",
        gpu_ids=[6, 7], num_workers=2, host_memory_admission=gate)
    assert pool._create_pool() == "pool"
    assert waits == [1] and gate.reserved_bytes == 0


def test_ram_timeout_during_sleep_is_not_swallowed_as_engine_failure():
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: None)
    pool._awake = pool._persistent = True
    def sleep():
        raise host_memory.HostMemoryTimeout("test timeout")
    pool._pool = SimpleNamespace(sleep=sleep)
    with pytest.raises(host_memory.HostMemoryTimeout, match="test timeout"):
        pool.release()
    assert not pool.active


def test_sleep_waves_restore_trainers_before_more_host_copies(monkeypatch):
    from generation.gen_workers import GenerationPool

    memory = {"total": 1700 * GIB, "available": 1100 * GIB}
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"))
    pool = object.__new__(GenerationPool)
    pool._persistent_worker_ids = tuple(range(8))
    pool._transient_worker_ids = ()
    pool._gpu_groups = [[rank] for rank in range(8)]
    pool.num_workers = 8
    pool.sleep_level = 1
    pool._stop_workers = lambda ranks: None
    events, peaks = [], []

    def control(command, ranks):
        if command == "host_memory":
            return {rank: {"backup_bytes": 70 * GIB} for rank in ranks}
        assert command == "sleep"
        events.append(("sleep", tuple(ranks)))
        memory["available"] -= len(ranks) * 70 * GIB
        peaks.append(memory["total"] - memory["available"])

    def restore(gpus):
        events.append(("restore", tuple(gpus)))
        # Returning a trainer to its GPU frees its previous CPU copy.
        memory["available"] += len(gpus) * 60 * GIB

    pool._vllm_control = control
    pool.sleep(after_sleep=restore, host_admission=gate)
    assert events == [event for start in range(0, 8, 2)
                      for event in [("sleep", (start, start + 1)),
                                    ("restore", (start, start + 1))]]
    assert max(peaks) <= 800 * GIB  # all-at-once peaked at 1160 GiB
    assert pool.num_workers == 8
    assert pool._persistent_worker_ids == tuple(range(8))
    assert gate.reserved_bytes == 0


def test_sleep_extension_uses_pageable_backups_and_reclaims_only_unused_cache(monkeypatch):
    from vllm_compat.sleep_memory import SleepMemoryExtension

    allocations = {
        1: SimpleNamespace(tag="weights", handle=(0, 70 * GIB)),
        2: SimpleNamespace(tag="kv_cache", handle=(0, 13 * GIB)),
    }
    allocator = SimpleNamespace(pointer_to_data=allocations)
    cumem = SimpleNamespace(PIN_MEMORY=True, CuMemAllocator=SimpleNamespace(
        get_instance=lambda: allocator))
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator", SimpleNamespace(cumem=cumem))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []

    def reclaim(torch, **kwargs):
        calls.append(kwargs)
        return {"rss_after": 5 * GIB}

    monkeypatch.setattr(host_memory, "reclaim_unused_host_memory", reclaim)
    result = SleepMemoryExtension().ttt_host_memory()
    assert result == {"backup_bytes": 70 * GIB, "rss_bytes": 5 * GIB}
    assert not cumem.PIN_MEMORY
    assert calls == [{"cuda_devices": (3,), "force": True}]
    assert len(allocations) == 2


def test_unaffordable_backup_releases_engine_instead_of_hanging(monkeypatch):
    from generation.gen_workers import GenerationPool

    memory = {"total": 200 * GIB, "available": 30 * GIB}
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(memory))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"))
    pool = object.__new__(GenerationPool)
    pool._persistent_worker_ids = (0, 1)
    pool._transient_worker_ids = ()
    pool._gpu_groups = [[0], [1]]
    pool.num_workers = pool.persistent_workers = 2
    pool.sleep_level = 1
    stopped, restored = [], []
    pool._stop_workers = lambda ranks: stopped.extend(ranks)

    def control(command, ranks):
        assert command == "host_memory", "cannot allocate a 70-GiB backup here"
        return {rank: {"backup_bytes": 70 * GIB} for rank in ranks}

    pool._vllm_control = control
    pool.sleep(after_sleep=lambda ids: restored.extend(ids), host_admission=gate)
    assert stopped == restored == [0, 1]
    assert pool._transient_worker_ids == (0, 1)
    assert pool.persistent_workers == 0 and pool.num_workers == 2


def test_partial_trainer_restore_respects_physical_gpu_mapping(monkeypatch):
    import torch
    import training.fast_distributed as fast_distributed
    import train_multy_CVaR as training

    calls = []
    trainer = object.__new__(training.ProcessDistributedTrainer)
    trainer._physical_ids = [1, 3, 5, 7]
    trainer._world_size = 4
    trainer._offloaded = True
    trainer._offloaded_ranks = {0, 1, 2, 3}
    trainer._dist_initialized = True
    trainer._process_label = "test"
    trainer._memory_fraction = .88
    trainer.model, trainer.optimizer = object(), object()
    trainer.backend = SimpleNamespace(
        restore_after_generation=lambda: calls.append(("local_restore", 0)),
        set_training_mode=lambda: None)
    trainer._command_queues = {
        rank: SimpleNamespace(put=lambda command, rank=rank:
                              calls.append((command["kind"], rank)))
        for rank in (1, 2, 3)}
    trainer._collect_event = lambda event, ranks: calls.append((event, set(ranks)))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(fast_distributed, "set_total_memory_ceiling", lambda *a: None)
    monkeypatch.setattr(fast_distributed, "validate_model_device", lambda *a: None)
    monkeypatch.setattr("training.trainers._restore_optimizer_state_to_parameters", lambda *a: None)
    monkeypatch.setattr(host_memory, "reclaim_unused_host_memory", lambda *a, **kw: None)
    trainer.restore_on_gpus([3, 7])
    assert calls == [("restore", 1), ("restore", 3), ("restored", {1, 3})]
    assert trainer._offloaded_ranks == {0, 2} and trainer.is_offloaded
    calls.clear()
    trainer.restore_on_gpus([1, 5])
    assert calls == [("restore", 2), ("local_restore", 0), ("restored", {2})]
    assert not trainer.is_offloaded
    trainer.restore_after_generation()  # idempotent after the last wave
    assert len(calls) == 3


def test_vllm_shutdown_reaps_children_outside_wrapper_process_group():
    import os
    import subprocess
    import psutil
    from generation import gen_workers as workers

    if os.name != "posix":
        pytest.skip("private process groups require POSIX")
    script = ("import subprocess, sys\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
              "print(child.pid, flush=True)\n"
              "sys.stdin.readline()\n")
    wrapper = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                 start_new_session=True)
    child = None
    try:
        child = psutil.Process(int(wrapper.stdout.readline()))
        assert os.getpgid(child.pid) != os.getpgid(wrapper.pid)

        def stop(_):
            wrapper.stdin.write("stop\n")
            wrapper.stdin.flush()

        pool = object.__new__(workers.GenerationPool)
        pool.backend = "vllm"
        pool.procs = [SimpleNamespace(pid=wrapper.pid, is_alive=lambda: wrapper.poll() is None,
            join=wrapper.wait, terminate=wrapper.terminate, kill=wrapper.kill)]
        pool.task_queues = [SimpleNamespace(put=stop, close=lambda: None, join_thread=lambda: None)]
        pool._ready_worker_ids, pool._sleep_capable_workers, pool._async_worker_ids = {0}, {0}, {0}
        pool._stop_workers([0])
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
        assert pool.procs == [None] and not pool._ready_worker_ids
    finally:
        if child is not None:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        for process in (wrapper, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        wrapper.stdin.close()
        wrapper.stdout.close()


def test_strategy_startup_charges_resident_tables_not_the_whole_checkpoint(monkeypatch):
    from generation import gen_workers as workers
    from models.strategist_models import strategist_engine_options

    monkeypatch.setenv("TTT_HOST_MEMORY_RESERVE_GIB", "16")
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=1200*GIB, available=1000*GIB))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"))
    reservations = []
    monkeypatch.setattr(workers, "GenerationPool",
                        lambda **kwargs: reservations.append(gate.reserved_bytes))
    pool = PhasedVLLMGenerationPool(
        lambda: None, lambda: None, model_name="nvidia/Qwen3.8-Flash-Next-NVFP4",
        num_workers=8, gpu_ids=list(range(8)), host_memory_admission=gate,
        vllm_engine_options=strategist_engine_options("qwen3.8-flash-next", "B200"))
    pool._create_pool()
    assert reservations == [int(4 * 8 * GIB * 1.05)]
    assert gate.reserved_bytes == 0
    pool._pool_kwargs["vllm_engine_options"] = strategist_engine_options("qwen3.8-flash-next")
    pool._create_pool()
    assert reservations[1] == int((48 + 4) * 8 * GIB * 1.05)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=1200*GIB, available=50*GIB))
    with pytest.raises(host_memory.HostMemoryUnavailable):
        pool._create_pool()


def test_shutdown_does_not_allocate_sleep_backups():
    from generation.gen_workers import PhasedVLLMGenerationPool

    calls = []
    pool = object.__new__(PhasedVLLMGenerationPool)
    pool._awake = True
    pool._pool = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    pool.release = lambda: pytest.fail("shutdown must not back up weights")
    pool.shutdown()
    assert calls == ["shutdown"] and not pool.active


def test_b200_cpu_parking_preserves_optimizer_and_mutable_buffers(monkeypatch):
    import copy
    import torch
    from models.model_backend import _ModelPlacementBackend
    from runtime import disk_offload

    monkeypatch.setattr(disk_offload, "DiskWeightOffload",
                        lambda *a, **kw: pytest.fail("B200 must not create disk snapshots"))
    model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 2))
    model[0].requires_grad_(False)
    model.register_buffer("counter", torch.tensor(0))
    model.tied = model[0].weight
    baseline = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model[1].parameters(), lr=.01)
    reference = torch.optim.Adam(baseline[1].parameters(), lr=.01)
    backend = _ModelPlacementBackend()
    backend.model = model
    backend.cfg = SimpleNamespace(hardware_profile="B200")
    backend._training_device_map = {"": 7}
    backend._trainer_is_offloaded = False
    identities = [id(p) for p in model.parameters()]
    moves = []
    original_to = model.to
    def move(target):
        moves.append(target)
        return original_to("cpu")  # Exercise GPU placement order without CUDA.
    monkeypatch.setattr(model, "to", move)
    for step in range(3):
        x = torch.ones(2, 5) * (step + 1)
        for network, adam in ((model, optimizer), (baseline, reference)):
            adam.zero_grad()
            network(x).square().mean().backward()
            adam.step()
            network.counter.add_(1)
        backend.offload_for_generation()
        assert model[0].weight.numel() == 15 and model[1].weight.numel() == 6
        backend.restore_after_generation()
        backend.restore_after_generation()
        assert not getattr(backend, "_disk_weights", None)
        assert model.tied is model[0].weight
        assert [id(p) for p in model.parameters()] == identities
        assert next(iter(optimizer.state)) is model[1].weight
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, baseline.state_dict()[name]), name
        for p, q in zip(model[1].parameters(), baseline[1].parameters()):
            for name, value in optimizer.state[p].items():
                assert torch.equal(value, reference.state[q][name])
    assert moves == ["cpu", "cuda:7"] * 3


def test_b200_transient_lifecycle_ignores_sleep_flags_and_reloads_adapter(monkeypatch):
    events = []

    def create(pool, **options):
        assert not options.get("vllm_enable_sleep_mode")
        assert pool._pool_kwargs["vllm_persistent_workers"] == 0
        assert pool._pool_kwargs["vllm_engine_options"]["safetensors_load_strategy"] == "lazy"
        events.append("load")
        return SimpleNamespace(num_workers=2, sleep_supported=False,
            sleep=lambda: pytest.fail("B200 must never create a sleep backup"),
            shutdown=lambda: events.append("shutdown"),
            generate_groups=lambda **kw: events.append(kw["adapter_path"]))

    monkeypatch.setattr(PhasedVLLMGenerationPool, "_create_pool", create)
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: events.append("released"),
        hardware_profile="B200", model_name="Qwen/Qwen3.8-27B", gpu_ids=[6, 7],
        vllm_sleep_level=1, vllm_persistent_workers=2)
    for step in range(2):
        pool.generate_groups(adapter_path=f"adapter{step}")
        pool.generate_groups(adapter_path=f"adapter{step}")
        pool.release()
        assert pool._pool is None and not pool.active
    assert events == ["load", "adapter0", "adapter0", "shutdown", "released",
                      "load", "adapter1", "adapter1", "shutdown", "released"]


@pytest.mark.parametrize("profile", ["RTX6000", "B200_full_ram"])
def test_full_ram_pools_keep_sleep_backups_and_load_only_once(monkeypatch, profile):
    events = []
    def create(pool, **options):
        assert options["vllm_enable_sleep_mode"] is True
        events.append("load")
        return SimpleNamespace(num_workers=4, persistent_workers=4, sleep_supported=True,
            sleep=lambda: events.append("sleep"), wake_up=lambda: events.append("wake"),
            shutdown=lambda: events.append("shutdown"),
            generate_groups=lambda **kw: events.append(kw["adapter_path"]))
    monkeypatch.setattr(PhasedVLLMGenerationPool, "_create_pool", create)
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: None, hardware_profile=profile,
        model_name="Qwen/Qwen3.8-27B", gpu_ids=[2, 3, 4, 5], num_workers=4, vllm_sleep_level=1)
    for step in (3, 4):
        pool.generate_groups(adapter_path=f"adapter{step}")
        pool.release()
    assert events == ["load", "adapter3", "sleep", "wake", "adapter4", "sleep"]
    pool.shutdown()
    assert events[-1] == "shutdown"


def test_b200_ram_gate_reserves_job_headroom_without_global_env_changes(monkeypatch):
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=1696*GIB, available=100*GIB))
    context = multiprocessing.get_context("spawn")
    assert HostMemoryAdmission(context).available_bytes() == 0
    gate = HostMemoryAdmission(context, hardware_profile="B200")
    assert gate.available_bytes() == 84*GIB
    full_ram = HostMemoryAdmission(context, hardware_profile="B200_full_ram")
    assert full_ram.available_bytes() == 95*GIB
    with gate.reservation(8*GIB, wait_for_external=False):
        assert gate.available_bytes() == 76*GIB
    activations = TokenWeightedActivationOffloadGate(
        context, 80*GIB, 32768, full_context_bytes=64*GIB)
    assert not activations.can_admit(32768)
    activations.host_admission.hardware_profile = "B200"
    assert activations.can_admit(32768)


@pytest.mark.parametrize("profile", ["B200", "B200_full_ram"])
def test_ram_wait_polls_each_second_then_reserves_and_continues(monkeypatch, profile):
    clock, waits = [0.0], []
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(host_memory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(host_memory, "host_memory_info", lambda:
                        None if not clock[0] else dict(total=200*GIB, available=(80 if clock[0] >= 2 else 16)*GIB))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile=profile)
    def retry(timeout):
        waits.append(timeout)
        clock[0] += timeout
    monkeypatch.setattr(gate._condition, "wait", retry)
    with gate.reservation(32*GIB, wait_for_external=False, stage="test startup"):
        assert gate.reserved_bytes == 32*GIB
    assert waits == [1, 1] and gate.reserved_bytes == 0


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("profile", ["B200", "B200_full_ram"])
def test_ram_timeout_stops_queued_work_without_another_two_hour_wait(monkeypatch, unknown, profile):
    clock, waits = [0.0], []
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(host_memory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(host_memory, "host_memory_info", lambda:
                        None if unknown else dict(total=200*GIB, available=16*GIB))
    gate = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile=profile)
    def retry(timeout):
        waits.append(timeout)
        clock[0] += timeout
    monkeypatch.setattr(gate._condition, "wait", retry)
    with pytest.raises(host_memory.HostMemoryUnavailable, match="2 hours"):
        with gate.reservation(32*GIB):
            pytest.fail("allocation must not start")
    assert clock[0] == 7200 and waits == [1] * 7200
    with pytest.raises(host_memory.HostMemoryUnavailable, match="2 hours"):
        with gate.reservation(0):
            pytest.fail("queued tasks must stop immediately after a peer's timeout")
    assert len(waits) == 7200 and gate.reserved_bytes == 0


def test_ram_timeout_is_not_hidden_by_worker_errors_and_cli_exits_resumably(capsys):
    from queue import Queue
    from training.trainers import ProcessDistributedTrainer
    trainer = object.__new__(ProcessDistributedTrainer)
    trainer._result_queue = Queue()
    trainer._result_queue.put(dict(event="error", rank=1, host_memory_timeout=True))
    with pytest.raises(host_memory.HostMemoryUnavailable, match="2 hours"):
        trainer._collect_event("offloaded", [1])
    def fail():
        raise host_memory.HostMemoryTimeout("test RAM timeout")
    root = Path(__file__).resolve().parents[1]
    guard = ast.parse((root / "train_multy_CVaR.py").read_text()).body[-1]
    with pytest.raises(SystemExit) as stopped:
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "ram-stop", "exec"),
             dict(__name__="__main__", main=fail, HostMemoryTimeout=host_memory.HostMemoryTimeout))
    assert stopped.value.code == 75
    assert "--resume" in capsys.readouterr().out


@pytest.mark.parametrize("profile", ["B200", "B200_full_ram"])
def test_b200_retries_cpu_allocation_errors_but_not_gpu_or_programming_errors(monkeypatch, profile):
    import errno
    clock, waits = [0.0], []
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(host_memory, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    failures = iter([MemoryError(), OSError(errno.ENOMEM, "Cannot allocate memory"),
                     RuntimeError("DefaultCPUAllocator: not enough memory"), None])
    expected = object()
    def allocate():
        error = next(failures)
        if error is not None:
            raise error
        return expected
    assert host_memory.retry_host_allocation(allocate, hardware_profile=profile, stage="test") is expected
    assert waits == [60, 60, 60]
    for error, profile in [(RuntimeError("CUDA out of memory"), "B200"),
                           (RuntimeError("bad model config"), "B200"),
                           (host_memory.HostMemoryUnavailable("already timed out"), "B200"),
                           (MemoryError(), "RTX6000")]:
        def fail():
            raise error
        with pytest.raises(type(error)):
            host_memory.retry_host_allocation(fail, hardware_profile=profile, stage="test")
    assert waits == [60, 60, 60]


def test_b200_allocation_failure_stops_after_two_hours(monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr(host_memory, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    monkeypatch.setattr(host_memory.gc, "collect", lambda: None)
    def allocate():
        calls.append(True)
        raise MemoryError()
    with pytest.raises(host_memory.HostMemoryUnavailable, match="2 hours"):
        host_memory.retry_host_allocation(allocate, hardware_profile="B200", stage="test")
    assert clock[0] == 7200 and len(calls) == 120


def test_auto_backend_preserves_ram_timeout_for_resumable_stop(monkeypatch):
    import models.model_backend as backend
    def fail():
        raise host_memory.HostMemoryUnavailable("RAM timeout")
    monkeypatch.setattr(backend, "UnslothBackend", lambda cfg: SimpleNamespace(load=fail))
    with pytest.raises(host_memory.HostMemoryUnavailable):
        backend._AutoFallbackBackend(SimpleNamespace()).load()


def test_generation_queue_wait_stops_when_a_shared_ram_wait_times_out():
    from queue import Queue
    from generation.gen_workers import GenerationPool
    admission = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile="B200")
    pool = object.__new__(GenerationPool)
    pool.host_memory_admission = admission
    messages = Queue()
    messages.put("ready")
    assert pool._get_worker_message(messages, timeout=.01) == "ready"
    admission.stop_waiting()
    with pytest.raises(host_memory.HostMemoryTimeout):
        pool._get_worker_message(messages, timeout=.01)


def test_cpu_parking_preserves_gradients_after_cooperative_sharding(monkeypatch):
    import torch
    import training.fast_distributed as distributed

    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    model[0].requires_grad_(False)
    model.register_buffer("counter", torch.tensor(0))
    identities = [id(parameter) for parameter in model.parameters()]
    expected_base = model[0].weight.clone()
    backend = SimpleNamespace(model=model, _trainer_is_offloaded=True)
    plans = []

    def balanced(placement, **kwargs):
        assert placement[0].weight.shape == (3, 3), "placement needs restored shapes"
        plans.append(kwargs["max_memory"])
        return kwargs["max_memory"]

    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace(
        dispatch_model=lambda *a, **kw: None,
        infer_auto_device_map=lambda *a, **kw: {"": 0}))
    monkeypatch.setitem(sys.modules, "accelerate.hooks", SimpleNamespace(
        remove_hook_from_submodules=lambda *a: None))
    monkeypatch.setitem(sys.modules, "accelerate.utils", SimpleNamespace(get_balanced_memory=balanced))
    monkeypatch.setattr(distributed, "set_long_rollout_memory_ceiling", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100*GIB, 180*GIB))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(host_memory, "reclaim_unused_host_memory", lambda *a, **kw: None)
    tensor_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *a, **kw:
                        tensor if a and isinstance(a[0], int) else tensor_to(tensor, *a, **kw))
    with exact_overflow.temporary_gpu_shards(backend, [0, 1]):
        assert torch.equal(model[0].weight, expected_base)
        model(torch.ones(2, 3)).sum().backward()
        expected_grad = model[1].weight.grad.clone()
        model.counter.add_(1)
    assert plans and backend._trainer_is_offloaded
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert torch.equal(model[0].weight, expected_base)
    assert torch.equal(model[1].weight.grad, expected_grad)
    assert model.counter.item() == 1
    assert [id(parameter) for parameter in model.parameters()] == identities


def test_b200_overflow_parking_charges_full_weights_and_waits(monkeypatch):
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Linear(128, 1))
    model[0].requires_grad_(False)
    monkeypatch.delenv("TTT_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(total=1696*GIB, available=100*GIB))
    monkeypatch.setattr(host_memory, "estimated_module_bytes", lambda model: 51*GIB)
    waits = []
    monkeypatch.setattr(host_memory, "wait_for_host_memory", lambda size, **kw: waits.append((size, kw)))
    exact_overflow.require_parking_capacity(model, 2, hardware_profile="B200")
    assert waits == [(int(1.1 * 102*GIB), dict(hardware_profile="B200", stage="cooperative trainer CPU offload"))]
    with pytest.raises(RuntimeError, match="insufficient host RAM"):
        exact_overflow.require_parking_capacity(model, 2)


def test_transient_handoff_restores_every_trainer_without_sleep():
    from generation.gen_workers import PhasedVLLMGenerationPool

    calls = []
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: None,
                                   gpu_ids=list(range(8)))
    pool._awake = True
    pool._persistent = False
    pool._pool = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    trainer = SimpleNamespace(host_memory_admission=None,
                              restore_on_gpus=lambda ids: calls.append(ids))
    pool.release_for_training(trainer)
    assert calls == ["shutdown", list(range(8))]
    assert not pool.active


def test_generation_reload_controls_keep_existing_model_defaults():
    from models.coder_profiles import _coder_vllm_persistent_workers
    from runtime.configuration import _build_arg_parser

    parser = _build_arg_parser()
    assert parser.parse_args([]).reload_coder_between_phases is None
    assert parser.parse_args([]).strategy_vllm_persistent_workers is None
    args = parser.parse_args(["--resume", "runs/ac1", "--reload-coder-between-phases",
                             "--strategy-vllm-persistent-workers", "0"])
    args.coder_model_profile = "qwen3.8-27b"
    assert _coder_vllm_persistent_workers(args) == 0
    assert args.strategy_vllm_persistent_workers == 0
    assert _coder_vllm_persistent_workers(SimpleNamespace(coder_model_profile="qwen3.8-27b")) is None
    assert _coder_vllm_persistent_workers(SimpleNamespace(coder_model_profile="gpt-oss-120b")) == 0


@pytest.mark.parametrize("strategy_persistent_workers", [None, 0])
def test_coder_reloads_current_adapter_with_optional_strategy_cache(monkeypatch, strategy_persistent_workers):
    from models.coder_profiles import _coder_vllm_persistent_workers
    from runtime.configuration import _build_arg_parser

    arguments = ["--reload-coder-between-phases"]
    if strategy_persistent_workers == 0:
        arguments += ["--strategy-vllm-persistent-workers", "0"]
    args = _build_arg_parser().parse_args(arguments)
    events = []

    def create(pool, **options):
        name = pool._pool_kwargs["model_name"]
        persistent = pool._pool_kwargs["vllm_persistent_workers"] != 0
        assert bool(options.get("vllm_enable_sleep_mode")) == persistent
        events.append((name, "load"))
        return SimpleNamespace(
            num_workers=2, persistent_workers=2 if persistent else 0,
            sleep_supported=persistent,
            shutdown=lambda: events.append((name, "shutdown")),
            sleep=lambda: events.append((name, "sleep")),
            wake_up=lambda: events.append((name, "wake")),
            generate_groups=lambda **kwargs: events.append((name, kwargs["adapter_path"])))

    monkeypatch.setattr(PhasedVLLMGenerationPool, "_create_pool", create)
    coder = PhasedVLLMGenerationPool(
        lambda: None, lambda: None, model_name="Qwen/Qwen3.8-27B", gpu_ids=[6, 7],
        num_workers=2, vllm_persistent_workers=_coder_vllm_persistent_workers(args))
    strategy = PhasedVLLMGenerationPool(
        lambda: None, lambda: None, model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        gpu_ids=[6, 7], num_workers=2,
        vllm_persistent_workers=args.strategy_vllm_persistent_workers, vllm_sleep_level=1)
    trainer = SimpleNamespace(host_memory_admission=None,
                              restore_on_gpus=lambda ids: events.append(("restore", ids)))
    for step in (1, 2):
        strategy.generate_groups(adapter_path=None)
        strategy.release()
        coder.generate_groups(adapter_path=f"adapter_step{step:03d}")
        coder.release_for_training(trainer)

    coder_name, strategy_name = coder._pool_kwargs["model_name"], strategy._pool_kwargs["model_name"]
    assert events.count((coder_name, "load")) == events.count((coder_name, "shutdown")) == 2
    assert (coder_name, "sleep") not in events
    assert (coder_name, "adapter_step001") in events and (coder_name, "adapter_step002") in events
    if strategy_persistent_workers == 0:
        assert strategy._pool is None and coder._pool is None
        assert events == [event for step in (1, 2) for event in (
            (strategy_name, "load"), (strategy_name, None), (strategy_name, "shutdown"),
            (coder_name, "load"), (coder_name, f"adapter_step{step:03d}"),
            (coder_name, "shutdown"), ("restore", [6, 7]))]
    else:
        assert events.count((strategy_name, "load")) == events.count((strategy_name, "wake")) == 1
        assert events.count((strategy_name, "sleep")) == 2
        assert (strategy_name, "shutdown") not in events
    assert events.count(("restore", [6, 7])) == 2


@pytest.mark.parametrize("available, released", [(700., []), (608.5, ["strategy"]),
                                                (300., ["strategy", "coder"])])
def test_parking_reclaims_only_the_sleeping_pools_it_needs(monkeypatch, available, released):
    state, calls = {"available": int(available * host_memory.GIB)}, []
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(
        total=1717 * host_memory.GIB, available=state["available"]))
    monkeypatch.setattr(host_memory, "estimated_module_bytes", lambda model: 51 * host_memory.GIB)
    monkeypatch.setattr(host_memory, "estimated_optimizer_bytes", lambda optimizer: 0)

    def pool(label, freed):
        def discard():
            calls.append(label)
            state["available"] += freed * host_memory.GIB
            return True
        return SimpleNamespace(discard_sleeping_host_residency=discard)

    exact_overflow.require_parking_capacity(
        object(), 8, generation_pools=(pool("strategy", 100), pool("coder", 300)))
    assert calls == released
    assert state["available"] >= int(1.1 * 51 * 8 * host_memory.GIB) + int(.1 * 1717 * host_memory.GIB)


def test_parking_does_not_stop_an_awake_pool_or_relax_the_safety_reserve(monkeypatch):
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: dict(
        total=1717 * host_memory.GIB, available=int(608.5 * host_memory.GIB)))
    monkeypatch.setattr(host_memory, "estimated_module_bytes", lambda model: 51 * host_memory.GIB)
    monkeypatch.setattr(host_memory, "estimated_optimizer_bytes", lambda optimizer: 0)
    pool = PhasedVLLMGenerationPool(lambda: None, lambda: None, num_workers=8)
    pool._awake = True
    pool._pool = SimpleNamespace(shutdown=lambda: pytest.fail("awake pool was stopped"))
    with pytest.raises(RuntimeError, match="insufficient host RAM"):
        exact_overflow.require_parking_capacity(object(), 8, generation_pools=(pool,))
    assert pool.active


def test_parking_refuses_unknown_memory_after_reclamation(monkeypatch):
    readings = iter([dict(total=1717 * host_memory.GIB, available=300 * host_memory.GIB), None])
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: next(readings))
    monkeypatch.setattr(host_memory, "estimated_module_bytes", lambda model: 51 * host_memory.GIB)
    monkeypatch.setattr(host_memory, "estimated_optimizer_bytes", lambda optimizer: 0)
    pool = SimpleNamespace(discard_sleeping_host_residency=lambda: True)
    with pytest.raises(RuntimeError, match="host RAM is unknown"):
        exact_overflow.require_parking_capacity(object(), 8, generation_pools=(pool,))


def test_main_registers_pools_before_recovery_and_overflow_passes_them_to_parking():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "train_multy_CVaR.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    registration = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Attribute) and target.attr == "_generation_pools"
                                for target in node.targets))
    assert ast.unparse(registration.value) == "(strategy_pool, gen_pool)"
    loop = next(node for node in main.body if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range" and any(isinstance(arg, ast.Name) and arg.id == "start_step"
                                                        for arg in node.iter.args))
    assert registration.lineno < loop.lineno
    source = ast.parse((root / "training/exact_overflow.py").read_text())
    call = next(node for node in ast.walk(source) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "require_parking_capacity")
    pools = next(item.value for item in call.keywords if item.arg == "generation_pools")
    assert ast.unparse(pools) == "getattr(trainer, '_generation_pools', ())"
