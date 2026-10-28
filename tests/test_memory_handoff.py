"""CPU-only regressions for RAM admission and the GPU phase handoff."""

from contextlib import nullcontext
import multiprocessing
from types import SimpleNamespace
import sys
import threading

import pytest

import runtime.host_memory as host_memory
from runtime.host_memory import GIB, HostMemoryAdmission, TokenWeightedActivationOffloadGate


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
    monkeypatch.setattr(training, "_restore_optimizer_state_to_parameters", lambda *a: None)
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


def test_shutdown_does_not_allocate_sleep_backups():
    from generation.gen_workers import PhasedVLLMGenerationPool

    calls = []
    pool = object.__new__(PhasedVLLMGenerationPool)
    pool._awake = True
    pool._pool = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    pool.release = lambda: pytest.fail("shutdown must not back up weights")
    pool.shutdown()
    assert calls == ["shutdown"] and not pool.active


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
