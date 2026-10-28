"""CPU-only scheduling/lifecycle checks for concurrent API planning and coding."""

from collections import Counter
from queue import Empty
import threading
from types import SimpleNamespace

import pytest

import generation.gen_workers as gen_workers
from generation.strategy_api import run_strategy_rounds
from _retire.strategy_pipeline import StrategyArrivalStream


class QuietBar:
    def update(self, *_args):
        pass

    def close(self):
        pass


def test_ready_jobs_use_all_engines_and_refill_before_slow_worker_finishes(monkeypatch):
    monkeypatch.setattr(gen_workers, "make_progress_bar", lambda *a, **k: QuietBar())
    pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
    pool.num_workers = 8
    pool.gen_micro_batch = 8
    pool.procs = [SimpleNamespace(exitcode=None) for _ in range(8)]
    tasks = []
    ready = []
    first_returned = threading.Event()

    class TaskQueue:
        def __init__(self, rank):
            self.rank = rank

        def put(self, task):
            step, adapter, jobs, options = task
            tasks.append((self.rank, step, jobs, options))
            assert adapter == "frozen-adapter"
            assert options["return_logprobs"] and options["full_policy_sampling"]
            assert options["top_k"] == 0
            for sample, prompt, count in jobs:
                assert count == 1
                ready.append((self.rank, sample, [(prompt, [1, 2], [-.1, -.2])]))

    class Results:
        def get(self, timeout):
            if not ready:
                raise Empty
            # Slow rank zero stays in flight while another engine takes new work.
            return ready.pop()

    pool.task_queues = [TaskQueue(rank) for rank in range(8)]
    pool.result_queue = Results()
    poll_count = 0

    def poll():
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return [dict(key=0, prompt_text="first plan", count=8, step_idx=0)], False
        if first_returned.is_set() and poll_count == 2:
            return [dict(key=1, prompt_text="next plan", count=8, step_idx=4000000)], True
        return [], True

    outputs = []
    for key, result in pool.iter_ready_jobs(
            poll, total_expected=16, adapter_path="frozen-adapter",
            max_new_tokens=500, temperature=.8, top_p=.95,
            top_k=20, return_logprobs=True, full_policy_sampling=True):
        if not outputs:
            assert {task[0] for task in tasks} == set(range(8))
            first_returned.set()
        outputs.append((key, result))
    assert Counter(key for key, _ in outputs) == {0: 8, 1: 8}
    assert next(i for i, (key, _) in enumerate(outputs) if key == 1) < 8
    samples = [sample for _, _, jobs, _ in tasks for sample, _, _ in jobs]
    assert len(samples) == len(set(samples)) == 16
    assert set(samples) == set(range(8)) | set(range(16, 24))
    assert all(result[0][2] == [-.1, -.2] for _, result in outputs)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "worker_error", "dead_worker"])
def test_ready_scheduler_rejects_incomplete_or_corrupt_work(monkeypatch, fault):
    monkeypatch.setattr(gen_workers, "make_progress_bar", lambda *a, **k: QuietBar())
    pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
    pool.num_workers = 1
    pool.gen_micro_batch = 2
    pool.procs = [SimpleNamespace(exitcode=1 if fault == "dead_worker" else None)]
    pool.task_queues = [SimpleNamespace(put=lambda task: None)]

    def result(timeout):
        if fault == "dead_worker":
            raise Empty
        if fault == "worker_error":
            return 0, None, {"error": "synthetic failure"}
        return 0, 999, [("unexpected", [])]

    pool.result_queue = SimpleNamespace(get=result)
    job = dict(key=0, prompt_text="plan", count=1, step_idx=0)
    jobs = [] if fault == "missing" else [job, job] if fault == "duplicate" else [job]
    with pytest.raises(RuntimeError):
        list(pool.iter_ready_jobs(
            lambda: (jobs, True), total_expected=1, adapter_path=None,
            max_new_tokens=10, temperature=1., top_p=1.))


@pytest.mark.parametrize("abort", [False, True])
def test_prepare_reuses_pool_and_abort_discards_inflight_workers(monkeypatch, abort):
    calls = []

    class Engines:
        num_workers = 8
        sleep_supported = True

        def __init__(self, **kwargs):
            calls.append("load")

        def iter_ready_jobs(self):
            yield 0, [("code", [1])]

        def sleep(self):
            calls.append("sleep")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(gen_workers, "GenerationPool", Engines)
    pool = gen_workers.PhasedVLLMGenerationPool(
        before_start=lambda: calls.append("offload"),
        after_stop=lambda: calls.append("restore"),
        model_name="test-coder", num_workers=8)
    pool.prepare()
    assert calls == ["offload", "load"]
    stream = pool.iter_ready_jobs()
    assert next(stream)[0] == 0
    if abort:
        stream.close()
        assert calls == ["offload", "load", "shutdown", "restore"]
    else:
        assert list(stream) == []
        pool.release()
        assert calls == ["offload", "load", "sleep", "restore"]
    assert not pool.active


def test_api_producer_cancellation_stops_history_and_later_rounds():
    started = threading.Event()
    finish_request = threading.Event()
    handled = []
    calls = []

    class Remote:
        def iter_group_jobs(self, **kwargs):
            calls.append(1)
            started.set()
            assert finish_request.wait(5)
            yield 0, [("plan", [])]

    def produce(publish, stop):
        run_strategy_rounds(
            Remote(), num_parents=1, num_strategies=3,
            prompt_builder=lambda *args: "prompt",
            result_handler=lambda *args: handled.append(args) or False,
            max_retries=0, generation_options={}, stop_event=stop)

    with StrategyArrivalStream(produce) as stream:
        assert started.wait(5)
    finish_request.set()
    stream._thread.join(timeout=5)
    assert not stream._thread.is_alive()
    assert calls == [1] and handled == []


def test_api_producer_propagates_errors():
    def produce(*_args):
        raise RuntimeError("synthetic API failure")

    with StrategyArrivalStream(produce) as stream:
        stream._thread.join(timeout=5)
        with pytest.raises(RuntimeError, match="synthetic API failure"):
            stream.poll()
