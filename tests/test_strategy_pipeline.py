"""CPU-only scheduling/lifecycle checks for local and API strategy pipelines."""

import ast
import asyncio
from collections import Counter
from pathlib import Path
from queue import Empty, Queue
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


def test_local_strategies_batch_all_parents_and_preserve_dependencies_and_retries(monkeypatch):
    monkeypatch.setattr(gen_workers, "make_progress_bar", lambda *a, **k: QuietBar())
    pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
    pool.num_workers, pool.tensor_parallel_size, pool.pipeline_parallel_size = 4, 2, 1
    pool.gen_micro_batch = 32
    pool._async_worker_ids = set(range(4))
    pool.procs = [SimpleNamespace(exitcode=None) for _ in range(4)]
    queued, calls, sessions = [], [], []
    contexts, handled, accepted, active = {}, set(), set(), set()

    def prompt(chain, stage, retry):
        assert stage == 0 or (chain, stage - 1) in accepted
        assert retry == 0 or (chain, stage, retry - 1) in handled
        text = f"parent={chain}; strategy={stage}; retry={retry}; reasoning_effort=xhigh"
        contexts[text] = chain, stage, retry
        return text

    def handle(chain, stage, text, retry):
        assert text == f"plan {chain}/{stage}/{retry}"
        assert (chain, stage, retry) not in handled
        handled.add((chain, stage, retry))
        if chain == 0 and stage == 0 and retry < 3:
            return True
        accepted.add((chain, stage))
        return False

    class Tasks:
        def __init__(self, rank):
            self.rank = rank

        def put(self, task):
            if len(task) == 1:
                if task[0] == "__strategies_done__":
                    assert not active and not queued
                sessions.append((self.rank, task[0]))
                return
            step, adapter, jobs, options = task
            assert adapter is None and len(jobs) == 1
            chain, text, count = jobs[0]
            assert count == 1 and chain not in active
            assert tuple(options[key] for key in (
                "max_new_tokens", "temperature", "top_p", "top_k", "min_p")) == (
                    262144, 1., 1., 0, 0.)
            assert not options["return_logprobs"] and not options["full_policy_sampling"]
            _chain, stage, retry = contexts[text]
            assert step == 17 + stage * 10000 + retry * 1000000
            assert options["seed_rank"] == chain % pool.num_workers
            active.add(chain)
            queued.append((self.rank, chain, stage, retry))
            calls.append((self.rank, chain, stage, retry))

    class Results:
        def get(self, timeout):
            if not handled:
                assert len(queued) == 10
                assert Counter(item[0] for item in queued) == {0: 3, 1: 3, 2: 2, 3: 2}
            # Let parent zero finish its retries and all stages while other parents wait.
            index = next((i for i, item in enumerate(queued) if item[1] == 0), 0)
            rank, chain, stage, retry = queued.pop(index)
            active.remove(chain)
            return rank, chain, [(f"plan {chain}/{stage}/{retry}", [])]

    pool.task_queues = [Tasks(rank) for rank in range(4)]
    pool.result_queue = Results()
    result = pool.run_sequential_chains(
        num_chains=10, num_stages=5, prompt_builder=prompt, result_handler=handle,
        adapter_path=None, max_new_tokens=262144, temperature=1., top_p=1.,
        top_k=0, min_p=0., step_idx=17, max_retries=3)
    assert result == dict(completed=50, retries=3)
    assert len(calls) == 53 and len(accepted) == 50
    assert calls[10][1:] == (0, 0, 1)
    assert sessions == ([(rank, "__strategies__") for rank in range(4)]
                        + [(rank, "__strategies_done__") for rank in range(4)])


def test_strategy_worker_accepts_followups_before_slow_request_finishes():
    async def run():
        tasks = Queue()
        events = {name: asyncio.Event() for name in ("slow", "fast", "followup")}
        release_slow = asyncio.Event()
        started = []

        async def generate(task):
            name = task[0]
            started.append(name)
            if name == "slow":
                await release_slow.wait()
            events[name].set()

        tasks.put(("slow", None, [], {}))
        tasks.put(("fast", None, [], {}))
        session = asyncio.create_task(gen_workers._serve_strategy_requests(tasks, generate))
        await events["fast"].wait()
        assert started == ["slow", "fast"] and not events["slow"].is_set()
        tasks.put(("followup", None, [], {}))
        await events["followup"].wait()
        assert not events["slow"].is_set()
        release_slow.set()
        await events["slow"].wait()
        tasks.put(("__strategies_done__",))
        tasks.put(("__control__", "sleep"))
        assert await session is False
        assert tasks.get_nowait() == ("__control__", "sleep")

    asyncio.run(asyncio.wait_for(run(), timeout=3))


@pytest.mark.parametrize("failure", [False, True])
def test_strategy_worker_cancels_inflight_requests_on_shutdown_or_error(failure):
    async def run():
        tasks = Queue()
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def generate(task):
            if task[0] == "fail":
                await started.wait()
                raise RuntimeError("synthetic strategy failure")
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        tasks.put(("slow", None, [], {}))
        session = asyncio.create_task(gen_workers._serve_strategy_requests(tasks, generate))
        await started.wait()
        if failure:
            tasks.put(("fail", None, [], {}))
            with pytest.raises(RuntimeError, match="synthetic strategy failure"):
                await session
        else:
            tasks.put(None)
            assert await session is True
        assert cancelled.is_set()

    asyncio.run(asyncio.wait_for(run(), timeout=3))


def test_strategy_worker_preserves_native_prompt_sampling_seed_and_context_limit():
    tree = ast.parse(Path(gen_workers.__file__).read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "_generate_strategy_request")
    encoded, requests, sampling = [], [], []
    results = Queue()

    def encode(prompt, **kwargs):
        encoded.append((prompt, kwargs))
        return [1, 2, 3]

    def params(**kwargs):
        sampling.append(kwargs)
        return kwargs

    async def generate(prompt, options, adapter, request_id):
        requests.append((prompt, options, adapter, request_id))
        return SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 5], text="strategy")])

    scope = dict(
        engine_options={"enable_lora": False}, llm=SimpleNamespace(
            get_tokenizer=lambda: SimpleNamespace(encode=encode)),
        seed=42, rank=2, max_seq_length=262144, result_queue=results,
        _vllm_rollout_seed=gen_workers._vllm_rollout_seed, SamplingParams=params,
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
        _async_final_output=generate, _lora_request=lambda path: path,
        _next_request_id=lambda kind: kind,
        _canonical_generated_text=lambda tokenizer, tokens, text: text)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "strategy worker", "exec"), scope)
    options = dict(max_new_tokens=262144, temperature=1., top_p=1., top_k=0,
                   min_p=0., seed_rank=1)
    prompt = "original native prompt; reasoning_effort=xhigh; retry reminder"
    task = (1020017, None, [(5, prompt, 1)], options)
    asyncio.run(scope[method.name](task))
    assert encoded == [(prompt, {"add_special_tokens": False})]
    assert sampling == [dict(
        n=1, max_tokens=262141, temperature=1., top_p=1., top_k=-1, min_p=0.,
        skip_special_tokens=True, output_kind="final",
        seed=gen_workers._vllm_rollout_seed(42, 1020017, 1, 5, 0))]
    assert requests[0][0] == {"prompt_token_ids": [1, 2, 3]}
    assert results.get_nowait() == (2, 5, [("strategy", [4, 5])])
    scope["max_seq_length"] = 3
    asyncio.run(scope[method.name](task))
    assert len(requests) == 1
    assert results.get_nowait() == (2, 5, [("", [])])


@pytest.mark.parametrize("fault", ["wrong_rank", "worker_error", "dead_worker", "exhausted"])
def test_local_strategy_batching_rejects_worker_failures_and_retry_exhaustion(monkeypatch, fault):
    monkeypatch.setattr(gen_workers, "make_progress_bar", lambda *a, **k: QuietBar())
    pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
    pool.num_workers, pool.tensor_parallel_size, pool.pipeline_parallel_size = 2, 2, 1
    pool.gen_micro_batch, pool._async_worker_ids = 8, {0, 1}
    pool.procs = [SimpleNamespace(exitcode=1 if fault == "dead_worker" else None)] * 2
    pool.task_queues = [SimpleNamespace(put=lambda task: None) for _ in range(2)]

    def receive(timeout):
        if fault == "dead_worker":
            raise Empty
        if fault == "worker_error":
            return 0, None, {"error": "synthetic engine failure"}
        return (1 if fault == "wrong_rank" else 0), 0, [("strategy", [])]

    pool.result_queue = SimpleNamespace(get=receive)
    with pytest.raises(RuntimeError):
        pool.run_sequential_chains(
            num_chains=2, num_stages=2, prompt_builder=lambda *args: "prompt",
            result_handler=lambda *args: fault == "exhausted", max_retries=0,
            adapter_path=None, max_new_tokens=262144, temperature=1., top_p=1.)


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
