"""CPU contracts for continuously refilled coder engines and phase-2 work."""

import ast
import asyncio
from array import array
from collections import Counter
from concurrent.futures import Future, wait, FIRST_COMPLETED
from contextlib import redirect_stdout
import io
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import generation.gen_workers as workers
from models.coder_profiles import _coder_effort_entries, _phase_coder_prompt_job


ROOT = Path(__file__).resolve().parents[1]


def load_nested(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class GenerationSchedulerTests(unittest.TestCase):
    def test_b200_and_rtx_use_the_same_fixed_job_scheduler(self):
        pool = workers.GenerationPool.__new__(workers.GenerationPool)
        pool.num_workers, pool.gen_micro_batch = 2, 2
        pool._async_worker_ids, pool._worker_options = {0, 1}, {}
        pool.iter_ready_jobs = Mock(side_effect=AssertionError("legacy path must keep its scheduler"))
        pool.task_queues = [SimpleNamespace(put=lambda task: None)] * 2
        pool.procs = [SimpleNamespace(exitcode=None)] * 2
        pool.result_queue = SimpleNamespace(get=lambda **kwargs: (0, 0, [("legacy", [7])]))
        for profile in ("RTX6000", "B200"):
            pool.hardware_profile = profile
            self.assertEqual(list(pool.iter_group_jobs(["prompt"], 1, None, 99, 1., 1., show_progress=False)),
                             [(0, [("legacy", [7])])])
        pool.iter_ready_jobs.assert_not_called()

    def test_refill_before_slow_request_finishes_and_keep_exact_counts(self):
        pool = workers.GenerationPool.__new__(workers.GenerationPool)
        pool.num_workers, pool.gen_micro_batch = 2, 2
        pool._async_worker_ids = {0, 1}
        pool.procs = [SimpleNamespace(exitcode=None)] * 2
        pool.shutdown = Mock()
        queued, markers, submitted, outputs = [], [], [], []

        class Tasks:
            def __init__(self, rank):
                self.rank = rank

            def put(self, task):
                if len(task) == 1:
                    markers.append((self.rank, task[0]))
                    return
                step, adapter, jobs, options = task
                self_case.assertEqual(adapter, "adapter_step7")
                self_case.assertEqual(len(jobs), 1)
                self_case.assertEqual((options["top_k"], options["min_p"]), (0, 0.))
                self_case.assertTrue(options["return_logprobs"] and options["full_policy_sampling"])
                for sample, prompt, count in jobs:
                    self_case.assertEqual(count, 1)
                    submitted.append((self.rank, sample, prompt, step))
                    queued.append((self.rank, sample, [(prompt, [7, 8], [-.1, -.2])]))

        class Results:
            def get(self, timeout):
                if not outputs:
                    self_case.assertEqual(Counter(rank for rank, *_ in submitted), {0: 2, 1: 2})
                    self_case.assertEqual(Counter((rank, prompt) for rank, _sample, prompt, _step in submitted),
                        {(0, "pilot xhigh"): 1, (1, "pilot xhigh"): 1,
                         (0, "pilot medium"): 1, (1, "pilot medium"): 1})
                # Hold rank 0/sample 0 until a replacement on that SAME engine completed.
                for index, (rank, sample, _result) in enumerate(queued):
                    if sample == 12:
                        self_case.assertTrue(any(item[1] == 0 for item in queued))
                        return queued.pop(index)
                index = next((i for i, item in enumerate(queued) if item[0] == 0 and item[1] != 0),
                             next((i for i, item in enumerate(queued) if item[1] != 0), 0))
                return queued.pop(index)

        self_case = self
        pool.task_queues = [Tasks(rank) for rank in range(2)]
        pool.result_queue = Results()
        polls = 0

        def poll():
            nonlocal polls
            polls += 1
            if polls == 1:
                return [dict(key=0, count=2, prompt_text="pilot xhigh", step_idx=7),
                        dict(key=1, count=2, prompt_text="pilot medium", step_idx=7)], False
            if polls == 2:
                return [dict(key=2, count=2, prompt_text="phase2 xhigh", step_idx=4000007)], True
            return [], True

        for key, result in pool.iter_ready_jobs(
                poll, total_expected=6, adapter_path="adapter_step7", max_new_tokens=262144,
                temperature=1., top_p=1., top_k=0, min_p=0., return_logprobs=True,
                full_policy_sampling=True, show_progress=False):
            outputs.append((key, result))
        self.assertEqual(Counter(key for key, _ in outputs), {0: 2, 1: 2, 2: 2})
        self.assertEqual(len({sample for _, sample, *_ in submitted}), 6)
        self.assertIn((0, 12, "phase2 xhigh", 4000007), submitted)
        self.assertEqual([marker for _, marker in markers],
                         ["__rollouts__", "__rollouts__", "__rollouts_done__", "__rollouts_done__"])
        self.assertTrue(all(result[0][2] == [-.1, -.2] for _, result in outputs))
        pool.shutdown.assert_not_called()

    def test_worker_accepts_new_requests_and_preserves_next_control_message(self):
        async def run():
            tasks = Queue()
            fast, replacement, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def generate(task):
                if task[0] == "slow":
                    await release.wait()
                elif task[0] == "fast":
                    fast.set()
                else:
                    replacement.set()

            tasks.put(("slow", None, [], {}))
            tasks.put(("fast", None, [], {}))
            session = asyncio.create_task(workers._serve_generation_requests(
                tasks, generate, done_marker="__rollouts_done__"))
            await fast.wait()
            tasks.put(("replacement", None, [], {}))
            await replacement.wait()
            self.assertFalse(release.is_set())
            release.set()
            await asyncio.sleep(.02)
            tasks.put(("__rollouts_done__",))
            tasks.put(("__control__", "sleep"))
            self.assertIs(await session, False)
            self.assertEqual(tasks.get_nowait(), ("__control__", "sleep"))
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_single_request_keeps_prompt_adapter_sampling_context_and_logprobs(self):
        results, sampling, requests, encoded = Queue(), [], [], []
        def encode(prompt, **kwargs):
            encoded.append((prompt, kwargs))
            return [1, 2, 3]
        async def generate(prompt, params, adapter, request_id):
            requests.append((prompt, params, adapter))
            candidate = SimpleNamespace(token_ids=[7, 8], text="program",
                logprobs=[{7: SimpleNamespace(logprob=-.1)}, {8: SimpleNamespace(logprob=-.2)}])
            return SimpleNamespace(outputs=[candidate])
        def params(**kwargs):
            sampling.append(kwargs)
            return kwargs
        scope = dict(engine_options=None, llm=SimpleNamespace(
            get_tokenizer=lambda: SimpleNamespace(encode=encode)), max_seq_length=262144,
            result_queue=results, rank=4, seed=42, array=array,
            _vllm_rollout_seed=workers._vllm_rollout_seed, SamplingParams=params,
            RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
            _async_final_output=generate, _lora_request=lambda value: value,
            _next_request_id=lambda kind: kind,
            _chosen_token_logprobs=workers._chosen_token_logprobs,
            _canonical_generated_text=lambda tokenizer, tokens, text: text)
        call = load_nested(ROOT / "generation/gen_workers.py", "_generate_single_request", scope)
        options = dict(max_new_tokens=262144, temperature=1., top_p=1., top_k=0, min_p=0.,
                       return_logprobs=True, full_policy_sampling=True, seed_rank=0)
        asyncio.run(call((4000007, "adapter_step7", [(29, "original xhigh retry prompt", 1)], options)))
        self.assertEqual(encoded, [("original xhigh retry prompt", {})])
        self.assertEqual(sampling, [dict(n=1, max_tokens=262141, temperature=1., top_p=1.,
            top_k=-1, min_p=0., repetition_penalty=1., logprobs=0, skip_special_tokens=True,
            output_kind="final", seed=workers._vllm_rollout_seed(42, 4000007, 0, 29, 0))])
        self.assertEqual(requests[0][2], "adapter_step7")
        rank, sample, result = results.get_nowait()
        self.assertEqual((rank, sample, result[0][:2]), (4, 29, ("program", [7, 8])))
        self.assertAlmostEqual(result[0][2][0], -.1)
        self.assertAlmostEqual(result[0][2][1], -.2)

    def test_phase2_polls_new_parent_during_first_parents_generation(self):
        records, prompt_jobs = [], []
        for source in range(4):
            future = Future()
            if source < 2:
                future.set_result(1.)
            records.append(dict(strategy_source_job_idx=source, _reward_future=future))
            prompt_jobs.append(dict(parent_group=source // 2, strategy_index=source % 2))
        allocations = []
        def allocate(ready, sources, **kwargs):
            self.assertTrue(all(record["_reward_future"].done() for record in ready))
            allocations.append(sources)
            return [2] * len(sources)
        scope = dict(prompt_jobs=prompt_jobs, pilot_programs_per_strategy=1,
                     programs_per_strategy=3, strategies_per_parent=2, step_idx=7,
                     _finish_strategy_pilots=allocate, wait=wait, FIRST_COMPLETED=FIRST_COMPLETED)
        call = load_nested(ROOT / "train_multy_CVaR.py", "_run_adaptive_followups_as_ready", scope)
        def generate(poll, expected):
            self.assertEqual(expected, 8)
            phases, done = poll()
            self.assertFalse(done)
            self.assertEqual(phases[0][:2], ([2, 2, 0, 0], 0))
            self.assertEqual(poll(), ([], False))  # A busy engine can keep decoding.
            for record in records[2:]:
                record["_reward_future"].set_result(1.)
            phases, done = poll()
            self.assertTrue(done)
            self.assertEqual(phases[0][:2], ([0, 0, 2, 2], 1))
        with redirect_stdout(io.StringIO()):
            counts = call(records, range(4), generate, pipelined=True)
        self.assertEqual(counts, [2, 2, 2, 2])
        self.assertEqual(allocations, [[0, 1], [2, 3]])

    def test_streamed_phase2_keeps_effort_variants_source_mapping_and_retries(self):
        cfg = SimpleNamespace(coder_template_kind="qwen3.8", coder_reasoning_effort="low",
            phase2_low_programs_per_strategy=2, phase2_medium_programs_per_strategy=1,
            phase2_xhigh_programs_per_strategy=1, max_new_tokens=262144, temperature=1., top_p=1.,
            sampling_top_k=0, sampling_min_p=0.)
        prompts = [dict(messages=f"parent {i}", prompt_text=f"parent {i}:low") for i in range(2)]
        records, deferred, jobs_seen = [], [], []
        class Pool:
            def iter_ready_jobs(self, poll, **options):
                self_case.assertEqual((options["total_expected"], options["adapter_path"]), (8, "adapter7"))
                jobs, done = poll()
                self_case.assertTrue(done)
                jobs_seen.extend(jobs)
                for job in jobs:
                    for _ in range(job["count"]):
                        yield job["key"], [(job["prompt_text"], [1, 2], [-.1, -.2])]
        self_case = self
        def queue(prompt, text, tokens, logprobs, **metadata):
            record = dict(prompt=prompt, text=text, tokens=tokens, logprobs=logprobs, **metadata)
            records.append(record)
            return record
        scope = dict(cfg=cfg, prompt_jobs=prompts, source_job_indices=[0, 1], step_idx=7,
            _coder_effort_entries=_coder_effort_entries, _phase_coder_prompt_job=_phase_coder_prompt_job,
            _render=lambda messages, **kwargs: f"{messages}:{kwargs['reasoning_effort']}",
            phase_prompt_job_cache={}, gen_pool=Pool(), adapter_path="adapter7", training_enabled=True,
            _uses_sequence_level_policy_ratio=lambda cfg: True, _queue_rollout=queue,
            _attach_strategy_plan=lambda *a: None,
            _defer_coder_format_retries=lambda batch, **kwargs: deferred.append((batch, kwargs)))
        call = load_nested(ROOT / "train_multy_CVaR.py", "_run_streamed_vllm_followups", scope)
        call(lambda: ([([4, 4], 0, 8, 8, 8)], True), 8)
        self.assertEqual(Counter(record["text"].split(":")[-1] for record in records),
                         {"low": 4, "medium": 2, "xhigh": 2})
        self.assertTrue(all(record["text"] == prompts[record["prompt"]]["prompt_text"] for record in records))
        self.assertEqual(Counter(record["strategy_source_job_idx"] for record in records), {0: 4, 1: 4})
        self.assertEqual(deferred, [(records, dict(seed_offset=4000000))])
        self.assertTrue(all(job["step_idx"] == 4000007 for job in jobs_seen))

    def test_phase2_starts_before_other_parents_pilots_finish(self):
        cfg = SimpleNamespace(coder_template_kind="qwen3.8", coder_reasoning_effort="low",
            pilot_medium_programs_per_strategy=0, pilot_xhigh_programs_per_strategy=0,
            phase2_low_programs_per_strategy=1, phase2_medium_programs_per_strategy=0,
            phase2_xhigh_programs_per_strategy=0, max_new_tokens=262144, temperature=1., top_p=1.,
            sampling_top_k=0, sampling_min_p=0.)
        prompts = [dict(parent_group=i // 2, strategy_index=i % 2,
                        messages=f"source{i}", prompt_text=f"source{i}:low") for i in range(4)]
        pilots, records, deferred, finished = [], [], [], []
        def queue(prompt, text, tokens, logprobs, **metadata):
            future = Future()
            future.set_result(1.)
            record = dict(_reward_future=future, text=text, **metadata)
            records.append(record)
            return record
        self_case = self
        class Pool:
            def iter_ready_jobs(self, poll, **options):
                pending, done = [], False
                while not done or pending:
                    jobs, done = poll()
                    pending.extend(jobs)
                    if not pending:
                        self_case.fail("expected unfinished pilot or phase-2 work")
                    # Generate ready phase 2 immediately, holding parent 1's pilots.
                    position = next((i for i, job in enumerate(pending)
                                     if job["step_idx"] >= 4000000), 0)
                    job = pending.pop(position)
                    if job["step_idx"] >= 4000000 and len(finished) == 2:
                        self_case.assertEqual(len(pilots), 2)
                        self_case.assertTrue(any(item["step_idx"] == 7 for item in pending))
                    self_case.assertEqual(job["count"], 1)
                    finished.append(job)
                    yield job["key"], [(job["prompt_text"], [1], [-.1])]
                self_case.assertEqual(options["total_expected"], 8)
        scope = dict(cfg=cfg, prompt_jobs=prompts, source_job_indices=list(range(4)), step_idx=7,
            pilot_programs_per_strategy=1, programs_per_strategy=2, strategies_per_parent=2,
            _coder_effort_entries=_coder_effort_entries, _phase_coder_prompt_job=_phase_coder_prompt_job,
            _render=lambda messages, **kwargs: f"{messages}:{kwargs['reasoning_effort']}",
            phase_prompt_job_cache={}, gen_pool=Pool(), adapter_path="adapter7", training_enabled=True,
            _uses_sequence_level_policy_ratio=lambda cfg: True, _queue_rollout=queue,
            _attach_strategy_plan=lambda *a: None, wait=wait, FIRST_COMPLETED=FIRST_COMPLETED,
            _finish_strategy_pilots=lambda ready, sources, **kwargs: [1] * len(sources),
            _defer_coder_format_retries=lambda batch, **kwargs: deferred.append((batch, kwargs)))
        streamed = load_nested(ROOT / "train_multy_CVaR.py", "_run_streamed_vllm_followups", scope)
        planner = load_nested(ROOT / "train_multy_CVaR.py", "_run_adaptive_followups_as_ready", scope)
        with redirect_stdout(io.StringIO()):
            counts = planner(pilots, range(4),
                lambda poll, total: streamed(poll, total, pilot_records=pilots),
                pipelined=True, streaming_pilots=True)
        self.assertEqual(counts, [1] * 4)
        self.assertEqual(len(pilots), 4)
        self.assertEqual(Counter(record["rollout_phase"] for record in records), {"pilot": 4, "adaptive": 4})
        self.assertEqual(Counter(record["strategy_source_job_idx"] for record in records), {0: 2, 1: 2, 2: 2, 3: 2})
        self.assertEqual(deferred, [(records, dict(seed_offset=0))])

    def test_worker_failure_and_generator_close_cancel_session(self):
        for fault in ("worker_error", "over_budget", "close"):
            with self.subTest(fault=fault):
                pool = workers.GenerationPool.__new__(workers.GenerationPool)
                pool.num_workers, pool.gen_micro_batch = 1, 2
                pool._async_worker_ids = {0}
                pool.task_queues = [SimpleNamespace(put=lambda task: None)]
                pool.procs = [SimpleNamespace(exitcode=None)]
                pool.shutdown = Mock()
                result = (0, None, dict(error="failure")) if fault == "worker_error" else (0, 0, [("ok", [1])])
                pool.result_queue = SimpleNamespace(get=lambda **kwargs: result)
                stream = pool.iter_ready_jobs(lambda: ([dict(key=0, count=3 if fault == "over_budget" else 2,
                    prompt_text="prompt", step_idx=0)], True), total_expected=2,
                    adapter_path=None, max_new_tokens=99, temperature=1., top_p=1., show_progress=False)
                if fault == "close":
                    next(stream)
                    stream.close()
                else:
                    with self.assertRaises(RuntimeError):
                        list(stream)
                pool.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
