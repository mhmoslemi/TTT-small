"""Retry contracts and real runner/scheduler routing without GPU dependencies."""

import ast
from collections import deque
from concurrent.futures import Future
from copy import deepcopy
import io
import json
from pathlib import Path
import queue
import re
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import nullcontext, redirect_stdout

from experiment_io import save_strategy_response
from output_retries import (coder_output_issue, coder_retry_prompt_job,
                            final_answer_scope, output_retry_messages,
                            retry_metadata, strategy_retry_needed)

ROOT = Path(__file__).resolve().parents[1]
VALID = "<think>check</think>\n```python\ndef run():\n    return 1\n```"


def functions_from_file(filename, names, scope):
    tree = ast.parse((ROOT / filename).read_text())
    selected = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.setdefault(node.name, node)
    nodes = list(selected.values())
    if {node.name for node in nodes} != set(names):
        raise AssertionError("missing tested production function")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)


class Bar:
    def __init__(self, total=0, **_kwargs):
        self.total = total
        self.n = 0
    def update(self, n):
        self.n += n
    def refresh(self):
        pass
    def close(self):
        pass


class InlinePool:
    def submit(self, fn, *args, **kwargs):
        result = Future()
        result.set_result(fn(*args, **kwargs))
        return result


class OutputContractTests(unittest.TestCase):
    def test_only_complete_final_python_blocks_pass(self):
        self.assertIsNone(coder_output_issue(VALID, require_final_marker=True))
        self.assertIsNone(coder_output_issue("```python\nthis is not valid Python\n```"))
        for text in ("", "plain prose", "def run(): pass", "```python\n```",
                     "```python\ndef run(): pass", "<think>```python\na=1\n```",
                     "```python\na=1\n```</think>no final program",
                     "```python\na=1\n```\n```python\na=2"):
            with self.subTest(text=text):
                self.assertIsNotNone(coder_output_issue(text))
        self.assertIsNone(coder_output_issue(
            "analysis ```python\ndraft\n```<|channel|>final<|message|>" + VALID))
        self.assertIsNone(coder_output_issue("assistantfinal```python\na=1\n```"))
        self.assertIsNotNone(coder_output_issue("```python\na=1\n```", require_final_marker=True))

    def test_reminder_does_not_mutate_or_stack_on_original(self):
        original = [{"role": "system", "content": "system"},
                    {"role": "user", "content": "parent and previous strategies"}]
        frozen = deepcopy(original)
        for kind in ("strategy", "coder"):
            retry = output_retry_messages(original, kind)
            self.assertEqual(original, frozen)
            self.assertEqual(retry[0], original[0])
            self.assertTrue(retry[-1]["content"].startswith(original[-1]["content"]))
            self.assertEqual(retry[-1]["content"].count("Your previous attempt"), 1)
            self.assertEqual(retry, output_retry_messages(original, kind))

    def test_strategy_three_retries_means_four_total_attempts(self):
        for attempt in range(3):
            self.assertTrue(strategy_retry_needed("missing", attempt, 3, label="s1"))
        self.assertFalse(strategy_retry_needed(None, 3, 3, label="s1"))
        with self.assertRaisesRegex(RuntimeError, "3 format retries.*4 attempts"):
            strategy_retry_needed("missing", 3, 3, label="s1")

    def test_each_strategy_attempt_and_prompt_survive_canonical_replacement(self):
        with TemporaryDirectory() as tmp:
            first = save_strategy_response(Path(tmp), 0, 2, 1, 3, "bad", attempt=0,
                                           prompt_text="original", extraction_issue="missing")
            second = save_strategy_response(Path(tmp), 0, 2, 1, 3, "<strategy>ok</strategy>",
                                            attempt=1, prompt_text="original + reminder")
            accepted = save_strategy_response(Path(tmp), 0, 2, 1, 3, "<strategy>ok</strategy>")
            self.assertEqual(first.read_text(), "bad")
            self.assertEqual(first.with_suffix(".prompt.txt").read_text(), "original")
            self.assertEqual(second.read_text(), accepted.read_text())
            self.assertFalse(json.loads(first.with_suffix(".meta.json").read_text())["accepted"])
            self.assertTrue(json.loads(second.with_suffix(".meta.json").read_text())["accepted"])

    def test_strategy_retries_remain_configured_and_coder_retry_is_opt_in(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            self.assertRegex(path.read_text(), r"(?m)^strategy_format_max_retries: 3$")
        source = (ROOT / "train_multy_CVaR.py").read_text()
        self.assertIn('"--strategy-format-max-retries"', source)
        self.assertIn('"--coder-retry"', source)
        launcher = (ROOT / "run.sh").read_text()
        self.assertIn("sh run.sh --coder-retry", launcher)


class RunnerRetryTests(unittest.TestCase):
    def runner(self, backend, *, x_grpo=False):
        cfg = SimpleNamespace(
            coder_template_kind="qwen3.8", coder_reasoning_effort="medium",
            group_size=3, seed=12, max_new_tokens=32000, temperature=.9,
            top_p=.92, sampling_top_k=0, sampling_min_p=0,
            generation_backend=backend, sandbox_timeout_s=600,
            coder_retry=True)
        scope = dict(
            cfg=cfg, Future=Future, coder_output_issue=coder_output_issue,
            coder_retry_prompt_job=coder_retry_prompt_job, retry_metadata=retry_metadata,
            parent_ctxs=[object()], step_idx=0, exp_dir=Path("unused"),
            x_grpo_mode=x_grpo, x_grpo_groups_per_context=1,
            queued_by_parent={0: 0}, queued_by_group={0: 0}, planned_by_group={0: 0},
            group_responses={0: []}, reward_futures={0: []},
            primary_rollout_records=[], coder_retry_records=[],
            deferred_coder_retry_batches=[],
            phase_prompt_job_cache={}, coder_retry_prompt_cache={},
            entropy_directory=None, rollout_io_futures=[],
            rollout_io_pool=InlinePool(), reward_pool=InlinePool(),
            isolated_eval=False, training_enabled=True, two_stage_rollouts=True,
            defer_gpu_evaluation=False, deferred_rollouts=[], vllm_logprob_records=[],
            source_job_indices=[0], adapter_path="unchanged_lora", model=object(),
            tokenizer=object(), cap_state={"value": 0}, RewardResult=SimpleNamespace,
            eval_progress_condition=threading.Condition(), eval_bar=Bar(3),
            make_progress_bar=Bar, _uses_sequence_level_policy_ratio=lambda cfg: False,
            _attach_strategy_plan=lambda *args: None,
        )
        scope["_record_evaluation_completion"] = lambda f: scope["eval_bar"].update(1)
        evaluated = []
        def evaluate(text, parent, timeout):
            evaluated.append(text)
            return SimpleNamespace(reward=1, valid=True, code="pass")
        scope["problem"] = SimpleNamespace(
            fail_score=0, require_final_code_marker=True, compute_reward=evaluate)
        saved = []
        scope["save_rollout_artifacts"] = lambda *args, **kw: saved.append((args, kw))
        scope["_render"] = lambda messages, rollout_phase=None: (
            "medium" + repr(messages))
        messages = [{"role": "user", "content": "original parent plus strategy"}]
        scope["prompt_jobs"] = [dict(
            parent_group=0, messages=messages, count=3, assigned_fold_index=0,
            prompt_text=scope["_render"](messages), coder_reasoning_effort="medium")]
        calls = []
        seeds = []
        scope["_seed_local_generation"] = seeds.append
        def generate(prompts, counts, logprobs):
            calls.append((prompts, counts))
            call = len(calls)
            for index in reversed(range(len(prompts))):
                outputs = []
                for ordinal in range(counts[index]):
                    # Pilot: one valid + one missing. Its retry also fails.
                    # Phase2: missing original, successful retry.
                    text = VALID if (call == 1 and ordinal == 0) or call == 4 else "no final output"
                    outputs.append((text, [101, 102], [-.1, -.2]) if logprobs
                                   else (text, [101, 102]))
                yield index, outputs
        def remote(**kw):
            self.assertEqual(kw["max_new_tokens"], 32000)
            self.assertEqual(kw["temperature"], .9)
            self.assertEqual(kw["top_p"], .92)
            self.assertEqual(kw["adapter_path"], "unchanged_lora")
            seeds.append(kw["step_idx"])
            yield from generate(kw["prompts_by_group"], kw["counts_by_group"], True)
        scope["gen_pool"] = (SimpleNamespace(iter_group_jobs=remote) if backend == "vllm" else None)
        scope["generate_prompt_jobs"] = lambda m, t, p, c, cfg, **kw: generate(p, c, False)
        functions_from_file("train_multy_CVaR.py", {
            "_coder_effort_for_rollout_phase", "_phase_coder_prompt_job",
            "_queue_rollout", "_submit_rollout", "_run_coder_format_retries",
            "_defer_coder_format_retries",
            "_drain_deferred_coder_format_retries",
            "_run_vllm_code_phase", "_run_local_code_phase"}, scope)
        return scope, calls, seeds, saved, evaluated

    def test_extra_attempts_preserve_prompts_phases_failures_and_budget(self):
        for backend in ("vllm", "hf"):
            for x_grpo in (False, True):
                with self.subTest(backend=backend, x_grpo=x_grpo), redirect_stdout(io.StringIO()):
                    s, calls, seeds, saved, evaluated = self.runner(backend, x_grpo=x_grpo)
                    original_job = deepcopy(s["prompt_jobs"][0])
                    run = s["_run_vllm_code_phase" if backend == "vllm" else "_run_local_code_phase"]
                    pilots = run([2], phase="pilot", seed_offset=0, progress_desc="pilots")
                    followups = run([1], phase="adaptive", seed_offset=4000000, progress_desc="phase2")
                    self.assertEqual(len(pilots), 2)  # allocator sees only originals
                    self.assertEqual(len(followups), 1)
                    self.assertEqual(s["queued_by_parent"], {0: 3})
                    self.assertEqual(s["planned_by_group"], {0: 3})
                    records = s["group_responses"][0]
                    self.assertEqual(len(records), 5)
                    self.assertEqual(len(s["coder_retry_records"]), 2)
                    self.assertEqual(len(s["primary_rollout_records"]), 3)
                    self.assertEqual(s["prompt_jobs"][0], original_job)
                    self.assertEqual([r["rollout_index"] for r in records], list(range(5)))
                    self.assertEqual([f.result().reward for f in s["reward_futures"][0]], [1, 0, 0, 0, 1])
                    self.assertEqual(len(evaluated), 2)  # missing artifacts never hit sandbox
                    self.assertEqual(s["eval_bar"].n, 5)
                    self.assertEqual(s["eval_bar"].total, 5)
                    self.assertEqual(len(saved), 5)
                    self.assertEqual(len(calls), 4)  # still-missing retry is NOT retried
                    self.assertEqual(len(set(seeds)), 4)
                    for index, expected in [(0, "medium"), (1, "medium"),
                                            (2, "medium"), (3, "medium")]:
                        self.assertTrue(calls[index][0][0].startswith(expected))
                    for rec in s["coder_retry_records"]:
                        parent = records[rec["retry_of_rollout"]]
                        self.assertEqual(rec["retry_attempt"], 1)
                        self.assertFalse(rec["counts_toward_allocation"])
                        self.assertEqual(rec["strategy_rollout_phase"], parent["strategy_rollout_phase"])
                        self.assertEqual(rec["strategy_source_job_idx"], 0)
                        job = s["prompt_jobs"][rec["job_idx"]]
                        self.assertIn("Your previous attempt", job["prompt_text"])
                        self.assertNotIn("no final output", job["prompt_text"])
                        self.assertEqual(saved[rec["rollout_index"]][1]["prompt_text"], job["prompt_text"])
                    if backend == "vllm":
                        self.assertEqual(len(s["vllm_logprob_records"]), 5)
                        self.assertTrue(all(r["behavior_logprobs"] == [-.1, -.2] for r in records))

    def test_already_retried_or_valid_records_never_generate_more(self):
        s, calls, *_ = self.runner("vllm")
        s["_run_coder_format_retries"]([
            dict(output_format_issue="missing", retry_attempt=1),
            dict(output_format_issue=None, retry_attempt=0)], seed_offset=0)
        self.assertFalse(calls)

    def test_coder_retry_disabled_keeps_failures_without_extra_generation(self):
        with redirect_stdout(io.StringIO()):
            s, calls, _, saved, evaluated = self.runner("vllm")
            s["cfg"].coder_retry = False
            records = s["_run_vllm_code_phase"](
                [3], phase="pilot", seed_offset=0, progress_desc="pilots")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(records), 3)
        self.assertEqual(len(s["primary_rollout_records"]), 3)
        self.assertEqual(s["coder_retry_records"], [])
        self.assertEqual(s["deferred_coder_retry_batches"], [])
        self.assertEqual(len(saved), 3)
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(
            [future.result().reward for future in s["reward_futures"][0]],
            [1, 0, 0])

    def test_adaptive_phases_finish_before_deferred_format_retries(self):
        for backend in ("vllm", "hf"):
            with self.subTest(backend=backend), redirect_stdout(io.StringIO()):
                s, calls, seeds, *_ = self.runner(backend)
                run = s["_run_vllm_code_phase" if backend == "vllm"
                        else "_run_local_code_phase"]
                pilots = run(
                    [2], phase="pilot", seed_offset=0,
                    progress_desc="pilots", defer_format_retries=True)
                self.assertEqual(len(pilots), 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(s["coder_retry_records"]), 0)

                followups = run(
                    [1], phase="adaptive", seed_offset=4_000_000,
                    progress_desc="phase2", defer_format_retries=True)
                self.assertEqual(len(followups), 1)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(s["coder_retry_records"]), 0)
                self.assertEqual(len(s["deferred_coder_retry_batches"]), 2)

                s["_drain_deferred_coder_format_retries"]()
                self.assertEqual(len(calls), 4)
                self.assertEqual(len(s["coder_retry_records"]), 2)
                self.assertEqual(s["deferred_coder_retry_batches"], [])
                self.assertEqual(
                    seeds,
                    [0, 4_000_000, 20_000_000, 24_000_000])

    def test_one_stage_generation_also_keeps_originals_and_adds_only_one_retry(self):
        tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        for backend, marker in (("vllm", "_run_vllm_code_phase"),
                                ("hf", "_run_local_code_phase")):
            with self.subTest(backend=backend), redirect_stdout(io.StringIO()):
                scope, calls, *_ = self.runner(backend)
                branch = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.If) and any(
                                  isinstance(child, ast.FunctionDef) and child.name == marker
                                  for child in node.body))
                scope["total_rollouts"] = 3
                exec(compile(ast.Module(body=branch.orelse, type_ignores=[]),
                             "ordinary_rollouts", "exec"), scope)
                self.assertEqual(len(scope["primary_rollout_records"]), 3)
                self.assertEqual(len(scope["coder_retry_records"]), 2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(scope["queued_by_parent"], {0: 3})
                self.assertTrue(all(p.startswith("medium") for prompts, _ in calls for p in prompts))

    def test_entropic_overlap_uses_all_attempts_and_actual_group_normalization(self):
        # Only tensor constructors are stubbed: execute the real overlap
        # builder, including prompt lookups, weights and per-record caching.
        class Values(list):
            size = property(len)
            def max(self):
                return max(self)
            def min(self):
                return min(self)
        class Tensor:
            def __init__(self, value, **kwargs):
                self.value = value
            def cpu(self):
                return self
        with redirect_stdout(io.StringIO()):
            s, *_ = self.runner("vllm")
            s["_run_vllm_code_phase"]([2], phase="pilot", seed_offset=0, progress_desc="pilot")
        s["np"] = SimpleNamespace(asarray=lambda value, **kw: Values(value), float64="float64")
        s["compute_group_advantages"] = lambda rewards, *a, **kw: (
            [r - sum(rewards) / len(rewards) for r in rewards], 1, "beta", {})
        functions_from_file("train_multy_CVaR.py", {"_prepare_entropic_overlap_group"}, s)
        fake_torch = SimpleNamespace(as_tensor=Tensor, tensor=Tensor, float32="float32", long="long")
        tokenize = lambda text, **kw: SimpleNamespace(input_ids=Tensor(text))
        records = s["group_responses"][0]
        with patch.dict("sys.modules", {"torch": fake_torch}):
            examples = s["_prepare_entropic_overlap_group"](
                tokenize, 0, records, s["reward_futures"][0], s["prompt_jobs"], {}, 1)
        self.assertEqual(len(examples), 3)
        self.assertEqual([ex["sample_weight"] for ex in examples], [1 / 3] * 3)
        for record, ex in zip(records, examples):
            self.assertIs(record["_entropic_overlap_example"], ex)
            self.assertEqual(ex["prompt_ids"].value, s["prompt_jobs"][record["job_idx"]]["prompt_text"])
        self.assertLess(examples[1]["advantage"], 0)  # original failure retained
        self.assertLess(examples[2]["advantage"], 0)  # failed retry retained too


class StrategyAttemptTests(unittest.TestCase):
    def scope(self, directory):
        tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        scope = dict(re=re, final_answer_scope=final_answer_scope,
                     output_retry_messages=output_retry_messages,
                     strategy_retry_needed=strategy_retry_needed,
                     save_strategy_response=save_strategy_response,
                     strategy_prompt_cache={}, strategy_attempt_prompts={},
                     strategy_format_max_retries=3, exp_dir=Path(directory), step_idx=0)
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in {
                         "_STRATEGY_FALLBACK", "_STRATEGY_BLOCK_RE", "_STRATEGY_FINAL_MARKERS"}
                             for t in node.targets)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), "constants", "exec"), scope)
        functions_from_file("train_multy_CVaR.py", {
            "_extract_final_strategy", "_strategy_prompt", "_handle_strategy_attempt",
            "_record_strategy_response", "_generate_strategy_batch"}, scope)
        scope["problem"] = SimpleNamespace(build_strategy_messages=lambda messages, **kw:
                                            deepcopy(messages) + [{"role": "user", "content": repr(kw)}])
        scope["_render_strategy"] = lambda messages, **kw: "HIGH:" + repr(messages)
        scope["strategy_chains"] = [dict(chain_id=0, parent_group=0, fold_index=0,
                                         source_indices=[0, 0], strategies=[])]
        scope["source_prompt_jobs"] = [dict(messages=[{"role": "user", "content": "task"}])]
        return scope

    def test_frozen_prompt_reused_and_only_valid_strategy_is_registered(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            s = self.scope(directory)
            chain = s["strategy_chains"][0]
            first = s["_strategy_prompt"](s["source_prompt_jobs"], chain, 0)
            s["source_prompt_jobs"][0]["messages"][0]["content"] = "changed task"
            for attempt in range(4):
                if attempt:
                    prompt = s["_strategy_prompt"](s["source_prompt_jobs"], chain, 0, retry=attempt)
                    self.assertNotIn("changed task", prompt)
                    self.assertTrue(prompt.startswith("HIGH:"))
                    self.assertEqual(prompt.count("Your previous attempt"), 1)
                response = "bad" if attempt < 3 else "<think>reason</think><strategy>final plan</strategy>"
                retry = s["_handle_strategy_attempt"](0, 0, response, attempt)
                self.assertEqual(retry, attempt < 3)
                self.assertEqual(len(chain["strategies"]), int(attempt == 3))
            self.assertEqual(chain["strategies"][0]["strategy"], "final plan")
            files = list((Path(directory) / "step00").glob("*_attempt*.meta.json"))
            self.assertEqual(len(files), 4)
            self.assertEqual(sum(json.loads(p.read_text())["accepted"] for p in files), 1)
            self.assertFalse(s["strategy_prompt_cache"])
            self.assertFalse(s["strategy_attempt_prompts"])

    def test_failed_last_attempt_saved_before_error_without_canonical_fallback(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            s = self.scope(directory)
            chain = s["strategy_chains"][0]
            for attempt in range(4):
                s["_strategy_prompt"](s["source_prompt_jobs"], chain, 0, retry=attempt)
                if attempt < 3:
                    self.assertTrue(s["_handle_strategy_attempt"](0, 0, "bad", attempt))
                else:
                    with self.assertRaisesRegex(RuntimeError, "4 attempts"):
                        s["_handle_strategy_attempt"](0, 0, "bad", attempt)
            step = Path(directory) / "step00"
            self.assertEqual(len(list(step.glob("*_attempt*.meta.json"))), 4)
            self.assertFalse((step / "step00_group00_fold00_strategy00.txt").exists())
            self.assertEqual(chain["strategies"], [])

    def test_api_batch_path_saves_at_arrival_and_obeys_same_retry_limit(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            s = self.scope(directory)
            s["cfg"] = SimpleNamespace(strategy_max_new_tokens=131072,
                                        strategy_temperature=.6, strategy_top_p=.95)
            s["strategies_per_parent"] = 2
            seeds = []
            def generate(**kw):
                seeds.append(kw["step_idx"])
                self.assertEqual(kw["adapter_path"], None)
                self.assertEqual(kw["max_new_tokens"], 131072)
                yield 0, [("bad" if len(seeds) < 4 else "<strategy>ok</strategy>", [])]
                # Saving is immediate, not delayed until the generator finishes.
                self.assertEqual(len(list((Path(directory) / "step00").glob("*_attempt*.meta.json"))), len(seeds))
            s["planning_pool"] = SimpleNamespace(iter_group_jobs=generate)
            for attempt in range(4):
                prompt = s["_strategy_prompt"](s["source_prompt_jobs"], s["strategy_chains"][0], 0, retry=attempt)
                needs_retry = s["_generate_strategy_batch"]([prompt], [0], 0, retry=attempt)
                self.assertEqual(needs_retry, {0: attempt < 3})
            self.assertEqual(len(set(seeds)), 4)

    def test_inprocess_strategy_path_retries_before_next_dependent_stage(self):
        tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                     and any(isinstance(child, ast.For)
                             and isinstance(child.target, ast.Name)
                             and child.target.id == "strategy_index" for child in node.body))
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            s = self.scope(directory)
            s.update(dict(
                cfg=SimpleNamespace(strategy_max_new_tokens=131072,
                                    strategy_temperature=.6, strategy_top_p=.95),
                strategies_per_parent=2, strategy_cfg=object(), model=object(),
                tokenizer=object(), cap_state={"value": 0}, make_progress_bar=Bar,
                _seed_local_generation=lambda *a: None,
                backend=SimpleNamespace(disable_adapter=nullcontext)))
            seen = []
            def generate(model, tokenizer, prompts, counts, cfg, **kw):
                seen.append(prompts[0])
                if len(seen) <= 3:
                    yield 0, [("bad", [])]
                else:
                    if len(seen) == 5:
                        self.assertIn("first accepted", prompts[0])
                        self.assertNotIn("bad", prompts[0])
                    yield 0, [("<strategy>first accepted</strategy>", [])]
            s["generate_prompt_jobs"] = generate
            exec(compile(ast.Module(body=[block], type_ignores=[]), "local_strategy", "exec"), s)
            self.assertEqual(len(seen), 5)
            self.assertEqual(len(s["strategy_chains"][0]["strategies"]), 2)


class StrategySchedulerTests(unittest.TestCase):
    def scheduler(self):
        scope = dict(
            deque=deque, queue=queue, make_progress_bar=Bar,
            terminal_log_only=lambda *args, **kwargs: None,
        )
        functions_from_file("gen_workers.py", {"run_sequential_chains"}, scope)
        results = queue.Queue()
        dispatched = []
        class TaskQueue:
            def __init__(self, rank):
                self.rank = rank
            def put(self, task):
                dispatched.append(task)
                _, _, jobs, _ = task
                chain, prompt, count = jobs[0]
                results.put((self.rank, chain, [(prompt, [])]))
        pool = SimpleNamespace(
            num_workers=2, tensor_parallel_size=1, pipeline_parallel_size=1,
            gen_micro_batch=1, task_queues=[TaskQueue(0), TaskQueue(1)],
            result_queue=results, procs=[SimpleNamespace(exitcode=None)] * 2)
        return scope["run_sequential_chains"], pool, dispatched

    def test_pipeline_retries_three_times_before_unlocking_dependents(self):
        run, pool, dispatched = self.scheduler()
        accepted = set()
        attempts = []
        def prompt(chain, stage, attempt):
            if stage:
                self.assertIn((chain, stage - 1), accepted)
            attempts.append((chain, stage, attempt))
            return f"{chain}/{stage}/{attempt}"
        def handle(chain, stage, response, attempt):
            if chain == stage == 0 and attempt < 3:
                return True
            accepted.add((chain, stage))
            return False
        with redirect_stdout(io.StringIO()):
            result = run(pool, num_chains=2, num_stages=2, max_retries=3,
                         prompt_builder=prompt, result_handler=handle, adapter_path=None,
                         max_new_tokens=131072, temperature=.6, top_p=.95)
        self.assertEqual(result, dict(completed=4, retries=3))
        self.assertEqual([a for c, s, a in attempts if c == s == 0], [0, 1, 2, 3])
        tasks = [t for t in dispatched if t[2][0][1].startswith("0/0/")]
        self.assertEqual(len({t[0] for t in tasks}), 4)
        self.assertTrue(all(t[3]["max_new_tokens"] == 131072 for t in tasks))

    def test_exhaustion_does_not_advance_chain(self):
        run, pool, dispatched = self.scheduler()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exhausted 3"):
            run(pool, num_chains=1, num_stages=2, max_retries=3,
                prompt_builder=lambda c, s, a: str(a), result_handler=lambda *args: True,
                adapter_path=None, max_new_tokens=131072, temperature=.6, top_p=.95)
        self.assertEqual(len(dispatched), 4)

    def test_phased_pool_exhaustion_stops_workers_without_sleep(self):
        tree = ast.parse((ROOT / "gen_workers.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "PhasedVLLMGenerationPool")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "run_sequential_chains")
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pool_wrapper", "exec"), scope)
        events = []
        def fail(**kw):
            raise RuntimeError("retry limit")
        pool = SimpleNamespace(run_sequential_chains=fail,
                               shutdown=lambda: events.append("shutdown"))
        wrapper = SimpleNamespace(_pool=pool, _awake=True, _persistent=True,
                                  _ensure_started=lambda: pool,
                                  _after_stop=lambda: events.append("after_stop"))
        with self.assertRaisesRegex(RuntimeError, "retry limit"):
            scope["run_sequential_chains"](wrapper)
        self.assertEqual(events, ["shutdown", "after_stop"])
        self.assertIsNone(wrapper._pool)
        self.assertFalse(wrapper._awake)


if __name__ == "__main__":
    unittest.main()
