"""Phase-specific coder prompt routing; no models or GPU packages required."""

import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from output_retries import coder_retry_prompt_job


def _runtime_scope(kind="qwen3.8", backend="vllm"):
    source = (Path(__file__).resolve().parents[1] / "train_multy_CVaR.py").read_text()
    tree = ast.parse(source)
    names = {"_coder_effort_for_rollout_phase",
             "_qwen38_pilot_effort_entries", "_phase_coder_prompt_job",
             "_render", "_run_vllm_code_phase", "_run_local_code_phase"}
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
    cfg = SimpleNamespace(
        coder_template_kind=kind, coder_reasoning_effort="medium",
        coder_preserve_thinking=True, thinking=True,
        generation_backend=backend, max_new_tokens=64, temperature=.6,
        top_p=.95, sampling_top_k=0, sampling_min_p=0.0)
    rendered = []
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            rendered.append(dict(options))
            return f"{options.get('reasoning_effort', 'generic')}:{messages!r}"
    scope = {
        "cfg": cfg, "tokenizer": Tokenizer(),
        "random": random,
        "_coder_messages_for_template": lambda messages, _kind: messages,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "phase_functions", "exec"),
         scope)
    return scope, rendered


class PhaseReasoningTests(unittest.TestCase):
    def test_native_template_receives_medium_pilot_and_phase2(self):
        scope, calls = _runtime_scope()
        messages = [{"role": "user", "content": "task and strategy"}]
        for phase, effort in [(None, "medium"), ("pilot", "medium"),
                              ("adaptive", "medium")]:
            scope["_render"](messages, rollout_phase=phase)
            self.assertEqual(calls[-1]["reasoning_effort"], effort)
            self.assertTrue(calls[-1]["enable_thinking"])
            self.assertTrue(calls[-1]["preserve_thinking"])
        self.assertEqual(scope["cfg"].coder_reasoning_effort, "medium")

    def test_other_coders_keep_original_template_controls(self):
        for kind in ["generic", "qwen3", "qwen-thinking", "gpt-oss"]:
            with self.subTest(kind=kind):
                scope, calls = _runtime_scope(kind)
                messages = [{"role": "user", "content": "task"}]
                pilot = scope["_render"](messages, rollout_phase="pilot")
                adaptive = scope["_render"](messages, rollout_phase="adaptive")
                self.assertEqual(pilot, adaptive)
                self.assertEqual(calls[0], calls[1])
                jobs = [dict(messages=messages, prompt_text=pilot, count=10)]
                self.assertEqual(scope["_phase_coder_prompt_job"](
                    jobs, 0, "adaptive", scope["cfg"], scope["_render"], {}), 0)
                self.assertEqual(len(jobs), 1)

    def test_variants_keep_pilot_prompts_intact_and_are_cached(self):
        scope, calls = _runtime_scope()
        scope["cfg"].coder_reasoning_effort = "low"
        messages = [{"role": "user", "content": "plan"}]
        jobs = [dict(messages=messages, prompt_text=scope["_render"](messages),
                     parent_group=2, strategy_index=1, assigned_fold_index=3,
                     count=14, coder_reasoning_effort="low")]
        original = copy.deepcopy(jobs[0])
        cache = {}
        index = scope["_phase_coder_prompt_job"](
            jobs, 0, "adaptive", scope["cfg"], scope["_render"], cache)
        self.assertEqual(index, 1)
        self.assertEqual(jobs[0], original)
        self.assertEqual(jobs[index]["coder_reasoning_effort"], "medium")
        self.assertEqual(jobs[index]["count"], 0)
        self.assertEqual(jobs[index]["parent_group"], 2)
        self.assertEqual(jobs[index]["strategy_index"], 1)
        self.assertEqual(jobs[index]["assigned_fold_index"], 3)
        count = len(calls)
        self.assertEqual(scope["_phase_coder_prompt_job"](
            jobs, 0, "adaptive", scope["cfg"], scope["_render"], cache), index)
        self.assertEqual(len(calls), count)
        self.assertEqual(scope["_phase_coder_prompt_job"](
            jobs, 0, "pilot", scope["cfg"], scope["_render"], cache), index)

    def test_base_effort_is_preserved_outside_pilots_and_phase2(self):
        scope, _ = _runtime_scope()
        effort = scope["_coder_effort_for_rollout_phase"]
        scope["cfg"].coder_reasoning_effort = "low"
        self.assertEqual(effort(scope["cfg"]), "low")
        self.assertEqual(effort(scope["cfg"], "pilot"), "medium")
        self.assertEqual(effort(scope["cfg"], "adaptive"), "medium")
        scope["cfg"].coder_reasoning_effort = "medium"
        jobs = [dict(prompt_text="already medium")]
        self.assertEqual(scope["_phase_coder_prompt_job"](
            jobs, 0, "adaptive", scope["cfg"], scope["_render"], {}), 0)

    def test_both_generation_backends_retain_correct_prompt_ids_for_training(self):
        for backend in ["vllm", "hf"]:
            with self.subTest(backend=backend):
                scope, calls = _runtime_scope(backend=backend)
                jobs = []
                for index in range(2):
                    messages = [{"role": "user", "content": f"strategy {index}"}]
                    jobs.append(dict(messages=messages,
                                     prompt_text=scope["_render"](messages),
                                     coder_reasoning_effort="medium", count=4,
                                     parent_group=0, strategy_index=index,
                                     assigned_fold_index=0))
                originals = copy.deepcopy(jobs)
                batches = []
                def generate(prompts, counts, with_logprobs=False):
                    batches.append((list(prompts), list(counts)))
                    # Deliberately return completion groups out of order.
                    for i in reversed(range(len(prompts))):
                        values = [(prompts[i], [4, 5], [-.4, -.5])
                                  for _ in range(counts[i])]
                        yield i, values if with_logprobs else [v[:2] for v in values]
                def remote_generate(**kwargs):
                    self.assertEqual(kwargs["temperature"], .6)
                    self.assertEqual(kwargs["max_new_tokens"], 64)
                    self.assertEqual(kwargs["adapter_path"], "current_adapter")
                    yield from generate(kwargs["prompts_by_group"],
                                        kwargs["counts_by_group"], True)
                def local_generate(_model, _tokenizer, prompts, counts, cfg, **kw):
                    self.assertEqual(cfg.coder_reasoning_effort, "medium")
                    yield from generate(prompts, counts)
                def queue(job_idx, text, tokens, logprobs=None, **kwargs):
                    # The production saver, scoring cache, and trainer all
                    # resolve prompt_jobs[record['job_idx']] in this same way.
                    self.assertEqual(text, jobs[job_idx]["prompt_text"])
                    return dict(job_idx=job_idx, text=text, token_ids=tokens,
                                behavior_logprobs=logprobs, **kwargs)
                scope.update({
                    "source_job_indices": [0, 1], "prompt_jobs": jobs,
                    "phase_prompt_job_cache": {}, "adapter_path": "current_adapter",
                    "step_idx": 0, "training_enabled": True,
                    "_uses_sequence_level_policy_ratio": lambda cfg: False,
                    "gen_pool": SimpleNamespace(iter_group_jobs=remote_generate),
                    "_queue_rollout": queue,
                    # Format retries have their own integration tests; this
                    # fixture tests routing of the original planned samples.
                    "_run_coder_format_retries": lambda *a, **k: None,
                    "_attach_strategy_plan": lambda *args: None,
                    "generate_prompt_jobs": local_generate, "model": object(),
                    "_seed_local_generation": lambda *args: None, "cap_state": {},
                    "make_progress_bar": lambda *a, **k: SimpleNamespace(
                        update=lambda n: None, close=lambda: None),
                })
                run = scope["_run_vllm_code_phase" if backend == "vllm"
                            else "_run_local_code_phase"]
                pilots = run([10, 10], phase="pilot", seed_offset=0, progress_desc="pilot")
                followups = run([0, 3], phase="adaptive", seed_offset=4000000,
                                progress_desc="adaptive")
                self.assertEqual(len(pilots) + len(followups), 23)
                self.assertEqual(jobs[:2], originals)
                self.assertEqual(batches[0][1], [1, 1, 9, 9])
                self.assertEqual(batches[1][1], [3])
                self.assertEqual(len(jobs), 4)
                for source_idx in (0, 1):
                    records = [r for r in pilots if r["strategy_source_job_idx"] == source_idx]
                    efforts = [jobs[r["job_idx"]]["coder_reasoning_effort"] for r in records]
                    self.assertEqual(efforts.count("medium"), 9)
                    self.assertEqual(efforts.count("xhigh"), 1)
                    self.assertTrue(all(r["rollout_phase"] == "pilot" for r in records))
                    for record in records:
                        job = jobs[record["job_idx"]]
                        self.assertTrue(job["prompt_text"].startswith(job["coder_reasoning_effort"] + ":"))
                        if backend == "vllm":
                            self.assertEqual(record["behavior_logprobs"], [-.4, -.5])
                for record in followups:
                    self.assertTrue(jobs[record["job_idx"]]["prompt_text"].startswith("medium:"))
                    self.assertEqual(record["job_idx"], 1)
                    self.assertEqual(record["strategy_source_job_idx"], 1)
                    if backend == "vllm":
                        self.assertEqual(record["behavior_logprobs"], [-.4, -.5])
                again = run([0, 1], phase="adaptive", seed_offset=4100000,
                            progress_desc="adaptive")
                self.assertEqual(again[0]["job_idx"], 1)
                self.assertEqual(len(jobs), 4)

    def test_xhigh_selection_is_stable_per_step_and_changes_no_budget(self):
        scope, _ = _runtime_scope()
        jobs = []
        for parent in range(3):
            for strategy in range(5):
                jobs.append(dict(parent_group=parent, strategy_index=strategy))
        indices = list(range(len(jobs)))
        counts = [10] * len(jobs)
        split = scope["_qwen38_pilot_effort_entries"](
            jobs, indices, counts, scope["cfg"], 7, "pilot")
        self.assertEqual(sum(count for _, count, _ in split), sum(counts))
        self.assertEqual(sum(effort == "xhigh" for _, _, effort in split), 6)
        self.assertEqual(
            split,
            scope["_qwen38_pilot_effort_entries"](
                jobs, indices, counts, scope["cfg"], 7, "pilot"))
        for parent in range(3):
            selected = {
                jobs[source]["strategy_index"]
                for source, count, effort in split
                if jobs[source]["parent_group"] == parent
                and effort == "xhigh" and count == 1
            }
            self.assertEqual(len(selected), 2)
        adaptive = scope["_qwen38_pilot_effort_entries"](
            jobs, indices, counts, scope["cfg"], 7, "adaptive")
        self.assertTrue(all(effort is None for _, _, effort in adaptive))

    def test_retry_remains_medium_and_in_pilot_allocation(self):
        scope, calls = _runtime_scope()
        messages = [{"role": "user", "content": "parent and strategy"}]
        jobs = [dict(messages=messages, prompt_text=scope["_render"](messages),
                     coder_reasoning_effort="medium", count=10,
                     parent_group=3, strategy_index=2, assigned_fold_index=1)]
        index = scope["_phase_coder_prompt_job"](
            jobs, 0, "pilot", scope["cfg"], scope["_render"], {},
            effort="xhigh")
        self.assertEqual(jobs[index]["coder_reasoning_effort"], "xhigh")
        record = dict(job_idx=index, strategy_rollout_phase="pilot", strategy_source_job_idx=0)
        frozen = copy.deepcopy(jobs)
        cache = {}
        retry = coder_retry_prompt_job(
            jobs, record, scope["_render"], cache,
            reasoning_effort="medium")
        self.assertEqual(calls[-1]["reasoning_effort"], "medium")
        self.assertEqual(jobs[retry]["coder_reasoning_effort"], "medium")
        self.assertEqual(jobs[retry]["count"], 0)
        self.assertEqual(jobs[retry]["parent_group"], 3)
        self.assertEqual(jobs[retry]["strategy_index"], 2)
        self.assertEqual(jobs[retry]["assigned_fold_index"], 1)
        self.assertEqual(jobs[0], frozen[0])
        self.assertIn("Your previous attempt", jobs[retry]["prompt_text"])
        self.assertEqual(coder_retry_prompt_job(
            jobs, record, scope["_render"], cache,
            reasoning_effort="medium"), retry)


if __name__ == "__main__":
    unittest.main()
