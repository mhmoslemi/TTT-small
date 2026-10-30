"""Native-boundary regressions without importing model, NumPy, or GPU packages."""

import ast
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
import __future__
import math
from pathlib import Path
from queue import Queue
import re
from types import SimpleNamespace
import unittest

from generation.output_retries import coder_output_issue


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), scope)


class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = {"<think>": 901, "</think>": 907} if vocab is None else vocab
        self.pieces = {value: key for key, value in self.vocab.items()}

    def get_added_vocab(self):
        return self.vocab

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[value] for value in ids)

    def encode(self, *args, **kwargs):
        raise AssertionError("generated text must never be re-encoded to infer a boundary")


class ImmediatePool:
    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future


class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = {"re": re, "_canonical_generated_text": lambda tokenizer, ids, text: text}
        load_functions("generation/gen_workers.py", {"native_final_answer_scope"}, cls.scope)
        load_functions("evaluation/reward.py", {"extract_python_code", "_extract_code_from_scope"}, cls.scope)

    def final(self, tokenizer, ids, required=True):
        return self.scope["native_final_answer_scope"](tokenizer, ids, require_final_marker=required)

    def test_hidden_native_boundary_selects_final_program_without_changing_ids(self):
        tokenizer = Tokenizer()
        tokenizer.pieces.update({1: "```python\nold_draft()\n```", 2: "```python\nfinal_program()\n```"})
        ids = [1, 907, 2]
        raw = tokenizer.decode([1, 2])  # vLLM may omit the special delimiter.
        final = self.final(tokenizer, ids)
        self.assertEqual(final, tokenizer.pieces[2])
        self.assertIsNone(coder_output_issue(raw, require_final_marker=True, final_answer_text=final))
        self.assertEqual(self.scope["extract_python_code"](raw, final_answer_text=final), "final_program()")
        self.assertEqual(ids, [1, 907, 2])
        self.assertEqual(raw, tokenizer.decode([1, 2]))

    def test_ordinary_tokens_spelling_a_boundary_cannot_end_reasoning(self):
        tokenizer = Tokenizer()
        tokenizer.pieces[1] = 'The example "</think>" is not a native token.\n```python\ndraft()\n```'
        final = self.final(tokenizer, [1])
        self.assertEqual(final, "")
        self.assertEqual(coder_output_issue(tokenizer.pieces[1], final_answer_text=final), "missing final Python block")
        self.assertIsNone(self.scope["extract_python_code"](tokenizer.pieces[1], final_answer_text=final))

    def test_final_program_delimiters_are_not_rescanned(self):
        tokenizer = Tokenizer()
        tokenizer.pieces.update({1: '```python\ntext = "', 2: '"\nother = "<think>"\n```'})
        final = self.final(tokenizer, [907, 1, 907, 2])
        self.assertEqual(final, '```python\ntext = "</think>"\nother = "<think>"\n```')
        self.assertIsNone(coder_output_issue(final, final_answer_text=final))
        self.assertEqual(self.scope["extract_python_code"](final, final_answer_text=final),
                         'text = "</think>"\nother = "<think>"')

    def test_missing_or_empty_final_and_unclosed_fences_still_fail(self):
        tokenizer = Tokenizer()
        tokenizer.pieces.update({1: "```python\ndraft()\n```", 2: "```python\nunfinished()"})
        self.assertEqual(self.final(tokenizer, [901, 1], required=False), "")
        self.assertEqual(self.scope["native_final_answer_scope"](
            tokenizer, [1], prompt_text="assistant\n<think>\n"), "")
        self.assertEqual(self.final(tokenizer, [907]), "")
        final = self.final(tokenizer, [907, 2])
        self.assertEqual(coder_output_issue(final, final_answer_text=final), "unclosed final Python block")

    def test_disabled_reasoning_and_legacy_tokenizers_retain_direct_answers(self):
        tokenizer = Tokenizer()
        tokenizer.pieces[1] = '```python\nvalue = "</think>"\n```'
        self.assertEqual(self.final(tokenizer, [1], required=False), tokenizer.pieces[1])
        self.assertIsNone(self.final(Tokenizer({}), [1], required=False))
        self.assertEqual(self.scope["extract_python_code"]("```python\nlegacy()\n```"), "legacy()")
        with self.assertRaisesRegex(RuntimeError, "native reasoning boundary"):
            self.final(Tokenizer({}), [1])

    def test_harmony_requires_real_channel_tokens_not_assistantfinal_text(self):
        tokenizer = Tokenizer({"<|channel|>": 71, "<|message|>": 73, "</think>": 907})
        tokenizer.pieces.update({1: "analysis", 2: "final", 3: "assistantfinal ```python\ndraft()\n```",
                                 4: "```python\nanswer()\n```"})
        self.assertEqual(self.final(tokenizer, [71, 1, 73, 3]), "")
        self.assertEqual(self.final(tokenizer, [71, 1, 73, 907, 4]), "")
        self.assertEqual(self.final(tokenizer, [71, 1, 73, 3, 71, 2, 73, 4]), tokenizer.pieces[4])

    def pipeline_scope(self, isolated):
        scope = dict(self.scope)
        evaluated, saved = [], []
        tokenizer = Tokenizer()
        tokenizer.pieces.update({1: "unfinished reasoning", 2: '```python\nvalue = "</think>"\n```'})

        def result(**kwargs):
            return SimpleNamespace(**kwargs)

        load_functions("problems/base.py", {"compute_reward"}, scope)
        scope.update(RewardResult=result, run_code=lambda code, **kwargs: (
            evaluated.append((code, kwargs)) or {"ok": False, "error": "synthetic stop"}))
        problem = SimpleNamespace(require_final_code_marker=True, fail_score=0., entrypoint="run", eval_cpus=1,
                                  preprocess=lambda code, parent: code)
        problem.compute_reward = scope["compute_reward"].__get__(problem)
        slots = Queue()
        slots.put(3)
        scope.update(
            tokenizer=tokenizer, problem=problem, cfg=SimpleNamespace(sandbox_timeout_s=1, generation_backend="hf"),
            math=math,
            prompt_jobs=[{"parent_group": 0, "prompt_text": "original prompt"}], parent_ctxs=[object()],
            group_responses=defaultdict(list), reward_futures=defaultdict(list),
            queued_by_parent=defaultdict(int), queued_by_group=defaultdict(int), planned_by_group=defaultdict(int),
            primary_rollout_records=[], coder_retry_records=[], rollout_io_futures=[],
            reward_pool=ImmediatePool(), rollout_io_pool=ImmediatePool(), Future=Future,
            coder_output_issue=coder_output_issue, retry_metadata=lambda record: {},
            entropy_directory=None, two_stage_rollouts=False, training_enabled=False,
            defer_gpu_evaluation=False, isolated_eval=isolated, exp_dir="unused", step_idx=0,
            save_rollout_artifacts=lambda *args, **kwargs: saved.append((args, kwargs)),
            _record_evaluation_completion=lambda future: None,
            host_admission=SimpleNamespace(reservation=lambda size: nullcontext()),
            isolated_memory_limit_bytes=1024, isolated_cpu_slots=slots)
        load_functions("train_multy_CVaR.py", {"_queue_rollout", "_submit_rollout", "_run_isolated_reward"}, scope)
        return scope, evaluated, saved

    def test_streamed_rollouts_and_both_evaluation_paths_share_native_scope(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                scope, evaluated, saved = self.pipeline_scope(isolated)
                raw = 'unfinished reasoning ```python\nvalue = "</think>"\n```'
                ids, logprobs = [1, 907, 2], [-1., -2., -3.]
                record = scope["_queue_rollout"](0, raw, ids, logprobs)
                self.assertIsNone(record["output_format_issue"])
                self.assertEqual(evaluated[0][0], 'value = "</think>"')
                self.assertEqual(saved[0][0][-1], raw)
                self.assertEqual(saved[0][1]["response_token_ids"], ids)
                self.assertEqual(record["behavior_logprobs"], logprobs)
                scope["_queue_rollout"](0, raw, [1, 2])
                self.assertEqual(len(evaluated), 1)  # Reasoning drafts never reach the sandbox.

    def test_pending_recovery_rechecks_saved_format_issue_using_exact_ids(self):
        scope, evaluated, _ = self.pipeline_scope(False)
        scope.update(parent_contexts=[object()], isolated_eval=False)
        parent = next(node for node in ast.parse((ROOT / "train_multy_CVaR.py").read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_finalize_pending_spo_rs_artifacts")
        evaluate = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        exec(compile(ast.Module(body=[evaluate], type_ignores=[]), "pending recovery", "exec"), scope)
        item = {"meta": {"parent_group": 0, "output_format_issue": "missing final Python block"},
                "response_path": SimpleNamespace(read_text=lambda **kwargs: "raw response missing visible boundary"),
                "prompt_path": SimpleNamespace(read_text=lambda **kwargs: "original prompt"),
                "response_token_ids": [1, 907, 2]}
        scope["evaluate"](item)
        self.assertIsNone(item["meta"]["output_format_issue"])
        self.assertEqual(evaluated[0][0], 'value = "</think>"')


if __name__ == "__main__":
    unittest.main()
