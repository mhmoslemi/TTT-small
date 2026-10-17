"""CPU diagnostics tests; numerical PyTorch checks run when torch is present."""

import ast
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import entropy_tools as entropy


class EntropyToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def observation(self, group=0, rollout=0, strategy=0, *, phase="pilot",
                    valid=True, tokens=3, code="def run():\n    return 1\n"):
        directory = self.root / "measurements"
        descriptor = entropy.observation_descriptor(directory, group, rollout)
        meta = dict(group=group, rollout=rollout, strategy_index=strategy,
                    strategy_rollout_phase=phase, valid=valid,
                    reward=2.0 if valid else 0.0, raw_score=.4 if valid else None,
                    n_response_tokens=tokens, failure_kind=None)
        return entropy.rollout_observation(descriptor, meta, code)

    def measured(self, observation, total):
        entropy._atomic_text(observation["_measurement"]["path"], entropy._json({
            "measured": True, "entropy_sum_nats": total,
            "token_count": observation["response_tokens"], "reason": None,
        }))

    def test_flag_off_and_reference_scoring_do_not_request_entropy(self):
        calls = []
        @entropy.measure_policy_entropy
        def score(model, examples, with_grad, chunk, **kwargs):
            calls.append(kwargs["return_entropy"])
            return ["logprobs"]
        self.assertEqual(score(None, [{}], True), ["logprobs"])
        tagged = {"_entropy_measurement": self.observation()["_measurement"]}
        self.assertEqual(score(None, [tagged], False), ["logprobs"])
        score(None, [tagged], True, measure_entropy=False)
        self.assertEqual(calls, [False, False, False])
        self.assertFalse((self.root / "measurements").exists())

    def test_retry_counts_first_completed_forward_only(self):
        calls, recorded = [], []
        descriptor = self.observation()["_measurement"]
        examples = [{"_entropy_measurement": descriptor}]
        @entropy.measure_policy_entropy
        def score(model, examples, with_grad, chunk, **kwargs):
            mode = kwargs["return_entropy"]
            calls.append(mode)
            return (["lp"], ["entropy"]) if mode else ["lp"]
        def record(desc, values):
            recorded.append(values)
            entropy._atomic_text(desc["path"], "{}")
        with patch.object(entropy, "_record_tensor", record):
            self.assertEqual(score(None, examples, True), ["lp"])
            # A backward OOM followed by another forward on the same sample.
            self.assertEqual(score(None, examples, True), ["lp"])
        self.assertEqual(calls, ["measure", False])
        self.assertEqual(recorded, ["entropy"])

    def test_failed_forward_writes_nothing_and_can_be_retried(self):
        descriptor = self.observation()["_measurement"]
        @entropy.measure_policy_entropy
        def score(*args, **kwargs):
            raise RuntimeError("simulated forward OOM")
        with self.assertRaisesRegex(RuntimeError, "simulated forward"):
            score(None, [{"_entropy_measurement": descriptor}], True)
        self.assertFalse(Path(descriptor["path"]).exists())

    def test_regularizer_entropy_return_is_unchanged(self):
        result = (["lp"], ["differentiable entropy"])
        @entropy.measure_policy_entropy
        def score(*args, **kwargs):
            self.assertIs(kwargs["return_entropy"], True)
            return result
        with patch.object(entropy, "_record_tensor"):
            actual = score(None, [{"_entropy_measurement":
                                   self.observation()["_measurement"]}], True,
                           return_entropy=True)
        self.assertIs(actual, result)

    def test_resume_attempt_does_not_reuse_previous_step_observations(self):
        first = entropy.begin_step(self.root, 0)
        second = entropy.begin_step(self.root, 0)
        self.assertNotEqual(first, second)
        self.assertNotEqual(entropy.observation_descriptor(first, 0, 0),
                            entropy.observation_descriptor(second, 0, 0))

    def test_token_weighted_entropy_failed_rollouts_and_missing_coverage(self):
        first = self.observation(rollout=0, tokens=2)
        failed = self.observation(rollout=1, tokens=8, valid=False)
        missing = self.observation(rollout=2, tokens=10)
        self.measured(first, 2.0)
        self.measured(failed, 4.0)
        summary = entropy.save_step(self.root, 0, [first, failed, missing],
                                    SimpleNamespace(strategies=True), {})
        self.assertAlmostEqual(summary["all"]["entropy_nats"], .6)
        self.assertAlmostEqual(summary["all"]["token_coverage"], .5)
        self.assertAlmostEqual(summary["all"]["valid_fraction"], 2/3)
        self.assertEqual(summary["all"]["measured_rollouts"], 2)
        self.assertEqual(summary["all"]["measured_tokens"], 10)
        self.assertAlmostEqual(summary["by_strategy"]["0"]["entropy_nats"], .6)
        self.assertEqual(summary["pilot"], summary["all"])
        samples = [json.loads(line) for line in (
            self.root / "step00/entropy_samples.jsonl").read_text().splitlines()]
        self.assertFalse(samples[2]["measured"])
        self.assertIsNone(samples[2]["entropy_sum_nats"])

    def test_count_mismatch_excluded_rather_than_biasing_average(self):
        row = self.observation(tokens=4)
        self.measured(row, 4.0)
        row["response_tokens"] = 5
        summary = entropy.save_step(
            self.root, 0, [row], SimpleNamespace(strategies=True), {})
        self.assertIsNone(summary["all"]["entropy_nats"])
        self.assertEqual(summary["all"]["measured_rollouts"], 0)

    def test_json_history_resume_upsert_and_svg(self):
        cfg = SimpleNamespace(strategies=True, strategies_per_parent=1)
        for step in [0, 1, 1, 2]:
            row = self.observation()
            self.measured(row, float(step + 1))
            entropy.save_step(self.root, step, [row], cfg, {})
        history = [json.loads(line) for line in
                   (self.root / "entropy.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in history], [0, 1, 2])
        self.assertTrue((self.root / "entropy.pdf").read_bytes().startswith(b"%PDF"))
        ET.parse(self.root / "strategy_diversity.svg")
        entropy.save_step(self.root, 1, [], cfg, {})
        history = [json.loads(line) for line in
                   (self.root / "entropy.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in history], [0, 1])
        self.assertIsNone(history[-1]["all"]["entropy_nats"])

    def test_missing_entropy_is_a_gap_not_a_zero_curve(self):
        summary = entropy.save_step(
            self.root, 0, [], SimpleNamespace(strategies=False), {})
        self.assertIsNone(summary["all"]["entropy_nats"])
        self.assertTrue((self.root / "entropy.pdf").read_bytes().startswith(b"%PDF"))

    def test_direct_run_has_no_strategy_only_metadata_or_artifact(self):
        row = self.observation(strategy=None, phase="ordinary")
        self.measured(row, 1.5)
        cfg = SimpleNamespace(
            strategies=False,
            strategy_model_name="must-not-leak-into-direct-run",
            model_name="Qwen/Qwen3-8B",
        )
        with patch.object(entropy, "_entropy_pdf") as entropy_pdf, \
                patch.object(entropy, "_line_plot") as strategy_plot:
            summary = entropy.save_step(self.root, 0, [row], cfg, {})
        entropy_pdf.assert_called_once()
        strategy_plot.assert_not_called()
        self.assertFalse((self.root / "strategy_diversity.svg").exists())
        self.assertFalse(summary["strategies_enabled"])
        self.assertIsNone(summary["strategy_model"])
        self.assertEqual(summary["by_strategy"], {})
        self.assertEqual(summary["by_parent_strategy"], {})
        self.assertIsNone(summary["strategy_entropy_aggregation"])
        self.assertIsNone(summary["pilot"])
        self.assertIsNone(summary["code_diversity"])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_direct_entropy_pdf_uses_one_full_width_axis(self):
        import matplotlib.pyplot as plt

        closed = []
        real_close = plt.close
        rows = [{"step": 0, "all": {"entropy_nats": 0.7},
                 "by_strategy": {}}]
        with patch.object(plt, "close", side_effect=closed.append):
            entropy._entropy_pdf(self.root / "entropy.pdf", rows)
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0].axes), 1)
        self.assertTrue((self.root / "entropy.pdf").read_bytes().startswith(b"%PDF"))
        real_close(closed[0])

    def test_strategy_entropy_is_saved_both_pooled_and_by_parent(self):
        rows = [self.observation(group=0, rollout=0, strategy=0, tokens=2),
                self.observation(group=1, rollout=0, strategy=0, tokens=3),
                self.observation(group=0, rollout=1, strategy=1, tokens=4)]
        for row, total in zip(rows, (1.0, 3.0, 8.0)):
            self.measured(row, total)
        summary = entropy.save_step(
            self.root, 0, rows,
            SimpleNamespace(strategies=True, strategies_per_parent=2), {})
        self.assertAlmostEqual(
            summary["by_strategy"]["0"]["entropy_nats"], 4.0 / 5.0)
        self.assertAlmostEqual(
            summary["by_strategy"]["1"]["entropy_nats"], 2.0)
        self.assertEqual(
            sorted(summary["by_parent_strategy"]), ["p0:s0", "p0:s1", "p1:s0"])

    def test_pilots_used_for_diversity_and_contexts_not_mixed(self):
        rows = [self.observation(rollout=0, strategy=0),
                self.observation(rollout=1, strategy=0),
                self.observation(rollout=2, strategy=1, code="x = 500\n"),
                self.observation(rollout=3, strategy=0, phase="phase2"),
                self.observation(group=1, rollout=0, strategy=0),
                self.observation(rollout=4, strategy=1, valid=False)]
        before = random.getstate()
        result = entropy.code_diversity(rows)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(result["population"], "valid_pilot_programs")
        self.assertEqual(result["valid_programs_compared"], 4)
        self.assertEqual(result["within_strategy"]["sampled_pairs"], 1)
        self.assertEqual(result["within_strategy"]["distance"], 0)
        self.assertEqual(result["across_strategies"]["sampled_pairs"], 2)
        self.assertGreater(result["across_strategies"]["distance"], .5)

    def test_code_signatures_ignore_formatting_and_comments(self):
        a = entropy._code_signature("def f():\n    return 2  # hi\n")
        b = entropy._code_signature("def f( ):\n  return 2\n")
        self.assertEqual(a, b)
        self.assertIsNone(entropy._code_signature(""))
        self.assertIsNone(entropy._code_signature('x = """'))

    def test_deterministic_bounded_pair_sampling(self):
        rows = [self.observation(rollout=i, strategy=i % 3,
                                 code=f"def f():\n return {i}\n")
                for i in range(30)]
        a = entropy.code_diversity(rows, max_pairs_per_parent=7)
        self.assertEqual(a, entropy.code_diversity(rows, max_pairs_per_parent=7))
        self.assertEqual(a["within_strategy"]["sampled_pairs"], 7)
        self.assertEqual(a["across_strategies"]["sampled_pairs"], 7)

    def test_training_integration_descriptors_and_epoch_guard(self):
        source = (Path(__file__).resolve().parents[1]
                  / "train_multy_CVaR.py").read_text()
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        for name in ["_prepare_binary_overlap_examples",
                     "_prepare_entropic_overlap_group", "train_step"]:
            self.assertIn('"_entropy_measurement"',
                          ast.get_source_segment(source, functions[name]))
        rank = ast.get_source_segment(source, functions["_train_rank_examples"])
        self.assertIn("measure_entropy=(epoch == 0)", rank)
        parser = ast.get_source_segment(source, functions["_build_arg_parser"])
        self.assertIn('"--measure-entropy"', parser)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class TorchEntropyTests(unittest.TestCase):
    def test_exact_entropy_and_detachment(self):
        import torch
        lp = torch.tensor([[.25, .25, .25, .25], [.9, .1, 0, 0]]).log()
        lp.requires_grad_()
        result = entropy.token_entropy(lp, detached=True)
        self.assertFalse(result.requires_grad)
        self.assertAlmostEqual(float(result[0]), math.log(4), places=6)
        self.assertAlmostEqual(float(result[1]), -.9*math.log(.9)-.1*math.log(.1),
                               places=6)
        self.assertTrue(entropy.token_entropy(lp).requires_grad)

    def _scorers(self):
        # Execute real scorer definitions without importing the GPU runner and
        # its unrelated optional dependencies. Use a tiny full-forward model.
        source = (Path(__file__).resolve().parents[1]
                  / "train_multy_CVaR.py").read_text()
        names = {"_score_hidden_token_chunks", "compute_token_logprobs",
                 "compute_batched_token_logprobs"}
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        scope = {"_token_entropy": entropy.token_entropy,
                 "measure_policy_entropy": entropy.measure_policy_entropy,
                 "_decoder_token_logprobs": lambda *a, **k: None,
                 "_shared_prefix_token_logprobs": lambda *a, **k: None}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "scorers", "exec"),
             scope)
        return scope

    def test_checkpointed_projection_gradients_identical(self):
        import torch
        scorer = self._scorers()["_score_hidden_token_chunks"]
        torch.manual_seed(13)
        head = torch.nn.Linear(5, 11)
        hidden = torch.randn(9, 5, requires_grad=True)
        targets = torch.arange(9)
        baseline = scorer(hidden, targets, head, chunk=3, with_grad=True,
                          return_entropy=False)
        baseline.sum().backward()
        expected = [hidden.grad.clone(), head.weight.grad.clone()]
        hidden.grad = None
        head.zero_grad()
        actual, ent = scorer(hidden, targets, head, chunk=3, with_grad=True,
                             return_entropy="measure")
        actual.sum().backward()
        self.assertTrue(torch.equal(actual, baseline))
        self.assertTrue(torch.equal(hidden.grad, expected[0]))
        self.assertTrue(torch.equal(head.weight.grad, expected[1]))
        self.assertFalse(ent.requires_grad)

    def test_batched_padding_response_only_and_gradients_unchanged(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 5)
                self.head = torch.nn.Linear(5, 16)
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.head(self.embed(input_ids)))
        scorer = self._scorers()["compute_batched_token_logprobs"]
        torch.manual_seed(11)
        model = Model()
        examples = [dict(prompt_ids=torch.tensor([[1, 2]]),
                         response_ids=torch.tensor([[3, 4, 5]])),
                    dict(prompt_ids=torch.tensor([[1, 2, 3]]),
                         response_ids=torch.tensor([[6]]))]
        baseline = scorer(model, examples, True, chunk=2)
        sum(x.sum() for x in baseline).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            for i, example in enumerate(examples):
                example["_entropy_measurement"] = entropy.observation_descriptor(
                    directory, 0, i)
            result = scorer(model, examples, True, chunk=2)
            sum(x.sum() for x in result).backward()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(result, baseline)))
            for parameter, gradient in zip(model.parameters(), expected):
                self.assertTrue(torch.equal(parameter.grad, gradient))
            for i, count in enumerate([3, 1]):
                observed = json.loads(Path(examples[i]["_entropy_measurement"][
                    "path"]).read_text())
                self.assertEqual(observed["token_count"], count)
                self.assertTrue(observed["measured"])


if __name__ == "__main__":
    unittest.main()
