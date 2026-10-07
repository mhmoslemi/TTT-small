"""Feedback retirement: loss-path, worker-interface, and resume regressions.

The archive diff supplies the exact pre-retirement code for structural parity
checks. These checks need no GPU stack; they do not replace GPU execution tests.
"""

import ast
import copy
import inspect
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = "train_multy_CVaR.py"
WORKER = "fast_distributed.py"


def previous_source(path):
    """Reverse only this file's archived unified diff, entirely in memory."""
    patch = (ROOT / "_retire/feedback_integration.patch").read_text().splitlines(True)
    start = patch.index(f"--- a/{path}\n") + 2
    end = next((i for i in range(start, len(patch)) if patch[i].startswith("--- a/")), len(patch))
    lines = (ROOT / path).read_text().splitlines(True)
    restored = []
    cursor = 0
    for line in patch[start:end]:
        if line.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            position = int(match.group(1)) - 1
            restored.extend(lines[cursor:position])
            cursor = position
        elif line.startswith(" "):
            assert lines[cursor] == line[1:]
            restored.append(line[1:])
            cursor += 1
        elif line.startswith("+"):
            assert lines[cursor] == line[1:]
            cursor += 1
        elif line.startswith("-"):
            restored.append(line[1:])
    restored.extend(lines[cursor:])
    return "".join(restored)


def functions(tree, prefix=""):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = prefix + node.name
            if isinstance(node, ast.FunctionDef):
                result[name] = node
            result.update(functions(node, name + "."))
    return result


class WithoutFeedback(ast.NodeTransformer):
    """Normalize old feedback-off loss expressions, without rewriting math."""
    def visit_If(self, node):
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if names & {"fb_on", "fb_adv", "fb_advantage", "feedback_advantage"}:
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [kw for kw in node.keywords if kw.arg != "feedback_advantage"]
        return self.generic_visit(node)


class FeedbackRetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.after = {p: ast.parse((ROOT / p).read_text()) for p in (TRAINER, WORKER)}
        cls.before = {p: ast.parse(previous_source(p)) for p in (TRAINER, WORKER)}

    def test_core_losses_scoring_and_memory_recovery_unchanged(self):
        old, new = functions(self.before[TRAINER]), functions(self.after[TRAINER])
        for name in ("rank_grpo_loss", "clipped_policy_loss", "compute_batched_token_logprobs",
                     "_detached_behavior_importance_ratio", "_clipped_policy_options",
                     "_run_oom_resilient_backward", "_training_microbatches"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))

    def test_loss_math_and_normalization_match_previous_feedback_off(self):
        variables = {"loss", "metrics", "policy_metrics", "effective_advantage", "eff_adv",
                     "kl_advantage", "kl_adv", "logp_difference", "logp_diff",
                     "average_difference", "avg_logp_diff", "weight"}
        def math(tree):
            normalized = WithoutFeedback().visit(copy.deepcopy(tree))
            expressions = []
            for node in ast.walk(normalized):
                if isinstance(node, ast.Assign):
                    targets = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                    if targets & variables:
                        expressions.append(ast.dump(node))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (node.func.attr == "append" and isinstance(node.func.value, ast.Name)
                            and node.func.value.id in {"batch_losses", "weighted_losses", "attempt_losses"}):
                        expressions.append(ast.dump(node))
            return sorted(expressions)
        for path in (TRAINER, WORKER):
            with self.subTest(path=path):
                self.assertEqual(math(self.before[path]), math(self.after[path]))

    def test_backward_optimizer_reduction_and_policy_forward_calls_unchanged(self):
        names = {"backward", "step", "clip_grad_norm_", "reduce_trainable_gradients",
                 "broadcast_trainable_parameters", "_clip_step_and_sync",
                 "_sum_gradients_to_primary", "compute_batched_token_logprobs"}
        def calls(tree):
            return sorted(ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and ((isinstance(n.func, ast.Name) and n.func.id in names)
                               or (isinstance(n.func, ast.Attribute) and n.func.attr in names)))
        for path in (TRAINER, WORKER):
            self.assertEqual(calls(self.before[path]), calls(self.after[path]), path)

    def test_worker_call_signatures_match_after_argument_removal(self):
        funcs = {}
        for tree in self.after.values():
            funcs.update(functions(tree))
        names = {"local_rank_update", "local_policy_update", "_train_rank_examples",
                 "_a3b_sequence_clipped_standard_loss"}
        for name in names:
            node = copy.deepcopy(funcs[name])
            node.body = [ast.Pass()]
            node.decorator_list = []
            scope = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         "signature", "exec"), scope)
            signature = inspect.signature(scope[name])
            count = 0
            for tree in self.after.values():
                for call in ast.walk(tree):
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == name):
                        self.assertTrue(all(kw.arg is not None for kw in call.keywords))
                        signature.bind(*[None for _ in call.args],
                                       **{kw.arg: None for kw in call.keywords})
                        count += 1
            self.assertGreater(count, 0, name)

    def test_constant_group_filter_and_eval_overlap_keep_feedback_off_behavior(self):
        def assigned(tree, name):
            return next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        class Disabled(ast.NodeTransformer):
            def visit_BoolOp(self, node):
                node = self.generic_visit(node)
                if isinstance(node.op, ast.And):
                    node.values = [v for v in node.values if not (
                        isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not)
                        and isinstance(v.operand, ast.Name) and v.operand.id == "fb_candidate_on")]
                return node
        for name in ("entropic_overlap_eligible", "overlap_eligible"):
            old = Disabled().visit(copy.deepcopy(assigned(self.before[TRAINER], name)))
            new = assigned(self.after[TRAINER], name)
            self.assertEqual(ast.dump(old), ast.dump(new))
        def gate(tree):
            return next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                        and {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
                        >= {"constant", "clipped_policy_mode"})
        for constant in (True, False):
            for clipped in (True, False):
                env = dict(constant=constant, clipped_policy_mode=clipped, fb_candidate_on=False)
                values = [eval(compile(ast.Expression(gate(t)), "gate", "eval"), env)
                          for t in (self.before[TRAINER], self.after[TRAINER])]
                self.assertEqual(*values)

    def test_worker_result_tuple_slots_remain_aligned(self):
        worker = functions(self.after[WORKER])["local_policy_update"]
        result = next(n.value for n in ast.walk(worker) if isinstance(n, ast.Return)
                      and isinstance(n.value, ast.Tuple)
                      and any(isinstance(e, ast.Name) and e.id == "attempt_ratio_count"
                              for e in n.value.elts))
        env = dict(packed=[1, 2, 3, 4], attempt_ratio_count=5, attempt_kl_error="error")
        self.assertEqual(eval(compile(ast.Expression(result), "result", "eval"), env),
                         (1, 2, 3, 4, 5, "error"))
        reads = [n.slice.value for n in ast.walk(worker) if isinstance(n, ast.Subscript)
                 and isinstance(n.value, ast.Name) and n.value.id == "result"
                 and isinstance(n.slice, ast.Constant)]
        self.assertEqual(max(reads), 5)
        parallel = functions(self.after[TRAINER])["ReplicatedDataParallelTrainer.train_policy"]
        returned = next(n.value for n in ast.walk(parallel) if isinstance(n, ast.Return)
                        and isinstance(n.value, ast.Tuple)
                        and any(isinstance(e, ast.Name) and e.id == "ratio_count" for e in n.value.elts))
        self.assertEqual([e.id for e in returned.elts],
                         ["total_loss", "total_logp_delta", "ratio_sum", "ratio_max", "ratio_count", "kl_error"])
        # accumulate appends quarantine count after those six fields.
        sums = [n for n in ast.walk(parallel) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "quarantined_examples" for t in n.targets)]
        self.assertEqual(ast.unparse(sums[0].value), "sum((result[6] for result in results))")

    def test_no_active_feedback_imports_payloads_flags_or_yaml_keys(self):
        for path in [*ROOT.glob("*.py"), *list((ROOT / "problems").glob("*.py"))]:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "feedback", path.name)
                    self.assertFalse((node.module or "").startswith("_retire"), path.name)
        for tree in self.after.values():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.arg)):
                    name = node.id if isinstance(node, ast.Name) else node.arg
                    self.assertFalse(name.startswith("fb_"), name)
                    self.assertNotIn(name, {"FeedbackStats", "FeedbackConfig", "feedback_advantage"})
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            self.assertFalse(key.value.startswith(("feedback", "fb_")), key.value)
                            self.assertNotIn(key.value, {"reprompt_text", "failure_signature"})
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(node.value.startswith(("--feedback", "--no-feedback")))
        for path in (ROOT / "configs").glob("*.yaml"):
            self.assertIsNone(re.search(r"(?m)^feedback(?:_|:)", path.read_text()), path.name)

    def test_failure_classification_preserved(self):
        from _retire.feedback import is_code_failure as previous
        source = ast.parse((ROOT / "problems/base.py").read_text())
        fn = functions(source)["is_code_failure"]
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "classifier", "exec"), scope)
        for kind in ("", "code", "constraint", "timeout", "infrastructure"):
            for parsed in (True, False):
                for ran in (True, False):
                    for message in ("SyntaxError", "run_failed: Timeout after 120s"):
                        result = SimpleNamespace(failure_kind=kind, parsed=parsed, ran=ran, msg=message)
                        self.assertEqual(scope["is_code_failure"](result), previous(result))

    def test_legacy_saved_and_yaml_feedback_settings_cannot_reactivate_feature(self):
        from test_retired_batch_growth import FixedBatchConfigTests
        helper = FixedBatchConfigTests()
        config = dict(helper.base_config(), feedback=True, feedback_lambda=10,
                      feedback_adaptive=True)
        cfg, merged = helper.load(config, saved=config)
        self.assertFalse(any(k == "feedback" or k.startswith("feedback_") for k in merged))
        self.assertFalse(hasattr(cfg, "feedback"))


if __name__ == "__main__":
    unittest.main()
