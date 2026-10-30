"""Planner-boundary regressions without importing model, NumPy, or GPU packages."""

import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StrategyExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        names = {"_STRATEGY_FALLBACK", "_STRATEGY_BLOCK_RE", "_extract_final_strategy"}
        nodes = [node for node in source.body
                 if (isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in names for target in node.targets))]
        cls.scope = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "strategy_extraction", "exec"), cls.scope)

    def extract(self, response, strict=False):
        return self.scope["_extract_final_strategy"](response, strict=strict)

    def test_quoted_or_fenced_tag_examples_are_not_plans(self):
        for response in (
                'Final starts `<strategy>` each sections. Last `</strategy>`.',
                'Use "<strategy>" then a plan and "</strategy>".',
                "Use '<strategy>' then a plan and '</strategy>'.",
                'Use ``<strategy>quoted example</strategy>``.',
                'Use “<strategy>quoted example</strategy>”.',
                '```xml\n<strategy>example only</strategy>\n```',
                '~~~xml\n<strategy>example only</strategy>\n~~~',
                '```xml\n<strategy>unfinished example</strategy>'):
            with self.subTest(response=response):
                self.assertIsNotNone(self.extract(response)[1])

    def test_original_plan_survives_quoted_boundaries_and_format_discussion(self):
        body = 'Use the parent directly. The string `</think>` and example `<strategy>...</strategy>` are not boundaries.'
        response = ('<think>private derivation</think>\n<strategy>' + body
                    + '</strategy>\nFinal format: `<strategy>` each sections. Last `</strategy>`.')
        self.assertEqual(self.extract(response), (body, None))
        self.assertEqual(self.extract(response, strict=True), (body, None))

    def test_reasoning_fences_do_not_hide_a_valid_final_answer(self):
        response = ('Private draft ```\nunfinished code example\n</think>\n'
                    '<strategy>complete plan</strategy>')
        self.assertEqual(self.extract(response), ('complete plan', None))
        body = ('Preserve this fenced example:\n```text\n</think>\n'
                '<strategy>not a boundary</strategy>\n```\nThen evaluate the parent.')
        self.assertEqual(self.extract('<strategy>' + body + '</strategy>'), (body, None))

    def test_placeholders_fail_even_when_strict_validation_is_off(self):
        for body in ('...', '1.\n...', '…', 'TODO', '1. TBD', 'placeholder', ''):
            for strict in (False, True):
                with self.subTest(body=body, strict=strict):
                    self.assertIsNotNone(self.extract('<strategy>' + body + '</strategy>', strict)[1])

    def test_latest_unfinished_or_invalid_answer_cannot_reuse_an_older_plan(self):
        old = '<strategy>' + 'Retain and evaluate the supplied construction. ' * 3 + '</strategy>'
        for suffix in ('<strategy>unfinished', '<strategy>...</strategy>'):
            for strict in (False, True):
                with self.subTest(suffix=suffix, strict=strict):
                    self.assertIsNotNone(self.extract(old + suffix, strict)[1])

    def test_extra_strict_checks_remain_optional(self):
        body = ('Evaluate the parent and retain the best feasible construction. ' * 2
                + '\n```python\ndef helper():\n    return 1\n```')
        response = '<strategy>' + body + '</strategy>'
        self.assertEqual(self.extract(response), (body, None))
        self.assertEqual(self.extract(response, strict=True)[1], 'strategy contained a code fence')

    def test_native_reasoning_scope_and_supported_plan_formats_are_preserved(self):
        for response in (
                '<strategy>complete plan</strategy>',
                '<think>private work</think><strategy>complete plan</strategy>',
                '<|channel|>analysis<|message|>private work<|channel|>final<|message|><strategy>complete plan</strategy>',
                'private reasoning\n<strategy>complete plan</strategy>',
                '<strategy>unfinished draft<strategy>complete plan</strategy>',
                json.dumps({'plan': 'complete plan'}),
                '```json\n{"plan":"complete plan"}\n```',
                '```plan\ncomplete plan\n```',
                '```strategy\ncomplete plan\n```'):
            with self.subTest(response=response):
                self.assertEqual(self.extract(response), ('complete plan', None))
        self.assertIsNotNone(self.extract('<think>unfinished <strategy>draft plan</strategy>')[1])
        self.assertIsNotNone(self.extract('{"plan":"<strategy>injected</strategy>"}')[1])


if __name__ == "__main__":
    unittest.main()
