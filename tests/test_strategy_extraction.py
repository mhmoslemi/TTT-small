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

    def test_fence_cannot_pair_across_reasoning_and_final_channels(self):
        body = ('Use the parent.\n```text\nimplementation details\n```\n'
                'Evaluate and return the best construction.')
        for marker in ('</think>', '</analysis>', '</reasoning>',
                       '<|channel|>final<|message|>', 'assistantfinal'):
            response = ('Private draft\n```\nunfinished example\n' + marker
                        + '\n<strategy>' + body + '</strategy>')
            with self.subTest(marker=marker):
                self.assertEqual(self.extract(response), (body, None))

    def test_unclosed_inner_fence_cannot_hide_outer_strategy_end(self):
        body = 'Use the parent.\n```text\nalgorithm details without a closing fence.'
        response = '<think>derivation</think>\n<strategy>' + body + '\n</strategy>'
        self.assertEqual(self.extract(response), (body, None))

    def test_repeated_reasoning_boundaries_do_not_leak_earlier_drafts(self):
        response = ('private work\n</think>\n<strategy>unfinished first draft\n'
                    '```\nmore reasoning\n</think>\n<strategy>'
                    'actual final plan\n```text\nexample\n```\nfinish</strategy>')
        self.assertEqual(self.extract(response),
                         ('actual final plan\n```text\nexample\n```\nfinish', None))

    def test_incomplete_final_cannot_close_a_block_from_reasoning(self):
        response = ('private work\n```\n<strategy>example</strategy>\n```\n'
                    '<strategy>draft\n</think>\n...```</strategy>')
        self.assertIsNotNone(self.extract(response)[1])

    def test_native_tags_in_a_complete_fenced_example_remain_literal(self):
        body = ('Use the parent.\n```text\n</think>\n'
                '<strategy>example only</strategy>\n```\nEvaluate the result.')
        for prefix in ('', '<think>private work</think>\n'):
            self.assertEqual(self.extract(prefix + '<strategy>' + body + '</strategy>'), (body, None))
        self.assertIsNotNone(self.extract('<think>```text\n</think>\n'
                                         '<strategy>example only</strategy>\n```')[1])

    def test_mask_preserves_offsets_and_retains_plan_text_exactly(self):
        from generation.strategy_prompts_openai import strategy_boundary_view
        body = 'Use α and β. The example `</strategy>` is literal.\n```text\nλ = 1\n```'
        raw = 'private ```\nwork\n</think>\n<strategy>' + body + '</strategy>'
        self.assertEqual(len(strategy_boundary_view(raw)), len(raw))
        self.assertEqual(self.extract(raw), (body, None))

    def test_inline_backticks_cannot_swallow_later_blocks(self):
        first = '<strategy>Use ```literal text```.</strategy>'
        self.assertEqual(self.extract(first + '<strategy>final plan</strategy>'), ('final plan', None))
        self.assertIsNotNone(self.extract(first + '<strategy>unfinished plan')[1])

    def test_saved_latest_run_false_rejections_extract_the_exact_final_plan(self):
        directory = ROOT / 'runs/erdos_gpt-oss-120b_1101-0306/step00'
        if not directory.is_dir():
            self.skipTest('saved run artifacts are not available')
        for name in ('step00_group00_fold00_strategy00_attempt00',
                     'step00_group00_fold00_strategy00_attempt03',
                     'step00_group03_fold00_strategy00_attempt00',
                     'step00_group05_fold00_strategy01_attempt02',
                     'step00_group07_fold00_strategy01_attempt00'):
            with self.subTest(attempt=name):
                raw = (directory / (name + '.txt')).read_text()
                final = raw.rsplit('</think>', 1)[1].strip()
                self.assertTrue(final.startswith('<strategy>') and final.endswith('</strategy>'))
                expected = final[len('<strategy>'):-len('</strategy>')].strip()
                self.assertEqual(self.extract(raw), (expected, None))
        raw = (directory / 'step00_group07_fold00_strategy02_attempt00.txt').read_text()
        self.assertIsNotNone(self.extract(raw)[1])

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
