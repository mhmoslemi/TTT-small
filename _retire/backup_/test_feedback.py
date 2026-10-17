"""Tests for the retired feature only; not part of active runtime discovery."""

import types
import unittest
from _retire.feedback import FeedbackConfig, is_code_failure, render_chat

RETIRED_KEYS = frozenset(["feedback","feedback_lambda","feedback_anneal_steps","feedback_anneal_shape","feedback_lambda_final","feedback_clip","feedback_chars","feedback_max_per_step","feedback_auto_fraction","feedback_include_constant_groups","feedback_inject_mode","feedback_normalize","feedback_adaptive","feedback_validity_floor","feedback_validity_target","feedback_max_reward_ratio","feedback_reward_scale_floor","feedback_max_per_signature","feedback_auto_signature_fraction"])


class RetiredFeedbackTests(unittest.TestCase):
    def test_feedback_dataclass_matches_yaml_contract(self):
        from dataclasses import fields

        feedback_keys = {
            ("feedback" if field.name == "enabled" else
             "feedback_lambda" if field.name == "lambda_f" else
             f"feedback_{field.name}")
            for field in fields(FeedbackConfig)
        }

        self.assertEqual(
            {key for key in RETIRED_KEYS
             if key == "feedback" or key.startswith("feedback_")},
            feedback_keys,
        )

    def test_feedback_only_accepts_code_failures(self):
        result = types.SimpleNamespace
        self.assertTrue(is_code_failure(result(
            failure_kind="code", parsed=True, ran=False, msg="SyntaxError")))
        self.assertFalse(is_code_failure(result(
            failure_kind="constraint", parsed=True, ran=True,
            msg="Invalid solution")))
        self.assertFalse(is_code_failure(result(
            failure_kind="timeout", parsed=True, ran=False,
            msg="Timeout after 120s")))
        self.assertFalse(is_code_failure(result(
            failure_kind="infrastructure", parsed=True, ran=False,
            msg="task files missing")))
        self.assertTrue(is_code_failure(result(
            failure_kind="", parsed=False, ran=False, msg="no_code_block")))
        self.assertFalse(is_code_failure(result(
            failure_kind="", parsed=True, ran=False,
            msg="run_failed: Timeout after 120s")))

    def test_feedback_reprompt_uses_rollout_thinking_mode(self):
        class Tokenizer:
            def __init__(self):
                self.enable_thinking = None

            def apply_chat_template(self, _messages, **kwargs):
                self.enable_thinking = kwargs["enable_thinking"]
                return "rendered"

        tokenizer = Tokenizer()
        self.assertEqual(
            render_chat(tokenizer, [{"role": "user", "content": "x"}],
                        enable_thinking=True),
            "rendered",
        )
        self.assertIs(tokenizer.enable_thinking, True)

    def test_feedback_caps_scale_from_current_batch(self):
        cfg = FeedbackConfig(enabled=True)
        self.assertEqual(cfg.resolve_caps(5, 16), (16, 4))
        self.assertEqual(cfg.resolve_caps(8, 64), (103, 26))

    def test_feedback_caps_keep_explicit_and_unlimited_modes(self):
        fixed = FeedbackConfig(
            enabled=True, max_per_step=12, max_per_signature=3)
        self.assertEqual(fixed.resolve_caps(20, 100), (12, 3))
        unlimited = FeedbackConfig(
            enabled=True, max_per_step=-1, max_per_signature=-1)
        self.assertEqual(unlimited.resolve_caps(5, 16), (0, 0))


if __name__ == "__main__":
    unittest.main()

