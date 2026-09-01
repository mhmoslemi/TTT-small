import io
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory.bandit import allocate_memory_arms, credit_memory_arms
from memory.bank import MemoryBank
from memory.config import MemoryConfig
from memory.curator import MemoryCurator
from memory.extractor import LessonExtractor
from memory.llm import LOOKUP_STEP_OFFSET
from memory.lookup import MemoryLookup
from memory.prompts import memory_protocol_block, render_memory_block
from memory.types import FAILURE, SUCCESS, Lesson, RolloutRecord


def causal_config(version="V2", group_size=8, **overrides):
    values = {
        "memory": True,
        "memory_version": version,
        "memory_lookup_mode": "select",
        "memory_lookup_max_select": 1,
        "memory_lookup_fallback": "none",
        "memory_arm_control_fraction": 0.2,
        "memory_arm_explore_fraction": 0.2,
        "memory_arm_max_lessons": 1,
        "memory_arm_exploration_c": 0.5,
        "memory_arm_comparison_n": 0,
        "memory_outcome_credit": True,
        "memory_text_reinforce": False,
        "group_size": group_size,
    }
    values.update(overrides)
    return MemoryConfig.from_dict(values, verbose=False)


def lesson(index, outcome=SUCCESS):
    return Lesson.create(
        title=f"method_{index}", summary=f"summary_{index}",
        lesson=f"operation_{index}", outcome=outcome, step=index,
    )


class _ExtractionLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete_many(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return [self.response for _ in prompts]


class MemoryVersionConfigTests(unittest.TestCase):
    def test_v1_preserves_historical_allocation(self):
        cfg = causal_config("V1")
        bank = MemoryBank(cfg)
        items = [lesson(i) for i in range(3)]
        bank.add_many(items)

        arms = allocate_memory_arms(8, [items[0]], bank, cfg, step=1)

        self.assertEqual(
            [(arm.name, arm.count) for arm in arms],
            [("selected", 5), ("no_memory", 2), ("explore", 1)],
        )
        self.assertEqual(cfg.arm_comparison_n, 0)

    def test_v2_resolves_and_guarantees_fixed_comparison_budget(self):
        cfg = causal_config("V2")
        bank = MemoryBank(cfg)
        items = [lesson(i) for i in range(3)]
        bank.add_many(items)

        arms = allocate_memory_arms(
            8, [items[0]], bank, cfg, step=1, reservations={})

        self.assertEqual(cfg.arm_comparison_n, 2)
        self.assertEqual(
            [(arm.name, arm.count) for arm in arms],
            [("selected", 4), ("no_memory", 2), ("explore", 2)],
        )
        self.assertEqual(sum(arm.count for arm in arms), 8)

    def test_v2_rejects_noncausal_or_impossible_settings(self):
        with self.assertRaisesRegex(ValueError, "outcome_credit"):
            causal_config("V2", memory_outcome_credit=False)
        with self.assertRaisesRegex(ValueError, "group_size >= 3"):
            causal_config("V2", group_size=2)

    def test_disabled_memory_accepts_either_version(self):
        cfg = MemoryConfig.from_dict(
            {"memory": False, "memory_version": "V2"}, verbose=False)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.version, "V2")


class V2BanditTests(unittest.TestCase):
    def test_batch_reservations_spread_exploration(self):
        cfg = causal_config()
        bank = MemoryBank(cfg)
        items = [lesson(i) for i in range(5)]
        bank.add_many(items)
        reservations = {items[0].id: 4}  # selected by all four parents
        explored = []

        for _ in range(4):
            arms = allocate_memory_arms(
                8, [items[0]], bank, cfg, step=2,
                reservations=reservations)
            explored.append(next(arm for arm in arms
                                 if arm.name == "explore").ids[0])

        self.assertEqual(len(set(explored)), 4)
        self.assertNotIn(items[0].id, explored)

    def test_credit_records_estimand_role_context_and_separate_metrics(self):
        cfg = causal_config()
        bank = MemoryBank(cfg)
        selected, explore = lesson(1), lesson(2)
        bank.add_many([selected, explore])
        observations = {
            "no_memory": {
                "memory_ids": [], "rewards": [0.0, 2.0],
                "valids": [True, False], "codes": ["control", "control"],
            },
            "selected": {
                "memory_ids": [selected.id],
                "rewards": [1.0, 3.0, 4.0, 5.0],
                "valids": [True, True, True, False],
                "codes": ["a", "b", "b", "c"],
            },
            "explore": {
                "memory_ids": [explore.id], "rewards": [0.0, 3.0],
                "valids": [False, True], "codes": ["x", "y"],
            },
        }

        updates = credit_memory_arms(
            bank, observations, parent_reward=1.5, step=4,
            parent_id="parent-123")

        self.assertEqual({update["n"] for update in updates}, {2})
        self.assertTrue(all("valid_rate_delta" in update for update in updates))
        self.assertTrue(all("exact_code_unique_rate" in update
                            and "control_exact_code_unique_rate" in update
                            for update in updates))
        row = selected.causal_history[0]
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["arm"], "selected")
        self.assertEqual(row["context_id"], "parent-123")
        self.assertEqual(row["parent_reward"], 1.5)
        self.assertEqual(selected.outcome_stats(2)["trials"], 1)
        with self.assertRaisesRegex(ValueError, "expected best@n=2"):
            bank.record_outcome(
                [selected.id], 1, 1, 1, 0.0, 5,
                comparison_n=1, arm="selected")

    def test_ucb_keeps_nonzero_uncertainty_when_observed_means_are_zero(self):
        cfg = causal_config()
        bank = MemoryBank(cfg)
        sparse, dense = lesson(1), lesson(9)
        bank.add_many([sparse, dense])
        for index in range(1):
            bank.record_outcome(
                [sparse.id], 2, 1, 0, 0.0, index,
                comparison_n=2, arm="explore")
        for index in range(10):
            bank.record_outcome(
                [dense.id], 2, 1, 0, 0.0, index,
                comparison_n=2, arm="explore")

        chosen = bank.exploration_lesson(c=0.5, reservations={})

        self.assertEqual(chosen.id, sparse.id)


class V2IdentityAndPromptTests(unittest.TestCase):
    def _tested_source(self):
        cfg = causal_config()
        bank = MemoryBank(cfg)
        source = lesson(1)
        source.uses = 7
        bank.add_many([source])
        bank.record_outcome(
            [source.id], 4, 3, 2, 0.25, 3,
            comparison_n=2, arm="selected", context_id="p")
        return cfg, bank, source

    def test_rewritten_lesson_inherits_lineage_not_causal_trials(self):
        cfg, bank, source = self._tested_source()
        rewritten = Lesson.create(
            source.title, source.summary, "a different intervention",
            source.outcome, step=9, scope=source.scope)
        rewritten._merged_from_ids = [source.id]

        MemoryCurator(cfg, bank, None, "task")._carry_counters([rewritten])

        self.assertEqual(rewritten.uses, 7)
        self.assertIn(source.id, rewritten.lineage_ids)
        self.assertEqual(rewritten.arm_trials, 0)
        self.assertEqual(rewritten.causal_history, [])

    def test_v1_rewrite_keeps_historical_counter_inheritance(self):
        cfg = causal_config("V1")
        bank = MemoryBank(cfg)
        source = lesson(1)
        source.uses = 7
        source.arm_trials = 3
        source.tail_uplift_sum = 0.75
        bank.add_many([source])
        rewritten = Lesson.create(
            source.title, source.summary, "a different intervention",
            source.outcome, step=9, scope=source.scope)
        rewritten._merged_from_ids = [source.id]

        MemoryCurator(cfg, bank, None, "task")._carry_counters([rewritten])

        self.assertEqual(rewritten.uses, 7)
        self.assertEqual(rewritten.arm_trials, 3)
        self.assertEqual(rewritten.tail_uplift_sum, 0.75)

    def test_exact_intervention_keeps_direct_causal_trials(self):
        cfg, bank, source = self._tested_source()
        unchanged = Lesson.create(
            source.title, "edited catalog summary", source.lesson,
            source.outcome, step=9, scope=source.scope)
        unchanged._merged_from_ids = [source.id]

        MemoryCurator(cfg, bank, None, "task")._carry_counters([unchanged])

        self.assertEqual(unchanged.arm_trials, 1)
        self.assertEqual(len(unchanged.causal_history), 1)

    def test_v2_rendering_is_contrastive_and_calls_lessons_hypotheses(self):
        record = RolloutRecord(
            parent_code="old_program()", parent_reward=1.0,
            code="new_program()", reward=2.0, valid=True)
        old_render = record.render_success(include_parent=False)
        new_render = record.render_success(include_parent=True)

        self.assertNotIn("old_program()", old_render)
        self.assertIn("parent program", new_render)
        self.assertIn("old_program()", new_render)
        self.assertIn("new_program()", new_render)

        success = lesson(1, SUCCESS)
        failure = lesson(2, FAILURE)
        v1 = render_memory_block([success, failure], version="V1")
        v2 = render_memory_block([success, failure], version="V2")
        self.assertIn("Observed to help", v1)
        self.assertNotIn("Observed to help", v2)
        self.assertIn("Hypotheses extracted", v2)
        self.assertIn("Precautions extracted", v2)
        self.assertIn("unconfirmed hypothesis", memory_protocol_block(""))

    def test_v2_serialization_preserves_trial_ledger_but_v1_shape_does_not_change(self):
        cfg, bank, source = self._tested_source()
        self.assertNotIn("causal_history", source.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            bank.save(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["memory_version"], "V2")
            self.assertEqual(payload["comparison_n"], 2)
            self.assertEqual(payload["lessons"][0]["causal_history"][0]["n"], 2)


class V2ExtractionEvidenceTests(unittest.TestCase):
    @staticmethod
    def _response():
        return json.dumps({
            "lessons": [
                {
                    "title": "unsupported strategy",
                    "summary": "claim a successful method",
                    "scope": "local",
                    "kind": "heuristic",
                    "lesson": "repeat the supposed successful change",
                    "importance": 2,
                    "closest_existing": "none",
                    "new_because": "new claim",
                },
                {
                    "title": "observed failure",
                    "summary": "avoid the rejected operation",
                    "scope": "local",
                    "kind": "pitfall",
                    "lesson": "check the verifier contract before returning",
                    "importance": 2,
                    "closest_existing": "none",
                    "new_because": "new precaution",
                },
            ]
        })

    @staticmethod
    def _failure_record():
        return RolloutRecord(
            parent_code="parent()", parent_reward=1.0,
            code="rejected()", reward=0.0, valid=False,
            msg="verification failed")

    @staticmethod
    def _success_record():
        return RolloutRecord(
            parent_code="parent()", parent_reward=1.0,
            code="accepted()", reward=2.0, valid=True)

    def test_v2_failure_only_batch_cannot_create_success_lesson(self):
        cfg = causal_config("V2")
        llm = _ExtractionLLM(self._response())
        extractor = LessonExtractor(cfg, llm, "task")

        notice = io.StringIO()
        with redirect_stdout(notice):
            result = extractor.extract([self._failure_record()], step=0)

        self.assertEqual([item.outcome for item in result.lessons], [FAILURE])
        self.assertIn("discarded 1 lesson", notice.getvalue())
        prompt = llm.prompts[0][0]["content"]
        self.assertIn("no accepted evidence", prompt)
        self.assertIn('kind="pitfall"', prompt)
        self.assertNotIn("why did the successes succeed", prompt)

    def test_v2_success_only_batch_cannot_create_pitfall(self):
        cfg = causal_config("V2")
        llm = _ExtractionLLM(self._response())
        extractor = LessonExtractor(cfg, llm, "task")

        with redirect_stdout(io.StringIO()):
            result = extractor.extract([self._success_record()], step=0)

        self.assertEqual([item.outcome for item in result.lessons], [SUCCESS])
        prompt = llm.prompts[0][0]["content"]
        self.assertIn("no rejected evidence", prompt)

    def test_v2_split_mode_also_enforces_source_outcome(self):
        cfg = causal_config("V2", memory_extract_mode="split")
        llm = _ExtractionLLM(self._response())
        extractor = LessonExtractor(cfg, llm, "task")

        with redirect_stdout(io.StringIO()):
            result = extractor.extract([self._failure_record()], step=0)

        self.assertEqual([item.outcome for item in result.lessons], [FAILURE])
        prompt = llm.prompts[0][0]["content"]
        self.assertNotIn("why did the successes succeed", prompt)
        self.assertIn('kind="pitfall"', prompt)

    def test_v1_keeps_historical_failure_only_parsing(self):
        cfg = causal_config("V1")
        llm = _ExtractionLLM(self._response())
        extractor = LessonExtractor(cfg, llm, "task")

        result = extractor.extract([self._failure_record()], step=0)

        self.assertEqual(
            [item.outcome for item in result.lessons], [SUCCESS, FAILURE])


class DependencyNoticeRoutingTests(unittest.TestCase):
    @staticmethod
    def _module():
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.random = SimpleNamespace()
        fake_yaml = types.ModuleType("yaml")
        module_name = "_train_multy_notice_test"
        path = Path(__file__).parents[1] / "train_multy.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            spec.loader.exec_module(module)
        return module

    def test_retained_stream_falls_back_after_diagnostic_closes(self):
        stream_type = self._module()._NoticeRoutingStream
        visible = io.StringIO()
        diagnostic = io.StringIO()
        stream = stream_type(visible, diagnostic, "test")
        diagnostic.close()

        stream.write("ordinary warning\n")
        stream.write("No prebuilt binary for CUDA\n")
        stream.flush()

        self.assertEqual(
            visible.getvalue(),
            "ordinary warning\nNo prebuilt binary for CUDA\n")

    def test_context_retained_stream_is_safe_after_teardown(self):
        module = self._module()
        visible = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dependency.log"
            with redirect_stderr(visible):
                with module._route_dependency_notices(path):
                    retained = sys.stderr
                    retained.write("No prebuilt binary for CUDA\n")
                retained.write("late transformers warning\n")
                retained.flush()

            self.assertIn("No prebuilt binary for CUDA", path.read_text())
        self.assertEqual(visible.getvalue(), "late transformers warning\n")


class _LookupLLM:
    def __init__(self, lesson_id):
        self.lesson_id = lesson_id
        self.steps = []

    def complete_many(self, prompts, **kwargs):
        self.steps.append(kwargs["step_idx"])
        return [json.dumps({"ids": [self.lesson_id], "why": "relevant"})
                for _ in prompts]


class V2LookupSeedTests(unittest.TestCase):
    def _lookup_step(self, version):
        cfg = causal_config(version)
        bank = MemoryBank(cfg)
        item = lesson(1)
        bank.add_many([item])
        llm = _LookupLLM(item.id)
        lookup = MemoryLookup(cfg, bank, llm, "task")
        lookup.select_batch(
            [SimpleNamespace(value=0.0, raw_score=None, code="parent")],
            step_idx=7, verbose=False)
        return llm.steps[0]

    def test_lookup_seed_namespace_is_versioned(self):
        self.assertEqual(self._lookup_step("V1"), 7)
        self.assertEqual(self._lookup_step("V2"), LOOKUP_STEP_OFFSET + 7)


if __name__ == "__main__":
    unittest.main()
