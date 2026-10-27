"""Resume budget checks without importing models, NumPy, or CUDA."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_io import (
    restart_changed_rollout_step, save_parent_selections,
)


class ResumeRolloutConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.saved = dict(
            groups_per_step=4, group_size=64, strategies_per_parent=4,
            programs_per_strategy=16, pilot_programs_per_strategy=11,
            pilot_medium_programs_per_strategy=0,
            pilot_xhigh_programs_per_strategy=0,
            phase2_low_programs_per_strategy=16,
            phase2_medium_programs_per_strategy=0,
            phase2_xhigh_programs_per_strategy=0,
        )
        self.current = dict(self.saved, groups_per_step=10, group_size=40,
                            programs_per_strategy=10,
                            pilot_programs_per_strategy=6,
                            pilot_medium_programs_per_strategy=3)
        (self.run / "config.json").write_text(json.dumps(self.saved))
        self.step = self.run / "step03"
        self.step.mkdir()

    def resume(self, cfg):
        with redirect_stdout(io.StringIO()):
            return restart_changed_rollout_step(cfg, self.run, 3)

    def test_fresh_and_completed_steps_use_new_counts(self):
        cfg = SimpleNamespace(**self.current)
        self.assertIsNone(self.resume(cfg))
        (self.step / "step03.parents.json").write_text('{"parents": []}')
        (self.step / "step03.summary.json").write_text('{}')
        self.assertIsNone(self.resume(cfg))

    def test_changed_budget_restarts_step_without_touching_checkpoint(self):
        (self.step / "step03.parents.json").write_text('{"parents": []}')
        (self.step / "step03_group00_rollout000.meta.json").write_text('{}')
        checkpoint = self.run / "training_state.pt"
        checkpoint.write_bytes(b'completed optimizer, adapter and sampler state')
        gradient = self.run / ".exact_spo_rs_overflow_step0003.pt"
        gradient.write_bytes(b'partial gradients must not be reused')
        marker = self.run / ".exact_sharded_recovery.json"
        marker.write_text(json.dumps(dict(step=3, overflow_gradient_file=gradient.name)))
        future = self.run / "step04"
        future.mkdir()
        (future / "step04.sampler.request.json").write_text('{}')
        cfg = SimpleNamespace(**self.current, learning_rate=0.001)
        backup = self.resume(cfg)
        self.assertTrue((backup / "step03/step03_group00_rollout000.meta.json").is_file())
        self.assertTrue((backup / "step04/step04.sampler.request.json").is_file())
        self.assertTrue((backup / marker.name).is_file())
        self.assertTrue((backup / gradient.name).is_file())
        self.assertFalse(self.step.exists())
        self.assertFalse(marker.exists())
        self.assertFalse(gradient.exists())
        self.assertFalse(future.exists())
        self.assertEqual(checkpoint.read_bytes(), b'completed optimizer, adapter and sampler state')
        self.assertEqual(json.loads((self.run / "config.json").read_text()), self.saved)
        self.assertEqual(cfg.groups_per_step, 10)
        self.assertIsNone(self.resume(cfg))

    def test_step_snapshot_survives_another_budget_change(self):
        cfg = SimpleNamespace(**self.current)
        save_parent_selections(self.run, 3, "PUCTSampler", [], [],
                               rollout_config=cfg)
        self.assertIsNone(self.resume(cfg))
        changed = SimpleNamespace(**dict(self.current, groups_per_step=12))
        backup = self.resume(changed)
        snapshot = json.loads((backup / "step03/step03.parents.json").read_text())
        self.assertEqual(snapshot["rollout_config"], self.current)
        self.assertEqual(changed.groups_per_step, 12)

    def test_changed_llm_parent_count_restarts_its_cached_request(self):
        (self.step / "step03.sampler.request.json").write_text(
            '{"num_parents": 8}')
        backup = self.resume(SimpleNamespace(**self.current))
        request = backup / "step03/step03.sampler.request.json"
        self.assertEqual(json.loads(request.read_text())["num_parents"], 8)
        self.assertFalse(self.step.exists())

    def test_unchanged_budget_or_effort_only_edits_keep_recovery(self):
        (self.step / "step03.parents.json").write_text('{"parents": []}')
        cfg = SimpleNamespace(**dict(self.saved, pilot_medium_programs_per_strategy=3))
        self.assertIsNone(self.resume(cfg))
        self.assertTrue(self.step.is_dir())
        self.assertFalse((self.run / "_restarted_steps").exists())

    def test_yaml_counts_override_saved_but_cli_is_stronger(self):
        source = (Path(__file__).resolve().parents[1]
                  / "train_multy_CVaR.py").read_text()
        load = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "load_config")
        overlay = next(n for n in load.body if isinstance(n, ast.If) and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "merged" and child.func.attr == "update"
            for child in ast.walk(n)))
        cli = next(n for n in load.body if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Tuple)
                   and any(isinstance(v, ast.Name) and v.id == "arg_name"
                           for v in n.target.elts))
        sampler = ModuleType("sampler")
        sampler.parent_selection_mode = lambda config: config.get("parent_selection", "puct")
        namespace = dict(
            saved=dict(self.saved, learning_rate=0.01),
            ydict=dict(self.current, learning_rate=0.02),
            merged=dict(self.current), profile_explicit_keys=set(),
            RETIRED_BATCH_GROWTH_KEYS=set(), resume_dir=self.run,
            args=SimpleNamespace(groups_per_step=12), skip=set(),
            _CLI_TO_CFG={}, parser=SimpleNamespace(get_default=lambda key: None),
        )
        with patch.dict(sys.modules, {"sampler": sampler}), redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[overlay], type_ignores=[]),
                         "resume_overlay", "exec"), namespace)
            self.assertEqual(namespace["merged"]["groups_per_step"], 10)
            self.assertEqual(namespace["merged"]["programs_per_strategy"], 10)
            self.assertEqual(namespace["merged"]["pilot_programs_per_strategy"], 6)
            self.assertEqual(namespace["merged"]["learning_rate"], 0.01)
            exec(compile(ast.Module(body=[cli], type_ignores=[]),
                         "cli_overlay", "exec"), namespace)
        self.assertEqual(namespace["merged"]["groups_per_step"], 12)


if __name__ == "__main__":
    unittest.main()
