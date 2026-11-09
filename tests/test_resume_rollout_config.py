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

    def test_changed_jev_parent_count_restarts_partial_comparisons(self):
        path = self.step / "step03.jev.json"
        path.write_text('{"requested": 8, "comparisons": [{"done": true}]}')
        backup = self.resume(SimpleNamespace(**self.current))
        saved = json.loads((backup / "step03/step03.jev.json").read_text())
        self.assertEqual(saved["requested"], 8)
        self.assertEqual(saved["comparisons"], [{"done": True}])
        self.assertFalse(self.step.exists())

    def test_changed_jev_threshold_restarts_generated_step_and_partial_training(self):
        cfg = SimpleNamespace(**self.saved, parent_selection="puct-jev",
                              jev_same_core_threshold=.68)
        save_parent_selections(self.run, 3, "PUCTJevSampler", [], [], rollout_config=cfg)
        cached = dict(requested=4, threshold=.9, selected=["old"], signature="old-signature",
                      comparisons=[{"same_core_idea": .8}], picks_info=["old"], decisions=["old"])
        path = self.step / "step03.jev.json"
        path.write_text(json.dumps(cached))
        (self.step / "step03_group00_rollout000.meta.json").write_text('{"reward": 1}')
        gradient = self.run / ".exact_spo_rs_overflow_step0003.pt"
        gradient.write_bytes(b'old partial gradients')
        marker = self.run / ".exact_sharded_recovery.json"
        marker.write_text(json.dumps(dict(step=3, overflow_gradient_file=gradient.name)))
        checkpoint = self.run / "training_state.pt"
        checkpoint.write_bytes(b'last completed training state')
        adapter = self.run / "adapter_step002"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b'last completed adapter')

        backup = self.resume(cfg)
        self.assertEqual(json.loads((backup / "step03/step03.jev.json").read_text()), cached)
        self.assertTrue((backup / "step03/step03.parents.json").is_file())
        self.assertTrue((backup / "step03/step03_group00_rollout000.meta.json").is_file())
        self.assertTrue((backup / gradient.name).is_file())
        self.assertTrue((backup / marker.name).is_file())
        self.assertFalse(gradient.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(list(self.step.iterdir()), [path])
        retained = json.loads(path.read_text())
        self.assertIsNone(retained["selected"])
        self.assertNotIn("picks_info", retained)
        self.assertNotIn("decisions", retained)
        for key in ("signature", "threshold", "comparisons"):
            self.assertEqual(retained[key], cached[key])
        self.assertEqual(checkpoint.read_bytes(), b'last completed training state')
        self.assertEqual((adapter / "adapter_model.safetensors").read_bytes(), b'last completed adapter')
        self.assertIsNone(self.resume(cfg))

    def test_jev_override_does_not_restart_unchanged_uncommitted_or_completed_step(self):
        path = self.step / "step03.jev.json"
        summary = self.step / "step03.summary.json"
        for threshold, selected, mode, completed in (
                (.9, ["old"], "puct-jev", False),
                (.68, None, "puct-jev", False),
                (.68, ["old"], "puct", False),
                (.68, ["old"], "puct-jev", True)):
            with self.subTest(threshold=threshold, selected=selected, mode=mode, completed=completed):
                path.write_text(json.dumps(dict(requested=4, threshold=.9, selected=selected)))
                if completed:
                    summary.write_text('{}')
                cfg = SimpleNamespace(**self.saved, parent_selection=mode,
                                      jev_same_core_threshold=threshold)
                self.assertIsNone(self.resume(cfg))
                self.assertTrue(path.is_file())
                self.assertFalse((self.run / "_restarted_steps").exists())

    def test_unchanged_budget_or_effort_only_edits_keep_recovery(self):
        (self.step / "step03.parents.json").write_text('{"parents": []}')
        cfg = SimpleNamespace(**dict(self.saved, pilot_medium_programs_per_strategy=3))
        self.assertIsNone(self.resume(cfg))
        self.assertTrue(self.step.is_dir())
        self.assertFalse((self.run / "_restarted_steps").exists())

    def test_hardware_and_gpu_count_change_preserves_generated_step_recovery(self):
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "train_multy_CVaR.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_recoverable_pending_spo_rs_artifacts")
        scope = dict(Path=Path, json=json)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "pending-recovery", "exec"), scope)
        for group in range(2):
            for rollout in range(2):
                base = self.step / f"step03_group{group:02d}_rollout{rollout:03d}"
                Path(str(base) + ".meta.json").write_text(json.dumps(dict(
                    step=3, status="evaluation_pending", group=group, parent_group=group,
                    rollout=rollout, n_response_tokens=12, counts_toward_allocation=True, retry_attempt=0)))
                for suffix in ("txt", "prompt.txt", "tokens.zlib", "behavior_logprobs.zlib"):
                    Path(str(base) + "." + suffix).touch()
        cfg = SimpleNamespace(groups_per_step=2, group_size=2, hardware_profile="B200_full_ram",
                              num_training_gpus=4, training_gpu_ids="2,3,4,5")
        recovered = scope["_recoverable_pending_spo_rs_artifacts"](self.run, 3, cfg)
        self.assertEqual(len(recovered), 4)
        recovered[0]["token_path"].unlink()
        self.assertIsNone(scope["_recoverable_pending_spo_rs_artifacts"](self.run, 3, cfg))

    def test_full_ram_resume_launcher_is_portable_and_checks_sidecars(self):
        import os
        import subprocess
        root = Path(__file__).resolve().parents[1]
        launcher = self.run / "resume_ac1_full_ram.sh"
        launcher.write_text((root / launcher.name).read_text())
        (self.run / "run_copy.sh").write_text('#!/bin/sh\nprintf "%s\\n" "$AVAILABLE_GPUS" "$TTT_CONFIG" "$@"\n')
        target = self.run / "runs/ac1_Qwen3.8-27B_1107-0624"
        step = target / "step03"
        step.mkdir(parents=True)
        (target / "training_state.pt").touch()
        base = step / "step03_group00_rollout000"
        Path(str(base) + ".meta.json").write_text('{}')
        for suffix in ("txt", "prompt.txt", "tokens.zlib", "behavior_logprobs.zlib"):
            Path(str(base) + "." + suffix).touch()
        result = subprocess.run(["sh", str(launcher)], text=True, capture_output=True,
                                env=dict(os.environ, AVAILABLE_GPUS="6,7"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["2,3,4,5", "configs/ac1.yaml",
                         "--resume", str(target), "--parent-selection", "puct"])
        Path(str(base) + ".tokens.zlib").unlink()
        result = subprocess.run(["sh", str(launcher)], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing recovery artifact", result.stderr)

    def test_yaml_counts_override_saved_but_cli_is_stronger(self):
        source = (Path(__file__).resolve().parents[1]
                  / "runtime/configuration.py").read_text()
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
        sampler = ModuleType('search.sampler')
        sampler.parent_selection_mode = lambda config: config.get("parent_selection", "puct")
        namespace = dict(
            saved=dict(self.saved, problem="erdos", num_seed_states=10,
                       jev_candidate_multiplier=5, jev_same_core_threshold=.90,
                       strategy_vllm_persistent_workers=2,
                       learning_rate=0.01, strategy_max_seq_length=131072,
                       strategy_max_new_tokens=65536, strategy_temperature=.6, strategy_top_p=.95,
                       strategy_top_k=50, strategy_min_p=.1),
            ydict=dict(self.current, num_seed_states=50, seed_min_points=40,
                       jev_candidate_multiplier=3, jev_same_core_threshold=.68,
                       seed_max_points=500, learning_rate=0.02, strategy_max_seq_length="auto",
                       strategy_max_new_tokens="auto", strategy_temperature=1., strategy_top_p=1.,
                       strategy_top_k=0, strategy_min_p=0.),
            merged=dict(self.current, problem="erdos"), profile_explicit_keys=set(),
            RETIRED_BATCH_GROWTH_KEYS=set(), resume_dir=self.run,
            args=SimpleNamespace(groups_per_step=12, reload_coder_between_phases=True,
                                 strategy_vllm_persistent_workers=0), skip=set(),
            _CLI_TO_CFG={}, parser=SimpleNamespace(get_default=lambda key: None),
        )
        with patch.dict(sys.modules, {'search.sampler': sampler}), redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[overlay], type_ignores=[]),
                         "resume_overlay", "exec"), namespace)
            self.assertEqual(namespace["merged"]["groups_per_step"], 10)
            self.assertEqual(namespace["merged"]["programs_per_strategy"], 10)
            self.assertEqual(namespace["merged"]["pilot_programs_per_strategy"], 6)
            self.assertEqual(namespace["merged"]["jev_candidate_multiplier"], 3)
            self.assertEqual(namespace["merged"]["jev_same_core_threshold"], .68)
            self.assertEqual(namespace["merged"]["learning_rate"], 0.01)
            self.assertEqual(namespace["merged"]["strategy_vllm_persistent_workers"], 2)
            self.assertEqual(tuple(namespace["merged"][key] for key in (
                "num_seed_states", "seed_min_points", "seed_max_points")), (10, 40, 99))
            self.assertEqual(tuple(namespace["merged"][key] for key in (
                "strategy_max_seq_length", "strategy_max_new_tokens", "strategy_temperature",
                "strategy_top_p", "strategy_top_k", "strategy_min_p")),
                ("auto", "auto", 1., 1., 0, 0.))
            namespace["saved"].update(seed_min_points=80, seed_max_points=400)
            exec(compile(ast.Module(body=[overlay], type_ignores=[]),
                         "resume_overlay", "exec"), namespace)
            self.assertEqual(tuple(namespace["merged"][key] for key in (
                "seed_min_points", "seed_max_points")), (80, 400))
            exec(compile(ast.Module(body=[cli], type_ignores=[]),
                         "cli_overlay", "exec"), namespace)
        self.assertEqual(namespace["merged"]["groups_per_step"], 12)
        self.assertTrue(namespace["merged"]["reload_coder_between_phases"])
        self.assertEqual(namespace["merged"]["strategy_vllm_persistent_workers"], 0)


if __name__ == "__main__":
    unittest.main()
