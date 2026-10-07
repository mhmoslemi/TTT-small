"""Fixed-batch cleanup and legacy resume checks without GPU dependencies."""

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from config_validation import (COMMON_OPTIONAL_KEYS, COMMON_REQUIRED_KEYS,
                               PROBLEM_OPTIONAL_KEYS, PROBLEM_REQUIRED_KEYS,
                               RETIRED_BATCH_GROWTH_KEYS)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "train_multy_CVaR.py"


def checkpoint_functions():
    names = {"_save_training_checkpoint", "_load_training_checkpoint"}
    tree = ast.parse(RUNNER.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    scope = {"Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), scope)
    return scope


class FixedBatchConfigTests(unittest.TestCase):
    def load(self, yaml_values, saved=None, extra_cli=()):
        numpy = ModuleType("numpy")
        numpy.integer = int
        yaml = ModuleType("yaml")
        yaml.safe_load = lambda stream: dict(yaml_values)
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = Path(directory) / "config.yaml"
            config.write_text("# parsed by the fixture\n")
            argv = [str(RUNNER), "--config", str(config)]
            if saved is not None:
                resume_dir = Path(directory) / "run"
                resume_dir.mkdir()
                (resume_dir / "config.json").write_text(json.dumps(saved))
                argv += ["--resume", str(resume_dir)]
            argv += list(extra_cli)
            with patch.dict(sys.modules, {"numpy": numpy, "yaml": yaml}), \
                 patch.object(sys, "argv", argv), \
                 patch.dict(os.environ, {"AVAILABLE_GPUS": "0,1"}), \
                 patch("gpu_runtime.query_gpu_memory", return_value={}), \
                 patch("gpu_runtime.detect_attention_heads", return_value=32):
                spec = importlib.util.spec_from_file_location("growth_cleanup_runner", RUNNER)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.load_config()

    def base_config(self):
        return dict(problem="circle_packing", model_name="Qwen/Qwen3-8B",
                    max_seq_length=32768, max_new_tokens=32768,
                    backend="hf", load_in_4bit=False, generation_backend="hf",
                    groups_per_step=3, group_size=36, target_modules=[],
                    thinking=False, training_gpu_id=0, gpu_ids="0,1",
                    train_examples_per_microbatch=1,
                    vllm_gpu_memory_utilization=.9,
                    vllm_tensor_parallel_size=0, vllm_pipeline_parallel_size=1,
                    available_gpu_ids="", evaluation_gpu_id=None,
                    reserve_last_gpu_for_evaluation=False, num_gpus=None)

    def test_legacy_yaml_and_saved_caps_cannot_reactivate_growth(self):
        values = self.base_config()
        values.update(dict(max_groups_per_step=99, max_group_size=999,
                           growth_force_step=0, growth_factor=4,
                           growth_valid_yield=0, growth_distinct_min=0))
        saved = dict(values, groups_per_step=5, group_size=20)
        cfg, merged = self.load(values, saved)
        self.assertEqual((cfg.groups_per_step, cfg.group_size), (5, 20))
        self.assertFalse(RETIRED_BATCH_GROWTH_KEYS & merged.keys())
        self.assertTrue(all(not hasattr(cfg, key) for key in RETIRED_BATCH_GROWTH_KEYS))
        cfg, merged = self.load(values, saved, ["--groups-per-step", "2", "--group-size", "12"])
        self.assertEqual((cfg.groups_per_step, cfg.group_size), (2, 12))
        self.assertFalse(RETIRED_BATCH_GROWTH_KEYS & merged.keys())

    def test_hierarchical_size_is_still_strategies_times_programs(self):
        values = self.base_config()
        values.update(dict(coder_model_name="Qwen/Qwen3-8B",
                           strategy_model_name="Qwen/Qwen3-8B",
                           strategies_per_parent=3, programs_per_strategy=12,
                           strategy_archive_top_r=2,
                           strategy_max_new_tokens=32768, strategy_max_seq_length=32768,
                           strategy_temperature=.6, strategy_top_p=.95,
                           strategy_thinking=True, strategy_reasoning_effort="high"))
        cfg, merged = self.load(values, extra_cli=["--strategies"])
        self.assertEqual(cfg.group_size, 36)
        self.assertEqual(cfg.groups_per_step, 3)
        self.assertFalse(RETIRED_BATCH_GROWTH_KEYS & merged.keys())

    def test_no_growth_cli_flags_or_main_loop_mutation(self):
        tree = ast.parse(RUNNER.read_text())
        parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_build_arg_parser")
        options = {node.value for node in ast.walk(parser) if isinstance(node, ast.Constant)
                   and isinstance(node.value, str) and node.value.startswith("--")}
        for key in RETIRED_BATCH_GROWTH_KEYS:
            self.assertNotIn("--" + key.replace("_", "-"), options)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        for node in ast.walk(main):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                self.assertNotIn(node.attr, {"groups_per_step", "group_size"})
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, {"grow_batch", "cur_g", "cur_k"})
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn(node.value, {"next_groups_per_step", "next_group_size"})
        self.assertNotIn("[init] batch growth", RUNNER.read_text())
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "grow_batch"
                             for node in tree.body))

    def test_all_nine_presets_remove_only_the_retired_keys(self):
        paths = list((ROOT / "configs").glob("*.yaml"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            text = path.read_text()
            keys = set(re.findall(r"(?m)^([a-zA-Z_][a-zA-Z_0-9]*):", text))
            problem = re.search(r"(?m)^problem:\s*([\w-]+)", text).group(1)
            required = COMMON_REQUIRED_KEYS | PROBLEM_REQUIRED_KEYS[problem]
            allowed = required | COMMON_OPTIONAL_KEYS | PROBLEM_OPTIONAL_KEYS[problem]
            with self.subTest(config=path.name):
                self.assertFalse(keys & RETIRED_BATCH_GROWTH_KEYS)
                self.assertFalse(required - keys)
                self.assertFalse(keys - allowed)
                self.assertIn("pilot_programs_per_strategy", keys)
                self.assertIn("strategy_format_max_retries", keys)


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.functions = checkpoint_functions()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.torch = ModuleType("torch")
        self.torch.save = lambda payload, path: Path(path).write_bytes(pickle.dumps(payload))
        self.torch.load = lambda path, **kw: pickle.loads(Path(path).read_bytes())
        self.patch = patch.dict(sys.modules, {"torch": self.torch})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_new_checkpoint_roundtrip_without_batch_growth_state(self):
        sampler = SimpleNamespace(state_dict=lambda: {"archive": [1, 2]})
        optimizer = SimpleNamespace(state_dict=lambda: {"moment": .5})
        tracker = SimpleNamespace(state_dict=lambda: {"value": 1.1})
        result = self.functions["_save_training_checkpoint"](
            self.path, 12, self.path / "adapter_latest", sampler, optimizer,
            spo_rs_tracker=tracker)
        payload = self.functions["_load_training_checkpoint"](self.path)
        self.assertEqual(payload, dict(version=2, next_step=12, adapter_dir="adapter_latest",
                                       sampler={"archive": [1, 2]}, optimizer={"moment": .5},
                                       spo_rs_tracker={"value": 1.1}))
        self.assertEqual(result, self.path / "training_state.pt")
        self.assertFalse((self.path / "training_state.pt.tmp").exists())

    def test_version_one_checkpoint_keeps_training_state_loadable(self):
        payload = dict(version=1, next_step=4, adapter_dir="adapter_step003",
                       sampler={"archive": [3]}, optimizer={"state": 7},
                       next_groups_per_step=24, next_group_size=999)
        self.torch.save(payload, self.path / "training_state.pt")
        loaded = self.functions["_load_training_checkpoint"](self.path)
        self.assertEqual(loaded, payload)
        # The loader no longer requires legacy batch fields, even for v1.
        payload.pop("next_groups_per_step")
        payload.pop("next_group_size")
        self.torch.save(payload, self.path / "training_state.pt")
        self.assertEqual(self.functions["_load_training_checkpoint"](self.path), payload)

    def test_missing_essential_state_and_unknown_versions_still_rejected(self):
        self.assertIsNone(self.functions["_load_training_checkpoint"](self.path))
        for payload in ({"version": 1}, {"version": 2}, {"version": 99}):
            self.torch.save(payload, self.path / "training_state.pt")
            with self.assertRaises(ValueError):
                self.functions["_load_training_checkpoint"](self.path)


class ArchiveTests(unittest.TestCase):
    def test_archived_controller_and_signals_retain_previous_behavior(self):
        from _retire.batch_growth import batch_for_step, collect_growth_signals, grow_batch
        cfg = SimpleNamespace(max_groups_per_step=8, max_group_size=128,
                              growth_valid_yield=.7, growth_distinct_min=4,
                              growth_factor=2, growth_force_step=10)
        self.assertEqual(grow_batch(3, 36, {"best_valid_yield": .8, "distinct_good": 4}, cfg), (6, 72))
        self.assertEqual(grow_batch(3, 36, {"best_valid_yield": .6, "distinct_good": 4}, cfg), (3, 36))
        self.assertEqual(batch_for_step(3, 36, 10, cfg), (8, 128))
        self.assertEqual(batch_for_step(3, 36, 9, cfg), (3, 36))
        stats = collect_growth_signals([
            ([{}, {}, {"retry_attempt": 1}], [True, False, True], [2, 0, 3],
             ["x", "", "recovery"], 1)])
        self.assertEqual(stats, {"best_valid_yield": .5, "distinct_good": 1})

    def test_active_sources_never_import_archive(self):
        for path in ROOT.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("_retire"), path.name)
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith("_retire") for alias in node.names), path.name)


if __name__ == "__main__":
    unittest.main()
