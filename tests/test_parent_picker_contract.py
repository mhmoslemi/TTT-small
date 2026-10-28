"""Supported picker contracts without importing model, NumPy or CUDA packages."""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from config_validation import validate_problem_config


ROOT = Path(__file__).resolve().parents[1]


class ParentPickerContractTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse((ROOT / 'search/sampler.py').read_text())
        names = {"parent_selection_mode", "can_migrate_parent_selection", "make_sampler"}
        nodes = [node for node in source.body
                 if (isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "PARENT_SELECTION_MODES"
                     for target in node.targets))]
        self.scope = {"PUCTSampler": Mock()}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "pickers", "exec"), self.scope)

    def test_registry_factory_cli_and_validation_agree(self):
        allowed = ("puct", "uct", "llm")
        self.assertEqual(self.scope["PARENT_SELECTION_MODES"], allowed)
        source = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        argument = next(node for node in ast.walk(source) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "--parent-selection")
        parser = argparse.ArgumentParser()
        statement = ast.copy_location(ast.Expr(argument), argument)
        exec(compile(ast.Module(body=[statement], type_ignores=[]),
                     "picker_cli", "exec"), {"p": parser})
        remote = ModuleType('search.llm_sampler')
        remote.LLMSampler = Mock()
        with patch.dict(sys.modules, {'search.llm_sampler': remote}):
            for mode in allowed:
                self.assertEqual(parser.parse_args(["--parent-selection", mode]).parent_selection, mode)
                validate_problem_config(dict(problem="erdos", parent_selection=mode),
                                        require_complete=False)
                self.assertEqual(self.scope["parent_selection_mode"]({"parent_selection": mode}), mode)
                factory = self.scope["make_sampler"]
                actual = factory(parent_selection=mode, num_seeds=2, exp_dir="run", problem="task")
                if mode == "llm":
                    self.assertIs(actual, remote.LLMSampler.return_value)
                    remote.LLMSampler.assert_called_once_with(exp_dir="run", problem="task", num_seeds=2)
                else:
                    self.assertIs(actual, self.scope["PUCTSampler"].return_value)
                    self.scope["PUCTSampler"].assert_called_with(use_uct=(mode == "uct"), num_seeds=2)
        for value in ("removed-picker", True, 3):
            with self.assertRaises(ValueError):
                self.scope["parent_selection_mode"]({"parent_selection": value})
            with self.assertRaises(ValueError):
                validate_problem_config(dict(problem="erdos", parent_selection=value),
                                        require_complete=False)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--parent-selection", "removed-picker"])

    def test_migration_is_only_from_puct_or_uct_to_llm(self):
        migrate = self.scope["can_migrate_parent_selection"]
        for source in ("puct", "uct", "llm"):
            for target in ("puct", "uct", "llm"):
                self.assertEqual(migrate(source, target),
                                 target == "llm" and source in ("puct", "uct"))
        self.assertEqual(self.scope["parent_selection_mode"]({"uct": True}), "uct")
        self.assertEqual(self.scope["parent_selection_mode"]({}), "puct")

    def test_llm_migration_preserves_checkpoint_and_defers_pending_step(self):
        source = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        method = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_migrate_parent_selection")
        checkpoint = Mock()
        scope = dict(json=json, Path=Path, _save_training_checkpoint=checkpoint)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "migration", "exec"), scope)
        sampler_module = ModuleType('search.sampler')
        sampler_module.can_migrate_parent_selection = self.scope["can_migrate_parent_selection"]
        remote = ModuleType('search.llm_sampler')
        migrated = SimpleNamespace(selection_mode="llm", load_state_dict=Mock())
        remote.LLMSampler = Mock(return_value=migrated)
        old = SimpleNamespace(selection_mode="puct", puct_c=1., max_buffer_size=2000,
                              topk_children=8, state_dict=lambda: {"states": ["saved-parent"]})
        optimizer, tracker, problem = object(), object(), object()
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'search.sampler': sampler_module, 'search.llm_sampler': remote}), \
                redirect_stdout(io.StringIO()):
            run = Path(folder)
            config = run / "config.json"
            config.write_text(json.dumps({"parent_selection": "puct", "model_name": "unchanged"}))
            args = (old, SimpleNamespace(parent_selection="llm"), run, 3,
                    run / "adapter_step002", optimizer, tracker)
            self.assertIs(scope[method.name](*args, recovery_pending=True, problem=problem), old)
            checkpoint.assert_not_called()
            remote.LLMSampler.assert_not_called()
            self.assertEqual(json.loads(config.read_text())["parent_selection"], "puct")
            self.assertIs(scope[method.name](*args, recovery_pending=False, problem=problem), migrated)
            migrated.load_state_dict.assert_called_once_with(
                {"states": ["saved-parent"], "selection_mode": "llm"})
            checkpoint.assert_called_once_with(run, 3, run / "adapter_step002",
                                               migrated, optimizer, tracker)
            self.assertEqual(json.loads(config.read_text()),
                             {"parent_selection": "llm", "uct": False, "model_name": "unchanged"})


if __name__ == "__main__":
    unittest.main()
