"""Seed configuration checks without importing NumPy or model packages."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from config_validation import validate_problem_config


ROOT = Path(__file__).resolve().parents[1]


class ErdosSeedTests(unittest.TestCase):
    def test_seed_count_and_inclusive_bounds_reach_the_generator(self):
        source = ast.parse((ROOT / "problems/erdos.py").read_text())
        method = next(node for node in ast.walk(source)
                      if isinstance(node, ast.FunctionDef) and node.name == "seed_states")
        numpy = MagicMock()
        numpy.max.return_value = 0.5
        scope = {"np": numpy, "List": list, "SeedState": SimpleNamespace}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "seeds", "exec"), scope)
        for cfg, count, low, high in (
                ({}, 10, 40, 99),
                ({"seed_min_points": 40, "seed_max_points": 500}, 50, 40, 500),
                ({"seed_min_points": 500, "seed_max_points": 500}, 1, 500, 500)):
            with self.subTest(cfg=cfg):
                numpy.reset_mock()
                rng = numpy.random.default_rng.return_value
                rng.integers.return_value = high
                problem = SimpleNamespace(cfg=cfg, seed=42, num_seed_states=count)
                seeds = scope["seed_states"](problem)
                self.assertEqual(len(seeds), count)
                self.assertEqual(rng.integers.call_count, count)
                rng.integers.assert_called_with(low, high + 1)
                numpy.ones.assert_called_with(high)
                numpy.random.default_rng.assert_called_with(42 + count - 1)

    def test_validation_accepts_ranges_and_rejects_invalid_or_misplaced_values(self):
        for values in ({}, {"seed_min_points": 40, "seed_max_points": 500},
                       {"seed_min_points": 500, "seed_max_points": 500}):
            validate_problem_config(dict(problem="erdos", **values), require_complete=False)
        for values in (
                {"seed_min_points": 501, "seed_max_points": 500},
                {"seed_min_points": 0}, {"seed_max_points": -1},
                {"seed_min_points": 40.5}, {"seed_max_points": True},
                {"num_seed_states": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_problem_config(dict(problem="erdos", **values), require_complete=False)
        for problem in ("ac1", "ac2"):
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                validate_problem_config(dict(problem=problem, problem_type=problem,
                                             seed_min_points=40), require_complete=False)


if __name__ == "__main__":
    unittest.main()
