import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment_io import StepPlotter
from plot_erdos_best_curve import load_best_curve


class StepPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_both_scripts_in_background_in_step_order(self):
        started, release = threading.Event(), threading.Event()
        def run(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(2))
            return SimpleNamespace(returncode=0)
        plotter = StepPlotter(self.root, "erdos")
        with patch("subprocess.run", side_effect=run) as mocked:
            try:
                plotter.submit(0)
                self.assertTrue(started.wait(2))
                # Queueing the next completed step does not wait for a plot.
                plotter.submit(1)
                self.assertEqual(mocked.call_count, 1)
            finally:
                release.set()
                plotter.close()
            self.assertEqual(mocked.call_count, 4)
            for call, step in zip(mocked.call_args_list, (0, 0, 1, 1)):
                command = call.args[0]
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[2], str(self.root.resolve()))
                self.assertEqual(command[3:5], ["--max-step", str(step)])
                self.assertEqual(call.kwargs["env"]["MPLBACKEND"], "Agg")
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(call.kwargs["timeout"], 120)
            self.assertIn("--saved-only", mocked.call_args_list[0].args[0])
            self.assertEqual(Path(mocked.call_args_list[1].args[0][1]).name,
                             "plot_erdos_best_curve.py")
        self.assertIn("completed step 1", (self.root / "plots.log").read_text())

    def test_non_erdos_gets_generic_curve_only(self):
        plotter = StepPlotter(self.root, "circle_packing")
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            plotter.submit(3)
            plotter.close()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(Path(run.call_args.args[0][1]).name,
                         "plot_erdos_best_curve.py")

    def test_failure_of_one_script_does_not_prevent_other_or_training(self):
        for failure in (SimpleNamespace(returncode=1),
                        subprocess.TimeoutExpired("plot", 120),
                        OSError("unavailable")):
            with self.subTest(failure=failure):
                plotter = StepPlotter(self.root, "erdos")
                with patch("subprocess.run", side_effect=[failure,
                           SimpleNamespace(returncode=0)]) as run:
                    with patch("sys.stdout", new=io.StringIO()):
                        plotter.submit(0)
                        plotter.close()
                self.assertEqual(run.call_count, 2)
        self.assertIn("plot refresh failed", (self.root / "plots.log").read_text())

    def _metas(self):
        (self.root / "config.json").write_text('{"problem":"erdos"}')
        for step, score in [(0, .45), (1, .4)]:
            folder = self.root / f"step{step:02d}"
            folder.mkdir()
            (folder / f"step{step:02d}_group0_rollout0.meta.json").write_text(
                json.dumps(dict(step=step, group=0, rollout=0, valid=True,
                                raw_score=score, construction=None)))

    def _solution_functions(self):
        # Exercise routing without needing numpy or matplotlib on this Mac.
        source = (Path(__file__).resolve().parents[1] / "plot_erdos.py").read_text()
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in {"load_metas", "pick", "main"}]
        scope = dict(argparse=argparse, json=json, Path=Path,
                     as_array=lambda value: None, run_program=Mock(),
                     extract_code=Mock())
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     "plot_erdos.py", "exec"), scope)
        return scope

    def test_both_loaders_exclude_later_steps(self):
        self._metas()
        steps, cumulative, per_step, valid_count = load_best_curve(self.root, max_step=0)
        self.assertEqual((steps, cumulative, per_step, valid_count),
                         ([0], [.45], [.45], 1))
        scope = self._solution_functions()
        metas = scope["load_metas"](self.root, max_step=0)
        self.assertEqual(len(metas), 1)
        self.assertEqual(scope["pick"](metas)["raw_score"], .45)

    def test_missing_construction_never_replays_in_auto_mode(self):
        self._metas()
        scope = self._solution_functions()
        with patch.object(sys, "argv", ["plot_erdos.py", str(self.root),
                                         "--max-step", "0", "--saved-only"]):
            with patch("sys.stdout", new=io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "forbids program replay"):
                    scope["main"]()
        scope["run_program"].assert_not_called()
        scope["extract_code"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
