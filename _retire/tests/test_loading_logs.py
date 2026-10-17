"""Replica loading logs: real descriptor capture, no GPU dependencies."""

import ast
import io
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout

from loading_logs import quiet_replica_load


ROOT = Path(__file__).resolve().parents[1]


class LoadingLogTests(unittest.TestCase):
    def test_python_and_native_output_are_quiet_only_during_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            script = textwrap.dedent(r"""
                import os, sys
                from loading_logs import quiet_replica_load
                print('first replica loading details', flush=True)
                with quiet_replica_load(sys.argv[1]):
                    print('duplicate loading details', flush=True)
                    print('Loading weights: 100%', file=sys.stderr, flush=True)
                    os.write(1, b'native download stdout\n')
                    os.write(2, b'native download stderr\n')
                print('replica 2/8 loaded', flush=True)
                print('training progress', file=sys.stderr, flush=True)
                os.write(1, b'native training stdout\n')
                os.write(2, b'native training stderr\n')
            """)
            result = subprocess.run(
                [sys.executable, "-c", script, str(log)], cwd=ROOT,
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [
                "first replica loading details", "replica 2/8 loaded",
                "native training stdout"])
            self.assertEqual(result.stderr.splitlines(), [
                "training progress", "native training stderr"])
            saved = log.read_text()
            for message in ("duplicate loading details", "Loading weights: 100%",
                            "native download stdout", "native download stderr"):
                self.assertIn(message, saved)
            self.assertNotIn("training progress", saved)

    def test_retained_library_handler_and_stream_survive_file_close(self):
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(console):
            log = Path(directory) / "worker.log"
            with quiet_replica_load(log):
                handler = logging.StreamHandler()
                retained = sys.stderr
                handler.emit(logging.makeLogRecord({"msg": "loading warning"}))
            handler.emit(logging.makeLogRecord({"msg": "training warning"}))
            retained.write("retained training stream\n")
            retained.flush()
            self.assertIn("loading warning", log.read_text())
            self.assertNotIn("training warning", log.read_text())
        self.assertEqual(console.getvalue(),
                         "training warning\nretained training stream\n")

    def test_failure_propagates_and_restores_console(self):
        console = io.StringIO()
        error = RuntimeError("model load failed")
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(console):
            log = Path(directory) / "worker.log"
            with self.assertRaises(RuntimeError) as caught:
                with quiet_replica_load(log):
                    print("load diagnostics")
                    raise error
            self.assertIs(caught.exception, error)
            self.assertIs(sys.stderr, console)
            print("subsequent output", file=sys.stderr)
            saved = log.read_text()
            self.assertIn("load diagnostics", saved)
            self.assertIn("RuntimeError: model load failed", saved)
            self.assertIn(str(log.resolve()), console.getvalue())
        self.assertIn("subsequent output", console.getvalue())

    def test_existing_worker_log_is_appended_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            for message in ("earlier startup", "later startup"):
                with quiet_replica_load(log):
                    print(message)
            self.assertIn("earlier startup", log.read_text())
            self.assertIn("later startup", log.read_text())

    def test_no_log_path_preserves_diagnostics(self):
        console = io.StringIO()
        with redirect_stdout(console), quiet_replica_load(None):
            print("visible startup")
        self.assertEqual(console.getvalue(), "visible startup\n")

    def test_quiet_context_is_only_for_additional_replica_loading(self):
        trainer = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        worker_tree = ast.parse((ROOT / "fast_distributed.py").read_text())
        worker = next(n for n in worker_tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "worker_main")
        def quiet_blocks(tree):
            return [n for n in ast.walk(tree) if isinstance(n, ast.With)
                    and any(isinstance(i.context_expr, ast.Call)
                            and isinstance(i.context_expr.func, ast.Name)
                            and i.context_expr.func.id == "quiet_replica_load"
                            for i in n.items)]
        threaded = next(n for n in trainer.body
                        if isinstance(n, ast.ClassDef)
                        and n.name == "ReplicatedDataParallelTrainer")
        self.assertEqual(len(quiet_blocks(trainer)), 1)
        self.assertEqual(len(quiet_blocks(threaded)), 1)
        self.assertEqual(len(quiet_blocks(worker)), 1)
        block = quiet_blocks(worker)[0]
        self.assertTrue(any(isinstance(n, ast.Import) and
                            any(a.name == "torch" for a in n.names)
                            for n in ast.walk(block)))
        self.assertFalse(any(isinstance(n, ast.While) for n in ast.walk(block)))
        self.assertLess(block.end_lineno, next(n.lineno for n in ast.walk(worker)
                                              if isinstance(n, ast.While)))
        main = next(n for n in trainer.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual(quiet_blocks(main), [])


if __name__ == "__main__":
    unittest.main()
