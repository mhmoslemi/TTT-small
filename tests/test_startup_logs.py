"""Startup presentation tests: no model, network, or GPU dependencies."""

import ast
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from startup_logs import StartupLog, dashboard_sections, render_dashboard


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class StartupLogTests(unittest.TestCase):
    def test_trainer_import_tolerates_legacy_terminal_output_module(self):
        """A stale logging helper must not abort a distributed update."""
        tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        compatibility_names = {
            "setting_log_only", "_native_bind_setting_log",
        }
        selected = []
        for node in tree.body:
            if (isinstance(node, ast.Import)
                    and any(alias.name == "terminal_output"
                            for alias in node.names)):
                selected.append(node)
            elif isinstance(node, ast.Assign):
                targets = {
                    target.id for target in node.targets
                    if isinstance(target, ast.Name)
                }
                if targets & compatibility_names:
                    selected.append(node)
            elif (isinstance(node, ast.FunctionDef)
                  and node.name == "bind_setting_log"):
                selected.append(node)
            elif (isinstance(node, ast.If)
                  and "hasattr(_terminal_output" in ast.unparse(node)):
                selected.append(node)

        legacy = ModuleType("terminal_output")
        legacy_calls = []
        legacy.terminal_log_only = lambda *args, **kwargs: legacy_calls.append(
            (args, kwargs))
        scope = {}
        code = compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            "trainer logging compatibility", "exec")
        with patch.dict(sys.modules, {"terminal_output": legacy}):
            exec(code, scope)

        scope["setting_log_only"]("legacy diagnostic")
        self.assertEqual(legacy_calls[0][0], ("legacy diagnostic",))
        self.assertIs(legacy.setting_log_only, scope["setting_log_only"])
        self.assertIs(legacy.bind_setting_log, scope["bind_setting_log"])
        self.assertIsNone(scope["bind_setting_log"]("/tmp/unused-setting.log"))

    def test_pre_directory_output_saved_and_resume_appends(self):
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(console):
            path = Path(directory) / "setting.log"
            for index in range(2):
                log = StartupLog()
                with log.capture():
                    print(f"[config] before bind {index}")
                    log.bind(path)
                    print(f"[memory] after bind {index}")
            saved = path.read_text()
        self.assertEqual(console.getvalue(), "")
        for index in range(2):
            self.assertIn(f"before bind {index}", saved)
            self.assertIn(f"after bind {index}", saved)
        self.assertEqual(saved.count("startup settings"), 2)

    def test_direct_run_setting_header_is_not_strategist_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setting.log"
            log = StartupLog(title="Single-Model Discovery")
            log.bind(path)
            saved = path.read_text()
        self.assertIn("Single-Model Discovery | startup settings", saved)
        self.assertNotIn("Strategist Bandit", saved)

    def test_warnings_errors_and_cached_stream_stay_visible(self):
        console = io.StringIO()
        with redirect_stdout(console):
            log = StartupLog()
            with log.capture():
                retained = sys.stdout
                print("[config] quiet detail")
                print("[warn] visible warning")
                print("[config] warning: conflicting setting")
                print("[error] visible error")
            retained.write("later training progress\n")
            retained.flush()
        output = console.getvalue()
        self.assertNotIn("quiet detail", output)
        self.assertIn("visible warning", output)
        self.assertIn("conflicting setting", output)
        self.assertIn("visible error", output)
        self.assertIn("later training progress", output)

    def test_help_and_early_config_failure_are_not_swallowed(self):
        for error in (SystemExit(0), ValueError("bad configuration")):
            console = io.StringIO()
            with redirect_stdout(console):
                log = StartupLog()
                with self.assertRaises(type(error)):
                    with log.capture():
                        print("configuration/help details")
                        raise error
                self.assertIs(sys.stdout, console)
            self.assertIn("configuration/help details", console.getvalue())

    def test_first_loading_progress_visible_but_kernel_details_saved(self):
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(console):
            log = StartupLog()
            log.bind(Path(directory) / "setting.log")
            with log.capture(loading=True):
                print("[backend=hf] loading coder ...")
                print("[memory] full kernel configuration")
                sys.stdout.write("\rLoading weights: 20%")
                sys.stdout.write("\rLoading weights: 100%")
                sys.stdout.write("\n")
            saved = log.path.read_text()
        self.assertIn("loading coder", console.getvalue())
        self.assertIn("Loading weights: 20%", console.getvalue())
        self.assertNotIn("full kernel configuration", console.getvalue())
        self.assertIn("full kernel configuration", saved)
        self.assertIn("Loading weights: 100%", saved)
        self.assertNotIn("Loading weights: 20%", saved)

    def test_summary_fields_not_inferred_from_config_or_surrounding_logs(self):
        log = StartupLog(console=io.StringIO())
        log.print("[config] detail: hidden")
        log.begin_summary()
        log.print("Model: Qwen/coder")
        log.print("Max new tokens: 262144")
        log.print("Future feature: preserve this setting")
        log.end_summary()
        log.print("[init] writing all rollouts to: somewhere")
        self.assertEqual(log.fields, {"Model": "Qwen/coder",
                                     "Max new tokens": "262144",
                                     "Future feature": "preserve this setting"})
        sections = dashboard_sections(log.fields, "/tmp/run")
        self.assertTrue(any(title == "OTHER SETTINGS" for title, _, _ in sections))

    def test_box_alignment_wrapping_and_color(self):
        fields = {"Problem": "erdos", "Model": "Qwen/" + "long-model-" * 14,
                  "Strategy model": "deepseek/strategist",
                  "Strategy sampling": "temperature=1, top_p=1, thinking=on, " * 4,
                  "Total rollouts/step": "108", "Advantage mode": "spo-rs"}
        for width in (44, 80, 112):
            for color in (False, True):
                with self.subTest(width=width, color=color):
                    output = render_dashboard(fields, "/tmp/" + "long-run-" * 20,
                                              width=width, color=color, resuming=True)
                    plain = ANSI.sub("", output)
                    self.assertEqual({len(line) for line in plain.splitlines()}, {width})
                    self.assertEqual("\033[" in output, color)
                    self.assertIn("STRATEGIST BANDIT", plain)
                    self.assertIn("RESUMING", plain)
                    self.assertIn("108", plain)

    def test_direct_dashboard_has_no_strategist_label(self):
        output = render_dashboard(
            {"Problem": "erdos", "Model": "Qwen/Qwen3-8B",
             "Total rollouts/step": "192"},
            "/tmp/direct-run", color=False)
        self.assertIn("SINGLE-MODEL DISCOVERY", output)
        self.assertNotIn("STRATEGIST BANDIT", output)

    def test_real_main_startup_uses_resolved_values_without_changing_behavior(self):
        tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        # Execute the actual startup through the first (mocked) model load.
        stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.With)
                    and any(isinstance(item.context_expr, ast.Call)
                            and isinstance(item.context_expr.func, ast.Name)
                            and item.context_expr.func.id == "_route_dependency_notices"
                            for item in n.items))
        main.body = main.body[:stop + 1] + [ast.parse("print('normal runtime output')").body[0]]
        effort = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_coder_effort_for_rollout_phase")
        code = compile(ast.fix_missing_locations(ast.Module(body=[effort, main], type_ignores=[])),
                       "startup main", "exec")
        base = dict(
            problem="erdos", model_name="Qwen/Qwen3.8-27B", training_model_name="Qwen/Qwen3.8-27B",
            strategy_model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", strategy_backend="local",
            backend="hf", generation_backend="vllm", training_gpu_ids="0,1,2,3,4,5,6,7",
            gpu_ids="0,1,2,3,4,5,6,7", num_training_gpus=8, evaluation_gpu_id=None,
            seed=42, deterministic=False, no_train=False, training_layout="replicated",
            fast=False, isolate_eval=True, strategies=True, reward_workers=2,
            vllm_tensor_parallel_size=1, vllm_pipeline_parallel_size=1, target=.380876,
            num_steps=50, uct=False, puct_c=1, advantage_mode="entropic",
            groups_per_step=3, group_size=36, learning_rate=4e-5, kl_penalty_coef=.1,
            max_new_tokens=262144, max_seq_length=262144, coder_template_kind="qwen3.8",
            coder_reasoning_effort="medium", temperature=1, top_p=1, thinking=True,
            strategies_per_parent=3, programs_per_strategy=12, pilot_programs_per_strategy=7,
            phase2_allocation_method="hurdle", strategy_archive_top_r=2, topk_children_per_parent=8,
            strategy_format_max_retries=3, strategy_max_new_tokens=131072,
            strategy_max_seq_length=131072, strategy_temperature=1, strategy_top_p=1,
            strategy_thinking=True, strategy_reasoning_effort="high", coder_model_profile="qwen3.8-27b",
            fused_long_attention=True, train_examples_per_microbatch=3, lora_rank=32,
            lora_alpha=32, lora_dropout=0., training_memory_fraction=.88, logprob_chunk=256,
            sandbox_timeout_s=1150, spo_rs_beta=2, spo_rs_d_half=.1, spo_rs_rho_min=.5,
            spo_rs_rho_max=.99, spo_rs_clip_epsilon_low=.2, spo_rs_clip_epsilon_high=.38,
            rank_update_epochs=1, x_grpo_contexts_per_step=2, x_grpo_budgets=[1, 2],
            x_grpo_relative_error=.99, x_grpo_entropy_coef=.001,
            rank_clip_epsilon_low=.2, rank_clip_epsilon_high=.38)

        cases = [
            (mode, True)
            for mode in ("entropic", "spo-rs", "x-grpo", "binary", "no-train")
        ] + [("entropic", False)]
        for mode, strategies in cases:
            with self.subTest(mode=mode, strategies=strategies), \
                    tempfile.TemporaryDirectory() as directory:
                cfg = SimpleNamespace(**dict(base, advantage_mode=mode,
                                             no_train=mode == "no-train",
                                             strategies=strategies))
                events = []
                binary = SimpleNamespace(enabled=mode == "binary", init_steps=4,
                                         lora_rank=16, clip_epsilon_low=.2, clip_epsilon_high=.38)
                problem = SimpleNamespace(entrypoint="run", metric_name="C₅ bound", maximize=False,
                                          seed_states=lambda: [1, 2])
                def load_config():
                    events.append("config")
                    print("[model-profile] detailed profile")
                    return cfg, vars(cfg).copy()
                def load_model():
                    events.append("load")
                    print("[memory] detailed kernel setup")
                    return object(), object()
                modules = {}
                for name, attrs in {
                    "problems.binary_coder": dict(BINARY_CODER_MODE="binary-coder",
                        BinaryCoderConfig=SimpleNamespace(from_mapping=lambda _: binary)),
                    "problems.registry": dict(get_problem=lambda *_: problem),
                    "experiment_io": dict(append_step_result=None, save_final_summary=None,
                        save_step_summary=None, make_experiment_dir=lambda *a, **k: directory),
                    "model_backend": dict(load_backend=lambda *a: SimpleNamespace(load=load_model)),
                }.items():
                    modules[name] = ModuleType(name)
                    vars(modules[name]).update(attrs)
                scope = dict(
                    Path=Path, os=os, load_config=load_config, _LOG_TIME_OFFSET_SECONDS=0,
                    _install_console_timestamps=lambda: None,
                    _install_terminal_log=lambda: SimpleNamespace(bind=lambda _: None),
                    bind_setting_log=lambda *args, **kwargs: None,
                    _pin_training_process=lambda _: events.append("pin"),
                    _uses_clipped_policy_loss=lambda m: m in {"spo-rs", "x-grpo", "rank"},
                    _uses_sequence_level_policy_ratio=lambda _: False,
                    _route_dependency_notices=lambda _: nullcontext())
                console = io.StringIO()
                with patch.dict(sys.modules, modules), redirect_stdout(console):
                    exec(code, scope)
                    scope["main"]()
                saved = (Path(directory) / "setting.log").read_text()
                output = console.getvalue()
                self.assertEqual(events, ["config", "pin", "load"])
                expected_title = ("STRATEGIST BANDIT" if strategies
                                  else "SINGLE-MODEL DISCOVERY")
                self.assertIn(expected_title, output)
                self.assertIn(
                    "Strategist Bandit | startup settings" if strategies
                    else "Single-Model Discovery | startup settings",
                    saved)
                self.assertIn("normal runtime output", output)
                if strategies:
                    self.assertIn("phase 2=medium", output)
                else:
                    self.assertIn("Coder reasoning: direct=medium", output)
                    self.assertNotIn("phase 2=", output)
                    self.assertNotIn("Strategy model:", output)
                self.assertIn("262144", output)
                self.assertNotIn("detailed profile", output)
                self.assertNotIn("detailed kernel setup", output)
                self.assertIn("detailed profile", saved)
                self.assertIn("detailed kernel setup", saved)
                self.assertIn("setting.log", output)
                self.assertIn("Total rollouts/step: 216" if mode == "x-grpo"
                              else "Total rollouts/step: 108", output)


if __name__ == "__main__":
    unittest.main()
