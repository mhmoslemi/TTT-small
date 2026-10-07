"""Coder context defaults and override precedence; no model/GPU dependencies."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from config_validation import (COMMON_OPTIONAL_KEYS, COMMON_REQUIRED_KEYS,
                               validate_problem_config)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "train_multy_CVaR.py"


def token_functions():
    names = {"_model_basename", "_coder_model_profile", "_profile_default",
             "_apply_coder_model_profile", "_resolve_coder_token_limits"}
    constants = {"_GPT_OSS_120B_MODEL_BASENAME",
                 "_QWEN3_8B_MODEL_BASENAME",
                 "_QWEN3_30B_A3B_THINKING_MODEL_BASENAME",
                 "_QWEN38_27B_MODEL_BASENAME"}
    nodes = [node for node in ast.parse(RUNNER.read_text()).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in constants
                         for t in node.targets))]
    scope = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), scope)
    return scope


class CoderTokenLimitTests(unittest.TestCase):
    def setUp(self):
        self.functions = token_functions()
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def resolve(self, config):
        explicit = set(config)
        self.functions["_apply_coder_model_profile"](config, explicit)
        self.functions["_resolve_coder_token_limits"](config)
        return config

    def test_profile_defaults_and_strategy_limits_unchanged(self):
        for name, expected in [("Qwen/Qwen3-8B", 40960),
                               ("Qwen/Qwen3.8-27B", 262144),
                               ("Qwen/Qwen3-30B-A3B-Thinking-2507", 262144),
                               ("openai/gpt-oss-120b", 131072)]:
            with self.subTest(model=name):
                config = self.resolve(dict(model_name=name,
                    strategy_max_seq_length=8192, strategy_max_new_tokens=4096))
                self.assertEqual(config["max_seq_length"], expected)
                self.assertEqual(config["max_new_tokens"], expected)
                self.assertNotIn("model_native_context_length", config)
                self.assertEqual(config["strategy_max_seq_length"], 8192)
                self.assertEqual(config["strategy_max_new_tokens"], 4096)

    def test_qwen3_8b_profile_forces_thinking_without_fake_effort(self):
        config = self.resolve(dict(
            model_name="Qwen/Qwen3-8B", thinking=False,
            temperature=0.73, top_p=0.81,
        ))
        self.assertTrue(config["thinking"])
        self.assertNotIn("coder_reasoning_effort", config)
        self.assertEqual(config["coder_template_kind"], "qwen3")
        self.assertEqual(config["temperature"], 0.73)
        self.assertEqual(config["top_p"], 0.81)
        self.assertEqual(config["training_layout"], "replicated")
        self.assertFalse(config["load_in_4bit"])

    def test_explicit_limits_still_override_profiles(self):
        config = self.resolve(dict(model_name="Qwen/Qwen3.8-27B",
                                   max_seq_length=32768, max_new_tokens=7000))
        self.assertEqual(config["max_seq_length"], 32768)
        self.assertEqual(config["max_new_tokens"], 7000)
        for key in ("max_seq_length", "max_new_tokens"):
            config = self.resolve(dict(model_name="Qwen/Qwen3.8-27B", **{key: 16000}))
            self.assertEqual(config[key], 16000)
            other = "max_new_tokens" if key == "max_seq_length" else "max_seq_length"
            self.assertEqual(config[other], 262144)

    def test_other_models_read_checkpoint_context_without_changing_layout(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for metadata in ({"max_position_embeddings": 65536},
                             {"max_position_embeddings": 999,
                              "text_config": {"max_position_embeddings": 65536}}):
                (path / "config.json").write_text(json.dumps(metadata))
                config = self.resolve(dict(model_name=directory, training_layout="sharded"))
                self.assertEqual(config["max_seq_length"], 65536)
                self.assertEqual(config["max_new_tokens"], 65536)
                self.assertNotIn("model_native_context_length", config)
                self.assertEqual(config["training_layout"], "sharded")
                self.assertNotIn("strict_exact_long_training", config)

    def test_hub_fallback_downloads_only_small_config(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"max_position_embeddings": 65536}))
            hub = ModuleType("huggingface_hub")
            hub.hf_hub_download = Mock(return_value=str(config_path))
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                config = self.resolve(dict(model_name="example/coder", max_new_tokens=1234))
            hub.hf_hub_download.assert_called_once_with("example/coder", "config.json")
            self.assertEqual(config["max_seq_length"], 65536)
            self.assertEqual(config["max_new_tokens"], 1234)

    def test_explicit_context_needs_no_metadata_lookup(self):
        hub = ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock(side_effect=AssertionError("unexpected lookup"))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            config = self.resolve(dict(model_name="example/coder", max_seq_length=10000))
            self.assertEqual(config["max_new_tokens"], 10000)
            explicit = dict(model_name="example/coder", max_seq_length=10000,
                            max_new_tokens=3000)
            self.assertEqual(self.resolve(dict(explicit)), explicit)
        hub.hf_hub_download.assert_not_called()

    def test_missing_or_invalid_metadata_fails_clearly_not_at_model_startup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for data in ("{}", "broken-json", '{"max_position_embeddings": false}',
                         '{"max_position_embeddings": -1}'):
                path.write_text(data)
                with self.assertRaisesRegex(ValueError, "Cannot resolve the coder context limit"):
                    self.resolve(dict(model_name=directory))

    def test_yaml_contract_allows_omission_and_validates_explicit_overrides(self):
        for key in ("max_seq_length", "max_new_tokens"):
            self.assertNotIn(key, COMMON_REQUIRED_KEYS)
            self.assertIn(key, COMMON_OPTIONAL_KEYS)
            with self.assertRaisesRegex(ValueError, key):
                validate_problem_config(dict(problem="erdos", **{key: 0}),
                                        require_complete=False)
        paths = list((ROOT / "configs").glob("*.yaml"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            text = path.read_text()
            self.assertIsNone(re.search(r"(?m)^(max_seq_length|max_new_tokens):", text))
            self.assertRegex(text, r"(?m)^strategy_max_seq_length: 131072$")
            self.assertRegex(text, r"(?m)^strategy_max_new_tokens: 131072$")

    def test_real_config_loader_profile_cli_and_resume_precedence(self):
        numpy = ModuleType("numpy")
        numpy.integer = int
        yaml = ModuleType("yaml")
        values = dict(problem="erdos", model_name="Qwen/Qwen3.8-27B",
                      backend="hf", load_in_4bit=False, generation_backend="hf",
                      groups_per_step=3, group_size=36, target_modules=[], thinking=True,
                      training_gpu_id=0, gpu_ids="0,1", train_examples_per_microbatch=1,
                      vllm_gpu_memory_utilization=.9, vllm_tensor_parallel_size=0,
                      vllm_pipeline_parallel_size=1, available_gpu_ids="",
                      evaluation_gpu_id=None, reserve_last_gpu_for_evaluation=False,
                      num_gpus=None)
        yaml.safe_load = lambda stream: dict(values)
        with TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"numpy": numpy, "yaml": yaml}), \
             patch.dict(os.environ, {"AVAILABLE_GPUS": "0,1"}), \
             patch("gpu_runtime.query_gpu_memory", return_value={}), \
             patch("gpu_runtime.detect_attention_heads", return_value=24):
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("# fixture parsed by mocked yaml\n")
            resume_dir = Path(directory) / "resume"
            resume_dir.mkdir()
            (resume_dir / "config.json").write_text(json.dumps(dict(values,
                max_seq_length=32768, max_new_tokens=32768,
                model_native_context_length=262144)))
            spec = importlib.util.spec_from_file_location("token_limit_runner", RUNNER)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cases = [([], (262144, 262144)),
                     (["--max-seq-length", "65536", "--max-new-tokens", "12000"], (65536, 12000)),
                     (["--resume", str(resume_dir)], (32768, 32768)),
                     (["--resume", str(resume_dir), "--max-seq-length", "262144",
                       "--max-new-tokens", "262144"], (262144, 262144))]
            for extra, expected in cases:
                with self.subTest(args=extra), patch.object(sys, "argv",
                        [str(RUNNER), "--config", str(config_path), *extra]):
                    cfg, merged = module.load_config()
                    self.assertEqual((cfg.max_seq_length, cfg.max_new_tokens), expected)
                    self.assertNotIn("model_native_context_length", merged)
                    self.assertFalse(hasattr(cfg, "model_native_context_length"))


if __name__ == "__main__":
    unittest.main()
