import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gen_workers import (
    GenerationPool,
    _vllm_engine_kwargs,
    _vllm_job_seed,
    _vllm_worker_loop,
)
from feedback import FeedbackConfig, is_code_failure, render_chat


class _Queue:
    def __init__(self, items=None):
        self.items = list(items or [])

    def get(self, timeout=None):
        return self.items.pop(0)

    def put(self, item):
        self.items.append(item)


class VLLMBackendTests(unittest.TestCase):
    def test_feedback_only_accepts_code_failures(self):
        result = types.SimpleNamespace
        self.assertTrue(is_code_failure(result(
            failure_kind="code", parsed=True, ran=False, msg="SyntaxError")))
        self.assertFalse(is_code_failure(result(
            failure_kind="constraint", parsed=True, ran=True,
            msg="Invalid solution")))
        self.assertFalse(is_code_failure(result(
            failure_kind="timeout", parsed=True, ran=False,
            msg="Timeout after 120s")))
        self.assertFalse(is_code_failure(result(
            failure_kind="infrastructure", parsed=True, ran=False,
            msg="task files missing")))
        self.assertTrue(is_code_failure(result(
            failure_kind="", parsed=False, ran=False, msg="no_code_block")))
        self.assertFalse(is_code_failure(result(
            failure_kind="", parsed=True, ran=False,
            msg="run_failed: Timeout after 120s")))

    def test_feedback_reprompt_uses_rollout_thinking_mode(self):
        class Tokenizer:
            def __init__(self):
                self.enable_thinking = None

            def apply_chat_template(self, _messages, **kwargs):
                self.enable_thinking = kwargs["enable_thinking"]
                return "rendered"

        tokenizer = Tokenizer()
        self.assertEqual(
            render_chat(tokenizer, [{"role": "user", "content": "x"}],
                        enable_thinking=True),
            "rendered",
        )
        self.assertIs(tokenizer.enable_thinking, True)

    def test_feedback_caps_scale_from_current_batch(self):
        cfg = FeedbackConfig(enabled=True)
        self.assertEqual(cfg.resolve_caps(5, 16), (16, 4))
        self.assertEqual(cfg.resolve_caps(8, 64), (103, 26))

    def test_feedback_caps_keep_explicit_and_unlimited_modes(self):
        fixed = FeedbackConfig(
            enabled=True, max_per_step=12, max_per_signature=3)
        self.assertEqual(fixed.resolve_caps(20, 100), (12, 3))
        unlimited = FeedbackConfig(
            enabled=True, max_per_step=-1, max_per_signature=-1)
        self.assertEqual(unlimited.resolve_caps(5, 16), (0, 0))

    def test_vllm_engine_kwargs_map_runtime_controls(self):
        kwargs = _vllm_engine_kwargs(
            model_name="org/model",
            max_seq_length=8192,
            load_in_4bit=True,
            lora_rank=64,
            gpu_memory_utilization=0.82,
            enforce_eager=True,
            enable_prefix_caching=False,
            max_num_seqs=12,
            seed=123,
        )

        self.assertEqual(kwargs["model"], "org/model")
        self.assertEqual(kwargs["max_model_len"], 8192)
        self.assertIs(kwargs["enable_lora"], True)
        self.assertEqual(kwargs["max_lora_rank"], 64)
        self.assertAlmostEqual(kwargs["gpu_memory_utilization"], 0.82)
        self.assertIs(kwargs["enforce_eager"], True)
        self.assertIs(kwargs["enable_prefix_caching"], False)
        self.assertEqual(kwargs["max_num_seqs"], 12)
        self.assertEqual(kwargs["seed"], 123)
        self.assertEqual(kwargs["quantization"], "bitsandbytes")

    def test_vllm_engine_kwargs_omit_optional_limits(self):
        kwargs = _vllm_engine_kwargs(
            model_name="org/model",
            max_seq_length=4096,
            load_in_4bit=False,
            lora_rank=32,
            gpu_memory_utilization=0.9,
            max_num_seqs=0,
            seed=None,
        )

        self.assertNotIn("max_num_seqs", kwargs)
        self.assertNotIn("seed", kwargs)
        self.assertNotIn("quantization", kwargs)

    def test_vllm_engine_rounds_adapter_capacity_up(self):
        kwargs = _vllm_engine_kwargs(
            model_name="org/model",
            max_seq_length=4096,
            load_in_4bit=False,
            lora_rank=24,
            gpu_memory_utilization=0.9,
        )
        self.assertEqual(kwargs["max_lora_rank"], 32)

    def test_vllm_job_seed_is_repeatable_and_separates_groups(self):
        self.assertIsNone(_vllm_job_seed(None, 3, 1, 2))
        first = _vllm_job_seed(42, 3, 1, 2)
        self.assertEqual(first, _vllm_job_seed(42, 3, 1, 2))
        self.assertNotEqual(first, _vllm_job_seed(42, 3, 1, 3))

    def test_generation_pool_rejects_unknown_backend_before_spawning(self):
        with self.assertRaisesRegex(ValueError, r"expected hf\|vllm"):
            GenerationPool("org/model", 1, backend="vision")

    def test_generation_pool_surfaces_worker_errors(self):
        pool = GenerationPool.__new__(GenerationPool)
        pool.num_workers = 1
        pool.gen_micro_batch = 0
        pool.task_queues = [_Queue()]
        pool.result_queue = _Queue([(0, None, {"error": "engine failed"})])
        pool.procs = [types.SimpleNamespace(exitcode=None)]

        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            list(pool.iter_group_jobs(
                prompts_by_group=["prompt"],
                group_size=1,
                adapter_path="/tmp/adapter",
                max_new_tokens=4,
                temperature=1.0,
                top_p=1.0,
                show_progress=False,
            ))

    def test_vllm_worker_preserves_result_queue_contract(self):
        fake_vllm = types.ModuleType("vllm")
        fake_request = types.ModuleType("vllm.lora.request")

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class LoRARequest:
            def __init__(self, name, adapter_id, path):
                self.name = name
                self.adapter_id = adapter_id
                self.path = path

        class Candidate:
            def __init__(self, prompt, idx):
                self.text = f"{prompt}-{idx}"
                self.token_ids = [idx + 1]

        class RequestOutput:
            def __init__(self, prompt, count):
                self.outputs = [Candidate(prompt, idx) for idx in range(count)]

        class LLM:
            def __init__(self, **_kwargs):
                pass

            def generate(self, prompts, sampling_params, **_kwargs):
                return [RequestOutput(prompt, params.n)
                        for prompt, params in zip(prompts, sampling_params)]

        fake_vllm.LLM = LLM
        fake_vllm.SamplingParams = SamplingParams
        fake_request.LoRARequest = LoRARequest
        task_queue = _Queue([
            (4, "/tmp/adapter_step004", [(7, "prompt", 2)], {
                "max_new_tokens": 10,
                "temperature": 0.8,
                "top_p": 0.95,
            }),
            None,
        ])
        result_queue = _Queue()
        ready_queue = _Queue()

        modules = {
            "vllm": fake_vllm,
            "vllm.lora": types.ModuleType("vllm.lora"),
            "vllm.lora.request": fake_request,
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            _vllm_worker_loop(
                rank=0,
                gpu_id=3,
                model_name="org/model",
                max_seq_length=1024,
                load_in_4bit=False,
                task_queue=task_queue,
                result_queue=result_queue,
                ready_queue=ready_queue,
                seed=42,
            )

        self.assertEqual(ready_queue.items, [("ready", 0, "")])
        self.assertEqual(result_queue.items, [
            (0, 7, [("prompt-0", [1]), ("prompt-1", [2])]),
        ])

    def test_backend_vllm_cli_expands_to_hf_training(self):
        # Config parsing itself does not need the heavy runtime dependencies.
        # Stub them so this test also runs on a CPU-only development machine.
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = lambda _stream: {}
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("{}\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--backend", "vllm",
                ]
                with patch.object(sys, "argv", argv):
                    cfg, merged = load_config()

        self.assertEqual(cfg.backend, "hf")
        self.assertEqual(cfg.generation_backend, "vllm")
        self.assertEqual(merged["backend"], "hf")
        self.assertEqual(merged["generation_backend"], "vllm")

    def test_batch_maxima_and_gpu_count_are_derived(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = lambda _stream: {}
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("{}\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--groups-per-step", "5", "--group-size", "16",
                    "--gpu-ids", "0,2,4,6,7", "--thinking",
                ]
                with patch.object(sys, "argv", argv):
                    cfg, merged = load_config()

        self.assertEqual(cfg.max_groups_per_step, 5)
        self.assertEqual(cfg.max_group_size, 16)
        self.assertEqual(cfg.num_gpus, 5)
        self.assertEqual(cfg.gpu_ids, "0,2,4,6,7")
        self.assertIs(cfg.thinking, True)
        self.assertEqual(merged["num_gpus"], 5)


if __name__ == "__main__":
    unittest.main()
