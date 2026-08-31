import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gen_workers import (
    GenerationPool,
    HybridHFGenerationPool,
    OnDemandGenerationPool,
    PhasedVLLMGenerationPool,
    distribute_jobs,
    _vllm_engine_kwargs,
    _vllm_job_seed,
    _vllm_worker_loop,
)
from feedback import FeedbackConfig, is_code_failure, render_chat
from gpu_runtime import (
    GPUMemory,
    allocate_gpu_roles,
    derive_vllm_tensor_parallel_size,
    resolve_memory_settings,
    validate_attention_heads,
)
from gpu_lease import gpu_lease


class _Queue:
    def __init__(self, items=None):
        self.items = list(items or [])

    def get(self, timeout=None):
        return self.items.pop(0)

    def put(self, item):
        self.items.append(item)


def _fake_complete_yaml(stream):
    values = {
        "problem": "circle_packing",
        "problem_type": "",
        "model_name": "Qwen/Qwen3-8B",
        "backend": "auto",
        "generation_backend": "hf",
        "vllm_gpu_memory_utilization": 0.9,
        "vllm_tensor_parallel_size": 0,
        "vllm_pipeline_parallel_size": 1,
        "groups_per_step": 8,
        "group_size": 64,
        "max_groups_per_step": None,
        "max_group_size": None,
        "training_gpu_id": 0,
        "available_gpu_ids": "",
        "evaluation_gpu_id": None,
        "reserve_last_gpu_for_evaluation": False,
        "gpu_ids": "",
        "num_gpus": None,
        "target_modules": [],
        "thinking": False,
    }
    if "problem: gpu_mode" in stream.read():
        values["problem"] = "gpu_mode"
    return values


class VLLMBackendTests(unittest.TestCase):
    def test_every_problem_yaml_is_standalone(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        self.assertFalse((config_dir / "defaults.yaml").exists())

        def top_level_keys(path):
            keys = set()
            for line in path.read_text().splitlines():
                if (not line or line[0].isspace() or line.startswith("#")
                        or ":" not in line):
                    continue
                keys.add(line.split(":", 1)[0])
            return keys

        paths = sorted(config_dir.glob("*.yaml"))
        required = top_level_keys(config_dir / "circle_packing.yaml")
        required.discard("reranker_enabled")
        self.assertGreaterEqual(len(required), 100)
        for path in paths:
            missing = required - top_level_keys(path)
            self.assertFalse(missing, f"{path.name} missing keys: {sorted(missing)}")

    def test_offline_auto_backend_selects_hf_without_importing_unsloth(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        sys.modules.pop("model_backend", None)
        unsloth_was_imported = "unsloth" in sys.modules

        try:
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(
                    os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
                from model_backend import HFBackend, load_backend

                backend = load_backend("auto", types.SimpleNamespace())

            self.assertIsInstance(backend, HFBackend)
            self.assertEqual("unsloth" in sys.modules, unsloth_was_imported)
        finally:
            sys.modules.pop("model_backend", None)

    def test_hf_memory_report_omits_inactive_vllm_knobs(self):
        cfg = {
            "generation_backend": "hf",
            "model_name": "Qwen/Qwen3-8B",
            "max_seq_length": 4096,
            "vllm_gpu_memory_utilization": "auto",
            "vllm_max_num_batched_tokens": "auto",
            "vllm_quantization": "",
            "gen_micro_batch": "auto",
            "logprob_chunk": "auto",
            "memory": False,
        }
        roles = allocate_gpu_roles([0], "erdos")
        memory = {0: GPUMemory(0, "L40S", 44.5, 43.7)}

        notes = resolve_memory_settings(cfg, roles, memory)

        self.assertFalse(any("vLLM" in note for note in notes))
        self.assertTrue(any("generation max sequences" in note for note in notes))

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
            quantization="bitsandbytes",
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            max_num_batched_tokens=4096,
            enable_sleep_mode=True,
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
        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertEqual(kwargs["pipeline_parallel_size"], 2)
        self.assertIs(kwargs["fully_sharded_loras"], True)
        self.assertEqual(kwargs["max_num_batched_tokens"], 4096)
        self.assertIs(kwargs["enable_chunked_prefill"], True)
        self.assertIs(kwargs["enable_sleep_mode"], True)

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

    def test_training_4bit_does_not_force_vllm_quantization(self):
        kwargs = _vllm_engine_kwargs(
            model_name="org/model",
            max_seq_length=4096,
            load_in_4bit=True,
            lora_rank=32,
            gpu_memory_utilization=0.9,
        )
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

    def test_small_prompt_jobs_rotate_across_all_workers(self):
        jobs = distribute_jobs(["a", "b", "c", "d"], 1, 3)
        assigned = [[group_idx for group_idx, _prompt, _count in worker]
                    for worker in jobs]
        self.assertEqual(assigned, [[0, 3], [1], [2]])

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

    def test_on_demand_pool_releases_shared_gpu(self):
        events = []

        class FakePool:
            num_workers = 1

            def __init__(self, **kwargs):
                events.append(("start", kwargs["gpu_ids"]))

            def iter_group_jobs(self, *args, **kwargs):
                yield 0, [("ok", [1])]

            def shutdown(self):
                events.append(("shutdown", None))

        pool = OnDemandGenerationPool(
            before_start=lambda: events.append(("offload", None)),
            after_stop=lambda: events.append(("restore", None)),
            model_name="org/model", num_workers=1, gpu_ids=[7],
        )
        with patch("gen_workers.GenerationPool", FakePool):
            got = list(pool.iter_group_jobs())
            pool.release()

        self.assertEqual(got, [(0, [("ok", [1])])])
        self.assertEqual(events, [
            ("offload", None), ("start", [7]),
            ("shutdown", None), ("restore", None),
        ])

    def test_phased_vllm_pool_sleeps_between_rollouts(self):
        events = []

        class FakePool:
            num_workers = 1
            sleep_supported = True

            def __init__(self, **kwargs):
                events.append(("start", kwargs["vllm_enable_sleep_mode"]))

            def sleep(self):
                events.append(("sleep", None))

            def wake_up(self):
                events.append(("wake", None))

            def iter_group_jobs(self, *args, **kwargs):
                events.append(("generate", None))
                yield 0, [("ok", [1])]

            def shutdown(self):
                events.append(("shutdown", None))

        with patch("gen_workers.GenerationPool", FakePool):
            pool = PhasedVLLMGenerationPool(
                before_start=lambda: events.append(("offload", None)),
                after_stop=lambda: events.append(("restore", None)),
                model_name="org/model", num_workers=1, gpu_ids=[7],
            )
            got = list(pool.iter_group_jobs())
            pool.release()
            pool.shutdown()

        self.assertEqual(got, [(0, [("ok", [1])])])
        self.assertEqual(events, [
            ("offload", None), ("start", True), ("sleep", None),
            ("restore", None), ("offload", None), ("wake", None),
            ("generate", None), ("sleep", None), ("restore", None),
            ("shutdown", None),
        ])

    def test_hybrid_hf_pool_splits_every_prompt_across_all_cards(self):
        calls = {}

        class FakeRemote:
            backend = "hf"
            num_workers = 2

            def iter_group_jobs(self, prompts_by_group, _group_size,
                                _adapter_path, _max_new_tokens, _temperature,
                                _top_p, **kwargs):
                counts = kwargs["counts_by_group"]
                calls["remote"] = list(counts)
                for idx, count in enumerate(counts):
                    if count:
                        yield idx, [("remote", [2])] * count

            def shutdown(self):
                calls["shutdown"] = True

        def local_iter(**kwargs):
            counts = kwargs["counts_by_group"]
            calls["local"] = list(counts)
            for idx, count in enumerate(counts):
                if count:
                    yield idx, [("local", [1])] * count

        pool = HybridHFGenerationPool(FakeRemote(), local_iter)
        got = list(pool.iter_group_jobs(
            ["a", "b"], 0, "/tmp/adapter", 4, 1.0, 1.0,
            counts_by_group=[8, 2], show_progress=False,
        ))

        totals = {0: 0, 1: 0}
        for group_idx, results in got:
            totals[group_idx] += len(results)
        self.assertEqual(calls["local"], [3, 0])
        self.assertEqual(calls["remote"], [5, 2])
        self.assertEqual(totals, {0: 8, 1: 2})

    def test_gpu_evaluation_lease_is_per_physical_device(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TTT_GPU_LEASE_DIR": tmp}):
            with gpu_lease(3):
                with self.assertRaises(TimeoutError):
                    with gpu_lease(3, timeout_s=0.02):
                        pass
                with gpu_lease(4, timeout_s=0.02):
                    pass
            with gpu_lease(3, timeout_s=0.02):
                pass

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

            def get_tokenizer(self):
                return types.SimpleNamespace(encode=lambda prompt: [1] * len(prompt))

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
        fake_yaml.safe_load = _fake_complete_yaml
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("{}\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--backend", "vllm",
                    "--training-gpu-id", "0", "--gpu-ids", "0,1",
                ]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "0,1"}):
                    cfg, merged = load_config()

        self.assertEqual(cfg.backend, "hf")
        self.assertEqual(cfg.generation_backend, "vllm")
        self.assertEqual(merged["backend"], "hf")
        self.assertEqual(merged["generation_backend"], "vllm")

    def test_batch_maxima_and_gpu_count_are_derived(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = _fake_complete_yaml
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("{}\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--groups-per-step", "5", "--group-size", "16",
                    "--gpu-ids", "1,0,2,4,6,7", "--thinking",
                    "--training-gpu-id", "1",
                ]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "1,0,2,4,6,7"}):
                    cfg, merged = load_config()

        self.assertEqual(cfg.max_groups_per_step, 5)
        self.assertEqual(cfg.max_group_size, 16)
        self.assertEqual(cfg.num_gpus, 6)
        self.assertEqual(cfg.gpu_ids, "1,0,2,4,6,7")
        self.assertIs(cfg.thinking, True)
        self.assertEqual(merged["num_gpus"], 6)

    def test_authoritative_gpu_role_table(self):
        ordinary_one = allocate_gpu_roles([7], "erdos")
        self.assertEqual(ordinary_one.training, 7)
        self.assertEqual(ordinary_one.generation, [7])
        self.assertIsNone(ordinary_one.evaluation)
        self.assertTrue(ordinary_one.sequential_generation)

        ordinary_many = allocate_gpu_roles([7, 2, 5], "denoising")
        self.assertEqual(ordinary_many.generation, [7, 2, 5])
        self.assertIsNone(ordinary_many.evaluation)
        self.assertTrue(ordinary_many.sequential_generation)

        gpu_one = allocate_gpu_roles([4], "gpu_mode")
        self.assertEqual(gpu_one.generation, [4])
        self.assertEqual(gpu_one.evaluation, 4)
        self.assertTrue(gpu_one.evaluation_shares_generation)

        gpu_two = allocate_gpu_roles([4, 9], "gpu_mode")
        self.assertEqual(gpu_two.generation, [4])
        self.assertEqual(gpu_two.evaluation, 9)
        self.assertFalse(gpu_two.evaluation_shares_generation)

        gpu_many = allocate_gpu_roles([4, 1, 3, 8], "gpu_mode")
        self.assertEqual(gpu_many.generation, [4, 1, 3])
        self.assertEqual(gpu_many.evaluation, 8)

    def test_tensor_parallel_head_divisibility_is_checked_early(self):
        validate_attention_heads(32, 4, "Qwen/Qwen3-8B")
        with self.assertRaisesRegex(ValueError, "32 attention heads"):
            validate_attention_heads(32, 6, "Qwen/Qwen3-8B")

    def test_tensor_parallel_factor_uses_every_gpu_via_replicas(self):
        self.assertEqual(derive_vllm_tensor_parallel_size(3, 32), 1)
        self.assertEqual(derive_vllm_tensor_parallel_size(6, 32), 2)
        self.assertEqual(derive_vllm_tensor_parallel_size(8, 32), 8)
        self.assertEqual(derive_vllm_tensor_parallel_size(7, None), 7)

    def test_gpu_mode_roles_type_and_tp_come_from_inventory(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = _fake_complete_yaml
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("problem: gpu_mode\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--problem", "gpu_mode", "--backend", "vllm",
                ]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "0,1,2,3,6"}):
                    cfg, _ = load_config()

        self.assertEqual(cfg.training_gpu_id, 0)
        self.assertEqual(cfg.gpu_ids, "0,1,2,3")
        self.assertEqual(cfg.evaluation_gpu_id, 6)
        self.assertEqual(cfg.vllm_tensor_parallel_size, 4)
        self.assertEqual(cfg.gpu_type, "H100")
        self.assertEqual(cfg.reward_workers, 1)

    def test_incompatible_all_card_tp_becomes_compatible_replicas(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = _fake_complete_yaml
        sys.modules.pop("train_multy", None)

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "empty.yaml"
                config_path.write_text("problem: gpu_mode\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--problem", "gpu_mode", "--backend", "vllm",
                ]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "0,1,2,3,4,5,6"}):
                    cfg, _ = load_config()

        self.assertEqual(cfg.gpu_ids, "0,1,2,3,4,5")
        self.assertEqual(cfg.vllm_tensor_parallel_size, 2)
        self.assertEqual(cfg.num_gpus, 6)


if __name__ == "__main__":
    unittest.main()
