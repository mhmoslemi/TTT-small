import io
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import call, mock_open, patch

from gen_workers import (
    GenerationPool,
    HybridHFGenerationPool,
    OnDemandGenerationPool,
    PhasedVLLMGenerationPool,
    distribute_jobs,
    _flashinfer_comm_guard_required,
    _prepare_flashinfer_comm_compat,
    _redirect_vllm_output,
    _vllm_engine_kwargs,
    _vllm_job_seed,
    _vllm_worker_loop,
)
from feedback import FeedbackConfig, is_code_failure, render_chat
from gpu_runtime import (
    GPUMemory,
    allocate_gpu_roles,
    derive_vllm_parallel_layout,
    derive_vllm_tensor_parallel_size,
    detect_attention_heads,
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
        "model_name": "Qwen/Qwen3-8B",
        "backend": "auto",
        "load_in_4bit": True,
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
        values.update({
            "problem": "gpu_mode",
            "problem_type": "trimul",
            "gpu_type": "H100",
            "reward_workers": 1,
        })
    return values


class VLLMBackendTests(unittest.TestCase):
    def test_gpt_oss_qlora_uses_trainable_checkpoint_only_for_training(self):
        from train_multy import (_resolve_training_backend,
                                 _resolve_training_model_name)

        self.assertEqual(
            _resolve_training_model_name(
                "openai/gpt-oss-120b", None, load_in_4bit=True),
            "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        )
        self.assertEqual(
            _resolve_training_model_name(
                "/models/openai/gpt-oss-20b/", None, load_in_4bit=True),
            "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        )
        self.assertEqual(
            _resolve_training_model_name(
                "openai/gpt-oss-120b", "local/trainable", True),
            "local/trainable",
        )
        self.assertEqual(
            _resolve_training_model_name(
                "openai/gpt-oss-120b", None, load_in_4bit=False),
            "openai/gpt-oss-120b",
        )
        self.assertEqual(
            _resolve_training_backend(
                "hf", "unsloth/gpt-oss-120b-unsloth-bnb-4bit"),
            "unsloth",
        )
        self.assertEqual(
            _resolve_training_backend(
                "auto", "unsloth/gpt-oss-20b-unsloth-bnb-4bit"),
            "unsloth",
        )
        self.assertEqual(
            _resolve_training_backend("hf", "Qwen/Qwen3-8B"),
            "hf",
        )

    def test_multi_gpu_training_uses_balanced_model_parallel_map(self):
        from model_backend import (_quantization_method, _training_device_map,
                                   _training_max_memory)

        self.assertEqual(
            _training_device_map(types.SimpleNamespace(num_training_gpus=3)),
            "balanced",
        )
        self.assertEqual(
            _training_device_map(types.SimpleNamespace(num_training_gpus=1)),
            {"": 0},
        )
        self.assertEqual(
            _training_max_memory(types.SimpleNamespace(
                training_max_memory_gib=[39.0, 38.5, 40.0])),
            {0: "39.0GiB", 1: "38.5GiB", 2: "40.0GiB"},
        )
        self.assertEqual(
            _quantization_method(types.SimpleNamespace(
                quantization_config={
                    "quant_method": "bitsandbytes", "_load_in_4bit": True,
                })),
            "bitsandbytes-4bit",
        )
        self.assertEqual(
            _quantization_method(types.SimpleNamespace(
                quantization_config={"quant_method": "mxfp4"})),
            "mxfp4",
        )

    def test_three_l40s_pass_gpt_oss_training_capacity_preflight(self):
        from train_multy import (_resolve_training_memory_budgets,
                                 _validate_known_training_capacity)

        memory = {
            gpu_id: GPUMemory(gpu_id, "L40S", 45.0, 44.0)
            for gpu_id in (0, 1, 2)
        }
        budgets = _resolve_training_memory_budgets([0, 1, 2], memory)

        self.assertEqual(budgets, [38.0, 38.0, 38.0])
        _validate_known_training_capacity(
            "unsloth/gpt-oss-120b-unsloth-bnb-4bit", budgets)
        with self.assertRaisesRegex(ValueError, "safe aggregate"):
            _validate_known_training_capacity(
                "unsloth/gpt-oss-120b-unsloth-bnb-4bit", [38.0])

    def test_three_gpu_gpt_oss_config_uses_all_cards_for_both_phases(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = _fake_complete_yaml
        sys.modules.pop("train_multy", None)
        memory = {
            gpu_id: GPUMemory(gpu_id, "L40S", 45.0, 44.0)
            for gpu_id in (0, 1, 2)
        }

        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "config.yaml"
                config_path.write_text("problem: circle_packing\n")
                argv = [
                    "train_multy.py", "--config", str(config_path),
                    "--model-name", "openai/gpt-oss-120b",
                    "--backend", "vllm",
                ]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "0,1,2"}), patch(
                        "gpu_runtime.query_gpu_memory", return_value=memory):
                    cfg, _ = load_config()

        self.assertEqual(cfg.training_gpu_ids, "0,1,2")
        self.assertEqual(cfg.gpu_ids, "0,1,2")
        self.assertEqual(cfg.num_training_gpus, 3)
        self.assertEqual(
            cfg.training_model_name,
            "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        )
        self.assertEqual(cfg.backend, "unsloth")
        self.assertEqual(cfg.training_max_memory_gib, [38.0, 38.0, 38.0])
        self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
        self.assertEqual(cfg.vllm_pipeline_parallel_size, 3)

    def test_training_process_exposes_every_ordered_training_gpu(self):
        from train_multy import _pin_training_process

        env = {}
        with patch.dict(os.environ, env, clear=True):
            _pin_training_process("7,2,5")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7,2,5")
            self.assertEqual(os.environ["TTT_TRAINING_GPU_ID"], "7")
            self.assertEqual(os.environ["TTT_TRAINING_GPU_IDS"], "7,2,5")

    def test_sharded_backend_restores_original_device_map(self):
        from model_backend import _ModelPlacementBackend

        events = []

        class Model:
            hf_device_map = {"layer0": 0, "layer1": 1, "layer2": 2}

            def to(self, device):
                events.append(("to", device))
                return self

        class Backend(_ModelPlacementBackend):
            pass

        accelerate = types.ModuleType("accelerate")
        hooks = types.ModuleType("accelerate.hooks")
        accelerate.dispatch_model = lambda model, **kwargs: events.append(
            ("dispatch", model, kwargs))
        hooks.remove_hook_from_submodules = lambda model: events.append(
            ("remove_hooks", model))

        backend = Backend()
        backend.model = Model()
        backend._remember_training_placement(backend.model)
        with patch.dict(sys.modules, {
                "accelerate": accelerate, "accelerate.hooks": hooks}):
            backend.offload_for_generation()
            backend.restore_after_generation()

        self.assertEqual(events[0][0], "remove_hooks")
        self.assertEqual(events[1], ("to", "cpu"))
        self.assertEqual(events[2][0], "dispatch")
        self.assertEqual(events[2][2]["device_map"], Model.hf_device_map)
        self.assertTrue(events[2][2]["force_hooks"])

    def test_saved_adapter_names_the_generation_base(self):
        from train_multy import _save_adapter

        class Model:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True)
                (Path(path) / "adapter_config.json").write_text(
                    '{"base_model_name_or_path": "train/bnb"}\n')

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = _save_adapter(
                Model(), tmp, 0, "openai/gpt-oss-120b")
            saved = __import__("json").loads(
                (Path(adapter_dir) / "adapter_config.json").read_text())

        self.assertEqual(
            saved["base_model_name_or_path"], "openai/gpt-oss-120b")

    def test_large_moe_lora_avoids_expert_wide_adapters(self):
        from model_backend import _resolve_lora_target_modules

        cfg = types.SimpleNamespace(
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            lora_rank=32,
        )
        model_config = types.SimpleNamespace(num_experts=512)

        targets = _resolve_lora_target_modules(cfg, model_config)

        self.assertEqual(targets, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(cfg.effective_target_modules, tuple(targets))

    def test_qwen_coder_next_uses_pipeline_when_replicas_do_not_fit(self):
        roles = allocate_gpu_roles([0, 1, 2], "erdos")
        memory = {
            gpu_id: GPUMemory(gpu_id, "H200", 140.4, 139.8)
            for gpu_id in roles.generation
        }
        cfg = {
            "model_name": "/models/Qwen3-Coder-Next",
            "generation_backend": "vllm",
            "max_seq_length": 32000,
            "memory": True,
            "memory_grant_context": True,
            "memory_token_budget": 3400,
            "vllm_gpu_memory_utilization": "auto",
            "vllm_quantization": "",
            "vllm_max_num_batched_tokens": "auto",
            "gen_micro_batch": "auto",
            "logprob_chunk": "auto",
        }

        heads = detect_attention_heads(cfg["model_name"])
        layout = derive_vllm_parallel_layout(cfg, roles, memory, heads)

        self.assertEqual(heads, 16)
        self.assertEqual(layout.tensor_parallel_size, 1)
        self.assertEqual(layout.pipeline_parallel_size, 3)
        self.assertEqual(layout.replicas, 1)
        self.assertGreater(
            layout.unsharded_stage_required_gib, layout.budget_gib)

        cfg["vllm_tensor_parallel_size"] = layout.tensor_parallel_size
        cfg["vllm_pipeline_parallel_size"] = layout.pipeline_parallel_size
        resolve_memory_settings(cfg, roles, memory)
        self.assertGreaterEqual(cfg["gen_micro_batch"], 1)

    def test_load_config_resolves_qwen_coder_next_to_tp1_pp3(self):
        fake_numpy = types.ModuleType("numpy")
        fake_yaml = types.ModuleType("yaml")

        def qwen_coder_config(stream):
            values = _fake_complete_yaml(stream)
            values.update({
                "model_name": "/models/Qwen3-Coder-Next",
                "generation_backend": "vllm",
                "max_seq_length": 32000,
                "memory": True,
                "memory_grant_context": True,
                "memory_token_budget": 3400,
                "vllm_max_num_batched_tokens": "auto",
                "gen_micro_batch": "auto",
                "logprob_chunk": "auto",
            })
            return values

        fake_yaml.safe_load = qwen_coder_config
        sys.modules.pop("train_multy", None)
        with patch.dict(sys.modules, {"numpy": fake_numpy, "yaml": fake_yaml}):
            from train_multy import load_config

            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "qwen.yaml"
                config_path.write_text("problem: erdos\n")
                argv = ["train_multy.py", "--config", str(config_path)]
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"AVAILABLE_GPUS": "0,1,2"}), patch(
                        "gpu_runtime.query_gpu_memory", return_value={}):
                    cfg, _ = load_config()

        self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
        self.assertEqual(cfg.vllm_pipeline_parallel_size, 3)
        self.assertEqual(cfg.num_gpus, 3)

    def test_small_model_keeps_parallel_replicas(self):
        roles = allocate_gpu_roles([0, 1, 2], "erdos")
        memory = {
            gpu_id: GPUMemory(gpu_id, "H100", 80.0, 79.0)
            for gpu_id in roles.generation
        }
        cfg = {
            "model_name": "Qwen/Qwen3-8B",
            "max_seq_length": 8192,
            "memory": False,
            "vllm_gpu_memory_utilization": "auto",
            "vllm_quantization": "",
        }

        layout = derive_vllm_parallel_layout(
            cfg, roles, memory, detect_attention_heads(cfg["model_name"]))

        self.assertEqual(layout.tensor_parallel_size, 1)
        self.assertEqual(layout.pipeline_parallel_size, 1)
        self.assertEqual(layout.replicas, 3)

    def test_four_l40s_run_keeps_four_qwen8b_rollout_replicas(self):
        roles = allocate_gpu_roles([0, 1, 2, 3], "erdos")
        memory = {
            gpu_id: GPUMemory(gpu_id, "L40S", 45.0, 44.4)
            for gpu_id in roles.generation
        }
        cfg = {
            "model_name": "Qwen/Qwen3-8B",
            "max_seq_length": 35400,
            "memory": True,
            "memory_grant_context": True,
            "memory_token_budget": 3400,
            "vllm_gpu_memory_utilization": "auto",
            "vllm_quantization": "",
        }

        layout = derive_vllm_parallel_layout(
            cfg, roles, memory, detect_attention_heads(cfg["model_name"]))

        self.assertEqual(layout.tensor_parallel_size, 1)
        self.assertEqual(layout.pipeline_parallel_size, 1)
        self.assertEqual(layout.replicas, 4)
        self.assertLessEqual(
            layout.unsharded_stage_required_gib, layout.budget_gib)

    def test_vllm_output_redirects_both_fds_to_append_log(self):
        opened = mock_open()
        opened.return_value.fileno.return_value = 73
        with patch("gen_workers.open", opened, create=True), patch(
                "gen_workers.os.makedirs") as makedirs, patch(
                "gen_workers.os.dup2") as dup2, patch("builtins.print"):
            handle = _redirect_vllm_output(
                "/tmp/example-run/vllm.log", 2, [4, 5])

        self.assertIs(handle, opened.return_value)
        opened.assert_called_once_with(
            "/tmp/example-run/vllm.log", "a", buffering=1,
            encoding="utf-8")
        makedirs.assert_called_once_with("/tmp/example-run", exist_ok=True)
        self.assertEqual(dup2.call_args_list, [call(73, 1), call(73, 2)])

    def test_known_dependency_notices_are_hidden_from_console(self):
        from train_multy import _NoticeRoutingStream

        visible = io.StringIO()
        diagnostic = io.StringIO()
        stream = _NoticeRoutingStream(visible, diagnostic, "dependency")
        stream.write("ordinary model loading output\n")
        stream.write(
            "Skipping import of cpp extensions due to incompatible torch "
            "version.")
        stream.write(" Please upgrade torch.\n")
        stream.write(
            "No prebuilt binary for CUDA 12.9, loading CUDA 12.8 instead.\n")
        stream.write(
            "Warning: You are sending unauthenticated requests to the HF Hub. "
            "Please set a HF_TOKEN.\n")
        stream.write(
            "W0831 torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an "
            "Enum subclass and is now natively supported by torch.compile as "
            "an opaque value type.\n")
        stream.write(
            "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n")

        self.assertEqual(visible.getvalue(), "ordinary model loading output\n")
        routed = diagnostic.getvalue()
        self.assertIn("Skipping import of cpp extensions", routed)
        self.assertIn("Please upgrade torch.", routed)
        self.assertIn("No prebuilt binary for CUDA 12.9", routed)
        self.assertIn("unauthenticated requests to the HF Hub", routed)
        self.assertIn("KernelPreference", routed)
        self.assertIn("`torch_dtype` is deprecated", routed)

    def test_notice_router_restores_logging_handlers_before_closing(self):
        from train_multy import _route_dependency_notices

        visible = io.StringIO()
        logger = logging.getLogger(
            "tests.notice_router_retained_handler")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.WARNING)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = Path(tmpdir) / "vllm.log"
                with patch("sys.stderr", visible):
                    with _route_dependency_notices(log_path):
                        handler = logging.StreamHandler()
                        logger.addHandler(handler)
                    logger.warning("warning emitted after routing context")

                self.assertIs(handler.stream, visible)
                self.assertIn(
                    "warning emitted after routing context",
                    visible.getvalue())
        finally:
            logger.handlers.clear()
            logging.Logger.manager.loggerDict.pop(logger.name, None)

    def test_every_problem_yaml_matches_its_problem_contract(self):
        import yaml

        from config_validation import (
            RERANKER_REQUIRED_KEYS, validate_problem_config,
        )

        config_dir = Path(__file__).resolve().parents[1] / "configs"
        self.assertFalse((config_dir / "defaults.yaml").exists())
        paths = sorted(config_dir.glob("*.yaml"))
        self.assertGreaterEqual(len(paths), 8)
        for path in paths:
            data = yaml.safe_load(path.read_text())
            validate_problem_config(data, source=path)
            self.assertFalse(data["reranker_enabled"], path.name)
            self.assertTrue(
                RERANKER_REQUIRED_KEYS <= set(data), path.name,
            )
            self.assertIn("memory_curate_max_new_tokens", data, path.name)
            self.assertIn("memory_hygiene_profile", data, path.name)
            self.assertIn("adam_beta1", data, path.name)
            self.assertIn("adam_beta2", data, path.name)
            self.assertIn("adam_epsilon", data, path.name)
            self.assertIn("weight_decay", data, path.name)

            if data["problem"] == "gpu_mode":
                self.assertEqual(data["gpu_type"], "H100", path.name)
                self.assertEqual(data["reward_workers"], 1, path.name)
                self.assertTrue(
                    (path.parents[1] / data["task_yaml"]).is_file(), path.name)
                self.assertTrue(
                    (path.parents[1] / data["lib_dir"]).is_dir(), path.name)
            else:
                self.assertGreaterEqual(data["eval_cpus"], 1, path.name)
                self.assertNotIn("gpu_type", data, path.name)
                self.assertNotIn("gpu_lease_timeout_s", data, path.name)

            if data["problem"] == "circle_packing":
                self.assertIn("num_circles", data, path.name)
                self.assertIn("degenerate_threshold", data, path.name)
            else:
                self.assertNotIn("num_circles", data, path.name)

            if data.get("problem_type") == "mla_decode_nvidia":
                self.assertIn("mla_seed_runtime_us", data, path.name)

    def test_memory_and_feedback_dataclasses_match_yaml_contract(self):
        from dataclasses import fields

        from config_validation import COMMON_REQUIRED_KEYS
        from feedback import FeedbackConfig
        from memory.config import MemoryConfig

        memory_keys = {
            "memory" if field.name == "enabled" else f"memory_{field.name}"
            for field in fields(MemoryConfig)
        }
        feedback_keys = {
            ("feedback" if field.name == "enabled" else
             "feedback_lambda" if field.name == "lambda_f" else
             f"feedback_{field.name}")
            for field in fields(FeedbackConfig)
        }

        self.assertEqual(
            {key for key in COMMON_REQUIRED_KEYS
             if key == "memory" or key.startswith("memory_")},
            memory_keys,
        )
        self.assertEqual(
            {key for key in COMMON_REQUIRED_KEYS
             if key == "feedback" or key.startswith("feedback_")},
            feedback_keys,
        )

    def test_complete_yaml_rejects_missing_disabled_reranker_switch(self):
        import yaml

        from config_validation import validate_problem_config

        path = Path(__file__).resolve().parents[1] / "configs" / "erdos.yaml"
        data = yaml.safe_load(path.read_text())
        data.pop("reranker_enabled")
        with self.assertRaisesRegex(ValueError, "reranker_enabled"):
            validate_problem_config(data, source=path)

    def test_problem_yaml_contract_rejects_cross_problem_fields(self):
        from config_validation import validate_problem_config

        with self.assertRaisesRegex(ValueError, "num_circles is not valid"):
            validate_problem_config({
                "problem": "erdos",
                "num_circles": 26,
            }, require_complete=False)

    def test_auto_reward_workers_accounts_for_cpus_per_evaluation(self):
        from train_multy import _resolve_reward_workers

        cfg = types.SimpleNamespace(reward_workers=0, num_gpus=3)
        problem = types.SimpleNamespace(eval_cpus=10)
        self.assertEqual(
            _resolve_reward_workers(cfg, problem, cpu_count=64), 6)

        cfg.reward_workers = 7
        self.assertEqual(
            _resolve_reward_workers(cfg, problem, cpu_count=64), 7)

    def test_cpu_sandbox_hides_gpus_and_enforces_thread_budget(self):
        import sandbox

        proc = types.SimpleNamespace(
            pid=123456789,
            returncode=1,
            communicate=lambda timeout=None: (b"", b""),
        )
        with patch("sandbox.subprocess.Popen", return_value=proc) as popen, \
                patch("sandbox._kill_tree"):
            sandbox.run_code(
                "def run():\n    return 1\n",
                entrypoint="run",
                timeout_s=1,
                max_cpus=6,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            self.assertEqual(env[key], "6")

    def test_problem_passes_eval_cpu_budget_to_sandbox(self):
        from problems.base import (ParentContext, Problem, RewardResult,
                                   SeedState)

        class ExampleProblem(Problem):
            def build_prompt(self, parent, memory=""):
                return []

            def preprocess(self, code, parent):
                return code

            def score(self, output, stdout):
                return RewardResult(reward=1.0, valid=True)

            def seed_states(self):
                return [SeedState()]

        problem = ExampleProblem({"eval_cpus": 6})
        with patch("problems.base.extract_python_code", return_value="pass"), \
                patch("problems.base.run_code", return_value={
                    "ok": True, "value": 1, "stdout": "",
                }) as run_code:
            result = problem.compute_reward("response", ParentContext(), 12)

        self.assertTrue(result.valid)
        run_code.assert_called_once_with(
            "pass", entrypoint="run", timeout_s=12, max_cpus=6)

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

    def test_flashinfer_comm_guard_targets_only_affected_python_and_versions(self):
        self.assertTrue(_flashinfer_comm_guard_required(
            "0.6.16.post3", (3, 11)))
        self.assertTrue(_flashinfer_comm_guard_required("0.6.15", (3, 10)))
        self.assertFalse(_flashinfer_comm_guard_required("0.6.17", (3, 11)))
        self.assertFalse(_flashinfer_comm_guard_required("0.6.16", (3, 12)))
        self.assertFalse(_flashinfer_comm_guard_required("unknown", (3, 11)))

    def test_flashinfer_comm_guard_propagates_to_vllm_children(self):
        stale_module = types.ModuleType("flashinfer.comm.fd_exchange")
        with patch("gen_workers.importlib.metadata.version",
                   return_value="0.6.16.post3"), patch(
                   "gen_workers._flashinfer_comm_guard_required",
                   return_value=True), patch.dict(
                   os.environ, {"PYTHONPATH": "/existing/path"},
                   clear=False), patch.dict(
                   sys.modules,
                   {"flashinfer.comm.fd_exchange": stale_module},
                   clear=False):
            changed = _prepare_flashinfer_comm_compat()

            self.assertTrue(changed)
            self.assertEqual(
                os.environ["TTT_VLLM_DISABLE_BROKEN_FLASHINFER_COMM"], "1")
            self.assertEqual(os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"], "0")
            self.assertTrue(os.environ["PYTHONPATH"].split(os.pathsep)[0].endswith(
                "vllm_compat"))
            self.assertIsNone(sys.modules["flashinfer.comm"])
            self.assertNotIn("flashinfer.comm.fd_exchange", sys.modules)

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
                events.append(("start", (
                    kwargs["vllm_enable_sleep_mode"],
                    kwargs["vllm_sleep_level"],
                )))

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
            got_again = list(pool.iter_group_jobs())
            pool.release()
            pool.shutdown()

        self.assertEqual(got, [(0, [("ok", [1])])])
        self.assertEqual(got_again, [(0, [("ok", [1])])])
        self.assertEqual(events, [
            ("offload", None), ("start", (True, 2)),
            ("generate", None), ("sleep", None), ("restore", None),
            ("offload", None), ("wake", None), ("generate", None),
            ("sleep", None), ("restore", None),
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
        allocator_confs = []

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
                allocator_confs.append(
                    os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

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
        with patch.dict(sys.modules, modules), patch.dict(
                os.environ, {
                    "PYTORCH_CUDA_ALLOC_CONF":
                    "max_split_size_mb:128,expandable_segments:True",
                }, clear=False):
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
                enable_sleep_mode=True,
            )

        self.assertEqual(allocator_confs, ["max_split_size_mb:128"])
        self.assertEqual(ready_queue.items, [("ready", 0, "")])
        self.assertEqual(result_queue.items, [
            (0, 7, [("prompt-0", [1]), ("prompt-1", [2])]),
        ])

    def test_vllm_worker_deep_sleep_discards_and_reloads_weights(self):
        fake_vllm = types.ModuleType("vllm")
        fake_request = types.ModuleType("vllm.lora.request")
        events = []

        class LLM:
            def __init__(self, **_kwargs):
                pass

            def sleep(self, level):
                events.append(("sleep", level))

            def wake_up(self, tags=None):
                events.append(("wake", tags))

            def collective_rpc(self, method):
                events.append(("rpc", method))

        fake_vllm.LLM = LLM
        fake_vllm.SamplingParams = object
        fake_request.LoRARequest = object
        task_queue = _Queue([
            ("__control__", "sleep"),
            ("__control__", "wake_up"),
            None,
        ])
        result_queue = _Queue()
        ready_queue = _Queue()
        control_queue = _Queue()

        with patch.dict(sys.modules, {
                "vllm": fake_vllm,
                "vllm.lora": types.ModuleType("vllm.lora"),
                "vllm.lora.request": fake_request,
                }):
            _vllm_worker_loop(
                rank=0,
                gpu_id=3,
                model_name="org/model",
                max_seq_length=1024,
                load_in_4bit=False,
                task_queue=task_queue,
                result_queue=result_queue,
                ready_queue=ready_queue,
                enable_sleep_mode=True,
                sleep_level=2,
                control_queue=control_queue,
            )

        self.assertEqual(ready_queue.items, [("ready", 0, "sleep:2")])
        self.assertEqual(events, [
            ("sleep", 2),
            ("wake", ["weights"]),
            ("rpc", "reload_weights"),
            ("wake", ["kv_cache"]),
        ])
        self.assertEqual(control_queue.items, [
            ("ok", 0, "sleep", ""),
            ("ok", 0, "wake_up", ""),
        ])

    def test_vllm_control_has_a_hard_timeout(self):
        import queue as queue_module

        pool = GenerationPool.__new__(GenerationPool)
        pool.backend = "vllm"
        pool.sleep_supported = True
        pool.num_workers = 1
        pool.task_queues = [_Queue()]
        pool.control_queue = queue_module.Queue()
        pool.procs = [types.SimpleNamespace(exitcode=None)]
        pool.vllm_log_path = "/tmp/run/vllm.log"

        with patch.dict(
                os.environ, {"TTT_VLLM_CONTROL_TIMEOUT_S": "0.01"}), patch.object(
                pool, "shutdown") as shutdown:
            with self.assertRaisesRegex(TimeoutError, "sleep exceeded"):
                pool._vllm_control("sleep")

        shutdown.assert_called_once_with()

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

    def test_gpu_mode_roles_type_and_replica_count_come_from_inventory(self):
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
        self.assertEqual(cfg.training_gpu_ids, "0,1,2,3")
        self.assertEqual(cfg.num_training_gpus, 4)
        self.assertEqual(cfg.gpu_ids, "0,1,2,3")
        self.assertEqual(cfg.evaluation_gpu_id, 6)
        self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
        self.assertEqual(cfg.vllm_pipeline_parallel_size, 1)
        self.assertEqual(cfg.gpu_type, "H100")
        self.assertEqual(cfg.reward_workers, 1)

    def test_six_fitting_gpus_become_six_parallel_replicas(self):
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
        self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
        self.assertEqual(cfg.vllm_pipeline_parallel_size, 1)
        self.assertEqual(cfg.num_gpus, 6)


if __name__ == "__main__":
    unittest.main()
