"""Hardware planning and host admission contracts, without torch/CUDA."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import generation.gen_workers as workers
import models.coder_profiles as coders
import models.strategist_models as strategists
import runtime.gpu_runtime as gpu
import runtime.hardware_profiles as hardware
import runtime.host_memory as host
from config_validation import validate_problem_config


ROOT = Path(__file__).resolve().parents[1]


class HardwareProfileTests(unittest.TestCase):
    def cards(self, b200=True):
        size = 183359 / 1024 if b200 else 95.6
        return {i: gpu.GPUMemory(i, "NVIDIA B200" if b200 else "RTX6000", size, size)
                for i in range(8)}

    def test_local_checkpoints_preserve_profiles_and_explicit_paths(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(hardware, "B200_MODEL_DIR", Path(folder)), redirect_stdout(io.StringIO()):
            for name, leaf in hardware.B200_LOCAL_MODELS.items():
                self.assertEqual(hardware.local_checkpoint(name, "B200"), name)
                path = Path(folder) / leaf
                path.mkdir()
                self.assertEqual(hardware.local_checkpoint(name, "B200"), str(path))
                self.assertEqual(hardware.local_checkpoint(name, "RTX6000"), name)
                self.assertEqual(hardware.checkpoint_identity(str(path)), name)
                self.assertEqual(coders._coder_model_profile(str(path)), coders._coder_model_profile(name))
                self.assertEqual(strategists.strategist_profile(str(path)), strategists.strategist_profile(name))
            explicit = "/my/custom/Qwen3-8B"
            self.assertEqual(hardware.local_checkpoint(explicit, "B200"), explicit)
            name = str(Path(folder) / "Qwen3-8B")
            self.assertTrue(coders._dual_resident_qwen3_8b_strategy_coder_pools(dict(
                hardware_profile="B200", model_name=name, coder_model_name=name,
                strategy_model_name=name, strategies=True, generation_backend="vllm")))

    def test_yaml_resume_refreshes_capacity_not_policy_and_rejects_typos(self):
        saved = dict(gen_micro_batch=3, logprob_chunk=128, spo_rs_beta=4,
                     temperature=1, max_seq_length=262144, model_name="saved-model")
        cfg = dict(saved)
        hardware.restore_hardware_settings(cfg, dict(hardware_profile="b200", logprob_chunk=512), saved)
        self.assertEqual(cfg["hardware_profile"], "B200")
        self.assertEqual((cfg["gen_micro_batch"], cfg["logprob_chunk"]), ("auto", 512))
        for key in ("spo_rs_beta", "temperature", "max_seq_length", "model_name"):
            self.assertEqual(cfg[key], saved[key])
        hardware.restore_hardware_settings(cfg, dict(hardware_profile="RTX6000"), dict(cfg))
        self.assertEqual(cfg["logprob_chunk"], "auto")
        cfg = dict(saved)
        hardware.restore_hardware_settings(cfg, {}, saved)
        self.assertEqual(cfg["gen_micro_batch"], 3)
        with self.assertRaises(ValueError):
            validate_problem_config(dict(problem="erdos", hardware_profile="B20"), require_complete=False)
        for path in (ROOT / "configs").glob("*.yaml"):
            self.assertIn("hardware_profile: RTX6000", path.read_text())
        source = (ROOT / "runtime/configuration.py").read_text()
        self.assertLess(source.index("restore_hardware_settings(merged"), source.index("# 3) CLI overlay"))

    def test_live_validation_never_invents_b200_memory(self):
        cfg = dict(hardware_profile="B200")
        for cards in ({}, self.cards(False)):
            with self.assertRaises(ValueError):
                hardware.validate_hardware_profile(cfg, range(8), cards)
        with patch.object(hardware, "host_memory_info", return_value=None):
            with self.assertRaises(RuntimeError):
                hardware.validate_hardware_profile(cfg, range(8), self.cards())
        with patch.object(hardware, "host_memory_info", return_value=dict(total=1200*host.GIB, available=1000*host.GIB)), redirect_stdout(io.StringIO()):
            hardware.validate_hardware_profile(cfg, range(8), self.cards())

    def test_all_profiled_models_fit_full_context_and_use_every_card(self):
        roles = gpu.allocate_gpu_roles(range(8), "erdos")
        cases = [
            ("Qwen/Qwen3-8B", 40960, False, 1),
            ("Qwen/Qwen3.8-27B", 262144, False, 1),
            ("Qwen/Qwen3-30B-A3B-Thinking-2507", 262144, False, 1),
            ("Qwen/Qwen3-Coder-30B-A3B-Instruct", 262144, False, 1),
            ("openai/gpt-oss-120b", 131072, False, 1),
            ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 131072, True, 1),
            ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 131072, True, 1),
            ("qwen3.8-flash-next", 262144, True, 1),
            ("Qwen/Qwen3.8-Flash-Next-FP8", 262144, True, 1),
            ("Qwen/Qwen3.8-Flash-Next", 262144, True, 4),
            ("deepseek-v4-flash-0731", 262144, True, 2),
        ]
        for model, context, frozen, expected_tp in cases:
            with self.subTest(model=model):
                cfg = dict(model_name=model, max_seq_length=context, hardware_profile="B200",
                           generation_backend="vllm", frozen_strategy=frozen,
                           strategy_inference_profile=bool(frozen and strategists.strategist_profile(model)),
                           vllm_gpu_memory_utilization="auto", vllm_runtime_reserve_gib=16)
                heads = gpu.known_attention_heads(model)
                layout = gpu.derive_vllm_parallel_layout(cfg, roles, self.cards(), heads)
                self.assertEqual(layout.tensor_parallel_size, expected_tp)
                self.assertEqual(layout.replicas * layout.tensor_parallel_size, 8)
                self.assertLessEqual(layout.unsharded_stage_required_gib, layout.budget_gib)
                cfg.update(vllm_tensor_parallel_size=layout.tensor_parallel_size, vllm_pipeline_parallel_size=1)
                gpu.resolve_memory_settings(cfg, roles, self.cards())
                self.assertEqual(cfg["max_seq_length"], context)
                self.assertGreaterEqual(cfg["gen_micro_batch"], 1)
                self.assertLessEqual(cfg["gen_micro_batch"], 256)
                self.assertEqual(cfg["vllm_max_num_batched_tokens"], 16384 if frozen else 8192)
                if frozen:
                    for parents in (1, 3, 4, 8):
                        folded = gpu.align_vllm_layout_to_concurrency(cfg, layout, 8, heads, parents)
                        self.assertLessEqual(folded.replicas, parents)
                        self.assertEqual(folded.replicas * folded.tensor_parallel_size * folded.pipeline_parallel_size, 8)

    def test_engine_options_and_handoff_match_rtx(self):
        for name in strategists.STRATEGIST_ALIASES:
            options = strategists.strategist_engine_options(name)
            self.assertEqual(options["safetensors_load_strategy"], "lazy")
            if name == "qwen3.8-flash-next":
                self.assertIs(options["enable_flashinfer_autotune"], False)
                self.assertTrue(options["engram_config"]["cpu_offload"])
            profile = strategists.strategist_profile(name)
            for tp in (1, 2, 4, 8):
                expected = strategists.strategist_memory_per_rank(name, tp, 262144)
                self.assertEqual(expected[0], (profile["weights_gib"] - profile["ple_gib"]) / tp)
                with patch.object(gpu, "_checkpoint_weight_gib", return_value=None):
                    for hardware_profile in ("RTX6000", "B200"):
                        self.assertEqual(gpu._vllm_memory_per_rank(dict(
                            model_name=name, max_seq_length=262144,
                            hardware_profile=hardware_profile, strategy_inference_profile=True), tp), expected)
        for hardware_profile in ("RTX6000", "B200"):
            cfg = SimpleNamespace(coder_model_profile="gpt-oss-120b", no_train=False,
                                  hardware_profile=hardware_profile)
            self.assertEqual(coders._coder_vllm_persistent_workers(cfg), 0)
        with patch.object(gpu, "_checkpoint_weight_gib", return_value=200.0):
            weight, _ = gpu._vllm_memory_per_rank(dict(
                model_name="qwen3.8-flash-next", max_seq_length=262144,
                hardware_profile="B200", strategy_inference_profile=True), 2)
            self.assertEqual(weight, 76.0)

    def test_b200_automatic_batch_scales_to_memory_and_real_work(self):
        roles = gpu.allocate_gpu_roles(range(8), "erdos")
        def resolve(profile, **overrides):
            cfg = dict(model_name="Qwen/Qwen3.8-27B", max_seq_length=262144,
                hardware_profile=profile, generation_backend="vllm", vllm_gpu_memory_utilization="auto",
                vllm_runtime_reserve_gib=16, vllm_tensor_parallel_size=1, vllm_pipeline_parallel_size=1,
                groups_per_step=8, group_size=512)
            cfg.update(overrides)
            gpu.resolve_memory_settings(cfg, roles, self.cards())
            return cfg
        full = resolve("B200")["gen_micro_batch"]
        self.assertGreater(full, 64)
        self.assertLessEqual(resolve("B200", groups_per_step=1, group_size=64)["gen_micro_batch"], 8)
        self.assertEqual(resolve("B200", gen_micro_batch=4)["gen_micro_batch"], 4)
        self.assertLessEqual(resolve("RTX6000")["gen_micro_batch"], 32)
        self.assertEqual(resolve("B200")["max_seq_length"], 262144)

    def test_b200_starts_workers_concurrently_like_rtx(self):
        events = []
        pool = workers.GenerationPool.__new__(workers.GenerationPool)
        pool.hardware_profile, pool.backend = "B200", "vllm"
        pool._staged_loading = False
        pool._start_worker = lambda rank: events.append(("start", rank))
        pool._wait_workers = lambda ids: events.append(("ready", tuple(ids)))
        pool._start_workers(range(8), initial=True)
        self.assertEqual(events, [("start", rank) for rank in range(8)] + [("ready", tuple(range(8)))])
        gate = object()
        phased = workers.PhasedVLLMGenerationPool(before_start=lambda: None, after_stop=lambda: None,
                    model_name="Qwen/Qwen3.8-27B", num_workers=8, hardware_profile="B200",
                    host_memory_admission=gate)
        self.assertIs(phased.host_memory_admission, gate)
        self.assertNotIn("host_memory_admission", phased._pool_kwargs)

    def test_ram_reserve_and_outstanding_work_are_not_spent_twice(self):
        context = SimpleNamespace(Value=lambda *a, **k: SimpleNamespace(value=0),
                                  Condition=threading.Condition, RLock=threading.RLock)
        gate = host.HostMemoryAdmission(context)
        info = dict(total=1200 * host.GIB, available=400 * host.GIB)
        with patch.object(host, "host_memory_info", return_value=info):
            self.assertEqual(gate.available_bytes(), 280 * host.GIB)
            with gate.reservation(100 * host.GIB, wait_for_external=False):
                self.assertEqual(gate.available_bytes(), 180 * host.GIB)
                info["available"] -= 100 * host.GIB
                self.assertEqual(gate.available_bytes(), 80 * host.GIB)
            self.assertEqual(gate.reserved_bytes, 0)
            with self.assertRaises(host.HostMemoryUnavailable):
                with gate.reservation(400 * host.GIB, wait_for_external=False):
                    self.fail("insufficient RAM must prevent allocation")

    def test_cgroup_ancestors_and_v1_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            membership = root / "membership"
            membership.write_text("0::/job/worker\n")
            (root / "job" / "worker").mkdir(parents=True)
            (root / "memory.max").write_text("max")
            (root / "job" / "memory.max").write_text(str(900 * host.GIB))
            (root / "job" / "memory.current").write_text(str(800 * host.GIB))
            self.assertIn((900*host.GIB, 100*host.GIB), host._cgroup_memory_bounds(root, membership))
            membership.write_text("5:memory:/job\n")
            (root / "memory" / "job").mkdir(parents=True)
            (root / "memory" / "job" / "memory.limit_in_bytes").write_text("1000")
            (root / "memory" / "job" / "memory.usage_in_bytes").write_text("1100")
            self.assertIn((1000, 0), host._cgroup_memory_bounds(root, membership))
        text = "MemTotal: 1258291200 kB\nMemAvailable: 1048576000 kB\n"
        with patch.object(Path, "read_text", return_value=text), patch.object(host, "_cgroup_memory_bounds", return_value=[(900*host.GIB, 100*host.GIB)]):
            self.assertEqual(host.host_memory_info(), dict(total=900*host.GIB, available=100*host.GIB))


if __name__ == "__main__":
    unittest.main()
