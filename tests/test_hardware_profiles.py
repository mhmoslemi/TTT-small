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
                self.assertEqual(hardware.local_checkpoint(name, "B200_full_ram"), str(path))
                self.assertEqual(hardware.local_checkpoint(name, "RTX6000"), name)
                self.assertEqual(hardware.checkpoint_identity(str(path)), name)
                self.assertEqual(coders._coder_model_profile(str(path)), coders._coder_model_profile(name))
                self.assertEqual(strategists.strategist_profile(str(path)), strategists.strategist_profile(name))
            explicit = "/my/custom/Qwen3-8B"
            self.assertEqual(hardware.local_checkpoint(explicit, "B200"), explicit)
            name = str(Path(folder) / "Qwen3-8B")
            for profile in ("RTX6000", "B200", "B200_full_ram"):
                self.assertEqual(coders._dual_resident_qwen3_8b_strategy_coder_pools(dict(
                    hardware_profile=profile, model_name=name, coder_model_name=name,
                    strategy_model_name=name, strategies=True, generation_backend="vllm")), profile != "B200")

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
            self.assertRegex(path.read_text(), r"(?m)^hardware_profile: (RTX6000|B200|B200_full_ram)[ \t]*(?:#.*)?$")
        source = (ROOT / "runtime/configuration.py").read_text()
        self.assertLess(source.index("restore_hardware_settings(merged"), source.index("# 3) CLI overlay"))

    def test_profile_changes_only_refresh_gpu_capacity(self):
        for previous in ("RTX6000", "B200"):
            for profile in ("RTX6000", "B200"):
                saved = dict(hardware_profile=previous, gen_micro_batch=7, logprob_chunk=128,
                             vllm_staged_loading=True, strategy_vllm_staged_loading=True,
                             strategy_vllm_persistent_workers=1, reload_coder_between_phases=True)
                cfg = dict(saved)
                hardware.restore_hardware_settings(cfg, dict(hardware_profile=profile), saved)
                expected = dict(saved, hardware_profile=profile)
                if previous != profile:
                    expected.update(gen_micro_batch="auto", logprob_chunk="auto",
                                    vllm_gpu_memory_utilization="auto",
                                    vllm_max_num_batched_tokens="auto")
                if profile == "B200":
                    expected["gen_micro_batch"] = "auto"
                self.assertEqual(cfg, expected)

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
            ("Qwen/Qwen3.8-Flash-Next-FP8", 262144, True, 2),
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
                self.assertEqual(cfg["vllm_max_num_batched_tokens"], 8192)
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
                    for hardware_profile in ("RTX6000", "B200", "B200_full_ram"):
                        actual = gpu._vllm_memory_per_rank(dict(
                            model_name=name, max_seq_length=262144,
                            hardware_profile=hardware_profile, strategy_inference_profile=True), tp)
                        extra = profile["ple_gib"] / tp if hardware_profile == "B200" else 0
                        self.assertEqual(actual, (expected[0] + extra, expected[1]))
                        options = strategists.strategist_engine_options(name, hardware_profile)
                        if profile["ple_gib"]:
                            self.assertEqual(options["engram_config"]["cpu_offload"], hardware_profile != "B200")
                        else:
                            self.assertEqual(options, strategists.strategist_engine_options(name))
        for hardware_profile in ("RTX6000", "B200"):
            cfg = SimpleNamespace(coder_model_profile="gpt-oss-120b", no_train=False,
                                  hardware_profile=hardware_profile)
            self.assertEqual(coders._coder_vllm_persistent_workers(cfg), 0)
        self.assertEqual(
            strategists.strategist_engine_options("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "B200"),
            strategists.strategist_engine_options("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"))

    def test_b200_fresh_and_resume_force_transient_coder_without_saved_flags(self):
        import os
        import sys
        from runtime import configuration

        preset = configuration.yaml.safe_load((ROOT / "configs/ac1.yaml").read_text())
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, AVAILABLE_GPUS="6,7", HF_HUB_OFFLINE="1"), \
             patch.object(gpu, "query_gpu_memory", return_value=self.cards()), \
             patch.object(gpu, "detect_attention_heads", return_value=32), \
             patch.object(configuration, "_isolated_evaluation_cpu_ids", return_value=[0, 1]), \
             patch.object(host, "host_memory_info", side_effect=AssertionError("no profile-only RAM preflight")), \
             redirect_stdout(io.StringIO()):
            os.environ.pop("TTT_HOST_MEMORY_RESERVE_GIB", None)
            config_path = Path(folder) / "ac1.yaml"
            argv = ["train_multy_CVaR.py", "--config", str(config_path),
                    "--advantage-mode", "spo-rs", "--fast", "--strategies", "--isolate-eval"]
            snapshots = {}
            for profile in ("RTX6000", "B200"):
                config_path.write_text(configuration.yaml.safe_dump(dict(preset, hardware_profile=profile)))
                for resume in (False, True):
                    if resume:
                        import json
                        saved = dict(merged, strategy_vllm_persistent_workers=0,
                                     reload_coder_between_phases=True,
                                     trainer_offload_dir="/old/low-ram/.offload")
                        (Path(folder) / "config.json").write_text(json.dumps(saved))
                    with patch.object(sys, "argv", argv + (["--resume", folder] if resume else [])):
                        cfg, merged = configuration.load_config()
                    self.assertEqual(cfg.vllm_tensor_parallel_size, 1)
                    self.assertEqual(cfg.num_training_gpus, 2)
                    self.assertEqual(cfg.strategy_vllm_persistent_workers, 0 if resume else None)
                    self.assertEqual(coders._coder_vllm_persistent_workers(cfg), 0 if resume or profile == "B200" else None)
                    self.assertEqual(getattr(cfg, "reload_coder_between_phases", False), resume)
                    self.assertIsNone(cfg.trainer_offload_dir)
                    self.assertNotIn("TTT_HOST_MEMORY_RESERVE_GIB", os.environ)
                    snapshots[profile, resume] = dict(vars(cfg))
                    snapshots[profile, resume].pop("hardware_profile")
            for resume in (False, True):
                rtx, b200 = snapshots["RTX6000", resume], snapshots["B200", resume]
                for key in ("gen_micro_batch", "strategy_gen_micro_batch"):
                    self.assertGreaterEqual(b200.pop(key), rtx.pop(key))
                self.assertEqual(rtx, b200)

    def test_run_copy_has_the_same_execution_flags_as_run2(self):
        normal = (ROOT / "run2.sh").read_text().split("exec python3 ", 1)[1].split('"$@"', 1)[0]
        copy = (ROOT / "run_copy.sh").read_text().split("exec python3 ", 1)[1].split('"$@"', 1)[0]
        self.assertEqual(normal, copy)
        self.assertNotIn("TTT_HOST_MEMORY_RESERVE_GIB", (ROOT / "run_copy.sh").read_text())

    def test_full_ram_resume_clears_low_ram_flags_and_uses_four_selected_gpus(self):
        import json
        import os
        import sys
        from runtime import configuration

        self.assertEqual(hardware.hardware_profile("b200_FULL_ram"), "B200_full_ram")
        preset = configuration.yaml.safe_load((ROOT / "configs/ac1.yaml").read_text())
        saved = dict(preset, parent_selection="puct", hardware_profile="B200", reload_coder_between_phases=True,
                     strategy_vllm_persistent_workers=0, trainer_offload_dir="/old/.offload",
                     training_gpu_ids="6,7", gpu_ids="6,7", num_training_gpus=2)
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, AVAILABLE_GPUS="2,3,4,5", HF_HUB_OFFLINE="1"), \
             patch.object(gpu, "query_gpu_memory", return_value=self.cards()), \
             patch.object(gpu, "detect_attention_heads", return_value=32), \
             patch.object(configuration, "_isolated_evaluation_cpu_ids", return_value=[0, 1]), \
             redirect_stdout(io.StringIO()):
            config = Path(folder) / "ac1.yaml"
            config.write_text(configuration.yaml.safe_dump(dict(preset, hardware_profile="B200_full_ram")))
            (Path(folder) / "config.json").write_text(json.dumps(saved))
            argv = ["train_multy_CVaR.py", "--config", str(config), "--resume", folder,
                    "--parent-selection", "puct", "--advantage-mode", "spo-rs", "--fast", "--strategies", "--isolate-eval"]
            with patch.object(sys, "argv", argv):
                cfg, merged = configuration.load_config()
            self.assertEqual(cfg.hardware_profile, "B200_full_ram")
            self.assertEqual(cfg.training_gpu_ids, "2,3,4,5")
            self.assertEqual(cfg.gpu_ids, "2,3,4,5")
            self.assertEqual(cfg.num_training_gpus, 4)
            self.assertEqual(cfg.parent_selection, "puct")
            self.assertEqual((cfg.groups_per_step, cfg.group_size), (8, 64))
            self.assertFalse(cfg.reload_coder_between_phases)
            self.assertIsNone(cfg.strategy_vllm_persistent_workers)
            self.assertIsNone(coders._coder_vllm_persistent_workers(cfg))
            self.assertIsNone(cfg.trainer_offload_dir)
            with patch.object(gpu, "query_gpu_memory", return_value={}), patch.object(sys, "argv", argv):
                fallback, _ = configuration.load_config()
            self.assertEqual(fallback.num_training_gpus, 4)
            self.assertEqual(fallback.vllm_tensor_parallel_size, 1)
            for name in strategists.STRATEGIST_ALIASES:
                self.assertEqual(strategists.strategist_engine_options(name, "B200_full_ram"),
                                 strategists.strategist_engine_options(name, "RTX6000"))
            memory = dict(total=1696*host.GIB, available=1400*host.GIB)
            self.assertEqual(host.host_memory_reserve_bytes(memory, "B200_full_ram"),
                             2 * host.GIB)

    def test_gpu_query_is_limited_to_launcher_devices(self):
        import os

        with patch.dict(os.environ, AVAILABLE_GPUS="6,7"), \
             patch.object(gpu.subprocess, "run", return_value=SimpleNamespace(
                 stdout="6, NVIDIA B200, 183359, 175000\n7, NVIDIA B200, 183359, 175000\n")) as query:
            self.assertEqual(set(gpu.query_gpu_memory()), {6, 7})
        self.assertIn("--id=6,7", query.call_args.args[0])

    def test_b200_automatic_batches_are_wider_without_changing_limits_or_rtx(self):
        roles = gpu.allocate_gpu_roles(range(8), "erdos")
        def resolve(profile, **overrides):
            cfg = dict(model_name="Qwen/Qwen3.8-27B", max_seq_length=262144,
                hardware_profile=profile, generation_backend="vllm", vllm_gpu_memory_utilization="auto",
                vllm_runtime_reserve_gib=16, vllm_tensor_parallel_size=1, vllm_pipeline_parallel_size=1,
                groups_per_step=8, group_size=512)
            cfg.update(overrides)
            gpu.resolve_memory_settings(cfg, roles, self.cards())
            return cfg
        for overrides in ({}, dict(groups_per_step=1, group_size=64), dict(frozen_strategy=True)):
            rtx = resolve("RTX6000", **overrides)
            for profile in ("B200", "B200_full_ram"):
                b200 = resolve(profile, **overrides)
                self.assertGreater(b200["gen_micro_batch"], rtx["gen_micro_batch"])
                self.assertLessEqual(b200["gen_micro_batch"], 128)
                for key in ("vllm_max_num_batched_tokens", "logprob_chunk",
                            "vllm_gpu_memory_utilization", "vllm_runtime_reserve_gib", "max_seq_length"):
                    self.assertEqual(b200[key], rtx[key], key)
        self.assertEqual(resolve("B200", gen_micro_batch=4)["gen_micro_batch"], 4)
        self.assertLessEqual(resolve("RTX6000")["gen_micro_batch"], 32)
        self.assertEqual(resolve("B200")["max_seq_length"], 262144)

        saved = dict(hardware_profile="B200_full_ram", gen_micro_batch=26)
        for requested in ("auto", 17):
            cfg = dict(saved)
            hardware.restore_hardware_settings(cfg, dict(hardware_profile="B200_full_ram",
                                                       gen_micro_batch=requested), saved)
            self.assertEqual(cfg["gen_micro_batch"], requested)

    def test_inference_and_evaluation_have_disjoint_physical_cores_on_b200_only(self):
        import os
        from runtime import configuration

        allowed = set(range(16)) | set(range(64, 80))
        def topology(path, *args, **kwargs):
            cpu = int(path.parent.parent.name.removeprefix("cpu")) % 64
            return str(cpu // 8 if path.name == "physical_package_id" else cpu % 8)
        with patch.object(os, "sched_getaffinity", return_value=allowed, create=True), \
             patch.object(os, "sched_setaffinity", create=True), \
             patch.object(Path, "read_text", topology):
            for profile in ("B200", "B200_full_ram", "RTX6000"):
                reserved = hardware.generation_cpu_ids(profile, 2)
                cfg = SimpleNamespace(hardware_profile=profile, generation_backend="vllm", gpu_ids="6,7")
                evaluation = configuration._isolated_evaluation_cpu_ids(cfg)
                self.assertFalse(set(reserved) & set(evaluation))
                self.assertEqual(set(reserved) | set(evaluation), allowed)
                self.assertEqual(len(reserved), 0 if profile == "RTX6000" else 12)
                for cpu in reserved:
                    self.assertIn((cpu + 64) % 128, reserved)
            with patch.object(Path, "read_text", side_effect=OSError("no topology")):
                self.assertEqual(hardware.generation_cpu_ids("B200_full_ram", 2), [])
            with patch.object(os, "sched_getaffinity", return_value={7}):
                self.assertEqual(hardware.generation_cpu_ids("B200", 2), [])

    def test_generation_stats_preserve_sampling_and_engine_memory_options(self):
        quiet = workers._vllm_engine_kwargs("Qwen/Qwen3.8-27B", 262144, False, 32, .9)
        stats = workers._vllm_engine_kwargs("Qwen/Qwen3.8-27B", 262144, False, 32, .9,
                                          enable_log_stats=True)
        self.assertTrue(quiet["disable_log_stats"])
        self.assertEqual(stats, dict(quiet, disable_log_stats=False))

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
