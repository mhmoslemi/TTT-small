"""Focused CPU-only contracts; no model, torch, or CUDA imports."""

import ast
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
from queue import Empty, Queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import generation.gen_workers as gen_workers
import runtime.gpu_runtime as gpu_runtime
import models.strategist_models as models
from config_validation import validate_problem_config


ROOT = Path(__file__).resolve().parents[1]


class StrategistModelsTests(unittest.TestCase):
    def test_native_token_limits_and_explicit_overrides(self):
        for name, context in (("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 131072),
                              ("qwen3.8-flash-next", 262144),
                              ("deepseek-v4-flash-0731", 262144)):
            self.assertEqual(models.resolve_strategy_token_limits(name),
                             (context, context))
            self.assertEqual(models.resolve_strategy_token_limits(name, "auto", 500),
                             (context, 500))
        self.assertEqual(models.resolve_strategy_token_limits("unknown", 4096, 2048),
                         (4096, 2048))
        self.assertEqual(models.resolve_strategy_token_limits("api", native_context=128000),
                         (128000, 128000))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"text_config": {"max_position_embeddings": 32768}}))
            self.assertEqual(models.resolve_strategy_token_limits(folder), (32768, 32768))
        for value in (True, 0, -1, "wrong"):
            with self.assertRaises(ValueError):
                models.resolve_strategy_token_limits("unused", value, 100)
        validate_problem_config(dict(problem="erdos", strategy_max_seq_length="auto",
                                     strategy_max_new_tokens="auto"), require_complete=False)

    def test_aliases_and_official_effort_levels(self):
        for alias, checkpoint in models.STRATEGIST_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(models.resolve_strategy_model(alias), checkpoint)
                self.assertEqual(models.resolve_strategy_model(checkpoint), checkpoint)
                profile = models.strategist_profile(checkpoint)
                self.assertEqual(models.resolve_strategy_effort(checkpoint, "auto"),
                                 profile["default_effort"])
                for effort in profile["efforts"]:
                    self.assertEqual(models.resolve_strategy_effort(checkpoint, effort), effort)
                with self.assertRaises(ValueError):
                    models.resolve_strategy_effort(checkpoint, "none")
        self.assertIsNone(models.strategist_profile("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"))
        self.assertEqual(models.resolve_strategy_effort("old-model", "auto"), "high")
        self.assertEqual(models.resolve_strategy_model("Qwen/Qwen3.8-Flash-Next-FP8"),
                         "Qwen/Qwen3.8-Flash-Next-FP8")
        with self.assertRaises(ValueError):
            models.resolve_strategy_effort("qwen3.8-flash-next", "high")
        with self.assertRaises(ValueError):
            models.resolve_strategy_effort("deepseek-v4-flash-0731", "medium")

    def test_native_renderers_preserve_effort_and_original_messages(self):
        messages = [{"role": "user", "content": "original task"}]
        tokenizer = Mock()
        for effort in ("low", "medium", "xhigh"):
            models.render_strategy_prompt(tokenizer, messages, "qwen3.8-flash-next",
                                          thinking=True, effort=effort)
            self.assertEqual(tokenizer.apply_chat_template.call_args.kwargs,
                             dict(tokenize=False, add_generation_prompt=True,
                                  enable_thinking=True, reasoning_effort=effort,
                                  preserve_thinking=True))
        def encode(turns, **kwargs):
            self.assertEqual(turns, messages)
            turns[0]["content"] = "mutated by encoder"
            return kwargs
        with patch.object(models, "_deepseek_encoder", return_value=encode):
            for effort in ("low", "high", "max"):
                rendered = models.render_strategy_prompt(
                    tokenizer, messages, "deepseek-v4-flash-0731",
                    thinking=True, effort=effort)
                self.assertEqual(rendered, dict(thinking_mode="thinking",
                                                reasoning_effort=effort))
        self.assertEqual(messages[0]["content"], "original task")

    def test_eight_cards_are_used_with_memory_fitting_replica_layouts(self):
        roles = gpu_runtime.allocate_gpu_roles(range(8), "erdos")
        cards = {i: gpu_runtime.GPUMemory(i, "RTX PRO 6000 Blackwell", 95.6, 95.6)
                 for i in range(8)}
        for alias, expected in (("qwen3.8-flash-next", (2, 4)),
                                ("deepseek-v4-flash-0731", (4, 2))):
            with self.subTest(alias=alias):
                context, _ = models.resolve_strategy_token_limits(alias)
                cfg = dict(model_name=models.resolve_strategy_model(alias),
                           strategy_inference_profile=True, max_seq_length=context,
                           generation_backend="vllm", gen_micro_batch="auto",
                           vllm_gpu_memory_utilization="auto",
                           vllm_runtime_reserve_gib=16, logprob_chunk="auto")
                heads = gpu_runtime.detect_attention_heads(cfg["model_name"])
                layout = gpu_runtime.derive_vllm_parallel_layout(cfg, roles, cards, heads)
                self.assertEqual((layout.tensor_parallel_size, layout.replicas), expected)
                self.assertLessEqual(layout.unsharded_stage_required_gib, layout.budget_gib)
                self.assertEqual(layout.replicas * layout.tensor_parallel_size, 8)
                single = gpu_runtime.align_vllm_layout_to_concurrency(
                    cfg, layout, 8, heads, 1)
                self.assertEqual((single.tensor_parallel_size, single.pipeline_parallel_size,
                                  single.replicas), (8, 1, 1))
                cfg["vllm_tensor_parallel_size"] = layout.tensor_parallel_size
                gpu_runtime.resolve_memory_settings(cfg, roles, cards)
                self.assertGreaterEqual(cfg["gen_micro_batch"], 8)
                self.assertEqual(cfg["max_seq_length"], context)
        cfg = dict(model_name="Qwen/Qwen3.8-27B", max_seq_length=131072,
                   vllm_runtime_reserve_gib=15, vllm_gpu_memory_utilization=.9)
        layout = gpu_runtime.derive_vllm_parallel_layout(cfg, roles, cards, 24)
        self.assertEqual((layout.tensor_parallel_size, layout.replicas), (1, 8))

    def test_engine_options_are_frozen_and_sm120_compatible(self):
        ordinary = gen_workers._vllm_engine_kwargs("Qwen/Qwen3.8-27B", 131072, False, 32, .9)
        self.assertTrue(ordinary["enable_lora"])
        self.assertNotIn("tokenizer_mode", ordinary)
        self.assertNotIn("generation_config", ordinary)
        self.assertNotIn("enable_flashinfer_autotune", ordinary)
        legacy = gen_workers._vllm_engine_kwargs(
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 131072, False, 32, .9,
            engine_options=models.strategist_engine_options(
                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"))
        self.assertEqual(legacy["generation_config"], "vllm")
        for alias in models.STRATEGIST_ALIASES:
            name = models.resolve_strategy_model(alias)
            context, _ = models.resolve_strategy_token_limits(name)
            options = models.strategist_engine_options(name)
            kwargs = gen_workers._vllm_engine_kwargs(
                name, context, False, 32, .9, tensor_parallel_size=4,
                engine_options=options)
            self.assertFalse(kwargs["enable_lora"])
            self.assertNotIn("max_lora_rank", kwargs)
            self.assertNotIn("fully_sharded_loras", kwargs)
            self.assertTrue(kwargs["enable_expert_parallel"])
            self.assertEqual(kwargs["max_model_len"], context)
            self.assertEqual(kwargs["generation_config"], "vllm")
            self.assertNotIn("speculative_config", kwargs)
            if alias.startswith("deepseek"):
                self.assertNotIn("enable_flashinfer_autotune", kwargs)
                self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
                self.assertEqual(kwargs["attention_config"], {"indexer_kv_dtype": "fp8"})
                self.assertEqual(kwargs["tokenizer_mode"], "deepseek_v4")
            else:
                self.assertEqual(kwargs["kv_cache_dtype"], "auto")
                self.assertEqual(kwargs["dtype"], "bfloat16")
                self.assertEqual(kwargs["quantization"], "modelopt")
                self.assertIs(kwargs["enable_flashinfer_autotune"], False)
                self.assertEqual(kwargs["engram_config"], {"cpu_offload": True})

    def test_startup_watchdog_allows_active_loading_but_rejects_stalls_and_errors(self):
        for scenario in ("log_activity", "ready_activity", "stalled", "missing_log", "dead", "error"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as folder:
                log_path = Path(folder) / "vllm.log"
                if scenario != "missing_log":
                    log_path.write_text("starting\n")
                pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
                pool.backend, pool.num_workers = "vllm", 2
                pool._startup_timeout, pool.vllm_log_path = 3., str(log_path)
                pool._ready_worker_ids, pool._sleep_capable_workers = set(), set()
                pool._async_worker_ids = set()
                pool.procs = [SimpleNamespace(exitcode=None) for _ in range(2)]
                pool.shutdown = Mock()
                elapsed = [0.]

                def receive(timeout):
                    elapsed[0] += timeout
                    tick = int(elapsed[0])
                    if scenario == "log_activity":
                        if tick in (2, 4):
                            with log_path.open("a") as handle:
                                handle.write("compiling\n")
                        if tick == 5:
                            return "ready", 0, "sleep:1;async"
                        if tick == 6:
                            return "ready", 1, "ready"
                    if scenario == "ready_activity":
                        if tick == 2:
                            return "ready", 0, "ready"
                        if tick == 4:
                            return "ready", 1, "ready"
                    if scenario == "dead":
                        pool.procs[1].exitcode = 1
                    if scenario == "error":
                        return "error", 1, "initialization failed"
                    raise Empty

                pool._ready_queue = SimpleNamespace(get=receive)
                with patch.object(gen_workers.time, "monotonic", side_effect=lambda: elapsed[0]), \
                        redirect_stdout(io.StringIO()):
                    if scenario in ("log_activity", "ready_activity"):
                        pool._wait_workers((0, 1))
                        self.assertEqual(pool._ready_worker_ids, {0, 1})
                        self.assertGreater(elapsed[0], pool._startup_timeout)
                        pool.shutdown.assert_not_called()
                        if scenario == "log_activity":
                            self.assertEqual(pool._sleep_capable_workers, {0})
                            self.assertEqual(pool._async_worker_ids, {0})
                    else:
                        error = TimeoutError if scenario in ("stalled", "missing_log") else RuntimeError
                        message = ("no progress" if error is TimeoutError else
                                   "exited during startup" if scenario == "dead" else "initialization failed")
                        with self.assertRaisesRegex(error, message):
                            pool._wait_workers((0, 1))
                        pool.shutdown.assert_called_once_with()
                        self.assertEqual(elapsed[0], 3. if error is TimeoutError else 1.)

    def test_qwen_qsa_native_kv_precision_matches_memory_planning(self):
        for name in ("qwen3.8-flash-next", "Qwen/Qwen3.8-Flash-Next",
                     "Qwen/Qwen3.8-Flash-Next-FP8"):
            for tp, expected_cache_gib in ((1, 8.0), (2, 4.0), (4, 4.0), (8, 4.0)):
                kwargs = gen_workers._vllm_engine_kwargs(
                    models.resolve_strategy_model(name), 262144, False, 32, .9,
                    tensor_parallel_size=tp,
                    engine_options=models.strategist_engine_options(name))
                self.assertEqual((kwargs["dtype"], kwargs["kv_cache_dtype"]),
                                 ("bfloat16", "auto"))
                self.assertEqual(kwargs["max_model_len"], 262144)
                _, cache_gib = models.strategist_memory_per_rank(name, tp, 262144)
                self.assertEqual(cache_gib, expected_cache_gib)
        _, deepseek_cache_gib = models.strategist_memory_per_rank(
            "deepseek-v4-flash-0731", 4, 262144)
        self.assertEqual(deepseek_cache_gib, 2.0)

    def test_new_pool_loading_is_host_memory_admitted(self):
        events = []
        @contextmanager
        def reservation(size, **kwargs):
            self.assertGreater(size, 400 * 2**30)
            self.assertFalse(kwargs["wait_for_external"])
            events.append("reserve")
            yield
            events.append("release")
        class Pool:
            num_workers, sleep_supported, persistent_workers = 4, True, 4
            def __init__(self, **kwargs):
                events.append("load")
        name = models.resolve_strategy_model("qwen3.8-flash-next")
        with patch.object(gen_workers, "GenerationPool", Pool):
            pool = gen_workers.PhasedVLLMGenerationPool(
                before_start=lambda: events.append("offload"), after_stop=lambda: None,
                model_name=name, num_workers=8, vllm_tensor_parallel_size=2,
                vllm_engine_options=models.strategist_engine_options(name),
                host_memory_admission=SimpleNamespace(reservation=reservation))
            pool.prepare()
        self.assertEqual(events, ["offload", "reserve", "load", "release"])

    def test_profile_does_not_override_yaml_sampling_or_token_limits(self):
        source = ast.parse((ROOT / "models/coder_profiles.py").read_text())
        method = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_apply_strategy_model_profile")
        scope = dict(strategist_profile=models.strategist_profile,
                     _profile_default=lambda cfg, key, value, explicit:
                         cfg.update({key: value}) if key not in explicit else None)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "profile", "exec"), scope)
        for alias in models.STRATEGIST_ALIASES:
            cfg = dict(strategy_max_seq_length=131072, strategy_max_new_tokens=131072,
                       strategy_temperature=1., strategy_top_p=1., strategy_sampling_top_k=0,
                       strategy_sampling_min_p=0., strategy_reasoning_effort="low",
                       strategy_thinking=True, strategy_vllm_sleep_level=1,
                       strategy_vllm_staged_loading=False)
            before = dict(cfg)
            with redirect_stdout(io.StringIO()):
                scope[method.name](cfg, alias, set(cfg))
            self.assertEqual(cfg, before)

    def test_legacy_thinking_profile_keeps_unfiltered_sampling(self):
        source = ast.parse((ROOT / "models/coder_profiles.py").read_text())
        methods = [n for n in source.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"_profile_default", "_apply_qwen3_8b_thinking_sampling"}]
        scope = dict(_is_qwen3_8b_sampling_model=lambda name: name == "Qwen/Qwen3-8B")
        exec(compile(ast.Module(body=methods, type_ignores=[]), "sampling", "exec"), scope)
        cfg = dict(model_name="other", strategies=True,
                   strategy_model_name="Qwen/Qwen3-8B", strategy_thinking=True,
                   strategy_temperature=1., strategy_top_p=1., strategy_min_p=0.)
        with redirect_stdout(io.StringIO()):
            scope["_apply_qwen3_8b_thinking_sampling"](cfg)
        self.assertEqual(tuple(cfg[key] for key in (
            "strategy_temperature", "strategy_top_p", "strategy_sampling_top_k",
            "strategy_sampling_min_p")), (1., 1., 0, 0.))

    def test_chain_retries_preserve_resolved_sampling_and_budget(self):
        pool = gen_workers.GenerationPool.__new__(gen_workers.GenerationPool)
        pool.num_workers = 2
        pool.tensor_parallel_size = pool.pipeline_parallel_size = 1
        pool.gen_micro_batch = 8
        pool.procs = [SimpleNamespace(exitcode=None)] * 2
        pool.result_queue = Queue()
        calls = []
        class Tasks:
            def __init__(self, rank):
                self.rank = rank
            def put(self, task):
                _, adapter, jobs, options = task
                assert adapter is None
                calls.append(options)
                pool.result_queue.put((self.rank, jobs[0][0], [("strategy", [])]))
        pool.task_queues = [Tasks(rank) for rank in range(2)]
        bar = SimpleNamespace(update=lambda *_: None, close=lambda: None)
        with patch.object(gen_workers, "make_progress_bar", return_value=bar), \
                redirect_stdout(io.StringIO()):
            result = pool.run_sequential_chains(
                num_chains=2, num_stages=2, adapter_path=None,
                prompt_builder=lambda *args: "native prompt",
                result_handler=lambda chain, stage, text, attempt:
                    chain == 0 and stage == 0 and attempt == 0,
                max_retries=1, max_new_tokens=262144,
                temperature=1., top_p=1., top_k=0, min_p=0.)
        self.assertEqual(result, dict(completed=4, retries=1))
        self.assertEqual(len(calls), 5)
        for options in calls:
            self.assertEqual(tuple(options[key] for key in (
                "max_new_tokens", "temperature", "top_p", "top_k", "min_p")),
                (262144, 1., 1., 0, 0.))

    def test_presets_and_runtime_preflight(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            text = path.read_text()
            self.assertIn("# strategy_model_name: qwen3.8-flash-next", text)
            self.assertIn("# strategy_model_name: deepseek-v4-flash-0731", text)
            self.assertIn("strategy_reasoning_effort: auto", text)
            for setting in ("strategy_max_seq_length: auto", "strategy_max_new_tokens: auto",
                            "strategy_temperature: 1", "strategy_top_p: 1",
                            "strategy_top_k: 0", "strategy_min_p: 0"):
                self.assertIn(setting, text)
        validate_problem_config(dict(problem="erdos", strategy_reasoning_effort="auto"),
                                require_complete=False)
        with patch.object(models.metadata, "version", return_value="0.28.0"):
            models.validate_strategist_runtime("old-model")
            with self.assertRaises(RuntimeError):
                models.validate_strategist_runtime("qwen3.8-flash-next")
        with patch.object(models.metadata, "version", return_value="0.30.0"), \
                patch.object(models, "_deepseek_encoder") as encoder:
            models.validate_strategist_runtime("deepseek-v4-flash-0731", prepare_encoder=True)
            encoder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
