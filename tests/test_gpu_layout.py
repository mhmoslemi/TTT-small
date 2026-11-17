"""CPU regressions for GPT-OSS rollout startup memory admission."""

import json
from pathlib import Path
import sys

import pytest

import runtime.gpu_runtime as gpu_runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trimul_evaluator(monkeypatch):
    import importlib.util

    directory = ROOT / "examples/gpu_mode/lib/bioml/trimul"
    modules = {}
    for name in ("task", "utils", "reference", "kernel_check", "eval"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_trimul_mask_flags_reach_input_generation(trimul_evaluator, monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    import yaml

    task = yaml.safe_load((ROOT / "examples/gpu_mode/lib/bioml/trimul/task.yml").read_text())
    assert {item["name"]: item["source"] for item in task["files"]}["kernel_check.py"] == "kernel_check.py"
    cases = task["tests"] + task["benchmarks"]
    path = tmp_path / "test_cases.txt"
    path.write_text("\n".join(";".join(f"{key}:{val}" for key, val in case.items())
                              for case in cases))
    parsed = trimul_evaluator["eval"].get_test_cases(str(path), None)
    assert [test.args for test in parsed] == cases
    assert all(isinstance(test.args["nomask"], bool) for test in parsed)

    # Use the real generator on CPU; only redirect its CUDA allocations.
    reference = trimul_evaluator["reference"]
    proxy = SimpleNamespace(**vars(torch))
    proxy.Generator = lambda **kwargs: torch.Generator()
    proxy.randn = lambda *args, **kwargs: torch.randn(*args, **{**kwargs, "device": "cpu"})
    monkeypatch.setattr(reference, "torch", proxy)
    unmasked = reference.generate_input(**parsed[0].args)[1]
    masked = reference.generate_input(**parsed[1].args)[1]
    assert torch.all(unmasked == 1)
    assert torch.any(masked == 0) and torch.any(masked == 1)
    with pytest.raises(TypeError, match="nomask must be a boolean"):
        reference.generate_input(**{**parsed[1].args, "nomask": "False"})


@pytest.mark.parametrize("mode,passes", [
    ("useful", True), ("unused", False), ("dummy", False),
    ("copy", False), ("warmup", False), ("probe_error", False),
])
def test_trimul_requires_contributing_triton_work(trimul_evaluator, monkeypatch, mode, passes):
    from types import SimpleNamespace
    import torch

    class JITFunction:
        fn = SimpleNamespace(__module__="submission")
        __name__ = "test_kernel"

        def run(self, x, out, *, grid, warmup):
            ir = "tt.load; tt.store" if mode == "copy" else "arith.mulf %x, %x : f32"
            if not warmup:
                out.copy_(x if mode == "copy" else x * 2)
            return SimpleNamespace(asm={"ttir": ir})

    original = JITFunction.run
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", SimpleNamespace(JITFunction=JITFunction))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    jit = JITFunction()

    def custom(data):
        x = data[0]
        if mode == "unused":
            return x * 2
        out = torch.empty_like(x)
        jit.run(x * 2 if mode == "copy" else x, out, grid=(1,), warmup=mode == "warmup")
        if mode == "probe_error" and torch.isnan(out).any():
            raise RuntimeError("probe failed")
        return x * 2 if mode in ("dummy", "warmup") else out

    custom.__module__ = "submission"

    def check(data, output):
        torch.rand(1)  # The real reference also initializes module parameters.
        return torch.allclose(output, data[0] * 2, rtol=.02, atol=.02), "wrong output"

    rng = torch.get_rng_state()
    good, message = trimul_evaluator["kernel_check"].check_triton_contribution(
        custom, (torch.arange(1., 9.),), trimul_evaluator["eval"]._clone_data, check)
    assert good is passes, message
    assert JITFunction.run is original
    assert torch.equal(rng, torch.get_rng_state())
    if mode == "probe_error":
        assert "probe failed" in message


def test_trimul_contribution_on_real_cuda(trimul_evaluator):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Triton in the evaluator environment")
    import triton
    import triton.language as tl

    @triton.jit
    def twice(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + idx, idx < N, other=0)
        tl.store(Y + idx, value * 2, idx < N)

    @triton.jit
    def copy(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + idx, tl.load(X + idx, idx < N, other=0), idx < N)

    for mode in ("useful", "dummy", "copy"):
        def custom(data):
            x = data[0]
            out = torch.empty_like(x)
            if mode == "copy":
                copy[(4,)](x * 2, out, x.numel(), 256)
            else:
                twice[(4,)](x, out, x.numel(), 256)
            return x * 2 if mode == "dummy" else out

        good, message = trimul_evaluator["kernel_check"].check_triton_contribution(
            custom, (torch.arange(1., 1025., device="cuda"),),
            trimul_evaluator["eval"]._clone_data,
            lambda data, out: (torch.allclose(out, data[0] * 2), "wrong output"))
        assert good is (mode == "useful"), message


def test_trimul_audit_diagnostics_are_saved():
    from types import SimpleNamespace
    from problems.gpu_mode import collect_logs

    run = SimpleNamespace(passed=False, stderr="", stdout="", result={
        "evaluator": "trimul-masked-triton-v1",
        "test.0.error": "No numerical submission Triton kernel executed",
    })
    logs = collect_logs(SimpleNamespace(runs={"test": SimpleNamespace(run=run)}))
    assert "trimul-masked-triton-v1" in logs
    assert "No numerical submission Triton kernel executed" in logs


def test_trimul_alone_shares_its_evaluator_with_generation():
    trimul = gpu_runtime.allocate_gpu_roles([6, 7], "gpu_mode", "trimul")
    assert trimul.generation == [6, 7]
    assert trimul.evaluation == 7
    assert trimul.evaluation_shares_generation

    mla = gpu_runtime.allocate_gpu_roles(
        [6, 7], "gpu_mode", "mla_decode_nvidia")
    assert mla.generation == [6]
    assert mla.evaluation == 7
    assert not mla.evaluation_shares_generation


@pytest.mark.parametrize("resume", [False, True])
def test_trimul_live_config_uses_one_evaluator_per_available_gpu(monkeypatch, tmp_path, resume):
    from runtime.configuration import load_config

    monkeypatch.setenv("AVAILABLE_GPUS", "6,7")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", lambda: {
        gpu: gpu_runtime.GPUMemory(physical_id=gpu, name="NVIDIA B200",
                                  total_gib=179., free_gib=179.) for gpu in (6, 7)})
    argv = ["train_multy_CVaR.py", "--config", str(ROOT / "configs/gpu_mode_trimul.yaml"),
            "--strategies", "--advantage-mode", "spo-rs", "--fast"]
    if resume:
        (tmp_path / "config.json").write_text(json.dumps({"reward_workers": 1}))
        argv += ["--resume", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = load_config()
    assert cfg.reward_workers == 2
    assert cfg.gpu_ids == "6,7"
    assert cfg.evaluation_shares_generation
    assert cfg.vllm_tensor_parallel_size == cfg.vllm_pipeline_parallel_size == 1


def test_trimul_b200_benchmark_identity_is_exact():
    from problems.gpu_mode import benchmark_environment_issue

    assert benchmark_environment_issue(
        "NVIDIA B200", (10, 0), "3.3.1", "B200", "3.3.1") is None
    assert "expected NVIDIA B200" in benchmark_environment_issue(
        "NVIDIA H100", (9, 0), "3.3.1", "B200", "3.3.1")
    assert "expected Triton 3.3.1" in benchmark_environment_issue(
        "NVIDIA B200", (10, 0), "3.4.0", "B200", "3.3.1")


def test_trimul_evaluations_use_both_gpus_without_overlapping_on_one(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    import threading
    import problems.gpu_mode as gpu_mode

    config = dict(kernel_python=sys.executable, problem_type="trimul",
                  kernel_gpu_id=7, gpu_ids="6,7", evaluation_shares_generation=True)
    problem = gpu_mode.GpuMode(config)
    assert problem.kernel_gpu_ids == [6, 7]
    active = set()
    calls = []
    verified_per_gpu = {6: [], 7: []}
    lock = threading.Lock()
    both_running = threading.Barrier(2)

    @contextmanager
    def lease(gpu_id, *args, **kwargs):
        with lock:
            assert gpu_id not in active
            active.add(gpu_id)
        try:
            yield
        finally:
            with lock:
                active.remove(gpu_id)

    def evaluate(code, lib_dir, task_yaml, problem_type, log_chars, timeout, gpu_id, *args):
        with lock:
            calls.append(gpu_id)
            verified_per_gpu[gpu_id].append(args[-1])
            call_number = len(calls)
        both_running.wait(timeout=5)
        # A failed benchmark must also return its GPU slot for the next wave.
        if call_number <= 2:
            return dict(ok=False, msg="Failed to pass test cases.", environment_verified=True)
        return dict(ok=True, score_us=123., environment_verified=True)

    monkeypatch.setattr("runtime.gpu_lease.gpu_lease", lease)
    monkeypatch.setattr(gpu_mode, "run_eval_with_timeout", evaluate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: problem.compute_reward(
            "```python\n@triton.jit\ndef kernel(): pass\n```", None, 900), range(4)))
    assert calls[:2] == [6, 7] or calls[:2] == [7, 6]
    assert sorted(calls) == [6, 6, 7, 7]
    assert verified_per_gpu == {6: [False, True], 7: [False, True]}
    assert not active
    assert [result.valid for result in results] == [False, False, True, True]
    assert sorted(problem._kernel_gpu_slots.queue) == [6, 7]

    # An explicitly selected standalone device still stays single-GPU.
    config["evaluation_shares_generation"] = False
    assert gpu_mode.GpuMode(config).kernel_gpu_ids == [7]


def test_trimul_evaluator_has_a_private_work_directory(monkeypatch, tmp_path):
    from types import ModuleType, SimpleNamespace
    import problems.gpu_mode as gpu_mode

    modules = {name: ModuleType(name) for name in (
        "libkernelbot", "libkernelbot.task", "libkernelbot.run_eval",
        "libkernelbot.submission", "libkernelbot.consts")}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    task_path = ROOT / "examples/gpu_mode/lib/bioml/trimul/task.yml"
    modules["libkernelbot.task"].make_task_definition = lambda path: (
        SimpleNamespace(task=path) if Path(path).is_file() else None)
    modules["libkernelbot.task"].build_task_config = lambda **options: options
    modules["libkernelbot.consts"].SubmissionMode = SimpleNamespace(LEADERBOARD="leaderboard")
    modules["libkernelbot.submission"].compute_score = lambda *args, **kwargs: .000123
    monkeypatch.setattr(gpu_mode, "ensure_python3_on_path", lambda _: sys.executable)
    probes = []

    def probe(python):
        probes.append(python)
        return dict(gpu="NVIDIA B200", capability=(10, 0), triton="3.3.1")

    monkeypatch.setattr(gpu_mode, "benchmark_environment_from_python", probe)
    monkeypatch.setattr(gpu_mode, "collect_logs", lambda *_: "")
    monkeypatch.setenv("TTT_FAST_RUNTIME_DIR", str(tmp_path))
    monkeypatch.chdir(ROOT)
    workdirs = []

    def run(config):
        workdir = Path.cwd()
        assert workdir != ROOT
        assert not (workdir / "submission.py").exists()
        (workdir / "submission.py").write_text(config["submission_content"])
        workdirs.append(workdir)
        return SimpleNamespace(success=True, runs={
            key: SimpleNamespace(run=SimpleNamespace(passed=True))
            for key in ("test", "leaderboard")})

    modules["libkernelbot.run_eval"].run_config = run
    sink = SimpleNamespace(send=lambda result: results.append(result), close=lambda: None)
    results = []
    for code in ("first", "second"):
        gpu_mode._eval_child(sink, code, "examples/gpu_mode/lib", str(task_path),
                             "trimul", 4000, expected_gpu="B200", expected_triton="3.3.1",
                             environment_verified=(code == "second"))
        assert Path.cwd() == ROOT
    assert len(set(workdirs)) == 2
    assert all(not path.exists() for path in workdirs)
    assert all(result["ok"] and result["score_us"] == pytest.approx(123.) for result in results)
    assert len(probes) == 1
    assert all(result["environment_verified"] for result in results)


@pytest.mark.parametrize("outcomes,expected_calls,exit_code", [
    ([False, True, True], 1, 112), ([True, True, True], 3, 0)])
def test_trimul_correctness_stops_at_first_failure(outcomes, expected_calls, exit_code):
    import ast
    from types import SimpleNamespace

    path = ROOT / "examples/gpu_mode/lib/bioml/trimul/eval.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "run_testing")
    calls, logs = [], []

    def check(pool, test):
        calls.append(test.index)
        return outcomes[test.index], "correctness diagnostic"

    scope = dict(multiprocessing=SimpleNamespace(Pool=object), TestCase=object,
                 PopcornOutput=object, run_single_test=check)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    result = scope["run_testing"](
        SimpleNamespace(log=lambda *values: logs.append(values)), None,
        [SimpleNamespace(index=i, spec=str(i)) for i in range(len(outcomes))])
    assert result == exit_code
    assert calls == list(range(expected_calls))
    assert logs[-1] == ("check", "pass" if exit_code == 0 else "fail")
    if exit_code:
        assert ("test.0.error", "correctness diagnostic") in logs


def test_trimul_preflight_cache_resets_at_batch_change_and_worker_failure(monkeypatch):
    import problems.gpu_mode as gpu_mode

    problem = gpu_mode.GpuMode(dict(kernel_python=sys.executable, kernel_gpu_id=7,
                                    gpu_type="B200", problem_type="trimul"))
    verified_args = []
    result = dict(ok=True, score_us=123., environment_verified=True)

    def evaluate(*args):
        verified_args.append(args[-1])
        return dict(result)

    monkeypatch.setattr(gpu_mode, "run_eval_with_timeout", evaluate)
    response = "```python\n@triton.jit\ndef kernel(): pass\n```"
    reward = lambda: problem.compute_reward(response, None, 900)
    assert reward().valid
    assert reward().valid
    result.clear()
    result.update(ok=False, msg="kernel_eval_timeout after 900s (process killed)")
    assert not reward().valid
    result.update(ok=True, score_us=123., environment_verified=True)
    assert reward().valid
    problem.begin_evaluation_batch()
    assert reward().valid
    problem.triton_version = "changed-version"
    result.clear()
    result.update(ok=False, msg="benchmark environment mismatch: wrong Triton")
    assert not reward().valid
    assert not reward().valid
    assert verified_args == [False, True, True, False, False, False, False]
    assert not problem._verified_kernel_environments


def test_gpu_mode_can_select_a_dedicated_kernel_interpreter():
    import sys
    from problems.gpu_mode import ensure_python3_on_path

    selected = ensure_python3_on_path(sys.executable)
    assert selected is not None
    assert Path(selected).resolve() == Path(sys.executable).resolve()


def test_kernel_python_is_checkout_relative_and_missing_path_fails_early(monkeypatch, tmp_path):
    import os
    from problems.gpu_mode import GpuMode, resolve_kernel_python

    configured = os.path.relpath(sys.executable, ROOT)
    monkeypatch.chdir(tmp_path)
    assert resolve_kernel_python(configured) == os.path.abspath(sys.executable)
    with pytest.raises(FileNotFoundError, match="correct kernel_python"):
        GpuMode({"kernel_python": str(tmp_path / "missing-env/bin/python3")})


def test_saved_kernel_recheck_uses_original_responses_and_separate_results(tmp_path):
    from types import SimpleNamespace
    from eval_saved_kernels import evaluate_saved
    from problems.base import RewardResult

    step = tmp_path / "step00"
    step.mkdir()
    source = step / "step00_group00_rollout000.txt"
    response = "Some reasoning.\n```python\nimport triton\n# final kernel\n```"
    source.write_text(response)
    meta = source.with_suffix(".meta.json")
    meta.write_text('{"reward": 0, "msg": "old infrastructure failure"}')
    source.with_suffix(".prompt.txt").write_text("must not be evaluated")
    source.with_suffix(".strategy.txt").write_text("must not be evaluated")
    calls = []

    def reward(text, parent, timeout_s):
        calls.append((text, timeout_s))
        return RewardResult(valid=True, parsed=True, ran=True,
                            raw_score=123., reward=1500 / 123., msg="runtime_us=123")

    output = tmp_path / "check"
    rows = evaluate_saved(SimpleNamespace(compute_reward=reward), step, output, 900)
    assert calls == [(response, 900)]
    assert rows[0]["raw_score"] == 123.
    assert json.loads((output / "results.json").read_text()) == rows
    assert (output / "step00_group00_rollout000.py").read_text() == "import triton\n# final kernel"
    assert source.read_text() == response
    assert json.loads(meta.read_text())["reward"] == 0


def test_trimul_config_is_512_max_effort_gpt_oss_rollouts():
    import yaml
    from models.coder_profiles import (_coder_effort_entries,
                                       _coder_model_profile)
    from types import SimpleNamespace

    config = yaml.safe_load((ROOT / "configs/gpu_mode_trimul.yaml").read_text())
    assert config["kernel_python"] == "../kenrel_env/bin/python3"
    assert config["model_name"] == config["coder_model_name"] == "openai/gpt-oss-120b"
    assert config["coder_reasoning_effort"] == "high"
    assert config["groups_per_step"] * config["group_size"] == 512
    assert (config["strategies_per_parent"]
            * config["programs_per_strategy"] == config["group_size"])
    assert _coder_model_profile(config["coder_model_name"])["reasoning_effort"] == "high"

    cfg = SimpleNamespace(
        coder_template_kind="gpt-oss", coder_reasoning_effort="high",
        coder_thinking_budget=0)
    sources = range(config["groups_per_step"]
                    * config["strategies_per_parent"])
    entries = _coder_effort_entries(
        [], sources,
        [config["pilot_programs_per_strategy"]] * len(sources),
        cfg, 0, "pilot")
    assert sum(count for _, count, _ in entries) == 384
    assert {effort for _, _, effort in entries} == {"high"}
    phase2 = _coder_effort_entries(
        [], sources,
        [config["programs_per_strategy"]
         - config["pilot_programs_per_strategy"]] * len(sources),
        cfg, 0, "adaptive")
    assert sum(count for _, count, _ in phase2) == 128
    assert {effort for _, _, effort in phase2} == {"high"}


def blackwell_cards(total_gib=94.97):
    return {
        gpu: gpu_runtime.GPUMemory(
            physical_id=gpu,
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            total_gib=total_gib, free_gib=total_gib)
        for gpu in range(8)
    }


@pytest.mark.parametrize("total_gib,quantization,expected_tp", [
    (94.97, "", 1), (141.0, "", 1), (48.0, "", 4),
    (94.97, "bitsandbytes", 2),
])
def test_gpt_oss_layout_and_worker_share_bounded_startup_contract(
        total_gib, quantization, expected_tp):
    from generation.gen_workers import _vllm_engine_kwargs

    cfg = dict(model_name="openai/gpt-oss-120b", max_seq_length=131072,
               vllm_gpu_memory_utilization=.9,
               vllm_runtime_reserve_gib=5., allow_gpt_oss_tp1=True,
               vllm_quantization=quantization)
    roles = gpu_runtime.allocate_gpu_roles(range(8), "erdos")
    layout = gpu_runtime.derive_vllm_parallel_layout(
        cfg, roles, blackwell_cards(total_gib), 64)
    assert layout.tensor_parallel_size == expected_tp
    assert layout.pipeline_parallel_size == 1
    assert layout.replicas * expected_tp == 8
    kwargs = _vllm_engine_kwargs(
        cfg["model_name"], 131072, False, 32, .9, quantization=quantization,
        tensor_parallel_size=expected_tp, enable_sleep_mode=True)
    assert kwargs["enable_lora"] and kwargs["enable_sleep_mode"]
    if quantization == "":
        assert kwargs["worker_cls"] == "vllm_compat.gpt_oss_worker.GptOssWorker"
    else:
        assert "worker_cls" not in kwargs


@pytest.mark.parametrize("resume", [False, True])
def test_erdos_gpt_oss_config_uses_eight_coders(monkeypatch, tmp_path, resume):
    import train_multy_CVaR as training
    import yaml

    monkeypatch.setenv("AVAILABLE_GPUS", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", blackwell_cards)
    # Pin this fixture's effort explicitly; the user's profile may be changed
    # independently of GPU placement. Also verify YAML retains precedence.
    config = yaml.safe_load((ROOT / "configs/erdos.yaml").read_text())
    config["coder_reasoning_effort"] = "low"
    config_path = tmp_path / "erdos.yaml"
    config_path.write_text(yaml.safe_dump(config))
    argv = ["train_multy_CVaR.py", "--config", str(config_path),
            "--backend", "unsloth", "--advantage-mode", "spo-rs", "--fast",
            "--strategies", "--coder-retry", "--fused-long-attention"]
    if resume:
        # Resume recomputes topology instead of retaining the TP=2 workaround.
        (tmp_path / "config.json").write_text(json.dumps({
            "allow_gpt_oss_tp1": True, "vllm_tensor_parallel_size": 2,
            "vllm_pipeline_parallel_size": 1,
        }))
        argv += ["--resume", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = training.load_config()
    assert cfg.vllm_tensor_parallel_size == 1
    assert cfg.vllm_pipeline_parallel_size == 1
    assert cfg.num_gpus == cfg.num_training_gpus == 8
    assert cfg.max_seq_length == cfg.max_new_tokens == 131072
    assert cfg.coder_reasoning_effort == "low"
    assert cfg.strategy_backend == "local" and not cfg.api_strategy
    assert cfg.training_layout == "replicated"
    assert cfg.gen_micro_batch == 0
    assert cfg.vllm_max_num_batched_tokens == 0


def test_qwen_keeps_original_worker_and_parallel_layout():
    from generation.gen_workers import _vllm_engine_kwargs

    cfg = dict(model_name="Qwen/Qwen3.8-27B", max_seq_length=131072,
               vllm_gpu_memory_utilization=.9, vllm_runtime_reserve_gib=15.)
    roles = gpu_runtime.allocate_gpu_roles(range(8), "erdos")
    layout = gpu_runtime.derive_vllm_parallel_layout(
        cfg, roles, blackwell_cards(), 24)
    assert (layout.tensor_parallel_size, layout.pipeline_parallel_size,
            layout.replicas) == (1, 1, 8)
    kwargs = _vllm_engine_kwargs(cfg["model_name"], 131072, False, 32, .9)
    assert "worker_cls" not in kwargs


def test_bounded_repack_preserves_bytes_and_uses_only_one_expert_of_scratch():
    import torch
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.arange(4 * 16 * 16, dtype=torch.int32).to(torch.uint8).view(4, 16, 16)
    original = weight.clone()
    calls = []
    sentinel_perm = object()

    def packing(experts, size_n, size_k, perm, is_a_8bit):
        assert experts.shape[0] == 1
        assert (size_n, size_k, perm, is_a_8bit) == (16, 32, sentinel_perm, False)
        calls.append(experts.shape[0])
        # A deterministic stand-in for the CUDA packing kernel, deliberately
        # changing both byte order and dtype/shape like the actual kernel.
        return experts.flatten().flip(0).contiguous().view(torch.int32).view(1, 4, 16)

    expected = torch.stack([
        expert.flatten().flip(0).contiguous().view(torch.int32).view(4, 16)
        for expert in original])
    actual = repack_experts_in_place(
        weight, 16, 32, sentinel_perm, False, repack=packing)
    assert torch.equal(actual, expected)
    assert actual.data_ptr() == weight.data_ptr()
    assert calls == [1] * 4


def test_bounded_repack_rejects_changed_storage_contract_before_overwriting():
    import torch
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.ones(2, 16, 16, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="storage contract"):
        repack_experts_in_place(
            weight, 16, 32, None, False,
            repack=lambda *args: torch.zeros(1, 16, 32, dtype=torch.uint8))
    assert torch.all(weight == 1)


@pytest.mark.parametrize("fail", [False, True])
def test_custom_worker_scopes_repacking_and_allocator_to_startup(monkeypatch, fail):
    from contextlib import contextmanager
    from types import SimpleNamespace

    def original(weight, size_n, size_k, perm, is_a_8bit):
        raise AssertionError("not called in this wiring check")

    marlin = SimpleNamespace(_repack_marlin_experts=original)
    monkeypatch.setitem(sys.modules,
        "vllm.model_executor.layers.quantization.utils",
        SimpleNamespace(marlin_utils_fp4=marlin))

    class Worker:
        split_limit = 20

        @contextmanager
        def _scoped_allocator_max_split(self, max_split_size_mb):
            previous = self.split_limit
            self.split_limit = max_split_size_mb
            try:
                yield
            finally:
                self.split_limit = previous

        def load_model(self, *, load_dummy_weights=False):
            # Match the allocator context entered by vLLM 0.28's loader.
            with self._scoped_allocator_max_split(max_split_size_mb=20):
                assert marlin._repack_marlin_experts is not original
                assert load_dummy_weights
                assert self.split_limit > 20
                if fail:
                    raise RuntimeError("startup failed")
                return "loaded"

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker",
                        SimpleNamespace(Worker=Worker))
    from vllm_compat.gpt_oss_worker import GptOssWorker
    try:
        worker = GptOssWorker()
        if fail:
            with pytest.raises(RuntimeError, match="startup failed"):
                worker.load_model(load_dummy_weights=True)
        else:
            assert worker.load_model(load_dummy_weights=True) == "loaded"
        assert worker.split_limit == 20
        assert marlin._repack_marlin_experts is original
    finally:
        sys.modules.pop("vllm_compat.gpt_oss_worker", None)


def test_cuda_marlin_packing_matches_installed_vllm():
    import inspect
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA packing comparison requires the GPU host")
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.randint(0, 256, (3, 256, 64), dtype=torch.uint8, device="cuda")
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    original = marlin_utils_fp4._repack_marlin_experts
    options = dict(size_n=256, size_k=128, is_a_8bit=False)
    if "perm" in inspect.signature(original).parameters:
        options["perm"] = perm
    expected = original(weight, **options)
    actual = repack_experts_in_place(
        weight.clone(), repack=original, **options)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def test_cuda_gpt_oss_scratch_reuse_and_sleep_wake():
    """Exercise real expert sizes and the private allocator on the GPU host.

    With vLLM's original 20 MiB split limit, cached conversion segments grow
    every iteration. Repeated conversions must instead reuse bounded scratch,
    produce identical packed weights, and survive the normal sleep/wake path.
    """
    import inspect
    import torch

    if not torch.cuda.is_available():
        pytest.skip("private-pool memory regression requires the GPU host")
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm_compat.gpt_oss_repack import repack_experts_in_place
    from vllm_compat.gpt_oss_worker import GptOssWorker

    # GPT-OSS-120B gate/up expert dimensions, large enough to use the 20 MiB
    # allocator segments that caused the production OOM (tiny tensors miss it).
    weight = torch.randint(0, 256, (3, 5760, 1440),
                           dtype=torch.uint8, device="cuda")
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    original = marlin_utils_fp4._repack_marlin_experts
    options = dict(size_n=5760, size_k=2880, is_a_8bit=False)
    if "perm" in inspect.signature(original).parameters:
        options["perm"] = perm
    expected = original(weight, **options)
    allocator = CuMemAllocator.get_instance()
    worker = object.__new__(GptOssWorker)
    tag = "gpt_oss_repack_test"
    with (allocator.use_memory_pool(tag=tag),
          worker._scoped_allocator_max_split(max_split_size_mb=20)):
        storage = weight.clone()
        before = allocator.get_current_usage()
        for _ in range(32):
            storage.copy_(weight)
            actual = repack_experts_in_place(
                storage, repack=original, **options)
        torch.cuda.synchronize()
        assert allocator.get_current_usage() - before < 128 * 1024**2
    assert torch.equal(actual, expected)
    allocator.sleep(offload_tags=(tag,))
    allocator.wake_up(tags=[tag])
    assert torch.equal(actual, expected)
