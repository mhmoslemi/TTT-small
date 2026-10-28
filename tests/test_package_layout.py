"""Package/import/resource checks without loading models or CUDA libraries."""

import ast
from pathlib import Path
import pickle
import subprocess
import sys
from types import ModuleType

from generation import gen_workers
from runtime.host_memory import validate_memory_handoff_runtime
from training.fast_distributed import worker_main


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "models": "model_backend strategist_models qwen_training_kernels gpt_oss_training",
    "training": "fast_distributed spo_rs advantage context_parallel exact_overflow training_checkpointing",
    "generation": "gen_workers strategy_api strategy_prompts strategy_prompts_openai output_retries",
    "search": "sampler llm_sampler rollout_allocation",
    "evaluation": "reward sandbox",
    "runtime": "gpu_runtime host_memory gpu_lease",
    "diagnostics": "startup_logs loading_logs terminal_output",
    "analysis": "entropy_tools plot_erdos plot_ac plot_erdos_best_curve plot_parent_tree",
}


def test_moved_modules_and_all_local_import_targets_exist():
    old_names = {name for names in PACKAGES.values() for name in names.split()}
    sources = [ROOT / "train_multy_CVaR.py", ROOT / "experiment_io.py"]
    assert not (ROOT / "logging").exists()
    assert (ROOT / "problems" / "registry.py").is_file()
    assert not (ROOT / "_retire" / "problems").exists()
    for package, names in PACKAGES.items():
        assert (ROOT / package / "__init__.py").is_file()
        for name in names.split():
            path = ROOT / package / f"{name}.py"
            assert path.is_file() and not (ROOT / f"{name}.py").exists()
            sources.append(path)
    sources.extend((ROOT / "vllm_compat").glob("*.py"))
    sources.extend((ROOT / "problems").glob("*.py"))
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                assert root not in old_names, (path, module)
                assert not module.startswith("_retire.problems"), (path, module)
                if root in PACKAGES or root == "problems":
                    target = ROOT.joinpath(*module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (path, module)


def test_problem_registry_loads_all_tasks_from_root_package(monkeypatch):
    from problems import get_problem, registry

    assert Path(registry.__file__).resolve() == ROOT / "problems" / "registry.py"

    class StubProblem:
        def __init__(self, cfg):
            self.cfg = cfg

    for name, module_name, class_name in (
            ("erdos", "erdos", "ErdosMinOverlap"),
            ("ac1", "ac_inequalities", "ACInequalities"),
            ("ac2", "ac_inequalities", "ACInequalities"),
            ("circle_packing", "circle_packing", "CirclePacking"),
            ("denoising", "denoising", "Denoising"),
            ("gpu_mode", "gpu_mode", "GpuMode")):
        module = ModuleType(f"problems.{module_name}")
        setattr(module, class_name, StubProblem)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        config = {"budget_s": 1300}
        problem = get_problem(name, config)
        assert isinstance(problem, StubProblem)
        assert problem.cfg["budget_s"] == 1300
        if name in ("ac1", "ac2"):
            assert problem.cfg["problem_type"] == name
        assert config == {"budget_s": 1300}


def test_spawn_worker_references_and_memory_handoff_use_same_checkout():
    for worker in (gen_workers._vllm_worker_loop, gen_workers._hf_worker_loop, worker_main):
        assert pickle.loads(pickle.dumps(worker)) is worker
    report = validate_memory_handoff_runtime(ROOT / "train_multy_CVaR.py")
    assert report["checkout"] == str(ROOT)
    assert set(report["files"]) == {
        "train_multy_CVaR.py", "runtime/host_memory.py",
        "generation/gen_workers.py", "vllm_compat/sleep_memory.py"}


def test_prompt_and_api_key_stay_at_project_root():
    for relative, setting, expected in (
            ("search/llm_sampler.py", "PROMPT_FILE", ROOT / "llm_sampler_prompt.txt"),
            ("generation/strategy_api.py", "OPENAI_KEY_FILE", ROOT / ".openai_api_key")):
        path = ROOT / relative
        tree = ast.parse(path.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == setting
                                  for target in node.targets))
        namespace = {"Path": Path, "__file__": str(path)}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), relative, "exec"), namespace)
        assert namespace[setting] == expected


def test_vllm_compat_hook_uses_project_root_after_move(monkeypatch):
    monkeypatch.setattr(gen_workers.importlib.metadata, "version", lambda _: "0.6.16")
    monkeypatch.setattr(gen_workers, "_flashinfer_comm_guard_required", lambda _: True)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("TTT_VLLM_DISABLE_BROKEN_FLASHINFER_COMM", "0")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "1")
    monkeypatch.setitem(sys.modules, "flashinfer.comm", None)
    assert gen_workers._prepare_flashinfer_comm_compat()
    import os
    assert str(ROOT / "vllm_compat") in os.environ["PYTHONPATH"].split(os.pathsep)


def test_plot_module_entrypoints_load_without_models():
    for module in ("analysis.plot_ac", "analysis.plot_erdos_best_curve", "analysis.plot_parent_tree"):
        result = subprocess.run([sys.executable, "-m", module, "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
