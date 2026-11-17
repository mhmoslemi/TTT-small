#!/usr/bin/env python3
"""Re-evaluate saved GPU-kernel rollouts without loading any generation models."""

import argparse
import json
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parent


def evaluate_saved(problem, step_dir, output_dir, timeout_s, limit=None):
    from evaluation.reward import extract_python_code
    from problems.base import ParentContext

    paths = sorted(path for path in Path(step_dir).glob("*_rollout*.txt")
                   if re.fullmatch(r"step\d+_group\d+_rollout\d+\.txt", path.name))
    if not paths:
        raise ValueError(f"No saved coder rollout .txt files in {step_dir}")
    if limit is not None:
        paths = paths[:limit]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(paths, 1):
        response = path.read_text(encoding="utf-8")
        code = extract_python_code(response)
        if code is not None:
            (output_dir / path.with_suffix(".py").name).write_text(code, encoding="utf-8")
        print(f"[{index}/{len(paths)}] evaluating {path.name}", flush=True)
        start = time.monotonic()
        result = problem.compute_reward(response, ParentContext(), timeout_s=timeout_s)
        results.append(dict(
            response_file=str(path), valid=result.valid, parsed=result.parsed,
            ran=result.ran, raw_score=result.raw_score, reward=result.reward,
            failure_kind=result.failure_kind, msg=result.msg, stdout=result.stdout,
            elapsed_s=time.monotonic() - start))
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        status = "PASS" if result.valid else "FAIL"
        print(f"  {status}: {result.msg}", flush=True)
        if result.failure_kind == "infrastructure":
            raise RuntimeError(f"Evaluator infrastructure failed: {result.msg}")
    passed = sum(row["valid"] for row in results)
    print(f"{passed}/{len(results)} passed. Results and extracted Python: {output_dir}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_dir", type=Path, help="Existing run's step directory")
    parser.add_argument("--gpu", required=True, type=int, help="Idle physical GPU for sequential evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gpu_mode_trimul.yaml",
                        help="Current YAML supplying the corrected kernel_python path")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N saved rollouts")
    args = parser.parse_args(argv)
    if args.gpu < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--gpu must be nonnegative and --limit must be positive")
    step_dir, config_path = args.step_dir.expanduser().resolve(), args.config.expanduser().resolve()
    saved_config = step_dir.parent / "config.json"
    if not step_dir.is_dir() or not saved_config.is_file():
        parser.error("step_dir must be a saved run's step directory with ../config.json")
    os.chdir(ROOT)

    # Keep evaluator scratch/cache files off the system disk, like the launchers.
    fast = Path(os.environ.get("TTT_FAST_RUNTIME_DIR", str(
        Path("/dev/shm/ttt-small-runtime") if os.access("/dev/shm", os.W_OK)
        else ROOT / ".runtime/fast")))
    for name, directory in {
            "TTT_FAST_RUNTIME_DIR": fast, "TMPDIR": fast / "tmp",
            "XDG_CACHE_HOME": ROOT / ".runtime/cache",
            "TORCH_EXTENSIONS_DIR": fast / "torch_extensions",
            "CUDA_CACHE_PATH": fast / "cuda"}.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import yaml
    from problems.gpu_mode import (
        GpuMode, benchmark_environment_from_python, benchmark_environment_issue,
        ensure_python3_on_path)

    cfg = json.loads(saved_config.read_text(encoding="utf-8"))
    if cfg.get("problem") != "gpu_mode":
        parser.error("This command only evaluates saved gpu_mode rollouts")
    current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["kernel_python"] = current["kernel_python"]
    cfg["kernel_gpu_id"] = args.gpu
    cfg["evaluation_shares_generation"] = False
    problem = GpuMode(cfg)
    python = ensure_python3_on_path(problem.kernel_python)
    identity = benchmark_environment_from_python(python)
    issue = benchmark_environment_issue(
        identity["gpu"], identity["capability"], identity["triton"],
        problem.gpu_type, problem.triton_version)
    if issue:
        raise RuntimeError(issue)
    print(f"Evaluator: {problem.kernel_python}; {identity['gpu']}; "
          f"Triton {identity['triton']}; physical GPU {args.gpu}", flush=True)
    output_dir = ROOT / ".runtime/kernel_eval_check" / step_dir.parent.name / step_dir.name
    results = evaluate_saved(
        problem, step_dir, output_dir,
        cfg.get("kernel_timeout_s") or cfg.get("sandbox_timeout_s", 900), args.limit)
    return 0 if any(row["valid"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
