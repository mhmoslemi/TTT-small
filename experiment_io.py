"""
Per-experiment file I/O.

Creates a directory under runs/ named from the main hyperparameters, then
writes one .txt and one .meta.json per rollout. Two-stage problems also write
the paired base-model planning response as .strategy.txt. ALL rollouts are
saved, including ones that failed extraction or validation.

Filenames:
    step03_group2_rollout17.txt        ← raw model response
    step03_group2_rollout17.strategy.txt ← raw planning-stage response
    step03_group2_rollout17.meta.json  ← reward, valid, msg, beta, advantage, etc.
    step03_group02_fold00_strategy01.txt ← strategy saved immediately

Directory name (problem-agnostic):
    runs/erdos_gpt-oss-120b_0602-2201/
    runs/circle_packing_n26_Qwen3-8B_0527-2201/

A config.json is also dumped at the root of the run dir.
"""

import json
import re
import time
from dataclasses import asdict
from pathlib import Path


class StepPlotter:
    """Refresh existing plots off the training thread, in step order.

    Use the same Python environment, headless CPU-only subprocesses, and a
    bounded timeout per script. A single worker avoids competing plot writes;
    --max-step prevents a queued plot from including a later unfinished step.
    Diagnostics go to plots.log, and failures never invalidate a checkpoint.
    """

    def __init__(self, run_dir, problem):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        self.run_dir = Path(run_dir).resolve()
        self.problem = str(problem).strip().lower()
        self.stopping = Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="step-plots")

    def submit(self, step):
        self.executor.submit(self._refresh, int(step))

    def close(self, *, wait=True):
        # On interruption, cancel queued plots; an already-running plot is
        # still bounded by the per-script timeout.
        if not wait:
            self.stopping.set()
        self.executor.shutdown(wait=wait, cancel_futures=not wait)

    def _refresh(self, step):
        import os
        import subprocess
        import sys

        scripts = [("plot_erdos_best_curve.py", [])]
        if self.problem in {"erdos", "erdos_min_overlap", "erdos_minimum_overlap"}:
            scripts.insert(0, ("plot_erdos.py", ["--saved-only"]))
        environment = dict(os.environ)
        environment.update({
            "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        script_dir = Path(__file__).resolve().parent
        try:
            with (self.run_dir / "plots.log").open("a", encoding="utf-8") as log:
                for script, options in scripts:
                    if self.stopping.is_set():
                        break
                    log.write(f"\n=== completed step {step}: {script} ===\n")
                    log.flush()
                    try:
                        result = subprocess.run(
                            [sys.executable, str(script_dir / script),
                             str(self.run_dir), "--max-step", str(step), *options],
                            env=environment, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            timeout=120, check=False,
                        )
                        if result.returncode:
                            raise RuntimeError(f"exit status {result.returncode}")
                    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                        log.write(f"plot refresh failed: {error}\n")
                        log.flush()
                        print(f"[plots] step {step}: {script} not updated "
                              f"({error}); see {self.run_dir / 'plots.log'}",
                              flush=True)
        except OSError as error:
            print(f"[plots] step {step}: cannot write plots.log: {error}", flush=True)


def _slugify(s: str) -> str:
    """Make a string safe for use in a directory name."""
    return re.sub(r"[^A-Za-z0-9._\-]", "_", str(s)).strip("_")


def _model_short(model_name: str) -> str:
    """Return just the last path component of a model name, slugified."""
    return _slugify(model_name.split("/")[-1])


def make_experiment_dir(cfg, root: str = "runs", resume_dir=None,
                        config_dict=None) -> Path:
    """
    Build a directory whose name encodes the key identifiers of this run.
    Full hyperparameters are always in config.json inside the directory.

    When ``resume_dir`` is provided, reuse it without rewriting the original
    configuration. ``config_dict`` lets callers persist problem-specific YAML
    keys in addition to the Config dataclass fields.
    """
    if resume_dir is not None:
        path = Path(resume_dir).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"resume directory not found: {path}")
        return path

    problem = _slugify(getattr(cfg, "problem", "run"))
    name_parts = [problem]

    # n<circles> tag for circle-packing problems
    num_circles = getattr(cfg, "num_circles", None)
    if problem in ("circle_packing", "circle", "circles") and num_circles is not None:
        name_parts.append(f"n{num_circles}")

    name_parts += [
        _model_short(cfg.model_name),
        time.strftime("%m%d-%H%M"),
    ]
    name = "_".join(name_parts)
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)

    if config_dict is not None:
        cfg_dict = dict(config_dict)
    else:
        try:
            cfg_dict = asdict(cfg)
        except TypeError:
            cfg_dict = {k: getattr(cfg, k) for k in dir(cfg)
                        if not k.startswith("_")
                        and not callable(getattr(cfg, k))}
    (path / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return path


def save_rollout(
    exp_dir: Path,
    step: int,
    group: int,
    rollout: int,
    response_text: str,
    meta: dict,
    prompt_text: str = None,
    strategy_text: str = None,
    artifacts_already_saved: bool = False,
):
    """
    Save one rollout as a .txt + .meta.json pair, plus optional prompt and
    planning-stage text files.

    meta should include at least: reward, valid, parsed, ran, msg.
    Anything JSON-serializable is fine.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"
    if not artifacts_already_saved:
        save_rollout_artifacts(
            exp_dir, step, group, rollout, response_text,
            prompt_text=prompt_text, strategy_text=strategy_text)

    # Make sure we can dump everything (numpy floats, bools, etc.)
    def _coerce(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if hasattr(v, "item"):  
            try:
                return v.item()
            except Exception:
                return str(v)
        if hasattr(v, "tolist"):  
            try:
                return v.tolist()
            except Exception:
                return str(v)
        return str(v)

    safe_meta = {k: _coerce(v) for k, v in meta.items()}
    meta_path = step_dir / f"{base}.meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(safe_meta, indent=2))
    meta_tmp.replace(meta_path)


def save_rollout_artifacts(
    exp_dir: Path,
    step: int,
    group: int,
    rollout: int,
    response_text: str,
    prompt_text: str = None,
    strategy_text: str = None,
    pending_meta: dict = None,
):
    """Persist generated text immediately, before reward evaluation finishes."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"

    def _atomic_text(path, value):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(str(value or ""), errors="replace")
        tmp.replace(path)

    _atomic_text(step_dir / f"{base}.txt", response_text)
    if strategy_text is not None:
        _atomic_text(step_dir / f"{base}.strategy.txt", strategy_text)
    if prompt_text is not None:
        _atomic_text(step_dir / f"{base}.prompt.txt", prompt_text)
    if pending_meta is not None:
        path = step_dir / f"{base}.meta.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(pending_meta, indent=2, default=str))
        tmp.replace(path)


def save_strategy_response(exp_dir: Path, step: int, parent_group: int,
                           fold: int, strategy: int,
                           response_text: str, *, attempt=None,
                           prompt_text=None, extraction_issue=None) -> Path:
    """Persist one planning response immediately after generation returns."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    suffix = "" if attempt is None else f"_attempt{int(attempt):02d}"
    base = (f"step{step:02d}_group{parent_group:02d}_fold{fold:02d}_"
            f"strategy{strategy:02d}{suffix}.txt")
    path = step_dir / base
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(str(response_text or ""), errors="replace")
    tmp.replace(path)
    if attempt is not None:
        prompt = (prompt_text if isinstance(prompt_text, str)
                  else json.dumps(prompt_text, ensure_ascii=False, indent=2))
        path.with_suffix(".prompt.txt").write_text(prompt, errors="replace")
        path.with_suffix(".meta.json").write_text(json.dumps({
            "attempt": int(attempt), "extraction_issue": extraction_issue,
            "accepted": extraction_issue is None,
        }, indent=2))
    return path


def save_parent_selections(exp_dir: Path, step: int, sampler_type: str,
                           parents: list, picks_info: list):
    """Persist the exact parents selected for a step before generation/training.

    The sampler checkpoint only contains nodes that survived archive pruning.
    This event file is deliberately independent of that checkpoint so a later
    tree plot can also reconstruct selections whose children were invalid,
    duplicates, or eventually pruned.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    selected = []
    for group, parent in enumerate(parents):
        info = picks_info[group] if group < len(picks_info) else {}
        selected.append({
            "group": group,
            "parent_id": str(parent.id),
            "parent_timestep": int(parent.timestep),
            "parent_reward": (float(parent.value)
                              if parent.value is not None else None),
            "parent_raw_score": (float(parent.raw_score)
                                 if parent.raw_score is not None else None),
            "parent_is_seed": bool(parent.is_seed),
            "ancestor_ids": [str(p.get("id")) for p in (parent.parents or [])
                             if p.get("id") is not None],
            "visit_count": int(info.get("n", 0)),
            "q_value": (float(info["Q"]) if info.get("Q") is not None else None),
            "prior": (float(info["P"]) if info.get("P") is not None else None),
            "exploration_bonus": (float(info["bonus"])
                                  if info.get("bonus") is not None else None),
            "selection_score": (float(info["score"])
                                if info.get("score") is not None else None),
        })
    payload = {
        "version": 1,
        "step": int(step),
        "sampler_type": str(sampler_type),
        "parents": selected,
    }
    path = step_dir / f"step{step:02d}.parents.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def save_step_summary(exp_dir: Path, step: int, summary: dict):
    """Write a per-step summary (group stats, best so far, timings)."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    (step_dir / f"step{step:02d}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


def append_step_result(exp_dir: Path, step: int, summary: dict):
    """Append one idempotent per-step result line to the run-level log."""
    step = int(step)
    path = Path(exp_dir) / "result.txt"
    marker = f"step={step:04d}\t"
    if path.is_file():
        try:
            if any(line.startswith(marker)
                   for line in path.read_text(errors="replace").splitlines()):
                return path
        except OSError:
            pass

    def _number(value):
        if value is None:
            return "unavailable"
        try:
            return f"{float(value):.12f}"
        except (TypeError, ValueError):
            return "unavailable"

    metric = str(summary.get("result_metric_name") or "raw metric")
    direction = ("maximize" if bool(summary.get("result_maximize", True))
                 else "minimize")
    step_raw = _number(summary.get("step_best_raw_score"))
    step_reward = _number(summary.get("step_best_reward"))
    run_raw = _number(summary.get("best_seen_raw_score"))
    run_reward = _number(summary.get("best_seen_reward"))
    found_step = summary.get("best_seen_step")
    found_step = ("unavailable" if found_step is None
                  else str(int(found_step)))
    line = (
        f"{marker}metric={metric}\tdirection={direction}\t"
        f"step_best_raw={step_raw}\tstep_best_reward={step_reward}\t"
        f"best_seen_raw={run_raw}\tbest_seen_reward={run_reward}\t"
        f"best_seen_found_step={found_step}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    return path


def save_final_summary(exp_dir: Path, best_value, best_code, best_step,
                       best_construction=None, best_raw_score=None):
    """
    Write the end-of-run summary.

    best_construction is the actual solution object (for Erdos, the h array).
    Written both into the summary and to its own file, because a plot or a
    verification wants the array on its own and should never have to re-run the
    program to get it.
    """
    out = {
        "best_value": float(best_value) if best_value is not None else None,
        "best_raw_score": (float(best_raw_score)
                           if best_raw_score is not None else None),
        "best_step": int(best_step) if best_step is not None else None,
        "best_code": best_code or "",
        "best_construction": best_construction,
    }
    (exp_dir / "final.summary.json").write_text(json.dumps(out, indent=2))
    if best_code:
        (exp_dir / "best_code.py").write_text(best_code)
    if best_construction:
        (exp_dir / "best_construction.json").write_text(
            json.dumps(best_construction))

if __name__ == "__main__":
    # Self-test
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        model_name="openai/gpt-oss-120b",
        problem="erdos", problem_type=None,
        num_steps=50, groups_per_step=8, group_size=64,
        learning_rate=4e-5, temperature=1.0, kl_penalty_coef=0.1,
    )
    p = make_experiment_dir(cfg, root="/tmp/runs_test")
    print(f"Created: {p}")
    save_rollout(p, step=0, group=0, rollout=0,
                 response_text="```python\nprint('hello')\n```",
                 meta={"reward": 0.0, "valid": False, "msg": "demo",
                       "advantage": 1.234})
    print("Saved demo rollout. Contents:")
    for f in sorted(p.iterdir()):
        print(" ", f.name)
