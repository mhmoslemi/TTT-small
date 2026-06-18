"""
Per-experiment file I/O.

Creates a directory under runs/ named from the main hyperparameters, then
writes one .txt and one .meta.json per rollout. ALL rollouts are saved,
including ones that failed extraction or validation.

Filenames:
    step03_group2_rollout17.txt        ← raw model response
    step03_group2_rollout17.meta.json  ← reward, valid, msg, beta, advantage, etc.

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


def _slugify(s: str) -> str:
    """Make a string safe for use in a directory name."""
    return re.sub(r"[^A-Za-z0-9._\-]", "_", str(s)).strip("_")


def _model_short(model_name: str) -> str:
    """Return just the last path component of a model name, slugified."""
    return _slugify(model_name.split("/")[-1])


def make_experiment_dir(cfg, root: str = "runs") -> Path:
    """
    Build a directory whose name encodes the key identifiers of this run.
    Full hyperparameters are always in config.json inside the directory.
    """
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

    try:
        cfg_dict = asdict(cfg)
    except TypeError:

        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg)
                    if not k.startswith("_") and not callable(getattr(cfg, k))}
    (path / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return path


def save_rollout(
    exp_dir: Path,
    step: int,
    group: int,
    rollout: int,
    response_text: str,
    meta: dict,
):
    """
    Save one rollout as a .txt + .meta.json pair.

    meta should include at least: reward, valid, parsed, ran, msg.
    Anything JSON-serializable is fine.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"
    (step_dir / f"{base}.txt").write_text(response_text, errors="replace")

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
    (step_dir / f"{base}.meta.json").write_text(json.dumps(safe_meta, indent=2))


def save_step_summary(exp_dir: Path, step: int, summary: dict):
    """Write a per-step summary (group stats, best so far, timings)."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    (step_dir / f"step{step:02d}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


def save_final_summary(exp_dir: Path, best_value, best_code, best_step):
    """Write the end-of-run summary."""
    out = {
        "best_value": float(best_value) if best_value is not None else None,
        "best_step": int(best_step) if best_step is not None else None,
        "best_code": best_code or "",
    }
    (exp_dir / "final.summary.json").write_text(json.dumps(out, indent=2))
    if best_code:
        (exp_dir / "best_code.py").write_text(best_code)



def save_elo_match(exp_dir: Path, step: int, cycle: int, match_idx: int,
                   meta: dict, prompt_text: str = "", response_text: str = ""):
    """Save one Elo tournament match under step{step}_Elo/.

    Mirrors save_rollout's layout: a .meta.json plus the raw judge response,
    and (optionally) the exact prompt the judge saw. `meta` should include the
    two state ids, which was shown as 1 vs 2, the parsed verdict, and the winner.
    """
    elo_dir = Path(exp_dir) / f"step{step:02d}_Elo"
    elo_dir.mkdir(parents=True, exist_ok=True)
    base = f"step{step:02d}_cycle{cycle:03d}_match{match_idx:04d}"

    def _coerce(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return str(v)
        return str(v)

    safe_meta = {k: _coerce(v) for k, v in meta.items()}
    (elo_dir / f"{base}.meta.json").write_text(json.dumps(safe_meta, indent=2))
    (elo_dir / f"{base}.response.txt").write_text(response_text or "", errors="replace")
    if prompt_text:
        (elo_dir / f"{base}.prompt.txt").write_text(prompt_text, errors="replace")


def save_elo_cycle_summary(exp_dir: Path, step: int, cycle: int, summary: dict):
    """Write per-cycle Elo standings (ratings, win counts) for quick inspection."""
    elo_dir = Path(exp_dir) / f"step{step:02d}_Elo"
    elo_dir.mkdir(parents=True, exist_ok=True)
    (elo_dir / f"step{step:02d}_cycle{cycle:03d}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )



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



