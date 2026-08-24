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
    # New configs store the base context length, before the runtime memory
    # allowance is added. The marker lets resume read older configs too.
    cfg_dict["_max_seq_length_includes_memory_topup"] = False
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
):
    """
    Save one rollout as a .txt + .meta.json pair (plus a .prompt.txt when the
    rendered prompt is supplied).

    meta should include at least: reward, valid, parsed, ran, msg.
    Anything JSON-serializable is fine.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"
    (step_dir / f"{base}.txt").write_text(response_text, errors="replace")
    if prompt_text is not None:
        (step_dir / f"{base}.prompt.txt").write_text(prompt_text, errors="replace")

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
