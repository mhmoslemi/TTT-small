#!/usr/bin/env python3
"""
Plot the step function h from an Erdos rollout.

Three routes to h, tried in this order:

  1. meta["construction"]   the array that was actually scored. Exact, no
                            execution. Runs from the patched trainer have it.
  2. --h-json FILE          an array you supply.
  3. replay                 re-execute the saved program. The fallback for
                            older runs, and only ever approximate: a mid-run
                            rollout was generated with its parent's h in scope
                            as `initial_h_values`, and older runs did not save
                            that either. meta["parent_construction"] supplies it
                            when present, but a program that is stochastic or
                            wall-clock bounded still will not reproduce.

Route 1 is why the trainer now writes `construction` and `parent_construction`
into every rollout meta. Replay is a last resort, not the design.

Usage:

    python plot_erdos.py runs/erdos_.../                    # best valid rollout
    python plot_erdos.py runs/erdos_.../ --step 13 --group 0 --rollout 2
    python plot_erdos.py runs/erdos_.../ --file best_code.py
    python plot_erdos.py runs/erdos_.../ --h-json h.json    # exact, no replay
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def as_array(value):
    """
    Coerce a meta field to a float array.

    experiment_io's _coerce falls back to str() for anything it cannot dump
    directly, so `construction` can arrive as a real list OR as the repr of one
    ("[0.498, 0.826, ...]"). Same for memory_ids. Handle both rather than
    assuming, since which one you get depends on the numpy dtype of the values
    at save time.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception:
            import ast as _ast
            try:
                value = _ast.literal_eval(text)
            except Exception:
                return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


# ----------------------------------------------------------------------
# The verifier, byte-for-byte what problems/erdos.py scores with
# ----------------------------------------------------------------------
def c5_of(h_values, n_points=None):
    """
    Returns (h_after_normalization, computed_c5).

    Mirrors verify_c5_solution: normalize the sum to n/2, then compute
    max(correlate(h, 1-h)) * dx. Does NOT clip, so an out-of-range value after
    normalization is visible here exactly as the real verifier would see it.
    """
    h = np.asarray(h_values, dtype=np.float64).ravel()
    n = int(n_points or h.shape[0])
    s = h.sum()
    if s != n / 2.0:
        h = h * ((n / 2.0) / s)
    dx = 2.0 / n
    corr = np.correlate(h, 1.0 - h, mode="full") * dx
    return h, float(np.max(corr))


def seed_construction(seed: int, index: int):
    """Reproduce seed `index` exactly as problems/erdos.py builds it."""
    rng = np.random.default_rng(seed + index)
    n_points = int(rng.integers(40, 100))     # drawn FIRST, consumes the stream
    base = np.ones(n_points) * 0.5
    pert = rng.uniform(-0.4, 0.4, n_points)
    pert = pert - np.mean(pert)
    return base + pert                        # no clipping, matches erdos.py


# ----------------------------------------------------------------------
# Finding and running a rollout
# ----------------------------------------------------------------------
def load_metas(run_dir: Path):
    out = []
    for meta_path in sorted(run_dir.glob("step*/*.meta.json")):
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        m["_meta_path"] = meta_path
        m["_txt_path"] = meta_path.with_suffix("").with_suffix(".txt")
        out.append(m)
    return out


def pick(metas, step=None, group=None, rollout=None):
    if step is not None:
        sel = [m for m in metas
               if m.get("step") == step
               and (group is None or m.get("group") == group)
               and (rollout is None or m.get("rollout") == rollout)]
        if not sel:
            raise SystemExit(f"no rollout matching step={step} group={group} "
                             f"rollout={rollout}")
        return sel[0]
    valid = [m for m in metas if m.get("valid") and m.get("raw_score") is not None]
    if not valid:
        raise SystemExit("no valid rollout in that run directory")
    return min(valid, key=lambda m: float(m["raw_score"]))   # lower C5 is better


def extract_code(text: str):
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return None
    return max(blocks, key=len).strip()       # the extractor takes a fenced block


def run_program(code: str, initial_h, budget_note=""):
    """
    Exec the program with `initial_h_values` in scope, exactly as preprocess
    arranges it, and call run() with no arguments, exactly as the sandbox does.
    """
    ns = {"__name__": "replay", "np": np, "numpy": np}
    if initial_h is not None:
        ns["initial_h_values"] = np.asarray(initial_h, dtype=np.float64)
    exec(compile(code, "<rollout>", "exec"), ns)
    if "run" not in ns:
        raise SystemExit("the program defines no top-level run()")
    return ns["run"]()                        # no args: the sandbox calls fn()


# ----------------------------------------------------------------------
# Orientation
# ----------------------------------------------------------------------
def canonical_orientation(h_values):
    """
    Put h in the paper's orientation (mass centered, tapering to ~0 at both
    ends). C5 is invariant under h -> 1-h and under reflection x -> 2-x, and
    both preserve sum(h) = n/2, so all four forms are equally valid solutions.
    The raw rollout is often the complement ("valley") or a mirror of the
    paper's "mountain"; pick the form with the least mass sitting at the ends.
    """
    h = np.asarray(h_values, dtype=np.float64).ravel()
    n = h.shape[0]
    edge = max(1, n // 10)

    def edge_mass(a):
        return float(a[:edge].mean() + a[-edge:].mean())

    candidates = (h, 1.0 - h, h[::-1], 1.0 - h[::-1])
    return min(candidates, key=edge_mass)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_step_function(h_values, c5_val, n_points, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_edges = np.linspace(0, 2, n_points + 1)
    y_steps = np.append(h_values, h_values[-1])

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(5.5, 2.2), dpi=300)
    ax.step(x_edges, y_steps, where="post", color="#4A80D0",
            linewidth=1.4, antialiased=True)
    ax.set_xlim(-0.02, 2.02)
    ax.set_ylim(-0.05, 1.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=8, color="#111111")
    ax.text(0.03, 0.92,
            f"{n_points}-piece function\n$c \\leq {c5_val:.9f}$",
            transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#B0B0B0", linewidth=0.8, alpha=0.9))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(Path(save_path).with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {save_path} and {Path(save_path).with_suffix('.pdf')}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot h from an Erdos rollout.")
    ap.add_argument("run_dir")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--group", type=int, default=None)
    ap.add_argument("--rollout", type=int, default=None)
    ap.add_argument("--file", default=None,
                    help="run a .py in the run dir (e.g. best_code.py) instead "
                         "of picking a rollout")
    ap.add_argument("--h-json", default=None,
                    help="a JSON list of h values; skips replay entirely and is "
                         "the only exact route for a mid-run rollout")
    ap.add_argument("--seed", type=int, default=42, help="cfg seed, for initial_h_values")
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--replay", action="store_true",
                    help="re-execute the program even when the saved "
                         "construction is available")
    ap.add_argument("--no-initial", action="store_true",
                    help="do not provide initial_h_values at all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    recorded = None
    label = ""
    h_raw = None

    # --- route 1: the saved solution, exact ---
    if not args.h_json and not args.file:
        metas = load_metas(run_dir)
        if not metas:
            raise SystemExit(f"no step*/*.meta.json under {run_dir}")
        m = pick(metas, args.step, args.group, args.rollout)
        recorded = m.get("raw_score")
        label = (f"step {m.get('step')} group {m.get('group')} "
                 f"rollout {m.get('rollout')}")
        print(f"selected {label}: recorded raw_score = {recorded}, "
              f"parent_is_seed = {m.get('parent_is_seed')}")
        saved = as_array(m.get("construction"))
        if saved is not None and not args.replay:
            h_raw = saved
            print(f"using the SAVED construction ({len(h_raw)} points): exact, "
                  f"no re-execution")
        else:
            if not args.replay:
                print("  this run predates construction saving; falling back to "
                      "replay, which is approximate")
            code = extract_code(m["_txt_path"].read_text(errors="replace"))
            if code is None:
                raise SystemExit("no ```python block in that rollout's response")
            parent_h = as_array(m.get("parent_construction"))
            if parent_h is not None:
                initial = parent_h
                print(f"initial_h_values = the SAVED parent construction "
                      f"(n={len(initial)}, C5={c5_of(initial)[1]:.9f})")
            elif args.no_initial:
                initial = None
            else:
                initial = seed_construction(args.seed, args.seed_index)
                print(f"initial_h_values = seed {args.seed_index} "
                      f"(n={len(initial)}, C5={c5_of(initial)[1]:.9f})")
                if m.get("parent_is_seed") is False:
                    print("  NOTE: the real parent was an evolved state and this "
                          "run did not save it, so the replay cannot be exact.")
            result = run_program(code, initial)
            if not (isinstance(result, tuple) and len(result) == 3):
                raise SystemExit(f"run() returned {type(result).__name__}, "
                                 f"expected a 3-tuple")
            h_raw, reported, n_rep = result
            print(f"program reported C5 = {float(reported):.9f}  n_points = {n_rep}")

    # --- route 2: h supplied directly ---
    elif args.h_json:
        h_raw = np.asarray(json.loads(Path(args.h_json).read_text()), dtype=float)
        label = f"from {Path(args.h_json).name}"
    # --- route 3: an explicit .py in the run dir ---
    else:
        code = Path(run_dir / args.file).read_text()
        label = args.file
        initial = None if args.no_initial else seed_construction(args.seed,
                                                                 args.seed_index)
        if initial is not None:
            print(f"initial_h_values = seed {args.seed_index} "
                  f"(n={len(initial)}, C5={c5_of(initial)[1]:.9f})")
        result = run_program(code, initial)
        if not (isinstance(result, tuple) and len(result) == 3):
            raise SystemExit(f"run() returned {type(result).__name__}, expected a "
                             f"3-tuple (h_values, c5_bound, n_points)")
        h_raw, reported, n_rep = result
        print(f"program reported C5 = {float(reported):.9f}  n_points = {n_rep}")

    h, computed = c5_of(h_raw)
    h = canonical_orientation(h)
    n = len(h)
    print(f"verifier computes  C5 = {computed:.9f}  over {n} points")
    if h.min() < -1e-12 or h.max() > 1 + 1e-12:
        print(f"  WARNING: after normalization h is outside [0,1] "
              f"({h.min():.4f}..{h.max():.4f}). The real verifier REJECTS this.")
    if recorded is not None:
        d = abs(computed - float(recorded))
        print(f"recorded raw_score = {float(recorded):.9f}   |diff| = {d:.2e}"
              + ("   (replay reproduced it)" if d < 1e-6 else
                 "   (replay DIVERGED: the program is stochastic, or it used a "
                 "parent construction this script could not supply)"))

    out = args.out or str(run_dir / "erdos_solution.png")
    plot_step_function(h, computed, n,
                       args.title or f"Erdos h  ({label})", out)


if __name__ == "__main__":
    main()