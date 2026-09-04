#!/usr/bin/env python3
"""
Plot how the per-step reward distribution evolves over a run.

For every run directory, every step*/ folder is scanned and each
*.meta.json rollout record contributes its `raw_score` (the evaluator's
number, e.g. the Erdos constant; only valid rollouts have one) and its
`reward` (the training reward the policy gradient saw; invalid rollouts get
the fail value, 0.0). One figure per run is written with three panels:

  1. raw_score distribution per step (valid rollouts), violin + median +
     best, with the config target drawn as a dashed line when known,
  2. gap to target |raw_score - target| per step on a log axis, so the
     structure near the target is visible even when early steps have
     far-away outliers (only drawn when the config has a target),
  3. reward distribution per step (valid rollouts by default; pass
     --include-invalid to fold the zeros in),
  4. valid fraction and rollout count per step.

A CSV of per-step summary statistics is written next to each figure.

Usage:

    python plot_reward_distributions.py                       # every run under runs/
    python plot_reward_distributions.py runs/erdos_Qwen3-8B_0827-1532
    python plot_reward_distributions.py runs/erdos_* --out output/reward_distributions
    python plot_reward_distributions.py --include-invalid --min-valid 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STEP_DIR = re.compile(r"^step(\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_run(run_dir: Path) -> list[dict]:
    """
    Return a list of per-step records, sorted by step index:

        {"step": int, "raw": np.ndarray (valid only), "reward_valid": np.ndarray,
         "reward_all": np.ndarray, "n": int, "n_valid": int}

    Steps with no rollout meta files (e.g. a partially written last step)
    are skipped.
    """
    steps = []
    for d in sorted(run_dir.iterdir()):
        m = _STEP_DIR.match(d.name)
        if not m or not d.is_dir():
            continue
        metas = sorted(d.glob("*.meta.json"))
        if not metas:
            continue
        raw, rew_valid, rew_all = [], [], []
        for f in metas:
            try:
                meta = json.loads(f.read_text())
            except Exception:
                continue
            reward = meta.get("reward")
            if reward is None:
                continue
            reward = float(reward)
            rew_all.append(reward)
            if meta.get("valid"):
                rew_valid.append(reward)
                rs = meta.get("raw_score")
                if rs is not None:
                    raw.append(float(rs))
        if not rew_all:
            continue
        steps.append({
            "step": int(m.group(1)),
            "raw": np.asarray(raw, dtype=float),
            "reward_valid": np.asarray(rew_valid, dtype=float),
            "reward_all": np.asarray(rew_all, dtype=float),
            "n": len(rew_all),
            "n_valid": len(rew_valid),
        })
    steps.sort(key=lambda s: s["step"])
    return steps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _violin_panel(ax, xs, samples, *, color, label, min_n, lower_is_better,
                  target=None):
    """Violin per step with median (dot) and best (marker) overlays."""
    keep = [(x, s) for x, s in zip(xs, samples) if len(s) >= min_n]
    if keep:
        pos = [x for x, _ in keep]
        data = [s for _, s in keep]
        parts = ax.violinplot(data, positions=pos, widths=0.85,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.45)
        med = [float(np.median(s)) for s in data]
        best = [float(s.min() if lower_is_better else s.max()) for s in data]
        ax.plot(pos, med, "o-", color=color, ms=3.5, lw=1.2, label=f"{label} median")
        ax.plot(pos, best, "^" if not lower_is_better else "v", color="black",
                ms=4, lw=0, label=f"{label} best")
    # Steps with too few samples: show whatever points exist.
    thin = [(x, s) for x, s in zip(xs, samples) if 0 < len(s) < min_n]
    for x, s in thin:
        ax.plot([x] * len(s), s, ".", color=color, alpha=0.6, ms=4)
    if target is not None:
        ax.axhline(target, ls="--", lw=1, color="crimson", label="target")
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, frameon=False)


def plot_run(run_dir: Path, steps: list[dict], out_png: Path, *,
             include_invalid: bool, min_n: int) -> None:
    cfg = _load_config(run_dir)
    problem = str(cfg.get("problem", ""))
    target = cfg.get("target")
    # Erdos's raw score is the constant being minimised; everything else in
    # this repo reports a higher-is-better raw score.
    lower_is_better = problem == "erdos"

    xs = [s["step"] for s in steps]
    raw = [s["raw"] for s in steps]
    rew = [s["reward_all"] if include_invalid else s["reward_valid"] for s in steps]
    n = np.array([s["n"] for s in steps])
    n_valid = np.array([s["n_valid"] for s in steps])

    has_target = isinstance(target, (int, float))
    n_panels = 4 if has_target else 3
    ratios = [3, 3, 3, 1.4] if has_target else [3, 3, 1.4]
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(max(8, 0.32 * len(xs) + 4), 3.3 * n_panels),
                             sharex=True, gridspec_kw={"height_ratios": ratios})
    if has_target:
        ax_raw, ax_gap, ax_rew, ax_n = axes
    else:
        ax_raw, ax_rew, ax_n = axes
        ax_gap = None

    _violin_panel(ax_raw, xs, raw, color="tab:blue", label="raw score", min_n=min_n,
                  lower_is_better=lower_is_better,
                  target=float(target) if isinstance(target, (int, float)) else None)
    ax_raw.set_ylabel("raw score" + (" (lower is better)" if lower_is_better else ""))
    ax_raw.set_title(f"{run_dir.name}   problem={problem or '?'}   "
                     f"steps={len(xs)}   rollouts={int(n.sum())}", fontsize=10)

    if ax_gap is not None:
        # Signed gap in the "worse" direction; zero/negative gaps (at or past
        # the target) cannot sit on a log axis, so clamp them to a tiny floor.
        sign = 1.0 if lower_is_better else -1.0
        gaps = [np.maximum(sign * (a - float(target)), 1e-9) for a in raw]
        _violin_panel(ax_gap, xs, gaps, color="tab:purple", label="gap", min_n=min_n,
                      lower_is_better=True)
        ax_gap.set_yscale("log")
        ax_gap.set_ylabel("|raw score - target| (log)")

    _violin_panel(ax_rew, xs, rew, color="tab:orange", label="reward", min_n=min_n,
                  lower_is_better=False)
    ax_rew.set_ylabel("reward" + (" (all rollouts)" if include_invalid
                                  else " (valid rollouts)"))

    frac = np.where(n > 0, n_valid / np.maximum(n, 1), 0.0)
    ax_n.bar(xs, frac, color="tab:green", alpha=0.55, width=0.85, label="valid fraction")
    ax_n.set_ylim(0, 1)
    ax_n.set_ylabel("valid frac")
    ax_n2 = ax_n.twinx()
    ax_n2.plot(xs, n, "k.-", lw=1, ms=3, label="rollouts/step")
    ax_n2.set_ylabel("rollouts")
    ax_n.set_xlabel("step")
    ax_n.grid(True, alpha=0.25)
    h1, l1 = ax_n.get_legend_handles_labels()
    h2, l2 = ax_n2.get_legend_handles_labels()
    ax_n.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
    ax_n.set_xticks(xs)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_csv(steps: list[dict], out_csv: Path) -> None:
    def stats(a: np.ndarray):
        if len(a) == 0:
            return [""] * 6
        return [f"{v:.6g}" for v in (a.min(), np.percentile(a, 25), np.median(a),
                                     a.mean(), np.percentile(a, 75), a.max())]

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "n", "n_valid", "valid_frac",
                    "raw_min", "raw_p25", "raw_median", "raw_mean", "raw_p75", "raw_max",
                    "reward_min", "reward_p25", "reward_median", "reward_mean",
                    "reward_p75", "reward_max"])
        for s in steps:
            frac = s["n_valid"] / s["n"] if s["n"] else 0.0
            w.writerow([s["step"], s["n"], s["n_valid"], f"{frac:.4f}",
                        *stats(s["raw"]), *stats(s["reward_valid"])])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and any(_STEP_DIR.match(c.name) for c in p.iterdir()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: all under --root)")
    ap.add_argument("--root", default="runs", help="where to look when no runs are given")
    ap.add_argument("--out", default="output/reward_distributions",
                    help="output directory for PNG + CSV files")
    ap.add_argument("--include-invalid", action="store_true",
                    help="fold invalid rollouts (reward = fail value) into the reward panel")
    ap.add_argument("--min-valid", type=int, default=2,
                    help="minimum samples for a violin; fewer are drawn as points")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs] if args.runs else _discover_runs(Path(args.root))
    if not runs:
        print("no run directories found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    done = 0
    for run_dir in runs:
        if not run_dir.is_dir():
            print(f"[skip] not a directory: {run_dir}", file=sys.stderr)
            continue
        steps = load_run(run_dir)
        if not steps:
            print(f"[skip] no rollout meta files: {run_dir}", file=sys.stderr)
            continue
        png = out_dir / f"{run_dir.name}.png"
        plot_run(run_dir, steps, png, include_invalid=args.include_invalid,
                 min_n=args.min_valid)
        write_csv(steps, out_dir / f"{run_dir.name}.csv")
        print(f"[ok] {run_dir.name}: {len(steps)} steps, "
              f"{sum(s['n'] for s in steps)} rollouts -> {png}")
        done += 1
    print(f"{done}/{len(runs)} runs plotted into {out_dir}/")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
