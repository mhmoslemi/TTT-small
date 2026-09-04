#!/usr/bin/env python3
"""
Per-step reward PDFs, stacked vertically, one figure per run.

Each step gets a row of three panels. In each panel y is the fraction of the
rollouts inside that panel's window per bin (each panel sums to one on its
own), fixed to 0..1:
  full range:  0.9 x min .. 1.1 x max of the step (-0.1 floor when min is 0)
  low tail:    min .. 1.2 x min, with a small margin either side
  high tail:   0.995 x max .. max, with a small margin either side
Rows are stacked top (step 0) to bottom (last step), each with its own x ticks.

Usage:

    python plot_reward_pdf.py                                 # every run under runs/
    python plot_reward_pdf.py runs/erdos_Qwen3-8B_0827-1532
    python plot_reward_pdf.py runs/erdos_* --bins 400 --valid-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advantage import evt_adaptive_alpha, gini_adaptive_alpha  # noqa: E402

_STEP_DIR = re.compile(r"^step(\d+)$")


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def max_possible_reward(cfg: dict) -> float | None:
    """Reward the config target would earn, mirroring the problem modules."""
    target = cfg.get("target")
    if not isinstance(target, (int, float)) or target <= 0:
        return None
    problem = str(cfg.get("problem", ""))
    if problem in ("erdos", "ac1", "ac2", "denoising"):
        return 1.0 / float(target)
    if problem == "gpu_mode":
        scale = cfg.get("score_scale")
        if isinstance(scale, (int, float)) and scale > 0:
            return float(scale) / float(target)
        return None
    if problem == "circle_packing":
        return float(target)
    return None


def load_run(run_dir: Path, valid_only: bool,
             min_rollouts: int = 0) -> list[tuple[int, np.ndarray]]:
    """
    Return [(step, rewards)] sorted by step. Steps with no rollouts, or with
    at most `min_rollouts` rollouts (counted before the valid-only filter),
    are skipped.
    """
    steps = []
    for d in sorted(run_dir.iterdir()):
        m = _STEP_DIR.match(d.name)
        if not m or not d.is_dir():
            continue
        rewards, n_total = [], 0
        for f in d.glob("*.meta.json"):
            try:
                meta = json.loads(f.read_text())
            except Exception:
                continue
            r = meta.get("reward")
            if r is None:
                continue
            n_total += 1
            if valid_only and not meta.get("valid"):
                continue
            rewards.append(float(r))
        if rewards and n_total > min_rollouts:
            steps.append((int(m.group(1)), np.asarray(rewards, dtype=float)))
    steps.sort(key=lambda t: t[0])
    return steps


def _hist_panel(ax, rewards, x_lo, x_hi, bins, *, clip):
    """
    Draw the 0..1 fraction-per-bin histogram of `rewards` over [x_lo, x_hi].
    Bins are normalised to the rollouts that fall inside the window, so every
    panel sums to one on its own (a tail panel is a distribution over the tail).
    """
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    edges = np.linspace(x_lo, x_hi, bins + 1)
    data = np.clip(rewards, x_lo, x_hi) if clip else rewards
    counts, _ = np.histogram(data, bins=edges)
    frac = counts / max(counts.sum(), 1)  # sums to 1 within this window
    # Step outline + fill: stays visible even when a bin is narrower than a
    # pixel (bars would vanish or fade at high bin counts).
    ax.stairs(frac, edges, fill=True, color="tab:blue", alpha=0.35)
    ax.stairs(frac, edges, color="tab:blue", lw=1.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _with_margin(lo, hi, frac=0.05):
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-3)
    return lo - frac * span, hi + frac * span


def plot_run(run_dir: Path, steps, out_png: Path, *, bins: int, row_height: float,
             evt_line: str = "threshold", gini_alpha_base: float = 0.2,
             gini_gamma: float = 1.0) -> None:
    cfg = _load_config(run_dir)

    n_rows = len(steps)
    # Three panels per step, each normalised to sum to one over its window,
    # all on the same 0..1 y scale:
    #   full:      0.9 x min .. 1.1 x max (floor -0.1 when the min is zero)
    #   low tail:  min .. 1.2 x min       (plus a small margin either side)
    #   high tail: 0.995 x max .. max     (plus a small margin either side)
    # Fourth column is a text-only panel (axes hidden) so the per-step
    # numbers never overlap the histograms.
    fig, axes = plt.subplots(n_rows, 4, sharex=False, squeeze=False,
                             figsize=(15, max(3.0, row_height * n_rows + 1.0)),
                             gridspec_kw={"width_ratios": [2, 1, 1, 1.7]})

    print(f"{run_dir.name}")
    print(f"  {'step':>4} {'G':>5} {'G>0':>5} {'max':>10} | "
          f"{'EVT k*':>6} {'tail >':>10} {'n>':>5} {'a_all':>7} {'a_pos':>7} "
          f"{'xi_gpd':>8} {'sigma':>9} {'KS':>6} {'method':>11} | "
          f"{'Gini_all':>8} {'a_all':>7} {'k':>5} {'tail >':>10} {'n>':>5} | "
          f"{'Gini_pos':>8} {'a_pos':>7} {'k':>5} {'tail >':>10} {'n>':>5}")
    for row, (step, rewards) in zip(axes, steps):
        ax_full, ax_low, ax_high, ax_txt = row
        ax_txt.axis("off")
        r_lo, r_hi = float(rewards.min()), float(rewards.max())

        x_lo = -0.1 if r_lo <= 0 else 0.9 * r_lo
        x_hi = 1.1 * r_hi
        _hist_panel(ax_full, rewards, x_lo, x_hi, bins, clip=True)
        ax_full.set_ylabel(f"step {step}", rotation=0, ha="right", va="center", fontsize=9)

        if r_lo <= 0:
            lo_a, lo_b = -0.1, 0.1
        else:
            lo_a, lo_b = _with_margin(r_lo, 1.2 * r_lo)
        _hist_panel(ax_low, rewards, lo_a, lo_b, bins, clip=False)

        if r_hi <= 0:
            hi_a, hi_b = -0.1, 0.1
        else:
            hi_a, hi_b = _with_margin(0.995 * r_hi, r_hi)
        _hist_panel(ax_high, rewards, hi_a, hi_b, bins, clip=False)

        # EVT adaptive alpha for this step (GPD fit to excesses, threshold by
        # KS minimisation). The red line marks where the upper tail starts:
        # the threshold R_(k*+1), i.e. the k*-th largest reward (alpha = k*/G
        # is a fraction of rollouts, not of the reward scale; --evt-line
        # alpha_max keeps the alpha * max line).
        # Four cutoffs per step (two-part / hurdle view of a zero-inflated
        # batch): each estimator once with alpha over ALL rollouts and once
        # over the positive (valid) rollouts only.
        #   red     EVT, alpha = k*/G          orange  EVT, alpha = k*/G_pos
        #   green   Gini over all rewards      purple  Gini over r > 0
        # The EVT fit itself always uses r > 0, so red and orange share the
        # same threshold and differ only in alpha.
        n_pos = int(np.sum(rewards > 0))
        alpha, k_star, threshold, evt = evt_adaptive_alpha(rewards)
        alpha_pos, _k_pos, _thr_pos, _evt_pos = evt_adaptive_alpha(
            rewards, positive_only=True)
        x_evt = threshold if evt_line == "threshold" else alpha * r_hi
        x_evt_pos = threshold if evt_line == "threshold" else alpha_pos * r_hi
        for ax in (ax_full, ax_high):
            ax.axvline(x_evt, color="red", lw=0.6)
            ax.axvline(x_evt_pos, color="darkorange", lw=0.6, ls=(0, (3, 2)))
        n_above_evt = int(np.sum(rewards > threshold))

        ga_alpha, ga_k, ga_thr, gini_all = gini_adaptive_alpha(
            rewards, alpha_base=gini_alpha_base, gamma=gini_gamma, positive_only=False)
        gp_alpha, gp_k, gp_thr, gini_pos = gini_adaptive_alpha(
            rewards, alpha_base=gini_alpha_base, gamma=gini_gamma, positive_only=True)
        for ax in (ax_full, ax_high):
            ax.axvline(ga_thr, color="green", lw=0.6)
            ax.axvline(gp_thr, color="purple", lw=0.6, ls=(0, (3, 2)))
        n_above_ga = int(np.sum(rewards > ga_thr))
        n_above_gp = int(np.sum(rewards > gp_thr))

        # Text panel to the right of the histograms, one colored line per method.
        lines = [
            ("red", f"EVT all   alpha={alpha:.3f}  k*={k_star}  tail>{threshold:.5g}  "
                    f"n_above={n_above_evt}"),
            ("darkorange", f"EVT >0    alpha={alpha_pos:.3f}  k*={k_star}  "
                           f"tail>{threshold:.5g}  n_above={n_above_evt}"),
            ("green", f"Gini all  G={gini_all:.3f}  alpha={ga_alpha:.3f}  k={ga_k}  "
                      f"tail>{ga_thr:.5g}  n_above={n_above_ga}"),
            ("purple", f"Gini >0   G={gini_pos:.3f}  alpha={gp_alpha:.3f}  k={gp_k}  "
                       f"tail>{gp_thr:.5g}  n_above={n_above_gp}"),
        ]
        ax_txt.text(0.0, 1.0, f"G={len(rewards)}  G>0={n_pos}  max={r_hi:.5g}",
                    transform=ax_txt.transAxes, ha="left", va="top",
                    fontsize=6.5, color="black", family="monospace")
        for i, (color, text) in enumerate(lines):
            ax_txt.text(0.0, 0.78 - 0.22 * i, text, transform=ax_txt.transAxes,
                        ha="left", va="top", fontsize=6.5, color=color,
                        family="monospace")

        print(f"  {step:>4} {len(rewards):>5} {n_pos:>5} {r_hi:>10.5g} | "
              f"{k_star:>6} {threshold:>10.5g} {n_above_evt:>5} {alpha:>7.4f} {alpha_pos:>7.4f} "
              f"{evt['xi_gpd']:>8.3f} {evt['sigma']:>9.3g} {evt['ks']:>6.3f} "
              f"{evt['method']:>11} | "
              f"{gini_all:>8.3f} {ga_alpha:>7.4f} {ga_k:>5} {ga_thr:>10.5g} {n_above_ga:>5} | "
              f"{gini_pos:>8.3f} {gp_alpha:>7.4f} {gp_k:>5} {gp_thr:>10.5g} {n_above_gp:>5}")

    axes[0][0].set_title("full range", fontsize=8)
    axes[0][1].set_title("low tail: min .. 1.2 x min", fontsize=8)
    axes[0][2].set_title("high tail: 0.995 x max .. max", fontsize=8)
    for ax in axes[-1][:3]:
        ax.set_xlabel("reward")
    fig.suptitle(f"{run_dir.name}   ({cfg.get('problem', '?')})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _discover_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and any(_STEP_DIR.match(c.name) for c in p.iterdir()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: all under --root)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="output/reward_pdf")
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--valid-only", action="store_true",
                    help="drop invalid rollouts (they otherwise sit at reward 0)")
    ap.add_argument("--row-height", type=float, default=0.8, help="inches per step panel")
    ap.add_argument("--min-rollouts", type=int, default=200,
                    help="only keep steps with more than this many rollouts")
    ap.add_argument("--min-steps", type=int, default=6,
                    help="only plot runs with more than this many kept steps")
    ap.add_argument("--evt-line", choices=["threshold", "alpha_max"], default="threshold",
                    help="where the red EVT line goes: the Hill tail threshold "
                         "R_(k*+1) (default) or alpha * max reward")
    ap.add_argument("--gini-alpha-base", type=float, default=0.2,
                    help="alpha_base for the green Gini-modulated alpha")
    ap.add_argument("--gini-gamma", type=float, default=1.0,
                    help="gamma exponent for the green Gini-modulated alpha")
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
        steps = load_run(run_dir, valid_only=args.valid_only,
                         min_rollouts=args.min_rollouts)
        if len(steps) <= args.min_steps:
            print(f"[skip] {run_dir.name}: only {len(steps)} steps with "
                  f">{args.min_rollouts} rollouts (need >{args.min_steps})",
                  file=sys.stderr)
            continue
        png = out_dir / f"{run_dir.name}.png"
        plot_run(run_dir, steps, png, bins=args.bins, row_height=args.row_height,
                 evt_line=args.evt_line, gini_alpha_base=args.gini_alpha_base,
                 gini_gamma=args.gini_gamma)
        print(f"[ok] {run_dir.name}: {len(steps)} steps -> {png} (+ .pdf)")
        done += 1
    print(f"{done}/{len(runs)} runs plotted into {out_dir}/")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
