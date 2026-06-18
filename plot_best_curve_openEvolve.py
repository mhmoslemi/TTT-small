#!/usr/bin/env python3
"""
Compare OpenEvolve (one or more models) vs TTT vs TTT+Elo on circle_packing.

  - OpenEvolve : dense best-so-far curve(s) parsed from log(s), sampled every --every.
                 One curve per model, each its own legend entry.
  - TTT        : one point per step at iteration (step+1)*STEP_ITERS.
  - TTT + Elo  : same.

Usage:
    python plot_best_curve.py                                  # newest log (labeled TTT_MODEL) + TTT lists
    python plot_best_curve.py --oe Qwen3-8B=logs/qwen.log --oe Llama3-8B=logs/llama.log
    python plot_best_curve.py logs/qwen.log --ttt-model Qwen3-8B --ylim 2.45 2.66 --target 2.635
    python plot_best_curve.py --no-openevolve                  # just TTT vs Elo
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless / SSH safe
import matplotlib.pyplot as plt
import numpy as np

# ============================ hyperparameters ============================
SAMPLE_EVERY = 64      # openevolve: plot one point every N iterations
STEP_ITERS   = 512     # iterations per TTT step (also the vertical-line spacing)
TTT_CUMMAX   = False   # True => plot cumulative best-so-far for the TTT series too
TTT_MODEL    = "Qwen3-8B"   # model used for the TTT / TTT+Elo runs below
YLIM = None            # e.g. (2.4, 2.65); None = autoscale
XLIM = None            # e.g. (0, 7680); None = autoscale
TARGET = 2.635          # horizontal reference line; ~2.635 for n=26 unit-square packing

# OpenEvolve runs to overlay: (model_label, log_or_json_path). path=None => newest in DEFAULT_LOG_DIR.
# Add more models here, or pass them on the CLI with --oe LABEL=PATH.
# OPENEVOLVE_RUNS = [
#     ("Qwen3-8B", None),
# ]

OPENEVOLVE_RUNS = [
    ("Qwen3-8B", '/work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs/openevolve_20260611_184912-qwen8B.log'),
    ("Qwen3-8B + Qwen3-32B", "/work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs/openevolve_20260611_210406.log"),
]


# best sum_radii per step (step 0..14). Paste new runs here.
TTT = [
    2.5261384438, 2.6035808140, 2.6083336157, 2.6141354338, 2.6235348955,
    2.6282124182, 2.6282124182, 2.6302148014, 2.6310935891, 2.6310935891,
    2.6310935891, 2.6310935891, 2.6358957386, 2.6342924021, 2.6342924021,
]
TTT_ELO = [
    2.5518054052, 2.6221449045, 2.6221772829, 2.6270835278, 2.6297131377,
    2.6313341376, 2.6330195957, 2.6313343652, 2.6313496673, 2.6340630826,
    2.6323260073, 2.6323260265, 2.6353127766, 2.6359674489, 2.6353127769,
]
# =========================================================================

DEFAULT_LOG_DIR = "/work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs"

# distinct colors for OpenEvolve models (kept away from the TTT orange / Elo teal)
OE_COLORS = ["#1f4e79", "#9467bd", "#8c564b", "#17becf", "#6a4c93", "#444444"]

_ITER_RE = re.compile(r"- Iteration (\d+):.*completed")
_SR_RE = re.compile(r"- Metrics:.*sum_radii=([0-9.]+)")


def parse_log(path):
    iters, srs, cur = [], [], None
    with open(path) as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                cur = int(m.group(1))
                continue
            if cur is not None:
                m = _SR_RE.search(line)
                if m:
                    iters.append(cur)
                    srs.append(float(m.group(1)))
                    cur = None
    return np.array(iters), np.array(srs)


def parse_json(path):
    with open(path) as f:
        data = json.load(f)
    return (np.array([d["iteration"] for d in data]),
            np.array([d["sum_radii"] for d in data]))


def best_so_far_curve(iters, srs, every):
    order = np.argsort(iters)
    iters, srs = iters[order], srs[order]
    cummax = np.maximum.accumulate(srs)
    grid = np.arange(0, int(iters.max()) + 1, every)
    idx = np.searchsorted(iters, grid, side="right") - 1
    mask = idx >= 0
    return grid[mask], cummax[idx[mask]]


def newest_log():
    cands = sorted(glob.glob(os.path.join(DEFAULT_LOG_DIR, "openevolve_*.log")),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def split_oe(s, default_label):
    """Parse a --oe / positional value of the form 'Label=path' or 'path'."""
    if "=" in s:
        label, path = s.split("=", 1)
        return label, path
    return default_label, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", help="single openevolve .log/.json (labeled with --ttt-model)")
    p.add_argument("--oe", action="append", default=[], metavar="LABEL=PATH",
                   help="openevolve run to overlay; repeatable")
    p.add_argument("--every", type=int, default=SAMPLE_EVERY)
    p.add_argument("--step-iters", type=int, default=STEP_ITERS, help="iterations per TTT step")
    p.add_argument("--ttt-model", default=TTT_MODEL, help="model label for the TTT / TTT+Elo series")
    p.add_argument("--ttt-cummax", action="store_true", default=TTT_CUMMAX)
    p.add_argument("--no-openevolve", action="store_true")
    p.add_argument("--ylim", type=float, nargs=2, default=YLIM, metavar=("LO", "HI"))
    p.add_argument("--xlim", type=float, nargs=2, default=XLIM, metavar=("LO", "HI"))
    p.add_argument("--target", type=float, default=TARGET)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    si = args.step_iters

    # ---- assemble the list of OpenEvolve runs: (label, path) ----
    runs = []
    if not args.no_openevolve:
        for s in args.oe:
            runs.append(split_oe(s, args.ttt_model))
        if args.log:
            runs.append(split_oe(args.log, args.ttt_model))
        if not runs:
            runs = list(OPENEVOLVE_RUNS)
        # resolve None paths to the newest log
        resolved = []
        for label, path in runs:
            path = path or newest_log()
            if path is not None:
                resolved.append((label, path))
        runs = resolved

    fig, ax = plt.subplots(figsize=(9, 5))

    # vertical step-boundary markers
    n_steps = max(len(TTT), len(TTT_ELO))
    xmax = args.xlim[1] if args.xlim else (n_steps + 1) * si
    for xb in range(si, int(xmax) + 1, si):
        ax.axvline(xb, color="0.8", ls="--", lw=0.8, zorder=0)

    # OpenEvolve dense curves, one per model
    for i, (label, path) in enumerate(runs):
        iters, srs = parse_json(path) if path.endswith(".json") else parse_log(path)
        if iters.size == 0:
            continue
        gx, gy = best_so_far_curve(iters, srs, args.every)
        ax.plot(gx, gy, lw=1.5, color=OE_COLORS[i % len(OE_COLORS)],
                label=f"OpenEvolve ({label})", zorder=2)

    # TTT series: one point per step at x = (step+1)*step_iters
    def add_series(values, color, label):
        y = np.array(values, dtype=float)
        if args.ttt_cummax:
            y = np.maximum.accumulate(y)
        x = (np.arange(len(y)) + 1) * si
        ax.plot(x, y, marker="o", ms=5, lw=1.5, color=color, label=label, zorder=3)

    add_series(TTT, "#e07b39", f"TTT ({args.ttt_model})")
    add_series(TTT_ELO, "#2a9d8f", f"TTT + Elo ({args.ttt_model})")

    if args.target is not None:
        ax.axhline(args.target, ls="--", lw=1, color="#aa3333", zorder=1,
                   label=f"target = {args.target:g}")

    ax.set_xlabel("iteration")
    ax.set_ylabel("best seen sum_radii")
    ax.set_title("circle_packing: OpenEvolve vs TTT vs TTT+Elo")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if args.ylim:
        ax.set_ylim(*args.ylim)
    if args.xlim:
        ax.set_xlim(*args.xlim)

    out_dir = os.path.dirname(runs[0][1]) if runs else "."
    out = args.out or os.path.join(out_dir, "compare_best.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("openevolve runs: " + (", ".join(f"{l}" for l, _ in runs) or "(none)"))
    print(f"TTT model: {args.ttt_model} | steps: {len(TTT)} / {len(TTT_ELO)} | step_iters: {si}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()