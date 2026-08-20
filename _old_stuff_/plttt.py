#!/usr/bin/env python3
"""
Best-so-far comparison plot across runs.

Reads the per-rollout .meta.json files a run writes under
runs/<name>/step<NN>/, takes the running maximum over valid rollouts, and draws
one curve per run. Every step where the running max improves is a discovery by
Def. 1, so every improvement gets a marker.

Not every improvement gets a LABEL. Runs converge into a band a few thousandths
wide, so labelling all of them puts every box in the same strip of the figure
and none of them are readable. Three things keep it legible:

  thinning     --annotate auto keeps the first, the last, and the largest
               jumps, up to --max-annot per run. The full list of discoveries
               is always printed to stdout, so nothing is lost, it just is not
               all on the figure.

  placement    labels are placed by trying a ladder of offsets and taking the
               first that clears every label already placed, measured on the
               real rendered boxes in display space. Anything pushed far from
               its point gets a leader line.

  zoom         --ymin clips the wasted lower region, and --split gives each run
               its own panel so the two runs cannot collide with each other at
               all.

Put this file at the repo root, next to train_multy.py.

    python plot_best_so_far.py \
        runs/circle_packing_n26_Qwen3-8B_0812-1928 \
        runs/circle_packing_n26_Qwen3-8B_0812-1933 \
        --labels "no memory" "memory" \
        --ymin 2.4 --out best_so_far.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")          # write files without needing a display
import matplotlib.pyplot as plt

_STEP_DIR_RE = re.compile(r"^step(\d+)$")

# Validated categorical palette (light surface), fixed order -- never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


# ----------------------------------------------------------------------
# Reading a run
# ----------------------------------------------------------------------
def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def step_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Numerically sorted (step_index, path). step10 must not sort before step2."""
    out = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_DIR_RE.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    return sorted(out)


def score_of(meta: dict, metric: str) -> Optional[float]:
    """
    The number to maximize. `raw_score` is the problem's own metric (sum of
    radii for circle packing); `reward` is what the trainer optimizes. They
    differ whenever a problem rescales, so which one you plot is a flag rather
    than a guess.
    """
    if not meta.get("valid"):
        return None
    value = meta.get(metric)
    if value is None and metric == "raw_score":
        value = meta.get("reward")     # problems that never set raw_score
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_run(run_dir: Path, metric: str) -> Dict[str, list]:
    """
    Steps, the per-step max, the running max, and rollout counts. A step with
    no valid rollout carries the previous running max forward, so the curve is
    defined at every step rather than skipping gaps.
    """
    steps, step_best, running, n_valid, n_total = [], [], [], [], []
    best = None

    for step_idx, sdir in step_dirs(run_dir):
        scores, total = [], 0
        for meta_path in sorted(sdir.glob("*.meta.json")):
            total += 1
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            s = score_of(meta, metric)
            if s is not None:
                scores.append(s)

        this = max(scores) if scores else None
        if this is not None:
            best = this if best is None else max(best, this)

        steps.append(step_idx)
        step_best.append(this)
        running.append(best)
        n_valid.append(len(scores))
        n_total.append(total)

    return {"steps": steps, "step_best": step_best, "running": running,
            "n_valid": n_valid, "n_total": n_total}


def improvements(steps: Sequence[int], running: Sequence[Optional[float]],
                 min_delta: float) -> List[Tuple[int, float]]:
    """Steps where the running max strictly improved, with the new value."""
    out, prev = [], None
    for step, value in zip(steps, running):
        if value is None:
            continue
        if prev is None or value > prev + min_delta:
            out.append((step, value))
            prev = value
        elif value > prev:
            prev = value       # improved by less than min_delta: track, do not mark
    return out


def thin(imps: List[Tuple[int, float]], max_n: int) -> List[Tuple[int, float]]:
    """
    Keep the first, the last, and the biggest jumps in between. The first shows
    where the run started, the last is the result, and the largest jumps are
    the only interior points whose exact value tells you much.
    """
    if max_n <= 0 or len(imps) <= max_n:
        return list(imps)
    keep = {0, len(imps) - 1}
    deltas = sorted(((imps[i][1] - imps[i - 1][1], i)
                     for i in range(1, len(imps))), reverse=True)
    for _, i in deltas:
        if len(keep) >= max_n:
            break
        keep.add(i)
    return [imps[i] for i in sorted(keep)]


def auto_label(run_dir: Path) -> str:
    cfg = read_config(run_dir)
    if "memory" in cfg:
        return "memory" if cfg.get("memory") else "no memory"
    return run_dir.name


# ----------------------------------------------------------------------
# Annotation placement
# ----------------------------------------------------------------------
# Tried in order. The first that clears everything already placed wins, so a
# label drifts outward only as far as it has to.
_OFFSETS = [
    (0, 18), (0, -24), (0, 42), (0, -48), (0, 66), (0, -72),
    (40, 18), (-40, 18), (40, -24), (-40, -24),
    (54, 48), (-54, 48), (54, -54), (-54, -54),
    (0, 92), (0, -98),
]


def _annotate(ax, x, y, dx, dy, color, fontsize):
    # No box -- just the number -- with a thin leader line in the series
    # colour back to its point, so a label can be pushed well clear of the
    # curve (and of other labels) without losing which point it belongs to.
    # The text itself stays in ink, not the series hue: a light categorical
    # color (yellow, aqua) is illegible as text, and identity is already
    # carried by the leader line + dot, not by coloring the number.
    return ax.annotate(
        f"{y:.6f}",
        xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha="center", va="bottom" if dy >= 0 else "top",
        fontsize=fontsize, color=INK_SECONDARY, zorder=6,
        arrowprops=dict(arrowstyle="-", linewidth=0.7, color=color,
                        alpha=0.75, shrinkA=0, shrinkB=3),
    )


def place_labels(ax, points: Sequence[Tuple[float, float]], color: str,
                 fontsize: float, placed: list) -> None:
    """
    Place one label per point, avoiding every box already in `placed`.

    Overlap is measured on the rendered text extents in display space, which is
    the only way to be right about it: a box's width depends on the font and
    the figure size, neither of which is knowable in data coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()                      # a renderer must exist before measuring
    renderer = fig.canvas.get_renderer()

    for (x, y) in points:
        chosen = None
        for dx, dy in _OFFSETS:
            ann = _annotate(ax, x, y, dx, dy, color, fontsize)
            try:
                bb = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.28)
            except Exception:
                chosen = ann
                break
            if not any(bb.overlaps(other) for other in placed):
                placed.append(bb)
                chosen = ann
                break
            ann.remove()
        if chosen is None:
            # Everything collided. Place it at the widest offset anyway rather
            # than dropping a discovery off the figure entirely.
            dx, dy = _OFFSETS[-1]
            chosen = _annotate(ax, x, y, dx, dy, color, fontsize)
            try:
                placed.append(chosen.get_window_extent(renderer=renderer))
            except Exception:
                pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot best-so-far per step for one or more runs.")
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one label per run; defaults to memory on/off from config.json")
    ap.add_argument("--metric", default="raw_score",
                    choices=["raw_score", "reward"],
                    help="which field to maximize (default: raw_score)")
    ap.add_argument("--out", default="best_so_far.png", help="output image path")

    ap.add_argument("--annotate", default="auto",
                    choices=["auto", "all", "ends", "none"],
                    help="auto = first, last and the biggest jumps (default)")
    ap.add_argument("--max-annot", type=int, default=5,
                    help="labels per run under --annotate auto (default: 5)")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="ignore improvements smaller than this entirely")

    ap.add_argument("--split", action="store_true",
                    help="one panel per run, sharing the x axis")
    ap.add_argument("--ymin", type=float, default=None,
                    help="clip the y axis from below; hides the early low region")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--target", type=float, default=None,
                    help="horizontal target line; read from config.json if omitted")
    ap.add_argument("--no-target", action="store_true")

    ap.add_argument("--fontsize", type=float, default=7.5)
    ap.add_argument("--figsize", nargs=2, type=float, default=None)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dirs = [Path(r).expanduser() for r in args.runs]
    for d in run_dirs:
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")

    labels = args.labels or [auto_label(d) for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"got {len(labels)} labels for {len(run_dirs)} runs")

    data = [read_run(d, args.metric) for d in run_dirs]

    target = args.target
    if target is None and not args.no_target:
        for d in run_dirs:
            t = read_config(d).get("target")
            if t is not None:
                target = float(t)
                break

    # ---- figure ----
    n = len(run_dirs)
    figsize = tuple(args.figsize) if args.figsize else (
        (11.0, 3.2 * n + 0.8) if args.split else (11.0, 6.0))
    if args.split:
        fig, axs = plt.subplots(n, 1, figsize=figsize, sharex=True)
        axes = list(axs) if n > 1 else [axs]
    else:
        fig, ax0 = plt.subplots(figsize=figsize)
        axes = [ax0] * n

    all_values, curves = [], []

    for i, (label, run) in enumerate(zip(labels, data)):
        ax = axes[i]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        pts = [(s, v) for s, v in zip(run["steps"], run["running"]) if v is not None]
        if not pts:
            print(f"[warn] {label}: no valid rollouts found, skipping")
            continue
        xs, ys = zip(*pts)
        all_values.extend(ys)

        # steps-post: the value holds until the next improvement, which is what
        # "best so far" means between discoveries.
        ax.plot(xs, ys, drawstyle="steps-post", linewidth=2.0,
                color=color, solid_capstyle="round", solid_joinstyle="round",
                label=label, zorder=3)

        # A dot at every step, not only at discoveries -- otherwise the
        # marker spacing along x looks arbitrary (dense where the run keeps
        # improving, empty for long flat stretches) when really every step
        # ran a rollout. Small, with a surface-colored ring so it stays
        # legible sitting on the line.
        ax.scatter(xs, ys, s=22, color=color, zorder=3.5,
                   edgecolors=SURFACE, linewidths=1.2)

        # Discoveries get the emphasized marker: bigger, same ring.
        imps = improvements(run["steps"], run["running"], args.min_delta)
        if imps:
            ix, iy = zip(*imps)
            ax.scatter(ix, iy, s=64, color=color, zorder=4,
                       edgecolors=SURFACE, linewidths=1.6)

        if args.annotate == "all":
            shown = imps
        elif args.annotate == "ends":
            shown = (imps[:1] + imps[-1:]) if len(imps) > 1 else imps
        elif args.annotate == "auto":
            shown = thin(imps, args.max_annot)
        else:
            shown = []
        curves.append((ax, shown, color))

        print(f"\n{label}  ({run_dirs[i].name})")
        print(f"  discoveries: {len(imps)}   "
              f"valid rollouts: {sum(run['n_valid'])}/{sum(run['n_total'])}")
        for step, value in imps:
            print(f"    step {step:>3}   {value:.6f}")
        if imps:
            print(f"  final: {imps[-1][1]:.6f} (first reached at step {imps[-1][0]})")

    # ---- axis cosmetics, before any text is measured ----
    ylabel = ("best valid metric so far" if args.metric == "raw_score"
              else "best reward so far")
    unique_axes = list(dict.fromkeys(axes))
    for i, ax in enumerate(unique_axes):
        if target is not None:
            ax.axhline(target, linestyle=(0, (5, 4)), linewidth=1.1,
                      color=INK_MUTED, zorder=2)
        ax.grid(True, axis="y", color=GRID, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
        ax.set_facecolor(SURFACE)
        ax.set_ylabel(labels[i] if args.split else ylabel,
                      color=INK_SECONDARY, fontsize=10.5)
    fig.patch.set_facecolor(SURFACE)
    if target is not None:
        all_values.append(target)

    axes[-1].set_xlabel("search step", color=INK_SECONDARY, fontsize=10.5)
    if args.split:
        fig.suptitle(args.title or "Best-so-far per step", color=INK_PRIMARY)
    else:
        axes[0].set_title(args.title or "Best-so-far per step",
                          color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
        leg = axes[0].legend(loc="lower right", frameon=False, fontsize=10.5)
        for text in leg.get_texts():
            text.set_color(INK_SECONDARY)

    if all_values:
        lo = args.ymin if args.ymin is not None else min(all_values)
        hi = args.ymax if args.ymax is not None else max(all_values)
        pad = (hi - lo) * 0.14 or max(abs(hi), 1.0) * 0.02
        for ax in unique_axes:
            ax.set_ylim(lo - pad * 0.35, hi + pad * 1.9)

    if target is not None:
        axes[0].annotate(f"target {target:.6f}", xy=(0.995, target),
                         xycoords=("axes fraction", "data"), ha="right",
                         va="bottom", fontsize=args.fontsize, color=INK_MUTED)

    fig.tight_layout()

    # ---- labels last, once the axes are final ----
    # Single-axis mode shares one collision set across runs so the two colours
    # cannot overlap each other; split mode gives each panel its own.
    if args.split:
        for ax, shown, color in curves:
            place_labels(ax, shown, color, args.fontsize, [])
    else:
        placed = []
        for ax, shown, color in curves:
            place_labels(ax, shown, color, args.fontsize, placed)

    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
