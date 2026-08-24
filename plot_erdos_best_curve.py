#!/usr/bin/env python3
"""Plot the best valid raw score seen through every training step.

Usage:
    python plot_erdos_best_curve.py runs/tmo123
    python3 plot_erdos_best_curve.py runs/tmo123 --out figures/progress.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


_STEP_RE = re.compile(r"step(\d+)")

# CVD-friendly palette and paper-style surface from the reference plot.
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _step_number(meta, path: Path):
    try:
        return int(meta["step"])
    except (KeyError, TypeError, ValueError):
        match = _STEP_RE.search(path.parent.name)
        return int(match.group(1)) if match else None


def _raw_score(meta, problem):
    """Return the raw metric, including old Erdos reward-only metadata."""
    raw = _finite_float(meta.get("raw_score"))
    if raw is not None:
        return raw
    reward = _finite_float(meta.get("reward"))
    if problem == "erdos" and reward is not None and reward > 0:
        # problems/erdos.py uses reward = 1 / (1e-8 + c5_bound).
        return (1.0 / reward) - 1e-8
    return None


def _run_info(run_dir):
    """Return config plus the metric name and optimization direction."""
    config_path = Path(run_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        config = {}

    problem = str(config.get("problem", "unknown")).strip().lower()
    problem_type = str(config.get("problem_type", "")).strip().lower()
    if problem in ("circle_packing", "circle", "circles"):
        return config, problem, "sum of radii", True
    if problem in ("erdos", "erdos_min_overlap", "erdos_minimum_overlap"):
        return config, "erdos", "C₅ bound", False
    if problem in ("denoising", "single_cell", "single_cell_analysis", "scrna"):
        return config, problem, "MSE", False
    if problem in ("ac1", "ac2", "ac_inequalities", "autocorrelation",
                   "autocorrelation_inequalities"):
        kind = problem if problem in ("ac1", "ac2") else (problem_type or "ac1")
        return config, problem, ("lower bound" if kind == "ac2" else "upper bound"), kind == "ac2"
    if problem in ("gpu_mode", "kernel", "kernel_engineering", "trimul",
                   "mla_decode_nvidia", "mla"):
        return config, problem, "runtime (microseconds)", False

    # Unknown/custom problems still work: reward has a repository-wide
    # higher-is-better contract. The plotted metric is therefore reward.
    return config, problem, "reward", True


def load_best_curve(run_dir):
    """Return (steps, cumulative_best, per_step_best, valid_rollout_count)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"not a directory: {run_dir}")

    _, problem, metric_name, maximize = _run_info(run_dir)

    best_by_step = {}
    seen_steps = set()
    valid_count = 0
    pattern = "step*/step*_group*_rollout*.meta.json"
    for path in sorted(run_dir.glob(pattern)):
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        step = _step_number(meta, path)
        if step is None or step < 0:
            continue
        seen_steps.add(step)
        if not meta.get("valid"):
            continue
        score = (_finite_float(meta.get("reward")) if metric_name == "reward"
                 else _raw_score(meta, problem))
        if score is None:
            continue
        valid_count += 1
        choose = max if maximize else min
        initial = -math.inf if maximize else math.inf
        best_by_step[step] = choose(best_by_step.get(step, initial), score)

    if not seen_steps:
        raise ValueError(f"no rollout metadata found under {run_dir}/step*/")
    if not best_by_step:
        raise ValueError("the run contains no valid rollout with a numeric score")

    steps = list(range(0, max(seen_steps) + 1))
    per_step = [best_by_step.get(step, math.nan) for step in steps]
    cumulative = []
    best = -math.inf if maximize else math.inf
    choose = max if maximize else min
    for value in per_step:
        if math.isfinite(value):
            best = choose(best, value)
        cumulative.append(best if math.isfinite(best) else math.nan)
    return steps, cumulative, per_step, valid_count


def improvement_indices(values, min_delta, maximize=False):
    """Indices where the running best improves by at least ``min_delta``."""
    out = []
    previous_labeled = None
    for i, value in enumerate(values):
        if not math.isfinite(value):
            continue
        delta = value - previous_labeled if maximize and previous_labeled is not None else (
            previous_labeled - value if previous_labeled is not None else math.inf
        )
        if previous_labeled is None or delta >= min_delta:
            out.append(i)
            previous_labeled = value
    return out


def plateau_index(values, tol=5e-10, maximize=False):
    """First step whose value equals the final best at displayed precision."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    final = finite[-1]
    for i, value in enumerate(values):
        reached = value >= final - tol if maximize else value <= final + tol
        if math.isfinite(value) and reached:
            return i
    return None


def halo(ax, x, y, color, base=9.5):
    """Soft glow around the point where the run reaches its final score."""
    for mult, alpha in ((3.6, 0.06), (2.8, 0.09), (2.1, 0.14), (1.5, 0.22)):
        ax.plot([x], [y], ls="none", marker="o", ms=base * mult,
                color=color, alpha=alpha, mec="none", zorder=3)
    ax.plot([x], [y], ls="none", marker="o", ms=base * 2.05,
            mfc="none", mec=color, mew=1.3, alpha=0.55, zorder=6)


def place_labels(fig, ax, annotations, direction="up", pad=3.0):
    """Lay labels out without overlaps while preserving vertical order."""
    if not annotations:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels_per_point = fig.dpi / 72.0
    boxes = [annotation.get_window_extent(renderer) for annotation in annotations]
    down = direction == "down"
    axes_box = ax.get_window_extent(renderer)
    order = sorted(range(len(annotations)), key=lambda i: boxes[i].y0,
                   reverse=down)
    placed = []

    def slide(box, go_down):
        x0, x1 = box.x0 - pad, box.x1 + pad
        y0, y1 = box.y0, box.y1
        height = y1 - y0
        for _ in range(len(annotations) + 1):
            collision = False
            for px0, px1, py0, py1 in placed:
                if x1 > px0 and x0 < px1 and y1 + pad > py0 and y0 - pad < py1:
                    if go_down:
                        y1, y0 = py0 - pad, py0 - pad - height
                    else:
                        y0, y1 = py1 + pad, py1 + pad + height
                    collision = True
            if not collision:
                break
        return x0, x1, y0, y1

    for i in order:
        box = boxes[i]
        x0, x1, y0, y1 = slide(box, down)
        if ((down and y0 < axes_box.y0 + 2)
                or (not down and y1 > axes_box.y1 - 2)):
            fx0, fx1, fy0, fy1 = slide(box, not down)
            if axes_box.y0 + 2 <= fy0 and fy1 <= axes_box.y1 - 2:
                x0, x1, y0, y1 = fx0, fx1, fy0, fy1
        dx, dy = annotations[i].xyann
        annotations[i].xyann = (dx, dy + (y0 - box.y0) / pixels_per_point)
        placed.append((x0, x1, y0, y1))

    fig.canvas.draw()
    for annotation in annotations:
        box = annotation.get_window_extent(renderer)
        shift = 0.0
        if box.y1 > axes_box.y1 - 2:
            shift = (axes_box.y1 - 2) - box.y1
        elif box.y0 < axes_box.y0 + 2:
            shift = (axes_box.y0 + 2) - box.y0
        if shift:
            dx, dy = annotation.xyann
            annotation.xyann = (dx, dy + shift / pixels_per_point)
    fig.canvas.draw()


def config_caption(run_dir, valid_count, metric_name, maximize):
    """Compact subtitle containing the settings that define this run."""
    config_path = Path(run_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        config = {}
    model = Path(str(config.get("model_name", "?"))).name
    groups = config.get("groups_per_step", "?")
    size = config.get("group_size", "?")
    memory = "memory on" if config.get("memory") else "memory off"
    problem = config.get("problem", "unknown")
    direction = "higher is better" if maximize else "lower is better"
    return (f"{problem} · {metric_name} · {model} · {groups} groups × {size} "
            f"rollouts · {memory} · {valid_count:,} valid rollouts · {direction}")


def plot_best_curve(run_dir, output=None, title=None, min_label_delta=1e-5,
                    max_step=None):
    run_dir = Path(run_dir)
    _, problem, metric_name, maximize = _run_info(run_dir)
    steps, cumulative, per_step, valid_count = load_best_curve(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    except ImportError as exc:
        raise ValueError(
            "matplotlib is required to render the plot; install it with "
            "`pip install matplotlib`"
        ) from exc

    if max_step is not None:
        keep = [i for i, step in enumerate(steps) if step <= max_step]
        steps = [steps[i] for i in keep]
        cumulative = [cumulative[i] for i in keep]
        per_step = [per_step[i] for i in keep]
        if not steps or not any(math.isfinite(v) for v in cumulative):
            raise ValueError(f"no valid score at or before step {max_step}")

    improvements = improvement_indices(cumulative, min_label_delta, maximize)
    final_index = plateau_index(cumulative, maximize=maximize)
    if final_index is not None and final_index not in improvements:
        improvements = sorted(improvements + [final_index])

    fig, ax = plt.subplots(figsize=(14.5, 6.4), dpi=150, facecolor="none")
    fig.subplots_adjust(left=0.075, right=0.982, top=0.84, bottom=0.13)
    ax.set_facecolor("none")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9)

    finite_points = [(step, value) for step, value in zip(steps, cumulative)
                     if math.isfinite(value)]
    xs = [point[0] for point in finite_points]
    ys = [point[1] for point in finite_points]
    ax.plot(xs, ys, color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=4)
    ax.plot(xs, ys, ls="none", marker="o", ms=4.8, color=BLUE,
            mec=SURFACE, mew=1.0, zorder=5)

    config_path = run_dir / "config.json"
    target = None
    if config_path.is_file():
        try:
            target = _finite_float(json.loads(config_path.read_text()).get("target"))
        except (OSError, json.JSONDecodeError):
            pass
    if target is not None:
        ax.axhline(target, color=INK_MUTED, linewidth=1.2,
                   linestyle=(0, (5, 4)), zorder=2)
        ax.text(steps[0] + 0.15, target, f"target  {target:.6f}",
                va="bottom", ha="left", fontsize=9, color=INK_SOFT, zorder=3)

    annotations = []
    for order, index in enumerate(improvements):
        x, y = steps[index], cumulative[index]
        if index == final_index:
            halo(ax, x, y, BLUE)
        ax.plot([x], [y], ls="none", marker="o", ms=9.5, color=BLUE,
                mec=SURFACE, mew=1.4, zorder=6)
        # Label the first, every third substantial improvement, and the final
        # plateau. Every improvement still receives the enlarged marker.
        if order != 0 and order % 3 != 0 and index != final_index:
            continue
        annotation = ax.annotate(
            f"{y:.9f}", xy=(x, y), xycoords="data",
            xytext=(7, 12), textcoords="offset points",
            fontsize=8.4, color=INK, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec=BLUE,
                      lw=0.7, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, alpha=0.75,
                            shrinkA=0, shrinkB=3),
            zorder=7,
        )
        annotations.append(annotation)

    values_for_limits = list(ys)
    if target is not None:
        values_for_limits.append(target)
    low, high = min(values_for_limits), max(values_for_limits)
    span = max(high - low, 1e-5)
    ax.set_ylim(low - 0.09 * span, high + 0.18 * span)
    ax.set_xlim(min(steps) - 0.6, max(steps) + 0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

    ax.set_xlabel("training step", fontsize=10.5, color=INK, labelpad=8)
    direction_word = "highest" if maximize else "lowest"
    ax.set_ylabel(f"{direction_word} {metric_name} seen so far",
                  fontsize=10.5, color=INK, labelpad=8)
    fig.suptitle(title or f"{problem}: best-so-far score vs. training step",
                 x=0.075, ha="left", y=0.95, fontsize=14,
                 color=INK, fontweight="bold")
    fig.text(0.075, 0.90, config_caption(run_dir, valid_count, metric_name, maximize),
             ha="left", fontsize=9.5, color=INK_SOFT)

    handles = [
        Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=6,
               mec=SURFACE, mew=1.0,
               label=f"{direction_word} valid {metric_name} so far"),
    ]
    if target is not None:
        handles.append(Line2D([], [], color=INK_MUTED, lw=1.2,
                              ls=(0, (5, 4)), label="target"))
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9.5,
              facecolor=SURFACE, edgecolor=GRID, borderpad=0.7)
    place_labels(fig, ax, annotations)

    png_path = Path(output) if output else run_dir / "best_so_far.png"
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    final_step = steps[final_index]
    final_best = cumulative[final_index]
    print(f"read {valid_count} valid rollouts across steps 0-{steps[-1]}")
    print(f"best {metric_name} = {final_best:.9f} by step {final_step}")
    print(f"wrote {png_path} and {pdf_path}")
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the cumulative best valid score for any problem."
    )
    parser.add_argument("run_dir", help="run directory containing stepXX folders")
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--title", default=None, help="custom plot title")
    parser.add_argument("--min-label-delta", type=float, default=1e-5,
                        help="minimum score change to mark as an improvement")
    parser.add_argument("--max-step", type=int, default=None,
                        help="plot only through this step")
    args = parser.parse_args()

    try:
        plot_best_curve(args.run_dir, output=args.out, title=args.title,
                        min_label_delta=args.min_label_delta,
                        max_step=args.max_step)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
