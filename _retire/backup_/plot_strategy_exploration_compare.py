"""Compare saved Erdos search outputs without rerunning generation/evaluation.

The two standalone figures measure distinct valid non-parent constructions and valid
parent-copy outputs per 100 attempted rollouts. Failures and retries remain
in the denominator. Novelty is relative to all selected parents in that step,
not the complete history of the run. Neither metric uses objective scores.

Construction matching uses the normalized piecewise-constant h function on
the exact union of both grids, with an absolute L-infinity tolerance. Reflection
and complementation are treated as equivalent (both preserve the C5 objective).
Distinctness means connected components of this tolerance-equivalence graph,
so it is conservative and independent of file order.
"""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = (
    ROOT / "runs/erdos_Qwen3-8B_1007-2324",
    ROOT / "runs/erdos_Qwen3-8B_1008-0537",
)
LABELS = ("With strategist", "Without strategist")
# Huawei-inspired red with a restrained charcoal comparison color.
COLORS = ("#C7000B", "#5A6470")


def construction(value, source):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError(f"Missing or invalid saved construction: {source}")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"Nonpositive construction mass: {source}")
    # Match the evaluator's normalization before comparing returned solutions.
    if total != vector.size / 2.0:
        vector = vector * ((vector.size / 2.0) / total)
    if not np.isfinite(vector).all():
        raise ValueError(f"Nonfinite normalized construction: {source}")
    return vector


@lru_cache(maxsize=256)
def aligned_grid_indices(n, m):
    """Compare every interval, without resampling or allocating the LCM grid."""
    grid = math.lcm(n, m)
    edges = np.union1d(np.arange(n + 1, dtype=np.int64) * (grid // n),
                       np.arange(m + 1, dtype=np.int64) * (grid // m))
    midpoint_twice = edges[:-1] + edges[1:]
    return midpoint_twice * n // (2 * grid), midpoint_twice * m // (2 * grid)


def equivalent(left, right, atol):
    if left.size == right.size:
        aligned_left, aligned_right = left, right
    else:
        i, j = aligned_grid_indices(left.size, right.size)
        aligned_left, aligned_right = left[i], right[j]
    for candidate in (aligned_right, aligned_right[::-1]):
        if np.max(np.abs(aligned_left - candidate)) <= atol:
            return True
        if np.max(np.abs(aligned_left - (1.0 - candidate))) <= atol:
            return True
    return False


def cluster_indices(vectors, atol):
    leaders = list(range(len(vectors)))

    def find(index):
        while leaders[index] != index:
            leaders[index] = leaders[leaders[index]]
            index = leaders[index]
        return index

    for i, left in enumerate(vectors):
        for j in range(i):
            a, b = find(i), find(j)
            if a != b and equivalent(left, vectors[j], atol):
                leaders[a] = b
    return [find(index) for index in range(len(vectors))]


def read_history(directory):
    rows = [json.loads(line) for line in
            (directory / "entropy.jsonl").read_text().splitlines() if line.strip()]
    return {int(row["step"]): row for row in rows}


def measure_step(directory, step, history, atol):
    files = sorted((directory / f"step{step:02d}").glob("*rollout*.meta.json"))
    records = [(path, json.loads(path.read_text())) for path in files]
    attempts = len(records)
    if attempts != int(history["all"]["rollouts"]):
        raise ValueError(f"Rollout count mismatch in {directory.name}, step {step}")
    if not attempts:
        raise ValueError(f"No attempts in {directory.name}, step {step}")

    # Include every selected parent, even if all its children were invalid.
    parents = {}
    for path, meta in records:
        key = meta.get("parent_id", meta.get("parent_group", meta["group"]))
        value = construction(meta.get("parent_construction"), path)
        if key in parents and not equivalent(value, parents[key], atol):
            raise ValueError(f"Inconsistent saved parent {key}: {path}")
        parents[key] = value

    novel_vectors, novel_paths, parent_copy_paths = [], [], []
    valid = parent_copies = own_parent_copies = 0
    for path, meta in records:
        if not meta["valid"]:
            continue
        valid += 1
        value = construction(meta.get("construction"), path)
        parent = construction(meta.get("parent_construction"), path)
        own_parent_copies += equivalent(value, parent, atol)
        if any(equivalent(value, candidate, atol) for candidate in parents.values()):
            parent_copies += 1
            parent_copy_paths.append(path.name)
            continue
        novel_vectors.append(value)
        novel_paths.append(path.name)

    clusters = cluster_indices(novel_vectors, atol)
    novel_count = len(set(clusters))
    assert 0 <= novel_count <= len(novel_vectors) <= valid <= attempts
    assert parent_copies + len(novel_vectors) == valid
    return {
        "step": step,
        "attempts": attempts,
        "retry_attempts": sum(bool(meta.get("retry_attempt")) for _, meta in records),
        "valid": valid,
        "valid_per_100_attempts": 100.0 * valid / attempts,
        "copies_of_any_selected_parent": parent_copies,
        "copies_of_own_parent": own_parent_copies,
        "parent_copies_per_100_attempts": 100.0 * parent_copies / attempts,
        "non_parent_valid_rollouts": len(novel_vectors),
        "distinct_non_parent_valid_constructions": novel_count,
        "distinct_non_parent_per_100_attempts": 100.0 * novel_count / attempts,
        "parent_copy_audit": parent_copy_paths,
        "cluster_audit": [
            {"file": path, "cluster": int(cluster)}
            for path, cluster in zip(novel_paths, clusters)
        ],
    }


def plot(series, steps, output, atol):
    figures = (
        ("diversity", "distinct_non_parent_per_100_attempts", "Exploration diversity",
         "Distinct non-parent outputs (%)"),
        ("parent_reproduction", "parent_copies_per_100_attempts", "Parent reproduction",
         "Parent-copy rate (%)"),
    )
    paths = []
    # Fixed single-column dimensions and embedded TrueType fonts. Do not smooth,
    # resample, or add uncertainty bands to these single-run observations.
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8.5, "axes.linewidth": 0.65,
        "axes.edgecolor": "#333333", "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#333333",
        "ytick.color": "#333333", "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }):
        for suffix, key, title, ylabel in figures:
            fig, axis = plt.subplots(figsize=(3.45, 2.6))
            fig.subplots_adjust(left=0.165, right=0.975, bottom=0.18, top=0.965)
            for rows, label, color in zip(series, LABELS, COLORS):
                axis.plot(
                    [row["step"] for row in rows], [row[key] for row in rows],
                    color=color, linewidth=1.6, linestyle="-",
                    solid_capstyle="round", solid_joinstyle="round",
                    label=label, clip_on=False,
                )
            axis.set_xlabel("Training step", labelpad=4)
            axis.set_ylabel(ylabel, labelpad=5)
            axis.set_xlim(min(steps) - 0.3, max(steps) + 0.3)
            maximum = max(row[key] for rows in series for row in rows)
            upper = 100 if key == "parent_copies_per_100_attempts" else max(
                5, math.ceil(maximum / 5) * 5
            )
            axis.set_ylim(0, upper)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            axis.tick_params(axis="both", direction="out", length=3, width=0.65, pad=3)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#DDE1E5", linewidth=0.45)
            axis.spines[["top", "right"]].set_visible(False)
            legend = axis.legend(
                loc="upper right" if suffix == "diversity" else "upper left",
                ncol=1, frameon=True, fancybox=False, framealpha=1,
                facecolor="white", edgecolor="#C7CBCF", fontsize=7.2,
                handlelength=1.6, handletextpad=0.5, borderpad=0.4,
                labelspacing=0.35, borderaxespad=0.65,
            )
            legend.get_frame().set_linewidth(0.55)
            figure_output = output.with_name(f"{output.stem}_{suffix}")
            pdf = figure_output.with_suffix(".pdf")
            fig.savefig(pdf, metadata={
                "Title": title,
                "Subject": (
                    "Saved observations over shared steps, per 100 attempted rollouts; "
                    f"construction tolerance={atol:g}. No objective-score comparison. "
                    "Not a fixed-parent causal evaluation."
                ),
            })
            fig.savefig(figure_output.with_suffix(".png"), dpi=400)
            plt.close(fig)
            paths.append(pdf)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-strategy", type=Path, default=DEFAULT_RUNS[0])
    parser.add_argument("--without-strategy", type=Path, default=DEFAULT_RUNS[1])
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=ROOT / "output/pdf/strategy_exploration_comparison")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol <= 0:
        parser.error("Construction tolerance must be positive and finite")
    runs = (args.with_strategy.resolve(), args.without_strategy.resolve())
    histories = [read_history(run) for run in runs]
    steps = sorted(set(histories[0]) & set(histories[1]))
    if not steps:
        parser.error("No shared completed steps")
    series = []
    for run, history, expected in zip(runs, histories, (True, False)):
        rows = []
        for step in steps:
            if bool(history[step]["strategies_enabled"]) != expected:
                raise ValueError(f"Unexpected strategy mode: {run}, step {step}")
            row = measure_step(run, step, history[step], args.atol)
            rows.append(row)
            print(f"{run.name} step {step:02d}: valid={row['valid']}/{row['attempts']}; "
                  f"distinct non-parent={row['distinct_non_parent_valid_constructions']}; "
                  f"parent copies={row['copies_of_any_selected_parent']}", flush=True)
        series.append(rows)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "shared_steps": steps,
        "construction_atol": args.atol,
        "construction_comparison": "normalized piecewise-constant functions on exact union grid, modulo reflection/complement",
        "deduplication": "within-step tolerance-connected components; excludes copies of every selected parent",
        "parent_reproduction": "valid outputs equivalent to any selected parent, without deduplicating repeated copies",
        "objective_scores_used": False,
        "denominator": "all attempted rollouts, including invalid outputs and retries",
        "compute_matched": False,
        "same_parent_contexts": False,
        "runs": {label: {"directory": str(run), "steps": rows}
                 for run, label, rows in zip(runs, LABELS, series)},
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    figures = plot(series, steps, output, args.atol)
    for label, rows in zip(LABELS, series):
        attempts = sum(row["attempts"] for row in rows)
        novel = sum(row["distinct_non_parent_valid_constructions"] for row in rows)
        copies = sum(row["copies_of_any_selected_parent"] for row in rows)
        print(f"{label}: {novel / attempts * 100:.3f} distinct non-parent and "
              f"{copies / attempts * 100:.3f} parent copies per 100 attempts "
              f"across shared steps ({attempts} attempts)")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
