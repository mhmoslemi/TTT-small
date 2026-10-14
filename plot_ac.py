#!/usr/bin/env python3
"""Plot the authoritative best-ever AC1/AC2 construction in a run directory.

New runs use the sampler's completed-step best snapshot, which includes an
unbeaten seed and valid solutions later pruned from the search archive. Older
runs fall back to rollout metadata. Plotting never re-executes a stochastic or
wall-clock-bounded candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from experiment_io import load_best_state_snapshot


_STEP_RE = re.compile(r"step(\d+)")


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _problem_kind(run_dir: Path):
    config_path = run_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {config_path}: {error}") from error
    problem = str(config.get("problem", "")).strip().lower()
    kind = problem if problem in {"ac1", "ac2"} else str(
        config.get("problem_type", "")).strip().lower()
    if kind not in {"ac1", "ac2"}:
        raise SystemExit(
            f"{run_dir} is not an AC1/AC2 run (problem={problem!r}, "
            f"problem_type={kind!r})")
    return kind


def _step_number(meta, path: Path):
    try:
        return int(meta["step"])
    except (KeyError, TypeError, ValueError):
        match = _STEP_RE.search(path.parent.name)
        return int(match.group(1)) if match else None


def load_metas(run_dir: Path, max_step=None):
    """Load only rollout metadata, excluding strategy-attempt sidecars."""
    out = []
    pattern = "step*/step*_group*_rollout*.meta.json"
    for path in sorted(run_dir.glob(pattern)):
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        step = _step_number(meta, path)
        if step is None or (max_step is not None and step > max_step):
            continue
        meta["step"] = step
        meta["_meta_path"] = path
        out.append(meta)
    return out


def pick(metas, kind, step=None, group=None, rollout=None):
    """Select an explicit rollout or the problem-native best valid rollout."""
    if step is not None:
        selected = [
            meta for meta in metas
            if int(meta.get("step", -1)) == step
            and (group is None or int(meta.get("group", -1)) == group)
            and (rollout is None
                 or int(meta.get("rollout", -1)) == rollout)
        ]
        if not selected:
            raise SystemExit(
                f"no rollout matching step={step} group={group} "
                f"rollout={rollout}")
        return selected[0]

    valid = [
        meta for meta in metas
        if bool(meta.get("valid"))
        and _finite_float(meta.get("raw_score")) is not None
    ]
    if not valid:
        raise SystemExit("no valid AC rollout with a numeric raw score")
    choose = max if kind == "ac2" else min
    return choose(valid, key=lambda meta: float(meta["raw_score"]))


def _saved_sequence(meta):
    value = meta.get("construction")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list) or not value:
        return None
    sequence = []
    for item in value:
        number = _finite_float(item)
        if number is None:
            return None
        sequence.append(number)
    return sequence


def evaluated_sequence_and_score(sequence, kind):
    """Apply the same defensive transforms and formula as the AC evaluator."""
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("numpy is required to verify and plot AC solutions") from error

    values = np.asarray(sequence, dtype=np.float64)
    values = np.maximum(values, 0.0)
    if float(values.sum()) < 0.01:
        raise SystemExit("saved construction has a near-zero coefficient sum")
    values = np.minimum(values, 1000.0)
    convolution = np.convolve(values, values)

    if kind == "ac1":
        score = 2.0 * len(values) * float(convolution.max()) / float(values.sum()) ** 2
        return values, float(score)

    count = len(convolution)
    dx = 1.0 / (count + 1)
    padded = np.concatenate(([0.0], convolution, [0.0]))
    left = padded[:-1]
    right = padded[1:]
    l2_squared = float(
        np.sum((dx / 3.0) * (left * left + left * right + right * right)))
    l1 = float(np.sum(np.abs(convolution)) / (count + 1))
    linf = float(np.max(np.abs(convolution)))
    if l1 <= 0.0 or linf <= 0.0:
        raise SystemExit("saved AC2 construction has zero convolution norm")
    return values, l2_squared / (l1 * linf)


def plot_sequence(sequence, score, kind, label, save_path, title=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit("matplotlib and numpy are required to render the plot") from error

    metric = "upper bound" if kind == "ac1" else "lower bound"
    edges = np.linspace(0.0, 1.0, len(sequence) + 1)
    steps = np.append(sequence, sequence[-1])

    fig, axis = plt.subplots(figsize=(7.0, 3.0), dpi=220, facecolor="none")
    axis.set_facecolor("none")
    axis.step(edges, steps, where="post", color="#2a78d6", linewidth=1.15)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("normalized support", fontsize=9.5)
    axis.set_ylabel("coefficient", fontsize=9.5)
    axis.grid(axis="y", color="#e6e5e1", linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#8a8985")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors="#52514e", labelsize=8.5)
    axis.set_title(title or f"{kind.upper()} best saved construction",
                   loc="left", fontsize=11, pad=10)
    axis.text(
        0.985, 0.96,
        f"{len(sequence):,} coefficients\n{metric} = {score:.9f}\n{label}",
        transform=axis.transAxes, ha="right", va="top", fontsize=8.2,
        color="#0b0b0b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb",
                  edgecolor="#d8d7d3", linewidth=0.7, alpha=0.94),
    )
    fig.tight_layout()

    png_path = Path(save_path)
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {png_path} and {pdf_path}")
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the exact best saved AC1/AC2 construction.")
    parser.add_argument("run_dir")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None,
                        help="select only rollouts at or before this step")
    parser.add_argument("--group", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    kind = _problem_kind(run_dir)
    explicit_rollout = any(
        value is not None for value in (args.step, args.group, args.rollout))
    selected = None
    if not explicit_rollout:
        selected = load_best_state_snapshot(run_dir, max_step=args.max_step)

    # Backward compatibility for old runs that predate authoritative snapshots,
    # and explicit rollout inspection requested on the command line.
    if selected is None:
        metas = load_metas(run_dir, max_step=args.max_step)
        if not metas:
            raise SystemExit(f"no saved best state or rollout metadata under "
                             f"{run_dir}/step*/")
        selected = pick(
            metas, kind, step=args.step, group=args.group,
            rollout=args.rollout)
    sequence = _saved_sequence(selected)
    if sequence is None:
        raise SystemExit(
            "best-ever state has no complete saved construction; refusing to "
            "plot a different, worse answer")

    evaluated, computed = evaluated_sequence_and_score(sequence, kind)
    recorded = float(selected["raw_score"])
    difference = abs(computed - recorded)
    tolerance = 1e-10 * max(1.0, abs(recorded))
    if difference > tolerance:
        raise SystemExit(
            f"saved construction does not reproduce its recorded score: "
            f"recorded={recorded:.12g}, computed={computed:.12g}")

    if "completed_step" in selected:
        origin = "seed" if selected.get("is_seed") else (
            f"step {selected.get('found_step')}")
        label = (f"best through step {selected.get('completed_step')} · "
                 f"found in {origin}")
    else:
        label = (f"step {selected.get('step')} · "
                 f"group {selected.get('group')} · "
                 f"rollout {selected.get('rollout')}")
    print(f"selected {label}; verified {kind.upper()} score={computed:.12g}")
    output = args.out or str(run_dir / f"{kind}_solution.png")
    plot_sequence(evaluated, computed, kind, label, output, title=args.title)


if __name__ == "__main__":
    main()
