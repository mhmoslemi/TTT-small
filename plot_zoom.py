"""
Zoom plot for the final N steps of a TTT-Discover run.

The tail rewards agree to ~10 digits, so raw values overlap into one line.
Instead we plot the GAP from SOTA (SOTA - reward) on a LOG axis, per rank, which
spreads the tiny differences out. Points at/above SOTA are marked separately
(their gap is <= 0 and can't be drawn on a log axis).

Usage:
    python plot_zoom.py <run_dir> [last_n] [out.png]

    python plot_zoom.py runs/circle_packing_n26_Qwen3-8B_0608-2252 6
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOTA = 2.6359830832


def load_top_csv(run_dir: Path):
    csv_path = run_dir / "results" / "all_top3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no {csv_path}. Run find.sh first.")
    by_step = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                reward = float(row["reward"])
            except (ValueError, KeyError):
                continue
            if str(row.get("valid", "")).strip().lower() == "true":
                by_step[step].append(reward)
    for s in by_step:
        by_step[s].sort(reverse=True)
    return by_step


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_zoom.py <run_dir> [last_n] [out.png]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else run_dir / "progress_zoom.png"

    by_step = load_top_csv(run_dir)
    if not by_step:
        print("no valid rows found")
        sys.exit(1)

    steps = sorted(by_step.keys())[-last_n:]
    max_rank = max(len(by_step[s]) for s in steps)
    max_rank = 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    cmap = plt.get_cmap("viridis")

    # ---- Panel 1: raw rewards, full precision, per rank ----
    for k in range(max_rank):
        xs = [s for s in steps if len(by_step[s]) > k]
        ys = [by_step[s][k] for s in steps if len(by_step[s]) > k]
        ax1.plot(xs, ys, "o-", ms=4, lw=1.0, alpha=0.8,
                 color=cmap(k / max(1, max_rank - 1)), label=f"rank {k+1}")
    ax1.axhline(SOTA, color="red", ls="--", lw=0.1, label=f"SOTA = {SOTA:.6f}")
    ax1.set_ylabel("sum of radii")
    ax1.set_title(f"Last {last_n} steps (raw) — {run_dir.name}")
    # ax1.legend(loc="lower right", fontsize=8, ncol=max_rank + 1)
    # ax1.legend('off')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False, axis="y")  # show absolute values

    # ---- Panel 2: gap from SOTA on a log axis ----
    reached = []  # (step, rank) pairs that met/beat SOTA (gap <= 0)
    for k in range(max_rank):
        xs, gaps = [], []
        for s in steps:
            if len(by_step[s]) <= k:
                continue
            gap = SOTA - by_step[s][k]
            if gap > 0:
                xs.append(s)
                gaps.append(gap)
            else:
                reached.append((s, by_step[s][k]))
        if xs:
            ax2.plot(xs, gaps, "o-", ms=4, lw=1.0, alpha=0.8,
                     color=cmap(k / max(1, max_rank - 1)), label=f"rank {k+1}")
    if reached:
        rs = [s for s, _ in reached]
        # draw them at the bottom of the log axis as "met/beat SOTA"
        ymin = ax2.get_ylim()[0] if ax2.has_data() else 1e-7
        ax2.scatter(rs, [ymin] * len(rs), marker="*", s=90, color="red",
                    zorder=5, label="met/beat SOTA")

    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    ax2.set_ylabel("SOTA − reward  (log)")
    ax2.set_title("Gap from SOTA (lower is better; log scale spreads the tail)")
    ax2.legend(loc="upper right", fontsize=8, ncol=max_rank + 1)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xticks(steps)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # textual tail summary
    print(f"\nlast {last_n} steps — rank-1 reward and gap from SOTA:")
    for s in steps:
        r1 = by_step[s][0]
        print(f"  step {s:>3}: {r1:.12f}   gap={SOTA - r1:+.3e}")


if __name__ == "__main__":
    main()