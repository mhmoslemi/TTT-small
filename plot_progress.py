"""
Plot TTT-Discover training progress from a run's results/all_top3.csv.

Two panels:
  (top)    best reward per step + running best-so-far, vs SOTA.
  (bottom) per-step rank spread (rank-1..rank-N rewards), so you can see how
           tightly the top rollouts cluster as training converges.

Usage:
    python plot_progress.py <run_dir> [out.png]

    python plot_progress.py runs/circle_packing_n26_Qwen3-8B_0608-2252
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOTA = 2.635983


# Y-axis limits (applied to both panels) — set to None to use automatic scaling
Y_MIN = 2.522   # e.g. 2.60
Y_MAX = 2.64    # e.g. 2.64

# X-axis (step) limit — set to None to show all steps
X_MAX_STEP = 14   # e.g. 500


def load_top_csv(run_dir: Path):
    """Read results/all_top3.csv -> {step: [rewards sorted desc]}.
    Columns: step,rank,reward,group,rollout,valid,msg,file."""
    csv_path = run_dir / "results" / "all_top3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"no {csv_path}. Run find.sh first to generate it.")
    by_step = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["step"])
                reward = float(row["reward"])
            except (ValueError, KeyError):
                continue
            valid = str(row.get("valid", "")).strip().lower() == "true"
            if valid:
                by_step[step].append(reward)
    for s in by_step:
        by_step[s].sort(reverse=True)
    return by_step


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_progress.py <run_dir> [out.png]")
        sys.exit(1)

    run_dir = Path(sys.argv[1]) 
    # out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "progress-no-Eloa.png"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "progress-Eloa-bad.png"
    by_step = load_top_csv(run_dir)
    if not by_step:
        print("no valid rows found in all_top3.csv")
        sys.exit(1)

    steps = sorted(s for s in by_step if X_MAX_STEP is None or s <= X_MAX_STEP)
    best_per_step = np.array([by_step[s][0] for s in steps])

    # Running best-so-far (monotonic non-decreasing).
    best_so_far = np.maximum.accumulate(best_per_step)

    # Max rank depth available (find.sh keeps up to 6).
    max_rank = max(len(by_step[s]) for s in steps)
    max_rank = 3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # ---- Panel 1: progress ----
    ax1.axhline(SOTA, color="red", ls="--", lw=1.2,
                label=f"SOTA = {SOTA:.6f}")
    ax1.plot(steps, best_per_step, "o-", ms=3, lw=1.0, alpha=0.55,
             color="tab:blue", label="best in step")
    ax1.plot(steps, best_so_far, "-", lw=2.2, color="tab:green",
             label="best so far")
    ax1.set_ylabel("sum of radii")
    ax1.set_title(f"Training progress — {run_dir.name}")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Zoom y to where the action is (ignore the very low early steps for scale),
    # but keep SOTA visible.
    lo = float(np.percentile(best_per_step, 5))
    _y1_lo = Y_MIN if Y_MIN is not None else min(lo, best_per_step.min())
    _y1_hi = Y_MAX if Y_MAX is not None else max(SOTA, best_per_step.max()) + 1e-3
    ax1.set_ylim(_y1_lo, _y1_hi)

    # ---- Panel 2: per-step rank spread ----
    cmap = plt.get_cmap("viridis")
    for k in range(max_rank):
        xs, ys = [], []
        for s in steps:
            if len(by_step[s]) > k:
                xs.append(s)
                ys.append(by_step[s][k])
        ax2.plot(xs, ys, "-", lw=1.0, alpha=0.8,
                 color=cmap(k / max(1, max_rank - 1)),
                 label=f"rank {k+1}")
    ax2.axhline(SOTA, color="red", ls="--", lw=1.0)
    ax2.set_xlabel("step")
    ax2.set_ylabel("sum of radii")
    ax2.set_title("Per-step top-rank spread (rank 1 = best rollout that step)")
    ax2.legend(loc="lower right", fontsize=8, ncol=max_rank)
    ax2.grid(True, alpha=0.3)
    # The interesting band is near SOTA once it converges; clip the low tail.
    all_ranked = np.concatenate([by_step[s] for s in steps])
    _y2_lo = Y_MIN if Y_MIN is not None else float(np.percentile(all_ranked, 10))
    _y2_hi = Y_MAX if Y_MAX is not None else max(SOTA, all_ranked.max()) + 1e-3
    ax2.set_ylim(_y2_lo, _y2_hi)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # quick textual summary
    final_best = best_so_far[-1]
    first_step_at_best = steps[int(np.argmax(best_per_step >= final_best))]
    print(f"final best-so-far: {final_best:.10f}  (SOTA {SOTA - final_best:+.2e})")
    print(f"first reached at step: {first_step_at_best}")


if __name__ == "__main__":
    main()