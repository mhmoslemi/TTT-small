"""
Re-score all rollouts of a given step (optionally one group) through the SAME
path training used: problem.compute_reward -> preprocess -> sandbox -> validate,
and save a circle-packing plot per rollout.

Usage:
    python score_step.py <run_dir> <step> [group]

    python score_step.py runs/circle_packing_n26_Qwen3-8B_0608-1530 36
    python score_step.py runs/circle_packing_n26_Qwen3-8B_0608-1530 36 7

Plots are written to:  <run_dir>/step<NN>/png/group<GG>_rollout<RRR>.png
"""

import sys
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from problems.registry import get_problem
from problems.base import ParentContext
from sandbox import run_code


# Known state-of-the-art for the current problem (circle packing n=26).
# SOTA-now = SOTA - rescored: positive => below SOTA, negative => beat SOTA.
SOTA = 2.635983


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    return json.loads(cfg_path.read_text())


def find_rollouts(run_dir: Path, step: int, group=None):
    step_dir = run_dir / f"step{step:02d}"
    if not step_dir.is_dir():
        raise FileNotFoundError(f"no {step_dir}")

    out = []
    pat = re.compile(r"step\d+_group(\d+)_rollout(\d+)\.txt$")
    for txt in sorted(step_dir.glob(f"step{step:02d}_group*_rollout*.txt")):
        m = pat.search(txt.name)
        if not m:
            continue
        g, r = int(m.group(1)), int(m.group(2))
        if group is not None and g != group:
            continue
        meta = Path(str(txt)[:-len(".txt")] + ".meta.json")
        out.append((g, r, txt, meta))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# ----------------------------------------------------------------------
# Plotting (mirrors plot_packing.py: color by radius, red edge on bad circles)
# ----------------------------------------------------------------------
def _collect_issues(centers, radii, tol=1e-9):
    issues = []
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        issues.append("NaN values present")
    for i in range(n):
        if radii[i] < 0:
            issues.append(f"Circle {i} negative radius")
        x, y = centers[i]
        r = radii[i]
        if (x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol):
            issues.append(f"Circle {i} outside unit square")
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            if dist - (radii[i] + radii[j]) < -tol:
                issues.append(f"Circles {i},{j} overlap")
    return issues


def plot_packing(centers, radii, sum_radii, save_to, title_prefix=""):
    n = len(radii)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False,
                                    linewidth=1.5, edgecolor="black"))

    cmap = plt.get_cmap("viridis")
    rmax = max(radii.max(), 1e-9) if n else 1e-9

    issues = _collect_issues(centers, radii)
    invalid_ids = set()
    for msg in issues:
        for tok in msg.replace(",", " ").split():
            if tok.isdigit():
                invalid_ids.add(int(tok))

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        edge = "red" if i in invalid_ids else "black"
        lw = 1.5 if i in invalid_ids else 0.5
        ax.add_patch(patches.Circle((x, y), r, facecolor=cmap(r / rmax),
                                    edgecolor=edge, linewidth=lw, alpha=0.65))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", weight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.3)

    title = f"{title_prefix}sum of radii = {sum_radii:.12f}"
    if issues:
        title += f"  [INVALID: {len(issues)} issue(s)]"
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close(fig)


def extract_centers_radii(code: str, prob, timeout_s):
    """Run the rollout's code through the sandbox (same preprocess as scoring)
    and pull (centers, radii) out of the return value, so the plot matches what
    the evaluator saw. Returns (centers, radii) or (None, None)."""
    from reward import extract_python_code
    src = extract_python_code(code)
    if src is None:
        return None, None
    full = prob.preprocess(src, ParentContext())
    out = run_code(full, entrypoint=prob.entrypoint, timeout_s=timeout_s)
    if not out.get("ok"):
        return None, None
    val = out.get("value")
    if not (isinstance(val, tuple) and len(val) == 3):
        return None, None
    try:
        centers = np.asarray(val[0], dtype=float)
        radii = np.asarray(val[1], dtype=float).ravel()
    except (ValueError, TypeError):
        return None, None
    return centers, radii


def main():
    if len(sys.argv) < 3:
        print("usage: python score_step.py <run_dir> <step> [group]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    step = int(sys.argv[2])
    group = int(sys.argv[3]) if len(sys.argv) > 3 else None

    cfg = load_run_config(run_dir)
    problem_name = cfg.get("problem", "circle_packing")
    timeout_s = float(cfg.get("sandbox_timeout_s", 30.0))
    prob = get_problem(problem_name, cfg)
    is_circle = problem_name in ("circle_packing", "circle", "circles")

    png_dir = Path('png_dir') / f"step{step:02d}" / "png"
    if is_circle:
        png_dir.mkdir(parents=True, exist_ok=True)

    rollouts = find_rollouts(run_dir, step, group)
    if not rollouts:
        print(f"no rollouts found for step {step}"
              + (f" group {group}" if group is not None else ""))
        sys.exit(1)

    print(f"run:       {run_dir.name}")
    print(f"problem:   {problem_name}   timeout: {timeout_s}s")
    print(f"SOTA:      {SOTA:.12f}   (SOTA-now = SOTA - rescored; negative => beat SOTA)")
    print(f"step:      {step}" + (f"   group: {group}" if group is not None else ""))
    print(f"rollouts:  {len(rollouts)}")
    if is_circle:
        print(f"plots ->   {png_dir}")
    print("-" * 84)
    print(f"{'grp':>3} {'roll':>4} | {'rescored':>16} {'SOTA-now':>16} | valid | msg")
    print("-" * 84)

    rows = []
    for g, r, txt, meta in rollouts:
        response = txt.read_text(errors="replace")

        # Same scoring path training used.
        res = prob.compute_reward(response, ParentContext(), timeout_s=timeout_s)
        sota_gap = SOTA - res.reward
        rows.append((g, r, res.reward, res.valid, res.msg, sota_gap))

        print(f"{g:>3} {r:>4} | {res.reward:16.12f} {sota_gap:+16.12f} "
              f"| {str(res.valid):>5} | {res.msg[:38]}")

        # Plot (circle packing only). Re-run to recover the actual geometry.
        if is_circle:
            centers, radii = extract_centers_radii(response, prob, timeout_s)
            if centers is not None and centers.ndim == 2 and centers.shape[1] == 2:
                save_to = png_dir / f"group{g:02d}_rollout{r:03d}.png"
                try:
                    plot_packing(centers, radii, float(np.sum(radii)),
                                 save_to,
                                 title_prefix=f"g{g} r{r}  ")
                except Exception as e:
                    print(f"        [plot failed for g{g} r{r}: {e!r}]")

    print("-" * 84)
    rescored = [x[2] for x in rows]
    valids = [x for x in rows if x[3]]
    best = max(rows, key=lambda x: x[2]) if rows else None
    print(f"rescored: min={min(rescored):.12f}  "
          f"mean={sum(rescored)/len(rescored):.12f}  "
          f"max={max(rescored):.12f}  valid={len(valids)}/{len(rows)}")
    if best is not None:
        print(f"best:     group {best[0]} rollout {best[1]}  "
              f"reward={best[2]:.12f}  SOTA-best={SOTA - best[2]:+.12f}")
        bg, br = best[0], best[1]
        print(f"          step{step:02d}/step{step:02d}_group{bg:02d}_rollout{br:03d}.txt")
        if is_circle:
            print(f"          png/group{bg:02d}_rollout{br:03d}.png")


if __name__ == "__main__":
    main()