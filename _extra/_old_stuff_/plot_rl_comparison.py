import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_0616 = '/work/mohammad/TTT-local/runs/circle_packing_n26_Qwen3-8B_0616-'
BASE_0617 = '/work/mohammad/TTT-local/runs/circle_packing_n26_Qwen3-8B_0617-'

RUNS = {
    'PPO':           BASE_0616 + '1808',
    'A2C':           BASE_0616 + '1833',
    'REINFORCE':     BASE_0616 + '2048',
    'TTT (lr=5e-4)': BASE_0616 + '2153',
    'TTT (lr=1)':    BASE_0617 + '0129',
    'TTT (lr=0)':    BASE_0617 + '0130',
}

COLORS = {
    'PPO':           '#e6194b',
    'A2C':           '#3cb44b',
    'REINFORCE':     '#4363d8',
    'TTT (lr=5e-4)': '#f58231',
    'TTT (lr=1)':    '#911eb4',
    'TTT (lr=0)':    '#42d4f4',
}

N_STEPS = 20


def load_best_per_step(run_dir: str) -> list[float | None]:
    """Return per-step best reward (max over all groups/rollouts); None if step missing."""
    results = []
    for step_idx in range(N_STEPS):
        step_name = f'step{step_idx:02d}'
        pattern = os.path.join(run_dir, step_name, '*.meta.json')
        files = glob.glob(pattern)
        if not files:
            results.append(None)
            continue
        best = -np.inf
        for fpath in files:
            with open(fpath) as f:
                meta = json.load(f)
            reward = meta.get('reward')
            if reward is not None and reward > best:
                best = reward
        results.append(best if best > -np.inf else None)
    return results


def cumulative_best(per_step: list[float | None]) -> list[float | None]:
    """Running maximum across steps."""
    running = []
    best_so_far = -np.inf
    for val in per_step:
        if val is not None and val > best_so_far:
            best_so_far = val
        running.append(best_so_far if best_so_far > -np.inf else None)
    return running


def main():
    all_per_step = {}
    all_cumulative = {}

    for label, run_dir in RUNS.items():
        if not os.path.isdir(run_dir):
            print(f'[WARN] Directory not found: {run_dir}')
            continue
        per_step = load_best_per_step(run_dir)
        all_per_step[label] = per_step
        all_cumulative[label] = cumulative_best(per_step)
        final_json = os.path.join(run_dir, 'final.summary.json')
        if os.path.exists(final_json):
            with open(final_json) as f:
                summary = json.load(f)
            print(f'{label:10s}  best_value={summary.get("best_value", "N/A"):.4f}  '
                  f'best_step={summary.get("best_step", "N/A")}')

    steps = list(range(N_STEPS))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle('Circle Packing n=26 — RL Algorithm Comparison (Qwen3-8B)',
                 fontsize=13, fontweight='bold')

    for label, values in all_cumulative.items():
        xs = [s for s, v in zip(steps, values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.plot(xs, ys, marker='o', markersize=5, linewidth=2,
                color=COLORS[label], label=label)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Best Sum of Radii (seen so far)', fontsize=11)
    ax.set_xticks(steps)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    out_path = '/work/mohammad/TTT-local/rl_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
