import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = [
    (
        ROOT / "runs/circle_packing_n26_Qwen3-8B_0910-0020",
        r"GRPO$_{\mathrm{rank}}$",
        "tab:blue",
    ),
    (
        ROOT / "runs/circle_packing_n26_Qwen3-8B_0910-1609",
        "TTT-noRL",
        "tab:orange",
    ),
]
TIMING_LOGS = [
    (ROOT / "res_ciclr_ours.txt", r"GRPO$_{\mathrm{rank}}$"),
    (ROOT / "res_ciclr_NoRL.txt", "TTT-noRL"),
]


def average_full_step(log_path):
    rollout_eval = {}
    training = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(
            r"\[step (\d+)\] rollout\+eval time: ([0-9.]+)s", line
        )
        if match:
            rollout_eval[int(match.group(1))] = float(match.group(2))
        match = re.search(
            r"\[step (\d+)\] TOTAL TRAINING TIME: ([0-9.]+)s", line
        )
        if match:
            training[int(match.group(1))] = float(match.group(2))

    completed_steps = sorted(set(rollout_eval) & set(training))
    average = sum(
        rollout_eval[step] + training[step] for step in completed_steps
    ) / len(completed_steps)
    return average, len(completed_steps)


def format_runtime(seconds):
    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{seconds:.1f}s ({minutes}m {remaining_seconds:02d}s)"


def best_so_far(run_dir):
    per_step = {}
    for path in run_dir.glob("step*/step*_group*_rollout*.meta.json"):
        try:
            record = json.loads(path.read_text())
            step = int(record["step"])
            raw_score = float(record["raw_score"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if math.isfinite(raw_score):
            per_step[step] = max(per_step.get(step, -math.inf), raw_score)

    steps = sorted(per_step)
    values = []
    running = -math.inf
    for step in steps:
        running = max(running, per_step[step])
        values.append(running)
    return steps, values


def meaningful_jumps(steps, values):
    return [
        (step, value)
        for index, (step, value) in enumerate(zip(steps, values))
        if index == 0 or value > values[index - 1] + 1e-9
    ]


annotation_offsets = {
    r"GRPO$_{\mathrm{rank}}$": {
        0: (7, 12), 2: (-8, 12), 3: (8, -25), 4: (-8, 12),
        5: (8, -34), 7: (-8, 18), 8: (-10, 42), 9: (18, -28),
        13: (8, 15),
    },
    "TTT-noRL": {
        0: (8, 12), 1: (8, 12), 2: (8, 14), 3: (-12, 30),
        4: (14, -34), 5: (12, 24), 19: (-12, -34), 20: (-8, 26),
        21: (6, -34), 22: (10, 30),
    },
}

fig, ax = plt.subplots(figsize=(14, 8))
all_steps = []
all_values = []

for run_dir, label, color in RUNS:
    steps, values = best_so_far(run_dir)
    jumps = meaningful_jumps(steps, values)
    all_steps.extend(steps)
    all_values.extend(values)

    ax.step(
        steps, values, where="post",
        linewidth=4.0 if label == r"GRPO$_{\mathrm{rank}}$" else 2.3,
        marker="o",
        markersize=4, color=color, label=label,
    )
    ax.scatter(
        [step for step, _ in jumps],
        [value for _, value in jumps],
        s=45, color=color, edgecolor="white", linewidth=0.8, zorder=3,
    )
    for step, value in jumps:
        dx, dy = annotation_offsets[label].get(step, (8, 12))
        ax.annotate(
            f"s{step}: {value:.6f}",
            xy=(step, value),
            xytext=(dx, dy),
            textcoords="offset points",
            color=color,
            fontsize=11.5,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
        )

ax.set_xlabel("Step")
ax.set_ylabel("Best raw score seen so far")
ax.set_title("Best Seen So Far by Step")
ax.set_xticks(range(min(all_steps), max(all_steps) + 1))
ax.set_ylim(min(all_values) - 0.08, max(all_values) + 0.25)
ax.grid(True, alpha=0.3)
ax.set_xlim([1.8,24])
ax.set_ylim([2.3,2.75])
ax.legend(loc="lower right", fontsize = 20)
timing_parts = []
for timing_log, label in TIMING_LOGS:
    average_seconds, completed_count = average_full_step(timing_log)
    timing_parts.append(
        f"{label}: {format_runtime(average_seconds)}"
    )
fig.text(
    0.5, 0.158,
    "Avg step time: " + "   |   ".join(timing_parts),
    ha="center", va="bottom", fontsize=14,
)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(ROOT / "best_seen_comparison.png", dpi=160)
