import re
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
log_text = (root / "res_ciclr_ours.txt").read_text(errors="replace")
pattern = re.compile(
    r"\[step (\d+)\] best-ever raw [^:]+: "
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

by_step = {}
for step_text, value_text in pattern.findall(log_text):
    by_step[int(step_text)] = float(value_text)

steps = sorted(by_step)
best = []
running_best = float("-inf")
for step in steps:
    running_best = max(running_best, by_step[step])
    best.append(running_best)

jumps = [
    (step, value)
    for i, (step, value) in enumerate(zip(steps, best))
    if i == 0 or value > best[i - 1]
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.step(steps, best, where="post", linewidth=2)
ax.scatter(steps, best, s=18)
ax.scatter(*zip(*jumps), s=45, color="tab:red", zorder=3)

offsets = {
    0: (8, 12),
    2: (-8, 12),
    3: (8, 12),
    4: (8, 12),
    5: (8, -30),
    7: (-8, 18),
    8: (-10, 42),
    9: (18, -28),
    13: (8, 12),
}
for step, value in jumps:
    dx, dy = offsets.get(step, (8, 12))
    ax.annotate(
        f"s{step}: {value:.6f}",
        xy=(step, value),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "linewidth": 0.7},
    )

ax.set_xlabel("Step")
ax.set_ylabel("Best seen so far")
ax.set_title("Best Seen So Far by Training Step")
ax.set_xticks(steps)
ax.set_ylim(min(best) - 0.08, max(best) + 0.22)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(root / "best_seen_so_far.png", dpi=160)
