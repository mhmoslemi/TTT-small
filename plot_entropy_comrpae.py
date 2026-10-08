"""One-off comparison of saved, full-vocabulary token-mean entropies."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
RUNS = [
    ("erdos_Qwen3-8B_1007-2324", "With strategist", "#6b21a8", True),
    ("erdos_Qwen3-8B_1008-0537", "Without strategist", "#35b6c7", False),
]


def read_measurements(name, strategies_enabled):
    directory = ROOT / "runs" / name
    history = [json.loads(line) for line in
               (directory / "entropy.jsonl").read_text().splitlines()
               if line.strip()]
    points = []
    for entry in sorted(history, key=lambda row: row["step"]):
        assert entry["strategies_enabled"] == strategies_enabled
        samples_path = directory / f"step{entry['step']:02d}" / "entropy_samples.jsonl"
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()
                   if line.strip()]
        measured = [sample for sample in samples if sample.get("measured")]
        total_tokens = sum(sample["token_count"] for sample in measured)
        assert total_tokens > 0
        value = math.fsum(sample["entropy_sum_nats"] for sample in measured) / total_tokens
        assert math.isfinite(value) and value >= 0
        assert math.isclose(value, entry["all"]["entropy_nats"],
                            rel_tol=1e-12, abs_tol=1e-12)
        assert len(measured) == entry["all"]["measured_rollouts"]
        assert len(samples) == entry["all"]["rollouts"]
        complete = len(measured) == len(samples)
        points.append((entry["step"], value, complete, len(measured), len(samples)))
    assert len({point[0] for point in points}) == len(points)
    return points


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
    })
    series = [
        (name, label, color, read_measurements(name, strategies_enabled))
        for name, label, color, strategies_enabled in RUNS
    ]
    common_steps = set.intersection(
        *({point[0] for point in points} for _, _, _, points in series))
    if not common_steps:
        raise ValueError("The runs have no shared measured steps")
    first_step, last_step = min(common_steps), max(common_steps)

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    figure.subplots_adjust(left=0.115, right=0.98, top=0.97, bottom=0.15)
    for name, label, color, points in series:
        points = [point for point in points
                  if first_step <= point[0] <= last_step]
        axis.plot([point[0] for point in points],
                  [point[1] for point in points],
                  color=color, linewidth=2.2, label=label)
        print(f"{name}: verified {len(points)} points from per-rollout sums and token counts")
    axis.set_xlim(first_step - 0.5, last_step + 0.5)
    axis.set_ylim(0, 0.70)
    tick_interval = 5 if last_step - first_step <= 30 else 10
    axis.xaxis.set_major_locator(MultipleLocator(tick_interval))
    axis.xaxis.set_minor_locator(MultipleLocator(tick_interval / 5))
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_xlabel("Training step", fontsize=16, labelpad=9)
    axis.set_ylabel("Generation entropy (nats)", fontsize=16, labelpad=10)
    axis.tick_params(labelsize=12, direction="out", length=5)
    axis.legend(loc="upper right", fontsize=12, frameon=True,
                facecolor="white", edgecolor="#d1d5db", framealpha=1)
    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "coder_entropy_comparison"
    figure.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Coder generation entropy: with and without strategist",
        "Subject": "Unsmoothed, pre-update, full-vocabulary Shannon entropy; token-weighted across reasoning and answer tokens. Sources: "
                   + "; ".join(run[0] for run in RUNS),
    })
    figure.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(figure)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
