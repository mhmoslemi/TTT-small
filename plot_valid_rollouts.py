import matplotlib.pyplot as plt


runs = {
    "Entropic (0823-2201)": [
        9.2, 41.2, 68.5, 59.0, 59.2, 79.1, 81.1, 77.9, 76.8, 79.3,
        87.7, 95.9, 96.5, 96.7, 95.5, 95.3, 97.3, 97.9, 96.7, 96.9,
        96.9, 95.7, 93.8, 96.7, 95.3,
    ],
    "Entropic (0824-0500)": [
        3.9, 37.1, 54.9, 55.5, 41.4, 56.2, 64.6, 78.7, 86.3, 83.8,
        90.0, 91.4, 89.1, 92.4, 91.8, 90.4, 89.5, 82.2, 84.4, 75.4,
        72.1, 64.1, 65.2, 67.2,
    ],
    "SPO-RS (0916-1749)": [
        10.2, 51.6, 71.1, 66.4, 78.5, 64.8, 87.9, 92.6, 88.7, 90.2,
        91.4, 93.4, 90.6, 68.0, 93.0, 93.8, 88.7, 90.2,
    ],
    "SPO-RS (0916-2140)": [7.2, 54.5, 50.6, 77.5, 68.2, 88.9, 83.4, 82.8],
    "two-LLM-SPO": [30.3, 77.3, 63.8, 49.0, 58.5, 73.5, 65.0, 75.3, 80.8],
    "two-LLM-noRL": [40.2, 71.2, 77.5, 81.0, 71.2, 71.5, 71.8, 78.7],
}

colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
markers = ["o", "s", "^", "D", "P", "X"]
line_styles = ["-", "--", "-.", ":", "-", "--"]

for (label, valid_percentages), color, marker, line_style in zip(
        runs.items(), colors, markers, line_styles):
    shown = valid_percentages[:9]
    plt.plot(
        range(len(shown)), shown,
        color=color,
        linestyle=line_style,
        marker=marker,
        linewidth=2.4,
        markersize=6,
        label=label,
    )

plt.xlabel("Step")
plt.ylabel("Valid rollouts (%)")
plt.ylim(0, 100)
plt.xlim(-0.2, 8.2)
plt.xticks(range(9))
plt.grid(alpha=0.25)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("valid_rollouts_by_step.png", dpi=200)
