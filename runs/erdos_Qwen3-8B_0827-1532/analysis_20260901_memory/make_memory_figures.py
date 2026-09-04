#!/usr/bin/env python3
"""Create report-ready figures for the within-run memory experiment.

The script reads the immutable rollout metadata in the parent run directory.
All derived files are written beside this script.
"""

from __future__ import annotations

import glob
import html
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EPS = 1e-12

COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "grid": "#E5E7EB",
    "axis": "#94A3B8",
    "control": "#94A3B8",
    "selected": "#167D8D",
    "explore": "#8B5CF6",
    "valid": "#138A72",
    "improve": "#2563A6",
    "win": "#159A80",
    "tie": "#CBD5E1",
    "loss": "#D95D4F",
}


def arm_color(arm):
    return COLORS["control"] if arm == "no_memory" else COLORS[arm]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_mean_ci(values, seed, draws=20000):
    values = list(values)
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(draws):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def expected_subsample_max(values, n):
    values = sorted(float(value) for value in values)
    denominator = math.comb(len(values), n)
    return sum(
        value * math.comb(index, n - 1) / denominator
        for index, value in enumerate(values)
        if index >= n - 1
    )


class Svg:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            """<style>
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17202A}
.title{font-size:30px;font-weight:750}.subtitle{font-size:17px;fill:#5D6D7E}
.panel{font-size:18px;font-weight:700}.label{font-size:15px}.small{font-size:13px;fill:#5D6D7E}
.tick{font-size:13px;fill:#475569}.value{font-size:14px;font-weight:700}.legend{font-size:14px;font-weight:650}
</style>""",
        ]

    def text(self, x, y, value, cls="label", anchor="start", fill=None, rotate=None):
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'class="{cls}"', f'text-anchor="{anchor}"']
        if fill:
            attrs.append(f'fill="{fill}"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x:.2f} {y:.2f})"')
        self.parts.append(f"<text {' '.join(attrs)}>{html.escape(str(value))}</text>")

    def line(self, x1, y1, x2, y2, stroke, width=1, dash=None, opacity=1.0):
        attrs = [
            f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"',
            f'stroke="{stroke}"', f'stroke-width="{width}"', f'opacity="{opacity}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f"<line {' '.join(attrs)}/>")

    def rect(self, x, y, width, height, fill, opacity=1.0, rx=0, stroke="none", stroke_width=0):
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" opacity="{opacity}" rx="{rx}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def circle(self, x, y, radius, fill, stroke="#FFFFFF", stroke_width=2):
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polyline(self, points, stroke, width=3):
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{encoded}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def save(self, path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts))


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(RUN / "step[0-9][0-9]" / "*.meta.json"))):
        item = json.loads(Path(path).read_text(errors="replace"))
        step = int(item["step"])
        if step <= 33:
            rows.append(item)
    return rows


def build_contrasts(rows):
    groups = defaultdict(list)
    for item in rows:
        if int(item["step"]) >= 1:
            groups[(int(item["step"]), int(item["group"]))].append(item)

    contrasts = []
    for (step, group), items in sorted(groups.items()):
        arms = defaultdict(list)
        for item in items:
            arms[item.get("memory_arm", "no_memory")].append(item)
        control = arms["no_memory"]
        for arm in ("selected", "explore"):
            treatment = arms[arm]
            n = min(len(treatment), len(control))
            treatment_valid = mean(bool(item["valid"]) for item in treatment)
            control_valid = mean(bool(item["valid"]) for item in control)
            treatment_code = mean(item.get("failure_kind") != "code" for item in treatment)
            control_code = mean(item.get("failure_kind") != "code" for item in control)
            treatment_improve = mean(
                bool(item["valid"]) and float(item["reward"]) > float(item["parent_value"]) + EPS
                for item in treatment
            )
            control_improve = mean(
                bool(item["valid"]) and float(item["reward"]) > float(item["parent_value"]) + EPS
                for item in control
            )
            tail = expected_subsample_max([item["reward"] for item in treatment], n) - expected_subsample_max(
                [item["reward"] for item in control], n
            )
            contrasts.append(
                {
                    "step": step,
                    "group": group,
                    "arm": arm,
                    "valid_treatment": treatment_valid,
                    "valid_control": control_valid,
                    "valid_delta": treatment_valid - control_valid,
                    "code_treatment": treatment_code,
                    "code_control": control_code,
                    "code_delta": treatment_code - control_code,
                    "improve_treatment": treatment_improve,
                    "improve_control": control_improve,
                    "improve_delta": treatment_improve - control_improve,
                    "tail": tail,
                    "treatment_has_valid": any(item["valid"] for item in treatment),
                    "control_has_valid": any(item["valid"] for item in control),
                }
            )
    return contrasts


def arm_rates(rows, arm):
    items = [
        item for item in rows
        if 1 <= int(item["step"]) <= 33 and item.get("memory_arm", "no_memory") == arm
    ]
    return {
        "n": len(items),
        "valid": mean(bool(item["valid"]) for item in items),
        "code": mean(item.get("failure_kind") != "code" for item in items),
        "improve": mean(
            bool(item["valid"]) and float(item["reward"]) > float(item["parent_value"]) + EPS
            for item in items
        ),
    }


def summarize(rows, contrasts):
    selected = [item for item in contrasts if item["arm"] == "selected"]
    explore = [item for item in contrasts if item["arm"] == "explore"]
    metrics = {}
    for name, key, seed in (
        ("scientific_validity", "valid_delta", 101),
        ("code_validity", "code_delta", 102),
        ("beats_parent", "improve_delta", 103),
    ):
        values = [item[key] for item in selected]
        metrics[name] = {"effect": mean(values), "ci95": bootstrap_mean_ci(values, seed)}

    tail = {}
    for arm, arm_rows, seed in (("selected", selected, 201), ("explore", explore, 202)):
        values = [item["tail"] for item in arm_rows]
        signs = Counter("win" if value > EPS else "loss" if value < -EPS else "tie" for value in values)
        tail[arm] = {
            "effect": mean(values),
            "median": statistics.median(values),
            "ci95": bootstrap_mean_ci(values, seed),
            "signs": dict(signs),
        }

    phase_defs = {
        "early": (1, 7),
        "middle": (8, 25),
        "late": (26, 33),
    }
    phases = {}
    for phase, (lo, hi) in phase_defs.items():
        items = [item for item in selected if lo <= item["step"] <= hi]
        signs = Counter("win" if item["tail"] > EPS else "loss" if item["tail"] < -EPS else "tie" for item in items)
        phases[phase] = {
            "n": len(items),
            "valid_delta": mean(item["valid_delta"] for item in items),
            "code_delta": mean(item["code_delta"] for item in items),
            "improve_delta": mean(item["improve_delta"] for item in items),
            "tail_mean": mean(item["tail"] for item in items),
            "tail_signs": {name: int(signs.get(name, 0)) for name in ("win", "tie", "loss")},
        }

    per_step = {}
    for step in range(1, 34):
        items = [item for item in selected if item["step"] == step]
        per_step[str(step)] = {
            "valid_delta": mean(item["valid_delta"] for item in items),
            "code_delta": mean(item["code_delta"] for item in items),
            "improve_delta": mean(item["improve_delta"] for item in items),
            "tail_mean": mean(item["tail"] for item in items),
        }

    valid_positive = sum(item["valid_delta"] > EPS for item in per_step.values())
    code_positive = sum(item["code_delta"] > EPS for item in per_step.values())
    late_tail_negative = sum(per_step[str(step)]["tail_mean"] < -EPS for step in range(26, 34))

    records = []
    incumbent = math.inf
    by_step = defaultdict(list)
    for item in rows:
        by_step[int(item["step"])].append(item)
    for step in range(34):
        valid = [item for item in by_step[step] if item.get("valid") and item.get("raw_score") is not None]
        if not valid:
            continue
        winner = min(valid, key=lambda item: float(item["raw_score"]))
        step_best = float(winner["raw_score"])
        improved = step_best < incumbent - 1e-15
        improvement = 0.0 if math.isinf(incumbent) or not improved else incumbent - step_best
        if improved:
            incumbent = step_best
        records.append(
            {
                "step": step,
                "cumulative_best": incumbent,
                "improved": improved,
                "improvement": improvement,
                "winner_arm": winner.get("memory_arm", "no_memory") if improved else None,
            }
        )

    attribution = defaultdict(float)
    for item in records:
        if item["step"] > 0 and item["improved"] and item["improvement"] > EPS:
            attribution[item["winner_arm"]] += item["improvement"]
    attribution_total = sum(attribution.values())
    attribution_share = {
        arm: attribution.get(arm, 0.0) / attribution_total
        for arm in ("selected", "explore", "no_memory")
    }

    return {
        "completed_steps": 34,
        "matched_parent_groups": len(selected),
        "arm_rates": {
            "no_memory": arm_rates(rows, "no_memory"),
            "selected": arm_rates(rows, "selected"),
            "explore": arm_rates(rows, "explore"),
        },
        "matched_effects": metrics,
        "tail": tail,
        "phases": phases,
        "per_step": per_step,
        "consistency": {
            "selected_validity_positive_steps": valid_positive,
            "selected_code_validity_positive_steps": code_positive,
            "late_tail_negative_steps": late_tail_negative,
        },
        "records": records,
        "record_improvement_share": attribution_share,
    }


def title(svg, heading, subtitle):
    svg.text(65, 48, heading, cls="title")
    svg.text(65, 76, subtitle, cls="subtitle")


def figure_overall(summary):
    svg = Svg(1500, 760)
    title(
        svg,
        "Selected memory improves feasibility, but not the maximum reliably",
        "Within-run comparison: 264 parents, with a no-memory control generated from the same parent.",
    )

    svg.text(80, 128, "A. Outcome rates", cls="panel")
    x0, y0, width, height = 95, 170, 760, 430
    ymax = 0.90
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        yy = y0 + height - tick / ymax * height
        svg.line(x0, yy, x0 + width, yy, COLORS["grid"])
        svg.text(x0 - 12, yy + 5, f"{tick:.0%}", cls="tick", anchor="end")
    svg.line(x0, y0, x0, y0 + height, COLORS["axis"])
    svg.line(x0, y0 + height, x0 + width, y0 + height, COLORS["axis"])

    categories = [
        ("Scientifically valid", "valid", "scientific_validity"),
        ("Code-valid", "code", "code_validity"),
        ("Beats its parent", "improve", "beats_parent"),
    ]
    control = summary["arm_rates"]["no_memory"]
    selected = summary["arm_rates"]["selected"]
    group_width = width / len(categories)
    bar_width = 68
    for index, (label, key, effect_key) in enumerate(categories):
        center = x0 + group_width * (index + 0.5)
        values = [(control[key], COLORS["control"], -43), (selected[key], COLORS["selected"], 43)]
        for value, color, offset in values:
            xx = center + offset - bar_width / 2
            yy = y0 + height - value / ymax * height
            svg.rect(xx, yy, bar_width, y0 + height - yy, color, rx=4)
            svg.text(xx + bar_width / 2, yy - 9, f"{value:.1%}", cls="value", anchor="middle")
        effect = summary["matched_effects"][effect_key]
        svg.text(center, y0 + height + 34, label, cls="legend", anchor="middle")
        svg.text(
            center,
            y0 + height + 57,
            f"memory effect {100 * effect['effect']:+.1f} pp",
            cls="small",
            anchor="middle",
            fill=COLORS["selected"],
        )

    svg.rect(520, 112, 18, 13, COLORS["control"], rx=2)
    svg.text(547, 124, "no memory", cls="legend")
    svg.rect(650, 112, 18, 13, COLORS["selected"], rx=2)
    svg.text(677, 124, "selected lesson", cls="legend")

    svg.text(930, 128, "B. Equal-budget scientific tail", cls="panel")
    svg.text(930, 151, "Mean best-of-13 reward difference versus no memory", cls="small")
    px0, py0, pwidth, pheight = 930, 205, 490, 315
    xmin, xmax = -0.12, 0.08
    xx = lambda value: px0 + (value - xmin) / (xmax - xmin) * pwidth
    for tick in (-0.10, -0.05, 0.0, 0.05):
        x = xx(tick)
        svg.line(x, py0, x, py0 + pheight, COLORS["grid"], width=1)
        svg.text(x, py0 + pheight + 23, f"{tick:+.2f}", cls="tick", anchor="middle")
    svg.line(xx(0), py0 - 8, xx(0), py0 + pheight, COLORS["ink"], width=2)

    tail_rows = [("selected lesson", "selected", 295), ("UCB explore lesson", "explore", 420)]
    for label, key, yy in tail_rows:
        effect = summary["tail"][key]["effect"]
        lo, hi = summary["tail"][key]["ci95"]
        color = COLORS[key]
        svg.text(px0, yy - 31, label, cls="legend")
        svg.line(xx(lo), yy, xx(hi), yy, color, width=5)
        svg.line(xx(lo), yy - 10, xx(lo), yy + 10, color, width=3)
        svg.line(xx(hi), yy - 10, xx(hi), yy + 10, color, width=3)
        svg.circle(xx(effect), yy, 9, color)
        svg.text(px0, yy + 36, f"mean {effect:+.3f}; 95% CI [{lo:+.3f}, {hi:+.3f}]", cls="small")

    svg.rect(925, 585, 500, 82, "#F8FAFC", rx=8, stroke="#E2E8F0", stroke_width=1)
    svg.text(945, 615, "Interpretation", cls="legend")
    svg.text(945, 640, "Memory clearly raises feasibility; both tail intervals cross zero.", cls="label")
    svg.text(945, 662, "So this run does not show a reliable average gain in the maximum.", cls="label")
    svg.text(80, 718, "Bootstrap intervals resample matched parent groups within this single run; they are not across-run confidence intervals.", cls="small")
    svg.save(HERE / "memory_overall_effect.svg")


def figure_phases(summary):
    svg = Svg(1500, 820)
    title(
        svg,
        "Memory stays useful for validity, but its max-seeking effect collapses late",
        "Selected-lesson rollouts compared with no-memory rollouts from the same parent.",
    )

    svg.text(80, 128, "A. Average matched effect by phase", cls="panel")
    x0, y0, width, height = 110, 165, 1280, 300
    ymin, ymax = -0.05, 0.35
    yy = lambda value: y0 + height - (value - ymin) / (ymax - ymin) * height
    for tick in (-0.05, 0.0, 0.1, 0.2, 0.3):
        y = yy(tick)
        svg.line(x0, y, x0 + width, y, COLORS["grid"])
        svg.text(x0 - 12, y + 5, f"{tick:+.0%}", cls="tick", anchor="end")
    svg.line(x0, yy(0), x0 + width, yy(0), COLORS["axis"], width=1.5)

    phase_defs = [("early", "steps 1–7"), ("middle", "steps 8–25"), ("late", "steps 26–33")]
    group_width = width / 3
    bar_width = 78
    for index, (key, label) in enumerate(phase_defs):
        center = x0 + group_width * (index + 0.5)
        phase = summary["phases"][key]
        for value, color, offset in (
            (phase["valid_delta"], COLORS["valid"], -50),
            (phase["improve_delta"], COLORS["improve"], 50),
        ):
            top = min(yy(value), yy(0))
            svg.rect(center + offset - bar_width / 2, top, bar_width, abs(yy(value) - yy(0)), color, rx=4)
            text_y = yy(value) - 9 if value >= 0 else yy(value) + 19
            svg.text(center + offset, text_y, f"{100 * value:+.1f} pp", cls="value", anchor="middle")
        svg.text(center, y0 + height + 32, label, cls="legend", anchor="middle")

    svg.rect(950, 112, 18, 13, COLORS["valid"], rx=2)
    svg.text(978, 124, "scientific-validity gain", cls="legend")
    svg.rect(1180, 112, 18, 13, COLORS["improve"], rx=2)
    svg.text(1208, 124, "beats-parent gain", cls="legend")

    svg.text(80, 535, "B. Which arm wins the matched best-of-13 comparison?", cls="panel")
    svg.text(80, 558, "Higher reward wins; ties mean identical best reward.", cls="small")
    bx0, bwidth = 310, 1050
    row_y = {"early": 600, "middle": 660, "late": 720}
    labels = {"early": "steps 1–7", "middle": "steps 8–25", "late": "steps 26–33"}
    for key in ("early", "middle", "late"):
        counts = summary["phases"][key]["tail_signs"]
        n = summary["phases"][key]["n"]
        svg.text(bx0 - 20, row_y[key] + 24, labels[key], cls="legend", anchor="end")
        cursor = bx0
        for outcome in ("win", "tie", "loss"):
            fraction = counts[outcome] / n
            segment = bwidth * fraction
            svg.rect(cursor, row_y[key], segment, 34, COLORS[outcome], rx=2 if outcome in ("win", "loss") else 0)
            if fraction >= 0.08:
                svg.text(cursor + segment / 2, row_y[key] + 23, f"{counts[outcome]} ({fraction:.0%})", cls="value", anchor="middle", fill="#FFFFFF" if outcome != "tie" else COLORS["ink"])
            cursor += segment

    legend_x = 810
    for index, outcome in enumerate(("win", "tie", "loss")):
        svg.rect(legend_x + index * 135, 535, 18, 13, COLORS[outcome], rx=2)
        svg.text(legend_x + index * 135 + 27, 547, outcome, cls="legend")

    svg.text(80, 792, "Validity gain was positive in 32/33 steps. In the eight late steps, mean tail uplift was negative in all eight.", cls="small")
    svg.save(HERE / "memory_effect_over_time.svg")


def figure_records(summary):
    svg = Svg(1500, 800)
    title(
        svg,
        "Exploratory memory produced rare, large early discoveries",
        "Cumulative best C5 bound; lower is better. Marker color shows the arm that set each new incumbent.",
    )

    x0, y0, width, height = 110, 150, 1280, 430
    records = summary["records"]
    xmin, xmax = 0, 33
    ymin = min(item["cumulative_best"] for item in records) - 0.000025
    ymax = max(item["cumulative_best"] for item in records) + 0.000030
    xx = lambda step: x0 + (step - xmin) / (xmax - xmin) * width
    yy = lambda value: y0 + (ymax - value) / (ymax - ymin) * height

    for tick in (0.3825, 0.3820, 0.3815, 0.3810):
        if ymin <= tick <= ymax:
            y = yy(tick)
            svg.line(x0, y, x0 + width, y, COLORS["grid"])
            svg.text(x0 - 14, y + 5, f"{tick:.5f}", cls="tick", anchor="end")
    for step in range(0, 34, 5):
        x = xx(step)
        svg.line(x, y0 + height, x, y0 + height + 7, COLORS["axis"])
        svg.text(x, y0 + height + 25, step, cls="tick", anchor="middle")
    svg.line(x0, y0, x0, y0 + height, COLORS["axis"])
    svg.line(x0, y0 + height, x0 + width, y0 + height, COLORS["axis"])
    svg.text(x0 + width / 2, y0 + height + 53, "training step", cls="legend", anchor="middle")

    points = [(xx(item["step"]), yy(item["cumulative_best"])) for item in records]
    svg.polyline(points, COLORS["ink"], width=3)
    for item in records:
        if item["improved"]:
            arm = item["winner_arm"]
            svg.circle(xx(item["step"]), yy(item["cumulative_best"]), 7, arm_color(arm))

    for step, label, dx, dy in (
        (1, "selected", -5, -28),
        (2, "explore", 22, -30),
        (3, "explore", 28, 26),
    ):
        item = next(record for record in records if record["step"] == step)
        svg.text(xx(step) + dx, yy(item["cumulative_best"]) + dy, f"step {step}: {label}", cls="small", anchor="middle")

    legend_x = 860
    for index, (arm, label) in enumerate((("selected", "selected lesson"), ("explore", "explore lesson"), ("no_memory", "no memory"))):
        svg.circle(legend_x + index * 175, 112, 7, arm_color(arm))
        svg.text(legend_x + index * 175 + 15, 117, label, cls="legend")

    final = records[-1]["cumulative_best"]
    svg.text(x0 + width - 8, yy(final) - 12, f"final {final:.9f}", cls="value", anchor="end")

    svg.text(80, 670, "Share of post-step-0 raw improvement by winning arm", cls="panel")
    svg.text(80, 694, "Descriptive attribution only; selected memory received 38 attempts per parent, explore and control received 13 each.", cls="small")
    bar_x, bar_y, bar_width, bar_height = 420, 720, 940, 42
    cursor = bar_x
    shares = summary["record_improvement_share"]
    for arm in ("selected", "explore", "no_memory"):
        segment = bar_width * shares[arm]
        svg.rect(cursor, bar_y, segment, bar_height, arm_color(arm), rx=3)
        if shares[arm] >= 0.08:
            svg.text(cursor + segment / 2, bar_y + 27, f"{shares[arm]:.1%}", cls="value", anchor="middle", fill="#FFFFFF")
        cursor += segment
    svg.text(bar_x + bar_width, bar_y - 9, f"no memory {shares['no_memory']:.1%}", cls="small", anchor="end")
    svg.save(HERE / "memory_record_discoveries.svg")


def convert(svg_path):
    converter = Path("/opt/homebrew/bin/rsvg-convert")
    if not converter.exists():
        return
    subprocess.run([str(converter), "-w", "2250", "-o", str(svg_path.with_suffix(".png")), str(svg_path)], check=True)
    subprocess.run([str(converter), "-f", "pdf", "-o", str(svg_path.with_suffix(".pdf")), str(svg_path)], check=True)


def main():
    rows = load_rows()
    contrasts = build_contrasts(rows)
    summary = summarize(rows, contrasts)
    (HERE / "memory_metrics.json").write_text(json.dumps(summary, indent=2))
    figure_overall(summary)
    figure_phases(summary)
    figure_records(summary)
    for name in ("memory_overall_effect", "memory_effect_over_time", "memory_record_discoveries"):
        convert(HERE / f"{name}.svg")
    print(json.dumps({
        "matched_groups": summary["matched_parent_groups"],
        "rates": summary["arm_rates"],
        "effects": summary["matched_effects"],
        "tail": summary["tail"],
        "phases": summary["phases"],
        "consistency": summary["consistency"],
        "record_improvement_share": summary["record_improvement_share"],
    }, indent=2))


if __name__ == "__main__":
    main()
