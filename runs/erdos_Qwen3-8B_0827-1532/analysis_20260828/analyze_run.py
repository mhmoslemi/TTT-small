#!/usr/bin/env python3
"""Read-only forensic analysis for erdos_Qwen3-8B_0827-1532.

The script uses only the Python standard library.  It reads the immutable run
artifacts from the parent directory and writes every derived artifact beside
this file.  It never rewrites a run artifact.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
TARGET_RECORD = 0.38092
EPS = 1e-12


def read_json(path: Path):
    return json.loads(path.read_text(errors="replace"))


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def safe_div(a, b):
    return a / b if b else 0.0


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def extract_python_code(response: str):
    """Mirror reward.extract_python_code without importing NumPy-dependent code."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in response and "</think>" not in response:
        return None
    matches = re.findall(r"```python\s*\n?(.*?)```", response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code:
            return code
    match = re.search(r"```python\s*\n?(.*)$", response, re.DOTALL)
    if match:
        code = re.sub(r"\n?```\s*$", "", match.group(1).strip()).strip()
        if code:
            return code
    matches = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code:
            return code
    stripped = response.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped
    return None


def structural_code_hash(code):
    if not code:
        return ""
    try:
        canonical = ast.dump(ast.parse(code), include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        return ""
    return sha(canonical)


def parse_memory_ids(value):
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = []
        if isinstance(parsed, (list, tuple)):
            return tuple(str(x) for x in parsed)
    return ()


def parse_vector(value):
    """Decode construction fields, which experiment_io serialized as strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return None


def expected_subsample_max(values, n):
    xs = sorted(float(x) for x in values)
    n = int(n)
    denom = math.comb(len(xs), n)
    return sum(
        x * math.comb(j, n - 1) / denom
        for j, x in enumerate(xs)
        if j >= n - 1
    )


def bootstrap_mean_ci(values, seed=42, draws=12000):
    """Percentile bootstrap over matched parent/group contrasts."""
    vals = list(values)
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(draws):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return [quantile(boots, 0.025), quantile(boots, 0.975)]


def failure_signature(msg):
    return re.sub(r"\d+", "#", (msg or "unknown").strip().lower())[:120]


def select_capped(indices, cap):
    idx = list(indices)
    if cap <= 0 or len(idx) <= cap:
        return idx
    stride = len(idx) / float(cap)
    return [idx[int(i * stride)] for i in range(cap)]


def select_balanced(indices, signatures, total_cap=0, per_signature_cap=0):
    buckets = {}
    for idx in indices:
        signature = signatures[idx] if idx < len(signatures) else "unknown"
        buckets.setdefault(signature or "unknown", []).append(idx)
    bucket_cap = (
        per_signature_cap
        if per_signature_cap > 0
        else total_cap
        if total_cap > 0
        else 0
    )
    if bucket_cap > 0:
        buckets = {
            signature: select_capped(items, bucket_cap)
            for signature, items in buckets.items()
        }
    out = []
    used = {signature: 0 for signature in buckets}
    while buckets and (total_cap <= 0 or len(out) < total_cap):
        for signature in list(buckets):
            if not buckets[signature] or (
                per_signature_cap > 0
                and used[signature] >= per_signature_cap
            ):
                del buckets[signature]
                continue
            out.append(buckets[signature].pop(0))
            used[signature] += 1
            if total_cap > 0 and len(out) >= total_cap:
                break
    return out


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_artifacts():
    config = read_json(RUN / "config.json")
    summaries = {}
    for path in sorted(RUN.glob("step*/step*.summary.json")):
        item = read_json(path)
        summaries[int(item["step"])] = item

    selections = {}
    for path in sorted(RUN.glob("step*/step*.parents.json")):
        item = read_json(path)
        selections[int(item["step"])] = item

    metas = []
    for path in sorted(RUN.glob("step*/step*_group*_rollout*.meta.json")):
        item = read_json(path)
        item["_meta_path"] = str(path)
        item["_txt_path"] = str(
            path.with_name(path.name.removesuffix(".meta.json") + ".txt")
        )
        item["memory_ids_parsed"] = parse_memory_ids(item.get("memory_ids"))
        item["construction_parsed"] = parse_vector(item.get("construction"))
        item["parent_construction_parsed"] = parse_vector(
            item.get("parent_construction")
        )
        metas.append(item)
    return config, summaries, selections, metas


def add_code_hashes(metas):
    for index, item in enumerate(metas, 1):
        text = Path(item["_txt_path"]).read_text(errors="replace")
        code = extract_python_code(text)
        item["response_hash"] = sha(text.strip())
        item["code_hash"] = sha(code.strip()) if code else ""
        item["structural_code_hash"] = structural_code_hash(code)
        if index % 2048 == 0:
            print(f"parsed {index}/{len(metas)} responses", flush=True)


def construction_key(values, digits=12):
    if not isinstance(values, list):
        return ""
    return sha(json.dumps([round(float(x), digits) for x in values], separators=(",", ":")))


def recompute_erdos_c5(values):
    """Pure-Python equivalent of max(correlate(h, 1-h, full) * 2/n)."""
    h_values = [float(x) for x in values]
    n = len(h_values)
    complement = [1.0 - x for x in h_values]
    candidates = []
    for lag in range(-(n - 1), n):
        total = 0.0
        for index, h_value in enumerate(h_values):
            shifted = index + lag
            if 0 <= shifted < n:
                total += h_value * complement[shifted]
        candidates.append(total * 2.0 / n)
    return max(candidates)


def step_rows(config, summaries, metas):
    grouped = defaultdict(list)
    for item in metas:
        grouped[int(item["step"])].append(item)

    cumulative_best = math.inf
    rows = []
    for step in sorted(grouped):
        items = grouped[step]
        valid = [x for x in items if x.get("valid")]
        raw = [float(x["raw_score"]) for x in valid if x.get("raw_score") is not None]
        step_best = min(raw) if raw else None
        if step_best is not None:
            cumulative_best = min(cumulative_best, step_best)

        improve = [
            x
            for x in valid
            if x.get("parent_value") is not None
            and float(x["reward"]) > float(x["parent_value"]) + EPS
        ]
        same = [
            x
            for x in valid
            if x.get("parent_value") is not None
            and abs(float(x["reward"]) - float(x["parent_value"])) <= EPS
        ]
        worse = [
            x
            for x in valid
            if x.get("parent_value") is not None
            and float(x["reward"]) < float(x["parent_value"]) - EPS
        ]

        group_map = defaultdict(list)
        for item in items:
            group_map[int(item["group"])].append(item)
        constant_groups = 0
        max_tie_counts = []
        unique_codes_per_group = []
        for group_items in group_map.values():
            rewards = [float(x["reward"]) for x in group_items]
            high = max(rewards)
            if high - min(rewards) < EPS:
                constant_groups += 1
            max_tie_counts.append(sum(abs(x - high) <= EPS for x in rewards))
            unique_codes_per_group.append(len({x["code_hash"] for x in group_items if x["code_hash"]}))

        summary = summaries.get(step, {})
        row = {
            "step": step,
            "rollouts": len(items),
            "valid": len(valid),
            "valid_fraction": safe_div(len(valid), len(items)),
            "code_valid_fraction": safe_div(
                sum(x.get("failure_kind") != "code" for x in items), len(items)
            ),
            "code_failures": sum(x.get("failure_kind") == "code" for x in items),
            "constraint_failures": sum(
                x.get("failure_kind") == "constraint" for x in items
            ),
            "timeouts": sum(x.get("failure_kind") == "timeout" for x in items),
            "step_best_raw": step_best,
            "cumulative_best_raw": cumulative_best if cumulative_best < math.inf else None,
            "valid_raw_median": median(raw),
            "valid_raw_mean": mean(raw),
            "parent_best_raw": min(
                float(x["parent_raw_score"])
                for x in items
                if x.get("parent_raw_score") is not None
            ),
            "improved_over_parent": len(improve),
            "same_as_parent": len(same),
            "worse_than_parent": len(worse),
            "improved_fraction_all": safe_div(len(improve), len(items)),
            "same_fraction_valid": safe_div(len(same), len(valid)),
            "unique_exact_codes": len({x["code_hash"] for x in items if x["code_hash"]}),
            "unique_structural_codes": len(
                {x["structural_code_hash"] for x in items if x["structural_code_hash"]}
            ),
            "unique_valid_exact_codes": len(
                {x["code_hash"] for x in valid if x["code_hash"]}
            ),
            "unique_valid_constructions_12dp": len(
                {
                    construction_key(x.get("construction_parsed"))
                    for x in valid
                    if x.get("construction_parsed")
                }
            ),
            "same_construction_as_parent": sum(
                x.get("construction_parsed") is not None
                and x.get("construction_parsed")
                == x.get("parent_construction_parsed")
                for x in valid
            ),
            "mean_unique_codes_per_group": mean(unique_codes_per_group),
            "constant_reward_groups": constant_groups,
            "mean_group_max_ties": mean(max_tie_counts),
            "response_tokens_total": sum(int(x.get("n_response_tokens", 0)) for x in items),
            "response_tokens_mean": mean(int(x.get("n_response_tokens", 0)) for x in items),
            "response_tokens_p95": quantile(
                [int(x.get("n_response_tokens", 0)) for x in items], 0.95
            ),
            "beta_mean": mean(float(x.get("beta", 0.0)) for x in items),
            "beta_median": median(float(x.get("beta", 0.0)) for x in items),
            "saturated_beta_groups": sum(
                max(float(x.get("beta", 0.0)) for x in group_items) >= 1_000_000
                for group_items in group_map.values()
            ),
            "positive_advantage_fraction": safe_div(
                sum(float(x.get("advantage", 0.0)) > 0 for x in items), len(items)
            ),
            "archive_size": summary.get("archive_size"),
            "distinct_good": summary.get("distinct_good"),
            "feedback_lambda_effective": summary.get("feedback_lambda_effective", 0.0),
            "feedback_teacher_rollouts": summary.get("feedback_teacher_rollouts", 0),
        }
        rows.append(row)
    return rows


def memory_contrasts(metas):
    grouped = defaultdict(list)
    for item in metas:
        grouped[(int(item["step"]), int(item["group"]))].append(item)

    rows = []
    for (step, group), items in sorted(grouped.items()):
        arms = defaultdict(list)
        for item in items:
            arms[item.get("memory_arm", "no_memory")].append(item)
        control = arms.get("no_memory")
        if not control:
            continue
        for arm in ("selected", "explore"):
            treatment = arms.get(arm)
            if not treatment:
                continue
            n = min(len(treatment), len(control))
            tail = expected_subsample_max([x["reward"] for x in treatment], n) - expected_subsample_max(
                [x["reward"] for x in control], n
            )
            t_valid = sum(bool(x["valid"]) for x in treatment)
            c_valid = sum(bool(x["valid"]) for x in control)
            t_code_valid = sum(x.get("failure_kind") != "code" for x in treatment)
            c_code_valid = sum(x.get("failure_kind") != "code" for x in control)
            t_improve = sum(
                bool(x["valid"]) and x["reward"] > x["parent_value"] + EPS
                for x in treatment
            )
            c_improve = sum(
                bool(x["valid"]) and x["reward"] > x["parent_value"] + EPS
                for x in control
            )
            rows.append(
                {
                    "step": step,
                    "group": group,
                    "arm": arm,
                    "memory_ids": ";".join(treatment[0]["memory_ids_parsed"]),
                    "treatment_rollouts": len(treatment),
                    "control_rollouts": len(control),
                    "matched_n": n,
                    "tail_uplift_reward": tail,
                    "valid_rate_treatment": safe_div(t_valid, len(treatment)),
                    "valid_rate_control": safe_div(c_valid, len(control)),
                    "valid_rate_delta": safe_div(t_valid, len(treatment))
                    - safe_div(c_valid, len(control)),
                    "code_valid_rate_delta": safe_div(t_code_valid, len(treatment))
                    - safe_div(c_code_valid, len(control)),
                    "parent_improvement_rate_delta": safe_div(t_improve, len(treatment))
                    - safe_div(c_improve, len(control)),
                    "treatment_has_valid": bool(t_valid),
                    "control_has_valid": bool(c_valid),
                }
            )
    return rows


def memory_summary(config, metas, contrasts, memory):
    arms = sorted({x.get("memory_arm", "no_memory") for x in metas})
    arm_stats = {}
    for arm in arms:
        items = [x for x in metas if x.get("memory_arm", "no_memory") == arm]
        valid = [x for x in items if x.get("valid")]
        arm_stats[arm] = {
            "rollouts": len(items),
            "rollout_fraction": safe_div(len(items), len(metas)),
            "valid_fraction": safe_div(len(valid), len(items)),
            "code_valid_fraction": safe_div(
                sum(x.get("failure_kind") != "code" for x in items), len(items)
            ),
            "parent_improvement_fraction": safe_div(
                sum(
                    x.get("valid")
                    and x.get("parent_value") is not None
                    and x["reward"] > x["parent_value"] + EPS
                    for x in items
                ),
                len(items),
            ),
        }

    contrast_stats = {}
    for arm in ("selected", "explore"):
        rows = [x for x in contrasts if x["arm"] == arm]
        tail_both = [
            x["tail_uplift_reward"]
            for x in rows
            if x["treatment_has_valid"] and x["control_has_valid"]
        ]
        contrast_stats[arm] = {
            "matched_groups": len(rows),
            "mean_tail_uplift": mean(x["tail_uplift_reward"] for x in rows),
            "median_tail_uplift": median(x["tail_uplift_reward"] for x in rows),
            "tail_uplift_mean_bootstrap_95": bootstrap_mean_ci(
                [x["tail_uplift_reward"] for x in rows], seed=101
            ),
            "tail_signs": dict(
                Counter(
                    "positive"
                    if x["tail_uplift_reward"] > EPS
                    else "negative"
                    if x["tail_uplift_reward"] < -EPS
                    else "zero"
                    for x in rows
                )
            ),
            "mean_valid_rate_delta": mean(x["valid_rate_delta"] for x in rows),
            "valid_delta_bootstrap_95": bootstrap_mean_ci(
                [x["valid_rate_delta"] for x in rows], seed=102
            ),
            "mean_code_valid_rate_delta": mean(
                x["code_valid_rate_delta"] for x in rows
            ),
            "code_valid_delta_bootstrap_95": bootstrap_mean_ci(
                [x["code_valid_rate_delta"] for x in rows], seed=103
            ),
            "mean_parent_improvement_rate_delta": mean(
                x["parent_improvement_rate_delta"] for x in rows
            ),
            "improvement_delta_bootstrap_95": bootstrap_mean_ci(
                [x["parent_improvement_rate_delta"] for x in rows], seed=104
            ),
            "both_valid_comparisons": len(tail_both),
            "both_valid_mean_tail_uplift": mean(tail_both),
            "both_valid_median_tail_uplift": median(tail_both),
            "both_valid_tail_signs": dict(
                Counter(
                    "positive" if x > EPS else "negative" if x < -EPS else "zero"
                    for x in tail_both
                )
            ),
            "valid_presence_patterns": {
                f"treatment_{t}_control_{c}": n
                for (t, c), n in Counter(
                    (x["treatment_has_valid"], x["control_has_valid"])
                    for x in rows
                ).items()
            },
        }

    by_lesson = {}
    for lesson in memory.get("lessons", []):
        lesson_id = lesson["id"]
        selected_rows = [
            x
            for x in contrasts
            if x["arm"] == "selected" and x["memory_ids"] == lesson_id
        ]
        by_lesson[lesson_id] = {
            "title": lesson.get("title"),
            "selector_uses": lesson.get("uses", 0),
            "selected_matched_groups": len(selected_rows),
            "selected_mean_tail_uplift": mean(
                x["tail_uplift_reward"] for x in selected_rows
            ),
            "selected_mean_valid_delta": mean(
                x["valid_rate_delta"] for x in selected_rows
            ),
            "selected_mean_code_valid_delta": mean(
                x["code_valid_rate_delta"] for x in selected_rows
            ),
        }

    phase_ranges = {
        "early_1_7": set(range(1, 8)),
        "middle_8_25": set(range(8, 26)),
        "late_26_33": set(range(26, 34)),
    }
    phases = {}
    for phase, steps in phase_ranges.items():
        phases[phase] = {}
        for arm in ("selected", "explore"):
            rows = [x for x in contrasts if x["arm"] == arm and x["step"] in steps]
            phases[phase][arm] = {
                "n": len(rows),
                "tail_uplift": mean(x["tail_uplift_reward"] for x in rows),
                "valid_delta": mean(x["valid_rate_delta"] for x in rows),
                "code_valid_delta": mean(x["code_valid_rate_delta"] for x in rows),
                "parent_improvement_delta": mean(
                    x["parent_improvement_rate_delta"] for x in rows
                ),
            }

    return {
        "arm_stats": arm_stats,
        "matched_contrasts": contrast_stats,
        "lessons": by_lesson,
        "phases": phases,
        "bank_counts": memory.get("counts", {}),
        "bank_stats": memory.get("stats", {}),
        "bank_usage": memory.get("usage", ""),
    }


def feedback_analysis(config, summaries, metas):
    by_step = defaultdict(list)
    for item in metas:
        by_step[int(item["step"])].append(item)
    for items in by_step.values():
        items.sort(key=lambda x: (int(x["group"]), int(x["rollout"])))

    picked = []
    active_code_failures = 0
    active_steps = []
    for step, summary in sorted(summaries.items()):
        expected = int(summary.get("feedback_teacher_rollouts", 0))
        if expected <= 0:
            continue
        active_steps.append(step)
        items = [x for x in by_step[step] if int(x.get("n_response_tokens", 0)) > 0]
        indices = [i for i, x in enumerate(items) if x.get("failure_kind") == "code"]
        active_code_failures += len(indices)
        signatures = [failure_signature(x.get("msg", "")) for x in items]
        keep = select_balanced(
            indices,
            signatures,
            int(summary.get("feedback_step_cap", 0)),
            int(summary.get("feedback_signature_cap", 0)),
        )
        if len(keep) != expected:
            raise RuntimeError(
                f"feedback reconstruction mismatch at step {step}: "
                f"{len(keep)} != {expected}"
            )
        for i in keep:
            picked.append({"step": step, "signature": signatures[i]})

    all_failures = Counter(x.get("failure_kind", "") for x in metas)
    code_signatures = Counter(
        failure_signature(x.get("msg", ""))
        for x in metas
        if x.get("failure_kind") == "code"
    )
    picked_signatures = Counter(x["signature"] for x in picked)
    rows = []
    for signature, occurrences in code_signatures.most_common():
        selected = picked_signatures[signature]
        rows.append(
            {
                "signature": signature,
                "occurrences": occurrences,
                "teacher_selected": selected,
                "selected_fraction": safe_div(selected, occurrences),
            }
        )

    code_valid = {
        step: 1.0
        - safe_div(
            sum(x.get("failure_kind") == "code" for x in items), len(items)
        )
        for step, items in by_step.items()
    }
    post_delta = []
    inactive_delta = []
    active_set = set(active_steps)
    for step in sorted(code_valid):
        if step + 1 not in code_valid:
            continue
        delta = code_valid[step + 1] - code_valid[step]
        (post_delta if step in active_set else inactive_delta).append(delta)

    return {
        "active_steps": active_steps,
        "teacher_forwards": len(picked),
        "teacher_fraction_of_all_rollouts": safe_div(len(picked), len(metas)),
        "active_step_code_failures": active_code_failures,
        "teacher_fraction_of_active_code_failures": safe_div(
            len(picked), active_code_failures
        ),
        "unique_selected_signatures": len(picked_signatures),
        "all_failure_counts": dict(all_failures),
        "top_code_signatures": rows[:20],
        "mean_next_step_code_valid_change_after_feedback": mean(post_delta),
        "mean_next_step_code_valid_change_without_feedback": mean(inactive_delta),
        "code_valid_by_step": code_valid,
        "warning": (
            "The run does not save feedback token advantages or repaired-program "
            "evaluations; these are exposure and behavioral diagnostics, not a "
            "causal estimate of feedback efficacy."
        ),
    }, rows


def tree_analysis(config, summaries, selections, metas):
    node_map = {x["node_id"]: x for x in metas}
    selection_events = []
    for selection_step, payload in sorted(selections.items()):
        for parent in payload.get("parents", []):
            item = dict(parent)
            item["selection_step"] = selection_step
            source = node_map.get(item["parent_id"])
            item["source_arm"] = source.get("memory_arm") if source else "seed"
            item["source_step"] = int(source["step"]) if source else None
            item["generated_parent"] = source is not None
            item["completed_expansion"] = selection_step in summaries
            selection_events.append(item)

    completed_generated = [
        x for x in selection_events if x["generated_parent"] and x["completed_expansion"]
    ]
    scheduled_generated = [x for x in selection_events if x["generated_parent"]]
    completed_unique_ids = {x["parent_id"] for x in completed_generated}
    scheduled_unique_ids = {x["parent_id"] for x in scheduled_generated}

    by_arm = {}
    for arm in sorted({x.get("memory_arm", "no_memory") for x in metas}):
        generated = [x for x in metas if x.get("memory_arm", "no_memory") == arm]
        expanded = [x for x in generated if x["node_id"] in completed_unique_ids]
        scheduled = [x for x in generated if x["node_id"] in scheduled_unique_ids]
        by_arm[arm] = {
            "generated": len(generated),
            "expanded_in_completed_step": len(expanded),
            "scheduled_including_incomplete_step34": len(scheduled),
            "completed_conversion_rate": safe_div(len(expanded), len(generated)),
        }

    seed_ids = {
        x["parent_id"]
        for x in selection_events
        if x.get("parent_is_seed")
    }
    lineage_rows = []
    for step, payload in sorted(selections.items()):
        parents = payload.get("parents", [])
        ancestor_sets = [set(x.get("ancestor_ids", [])) for x in parents]
        roots = [
            x.get("ancestor_ids", [])[-1]
            if x.get("ancestor_ids")
            else x["parent_id"]
            for x in parents
        ]
        first_children = [
            x.get("ancestor_ids", [])[-2]
            if len(x.get("ancestor_ids", [])) >= 2
            else x["parent_id"]
            for x in parents
        ]
        pairwise_jaccard = []
        for i in range(len(ancestor_sets)):
            for j in range(i + 1, len(ancestor_sets)):
                union = ancestor_sets[i] | ancestor_sets[j]
                pairwise_jaccard.append(
                    safe_div(len(ancestor_sets[i] & ancestor_sets[j]), len(union))
                )
        common = set.intersection(*ancestor_sets) if ancestor_sets and all(ancestor_sets) else set()
        lengths = [len(x) for x in ancestor_sets]
        root_counts = Counter(roots)
        child_counts = Counter(first_children)
        lineage_rows.append(
            {
                "step": step,
                "parents": len(parents),
                "unique_parent_ids": len({x["parent_id"] for x in parents}),
                "unique_parent_raw_12dp": len(
                    {round(float(x["parent_raw_score"]), 12) for x in parents}
                ),
                "unique_root_seeds": len(root_counts),
                "dominant_root_fraction": safe_div(max(root_counts.values()), len(parents)),
                "unique_step0_branches": len(child_counts),
                "dominant_step0_branch_fraction": safe_div(
                    max(child_counts.values()), len(parents)
                ),
                "common_ancestor_count": len(common),
                "common_ancestor_fraction_of_median_depth": safe_div(
                    len(common), median(lengths) or 1
                ),
                "mean_pairwise_ancestor_jaccard": mean(pairwise_jaccard),
                "mean_parent_age": mean(
                    step - int(x.get("parent_timestep", step)) for x in parents
                ),
                "max_parent_age": max(
                    [step - int(x.get("parent_timestep", step)) for x in parents],
                    default=0,
                ),
                "zero_visit_parents": sum(int(x.get("visit_count", 0)) == 0 for x in parents),
            }
        )

    final_archive = summaries[max(summaries)]["archive_size"]
    nonseed_archive = final_archive - int(config.get("num_seed_states", len(seed_ids)))
    valid_rollouts = sum(bool(x.get("valid")) for x in metas)
    return {
        "selection_events": len(selection_events),
        "generated_parent_events_completed": len(completed_generated),
        "generated_parent_unique_completed": len(completed_unique_ids),
        "generated_parent_unique_scheduled_including_step34": len(scheduled_unique_ids),
        "rollout_to_completed_parent_rate": safe_div(len(completed_unique_ids), len(metas)),
        "valid_rollout_to_completed_parent_rate": safe_div(
            len(completed_unique_ids), valid_rollouts
        ),
        "all_selected_parents_zero_visit_fraction": safe_div(
            sum(int(x.get("visit_count", 0)) == 0 for x in selection_events),
            len(selection_events),
        ),
        "parent_age_mean_completed": mean(
            x["selection_step"] - int(x["parent_timestep"])
            for x in selection_events
            if x["completed_expansion"]
        ),
        "parent_age_one_fraction_completed": safe_div(
            sum(
                x["selection_step"] - int(x["parent_timestep"]) == 1
                for x in selection_events
                if x["completed_expansion"]
            ),
            sum(x["completed_expansion"] for x in selection_events),
        ),
        "final_archive_size": final_archive,
        "final_nonseed_archive": nonseed_archive,
        "final_archive_fraction_of_all_rollouts": safe_div(nonseed_archive, len(metas)),
        "final_archive_fraction_of_valid_rollouts": safe_div(
            nonseed_archive, valid_rollouts
        ),
        "conversion_by_memory_arm": by_arm,
        "lineage_final_step": lineage_rows[-1],
        "root_seed_ids_seen": len(seed_ids),
    }, lineage_rows


def overall_summary(config, summaries, selections, metas, steps, memory_stats, feedback, tree):
    valid = [x for x in metas if x.get("valid") and x.get("raw_score") is not None]
    best = min(float(x["raw_score"]) for x in valid)
    best_items = [x for x in valid if float(x["raw_score"]) == best]
    best_item = best_items[0]
    best_recomputed = recompute_erdos_c5(best_item["construction_parsed"])
    first_practical = min(
        int(x["step"])
        for x in valid
        if float(x["raw_score"]) <= best + 1e-12
    )
    step0_parents = selections[0]["parents"]
    seed_best = min(float(x["parent_raw_score"]) for x in step0_parents)
    step0_best = steps[0]["step_best_raw"]
    final_step = max(summaries)
    return {
        "configured_steps": int(config["num_steps"]),
        "completed_steps": len(summaries),
        "completed_step_range": [min(summaries), final_step],
        "incomplete_parent_only_steps": sorted(set(selections) - set(summaries)),
        "rollouts": len(metas),
        "rollouts_per_completed_step": int(config["groups_per_step"])
        * int(config["group_size"]),
        "valid_rollouts": len(valid),
        "valid_fraction": safe_div(len(valid), len(metas)),
        "failure_counts": dict(Counter(x.get("failure_kind", "") for x in metas)),
        "best_seed_raw": seed_best,
        "best_step0_raw": step0_best,
        "best_final_raw": best,
        "best_reward": max(float(x["reward"]) for x in valid),
        "best_source_meta": best_item["_meta_path"],
        "best_recomputed_raw": best_recomputed,
        "best_recompute_absolute_error": abs(best_recomputed - best),
        "best_exact_first_step": min(int(x["step"]) for x in best_items),
        "best_within_1e_12_first_step": first_practical,
        "seed_to_final_relative_reduction": safe_div(seed_best - best, seed_best),
        "step0_to_final_relative_reduction": safe_div(step0_best - best, step0_best),
        "target": float(config["target"]),
        "target_gap": best - float(config["target"]),
        "target_hits": sum(float(x["raw_score"]) <= float(config["target"]) for x in valid),
        "stated_record": TARGET_RECORD,
        "record_gap": best - TARGET_RECORD,
        "record_hits": sum(float(x["raw_score"]) <= TARGET_RECORD for x in valid),
        "last_steps_without_practical_best_improvement": final_step - first_practical,
        "last_three_distinct_good": [
            summaries[s].get("distinct_good") for s in range(final_step - 2, final_step + 1)
        ],
        "response_tokens_total": sum(int(x.get("n_response_tokens", 0)) for x in metas),
        "response_tokens_mean": mean(int(x.get("n_response_tokens", 0)) for x in metas),
        "response_tokens_p95": quantile(
            [int(x.get("n_response_tokens", 0)) for x in metas], 0.95
        ),
        "unique_exact_codes": len({x["code_hash"] for x in metas if x["code_hash"]}),
        "unique_structural_codes": len(
            {x["structural_code_hash"] for x in metas if x["structural_code_hash"]}
        ),
        "unique_valid_exact_codes": len(
            {x["code_hash"] for x in valid if x["code_hash"]}
        ),
        "unique_valid_constructions_12dp": len(
            {
                construction_key(x.get("construction_parsed"))
                for x in valid
                if x.get("construction_parsed")
            }
        ),
        "memory": memory_stats,
        "feedback": feedback,
        "tree": tree,
        "evidence_limits": [
            "This is one non-deterministic combined-treatment run; it cannot identify the total method effect versus a plain run.",
            "Memory has a within-parent randomized no-memory arm, so its arm contrasts are the strongest causal evidence available here.",
            "Feedback has no simultaneous no-feedback arm and repaired teacher outputs are not evaluated or saved.",
            "The copied artifacts omit training logs, token-level feedback statistics, adapter checkpoints, and wall-clock timing.",
        ],
    }


class Svg:
    def __init__(self, width=1200, height=760):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#17202a}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#566573}.axis{stroke:#aab7b8;stroke-width:1}.grid{stroke:#e5e8e8;stroke-width:1}.label{font-size:12px}.legend{font-size:13px;font-weight:600}</style>',
        ]

    def line(self, x1, y1, x2, y2, stroke="#333", width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{extra}/>'
        )

    def rect(self, x, y, w, h, fill, stroke="none", opacity=1.0, rx=0):
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" opacity="{opacity}" rx="{rx}"/>'
        )

    def text(self, x, y, value, cls="label", anchor="start", fill=None):
        style = f' style="fill:{fill}"' if fill else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{style}>{html.escape(str(value))}</text>'
        )

    def polyline(self, points, stroke, width=2.5):
        data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{data}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def circle(self, x, y, r, fill):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}"/>')

    def save(self, path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts))


def line_panel(svg, rows, xkey, series, box, y_min, y_max, title, y_ticks, feedback_steps=None):
    x0, y0, width, height = box
    steps = [float(r[xkey]) for r in rows]
    xmin, xmax = min(steps), max(steps)

    def xx(value):
        return x0 + safe_div(value - xmin, xmax - xmin or 1) * width

    def yy(value):
        return y0 + height - safe_div(value - y_min, y_max - y_min or 1) * height

    if feedback_steps:
        for step in feedback_steps:
            half = width / (xmax - xmin + 1) / 2
            svg.rect(xx(step) - half, y0, 2 * half, height, "#fef9e7", opacity=0.9)
    for value in y_ticks:
        svg.line(x0, yy(value), x0 + width, yy(value), stroke="#e5e8e8")
        svg.text(x0 - 10, yy(value) + 4, f"{value:.6g}", anchor="end")
    svg.line(x0, y0, x0, y0 + height, stroke="#7f8c8d")
    svg.line(x0, y0 + height, x0 + width, y0 + height, stroke="#7f8c8d")
    for step in range(int(xmin), int(xmax) + 1, 5):
        svg.text(xx(step), y0 + height + 20, step, anchor="middle")
    svg.text(x0, y0 - 12, title, cls="legend")
    for label, key, color in series:
        points = [(xx(float(r[xkey])), yy(float(r[key]))) for r in rows if r[key] is not None]
        svg.polyline(points, color)
        for x, y in points:
            svg.circle(x, y, 2.4, color)
    return xx, yy


def plot_dynamics(steps, feedback_steps, target):
    svg = Svg(1200, 780)
    svg.text(60, 42, "Search dynamics: strong early descent, then a flat tail", cls="title")
    svg.text(60, 66, "Yellow bands are steps where adaptive code-feedback ran.", cls="subtitle")

    best_values = [r["cumulative_best_raw"] for r in steps]
    low = min(min(best_values), target, TARGET_RECORD) - 0.00002
    high = min(max(best_values), 0.3830) + 0.00003
    xx, yy = line_panel(
        svg,
        steps,
        "step",
        [("cumulative best", "cumulative_best_raw", "#2471a3")],
        (95, 105, 1040, 270),
        low,
        high,
        "C5 bound (lower is better; y-axis clipped above 0.383)",
        [0.3810, 0.3820, high],
        feedback_steps,
    )
    svg.line(95, yy(target), 1135, yy(target), stroke="#c0392b", width=2, dash="7 5")
    svg.line(95, yy(TARGET_RECORD), 1135, yy(TARGET_RECORD), stroke="#d68910", width=2, dash="3 4")
    svg.text(1128, yy(target) - 7, f"target {target:.6f}", anchor="end", fill="#c0392b")
    svg.text(1128, yy(TARGET_RECORD) + 16, f"stated record {TARGET_RECORD:.5f}", anchor="end", fill="#d68910")

    line_panel(
        svg,
        steps,
        "step",
        [
            ("scientifically valid", "valid_fraction", "#1e8449"),
            ("code-valid", "code_valid_fraction", "#884ea0"),
        ],
        (95, 460, 1040, 220),
        0.0,
        1.0,
        "Validity fractions",
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        feedback_steps,
    )
    svg.line(770, 82, 810, 82, stroke="#2471a3", width=3)
    svg.text(820, 87, "cumulative best", cls="legend")
    svg.line(770, 432, 810, 432, stroke="#1e8449", width=3)
    svg.text(820, 437, "scientifically valid", cls="legend")
    svg.line(970, 432, 1010, 432, stroke="#884ea0", width=3)
    svg.text(1020, 437, "code-valid", cls="legend")
    svg.text(615, 735, "completed training step", anchor="middle", cls="legend")
    svg.save(HERE / "run_dynamics.svg")


def plot_memory(memory_stats):
    svg = Svg(1200, 760)
    svg.text(60, 42, "Memory arms: feasibility gain is clear; tail-quality gain is not", cls="title")
    svg.text(60, 66, "Each comparison is matched to the same parent and a no-memory arm.", cls="subtitle")

    phases = ["early_1_7", "middle_8_25", "late_26_33"]
    phase_labels = ["steps 1–7", "steps 8–25", "steps 26–33"]
    metrics = [
        ("valid_delta", "validity delta", "#1e8449"),
        ("parent_improvement_delta", "parent-improvement delta", "#2471a3"),
    ]
    x0, y0, width, height = 100, 125, 1010, 260
    svg.line(x0, y0 + height, x0 + width, y0 + height, stroke="#7f8c8d")
    svg.line(x0, y0, x0, y0 + height, stroke="#7f8c8d")
    y_min, y_max = -0.12, 0.36
    for tick in [-0.1, 0.0, 0.1, 0.2, 0.3]:
        yy = y0 + height - safe_div(tick - y_min, y_max - y_min) * height
        svg.line(x0, yy, x0 + width, yy, stroke="#e5e8e8")
        svg.text(x0 - 10, yy + 4, f"{tick:+.0%}", anchor="end")
    group_w = width / len(phases)
    bar_w = 34
    colors = {"selected": "#2874a6", "explore": "#a569bd"}
    for pi, phase in enumerate(phases):
        center = x0 + group_w * (pi + 0.5)
        svg.text(center, y0 + height + 24, phase_labels[pi], anchor="middle", cls="legend")
        offsets = [-58, -18, 30, 70]
        oi = 0
        for arm in ("selected", "explore"):
            for key, _label, _metric_color in metrics:
                value = memory_stats["phases"][phase][arm][key]
                yy0 = y0 + height - safe_div(0 - y_min, y_max - y_min) * height
                yyv = y0 + height - safe_div(value - y_min, y_max - y_min) * height
                top, bh = min(yy0, yyv), abs(yyv - yy0)
                shade = colors[arm]
                opacity = 1.0 if key == "valid_delta" else 0.55
                svg.rect(center + offsets[oi] - bar_w / 2, top, bar_w, bh, shade, opacity=opacity, rx=2)
                oi += 1

    svg.text(100, 445, "Matched best-of-13 reward uplift", cls="legend")
    x0, y0, width, height = 100, 480, 1010, 180
    y_min, y_max = -0.12, 0.14
    for tick in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        yy = y0 + height - safe_div(tick - y_min, y_max - y_min) * height
        svg.line(x0, yy, x0 + width, yy, stroke="#e5e8e8")
        svg.text(x0 - 10, yy + 4, f"{tick:+.2f}", anchor="end")
    group_w = width / len(phases)
    for pi, phase in enumerate(phases):
        center = x0 + group_w * (pi + 0.5)
        for ai, arm in enumerate(("selected", "explore")):
            value = memory_stats["phases"][phase][arm]["tail_uplift"]
            yy0 = y0 + height - safe_div(0 - y_min, y_max - y_min) * height
            clipped = min(y_max, max(y_min, value))
            yyv = y0 + height - safe_div(clipped - y_min, y_max - y_min) * height
            svg.rect(center + (-28 if ai == 0 else 8), min(yy0, yyv), 28, abs(yyv - yy0), colors[arm], rx=2)
            svg.text(center + (-14 if ai == 0 else 22), min(yy0, yyv) - 7, f"{value:+.3f}", anchor="middle")
    svg.rect(760, 82, 20, 12, colors["selected"])
    svg.text(790, 93, "selected lesson", cls="legend")
    svg.rect(930, 82, 20, 12, colors["explore"])
    svg.text(960, 93, "UCB explore lesson", cls="legend")
    svg.text(100, 710, "Solid bars: validity; translucent bars: beats-parent rate. Bottom bars are reward uplift.", cls="subtitle")
    svg.save(HERE / "memory_effects.svg")


def plot_tree(overall, lineage_rows, steps):
    svg = Svg(1200, 800)
    svg.text(60, 42, "Tree efficiency: most samples train the policy but never guide a later expansion", cls="title")
    svg.text(60, 66, "Parent counts use completed steps; step 34 only saved a selection event.", cls="subtitle")

    counts = [
        ("generated", overall["rollouts"], "#5dade2"),
        ("scientifically valid", overall["valid_rollouts"], "#58d68d"),
        ("in final non-seed archive", overall["tree"]["final_nonseed_archive"], "#f5b041"),
        ("expanded as a later parent", overall["tree"]["generated_parent_unique_completed"], "#af7ac5"),
    ]
    max_count = counts[0][1]
    x0, y0, max_w = 285, 115, 830
    for i, (label, count, color) in enumerate(counts):
        y = y0 + i * 68
        width = max_w * count / max_count
        svg.text(x0 - 18, y + 25, label, anchor="end", cls="legend")
        svg.rect(x0, y, width, 38, color, rx=5)
        value = f"{count:,}  ({count / max_count:.2%})"
        if width > 0.8 * max_w:
            svg.text(x0 + width - 12, y + 25, value, cls="legend", anchor="end", fill="#ffffff")
        else:
            svg.text(x0 + width + 12, y + 25, value, cls="legend")

    svg.text(60, 430, "Lineage concentration among the 8 selected parents", cls="legend")
    rows = [r for r in lineage_rows if r["step"] > 0]
    line_panel(
        svg,
        rows,
        "step",
        [
            ("dominant seed root", "dominant_root_fraction", "#c0392b"),
            ("shared ancestry", "common_ancestor_fraction_of_median_depth", "#884ea0"),
        ],
        (95, 470, 1040, 215),
        0.0,
        1.0,
        "Fraction",
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    svg.line(755, 425, 795, 425, stroke="#c0392b", width=3)
    svg.text(805, 430, "dominant seed root", cls="legend")
    svg.line(955, 425, 995, 425, stroke="#884ea0", width=3)
    svg.text(1005, 430, "ancestors shared by all", cls="legend")
    svg.text(615, 743, "selection step", anchor="middle", cls="legend")
    svg.text(60, 775, "All 280 saved parent selections had visit_count = 0, so selected nodes were always unvisited leaves.", cls="subtitle")
    svg.save(HERE / "tree_efficiency.svg")


def main():
    print(f"reading {RUN}", flush=True)
    config, summaries, selections, metas = load_artifacts()
    add_code_hashes(metas)
    memory = read_json(RUN / "memory.json")

    steps = step_rows(config, summaries, metas)
    contrasts = memory_contrasts(metas)
    memory_stats = memory_summary(config, metas, contrasts, memory)
    feedback, feedback_rows = feedback_analysis(config, summaries, metas)
    tree, lineage_rows = tree_analysis(config, summaries, selections, metas)
    overall = overall_summary(
        config,
        summaries,
        selections,
        metas,
        steps,
        memory_stats,
        feedback,
        tree,
    )

    write_csv(HERE / "step_metrics.csv", steps)
    write_csv(HERE / "memory_group_contrasts.csv", contrasts)
    write_csv(HERE / "feedback_signatures.csv", feedback_rows)
    write_csv(HERE / "lineage_metrics.csv", lineage_rows)
    (HERE / "metrics.json").write_text(json.dumps(overall, indent=2, sort_keys=True))

    plot_dynamics(steps, feedback["active_steps"], float(config["target"]))
    plot_memory(memory_stats)
    plot_tree(overall, lineage_rows, steps)
    print(json.dumps({
        "completed_steps": overall["completed_steps"],
        "rollouts": overall["rollouts"],
        "best_final_raw": overall["best_final_raw"],
        "target_gap": overall["target_gap"],
        "record_gap": overall["record_gap"],
        "valid_fraction": overall["valid_fraction"],
        "tree_parent_rate": overall["tree"]["rollout_to_completed_parent_rate"],
        "selected_memory_valid_delta": overall["memory"]["matched_contrasts"]["selected"]["mean_valid_rate_delta"],
        "selected_memory_tail_uplift": overall["memory"]["matched_contrasts"]["selected"]["mean_tail_uplift"],
        "feedback_teacher_forwards": overall["feedback"]["teacher_forwards"],
    }, indent=2))


if __name__ == "__main__":
    main()
