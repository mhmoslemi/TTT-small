"""Budget-neutral memory arms and causal tail credit.

The unit of intervention is a prompt arm under one parent.  The parent still
receives exactly K rollouts; K is merely divided among selected-memory,
no-memory, and under-tested-memory prompts.  Outcome credit compares treatment
and null arms at the same effective sample size, avoiding the automatic
best-of-more advantage of a larger arm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

from memory.types import Lesson


@dataclass
class MemoryArm:
    name: str
    lessons: List[Lesson]
    count: int

    @property
    def ids(self) -> List[str]:
        return [lesson.id for lesson in self.lessons]


def _fractional_counts(total: int, weights: Sequence[float]) -> List[int]:
    """Largest-remainder allocation that sums exactly to ``total``."""
    total = max(0, int(total))
    clean = [max(0.0, float(w)) for w in weights]
    scale = sum(clean)
    if total == 0:
        return [0] * len(clean)
    if scale <= 0:
        out = [0] * len(clean)
        if out:
            out[0] = total
        return out
    raw = [total * w / scale for w in clean]
    out = [int(math.floor(x)) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def _ensure_minimum_counts(counts: List[int], minimum: int) -> List[int]:
    """Move budget between active arms until each has ``minimum`` samples."""
    counts = list(counts)
    minimum = int(minimum)
    if minimum <= 0 or len(counts) <= 1:
        return counts
    if minimum * len(counts) > sum(counts):
        raise ValueError(
            f"group budget cannot provide {minimum} rollout(s) to each of "
            f"{len(counts)} active memory arms")
    for receiver in range(len(counts)):
        while counts[receiver] < minimum:
            donors = [idx for idx in range(len(counts))
                      if idx != receiver and counts[idx] > minimum]
            if not donors:
                raise ValueError("unable to satisfy V2 memory arm minimums")
            donor = max(donors, key=lambda idx: counts[idx] - minimum)
            counts[donor] -= 1
            counts[receiver] += 1
    return counts


def allocate_memory_arms(group_size: int, selected: Sequence[Lesson], bank,
                         cfg, step: int,
                         reservations: Optional[Dict[str, int]] = None
                         ) -> List[MemoryArm]:
    """Allocate a parent's existing rollout budget across prompt variants."""
    k = max(0, int(group_size))
    # Preserve the public meaning of lookup_mode=none: extraction can remain
    # active for ablation, but no bank entry reaches a rollout prompt.
    if getattr(cfg, "lookup_mode", "select") == "none":
        return [MemoryArm("no_memory", [], k)]
    selected = list(selected or ())[:int(cfg.arm_max_lessons)]
    control_f = float(cfg.arm_control_fraction)
    explore_f = float(cfg.arm_explore_fraction)

    # Exact legacy path: one prompt and all K samples.
    if control_f <= 0 and explore_f <= 0:
        return [MemoryArm("selected" if selected else "no_memory", selected, k)]

    excluded = {lesson.id for lesson in selected}
    is_v2 = bool(getattr(cfg, "is_v2", False))
    explore_kwargs = ({"reservations": reservations} if is_v2 else {})
    explorer = bank.exploration_lesson(
        excluded, step=step, c=float(cfg.arm_exploration_c),
        **explore_kwargs) if bank else None

    selected_f = max(0.0, 1.0 - control_f - explore_f)
    specs = [
        ("selected", selected, selected_f if selected else 0.0),
        ("no_memory", [], control_f + (selected_f if not selected else 0.0)),
        ("explore", [explorer] if explorer is not None else [],
         explore_f if explorer is not None else 0.0),
    ]
    if explorer is None:
        # Preserve total budget when the bank has no alternative lesson.
        specs[1] = ("no_memory", [], specs[1][2] + explore_f)

    positive = [(name, lessons, weight) for name, lessons, weight in specs
                if weight > 0]
    counts = _fractional_counts(k, [x[2] for x in positive])
    if is_v2 and len(positive) > 1:
        counts = _ensure_minimum_counts(
            counts, int(getattr(cfg, "arm_comparison_n", 0) or 0))
    arms = [MemoryArm(name, list(lessons), count)
            for (name, lessons, _), count in zip(positive, counts) if count > 0]
    if is_v2 and explorer is not None and reservations is not None \
            and any(arm.name == "explore" for arm in arms):
        reservations[explorer.id] = int(reservations.get(explorer.id, 0)) + 1
    if not arms and k:
        arms = [MemoryArm("no_memory", [], k)]
    return arms


def expected_subsample_max(values: Sequence[float], n: int) -> float:
    """Expected maximum of a uniformly drawn size-n subset, without replacement."""
    xs = sorted(float(x) for x in values)
    n = int(n)
    if n <= 0 or n > len(xs):
        raise ValueError("n must be in [1, len(values)]")
    denom = math.comb(len(xs), n)
    # Sorted item j can be the maximum exactly when it is chosen and the other
    # n-1 items come from the j items below it.
    return sum(x * math.comb(j, n - 1) / denom
               for j, x in enumerate(xs) if j >= n - 1)


def _distinct_programs(values: Sequence[str]) -> int:
    return len({str(value).strip() for value in values or ()
                if str(value).strip()})


def credit_memory_arms(bank, observations: Dict[str, Dict], parent_reward: float,
                       step: int, parent_id: str = "") -> List[Dict]:
    """Credit memory arms against this parent's randomized null arm."""
    control = observations.get("no_memory")
    if bank is None or not control or not control.get("rewards"):
        return []
    control_rewards = list(control["rewards"])
    is_v2 = bool(getattr(getattr(bank, "cfg", None), "is_v2", False))
    fixed_n = (int(getattr(bank.cfg, "arm_comparison_n", 0) or 0)
               if is_v2 else 0)
    if is_v2 and fixed_n <= 0:
        raise ValueError("memory V2 requires a fixed comparison_n")
    updates = []
    for name, obs in observations.items():
        ids = list(obs.get("memory_ids") or ())
        rewards = list(obs.get("rewards") or ())
        if name == "no_memory" or not ids or not rewards:
            continue
        n = fixed_n if is_v2 else min(len(rewards), len(control_rewards))
        if n > len(rewards) or n > len(control_rewards):
            raise ValueError(
                f"memory arm {name!r} cannot support fixed best@n={n}: "
                f"treatment={len(rewards)}, control={len(control_rewards)}")
        uplift = (expected_subsample_max(rewards, n)
                  - expected_subsample_max(control_rewards, n))
        valid = sum(bool(v) for v in obs.get("valids", ()))
        control_valid = sum(bool(v) for v in control.get("valids", ()))
        improved = sum(float(r) > float(parent_reward) for r in rewards)
        if is_v2:
            distinct = _distinct_programs(obs.get("codes", ()))
            control_distinct = _distinct_programs(control.get("codes", ()))
            bank.record_outcome(
                ids, rollouts=len(rewards), valid=valid, improved=improved,
                tail_uplift=uplift, step=step, comparison_n=n, arm=name,
                context_id=parent_id, parent_reward=parent_reward,
                control_rollouts=len(control_rewards),
                control_valid=control_valid, distinct_codes=distinct,
                control_distinct_codes=control_distinct)
            updates.append({
                "arm": name, "ids": ids, "n": n,
                "tail_uplift": uplift,
                "valid": valid, "rollouts": len(rewards),
                "control_valid": control_valid,
                "control_rollouts": len(control_rewards),
                "valid_rate_delta": (
                    valid / len(rewards)
                    - control_valid / len(control_rewards)),
                "distinct_codes": distinct,
                "control_distinct_codes": control_distinct,
                "exact_code_unique_rate": distinct / len(rewards),
                "control_exact_code_unique_rate": (
                    control_distinct / len(control_rewards)),
                "improved": improved,
                "parent_id": str(parent_id or ""),
            })
        else:
            bank.record_outcome(
                ids, rollouts=len(rewards), valid=valid, improved=improved,
                tail_uplift=uplift, step=step)
            updates.append({"arm": name, "ids": ids, "n": n,
                            "tail_uplift": uplift, "valid": valid,
                            "rollouts": len(rewards), "improved": improved})
    return updates
