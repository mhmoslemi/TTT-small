"""Retired batch-growth mechanism, archived for reference only.

Nothing in the active runner imports this module. G here means groups per
step and K means rollouts per group (the old log naming, not X-GRPO naming).

Former settings (also removed from the YAML presets):
    max_groups_per_step: null
    max_group_size: null
    growth_force_step: 10
    growth_valid_yield: 0.7
    growth_distinct_min: 4
    growth_factor: 2.0

Former wiring:
  * Resolve null maxima to the starting batch sizes. CLI starting sizes also
    replace their maxima when no explicit CLI maximum is given.
  * Strategy mode pins max_group_size to strategies_per_parent *
    programs_per_strategy.
  * Before a step at/after growth_force_step, force both sizes to their caps.
  * After earlier steps, call grow_batch with the signals collected below.
  * Persist the resulting sizes as next_groups_per_step / next_group_size in
    version-1 training checkpoints and restore them before the next step.

The active runner now uses fixed resolved configuration sizes, writes v2
checkpoints without growth state, and accepts v1 checkpoints while ignoring
their next-batch fields. Pilot/phase-2 allocation is a separate active feature.
"""


def resolve_growth_caps(config, *, groups_overridden=False,
                        group_size_overridden=False,
                        max_groups_overridden=False,
                        max_group_size_overridden=False):
    """Archived configuration resolution; returns a copy."""
    merged = dict(config)
    if groups_overridden and not max_groups_overridden:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if group_size_overridden and not max_group_size_overridden:
        merged["max_group_size"] = int(merged["group_size"])
    if merged["max_groups_per_step"] is None:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if merged["max_group_size"] is None:
        merged["max_group_size"] = int(merged["group_size"])
    if int(merged["max_groups_per_step"]) < int(merged["groups_per_step"]):
        raise ValueError("max_groups_per_step cannot be below groups_per_step")
    if int(merged["max_group_size"]) < int(merged["group_size"]):
        raise ValueError("max_group_size cannot be below group_size")
    return merged


def collect_growth_signals(groups):
    """Groups contain (records, valid_flags, rewards, code_strings, parent_value).

    Recovery attempts were excluded from these signals. Code hashes retained
    the original whitespace-stripped deduplication behavior.
    """
    best_valid_yield = 0.0
    distinct_good_hashes = set()
    for records, valids, rewards, codes, parent_value in groups:
        planned_valids = [valid for record, valid in zip(records, valids)
                         if not record.get("retry_attempt", 0)]
        if planned_valids:
            best_valid_yield = max(
                best_valid_yield, sum(planned_valids) / len(planned_valids))
        parent_value = float(parent_value) if parent_value is not None else 0.0
        for record, valid, reward, code in zip(records, valids, rewards, codes):
            if (not record.get("retry_attempt", 0)
                    and valid and code and reward > parent_value):
                distinct_good_hashes.add(hash(code.strip()))
    return {"best_valid_yield": float(best_valid_yield),
            "distinct_good": len(distinct_good_hashes)}


def batch_for_step(cur_g, cur_k, step, cfg):
    """Former pre-step forced-cap override."""
    if step >= cfg.growth_force_step:
        return int(cfg.max_groups_per_step), int(cfg.max_group_size)
    return cur_g, cur_k


def grow_batch(cur_g, cur_k, stats, cfg):
    """
    Monotonic ratchet. Grow (G, K) toward (max_groups_per_step, max_group_size)
    only when BOTH signals from the step just finished clear their thresholds:
      - best_valid_yield: the best group's valid fraction, and
      - distinct_good: the count of unique children that beat their parent.
    Otherwise hold. Never shrinks. The step >= growth_force_step override lives
    in the caller, not here.
    """
    g_max = int(cfg.max_groups_per_step)
    k_max = int(cfg.max_group_size)
    stats = stats or {}
    grow = (float(stats.get("best_valid_yield", 0.0)) >= cfg.growth_valid_yield
            and int(stats.get("distinct_good", 0)) >= cfg.growth_distinct_min)
    if grow:
        cur_g = min(g_max, int(round(cur_g * cfg.growth_factor)))
        cur_k = min(k_max, int(round(cur_k * cfg.growth_factor)))
    return cur_g, cur_k
