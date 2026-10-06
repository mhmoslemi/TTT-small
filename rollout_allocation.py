"""Fixed-budget allocation for adaptive two-stage rollout generation.

The discovery objective is the best reward found, not the identity of the
strategy with the largest mean reward.  After the pilot, this module treats
each strategy as a zero-inflated Bayesian arm and greedily maximizes the
posterior expected improvement of the best reward under the remaining fixed
budget.  The greedy planning loop is CPU-only: it returns all phase-2 counts
before any phase-2 generation begins.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


_POSTERIOR_SCENARIOS = 4096


def _equal_allocation(total: int, count: int) -> list[int]:
    """Return a deterministic balanced integer allocation."""
    if count <= 0:
        if total:
            raise ValueError("cannot allocate a positive budget to no arms")
        return []
    base, remainder = divmod(int(total), int(count))
    return [base + (1 if index < remainder else 0)
            for index in range(count)]


def allocate_posterior_expected_best(
        reward_groups: Sequence[Sequence[float]],
        valid_groups: Sequence[Sequence[bool]],
        total: int,
        *,
        fail_reward: float,
        seed: int,
        posterior_scenarios: int = _POSTERIOR_SCENARIOS,
        ) -> tuple[list[int], list[dict], dict]:
    """Allocate ``total`` future samples to maximize expected best reward.

    Each arm has a Beta posterior for producing a valid rollout and a weak
    empirical-Bayes Normal-Inverse-Gamma posterior for the reward conditional
    on validity.  Monte Carlo scenarios integrate both parameter uncertainty
    and future rollout noise.  For every unit of the fixed budget, the greedy
    planner assigns the unit with the largest marginal increase in the
    expected maximum.  Because expected maximum is monotone submodular, this
    is the standard greedy fixed-budget approximation; it is not a sequence of
    actual generations or evaluation barriers.

    Returns ``(allocations, arm_diagnostics, summary)``.  Allocations are
    non-negative integers summing exactly to ``total``.
    """
    total = int(total)
    if total < 0:
        raise ValueError("phase-2 rollout budget must be non-negative")
    if len(reward_groups) != len(valid_groups):
        raise ValueError("reward_groups and valid_groups must align")
    n_arms = len(reward_groups)
    if n_arms == 0:
        if total:
            raise ValueError("cannot allocate a positive budget to no arms")
        return [], [], {
            "method": "posterior-expected-best",
            "pilot_best_reward": float(fail_reward),
            "projected_expected_gain": 0.0,
            "posterior_scenarios": 0,
            "fallback_equal": True,
        }
    if posterior_scenarios < 256:
        raise ValueError("posterior_scenarios must be at least 256")

    rewards = []
    valid_masks = []
    valid_rewards = []
    all_rewards = []
    for arm_index, (arm_rewards, arm_valid) in enumerate(
            zip(reward_groups, valid_groups)):
        values = np.asarray(list(arm_rewards), dtype=np.float64)
        mask = np.asarray(list(arm_valid), dtype=np.bool_)
        if values.ndim != 1 or mask.ndim != 1 or len(values) != len(mask):
            raise ValueError(
                f"pilot rewards/validity for arm {arm_index} must be aligned "
                "one-dimensional sequences")
        if len(values) == 0:
            raise ValueError(f"arm {arm_index} has no pilot observations")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"arm {arm_index} has a non-finite pilot reward")
        rewards.append(values)
        valid_masks.append(mask)
        selected = values[mask]
        valid_rewards.append(selected)
        all_rewards.extend(float(value) for value in values)

    pilot_best = float(max(all_rewards))
    pooled_valid = np.concatenate(
        [values for values in valid_rewards if len(values)], axis=0
    ) if any(len(values) for values in valid_rewards) else np.asarray(
        [], dtype=np.float64)

    diagnostics = []
    for values, mask in zip(rewards, valid_masks):
        valid_count = int(mask.sum())
        diagnostics.append({
            "pilot_count": int(len(values)),
            "valid_count": valid_count,
            "posterior_valid_probability": float(
                (valid_count + 1.0) / (len(values) + 2.0)),
            "pilot_reward_mean": float(values.mean()),
            "pilot_reward_variance": float(values.var(ddof=0)),
            "pilot_best_reward": float(values.max()),
            "initial_expected_improvement": 0.0,
            "final_marginal_expected_improvement": 0.0,
        })

    if total == 0:
        return [0] * n_arms, diagnostics, {
            "method": "posterior-expected-best",
            "pilot_best_reward": pilot_best,
            "projected_expected_gain": 0.0,
            "posterior_scenarios": int(posterior_scenarios),
            "fallback_equal": False,
        }

    # If every pilot failed, the data contain no information about the reward
    # conditional on validity and every arm has the same Beta posterior.  An
    # equal allocation is then the exact symmetry-preserving answer; inventing
    # a reward scale would create a meaningless preference.
    if pooled_valid.size == 0:
        allocations = _equal_allocation(total, n_arms)
        return allocations, diagnostics, {
            "method": "posterior-expected-best",
            "pilot_best_reward": pilot_best,
            "projected_expected_gain": 0.0,
            "posterior_scenarios": int(posterior_scenarios),
            "fallback_equal": True,
        }

    pooled_mean = float(pooled_valid.mean())
    if pooled_valid.size > 1:
        pooled_scale = float(pooled_valid.std(ddof=1))
    else:
        pooled_scale = 0.0
    observed_span = float(max(all_rewards) - min(all_rewards))
    scale_reference = max(
        abs(pooled_mean - float(fail_reward)),
        abs(pooled_mean), observed_span, np.finfo(np.float64).eps,
    )
    if not _is_positive_finite(pooled_scale):
        pooled_scale = 0.05 * scale_reference
    pooled_scale = max(pooled_scale, np.finfo(np.float64).eps)

    # A weak common prior lets an arm with few valid pilots retain uncertainty
    # without allowing a single observation to imply zero conditional reward
    # variance.  Its centre and scale are learned from this parent's complete
    # valid pilot set, so the allocation remains reward-scale invariant.
    prior_kappa = 0.5
    prior_alpha = 2.5
    prior_beta = (prior_alpha - 1.0) * pooled_scale * pooled_scale
    rng = np.random.default_rng(int(seed) % (2 ** 32))
    future_rewards = []

    for values, mask, selected in zip(
            rewards, valid_masks, valid_rewards):
        pilot_count = int(len(values))
        valid_count = int(mask.sum())

        posterior_valid = rng.beta(
            valid_count + 1.0,
            pilot_count - valid_count + 1.0,
            size=posterior_scenarios,
        )

        if valid_count:
            sample_mean = float(selected.mean())
            centered_sum = float(
                np.square(selected - sample_mean).sum())
        else:
            sample_mean = pooled_mean
            centered_sum = 0.0
        posterior_kappa = prior_kappa + valid_count
        posterior_mean = (
            prior_kappa * pooled_mean + valid_count * sample_mean
        ) / posterior_kappa
        posterior_alpha = prior_alpha + 0.5 * valid_count
        mean_offset = sample_mean - pooled_mean
        posterior_beta = (
            prior_beta
            + 0.5 * centered_sum
            + (prior_kappa * valid_count * mean_offset * mean_offset)
            / (2.0 * posterior_kappa)
        )
        posterior_beta = max(
            float(posterior_beta), np.finfo(np.float64).tiny)

        precision = rng.gamma(
            shape=posterior_alpha,
            scale=1.0 / posterior_beta,
            size=posterior_scenarios,
        )
        conditional_variance = 1.0 / np.maximum(
            precision, np.finfo(np.float64).tiny)
        conditional_mean = rng.normal(
            loc=posterior_mean,
            scale=np.sqrt(conditional_variance / posterior_kappa),
            size=posterior_scenarios,
        )
        conditional_draws = rng.normal(
            loc=conditional_mean[None, :],
            scale=np.sqrt(conditional_variance)[None, :],
            size=(total, posterior_scenarios),
        )
        # fail_reward is the lower endpoint used by the discovery engine.
        conditional_draws = np.maximum(
            conditional_draws, float(fail_reward))
        is_valid = rng.random(
            (total, posterior_scenarios)) < posterior_valid[None, :]
        future_rewards.append(np.where(
            is_valid, conditional_draws, float(fail_reward)))

    allocations = [0] * n_arms
    scenario_best = np.full(
        posterior_scenarios, pilot_best, dtype=np.float64)
    last_marginal = [0.0] * n_arms

    initial_gains = []
    for arm_index in range(n_arms):
        improvement = np.maximum(
            scenario_best, future_rewards[arm_index][0]) - scenario_best
        initial_gains.append(float(improvement.mean()))
        diagnostics[arm_index]["initial_expected_improvement"] = float(
            initial_gains[-1])

    for _ in range(total):
        gains = []
        for arm_index in range(n_arms):
            sample_index = allocations[arm_index]
            candidate = future_rewards[arm_index][sample_index]
            gains.append(float(
                (np.maximum(scenario_best, candidate) - scenario_best).mean()
            ))

        best_gain = max(gains)
        # When all posterior marginal gains have numerically vanished, the
        # objective is indifferent.  Balance the unused budget rather than
        # letting an arbitrary arm index absorb it.
        if best_gain <= np.finfo(np.float64).eps * scale_reference:
            remaining = total - sum(allocations)
            balanced = _equal_allocation(remaining, n_arms)
            allocations = [current + extra for current, extra in zip(
                allocations, balanced)]
            break

        chosen = max(
            range(n_arms),
            key=lambda index: (
                gains[index],
                initial_gains[index],
                diagnostics[index]["pilot_best_reward"],
                diagnostics[index]["pilot_reward_mean"],
                -index,
            ),
        )
        last_marginal[chosen] = gains[chosen]
        scenario_best = np.maximum(
            scenario_best,
            future_rewards[chosen][allocations[chosen]],
        )
        allocations[chosen] += 1

    if sum(allocations) != total or any(count < 0 for count in allocations):
        raise RuntimeError("posterior expected-best allocation lost budget")
    for index, marginal in enumerate(last_marginal):
        diagnostics[index]["final_marginal_expected_improvement"] = float(
            marginal)

    return allocations, diagnostics, {
        "method": "posterior-expected-best",
        "pilot_best_reward": pilot_best,
        "projected_expected_gain": float(
            np.maximum(scenario_best - pilot_best, 0.0).mean()),
        "posterior_scenarios": int(posterior_scenarios),
        "fallback_equal": False,
    }


def _is_positive_finite(value: float) -> bool:
    """Small local predicate kept dependency-free apart from NumPy."""
    return bool(np.isfinite(value) and value > 0.0)
