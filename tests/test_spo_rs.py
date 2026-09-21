import math

import numpy as np
import pytest

from spo_rs import SPORSTracker


def _tracker(**overrides):
    options = dict(entropic_beta=2.0, d_half=0.06, rho_min=0.875, rho_max=0.96)
    options.update(overrides)
    return SPORSTracker(**options)


def test_transformed_rewards_matches_exponential_reward_transform():
    tracker = _tracker(entropic_beta=1.5)
    rewards = [0.0, 1.0, -2.0, 0.5]
    transformed = tracker.transformed_rewards(rewards)
    assert np.allclose(transformed, np.exp(1.5 * np.asarray(rewards)))


def test_transformed_rewards_rejects_empty_or_nonfinite():
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.transformed_rewards([])
    with pytest.raises(ValueError):
        tracker.transformed_rewards([float("nan")])


def test_normalized_advantages_is_none_before_initialization():
    tracker = _tracker()
    transformed = tracker.transformed_rewards([0.1, 0.2, 0.3])
    assert tracker.normalized_advantages(transformed) is None


def test_normalized_advantages_matches_eq10_using_pre_step_baseline():
    beta = 2.0
    tracker = _tracker(entropic_beta=beta)
    seed_transformed = tracker.transformed_rewards([0.1, 0.4, -0.2, 0.05])
    tracker.update(seed_transformed, divergence=0.0, policy_adapter="adapter0", step=0)
    v0 = tracker.baseline()

    # Eq. 10: A_i = (1/beta) * (exp(beta*R_i)/v_hat_{t-1} - 1), using the
    # value saved *before* this step's update.
    transformed_1 = tracker.transformed_rewards([0.3, -0.1, 0.6])
    advantages = tracker.normalized_advantages(transformed_1)
    expected = (transformed_1 / v0 - 1.0) / beta
    assert np.allclose(advantages, expected)

    # Updating the tracker afterward must not retroactively change advantages
    # already computed this step (read-old-then-write-new).
    tracker.update(transformed_1, divergence=0.02, policy_adapter="adapter1", step=1)
    assert np.allclose(advantages, expected)
    assert abs(tracker.baseline() - v0) > 1e-6


def test_initialization_matches_eq18_and_eq19():
    rho_min = 0.875
    tracker = _tracker(rho_min=rho_min, rho_max=0.96)
    rewards = [0.2, -0.3, 0.7, 0.1, -0.05]
    transformed = tracker.transformed_rewards(rewards)
    result = tracker.update(transformed, divergence=0.0, policy_adapter="a0", step=0)

    expected_v0 = float(np.mean(transformed))
    expected_n0 = len(rewards) / (1.0 - rho_min)

    assert result["initialized"] is True
    assert math.isclose(tracker.baseline(), expected_v0, rel_tol=1e-12)
    assert math.isclose(result["effective_count_after"], expected_n0, rel_tol=1e-12)


def test_update_matches_eq13_through_eq17():
    beta, d_half, rho_min, rho_max = 2.0, 0.06, 0.875, 0.96
    tracker = _tracker(entropic_beta=beta, d_half=d_half, rho_min=rho_min, rho_max=rho_max)

    seed_transformed = tracker.transformed_rewards([0.1, 0.4, -0.2, 0.05])
    seed_result = tracker.update(
        seed_transformed, divergence=0.0, policy_adapter="a0", step=0)
    v_before = seed_result["value_after"]
    n_before = seed_result["effective_count_after"]

    divergence = 0.04
    rewards_1 = [0.3, -0.1, 0.6, 0.0, 0.2]
    transformed_1 = tracker.transformed_rewards(rewards_1)
    result = tracker.update(
        transformed_1, divergence=divergence, policy_adapter="a1", step=1)

    m1 = len(rewards_1)
    expected_rho = min(rho_max, max(rho_min, 2.0 ** (-divergence / d_half)))  # Eq. 13
    expected_n_after = expected_rho * n_before + m1                          # Eq. 14
    expected_eta = m1 / expected_n_after                                     # Eq. 15
    r_bar = float(np.mean(transformed_1))                                    # Eq. 11
    expected_v_eq16 = v_before + expected_eta * (r_bar - v_before)
    expected_v_eq17 = (
        expected_rho * n_before * v_before + float(np.sum(transformed_1))
    ) / expected_n_after

    # The paper states Eq. 16 and Eq. 17 are equivalent; confirm that holds
    # numerically before checking the implementation (which uses Eq. 17) against it.
    assert math.isclose(expected_v_eq16, expected_v_eq17, rel_tol=1e-9)

    assert math.isclose(result["rho"], expected_rho, rel_tol=1e-12)
    assert math.isclose(result["effective_count_after"], expected_n_after, rel_tol=1e-12)
    assert math.isclose(result["eta"], expected_eta, rel_tol=1e-12)
    assert math.isclose(result["value_after"], expected_v_eq17, rel_tol=1e-9)
    assert math.isclose(tracker.baseline(), expected_v_eq17, rel_tol=1e-9)


def test_rho_clips_to_bounds():
    tracker = _tracker(d_half=0.06, rho_min=0.875, rho_max=0.96)
    seed = tracker.transformed_rewards([0.1, 0.2])
    tracker.update(seed, divergence=0.0, policy_adapter="a0", step=0)

    # A large policy divergence floors rho at rho_min.
    large_div = tracker.transformed_rewards([0.1, 0.2])
    result = tracker.update(large_div, divergence=10.0, policy_adapter="a1", step=1)
    assert result["rho"] == pytest.approx(tracker.rho_min)

    # A missing/unattainable KL measurement (inf) also floors rho at rho_min.
    missing_div = tracker.transformed_rewards([0.1, 0.2])
    result2 = tracker.update(missing_div, divergence=math.inf, policy_adapter="a2", step=2)
    assert result2["rho"] == pytest.approx(tracker.rho_min)

    # A near-zero divergence caps rho at rho_max.
    tiny_div = tracker.transformed_rewards([0.1, 0.2])
    result3 = tracker.update(tiny_div, divergence=1e-9, policy_adapter="a3", step=3)
    assert result3["rho"] == pytest.approx(tracker.rho_max)


def test_state_dict_round_trip_and_mismatch_rejection():
    tracker = _tracker()
    rewards = tracker.transformed_rewards([0.1, -0.2, 0.3])
    tracker.update(rewards, divergence=0.0, policy_adapter="a0", step=0)

    restored = _tracker()
    restored.load_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()

    mismatched = _tracker(entropic_beta=3.0)
    with pytest.raises(ValueError):
        mismatched.load_state_dict(tracker.state_dict())


def test_consecutive_policy_divergence_matches_eq12_equal_context_weighting():
    # Context 0 has two rollouts, each with per-token log-ratio 0.5 -> sum 1.0.
    # Context 1 has one rollout with identical current/previous logprobs -> 0.0.
    current_scores = [[0.5, 0.5], [1.5, 1.5], [2.0, 2.0]]
    previous_scores = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    context_ids = [0, 0, 1]

    divergence, missing, per_context = SPORSTracker.consecutive_policy_divergence(
        current_scores, previous_scores, context_ids)

    assert missing == 0
    assert per_context[0] == pytest.approx(1.0)
    assert per_context[1] == pytest.approx(0.0)
    # Eq. 12 weights contexts equally (mean of 1.0 and 0.0 = 0.5), not by
    # rollout count (which would give (1.0+1.0+0.0)/3 = 0.667).
    assert divergence == pytest.approx(0.5)
