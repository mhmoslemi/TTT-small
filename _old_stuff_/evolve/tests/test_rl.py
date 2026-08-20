import numpy as np
import pytest

from rl.advantage import (clip_advantages, combine, feedback_advantages,
                          group_relative_advantages)


def test_advantages_sum_to_zero_for_any_beta():
    """omega = G * softmax, so sum(omega) = G and sum(A) = 0 -- the baseline
    falls out for free, with no std division to flatten the tilt."""
    r = [0.0, 1.0, 2.5, 2.5, 0.3]
    for beta in (0.01, 0.5, 1.0, 5.0, 50.0):
        assert group_relative_advantages(r, beta).sum() == pytest.approx(0.0, abs=1e-9)

def test_beta_zero_gives_exactly_zero_not_mean_seeking():
    """The paper calls beta=0 'mean-seeking'; it is actually no signal at all."""
    assert np.allclose(group_relative_advantages([0.0, 1.0, 9.0], 0.0), 0.0)

def test_small_beta_is_the_mean_seeking_regime():
    """A ~ beta (r - rbar) for small beta."""
    r = np.array([0.0, 1.0, 2.0]); beta = 1e-3
    got = group_relative_advantages(r, beta)
    assert np.allclose(got, beta * (r - r.mean()), atol=1e-6)

def test_large_beta_is_max_seeking():
    """Unique max -> G-1, everyone else -> -1."""
    r = [0.1, 0.2, 5.0]
    a = group_relative_advantages(r, 500.0)
    assert a[2] == pytest.approx(len(r) - 1, abs=1e-6)
    assert a[0] == pytest.approx(-1.0, abs=1e-6)
    assert a[1] == pytest.approx(-1.0, abs=1e-6)

def test_beta_monotonically_concentrates_on_the_best():
    r = [0.0, 1.0, 2.0]
    prev = -np.inf
    for beta in (0.1, 1.0, 5.0, 20.0):
        cur = group_relative_advantages(r, beta)[2]
        assert cur > prev
        prev = cur

def test_identical_rewards_give_no_signal():
    assert np.allclose(group_relative_advantages([1.5] * 4, 3.0), 0.0)

def test_singleton_and_empty_groups():
    assert group_relative_advantages([2.0], 1.0).tolist() == [0.0]
    assert group_relative_advantages([], 1.0).size == 0

def test_no_overflow_at_extreme_beta_and_reward():
    out = group_relative_advantages([1e6, 2e6], 1e3)
    assert np.isfinite(out).all()

def test_feedback_advantage_sign():
    """Positive where the feedback makes the sampled token more plausible."""
    fb = feedback_advantages([-1.0, -5.0], [-2.0, -1.0])
    assert fb[0] > 0 and fb[1] < 0

def test_feedback_advantage_shape_mismatch_raises():
    with pytest.raises(ValueError):
        feedback_advantages([-1.0], [-1.0, -2.0])

def test_combine_success_carries_only_the_reward_term():
    out = combine(0.7, np.array([5.0, -5.0]), failed=False, lambda_f=1.0, num_tokens=2)
    assert np.allclose(out, 0.7)

def test_combine_failure_adds_the_dense_term():
    out = combine(-1.0, np.array([2.0, -2.0]), failed=True, lambda_f=0.5, num_tokens=2)
    assert np.allclose(out, [-1.0 + 1.0, -1.0 - 1.0])

def test_combine_lambda_zero_disables_feedback():
    out = combine(-1.0, np.array([9.0, 9.0]), failed=True, lambda_f=0.0, num_tokens=2)
    assert np.allclose(out, -1.0)

def test_combine_rejects_length_mismatch():
    with pytest.raises(ValueError):
        combine(0.0, np.array([1.0]), failed=True, lambda_f=1.0, num_tokens=3)

def test_advantage_clip_bounds_the_unbounded_feedback_term():
    assert np.allclose(clip_advantages(np.array([-100.0, 100.0]), 10.0), [-10.0, 10.0])
    assert np.allclose(clip_advantages(np.array([-100.0]), 0), [-100.0])
