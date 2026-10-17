import ast
from concurrent.futures import Future
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spo_rs import (BOUNDED_SHAPING_VERSION, BoundedSPORSTracker,
                    SPORSTracker, validate_shaping_resume)


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


ROOT = Path(__file__).resolve().parents[1]


def tracker(**options):
    return BoundedSPORSTracker(**{
        "entropic_beta": 2.0, "d_half": 0.06,
        "rho_min": 0.875, "rho_max": 0.96, **options,
    })


def initialize(t, seeds=(1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4)):
    t.update(t.transformed_rewards(seeds), raw_rewards=seeds,
             divergence=0.0, policy_adapter="adapter_initial", step=-1)
    t.prepare_step()
    return t


def quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), q * weights.sum())
    return values[order[min(index, len(values) - 1)]]


def test_bounded_transform_and_advantage_match_proposal():
    t = initialize(tracker())
    info = t.shaping_info()
    assert info["shaping_center"] == 2.4
    assert info["shaping_scale"] == pytest.approx(0.2)
    rewards = np.array([0, 2.2, 2.4, 2.400001, 2.6, 3.0])
    u = 0.5 + np.arctan((rewards - 2.4) / info["shaping_scale"]) / math.pi
    g = np.exp(2.5 * u)
    np.testing.assert_allclose(t.transformed_rewards(rewards), g, rtol=1e-14)
    a = t.normalized_advantages(g)
    np.testing.assert_allclose(a, (g / t.baseline() - 1) / 2)
    assert np.all(np.diff(a) > 0)
    assert np.all(a >= math.expm1(-2.5) / 2)
    assert np.all(a <= math.expm1(2.5) / 2)


def test_history_replay_uses_same_transform_and_all_rewards():
    seeds = np.array([1, 1.5, 1.8, 2, 2.1, 2.2, 2.3, 2.4])
    t = initialize(tracker(), seeds)
    history, weights = seeds.copy(), np.full(len(seeds), 8.0)
    for step in range(6):
        t.prepare_step()
        valid = history > 0
        b = quantile(history[valid], weights[valid], 0.9)
        s = b - quantile(history[valid], weights[valid], 0.75)
        assert s > 0
        transform = lambda r: np.exp(2.5 * (
            0.5 + np.arctan((r - b) / s) / math.pi))
        expected_v = np.dot(weights, transform(history)) / weights.sum()
        assert t.baseline() == pytest.approx(expected_v, rel=1e-13)
        rewards = np.array([0.0, 0.0, 2.1 + step * 0.1, 2.6 + step * 0.2])
        a = t.normalized_advantages(t.transformed_rewards(rewards))
        np.testing.assert_allclose(a, (transform(rewards) / expected_v - 1) / 2)
        update = t.update(t.transformed_rewards(rewards), raw_rewards=rewards,
                          divergence=0.02, policy_adapter=f"a{step}", step=step)
        weights = np.r_[weights * update["rho"], np.ones(len(rewards))]
        history = np.r_[history, rewards]
        assert update["group_size"] == len(rewards)
        assert update["effective_count_after"] == pytest.approx(weights.sum())
        assert update["value_after"] == pytest.approx(
            np.dot(weights, transform(history)) / weights.sum(), rel=1e-13)
        assert len(t.state_dict()["history_rewards"]) == len(history)


def test_step_baseline_is_global_frozen_not_parent_centered():
    t = initialize(tracker())
    before = t.state_dict()
    groups = [np.array([0.0, 2.5]), np.array([2.7, 2.8, 2.9])]
    a = [t.normalized_advantages(t.transformed_rewards(r)) for r in groups]
    combined = t.normalized_advantages(t.transformed_rewards(np.concatenate(groups)))
    np.testing.assert_array_equal(np.concatenate(a), combined)
    assert not np.isclose(a[1].sum(), 0.0)  # No parent-wise recentering.
    assert t.state_dict() == before


@pytest.mark.parametrize("seeds", [[0, 0], [2, 2], [-1, -1], [1]])
def test_tied_initial_history_and_all_failures_remain_finite(seeds):
    t = initialize(tracker(), seeds)
    for step in range(2):
        r = np.array([0.0, 0.0, 1.0, 2.0])
        g = t.transformed_rewards(r)
        assert np.isfinite(t.normalized_advantages(g)).all()
        t.update(g, raw_rewards=r, divergence=0, policy_adapter="a", step=step)
        t.prepare_step()
        assert t.shaping_info()["shaping_scale"] > 0


def test_no_fixed_scale_floor_erases_small_improvements():
    seeds = np.linspace(2.0, 2.0 + 1e-10, 20)
    t = initialize(tracker(), seeds)
    assert 0 < t.shaping_info()["shaping_scale"] < 1e-10
    r = np.array([seeds[-1], seeds[-1] + 1e-11])
    a = t.normalized_advantages(t.transformed_rewards(r))
    assert a[1] > a[0] + 0.01


def test_checkpoint_roundtrip_and_preview_do_not_double_update_history():
    t = initialize(tracker())
    for step in range(3):
        t.prepare_step()
        restored = tracker()
        restored.load_state_dict(json.loads(json.dumps(t.state_dict())))
        restored.prepare_step()
        before = t.state_dict()
        r = [0.0, 2.5 + step * 0.1, 2.6 + step * 0.2]
        g = t.transformed_rewards(r)
        np.testing.assert_array_equal(restored.transformed_rewards(r), g)
        preview = deepcopy(t)
        expected = preview.update(g, raw_rewards=r, divergence=0.04,
                                  policy_adapter=f"a{step}", step=step)
        assert t.state_dict() == before
        actual = restored.update(g, raw_rewards=r, divergence=0.04,
                                 policy_adapter=f"a{step}", step=step)
        assert actual == expected
        assert restored.state_dict() == preview.state_dict()
        t = restored


def test_corrupt_or_wrong_objective_checkpoint_is_rejected():
    t = initialize(tracker())
    state = t.state_dict()
    for field, value in [("value", 999), ("shaping_width", 1),
                         ("history_weights", [1]), ("shaping_scale", 0)]:
        bad = deepcopy(state)
        bad[field] = value
        with pytest.raises(ValueError):
            tracker().load_state_dict(bad)
    with pytest.raises(ValueError):
        SPORSTracker(entropic_beta=2, d_half=.06, rho_min=.875,
                     rho_max=.96).load_state_dict(state)
    with pytest.raises(ValueError):
        t.update(t.transformed_rewards([1]), divergence=0,
                 policy_adapter="a", step=0)


def test_old_adaptive_runs_are_not_silently_reinterpreted():
    old = {"advantage_mode": "spo-rs", "spo_rs_adaptive_beta": True}
    with pytest.raises(ValueError, match="retired adaptive-beta"):
        validate_shaping_resume(old)
    validate_shaping_resume({**old, "spo_rs_adaptive_beta": False})
    validate_shaping_resume({**old, "spo_rs_reward_shaping": BOUNDED_SHAPING_VERSION})


def test_overlap_can_train_ready_examples_without_pending_rewards():
    # Load just the pure completion helper, not the training program.
    tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_complete_spo_rs_overlap_examples")
    scope = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "helper", "exec"), scope)
    complete = scope[helper.name]
    t = initialize(tracker())
    ready, pending = Future(), Future()
    ready.set_result(SimpleNamespace(reward=2.6))
    example = {"_overlap_reward_future": ready}
    expected = t.normalized_advantages(t.transformed_rewards([2.6]))[0]
    assert complete([example], t)[0]["advantage"] == expected
    assert not pending.done()
    with pytest.raises(RuntimeError, match="before its reward"):
        complete([{"_overlap_reward_future": pending}], t)


def test_both_modes_share_overlap_and_every_update_supplies_raw_rewards():
    tree = ast.parse((ROOT / "train_multy_CVaR.py").read_text())
    updates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "spo_rs_overlap_eligible"
                for t in node.targets):
            assert "adaptive_spo_beta" not in ast.unparse(node.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"spo_rs_tracker", "preview_tracker"}):
            updates.append(node)
            assert "raw_rewards" in {kw.arg for kw in node.keywords}
    assert len(updates) == 5
