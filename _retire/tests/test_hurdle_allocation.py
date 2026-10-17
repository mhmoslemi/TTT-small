"""Allocation invariants and independent checks of the hurdle prediction math."""

import ast
from concurrent.futures import Future
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rollout_allocation import (
    _dirichlet_cdf_moments,
    _hurdle_bandwidth,
    _hurdle_kernel_cdfs,
    allocate_hurdle_expected_best,
)


def allocate(rewards, total=24, parent=1.0, valid=None):
    if valid is None:
        valid = [[True] * len(arm) for arm in rewards]
    return allocate_hurdle_expected_best(
        rewards, valid, total, parent_reward=parent)


@pytest.mark.parametrize("arms,budget", [(1,1), (3,17), (4,24), (7,43)])
def test_hurdle_preserves_budget_and_symmetry(arms, budget):
    result = allocate([[0.0, 1.0, 1.2, 1.4]] * arms, total=budget)
    allocations, diagnostics, summary = result
    assert sum(allocations) == budget
    assert all(isinstance(n, int) and n >= 0 for n in allocations)
    assert max(allocations) - min(allocations) <= 1
    assert result == allocate([[0.0, 1.0, 1.2, 1.4]] * arms, total=budget)
    assert summary["projected_expected_gain"] > 0
    assert all(d["improvement_count"] == 2 for d in diagnostics)


def test_hurdle_no_positive_pilots_do_not_invent_a_scale():
    counts, diagnostics, summary = allocate(
        [[1.0]*4, [-10, 0, .5, .9], [0]*4], total=14)
    assert counts == [5, 5, 4]
    assert summary["fallback_equal"]
    assert not summary["gain_estimate_available"]
    assert summary["incumbent_reward"] == 1.0
    assert all(d["improvement_count"] == 0 for d in diagnostics)


def test_hurdle_failure_severity_cannot_inflate_upside():
    rewards = [[1.01, 1.02, 1.0, .9, 0], [1.03, 1.04, 1.0, .99, .98]]
    first = allocate(rewards)
    rewards[0][-2:] = [-1000, -1e6]
    second = allocate(rewards)
    assert first[0] == second[0]
    assert first[2] == second[2]
    for d1, d2 in zip(first[1], second[1]):
        for key in ("initial_expected_improvement", "final_marginal_expected_improvement",
                    "posterior_improvement_probability", "mean_positive_improvement"):
            assert d1[key] == d2[key]


def test_invalid_high_rewards_are_never_counted_as_improvements():
    result = allocate([[1.2, 1e10], [1.1, 1]], valid=[[True, False], [True, True]])
    assert result[1][0]["improvement_count"] == 1
    assert result[2]["incumbent_reward"] == 1.2
    assert result[0] == allocate([[1.2, 0], [1.1, 1]],
                                valid=[[True, False], [True, True]])[0]


def test_unobserved_arm_retains_uncertainty_but_does_not_win_by_failing():
    counts, diag, _ = allocate([[1.1,1.2,1.3,1.4], [0]*4])
    assert counts[0] > counts[1]
    assert 0 < diag[1]["posterior_improvement_probability"] < diag[0]["posterior_improvement_probability"]
    assert 0 < diag[1]["initial_expected_improvement"] < diag[0]["initial_expected_improvement"]


def test_tiny_improvements_and_reward_scale_translation():
    rewards = [[1, 1.01, 1.02, .8], [0, 1.005, 1, .4]]
    counts, _, summary = allocate(rewards)
    factor, offset = 1e-7, -3e-6
    scaled = [[factor*r+offset for r in arm] for arm in rewards]
    counts2, diag2, summary2 = allocate(scaled, parent=factor+offset)
    assert counts2 == counts
    assert diag2[0]["improvement_count"] == 2
    assert summary2["projected_expected_gain"] == pytest.approx(
        factor*summary["projected_expected_gain"], rel=1e-8)


@pytest.mark.parametrize("rewards,valid,total,parent", [
    ([[1]], [[True]], -1, 0), ([], [], 1, 0),
    ([[1]], [], 1, 0), ([[]], [[]], 1, 0),
    ([[1,2]], [[True]], 1, 0), ([[math.nan]], [[True]], 1, 0),
    ([[1]], [[True]], 1, math.inf),
])
def test_hurdle_rejects_invalid_input(rewards, valid, total, parent):
    with pytest.raises(ValueError):
        allocate_hurdle_expected_best(rewards, valid, total, parent_reward=parent)


def test_zero_budget_and_no_arms():
    assert allocate([[1.1], [0]], total=0)[0] == [0,0]
    assert allocate([], total=0)[0] == []


def test_dirichlet_cdf_moments_against_beta_closed_form():
    # F=W_1 with W_1~Beta(3,1): E[F^m]=3/(3+m).
    stream = _dirichlet_cdf_moments(np.array([[0.0], [1.0]]), [1,3])
    for m in range(45):
        assert next(stream)[0] == pytest.approx(3/(3+m), rel=1e-12)


def test_dirichlet_moments_keep_same_arm_parameter_uncertainty():
    cdfs = np.array([[.2, .4], [.7, .9], [1., 1.]])
    alpha = np.array([1,2,4])
    stream = _dirichlet_cdf_moments(cdfs, alpha)
    next(stream)
    mean = next(stream)
    second = next(stream)
    expected_mean = alpha @ cdfs / alpha.sum()
    expected_var = (alpha @ cdfs**2 / alpha.sum() - expected_mean**2) / (alpha.sum()+1)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(second, expected_mean**2 + expected_var)
    assert np.all(second > mean**2)


def test_expected_best_matches_independent_predictive_simulation():
    # One arm removes allocation uncertainty; simulate actual draws independently
    # of the CDF/moment integrator to check the entire expected-best calculation.
    rewards = [0, 1, 1.1, 1.2, 1.4]
    budget = 7
    _, _, summary = allocate([rewards], total=budget)
    positives = np.array([.1,.2,.4])
    h = _hurdle_bandwidth(positives / positives.max()) * positives.max()
    rng = np.random.default_rng(983)
    draws = 200_000
    weights = rng.dirichlet([3,1,1,1,1], size=draws)
    cumulative = np.cumsum(weights, axis=1)
    best = np.full(draws, .4)
    for _ in range(budget):
        component = (rng.random(draws)[:,None] > cumulative).sum(axis=1)
        center = np.zeros(draws)
        for i, value in enumerate(positives, 2):
            center[component == i] = value
        prior = component == 1
        center[prior] = rng.choice(positives, size=int(prior.sum()))
        z = np.abs(center + h*rng.standard_normal(draws))
        z[component == 0] = 0
        best = np.maximum(best, z)
    realized = best - .4
    se = realized.std() / np.sqrt(draws)
    assert abs(summary["projected_expected_gain"] - realized.mean()) < 5*se


def test_logged_step1_parent2_dominant_arm_is_no_longer_starved():
    # Equal-size observed pilots from the audited run, not phase-2 outcomes.
    parent = 2.624372804170494
    dominant = [0, parent, parent, 2.624372805062, 2.624375854524,
                2.624376276324, 2.624385111046, 2.624386952123,
                2.624437217516, 2.624468648306, 2.624927759387]
    dominated = [0]*4 + [2.611613081151] + [parent]*4 + [2.624447900115, 2.624812415075]
    counts, diag, _ = allocate([dominant, dominated], parent=parent,
                              valid=[[x>0 for x in dominant], [x>0 for x in dominated]])
    assert diag[0]["initial_expected_improvement"] > diag[1]["initial_expected_improvement"]
    assert counts[0] > counts[1]


def test_positive_kernel_is_stochastically_monotone():
    cdfs = _hurdle_kernel_cdfs([0.01, .1, .5], np.linspace(0, 2, 200), .2)
    assert np.all(cdfs[0] >= cdfs[1]-1e-15)
    assert np.all(cdfs[1] >= cdfs[2]-1e-15)


def _runner_helpers(namespace):
    """Execute the actual nested orchestration with CPU-only finished futures."""
    path = Path(__file__).resolve().parents[1] / "train_multy_CVaR.py"
    tree = ast.parse(path.read_text())
    wanted = {"_finish_strategy_pilots", "_print_adaptive_phase2_results",
              "_strategy_plan_metadata", "_attach_strategy_plan",
              "_allocate_largest_remainder"}
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(nodes) == len(wanted)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("method", ["hurdle", "bandit", "rule_based"])
def test_finished_pilots_to_phase2_logs_and_metadata(method, capsys):
    jobs = [{"parent_group":0, "strategy_index":i} for i in range(2)]
    records = []
    for arm, rewards in enumerate([[0, 1.1], [1.2, 1.3]]):
        for reward in rewards:
            future = Future()
            future.set_result(SimpleNamespace(reward=reward, valid=reward>0,
                                             raw_score=reward or None, stdout=""))
            records.append({"strategy_source_job_idx":arm, "_reward_future":future})
    namespace = dict(np=np, math=math, prompt_jobs=jobs, parents=[SimpleNamespace(value=1.0)],
                     cfg=SimpleNamespace(seed=42, fail_score=0.0, group_size=8),
                     problem=SimpleNamespace(maximize=True, metric_name="score"),
                     step_idx=0, pilot_programs_per_strategy=2, strategies_per_parent=2,
                     programs_per_strategy=4, phase2_allocation_method=method,
                     strategy_pilot_diagnostics=[], _ANSI_ORANGE="", _ANSI_YELLOW="", _ANSI_RESET="",
                     group_responses={0:records})
    _runner_helpers(namespace)
    counts = namespace["_finish_strategy_pilots"](records, [0,1], rewards_already_ready=True)
    assert sum(counts) == 4
    if method == "hurdle":
        assert records[0]["strategy_hurdle"]["parent_reward"] == 1.0
        assert records[0]["strategy_hurdle"]["improvement_count"] == 1
    namespace["_print_adaptive_phase2_results"]()
    output = capsys.readouterr().out
    assert "rollout phase 2 complete" in output
    assert "best raw after new rollouts" in output
    if method == "hurdle":
        assert "posterior-improve=" in output
        assert "improved=1/2" in output


@pytest.mark.parametrize("method", ["hurdle", "bandit", "rule_based"])
def test_config_accepts_each_allocation_method(method):
    from config_validation import validate_problem_config
    validate_problem_config({"problem":"erdos", "phase2_allocation_method":method},
                            require_complete=False)


@pytest.mark.parametrize("saved_method", [None, "rule_based", "bandit", "hurdle"])
def test_new_preset_does_not_override_saved_allocation(saved_method, tmp_path, monkeypatch):
    import yaml
    from train_multy_CVaR import load_config
    root = Path(__file__).resolve().parents[1]
    saved = yaml.safe_load((root / "configs/erdos.yaml").read_text())
    if saved_method is None:
        saved.pop("phase2_allocation_method", None)
    else:
        saved["phase2_allocation_method"] = saved_method
    run = tmp_path / "resume"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(saved))
    monkeypatch.setattr(sys, "argv", ["train_multy_CVaR.py", "--resume", str(run),
                                     "--strategies"])
    cfg, _ = load_config()
    assert cfg.phase2_allocation_method == (saved_method or "rule_based")
