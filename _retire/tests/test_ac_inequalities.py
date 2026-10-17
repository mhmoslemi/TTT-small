"""AC1/AC2 evaluator, reward, and two-stage prompt contracts."""

import math

import pytest

from problems.ac_inequalities import ACInequalities
from problems.base import ParentContext


def make_problem(problem_type):
    return ACInequalities({
        "problem_type": problem_type,
        "target": 1.5 if problem_type == "ac1" else 0.96,
        "fail_score": 0.0,
        "budget_s": 1000,
        "eval_cpus": 2,
        "num_seed_states": 1,
        "seed": 42,
    })


def test_ac1_reward_is_the_exact_reciprocal_without_epsilon():
    problem = make_problem("ac1")
    result = problem.score([1.0, 1.0], "")
    assert result.valid
    assert result.raw_score == pytest.approx(2.0)
    assert result.reward == pytest.approx(0.5)
    seed = problem.seed_states()[0]
    assert seed.value == pytest.approx(1.0 / seed.raw_score)


def test_ac2_score_is_finite_and_uses_the_authoritative_evaluator():
    result = make_problem("ac2").score([1.0], "")
    assert result.valid
    assert result.raw_score == pytest.approx(2.0 / 3.0)
    assert result.reward == pytest.approx(result.raw_score)


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_invalid_scores_are_not_marked_valid(problem_type):
    problem = make_problem(problem_type)
    for output in ([], [0.0], [math.inf], [True]):
        result = problem.score(output, "")
        assert not result.valid
        assert result.reward == problem.fail_score


@pytest.mark.parametrize("problem_type,entrypoint", [
    ("ac1", "propose_candidate"),
    ("ac2", "construct_function"),
])
def test_two_stage_prompts_preserve_exact_math_and_compact_handoff(
        problem_type, entrypoint):
    problem = make_problem(problem_type)
    parent = ParentContext(
        code="def previous():\n    pass",
        raw_score=1.2,
        construction=[0.125, 0.25, 0.5],
    )
    base = problem.build_prompt(parent)
    text = base[-1]["content"]
    assert "Authoritative evaluator" in text
    assert f"Define `{entrypoint}(" in text
    assert "height_sequence_1" in text
    assert "[0.125, 0.25, 0.5]" not in text
    assert problem.saves_construction

    strategy_text = problem.build_strategy_messages(base)[-1]["content"]
    assert "full *linear* self-convolution" in strategy_text
    assert "normal, readable English prose" in strategy_text
    assert "Never form a\n  dense n-by-n identity" in strategy_text
    assert "<strategy>" in strategy_text
    if problem_type == "ac1":
        assert "2*n*max(convolve" in strategy_text
    else:
        assert "L2sq/(L1*Linf)" in strategy_text

    coder_text = problem.build_code_messages(base, "Detailed plan")[-1]["content"]
    assert "Treat the strategy as planning guidance" in coder_text
    assert "Mandatory compact-source and return contract" in coder_text
    assert "at most 32 scalar" in coder_text
    assert "Do not spend the\n  response re-deriving" in coder_text
    assert problem.coder_retry_uses_failed_response is True

    problem.two_stage_rollouts = False
    direct_text = problem.build_prompt(parent)[-1]["content"]
    assert "Return exactly one complete fenced Python code block" in direct_text


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_ac_strategy_integrity_gate_rejects_decoding_and_collapse(problem_type):
    problem = make_problem(problem_type)
    assert problem.strategy_output_issue(
        "A detailed readable plan with formulas, dimensions, and validation.") is None
    assert "tokenizer-decoding" in problem.strategy_output_issue(
        "planĠwithĊbyteĠmarkers")
    assert "tokenizer-decoding" in problem.strategy_output_issue(
        "otherwise readable plan with one damaged character \ufffd")
    assert "collapsed" in problem.strategy_output_issue("x" * 300)
    assert "backend error" in problem.strategy_output_issue(
        "Detailed plan followed by ErrorMessage: ChatCoder failed")


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_ac_strategy_normalizer_repairs_byte_surface_locally(problem_type):
    problem = make_problem(problem_type)
    surface = "<strategy>HelloĠworldĊAĠæ¢B</strategy>"
    repaired = problem.normalize_strategy_response(surface)
    assert repaired.startswith("<strategy>Hello world\nA ")
    assert repaired.endswith("B</strategy>")
    assert "Ġ" not in repaired and "Ċ" not in repaired
    assert "\ufffd" in repaired

