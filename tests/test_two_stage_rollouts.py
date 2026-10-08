from problems.base import ParentContext, Problem, RewardResult, SeedState
from problems.erdos import ErdosMinOverlap
from train_multy_CVaR import (
    _STRATEGY_FALLBACK,
    _apply_qwen3_8b_thinking_sampling,
    _extract_final_strategy,
)
from rollout_allocation import allocate_posterior_expected_best


class _DummyProblem(Problem):
    def build_prompt(self, parent):
        raise NotImplementedError

    def preprocess(self, code, parent):
        raise NotImplementedError

    def score(self, output, stdout):
        raise NotImplementedError

    def seed_states(self):
        raise NotImplementedError


def _problem():
    return _DummyProblem({})


def _content(messages):
    return str(messages[-1]["content"])


def test_build_strategy_messages_forbids_code_and_requires_wrapper():
    messages = [{"role": "user", "content": "Solve the task."}]
    staged = _problem().build_strategy_messages(messages)
    content = _content(staged)
    assert "Do not write Python code or a code fence in this stage" in content
    assert "<strategy>...</strategy>" in content


def test_build_strategy_messages_chains_previous_strategies():
    messages = [{"role": "user", "content": "Solve the task."}]
    staged = _problem().build_strategy_messages(
        messages, previous_strategies=["do X first", "do Y instead"])
    content = _content(staged)
    assert "<previous_strategy_1>\ndo X first\n</previous_strategy_1>" in content
    assert "<previous_strategy_2>\ndo Y instead\n</previous_strategy_2>" in content
    assert "materially different approach" in content


def test_build_strategy_messages_without_previous_has_no_history_block():
    messages = [{"role": "user", "content": "Solve the task."}]
    content = _content(_problem().build_strategy_messages(messages))
    assert "<previous_strategy_" not in content


def test_build_strategy_messages_appends_to_trailing_user_turn():
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "base prompt"}]
    staged = _problem().build_strategy_messages(messages)
    assert len(staged) == 2
    assert staged[0] == {"role": "system", "content": "sys"}
    assert staged[1]["content"].startswith("base prompt")


def test_build_strategy_messages_appends_new_turn_when_last_is_not_user():
    messages = [{"role": "user", "content": "base prompt"},
                {"role": "assistant", "content": "prior reply"}]
    staged = _problem().build_strategy_messages(messages)
    assert len(staged) == 3
    assert staged[-1]["role"] == "user"


def test_build_code_messages_embeds_strategy_and_requires_code_only():
    messages = [{"role": "user", "content": "Solve the task."}]
    staged = _problem().build_code_messages(messages, "use approach A")
    content = _content(staged)
    assert "<strategy>\nuse approach A\n</strategy>" in content
    assert "```python" in content
    assert "Do not output analysis, reasoning, a strategy" in content


def test_erdos_direct_prompt_and_strategy_coder_share_compact_contract():
    problem = ErdosMinOverlap({"budget_s": 60})
    parent = ParentContext()

    problem.two_stage_rollouts = True
    strategist_base = _content(problem.build_prompt(parent))
    assert "## Mandatory compact-source contract" not in strategist_base
    staged = _content(problem.build_code_messages(
        [{"role": "user", "content": strategist_base}], "detailed plan"))

    problem.two_stage_rollouts = False
    direct = _content(problem.build_prompt(parent))
    for required in (
            "## Mandatory compact-source contract",
            "A list/tuple/array literal may contain at most 32 scalar",
            "at most 300 lines and about 12,000"):
        assert required in direct
        assert required in staged
    assert "Return exactly one complete fenced Python code block" in direct
    assert "Return only exactly one fenced Python code block" in staged


def test_extract_final_strategy_selects_last_complete_block():
    raw = (
        "analysis " + ("draft one padding text " * 6)
        + "<strategy>" + "first draft padding text " * 6 + "</strategy>"
        + " more thinking "
        + "assistantfinal<strategy>"
        + "real final strategy padding text " * 6
        + "</strategy>"
    )
    body, reason = _extract_final_strategy(raw)
    assert reason is None
    assert "real final strategy" in body
    assert "first draft" not in body


def test_extract_final_strategy_prefers_final_channel_over_analysis_placeholder():
    # Mirrors the documented GPT-OSS failure mode: the analysis channel
    # quotes the output instruction, including a dummy <strategy> pair.
    raw = (
        "analysis The instructions say: Return only the strategy between "
        "<strategy> and </strategy> tags. " + ("padding " * 5)
        + "assistantfinal<strategy>"
        + "the actual detailed strategy text goes here and is long enough "
        "to clear the minimum length threshold required for extraction "
        + "</strategy>"
    )
    body, reason = _extract_final_strategy(raw)
    assert reason is None
    assert "actual detailed strategy" in body


def test_extract_final_strategy_rejects_short_block():
    raw = "assistantfinal<strategy>too short</strategy>"
    body, reason = _extract_final_strategy(raw)
    assert body == _STRATEGY_FALLBACK
    assert reason == "missing complete final strategy"


def test_extract_final_strategy_empty_response_returns_fallback():
    body, reason = _extract_final_strategy("")
    assert body == _STRATEGY_FALLBACK
    assert reason == "empty response"


def test_extract_final_strategy_recovers_unclosed_final_tag():
    # Mirrors the real step15_group00_fold00_strategy03.txt pattern: a real,
    # long final strategy that uses a bare ``` fence for a math formula and
    # simply ends -- generation stopped -- without ever closing </strategy>.
    tail = (
        "**Goal** - minimise the objective.\n\n"
        "```\nC5(h) = max_k sum_i h[i] * (1 - h[i+k]) * dx\n```\n\n"
        + "Further detailed steps of the plan follow here. " * 4
    )
    raw = "analysis thinking about it " + "assistantfinal<strategy>\n" + tail
    body, reason = _extract_final_strategy(raw)
    assert reason is None
    assert body == tail.strip()


def test_extract_final_strategy_rejects_unclosed_tag_containing_real_code():
    tail = (
        "Here is the plan. " * 3
        + "```python\ndef run():\n    return 1\n"
    )
    raw = "analysis thinking " + "assistantfinal<strategy>\n" + tail
    body, reason = _extract_final_strategy(raw)
    assert body == _STRATEGY_FALLBACK
    assert reason == "strategy contained a code fence"


def test_extract_final_strategy_bare_fence_is_not_treated_as_code():
    raw = (
        "assistantfinal<strategy>"
        + "A strategy that shows a formula in a plain fence. " * 3
        + "\n```\nO[k] = sum h[i]\n```\n"
        + "More explanation follows to pad this out past the minimum length."
        + "</strategy>"
    )
    body, reason = _extract_final_strategy(raw)
    assert reason is None
    assert "O[k] = sum h[i]" in body


def test_extract_final_strategy_falls_back_to_earlier_clean_block():
    # The second block is well over the 80-char minimum on its own, so
    # skipping it can only be explained by the code check, not by length --
    # isolating that the code-triggered skip (not just the length check)
    # keeps searching backward for an earlier clean block.
    code_block = (
        "Here is a plan that unfortunately embeds actual code: ```python\n"
        "def run():\n    return compute_something(1, 2, 3)\n"
        "``` which should be rejected for containing real code."
    )
    assert len(code_block) >= 80
    raw = (
        "assistantfinal"
        + "<strategy>" + "first clean strategy padding text here " * 4 + "</strategy>"
        + " then it reconsiders and writes "
        + f"<strategy>{code_block}</strategy>"
    )
    body, reason = _extract_final_strategy(raw)
    assert reason is None
    assert "first clean strategy" in body


def test_extract_final_strategy_never_recovers_raw_analysis_text():
    # The model never reaches a final answer at all -- there is no strategy
    # to recover by any means. The coder must get the strategist's final
    # strategy, never its private thinking process, even when that thinking
    # process contains substantial real, on-topic reasoning.
    raw = (
        "analysis " + (
            "We think the best approach is to use a subgradient descent "
            "method combined with projection onto the feasible set. "
        ) * 5
    )
    body, reason = _extract_final_strategy(raw)
    assert body == _STRATEGY_FALLBACK
    assert reason == "missing complete final strategy"


def test_extract_final_strategy_last_resort_rejects_bare_fence_scratch_code_in_final_channel():
    # Reaches the final channel (so the last-resort tier is reachable), but
    # fails the earlier no-wrapper-candidate check because of the stray,
    # unclosed "<strategy" mention -- isolating the last-resort tier's own
    # stricter any-fence check, rather than the (now unreachable) no-final-
    # marker path.
    raw = (
        "assistantfinal"
        "note: remember the <strategy tag format next time.\n"
        "```\ndef helper(x):\n    return x + 1\n```\n"
        + "More rambling thoughts follow to pad this out well past eighty "
        "characters total. " * 2
    )
    body, reason = _extract_final_strategy(raw)
    assert body == _STRATEGY_FALLBACK
    assert reason == "strategy contained a code fence"


def test_extract_final_strategy_last_resort_caps_to_tail_within_final_channel():
    # Same stray-tag-mention trick to reach the last-resort tier, but with
    # fence-free filler so the length cap -- not the code check -- is what's
    # under test, now scoped to final-channel content only.
    head_marker = "UNIQUE_HEAD_MARKER_SHOULD_BE_DROPPED"
    tail_marker = "UNIQUE_TAIL_MARKER_SHOULD_SURVIVE"
    raw = (
        "assistantfinal"
        "note: remember the <strategy tag format next time.\n"
        + head_marker + " "
        + "filler reasoning text that just keeps going on and on. " * 200
        + tail_marker
    )
    body, reason = _extract_final_strategy(raw)
    assert reason == "used unformatted reasoning as a last resort"
    assert tail_marker in body
    assert head_marker not in body


def test_bandit_allocation_is_fixed_budget_and_deterministic():
    rewards = [
        [0.0, 2.0, 2.1, 0.0, 2.2, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [1.8, 1.9, 1.85, 1.95, 1.9, 1.88],
    ]
    valid = [
        [False, True, True, False, True, False],
        [False, False, True, False, False, False],
        [True, True, True, True, True, True],
    ]
    first = allocate_posterior_expected_best(
        rewards, valid, 27, fail_reward=0.0, seed=1234)
    second = allocate_posterior_expected_best(
        rewards, valid, 27, fail_reward=0.0, seed=1234)

    assert first == second
    assert sum(first[0]) == 27
    assert all(count >= 0 for count in first[0])
    assert first[2]["method"] == "posterior-expected-best"


def test_bandit_all_invalid_pilots_preserve_symmetry():
    allocations, diagnostics, summary = allocate_posterior_expected_best(
        [[0.0] * 6 for _ in range(4)],
        [[False] * 6 for _ in range(4)],
        14,
        fail_reward=0.0,
        seed=5,
    )

    assert allocations == [4, 4, 3, 3]
    assert sum(allocations) == 14
    assert summary["fallback_equal"] is True
    assert all(item["valid_count"] == 0 for item in diagnostics)


def test_bandit_productive_arm_beats_all_zero_arm():
    allocations, diagnostics, _summary = allocate_posterior_expected_best(
        [
            [2.0, 2.1, 2.2, 2.15, 2.25, 2.18, 2.3, 2.22],
            [0.0] * 8,
        ],
        [
            [True] * 8,
            [False] * 8,
        ],
        24,
        fail_reward=0.0,
        seed=99,
    )

    assert sum(allocations) == 24
    assert allocations[0] > allocations[1]
    assert diagnostics[0]["posterior_valid_probability"] > diagnostics[1][
        "posterior_valid_probability"]


def test_qwen_sampling_profile_does_not_override_explicit_coder_yaml():
    config = {
        "model_name": "Qwen/Qwen3-8B",
        "thinking": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "strategies": True,
        "strategy_model_name": "Qwen/Qwen3-8B",
        "strategy_thinking": True,
        "strategy_temperature": 1.0,
        "strategy_top_p": 1.0,
        "strategy_top_k": 0,
        "strategy_min_p": 0.0,
    }
    _apply_qwen3_8b_thinking_sampling(
        config, {
            "temperature", "top_p", "top_k", "min_p",
            "strategy_temperature", "strategy_top_p", "strategy_top_k",
            "strategy_min_p",
        })

    assert config["temperature"] == 1.0
    assert config["top_p"] == 1.0
    assert config["sampling_top_k"] == 0
    assert config["sampling_min_p"] == 0.0
    assert config["qwen3_8b_thinking_sampling"] is False
    assert config["strategy_temperature"] == 1.0
    assert config["strategy_top_p"] == 1.0
    assert config["strategy_sampling_top_k"] == 0
    assert config["strategy_sampling_min_p"] == 0.0
    assert config["strategy_qwen3_8b_thinking_sampling"] is False


def test_qwen_coder_sampling_keeps_yaml_values_and_disables_filters():
    config = {
        "model_name": "Qwen/Qwen3-8B",
        "thinking": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "strategies": False,
    }
    _apply_qwen3_8b_thinking_sampling(config, frozenset())

    assert config["temperature"] == 1.0
    assert config["top_p"] == 1.0
    assert config["sampling_top_k"] == 0
    assert config["sampling_min_p"] == 0.0
    assert config["qwen3_8b_thinking_sampling"] is False
