from problems.base import ParentContext, Problem, RewardResult, SeedState
from train_multy_CVaR import _STRATEGY_FALLBACK, _extract_final_strategy


class _DummyProblem(Problem):
    def build_prompt(self, parent, memory="", memory_protocol=False):
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
