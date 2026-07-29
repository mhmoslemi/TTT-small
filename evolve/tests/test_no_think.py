"""
Making enable_thinking=False actually stick.

apply_chat_template forwards unknown kwargs into the Jinja context rather than
raising, so a template that never references enable_thinking ignores it in
silence -- and the model opens <think> anyway. That silence is how the bug
survived a round of "fixes".
"""
import pytest

from config import Config
from llm.backbone import Backbone


class _Tok:
    """A template that IGNORES enable_thinking, like the one that broke this."""
    chat_template = "yes"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        return "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, t):
        return 0


class _HonouringTok(_Tok):
    """A template that DOES honour it, emitting a closed empty block."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True,
                            **kwargs):
        tail = "" if enable_thinking else "<think>\n\n</think>\n\n"
        return f"<|im_start|>assistant\n{tail}"


def _backbone(tok, **over):
    cfg = Config().model
    cfg.enable_thinking = False      # these tests are about suppressing it
    for k, v in over.items():
        setattr(cfg, k, v)

    class Stub:
        name = "stub"
        def load(self): return object(), tok

    return Backbone(cfg, backend=Stub()).load()


MSG = [{"role": "user", "content": "hi"}]


def test_a_template_that_ignores_the_kwarg_gets_a_prefilled_block():
    out = _backbone(_Tok()).render(MSG)
    assert out.endswith("<think>\n\n</think>\n\n")

def test_a_template_that_honours_the_kwarg_is_left_alone():
    out = _backbone(_HonouringTok()).render(MSG)
    assert out.count("</think>") == 1        # not double-prefilled

def test_thinking_on_means_no_prefill():
    out = _backbone(_Tok(), enable_thinking=True).render(MSG)
    assert "</think>" not in out

def test_the_guard_can_be_switched_off():
    out = _backbone(_Tok(), force_no_think=False).render(MSG)
    assert "</think>" not in out

def test_an_empty_prefill_is_a_no_op():
    out = _backbone(_Tok(), no_think_prefill="").render(MSG)
    assert "</think>" not in out

def test_what_happened_is_reported_once(capsys):
    backbone = _backbone(_Tok())
    backbone.render(MSG)
    backbone.render(MSG)
    out = capsys.readouterr().out
    assert "prefilling" in out
    assert out.count("[backbone]") == 1      # said once, not per rollout

def test_no_chat_template_still_renders():
    class Bare:
        chat_template = None
        pad_token_id = 0
        def convert_tokens_to_ids(self, t): return 0

    assert "hi" in _backbone(Bare()).render(MSG)


# ---------------- the timeout is surfaced to the model ----------------
def _example(timeout=100.0):
    from examples.circle_packing.env import build
    cfg = Config()
    cfg.example.params = {"num_circles": 26, "entrypoint": "run_packing"}
    cfg.verifier.timeout_s = timeout
    return build(cfg)


def test_the_prompt_says_nothing_about_the_timeout():
    """Kept out of the prompt to stay faithful to the reference wording. The
    guidance lives in the verifier FEEDBACK instead, where it only appears
    once a timeout has actually happened."""
    assert "seconds" not in _example(100.0).instruction()

def test_timeout_feedback_says_what_to_do_differently():
    example = _example(timeout=2.0)          # must actually exceed the limit
    result = example.verify(
        "```python\nimport time\ndef run_packing():\n    time.sleep(30)\n```", None)
    assert not result.valid
    assert "Timeout" in result.msg
    assert "BOUNDED local refinement" in result.feedback
    assert "vectorize" in result.feedback.lower()
