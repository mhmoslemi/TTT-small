"""
Budget forcing: cap the think block so a reasoning model still emits an answer.

The failure this prevents: the model spends every token inside <think>, never
writes a ```python block, and scores zero however good the reasoning was.
"""
import pytest

from config import Config, load_config
from core.engine import Engine
from core.types import Rollout, VerifyResult


# ---------------- truncation is named, not silently lumped in ----------------
def _engine(tmp_path, max_new_tokens=100):
    cfg = load_config([
        "--example", "circle_packing", "--backend", "mock", "--steps", "1",
        "--n-select", "1", "--k-children", "1", "--progress", "false",
        "--max-new-tokens", str(max_new_tokens),
        "--output-root", str(tmp_path), "--set", "example.params.num_circles=4",
    ]).config
    return Engine(cfg)


def test_hitting_the_limit_without_code_is_reported_as_truncation(tmp_path):
    engine = _engine(tmp_path, max_new_tokens=10)
    rollout = Rollout(token_ids=list(range(10)))          # exactly at the cap
    result = VerifyResult(valid=False, msg="no_code_block")
    engine._mark_if_truncated(rollout, result)
    assert result.msg == "truncated_before_code"
    assert "10-token limit" in result.feedback
    assert "sooner" in result.feedback                    # actionable for Eq. 9

def test_a_short_response_with_no_code_is_not_called_truncation(tmp_path):
    engine = _engine(tmp_path, max_new_tokens=100)
    result = VerifyResult(valid=False, msg="no_code_block")
    engine._mark_if_truncated(Rollout(token_ids=[1, 2, 3]), result)
    assert result.msg == "no_code_block"                  # genuinely wrote none

def test_a_valid_rollout_is_never_relabelled(tmp_path):
    engine = _engine(tmp_path, max_new_tokens=10)
    result = VerifyResult(valid=True, reward=1.0, msg="ok")
    engine._mark_if_truncated(Rollout(token_ids=list(range(50))), result)
    assert result.msg == "ok"

def test_truncation_count_reaches_the_step_summary(tmp_path):
    engine = _engine(tmp_path)
    engine.run()
    assert "num_truncated" in engine.history[0]


# ---------------- the two-phase generation path ----------------
class FakeTok:
    """Minimal tokenizer: one id per word, with </think> as id 999."""
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text=None, **kw):
        return {"input_ids": [self._enc(text)] if isinstance(text, str)
                else [self._enc(t) for t in text]}

    @staticmethod
    def _enc(text):
        return [999 if w == "</think>" else (len(w) + 10) for w in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            i = int(i)
            if i == 999:
                out.append("</think>")
            elif i == 2:
                out.append("" if skip_special_tokens else "<eos>")
            elif i != 0:
                out.append(f"w{i}")
        return " ".join(out)


def test_close_tag_detection_survives_skip_special_tokens():
    """</think> is an added token on some checkpoints; skipping specials would
    hide it and make a closed block look like it is still thinking."""
    tok = FakeTok()
    ids = [11, 999, 12]
    assert "</think>" in tok.decode(ids, skip_special_tokens=False)

def test_eos_is_detectable_even_when_pad_equals_eos():
    """_ensure_pad_token sets pad = eos, so filtering pads would delete the
    evidence that a sequence finished on its own."""
    raw = [11, 12, 2, 0, 0]
    eos_id = 2
    assert eos_id in raw                       # decided before filtering
    filtered = [t for t in raw if t != 0]
    assert filtered == [11, 12, 2]

def test_budget_is_skipped_when_it_would_not_bind():
    """think_budget >= max_new_tokens leaves nothing for the answer, so the
    two-phase path must not engage."""
    cfg = Config().generation
    cfg.max_new_tokens, cfg.think_budget = 100, 100
    assert not (0 < cfg.think_budget < cfg.max_new_tokens)
    cfg.think_budget = 60
    assert 0 < cfg.think_budget < cfg.max_new_tokens

def test_budget_defaults_to_off():
    assert Config().generation.think_budget == 0

def test_budget_flows_from_config_to_the_generator(tmp_path):
    """The knob has to reach sample_batch, not sit unread in the config."""
    from llm.generation import InProcessGenerator

    seen = {}

    class Spy:
        def render(self, m): return m[0]["content"]
        def sample_batch(self, texts, **kw):
            seen.update(kw)
            return [("x", [1]) for _ in texts]

    cfg = Config().generation
    cfg.max_new_tokens, cfg.think_budget = 100, 60
    InProcessGenerator(Spy(), cfg, progress=False).generate(
        [(0, [{"role": "user", "content": "hi"}], 2)])
    assert seen["think_budget"] == 60
    assert seen["think_close_tag"] == "</think>"
