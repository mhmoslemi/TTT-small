"""Batched generation: one generate() call for the whole step's B_t."""
import pytest

from config import Config
from llm.generation import InProcessGenerator, _is_oom


class RecordingBackbone:
    """Counts generate() calls and reports the batch size of each."""

    def __init__(self, fail_above=None):
        self.batches = []
        self.fail_above = fail_above

    def render(self, messages):
        return messages if isinstance(messages, str) else messages[0]["content"]

    def sample_batch(self, prompt_texts, max_new_tokens=0, temperature=1.0,
                     top_p=1.0, on_step=None, **kwargs):
        texts = list(prompt_texts)
        if self.fail_above is not None and len(texts) > self.fail_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.batches.append(len(texts))
        if on_step is not None:
            on_step()
        return [(f"resp:{t}", [1, 2, 3]) for t in texts]


def _cfg(**kw):
    cfg = Config().generation
    cfg.max_new_tokens = 4
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _jobs():
    # Two leaf targets asking for 4, one virtual target asking for 1: B_t = 9.
    return [(0, [{"role": "user", "content": "aaaa"}], 4),
            (1, [{"role": "user", "content": "bb"}], 4),
            (2, [{"role": "user", "content": "cccccc"}], 1)]


def test_whole_step_goes_through_one_call():
    backbone = RecordingBackbone()
    out = InProcessGenerator(backbone, _cfg(), progress=False).generate(_jobs())
    assert backbone.batches == [9]              # not [4, 4, 1]

def test_every_group_gets_exactly_the_count_it_asked_for():
    out = InProcessGenerator(RecordingBackbone(), _cfg(), progress=False).generate(_jobs())
    assert {g: len(s) for g, s in out.items()} == {0: 4, 1: 4, 2: 1}

def test_samples_are_routed_back_to_the_right_group():
    """Length sorting reorders the flat batch; the mapping must survive it."""
    out = InProcessGenerator(RecordingBackbone(), _cfg(), progress=False).generate(_jobs())
    assert all(text == "resp:aaaa" for text, _ in out[0])
    assert all(text == "resp:bb" for text, _ in out[1])
    assert all(text == "resp:cccccc" for text, _ in out[2])

def test_batch_size_caps_the_call():
    backbone = RecordingBackbone()
    InProcessGenerator(backbone, _cfg(batch_size=4), progress=False).generate(_jobs())
    assert backbone.batches == [4, 4, 1]
    assert sum(backbone.batches) == 9

def test_batch_size_zero_means_all_of_b_t():
    backbone = RecordingBackbone()
    InProcessGenerator(backbone, _cfg(batch_size=0), progress=False).generate(_jobs())
    assert backbone.batches == [9]

def test_prompts_are_length_sorted_to_limit_padding_waste():
    """Shorter prompts first, so a batch is as uniform as it can be."""
    seen = []

    class Spy(RecordingBackbone):
        def sample_batch(self, prompt_texts, **kw):
            seen.extend(prompt_texts)
            return super().sample_batch(prompt_texts, **kw)

    InProcessGenerator(Spy(), _cfg(), progress=False).generate(_jobs())
    assert [len(t) for t in seen] == sorted(len(t) for t in seen)

def test_oom_halves_the_batch_and_still_returns_everything():
    backbone = RecordingBackbone(fail_above=4)
    out = InProcessGenerator(backbone, _cfg(), progress=False).generate(_jobs())
    assert max(backbone.batches) <= 4
    assert sum(backbone.batches) == 9
    assert {g: len(s) for g, s in out.items()} == {0: 4, 1: 4, 2: 1}

def test_oom_on_a_single_sequence_propagates():
    """Nothing left to halve -- surface it rather than loop forever."""
    gen = InProcessGenerator(RecordingBackbone(fail_above=0), _cfg(), progress=False)
    with pytest.raises(RuntimeError, match="out of memory"):
        gen.generate([(0, [{"role": "user", "content": "x"}], 1)])

def test_a_non_oom_error_is_not_retried():
    class Broken(RecordingBackbone):
        def sample_batch(self, prompt_texts, **kw):
            raise ValueError("shape mismatch")

    gen = InProcessGenerator(Broken(), _cfg(), progress=False)
    with pytest.raises(ValueError, match="shape mismatch"):
        gen.generate(_jobs())

def test_empty_job_list():
    assert InProcessGenerator(RecordingBackbone(), _cfg(), progress=False).generate([]) == {}

@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("CUDA out of memory"), True),
    (RuntimeError("Out Of Memory on device"), True),
    (ValueError("shape mismatch"), False),
])
def test_oom_detection(exc, expected):
    assert _is_oom(exc) is expected
