import numpy as np
import pytest

from config import Config
from core.types import OUTCOME_FAILURE, OUTCOME_SUCCESS, Lesson
from llm.mock import MockBackbone
from memory.bank import MemoryBank
from memory.extractor import LessonExtractor, parse_lessons
from memory.retrieval import HashEmbedder, top_m_by_cosine


def _cfg(**kw):
    c = Config().memory
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(dim=128)
    a, b = e.encode(["hexagonal packing"]), e.encode(["hexagonal packing"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)

def test_similar_text_scores_higher_than_unrelated():
    e = HashEmbedder(dim=512)
    v = e.encode(["use a hexagonal lattice for circle packing",
                  "hexagonal lattice packing of circles",
                  "sort the database index by timestamp"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])

def test_top_m_ordering_and_bounds():
    m = np.eye(4, dtype=np.float32)
    assert top_m_by_cosine(m[2], m, 2)[0] == 2
    assert len(top_m_by_cosine(m[0], m, 99)) == 4
    assert top_m_by_cosine(m[0], m, 0) == []

def test_bank_is_additive_and_retrieves_relevant_lessons():
    bank = MemoryBank(_cfg(top_m=1, embedding_backend="hash"))
    bank.add([Lesson(title="hexagonal", summary="use a hexagonal lattice"),
              Lesson(title="database", summary="index by timestamp",
                     outcome=OUTCOME_FAILURE)])
    assert len(bank) == 2
    bank.add([Lesson(title="corners", summary="pack corners first")])
    assert len(bank) == 3                       # M_t = M_{t-1} u new
    hit = bank.retrieve("hexagonal lattice packing", 1)
    assert hit and hit[0].title == "hexagonal"

def test_bank_cap_evicts_oldest():
    bank = MemoryBank(_cfg(max_bank_size=2))
    bank.add([Lesson(title=f"l{i}") for i in range(4)])
    assert len(bank) == 2 and bank.lessons[0].title == "l2"

def test_empty_bank_retrieves_nothing():
    assert MemoryBank(_cfg()).retrieve("anything", 5) == []

def test_bank_roundtrip(tmp_path):
    bank = MemoryBank(_cfg())
    bank.add([Lesson(title="t", summary="s", body="b", outcome=OUTCOME_FAILURE, step=3)])
    bank.save(tmp_path / "m.json")
    other = MemoryBank(_cfg())
    assert other.load(tmp_path / "m.json") == 1
    assert other.lessons[0].outcome == OUTCOME_FAILURE and other.lessons[0].step == 3
    assert other.lessons[0].embedding is not None    # re-embedded on load

def test_parse_strict_json():
    got = parse_lessons(
        '[{"title": "a", "summary": "b", "lesson": "c"}]', OUTCOME_SUCCESS, 1, 3)
    assert len(got) == 1 and got[0].title == "a" and got[0].body == "c"

def test_parse_fenced_json_with_prose_around_it():
    text = 'Sure!\n```json\n[{"title": "x", "summary": "y", "lesson": "z"}]\n```\nDone.'
    assert parse_lessons(text, OUTCOME_SUCCESS, 0, 3)[0].title == "x"

def test_parse_respects_the_limit():
    blob = "[" + ",".join('{"title": "t", "summary": "s", "lesson": "l"}' for _ in range(9)) + "]"
    assert len(parse_lessons(blob, OUTCOME_SUCCESS, 0, 3)) == 3

def test_parse_falls_back_to_a_numbered_list():
    text = ("1. Pack the corners first because they waste the most area\n"
            "2. Use scipy SLSQP rather than a hand rolled gradient step\n")
    got = parse_lessons(text, OUTCOME_FAILURE, 2, 5)
    assert len(got) == 2 and all(l.outcome == OUTCOME_FAILURE for l in got)

def test_parse_garbage_returns_nothing():
    assert parse_lessons("...", OUTCOME_SUCCESS, 0, 3) == []
    assert parse_lessons("", OUTCOME_SUCCESS, 0, 3) == []

def test_extractor_makes_one_call_per_group_not_per_response():
    """The whole point of Sec. 2.2: the group is summarized jointly."""
    llm = MockBackbone(seed=0)
    ex = LessonExtractor(_cfg(lessons_per_group=2), llm, "problem")
    ex.extract([("code", 1.0)] * 8, [("bad", "boom")] * 8, step=0)
    assert llm.calls["chat_batch"] == 1          # one batched call...

def test_extractor_skips_an_empty_group():
    llm = MockBackbone(seed=0)
    ex = LessonExtractor(_cfg(), llm, "p")
    assert ex.extract([], [], step=0) == []
    assert llm.calls["chat_batch"] == 0

def test_extractor_tags_outcomes():
    llm = MockBackbone(seed=1)
    ex = LessonExtractor(_cfg(lessons_per_group=1), llm, "p")
    got = ex.extract([("c", 1.0)], [("c", "err")], step=4)
    assert {l.outcome for l in got} == {OUTCOME_SUCCESS, OUTCOME_FAILURE}
    assert all(l.step == 4 for l in got)
