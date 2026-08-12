"""
Offline check of the whole memory path with a stub LLM. No GPU, no model, no
network:

    cd <repo root> && python -m memory.selftest

Covers the parts that fail silently at 3am on a cluster rather than loudly:
the master flag ignoring memory_* keys, JSON parsing of the extraction reply
(including a fenced reply and a reply with prose around the array), dedup,
eviction, retrieval ranking, save/load round-trip, and the fact that
inject_memories does not mutate the caller's message list.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from memory_v1.bank import MemoryBank
from memory_v1.config import MemoryConfig
from memory_v1.embedding import Embedder
from memory_v1.extractor import LessonExtractor
from memory_v1.prompts import inject_memories, parse_lessons
from memory_v1.types import FAILURE, SUCCESS, RolloutRecord

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


class StubLLM:
    """Returns a canned reply per call, cycling through the shapes we must handle."""

    def __init__(self):
        self.calls = 0
        self.replies = [
            json.dumps([
                {"title": "Seed corners first",
                 "summary": "Place circles in the four corners before the interior.",
                 "lesson": "Every high-reward attempt fixed corner circles first, "
                           "then grew interior radii against them."},
                {"title": "Polish with SLSQP",
                 "summary": "A local refinement pass adds a few thousandths.",
                 "lesson": "Running a constrained local optimizer after the "
                           "constructive phase consistently improved the sum."},
            ]),
            "Here is the analysis.\n```json\n"
            + json.dumps([
                {"title": "Unpack the return tuple",
                 "summary": "Returning a list instead of ndarray crashes the validator.",
                 "lesson": "The validator indexes centers as an array; return "
                           "np.asarray for both centers and radii."},
            ])
            + "\n```\nThat covers it.",
        ]

    def complete(self, messages, adapter_path=None):
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        return reply


def main():
    print("memory selftest")

    # --- master flag --------------------------------------------------
    print("\nmaster flag")
    off = MemoryConfig.from_dict({"memory": False, "memory_top_m": 99}, verbose=False)
    check("disabled config ignores memory_* keys", off.enabled is False and off.top_m == 5)
    on = MemoryConfig.from_dict({"memory": True, "memory_top_m": 3,
                                 "memory_lessons_per_call": 2}, verbose=False)
    check("enabled config reads memory_* keys", on.enabled and on.top_m == 3)
    check("unknown memory_* keys do not raise",
          MemoryConfig.from_dict({"memory": True, "memory_nonsense": 1},
                                 verbose=False).enabled)

    # --- parsing ------------------------------------------------------
    print("\nparsing")
    fenced = "```json\n[{\"title\":\"a\",\"summary\":\"b\",\"lesson\":\"c\"}]\n```"
    check("fenced JSON", len(parse_lessons(fenced, SUCCESS, 0, 3)) == 1)
    noisy = "Sure!\n[{\"title\":\"a\",\"summary\":\"b\",\"lesson\":\"c\"}]\nHope that helps."
    check("JSON with prose around it", len(parse_lessons(noisy, SUCCESS, 0, 3)) == 1)
    thinking = "<think>hmm</think>[{\"title\":\"a\",\"lesson\":\"c\"}]"
    check("think block stripped", len(parse_lessons(thinking, SUCCESS, 0, 3)) == 1)
    check("garbage returns nothing", parse_lessons("no json here", SUCCESS, 0, 3) == [])
    check("expected count caps output",
          len(parse_lessons(json.dumps([{"title": str(i), "lesson": "x" * 30}
                                        for i in range(9)]), SUCCESS, 0, 3)) == 3)

    # --- bank ---------------------------------------------------------
    print("\nbank")
    cfg = MemoryConfig.from_dict({"memory": True, "memory_top_m": 2,
                                  "memory_lessons_per_call": 2,
                                  "memory_embed_backend": "hash",
                                  "memory_max_lessons": 4}, verbose=False)
    embedder = Embedder("hash", dim=512, verbose=False)
    bank = MemoryBank(cfg, embedder)

    lessons = parse_lessons(StubLLM().replies[0], SUCCESS, 0, 2)
    check("add_many inserts", bank.add_many(lessons) == 2)
    check("re-adding the same lessons is a no-op", bank.add_many(lessons) == 0)

    fail_lessons = parse_lessons(
        json.dumps([{"title": "Validator wants arrays",
                     "summary": "Return ndarray, not list.",
                     "lesson": "The validator indexes centers as an ndarray."}]),
        FAILURE, 1, 1)
    bank.add_many(fail_lessons)
    check("counts split by outcome",
          bank.counts() == {"total": 3, "success": 2, "failure": 1})

    hits = bank.retrieve("local optimizer refinement pass after construction")
    check("retrieve returns top_m", len(hits) == 2)
    check("retrieve ranks the on-topic lesson first",
          "SLSQP" in hits[0].title or "optimizer" in hits[0].lesson.lower())
    check("retrieve counts a use", hits[0].uses >= 1)

    empty = MemoryBank(cfg, embedder)
    check("retrieve on an empty bank returns []", empty.retrieve("anything") == [])

    scoped = MemoryConfig.from_dict({"memory": True, "memory_top_m": 5,
                                     "memory_embed_backend": "hash",
                                     "memory_retrieval_scope": "failure"},
                                    verbose=False)
    bank.cfg = scoped
    only_fail = bank.retrieve("array")
    check("retrieval_scope filters by outcome",
          all(l.outcome == FAILURE for l in only_fail) and len(only_fail) == 1)
    bank.cfg = cfg

    # --- eviction -----------------------------------------------------
    print("\neviction")
    for i in range(6):
        bank.add_many(parse_lessons(
            json.dumps([{"title": f"filler {i}",
                         "summary": f"unrelated topic number {i}",
                         "lesson": f"a distinct filler lesson about widget {i}"}]),
            SUCCESS, 2 + i, 1))
    check("bank respects max_lessons", len(bank) <= 4)

    # --- persistence --------------------------------------------------
    print("\npersistence")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.json"
        bank.save(path)
        check("save writes the file", path.exists())
        reloaded = MemoryBank(cfg, embedder)
        n = reloaded.load(path)
        check("load round-trips every lesson", n == len(bank))
        check("reloaded bank still retrieves",
              isinstance(reloaded.retrieve("widget"), list))

    # --- injection ----------------------------------------------------
    print("\ninjection")
    base = [{"role": "user", "content": "ORIGINAL"}]
    out = inject_memories(base, hits, mode="append")
    check("append keeps one message", len(out) == 1)
    check("append preserves the original text", "ORIGINAL" in out[0]["content"])
    check("append adds the lessons", "Lessons from earlier attempts" in out[0]["content"])
    check("caller's list is untouched", base[0]["content"] == "ORIGINAL")
    sysmode = inject_memories(base, hits, mode="system")
    check("system mode prepends a system message",
          len(sysmode) == 2 and sysmode[0]["role"] == "system")
    check("no lessons means no change", inject_memories(base, [])[0]["content"] == "ORIGINAL")

    # --- extractor ----------------------------------------------------
    print("\nextractor")
    records = [
        RolloutRecord(step=0, group=0, rollout=i, response="<strategy>grid</strategy>",
                      code="def run_packing(): pass", reward=2.6 - 0.01 * i,
                      valid=True, parsed=True, ran=True)
        for i in range(3)
    ] + [
        RolloutRecord(step=0, group=0, rollout=10 + i, response="broken",
                      reward=0.0, valid=False, parsed=True, ran=False,
                      msg=f"run_failed: ValueError at line {i}",
                      stdout="Traceback ...")
        for i in range(5)
    ]
    ex = LessonExtractor(cfg, StubLLM(), "circle packing, n=26", fail_score=0.0)
    succ, fail = ex.partition(records)
    check("partition splits on reward and validity", len(succ) == 3 and len(fail) == 5)
    check("failure signature strips line numbers",
          len({r.failure_signature() for r in fail}) == 1)
    produced = ex.extract(records, step=0, verbose=False)
    check("extract makes both calls", ex.llm.calls == 2)
    check("extract labels outcomes",
          any(l.outcome == SUCCESS for l in produced)
          and any(l.outcome == FAILURE for l in produced))

    empty_side = LessonExtractor(cfg, StubLLM(), "task", fail_score=0.0)
    empty_side.extract([r for r in records if r.valid], step=0, verbose=False)
    check("no failures means one call only", empty_side.llm.calls == 1)

    class BrokenLLM:
        def complete(self, messages, adapter_path=None):
            raise RuntimeError("cuda oom")

    broken = LessonExtractor(cfg, BrokenLLM(), "task", fail_score=0.0)
    check("a failed extraction call does not raise",
          broken.extract(records, step=0, verbose=False) == [])

    print("\n" + ("all checks passed" if not FAILS
                  else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
