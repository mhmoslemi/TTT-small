"""
The memory maker (Sec. 2.2).

After the B_t rollouts of step t are evaluated, split them into

    S_t = {(x_i, a_i, s'_i, r_i) : r_i > 0}
    F_t = {(x_i, a_i, f_i)       : r_i = 0 or the attempt failed}

and make exactly two LLM calls, one per group, each returning L lessons. Both
use the same backbone as the generator with dedicated prompts, so nothing extra
is loaded.

Sampling from the group. A step can produce 512 rollouts and the prompts cannot
hold them all, so each side is subsampled to max_examples_per_call:

  successes  the highest-reward ones. The paper wants the choices that separate
             strong attempts from weak ones, and the top of the group is where
             that signal lives.

  failures   round-robin over distinct verifier messages. Taking the first k
             failures instead would usually mean k copies of one crash, and a
             lesson extracted from k copies of one crash is a note about one
             trajectory, which is exactly what Sec. 2.2 says these are not.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

from memory.prompts import (build_negative_messages, build_positive_messages,
                            parse_lessons)
from memory.types import FAILURE, SUCCESS, Lesson, RolloutRecord


class LessonExtractor:
    def __init__(self, cfg, llm, meta_description: str, fail_score: float = 0.0):
        self.cfg = cfg
        self.llm = llm
        self.meta_description = meta_description or "(no task description)"
        self.fail_score = float(fail_score)
        self.last_raw: Dict[str, str] = {}      # outcome -> raw response, for logging

    # ------------------------------------------------------------------
    def partition(self, records: Sequence[RolloutRecord]):
        successes, failures = [], []
        for rec in records:
            (successes if rec.is_success(self.fail_score) else failures).append(rec)
        return successes, failures

    # ------------------------------------------------------------------
    def _pick_successes(self, records: List[RolloutRecord]) -> List[RolloutRecord]:
        k = int(self.cfg.max_examples_per_call)
        return sorted(records, key=lambda r: r.reward, reverse=True)[:k]

    def _pick_failures(self, records: List[RolloutRecord]) -> List[RolloutRecord]:
        k = int(self.cfg.max_examples_per_call)
        buckets: Dict[str, List[RolloutRecord]] = {}
        for rec in records:
            buckets.setdefault(rec.failure_signature(), []).append(rec)
        picked: List[RolloutRecord] = []
        # One from each distinct failure first, then second ones, and so on.
        while len(picked) < k and any(buckets.values()):
            for sig in list(buckets):
                if not buckets[sig]:
                    del buckets[sig]
                    continue
                picked.append(buckets[sig].pop(0))
                if len(picked) >= k:
                    break
        return picked

    # ------------------------------------------------------------------
    def extract(self, records: Sequence[RolloutRecord], step: int,
                adapter_path=None, verbose: bool = True) -> List[Lesson]:
        """Returns up to 2L lessons. Never raises: a failed call costs a step."""
        successes, failures = self.partition(records)
        L = int(self.cfg.lessons_per_call)
        out: List[Lesson] = []
        self.last_raw = {}
        t0 = time.time()

        if successes:
            picked = self._pick_successes(successes)
            messages = build_positive_messages(
                self.meta_description, picked, L,
                int(self.cfg.max_chars_per_example))
            out += self._one_call(messages, SUCCESS, step, L, adapter_path)

        if failures:
            picked = self._pick_failures(failures)
            messages = build_negative_messages(
                self.meta_description, picked, L,
                int(self.cfg.max_chars_per_example),
                int(self.cfg.feedback_chars))
            out += self._one_call(messages, FAILURE, step, L, adapter_path)

        if verbose:
            n_pos = sum(1 for l in out if l.outcome == SUCCESS)
            n_neg = len(out) - n_pos
            print(f"[memory] step {step}: |S_t|={len(successes)} |F_t|={len(failures)}"
                  f" -> {n_pos} positive + {n_neg} negative lessons"
                  f" ({time.time() - t0:.1f}s)")
        return out

    def _one_call(self, messages, outcome: str, step: int, L: int,
                  adapter_path) -> List[Lesson]:
        try:
            raw = self.llm.complete(messages, adapter_path=adapter_path)
        except Exception as e:
            print(f"[memory] {outcome} extraction call failed: {e!r}")
            return []
        self.last_raw[outcome] = raw
        lessons = parse_lessons(raw, outcome, step, L)
        if not lessons:
            print(f"[memory] {outcome} extraction returned nothing parsable "
                  f"({len(raw or '')} chars)")
        return lessons


# ----------------------------------------------------------------------
def build_meta_description(problem, cfg) -> str:
    """
    The `d` shown to the memory maker. Assembled from the Problem attributes
    that every problem in problems/ already defines, so adding a problem does
    not mean touching this module.
    """
    direction = "higher is better" if getattr(problem, "maximize", True) \
        else "lower is better"
    lines = [
        f"name: {getattr(problem, 'name', 'unknown')}",
        f"entrypoint: {getattr(problem, 'entrypoint', '?')}",
        f"metric: {getattr(problem, 'metric_name', 'score')} ({direction})",
    ]
    ptype = getattr(cfg, "problem_type", "") or ""
    if ptype:
        lines.append(f"variant: {ptype}")
    target = getattr(problem, "target", None)
    if target is not None:
        lines.append(f"target: {target}")
    lines.append("The agent writes a Python program; a verifier runs it and "
                 "returns a scalar reward plus textual feedback.")
    return "\n".join(lines)
