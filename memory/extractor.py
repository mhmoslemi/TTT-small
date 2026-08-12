"""
The memory maker (Sec. 2.2).

After the B_t rollouts of step t are evaluated, split them into

    S_t = {(x_i, a_i, s'_i, r_i) : r_i > 0}
    F_t = {(x_i, a_i, f_i)       : r_i = 0 or the attempt failed}

and make two LLM calls, one per group, using the same backbone as the generator
with dedicated prompts.

Both calls carry the current bank as a catalog and are told to write only what
is new, or to name an existing id under "reinforce" when the batch merely
confirms something already recorded. That is why lessons_per_call is a ceiling
here rather than a quota: a step that discovered nothing new should raise the
importance of what it confirmed and add nothing, which is strictly better than
adding L restatements for the dedup filter to absorb.

Sampling from the group. A step can produce hundreds of rollouts and the
prompts cannot hold them all:

  successes  the highest-reward ones, since that is where the choices that
             separate strong from weak attempts live.

  failures   round-robin over distinct verifier messages. Taking the first k
             instead would usually mean k copies of one crash, and a lesson
             extracted from k copies of one crash is a note about one
             trajectory, which is what Sec. 2.2 says these are not.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from memory.prompts import (ExtractionResult, build_negative_messages,
                            build_positive_messages, parse_extraction)
from memory.types import FAILURE, SUCCESS, Lesson, RolloutRecord


class LessonExtractor:
    def __init__(self, cfg, llm, meta_description: str, bank=None,
                 fail_score: float = 0.0):
        self.cfg = cfg
        self.llm = llm
        self.bank = bank
        self.meta_description = meta_description or "(no task description)"
        self.fail_score = float(fail_score)
        self.last_raw: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def partition(self, records: Sequence[RolloutRecord]):
        successes, failures = [], []
        for rec in records:
            (successes if rec.is_success(self.fail_score) else failures).append(rec)
        return successes, failures

    def _pick_successes(self, records: List[RolloutRecord]) -> List[RolloutRecord]:
        k = int(self.cfg.max_examples_per_call)
        return sorted(records, key=lambda r: r.reward, reverse=True)[:k]

    def _pick_failures(self, records: List[RolloutRecord]) -> List[RolloutRecord]:
        k = int(self.cfg.max_examples_per_call)
        buckets: Dict[str, List[RolloutRecord]] = {}
        for rec in records:
            buckets.setdefault(rec.failure_signature(), []).append(rec)
        picked: List[RolloutRecord] = []
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
                adapter_path=None, step_idx: Optional[int] = None
                ) -> ExtractionResult:
        """Up to 2L lessons plus reinforce ids. Never raises."""
        successes, failures = self.partition(records)
        L = int(self.cfg.lessons_per_call)
        catalog = self.bank.catalog() if self.bank is not None else []
        require_full = bool(getattr(self.cfg, "require_full_lessons", False))
        step_idx = step if step_idx is None else step_idx

        result = ExtractionResult()
        self.last_raw = {}

        if successes:
            messages = build_positive_messages(
                self.meta_description, self._pick_successes(successes), L,
                int(self.cfg.max_chars_per_example), catalog, require_full)
            self._one_call(messages, SUCCESS, step, L, adapter_path,
                           step_idx, result)

        if failures:
            messages = build_negative_messages(
                self.meta_description, self._pick_failures(failures), L,
                int(self.cfg.max_chars_per_example),
                int(self.cfg.feedback_chars), catalog, require_full)
            self._one_call(messages, FAILURE, step, L, adapter_path,
                           step_idx, result)

        result.reinforce = list(dict.fromkeys(result.reinforce))
        self._counts = (len(successes), len(failures))
        return result

    def _one_call(self, messages, outcome: str, step: int, L: int,
                  adapter_path, step_idx: int, into: ExtractionResult) -> None:
        try:
            raw = self.llm.complete(messages, adapter_path=adapter_path,
                                    step_idx=step_idx)
        except Exception as e:
            print(f"[memory] {outcome} extraction call failed: {e!r}")
            return
        self.last_raw[outcome] = raw
        parsed = parse_extraction(raw, outcome, step, L)
        if not parsed.lessons and not parsed.reinforce:
            print(f"[memory] {outcome} extraction returned nothing parsable "
                  f"({len(raw or '')} chars)")
        into.lessons.extend(parsed.lessons)
        into.reinforce.extend(parsed.reinforce)

    # ------------------------------------------------------------------
    def update(self, records: Sequence[RolloutRecord], step: int,
               adapter_path=None, verbose: bool = True) -> Dict:
        """
        Extract, reinforce, insert. Returns stats and prints one line, so the
        trainer's call site stays a single statement.
        """
        t0 = time.time()
        result = self.extract(records, step, adapter_path=adapter_path)
        n_succ, n_fail = getattr(self, "_counts", (0, 0))

        reinforced = 0
        added = 0
        if self.bank is not None:
            reinforced = self.bank.reinforce(result.reinforce)
            added = self.bank.add_many(result.lessons)

        stats = {
            "step": step,
            "successes": n_succ,
            "failures": n_fail,
            "proposed": len(result.lessons),
            "added": added,
            "reinforced": reinforced,
            "seconds": time.time() - t0,
        }
        if verbose:
            counts = self.bank.counts() if self.bank is not None else {}
            print(f"[memory] step {step}: |S_t|={n_succ} |F_t|={n_fail}  "
                  f"proposed={len(result.lessons)} added={added} "
                  f"reinforced={reinforced}  "
                  f"bank={counts.get('total', 0)} "
                  f"({counts.get('success', 0)} positive / "
                  f"{counts.get('failure', 0)} negative)  "
                  f"{stats['seconds']:.1f}s")
        return stats


# ----------------------------------------------------------------------
def build_meta_description(problem, cfg) -> str:
    """
    The `d` shown to the memory maker. Assembled from Problem attributes every
    problem in problems/ already defines, so adding a problem does not mean
    touching this module.
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
