"""
The memory maker (Sec. 2.2), with two changes driven by the measurements.

1. It sees the PARENT, not just the children. §4.3: "extraction should compare
   children with parents so that useful changes remain identifiable even when
   absolute rewards have plateaued." The median within-group reward spread was
   0.000000, with zero spread in 14 of 37 steps for MEM-B. Handed k
   identical-scoring programs and asked what made them good, the extractor
   restated the construction, because with no delta there was nothing else to
   say. Each record now carries its parent's program and score, so the question
   becomes what changed.

2. Every proposed lesson is checked by hygiene.py before it can enter the bank.
   Global-scope lessons carrying code, over-long code blocks, and coordinate or
   lattice construction at any scope are rejected and counted. This is what
   stands between the bank and another 979e6dc0.

Sampling from the group is unchanged: successes by highest reward, failures
round-robin over distinct verifier messages so one repeated crash cannot fill the
prompt.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from memory.hygiene import HygieneStats, violation
from memory.prompts import (ExtractionResult, build_contrast_messages,
                            build_negative_messages, build_positive_messages,
                            parse_extraction)
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
        self._counts = (0, 0)

    # ------------------------------------------------------------------
    def partition(self, records: Sequence[RolloutRecord]):
        successes, failures = [], []
        for rec in records:
            (successes if rec.is_success(self.fail_score) else failures).append(rec)
        return successes, failures

    def _pick_successes(self, records: List[RolloutRecord]) -> List[RolloutRecord]:
        """
        Highest reward first, but when the batch has any measurable improvement
        over parents, prefer the biggest improvers. On a plateau those coincide;
        off a plateau, "what improved" beats "what scored highest", since the
        highest scorer may simply be the parent unchanged.
        """
        k = int(self.cfg.max_examples_per_call)
        deltas = [r.delta() for r in records if r.delta() is not None]
        if deltas and max(deltas) > 1e-12:
            return sorted(records,
                          key=lambda r: (r.delta() if r.delta() is not None else 0.0,
                                         r.reward), reverse=True)[:k]
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
                adapter_path=None) -> ExtractionResult:
        successes, failures = self.partition(records)
        L = int(self.cfg.lessons_per_call)
        catalog = self.bank.catalog() if self.bank is not None else []
        require_full = bool(getattr(self.cfg, "require_full_lessons", False))
        max_code = int(getattr(self.cfg, "max_code_lines", 4))

        # `extract_from` decides which sides are called at all. With
        # "failure" the positive call is skipped entirely, so a step costs one
        # LLM call rather than two.
        source = str(getattr(self.cfg, "extract_from", "both"))
        want_pos = source in ("both", "success")
        want_neg = source in ("both", "failure")

        # Contrastive: one call over both halves. The tag is SUCCESS by
        # default; a lesson declaring kind="pitfall" is routed to FAILURE by the
        # parser, so both outcome types can come out of the single call.
        if str(getattr(self.cfg, "extract_mode", "contrast")) == "contrast" \
                and (successes or failures):
            messages = build_contrast_messages(
                self.meta_description,
                self._pick_successes(successes) if successes else [],
                self._pick_failures(failures) if failures else [],
                L, int(self.cfg.max_chars_per_example),
                int(self.cfg.feedback_chars), catalog, max_code)
            result = ExtractionResult()
            self.last_raw = {}
            self._counts = (len(successes), len(failures))
            self._one_call(messages, SUCCESS, step, L, adapter_path, result)
            result.reinforce = list(dict.fromkeys(result.reinforce))
            return result

        prompts, tags = [], []
        if successes and want_pos:
            prompts.append(build_positive_messages(
                self.meta_description, self._pick_successes(successes), L,
                int(self.cfg.max_chars_per_example), catalog, require_full,
                max_code))
            tags.append(SUCCESS)
        if failures and want_neg:
            prompts.append(build_negative_messages(
                self.meta_description, self._pick_failures(failures), L,
                int(self.cfg.max_chars_per_example),
                int(self.cfg.feedback_chars), catalog, require_full, max_code))
            tags.append(FAILURE)

        result = ExtractionResult()
        self.last_raw = {}
        self._counts = (len(successes), len(failures))
        if not prompts:
            return result

        from memory.llm import EXTRACT_STEP_OFFSET
        try:
            replies = self.llm.complete_many(
                prompts, adapter_path=adapter_path,
                step_idx=EXTRACT_STEP_OFFSET + int(step),
                max_new_tokens=int(self.cfg.max_new_tokens),
                temperature=float(self.cfg.temperature))
        except Exception as e:
            print(f"[memory] extraction call failed ({e!r})")
            return result

        for tag, raw in zip(tags, replies):
            self.last_raw[tag] = raw
            parsed = parse_extraction(raw, tag, step, L)
            if not parsed.lessons and not parsed.reinforce:
                print(f"[memory] {tag} extraction returned nothing parsable "
                      f"({len(raw or '')} chars)")
            result.lessons.extend(parsed.lessons)
            result.reinforce.extend(parsed.reinforce)

        result.reinforce = list(dict.fromkeys(result.reinforce))
        return result

    def _one_call(self, messages, tag, step, L, adapter_path,
                  into: ExtractionResult) -> None:
        from memory.llm import EXTRACT_STEP_OFFSET
        try:
            raw = self.llm.complete_many(
                [messages], adapter_path=adapter_path,
                step_idx=EXTRACT_STEP_OFFSET + int(step),
                max_new_tokens=int(self.cfg.max_new_tokens),
                temperature=float(self.cfg.temperature))[0]
        except Exception as e:
            print(f"[memory] extraction call failed ({e!r})")
            return
        self.last_raw[tag] = raw
        parsed = parse_extraction(raw, tag, step, L)
        if not parsed.lessons and not parsed.reinforce:
            print(f"[memory] extraction returned nothing parsable "
                  f"({len(raw or '')} chars)")
        into.lessons.extend(parsed.lessons)
        into.reinforce.extend(parsed.reinforce)
        if parsed.reflection and not into.reflection:
            into.reflection = parsed.reflection

    # ------------------------------------------------------------------
    def screen(self, lessons: Sequence[Lesson], hygiene: HygieneStats
               ) -> List[Lesson]:
        """Drop anything that hands over a construction rather than an operation."""
        kept = []
        for lesson in lessons:
            reason = violation(lesson.lesson, lesson.scope, self.cfg)
            if reason:
                hygiene.reject(reason)
                continue
            hygiene.keep()
            kept.append(lesson)
        return kept

    # ------------------------------------------------------------------
    def update(self, records: Sequence[RolloutRecord], step: int,
               adapter_path=None, verbose: bool = True) -> Dict:
        """Extract, screen, reinforce, insert. Prints one line."""
        t0 = time.time()
        result = self.extract(records, step, adapter_path=adapter_path)
        n_succ, n_fail = self._counts

        hygiene = HygieneStats()
        kept = self.screen(result.lessons, hygiene)

        reinforced = added = 0
        if self.bank is not None:
            reinforced = self.bank.reinforce(result.reinforce)
            added = self.bank.add_many(kept)
            self.bank.stats["rejected"] += hygiene.rejected

        stats = {"step": step, "successes": n_succ, "failures": n_fail,
                 "proposed": len(result.lessons), "kept": len(kept),
                 "rejected": hygiene.rejected, "added": added,
                 "reinforced": reinforced, "seconds": time.time() - t0}

        if verbose:
            counts = self.bank.counts() if self.bank is not None else {}
            src = str(getattr(self.cfg, "extract_from", "both"))
            tag = "" if src == "both" else f" [{src}-only]"
            print(f"[memory] step {step}{tag}: |S_t|={n_succ} |F_t|={n_fail}  "
                  f"proposed={len(result.lessons)} added={added} "
                  f"reinforced={reinforced}  {hygiene.line()}  "
                  f"{stats['seconds']:.1f}s")
            if counts:
                print(f"[memory] bank={counts['total']} "
                      f"({counts['success']}+/{counts['failure']}-, "
                      f"{counts['local']} local/{counts['global']} global)  "
                      + (self.bank.usage_summary() if self.bank else ""))
        return stats


# ----------------------------------------------------------------------
def build_meta_description(problem, cfg) -> str:
    """
    The `d` shown to the maker and the selector. Built from Problem attributes
    every problem already defines, so adding a problem needs no edit here.
    """
    direction = "higher is better" if getattr(problem, "maximize", True) \
        else "lower is better"
    lines = [f"name: {getattr(problem, 'name', 'unknown')}",
             f"entrypoint: {getattr(problem, 'entrypoint', '?')}",
             f"metric: {getattr(problem, 'metric_name', 'score')} ({direction})"]
    ptype = getattr(cfg, "problem_type", "") or ""
    if ptype:
        lines.append(f"variant: {ptype}")
    target = getattr(problem, "target", None)
    if target is not None:
        lines.append(f"target: {target}")
    lines.append("The agent writes a Python program; a verifier runs it and "
                 "returns a scalar reward plus textual feedback.")
    return "\n".join(lines)