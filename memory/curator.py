"""
Periodic curation of the whole bank (Dynamic Cheatsheet, DC-RS).

The additive design accumulated 185 lessons containing nine near-copies of one
idea, because every dedup mechanism it had was a threshold on a similarity score
and no threshold separates a paraphrase from a new claim. Curation replaces the
threshold with a decision: the model is handed the entire bank and returns the
bank it wants to keep.

The forcing function is the one DC uses. Anything not explicitly carried into the
rewrite is gone. That converts "should I add this?" -- which is cheap to answer
yes to -- into "is this worth one of my slots?", which is not.

Runs every `curate_every` steps, never on step 0, and only when the bank is large
enough to be worth it. One extra LLM call on those steps.

Safety properties, because a bad rewrite could destroy the run's accumulated
memory in a single call:
  - the rewritten bank is rejected wholesale if it comes back empty, malformed,
    or shorter than `curate_min_keep_frac` of the input
  - ids are regenerated from content, and the usage/confirmation counters of any
    lesson whose text is unchanged are carried across
  - the pre-curation bank is written to memory.pre-curate-<step>.json first
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from memory.prompts import GENERALIZATION_RULES, _candidate_spans
from memory.types import (FAILURE, GLOBAL, LOCAL, SUCCESS, Lesson,
                          clamp_importance, content_tokens, jaccard)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_curation_messages(meta_description: str, entries: Sequence[str],
                            max_items: int, max_code: int = 4) -> List[Dict]:
    user = (
        "# MEMORY CURATOR\n\n"
        "You maintain the lesson memory for an automated program-search run. "
        "Your goal is a compact, high-value memory: you are rewriting it in "
        "full, and anything you do not carry forward is permanently lost.\n\n"
        f"## Task the memory serves\n{meta_description}\n\n"
        "## Current memory\n\n" + "\n\n".join(entries) + "\n\n"
        "## What to do\n"
        "**Selective retention.** Keep only entries that would change what a "
        "future attempt writes. Discard anything redundant, trivial, or so "
        "specific to one program that it does not generalize. Matched tail-uplift "
        "trials are the strongest evidence: protect relevant positive entries, "
        "be skeptical of repeatedly negative ones, and do not mistake an "
        "under-tested entry for a disproven one.\n\n"
        "**Merge duplicates.** Several entries here state the same idea in "
        "different words. Merge each such group into ONE entry, written better "
        "than any of its inputs, and sum their usage counts.\n\n"
        "**Refine.** Where an entry is vague, sharpen it. Where two entries "
        "conflict, keep the one the evidence supports and say so. Where an entry "
        "is a description of a program rather than an insight, either rewrite it "
        "as the insight or drop it.\n\n"
        "**Watch for monoculture.** If one idea dominates the memory, that is a "
        "warning sign rather than a signal of quality: a memory in which every "
        "entry points the same direction has stopped being able to suggest an "
        "alternative. Prefer a smaller memory that spans several distinct "
        "approaches over a larger one that repeats a single approach.\n\n"
        f"Return at most {max_items} entries. Fewer is better than more.\n\n"
        + GENERALIZATION_RULES.format(max_code=max_code) + "\n\n"
        "## Output format\n\n"
        "Return ONLY a JSON object, no prose and no fences:\n\n"
        "{\n"
        '  "notes": "<2-4 sentences: what you merged, what you dropped, why>",\n'
        '  "lessons": [\n'
        "    {\n"
        '      "title": "<plain English, under 10 words>",\n'
        '      "summary": "<one plain sentence>",\n'
        '      "scope": "local" | "global",\n'
        '      "outcome": "success" | "failure",\n'
        '      "lesson": "<the content, written for a model to reuse>",\n'
        '      "importance": <integer 1-5>,\n'
        '      "merged_from": ["<id>", ...]\n'
        "    }\n"
        "  ]\n"
        "}"
    )
    return [{"role": "user", "content": user}]


def curation_entries(bank, body_chars: int = 400) -> List[str]:
    """
    Full entries for the curator: unlike the lookup index, this includes the
    bodies, because merging two lessons requires reading both.
    """
    out = []
    for l in sorted(bank.lessons, key=lambda x: (-x.importance, x.step)):
        body = (l.lesson or l.summary).replace("\n", " ")[:body_chars]
        tail = l.mean_tail_uplift()
        out.append(f"[{l.id}] ({l.scope}/{l.outcome}, imp {l.importance:.1f}, "
                   f"step {l.step}, used {l.uses}x, confirmed {l.confirmations}x, "
                   f"matched tail {tail:+.6g} over {l.arm_trials} trials, "
                   f"wins {l.arm_tail_wins})\n"
                   f"  {l.title}\n  {body}")
    return out


def parse_curation(response_text: str, step: int) -> List[Lesson]:
    text = _THINK_RE.sub("", response_text or "").strip()
    if not text:
        return []
    payload = None
    for span in _candidate_spans(text):
        try:
            payload = json.loads(span)
            break
        except Exception:
            continue
    items = payload.get("lessons") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        body = str(item.get("lesson", item.get("content", ""))).strip()
        summary = str(item.get("summary", "")).strip()
        if not body and not summary:
            continue
        scope = str(item.get("scope", LOCAL)).strip().lower()
        outcome = str(item.get("outcome", SUCCESS)).strip().lower()
        lesson = Lesson.create(
            title=str(item.get("title", "")).strip() or summary[:80],
            summary=summary or body[:200], lesson=body or summary,
            outcome=outcome if outcome in (SUCCESS, FAILURE) else SUCCESS,
            step=step, scope=scope if scope in (LOCAL, GLOBAL) else LOCAL,
            importance=clamp_importance(item.get("importance", 3.0)))
        # Kept only until _carry_counters runs. Exact provenance is safer than
        # guessing from text when the curator merged several old entries.
        merged_from = item.get("merged_from", [])
        if isinstance(merged_from, list):
            lesson._merged_from_ids = [str(x).strip().lower()
                                       for x in merged_from if str(x).strip()]
        out.append(lesson)
    return out


class MemoryCurator:
    def __init__(self, cfg, bank, llm, meta_description: str, extractor=None):
        self.cfg = cfg
        self.bank = bank
        self.llm = llm
        self.meta_description = meta_description or "(no task description)"
        self.extractor = extractor          # reused for the hygiene screen
        self.last_notes = ""

    def due(self, step: int) -> bool:
        every = int(getattr(self.cfg, "curate_every", 0) or 0)
        if every <= 0 or step <= 0 or step % every != 0:
            return False
        return len(self.bank) >= int(getattr(self.cfg, "curate_min_bank", 20))

    def run(self, step: int, adapter_path=None, verbose: bool = True) -> Dict:
        t0 = time.time()
        before = len(self.bank)
        entries = curation_entries(self.bank)
        messages = build_curation_messages(
            self.meta_description, entries,
            int(getattr(self.cfg, "curate_max_items", 60)),
            int(getattr(self.cfg, "max_code_lines", 4)))

        from memory.llm import CURATE_STEP_OFFSET
        try:
            reply = self.llm.complete_many(
                [messages], adapter_path=adapter_path,
                step_idx=CURATE_STEP_OFFSET + int(step),
                max_new_tokens=int(getattr(self.cfg, "curate_max_new_tokens", 4096)),
                temperature=float(getattr(self.cfg, "temperature", 0.7)))[0]
        except Exception as e:
            print(f"[memory] curation call failed ({e!r}); bank left as is")
            return {"step": step, "before": before, "after": before, "applied": False}

        new_lessons = parse_curation(reply, step)

        # Screen the rewrite exactly like fresh extraction: curation must not be
        # a way for a construction to re-enter the bank.
        if self.extractor is not None and new_lessons:
            from memory.hygiene import HygieneStats
            h = HygieneStats()
            new_lessons = self.extractor.screen(new_lessons, h)

        min_frac = float(getattr(self.cfg, "curate_min_keep_frac", 0.25))
        if not new_lessons or len(new_lessons) < max(1, int(before * min_frac)):
            print(f"[memory] curation returned {len(new_lessons)} of {before} "
                  f"lesson(s), below the {min_frac:.0%} floor; REJECTED, bank "
                  f"left as is")
            return {"step": step, "before": before, "after": before, "applied": False}

        # Snapshot before replacing anything.
        if self.bank._path is not None:
            snap = Path(self.bank._path).with_name(
                f"memory.pre-curate-{step:03d}.json")
            try:
                self.bank.save(snap)
            except Exception:
                pass

        carried = self._carry_counters(new_lessons)
        self.bank.lessons = new_lessons
        self.bank._by_id = {l.id: l for l in new_lessons}
        self.bank.stats["curations"] = self.bank.stats.get("curations", 0) + 1
        self.bank.stats["curated_away"] = (self.bank.stats.get("curated_away", 0)
                                           + max(0, before - len(new_lessons)))

        if verbose:
            print(f"[memory] curation at step {step}: {before} -> "
                  f"{len(new_lessons)} lessons, {carried} counter(s) carried "
                  f"over  {time.time() - t0:.1f}s")
        return {"step": step, "before": before, "after": len(new_lessons),
                "applied": True, "carried": carried}

    def _carry_counters(self, new_lessons: Sequence[Lesson]) -> int:
        """
        A rewritten lesson keeps the usage history of the old one it came from,
        matched on token overlap. Without this every curation resets `uses` to
        zero and the selector loses the only evidence it has about which lessons
        have been worth choosing.
        """
        old = [(l, l.tokens()) for l in self.bank.lessons]
        old_by_id = {l.id: l for l in self.bank.lessons}
        claimed = set()
        carried = 0
        for new in new_lessons:
            sources = [old_by_id[x] for x in getattr(
                new, "_merged_from_ids", []) if x in old_by_id and x not in claimed]
            if not sources:
                toks = new.tokens()
                best, best_j = None, 0.0
                for cand, ctoks in old:
                    if cand.id in claimed:
                        continue
                    j = jaccard(toks, ctoks)
                    if j > best_j:
                        best, best_j = cand, j
                if best is not None and best_j >= 0.4:
                    sources = [best]
            if not sources:
                continue

            claimed.update(l.id for l in sources)
            new.uses = sum(l.uses for l in sources)
            new.confirmations = sum(l.confirmations for l in sources)
            new.arm_trials = sum(l.arm_trials for l in sources)
            new.arm_rollouts = sum(l.arm_rollouts for l in sources)
            new.arm_valid = sum(l.arm_valid for l in sources)
            new.arm_parent_improvements = sum(
                l.arm_parent_improvements for l in sources)
            new.arm_tail_wins = sum(l.arm_tail_wins for l in sources)
            new.tail_uplift_sum = sum(l.tail_uplift_sum for l in sources)
            tested = [l for l in sources if l.arm_trials > 0]
            if tested:
                new.tail_uplift_best = max(l.tail_uplift_best for l in tested)
                new.last_outcome_step = max(l.last_outcome_step for l in tested)
            carried += len(sources)
        return carried
