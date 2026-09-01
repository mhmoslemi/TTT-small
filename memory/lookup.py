"""
Memory lookup: the model reads its own index and asks for what it wants.

This replaces Eq. 7 entirely. There is no query vector, no cosine, no top-m.

Why, from the measurements: cosine retrieval left 87-89% of lessons never
retrieved while the top five took 60-77% of all retrievals, and because the
query was program text and the similarity lexical, 88-96% of retrieval mass went
to the 23-34% of entries that were code. The retriever was systematically
selecting the least transferable content, and all 48 lessons about global search
(the behavior that actually produced an escape) got zero retrievals.

Three modes:

  select   one call per parent, all parents batched into a single pool round.
           The model sees the whole index and returns ids. Cost is one extra
           generation round per step, roughly one short prompt per parent.
  all      no call; every lesson goes into every prompt. Cheapest in calls,
           most expensive in rollout prompt tokens.
  none     never inject, for measuring extraction alone.

The selection is recorded per parent, so which lessons the model CHOSE is now an
observable with a stated reason, rather than something inferred from a cosine.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from memory.prompts import build_lookup_messages, parent_block, parse_lookup
from memory.types import Lesson


class MemoryLookup:
    def __init__(self, cfg, bank, llm, meta_description: str):
        self.cfg = cfg
        self.bank = bank
        self.llm = llm
        self.meta_description = meta_description or "(no task description)"
        self.last_reasons: Dict[int, str] = {}

    # ------------------------------------------------------------------
    def _fallback(self) -> List[str]:
        mode = getattr(self.cfg, "lookup_fallback", "none")
        k = int(self.cfg.lookup_max_select)
        if mode == "none" or k <= 0 or not self.bank.lessons:
            return []
        if mode == "recent":
            items = sorted(self.bank.lessons, key=lambda l: l.step, reverse=True)
        else:
            items = sorted(self.bank.lessons,
                           key=lambda l: (self.bank.evidence_mean(l),
                                          l.importance, l.step), reverse=True)
        return [l.id for l in items[:k]]

    # ------------------------------------------------------------------
    def select_batch(self, parent_ctxs: Sequence, step_idx: int,
                     adapter_path=None, verbose: bool = True
                     ) -> Dict[int, List[Lesson]]:
        """
        Returns {group_index: [Lesson, ...]} in the order the model asked for.

        An empty bank returns empty selections without any LLM call, which is
        what keeps step 0 byte-identical to a no-memory run.
        """
        n = len(parent_ctxs)
        empty = {g: [] for g in range(n)}
        if not self.bank.lessons or self.cfg.lookup_mode == "none":
            return empty

        if self.cfg.lookup_mode == "all":
            # Every lesson, every prompt. No selection call. build_injection
            # still applies the token budget, so a large bank is truncated
            # rather than blowing up the context.
            all_lessons = self.bank.fetch(self.bank.ids(), count_use=True)
            return {g: list(all_lessons) for g in range(n)}

        catalog = self.bank.catalog()
        valid = self.bank.catalog_ids()
        k = int(self.cfg.lookup_max_select)
        if k <= 0 or not catalog:
            return empty

        prompts = []
        for pc in parent_ctxs:
            reward = getattr(pc, "value", 0.0) or 0.0
            raw = getattr(pc, "raw_score", None)
            head = f"score = {reward:.6f}"
            if raw is not None:
                head += f"  (metric = {raw:.6f})"
            prompts.append(build_lookup_messages(
                self.meta_description,
                parent_block(head, getattr(pc, "code", "") or ""),
                catalog, k))

        t0 = time.time()
        try:
            lookup_step = int(step_idx)
            if bool(getattr(self.cfg, "is_v2", False)):
                from memory.llm import LOOKUP_STEP_OFFSET
                lookup_step += LOOKUP_STEP_OFFSET
            replies = self.llm.complete_many(
                prompts, adapter_path=adapter_path, step_idx=lookup_step,
                max_new_tokens=int(self.cfg.lookup_max_new_tokens),
                temperature=float(self.cfg.lookup_temperature))
        except Exception as e:
            print(f"[memory] lookup call failed ({e!r}); "
                  f"falling back to '{getattr(self.cfg, 'lookup_fallback', 'none')}'")
            fb = self._fallback()
            return {g: self.bank.fetch(fb) for g in range(n)}

        out, n_empty, n_ids = {}, 0, 0
        for g in range(n):
            reply = replies[g] if g < len(replies) else ""
            res = parse_lookup(reply, valid, k)
            self.last_reasons[g] = res.why
            if not res.ids:
                n_empty += 1
                res_ids = self._fallback()
            else:
                res_ids = res.ids
            n_ids += len(res_ids)
            out[g] = self.bank.fetch(res_ids)

        self.bank.stats["lookups"] += n
        self.bank.stats["selections"] += n_ids
        self.bank.stats["empty_selections"] += n_empty

        if verbose:
            print(f"[memory] step {step_idx}: lookup over {len(catalog)} lessons "
                  f"-> {n_ids} selections across {n} parents "
                  f"({n_empty} chose nothing)  {time.time() - t0:.1f}s")
        return out
