"""
The memory bank M.

No vectors. The bank is a list of lessons plus an id index, and everything that
used to be a nearest-neighbour question is now either an exact id lookup or a
token-set comparison:

  selection    the model reads catalog() and names ids. See lookup.py.
  dedup        normalized-title equality, then token-set Jaccard. Measured on
               the v2 bank, identical-title pairs had cosine 0.45-0.95 under the
               old hashed embedder, so no threshold on that scale separated them;
               title normalization catches them outright.
  eviction     lowest importance, then least used, then oldest.

`uses` now means the model chose this lesson, not that a cosine ranked it top-5.
That makes the catalog's "used Nx" column a real signal for the selector, and it
makes the retrieval-concentration measurement causal rather than an artifact of
the query being program text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from memory.types import (FAILURE, GLOBAL, LOCAL, SUCCESS, Lesson,
                          clamp_importance, jaccard, normalize_title)


class MemoryBank:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lessons: List[Lesson] = []
        self._by_id: Dict[str, Lesson] = {}
        self._path: Optional[Path] = None
        self.stats = {"proposed": 0, "added": 0, "duplicates": 0,
                      "reinforced": 0, "evicted": 0, "rejected": 0,
                      "lookups": 0, "selections": 0, "empty_selections": 0}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.lessons)

    def counts(self) -> Dict[str, int]:
        return {
            "total": len(self.lessons),
            "success": sum(1 for l in self.lessons if l.outcome == SUCCESS),
            "failure": sum(1 for l in self.lessons if l.outcome == FAILURE),
            "local": sum(1 for l in self.lessons if l.scope == LOCAL),
            "global": sum(1 for l in self.lessons if l.scope == GLOBAL),
        }

    def by_id(self, lesson_id: str) -> Optional[Lesson]:
        return self._by_id.get((lesson_id or "").strip().lower())

    def ids(self) -> List[str]:
        return [l.id for l in self.lessons]

    def fetch(self, ids: Sequence[str], count_use: bool = True) -> List[Lesson]:
        """
        Resolve chosen ids to lessons, in the order asked for. Unknown ids are
        skipped. This is the whole of what used to be Eq. 7.
        """
        out = []
        for lesson_id in ids or ():
            lesson = self.by_id(lesson_id)
            if lesson is None:
                continue
            if count_use:
                lesson.uses += 1
            out.append(lesson)
        return out

    # ------------------------------------------------------------------
    def catalog(self, limit: Optional[int] = None,
                chars: Optional[int] = None) -> List[str]:
        """
        The index the model reads, one line per lesson. limit=0 means the whole
        bank, which is the default and the intent: the model should see
        everything it has and decide for itself.

        When a limit does apply, order is importance then recency, so the
        entries hidden are the weakest rather than an arbitrary slice.
        """
        limit = int(self.cfg.catalog_max_lessons if limit is None else limit)
        chars = int(self.cfg.catalog_chars if chars is None else chars)
        if not self.lessons:
            return []
        items = self.lessons
        if limit > 0 and len(items) > limit:
            items = sorted(items, key=lambda l: (l.importance, l.step),
                           reverse=True)[:limit]
        # Stable, readable order for the model: oldest first, so ids it has seen
        # before stay in the same place between steps.
        items = sorted(items, key=lambda l: (l.step, l.id))
        return [l.catalog_line(chars) for l in items]

    def catalog_ids(self, limit: Optional[int] = None) -> List[str]:
        """The ids that appear in catalog(), for validating a selection."""
        limit = int(self.cfg.catalog_max_lessons if limit is None else limit)
        items = self.lessons
        if limit > 0 and len(items) > limit:
            items = sorted(items, key=lambda l: (l.importance, l.step),
                           reverse=True)[:limit]
        return [l.id for l in items]

    # ------------------------------------------------------------------
    def add_many(self, lessons: Sequence[Lesson]) -> int:
        """Insert new lessons, reinforcing rather than storing duplicates."""
        if not lessons:
            return 0
        self.stats["proposed"] += len(lessons)
        added = 0
        for lesson in lessons:
            dup = self._duplicate_of(lesson)
            if dup is not None:
                # Re-deriving something already recorded is independent
                # confirmation, not waste. Reinforce and drop the copy.
                self._bump(dup, float(self.cfg.reinforce_delta))
                self.stats["duplicates"] += 1
                continue
            self.lessons.append(lesson)
            self._by_id[lesson.id] = lesson
            added += 1
        self.stats["added"] += added
        self._evict()
        return added

    def reinforce(self, ids: Sequence[str], delta: Optional[float] = None) -> int:
        delta = float(self.cfg.reinforce_delta if delta is None else delta)
        hit = 0
        for lesson_id in ids or ():
            lesson = self.by_id(lesson_id)
            if lesson is not None:
                self._bump(lesson, delta)
                hit += 1
        return hit

    def _bump(self, lesson: Lesson, delta: float) -> None:
        lesson.importance = clamp_importance(lesson.importance + delta)
        lesson.confirmations += 1
        self.stats["reinforced"] += 1

    def _duplicate_of(self, lesson: Lesson) -> Optional[Lesson]:
        if lesson.id in self._by_id:
            return self._by_id[lesson.id]
        key = normalize_title(lesson.title)
        thr = float(getattr(self.cfg, "dedup_jaccard", 0.0) or 0.0)
        toks = lesson.tokens()
        best, best_j = None, 0.0
        for other in self.lessons:
            if other.outcome != lesson.outcome:
                continue
            if key and normalize_title(other.title) == key:
                return other          # same title once stopwords are stripped
            if thr <= 0:
                continue
            j = jaccard(toks, other.tokens())
            if j > best_j:
                best, best_j = other, j
        return best if best_j >= thr else None

    def _evict(self) -> None:
        cap = int(getattr(self.cfg, "max_lessons", 0) or 0)
        if cap <= 0 or len(self.lessons) <= cap:
            return
        order = sorted(range(len(self.lessons)),
                       key=lambda i: (self.lessons[i].importance,
                                      self.lessons[i].uses,
                                      self.lessons[i].step))
        drop = set(order[: len(self.lessons) - cap])
        self.lessons = [l for i, l in enumerate(self.lessons) if i not in drop]
        self._by_id = {l.id: l for l in self.lessons}
        self.stats["evicted"] += len(drop)

    # ------------------------------------------------------------------
    def usage_summary(self) -> str:
        """
        The §4.2 measurement, computed inline so a run reports its own retrieval
        concentration instead of needing a post-hoc script.
        """
        if not self.lessons:
            return "bank empty"
        uses = sorted((l.uses for l in self.lessons), reverse=True)
        total = sum(uses) or 1
        never = sum(1 for u in uses if u == 0)
        top5 = sum(uses[:5])
        return (f"{len(uses)} lessons, {100 * never / len(uses):.0f}% never "
                f"chosen, top-5 hold {100 * top5 / total:.0f}% of selections")

    # ------------------------------------------------------------------
    def attach(self, path) -> None:
        self._path = Path(path)
        if self._path.exists():
            self.load(self._path)

    def save(self, path=None) -> Optional[Path]:
        target = Path(path) if path is not None else self._path
        if target is None:
            return None
        payload = {"counts": self.counts(), "stats": self.stats,
                   "usage": self.usage_summary(),
                   "lessons": [l.to_dict() for l in self.lessons]}
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(target)          # atomic, so a crash cannot truncate it
        return target

    def load(self, path) -> int:
        data = json.loads(Path(path).read_text())
        self.lessons = [Lesson.from_dict(d) for d in data.get("lessons", [])]
        self._by_id = {l.id: l for l in self.lessons}
        return len(self.lessons)
