"""
The memory bank M (Sec. 2.2, Eq. 7).

Additive: M_t = M_{t-1} + L+_t + L-_t. Nothing is rewritten, only appended,
deduplicated on entry, reinforced when a later step confirms it, and evicted
when the cap is hit.

Departures from the bare paper text, all switchable:

  dedup_threshold     a second line of defence. The extraction prompt already
                      shows the maker what is recorded and asks for new
                      material only, so this should now fire rarely; when it
                      does, the existing lesson is reinforced instead of the
                      duplicate being silently dropped.

  importance          the maker rates each lesson 1-5, and a confirmation from
                      a later step raises it. It shifts retrieval rank by at
                      most importance_weight and it decides eviction order, so
                      a finding that keeps proving out outlives one that was
                      guessed once.

  max_lessons         an unbounded bank is fine for 50 steps and not fine for
                      a long run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from memory.embedding import Embedder, cosine_scores
from memory.types import (FAILURE, IMPORTANCE_DEFAULT, IMPORTANCE_MAX, SUCCESS,
                          Lesson, clamp_importance)


class MemoryBank:
    def __init__(self, cfg, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.lessons: List[Lesson] = []
        self._matrix: Optional[np.ndarray] = None      # (n, d), row i <-> lessons[i]
        self._path: Optional[Path] = None
        self.stats = {"proposed": 0, "added": 0, "duplicates": 0,
                      "reinforced": 0, "evicted": 0, "retrievals": 0}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.lessons)

    def counts(self) -> Dict[str, int]:
        return {
            "total": len(self.lessons),
            "success": sum(1 for l in self.lessons if l.outcome == SUCCESS),
            "failure": sum(1 for l in self.lessons if l.outcome == FAILURE),
        }

    def by_id(self, lesson_id: str) -> Optional[Lesson]:
        lesson_id = (lesson_id or "").strip().lower()
        for l in self.lessons:
            if l.id == lesson_id:
                return l
        return None

    def catalog(self, limit: Optional[int] = None,
                chars: Optional[int] = None) -> List[str]:
        """
        The 'already recorded' list handed to the memory maker. Highest
        importance first, then most recent, so when the cap bites it is the
        weakest entries that are hidden rather than an arbitrary slice.
        Embeddings are not included.
        """
        limit = int(self.cfg.catalog_max_lessons if limit is None else limit)
        chars = int(self.cfg.catalog_chars if chars is None else chars)
        if limit <= 0 or not self.lessons:
            return []
        ranked = sorted(self.lessons,
                        key=lambda l: (l.importance, l.step), reverse=True)
        return [l.catalog_line(chars) for l in ranked[:limit]]

    # ------------------------------------------------------------------
    def add_many(self, lessons: Sequence[Lesson]) -> int:
        """Insert new lessons, reinforcing rather than storing duplicates."""
        if not lessons:
            return 0
        self.stats["proposed"] += len(lessons)

        vectors = self.embedder.encode([l.text_for_embedding() for l in lessons])
        added = 0
        for lesson, vec in zip(lessons, vectors):
            dup = self._duplicate_of(lesson, vec)
            if dup is not None:
                # The maker re-derived something already recorded. Treat that
                # as independent confirmation rather than throwing it away.
                self._bump(dup, float(self.cfg.reinforce_delta))
                self.stats["duplicates"] += 1
                continue
            lesson.embedding = [float(x) for x in vec]
            self.lessons.append(lesson)
            self._matrix = (vec[None, :] if self._matrix is None
                            else np.vstack([self._matrix, vec[None, :]]))
            added += 1

        self.stats["added"] += added
        self._evict()
        return added

    def reinforce(self, ids: Sequence[str], delta: Optional[float] = None) -> int:
        """Raise the importance of lessons the maker said this batch confirmed."""
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

    def _duplicate_of(self, lesson: Lesson, vec: np.ndarray) -> Optional[Lesson]:
        for l in self.lessons:
            if l.id == lesson.id:
                return l
        thr = float(getattr(self.cfg, "dedup_threshold", 0.0) or 0.0)
        if thr <= 0 or self._matrix is None or not self.lessons:
            return None
        same = np.array([l.outcome == lesson.outcome for l in self.lessons])
        if not same.any():
            return None
        scores = cosine_scores(vec, self._matrix)
        scores = np.where(same, scores, -np.inf)
        best = int(np.argmax(scores))
        return self.lessons[best] if scores[best] >= thr else None

    def _evict(self) -> None:
        cap = int(getattr(self.cfg, "max_lessons", 0) or 0)
        if cap <= 0 or len(self.lessons) <= cap:
            return
        # Lowest importance goes first, then least used, then oldest.
        order = sorted(range(len(self.lessons)),
                       key=lambda i: (self.lessons[i].importance,
                                      self.lessons[i].uses,
                                      self.lessons[i].step))
        drop = set(order[: len(self.lessons) - cap])
        keep = [i for i in range(len(self.lessons)) if i not in drop]
        self.lessons = [self.lessons[i] for i in keep]
        self._matrix = self._matrix[keep] if self._matrix is not None else None
        self.stats["evicted"] += len(drop)

    # ------------------------------------------------------------------
    def retrieve(self, query_text: str, m: Optional[int] = None) -> List[Lesson]:
        """
        Eq. 7 with an importance tilt.

        rank = cosine + w * (importance - 3) / 2

        so importance shifts the score by at most +/- w and a lesson left at the
        default importance of 3 ranks exactly as plain cosine would. Returned
        best-first, which is the order build_injection trims from.
        """
        m = int(self.cfg.top_m if m is None else m)
        if m <= 0 or not self.lessons or self._matrix is None:
            return []

        scope = getattr(self.cfg, "retrieval_scope", "both")
        idx = list(range(len(self.lessons)))
        if scope == "success":
            idx = [i for i in idx if self.lessons[i].outcome == SUCCESS]
        elif scope == "failure":
            idx = [i for i in idx if self.lessons[i].outcome == FAILURE]
        if not idx:
            return []

        query = self.embedder.encode_one(query_text)
        cos = cosine_scores(query, self._matrix[idx])

        w = float(getattr(self.cfg, "importance_weight", 0.0) or 0.0)
        imp = np.array([self.lessons[i].importance for i in idx], dtype=np.float32)
        score = cos + w * ((imp - IMPORTANCE_DEFAULT) / 2.0)

        floor = float(getattr(self.cfg, "min_similarity", 0.0) or 0.0)
        out = []
        for j in np.argsort(-score)[:m]:
            j = int(j)
            if cos[j] < floor:       # the floor is on similarity, not on rank
                break
            lesson = self.lessons[idx[j]]
            lesson.uses += 1
            out.append(lesson)
        self.stats["retrievals"] += 1
        return out

    # ------------------------------------------------------------------
    def attach(self, path) -> None:
        """Set the file this bank writes to, and load it if it already exists."""
        self._path = Path(path)
        if self._path.exists():
            self.load(self._path)

    def save(self, path=None) -> Optional[Path]:
        target = Path(path) if path is not None else self._path
        if target is None:
            return None
        payload = {
            "counts": self.counts(),
            "stats": self.stats,
            "lessons": [l.to_dict() for l in self.lessons],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(target)      # atomic, so a crash mid-write cannot truncate
        return target

    def load(self, path) -> int:
        data = json.loads(Path(path).read_text())
        self.lessons = [Lesson.from_dict(d) for d in data.get("lessons", [])]

        missing = [i for i, l in enumerate(self.lessons) if not l.embedding]
        if missing:
            vecs = self.embedder.encode(
                [self.lessons[i].text_for_embedding() for i in missing])
            for i, v in zip(missing, vecs):
                self.lessons[i].embedding = [float(x) for x in v]

        dims = {len(l.embedding) for l in self.lessons if l.embedding}
        if len(dims) > 1 or (dims and dims.pop() != self.embedder.dim):
            # A bank written by a different embedder cannot be scored against
            # this one. Re-embed rather than silently ranking on noise.
            vecs = self.embedder.encode(
                [l.text_for_embedding() for l in self.lessons])
            for l, v in zip(self.lessons, vecs):
                l.embedding = [float(x) for x in v]

        self._matrix = (np.array([l.embedding for l in self.lessons],
                                 dtype=np.float32)
                        if self.lessons else None)
        return len(self.lessons)
