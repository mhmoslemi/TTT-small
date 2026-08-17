"""
The memory bank M (Sec. 2.2, Eq. 7).

Additive by construction: M_t = M_{t-1} + L+_t + L-_t. Nothing is ever
rewritten, only appended, deduplicated on entry, and eventually evicted when
the cap is hit.

Two departures from the bare paper text, both deliberate and both switchable:

  dedup_threshold   the extractor is asked for L lessons every step whether or
                    not the step produced L lessons' worth of new evidence, so
                    a long run accumulates near-duplicates that then crowd out
                    the top-m slots. A new lesson whose cosine to an existing
                    same-outcome lesson exceeds the threshold is dropped and
                    the survivor's step is left alone. Set
                    memory_dedup_threshold to 0 to keep everything.

  max_lessons       an unbounded bank is fine for 50 steps and not fine for a
                    long run; eviction is oldest-first among the least-used, so
                    lessons that keep being retrieved survive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from memory.embedding import Embedder, cosine_scores
from memory.types import FAILURE, SUCCESS, Lesson


class MemoryBank:
    def __init__(self, cfg, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.lessons: List[Lesson] = []
        self._matrix: Optional[np.ndarray] = None      # (n, d), row i <-> lessons[i]
        self._path: Optional[Path] = None
        self.stats = {"proposed": 0, "added": 0, "duplicates": 0, "evicted": 0,
                      "retrievals": 0}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.lessons)

    def counts(self) -> Dict[str, int]:
        return {
            "total": len(self.lessons),
            "success": sum(1 for l in self.lessons if l.outcome == SUCCESS),
            "failure": sum(1 for l in self.lessons if l.outcome == FAILURE),
        }

    # ------------------------------------------------------------------
    def add_many(self, lessons: Sequence[Lesson]) -> int:
        """Insert new lessons, skipping duplicates. Returns how many landed."""
        if not lessons:
            return 0
        self.stats["proposed"] += len(lessons)

        vectors = self.embedder.encode([l.text_for_embedding() for l in lessons])
        added = 0
        for lesson, vec in zip(lessons, vectors):
            if self._is_duplicate(lesson, vec):
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

    def _is_duplicate(self, lesson: Lesson, vec: np.ndarray) -> bool:
        if any(l.id == lesson.id for l in self.lessons):
            return True
        thr = float(getattr(self.cfg, "dedup_threshold", 0.0) or 0.0)
        if thr <= 0 or self._matrix is None or not len(self.lessons):
            return False
        same = np.array([l.outcome == lesson.outcome for l in self.lessons])
        if not same.any():
            return False
        scores = cosine_scores(vec, self._matrix)
        return bool(scores[same].max() >= thr)

    def _evict(self) -> None:
        cap = int(getattr(self.cfg, "max_lessons", 0) or 0)
        if cap <= 0 or len(self.lessons) <= cap:
            return
        # Keep the most-used, break ties by recency. Rank ascending so the
        # first `drop` entries are the ones to lose.
        order = sorted(range(len(self.lessons)),
                       key=lambda i: (self.lessons[i].uses, self.lessons[i].step))
        drop = set(order[: len(self.lessons) - cap])
        keep = [i for i in range(len(self.lessons)) if i not in drop]
        self.lessons = [self.lessons[i] for i in keep]
        self._matrix = self._matrix[keep] if self._matrix is not None else None
        self.stats["evicted"] += len(drop)

    # ------------------------------------------------------------------
    def retrieve(self, query_text: str, m: Optional[int] = None) -> List[Lesson]:
        """Eq. 7: the top-m lessons by cosine similarity to the parent state."""
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
        scores = cosine_scores(query, self._matrix[idx])
        floor = float(getattr(self.cfg, "min_similarity", 0.0) or 0.0)

        order = np.argsort(-scores)[:m]
        out = []
        for j in order:
            if scores[j] < floor:
                break
            lesson = self.lessons[idx[int(j)]]
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
        raw = data.get("lessons", [])
        self.lessons = [Lesson.from_dict(d) for d in raw]

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
