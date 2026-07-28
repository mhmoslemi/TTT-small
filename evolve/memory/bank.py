"""
The memory bank M (Sec. 2.2).

Updated additively -- M_t = M_{t-1} u L+_t u L-_t -- so a lesson learned early
stays available for the whole run. memory.max_bank_size is an operational
guard rail, not part of the method: at 0 (the default) nothing is ever evicted.

Retrieval is top-m cosine over lesson embeddings against the parent state
(Eq. 7); the hits are inserted into the generation prompt.
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from core.types import Lesson
from memory.retrieval import build_embedder, top_m_by_cosine


class MemoryBank:
    def __init__(self, cfg, embedder=None, backbone=None):
        self.cfg = cfg
        self.embedder = embedder or build_embedder(cfg, backbone=backbone)
        self.lessons: List[Lesson] = []
        self._matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.lessons)

    # ------------------------------------------------------------------
    def add(self, lessons: Sequence[Lesson]) -> int:
        """Embed and append. Returns how many were stored."""
        fresh = [l for l in lessons if (l.title or l.summary or l.body)]
        if not fresh:
            return 0

        vectors = self.embedder.encode([l.embed_text() for l in fresh])
        for lesson, vec in zip(fresh, vectors):
            lesson.embedding = vec
        self.lessons.extend(fresh)

        cap = int(self.cfg.max_bank_size or 0)
        if cap > 0 and len(self.lessons) > cap:
            # Evict oldest first; recent evidence is the more relevant.
            self.lessons = self.lessons[-cap:]

        self._matrix = None
        return len(fresh)

    # ------------------------------------------------------------------
    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.lessons:
                self._matrix = np.zeros((0, 1), dtype=np.float32)
            else:
                self._matrix = np.stack([l.embedding for l in self.lessons])
        return self._matrix

    def retrieve(self, query_text: str, m: Optional[int] = None) -> List[Lesson]:
        """R(p) = Top-m sim(e(p), e(l)) over the bank."""
        m = int(self.cfg.top_m if m is None else m)
        if not self.lessons or m <= 0:
            return []
        query = self.embedder.encode([query_text or ""])[0]
        idx = top_m_by_cosine(query, self._ensure_matrix(), m)
        return [self.lessons[i] for i in idx]

    def render(self, lessons: Sequence[Lesson]) -> str:
        if not lessons:
            return ""
        return "\n\n".join(f"{i}. {l.render()}" for i, l in enumerate(lessons, 1))

    # ------------------------------------------------------------------
    def counts(self) -> dict:
        out = {"total": len(self.lessons)}
        for lesson in self.lessons:
            out[lesson.outcome] = out.get(lesson.outcome, 0) + 1
        return out

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(
            [l.to_dict() for l in self.lessons], indent=2))

    def load(self, path) -> int:
        data = json.loads(Path(path).read_text())
        restored = [Lesson(**d) for d in data]
        self.lessons = []
        self._matrix = None
        return self.add(restored)
