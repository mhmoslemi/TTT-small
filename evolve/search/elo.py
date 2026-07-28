"""
Elo debate signal (Sec. 2.1).

The rank signal sees only scalar rewards, so two candidates with the same reward
are indistinguishable to it. The debate signal adds a qualitative comparison: an
LLM judge decides which of two nodes is the more promising continuation, and the
outcomes are folded into Elo ratings.

    p_ij = 1 / (1 + 10^((E_j - E_i) / scale))
    E_i <- E_i + K (y_ij - p_ij)
    E_j <- E_j + K ((1 - y_ij) - (1 - p_ij))

with y_ij in {0, 0.5, 1}. Ratings are then standardized over D to give E~(s).

`scale` is configurable because the paper writes the expectation as
10^(E_j - E_i) with no denominator, i.e. scale = 1.0, which is ~400x steeper
than conventional Elo: a one-point rating gap already implies p = 0.09. That is
self-consistent with a small K, but a habitual K = 24 under scale = 1.0 makes
ratings swing wildly on a single match. Default here is the classic 400.0; set
elo.scale = 1.0 for the literal equation.
"""

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from core.types import Verdict
from search.signals import standardize


class EloRatings:
    def __init__(self, k_factor: float = 24.0, initial_rating: float = 0.0,
                 scale: float = 400.0):
        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self.scale = max(float(scale), 1e-9)
        self.ratings: Dict[str, float] = {}
        self.matches: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def ensure(self, ids: Iterable[str]) -> None:
        """All nodes start at the same rating, so differences come only from play."""
        for i in ids:
            self.ratings.setdefault(i, self.initial_rating)
            self.matches.setdefault(i, 0)

    def expected(self, a: str, b: str) -> float:
        """p_ij: probability that a beats b."""
        ra = self.ratings.get(a, self.initial_rating)
        rb = self.ratings.get(b, self.initial_rating)
        exponent = (rb - ra) / self.scale
        # Guard the 10**x overflow that scale=1.0 makes reachable.
        if exponent > 30:
            return 0.0
        if exponent < -30:
            return 1.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    def update(self, a: str, b: str, y_a: float) -> None:
        self.ensure([a, b])
        p_a = self.expected(a, b)
        k = self.k_factor
        self.ratings[a] += k * (y_a - p_a)
        self.ratings[b] += k * ((1.0 - y_a) - (1.0 - p_a))
        self.matches[a] += 1
        self.matches[b] += 1

    def apply(self, verdicts: Iterable[Verdict]) -> int:
        n = 0
        for v in verdicts:
            self.update(v.node_a, v.node_b, v.y)
            n += 1
        return n

    # ------------------------------------------------------------------
    def standardized(self, ids: Sequence[str]) -> np.ndarray:
        """
        E~(s) over the given ids. Nodes that never played sit at the initial
        rating, so they standardize to whatever the unplayed population implies
        -- deliberately neutral rather than penalised.
        """
        raw = [self.ratings.get(i, self.initial_rating) for i in ids]
        return standardize(raw)

    def standings(self, ids: Sequence[str]) -> List[dict]:
        rows = [{"id": i,
                 "rating": self.ratings.get(i, self.initial_rating),
                 "matches": self.matches.get(i, 0)} for i in ids]
        rows.sort(key=lambda r: r["rating"], reverse=True)
        return rows
