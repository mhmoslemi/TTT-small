"""
Optional pairwise-comparison signal.

Scalar rewards cannot separate two nodes that scored the same, and sometimes
you have a comparator that can -- a human, a stronger model, a slow but
accurate simulator. Feed those judgements in here and the resulting ratings
become the second half of the global node logit.

    p_ij = 1 / (1 + 10^((E_j - E_i) / scale))
    E_i <- E_i + K (y_ij - p_ij)
    E_j <- E_j + K ((1 - y_ij) - (1 - p_ij))

with y in {0, 0.5, 1}. Entirely optional: leave it out and selection runs on
the rank signal alone (set alpha = 1.0 to be explicit about it).

`scale` is exposed because the convention matters. Classic Elo uses 400, where
a 400-point gap means a 10:1 expected score. The formulation in some papers
omits the denominator entirely, which is ~400x steeper -- a one-point gap
already implies p = 0.09 -- and needs a correspondingly tiny K. Pick one and
set K to match.
"""

import itertools
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .signals import standardize


class EloRatings:
    def __init__(self, k_factor: float = 24.0, initial_rating: float = 0.0,
                 scale: float = 400.0):
        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self.scale = max(float(scale), 1e-9)
        self.ratings: Dict[str, float] = {}
        self.matches: Dict[str, int] = {}

    def ensure(self, ids: Iterable[str]) -> None:
        """Everyone starts equal, so ratings only ever reflect actual results."""
        for i in ids:
            self.ratings.setdefault(i, self.initial_rating)
            self.matches.setdefault(i, 0)

    def expected(self, a: str, b: str) -> float:
        """Probability that a beats b under the current ratings."""
        ra = self.ratings.get(a, self.initial_rating)
        rb = self.ratings.get(b, self.initial_rating)
        exponent = (rb - ra) / self.scale
        if exponent > 30:      # guard the overflow a small scale makes reachable
            return 0.0
        if exponent < -30:
            return 1.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    def update(self, a: str, b: str, score_a: float) -> None:
        """score_a: 1.0 if a won, 0.0 if b won, 0.5 for a draw."""
        self.ensure([a, b])
        p_a = self.expected(a, b)
        self.ratings[a] += self.k_factor * (score_a - p_a)
        self.ratings[b] += self.k_factor * ((1.0 - score_a) - (1.0 - p_a))
        self.matches[a] += 1
        self.matches[b] += 1

    def play(self, pairs: Sequence[Tuple[str, str]], compare) -> int:
        """
        Run a schedule of comparisons.

        `compare(a_id, b_id)` returns 1.0 / 0.0 / 0.5. Returns how many were
        played; a comparator that returns None skips that pair.
        """
        played = 0
        for a, b in pairs:
            outcome = compare(a, b)
            if outcome is None:
                continue
            self.update(a, b, float(outcome))
            played += 1
        return played

    def standardized(self, ids: Sequence[str]) -> np.ndarray:
        """E~(s). Nodes that never played sit at the initial rating."""
        return standardize([self.ratings.get(i, self.initial_rating) for i in ids])

    def as_dict(self, ids: Sequence[str]) -> Dict[str, float]:
        """Standardized ratings keyed by id, ready to hand to DPUCT.select()."""
        return {i: float(v) for i, v in zip(ids, self.standardized(ids))}

    def standings(self, ids: Optional[Sequence[str]] = None) -> List[dict]:
        ids = list(ids if ids is not None else self.ratings)
        rows = [{"id": i,
                 "rating": self.ratings.get(i, self.initial_rating),
                 "matches": self.matches.get(i, 0)} for i in ids]
        rows.sort(key=lambda r: r["rating"], reverse=True)
        return rows


def build_pairings(ids: Sequence[str], mode: str = "round_robin",
                   num_matches: int = 60, rounds: int = 1,
                   rng: Optional[random.Random] = None) -> List[Tuple[str, str]]:
    """
    round_robin : every unordered pair, shuffled
    random      : up to num_matches distinct pairs
    neighbors   : rank-adjacent pairs only (len(ids) - 1 matches); assumes the
                  caller passed ids already sorted by their quality estimate

    Presentation order within a pair is randomized, so a comparator with a
    position bias does not systematically favour whoever is shown first.
    """
    rng = rng or random.Random()
    ids = list(ids)
    if len(ids) < 2:
        return []

    if mode == "neighbors":
        base = list(zip(ids[:-1], ids[1:]))
    elif mode == "random":
        base = list(itertools.combinations(ids, 2))
        rng.shuffle(base)
        base = base[: max(0, int(num_matches))]
    else:
        base = list(itertools.combinations(ids, 2))
        rng.shuffle(base)

    pairs: List[Tuple[str, str]] = []
    for _ in range(max(1, int(rounds))):
        chunk = list(base)
        rng.shuffle(chunk)
        pairs.extend((b, a) if rng.random() < 0.5 else (a, b) for a, b in chunk)
    return pairs
