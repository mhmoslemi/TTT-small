"""Elo math + tournament pairing helpers for the Multi-Agent re-ranker.

Kept separate from orchestration so the numerical core has no I/O and is easy
to test in isolation.
"""
from __future__ import annotations
import itertools
import math
import random
from typing import Dict, List, Optional, Tuple


def expected_score(rating_a: float, rating_b: float) -> float:
    """Classic Elo expectation: probability that A beats B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_pair(ratings: Dict[str, float], a_id: str, b_id: str,
                score_a: float, k_factor: float) -> None:
    """Apply one Elo update in place.
    score_a is 1.0 if A won, 0.0 if B won, 0.5 for a draw."""
    ra, rb = ratings[a_id], ratings[b_id]
    ea = expected_score(ra, rb)
    eb = 1.0 - ea
    ratings[a_id] = ra + k_factor * (score_a - ea)
    ratings[b_id] = rb + k_factor * ((1.0 - score_a) - eb)


def build_pairings(ids: List[str], mode: str = "round_robin",
                   num_random_matches: int = 60, both_orders: bool = False,
                   rng: Optional[random.Random] = None) -> List[Tuple[str, str]]:
    """Return the list of (id_a, id_b) matches to play."""
    rng = rng or random.Random()
    if len(ids) < 2:
        return []
    if mode == "round_robin":
        pairs = list(itertools.combinations(ids, 2))
        if both_orders:
            pairs += [(b, a) for (a, b) in pairs]
        rng.shuffle(pairs)
        return pairs
    # "random": sample distinct unordered pairs without replacement, up to a cap.
    all_pairs = list(itertools.combinations(ids, 2))
    rng.shuffle(all_pairs)
    return all_pairs[: max(0, int(num_random_matches))]


def softmax_from_ratings(ratings: Dict[str, float], temp: float = 100.0
                         ) -> Dict[str, float]:
    """Turn Elo ratings into a probability distribution (sums to 1).
    `temp` is expressed in Elo points: a ~temp gap is a factor-e difference in
    weight. Empty/degenerate inputs yield a uniform distribution."""
    if not ratings:
        return {}
    ids = list(ratings.keys())
    vals = [ratings[i] for i in ids]
    temp = max(float(temp), 1e-6)
    m = max(vals)                                   # numerical stabilization
    exps = [math.exp((v - m) / temp) for v in vals]
    z = sum(exps)
    if z <= 0.0 or not math.isfinite(z):
        u = 1.0 / len(ids)
        return {i: u for i in ids}
    return {i: e / z for i, e in zip(ids, exps)}