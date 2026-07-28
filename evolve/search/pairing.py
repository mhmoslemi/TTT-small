"""Which node pairs to compare in the Elo debate each step."""

import itertools
import random
from typing import List, Optional, Sequence, Tuple


def build_pairings(ids: Sequence[str], mode: str = "round_robin",
                   num_matches: int = 60, rounds: int = 1,
                   rng: Optional[random.Random] = None) -> List[Tuple[str, str]]:
    """
    round_robin : every unordered pair, shuffled
    random      : up to num_matches distinct pairs
    neighbors   : compare rank-adjacent candidates only (len(ids) - 1 matches).
                  Cheapest useful schedule -- assumes `ids` arrives sorted by
                  the caller's quality estimate.
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
    else:  # round_robin
        base = list(itertools.combinations(ids, 2))
        rng.shuffle(base)

    pairs: List[Tuple[str, str]] = []
    for _ in range(max(1, int(rounds))):
        chunk = list(base)
        rng.shuffle(chunk)
        # Randomize presentation order so the judge's position bias does not
        # systematically favour whoever is shown first.
        pairs.extend((b, a) if rng.random() < 0.5 else (a, b) for a, b in chunk)
    return pairs
