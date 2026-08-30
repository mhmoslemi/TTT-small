"""
Dataset-level signals that feed the global node logit.

Two independent estimates of how good a node is *relative to the whole archive*,
rather than relative to its siblings:

    rank        where its subtree max sits in the ordering of all subtree maxima
    comparison  an Elo rating from pairwise judgements (optional)

both standardized, then blended:

    L_D(s) = alpha * R~(s) + (1 - alpha) * E~(s)

One caveat worth knowing when you tune alpha. Standardizing a rank gives you a
normalized quantile: over N nodes the ranks are a fixed permutation of 1..N, so
the mean and spread depend only on N and R~ lands in roughly +/-1.73 whatever
your reward distribution looks like. A standardized Elo rating has real tails.
So alpha blends two differently-shaped signals rather than interpolating
between like scales -- alpha = 0.5 is not "half of each" in any strict sense.
"""

from typing import Optional, Sequence

import numpy as np


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ascending ranks starting at 1; tied values share their average rank."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=float)

    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)   # mean of ranks i+1 .. j
        i = j
    return ranks


def standardize(values: Sequence[float]) -> np.ndarray:
    """(x - mean) / std, with a degenerate spread giving zeros rather than NaN."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def rank_signal(subtree_maxima: Sequence[float]) -> np.ndarray:
    """
    R~(s): standardized rank of W_m over the archive.

    Ranks rather than raw values, so the signal does not care about your reward
    scale, outliers, or how a particular problem happens to be calibrated.
    """
    return standardize(average_ranks(subtree_maxima))


def blend(rank_sig: Sequence[float], comparison_sig: Optional[Sequence[float]],
          alpha: float) -> np.ndarray:
    """L_D(s) = alpha * R~(s) + (1 - alpha) * E~(s)."""
    rank_arr = np.asarray(rank_sig, dtype=float)
    if comparison_sig is None or len(comparison_sig) == 0:
        return float(alpha) * rank_arr
    other = np.asarray(comparison_sig, dtype=float)
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * rank_arr + (1.0 - a) * other


def minmax(values: Sequence[float]) -> np.ndarray:
    """Rescale to [0, 1] by the observed spread; constant input gives zeros."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def softmax(logits: Sequence[float], tau: float = 1.0) -> np.ndarray:
    """Temperature softmax, stabilized, with a uniform fallback."""
    arr = np.asarray(logits, dtype=float)
    if arr.size == 0:
        return arr
    z = arr / max(float(tau), 1e-6)
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(arr, 1.0 / arr.size)
    return e / total
