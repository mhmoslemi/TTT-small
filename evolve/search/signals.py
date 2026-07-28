"""
Dataset-level signals feeding the global node logit (Sec. 2.1).

The rank statistic is R_rk(s) = rank_D(W_m(s)), ascending, ties averaged, then
standardized over D:

    R~(s) = (R_rk(s) - mu_D,R) / sigma_D,R

Ranks rather than raw rewards, so the signal does not depend on reward scale,
outliers, or how a problem happens to calibrate its metric.

Note on what standardizing a rank does: over N nodes the ranks are a fixed
permutation of 1..N, so mu and sigma are determined by N alone and R~ is a
normalized quantile bounded near +/-1.73 whatever the reward distribution is.
The Elo signal it is blended with in Eq. 3 has genuine tails, so alpha mixes
two differently-shaped signals rather than interpolating between like scales.
"""

from typing import Sequence

import numpy as np


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ascending ranks starting at 1, ties sharing their average rank."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=float)

    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_vals = arr[order]

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # positions i..j-1 are tied -> average of ranks (i+1)..j
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def standardize(values: Sequence[float]) -> np.ndarray:
    """(x - mean) / std. A degenerate spread yields all zeros, not NaN."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def rank_signal(subtree_maxima: Sequence[float]) -> np.ndarray:
    """R~(s) over the archive."""
    return standardize(average_ranks(subtree_maxima))


def global_node_logits(rank_sig: Sequence[float], elo_sig: Sequence[float],
                       alpha: float) -> np.ndarray:
    """
    Eq. 3:  L_D(s) = alpha * R~(s) + (1 - alpha) * E~(s)

    alpha = 1 uses the subtree-reward rank alone, alpha = 0 the Elo rating alone.
    """
    rank_arr = np.asarray(rank_sig, dtype=float)
    if elo_sig is None or len(elo_sig) == 0:
        return float(alpha) * rank_arr
    elo_arr = np.asarray(elo_sig, dtype=float)
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * rank_arr + (1.0 - a) * elo_arr


def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    """Map to [0, 1] by the observed spread; constant input maps to zeros."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
