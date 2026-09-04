"""
Entropic objective with adaptive \beta.

Paper section 3.2: instead of optimizing the expected reward, we optimize

    J_\beta(\theta; s) = log E_{a~\pi_\theta}[exp(\beta · R(s,a))]

The gradient gives a reweighted policy gradient with weights

    w_\beta(a) = exp(\beta·R(a)) / E[exp(\beta·R)]
    A(a)   = w_\beta(a) - 1                # baselined advantage

As \beta → inf, this picks out the single best action. The trick is choosing \beta
adaptively for each parent

Implementation is done using ideas from discover/ttt_discover/rl/train.py::compute_advantages
under the `entropic_adaptive_beta` branch.
"""

import math
import numpy as np


def _kl_to_uniform(beta: float, rewards: np.ndarray) -> float:
    """
    KL(q_\beta || uniform) where q_\beta(i) ->  exp(\beta * r_i).
    Numerically stabilized by subtracting r.max().
    """
    K = len(rewards)
    if K <= 1:
        return 0.0
    logK = math.log(K)
    logits = beta * (rewards - rewards.max())
    # log q_\beta
    log_Z = np.log(np.exp(logits).sum())
    log_q = logits - log_Z
    q = np.exp(log_q)
    # KL(q || uniform) = sum q * (log q + log K)
    return float((q * (log_q + logK)).sum())


def entropic_adaptive_advantages(
    rewards: np.ndarray,
    gamma: float = math.log(2),
    beta_max: float = 1e6,
    n_bisect: int = 60,
    eps: float = 1e-12,
):
    """
    Compute leave-one-out entropic advantages for one group.

    rewards: shape (K,) — rewards of the K rollouts from the same parent.

    Returns:
      advantages: shape (K,) — what the policy gradient gets weighted by
      beta:       the temperature found by bisection

    All-equal-reward groups get a zero advantage vector (no gradient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = r.shape[0]

    if K < 2 or float(r.max() - r.min()) < eps:
        return np.zeros_like(r), 0.0

    # Step 1: find \beta with KL(q_\beta || uniform) = gamma via bisection
    lo, hi = 0.0, 1.0

    # If even hi=1 has KL > gamma, then \beta is in (0, 1)
    if _kl_to_uniform(hi, r) < gamma:
        # Need to grow hi until KL exceeds gamma
        while hi < beta_max and _kl_to_uniform(hi, r) < gamma:
            hi *= 2.0
        if _kl_to_uniform(hi, r) < gamma:
            # Saturated; \beta = beta_max (effectively argmax)
            beta = hi
        else:
            beta = None
    else:
        beta = None

    if beta is None:
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if _kl_to_uniform(mid, r) < gamma:
                lo = mid
            else:
                hi = mid
        beta = hi

    # Step 2: LOO entropic weights
    # w_n = e^{\beta(r_n - r_max)} / Z_{-n}
    # where Z_{-n} = (sum_m e^{\beta(r_m - r_max)} - e^{\beta(r_n - r_max)}) / (K-1)
    shift = r - r.max()
    e = np.exp(beta * shift)
    total = e.sum()
    Z_loo = (total - e) / (K - 1)
    w = e / (Z_loo + eps)
    advantages = w - 1.0

    return advantages, beta


def grpo_advantages(rewards: np.ndarray, eps: float = 1e-12, normalize_std: bool = True):
    """
    Standard GRPO group-relative advantages for one group.

    A_n = (r_n - mean(r)) / (std(r) + eps)   (or just r_n - mean(r) when
    normalize_std is False).

    Returns:
      advantages: shape (K,)
      scale:      the std used for normalization (0.0 for degenerate groups)

    All-equal-reward groups get a zero advantage vector (no gradient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = r.shape[0]
    if K < 2 or float(r.max() - r.min()) < eps:
        return np.zeros_like(r), 0.0
    centered = r - r.mean()
    std = float(r.std())
    if not normalize_std:
        return centered, std
    return centered / (std + eps), std


def upper_tail_advantages(
    rewards: np.ndarray,
    alpha: float = 0.2,
    lam: float = 0.5,
    eps: float = 1e-12,
):
    """
    GRPO advantage blended with a sparse right-tail (upper VaR) term.

    q      = Quantile(r, 1 - alpha)                       # upper VaR
    A_up_n = (r_n - q) / (std + eps)   if r_n > q else 0  # tail-only bonus
    A_n    = (1 - lam) * (r_n - mean) / (std + eps) + lam * A_up_n

    alpha is the tail mass (0.2 -> 80th percentile cutoff, 0.1 -> 90th).
    lam = 0 recovers plain GRPO; lam = 1 rewards only rollouts above the
    cutoff and gives no gradient to the rest.

    Returns:
      advantages: shape (K,)
      q:          the (1 - alpha)-quantile threshold used (0.0 for degenerate groups)

    All-equal-reward groups get a zero advantage vector (no gradient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = r.shape[0]
    if K < 2 or float(r.max() - r.min()) < eps:
        return np.zeros_like(r), 0.0
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    std = float(r.std())
    scale = std + eps
    q = float(np.quantile(r, 1.0 - alpha))
    base = (r - r.mean()) / scale
    upper = np.where(r > q, (r - q) / scale, 0.0)
    return (1.0 - lam) * base + lam * upper, q


def hill_estimates(rewards: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Hill estimates xi_k for k = 1 .. n-1 over the positive rewards:

        xi_k = (1/k) * sum_{i<=k} ln R_(i) - ln R_(k+1),   R_(1) >= R_(2) >= ...

    Returns an array indexed by k-1 (empty if fewer than two positive values).
    """
    r = np.asarray(rewards, dtype=np.float64)
    r = r[r > 0]
    n = r.shape[0]
    if n < 2:
        return np.zeros(0)
    logs = np.log(np.sort(r)[::-1] + eps)
    ks = np.arange(1, n)
    return np.cumsum(logs[:-1]) / ks - logs[1:]


def _gpd_fit(y: np.ndarray):
    """
    Maximum-likelihood GPD fit to excesses y >= 0 (Grimshaw's profile
    likelihood, no scipy). Returns (xi, sigma, loglik) or None when the
    excesses are degenerate (all zero).
    """
    y = np.asarray(y, dtype=np.float64)
    k = y.shape[0]
    ymax = float(y.max())
    ymean = float(y.mean())
    if k < 2 or ymax <= 0.0 or ymean <= 0.0:
        return None

    def profile(theta):
        # theta = xi / sigma. For theta -> 0 the GPD is exponential.
        if abs(theta) < 1e-12:
            return 0.0, ymean, -k * math.log(ymean) - k
        z = np.log1p(theta * y)
        xi = float(z.mean())
        if xi == 0.0 or theta * xi <= 0.0:
            return None
        sigma = xi / theta
        if sigma <= 0.0:
            return None
        ll = -k * math.log(sigma) - (1.0 + 1.0 / xi) * float(z.sum())
        return xi, sigma, ll

    lo = -1.0 / ymax
    # Candidate thetas: dense on (lo, 0) and log-spaced on (0, big).
    neg = lo * (1.0 - np.logspace(-8, 0, 60))[::-1]   # from just above lo up to ~0
    pos = np.logspace(-8, 4, 120) / ymean
    cands = np.concatenate([neg, [0.0], pos])
    best = None
    for th in cands:
        out = profile(float(th))
        if out is None or not np.isfinite(out[2]):
            continue
        if best is None or out[2] > best[3]:
            best = (out[0], out[1], float(th), out[2])
    if best is None:
        return None
    # Golden-section refinement around the best grid point.
    idx = int(np.argmin(np.abs(cands - best[2])))
    a = float(cands[max(idx - 1, 0)])
    b = float(cands[min(idx + 1, cands.shape[0] - 1)])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = profile(c)
        fd = profile(d)
        vc = fc[2] if fc is not None else -np.inf
        vd = fd[2] if fd is not None else -np.inf
        if vc < vd:
            a = c
        else:
            b = d
    out = profile(0.5 * (a + b))
    if out is not None and np.isfinite(out[2]) and out[2] >= best[3]:
        return out[0], out[1], out[2]
    return best[0], best[1], best[3]


def _gpd_cdf(y: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    if abs(xi) < 1e-8:
        return 1.0 - np.exp(-y / sigma)
    z = 1.0 + xi * y / sigma
    z = np.clip(z, 1e-300, None)
    return 1.0 - z ** (-1.0 / xi)


def evt_adaptive_alpha(
    rewards: np.ndarray,
    k_min: int = 5,
    max_tail_frac: float = 0.5,
    rel_tie_tol: float = 1e-9,
    positive_only: bool = False,
):
    """
    Extreme-value tail cutoff for one batch, chosen as correctly as the
    sample allows:

      1. Sort positive rewards descending. For every candidate tail size k
         (k_min <= k <= max_tail_frac * n) take threshold u = R_(k+1) and the
         excesses y_i = R_(i) - u, i <= k (the Pickands-Balkema-de Haan
         regime: excesses over a high threshold follow a GPD).
      2. Fit the GPD (xi, sigma) by maximum likelihood. xi may be negative,
         which is the right regime for bounded rewards; the Hill estimator
         alone cannot express that.
      3. Pick k* minimising the Kolmogorov-Smirnov distance between the
         empirical excess CDF and the fitted GPD (Clauset-Shalizi-Newman
         threshold selection). That is the point where the data actually
         start behaving like a tail.
      4. alpha = k* / G, threshold = R_(k*+1); the tail is everything above.
         G is the full batch size, or only the positive (valid) rollouts when
         positive_only=True (hurdle / zero-inflated view: alpha conditional
         on clearing the zero hurdle). The fit itself always uses r > 0.

    Degenerate batches: if every candidate tail is a set of exact ties (no
    excess), EVT does not apply and the tail is taken to be the rollouts tied
    at the maximum. Fewer than k_min + 2 positive rewards give alpha = 0.

    Returns (alpha, k_star, threshold, info) with info holding xi_gpd,
    sigma, ks, xi_hill (Hill estimate at k*), method, and the Hill array.
    """
    r_all = np.asarray(rewards, dtype=np.float64)
    G_total = int(r_all.shape[0])
    pos = r_all[r_all > 0]
    n = int(pos.shape[0])
    G = n if positive_only else G_total
    top = float(r_all.max()) if G_total else 0.0
    info = {"xi_gpd": float("nan"), "sigma": float("nan"), "ks": float("nan"),
            "xi_hill": float("nan"), "method": "none", "hill": np.zeros(0)}
    if G_total == 0 or n == 0:
        return 0.0, 0, top, info

    desc = np.sort(pos)[::-1]
    hill = hill_estimates(desc)
    info["hill"] = hill

    k_max = min(n - 1, int(max_tail_frac * n))
    if n < k_min + 2 or k_max < k_min:
        # Too few positives for a tail fit: tail = ties at the maximum.
        n_tie = int(np.sum(desc >= desc[0] * (1.0 - rel_tie_tol)))
        k = min(n_tie, max(n - 1, 1))
        thr = float(desc[min(k, n - 1)])
        info["method"] = "ties_at_max"
        return k / G, k, thr, info

    best = None  # (ks, k, xi, sigma)
    for k in range(k_min, k_max + 1):
        u = desc[k]
        y = desc[:k] - u
        if float(y.max()) <= 0.0:
            continue  # all tied at the threshold: no excesses to fit
        fit = _gpd_fit(y)
        if fit is None:
            continue
        xi, sigma, _ll = fit
        ys = np.sort(y)
        F = _gpd_cdf(ys, xi, sigma)
        i = np.arange(1, k + 1)
        ks = float(np.max(np.maximum(np.abs(i / k - F), np.abs((i - 1) / k - F))))
        if not np.isfinite(ks):
            continue
        if best is None or ks < best[0]:
            best = (ks, k, xi, sigma)

    if best is None:
        n_tie = int(np.sum(desc >= desc[0] * (1.0 - rel_tie_tol)))
        k = min(n_tie, n - 1)
        thr = float(desc[min(k, n - 1)])
        info["method"] = "ties_at_max"
        return k / G, k, thr, info

    ks, k_star, xi, sigma = best
    info.update({"xi_gpd": float(xi), "sigma": float(sigma), "ks": float(ks),
                 "xi_hill": float(hill[k_star - 1]) if hill.shape[0] >= k_star else float("nan"),
                 "method": "gpd_ks"})
    return k_star / G, int(k_star), float(desc[k_star]), info


def hill_adaptive_alpha(
    rewards: np.ndarray,
    k_min: int = 5,
    max_tail_frac: float = 0.5,
    window: int | None = None,
    c: float = 1.0,
):
    """
    Plain Hill-plateau rule, kept for comparison with evt_adaptive_alpha.

    k* is the first k >= k_min whose next `window` Hill estimates all lie
    within c standard errors (SE_k = xi_k / sqrt(k)) of xi_k, i.e. the
    estimator has stabilised relative to its own sampling noise. Falls back
    to the window with the smallest spread when no plateau exists.

    Returns (alpha, k_star, threshold, xi) where xi is the Hill array.
    """
    r_all = np.asarray(rewards, dtype=np.float64)
    G = int(r_all.shape[0])
    pos = r_all[r_all > 0]
    n = int(pos.shape[0])
    if G == 0 or n < k_min + 2:
        return 0.0, 0, float(r_all.max()) if G else 0.0, np.zeros(0)
    desc = np.sort(pos)[::-1]
    xi = hill_estimates(desc)
    k_max = max(k_min, min(n - 1, int(max_tail_frac * n)))
    if window is None:
        window = max(3, n // 20)
    k_star = None
    for k in range(k_min, k_max - window + 1):
        seg = xi[k - 1:k - 1 + window]
        se = max(abs(seg[0]) / math.sqrt(k), 1e-12)
        if np.all(np.abs(seg - seg[0]) <= c * se):
            k_star = k
            break
    if k_star is None:
        best, best_sd = k_min, np.inf
        for k in range(k_min, max(k_min, k_max - window) + 1):
            sd = float(np.std(xi[k - 1:k - 1 + window]))
            if sd < best_sd:
                best, best_sd = k, sd
        k_star = best
    k_star = int(min(max(k_star, 1), k_max))
    return k_star / G, k_star, float(desc[min(k_star, n - 1)]), xi


def gini_coefficient(rewards: np.ndarray, eps: float = 1e-12,
                     include_zeros: bool = False) -> float:
    """
    Gini of the rewards:

        Gini = sum_i sum_j |r_i - r_j| / (2 n sum_i r_i)

    Over the positive rewards r+ by default; include_zeros=True keeps the
    zero (invalid) rollouts in the sum, which pushes Gini toward 1 when most
    of the batch is invalid. 1 -> one reward dominates, 0 -> all equal.
    Returns 0.0 when fewer than two rewards remain or the total is zero.
    """
    r = np.asarray(rewards, dtype=np.float64)
    r = r[r >= 0] if include_zeros else r[r > 0]
    n = r.shape[0]
    if n < 2:
        return 0.0
    total = float(r.sum())
    if total <= eps:
        return 0.0
    # O(n log n) form of the double sum via sorted ranks.
    srt = np.sort(r)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * srt)) / (n * total) - (n + 1.0) / n)


def gini_adaptive_alpha(
    rewards: np.ndarray,
    alpha_base: float = 0.2,
    gamma: float = 1.0,
    positive_only: bool = True,
):
    """
    Gini-modulated upper-tail mass:

        alpha = alpha_base * (1 - Gini) ** gamma

    positive_only=True (default): Gini over r+ and k = round(alpha * n_pos),
    the hurdle view where alpha is conditional on a valid rollout.
    positive_only=False: Gini over all rewards, zeros included, and
    k = round(alpha * G) over the whole batch.

    High inequality (Gini -> 1) shrinks alpha toward the single best rollout;
    low inequality (Gini -> 0) lets alpha grow back to alpha_base.

    Returns (alpha, k, threshold, gini) where k = round(alpha * G) is the
    number of tail rollouts (at least 1 when any positive reward exists) and
    threshold is R_(k+1), the reward above which rollouts count as the tail.
    """
    r_all = np.asarray(rewards, dtype=np.float64)
    G_total = r_all.shape[0]
    pos = r_all[r_all > 0]
    if G_total == 0 or pos.shape[0] == 0:
        return 0.0, 0, float(r_all.max()) if G_total else 0.0, 0.0
    G = pos.shape[0] if positive_only else G_total
    gini = gini_coefficient(pos if positive_only else r_all,
                            include_zeros=not positive_only)
    alpha = float(alpha_base) * (1.0 - gini) ** float(gamma)
    alpha = float(min(max(alpha, 0.0), 1.0))
    k = int(max(1, round(alpha * G)))
    desc = np.sort(pos)[::-1]
    threshold = float(desc[min(k, desc.shape[0] - 1)])
    return alpha, k, threshold, gini


if __name__ == "__main__":
    # Case 1: all rewards equal -> zero advantage
    r = np.array([0.5, 0.5, 0.5, 0.5])
    a, b = entropic_adaptive_advantages(r)
    print(f"all equal -> adv={a}, beta={b}")

    # Case 2: one big outlier -> outlier gets the positive signal
    r = np.array([0.1, 0.1, 0.1, 0.1, 2.5])
    a, b = entropic_adaptive_advantages(r)
    print(f"one outlier -> adv={a}, beta={b:.4f}")
    print(f"  sum adv (should ~= 0 in expectation but is LOO so not exactly): {a.sum():.4f}")

    # Case 3: small differences -> \beta grows large, still concentrates on best
    r = np.array([1.0, 1.001, 1.002, 1.003])
    a, b = entropic_adaptive_advantages(r)
    print(f"small diffs -> adv={a}, beta={b:.4f}")

    # Case 4: large spread -> \beta is small (KL budget hits fast)
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    a, b = entropic_adaptive_advantages(r)
    print(f"big spread -> adv={a}, beta={b:.4f}")
