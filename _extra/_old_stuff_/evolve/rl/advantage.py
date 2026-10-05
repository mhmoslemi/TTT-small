"""
Advantages for the test-time RL update (Sec. 2.3).

Two signals, combined per token:

  Group-relative reward (Eq. 8). Within a group g_p of responses sharing one
  prompt, tilt the rewards exponentially:

      omega_i = G_p exp(beta r_i) / sum_j exp(beta r_j)      A_rew_i = omega_i - 1

  omega is G_p times a softmax, so sum_i omega_i = G_p and the advantages sum to
  zero by construction -- a baseline falls out for free, with no std division
  that would flatten the tilt back out. As beta -> infinity the weight
  concentrates on the best response (advantage -> G_p - 1, the rest -> -1),
  which is the max-seeking limit the discovery objective wants.

  Note beta = 0 gives omega_i = 1 and A_rew_i = 0 exactly -- no signal at all,
  not mean-seeking. The mean-seeking regime is small positive beta, where
  A_rew_i ~ beta (r_i - r_bar).

  Feedback-based failure signal (Eq. 9). For a failed response, re-run the
  rollout policy with the verifier's feedback spliced into the context and read
  off the per-token log-prob difference:

      A_fb_{i,l} = log q(y_il | x_p, f_i, y_i<l) - log pi(y_il | x_p, y_i<l)

  Same weights either side; the only difference is whether the feedback is in
  context. Positive where the feedback makes the sampled token more plausible,
  negative where it makes it less. Detached -- it is a target, not a path for
  gradients.

Combined:  A_{i,l} = A_rew_i + lambda_f d_i A_fb_{i,l},  d_i = 1{response failed}
"""

from typing import Optional, Sequence

import numpy as np


def group_relative_advantages(rewards: Sequence[float], beta: float) -> np.ndarray:
    """Eq. 8. Returns one scalar advantage per response in the group."""
    r = np.asarray(rewards, dtype=np.float64)
    g = r.size
    if g == 0:
        return np.zeros(0, dtype=np.float64)
    if g == 1:
        # A single response has nothing to be relative to.
        return np.zeros(1, dtype=np.float64)

    z = float(beta) * r
    z -= z.max()                                   # stabilize before exp
    e = np.exp(z)
    total = e.sum()
    if not np.isfinite(total) or total <= 0:
        return np.zeros(g, dtype=np.float64)
    omega = g * e / total
    return omega - 1.0


def feedback_advantages(teacher_logprobs: Sequence[float],
                        policy_logprobs: Sequence[float]) -> np.ndarray:
    """Eq. 9, per token. Both inputs come from the same frozen parameters."""
    teacher = np.asarray(teacher_logprobs, dtype=np.float64)
    policy = np.asarray(policy_logprobs, dtype=np.float64)
    if teacher.shape != policy.shape:
        raise ValueError(
            f"teacher/policy logprob shapes differ: {teacher.shape} vs {policy.shape}")
    return teacher - policy


def combine(a_rew: float, a_fb: Optional[np.ndarray], failed: bool,
            lambda_f: float, num_tokens: int) -> np.ndarray:
    """
    A_{i,l} = A_rew_i + lambda_f d_i A_fb_{i,l}, broadcast over the response.

    A successful response carries only its group-relative reward; a failed one
    also carries the dense token-level correction.
    """
    out = np.full(num_tokens, float(a_rew), dtype=np.float64)
    if failed and a_fb is not None and lambda_f != 0.0:
        fb = np.asarray(a_fb, dtype=np.float64)
        if fb.size != num_tokens:
            raise ValueError(
                f"feedback advantage has {fb.size} tokens, response has {num_tokens}")
        out = out + float(lambda_f) * fb
    return out


def clip_advantages(advantages: np.ndarray, limit: Optional[float]) -> np.ndarray:
    """
    Operational guard rail, not part of the paper.

    A_fb is a difference of log-probabilities and is unbounded below: one token
    the feedback-conditioned teacher considers near-impossible can produce a
    -30 advantage that dominates the whole batch. Set rl.advantage_clip to 0 to
    disable.
    """
    if not limit or limit <= 0:
        return advantages
    return np.clip(advantages, -float(limit), float(limit))
