"""
The clipped policy-gradient objective (Eq. 10, 11).

    rho_{i,l}(theta) = pi_theta(y_il | x_p, y_i<l) / pi_thetabar(y_il | ...)

    L(theta) = -(1/N_t) sum_p sum_{i in g_p} sum_l
                   min( rho A , clip(rho, 1-eps, 1+eps) A )
               + eta_KL E_{c ~ C_t} [ D_KL( pi_theta(.|c) || pi_theta0(.|c) ) ]

N_t is the total sampled token count over the WHOLE batch, so these functions
return unnormalized sums and the trainer divides once by N_t. That matters when
gradients are accumulated across microbatches: normalizing per microbatch would
weight a short response the same as a long one and silently change the objective.

theta_0 is the policy before any test-time adaptation -- the backbone with the
LoRA adapter disabled -- not the previous step's policy. The KL anchors the run
to where it started rather than damping step-to-step movement.

torch is imported lazily so the rest of the framework imports without it.
"""


def policy_loss_sum(logp_new, logp_old, advantages, clip_epsilon: float):
    """
    Summed token-level PPO surrogate over one microbatch.

    With one update per rollout batch theta == thetabar on entry, so rho == 1
    and the clip is inactive; the gradient still flows through logp_new. The
    machinery earns its keep once rl.updates_per_step > 1.
    """
    import torch

    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.min(unclipped, clipped).sum()


def kl_sum(logp_policy, logp_reference):
    """
    Summed D_KL(pi_theta || pi_theta_0), k3 estimator:

        exp(d) - d - 1,   d = log pi_ref - log pi_theta

    Non-negative by construction and lower variance than the naive -d, which can
    go negative on a finite sample and so reward drifting away from theta_0.
    """
    import torch

    diff = logp_reference - logp_policy
    return (torch.exp(diff) - diff - 1.0).sum()


def microbatch_loss(logp_new, logp_old, logp_reference, advantages,
                    clip_epsilon: float, kl_coef: float, normalizer: float):
    """
    One microbatch's contribution to Eq. 11, already divided by N_t so the
    caller can simply .backward() each microbatch in turn.

    Returns (loss, parts) with parts detached, as unnormalized sums, so a caller
    can aggregate them across microbatches and divide once for reporting.
    """
    import torch

    policy = policy_loss_sum(logp_new, logp_old, advantages, clip_epsilon)
    if kl_coef and logp_reference is not None:
        kl = kl_sum(logp_new, logp_reference)
    else:
        kl = torch.zeros((), device=logp_new.device, dtype=logp_new.dtype)

    total = (policy + float(kl_coef) * kl) / max(float(normalizer), 1.0)
    parts = {
        "policy_loss_sum": float(policy.detach()),
        "kl_sum": float(kl.detach()),
        "tokens": int(logp_new.numel()),
    }
    return total, parts
