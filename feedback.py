"""
Feedback-based failure signal (Sec. 2.3, Eq. 9).

A scalar reward says a rollout failed. It does not say which tokens caused the
failure. Eq. 9 recovers that by reusing the rollout policy as a
feedback-conditioned self-teacher:

    A^fb_{i,l} = log q_thetabar(y_{i,l} | x_p, f_i, y_{i,<l})
               - log pi_thetabar(y_{i,l} | x_p,      y_{i,<l})

Same weights on both sides. The only difference is whether the verifier's
complaint is in the context. Where the token becomes more likely once the model
can see the error message, the feedback endorses it; where it becomes less
likely, the feedback blames it. Detached, so it shapes the advantage and is
never differentiated through.

Combined per Sec. 2.3:

    A_{i,l} = A^rew_i + lambda_f * d_i * A^fb_{i,l}

with d_i the failure indicator, so successful rollouts are untouched.

ONE FORWARD PASS, NOT TWO. The second term above is log pi_thetabar, the
rollout policy. The trainer accumulates gradients over every example and calls
optimizer.step() once at the end, so throughout that loop theta == thetabar and
`cur_lp.detach()` from the existing forward IS log pi_thetabar. Only the
feedback-conditioned forward is new, and it runs under no_grad.

Cost: one extra prompt+response forward per selected FAILED rollout. By default
the teacher budget scales with the current rollout batch instead of being a
fixed count, so adaptive changes to G or K preserve the same compute fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

PREFIX = "feedback_"
_MISSING = object()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class FeedbackConfig:
    enabled: bool = False

    lambda_f: float = 0.2         # lambda_f at step 0
    # Annealing. anneal_steps=0 keeps lambda_f constant (Sec. 2.3 as written).
    # anneal_steps=10 with lambda_final=0 means the feedback term is fully off
    # from step 10 onward, and the teacher forward is skipped entirely once the
    # coefficient reaches zero, so the late steps cost exactly what a no-feedback
    # run costs.
    anneal_steps: int = 0
    anneal_shape: str = "linear"  # linear | cosine
    lambda_final: float = 0.0
    clip: float = 5.0             # clamp |A^fb| per token; 0 disables (paper has no clip)
    chars: int = 1200             # verifier text budget in the reprompt
    # 0 = automatic fraction of G*K; -1 = every failure; >0 = explicit cap.
    max_per_step: int = 0
    auto_fraction: float = 0.20
    include_constant_groups: bool = True
    inject_mode: str = "append"   # append | user_turn
    normalize: bool = False       # standardize A^fb within a response

    # --- adaptive repair budget ---
    # A fixed time schedule cannot tell whether failures are still the
    # bottleneck. When enabled, the scheduled coefficient is multiplied by a
    # validity-deficit controller: full below `validity_floor`, zero at/above
    # `validity_target`, linear between them.
    adaptive: bool = False
    validity_floor: float = 0.5
    validity_target: float = 0.9
    # Bound mean |feedback advantage| relative to the scalar reward advantage.
    # This is a cheap trust-region proxy: feedback may repair the proposal, but
    # cannot silently become the main scientific objective.
    max_reward_ratio: float = 0.0  # 0 disables the bound (legacy)
    reward_scale_floor: float = 0.25
    # 0 = automatic fraction of the step cap; -1 = unlimited; >0 = explicit.
    max_per_signature: int = 0
    auto_signature_fraction: float = 0.25

    @classmethod
    def from_dict(cls, d: Dict[str, Any], verbose: bool = True) -> "FeedbackConfig":
        d = dict(d or {})
        enabled = bool(d.get("feedback", False))

        if not enabled:
            defaults = cls()
            ignored = sorted(
                k for k, v in d.items()
                if k.startswith(PREFIX) and v is not None
                and getattr(defaults, _name(k), _MISSING) != v
            )
            if verbose and ignored:
                print(f"[feedback] disabled (--feedback not set); ignoring "
                      f"{len(ignored)} feedback_* key(s): {', '.join(ignored)}")
            return cls(enabled=False)

        kwargs: Dict[str, Any] = {"enabled": True}
        known = {f.name: f for f in fields(cls)}
        unknown = []
        for key, value in d.items():
            if not key.startswith(PREFIX) or value is None:
                continue
            name = _name(key)
            if name not in known or name == "enabled":
                unknown.append(key)
                continue
            target = known[name].type
            try:
                if target is bool or target == "bool":
                    value = _as_bool(value)
                elif target is int or target == "int":
                    value = int(value)
                elif target is float or target == "float":
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"[feedback] bad value for {key}: {value!r}")
            kwargs[name] = value

        if verbose and unknown:
            print(f"[feedback] unknown feedback_* key(s) ignored: "
                  f"{', '.join(sorted(unknown))}")

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.lambda_f < 0:
            raise ValueError("feedback_lambda must be >= 0")
        if self.lambda_final < 0:
            raise ValueError("feedback_lambda_final must be >= 0")
        if self.clip < 0:
            raise ValueError("feedback_clip must be >= 0")
        if self.chars < 0:
            raise ValueError("feedback_chars must be >= 0")
        if self.max_per_step < -1:
            raise ValueError("feedback_max_per_step must be -1, 0, or positive")
        if not 0.0 < self.auto_fraction <= 1.0:
            raise ValueError("feedback_auto_fraction must be in (0, 1]")
        if self.anneal_steps < 0:
            raise ValueError("feedback_anneal_steps must be >= 0")
        if self.anneal_shape not in ("linear", "cosine"):
            raise ValueError("feedback_anneal_shape must be linear|cosine")
        if self.inject_mode not in ("append", "user_turn"):
            raise ValueError("feedback_inject_mode must be append|user_turn")
        if not 0.0 <= self.validity_floor <= 1.0:
            raise ValueError("feedback_validity_floor must be in [0, 1]")
        if not 0.0 <= self.validity_target <= 1.0:
            raise ValueError("feedback_validity_target must be in [0, 1]")
        if self.validity_target <= self.validity_floor:
            raise ValueError(
                "feedback_validity_target must exceed feedback_validity_floor")
        if self.max_reward_ratio < 0:
            raise ValueError("feedback_max_reward_ratio must be >= 0")
        if self.reward_scale_floor < 0:
            raise ValueError("feedback_reward_scale_floor must be >= 0")
        if self.max_per_signature < -1:
            raise ValueError(
                "feedback_max_per_signature must be -1, 0, or positive")
        if not 0.0 < self.auto_signature_fraction <= 1.0:
            raise ValueError(
                "feedback_auto_signature_fraction must be in (0, 1]")

    def resolve_caps(self, groups: int, group_size: int) -> tuple[int, int]:
        """Return concrete (step, signature) caps for the current G and K.

        The defaults preserve the user's tuned G=5, K=16 values: 20% of 80 is
        16 teacher forwards, and 25% of that cap is 4 copies of one signature.
        Ceil keeps small experiments from accidentally turning feedback off.
        A returned zero means unlimited, matching select_balanced().
        """
        import math

        batch = max(0, int(groups)) * max(0, int(group_size))
        if self.max_per_step > 0:
            total_cap = int(self.max_per_step)
        elif self.max_per_step == 0:
            total_cap = (max(1, int(math.ceil(batch * self.auto_fraction)))
                         if batch else 0)
        else:  # -1: explicitly unlimited
            total_cap = 0

        if self.max_per_signature > 0:
            signature_cap = int(self.max_per_signature)
        elif self.max_per_signature == 0:
            basis = total_cap or batch
            signature_cap = (
                max(1, int(math.ceil(basis * self.auto_signature_fraction)))
                if basis else 0)
        else:  # -1: explicitly unlimited
            signature_cap = 0

        if total_cap > 0 and signature_cap > 0:
            signature_cap = min(signature_cap, total_cap)
        return total_cap, signature_cap

    def lambda_at(self, step: int) -> float:
        """
        The coefficient in force at this step.

        Keyed on the absolute step rather than advanced by a counter, so a step
        is reproducible on its own and a restart cannot land on a different
        point of the schedule than a fresh run would.
        """
        n = int(self.anneal_steps or 0)
        if n <= 0:
            return float(self.lambda_f)
        if step >= n:
            return float(self.lambda_final)
        frac = float(step) / float(n)
        if self.anneal_shape == "cosine":
            import math
            w = 0.5 * (1.0 + math.cos(math.pi * frac))
        else:
            w = 1.0 - frac
        return float(self.lambda_final + (self.lambda_f - self.lambda_final) * w)

    def schedule_preview(self, n: int = 12) -> str:
        if self.anneal_steps <= 0:
            return f"constant {self.lambda_f}"
        pts = [f"{s}:{self.lambda_at(s):.3f}" for s in range(0, min(n, self.anneal_steps + 2))]
        return "  ".join(pts)

    def effective_lambda(self, step: int, valid_fraction: float) -> float:
        """Scheduled coefficient, optionally gated by the observed validity."""
        base = self.lambda_at(step)
        if not self.adaptive:
            return base
        valid = max(0.0, min(1.0, float(valid_fraction)))
        if valid >= self.validity_target:
            return 0.0
        if valid <= self.validity_floor:
            return base
        scale = ((self.validity_target - valid)
                 / (self.validity_target - self.validity_floor))
        return base * scale

    def describe(self) -> str:
        if not self.enabled:
            return "feedback signal OFF"
        sched = (f"lambda_f={self.lambda_f}" if self.anneal_steps <= 0
                 else f"lambda_f {self.lambda_f}->{self.lambda_final} over "
                      f"{self.anneal_steps} steps ({self.anneal_shape})")
        cap = (f"auto {self.auto_fraction:.0%} of G*K"
               if self.max_per_step == 0
               else "all" if self.max_per_step < 0
               else str(self.max_per_step))
        sig_cap = (f"auto {self.auto_signature_fraction:.0%} of cap"
                   if self.max_per_signature == 0
                   else "all" if self.max_per_signature < 0
                   else str(self.max_per_signature))
        return (f"feedback signal ON  {sched}  "
                f"clip={self.clip or 'none'}  "
                f"constant_groups={'in' if self.include_constant_groups else 'out'}  "
                f"cap={cap}  signature-cap={sig_cap}"
                + (f"  adaptive-validity={self.validity_floor:.0%}-"
                   f"{self.validity_target:.0%}"
                   if self.adaptive else "")
                + (f"  fb/reward<={self.max_reward_ratio:.2f}"
                   if self.max_reward_ratio > 0 else ""))


def _name(key: str) -> str:
    """feedback_lambda maps to the field lambda_f; the rest strip the prefix."""
    name = key[len(PREFIX):]
    return "lambda_f" if name == "lambda" else name


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# ----------------------------------------------------------------------
# reprompt(x_p, f_i)
# ----------------------------------------------------------------------
_HEADER = "## Verifier output for the attempt below"

_PREAMBLE = (
    "The program produced in response to the instructions above was executed "
    "and rejected. The verifier reported the following. Treat it as ground "
    "truth about what went wrong."
)

_CLOSER = (
    "Write the program again, correcting the cause of this failure while "
    "keeping everything that was already right."
)


def format_feedback(msg: str, stdout: str, limit: int = 1200) -> str:
    """
    f_i. The verifier's tag plus the tail of stdout, which is where the
    traceback lives; the head is usually progress noise.
    """
    parts = []
    if msg:
        parts.append(f"verifier: {msg}")
    else:
        parts.append("verifier: invalid (no message)")
    tail = (stdout or "").strip()
    if tail:
        parts.append("stdout tail:\n" + tail[-limit:])
    return "\n".join(parts)


def build_reprompt(messages: List[Dict], feedback_text: str,
                   mode: str = "append") -> List[Dict]:
    """
    x_p augmented with f_i, per Sec. 2.3. Returns a NEW list.

    Two modes, and the difference matters more than it looks:

      append     the feedback is folded into the same user message. The teacher
                 sees one instruction that happens to mention a past failure.

      user_turn  the feedback arrives as a separate later user turn. The
                 teacher sees a conversation where its work was rejected.

    `append` is the default because the response y_i was itself generated from
    a single-user-message prompt, so the teacher's context stays closer to the
    rollout's context and log q - log pi isolates the feedback rather than also
    picking up a change in conversational shape.
    """
    block = f"{_HEADER}\n\n{_PREAMBLE}\n\n{feedback_text}\n\n{_CLOSER}"
    out = [dict(m) for m in messages]
    if not feedback_text:
        return out

    if mode == "user_turn":
        return out + [{"role": "user", "content": block}]

    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = out[i]["content"] + "\n\n" + block
            return out

    out.append({"role": "user", "content": block})
    return out


def render_chat(tokenizer, messages: List[Dict]) -> str:
    """Same call and fallback the trainer uses for the rollout prompt."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def is_failure(res, fail_score: float = 0.0) -> bool:
    """
    d_i = 1{r_i = 0 or the attempt failed}. `valid` is checked as well as the
    reward so a problem with a negative fail_score cannot mislabel an invalid
    rollout as a success.
    """
    return not (bool(getattr(res, "valid", False))
                and float(getattr(res, "reward", 0.0)) > float(fail_score))


def select_capped(indices: Sequence[int], cap: int) -> List[int]:
    """
    Even stride rather than a prefix. Taking the first `cap` failures would draw
    them all from the first group or two, and a cap is supposed to reduce cost
    without changing which part of the batch the signal comes from.
    """
    idx = list(indices)
    if cap <= 0 or len(idx) <= cap:
        return idx
    step = len(idx) / float(cap)
    return [idx[int(i * step)] for i in range(cap)]


def select_balanced(indices: Sequence[int], signatures: Sequence[str],
                    total_cap: int = 0, per_signature_cap: int = 0) -> List[int]:
    """Round-robin failure selection so one repeated crash cannot dominate."""
    buckets = {}
    for idx in indices:
        sig = signatures[idx] if idx < len(signatures) else "unknown"
        buckets.setdefault(sig or "unknown", []).append(idx)
    # Spread each signature's candidates across the whole batch before the
    # cross-signature round robin. Otherwise a common failure would still take
    # all of its retained examples from the earliest parent groups.
    bucket_cap = (per_signature_cap if per_signature_cap > 0
                  else total_cap if total_cap > 0 else 0)
    if bucket_cap > 0:
        buckets = {sig: select_capped(items, bucket_cap)
                   for sig, items in buckets.items()}
    out = []
    used = {sig: 0 for sig in buckets}
    while buckets and (total_cap <= 0 or len(out) < total_cap):
        for sig in list(buckets):
            if not buckets[sig] or (per_signature_cap > 0
                                    and used[sig] >= per_signature_cap):
                del buckets[sig]
                continue
            out.append(buckets[sig].pop(0))
            used[sig] += 1
            if total_cap > 0 and len(out) >= total_cap:
                break
    return out


def bound_feedback_advantage(adv, reward_advantage: float,
                             cfg: FeedbackConfig):
    """Limit feedback's mean magnitude relative to the max-seeking signal."""
    ratio = float(getattr(cfg, "max_reward_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return adv, 1.0
    mean_abs = float(adv.abs().mean().item())
    target = ratio * max(abs(float(reward_advantage)),
                         float(getattr(cfg, "reward_scale_floor", 0.0)))
    if mean_abs <= target or mean_abs <= 1e-12:
        return adv, 1.0
    scale = target / mean_abs
    return adv * scale, scale


# ----------------------------------------------------------------------
# A^fb (Eq. 9)
# ----------------------------------------------------------------------
def feedback_advantage(compute_token_logprobs, model, tokenizer,
                       reprompt_text: str, response_ids, rollout_logprobs,
                       cfg: FeedbackConfig, lam: Optional[float] = None,
                       chunk: int = 0):
    """
    Returns the detached per-token A^fb aligned 1:1 with response_ids, already
    multiplied by the coefficient in force, or None if it could not be computed.
    Pass `lam` from cfg.lambda_at(step) when annealing; it defaults to the
    unannealed lambda_f.

    `rollout_logprobs` must be log pi_thetabar for this response, detached. In
    the trainer that is cur_lp.detach(), which is exact as long as no optimizer
    step has happened since the rollouts were sampled.
    """
    import torch

    try:
        rp_ids = tokenizer(reprompt_text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            q_lp = compute_token_logprobs(model, rp_ids, response_ids,
                                          with_grad=False, chunk=chunk)
    except Exception as e:
        print(f"[feedback] teacher forward failed ({e!r}); skipping this rollout")
        return None

    if q_lp.shape[0] != rollout_logprobs.shape[0]:
        # Cannot align, so it cannot be attributed to tokens. Drop it rather
        # than silently applying a shifted credit assignment.
        return None

    adv = (q_lp - rollout_logprobs).detach()

    if cfg.normalize:
        std = adv.std()
        if float(std) > 1e-6:
            adv = (adv - adv.mean()) / std

    if cfg.clip and cfg.clip > 0:
        # Not in the paper. A single token can hit a log-ratio of 10 or more
        # when the feedback names an identifier verbatim, and unclipped that one
        # token dominates a whole response's update. clip=0 restores Eq. 9 as
        # written.
        adv = adv.clamp(-float(cfg.clip), float(cfg.clip))

    lam = float(cfg.lambda_f if lam is None else lam)
    return lam * adv


class FeedbackStats:
    """Per-step diagnostics, printed by the trainer."""

    def __init__(self):
        self.n = 0
        self.skipped = 0
        self.sum_abs = 0.0
        self.sum_pos = 0.0
        self.max_abs = 0.0

    def add(self, adv) -> None:
        self.n += 1
        a = adv.abs()
        self.sum_abs += float(a.mean().item())
        self.sum_pos += float((adv > 0).float().mean().item())
        self.max_abs = max(self.max_abs, float(a.max().item()))

    def line(self, step_idx: int, lam: float) -> str:
        if self.n == 0:
            return (f"[step {step_idx}] feedback: no failed rollouts scored "
                    f"(lambda={lam:.3f})"
                    + (f", {self.skipped} skipped" if self.skipped else ""))
        return (f"[step {step_idx}] feedback: {self.n} failed rollouts scored"
                + (f" ({self.skipped} skipped)" if self.skipped else "")
                + f"  lambda={lam:.3f}"
                f"  mean|lambda*A^fb|={self.sum_abs / self.n:.4f}"
                f"  max={self.max_abs:.3f}"
                f"  endorsed={100 * self.sum_pos / self.n:.0f}% of tokens")
