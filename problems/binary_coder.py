"""Verified-usability objective for the initial coder training phase.

This objective is deliberately absolute rather than group-relative.  Every
verified usable program receives +1 and every other program receives -1, so an
all-failure group still carries a negative training signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


BINARY_CODER_MODE = "binary-coder"


@dataclass(frozen=True)
class BinaryCoderConfig:
    enabled: bool = False
    init_steps: int = 0
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.38
    lora_rank: int = 32

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BinaryCoderConfig":
        cfg = cls(
            enabled=bool(values.get("binary_coder_training", False)),
            init_steps=int(values.get("binary_coder_init_steps", 0)),
            clip_epsilon_low=float(values.get(
                "binary_coder_clip_epsilon_low", 0.2)),
            clip_epsilon_high=float(values.get(
                "binary_coder_clip_epsilon_high", 0.38)),
            lora_rank=int(values.get(
                "binary_coder_lora_rank", values.get("lora_rank", 32))),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.init_steps < 0:
            raise ValueError("binary_coder_init_steps must be nonnegative")
        if self.enabled and self.init_steps < 1:
            raise ValueError(
                "binary_coder_init_steps must be positive when "
                "binary_coder_training is enabled")
        if self.lora_rank < 1:
            raise ValueError("binary_coder_lora_rank must be positive")
        for name, value in (
                ("binary_coder_clip_epsilon_low", self.clip_epsilon_low),
                ("binary_coder_clip_epsilon_high", self.clip_epsilon_high)):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")

    def active(self, step: int) -> bool:
        return self.enabled and int(step) < self.init_steps

    def frozen(self, step: int) -> bool:
        return self.enabled and int(step) >= self.init_steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "binary_coder_training": self.enabled,
            "binary_coder_init_steps": self.init_steps,
            "binary_coder_clip_epsilon_low": self.clip_epsilon_low,
            "binary_coder_clip_epsilon_high": self.clip_epsilon_high,
            "binary_coder_lora_rank": self.lora_rank,
        }


def _unchanged_parent_construction(result: Any, parent: Any) -> bool:
    current = getattr(result, "construction", None)
    previous = getattr(parent, "construction", None)
    if current is None or previous is None:
        return False
    try:
        current_values = np.asarray(current, dtype=np.float64).reshape(-1)
        previous_values = np.asarray(previous, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        current_values.size > 0
        and current_values.shape == previous_values.shape
        and np.isfinite(current_values).all()
        and np.isfinite(previous_values).all()
        and np.allclose(
            current_values, previous_values, rtol=0.0, atol=1e-12))


def verified_usability(result: Any, parent: Any, *, fail_score: float
                       ) -> tuple[bool, str]:
    """Apply an absolute, deterministic usability gate to one rollout."""
    if not bool(getattr(result, "parsed", False)):
        return False, "unparsed"
    if not str(getattr(result, "code", "") or "").strip():
        return False, "empty_code"
    if not bool(getattr(result, "ran", False)):
        return False, "runtime_failure"
    if not bool(getattr(result, "valid", False)):
        return False, "verifier_rejected"

    try:
        reward = float(getattr(result, "reward"))
    except (TypeError, ValueError):
        return False, "invalid_reward"
    if not math.isfinite(reward) or reward <= float(fail_score):
        return False, "nonpositive_verified_reward"

    raw_score = getattr(result, "raw_score", None)
    if raw_score is not None:
        try:
            if not math.isfinite(float(raw_score)):
                return False, "nonfinite_raw_score"
        except (TypeError, ValueError):
            return False, "invalid_raw_score"

    construction = getattr(result, "construction", None)
    if construction is not None:
        try:
            values = np.asarray(construction, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False, "invalid_construction"
        if values.size and not np.isfinite(values).all():
            return False, "invalid_construction"
    if _unchanged_parent_construction(result, parent):
        return False, "unchanged_parent"
    return True, "verified_usable"


def binary_coder_advantages(
        results: Sequence[Any], parent: Any, *, fail_score: float
        ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return fixed +/-1 advantages and per-rollout gate diagnostics."""
    diagnostics = []
    advantages = np.empty(len(results), dtype=np.float64)
    for index, result in enumerate(results):
        usable, reason = verified_usability(
            result, parent, fail_score=fail_score)
        advantages[index] = 1.0 if usable else -1.0
        diagnostics.append({
            "usable": bool(usable),
            "reason": reason,
            "advantage": float(advantages[index]),
        })
    return advantages, diagnostics


def binary_coder_clipped_loss(
        current_logprobs, old_logprobs, advantage, *,
        clip_epsilon_low: float, clip_epsilon_high: float,
        return_tensor_metrics: bool = False):
    """Length-normalized asymmetric PPO loss for a binary coder outcome."""
    import torch

    low = float(clip_epsilon_low)
    high = float(clip_epsilon_high)
    if (not math.isfinite(low) or not 0.0 < low < 1.0
            or not math.isfinite(high) or not 0.0 < high < 1.0):
        raise ValueError("binary coder clip distances must be in (0, 1)")

    current = current_logprobs.to(dtype=torch.float64)
    old = torch.as_tensor(
        old_logprobs, dtype=current.dtype, device=current.device).detach()
    if current.ndim != 1 or current.numel() == 0 or old.shape != current.shape:
        raise ValueError(
            "binary coder current/old logprobs must be matching vectors")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("binary coder logprobs must be finite")

    try:
        advantage_value = float(advantage)
    except (TypeError, ValueError):
        raise ValueError("binary coder advantage must be exactly -1 or +1")
    if advantage_value not in (-1.0, 1.0):
        raise ValueError("binary coder advantage must be exactly -1 or +1")
    adv = current.new_tensor(advantage_value).detach()

    ratios = torch.exp(current - old)
    clipped_ratios = ratios.clamp(1.0 - low, 1.0 + high)
    policy_loss = -torch.minimum(
        ratios * adv, clipped_ratios * adv).mean()
    clipped_fraction = (
        ratios.detach() != clipped_ratios.detach()).double().mean()
    if not torch.isfinite(policy_loss).all():
        raise FloatingPointError("nonfinite binary coder clipped loss")

    zero = current.new_zeros(())
    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_estimate": zero,
        "entropy_estimate": zero,
        "ratio": ratios.mean().detach(),
        "prefix_ratio_max": current.new_ones(()),
        "clipped": clipped_fraction.detach(),
    }
    if return_tensor_metrics:
        return policy_loss, metrics
    return policy_loss, {
        key: float(value.item()) for key, value in metrics.items()
    }
