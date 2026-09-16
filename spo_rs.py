"""Run-wide KL-adaptive entropic value tracker for SPO-RS."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


class SPORSTracker:
    """Track one entropic value for the entire discovery run."""

    VERSION = 2

    def __init__(self, *, entropic_beta: float, d_half: float,
                 rho_min: float, rho_max: float):
        self.entropic_beta = float(entropic_beta)
        self.d_half = float(d_half)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self._validate_options()
        self._value: float | None = None
        self._effective_count = 0.0
        self._last_policy_adapter: str | None = None
        self._last_step = -1
        self._updates = 0

    def _validate_options(self) -> None:
        if not math.isfinite(self.entropic_beta) or self.entropic_beta <= 0.0:
            raise ValueError("SPO-RS entropic beta must be finite and positive")
        if not math.isfinite(self.d_half) or self.d_half <= 0.0:
            raise ValueError("SPO-RS D_half must be finite and positive")
        if (not math.isfinite(self.rho_min)
                or not math.isfinite(self.rho_max)
                or not 0.0 < self.rho_min <= self.rho_max < 1.0):
            raise ValueError("SPO-RS rho bounds must satisfy 0 < min <= max < 1")

    def __len__(self) -> int:
        return int(self._value is not None)

    @property
    def initialized(self) -> bool:
        return self._value is not None

    @property
    def last_policy_adapter(self) -> str | None:
        return self._last_policy_adapter

    def baseline(self) -> float | None:
        return None if self._value is None else float(self._value)

    def transformed_rewards(self, rewards: Sequence[float]) -> np.ndarray:
        values = np.asarray(rewards, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("SPO-RS rewards must be a nonempty finite vector")
        transformed = np.exp(self.entropic_beta * values)
        if not np.isfinite(transformed).all() or np.any(transformed <= 0.0):
            raise FloatingPointError(
                "nonfinite SPO-RS exponential rewards; reduce spo_rs_beta")
        return transformed

    def normalized_advantages(
        self, transformed_rewards: Sequence[float]
    ) -> np.ndarray | None:
        """Use the saved pre-step value for every rollout in this step."""
        baseline = self.baseline()
        if baseline is None:
            return None
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if (values.ndim != 1 or values.size == 0
                or not np.isfinite(values).all() or np.any(values <= 0.0)):
            raise ValueError(
                "SPO-RS transformed rewards must be finite and positive")
        advantages = (values / baseline - 1.0) / self.entropic_beta
        if not np.isfinite(advantages).all():
            raise FloatingPointError("nonfinite SPO-RS normalized advantages")
        return advantages

    @staticmethod
    def consecutive_policy_divergence(
        current_scores: Sequence[Sequence[float] | None],
        previous_scores: Sequence[Sequence[float] | None],
        context_ids: Sequence[int],
        *, required_context_ids: Sequence[int] | None = None,
    ) -> tuple[float, int, dict[int, float]]:
        """Estimate mean KL(pi_t || pi_{t-1}) on current-step contexts.

        Current responses are sampled from ``pi_t``. Their trajectory
        log-likelihood difference is therefore a direct Monte Carlo estimate
        of the forward KL. Trajectories are first averaged within a selected
        parent and then the parents are weighted equally.
        """
        if not (len(current_scores) == len(previous_scores) == len(context_ids)):
            raise ValueError("SPO-RS policy scores and contexts must align")
        required_contexts = tuple(dict.fromkeys(
            int(x) for x in (
                context_ids if required_context_ids is None
                else required_context_ids)))
        by_context: dict[int, list[float]] = {
            context_id: [] for context_id in required_contexts
        }
        missing = 0
        for current_values, previous_values, context_id in zip(
                current_scores, previous_scores, context_ids):
            if current_values is None or previous_values is None:
                missing += 1
                continue
            current = np.asarray(current_values, dtype=np.float64)
            previous = np.asarray(previous_values, dtype=np.float64)
            if (current.ndim != 1 or current.size == 0
                    or current.shape != previous.shape
                    or not np.isfinite(current).all()
                    or not np.isfinite(previous).all()):
                missing += 1
                continue
            by_context[int(context_id)].append(float(np.sum(
                current - previous, dtype=np.float64)))

        if any(not values for values in by_context.values()):
            return math.inf, missing, {}
        context_estimates = {
            context_id: float(math.fsum(values) / len(values))
            for context_id, values in by_context.items()
        }
        divergence = float(
            math.fsum(context_estimates.values()) / len(context_estimates))
        # A finite Monte Carlo estimate can be slightly negative even though
        # the population KL is nonnegative.
        return max(0.0, divergence), missing, context_estimates

    def update(self, transformed_rewards: Sequence[float], *,
               divergence: float, policy_adapter: str | Path,
               step: int) -> dict:
        """Update once, after this step's advantages have been formed."""
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if (values.ndim != 1 or values.size == 0
                or not np.isfinite(values).all() or np.any(values <= 0.0)):
            raise ValueError(
                "SPO-RS transformed rewards must be finite and positive")
        current_sum = float(math.fsum(float(value) for value in values))
        current_count = int(values.size)
        current_mean = current_sum / current_count
        value_before = self._value
        count_before = float(self._effective_count)

        if value_before is None:
            rho = None
            eta = 1.0
            value_after = current_mean
            count_after = current_count / (1.0 - self.rho_min)
        else:
            divergence = float(divergence)
            if math.isnan(divergence) or divergence < 0.0:
                raise ValueError("SPO-RS policy divergence must be nonnegative")
            raw_rho = (0.0 if math.isinf(divergence)
                       else 2.0 ** (-divergence / self.d_half))
            rho = min(self.rho_max, max(self.rho_min, raw_rho))
            retained_count = rho * count_before
            count_after = retained_count + current_count
            value_after = (
                retained_count * value_before + current_sum
            ) / count_after
            eta = current_count / count_after

        if (not math.isfinite(value_after) or value_after <= 0.0
                or not math.isfinite(count_after) or count_after <= 0.0):
            raise FloatingPointError("nonfinite SPO-RS tracker update")
        self._value = float(value_after)
        self._effective_count = float(count_after)
        self._last_policy_adapter = Path(policy_adapter).name
        self._last_step = int(step)
        self._updates += 1
        return {
            "initialized": value_before is None,
            "value_before": value_before,
            "value_after": value_after,
            "effective_count_before": count_before,
            "effective_count_after": count_after,
            "group_mean": current_mean,
            "group_size": current_count,
            "divergence": (
                None if value_before is None or not math.isfinite(divergence)
                else float(divergence)),
            "divergence_missing": bool(
                value_before is not None and not math.isfinite(divergence)),
            "rho": rho,
            "eta": eta,
            "policy_adapter": self._last_policy_adapter,
            "step": self._last_step,
            "updates": self._updates,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "entropic_beta": self.entropic_beta,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "value": self._value,
            "effective_count": self._effective_count,
            "last_policy_adapter": self._last_policy_adapter,
            "last_step": self._last_step,
            "updates": self._updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != self.VERSION:
            raise ValueError("unsupported SPO-RS tracker state")
        expected = {
            "entropic_beta": self.entropic_beta,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
        }
        for name, value in expected.items():
            saved = float(state.get(name, math.nan))
            if not math.isclose(saved, value, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"SPO-RS tracker {name}={saved} does not match config {value}")

        value = state.get("value")
        effective_count = float(state.get("effective_count", 0.0))
        if value is None:
            if effective_count != 0.0:
                raise ValueError("uninitialized SPO-RS tracker has history")
            self._value = None
            self._effective_count = 0.0
        else:
            value = float(value)
            if (not math.isfinite(value) or value <= 0.0
                    or not math.isfinite(effective_count)
                    or effective_count <= 0.0):
                raise ValueError("invalid SPO-RS tracker state")
            self._value = value
            self._effective_count = effective_count
        adapter = state.get("last_policy_adapter")
        self._last_policy_adapter = None if adapter is None else Path(adapter).name
        self._last_step = int(state.get("last_step", -1))
        self._updates = int(state.get("updates", int(self._value is not None)))
