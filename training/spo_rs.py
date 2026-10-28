"""Historical SPO-RS trackers: exponential and bounded adaptive shaping."""

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

    def prepare_step(self) -> None:
        """Freeze the historical baseline before any current-step rewards."""

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
               step: int, raw_rewards: Sequence[float] | None = None) -> dict:
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


BOUNDED_SHAPING_VERSION = "historical-arctan-1.25-v1"


def validate_shaping_resume(saved: dict) -> None:
    """Never silently reuse gradients from the replaced batch objective."""
    if (saved.get("advantage_mode") == "spo-rs"
            and saved.get("spo_rs_adaptive_beta", False)
            and saved.get("spo_rs_reward_shaping") != BOUNDED_SHAPING_VERSION):
        raise ValueError(
            "this run used the retired adaptive-beta SPO objective, not the "
            "historical bounded shaping; use the previous code to resume it "
            "or start a fresh run with the new objective")


class BoundedSPORSTracker(SPORSTracker):
    """SPO's historical moving average with adaptive, bounded reward shaping.

    At the start of a step, using only decayed *previous* rewards above the
    failure score, b = Q90 and s = Q90 - Q75 (weighted inverse-CDF quantiles).
    Freeze u(r) = 1/2 + atan((r-b)/s)/pi, g(r) = exp(beta * 1.25 * u(r)),
    and v = weighted_mean(g(history)) for every parent and rollout this step.
    A = (g/v - 1)/beta. Failures remain in v and the update; no rollout is
    filtered out of training. This is a monotone upper-tail reward surrogate,
    not the original exp(beta*r) objective or an exact maximum operator.

    Raw scalar rewards and their decay weights are checkpointed so changing
    b/s never mixes differently transformed rewards in the baseline. No
    model tensors or rollout text are retained here.
    """

    VERSION = 3
    WIDTH = 1.25

    def __init__(self, *, fail_score=0.0, **options):
        super().__init__(**options)
        self.fail_score = float(fail_score)
        if not math.isfinite(self.fail_score):
            raise ValueError("SPO-RS fail_score must be finite")
        # Avoid exponent overflow even for user-specified beta values.
        if self.entropic_beta * self.WIDTH > math.log(np.finfo(float).max) - 2:
            raise ValueError("SPO-RS beta is too large for bounded shaping")
        self._history_rewards = np.empty(0, dtype=np.float64)
        self._history_weights = np.empty(0, dtype=np.float64)
        self._center: float | None = None
        self._scale: float | None = None

    @staticmethod
    def _rewards(rewards):
        values = np.asarray(rewards, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("SPO-RS rewards must be a nonempty finite vector")
        return values

    def _calibrate(self, rewards, weights):
        successful = (rewards > self.fail_score) & (weights > 0.0)
        values, mass = rewards[successful], weights[successful]
        if not values.size:
            # All-failure history has no upper-tail scale. Retain any prior
            # calibration; otherwise use a finite neutral initial scale.
            if self._center is None:
                self._center = self.fail_score
                self._scale = max(1.0, abs(self.fail_score))
            return
        order = np.argsort(values, kind="stable")
        values, mass = values[order], mass[order]
        cumulative = np.cumsum(mass, dtype=np.float64)

        def quantile(q):
            index = min(int(np.searchsorted(cumulative, q * cumulative[-1],
                                            side="left")), len(values) - 1)
            return float(values[index])

        self._center = quantile(0.90)
        scale = self._center - quantile(0.75)
        if scale <= 0.0:
            # Ties must not create a zero denominator or a fixed precision
            # floor that erases small but real improvements.
            scale = self._scale or (quantile(0.75) - quantile(0.25))
            if scale <= 0.0:
                scale = float(values[-1] - values[0])
            if scale <= 0.0:
                scale = max(1.0, abs(self._center - self.fail_score))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("invalid historical SPO-RS shaping scale")
        self._scale = scale

    def prepare_step(self) -> None:
        if not self.initialized:
            raise RuntimeError("bounded SPO-RS requires seed history first")
        self._calibrate(self._history_rewards, self._history_weights)
        values = self.transformed_rewards(self._history_rewards)
        self._value = float(np.dot(values, self._history_weights)
                            / self._history_weights.sum())

    def transformed_rewards(self, rewards: Sequence[float]) -> np.ndarray:
        values = self._rewards(rewards)
        if self._center is None:
            # The only pre-initialization call is the verified seed batch.
            self._calibrate(values, np.ones(values.size, dtype=np.float64))
        with np.errstate(over="ignore"):
            utility = 0.5 + np.arctan(
                (values - self._center) / self._scale) / math.pi
        return np.exp(self.entropic_beta * self.WIDTH * utility)

    def shaping_info(self) -> dict:
        return {
            "reward_shaping": BOUNDED_SHAPING_VERSION,
            "shaping_width": self.WIDTH,
            "shaping_center": self._center,
            "shaping_scale": self._scale,
            "shaping_history_count": int(self._history_rewards.size),
            "beta": self.entropic_beta,
        }

    def update(self, transformed_rewards, *, divergence, policy_adapter,
               step, raw_rewards=None) -> dict:
        if raw_rewards is None:
            raise ValueError("bounded SPO-RS update requires raw reward history")
        rewards = self._rewards(raw_rewards)
        transformed = np.asarray(transformed_rewards, dtype=np.float64)
        if (transformed.shape != rewards.shape
                or not np.array_equal(transformed, self.transformed_rewards(rewards))):
            raise ValueError("bounded SPO-RS raw/transformed rewards do not match")
        info = self.shaping_info()
        result = super().update(
            transformed, divergence=divergence,
            policy_adapter=policy_adapter, step=step)
        if result["initialized"]:
            weights = np.full(rewards.size, 1.0 / (1.0 - self.rho_min))
        else:
            self._history_weights *= result["rho"]
            weights = np.ones(rewards.size, dtype=np.float64)
        self._history_rewards = np.concatenate((self._history_rewards, rewards))
        self._history_weights = np.concatenate((self._history_weights, weights))
        # Keep this step's calibration frozen until the *next* prepare_step.
        return {**result, **info}

    def state_dict(self) -> dict:
        return {
            **super().state_dict(), **self.shaping_info(),
            "fail_score": self.fail_score,
            "history_rewards": self._history_rewards.tolist(),
            "history_weights": self._history_weights.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if (not isinstance(state, dict)
                or state.get("reward_shaping") != BOUNDED_SHAPING_VERSION
                or state.get("shaping_width") != self.WIDTH
                or state.get("fail_score") != self.fail_score):
            raise ValueError("incompatible bounded SPO-RS shaping checkpoint")
        super().load_state_dict(state)
        rewards = np.asarray(state.get("history_rewards"), dtype=np.float64)
        weights = np.asarray(state.get("history_weights"), dtype=np.float64)
        if (rewards.ndim != 1 or weights.shape != rewards.shape
                or not np.isfinite(rewards).all()
                or not np.isfinite(weights).all() or np.any(weights < 0.0)
                or bool(rewards.size) != self.initialized
                or not math.isclose(float(weights.sum()), self._effective_count,
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("invalid bounded SPO-RS reward history")
        self._history_rewards = rewards.copy()
        self._history_weights = weights.copy()
        self._center = state.get("shaping_center")
        self._scale = state.get("shaping_scale")
        if self.initialized:
            if (self._center is None or not math.isfinite(self._center)
                    or self._scale is None or not math.isfinite(self._scale)
                    or self._scale <= 0.0):
                raise ValueError("invalid saved SPO-RS shaping calibration")
            expected = float(np.dot(self.transformed_rewards(rewards), weights)
                             / weights.sum())
            if not math.isclose(expected, self._value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("bounded SPO-RS baseline/history mismatch")
