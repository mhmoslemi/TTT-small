"""Persistent KL-adaptive entropic value tracker for SPO-RS."""

from __future__ import annotations

from array import array
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np


class SPORSTracker:
    """Track the entropic value of each exact rendered conditioning prompt."""

    VERSION = 1
    POLICY_LOGPROB_SOURCE = "forward-score-v1"

    def __init__(self, *, entropic_beta: float, d_half: float,
                 rho_min: float, rho_max: float):
        self.entropic_beta = float(entropic_beta)
        self.d_half = float(d_half)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self._validate_options()
        self._entries: dict[str, dict] = {}

    def _validate_options(self) -> None:
        if not math.isfinite(self.entropic_beta) or self.entropic_beta <= 0.0:
            raise ValueError("SPO-RS entropic beta must be finite and positive")
        if not math.isfinite(self.d_half) or self.d_half <= 0.0:
            raise ValueError("SPO-RS D_half must be finite and positive")
        if (not math.isfinite(self.rho_min)
                or not math.isfinite(self.rho_max)
                or not 0.0 < self.rho_min <= self.rho_max < 1.0):
            raise ValueError("SPO-RS rho bounds must satisfy 0 < min <= max < 1")

    @staticmethod
    def prompt_key(prompt_text: str) -> str:
        return hashlib.sha256(str(prompt_text).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._entries)

    def baseline(self, prompt_key: str) -> float | None:
        entry = self._entries.get(str(prompt_key))
        return None if entry is None else float(entry["value"])

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
        self, prompt_key: str, transformed_rewards: Sequence[float]
    ) -> np.ndarray | None:
        baseline = self.baseline(prompt_key)
        if baseline is None:
            return None
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("SPO-RS tracker contains a nonpositive value")
        values = np.asarray(transformed_rewards, dtype=np.float64)
        advantages = values / baseline - 1.0
        if not np.isfinite(advantages).all():
            raise FloatingPointError("nonfinite SPO-RS normalized advantages")
        return advantages

    @staticmethod
    def initial_advantages(
        transformed_rewards: Sequence[float],
    ) -> np.ndarray | None:
        """Cross-fit a first-visit baseline without using a sample on itself."""
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if (values.ndim != 1 or values.size == 0
                or not np.isfinite(values).all() or np.any(values <= 0.0)):
            raise ValueError(
                "SPO-RS transformed rewards must be finite and positive")
        if values.size < 2:
            return None

        # For sample i, the mean of all j != i is independent of y_i under
        # the frozen sampling policy. It can therefore initialize the positive
        # value scale without turning the sample's own reward into its baseline.
        count = int(values.size - 1)
        baselines = np.empty_like(values)
        for index in range(values.size):
            try:
                other_sum = math.fsum(
                    float(value) for offset, value in enumerate(values)
                    if offset != index)
            except OverflowError as error:
                raise FloatingPointError(
                    "nonfinite SPO-RS first-visit baseline") from error
            baselines[index] = other_sum / count
        if (not np.isfinite(baselines).all()
                or np.any(baselines <= 0.0)):
            raise FloatingPointError(
                "nonfinite SPO-RS first-visit baseline")
        advantages = values / baselines - 1.0
        if not np.isfinite(advantages).all():
            raise FloatingPointError(
                "nonfinite SPO-RS first-visit advantages")
        return advantages

    def anchor_requests(self, prompt_keys: Iterable[str]) -> list[dict]:
        """Return prior-policy trajectories that need current-policy scores."""
        requests = []
        for prompt_key in dict.fromkeys(str(key) for key in prompt_keys):
            entry = self._entries.get(prompt_key)
            if entry is None:
                continue
            # Sampling-time logprobs and explicit forward rescoring are not
            # numerically interchangeable. Older checkpoints did not record
            # the source, so discard those anchors once instead of reporting
            # false policy drift for an unchanged model.
            if (entry.get("policy_logprob_source")
                    != self.POLICY_LOGPROB_SOURCE):
                continue
            prompt_ids = list(entry.get("prompt_ids", ()))
            responses = entry.get("response_ids", ())
            old_logprobs = entry.get("policy_logprobs", ())
            for response_ids, old_values in zip(responses, old_logprobs):
                response_ids = list(response_ids)
                old_values = list(old_values)
                if response_ids and len(response_ids) == len(old_values):
                    requests.append({
                        "prompt_key": prompt_key,
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "old_logprobs": old_values,
                    })
        return requests

    @staticmethod
    def divergences_from_scores(
        prompt_keys: Iterable[str], requests: Sequence[dict],
        current_scores: Sequence[Sequence[float] | None],
    ) -> tuple[dict[str, float], int]:
        """Estimate current-to-prior token KL on prior-policy trajectories.

        For a sampled prior-policy token and ``r = pi_current / pi_prior``, the
        nonnegative f-divergence integrand ``r log r - r + 1`` has expectation
        ``KL(pi_current || pi_prior)`` at that prefix. Prefix importance ratios
        move the state-prefix expectation from the prior policy to the current
        policy; summing over tokens gives a trajectory-level KL estimate.
        """
        if len(requests) != len(current_scores):
            raise ValueError("SPO-RS anchor requests and scores must align")
        terms: dict[str, list[float]] = {}
        missing = 0
        for request, current_values in zip(requests, current_scores):
            old = np.asarray(request["old_logprobs"], dtype=np.float64)
            if current_values is None:
                missing += 1
                continue
            current = np.asarray(current_values, dtype=np.float64)
            if (current.shape != old.shape or current.ndim != 1
                    or current.size == 0 or not np.isfinite(current).all()
                    or not np.isfinite(old).all()):
                missing += 1
                continue
            log_ratio = current - old
            prefix_log_ratio = np.concatenate((
                np.zeros(1, dtype=np.float64),
                np.cumsum(log_ratio[:-1], dtype=np.float64),
            ))
            if (np.any(log_ratio > 700.0)
                    or np.any(prefix_log_ratio > 700.0)):
                trajectory_term = math.inf
            else:
                ratio = np.exp(log_ratio)
                token_terms = ratio * log_ratio - ratio + 1.0
                token_terms = np.maximum(token_terms, 0.0)
                prefix_ratio = np.exp(prefix_log_ratio)
                trajectory_term = float(np.sum(prefix_ratio * token_terms))
            terms.setdefault(str(request["prompt_key"]), []).append(
                trajectory_term)

        divergences = {}
        for prompt_key in dict.fromkeys(str(key) for key in prompt_keys):
            chunks = terms.get(prompt_key)
            if not chunks:
                divergences[prompt_key] = math.inf
                continue
            divergences[prompt_key] = max(
                0.0, float(sum(chunks)) / len(chunks))
        return divergences, missing

    @staticmethod
    def _compact_anchors(prompt_ids, anchors) -> tuple[array, list[array], list[array]]:
        compact_prompt = array("I", (int(value) for value in prompt_ids))
        compact_responses = []
        compact_logprobs = []
        for response_ids, policy_logprobs in anchors:
            ids = [int(value) for value in response_ids]
            values = [float(value) for value in policy_logprobs]
            if (not ids or len(ids) != len(values)
                    or not all(math.isfinite(value) for value in values)):
                continue
            compact_responses.append(array("I", ids))
            compact_logprobs.append(array("f", values))
        return compact_prompt, compact_responses, compact_logprobs

    def update(self, prompt_key: str, transformed_rewards: Sequence[float], *,
               divergence: float, prompt_ids: Sequence[int], anchors,
               step: int) -> dict:
        """Update after advantages were formed from the pre-update value."""
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("SPO-RS transformed rewards must be finite and nonempty")
        if np.any(values <= 0.0):
            raise ValueError("SPO-RS transformed rewards must be positive")
        prompt_key = str(prompt_key)
        current_mean = float(values.mean())
        group_size = int(values.size)
        previous = self._entries.get(prompt_key)
        compact_prompt, compact_responses, compact_logprobs = (
            self._compact_anchors(prompt_ids, anchors))

        if previous is None:
            value_before = None
            count_before = 0.0
            rho = None
            eta = 1.0
            value_after = current_mean
            count_after = group_size / (1.0 - self.rho_min)
            visits = 1
        else:
            value_before = float(previous["value"])
            count_before = float(previous["effective_count"])
            divergence = float(divergence)
            if math.isnan(divergence) or divergence < 0.0:
                raise ValueError("SPO-RS policy divergence must be nonnegative")
            raw_rho = (0.0 if math.isinf(divergence)
                       else 2.0 ** (-divergence / self.d_half))
            rho = min(self.rho_max, max(self.rho_min, raw_rho))
            count_after = rho * count_before + group_size
            eta = group_size / count_after
            value_after = value_before + eta * (current_mean - value_before)
            visits = int(previous.get("visits", 0)) + 1

        if (not math.isfinite(value_after) or value_after <= 0.0
                or not math.isfinite(count_after) or count_after <= 0.0):
            raise FloatingPointError("nonfinite SPO-RS tracker update")
        self._entries[prompt_key] = {
            "value": value_after,
            "effective_count": count_after,
            "prompt_ids": compact_prompt,
            "response_ids": compact_responses,
            "policy_logprobs": compact_logprobs,
            "policy_logprob_source": self.POLICY_LOGPROB_SOURCE,
            "last_step": int(step),
            "visits": visits,
        }
        return {
            "initialized": previous is None,
            "value_before": value_before,
            "value_after": value_after,
            "effective_count_before": count_before,
            "effective_count_after": count_after,
            "group_mean": current_mean,
            "group_size": group_size,
            "divergence": (
                None if previous is None or not math.isfinite(divergence)
                else float(divergence)),
            "divergence_missing": bool(
                previous is not None and not math.isfinite(divergence)),
            "rho": rho,
            "eta": eta,
            "anchor_count": len(compact_responses),
            "visits": visits,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "entropic_beta": self.entropic_beta,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "entries": self._entries,
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
        entries = state.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("SPO-RS tracker entries are missing")
        restored = {}
        for prompt_key, entry in entries.items():
            value = float(entry["value"])
            effective_count = float(entry["effective_count"])
            if (not math.isfinite(value) or value <= 0.0
                    or not math.isfinite(effective_count)
                    or effective_count <= 0.0):
                raise ValueError("invalid SPO-RS tracker entry")
            prompt_ids, responses, logprobs = self._compact_anchors(
                entry.get("prompt_ids", ()),
                zip(entry.get("response_ids", ()),
                    entry.get("policy_logprobs", ())),
            )
            restored[str(prompt_key)] = {
                "value": value,
                "effective_count": effective_count,
                "prompt_ids": prompt_ids,
                "response_ids": responses,
                "policy_logprobs": logprobs,
                "policy_logprob_source": str(entry.get(
                    "policy_logprob_source", "legacy-sampling")),
                "last_step": int(entry.get("last_step", -1)),
                "visits": int(entry.get("visits", 1)),
            }
        self._entries = restored
