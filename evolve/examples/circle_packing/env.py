"""
Circle packing: place n circles in the unit square, maximize the sum of radii.

Same environment as the reference TTT-Discover implementation -- same validator,
same entrypoint contract, same prompt content, same reward (sum of radii, 0 if
invalid) -- so results are comparable across the two frameworks. What differs is
everything upstream: which parent gets expanded, what memories are in context,
and how the update is computed.

All knobs come from example.params; nothing here is hardcoded.
"""

import inspect
from typing import Any, List, Optional

import numpy as np

from core.types import Node, VerifyResult
from envs.base import Example
from examples.circle_packing import prompts
from examples.circle_packing.validator import validate_packing

_VALIDATOR_SRC = inspect.getsource(validate_packing)


class CirclePacking(Example):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_circles = int(self.params.get("num_circles", 26))
        target = self.params.get("target")
        if target is None:
            # Best known values for the sizes the reference implementation ships.
            target = {26: 2.635983, 32: 2.940}.get(self.num_circles)
        self.target = target

    # ------------------------------------------------------------------
    # The problem description d
    # ------------------------------------------------------------------
    def meta_description(self) -> str:
        target = "unknown" if self.target is None else f"{self.target}"
        return prompts.META.format(
            n=self.num_circles,
            validator_src=_VALIDATOR_SRC,
            metric_name=self.metric_name,
            target=target,
            direction="higher is better" if self.maximize else "lower is better",
        )

    def instruction(self) -> str:
        return prompts.INSTRUCTION.format(n=self.num_circles,
                                          entrypoint=self.entrypoint)

    # ------------------------------------------------------------------
    # Transition T_d
    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: Optional[Node]) -> str:
        return (prompts.PRELUDE + _VALIDATOR_SRC
                + "\n\n# ---- model code below ----\n" + code)

    # ------------------------------------------------------------------
    # Evaluator R_d, F_d
    # ------------------------------------------------------------------
    def score(self, value: Any, stdout: str) -> VerifyResult:
        n = self.num_circles
        res = VerifyResult(reward=self.fail_reward, stdout=stdout)

        if not (isinstance(value, tuple) and len(value) == 3):
            res.msg = "bad_return_shape"
            res.feedback = (
                f"{self.entrypoint}() must return a 3-tuple "
                f"(centers, radii, sum_radii); it returned {type(value).__name__}.")
            return res

        centers, radii, _ = value
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array: {e}"
            res.feedback = f"centers/radii could not be read as float arrays: {e}"
            return res

        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != n:
            res.msg = f"bad_centers_shape: {centers.shape}"
            res.feedback = (f"centers must have shape ({n}, 2); "
                            f"got {tuple(centers.shape)}.")
            return res
        if radii.shape != (n,):
            res.msg = f"bad_radii_shape: {radii.shape}"
            res.feedback = f"radii must have shape ({n},); got {tuple(radii.shape)}."
            return res

        valid, msg = validate_packing(centers, radii)
        res.valid = valid
        res.msg = msg
        if valid:
            total = float(np.sum(radii))
            res.reward = total
            res.raw_score = total
            res.feedback = ""
        else:
            # The validator names the offending circle, which is exactly the kind
            # of localized signal Eq. 9 can turn into token-level credit.
            res.feedback = (
                f"The packing is geometrically invalid: {msg}\n"
                f"Every pair of circles must be non-overlapping and every circle "
                f"must lie inside the unit square.")
        return res

    # ------------------------------------------------------------------
    def seed_nodes(self, count: int) -> List[dict]:
        return [{} for _ in range(max(1, count))]


def build(cfg) -> CirclePacking:
    """Entry point the registry calls."""
    return CirclePacking(cfg)
