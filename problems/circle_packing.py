"""
Circle packing.

  - entrypoint:  run_packing
  - validator:   validate_packing (byte-identical to the paper's)
  - reward:      sum of radii if valid else 0   (maximize)
  - get_question template reproduced verbatim; the validator source is injected
    into the prompt with inspect.getsource, exactly like the original.
"""

from __future__ import annotations
import inspect
from typing import Any, List
import numpy as np

from problems.base import (
    Problem, ParentContext, RewardResult, SeedState, render_state_context,
)


# ----------------------------------------------------------------------
# Validator (verbatim copy of the paper's / examples/circle_packing/env.py)
# ----------------------------------------------------------------------
def validate_packing(centers, radii):
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"

    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"

    return True, "ok"


_VALIDATOR_SRC = inspect.getsource(validate_packing)


class CirclePacking(Problem):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_circles = int(cfg.get("num_circles", 26))
        if self.target is None:
            self.target = 2.636 if self.num_circles == 26 else 2.940

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)
        n = self.num_circles
        user = f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {n} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{_VALIDATOR_SRC}
```

{state_ctx}

Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({n}, 2) and radii has shape ({n},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""

        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = (
            "import numpy as np\n"
            "import math\n"
            "try:\n"
            "    from scipy.optimize import minimize\n"
            "except ImportError:\n"
            "    minimize = None\n\n"
            + _VALIDATOR_SRC + "\n"
        )
        return prelude + "\n# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    # def score(self, output: Any, stdout: str) -> RewardResult:
    #     res = RewardResult(reward=self.fail_score)
    #     if not (isinstance(output, tuple) and len(output) == 3):
    #         res.msg = "bad_return_shape"
    #         return res
    #     centers, radii, _ = output
    #     centers = np.asarray(centers)
    #     radii = np.asarray(radii).ravel()

    #     if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != self.num_circles:
    #         res.msg = f"bad_centers_shape: {centers.shape}"
    #         return res

    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, tuple) and len(output) == 3):
            res.msg = "bad_return_shape"
            return res
        centers, radii, _ = output
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array: {e}"
            return res

        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != self.num_circles:
            res.msg = f"bad_centers_shape: {centers.shape}"
            return res
        if radii.shape != (self.num_circles,):
            res.msg = f"bad_radii_shape: {radii.shape}"
            return res

        valid, msg = validate_packing(centers, radii)
        res.valid = valid
        res.msg = msg
        if valid:
            s = float(np.sum(radii))
            res.reward = s
            res.raw_score = s
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        return [SeedState(code="", value=0.0, raw_score=0.0)
                for _ in range(self.num_seed_states)]
