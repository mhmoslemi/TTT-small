"""
Circle packing.

  - entrypoint:  run_packing
  - validator:   validate_packing (byte-identical to the paper's)
  - reward:      sum of radii if valid else 0   (maximize)

The prompt is memory-aware. build_prompt takes an optional `memory` block and
places it BETWEEN the parent state and the instruction, per Fig. 1's ordering,
rather than having the trainer staple it onto the end. Two things change when
memory is present:

  - the fixed "Consider:" hint list is replaced. Those four hints are the same
    every step and compete with the retrieved lessons for the model's attention;
    with memory on, the analysis step is to consult the lessons instead.
  - the prompt asks the model to note where the lessons do NOT apply. That is a
    pressure valve: v1's failure was a lesson reaching 99% adoption, and a prompt
    that only ever asks "how do I use this" has no way to reject it.
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


# ----------------------------------------------------------------------
# Prompt sections
# ----------------------------------------------------------------------
_ANALYSIS_NO_MEMORY = """Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?"""

_ANALYSIS_WITH_MEMORY = """## 1. Analysis and strategy

Work through the recorded lessons above before you write anything:
- Which of them bear on the program you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. The lessons are evidence
  from earlier attempts, not requirements, and some of them will be wrong or
  irrelevant for this state.
- Is anything the lessons recommend already present in the program above and
  still not working? If so, the lesson has been tried and the improvement lies
  somewhere it does not cover.

Then decide what to change. A lesson tells you an idea; you decide the
implementation. Do not copy any expression from a lesson verbatim, and do not let
a lesson choose your overall arrangement for you.

If none of the lessons is useful here, ignore them and reason from first
principles about the packing itself: boundary and corner occupancy, whether a
different arrangement family would do better, gaps that could absorb a
repositioned circle, and whether the optimization formulation itself is the
limit."""

_MEMORY_HEADER = """## Lessons from earlier attempts at this task

Extracted from programs already generated and evaluated in this same search.
They are empirical findings, not part of the specification above, and they do
not override any rule stated in it."""

_V2_MEMORY_HEADER = """## Candidate hypotheses from earlier attempts

These are unconfirmed hypotheses extracted from evaluated programs in this
search. They may be irrelevant or harmful and never override the task rules."""

_V2_ANALYSIS = """## 1. Analysis and strategy

Use the same review procedure whether or not a memory hypothesis was assigned:
- If one is present, decide whether it applies to the given program and what it
  would change. It may be wrong or irrelevant.
- If it recommends something already present and still ineffective, treat that
  avenue as spent.
- If none is present or useful, reason from first principles about boundaries,
  gaps, arrangement families, and the optimization formulation.

Choose the implementation yourself. Do not copy an expression from a hypothesis
verbatim and do not let it dictate the complete arrangement."""


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
    def build_prompt(self, parent: ParentContext, memory: str = "",
                     memory_protocol: bool = False) -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)
        n = self.num_circles

        memory_section = ""
        if memory_protocol:
            candidate = ((memory or "").strip()
                         or "(No memory hypothesis was assigned to this control arm.)")
            memory_section = f"\n{_V2_MEMORY_HEADER}\n\n{candidate}\n"
        elif memory and memory.strip():
            memory_section = f"\n{_MEMORY_HEADER}\n\n{memory.strip()}\n"
        if memory_protocol:
            analysis = _V2_ANALYSIS
        else:
            analysis = (_ANALYSIS_WITH_MEMORY
                        if memory_section else _ANALYSIS_NO_MEMORY)

        user = f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {n} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{_VALIDATOR_SRC}
```

{state_ctx}
{memory_section}
{analysis}

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({n}, 2) and radii has shape ({n},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- You can use up to {self.eval_cpus} CPUs.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not catch exceptions in order to return a degenerate packing. A program that
  returns all-zero or near-zero radii when something goes wrong scores the same as
  one that crashes, and it hides the error that would have told you what to fix.
  Let it fail loudly instead.
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
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, tuple) and len(output) == 3):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        centers, radii, _ = output
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array: {e}"
            res.failure_kind = "code"
            return res

        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != self.num_circles:
            res.msg = f"bad_centers_shape: {centers.shape}"
            res.failure_kind = "code"
            return res
        if radii.shape != (self.num_circles,):
            res.msg = f"bad_radii_shape: {radii.shape}"
            res.failure_kind = "code"
            return res

        valid, msg = validate_packing(centers, radii)

        # A packing the validator accepts but whose radii are all (near) zero is
        # not a solution: it is a program that detected its own failure and
        # returned something harmless. Accepting it at reward 0 makes defensive
        # coding free, inflates the valid rate, and gives the memory extractor a
        # pile of "returned zeros" rollouts to learn from. Reject it instead.
        if valid:
            s = float(np.sum(radii))
            if s <= self.degenerate_threshold:
                res.valid = False
                res.msg = (f"degenerate_packing: sum of radii {s:.3e} <= "
                           f"{self.degenerate_threshold:.3e}")
                return res
            res.valid = True
            res.msg = msg
            res.reward = s
            res.raw_score = s
            return res

        res.valid = False
        res.msg = msg
        return res

    @property
    def degenerate_threshold(self) -> float:
        """
        Below this the packing is treated as invalid. Default is deliberately
        tiny: it should catch all-zero and numerically-collapsed returns without
        rejecting a genuinely poor but real packing. Override with
        `degenerate_threshold` in the problem config.
        """
        return float(self.cfg.get("degenerate_threshold", 1e-6))

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        return [SeedState(code="", value=0.0, raw_score=0.0)
                for _ in range(self.num_seed_states)]
