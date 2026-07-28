"""
Circle-packing prompt text.

Split into the two halves Fig. 1 distinguishes: META is the problem description
d, stable for the whole run and identical in every prompt; INSTRUCTION is what
to do this turn. Keeping them apart is what lets the builder slot the parent
node and the retrieved lessons between them.
"""

META = """You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {n} circles in a unit square [0,1]x[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{validator_src}
```

Target {metric_name}: {target} ({direction})."""


INSTRUCTION = """Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

HARD TIME LIMIT: {entrypoint}() is killed after {timeout:.0f} seconds and scores
ZERO. This is the constraint that most often costs points. Budget for it:
- A global optimizer over all 3n variables (differential_evolution, basinhopping,
  dual_annealing) will NOT finish in time at n={n}. Do not use one.
- A rejection-sampling objective that returns a large constant for any invalid
  configuration gives the optimizer nothing to descend and wastes the budget.
- What fits: a constructive layout, then a bounded local refinement -- a fixed
  iteration count of SLSQP, or a few thousand hill-climbing steps. Vectorize the
  pairwise distances with numpy; a Python double loop over pairs is too slow to
  iterate usefully.
- Leave margin. Aim to return in well under {timeout:.0f}s.

Rules:
- You must define the {entrypoint} function: def {entrypoint}() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({n}, 2) and radii has shape ({n},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Think first, then answer — but only once each.

While thinking: decide the geometry and the refinement method. Keep it to the
decision. Do NOT draft the program in your reasoning, do not write code you are
going to rewrite, and do not restate the problem or re-derive the area bound.
Settle the approach and stop.

Then output the complete program between ```python and ``` — exactly one
program, and nothing after the closing fence. No summary, no review, no second
attempt, no explanation of what you just wrote.

A complete working program scores; an unfinished one scores zero however good
the reasoning behind it was. Write the program once, and write it to run."""


PRELUDE = """import numpy as np
import math
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

"""
