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

Rules:
- You must define the {entrypoint} function: def {entrypoint}() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({n}, 2) and radii has shape ({n},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```."""


PRELUDE = """import numpy as np
import math
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

"""
