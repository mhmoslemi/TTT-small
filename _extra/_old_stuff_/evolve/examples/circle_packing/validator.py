"""
The circle-packing validator.

Kept byte-identical to the reference implementation's, and deliberately in its
own module: its SOURCE is injected into both the prompt and the sandbox prelude
via inspect.getsource, so the model is shown the exact function that will judge
it. Editing this file changes the prompt and the verifier together, which is the
only way they can be guaranteed not to drift apart.
"""

import numpy as np


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
