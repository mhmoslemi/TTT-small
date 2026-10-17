"""
Reward for circle packing.

Given Python code that defines `run_packing()` returning (centers, radii, sum_radii),
we:
  1. Run it in the sandbox
  2. Validate that the circles fit in [0,1]^2 and don't overlap
  3. Reward = sum of radii if valid, else 0

The validator is byte-for-byte the one used in the paper's examples/circle_packing/env.py
so the reward is comparable.
"""

import re
import inspect
import numpy as np

from sandbox import run_code


# ----------------------------------------------------------------------
# Validator (copy of the paper's, kept verbatim for compatibility)
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


# ----------------------------------------------------------------------
# Code extraction
# ----------------------------------------------------------------------
def _extract_code_from_scope(text: str) -> str | None:
    """Extract the last code artifact from one answer scope."""
    # A reasoning model can show several programs. Locate the LAST Python
    # opener first, then take its complete body or (for the existing recovery
    # behavior) its unterminated body through EOS. Checking complete matches
    # first would incorrectly select an older draft when the true final fence
    # is the one truncated at EOS.
    openers = list(re.finditer(r"```python\s*\n?", text, re.IGNORECASE))
    if openers:
        body_start = openers[-1].end()
        closing = text.find("```", body_start)
        body_end = closing if closing >= 0 else len(text)
        code = text[body_start:body_end].strip()
        if code:
            return code

    matches = re.findall(r"```\s*\n?(.*?)```", text, flags=re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code:
            return code

    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped
    return None


def extract_python_code(response: str, *, require_final_marker: bool = False
                        ) -> str | None:
    """
    Pull the final Python answer out of a response.

    When a reasoning-model final-answer delimiter is present, only text after
    the last delimiter is eligible. This prevents executable-looking Python
    drafts in the visible reasoning trace from being evaluated as the answer.
    Models without an explicit delimiter retain the existing last-fence
    behavior unless ``require_final_marker`` is set for a reasoning template
    whose completed answers are contractually delimited.

    Returns None if the final-answer scope contains no code. Returns code with
    no fences otherwise.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    final_markers = (
        "<|channel|>final<|message|>",
        "</think>",
        "</analysis>",
        "</reasoning>",
    )
    marker_end = -1
    for marker in final_markers:
        position = response.lower().rfind(marker.lower())
        if position >= 0:
            marker_end = max(marker_end, position + len(marker))
    if marker_end >= 0:
        # Strictly scope extraction to the final answer. If the model ended its
        # reasoning but never emitted final code, failing the rollout is safer
        # than executing one of its earlier experimental snippets.
        return _extract_code_from_scope(response[marker_end:])

    if require_final_marker:
        return None
    if re.search(r"<think\b", response, flags=re.IGNORECASE):
        # An opening reasoning tag with no closing marker has no final answer.
        return None
    return _extract_code_from_scope(response)


# ----------------------------------------------------------------------
# Main reward function
# ----------------------------------------------------------------------
def compute_reward(response: str, num_circles: int, timeout_s: float = 60.0):
    """
    Score a model response for the circle packing problem.

    Returns a dict:
      {
        "reward":     float (sum of radii, or 0.0),
        "valid":      bool (did packing pass validation?),
        "parsed":     bool (did we extract code at all?),
        "ran":        bool (did the code execute without error?),
        "msg":        str (human-readable status),
        "stdout":     str,
        "centers":    np.ndarray or None,
        "radii":      np.ndarray or None,
      }
    """
    out = {
        "reward": 0.0, "valid": False, "parsed": False, "ran": False,
        "msg": "", "stdout": "", "centers": None, "radii": None,
    }

    code = extract_python_code(response)
    if code is None:
        out["msg"] = "no_code_block"
        return out
    out["parsed"] = True

    # Inject the validator into the sandbox so the model can call it if it tries.
    # Also import numpy/math at the top — many models forget the imports.
    prelude = (
        "import numpy as np\n"
        "import math\n"
        "try:\n"
        "    from scipy.optimize import minimize\n"
        "except ImportError:\n"
        "    minimize = None\n"
        "\n"
        + inspect.getsource(validate_packing)
        + "\n"
    )
    full_code = prelude + "\n# ---- model code below ----\n" + code

    sandbox_out = run_code(full_code, entrypoint="run_packing", timeout_s=timeout_s)
    out["stdout"] = sandbox_out.get("stdout", "")

    if not sandbox_out["ok"]:
        out["msg"] = f"run_failed: {sandbox_out.get('error', 'unknown')}"
        return out
    out["ran"] = True

    value = sandbox_out.get("value")
    if not (isinstance(value, tuple) and len(value) == 3):
        out["msg"] = "bad_return_shape"
        return out

    centers, radii, _ = value
    centers = np.asarray(centers)
    radii = np.asarray(radii).ravel()

    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != num_circles:
        out["msg"] = f"bad_centers_shape: {centers.shape}"
        return out
    if radii.shape != (num_circles,):
        out["msg"] = f"bad_radii_shape: {radii.shape}"
        return out

    valid, msg = validate_packing(centers, radii)
    out["valid"] = valid
    out["msg"] = msg
    out["centers"] = centers
    out["radii"] = radii

    if valid:
        out["reward"] = float(np.sum(radii))

    return out


if __name__ == "__main__":
    # Simple hexagonal lattice for n=26 — should be valid, ~2.6 sum of radii
    test_response = """Here is my solution:
```python
import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    r = 0.1
    centers = []
    radii = []
    rows = 5
    for row in range(rows):
        y = r + row * r * np.sqrt(3)
        offset = r if row % 2 else 2*r
        for col in range(5):
            x = offset + col * 2*r
            if len(centers) < n:
                centers.append([x, y])
                radii.append(r)
    while len(centers) < n:
        centers.append([0.5, 0.5 + 0.1*len(centers)])
        radii.append(0.001)
    centers = np.array(centers)
    radii = np.array(radii)
    return centers, radii, float(radii.sum())
```
"""
    result = compute_reward(test_response, num_circles=26)
    print("reward:", result["reward"])
    print("valid:", result["valid"])
    print("msg:", result["msg"])
