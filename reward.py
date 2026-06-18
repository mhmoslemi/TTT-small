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
def extract_python_code(response: str) -> str | None:
    """
    Pull Python code out of a response. We try four strategies, in order:

      1. A ```python ... ``` block (well-formed)
      2. A ```python ... <end of string>  (model hit max_new_tokens mid-block)
      3. A generic ``` ... ``` block (no language tag)
      4. The raw response if it starts with 'import' or 'def '

    Returns None only if all four fail. Returns code with no fences.
    """
    # 0) Strip any <think>...</think> reasoning block (Qwen3, R1, etc.).
    #    If the closing </think> is missing (model still inside thinking when
    #    truncated), we discard everything up to and including the open tag too,
    #    because there's no actual code in there anyway.
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in response and "</think>" not in response:
        # Unterminated thinking — model never produced code
        return None

    # 1) Well-formed ```python ... ``` block, take the LAST one
    matches = re.findall(r"```python\s*\n?(.*?)```", response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code:
            return code

    # 2) Unterminated ```python ... <EOS>: take everything after the last ```python
    m = re.search(r"```python\s*\n?(.*)$", response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        # Strip a trailing ``` if it leaked in
        code = re.sub(r"\n?```\s*$", "", code).strip()
        if code:
            return code

    # 3) Any ``` ... ``` block
    matches = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code:
            return code

    # 4) Raw response if it smells like Python
    stripped = response.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped

    return None


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