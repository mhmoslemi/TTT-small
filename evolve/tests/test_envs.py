import numpy as np
import pytest

from config import Config
from core.types import Node
from envs.base import extract_python_code
from envs.sandbox import run_code
from examples.circle_packing.env import CirclePacking, build
from examples.circle_packing.validator import validate_packing


# ---------------- code extraction ----------------
def test_last_fenced_block_wins():
    """Models often show a draft before the final answer."""
    text = "```python\nx = 1\n```\nOn reflection:\n```python\nx = 2\n```"
    assert extract_python_code(text) == "x = 2"

def test_bare_and_unlabelled_fences():
    assert extract_python_code("```\ndef run(): pass\n```") == "def run(): pass"
    assert extract_python_code("import numpy as np\nz = 1") == "import numpy as np\nz = 1"

def test_no_code_returns_none():
    assert extract_python_code("I think hexagonal packing is best.") is None
    assert extract_python_code("") is None


# ---------------- sandbox ----------------
def test_sandbox_returns_the_value():
    out = run_code("def f():\n    return 41 + 1\n", "f", timeout_s=20)
    assert out["ok"] and out["value"] == 42

def test_sandbox_reports_traceback_not_just_the_message():
    """Eq. 9 is only as informative as the feedback text."""
    out = run_code("def f():\n    return 1 / 0\n", "f", timeout_s=20)
    assert not out["ok"]
    assert "ZeroDivisionError" in out["error"]
    # The raising frame and the offending source line both survive, which is
    # what makes the feedback specific enough to localize.
    assert "in f" in out["traceback"] and "return 1 / 0" in out["traceback"]

def test_sandbox_kills_an_infinite_loop():
    out = run_code("def f():\n    while True: pass\n", "f", timeout_s=2)
    assert not out["ok"] and "Timeout" in out["error"]

def test_sandbox_captures_stdout():
    out = run_code("def f():\n    print('hello')\n    return 1\n", "f", timeout_s=20)
    assert out["ok"] and "hello" in out["stdout"]

def test_sandbox_missing_entrypoint_is_an_error_not_a_crash():
    out = run_code("def other(): return 1\n", "f", timeout_s=20)
    assert not out["ok"]


# ---------------- validator ----------------
def test_validator_accepts_a_legal_packing():
    ok, msg = validate_packing(np.array([[0.25, 0.25], [0.75, 0.75]]),
                               np.array([0.2, 0.2]))
    assert ok and msg == "ok"

def test_validator_rejects_overlap_out_of_bounds_negative_and_nan():
    over = validate_packing(np.array([[0.5, 0.5], [0.55, 0.5]]), np.array([0.2, 0.2]))
    assert not over[0] and "overlap" in over[1]
    oob = validate_packing(np.array([[0.95, 0.5]]), np.array([0.2]))
    assert not oob[0] and "outside" in oob[1]
    neg = validate_packing(np.array([[0.5, 0.5]]), np.array([-0.1]))
    assert not neg[0] and "negative" in neg[1]
    nan = validate_packing(np.array([[np.nan, 0.5]]), np.array([0.1]))
    assert not nan[0] and "NaN" in nan[1]


# ---------------- the example ----------------
def _example(n=2, **params):
    cfg = Config()
    cfg.example.params = {"num_circles": n, "entrypoint": "run_packing", **params}
    cfg.verifier.timeout_s = 30
    return build(cfg)

def test_params_drive_the_example_no_hardcoding():
    ex = _example(n=7, target=1.234, metric_name="custom")
    assert ex.num_circles == 7 and ex.target == 1.234 and ex.metric_name == "custom"

def test_target_defaults_only_for_known_sizes():
    assert _example(n=26).target == pytest.approx(2.635983)
    assert _example(n=32).target == pytest.approx(2.940)
    assert _example(n=7).target is None

def test_meta_shows_the_validator_the_run_will_actually_use():
    meta = _example(n=5).meta_description()
    assert "pack 5 circles" in meta
    assert "def validate_packing" in meta        # injected, cannot drift

def test_instruction_names_the_configured_entrypoint():
    assert "run_packing" in _example(n=3).instruction()

def test_end_to_end_reward_is_the_sum_of_radii():
    ex = _example(n=2)
    response = """```python
import numpy as np
def run_packing():
    centers = np.array([[0.25, 0.25], [0.75, 0.75]])
    radii = np.array([0.2, 0.2])
    return centers, radii, 0.4
```"""
    res = ex.verify(response, None)
    assert res.valid and res.reward == pytest.approx(0.4)
    assert res.feedback == ""                    # nothing to correct

def test_invalid_geometry_scores_zero_and_explains_why():
    ex = _example(n=2)
    response = """```python
import numpy as np
def run_packing():
    return np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0.4, 0.4]), 0.8
```"""
    res = ex.verify(response, None)
    assert not res.valid and res.reward == 0.0
    assert "overlap" in res.feedback

def test_wrong_shape_is_rejected_with_actionable_feedback():
    ex = _example(n=5)
    response = """```python
import numpy as np
def run_packing():
    return np.array([[0.5, 0.5]]), np.array([0.1]), 0.1
```"""
    res = ex.verify(response, None)
    assert not res.valid and "(5, 2)" in res.feedback

def test_runtime_error_feedback_carries_the_traceback():
    ex = _example(n=2)
    res = ex.verify("```python\ndef run_packing():\n    raise RuntimeError('boom')\n```", None)
    assert not res.valid
    assert "boom" in res.feedback and "Traceback" in res.feedback

def test_missing_code_block_is_a_failure_with_feedback():
    res = _example(n=2).verify("I would use a hexagonal lattice.", None)
    assert not res.valid and res.msg == "no_code_block"
    assert "code block" in res.feedback

def test_parent_rendering_switches_on_whether_there_is_one():
    ex = _example(n=2)
    assert "from scratch" in ex.render_parent(None)
    node = Node(code="x = 1", reward=1.5, raw_score=1.5, valid=True)
    rendered = ex.render_parent(node)
    assert "1.500000" in rendered and "x = 1" in rendered

def test_feedback_truncation_keeps_both_ends():
    ex = _example(n=2)
    ex.feedback_max_chars = 60
    out = ex.truncate_feedback("HEAD" + "x" * 500 + "TAIL")
    assert out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out
