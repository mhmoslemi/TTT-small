"""
Code extraction and reasoning stripping.

The failure this guards: a reasoning model writes draft programs inside
<think> while arguing with itself, then writes the real one after. Taking the
last fenced block blindly, or not stripping <think> at all, submits the wrong
program -- or a truncated fragment of one.
"""
import pytest

from envs.base import extract_python_code, has_complete_code_block, strip_reasoning


# ---------------- reasoning is removed before anything else ----------------
def test_think_block_is_stripped():
    assert "draft" not in strip_reasoning("<think>draft</think>\nreal")

def test_unterminated_think_returns_none():
    """Still reasoning when the budget ran out -- there is no submitted answer."""
    assert strip_reasoning("<think>still going and going") is None

def test_no_think_block_passes_through():
    assert strip_reasoning("just text") == "just text"

def test_a_draft_inside_think_is_not_submitted():
    response = ("<think>Maybe:\n```python\nBAD = 1\n```\nno, wrong.</think>\n"
                "```python\nGOOD = 1\n```")
    assert extract_python_code(response) == "GOOD = 1"

def test_truncated_mid_think_yields_no_code_even_with_a_block_inside():
    response = "<think>trying\n```python\nDRAFT = 1\n```\nhmm, but what if"
    assert extract_python_code(response) is None


# ---------------- the four extraction passes ----------------
def test_pass1_well_formed_block():
    assert extract_python_code("```python\nx = 1\n```") == "x = 1"

def test_pass1_takes_the_last_complete_block():
    assert extract_python_code(
        "```python\nfirst = 1\n```\nbetter:\n```python\nsecond = 2\n```") == "second = 2"

def test_pass1_skips_empty_trailing_blocks():
    """A degenerate tail can leave an empty fence after the real program."""
    assert extract_python_code(
        "```python\nreal = 1\n```\nand\n```python\n```") == "real = 1"

def test_pass2_unterminated_block_is_still_recovered():
    """Ran out of budget mid-program; it often still parses and runs."""
    assert extract_python_code(
        "here:\n```python\ndef run():\n    return 1") == "def run():\n    return 1"

def test_pass3_untagged_fence():
    assert extract_python_code("```\ny = 2\n```") == "y = 2"

def test_pass4_bare_program():
    assert extract_python_code("import numpy as np\nz = 3") == "import numpy as np\nz = 3"

def test_prose_yields_nothing():
    assert extract_python_code("I would use a hexagonal lattice.") is None
    assert extract_python_code("") is None
    assert extract_python_code(None) is None


# ---------------- the stop criterion's predicate ----------------
def test_complete_block_is_detected():
    assert has_complete_code_block("```python\nx = 1\n```")

def test_open_block_is_not_complete():
    assert not has_complete_code_block("```python\nx = 1")

def test_a_block_inside_think_does_not_count_as_done():
    """Stopping there would submit a draft the model was still arguing with."""
    assert not has_complete_code_block("<think>```python\ndraft = 1\n```")

def test_block_after_a_closed_think_counts():
    assert has_complete_code_block("<think>reasoned</think>\n```python\nx = 1\n```")


# ---------------- the real-world degenerate transcript ----------------
def test_the_observed_loop_extracts_the_first_real_program():
    """
    Condensed from an actual rollout: forced </think>, a program, then the model
    reopened <strategy> and started a second, truncated one.
    """
    response = (
        "<think>lots of reasoning</think>\n"
        "I have analysed enough. Final program:\n"
        "```python\nimport numpy as np\ndef run_packing():\n    return 1\n```\n"
        "</think>\n<strategy>reconsidering</strategy>\n"
        "```python\ncenters = centers[:8] + np.array([0.5, 0.5], ...")
    code = extract_python_code(response)
    assert code is not None
    # The complete block wins over the truncated fragment that follows it.
    assert "def run_packing" in code
    assert "centers[:8]" not in code
