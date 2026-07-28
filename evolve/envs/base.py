"""
The Example interface (the "problem description d" of Def. 1).

An example owns everything problem-specific: what the candidate space is, how a
response becomes a candidate (the transition T_d), and how a candidate is scored
(the reward R_d and the feedback F_d). The framework owns everything else and
never imports a concrete example -- core/registry.py resolves them by name.

An example reads its knobs from `example.params`, an open namespace in the
config, so adding a problem never means touching the framework's schema.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.types import Node, VerifyResult
from envs.sandbox import run_code

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>", re.IGNORECASE)
_CLOSE_THINK = re.compile(r"</think>", re.IGNORECASE)


def strip_reasoning(response: str) -> Optional[str]:
    """
    Remove <think>...</think> so only the answer is considered.

    Returns None when a block was opened and never closed: the model was still
    reasoning when it ran out of budget, and any fenced code inside a think
    block is a draft it was arguing with, not the program it chose to submit.
    """
    if not response:
        return None
    cleaned = _THINK_BLOCK.sub("", response)
    if _OPEN_THINK.search(cleaned) and not _CLOSE_THINK.search(cleaned):
        return None
    # A forced close can leave a stray closing tag with no opener.
    return _CLOSE_THINK.sub("", cleaned)


def extract_python_code(response: str) -> Optional[str]:
    """
    Pull the submitted program out of a response, in four passes.

    Ported from the reference implementation, whose ordering matters:

      1. a well-formed ```python ... ``` block
      2. an unterminated ```python ... <end of output>, i.e. the model ran out
         of budget mid-program -- still worth running, it often parses
      3. a generic ``` ... ``` block with no language tag
      4. the raw response, when it is plainly a program

    Reasoning is stripped first. Without that, a draft the model wrote inside
    <think> and then rejected can be picked up as the answer.
    """
    cleaned = strip_reasoning(response)
    if not cleaned:
        return None

    # 1) Well-formed block. Last one wins: the prompt asks for the final
    #    program last, and earlier blocks are usually partial sketches.
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", cleaned,
                        re.DOTALL | re.IGNORECASE)
    for code in reversed(blocks):
        if code.strip():
            return code.strip()

    # 2) Opened a block and ran out of budget before closing it.
    match = re.search(r"```(?:python|py)?\s*\n?(.*)$", cleaned,
                      re.DOTALL | re.IGNORECASE)
    if match:
        code = re.sub(r"\n?```\s*$", "", match.group(1)).strip()
        if code:
            return code

    # 3) Any fenced block at all.
    blocks = re.findall(r"```\s*\n?(.*?)```", cleaned, re.DOTALL)
    for code in reversed(blocks):
        if code.strip():
            return code.strip()

    # 4) Unfenced, but it smells like a program.
    stripped = cleaned.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped
    return None


def has_complete_code_block(response: str) -> bool:
    """True once a closed ```python block exists outside any reasoning block."""
    cleaned = strip_reasoning(response)
    if not cleaned:
        return False
    return bool(re.search(r"```(?:python|py)?\s*\n.*?\n?```", cleaned,
                          re.DOTALL | re.IGNORECASE))


class Example(ABC):
    """One use-case. Subclasses live under examples/<name>/env.py."""

    name: str = "example"
    entrypoint: str = "run"
    metric_name: str = "score"
    maximize: bool = True

    def __init__(self, cfg):
        """cfg is the full Config; params live in cfg.example.params."""
        self.cfg = cfg
        self.params: Dict[str, Any] = dict(cfg.example.params or {})
        self.fail_reward = float(cfg.verifier.fail_reward)
        self.timeout_s = float(cfg.verifier.timeout_s)
        self.max_cpus = int(cfg.verifier.max_cpus)
        self.feedback_max_chars = int(cfg.verifier.feedback_max_chars)
        self.entrypoint = str(self.params.get("entrypoint", self.entrypoint))
        self.metric_name = str(self.params.get("metric_name", self.metric_name))
        self.maximize = bool(self.params.get("maximize", self.maximize))

    # ---- the problem description d -----------------------------------
    @abstractmethod
    def meta_description(self) -> str:
        """The stable statement of the problem, shown in every prompt."""

    @abstractmethod
    def instruction(self) -> str:
        """What to do this turn: output format, rules, constraints."""

    def render_parent(self, node: Optional[Node]) -> str:
        """The parent candidate s_t as the prompt should show it."""
        if node is None or node.is_root or not node.code:
            return ("No previous program exists. Write one from scratch.")
        score = node.display_score()
        return (f"The previous program achieved {self.metric_name} = {score:.6f}.\n"
                f"```python\n{node.code}\n```")

    def render_for_judge(self, node: Node) -> str:
        """How a candidate is shown to the Elo judge."""
        score = node.display_score()
        status = "valid" if node.valid else f"INVALID ({node.msg})"
        code = (node.code or "(no code)")[:4000]
        return (f"{self.metric_name}: {score:.6f}  [{status}]\n"
                f"```python\n{code}\n```")

    # ---- the transition T_d ------------------------------------------
    def transition(self, response_text: str, parent: Optional[Node]) -> Optional[str]:
        """Response -> candidate code. None means nothing runnable was produced."""
        return extract_python_code(response_text)

    @abstractmethod
    def preprocess(self, code: str, parent: Optional[Node]) -> str:
        """The full program to execute: prelude + verifier source + model code."""

    # ---- the evaluator R_d, F_d --------------------------------------
    @abstractmethod
    def score(self, value: Any, stdout: str) -> VerifyResult:
        """Turn the entrypoint's return value into a reward and feedback."""

    def seed_nodes(self, count: int) -> List[dict]:
        """Root states s_0. Empty by default: 'write a program from scratch'."""
        return [{} for _ in range(max(1, count))]

    # ---- default verification path -----------------------------------
    def verify(self, response_text: str, parent: Optional[Node]) -> VerifyResult:
        """
        Run the candidate and produce (r, f).

        Every failure path sets `feedback`, because a failed rollout with empty
        feedback contributes nothing to Eq. 9 -- the dense token-level signal is
        exactly the feedback made differentiable.
        """
        code = self.transition(response_text, parent)
        if code is None:
            return VerifyResult(
                reward=self.fail_reward, valid=False, msg="no_code_block",
                feedback="No ```python code block was found in the response. "
                         "The final answer must contain one fenced Python block.")

        out = run_code(self.preprocess(code, parent), entrypoint=self.entrypoint,
                       timeout_s=self.timeout_s, max_cpus=self.max_cpus)
        stdout = out.get("stdout", "") or ""

        if not out.get("ok"):
            error = out.get("error", "unknown error")
            traceback_text = out.get("traceback", "") or ""
            feedback = f"Execution failed: {error}"
            if traceback_text:
                feedback += f"\n\nTraceback:\n{traceback_text}"
            return VerifyResult(
                reward=self.fail_reward, valid=False, code=code, stdout=stdout,
                msg=f"run_failed: {error}"[:300],
                feedback=self.truncate_feedback(feedback))

        result = self.score(out.get("value"), stdout)
        result.code = code
        if not result.stdout:
            result.stdout = stdout
        if not result.valid and not result.msg:
            result.msg = "invalid"
        if not result.feedback and not result.valid:
            result.feedback = self.truncate_feedback(
                f"The candidate ran but was rejected: {result.msg}")
        return result

    def truncate_feedback(self, text: str) -> str:
        limit = self.feedback_max_chars
        if limit <= 0 or len(text) <= limit:
            return text
        # Keep both ends: the exception type is at the top, the raising frame
        # at the bottom, and the middle of a traceback is the least useful part.
        head = limit * 2 // 3
        tail = limit - head
        return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"
