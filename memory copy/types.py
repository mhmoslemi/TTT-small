"""
Records passed around the memory module (Sec. 2.2).

Two types:

  RolloutRecord   one evaluated rollout as the memory maker sees it. Built in
                  train_step from the same RewardResult the sampler consumes,
                  so no problem file has to change.

  Lesson          one extracted memory. The paper stores "a title, a short
                  summary, the full lesson, its outcome type, the extraction
                  step, and an embedding used for retrieval" -- those are
                  exactly the fields here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SUCCESS = "success"
FAILURE = "failure"


def _clip(s: str, limit: int) -> str:
    s = s or ""
    if limit <= 0 or len(s) <= limit:
        return s
    head = limit // 2
    tail = limit - head
    return s[:head] + "\n...[truncated]...\n" + s[-tail:]


_STRATEGY_RE = re.compile(r"<strategy>(.*?)</strategy>", re.DOTALL | re.IGNORECASE)


def extract_strategy(response_text: str) -> str:
    """
    The problem prompts ask for a <strategy> block before the program. When it
    is there it is the single most informative part of a response for lesson
    extraction, because it states the intent rather than the implementation.
    """
    m = _STRATEGY_RE.search(response_text or "")
    return m.group(1).strip() if m else ""


@dataclass
class RolloutRecord:
    """One (x_i, a_i, s'_i, r_i) or (x_i, a_i, f_i) tuple from step t."""

    step: int = 0
    group: int = 0
    rollout: int = 0

    parent_summary: str = ""      # what x_p described: the parent's standing
    response: str = ""            # a_i, the raw model output
    code: str = ""                # s'_i, the extracted program (may be empty)
    reward: float = 0.0           # r_i
    raw_score: Optional[float] = None
    valid: bool = False
    parsed: bool = False
    ran: bool = False
    msg: str = ""                 # verifier tag: no_code_block / run_failed: ...
    stdout: str = ""              # the executor's own words

    def is_success(self, fail_score: float = 0.0) -> bool:
        """
        Paper: S_t is {r_i > 0}, F_t is {r_i = 0 or the attempt failed}.

        `valid` is required as well as the reward test because a problem whose
        fail_score is negative would otherwise route a merely-invalid rollout
        into the positive group.
        """
        return bool(self.valid) and float(self.reward) > float(fail_score)

    def feedback(self, limit: int = 800) -> str:
        """f_i: everything the verifier said about why this attempt failed."""
        parts = []
        if self.msg:
            parts.append(f"verifier: {self.msg}")
        elif not self.valid:
            parts.append("verifier: invalid (no message)")
        if self.stdout and self.stdout.strip():
            # The tail carries the traceback; the head is usually progress noise.
            parts.append("stdout tail:\n" + (self.stdout.strip()[-limit:]))
        return "\n".join(parts) if parts else "no verifier output"

    def render_success(self, max_chars: int = 1500) -> str:
        strategy = extract_strategy(self.response)
        score = (f"{self.raw_score:.6f}" if self.raw_score is not None
                 else f"{self.reward:.6f}")
        body = [f"reward = {self.reward:.6f}  (metric = {score})"]
        if strategy:
            body.append("stated strategy:\n" + _clip(strategy, max_chars // 2))
        body.append("program:\n```python\n"
                    + _clip(self.code or "(no code captured)", max_chars)
                    + "\n```")
        return "\n".join(body)

    def render_failure(self, max_chars: int = 1500, feedback_chars: int = 800) -> str:
        strategy = extract_strategy(self.response)
        body = [f"reward = {self.reward:.6f}  (failed)"]
        if strategy:
            body.append("stated strategy:\n" + _clip(strategy, max_chars // 3))
        if self.code:
            body.append("attempted program:\n```python\n"
                        + _clip(self.code, max_chars) + "\n```")
        else:
            body.append("attempted response (no parsable program):\n"
                        + _clip(self.response, max_chars // 2))
        body.append("verifier feedback:\n" + self.feedback(feedback_chars))
        return "\n".join(body)

    def failure_signature(self) -> str:
        """
        A coarse key for grouping identical failures, so one repeated crash does
        not fill the whole extraction prompt. Digits are stripped because line
        numbers and array sizes differ between otherwise identical errors.
        """
        base = (self.msg or "unknown").strip().lower()
        base = re.sub(r"\d+", "#", base)
        return base[:120]


@dataclass
class Lesson:
    """One entry in the memory bank M."""

    id: str = ""
    title: str = ""
    summary: str = ""
    lesson: str = ""
    outcome: str = SUCCESS        # SUCCESS | FAILURE
    step: int = 0
    embedding: Optional[List[float]] = None
    uses: int = 0                 # how many prompts have retrieved it

    @staticmethod
    def make_id(title: str, lesson: str, outcome: str) -> str:
        h = hashlib.blake2b(
            f"{outcome}|{title}|{lesson}".encode("utf-8", "replace"),
            digest_size=8,
        )
        return h.hexdigest()

    @classmethod
    def create(cls, title: str, summary: str, lesson: str,
               outcome: str, step: int) -> "Lesson":
        title = (title or "").strip() or "untitled"
        summary = (summary or "").strip()
        lesson = (lesson or "").strip()
        return cls(
            id=cls.make_id(title, lesson, outcome),
            title=title, summary=summary, lesson=lesson,
            outcome=outcome, step=int(step),
        )

    def text_for_embedding(self) -> str:
        return f"{self.title}\n{self.summary}\n{self.lesson}"

    def render(self, max_chars: int = 900) -> str:
        tag = "WHAT WORKED" if self.outcome == SUCCESS else "WHAT FAILED"
        body = self.lesson or self.summary
        return f"[{tag}] {self.title}\n{_clip(body, max_chars)}"

    def to_dict(self) -> Dict:
        d = {
            "id": self.id, "title": self.title, "summary": self.summary,
            "lesson": self.lesson, "outcome": self.outcome, "step": self.step,
            "uses": self.uses,
        }
        if self.embedding is not None:
            d["embedding"] = [float(x) for x in self.embedding]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Lesson":
        return cls(
            id=d.get("id", ""), title=d.get("title", ""),
            summary=d.get("summary", ""), lesson=d.get("lesson", ""),
            outcome=d.get("outcome", SUCCESS), step=int(d.get("step", 0)),
            embedding=d.get("embedding"), uses=int(d.get("uses", 0)),
        )
