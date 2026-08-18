"""
Records passed around the memory module (Sec. 2.2).

  RolloutRecord   one evaluated rollout as the memory maker sees it. Built in
                  train_step from the same RewardResult the sampler consumes,
                  so no problem file has to change.

  Lesson          one entry in the bank. The paper stores "a title, a short
                  summary, the full lesson, its outcome type, the extraction
                  step, and an embedding used for retrieval". `importance` is
                  added on top: the extractor assigns it, and it is raised
                  again whenever a later step confirms the same lesson, so a
                  finding that keeps proving out survives eviction and outranks
                  an equally similar but unconfirmed neighbour.

Field convention, enforced in the extraction prompts:

  title    written for a human reading the log. Plain English, no shorthand.
  summary  one plain sentence, also human-facing.
  lesson   written for the model that will read it back. Compression is
           allowed and encouraged: symbols, parameter names, terse notation.
           Nobody has to be able to read it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

SUCCESS = "success"
FAILURE = "failure"

IMPORTANCE_MIN = 1.0
IMPORTANCE_MAX = 5.0
IMPORTANCE_DEFAULT = 3.0


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

    parent_summary: str = ""
    response: str = ""
    code: str = ""
    reward: float = 0.0
    raw_score: Optional[float] = None
    valid: bool = False
    parsed: bool = False
    ran: bool = False
    msg: str = ""
    stdout: str = ""

    def is_success(self, fail_score: float = 0.0) -> bool:
        """
        Paper: S_t is {r_i > 0}, F_t is {r_i = 0 or the attempt failed}.

        `valid` is required as well as the reward test because a problem whose
        fail_score is negative would otherwise route a merely-invalid rollout
        into the positive group.
        """
        return bool(self.valid) and float(self.reward) > float(fail_score)

    def feedback(self, limit: int = 800) -> str:
        parts = []
        if self.msg:
            parts.append(f"verifier: {self.msg}")
        elif not self.valid:
            parts.append("verifier: invalid (no message)")
        if self.stdout and self.stdout.strip():
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
        Coarse key for grouping identical failures, so one repeated crash does
        not fill the extraction prompt. Digits are stripped because line
        numbers and array sizes differ between otherwise identical errors.
        """
        base = (self.msg or "unknown").strip().lower()
        base = re.sub(r"\d+", "#", base)
        return base[:120]


@dataclass
class Lesson:
    """One entry in the memory bank M."""

    id: str = ""
    title: str = ""               # human-facing
    summary: str = ""             # human-facing
    lesson: str = ""              # model-facing, compression allowed
    outcome: str = SUCCESS
    step: int = 0
    importance: float = IMPORTANCE_DEFAULT
    embedding: Optional[List[float]] = None
    uses: int = 0                 # how many prompts have retrieved it
    confirmations: int = 0        # how many later steps reinforced it

    @staticmethod
    def make_id(title: str, lesson: str, outcome: str) -> str:
        h = hashlib.blake2b(
            f"{outcome}|{title}|{lesson}".encode("utf-8", "replace"),
            digest_size=4,
        )
        return h.hexdigest()       # 8 hex chars, short enough to cite in a prompt

    @classmethod
    def create(cls, title: str, summary: str, lesson: str, outcome: str,
               step: int, importance: float = IMPORTANCE_DEFAULT) -> "Lesson":
        title = (title or "").strip() or "untitled"
        summary = (summary or "").strip()
        lesson = (lesson or "").strip()
        return cls(
            id=cls.make_id(title, lesson, outcome),
            title=title, summary=summary, lesson=lesson,
            outcome=outcome, step=int(step),
            importance=clamp_importance(importance),
        )

    def text_for_embedding(self) -> str:
        return f"{self.title}\n{self.summary}\n{self.lesson}"

    def render(self, max_chars: int = 900) -> str:
        body = self.lesson or self.summary
        return f"{self.title}\n   {_clip(body, max_chars)}"

    def catalog_line(self, chars: int = 160) -> str:
        """One line for the 'already recorded' list shown to the memory maker."""
        tag = "+" if self.outcome == SUCCESS else "-"
        text = (self.summary or self.lesson or "").replace("\n", " ")
        return (f"[{self.id}] ({tag} imp {self.importance:.1f}) "
                f"{self.title} :: {text[:chars]}")

    def to_dict(self) -> Dict:
        d = {
            "id": self.id, "title": self.title, "summary": self.summary,
            "lesson": self.lesson, "outcome": self.outcome, "step": self.step,
            "importance": float(self.importance), "uses": self.uses,
            "confirmations": self.confirmations,
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
            importance=clamp_importance(d.get("importance", IMPORTANCE_DEFAULT)),
            embedding=d.get("embedding"), uses=int(d.get("uses", 0)),
            confirmations=int(d.get("confirmations", 0)),
        )


def clamp_importance(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return IMPORTANCE_DEFAULT
    return max(IMPORTANCE_MIN, min(IMPORTANCE_MAX, x))
