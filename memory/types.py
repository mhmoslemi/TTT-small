"""
Records passed around the memory module.

Changes from the embedding version:

  Lesson.embedding is gone. Nothing is vectorized any more.

  Lesson.scope is new, and it is the field that addresses the Table 3 failure.
  A `local` lesson describes an operation or repair that composes with any
  global layout (clamp positions to the box; shrink a pair to tangency). A
  `global` lesson describes something that determines the overall structure
  (which layout to start from, which optimizer formulation to use). Local
  lessons may carry a few lines of code. Global lessons may not carry code at
  all, because a global rule expressed as code IS the solution, and copying it
  is what turned one memory slot into MEM-C's plateau.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

SUCCESS = "success"
FAILURE = "failure"

LOCAL = "local"
GLOBAL = "global"

IMPORTANCE_MIN = 1.0
IMPORTANCE_MAX = 5.0
IMPORTANCE_DEFAULT = 3.0

_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on", "at",
    "by", "is", "are", "be", "as", "that", "this", "it", "its", "from", "into",
    "during", "when", "if", "then", "so", "use", "using", "used", "ensure",
    "avoid", "prevent", "correct", "proper", "properly", "always", "must",
    "should", "optimization", "optimizer", "process", "step", "steps",
}


def _clip(s: str, limit: int) -> str:
    s = s or ""
    if limit <= 0 or len(s) <= limit:
        return s
    head = limit // 2
    return s[:head] + "\n...[truncated]...\n" + s[-(limit - head):]


def content_tokens(text: str) -> set:
    """Stopword-stripped token set, used for lexical dedup."""
    return {w for w in _WORD_RE.findall((text or "").lower())
            if w not in _STOP and len(w) > 2}


def normalize_title(title: str) -> str:
    """
    Collapse the near-identical titles the extractor keeps producing.
    "Ensure Bounds Compatibility in Optimization" and "Ensure Bounds
    Compatibility" reduce to the same key.
    """
    toks = sorted(content_tokens(title))
    return " ".join(toks)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


_STRATEGY_RE = re.compile(r"<strategy>(.*?)</strategy>", re.DOTALL | re.IGNORECASE)


def extract_strategy(response_text: str) -> str:
    m = _STRATEGY_RE.search(response_text or "")
    return m.group(1).strip() if m else ""


@dataclass
class RolloutRecord:
    """One (x_i, a_i, s'_i, r_i) or (x_i, a_i, f_i) tuple from step t."""

    step: int = 0
    group: int = 0
    rollout: int = 0

    parent_summary: str = ""
    parent_code: str = ""          # for the child-vs-parent comparison
    parent_reward: Optional[float] = None
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
        `valid` is required as well, so a problem with a negative fail_score
        cannot route an invalid rollout into the positive group.
        """
        return bool(self.valid) and float(self.reward) > float(fail_score)

    def delta(self) -> Optional[float]:
        """r_i minus the parent's reward. None when the parent score is unknown."""
        if self.parent_reward is None:
            return None
        return float(self.reward) - float(self.parent_reward)

    def feedback(self, limit: int = 800) -> str:
        parts = [f"verifier: {self.msg}" if self.msg
                 else "verifier: invalid (no message)"]
        tail = (self.stdout or "").strip()
        if tail:
            parts.append("stdout tail:\n" + tail[-limit:])
        return "\n".join(parts)

    def render_success(self, max_chars: int = 1500) -> str:
        strategy = extract_strategy(self.response)
        d = self.delta()
        head = f"reward = {self.reward:.6f}"
        if self.raw_score is not None:
            head += f"  (metric = {self.raw_score:.6f})"
        if d is not None:
            head += (f"  |  parent = {self.parent_reward:.6f}, "
                     f"change = {d:+.6f}")
        body = [head]
        if strategy:
            body.append("stated strategy:\n" + _clip(strategy, max_chars // 2))
        body.append("program:\n```python\n"
                    + _clip(self.code or "(no code captured)", max_chars) + "\n```")
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
        base = re.sub(r"\d+", "#", (self.msg or "unknown").strip().lower())
        return base[:120]


@dataclass
class Lesson:
    """One entry in the memory bank M."""

    id: str = ""
    title: str = ""               # human-facing
    summary: str = ""             # human-facing, and what the selector reads
    lesson: str = ""              # model-facing body
    scope: str = LOCAL            # local | global
    outcome: str = SUCCESS        # success | failure
    step: int = 0
    importance: float = IMPORTANCE_DEFAULT
    uses: int = 0                 # times the model CHOSE this lesson
    confirmations: int = 0        # times a later step reinforced it

    @staticmethod
    def make_id(title: str, lesson: str, outcome: str) -> str:
        h = hashlib.blake2b(
            f"{outcome}|{title}|{lesson}".encode("utf-8", "replace"),
            digest_size=4)
        return h.hexdigest()      # 8 hex chars, short enough for the model to cite

    @classmethod
    def create(cls, title: str, summary: str, lesson: str, outcome: str,
               step: int, scope: str = LOCAL,
               importance: float = IMPORTANCE_DEFAULT) -> "Lesson":
        title = (title or "").strip() or "untitled"
        return cls(
            id=cls.make_id(title, (lesson or "").strip(), outcome),
            title=title, summary=(summary or "").strip(),
            lesson=(lesson or "").strip(),
            scope=(scope if scope in (LOCAL, GLOBAL) else LOCAL),
            outcome=outcome, step=int(step),
            importance=clamp_importance(importance))

    def tokens(self) -> set:
        return content_tokens(f"{self.title} {self.summary} {self.lesson}")

    def render(self, max_chars: int = 900) -> str:
        return f"{self.title}\n   {_clip(self.lesson or self.summary, max_chars)}"

    def catalog_line(self, chars: int = 200) -> str:
        """
        One line of the index the model reads when choosing. Carries enough to
        decide with, and nothing that could be copied: no body, no code.
        """
        tag = "worked" if self.outcome == SUCCESS else "failed"
        text = (self.summary or self.lesson or "").replace("\n", " ")
        return (f"{self.id}  [{self.scope}/{tag}, imp {self.importance:.1f}, "
                f"step {self.step}, used {self.uses}x]  {self.title}"
                f" :: {text[:chars]}")

    def to_dict(self) -> Dict:
        return {"id": self.id, "title": self.title, "summary": self.summary,
                "lesson": self.lesson, "scope": self.scope,
                "outcome": self.outcome, "step": self.step,
                "importance": float(self.importance), "uses": self.uses,
                "confirmations": self.confirmations}

    @classmethod
    def from_dict(cls, d: Dict) -> "Lesson":
        return cls(id=d.get("id", ""), title=d.get("title", ""),
                   summary=d.get("summary", ""), lesson=d.get("lesson", ""),
                   scope=d.get("scope", LOCAL),
                   outcome=d.get("outcome", SUCCESS), step=int(d.get("step", 0)),
                   importance=clamp_importance(d.get("importance", IMPORTANCE_DEFAULT)),
                   uses=int(d.get("uses", 0)),
                   confirmations=int(d.get("confirmations", 0)))


def clamp_importance(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return IMPORTANCE_DEFAULT
    return max(IMPORTANCE_MIN, min(IMPORTANCE_MAX, x))
