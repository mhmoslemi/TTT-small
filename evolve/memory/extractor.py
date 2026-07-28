"""
Lesson extraction (Sec. 2.2).

After the B_t rollouts of step t are verified they are split by outcome

    S_t = {(x_i, a_i, s'_i, r_i) : r_i > 0}
    F_t = {(x_i, a_i, f_i)       : r_i = 0 or the attempt failed}

and each group is processed in ONE LLM call, not one call per response. That is
the point: shown the whole group at once, the model can tell a systematic
pattern from an accident of a single trajectory. Exactly 2L lessons per step.
"""

import json
import re
from typing import List, Optional, Sequence, Tuple

from core.types import OUTCOME_FAILURE, OUTCOME_SUCCESS, Lesson
from memory import prompts as memory_prompts

_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_lessons(text: str, outcome: str, step: int, limit: int) -> List[Lesson]:
    """
    Parse the extractor's reply. Tries strict JSON, then a fenced block, then a
    numbered-list fallback, so a model that ignores the format still contributes.
    """
    if not text:
        return []

    candidates = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.append(block.group(0))
    candidates.append(text)

    for raw in candidates:
        try:
            data = json.loads(raw.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        out = []
        for item in data[:limit]:
            if not isinstance(item, dict):
                continue
            out.append(Lesson(
                title=str(item.get("title", ""))[:200],
                summary=str(item.get("summary", ""))[:1000],
                body=str(item.get("lesson", item.get("body", "")))[:4000],
                outcome=outcome,
                step=step,
            ))
        if out:
            return out

    # Fallback: split a numbered or bulleted list into lessons.
    chunks = re.split(r"\n\s*(?:\d+[.)]|[-*])\s+", "\n" + text)
    out = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue
        first, _, rest = chunk.partition("\n")
        out.append(Lesson(title=first.strip()[:200], summary=first.strip()[:1000],
                          body=(rest.strip() or first.strip())[:4000],
                          outcome=outcome, step=step))
        if len(out) >= limit:
            break
    return out


class LessonExtractor:
    def __init__(self, cfg, llm, problem_description: str = ""):
        self.cfg = cfg
        self.llm = llm
        self.problem_description = problem_description

    # ------------------------------------------------------------------
    def _render_success(self, items: Sequence[Tuple[str, float]]) -> str:
        limit = int(self.cfg.max_chars_per_example)
        blocks = []
        for i, (code, reward) in enumerate(items, 1):
            blocks.append(f"### Attempt {i} — reward {reward:.6f}\n"
                          f"```python\n{(code or '').strip()[:limit]}\n```")
        return "\n\n".join(blocks)

    def _render_failure(self, items: Sequence[Tuple[str, str]]) -> str:
        limit = int(self.cfg.max_chars_per_example)
        blocks = []
        for i, (code, feedback) in enumerate(items, 1):
            blocks.append(f"### Attempt {i} — FAILED\n"
                          f"Verifier said: {(feedback or 'no feedback').strip()[:1200]}\n"
                          f"```python\n{(code or '(no code extracted)').strip()[:limit]}\n```")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    def extract(self, successes: Sequence[Tuple[str, float]],
                failures: Sequence[Tuple[str, str]], step: int) -> List[Lesson]:
        """One call per non-empty group; up to 2L lessons."""
        L = int(self.cfg.lessons_per_group)
        if L <= 0:
            return []

        calls = []
        if successes:
            calls.append((
                memory_prompts.positive_prompt(
                    self.problem_description, self._render_success(successes),
                    len(successes), L),
                OUTCOME_SUCCESS))
        if failures:
            calls.append((
                memory_prompts.negative_prompt(
                    self.problem_description, self._render_failure(failures),
                    len(failures), L),
                OUTCOME_FAILURE))
        if not calls:
            return []

        replies = self.llm.chat_batch(
            [messages for messages, _ in calls],
            max_new_tokens=int(self.cfg.extractor_max_tokens),
            temperature=float(self.cfg.extractor_temperature),
        )

        lessons: List[Lesson] = []
        for reply, (_, outcome) in zip(replies, calls):
            lessons.extend(parse_lessons(reply, outcome, step, L))
        return lessons
