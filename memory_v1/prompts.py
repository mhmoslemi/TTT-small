"""
Memory-maker prompts, the parser for what they return, and the injection of
retrieved lessons into a generation prompt.

Two things worth being explicit about, because both are places the paper is
tighter than any implementation can be for free:

1. The extraction prompts process a whole group in ONE call. Sec. 2.2 is
   specific about this: "the extracted lessons summarize patterns across the
   entire group; they are not separate notes for individual responses."
   Processing jointly is what lets the model separate a systematic pattern
   from an accident of one trajectory, so the examples are numbered and the
   prompt asks for patterns that hold across several of them.

2. Fig. 1 orders the generation prompt as
   [ meta d | parent node | top-m memories | instruction ].
   Every problem in problems/ emits a single user message that already
   contains d, the parent, and the instruction as one block, so inserting the
   memories in the middle would mean editing all six problem files. inject
   mode "append" therefore puts the memory block at the END of the last user
   message, after the instruction, and restates in one line that the lessons
   apply to the program being written. Mode "system" puts them in a separate
   system message instead. Both are deviations from the figure's exact order;
   "append" keeps the memories nearest the generation point, which is the
   position recency favours.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

from memory_v1.types import FAILURE, SUCCESS, Lesson, RolloutRecord

# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
_SCHEMA = (
    'Return ONLY a JSON array, no prose before or after it, no markdown '
    'fences. Exactly {n} objects, each with these keys:\n'
    '  "title":   under 10 words, specific, not a restatement of the task\n'
    '  "summary": one sentence, the lesson in its shortest useful form\n'
    '  "lesson":  2 to 5 sentences, concrete and actionable, naming the '
    'technique, parameter, or structure involved\n'
)

_QUALITY_BAR = (
    "Rules for every lesson:\n"
    "- It must generalize past the specific attempts shown. A lesson that only "
    "describes one attempt is useless later.\n"
    "- It must be specific enough to act on. \"be more careful\" and \"try "
    "harder\" are not lessons; \"push circles into the four corners first, then "
    "grow the interior radii\" is.\n"
    "- Do not restate the problem statement or the scoring rule.\n"
    "- Do not include code blocks. Refer to techniques by name.\n"
    "- If the attempts do not support {n} distinct lessons, still return {n} "
    "objects, but make the weaker ones narrower rather than vaguer.\n"
)


def build_positive_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int,
                            max_chars_per_example: int) -> List[Dict]:
    """prompt+ over S_t: strategies shared by the successful attempts."""
    blocks = []
    for i, rec in enumerate(records, 1):
        blocks.append(f"### Successful attempt {i}\n"
                      + rec.render_success(max_chars_per_example))
    user = (
        f"You are analyzing a batch of SUCCESSFUL attempts at the following "
        f"task, in order to write down what made them work.\n\n"
        f"## Task\n{meta_description}\n\n"
        f"## Attempts ({len(records)} of them, all valid)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        f"Identify the strategies these attempts have IN COMMON, and the "
        f"choices that separate the higher-reward ones from the lower-reward "
        f"ones. Write {num_lessons} lessons that a future attempt at this same "
        f"task should follow.\n\n"
        + _QUALITY_BAR.format(n=num_lessons) + "\n"
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


def build_negative_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int,
                            max_chars_per_example: int,
                            feedback_chars: int) -> List[Dict]:
    """prompt- over F_t: common failure modes and how to prevent them."""
    blocks = []
    for i, rec in enumerate(records, 1):
        blocks.append(f"### Failed attempt {i}\n"
                      + rec.render_failure(max_chars_per_example, feedback_chars))
    user = (
        f"You are analyzing a batch of FAILED attempts at the following task, "
        f"in order to write down how to avoid repeating them.\n\n"
        f"## Task\n{meta_description}\n\n"
        f"## Attempts ({len(records)} of them, all invalid or crashed)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        f"Identify the failure modes that recur across these attempts, not the "
        f"one-off accidents. For each, state the preventative measure: what a "
        f"future attempt should do differently so the same verifier message "
        f"does not come back. Write {num_lessons} lessons.\n\n"
        + _QUALITY_BAR.format(n=num_lessons) + "\n"
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _candidate_json_spans(text: str) -> List[str]:
    """Ordered guesses at where the JSON array is, most likely first."""
    out = []
    for m in _FENCE_RE.finditer(text):
        out.append(m.group(1).strip())
    # Widest bracketed span: tolerates a trailing sentence after the array.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        out.append(text[start:end + 1])
    out.append(text.strip())
    return out


def parse_lessons(response_text: str, outcome: str, step: int,
                  expected: int) -> List[Lesson]:
    """
    Turn one memory-maker response into Lesson objects.

    A model that ignores the schema should cost a step's lessons, not the run,
    so every failure path here returns what it could recover (possibly nothing)
    instead of raising.
    """
    text = _THINK_RE.sub("", response_text or "").strip()
    if not text:
        return []

    items = None
    for span in _candidate_json_spans(text):
        try:
            parsed = json.loads(span)
        except Exception:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("lessons", parsed.get("items"))
        if isinstance(parsed, list) and parsed:
            items = parsed
            break

    if items is None:
        items = _parse_loose(text)

    lessons = []
    for item in items[: max(expected, 0) or None]:
        if not isinstance(item, dict):
            if isinstance(item, str) and item.strip():
                lessons.append(Lesson.create(
                    title=item.strip()[:80], summary=item.strip(),
                    lesson=item.strip(), outcome=outcome, step=step))
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        body = str(item.get("lesson", item.get("detail", ""))).strip()
        if not body and not summary:
            continue
        lessons.append(Lesson.create(title=title or summary[:80],
                                     summary=summary or body[:200],
                                     lesson=body or summary,
                                     outcome=outcome, step=step))
    return lessons


_NUMBERED_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s+(.{20,})$", re.MULTILINE)


def _parse_loose(text: str) -> List[Dict]:
    """Last resort: read a numbered or bulleted list as one lesson per item."""
    out = []
    for m in _NUMBERED_RE.finditer(text):
        body = m.group(1).strip()
        head = body.split(".")[0][:80]
        out.append({"title": head, "summary": body[:200], "lesson": body})
    return out


# ----------------------------------------------------------------------
# Retrieval query + injection
# ----------------------------------------------------------------------
def parent_query_text(meta_description: str, parent_summary: str,
                      parent_code: str, limit: int = 4000) -> str:
    """
    e(p) in Eq. 7 is the embedding of the parent state. The task description is
    included because it anchors the vector to this problem, which matters when
    a bank is reloaded across runs; the code carries the actual position in
    solution space.
    """
    parts = [meta_description or "", parent_summary or ""]
    if parent_code:
        parts.append(parent_code[:limit])
    return "\n".join(p for p in parts if p)


def render_memory_block(lessons: Sequence[Lesson], max_chars: int = 900) -> str:
    if not lessons:
        return ""
    successes = [l for l in lessons if l.outcome == SUCCESS]
    failures = [l for l in lessons if l.outcome == FAILURE]
    lines = ["## Lessons from earlier attempts at this task",
             "These were extracted from previously evaluated programs. Apply "
             "the ones that are relevant; they are evidence, not orders."]
    if successes:
        lines.append("\n### Strategies that worked")
        for i, l in enumerate(successes, 1):
            lines.append(f"{i}. {l.title}\n   {l.lesson[:max_chars]}")
    if failures:
        lines.append("\n### Failure modes to avoid")
        for i, l in enumerate(failures, 1):
            lines.append(f"{i}. {l.title}\n   {l.lesson[:max_chars]}")
    return "\n".join(lines)


_INJECT_TAIL = ("\nUse the lessons above where they apply to the program you "
                "are about to write. Ignore any that do not fit this state.")


def inject_memories(messages: List[Dict], lessons: Sequence[Lesson],
                    mode: str = "append", max_chars: int = 900) -> List[Dict]:
    """
    Return a NEW message list with the retrieved lessons added. The input list
    is not mutated, because problems build it fresh per group and a shared
    reference would accumulate blocks across steps.
    """
    if not lessons:
        return list(messages)

    block = render_memory_block(lessons, max_chars)
    out = [dict(m) for m in messages]

    if mode == "system":
        return [{"role": "system", "content": block}] + out

    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = out[i]["content"] + "\n\n" + block + _INJECT_TAIL
            return out

    out.append({"role": "user", "content": block + _INJECT_TAIL})
    return out
