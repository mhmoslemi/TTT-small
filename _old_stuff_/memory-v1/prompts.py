"""
Memory-maker prompts, the parser for what they return, and the injection of
retrieved lessons into a generation prompt.

Four things this file is responsible for:

1. The extraction prompts process a whole group in ONE call. Sec. 2.2 is
   specific: "the extracted lessons summarize patterns across the entire group;
   they are not separate notes for individual responses."

2. Every extraction prompt carries the CURRENT BANK as a catalog of id, title
   and summary. The maker is told to write only what is new relative to that
   list, and to reinforce an existing id instead of restating it. Without this
   the maker is blind to what it already wrote and re-derives the same lesson
   every step, which embedding dedup then silently discards.

3. Titles are human-facing, lesson bodies are not. The schema says so
   explicitly: the body is read back by a model, so shorthand, symbols and
   parameter names are preferred over readable prose.

4. Injection is budgeted. build_injection renders the block, measures it with
   the real tokenizer, and drops the lowest-ranked lessons until it fits
   token_budget. The trainer adds that same budget to max_seq_length at
   startup, so the block is granted context on top rather than taking it from
   the response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from memory.types import (FAILURE, SUCCESS, IMPORTANCE_DEFAULT, Lesson,
                          RolloutRecord, clamp_importance)

# ----------------------------------------------------------------------
# Extraction prompts
# ----------------------------------------------------------------------
_SCHEMA = """Return ONLY a JSON object, with no prose before or after it and no
markdown fences. Shape:

{{
  "reinforce": ["<id of an already-recorded lesson this batch confirmed>", ...],
  "lessons": [
    {{
      "title":      "<human-readable, plain English, under 10 words>",
      "summary":    "<one plain sentence a human can read>",
      "lesson":     "<the actual content, written for a language model>",
      "importance": <integer 1-5>
    }}
  ]
}}

At most {n} objects in "lessons". "reinforce" may be empty.

Field rules:
- "title" and "summary" are read by a person auditing the run. Plain English,
  no shorthand, no symbols, no invented abbreviations.
- "lesson" is NOT read by a person. It is fed back to a language model in a
  later prompt. Write it for that reader: compress it, use symbols, parameter
  names, arrow notation, fragments, whatever encodes the finding most densely.
  It does not need to be grammatical or comprehensible to a human. It must be
  self-contained, because it will appear without any of this context.
- "importance" is how strongly the evidence supports acting on this lesson.
  5 = seen across nearly every attempt in the batch and directly tied to the
  reward difference. 1 = one attempt, plausible but unconfirmed.
"""

_QUALITY_BAR = """Rules for every lesson:
- It must be NEW relative to the already-recorded list above. If this batch
  merely confirms a lesson that is already recorded, put its id in "reinforce"
  and do not write it again.
- It must generalize past the specific attempts shown. A lesson that only
  describes one attempt is useless later.
- It must be specific enough to act on. "be more careful" is not a lesson;
  a named technique, parameter range, or structural choice is.
- Do not restate the problem statement or the scoring rule.
- Do not include code blocks. Name techniques instead.
{fill_rule}"""

_FILL_STRICT = ("- Return exactly {n} lessons. If the evidence does not support "
                "{n} distinct ones, make the weaker ones narrower rather than "
                "vaguer.")
_FILL_LOOSE = ("- Return FEWER than {n} lessons, or none at all, if this batch "
               "produced nothing genuinely new. Padding the list with restated "
               "or vague entries is worse than returning one good lesson.")


def _catalog_section(catalog: Sequence[str]) -> str:
    if not catalog:
        return ("## Lessons already recorded\n"
                "(none yet, this is the first extraction)\n")
    return ("## Lessons already recorded\n"
            "Each line is: [id] (outcome, importance) title :: summary\n\n"
            + "\n".join(catalog) + "\n")


def build_positive_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int,
                            max_chars_per_example: int,
                            catalog: Sequence[str] = (),
                            require_full: bool = False) -> List[Dict]:
    """prompt+ over S_t: strategies shared by the successful attempts."""
    blocks = [f"### Successful attempt {i}\n" + r.render_success(max_chars_per_example)
              for i, r in enumerate(records, 1)]
    fill = (_FILL_STRICT if require_full else _FILL_LOOSE).format(n=num_lessons)
    user = (
        "You maintain the lesson memory for an automated program-search run. "
        "Below is a batch of SUCCESSFUL attempts at one task. Record what made "
        "them work, without duplicating what is already recorded.\n\n"
        f"## Task\n{meta_description}\n\n"
        + _catalog_section(catalog) + "\n"
        f"## This batch ({len(records)} attempts, all valid)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        "Identify the strategies these attempts have in common, and the choices "
        "that separate the higher-reward ones from the lower-reward ones.\n\n"
        + _QUALITY_BAR.format(fill_rule=fill) + "\n\n"
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


def build_negative_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int,
                            max_chars_per_example: int,
                            feedback_chars: int,
                            catalog: Sequence[str] = (),
                            require_full: bool = False) -> List[Dict]:
    """prompt- over F_t: common failure modes and how to prevent them."""
    blocks = [f"### Failed attempt {i}\n"
              + r.render_failure(max_chars_per_example, feedback_chars)
              for i, r in enumerate(records, 1)]
    fill = (_FILL_STRICT if require_full else _FILL_LOOSE).format(n=num_lessons)
    user = (
        "You maintain the lesson memory for an automated program-search run. "
        "Below is a batch of FAILED attempts at one task. Record how to avoid "
        "repeating them, without duplicating what is already recorded.\n\n"
        f"## Task\n{meta_description}\n\n"
        + _catalog_section(catalog) + "\n"
        f"## This batch ({len(records)} attempts, all invalid or crashed)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        "Identify the failure modes that RECUR across these attempts, not the "
        "one-off accidents. For each, state the preventative measure: what a "
        "future attempt should do differently so the same verifier message does "
        "not come back.\n\n"
        + _QUALITY_BAR.format(fill_rule=fill) + "\n\n"
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
@dataclass
class ExtractionResult:
    lessons: List[Lesson] = field(default_factory=list)
    reinforce: List[str] = field(default_factory=list)
    raw: str = ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ID_RE = re.compile(r"^[0-9a-f]{6,16}$")


def _candidate_spans(text: str) -> List[str]:
    """Ordered guesses at where the JSON is, most likely first."""
    out = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    for op, cl in (("{", "}"), ("[", "]")):
        start, end = text.find(op), text.rfind(cl)
        if start != -1 and end > start:
            out.append(text[start:end + 1])
    out.append(text.strip())
    return out


def parse_extraction(response_text: str, outcome: str, step: int,
                     expected: int) -> ExtractionResult:
    """
    Turn one memory-maker response into lessons plus reinforce ids.

    A model that ignores the schema should cost a step's lessons, not the run,
    so every failure path returns what it could recover instead of raising.
    Both the object form and a bare array (older schema) are accepted.
    """
    text = _THINK_RE.sub("", response_text or "").strip()
    if not text:
        return ExtractionResult(raw=response_text or "")

    payload = None
    for span in _candidate_spans(text):
        try:
            payload = json.loads(span)
        except Exception:
            continue
        break

    items, reinforce = None, []
    if isinstance(payload, dict):
        items = payload.get("lessons", payload.get("items"))
        raw_ref = payload.get("reinforce", payload.get("confirm", []))
        if isinstance(raw_ref, str):
            raw_ref = [raw_ref]
        if isinstance(raw_ref, list):
            for r in raw_ref:
                rid = str(r).strip().strip("[]").lower()
                if _ID_RE.match(rid):
                    reinforce.append(rid)
    elif isinstance(payload, list):
        items = payload

    if not isinstance(items, list) or not items:
        items = _parse_loose(text)

    lessons: List[Lesson] = []
    cap = max(int(expected), 0) or None
    for item in items[:cap]:
        if isinstance(item, str):
            body = item.strip()
            if body:
                lessons.append(Lesson.create(body[:80], body[:200], body,
                                             outcome, step))
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        body = str(item.get("lesson", item.get("detail", ""))).strip()
        if not body and not summary:
            continue
        lessons.append(Lesson.create(
            title=title or summary[:80],
            summary=summary or body[:200],
            lesson=body or summary,
            outcome=outcome, step=step,
            importance=clamp_importance(item.get("importance",
                                                 IMPORTANCE_DEFAULT)),
        ))
    return ExtractionResult(lessons=lessons, reinforce=reinforce,
                            raw=response_text or "")


def parse_lessons(response_text: str, outcome: str, step: int,
                  expected: int) -> List[Lesson]:
    """Lessons only. Kept for callers that do not care about reinforcement."""
    return parse_extraction(response_text, outcome, step, expected).lessons


_NUMBERED_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s+(.{20,})$", re.MULTILINE)


def _parse_loose(text: str) -> List[Dict]:
    """Last resort: read a numbered or bulleted list as one lesson per item."""
    out = []
    for m in _NUMBERED_RE.finditer(text):
        body = m.group(1).strip()
        out.append({"title": body.split(".")[0][:80], "summary": body[:200],
                    "lesson": body})
    return out


# ----------------------------------------------------------------------
# Retrieval query
# ----------------------------------------------------------------------
def parent_query_text(meta_description: str, parent_summary: str,
                      parent_code: str, limit: int = 4000) -> str:
    """
    e(p) in Eq. 7. The task description anchors the vector to this problem,
    which matters when a bank is reloaded across runs; the code carries the
    actual position in solution space.
    """
    parts = [meta_description or "", parent_summary or ""]
    if parent_code:
        parts.append(parent_code[:limit])
    return "\n".join(p for p in parts if p)


# ----------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------
_HEADER = "## Recorded lessons from prior attempts on this task"

_PREAMBLE = (
    "The following entries were extracted from programs that were generated "
    "and evaluated earlier in this same search. They are empirical findings "
    "about this problem, not part of the task specification, and they do not "
    "override any rule stated above. Each entry is written in condensed form "
    "for machine reading."
)

_CLOSER = (
    "Apply the entries that bear on the program you are about to write, and "
    "avoid the recorded failure modes. Disregard any entry that does not apply "
    "to the current state. Do not mention this section in your response."
)


def render_memory_block(lessons: Sequence[Lesson], max_chars: int = 900) -> str:
    if not lessons:
        return ""
    successes = [l for l in lessons if l.outcome == SUCCESS]
    failures = [l for l in lessons if l.outcome == FAILURE]
    lines = [_HEADER, "", _PREAMBLE]
    if successes:
        lines.append("\n### Confirmed effective (from valid, high-reward programs)")
        for i, l in enumerate(successes, 1):
            lines.append(f"{i}. {l.render(max_chars)}")
    if failures:
        lines.append("\n### Known failure modes (from invalid or crashed programs)")
        for i, l in enumerate(failures, 1):
            lines.append(f"{i}. {l.render(max_chars)}")
    lines.append("\n" + _CLOSER)
    return "\n".join(lines)


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return max(1, len(text) // 4)


def build_injection(lessons: Sequence[Lesson], tokenizer=None,
                    token_budget: int = 0, max_chars: int = 900
                    ) -> Tuple[str, int, List[Lesson]]:
    """
    Render the block and make it fit.

    Lessons arrive ranked best-first from MemoryBank.retrieve, so trimming
    drops from the tail. If a single lesson still exceeds the budget its body
    is shortened by bisection rather than dropped, because an empty block and a
    truncated one are not the same signal to the model.

    Returns (block_text, token_count, lessons_actually_included).
    """
    kept = list(lessons)
    if not kept:
        return "", 0, []

    while kept:
        text = render_memory_block(kept, max_chars)
        n = count_tokens(tokenizer, text)
        if token_budget <= 0 or n <= token_budget:
            return text, n, kept
        if len(kept) == 1:
            break
        kept.pop()

    lo, hi = 80, max_chars
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(tokenizer, render_memory_block(kept, mid)) <= token_budget:
            lo = mid
        else:
            hi = mid - 1
    text = render_memory_block(kept, lo)
    return text, count_tokens(tokenizer, text), kept


def inject_block(messages: List[Dict], block: str,
                 mode: str = "append") -> List[Dict]:
    """
    Return a NEW message list with the block added. The input is not mutated,
    because problems build it fresh per group and a shared reference would
    accumulate blocks across steps.

    An empty block returns a copy of the input unchanged, which is what makes
    step 0 byte-identical to a no-memory run.
    """
    out = [dict(m) for m in messages]
    if not block:
        return out

    if mode == "system":
        return [{"role": "system", "content": block}] + out

    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = out[i]["content"] + "\n\n" + block
            return out

    out.append({"role": "user", "content": block})
    return out
