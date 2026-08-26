"""
Prompts for the four things the module asks the model to do: extract lessons,
curate the bank, choose which lessons to use, and read the chosen lessons back.

Three design borrowings, each tied to a measured failure:

1. CONTRASTIVE EXTRACTION (ReasoningBank, self-contrast). One call sees the
   successes AND the failures of the step and is asked why some worked and
   others did not. The split prompt+/prompt- design asks "what makes these good"
   at steps where the median within-group reward spread is exactly 0.000000, so
   it restates the current construction. A contrast between the batch's two
   halves exists even when the successes are indistinguishable from each other.

2. GENERALIZATION BY PROHIBITION, not by request. ReasoningBank forbids naming
   specific websites, queries, and string contents. The analogue for program
   search is to forbid coordinates, numeric constants tied to the instance size,
   and whole layouts. v1 asked for generality and stored a coordinate formula
   that reached 99% of programs. Naming what may not appear is enforceable;
   asking for generality is not.

3. CURATION, not accumulation (Dynamic Cheatsheet). See curator.py. Additive
   banks accumulate paraphrases; a periodic rewrite where anything not carried
   forward is dropped forces the model to decide what is worth keeping.

Reflection precedes summary in both extraction prompts, again from
ReasoningBank: state why before stating what.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from memory.types import (FAILURE, GLOBAL, IMPORTANCE_DEFAULT, LOCAL, SUCCESS,
                          Lesson, RolloutRecord, clamp_importance)

# ----------------------------------------------------------------------
# Rules shared by every extraction prompt
# ----------------------------------------------------------------------
GENERALIZATION_RULES = """## What a lesson may and may not contain

A lesson is an insight that transfers to the next attempt. It is NOT a record of
what these programs did.

NEVER include, at any scope:
- coordinates, positions, or any expression that computes them
- numeric constants tied to this instance size, lattice constants, or magic numbers
- a complete arrangement, layout, seed configuration, or initialization scheme
- more than {max_code} lines of code, in any form, fenced or unfenced
- the specific score any program reached

The test: if a future attempt could paste your lesson into its program and
thereby have its structure decided for it, the lesson is too specific. Rewrite it
as the idea a reader would need in order to derive that structure themselves.

Declare a "scope" for each lesson:
  "local"   an operation, repair, check, or diagnostic that drops into any
            candidate without deciding its overall approach. May carry a few
            lines of code.
  "global"  anything determining the overall structure: which family of method
            to search in, which formulation to hand the optimizer, what to treat
            as the decision variables. MUST contain no code at all.

The asymmetry is measured, not stylistic. A global rule written as code stops
being a lesson and becomes the answer: every later program copies it, the search
stops exploring the space that rule fixes, and the run plateaus."""

_SCHEMA = """## Output format

Return ONLY a JSON object, no prose around it and no markdown fences:

{{
  "reflection": "<2-4 sentences: why did the successes succeed and the failures
                 fail? Write this FIRST and reason in it, then derive the items
                 below from it.>",
  "reinforce": ["<id from the index that this batch re-confirmed>", ...],
  "lessons": [
    {{
      "title":      "<human-readable, plain English, under 10 words>",
      "summary":    "<one plain sentence a human can read>",
      "scope":      "local" | "global",
      "kind":       "operation" | "heuristic" | "diagnostic" | "pitfall",
      "lesson":     "<the content, written for a language model to reuse>",
      "importance": <integer 1-5>,
      "closest_existing": "<id from the index this most resembles, or none>",
      "new_because": "<one clause: what this adds that the closest one lacks>"
    }}
  ]
}}

At most {n} lessons IN TOTAL. Returning fewer, or none, is correct when this
batch produced nothing that generalizes. An empty list is a valid answer and is
better than a restatement.

- "title"/"summary" are read by a person auditing the run: plain English.
- "lesson" is not read by a person. It is fed back to a language model later.
  Compress it, but keep it self-contained.
- "importance": 5 = held across nearly every attempt here and tied to the
  outcome difference. 1 = one attempt, plausible, unconfirmed.
- "closest_existing" and "new_because" are mandatory. Look up the nearest entry
  in the index before writing. If you cannot say what yours adds, it is not new:
  put that id in "reinforce" and drop it from "lessons"."""


def _index_section(catalog: Sequence[str]) -> str:
    if not catalog:
        return ("## Lessons already recorded\n"
                "(none yet, this is the first extraction)\n")
    return ("## Lessons already recorded\n"
            "id  [scope/outcome, importance, step, times used]  title :: summary\n\n"
            + "\n".join(catalog) + "\n")


def _delta_note(records: Sequence[RolloutRecord]) -> str:
    deltas = [r.delta() for r in records if r.delta() is not None]
    if not deltas:
        return ""
    best, worst = max(deltas), min(deltas)
    if abs(best) < 1e-12 and abs(worst) < 1e-12:
        return ("\n**Nothing here improved on its parent.** Every attempt scored "
                "exactly what it started from. Do not describe what these "
                "programs do, because that is already recorded and is not "
                "working. Either name the specific thing blocking progress, at "
                "`global` scope and without code, or return no lessons at all.\n")
    return (f"\nEach attempt's change relative to its parent is shown, ranging "
            f"from {worst:+.6f} to {best:+.6f}. What matters is what CHANGED.\n")


# ----------------------------------------------------------------------
# Contrastive extraction (default): successes and failures in one call
# ----------------------------------------------------------------------
def build_contrast_messages(meta_description: str,
                            successes: Sequence[RolloutRecord],
                            failures: Sequence[RolloutRecord],
                            num_lessons: int,
                            max_chars_per_example: int,
                            feedback_chars: int,
                            catalog: Sequence[str] = (),
                            max_code: int = 4) -> List[Dict]:
    """
    One call over both halves of the batch.

    The instruction to compare rather than to summarize is the whole point: on a
    plateau the accepted programs are indistinguishable from each other, but they
    are still distinguishable from the rejected ones.
    """
    blocks = []
    for i, r in enumerate(successes, 1):
        blocks.append(f"### ACCEPTED attempt {i}\n" + r.render_success(max_chars_per_example))
    for i, r in enumerate(failures, 1):
        blocks.append(f"### REJECTED attempt {i}\n"
                      + r.render_failure(max_chars_per_example, feedback_chars))

    user = (
        "You maintain the lesson memory for an automated program-search run. "
        "Below is one batch of attempts at a single task. Some were accepted by "
        "the verifier and some were rejected.\n\n"
        f"## Task\n{meta_description}\n\n"
        + _index_section(catalog) + "\n"
        f"## This batch ({len(successes)} accepted, {len(failures)} rejected)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        "Compare and contrast these attempts. Think first: why did some succeed "
        "while others failed, and what separates the higher-scoring accepted ones "
        "from the lower-scoring ones?\n"
        "- Identify strategies that consistently led to a better outcome.\n"
        "- Identify mistakes or inefficiencies in the rejected attempts, and "
        "state the preventative measure for each.\n"
        "- Prefer insights that hold beyond this particular batch.\n"
        + _delta_note(list(successes) + list(failures))
        + "\n" + GENERALIZATION_RULES.format(max_code=max_code) + "\n\n"
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


# ----------------------------------------------------------------------
# Split extraction (prompt+ / prompt-), kept for the ablation
# ----------------------------------------------------------------------
def build_positive_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int, max_chars_per_example: int,
                            catalog: Sequence[str] = (),
                            require_full: bool = False,
                            max_code: int = 4) -> List[Dict]:
    blocks = [f"### Attempt {i}\n" + r.render_success(max_chars_per_example)
              for i, r in enumerate(records, 1)]
    user = (
        "You maintain the lesson memory for an automated program-search run. "
        "Below is a batch of attempts the verifier ACCEPTED.\n\n"
        f"## Task\n{meta_description}\n\n"
        + _index_section(catalog) + "\n"
        f"## This batch ({len(records)} accepted attempts)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        "Think first about WHY these succeeded, then derive what separates the "
        "higher-scoring ones from the lower-scoring ones and what each changed "
        "relative to its parent."
        + _delta_note(records)
        + "\n" + GENERALIZATION_RULES.format(max_code=max_code) + "\n\n"
        + ("Return exactly {n} lessons.\n\n".format(n=num_lessons) if require_full else "")
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


def build_negative_messages(meta_description: str,
                            records: Sequence[RolloutRecord],
                            num_lessons: int, max_chars_per_example: int,
                            feedback_chars: int, catalog: Sequence[str] = (),
                            require_full: bool = False,
                            max_code: int = 4) -> List[Dict]:
    blocks = [f"### Attempt {i}\n"
              + r.render_failure(max_chars_per_example, feedback_chars)
              for i, r in enumerate(records, 1)]
    user = (
        "You maintain the lesson memory for an automated program-search run. "
        "Below is a batch of attempts the verifier REJECTED.\n\n"
        f"## Task\n{meta_description}\n\n"
        + _index_section(catalog) + "\n"
        f"## This batch ({len(records)} rejected attempts)\n\n"
        + "\n\n".join(blocks)
        + "\n\n## What to produce\n"
        "Reflect first on WHY these failed, then state the lessons or "
        "preventative strategies. Identify the failure modes that RECUR here, "
        "not the one-off accidents.\n\n"
        "A caution specific to this channel: 'catch the error and return a safe "
        "default' is almost never the right lesson. A program that detects its "
        "own failure and returns a degenerate answer has not solved anything, and "
        "a memory full of defensive coding produces valid, worthless output. "
        "State what would make the computation correct, not how to survive it "
        "being wrong.\n"
        + "\n" + GENERALIZATION_RULES.format(max_code=max_code) + "\n\n"
        + ("Return exactly {n} lessons.\n\n".format(n=num_lessons) if require_full else "")
        + _SCHEMA.format(n=num_lessons)
    )
    return [{"role": "user", "content": user}]


# ----------------------------------------------------------------------
# Parsing extraction
# ----------------------------------------------------------------------
@dataclass
class ExtractionResult:
    lessons: List[Lesson] = field(default_factory=list)
    reinforce: List[str] = field(default_factory=list)
    reflection: str = ""
    raw: str = ""
    rejected: List[Tuple[str, str]] = field(default_factory=list)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ID_RE = re.compile(r"^[0-9a-f]{6,16}$")


def _candidate_spans(text: str) -> List[str]:
    out = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    for op, cl in (("{", "}"), ("[", "]")):
        a, b = text.find(op), text.rfind(cl)
        if a != -1 and b > a:
            out.append(text[a:b + 1])
    out.append(text.strip())
    return out


def _clean_ids(raw) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    out = []
    if isinstance(raw, list):
        for r in raw:
            rid = str(r).strip().strip("[]").lower()
            if _ID_RE.match(rid):
                out.append(rid)
    return out


def parse_extraction(response_text: str, outcome: str, step: int,
                     expected: int) -> ExtractionResult:
    """
    Never raises: a malformed reply costs one step's lessons, not the run.

    `outcome` is the default label. In contrastive mode a lesson may declare
    "kind": "pitfall", which routes it to the failure side, since that is what
    the outcome tag is for downstream.
    """
    text = _THINK_RE.sub("", response_text or "").strip()
    if not text:
        return ExtractionResult(raw=response_text or "")

    payload = None
    for span in _candidate_spans(text):
        try:
            payload = json.loads(span)
            break
        except Exception:
            continue

    items, reinforce, reflection = None, [], ""
    if isinstance(payload, dict):
        items = payload.get("lessons", payload.get("items"))
        reinforce = _clean_ids(payload.get("reinforce", payload.get("confirm", [])))
        reflection = str(payload.get("reflection", ""))[:1200]
    elif isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        items = []

    lessons = []
    cap = max(int(expected), 0) or None
    for item in items[:cap]:
        if isinstance(item, str):
            body = item.strip()
            if body:
                lessons.append(Lesson.create(body[:80], body[:200], body, outcome, step))
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        body = str(item.get("lesson", item.get("content", item.get("detail", "")))).strip()
        if not body and not summary:
            continue
        scope = str(item.get("scope", LOCAL)).strip().lower()
        if scope not in (LOCAL, GLOBAL):
            scope = LOCAL
        kind = str(item.get("kind", "")).strip().lower()
        tag = FAILURE if kind == "pitfall" else (SUCCESS if kind else outcome)
        lessons.append(Lesson.create(
            title=title or summary[:80], summary=summary or body[:200],
            lesson=body or summary, outcome=tag, step=step, scope=scope,
            importance=clamp_importance(item.get("importance", IMPORTANCE_DEFAULT))))
        nb = str(item.get("new_because", "")).strip().lower()
        ce = _clean_ids(item.get("closest_existing"))
        if ce and (not nb or nb in ("none", "n/a", "nothing", "-")):
            reinforce.extend(ce)
            lessons.pop()

    return ExtractionResult(lessons=lessons,
                            reinforce=list(dict.fromkeys(reinforce)),
                            reflection=reflection, raw=response_text or "")


# ----------------------------------------------------------------------
# Lookup: the model reads the index and names what it wants
# ----------------------------------------------------------------------
_LOOKUP_SCHEMA = """Reply with ONLY a JSON object, no prose and no fences:

{{"ids": ["<id>", ...], "why": "<one short clause per id, same order>"}}

At most {k} ids. An EMPTY list is the right answer when nothing recorded bears on
this state, and is preferred over filling the list. Use only ids from the index."""


def build_lookup_messages(meta_description: str, parent_block: str,
                          catalog: Sequence[str], max_select: int) -> List[Dict]:
    user = (
        "You are about to modify a candidate program for the task below. First "
        "decide which of your recorded lessons are worth having in front of "
        "you.\n\n"
        f"## Task\n{meta_description}\n\n"
        f"## Current candidate\n{parent_block}\n\n"
        "## Lesson index (the complete memory)\n"
        "id  [scope/outcome, importance, step, times used, causal tail evidence]"
        "  title :: summary\n\n"
        + "\n".join(catalog)
        + "\n\n## What to choose\n"
        "Pick lessons that bear on THIS candidate and what you would change about "
        "it. Prefer one that would change what you write over one that describes "
        "what the candidate already does. Ignore anything restating the task. A "
        "lesson many steps have used is not automatically right for this state; if "
        "the run appears stuck, the frequently-used entries are the likeliest "
        "cause and the least useful choice. Prefer positive matched tail-uplift "
        "evidence when it is relevant, but do not confuse an under-tested lesson "
        "with a bad one; the exploration arm handles uncertainty separately.\n\n"
        + _LOOKUP_SCHEMA.format(k=max_select)
    )
    return [{"role": "user", "content": user}]


@dataclass
class LookupResult:
    ids: List[str] = field(default_factory=list)
    why: str = ""
    raw: str = ""


def parse_lookup(response_text: str, valid_ids: Sequence[str],
                 max_select: int) -> LookupResult:
    text = _THINK_RE.sub("", response_text or "").strip()
    if not text:
        return LookupResult(raw=response_text or "")
    allowed = set(valid_ids)
    ids, why, payload = [], "", None
    for span in _candidate_spans(text):
        try:
            payload = json.loads(span)
            break
        except Exception:
            continue
    if isinstance(payload, dict):
        ids = _clean_ids(payload.get("ids", payload.get("lessons", [])))
        why = str(payload.get("why", ""))[:400]
    elif isinstance(payload, list):
        ids = _clean_ids(payload)
    if not ids:
        ids = [t for t in re.findall(r"\b[0-9a-f]{8}\b", text.lower()) if t in allowed]
    seen, out = set(), []
    for i in ids:
        if i in allowed and i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= max_select:
            break
    return LookupResult(ids=out, why=why, raw=response_text or "")


def parent_block(meta_reward: str, parent_code: str, limit: int = 3000) -> str:
    body = [meta_reward]
    if parent_code and parent_code.strip():
        body.append("```python\n" + parent_code[:limit] + "\n```")
    else:
        body.append("(no program yet; this is a seed state)")
    return "\n".join(body)


# ----------------------------------------------------------------------
# Rendering the chosen lessons for the generation prompt
# ----------------------------------------------------------------------
def render_memory_block(lessons: Sequence[Lesson], max_chars: int = 900) -> str:
    """
    The block handed to problem.build_prompt. The problem decides where it goes
    and writes its own framing around it, so this is the content only.
    """
    if not lessons:
        return ""
    lines = []
    for tag, group in (("Observed to help", SUCCESS), ("Observed to fail", FAILURE)):
        chosen = [l for l in lessons if l.outcome == group]
        if not chosen:
            continue
        lines.append(f"**{tag}**")
        for i, l in enumerate(chosen, 1):
            lines.append(f"{i}. [{l.scope}] {l.render(max_chars)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)
    try:
        from model_io import text_tokenizer
        decoder = text_tokenizer(tokenizer)
        return len(decoder(text, add_special_tokens=False).input_ids)
    except Exception:
        return max(1, len(text) // 4)


def build_injection(lessons: Sequence[Lesson], tokenizer=None,
                    token_budget: int = 0, max_chars: int = 900
                    ) -> Tuple[str, int, List[Lesson]]:
    """
    Render and fit. Lessons arrive in the order the model asked for them, so
    trimming drops from the tail: its last choice is the one it wanted least.
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
    Fallback for problems that do not accept a memory argument. Preferred path is
    problem.build_prompt(parent, memory=block), which places the block where the
    problem wants it rather than after the instruction.
    """
    out = [dict(m) for m in messages]
    if not block:
        return out
    if mode == "system":
        return [{"role": "system", "content": block}] + out
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            content = out[i].get("content", "")
            if isinstance(content, list):
                out[i]["content"] = list(content) + [
                    {"type": "text", "text": block}
                ]
            else:
                out[i]["content"] = str(content) + "\n\n" + block
            return out
    out.append({"role": "user", "content": block})
    return out
