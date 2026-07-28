"""
Extraction prompts for the memory maker (Sec. 2.2).

Two prompts, one per group. Both insist on lessons that generalize across the
whole group rather than notes about individual responses -- that is what lets
the model separate a systematic pattern from an accident of one trajectory.

Output is requested as a JSON array so it can be parsed; memory/extractor.py
falls back to a lenient parser when the model does not comply.
"""

SYSTEM = (
    "You are analysing a batch of attempts at a hard search problem. "
    "You write short, concrete, reusable lessons for whoever attempts it next."
)

_FORMAT = """
Return ONLY a JSON array with exactly {n} objects, no prose around it:

[
  {{"title": "<6 words or fewer>",
    "summary": "<one sentence, the actionable rule>",
    "lesson": "<2-4 sentences: the pattern, the evidence for it, and what to do>"}}
]
"""

POSITIVE = """The following {count} attempts at this problem SUCCEEDED, with the reward each achieved.

Problem
-------
{problem}

Successful attempts
-------------------
{examples}

Identify the {n} most useful strategies these successes SHARE. Consider the
approach taken, the structure of the solution, and any technique that repeatedly
paid off. Rank them by how much they contributed to the reward.

Rules:
- Lessons must describe patterns ACROSS the group, not one attempt. Never write
  "attempt 3 did X"; write the rule that attempt 3 is evidence for.
- Be specific and technical. "Be careful" is worthless; name the technique.
- Skip anything that is already obvious from the problem statement.
""" + _FORMAT

NEGATIVE = """The following {count} attempts at this problem FAILED, each with the error or
rejection reason the verifier reported.

Problem
-------
{problem}

Failed attempts
---------------
{examples}

Identify the {n} most common FAILURE MODES, and for each the concrete
preventative measure that would have avoided it.

Rules:
- Group failures by root cause; the same underlying mistake reported two ways is
  one lesson, not two.
- Lessons must describe patterns ACROSS the group, not one attempt.
- State the fix in a way the next attempt can act on before it writes any code.
- Prefer the failures that recur; ignore one-off typos.
""" + _FORMAT


def positive_prompt(problem: str, examples: str, count: int, n: int) -> list:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": POSITIVE.format(
            problem=problem, examples=examples, count=count, n=n)},
    ]


def negative_prompt(problem: str, examples: str, count: int, n: int) -> list:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": NEGATIVE.format(
            problem=problem, examples=examples, count=count, n=n)},
    ]
