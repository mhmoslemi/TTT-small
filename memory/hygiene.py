"""
Extraction hygiene: the Table 3 guard.

The measured failure was not that lessons were too long or too many. It was
that the single most-injected lesson in each run was an instantiated
construction, and copying it fixed the global solution structure. MEM-C's
979e6dc0 was a coordinate formula, appeared in 19.3% of all injections, and its
literal expression showed up in 99.0% of programs after step 5 against 0.0% in
the control. The prose lesson stating the same hexagonal idea, 3a29df87, was
never injected and caused no lock-in.

So the rule this module enforces is not "be general", which the prompt already
asked for and did not get. It is mechanical:

  a lesson may describe an OPERATION, never hand over a CONSTRUCTION

and it is checked after generation rather than trusted. Three tests:

  1. global-scope lessons carry no code at all. A global rule expressed as code
     is the solution.
  2. no code block longer than max_code_lines. A four-line repair composes; a
     twelve-line block is a program.
  3. no coordinate/layout construction in any code, at any scope. Assignments
     to position variables, grid-index arithmetic, and the lattice constants
     are rejected by pattern.

Every rejection is counted and reported, so "the prompt was fixed" is a number
per step rather than an assumption.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from memory.types import GLOBAL, LOCAL

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)

# Assignments that place things in the solution space, rather than repairing
# something already placed.
_CONSTRUCTION_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(x|y|cx|cy|px|py)\s*=\s*[-+0-9(]", "assigns a coordinate"),
    (r"\bcenters?\s*\[[^\]]*\]\s*=", "writes into a centers array"),
    (r"\b(rows|cols|ncols|nrows)\s*=\s*int\s*\(", "computes grid indices"),
    (r"np\.ceil\s*\(\s*np\.sqrt", "grid-size construction"),
    (r"np\.sqrt\s*\(\s*3\s*\)", "hexagonal lattice constant"),
    (r"\brow\s*%\s*2\b|\bcol\s*%\s*2\b", "row/column parity offset"),
    (r"\b(linspace|meshgrid|mgrid)\s*\(", "builds a coordinate lattice"),
    (r"for\s+(row|col|i|j)\s+in\s+range\([^)]*\)\s*:\s*(\n|.)*?(x|y|centers)\s*=",
     "loop that lays out positions"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), why) for p, why in _CONSTRUCTION_PATTERNS]


def code_blocks(text: str) -> List[str]:
    """Fenced blocks, plus inline stretches that are obviously code."""
    blocks = [m.group(1) for m in _FENCE_RE.finditer(text or "")]
    if not blocks:
        # The extractor is told not to fence, so also treat lines that look like
        # statements as code. Otherwise the ban is trivially bypassed by
        # dropping the backticks, which is exactly what 1644ca11 did.
        lines = [l for l in (text or "").splitlines()
                 if re.search(r"[=\[]\s*\S", l) and re.search(r"\w\s*[=(\[]", l)]
        if len(lines) >= 2:
            blocks = ["\n".join(lines)]
    return blocks


def code_line_count(text: str) -> int:
    total = 0
    for b in code_blocks(text):
        total += len([l for l in b.splitlines() if l.strip()])
    return total


def violation(lesson_text: str, scope: str, cfg) -> Optional[str]:
    """
    Returns a reason string when the lesson must be rejected, else None.
    """
    if not getattr(cfg, "forbid_constructions", True):
        return None

    text = lesson_text or ""
    blocks = code_blocks(text)
    n_lines = code_line_count(text)

    if scope == GLOBAL and blocks and not getattr(cfg, "global_scope_allows_code", False):
        return ("global-scope lesson contains code; a global rule written as "
                "code is the solution itself")

    max_lines = int(getattr(cfg, "max_code_lines", 4) or 0)
    if max_lines > 0 and n_lines > max_lines:
        return f"code block is {n_lines} lines, limit is {max_lines}"

    joined = "\n".join(blocks) if blocks else ""
    for rx, why in _COMPILED:
        if rx.search(joined):
            return f"code {why}"
    return None


class HygieneStats:
    def __init__(self):
        self.kept = 0
        self.rejected = 0
        self.reasons = {}

    def keep(self):
        self.kept += 1

    def reject(self, reason: str):
        self.rejected += 1
        key = reason.split(";")[0][:60]
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def line(self) -> str:
        if not self.rejected:
            return f"hygiene: {self.kept} kept, 0 rejected"
        top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:2]
        detail = "; ".join(f"{k} x{v}" for k, v in top)
        return (f"hygiene: {self.kept} kept, {self.rejected} rejected as "
                f"constructions ({detail})")
