"""Prompt construction + verdict parsing for the Multi-Agent Elo re-ranker.

Comparison prompts follow the AI co-scientist Ranking-agent format (single-turn
comparison and simulated multi-turn scientific debate), adapted to discovery:
the two "hypotheses" being compared are two candidate *programs/strategies* from
the reuse buffer. The judge is asked which program is the more promising
direction to EXPAND NEXT, not which already has the higher measured reward
(the PUCT Q-term already accounts for the measured reward).

Personas/criteria are problem-aware: circle_packing uses a computational-geometry
rubric; everything else falls back to a generic discovery rubric. The re-ranker
passes metric_name to pick the right preset, so the same code path serves every
problem in the registry.
"""
from __future__ import annotations
import re
from typing import List, Optional

# Matches "better idea: 1", "better hypothesis: <2>", etc. (case-insensitive).
_VERDICT_RE = re.compile(
    r"better\s+(?:idea|hypothesis)\s*:?\s*<?\s*([12])",
    re.IGNORECASE,
)


def default_goal(metric_name: str, maximize: bool, target=None) -> str:
    direction = "maximize" if maximize else "minimize"
    tgt = f" (target {target})" if target is not None else ""
    return (f"Discover a program whose output {direction}s the metric "
            f"'{metric_name}'{tgt}. We search by repeatedly expanding promising "
            f"programs sampled from a buffer.")


# ----------------------------------------------------------------------
# Generic discovery rubric (fallback for problems without a custom preset)
# ----------------------------------------------------------------------
DEFAULT_CRITERIA = (
    "- Likelihood the underlying approach can be pushed to a stronger solution\n"
    "- Novelty / originality of the search strategy (not just parameter tweaks)\n"
    "- Exploitation of problem structure\n"
    "- Robustness, i.e. the chance it keeps producing valid solutions\n"
    "- Headroom: how much further this direction can plausibly improve"
)

_GENERIC_SYSTEM_DEBATE = (
    "You are an expert in comparative analysis, simulating a panel of "
    "domain experts in a structured discussion to evaluate two competing "
    "candidate programs for a scientific discovery search. The objective "
    "is to rigorously determine which program is the more promising "
    "direction to expand next, based on the criteria below. The experts "
    "have no prior bias toward either program and focus only on the "
    "optimal choice, given that only one can be expanded."
)

_GENERIC_SYSTEM_SINGLE = (
    "You are an expert evaluator comparing two candidate programs for a "
    "scientific discovery search, determining which is the more promising "
    "direction to expand next."
)


# ----------------------------------------------------------------------
# Circle-packing preset (computational-geometry rubric)
# ----------------------------------------------------------------------
_CIRCLE_SYSTEM = (
    "You are an expert computational geometer and numerical optimization "
    "specialist evaluating two algorithmic strategies for the Circle Packing "
    "problem. The objective is to pack N circles within a [0,1]x[0,1] unit "
    "square to maximize the sum of their radii without any overlap.\n\n"
    "CRITICAL DIRECTIVE: DO NOT evaluate these scripts based on code "
    "readability, elegance, comments, or software engineering best practices. "
    "A messy, hardcoded, or unconventional script that uses superior geometric "
    "heuristics to achieve a tighter packing is strictly better than a clean, "
    "standard script that gets trapped in a local minimum."
)

_CIRCLE_CRITERIA = (
    "* Initialization Strategy: Does the algorithm seed the centers "
    "intelligently (e.g., hexagonal lattices, staggered grids, boundary-hugging) "
    "rather than relying on pure uniform random placement?\n"
    "* Optimization Engine: Does it utilize a robust method for non-convex "
    "spaces (e.g., SLSQP with strict bounds, simulated annealing, basin-hopping, "
    "or physics-based repulsion/expansion)?\n"
    "* Local Optima Escape: Does the algorithm include perturbation mechanics "
    "(jiggling, swapping, targeted radius inflation) to break out of jammed "
    "configurations?\n"
    "* Constraint Handling: How mathematically sound is the enforcement of the "
    "`x, y +/- r <= 1` boundaries and the `dist >= r_i + r_j` overlap rules?"
)


def _select_preset(metric_name: str):
    """Return (system_prompt, criteria, intro_line, slot_label) for the problem.

    Keyed off metric_name (cheap, already known to the re-ranker). Add new
    presets here as you tune per-problem rubrics. slot_label controls how the
    two candidates are introduced ('Approach' vs 'Program') to match the rubric.
    """
    m = (metric_name or "").strip().lower()
    if m in ("sum of radii", "sum_of_radii"):
        return (
            _CIRCLE_SYSTEM,
            _CIRCLE_CRITERIA,
            ("Evaluate the structural potential of the two approaches based "
             "purely on these mathematical criteria:"),
            "Approach",
        )
    # default / fallback
    return (
        None,  # signals: use generic system prompts chosen by debate flag
        DEFAULT_CRITERIA,
        ("Criteria for which program is the better direction to expand:"),
        "Program",
    )


def _truncate(code: str, max_chars: int) -> str:
    code = code or ""
    if max_chars and len(code) > max_chars:
        return code[:max_chars] + "\n# ... [truncated for comparison] ...\n"
    return code


def build_comparison_messages(
    goal: str,
    code1: str,
    code2: str,
    criteria: str = None,
    debate: bool = True,
    max_code_chars: int = 4000,
    metric_name: str = "",
) -> List[dict]:
    """Build chat messages for one pairwise match.

    debate=True  -> simulated multi-turn expert panel (ONE llm call that emits
                    the whole transcript and ends with the verdict).
    debate=False -> single concise comparison.
    Both must end with: better idea: <1 or 2>.

    A problem-specific preset (persona + rubric) is chosen from metric_name. If
    `criteria` is passed explicitly it overrides the preset's criteria; otherwise
    the preset's criteria are used.
    """
    c1 = _truncate(code1, max_code_chars)
    c2 = _truncate(code2, max_code_chars)

    preset_system, preset_criteria, crit_intro, slot = _select_preset(metric_name)
    crit = criteria if criteria is not None else preset_criteria

    if debate:
        system = preset_system or _GENERIC_SYSTEM_DEBATE
        user = f"""Goal: {goal}

{crit_intro}
{crit}

{slot} 1:
```python
{c1}
```

{slot} 2:
```python
{c2}
```

Debate procedure:
The discussion unfolds over a few turns (typically 3 to 5).
Turn 1: concisely summarize the core strategy of each candidate.
Subsequent turns:
  * Critically evaluate each candidate against the Goal and Criteria.
  * Identify weaknesses, limitations, or likely failure modes of each.
  * Judge which candidate has more headroom to reach a stronger solution.

Termination and judgment:
Once the discussion reaches sufficient depth, give a conclusive judgment that
succinctly states the rationale, then indicate the superior candidate by writing
the phrase "better idea: " followed by "1" or "2".
"""
    else:
        system = preset_system or _GENERIC_SYSTEM_SINGLE
        user = f"""Goal: {goal}

{crit_intro}
{crit}

Judge only the merit and headroom of each approach. Do not assume that the
already-measured score decides the comparison.

{slot} 1:
```python
{c1}
```

{slot} 2:
```python
{c2}
```

Briefly reason about the mathematical and optimization superiority of one
{slot.lower()} over the other, then conclude with exactly the phrase:
"better idea: <1 or 2>".
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_verdict(text: str) -> Optional[int]:
    """Return 1, 2, or None. Uses the LAST occurrence (the verdict is emitted
    at the end of the response)."""
    if not text:
        return None
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])



