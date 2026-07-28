"""
The backbone acting as pairwise judge for the Elo debate (Sec. 2.1).

For a pair (s_i, s_j) the judge decides which is the more promising, correct or
useful CONTINUATION -- not which currently scores higher. The reward already
carries "which scores higher"; the debate exists to add what the reward misses,
so the prompt pushes the judge toward headroom and soundness rather than a
restatement of the numbers.

Outcome y_ij in {0, 0.5, 1}: 1 = A preferred, 0 = B preferred, 0.5 = tie.
An unparseable reply is recorded as a tie, which leaves both ratings unchanged
rather than inventing a winner.
"""

import re
from typing import List, Sequence, Tuple

from core.types import Verdict

SYSTEM = (
    "You are judging two candidate solutions to a hard search problem. "
    "You decide which is the more promising basis for further work."
)

TEMPLATE = """Problem
-------
{problem}

Candidate A
-----------
{a}

Candidate B
-----------
{b}

Which candidate is the more promising, correct, or useful CONTINUATION of the
search? Judge the approach's headroom and soundness, not only the score it has
reached: a slightly weaker candidate built on an idea with room to grow beats a
saturated one. Weigh correctness, whether the reasoning is sound, and how much
better the approach could plausibly get.

Answer in at most four sentences, then a final line exactly of the form:
VERDICT: A
VERDICT: B
VERDICT: TIE
"""

_VERDICT = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(A|B|TIE|DRAW|EQUAL)\b", re.IGNORECASE)


def parse_verdict(text: str, allow_ties: bool = True) -> Tuple[float, str]:
    """Return (y_a, rationale). Unparseable -> tie, which is a no-op update."""
    if not text:
        return 0.5, ""
    matches = _VERDICT.findall(text)
    if matches:
        token = matches[-1].upper()          # last statement wins
        if token == "A":
            return 1.0, text.strip()
        if token == "B":
            return 0.0, text.strip()
        return (0.5 if allow_ties else 0.5), text.strip()

    # No tagged verdict: accept an unambiguous closing statement.
    tail = text.strip()[-200:].lower()
    says_a = "candidate a" in tail
    says_b = "candidate b" in tail
    if says_a and not says_b:
        return 1.0, text.strip()
    if says_b and not says_a:
        return 0.0, text.strip()
    return 0.5, text.strip()


class EloJudge:
    def __init__(self, cfg, llm, problem_description: str = ""):
        """cfg is an EloConfig."""
        self.cfg = cfg
        self.llm = llm
        self.problem_description = problem_description

    def build_prompt(self, a_render: str, b_render: str) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(
                problem=self.problem_description, a=a_render, b=b_render)},
        ]

    def judge(self, pairs: Sequence[Tuple[str, str]], renders: dict
              ) -> List[Verdict]:
        """
        pairs: (node_id_a, node_id_b). renders: node_id -> text shown to the judge.
        """
        pairs = [(a, b) for a, b in pairs if a in renders and b in renders]
        if not pairs:
            return []

        prompts = [self.build_prompt(renders[a], renders[b]) for a, b in pairs]
        replies = self.llm.chat_batch(
            prompts,
            max_new_tokens=int(self.cfg.judge_max_tokens),
            temperature=float(self.cfg.judge_temperature),
            batch_size=int(self.cfg.judge_batch_size),
        )

        out = []
        for (a, b), reply in zip(pairs, replies):
            y, rationale = parse_verdict(reply, self.cfg.allow_ties)
            out.append(Verdict(node_a=a, node_b=b, y=y,
                               rationale=rationale[:2000], raw=reply[:4000]))
        return out
