"""
Data carried between components.

Nothing here imports torch or any example, so the search, memory and advantage
math can be exercised without a model loaded.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Target kinds (Sec. 2.1, end): a selected leaf deepens the tree with k
# children, a selected virtual action widens its parent with exactly 1.
LEAF_EXPAND = "leaf"
VIRTUAL_EXPAND = "virtual"

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Node:
    """One evaluated candidate s. Nodes live in the tree and in the archive D."""
    id: str = field(default_factory=new_id)
    parent_id: Optional[str] = None
    step: int = 0                     # step at which it was created
    depth: int = 0

    # Verifier output. reward is Q(s) = R_d(s); invalid candidates get 0.
    reward: float = 0.0
    raw_score: Optional[float] = None   # the true metric, for display
    valid: bool = False
    feedback: str = ""                  # F_d(s)
    msg: str = ""

    # Provenance
    code: str = ""
    response: str = ""
    is_root: bool = False

    # Dynamic statistics, recomputed every step by SearchTree.recompute().
    subtree_size: int = 1               # m_s = |T(s)|
    subtree_max: float = 0.0            # W_m(s) = max_{y in T(s)} Q(y)

    @property
    def failed(self) -> bool:
        """d_i in Sec. 2.3: zero reward or the attempt failed outright."""
        return (not self.valid) or self.reward <= 0.0

    def display_score(self) -> float:
        return self.raw_score if self.raw_score is not None else self.reward


@dataclass
class Target:
    """A selected generation site: expand this node, produce num_children."""
    kind: str                          # LEAF_EXPAND | VIRTUAL_EXPAND
    node_id: str
    num_children: int
    score: float = 0.0
    # Eq. 6 breakdown, kept for logging.
    value: float = 0.0                 # V(p, a)
    prior: float = 0.0                 # pi_D(a | p)
    bonus: float = 0.0                 # c * prior * sqrt(m_p) / (1 + m_p,a)
    parent_id: Optional[str] = None

    @property
    def key(self):
        return (self.kind, self.node_id)


@dataclass
class Rollout:
    """One sampled response y_i, before verification."""
    id: str = field(default_factory=new_id)
    target_key: Any = None
    parent_id: Optional[str] = None    # node the prompt was built from
    group_id: int = 0                  # g_p: rollouts sharing a prompt x_p
    prompt_messages: List[dict] = field(default_factory=list)
    prompt_text: str = ""
    response_text: str = ""
    token_ids: Optional[List[int]] = None
    prompt_token_ids: Optional[List[int]] = None


@dataclass
class VerifyResult:
    """Verifier output for one candidate: the reward r and the feedback f."""
    reward: float = 0.0
    raw_score: Optional[float] = None
    valid: bool = False
    feedback: str = ""
    msg: str = ""
    code: str = ""
    stdout: str = ""

    @property
    def failed(self) -> bool:
        return (not self.valid) or self.reward <= 0.0


@dataclass
class Lesson:
    """One entry of the memory bank M (Sec. 2.2)."""
    id: str = field(default_factory=new_id)
    title: str = ""
    summary: str = ""
    body: str = ""
    outcome: str = OUTCOME_SUCCESS     # OUTCOME_SUCCESS | OUTCOME_FAILURE
    step: int = 0
    embedding: Optional[Any] = None    # np.ndarray, set by the embedder

    def render(self) -> str:
        tag = "WHAT WORKED" if self.outcome == OUTCOME_SUCCESS else "WHAT FAILED"
        parts = [f"[{tag}] {self.title}".strip()]
        if self.summary:
            parts.append(self.summary)
        if self.body and self.body != self.summary:
            parts.append(self.body)
        return "\n".join(parts)

    def embed_text(self) -> str:
        return "\n".join(x for x in (self.title, self.summary, self.body) if x)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "summary": self.summary,
                "body": self.body, "outcome": self.outcome, "step": self.step}


@dataclass
class Verdict:
    """One pairwise debate outcome y_ij in {0, 0.5, 1}."""
    node_a: str
    node_b: str
    y: float                           # 1.0 = a preferred, 0.0 = b, 0.5 = tie
    rationale: str = ""
    raw: str = ""
