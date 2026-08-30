"""
The search tree, and the two statistics D-PUCT is built on.

A node carries a scalar `value` (higher is better) and an arbitrary `payload`
that the library never inspects -- put your board state, your program, your
molecule there. Everything else is derived.

Per selection round the tree recomputes, bottom-up:

    m_s    = |T(s)|                    subtree size
    W_m(s) = max_{y in T(s)} Q(y)      subtree MAX

The max is the whole point. Classic MCTS averages a subtree's returns, which is
right when you will have to play the average outcome. It is wrong when you get
to keep the single best thing you find, because averaging buries an exceptional
descendant under its mediocre siblings. W_m keeps it visible at every ancestor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4


def _new_id() -> str:
    return uuid4().hex[:12]


@dataclass
class Node:
    """One candidate in the tree."""

    id: str = field(default_factory=_new_id)
    parent_id: Optional[str] = None
    depth: int = 0
    value: float = 0.0          # Q(s), higher is better
    payload: Any = None         # yours; the library never looks inside
    is_root: bool = False

    # Recomputed by Tree.recompute(); do not set by hand.
    subtree_size: int = 1       # m_s
    subtree_max: float = 0.0    # W_m(s)


class Tree:
    """
    A search tree that doubles as a flat archive.

    The tree keeps local trajectories; the archive (`nodes()`) is what
    dataset-level statistics such as the rank signal are computed over.
    """

    def __init__(self, max_size: int = 0):
        """max_size caps the archive; 0 means unbounded."""
        self.max_size = int(max_size or 0)
        self._nodes: Dict[str, Node] = {}
        self._children: Dict[str, List[str]] = {}
        self._roots: List[str] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add_root(self, value: float = 0.0, payload: Any = None) -> Node:
        node = Node(value=float(value), payload=payload, is_root=True, depth=0)
        self._nodes[node.id] = node
        self._children[node.id] = []
        self._roots.append(node.id)
        return node

    def add_child(self, parent_id: str, value: float = 0.0,
                  payload: Any = None) -> Node:
        parent = self._nodes[parent_id]
        node = Node(parent_id=parent_id, depth=parent.depth + 1,
                    value=float(value), payload=payload)
        self._nodes[node.id] = node
        self._children[node.id] = []
        self._children[parent_id].append(node.id)
        return node

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self) -> List[Node]:
        """The flat archive, roots included."""
        return list(self._nodes.values())

    def roots(self) -> List[Node]:
        return [self._nodes[i] for i in self._roots]

    def children(self, node_id: str) -> List[Node]:
        return [self._nodes[c] for c in self._children.get(node_id, ())]

    def child_ids(self, node_id: str) -> List[str]:
        return list(self._children.get(node_id, ()))

    def is_leaf(self, node_id: str) -> bool:
        return not self._children.get(node_id)

    def parent_of(self, node: Node) -> Optional[Node]:
        return self._nodes.get(node.parent_id) if node.parent_id else None

    def ancestors(self, node_id: str) -> List[str]:
        out, cur = [], self._nodes[node_id].parent_id
        while cur:
            out.append(cur)
            cur = self._nodes[cur].parent_id
        return out

    def expanded(self) -> List[Node]:
        """
        Non-root nodes -- the ones that were actually generated.

        Roots are the initial state, not candidates the search produced. They
        are excluded from archive statistics because a root's W_m equals the
        global best, which would distort every rank.
        """
        return [n for n in self._nodes.values() if not n.is_root]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def recompute(self) -> None:
        """Refresh m_s and W_m(s) everywhere. Deepest first, so one pass does it."""
        for node in sorted(self._nodes.values(), key=lambda n: n.depth,
                           reverse=True):
            size = 1
            best = node.value
            for cid in self._children.get(node.id, ()):
                child = self._nodes[cid]
                size += child.subtree_size
                if child.subtree_max > best:
                    best = child.subtree_max
            node.subtree_size = size
            node.subtree_max = best

    def best(self) -> Optional[Node]:
        """argmax over generated nodes -- the answer the search returns."""
        candidates = self.expanded()
        return max(candidates, key=lambda n: n.value) if candidates else None

    def top_k(self, k: int, key: str = "subtree_max") -> List[Node]:
        candidates = self.expanded()
        candidates.sort(key=lambda n: getattr(n, key), reverse=True)
        return candidates[:k]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def prune(self) -> int:
        """
        Drop the weakest nodes when the archive exceeds max_size.

        Ancestors of a survivor are always kept, or the survivor would be
        detached from the tree; roots are never dropped. The cap is therefore
        soft, and can be exceeded once ancestors are added back.
        """
        cap = self.max_size
        if cap <= 0 or len(self._nodes) <= cap:
            return 0

        ranked = sorted(self.expanded(),
                        key=lambda n: (n.value, n.subtree_max), reverse=True)
        keep = set(self._roots)
        for node in ranked:
            if len(keep) >= cap:
                break
            keep.add(node.id)
            keep.update(self.ancestors(node.id))

        dropped = [nid for nid in self._nodes if nid not in keep]
        for nid in dropped:
            parent_id = self._nodes[nid].parent_id
            if parent_id in self._children:
                self._children[parent_id] = [
                    c for c in self._children[parent_id] if c != nid]
            self._nodes.pop(nid, None)
            self._children.pop(nid, None)
        if dropped:
            self.recompute()
        return len(dropped)

    def subtree(self, node_id: str) -> Iterator[Node]:
        """T(s), including s."""
        stack = [node_id]
        while stack:
            cur = stack.pop()
            yield self._nodes[cur]
            stack.extend(self._children.get(cur, ()))

    # ------------------------------------------------------------------
    # Drawing (lazy imports: viz is optional and matplotlib doubly so)
    # ------------------------------------------------------------------
    def render(self, **kwargs) -> str:
        """Text drawing of the tree. See dpuct.viz.render_text for options."""
        from .viz import render_text
        return render_text(self, **kwargs)

    def show(self, **kwargs) -> None:
        """print(self.render(...))."""
        print(self.render(**kwargs))

    def draw(self, **kwargs):
        """Matplotlib drawing. See dpuct.viz.draw for options."""
        from .viz import draw
        return draw(self, **kwargs)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        best = self.best()
        return {
            "nodes": len(self._nodes),
            "roots": len(self._roots),
            "leaves": sum(1 for n in self._nodes if self.is_leaf(n)),
            "max_depth": max((n.depth for n in self._nodes.values()), default=0),
            "best_value": best.value if best else None,
        }
