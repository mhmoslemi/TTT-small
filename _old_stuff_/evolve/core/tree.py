"""
Search tree + flat archive D (paper Sec. 1).

The tree preserves local search trajectories; the archive is the global record
every dataset-level statistic is computed over. Both are the same node objects,
viewed two ways.

Per step the engine calls recompute(), which refreshes

    m_s    = |T(s)|                   subtree size
    W_m(s) = max_{y in T(s)} Q(y)     subtree max, Eq. 2

bottom-up. W_m is the statistic the whole framework is built around: it keeps an
exceptional descendant visible at every ancestor instead of averaging it away.
"""

from typing import Dict, Iterable, List, Optional

from core.types import Node


class SearchTree:
    def __init__(self, max_archive_size: int = 0, fail_reward: float = 0.0):
        self.max_archive_size = int(max_archive_size or 0)
        self.fail_reward = float(fail_reward)
        self._nodes: Dict[str, Node] = {}
        self._children: Dict[str, List[str]] = {}
        self._roots: List[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_root(self, **kwargs) -> Node:
        node = Node(parent_id=None, depth=0, is_root=True, **kwargs)
        self._nodes[node.id] = node
        self._children[node.id] = []
        self._roots.append(node.id)
        return node

    def add_child(self, parent_id: str, **kwargs) -> Node:
        parent = self._nodes[parent_id]
        node = Node(parent_id=parent_id, depth=parent.depth + 1, **kwargs)
        self._nodes[node.id] = node
        self._children[node.id] = []
        self._children[parent_id].append(node.id)
        return node

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self) -> List[Node]:
        """The flat archive D."""
        return list(self._nodes.values())

    def roots(self) -> List[Node]:
        return [self._nodes[i] for i in self._roots]

    def children(self, node_id: str) -> List[Node]:
        return [self._nodes[c] for c in self._children.get(node_id, [])]

    def child_ids(self, node_id: str) -> List[str]:
        return list(self._children.get(node_id, []))

    def is_leaf(self, node_id: str) -> bool:
        return not self._children.get(node_id)

    def parent_of(self, node: Node) -> Optional[Node]:
        return self._nodes.get(node.parent_id) if node.parent_id else None

    def evaluated(self) -> List[Node]:
        """Nodes carrying a real verifier result (roots may be empty seeds)."""
        return [n for n in self._nodes.values() if not n.is_root]

    def ancestors(self, node_id: str) -> List[str]:
        out, cur = [], self._nodes[node_id].parent_id
        while cur:
            out.append(cur)
            cur = self._nodes[cur].parent_id
        return out

    # ------------------------------------------------------------------
    # Dynamic statistics
    # ------------------------------------------------------------------
    def recompute(self) -> None:
        """Refresh m_s and W_m(s) for every node, deepest first."""
        for node in sorted(self._nodes.values(), key=lambda n: n.depth, reverse=True):
            size = 1
            best = node.reward
            for cid in self._children.get(node.id, ()):
                child = self._nodes[cid]
                size += child.subtree_size
                if child.subtree_max > best:
                    best = child.subtree_max
            node.subtree_size = size
            node.subtree_max = best

    def best(self) -> Optional[Node]:
        """argmax_{s in D} R(s), ignoring unevaluated roots."""
        candidates = [n for n in self._nodes.values() if not n.is_root]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.reward)

    def top_k(self, k: int, key: str = "subtree_max") -> List[Node]:
        candidates = [n for n in self._nodes.values() if not n.is_root]
        candidates.sort(key=lambda n: getattr(n, key), reverse=True)
        return candidates[:k]

    # ------------------------------------------------------------------
    # Archive cap
    # ------------------------------------------------------------------
    def prune(self) -> int:
        """
        Cap |D| at max_archive_size, keeping the best nodes by reward.

        Ancestors of a kept node are kept too, otherwise the survivor would be
        detached from the tree. Roots are never dropped, so the cap is soft: it
        can be exceeded once ancestors are added back.
        """
        cap = self.max_archive_size
        if cap <= 0 or len(self._nodes) <= cap:
            return 0

        ranked = sorted(
            (n for n in self._nodes.values() if not n.is_root),
            key=lambda n: (n.reward, n.subtree_max, -n.step),
            reverse=True,
        )
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
                    c for c in self._children[parent_id] if c != nid
                ]
            self._nodes.pop(nid, None)
            self._children.pop(nid, None)
        if dropped:
            self.recompute()
        return len(dropped)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "roots": list(self._roots),
            "nodes": [
                {
                    "id": n.id, "parent_id": n.parent_id, "step": n.step,
                    "depth": n.depth, "reward": n.reward, "raw_score": n.raw_score,
                    "valid": n.valid, "is_root": n.is_root, "msg": n.msg,
                    "subtree_size": n.subtree_size, "subtree_max": n.subtree_max,
                    "code": n.code, "feedback": n.feedback,
                }
                for n in self._nodes.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, max_archive_size: int = 0) -> "SearchTree":
        tree = cls(max_archive_size=max_archive_size)
        raw = list(data.get("nodes", []))
        # Parents must exist before their children.
        for d in sorted(raw, key=lambda d: d["depth"]):
            node = Node(**{k: v for k, v in d.items()
                           if k in Node.__dataclass_fields__})
            tree._nodes[node.id] = node
            tree._children.setdefault(node.id, [])
            if node.parent_id:
                tree._children.setdefault(node.parent_id, []).append(node.id)
            else:
                tree._roots.append(node.id)
        tree.recompute()
        return tree


def iter_subtree(tree: SearchTree, node_id: str) -> Iterable[Node]:
    """T(s), including s itself."""
    stack = [node_id]
    while stack:
        cur = stack.pop()
        yield tree.get(cur)
        stack.extend(tree.child_ids(cur))
