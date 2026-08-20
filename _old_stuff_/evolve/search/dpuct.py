"""
D-PUCT: dynamic PUCT over the node dataset (Sec. 2.1).

Standard PUCT scores a child s of parent p with a summed subtree value and a
softmax-over-siblings prior. D-PUCT changes both halves:

  exploitation   W(s)/m_s  ->  W_m(s), the subtree MAX (Eq. 2)
  prior          softmax over sibling Q  ->  a parent-local softmax over
                 *global* node logits (Eq. 3), extended with a virtual child
                 that represents "sample one more sibling here" (Eq. 4)

    L_p(a) = L_D(a)                    for a real child
             mu_L(p) + lambda sigma_L(p)   for the virtual action s-hat   (Eq. 4)

    pi_D(a | p) = softmax_{A(p)} ( L_p(a) / tau )                         (Eq. 5)

    D-PUCT(p, a) = V(p, a) + c pi_D(a | p) sqrt(m_p) / (1 + m_p,a)        (Eq. 6)

with V(p, a) = W_m(a) and m_p,a = m_a for a real child, and m_p,s-hat = 0 for
the virtual action, so its score is carried by the prior and the bonus.

Selection modes
---------------
"node" (default) reads the paper's "select top-n nodes based on scores in
Eq. (6)" literally: every node contributes exactly one generation target.

    leaf node        -> expand it into k children   (deepen)
    node with kids   -> its virtual action, 1 child (widen)

An internal node is therefore never itself a generation target, which is what
removes the paper's undefined case -- Sec. 2.1 defines an outcome only for a
selected virtual action and a selected leaf. Its children remain reachable:
each is a target in its own right, and the parent stays reachable through its
virtual action. Batch size lands in [n, nk] exactly as stated.

"action_descend" scores (parent, action) pairs instead and walks down through a
chosen internal child until it reaches a leaf or a virtual action.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.tree import SearchTree
from core.types import LEAF_EXPAND, VIRTUAL_EXPAND, Target
from search.signals import global_node_logits, minmax_normalize, rank_signal

VIRTUAL_KEY = "__virtual__"


def softmax(logits: Sequence[float], tau: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    if arr.size == 0:
        return arr
    t = max(float(tau), 1e-6)
    z = arr / t
    z -= z.max()
    e = np.exp(z)
    total = e.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(arr, 1.0 / arr.size)
    return e / total


@dataclass
class ParentPrior:
    """pi_D(. | p) over A(p) = C(p) union {s-hat}."""
    parent_id: str
    priors: Dict[str, float] = field(default_factory=dict)
    logits: Dict[str, float] = field(default_factory=dict)
    mu: float = 0.0
    sigma: float = 0.0

    def get(self, action_key: str) -> float:
        return self.priors.get(action_key, 0.0)


class DPUCT:
    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Eq. 3 — global node logits over the archive
    # ------------------------------------------------------------------
    def compute_logits(self, tree: SearchTree,
                       elo_standardized: Optional[Dict[str, float]] = None
                       ) -> Dict[str, float]:
        """
        L_D(s) for every generated node. Roots are the initial state s_0, not
        generated candidates, so they stay out of D's statistics -- a root's
        W_m equals the global best, which would distort every rank.
        """
        nodes = tree.evaluated()
        if not nodes:
            return {}
        ids = [n.id for n in nodes]
        rank_sig = rank_signal([n.subtree_max for n in nodes])

        elo_sig = None
        if elo_standardized:
            elo_sig = [float(elo_standardized.get(i, 0.0)) for i in ids]

        logits = global_node_logits(rank_sig, elo_sig, self.cfg.alpha)
        return {i: float(v) for i, v in zip(ids, logits)}

    # ------------------------------------------------------------------
    # Eq. 4 + 5 — virtual child and the parent-local prior
    # ------------------------------------------------------------------
    def parent_priors(self, tree: SearchTree, logits: Dict[str, float]
                      ) -> Dict[str, ParentPrior]:
        out: Dict[str, ParentPrior] = {}
        lam = float(self.cfg.lambda_virtual)

        for parent in tree.nodes():
            child_ids = tree.child_ids(parent.id)
            action_logits: Dict[str, float] = {
                cid: float(logits.get(cid, 0.0)) for cid in child_ids
            }

            if child_ids:
                vals = np.asarray(list(action_logits.values()), dtype=float)
                mu = float(vals.mean())
                # Population std (1/n_p), as written in Sec. 2.1.
                sigma = float(vals.std())
            else:
                # n_p = 0: Eq. 4 is defined only for n_p >= 1. Fall back to the
                # parent's own standing, which is the only evidence available.
                mu = float(logits.get(parent.id, 0.0))
                sigma = 0.0

            action_logits[VIRTUAL_KEY] = mu + lam * sigma

            keys = list(action_logits)
            probs = softmax([action_logits[k] for k in keys], self.cfg.tau)
            out[parent.id] = ParentPrior(
                parent_id=parent.id,
                priors={k: float(p) for k, p in zip(keys, probs)},
                logits=action_logits,
                mu=mu,
                sigma=sigma,
            )
        return out

    # ------------------------------------------------------------------
    # Eq. 6 — scores
    # ------------------------------------------------------------------
    def _value_table(self, tree: SearchTree) -> Dict[str, float]:
        """
        V(p, a) = W_m(a). Optionally rescaled to [0, 1] by the archive spread:
        W_m is in raw reward units while the prior is a probability, so without
        rescaling c has to be retuned for every problem.
        """
        nodes = tree.nodes()
        raw = [n.subtree_max for n in nodes]
        if self.cfg.normalize_exploitation:
            evaluated = tree.evaluated()
            if evaluated:
                lo = min(n.subtree_max for n in evaluated)
                hi = max(n.subtree_max for n in evaluated)
                if hi - lo > 1e-12:
                    return {n.id: (n.subtree_max - lo) / (hi - lo) for n in nodes}
            return {n.id: 0.0 for n in nodes}
        return {n.id: float(v) for n, v in zip(nodes, raw)}

    def _bonus(self, prior: float, m_p: int, m_pa: int) -> float:
        return float(self.cfg.c_puct) * prior * math.sqrt(max(m_p, 0)) / (1.0 + m_pa)

    def _virtual_value(self, tree: SearchTree, parent_id: str,
                       values: Dict[str, float]) -> float:
        if self.cfg.virtual_value_mode == "parent_mean":
            kids = tree.child_ids(parent_id)
            if kids:
                return float(np.mean([values.get(c, 0.0) for c in kids]))
            return 0.0
        return 0.0  # "zero": Eq. 6's virtual action is prior + bonus only

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, tree: SearchTree, logits: Dict[str, float]) -> List[Target]:
        priors = self.parent_priors(tree, logits)
        values = self._value_table(tree)
        if self.cfg.selection_mode == "action_descend":
            candidates = self._targets_action_descend(tree, priors, values)
        else:
            candidates = self._targets_per_node(tree, priors, values)

        candidates.sort(key=lambda t: t.score, reverse=True)
        picked: List[Target] = []
        seen = set()
        for t in candidates:
            if t.key in seen:
                continue
            seen.add(t.key)
            picked.append(t)
            if len(picked) >= int(self.cfg.n_select):
                break
        return picked

    def _targets_per_node(self, tree: SearchTree,
                          priors: Dict[str, ParentPrior],
                          values: Dict[str, float]) -> List[Target]:
        k = int(self.cfg.k_children)
        out: List[Target] = []

        for node in tree.nodes():
            if tree.is_leaf(node.id):
                # Deepen: scored as the action "node" taken at its parent.
                parent = tree.parent_of(node)
                if parent is None:
                    # A root with no children: nothing competes with it locally.
                    prior, m_p = 1.0, node.subtree_size
                    parent_id = None
                else:
                    prior = priors[parent.id].get(node.id)
                    m_p = parent.subtree_size
                    parent_id = parent.id
                value = values.get(node.id, 0.0)
                bonus = self._bonus(prior, m_p, node.subtree_size)
                out.append(Target(
                    kind=LEAF_EXPAND, node_id=node.id, num_children=k,
                    score=value + bonus, value=value, prior=prior, bonus=bonus,
                    parent_id=parent_id,
                ))
            else:
                # Widen: the virtual action at this node. m_p,s-hat = 0.
                prior = priors[node.id].get(VIRTUAL_KEY)
                value = self._virtual_value(tree, node.id, values)
                bonus = self._bonus(prior, node.subtree_size, 0)
                out.append(Target(
                    kind=VIRTUAL_EXPAND, node_id=node.id, num_children=1,
                    score=value + bonus, value=value, prior=prior, bonus=bonus,
                    parent_id=node.id,
                ))
        return out

    def _targets_action_descend(self, tree: SearchTree,
                                priors: Dict[str, ParentPrior],
                                values: Dict[str, float]) -> List[Target]:
        """Score every (parent, action) pair, then resolve to a generation site."""
        k = int(self.cfg.k_children)
        scored = []

        for parent in tree.nodes():
            pp = priors[parent.id]
            m_p = parent.subtree_size
            for cid in tree.child_ids(parent.id):
                child = tree.get(cid)
                prior = pp.get(cid)
                value = values.get(cid, 0.0)
                bonus = self._bonus(prior, m_p, child.subtree_size)
                scored.append((value + bonus, parent.id, cid, value, prior, bonus))
            prior = pp.get(VIRTUAL_KEY)
            value = self._virtual_value(tree, parent.id, values)
            bonus = self._bonus(prior, m_p, 0)
            scored.append((value + bonus, parent.id, VIRTUAL_KEY, value, prior, bonus))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Target] = []
        for score, parent_id, action, value, prior, bonus in scored:
            if action == VIRTUAL_KEY:
                out.append(Target(kind=VIRTUAL_EXPAND, node_id=parent_id,
                                  num_children=1, score=score, value=value,
                                  prior=prior, bonus=bonus, parent_id=parent_id))
                continue
            # Walk down through internal children until a leaf or a virtual action.
            cur, guard = action, 0
            while not tree.is_leaf(cur) and guard < 256:
                guard += 1
                pp = priors[cur]
                best_action = max(pp.priors, key=lambda a: pp.priors[a])
                if best_action == VIRTUAL_KEY:
                    break
                cur = best_action
            if tree.is_leaf(cur):
                out.append(Target(kind=LEAF_EXPAND, node_id=cur, num_children=k,
                                  score=score, value=value, prior=prior,
                                  bonus=bonus, parent_id=tree.get(cur).parent_id))
            else:
                out.append(Target(kind=VIRTUAL_EXPAND, node_id=cur, num_children=1,
                                  score=score, value=value, prior=prior,
                                  bonus=bonus, parent_id=cur))
        return out
