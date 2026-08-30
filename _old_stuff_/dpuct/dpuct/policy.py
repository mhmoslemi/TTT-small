"""
D-PUCT: dynamic PUCT over a node dataset.

Classic PUCT scores a child s of parent p as

    PUCT(s) = W(s)/m_s + c * P(s|p) * sqrt(m_p) / (1 + m_s)

with W(s) summed over the subtree and P a softmax over sibling values. D-PUCT
changes both halves and adds a third action:

  exploitation   the subtree MAX W_m(s), not the mean. You keep the best result
                 you find, so an exceptional descendant should keep its whole
                 ancestry attractive instead of being averaged away.

  prior          a parent-local softmax over GLOBAL node logits (how a node
                 ranks against the entire archive), not over sibling values.
                 A node in a strong neighbourhood no longer looks good merely
                 because its siblings are weak.

  virtual child  an explicit action meaning "sample one more child here",
                 priced optimistically at mu_L(p) + lambda * sigma_L(p). This
                 is what puts widening and deepening on one comparable scale --
                 and it prices variance among siblings as upside, which is
                 correct when only the best result survives.

    L_p(a) = L_D(a)                          for a real child
             mu_L(p) + lambda * sigma_L(p)   for the virtual action

    pi_D(a|p) = softmax over A(p) of L_p(a) / tau

    score(p, a) = V(p, a) + c * pi_D(a|p) * sqrt(m_p) / (1 + m_p,a)

with V = W_m(a), m_p,a = m_a for a real child, and m_p,a = 0 for the virtual
action so its score rests on the prior and the bonus.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import (LEAF_EXPAND, SELECTION_ACTION_DESCEND, VIRTUAL_EXPAND,
                     DPUCTConfig)
from .signals import blend, rank_signal, softmax
from .tree import Node, Tree

VIRTUAL_KEY = "__virtual__"


@dataclass
class Target:
    """
    A selected place to do work.

    kind == "leaf"    : expand this node into `num_children` children
    kind == "virtual" : add exactly one more child to this node
    """

    kind: str
    node_id: str
    num_children: int
    score: float = 0.0
    value: float = 0.0      # V(p, a), the exploitation term
    prior: float = 0.0      # pi_D(a | p)
    bonus: float = 0.0      # the exploration term
    parent_id: Optional[str] = None

    @property
    def key(self):
        return (self.kind, self.node_id)

    @property
    def is_leaf_expansion(self) -> bool:
        return self.kind == LEAF_EXPAND

    def __repr__(self) -> str:
        return (f"Target({self.kind}, node={self.node_id[:8]}, "
                f"x{self.num_children}, score={self.score:.4f} "
                f"= V {self.value:.4f} + bonus {self.bonus:.4f})")


@dataclass
class ParentPrior:
    """pi_D(. | p) over A(p) = children(p) + {virtual}."""

    parent_id: str
    priors: Dict[str, float] = field(default_factory=dict)
    logits: Dict[str, float] = field(default_factory=dict)
    mu: float = 0.0
    sigma: float = 0.0

    def get(self, action_key: str) -> float:
        return self.priors.get(action_key, 0.0)


class DPUCT:
    """
    The selection policy. Stateless with respect to the tree: hand it a tree,
    get back targets.

        policy = DPUCT(DPUCTConfig(n_select=4, k_children=8))
        targets = policy.select(tree)
    """

    def __init__(self, config: Optional[DPUCTConfig] = None):
        self.config = (config or DPUCTConfig()).validate()

    # ------------------------------------------------------------------
    def global_logits(self, tree: Tree,
                      comparison: Optional[Dict[str, float]] = None
                      ) -> Dict[str, float]:
        """
        L_D(s) for every generated node.

        `comparison` maps node id -> standardized comparison score, e.g.
        EloRatings.as_dict(...). Omit it to run on the rank signal alone.
        """
        nodes = tree.expanded()
        if not nodes:
            return {}
        ids = [n.id for n in nodes]
        ranks = rank_signal([n.subtree_max for n in nodes])
        other = ([float(comparison.get(i, 0.0)) for i in ids]
                 if comparison else None)
        return {i: float(v)
                for i, v in zip(ids, blend(ranks, other, self.config.alpha))}

    # ------------------------------------------------------------------
    def parent_priors(self, tree: Tree, logits: Dict[str, float]
                      ) -> Dict[str, ParentPrior]:
        """Eq. 4 and 5: the virtual child's logit, then a per-parent softmax."""
        out: Dict[str, ParentPrior] = {}
        lam = float(self.config.lambda_virtual)

        for parent in tree.nodes():
            child_ids = tree.child_ids(parent.id)
            action_logits = {cid: float(logits.get(cid, 0.0)) for cid in child_ids}

            if child_ids:
                vals = np.asarray(list(action_logits.values()), dtype=float)
                mu, sigma = float(vals.mean()), float(vals.std())
            else:
                # No children yet, so there is no sibling distribution to
                # estimate from; fall back to the parent's own standing.
                mu, sigma = float(logits.get(parent.id, 0.0)), 0.0

            action_logits[VIRTUAL_KEY] = mu + lam * sigma
            keys = list(action_logits)
            probs = softmax([action_logits[k] for k in keys], self.config.tau)

            out[parent.id] = ParentPrior(
                parent_id=parent.id,
                priors={k: float(p) for k, p in zip(keys, probs)},
                logits=action_logits, mu=mu, sigma=sigma)
        return out

    # ------------------------------------------------------------------
    def _values(self, tree: Tree) -> Dict[str, float]:
        """V(p, a) = W_m(a), optionally rescaled so c transfers across problems."""
        nodes = tree.nodes()
        if not self.config.normalize_exploitation:
            return {n.id: float(n.subtree_max) for n in nodes}

        expanded = tree.expanded()
        if expanded:
            lo = min(n.subtree_max for n in expanded)
            hi = max(n.subtree_max for n in expanded)
            if hi - lo > 1e-12:
                return {n.id: (n.subtree_max - lo) / (hi - lo) for n in nodes}
        return {n.id: 0.0 for n in nodes}

    def _bonus(self, prior: float, m_p: int, m_pa: int) -> float:
        return self.config.c_puct * prior * math.sqrt(max(m_p, 0)) / (1.0 + m_pa)

    def _virtual_value(self, tree: Tree, parent_id: str,
                       values: Dict[str, float]) -> float:
        if self.config.virtual_value_mode == "parent_mean":
            kids = tree.child_ids(parent_id)
            return float(np.mean([values.get(c, 0.0) for c in kids])) if kids else 0.0
        return 0.0

    # ------------------------------------------------------------------
    def score_all(self, tree: Tree,
                  comparison: Optional[Dict[str, float]] = None) -> List[Target]:
        """Every candidate target, scored and sorted. Useful for inspection."""
        logits = self.global_logits(tree, comparison)
        priors = self.parent_priors(tree, logits)
        values = self._values(tree)
        targets = (self._targets_descend(tree, priors, values)
                   if self.config.selection_mode == SELECTION_ACTION_DESCEND
                   else self._targets_per_node(tree, priors, values))
        targets.sort(key=lambda t: t.score, reverse=True)
        return targets

    def select(self, tree: Tree,
               comparison: Optional[Dict[str, float]] = None,
               n: Optional[int] = None) -> List[Target]:
        """
        The top-n targets for this round.

        Call tree.recompute() first, or use SearchLoop, which does it for you.
        """
        limit = int(n if n is not None else self.config.n_select)
        picked: List[Target] = []
        seen = set()
        for target in self.score_all(tree, comparison):
            if target.key in seen:
                continue
            seen.add(target.key)
            picked.append(target)
            if len(picked) >= limit:
                break
        return picked

    # ------------------------------------------------------------------
    def _targets_per_node(self, tree: Tree, priors: Dict[str, ParentPrior],
                          values: Dict[str, float]) -> List[Target]:
        """One target per node: leaves deepen, internal nodes widen."""
        k = self.config.k_children
        out: List[Target] = []

        for node in tree.nodes():
            if tree.is_leaf(node.id):
                parent = tree.parent_of(node)
                if parent is None:
                    prior, m_p, parent_id = 1.0, node.subtree_size, None
                else:
                    prior = priors[parent.id].get(node.id)
                    m_p, parent_id = parent.subtree_size, parent.id
                value = values.get(node.id, 0.0)
                bonus = self._bonus(prior, m_p, node.subtree_size)
                out.append(Target(LEAF_EXPAND, node.id, k, value + bonus,
                                  value, prior, bonus, parent_id))
            else:
                prior = priors[node.id].get(VIRTUAL_KEY)
                value = self._virtual_value(tree, node.id, values)
                bonus = self._bonus(prior, node.subtree_size, 0)
                out.append(Target(VIRTUAL_EXPAND, node.id, 1, value + bonus,
                                  value, prior, bonus, node.id))
        return out

    def _targets_descend(self, tree: Tree, priors: Dict[str, ParentPrior],
                         values: Dict[str, float]) -> List[Target]:
        """Score (parent, action) pairs, then walk down to a concrete site."""
        k = self.config.k_children
        scored = []

        for parent in tree.nodes():
            pp = priors[parent.id]
            m_p = parent.subtree_size
            for cid in tree.child_ids(parent.id):
                child = tree.get(cid)
                prior, value = pp.get(cid), values.get(cid, 0.0)
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
                out.append(Target(VIRTUAL_EXPAND, parent_id, 1, score,
                                  value, prior, bonus, parent_id))
                continue
            cur, guard = action, 0
            while not tree.is_leaf(cur) and guard < 256:
                guard += 1
                pp = priors[cur]
                best = max(pp.priors, key=lambda a: pp.priors[a])
                if best == VIRTUAL_KEY:
                    break
                cur = best
            if tree.is_leaf(cur):
                out.append(Target(LEAF_EXPAND, cur, k, score, value, prior,
                                  bonus, tree.get(cur).parent_id))
            else:
                out.append(Target(VIRTUAL_EXPAND, cur, 1, score, value, prior,
                                  bonus, cur))
        return out

    # ------------------------------------------------------------------
    def explain(self, tree: Tree, comparison: Optional[Dict[str, float]] = None,
                limit: int = 10) -> str:
        """A readable score table -- the fastest way to see why it picked what it did."""
        rows = self.score_all(tree, comparison)[:limit]
        if not rows:
            return "(no candidate targets)"
        lines = [f"{'kind':<8} {'node':<10} {'x':>3} {'score':>9} "
                 f"{'V':>9} {'prior':>7} {'bonus':>9}",
                 "-" * 60]
        for t in rows:
            lines.append(f"{t.kind:<8} {t.node_id[:8]:<10} {t.num_children:>3} "
                         f"{t.score:>9.4f} {t.value:>9.4f} {t.prior:>7.3f} "
                         f"{t.bonus:>9.4f}")
        return "\n".join(lines)
