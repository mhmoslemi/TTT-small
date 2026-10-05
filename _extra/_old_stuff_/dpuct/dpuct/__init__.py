"""
dpuct — max-seeking tree search you can drop into an existing MCTS.

Classic MCTS backs up the *mean* return of a subtree, which is right when you
have to live with the average outcome. It is wrong when you keep only the single
best thing the search finds -- an exceptional descendant gets averaged away by
its mediocre siblings, and the branch that produced it stops looking attractive.

D-PUCT changes three things:

  * back up the subtree MAX, so one exceptional result keeps its whole ancestry
    attractive;
  * build the prior from how a node ranks against the ENTIRE archive, not just
    against its siblings, so a node is not flattered by weak company;
  * add an explicit "sample one more child here" action, priced optimistically
    from the spread of the existing children -- which makes widening and
    deepening directly comparable, and treats variance as upside.

Two ways in.

Already have a search loop? Use the policy as your selection step:

    from dpuct import Tree, DPUCT, DPUCTConfig

    tree = Tree()
    root = tree.add_root(payload=initial_state)
    ...
    tree.recompute()
    for target in DPUCT(DPUCTConfig(n_select=4)).select(tree):
        if target.is_leaf_expansion:
            ...  # generate target.num_children children of target.node_id
        else:
            ...  # generate exactly one more child of target.node_id

Starting from scratch? Supply an expand function and let the loop drive:

    from dpuct import search

    def expand(node, num_children):
        return [(score(c), c) for c in propose(node.payload, num_children)]

    result = search(expand, rounds=20)
    print(result.best_value, result.best_payload)
"""

from .config import (LEAF_EXPAND, SELECTION_ACTION_DESCEND, SELECTION_NODE,
                     VIRTUAL_EXPAND, DPUCTConfig)
from .elo import EloRatings, build_pairings
from .loop import RoundStats, SearchLoop, SearchResult, search
from .policy import DPUCT, ParentPrior, Target
from .signals import (average_ranks, blend, minmax, rank_signal, softmax,
                      standardize)
from .tree import Node, Tree
from .viz import (draw, layout, path_to_best, render_text, to_dot,
                  to_mermaid)

__version__ = "0.1.0"

__all__ = [
    # core
    "Tree", "Node", "DPUCT", "DPUCTConfig", "Target", "ParentPrior",
    # driver
    "search", "SearchLoop", "SearchResult", "RoundStats",
    # optional comparison signal
    "EloRatings", "build_pairings",
    # drawing
    "render_text", "draw", "to_dot", "to_mermaid", "path_to_best", "layout",
    # signal primitives, exposed for inspection and custom priors
    "rank_signal", "standardize", "average_ranks", "blend", "minmax", "softmax",
    # constants
    "LEAF_EXPAND", "VIRTUAL_EXPAND", "SELECTION_NODE", "SELECTION_ACTION_DESCEND",
    "__version__",
]
