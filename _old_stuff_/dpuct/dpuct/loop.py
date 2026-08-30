"""
An optional driver, for when you do not already have a search loop.

If you have your own MCTS, ignore this and use DPUCT.select() directly as the
selection step. If you are starting from scratch, SearchLoop is the whole
algorithm:

    recompute statistics -> select targets -> expand them -> repeat

You supply one function: given a node and how many children to produce, return
those children. Everything else is handled.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import DPUCTConfig
from .policy import DPUCT, Target
from .tree import Node, Tree

# expand(node, num_children) -> [(value, payload), ...]
ExpandFn = Callable[[Node, int], Sequence[Tuple[float, Any]]]

# compare(a_id, b_id) -> 1.0 | 0.0 | 0.5 | None
CompareFn = Callable[[str, str], Optional[float]]


@dataclass
class RoundStats:
    round: int
    targets: int = 0
    leaf_targets: int = 0
    virtual_targets: int = 0
    children: int = 0
    best_value: float = 0.0
    archive_size: int = 0
    comparisons: int = 0

    def __repr__(self) -> str:
        return (f"round {self.round:>3}  targets={self.targets:>3} "
                f"({self.leaf_targets}L/{self.virtual_targets}V) "
                f"children={self.children:>4} |D|={self.archive_size:>5} "
                f"best={self.best_value:.6f}")


@dataclass
class SearchResult:
    tree: Tree
    best: Optional[Node]
    history: List[RoundStats] = field(default_factory=list)

    @property
    def best_value(self) -> Optional[float]:
        return self.best.value if self.best else None

    @property
    def best_payload(self) -> Any:
        return self.best.payload if self.best else None

    def curve(self) -> List[float]:
        """Best-so-far after each round; the thing you actually plot."""
        return [s.best_value for s in self.history]


class SearchLoop:
    def __init__(self, expand: ExpandFn,
                 config: Optional[DPUCTConfig] = None,
                 compare: Optional[CompareFn] = None,
                 elo=None, elo_top_k: int = 16,
                 elo_pairing: str = "round_robin", elo_matches: int = 30,
                 on_round: Optional[Callable[[RoundStats], None]] = None):
        """
        expand   : (node, num_children) -> [(value, payload), ...]
                   Return fewer than asked if you like; the loop adapts.
        compare  : optional pairwise judge enabling the Elo signal. Needs `elo`.
        elo      : an EloRatings instance. Supply both, or neither.
        on_round : called with RoundStats after each round, for progress output.
        """
        self.expand = expand
        self.config = (config or DPUCTConfig()).validate()
        self.policy = DPUCT(self.config)
        self.compare = compare
        self.elo = elo
        self.elo_top_k = int(elo_top_k)
        self.elo_pairing = elo_pairing
        self.elo_matches = int(elo_matches)
        self.on_round = on_round

    # ------------------------------------------------------------------
    def _run_comparisons(self, tree: Tree) -> Tuple[Optional[Dict[str, float]], int]:
        if self.elo is None or self.compare is None or self.config.alpha >= 1.0:
            # alpha = 1 ignores the comparison term, so playing matches would
            # cost whatever a comparison costs and change nothing.
            return None, 0

        from .elo import build_pairings

        candidates = tree.top_k(self.elo_top_k)
        if len(candidates) < 2:
            return None, 0
        ids = [n.id for n in candidates]
        self.elo.ensure(ids)
        played = self.elo.play(
            build_pairings(ids, self.elo_pairing, self.elo_matches), self.compare)
        all_ids = [n.id for n in tree.expanded()]
        return self.elo.as_dict(all_ids), played

    # ------------------------------------------------------------------
    def run(self, rounds: int, tree: Optional[Tree] = None,
            root_payload: Any = None, root_value: float = 0.0) -> SearchResult:
        tree = tree if tree is not None else Tree(self.config.max_archive_size)
        if not tree.roots():
            tree.add_root(value=root_value, payload=root_payload)

        history: List[RoundStats] = []
        for index in range(int(rounds)):
            tree.recompute()
            comparison, played = self._run_comparisons(tree)
            targets = self.policy.select(tree, comparison)
            if not targets:
                break

            produced = 0
            for target in targets:
                node = tree.get(target.node_id)
                for value, payload in self.expand(node, target.num_children) or ():
                    tree.add_child(target.node_id, value=value, payload=payload)
                    produced += 1

            tree.recompute()
            tree.prune()
            best = tree.best()

            stats = RoundStats(
                round=index, targets=len(targets),
                leaf_targets=sum(1 for t in targets if t.is_leaf_expansion),
                virtual_targets=sum(1 for t in targets if not t.is_leaf_expansion),
                children=produced, archive_size=len(tree),
                best_value=best.value if best else float("-inf"),
                comparisons=played)
            history.append(stats)
            if self.on_round:
                self.on_round(stats)

        tree.recompute()
        return SearchResult(tree=tree, best=tree.best(), history=history)


def search(expand: ExpandFn, rounds: int = 10,
           config: Optional[DPUCTConfig] = None, *,
           tree: Optional[Tree] = None, root_payload: Any = None,
           root_value: float = 0.0, **kwargs) -> SearchResult:
    """
    One-call entry point.

        result = search(expand=my_expand, rounds=20,
                        config=DPUCTConfig(n_select=4, k_children=8),
                        root_payload=starting_state)
        print(result.best_value, result.best_payload)

    Anything in **kwargs goes to SearchLoop (compare, elo, on_round, ...); the
    root arguments are routed to run(), where they belong.
    """
    return SearchLoop(expand, config=config, **kwargs).run(
        rounds, tree=tree, root_payload=root_payload, root_value=root_value)
