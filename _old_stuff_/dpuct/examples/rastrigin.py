"""
A worked example with no LLM in sight: find the global maximum of a rugged
function by refining candidates in a tree.

Each node holds a point x. Expanding a node perturbs x, with the step size
shrinking as the tree deepens -- coarse exploration near the root, fine
refinement further down. The landscape is deliberately multimodal, so a search
that chases the best local average gets stuck in whichever basin it started in.

Run: python examples/rastrigin.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpuct import DPUCTConfig, SearchLoop

DIM = 2


def rastrigin(point):
    """Negated Rastrigin: many local optima, one global max of 0 at the origin."""
    return -sum(x * x - 10.0 * math.cos(2.0 * math.pi * x) + 10.0 for x in point)


def make_expand(rng, span=5.12):
    def expand(node, num_children):
        base = node.payload if node.payload is not None else [0.0] * DIM
        # Shrink the step with depth: explore coarsely, refine finely.
        step = span / (2.0 ** min(node.depth, 8))
        out = []
        for _ in range(num_children):
            child = [max(-span, min(span, x + rng.gauss(0.0, step))) for x in base]
            out.append((rastrigin(child), child))
        return out
    return expand


def main():
    rng = random.Random(0)
    loop = SearchLoop(
        expand=make_expand(rng),
        config=DPUCTConfig(n_select=4, k_children=6, c_puct=0.7,
                           alpha=1.0, lambda_virtual=1.0, tau=1.0,
                           max_archive_size=2000),
        on_round=print,
    )
    start = [rng.uniform(-5.12, 5.12) for _ in range(DIM)]
    result = loop.run(rounds=25, root_payload=start,
                      root_value=rastrigin(start))

    print(f"\nstart      f = {rastrigin(start):.6f}  at {[round(x, 3) for x in start]}")
    print(f"best found f = {result.best_value:.6f}  at "
          f"{[round(x, 3) for x in result.best_payload]}")
    print(f"global max f = 0.0 at the origin")
    print(f"tree: {result.tree.summary()}")


if __name__ == "__main__":
    main()
