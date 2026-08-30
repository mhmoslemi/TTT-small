# dpuct

Max-seeking tree search you can drop into an existing MCTS. Pure Python + numpy,
no ML dependencies.

## Why

Classic MCTS backs up the **mean** return of a subtree. That is correct for
games — you have to play the average outcome, so the average is what you want
to estimate.

It is wrong for discovery. When you are searching for one good molecule, one
fast kernel, one correct proof, you keep the best thing you find and discard
everything else. Under a mean backup a branch holding one brilliant result and
nine failures scores *worse* than a branch of nine mediocre ones — so the search
walks away from the brilliant result.

|              | classic PUCT                  | D-PUCT                                  |
|--------------|-------------------------------|-----------------------------------------|
| **backup**   | subtree mean                  | subtree **max**, `W_m(s)`               |
| **prior**    | softmax over *sibling* values | softmax over **archive-wide** ranks     |
| **actions**  | descend into a child          | descend, **or sample one more child here** |

The third is the one you cannot easily bolt on yourself. D-PUCT adds an explicit
"generate a new sibling" action priced at `μ_L(p) + λ·σ_L(p)` — the mean logit of
the existing children plus λ standard deviations. The `σ` term treats *variance*
among a parent's children as upside, which is right when only the best survivor
ships: a parent that produced {0, 0.1, 9} is a better bet for one more sample
than one that produced {1, 1, 1}.

## Install

```bash
pip install -e .        # or just: pip install numpy
```

## Use it as a selection step

If you already have a search loop, this is the whole integration:

```python
from dpuct import Tree, DPUCT, DPUCTConfig

tree = Tree()
root = tree.add_root(payload=initial_state)
policy = DPUCT(DPUCTConfig(n_select=4, k_children=8, c_puct=1.0))

for _ in range(rounds):
    tree.recompute()                        # refresh W_m and m_s
    for target in policy.select(tree):
        node = tree.get(target.node_id)
        for value, payload in my_expand(node, target.num_children):
            tree.add_child(node.id, value=value, payload=payload)

print(tree.best().payload)
```

Three rules:

1. Call `tree.recompute()` before `select()` — statistics are not maintained
   incrementally.
2. `value` is always **higher-is-better**. Minimizing? Store the negation.
3. Honour `target.num_children`: `k` for a `leaf` target, exactly 1 for a
   `virtual` one. Returning fewer is fine.

## Or let it drive

```python
from dpuct import search, DPUCTConfig

def expand(node, num_children):
    return [(score(c), c) for c in propose(node.payload, num_children)]

result = search(expand, rounds=20,
                config=DPUCTConfig(n_select=4, k_children=8),
                root_payload=start)

print(result.best_value, result.best_payload)
print(result.curve())        # best-so-far per round
```

## Seeing why it chose something

```python
print(policy.explain(tree))
```
```
kind     node          x     score         V   prior     bonus
--------------------------------------------------------------
virtual  72703c9c      1    2.0674    0.0000   0.431    2.0674
leaf     f1015a50      6    1.9905    1.0000   0.597    0.9905
```

## Drawing the tree

Seeing the tree is how you tell whether `c_puct` is doing what you think.

```python
tree.show()                                    # text, no dependencies
tree.show(targets=policy.select(tree))         # mark what was just selected
tree.show(max_depth=3, max_children=2)         # trim a big tree
tree.draw(max_depth=6)                         # matplotlib
```

```
@ Q=0 W=9 m=9
|-- o Q=0.1 W=9 m=6
|   |-- * Q=9 W=9 m=1
|   |-- o Q=0 W=0 m=1
|   `-- ... 3 weaker sibling(s)
`-- o Q=1 W=1.1 m=2
    `-- o Q=1.1 W=1.1 m=1
@ root   * best   > selected   o node
```

Each node shows the three numbers selection is computed from: `Q` (own value),
`W` (subtree max), `m` (subtree size). `Q=0.1 W=9` on that branch is the library
in one line — its own value is 0.1, but the best thing beneath it is a 9. A mean
backup would have shown you 0.9 and moved on.

| renderer | needs | good for |
|---|---|---|
| `render_text` / `tree.show()` | nothing | terminals, logs, quick checks |
| `to_mermaid` | nothing | GitHub markdown, notebook viewers |
| `to_dot` | nothing (graphviz to rasterize) | publication figures |
| `draw` / `tree.draw()` | matplotlib | exploring a big tree |

Trimming keeps the **best** children, not the first ones, and truncation is
always reported rather than silently hiding nodes. `path_to_best(tree)` returns
the winning lineage, useful as a `highlight` set.

Reading the matplotlib plot: a single deep spike with no fan means `c_puct` is
too low; a wide bush that never descends means it is too high.

## Configuration

| parameter | symbol | what it does |
|---|---|---|
| `n_select` | n | targets returned per round |
| `k_children` | k | children when a leaf is selected (virtual always gives 1) |
| `c_puct` | c | **explore/exploit dial — tune this first.** 0 = greedy on `W_m` |
| `alpha` | α | 1.0 = ranks only, 0.0 = pairwise comparisons only |
| `lambda_virtual` | λ | optimism of the virtual child; 0 = no credit for variance |
| `tau` | τ | prior temperature; small = peaked, large = flat |
| `normalize_exploitation` | | rescale `W_m` to [0,1] so `c` transfers across problems |
| `virtual_value_mode` | | `zero` (prior + bonus only) or `parent_mean` |
| `selection_mode` | | `node` (one target per node) or `action_descend` |
| `max_archive_size` | | cap on stored nodes; 0 = unbounded |

A round of `n` targets produces between `n` and `n·k` children — a bound you can
budget against.

**Tuning order:** `c_puct` first, everything else is second-order. Then
`k_children` for evaluation budget, then `lambda_virtual` if the search is
deepening one lineage and ignoring alternatives, then `alpha` once you actually
have a comparator worth calling.

## Optional: pairwise comparisons

Scalar rewards cannot separate two nodes that scored identically. If you have a
comparator that can — a human, a stronger model, a slow accurate simulator —
feed its judgements in as Elo ratings:

```python
from dpuct import EloRatings, build_pairings

elo = EloRatings(k_factor=24, scale=400)
ids = [n.id for n in tree.top_k(16)]
elo.ensure(ids)
elo.play(build_pairings(ids, "round_robin"), my_comparator)

targets = policy.select(tree, comparison=elo.as_dict(ids))   # needs alpha < 1
```

`my_comparator(a_id, b_id)` returns `1.0` / `0.0` / `0.5`, or `None` to skip.

On `scale`: classic Elo uses 400, where a 400-point gap means a 10:1 expected
score. Some formulations omit the denominator, which is ~400× steeper — a
one-point gap already implies p = 0.09 — and needs a correspondingly tiny
`k_factor`. Pick a convention and set `k_factor` to match.

## When not to use this

If you have to live with the average outcome — a game you must actually play, a
policy you must actually deploy — the mean backup is correct and standard MCTS
is the right tool. D-PUCT is for search where failures are free and only the
best survivor ships.

## Layout

```
dpuct/
├── dpuct/
│   ├── tree.py      Tree and Node; W_m and m_s
│   ├── config.py    DPUCTConfig, validated
│   ├── signals.py   rank signal, standardization, softmax
│   ├── elo.py       optional pairwise-comparison ratings
│   ├── policy.py    the selection rule
│   ├── viz.py       text / mermaid / graphviz / matplotlib drawing
│   └── loop.py      optional driver: SearchLoop / search()
├── examples/rastrigin.py       a runnable non-LLM search
├── notebooks/tutorial.ipynb    start here
└── tests/                      60 tests
```

```bash
python -m pytest tests/ -q
python examples/rastrigin.py
jupyter notebook notebooks/tutorial.ipynb
```

## Provenance

Extracted from the `evolve/` framework in this repository, where D-PUCT drives
LLM-generated program search. This package is standalone — nothing here imports
a model, and the two copies are independent for now.
