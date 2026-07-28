# EVOLVE

Memory-augmented D-PUCT search over a node dataset, with failure-aware
max-seeking test-time RL.

Agentic discovery is scored by the **best** valid candidate a search produces,
not the average, and every component here is built around that asymmetry:
`W_m(s) = max_{y∈T(s)} Q(y)` keeps an exceptional descendant visible at every
ancestor, the virtual child prices *variance* among siblings as upside, and the
β-tilt in the RL update interpolates from mean-seeking to max-seeking.

## Quick start

```bash
pip install -r requirements.txt      # see the note about Unsloth inside

bash run.sh                          # circle packing, defaults
DRY_RUN=1 bash run.sh                # resolve config + provenance, run nothing
python main.py --backend mock --steps 3   # full loop, no model, no GPU
python -m pytest tests/ -q           # 98 tests, no GPU required
```

## Configuration

Five layers, highest priority first. Every key is tagged with where it came
from; `--print-config` shows the whole table.

| Priority | Layer | Origin tag |
|---|---|---|
| 1 | CLI flags from `run.sh` | `cli` |
| 2 | `FRAMEWORK_OVERRIDES` in `config.py` | `config.py:override` |
| 3 | `examples/<name>/config.yaml` | `yaml:example` |
| 4 | `configs/base.yaml` | `yaml:base` |
| 5 | dataclass defaults in `config.py` | `config.py:default` |

Layers 2 and 5 are both `config.py` and play different roles. The dataclass
defaults are the floor — the schema, plus a value when nobody else supplies
one. `FRAMEWORK_OVERRIDES` is the ceiling below the CLI: an explicit dict of
pins that outrank every YAML file. Without that split, putting `config.py`
above YAML would make every example's `config.yaml` unreachable.

Every leaf has a dotted path, settable three equivalent ways:

```yaml
search:            # nested YAML
  alpha: 0.5
search.alpha: 0.5  # dotted YAML
```
```bash
--alpha 0.5                 # named flag
--set search.alpha=0.5      # generic, reaches any leaf
```

`example.params` is an open namespace: arbitrary keys are accepted there and
passed to the example untouched, so problem knobs never enter the framework
schema. Every other path must exist — a typo raises with a suggestion instead
of silently doing nothing.

## Layout

```
config.py           the five-layer resolver + the full schema
run.sh              launcher; every variable is opt-in
main.py             entry point
configs/base.yaml   framework-wide YAML

core/     types, tree + archive D, registry, engine (Algorithm 1)
search/   signals (R~), elo (E~), pairing, dpuct (Eq. 3-6)
memory/   bank, extractor (2L lessons/step), retrieval (Eq. 7), prompts
rl/       advantage (Eq. 8, 9), objective (Eq. 10, 11), trainer
llm/      backend (Unsloth↔HF), backbone, generation, judge, mock
envs/     Example ABC, subprocess sandbox
prompting/ builder — [ d | parent | top-m memories | instruction ]
runio/    run directories, rollout logs, tree snapshots
examples/circle_packing/   config.yaml, env.py, prompts.py, validator.py
```

## Where the paper maps into the code

| Paper | Code |
|---|---|
| Eq. 2 — `W_m(s)`, `m_s` | `core/tree.py::SearchTree.recompute` |
| Eq. 3 — global node logit | `search/signals.py::global_node_logits` |
| Elo debate | `search/elo.py`, `llm/judge.py` |
| Eq. 4 — virtual child | `search/dpuct.py::DPUCT.parent_priors` |
| Eq. 5 — parent-local prior | same, temperature softmax over `A(p)` |
| Eq. 6 — D-PUCT score | `search/dpuct.py::DPUCT.select` |
| Sec. 2.2 — 2L lessons | `memory/extractor.py::LessonExtractor.extract` |
| Eq. 7 — top-m retrieval | `memory/bank.py::MemoryBank.retrieve` |
| Eq. 8 — group-relative tilt | `rl/advantage.py::group_relative_advantages` |
| Eq. 9 — feedback teacher | `rl/trainer.py::_feedback_advantage` |
| Eq. 10, 11 — clipped loss | `rl/objective.py` |
| Algorithm 1 | `core/engine.py::Engine.run_step` |

## Decisions the paper leaves open

Each is a config knob, defaulted to the reading argued in the code comments.

- **`search.selection_mode`** (`node`). §2.1 defines an outcome only for a
  selected virtual action (1 child) and a selected leaf (k children), leaving a
  selected *internal* child undefined. In `node` mode every node contributes
  exactly one target — leaves deepen, nodes with children offer their virtual
  action — so an internal node is never itself a target and the case cannot
  arise. `action_descend` instead walks down through internal children.
- **`search.virtual_value_mode`** (`zero`). Eq. 6 does not define `V(p, ŝ)`;
  the text says the score is "driven by the prior and the exploration bonus".
- **`elo.scale`** (`400.0`). The paper writes `p_ij = (1 + 10^(E_j − E_i))^{-1}`,
  i.e. scale 1.0 — about 400× steeper than conventional Elo, where a one-point
  gap already means `p = 0.09`. Set `1.0` for the literal equation.
- **`search.normalize_exploitation`** (`true`). `V = W_m` is in raw reward units
  while the prior is a probability, so `c` does not transfer across problems
  unless `W_m` is rescaled. Set `false` for the literal Eq. 6.
- **`rl.advantage_clip`** (`10.0`) and **`rl.skip_degenerate_batches`** (`true`)
  are guard rails, not method: `A_fb` is a log-prob difference and unbounded
  below, and β-tilting identical rewards yields exactly zero advantage.

## Ablations

```bash
RL_ENABLED=false bash run.sh              # search + memory only
MEMORY_ENABLED=false bash run.sh          # no lesson bank
ELO_ENABLED=false ALPHA=1.0 bash run.sh   # rank signal only (Eq. 3)
ALPHA=0.0 bash run.sh                     # Elo signal only
BETA=20 bash run.sh                       # near-max-seeking RL
LAMBDA_VIRTUAL=0 bash run.sh              # no optimism on unseen siblings
```

## Adding an example

Create `examples/<name>/` with a `config.yaml` and an `env.py` exposing
`build(cfg) -> Example`. Implement `meta_description`, `instruction`,
`preprocess` and `score`; read every knob from `example.params`. No framework
file changes — `core/registry.py` resolves examples by import path.

## Output

```
runs/<example>_<n>_<model>_<time>/
  config.json          resolved config
  provenance.txt       which layer set each key
  tree.json            search tree + archive
  memory.json          the lesson bank
  step00/
    step00_target00_rollout000.txt        raw response
    step00_target00_rollout000.meta.json  reward, valid, feedback
    step00.summary.json
    step00_elo/                           judge prompts, replies, standings
  best_code.py  final.summary.json
```

Every rollout is written, including failures — they are the input to the
negative-lesson extractor and to Eq. 9, so a log that kept only successes would
hide half the training signal.
