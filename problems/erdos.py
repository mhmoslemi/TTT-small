"""
Erdos' Minimum Overlap Problem.

Two changes from the original:

  build_prompt takes `memory` and places the retrieved lessons between the
  parent state and the instruction, and adapts the instruction when they are
  present, rather than having the trainer staple the block onto the end.

  The compute budget is config-driven. The prompt used to hardcode
  "budget_s=1000", and the sandbox calls run() with NO arguments, so that
  default is what actually executes: every rollout may burn 1000 seconds of
  optimization. At 512 rollouts a step that is 4 to 70 hours depending on how
  many evaluations run in parallel. `budget_s` in the problem config now sets
  both the number in the prompt and the sandbox ceiling, so the two cannot
  drift apart.
"""

from __future__ import annotations
import inspect
from typing import Any, List
import numpy as np
from problems.base import (
    Problem, ParentContext, RewardResult, SeedState, render_state_context,
)


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


_VERIFIER_SRC = (
    "import numpy as np\n\n"
    + inspect.getsource(verify_c5_solution) + "\n\n"
    + inspect.getsource(evaluate_erdos_solution) + "\n\n"
)


class ErdosMinOverlap(Problem):
    name = "erdos"
    entrypoint = "run"
    metric_name = "C\u2085 bound"
    maximize = False

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.3808
        # What the model is told it has, and what the sandbox actually allows.
        # The sandbox gets headroom so a program that respects its budget is
        # killed by its own clock rather than by the harness, which produces a
        # returned best-so-far instead of a lost rollout.
        self.budget_s = float(cfg.get("budget_s", 60.0))
        self.n_cpus = int(cfg.get("eval_cpus", 2))

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)

        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(parent.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        memory_section = ""
        if memory and memory.strip():
            memory_section = f"""
## Lessons from earlier attempts at this problem

Extracted from programs already generated and evaluated in this same search.
Empirical findings, not part of the specification above, and they do not
override any constraint stated in it.

{memory.strip()}
"""

        if memory_section:
            code_section = '''Work through the lessons above before writing anything:
- Which bear on the construction you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. Some will be wrong or
  irrelevant for this state.
- Is anything they recommend already in the algorithm above and still not
  improving the bound? Then that avenue is spent and the gain is elsewhere.

Then reason about how to improve the construction. Aim for something different
from the algorithm above: a different algorithmic idea, different heuristics, a
different parameterization or sweep. A lesson gives you an idea; you choose the
implementation, and you should not copy any expression from one verbatim.
Unless you make a meaningful improvement, you will not be rewarded.'''
        elif parent.code and parent.code.strip():
            code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.'''
        else:
            code_section = '''Write code to optimize this construction.'''

        user = f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {self.budget_s:.0f}s for your code to run
- **CPUs**: {self.n_cpus} available

## Rules
- Define `run(seed=42, budget_s={self.budget_s:.0f}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- It is called with NO arguments, so your default for `budget_s` is the one that
  runs. Respect it: track elapsed time and return your best solution before it
  expires, rather than being killed with nothing to show
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}{memory_section}
{code_section}
'''
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = _VERIFIER_SRC
        if parent.construction is not None:
            prelude += f"initial_h_values = np.array({list(parent.construction)!r})\n\n"
        return prelude + "# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not verify_erdos_solution(output):
            res.msg = "Invalid solution."
            return res
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        res.valid = True
        res.raw_score = float(c5_bound)
        res.reward = float(1.0 / (1e-8 + c5_bound))
        res.construction = list(np.asarray(h_values).ravel())
        res.msg = f"C5 bound: {c5_bound}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        seeds: List[SeedState] = []
        for i in range(self.num_seed_states):
            rng = np.random.default_rng(self.seed + i)
            n_points = int(rng.integers(40, 100))
            construction = np.ones(n_points) * 0.5
            perturbation = rng.uniform(-0.4, 0.4, n_points)
            perturbation = perturbation - np.mean(perturbation)
            construction = construction + perturbation
            dx = 2.0 / n_points
            correlation = np.correlate(construction, 1 - construction, mode="full") * dx
            c5_bound = float(np.max(correlation))
            seeds.append(SeedState(
                code="",
                value=float(1.0 / (1e-8 + c5_bound)),
                raw_score=c5_bound,
                construction=list(construction),
            ))
        return seeds