"""
Autocorrelation Inequalities (AC1 and AC2).

  - problem_type "ac1": entrypoint propose_candidate, minimize upper bound,
                        reward = 1/(1e-8 + result)
  - problem_type "ac2": entrypoint construct_function, maximize lower bound,
                        reward = result

"""

from __future__ import annotations

import inspect
from typing import Any, List

import numpy as np

from problems.base import (
    Problem, ParentContext, RewardResult, SeedState, render_state_context,
)

from examples.ac_inequalities.prompt import (
    AC1_EVAL_FUNCTION,
    AC1_LITERATURE,
    AC2_LITERATURE,
    ae_verifier_program,
    example_ae_program_random_init,
    thetaevolve_initial_program_prev_init,
)

def evaluate_sequence(sequence: list) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    if not isinstance(sequence, list):
        return np.inf
    if not sequence:
        return np.inf
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]
    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return np.inf
    return float(2 * n * max_b / (sum_a**2))


evaluate_sequence_ac1 = evaluate_sequence


def evaluate_sequence(sequence: list) -> float:
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")
    if not sequence:
        raise ValueError("Empty sequence")
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


evaluate_sequence_ac2 = evaluate_sequence


def verify_ac1_solution(result) -> bool:
    try:
        value = evaluate_sequence_ac1(result)
        if value == np.inf:
            return False
    except Exception:
        return False
    return True


def verify_ac2_solution(result) -> bool:
    try:
        value = evaluate_sequence_ac2(result)
        if value == np.inf:
            return False
    except Exception:
        return False
    return True


class ACInequalities(Problem):
    name = "ac_inequalities"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.problem_type = str(cfg.get("problem_type", "ac1")).lower()
        if self.problem_type not in ("ac1", "ac2"):
            raise ValueError(f"problem_type must be 'ac1' or 'ac2', got {self.problem_type}")
        self.budget_s = int(cfg.get("budget_s", 1000))

        if self.problem_type == "ac1":
            self.entrypoint = "propose_candidate"
            self.metric_name = "upper bound"
            self.maximize = False
            if self.target is None:
                self.target = 1.5030
            self._evaluate = evaluate_sequence_ac1
            self._verify = verify_ac1_solution
            self._eval_src = inspect.getsource(evaluate_sequence_ac1)
        else:
            self.entrypoint = "construct_function"
            self.metric_name = "lower bound"
            self.maximize = True
            if self.target is None:
                self.target = 0.97
            self._evaluate = evaluate_sequence_ac2
            self._verify = verify_ac2_solution
            self._eval_src = inspect.getsource(evaluate_sequence_ac2)

    # ------------------------------------------------------------------
    def _state_ctx(self, parent: ParentContext) -> str:
        ctx = render_state_context(self.metric_name, self.target, parent,
                                   maximize=self.maximize)
        if parent.construction is not None and len(parent.construction) > 0:
            ctx += f"\nLength of the construction: {len(parent.construction)}"
        return ctx

    def build_prompt(self, parent: ParentContext) -> List[dict]:
        state_ctx = self._state_ctx(parent)
        budget_s = self.budget_s

        if self.problem_type == "ac1":
            user = f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

{AC1_EVAL_FUNCTION}

{AC1_LITERATURE}

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with {budget_s}s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{state_ctx}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''
        else:
            user = f''''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step functions, that maximizes the following evaluation function:

```python
{ae_verifier_program}
```

{AC2_LITERATURE}
Your task is to write a search function, construct_function(), that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with {budget_s}s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{state_ctx}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded, if you are stuck you should think about how to get unstuck.

Rules:
- You must define the `construct_function` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked. Do not import height_sequence_1 yourself; it will already be available.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = "import numpy as np\n\n" + self._eval_src + "\n\n"
        if parent.construction is not None:
            prelude += f"height_sequence_1 = np.array({list(parent.construction)!r})\n\n"
        return prelude + "# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not self._verify(output):
            res.msg = "Invalid solution."
            return res
        result = float(self._evaluate(output))
        res.valid = True
        res.raw_score = result
        res.construction = list(output)
        res.msg = f"Success; raw_score={result}"
        if self.problem_type == "ac1":
            res.reward = float(1.0 / (1e-8 + result))
        else:
            res.reward = result
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        seeds: List[SeedState] = []
        for i in range(self.num_seed_states):
            rng = np.random.default_rng(self.seed + i)
            construction = [float(rng.random())] * int(rng.integers(1000, 8000))
            if self.problem_type == "ac1":
                raw = float(evaluate_sequence_ac1(construction))
                value = float(1.0 / (1e-8 + raw))
                code = example_ae_program_random_init(self.budget_s)
            else:
                raw = float(evaluate_sequence_ac2(construction))
                value = raw
                code = thetaevolve_initial_program_prev_init
            seeds.append(SeedState(code=code, value=value, raw_score=raw,
                                   construction=construction))
        return seeds
