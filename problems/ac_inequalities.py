"""
Autocorrelation Inequalities (AC1 and AC2).

  - problem_type "ac1": entrypoint propose_candidate, minimize upper bound,
                        reward = 1/result
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
    AC1_LITERATURE,
    AC2_LITERATURE,
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


class ACInequalities(Problem):
    two_stage_rollouts = True
    saves_construction = True
    # A malformed AC answer commonly ends after a long mathematical
    # derivation.  Give the recovery turn that work so it completes the
    # program instead of repeating the derivation from the beginning.
    coder_retry_uses_failed_response = True
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
            self._eval_src = inspect.getsource(evaluate_sequence_ac1)
        else:
            self.entrypoint = "construct_function"
            self.metric_name = "lower bound"
            self.maximize = True
            if self.target is None:
                self.target = 0.97
            self._evaluate = evaluate_sequence_ac2
            self._eval_src = inspect.getsource(evaluate_sequence_ac2)

    # ------------------------------------------------------------------
    def _state_ctx(self, parent: ParentContext) -> str:
        ctx = render_state_context(self.metric_name, self.target, parent,
                                   maximize=self.maximize)
        if parent.construction is not None and len(parent.construction) > 0:
            ctx += f"\nLength of the construction: {len(parent.construction)}"
        return ctx

    @staticmethod
    def _append_user_instruction(messages: List[dict], instruction: str
                                 ) -> List[dict]:
        staged = [dict(message) for message in messages]
        if staged and staged[-1].get("role") == "user":
            staged[-1]["content"] = (
                str(staged[-1].get("content", "")).rstrip()
                + "\n\n" + instruction.strip() + "\n"
            )
        else:
            staged.append({"role": "user", "content": instruction.strip()})
        return staged

    def build_strategy_messages(
            self, messages: List[dict],
            previous_strategies: List[str] | None = None) -> List[dict]:
        common = '''
Autocorrelation-specific implementation constraints:
- If a current construction exists, `height_sequence_1` is already supplied
  to the executed program as a one-dimensional NumPy array. Never copy,
  serialize, reconstruct, or hard-code that array. Plan a compact algorithm
  that reads it directly. Never propose a long explicit numeric table or an
  unrolled pattern.
- The entrypoint must return a non-empty Python `list` of finite real numbers.
  Keep the returned coefficients explicitly nonnegative and no larger than
  1000; do not depend on the evaluator's defensive clipping to repair a search
  candidate. Retain a separately copied, exactly re-evaluated best candidate
  and a concrete valid fallback throughout the run.
- The authoritative evaluator uses full *linear* self-convolution, producing
  2*n-1 coefficients. It is not circular convolution. Derive every surrogate,
  gradient, or convolution index against that definition and accept candidates
  only by calling the supplied exact `evaluate_sequence`.
- Positive rescaling below the coefficient cap leaves the objective unchanged.
  Do not claim that changing scale alone is an improvement, and do not let
  normalization interact with clipping in a way that changes the candidate.
- Products of the submitted coefficients make the search generally nonconvex.
  A direct CVXPY call is not a solution unless the stated subproblem satisfies
  the solver's rules. Explain any relaxation or alternating subproblem, how a
  candidate is recovered, and how it is checked by the original evaluator.
- The supplied parents can contain several thousand coefficients. Never form a
  dense n-by-n identity, Toeplitz/convolution operator, Hessian, or constraint
  matrix, and never estimate a gradient with one evaluator call per
  coordinate. Those constructions have quadratic memory or evaluation count
  and are not feasible here. An LP is acceptable only with a genuinely sparse
  or matrix-free operator whose storage and solver behavior you spell out.
- The exact evaluator itself performs the authoritative full convolution, so
  exact acceptance calls are the scarce resource. Use cheap vectorized or FFT
  surrogates to propose candidates if useful, then spend a bounded number of
  exact calls on acceptance. Adapt that count from measured elapsed time and
  retain enough headroom to return the fallback.
- Derive dimensions from runtime inputs. If changing resolution, specify the
  exact interpolation/resampling rule, normalization, validity checks, and how
  the new candidate is compared with the parent. Explain initialization when
  no parent is available.'''
        if self.problem_type == "ac1":
            mathematical = '''
AC1 mathematical checks for the final plan:
- For a returned sequence a of length n, the exact lower-is-better objective is
  2*n*max(convolve(a, a, mode="full"))/sum(a)^2, with sum(a) >= 0.01 after
  defensive clipping. Include every one of the 2*n-1 convolution positions.
- Any normalization or LP direction must preserve nonnegativity and must be
  evaluated with that exact ratio. Do not infer monotonic improvement from a
  surrogate bound without an exact acceptance comparison.'''
        else:
            mathematical = '''
AC2 mathematical checks for the final plan:
- Let c=convolve(a,a,mode="full") have m=2*n-1 entries and let
  y=(0,c_0,...,c_{m-1},0). The evaluator uses uniform spacing
  dx=1/(m+1), the piecewise-linear integral
  L2sq=sum_i dx/3*(y_i^2+y_i*y_{i+1}+y_{i+1}^2),
  L1=sum(abs(c))/(m+1), and Linf=max(abs(c)). The exact higher-is-better
  objective is L2sq/(L1*Linf).
- Keep L1 and Linf strictly positive and use the same endpoint convention and
  spacing in any analytic gradient or surrogate. Re-evaluate every accepted
  candidate with the supplied function rather than trusting an internal score.'''
        staged = self._append_user_instruction(messages, common + mathematical)
        return super().build_strategy_messages(
            staged, previous_strategies=previous_strategies)

    def build_code_messages(self, messages: List[dict],
                            strategy: str) -> List[dict]:
        staged = super().build_code_messages(messages, strategy)
        entrypoint = ("propose_candidate" if self.problem_type == "ac1"
                      else "construct_function")
        return self._append_user_instruction(staged, f'''
## Mandatory compact-source and return contract

- `height_sequence_1` is runtime input when a parent construction exists.
  Reference it directly; NEVER assign a pasted copy of its values, serialize
  it, or reconstruct it as a literal. If no parent was announced, create a
  starting vector under a different name.
- Do not emit any long numeric literal, lookup table, repeated-value list, or
  unrolled pattern. A list/tuple/array literal may contain at most 32 scalar
  values. Build every larger vector algorithmically with NumPy, a compact
  formula, interpolation, tiling, or a loop.
- Keep the source concise: at most 300 lines and about 12,000 characters.
  Finish `{entrypoint}`, return a non-empty Python list of finite coefficients,
  and close the single Python fence well before the generation limit.
- Use the supplied `evaluate_sequence` for final acceptance. Preserve a valid
  best-so-far fallback and return it before the configured deadline.
- The expert plan has already done the mathematical design. Do not spend the
  response re-deriving the problem on toy lengths or narrating alternatives.
  Resolve only implementation details in private reasoning, then emit the
  complete program. A simple executable implementation with the parent as a
  fallback is preferable to an ambitious program that is never finished.

These are hard output constraints even if the proposed strategy conflicts
with them.''')

    def build_prompt(self, parent: ParentContext) -> List[dict]:
        state_ctx = self._state_ctx(parent)
        budget_s = self.budget_s
        if self.problem_type == "ac1":
            direction = "minimize"
            metric = "the AC1 upper bound"
            entrypoint = "propose_candidate"
            literature = AC1_LITERATURE
        else:
            direction = "maximize"
            metric = "the AC2 lower bound"
            entrypoint = "construct_function"
            literature = AC2_LITERATURE

        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = f'''
The current construction is already available to the executed program as the
one-dimensional NumPy array `height_sequence_1` (length
{len(parent.construction)}). Use it directly as one possible initialization;
do not copy its values into the response. You may also explore genuinely
different compactly generated initializations.'''

        direct_output_section = ""
        if not self.two_stage_rollouts:
            direct_output_section = f'''
## Required output

Return exactly one complete fenced Python code block, beginning with
```python and ending with ```, and nothing else. The extracted block must
define `{entrypoint}` at top level. Do not output a strategy, analysis, prose,
notes, example usage, a second code block, or a partial snippet. Do not embed
the parent construction as data. Finish the program and close the fence well
before the response-token limit.'''

        correction_instruction = (
            "The downstream coder must treat a strategy as guidance and "
            "correct any conflict with the evaluator or interface."
            if self.two_stage_rollouts else
            "The returned program must follow the evaluator and interface "
            "even if an exploratory idea conflicts with them."
        )

        user = f'''You are an expert in harmonic analysis, autocorrelation
inequalities, and numerical optimization. Write a compact search program whose
returned nonnegative step-function heights {direction} {metric}.

## Authoritative evaluator

The evaluator below is read-only and is the only definition of the score:
```python
{self._eval_src}
```

{literature}

## Runtime contract and resources

- Define `{entrypoint}(seed={self.seed}, budget_s={budget_s}, **kwargs)`.
- The harness invokes `{entrypoint}()` with no arguments, so those defaults are
  active. Return a non-empty Python list of finite real coefficients.
- `evaluate_sequence` is pre-imported and may be called during search. Do not
  redefine or import it.
- Time budget: {budget_s} seconds. Check the deadline inside expensive loops
  and solver calls, preserve the best exact-scoring valid candidate, and return
  it with headroom before the hard sandbox timeout.
- Candidate failure receives the configured failure reward; do not hide an
  error by returning an empty or non-finite sequence.
- You may use NumPy, SciPy, CVXPY, math, and the Python standard library. Use
  at most {self.eval_cpus} CPUs.
- Make helper functions top-level. Do not use nested-function closures or
  lambda functions. Do not use filesystem or network I/O.
- Construct large arrays algorithmically. Do not paste long numeric literals,
  lookup tables, repeated-value lists, or unrolled patterns.
- Runtime parent lengths are commonly in the thousands. Do not allocate dense
  n-by-n matrices (including `np.eye(n)`), dense convolution/Toeplitz
  operators, or use one finite-difference evaluation per coordinate. Generated
  programs must keep auxiliary storage linear or near-linear in n. If using a
  solver, use a truly sparse/matrix-free formulation and set an internal stop
  condition that leaves time to return the saved fallback.

{state_ctx}
{construction_section}

Reason from the exact evaluator and improve meaningfully over the supplied
parent/search program when one exists. {correction_instruction}
{direct_output_section}
'''
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
        if not isinstance(output, list) or not output:
            res.msg = "bad_return_sequence"
            res.failure_kind = "code"
            return res
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not np.isfinite(x) for x in output):
            res.msg = "bad_return_values"
            res.failure_kind = "code"
            return res
        # AC entrypoints return only the construction, not a model-reported
        # score. Unlike Erdos there is therefore no claimed-vs-computed scalar
        # and no numerical comparison tolerance to tune: independently run the
        # authoritative evaluator once and trust only that result.
        try:
            result = float(self._evaluate(output))
        except Exception as exc:
            res.msg = f"invalid_solution: {exc}"
            res.failure_kind = "constraint"
            return res
        if not np.isfinite(result) or result <= 0.0:
            res.msg = f"invalid_score: {result}"
            res.failure_kind = "constraint"
            return res
        res.valid = True
        res.raw_score = result
        res.construction = list(output)
        res.msg = f"Success; raw_score={result}"
        if self.problem_type == "ac1":
            res.reward = float(1.0 / result)
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
                value = float(1.0 / raw)
                code = example_ae_program_random_init(self.budget_s)
            else:
                raw = float(evaluate_sequence_ac2(construction))
                value = raw
                code = thetaevolve_initial_program_prev_init
            seeds.append(SeedState(code=code, value=value, raw_score=raw,
                                   construction=construction))
        return seeds
