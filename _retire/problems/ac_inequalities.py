"""
Autocorrelation Inequalities (AC1 and AC2).

  - problem_type "ac1": entrypoint propose_candidate, minimize upper bound,
                        reward = 1/result
  - problem_type "ac2": entrypoint construct_function, maximize lower bound,
                        reward = result

"""

from __future__ import annotations

import inspect
import re
from typing import Any, List

import numpy as np

from _retire.problems.base import (
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


def _ac_bytelevel_decoder_table():
    """Inverse GPT-2/ByteLevel alphabet used only for AC plan recovery."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {
        chr(character): value
        for value, character in zip(byte_values, unicode_values)
    }


_AC_BYTELEVEL_DECODE = _ac_bytelevel_decoder_table()
_AC_BYTELEVEL_MARKERS = ("\u0120", "\u010a", "\u0109", "\u010d")


def _repair_ac_bytelevel_surface(text: str) -> str:
    """Repair a serialized byte-token surface for AC strategy text only."""
    original = str(text or "")
    if not any(marker in original for marker in _AC_BYTELEVEL_MARKERS):
        return original
    payload = bytearray()
    for character in original:
        value = _AC_BYTELEVEL_DECODE.get(character)
        if value is None:
            payload.extend(character.encode("utf-8"))
        else:
            payload.append(value)
    return payload.decode("utf-8", errors="replace")

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
Autocorrelation-specific scientific and execution requirements:
- Write the final plan as normal, readable English prose with ordinary spaces
  and line breaks. Never emit serialized tokenizer surfaces, escaped token
  streams, tool/backend error messages, execution traces, or mixed-language
  fragments. Before closing the block, confirm that all six requested sections
  are present in order and readable independently of the reasoning transcript.
- If a current construction exists, `height_sequence_1` is already supplied
  to the executed program as a one-dimensional NumPy array. Never copy,
  serialize, reconstruct, or hard-code that array. Read it directly as a
  possible initialization; this does not require retaining its resolution
  or construction family.
- The entrypoint must return a non-empty Python `list` of finite real numbers.
  Keep the returned coefficients explicitly nonnegative and no larger than
  1000; do not depend on the evaluator's defensive clipping to repair a search
  candidate. Retain a separately copied, exactly re-evaluated best candidate
  and a concrete valid fallback throughout the run.
- The authoritative evaluator uses full *linear* self-convolution, producing
  2*n-1 coefficients. It is not circular convolution. Derive every surrogate,
  gradient, or convolution index against that definition. Verify updates to
  the saved best candidate and the final return with `evaluate_sequence`.
  Internal search iterates may worsen or use surrogates; do not mistake their
  surrogate scores for verified improvements.
- Positive rescaling below the coefficient cap leaves the objective unchanged.
  Do not claim that changing scale alone is an improvement, and do not let
  normalization interact with clipping in a way that changes the candidate.
- Products of the submitted coefficients make the search generally nonconvex.
  A direct CVXPY call is not a solution unless the stated subproblem satisfies
  the solver's rules. Explain any relaxation or alternating subproblem, how a
  candidate is recovered, and how it is checked by the original evaluator.
- Choose algorithms for the actual problem size and execution resources.
  Account for arrays, temporary copies, solver workspace, and evaluation cost.
  Dense operators, finite differences, and LP methods are options when their
  costs fit; for larger searches, consider structured, sparse, matrix-free,
  FFT, analytic-gradient, or reduced-subproblem approaches as appropriate.
  Explain the chosen formulation's scaling, rather than assuming a library
  call makes a large allocation or solve affordable.
- Schedule expensive evaluations and solver calls using elapsed time inside
  the generated program. Surrogates may guide exploration, but leave time for
  exact validation and returning the saved best valid candidate. Do not impose
  an arbitrary number of evaluations independent of the actual runtime.
- Sequence length and construction family are search choices, not fixed by
  the parent, seed program, or literature example. Derive dimensions from the
  candidate being evaluated. If changing resolution, explain the construction
  or resampling rule, normalization, validity checks, and exact comparison.
  Explain initialization when no parent is available. The coefficient cap of
  1000 is a value bound, not a limit on the number of coefficients.'''
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
  spacing in any analytic gradient or surrogate. Verify the saved best and
  final candidate with the supplied function rather than trusting an internal
  score.'''
        staged = self._append_user_instruction(messages, common + mathematical)
        return super().build_strategy_messages(
            staged, previous_strategies=previous_strategies)

    def strategy_output_issue(self, strategy: str) -> str | None:
        """Reject corrupted AC plans before they reach another model.

        DeepSeek/Qwen occasionally returns a syntactically valid strategy block
        whose prose is still a byte-token surface or a collapsed backend-error
        fragment.  Those strings are not useful exploration: the coder cannot
        reliably interpret them.  This gate is deliberately AC-only and checks
        text integrity rather than mathematical quality.
        """
        text = str(strategy or "")
        byte_markers = sum(text.count(marker)
                           for marker in ("\u0120", "\u010a", "\u0109", "\u010d"))
        if byte_markers or "\ufffd" in text:
            return "AC strategy contained tokenizer-decoding artifacts"
        if re.search(r"\S{240,}", text):
            return "AC strategy contained collapsed or unreadable text"
        if re.search(
                r"\bErrorMessage\s*:|\bError during execution\b|"
                r"\bChatCoder\b|/opt/[^\s]+/project\b",
                text, flags=re.IGNORECASE):
            return "AC strategy contained backend error text"
        return None

    def normalize_strategy_response(self, response: str) -> str:
        """Repair byte-token surfaces only for AC strategy generation."""
        return _repair_ac_bytelevel_surface(response)


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

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable.\x20
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.\x20
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to {self.eval_cpus} CPUs.
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

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable.\x20
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.\x20
Unless you make a meaningful improvement, you will not be rewarded, if you are stuck you should think about how to get unstuck.

Rules:
- You must define the `construct_function` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to {self.eval_cpus} CPUs.
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
