"""Fourier sign-uncertainty (C4) Hermite-coefficient problem.

This implements the first AlphaEvolve/Goncalves Hermite formulation.  A
candidate returns the free coefficients c_0, ..., c_{m-1}.  The verifier adds
c_m so that

    Q(z) = sum_{k=0}^m c_k H_{4k}(z)

satisfies Q(0) = 0, orients Q to be positive at infinity, removes the forced
z**2 factor, and finds the largest *sign-changing* positive real root.  The
certified construction score is r_max**2 / (2*pi), and lower is better.

All polynomial construction and root multiplicities are computed over exact
rationals with SymPy.  In particular, an even-multiplicity root is not counted
as a sign change; this is the material distinction missing from evaluators
that simply select the largest positive root.
"""

from __future__ import annotations

import inspect
import math
from numbers import Real
from typing import Any, List

import numpy as np
import sympy as sp

from problems.base import (
    ParentContext,
    Problem,
    RewardResult,
    SeedState,
    render_state_context,
)


_C4_X = sp.symbols("x")

# Published free coefficients.  In both cases the verifier derives one more
# coefficient (of H_12) to impose Q(0) = 0.
_GONCALVES_FREE_COEFFICIENTS = (
    -113.0 / 100.0,
    1.0 / 25.0,
    1.0 / 3240.0,
)
_ALPHAEVOLVE_FREE_COEFFICIENTS = (
    0.3292519302257546,
    -0.01158510802599293,
    -8.921606035407065e-05,
)


def _coerce_c4_coefficients(coefficients: Any, max_coeff_count: int) -> np.ndarray:
    """Return a finite, nonzero, one-dimensional float coefficient vector."""
    if isinstance(coefficients, np.ndarray):
        raw = coefficients
    elif isinstance(coefficients, (list, tuple)):
        raw = np.asarray(coefficients)
    else:
        raise ValueError("run() must return a list, tuple, or numpy array")

    if raw.ndim != 1:
        raise ValueError(f"coefficients must be one-dimensional, got shape {raw.shape}")
    if raw.size < 1:
        raise ValueError("coefficient list must not be empty")
    if raw.size > max_coeff_count:
        raise ValueError(
            f"too many free coefficients: {raw.size} > {max_coeff_count}"
        )

    values = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise ValueError("every coefficient must be a real numeric scalar")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("coefficients must not contain NaN or infinity")
        values.append(converted)

    result = np.asarray(values, dtype=np.float64)
    if not np.any(result != 0.0):
        raise ValueError("coefficients must not all be zero")
    return result


def _hermite_4k_polynomials(count: int) -> list[sp.Expr]:
    """Return physicists' Hermite polynomials H_0, H_4, ..., H_{4(count-1)}."""
    return [
        sp.polys.orthopolys.hermite_poly(n=4 * k, x=_C4_X, polys=False)
        for k in range(count)
    ]


def _construct_c4_polynomial(
    coefficients: Any, max_coeff_count: int
) -> tuple[np.ndarray, sp.Poly]:
    """Construct the exactly rational, positively oriented Hermite polynomial."""
    coeffs = _coerce_c4_coefficients(coefficients, max_coeff_count)
    rational_coeffs = [sp.Rational(str(float(value))) for value in coeffs]

    # There are m submitted/free coefficients and one dependent coefficient.
    m = len(rational_coeffs)
    hermites = _hermite_4k_polynomials(m + 1)
    partial = sp.Add(
        *(rational_coeffs[k] * hermites[k] for k in range(m))
    )
    value_of_last_basis_at_zero = hermites[m].subs(_C4_X, 0)
    if value_of_last_basis_at_zero == 0:
        raise ValueError(f"H_{4 * m}(0) unexpectedly equals zero")
    last_coefficient = (
        -partial.subs(_C4_X, 0) / value_of_last_basis_at_zero
    )

    polynomial = sp.Poly(
        sp.expand(partial + last_coefficient * hermites[m]),
        _C4_X,
        domain=sp.QQ,
    )
    if polynomial.is_zero:
        raise ValueError("constructed polynomial is identically zero")
    if polynomial.LC() < 0:
        polynomial = -polynomial
    if polynomial.eval(0) != 0:
        raise ValueError("internal error: constructed polynomial does not vanish at zero")
    return coeffs, polynomial


def _largest_positive_sign_changing_root(polynomial: sp.Poly) -> sp.Expr:
    """Return the largest positive root with odd multiplicity, exactly.

    For a real polynomial, crossing a real root changes its sign if and only if
    that root has odd multiplicity.  Using exact multiplicities avoids the
    epsilon-probing ambiguity of a floating-point sign test.
    """
    x_squared = sp.Poly(_C4_X**2, _C4_X, domain=sp.QQ)
    quotient, remainder = polynomial.div(x_squared)
    if not remainder.is_zero:
        raise ValueError("constructed polynomial is not exactly divisible by x^2")
    if quotient.is_zero:
        raise ValueError("P(x)/x^2 is identically zero")

    roots_with_multiplicity = quotient.real_roots(
        multiple=False,
        radicals=False,
    )
    sign_changing_positive_roots = []
    for root, multiplicity in roots_with_multiplicity:
        if int(multiplicity) % 2 == 0:
            continue
        if root.is_positive is True:
            sign_changing_positive_roots.append(root)
            continue
        if root.is_zero is True or root.is_negative is True:
            continue
        # real_roots() returns certified real algebraic roots.  This fallback
        # is only for a root whose assumptions do not expose its sign.
        if float(sp.N(root, 100)) > 0.0:
            sign_changing_positive_roots.append(root)

    if not sign_changing_positive_roots:
        raise ValueError("P(x)/x^2 has no positive sign-changing real root")

    # SymPy documents real_roots() as sorted in increasing real order.  The
    # filtered list therefore remains sorted, without a floating comparison.
    return sign_changing_positive_roots[-1]


def compute_c4_bound(
    coefficients: Any, max_coeff_count: int = 8
) -> tuple[float, float]:
    """Recompute the Hermite construction's C4 upper bound and r_max."""
    _, polynomial = _construct_c4_polynomial(coefficients, max_coeff_count)
    exact_r_max = _largest_positive_sign_changing_root(polynomial)
    c4_bound = float(sp.N(exact_r_max**2 / (2 * sp.pi), 80))
    r_max = float(sp.N(exact_r_max, 80))
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("computed r_max is not positive and finite")
    if not math.isfinite(c4_bound) or c4_bound <= 0.0:
        raise ValueError("computed C4 bound is not positive and finite")
    return c4_bound, r_max


def evaluate_c4_coefficients(
    coefficients: Any, max_coeff_count: int = 8
) -> float:
    """Return the authoritative lower-is-better C4 upper bound."""
    return compute_c4_bound(coefficients, max_coeff_count)[0]


_C4_VERIFIER_FUNCTIONS = (
    _coerce_c4_coefficients,
    _hermite_4k_polynomials,
    _construct_c4_polynomial,
    _largest_positive_sign_changing_root,
    compute_c4_bound,
)


def _verifier_source(max_coeff_count: int) -> str:
    source = (
        "import math\n"
        "from numbers import Real\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "import sympy as sp\n\n"
        "_C4_X = sp.symbols('x')\n\n"
    )
    source += "\n\n".join(
        inspect.getsource(function) for function in _C4_VERIFIER_FUNCTIONS
    )
    source += (
        "\n\n"
        f"C4_MAX_COEFF_COUNT = {int(max_coeff_count)}\n\n"
        "def evaluate_c4_coefficients(coefficients):\n"
        "    return compute_c4_bound(\n"
        "        coefficients, C4_MAX_COEFF_COUNT\n"
        "    )[0]\n\n"
    )
    return source


class ErdosC4Uncertainty(Problem):
    """TTT problem wrapper for the exact Hermite C4 construction."""

    name = "erdos-c4"
    entrypoint = "run"
    metric_name = "C₄ upper bound"
    maximize = False
    saves_construction = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.budget_s = float(cfg.get("budget_s", 280.0))
        self.max_coeff_count = int(cfg.get("max_coeff_count", 8))
        self.benchmark_c4 = float(
            cfg.get("benchmark_c4", 0.3215872333529007)
        )
        if self.budget_s <= 0.0 or not math.isfinite(self.budget_s):
            raise ValueError("budget_s must be positive and finite")
        if self.max_coeff_count < 3:
            raise ValueError("max_coeff_count must be at least 3")
        if self.benchmark_c4 <= 0.0 or not math.isfinite(self.benchmark_c4):
            raise ValueError("benchmark_c4 must be positive and finite")
        if self.target is None:
            self.target = self.benchmark_c4

    def build_prompt(
        self,
        parent: ParentContext,
        memory: str = "",
        memory_protocol: bool = False,
    ) -> List[dict]:
        state_ctx = render_state_context(
            self.metric_name,
            self.target,
            parent,
            maximize=self.maximize,
        )

        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = f"""
The current construction is available as the one-dimensional NumPy array
`initial_coefficients`. It contains {len(parent.construction)} free
coefficients. You may change both their values and their count, but the returned
count must be between 1 and {self.max_coeff_count}.
"""

        memory_section = ""
        if memory_protocol:
            candidate = (
                (memory or "").strip()
                or "(No memory hypothesis was assigned to this control arm.)"
            )
            memory_section = f"""
## Candidate hypotheses from earlier attempts

These are unconfirmed hypotheses extracted from programs generated and
evaluated in this search. They may be wrong or irrelevant and do not override
the mathematical specification.

{candidate}
"""
        elif memory and memory.strip():
            memory_section = f"""
## Lessons from earlier attempts

These are empirical observations from this search, not part of the
specification, and they may be wrong or irrelevant.

{memory.strip()}
"""

        if memory_protocol:
            instruction = """Review the assigned hypothesis, if any, and decide
whether it applies to this construction. Then produce a meaningfully improved
search algorithm; do not copy a lesson expression verbatim."""
        elif memory_section:
            instruction = """Assess which lessons actually apply, then produce
a meaningfully improved search algorithm. Treat already-tried, unsuccessful
ideas as spent."""
        elif parent.code and parent.code.strip():
            instruction = """Reason about how to improve the previous search
algorithm through a different parameterization, optimizer, initialization,
precision strategy, or exploration schedule."""
        else:
            instruction = "Write code to search for the best coefficient vector."

        user = f'''You are an expert in harmonic analysis, exact polynomial
arithmetic, and numerical optimization. Find the smallest verified upper bound
on the Fourier sign-uncertainty constant C₄ using the Hermite construction
below.

## Construction

For m submitted free coefficients c₀, ..., c_{{m-1}}, the fixed evaluator
forms

    Q(z) = sum_{{k=0}}^m c_k H_{{4k}}(z),

where H_n is the physicists' Hermite polynomial. The evaluator, not your code,
chooses the final coefficient c_m exactly so that Q(0) = 0 and flips the whole
polynomial when needed so that it is positive at infinity.

It then removes the forced z² factor and finds r_max, the largest positive real
root across which Q(z)/z² changes sign. Roots of even multiplicity do not count.
The reported upper bound is

    U(c) = r_max² / (2π).

Lower values are better. Overall coefficient scale is irrelevant; only the
ratios matter. The final scorer reconstructs everything independently with
exact rational polynomial arithmetic and does not trust a claimed score.

## Budget and resources

- Time budget: {self.budget_s:.0f} seconds
- CPUs available to this candidate: {self.eval_cpus}
- At most {self.max_coeff_count} submitted/free coefficients

## Program contract

- Define `run(seed={self.seed}, budget_s={self.budget_s:.0f}, **kwargs)`.
- `run()` is invoked with no arguments, so its defaults are the active values.
- Return only a non-empty one-dimensional list, tuple, or NumPy array of finite
  real free coefficients. Do not return a claimed bound or root.
- `evaluate_c4_coefficients(coefficients)` is pre-imported. It performs the
  exact authoritative calculation and returns the lower-is-better bound. Exact
  root isolation is expensive, so a fast numerical surrogate may be useful for
  inner search, followed by exact checks of promising candidates.
- `C4_MAX_COEFF_COUNT` is pre-imported and equals {self.max_coeff_count}.
- Respect `budget_s` and return the best verified coefficients before timeout.
- You may use NumPy, SciPy, SymPy, CVXPY, and the Python standard library.
- Use at most {self.eval_cpus} CPUs. Make helper functions top-level; do not use
  closures or lambda functions.
- No filesystem or network I/O.

{state_ctx}
{construction_section}{memory_section}
{instruction}

## Output format

First give a strategy under 100 words between <strategy> and </strategy> tags.
Then provide exactly one ```python block containing the complete program. No
prose may follow the closing fence, and there must be no other code block.
'''
        return [{"role": "user", "content": user}]

    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = _verifier_source(self.max_coeff_count)
        if parent.construction is None:
            prelude += "initial_coefficients = None\n\n"
        else:
            prelude += (
                "initial_coefficients = np.array("
                f"{list(parent.construction)!r}, dtype=np.float64)\n\n"
            )
        return prelude + "# ---- model code below ----\n" + code

    def score(self, output: Any, stdout: str) -> RewardResult:
        result = RewardResult(reward=self.fail_score)
        if not isinstance(output, (list, tuple, np.ndarray)):
            result.msg = "bad_return_coefficients"
            result.failure_kind = "code"
            return result

        try:
            coefficients = _coerce_c4_coefficients(
                output,
                self.max_coeff_count,
            )
            c4_bound, r_max = compute_c4_bound(
                coefficients,
                self.max_coeff_count,
            )
        except Exception as exc:
            result.msg = f"Invalid C4 construction: {exc}"
            result.failure_kind = "constraint"
            return result

        result.valid = True
        result.raw_score = c4_bound
        result.reward = float(self.benchmark_c4 / c4_bound)
        result.construction = coefficients.tolist()
        result.msg = (
            f"C4 upper bound: {c4_bound:.12g}; "
            f"r_max: {r_max:.12g}; free coefficients: {len(coefficients)}"
        )
        return result

    def seed_states(self) -> List[SeedState]:
        published = (
            _GONCALVES_FREE_COEFFICIENTS,
            _ALPHAEVOLVE_FREE_COEFFICIENTS,
        )
        evaluated = []
        for coefficients in published:
            c4_bound, _ = compute_c4_bound(
                coefficients,
                self.max_coeff_count,
            )
            evaluated.append((list(coefficients), c4_bound))

        seeds: List[SeedState] = []
        for index in range(self.num_seed_states):
            coefficients, c4_bound = evaluated[index % len(evaluated)]
            seeds.append(
                SeedState(
                    code="",
                    value=float(self.benchmark_c4 / c4_bound),
                    raw_score=c4_bound,
                    construction=list(coefficients),
                )
            )
        return seeds
