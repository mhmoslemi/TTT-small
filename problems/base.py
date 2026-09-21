"""
Base classes for problems.
The reward convention across ALL problems is "higher is better", so the PUCT
sampler and the entropic advantage do not need to know whether the underlying
metric is minimized or maximized. 
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from reward import extract_python_code   
from sandbox import run_code


STRATEGY_OUTPUT_CONTRACT = '''## Mandatory final-output contract

Your ONLY deliverable in this turn is the complete strategy plan for the coder.
Earlier requests to write a program describe the downstream coder's job; they
do not change your role here. Do not return Python code, a code fence, an
unwrapped explanation, or a promise to provide the plan later.

HANDOFF BOUNDARY: We extract ONLY the contents of your final <strategy> block
and send them, together with the original task inputs, to a separate coder
agent. The coder cannot see your thinking, analysis, earlier discussion, or
anything else outside that block. Subsequent strategists also receive only
your extracted plan. You are the domain expert and algorithm designer; the
coder's role is to implement your specification. Put every implementation-
relevant conclusion, mathematical definition, design decision, and concrete
step in the block itself. Nothing is communicated merely because you worked
it out earlier in your response. Never refer to unseen reasoning with phrases
such as "as derived above" or leave essential choices for the coder to invent.

After any reasoning, start your final answer with the literal opening tag
<strategy> on its own line. Put the entire substantive implementation plan,
including all six requested sections, INSIDE this single block. End it with
the literal closing tag </strategy> on its own line, then stop immediately.
Do not nest blocks, escape the tags, put them in backticks, or quote an example
block in place of your actual plan. Add no explanation or text after the
closing tag. Keep exploratory reasoning outside the final block.

The block must be a detailed, self-contained coding specification, not an
abstract, a short recap of your analysis, or six headings with a few vague
bullets. Expand the actual algorithm into numbered substeps with formulas,
inputs, outputs, dimensions, parameter choices, and failure handling wherever
needed. A coder should be able to translate these steps into a complete
program without doing new mathematical or algorithmic research. Spend the
available answer space on this final deliverable: remove repeated discussion,
but do not omit the details required to implement it correctly.

A response without both tags and a complete plan between them is unusable:
the coder will receive none of your intended plan. Reserve enough output space
to finish the plan and close the block before ending. Before finishing, read
the block as if you could see ONLY it and the original task: fill in missing
steps, resolve conflicting dimensions or settings, and check that every
nontrivial operation has an implementation rule. Check that the real plan is
enclosed exactly once and that the last non-whitespace characters of your
response are exactly </strategy>.'''


# ----------------------------------------------------------------------
# Data carried between the engine and the problems
# ----------------------------------------------------------------------
@dataclass
class SeedState:
    """One initial archive entry."""
    code: str = ""
    value: float = 0.0                 # reward (higher=better), used by the sampler
    raw_score: Optional[float] = None  # true metric, shown in the prompt
    construction: Optional[list] = None  # injected global (height_sequence_1 / initial_h_values)


@dataclass
class ParentContext:
    """Everything a prompt/preprocess needs from the parent state."""
    code: str = ""
    value: float = 0.0
    raw_score: Optional[float] = None
    construction: Optional[list] = None


@dataclass
class RewardResult:
    reward: float = 0.0
    raw_score: Optional[float] = None
    valid: bool = False
    parsed: bool = False
    ran: bool = False
    msg: str = ""
    stdout: str = ""
    code: str = ""
    construction: Optional[list] = None  
    # Failure stage used only to decide whether token-level feedback applies.
    # code = malformed source/runtime/interface bug; constraint = a runnable
    # program rejected by the scientific verifier; timeout/infrastructure are
    # explicitly excluded from feedback.
    failure_kind: str = ""


# ----------------------------------------------------------------------
# Prompt helper
# ----------------------------------------------------------------------
def render_state_context(metric_name: str, target, parent: ParentContext,
                         maximize: bool = True) -> str:
    direction = "higher is better" if maximize else "lower is better"
    if parent.code and parent.code.strip():
        shown = parent.raw_score if parent.raw_score is not None else parent.value
        return (
            f"Target {metric_name}: {target} ({direction}).\n"
            f"Your previous program achieved {metric_name} = {shown:.6f}.\n"
            f"Here is the previous program:\n"
            f"```python\n{parent.code}\n```\n"
        )
    return (
        f"Target {metric_name}: {target} ({direction}).\n"
        f"No previous program. Write one from scratch.\n"
    )


# ----------------------------------------------------------------------
# Problem ABC
# ----------------------------------------------------------------------
class Problem(ABC):
    name: str = "base"
    entrypoint: str = "run"         
    metric_name: str = "score"
    maximize: bool = True

    # Whether RewardResult.construction is the actual solution object and
    # therefore worth writing into every rollout's meta. True only where the
    # construction cannot be recovered by re-running the program: Erdos returns
    # an h array produced by a stochastic, wall-clock-bounded optimizer, so a
    # replay does not reproduce it. Circle packing and gpu_mode carry no
    # construction at all, and their programs are the artifact.
    saves_construction: bool = False
    two_stage_rollouts: bool = False
    retry_truncated_code: bool = False

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.target = self.cfg.get("target")
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.num_seed_states = int(self.cfg.get("num_seed_states", 8))
        self.seed = int(self.cfg.get("seed", 42))
        self.eval_cpus = int(self.cfg.get("eval_cpus", 1))
        if self.eval_cpus < 1:
            raise ValueError("eval_cpus must be >= 1")

    # ---- prompt / sandbox program / scoring (subclasses implement) ----
    @abstractmethod
    def build_prompt(self, parent: ParentContext, memory: str = "",
                     memory_protocol: bool = False) -> List[dict]:
        """
        Build the chat messages for one parent.

        `memory` is a pre-rendered block of retrieved lessons, or "" when memory
        is off or nothing was selected. A problem that accepts it should place it
        between the parent state and the instruction, and adapt the instruction
        to it; a problem that ignores it still works, and the trainer falls back
        to appending the block itself.

        In memory V2, `memory_protocol` is true for every arm in a matched
        comparison. The no-memory control then receives the same reasoning
        wrapper with an explicit empty hypothesis, isolating lesson content.
        """
        ...

    def build_strategy_messages(
            self, messages: List[dict],
            previous_strategies: Optional[List[str]] = None) -> List[dict]:
        staged = [dict(message) for message in messages]
        previous_strategies = list(previous_strategies or [])
        history = ""
        if previous_strategies:
            rendered = []
            for index, strategy in enumerate(previous_strategies, start=1):
                rendered.append(
                    f"<previous_strategy_{index}>\n{strategy.strip()}\n"
                    f"</previous_strategy_{index}>")
            history = (
                "The following strategies were already proposed for this same "
                "task and parent context:\n\n"
                + "\n\n".join(rendered)
                + "\n\nPropose a materially different approach. Do not merely "
                  "rename variables, reorder steps, or make superficial "
                  "parameter changes. Treat these plans as unverified proposals, "
                  "not established facts. Explain the substantive difference "
                  "in your final plan, and make that plan independently usable "
                  "without referring back to earlier strategies.\n\n"
            )
        instruction = (
            "## Strategy-stage output\n\n"
            + history
            + '''You are the domain expert, research lead, and algorithm designer.
Your work will be handed to a separate implementation-only coder agent. You
must decide and specify the mathematics, algorithm, and execution plan; the
coder translates that specification into working Python. Do not delegate the
scientific design to the coder through vague instructions or unexplained
algorithm names.

The coder receives the original task inputs and ONLY the contents of your
final <strategy> block. It will NOT see your thinking, analysis, or discussion
outside that block, even if those contain the most important parts of your
solution. Subsequent strategists also see only the extracted plan. Transfer
all implementation-relevant conclusions into a very detailed, step-by-step
final specification, with concise technical justifications for consequential
choices. The final plan must stand on its own, without access to your reasoning
transcript or earlier strategies. Long analysis followed by a short summary
does not fulfill this task.

You may reason before the final answer. End with exactly one complete
<strategy>...</strategy> block containing the following numbered sections,
using descriptive subheadings and ordered substeps. Develop each applicable
section fully. The implementation procedure should be the most detailed part
of the handoff; headings and one-line instructions alone are insufficient:

1. Approach and rationale
State the central idea, why it could improve the supplied parent or solve the
task, and the bottleneck it addresses. Choose a coherent primary approach.
Identify its important limitations and distinguish justified facts from
heuristics or hypotheses; do not claim untested improvements or optimality.

2. Mathematical specification
Define the representation, variables, exact objective, constraints, and any
derived formulas needed by the algorithm. Check indexing, boundaries, units,
and optimization direction against the supplied evaluator. Justify any claimed
equivalence or convexity. If using a surrogate, relaxation, smoothing, or a
restricted search space, explain what changes and how candidates will still be
judged using the original objective. Name a solver only with a formulation it
actually supports, including how it handles each constraint.

3. Implementation procedure
Give an executable-on-paper sequence from runtime inputs to returned output.
For each stage, specify its inputs and shapes, the exact operation, outputs,
and the condition for moving to the next stage. Define helper responsibilities
and their interfaces. Cover parent validation, initialization, the core
update/search loop, candidate acceptance, best-candidate storage, termination,
and final return. Supply the formulas or language-independent pseudocode for
every nontrivial operation, including constraint handling and repair. If
gradients are needed, give them or a precise computation and checking procedure.
For solver-based steps, specify the decision vector, objective, constraint
functions, derivative handling, initial point, stopping options, and what to
do with each relevant solver outcome. Keep indexing and dimensions consistent
throughout; if changing representation or resolution, specify how to transform
the supplied parent, restore feasibility, and compare candidates. Explain how
to proceed without a parent as well.

Instructions such as "optimize", "tune", "project to feasibility", "adjust the
dimension", or "use a solver" must be expanded into concrete operations the
coder can implement. Do not leave essential helper algorithms implicit. Keep
the proposed program within the allowed libraries and source-size limits;
those limits do not justify omitting details from the strategy specification.

4. Search choices and useful variations
Give a complete, coherent default configuration, with justified starting
settings and practical ranges or explicit adaptation rules for consequential
choices. Explain when each adjustment is triggered and how it is applied.
Several programs will be sampled from this plan:
identify a small set of meaningful variations within the primary approach
(for example initialization, representation, update schedule, or neighborhood)
and when each is worth trying. Separate correctness requirements from choices
the coder may vary. Do not force every program to execute every variation, or
arbitrarily lock the search to one dimension or unexplained parameter value.

5. Compute budget and robustness
Estimate the dominant time/memory costs for the proposed sizes and available
hardware, distinguishing estimates from measurements you have not made.
Specify a feasible schedule, where to check deadlines inside expensive loops
or solver calls, how to stop them, and how to retain the best valid candidate.
Give explicit responses to numerical instability, invalid proposals, solver
failure, and lack of improvement. Specify the construction or recovery steps
for a concrete valid fallback that can be returned before the time limit.

6. Validation and return contract
Specify checks for feasibility, finite values, objective consistency, and the
required interface/output. Re-evaluate the final candidate with the actual
task metric. Include small sanity checks that could expose mistakes in the
proposed formulas or implementation; never imply you already ran them.

Write a thorough, professionally structured implementation specification.
Carry over all necessary decisions and technical details from your analysis
into this final artifact, while keeping exploratory self-dialogue outside it.
Do not shorten the actual plan merely because you already explained something
in your thinking: that explanation is invisible to the coder. Avoid redundant
task restatements and pasted input arrays, and do not write Python source or
include code fences. Reserve enough response space to finish every section
in detail and close </strategy>; put nothing after that closing tag.'''
            + "\n\n" + STRATEGY_OUTPUT_CONTRACT
        )
        if staged and staged[-1].get("role") == "user":
            staged[-1]["content"] = (
                str(staged[-1].get("content", "")).rstrip()
                + "\n\n" + instruction + "\n"
            )
        else:
            staged.append({"role": "user", "content": instruction})
        return staged

    def build_code_messages(self, messages: List[dict],
                            strategy: str) -> List[dict]:
        staged = [dict(message) for message in messages]
        instruction = f"""## Strategy from the reasoning-model planning stage

<strategy>
{strategy.strip()}
</strategy>

## Code-stage output

Use the task information and strategy above to produce the complete solution.
Treat the strategy as planning guidance: preserve useful ideas, but correct any
mistake or conflict with the task, required interface, or current parent
construction. Do not output analysis, reasoning, a strategy, notes, or example
usage. Return only exactly one fenced Python code block, beginning with
```python and ending with ```."""
        if staged and staged[-1].get("role") == "user":
            staged[-1]["content"] = (
                str(staged[-1].get("content", "")).rstrip()
                + "\n\n" + instruction + "\n"
            )
        else:
            staged.append({"role": "user", "content": instruction})
        return staged

    @abstractmethod
    def preprocess(self, code: str, parent: ParentContext) -> str:
        """Return the full program to execute (prelude + verifier + construction + code)."""
        ...

    @abstractmethod
    def score(self, output: Any, stdout: str) -> RewardResult:
        """Validate the sandbox return value and turn it into a RewardResult."""
        ...

    @abstractmethod
    def seed_states(self) -> List[SeedState]:
        ...

    # ---- default reward path (subprocess sandbox) --------------------
    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float, *, cpu_id: Optional[int] = None
                       ) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        code = extract_python_code(response_text)
        if code is None:
            res.msg = "no_code_block"
            res.failure_kind = "code"
            return res
        res.parsed = True
        res.code = code

        full_code = self.preprocess(code, parent)
        out = run_code(
            full_code,
            entrypoint=self.entrypoint,
            timeout_s=timeout_s,
            max_cpus=(1 if cpu_id is not None else self.eval_cpus),
            cpu_id=cpu_id,
        )
        diagnostics = [out.get("stdout", ""), out.get("traceback", ""),
                       out.get("stderr", "")]
        res.stdout = "\n".join(str(x).strip() for x in diagnostics if x).strip()
        if not out.get("ok"):
            error = str(out.get("error", "unknown"))
            res.msg = f"run_failed: {error}"
            res.failure_kind = ("timeout" if "timeout" in error.lower()
                                else "code")
            return res
        res.ran = True

        scored = self.score(out.get("value"), res.stdout)
        # carry engine-level fields the scorer does not set
        scored.parsed = True
        scored.ran = True
        scored.code = code
        if not scored.stdout:
            scored.stdout = res.stdout
        if not scored.valid and not scored.msg:
            scored.msg = "invalid"
        if not scored.valid and not scored.failure_kind:
            scored.failure_kind = "constraint"
        return scored
