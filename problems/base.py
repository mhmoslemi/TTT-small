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

from evaluation.reward import extract_python_code
from evaluation.sandbox import run_code
from generation.strategy_prompts import LOCAL_STRATEGY_PROMPT


CODER_EXECUTION_CONTRACT = '''## Coder capabilities

The coder is a one-shot, text-only LLM. It has no CPU, GPU, Python interpreter,
terminal, tools, or execution environment. It can only use internal
language-model reasoning and write source code in a single response. It cannot
run or test the program, execute numerical calculations, inspect results, or
revise the program after execution feedback.

The harness executes the submitted program only after the coder's response
is complete. Any hardware or libraries described in this task belong to that
execution environment, not to the coder. Implement required calculations,
candidate checks, scoring, and fallback handling as source-code operations for
the harness to execute. Do not perform these operations while composing the
response, claim measured results, or include a testing/development checklist.
The coder's deliverable is the program source, not a computed solution.'''


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
    # The parent's own parent's true metric ("before"; raw_score is "after").
    parent_raw_score: Optional[float] = None
    # What the run that produced the parent printed (see observation_text).
    observation: str = ""


# The previous program's stdout is shown to the model whole. The only bound is
# a safety cap against a runaway print loop (the sandbox itself stops at 256
# KiB); beyond it the start and the end are kept. 0 disables the cap. Problem
# config key: parent_output_max_chars.
PARENT_OUTPUT_MAX_CHARS = 20000


def observation_text(stdout, max_chars: int) -> str:
    """The program's own stdout (no traceback/stderr), kept for the next prompt."""
    text = str(stdout or "").strip()
    if max_chars and len(text) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        text = (text[:head].rstrip()
                + f"\n[... {len(text) - max_chars} characters omitted ...]\n"
                + text[-tail:].lstrip())
    return text


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
    # Failure stage preserved in rollout diagnostics.
    # code = malformed source/runtime/interface bug; constraint = a runnable
    # program rejected by the scientific verifier; timeout/infrastructure are
    # distinct from code failures.
    failure_kind: str = ""
    # stdout the program itself printed, for the next prompt (observation_text).
    observation: str = ""


def is_code_failure(res) -> bool:
    """Classify code failures separately from verifier, timeout and infra failures.

    Explicit failure kinds are authoritative. The fallback supports older
    RewardResult-like objects without losing their original classification.
    """
    kind = str(getattr(res, "failure_kind", "") or "").strip().lower()
    if kind:
        return kind == "code"
    if not bool(getattr(res, "parsed", False)):
        return True
    if bool(getattr(res, "ran", False)):
        return False
    msg = str(getattr(res, "msg", "") or "").lower()
    return "timeout" not in msg


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


def render_ttt_state_context(metric_name: str, target, parent: ParentContext,
                             maximize: bool = True) -> str:
    """Parent block of the TTT-Discover prompt (tinker_utils/state.py::State.to_prompt).

    Same wording and layout, with three deliberate differences:
      * the previous program's stdout is shown whole (no 500-character tail);
      * a parent without a "before" score says "(lower|higher is better)"
        according to the metric, where the reference hard-codes "higher";
      * `before`/`after` are the problem's true metric (raw_score), because
        `value` here is a reward (1/raw for AC1), not the metric itself.
    Callers keep the old render_state_context text for the no-parent case.
    """
    better = "higher" if maximize else "lower"
    ctx = (f"You are iteratively optimizing {metric_name}.\n"
           f"Here is the last code we ran:\n{parent.code}")
    after = parent.raw_score
    before = parent.parent_raw_score
    if after is not None:
        gap = (target - after) if maximize else (after - target)
        if before is not None and parent.construction:
            ctx += (f"\nHere is the {metric_name} before and after running the "
                    f"code above ({better} is better): {before:.6f} -> {after:.6f}")
        else:
            ctx += f"\nCurrent {metric_name} ({better} is better): {after:.6f}"
        ctx += (f"\nTarget: {target}. Current gap: {gap:.6f}. "
                "Further improvements will also be generously rewarded.")
    else:
        ctx += f"\nTarget {metric_name}: {target}"
    output = str(parent.observation or "").strip()
    if output:
        ctx += f"\n\n--- Previous Program Output ---\n{output}\n--- End Output ---"
    return ctx


# ----------------------------------------------------------------------
# Problem ABC
# ----------------------------------------------------------------------
class Problem(ABC):
    name: str = "base"
    entrypoint: str = "run"         
    metric_name: str = "score"
    maximize: bool = True

    # Whether RewardResult.construction is the actual solution object and
    # therefore worth writing into every rollout's meta. Enable it where the
    # construction cannot be recovered by re-running the program: Erdos and the
    # AC problems return arrays produced by stochastic, wall-clock-bounded
    # optimizers, so a replay need not reproduce them. Circle packing and
    # gpu_mode carry no construction at all, and their programs are the artifact.
    saves_construction: bool = False
    two_stage_rollouts: bool = False

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.target = self.cfg.get("target")
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.num_seed_states = int(self.cfg.get("num_seed_states", 8))
        self.seed = int(self.cfg.get("seed", 42))
        self.eval_cpus = int(self.cfg.get("eval_cpus", 1))
        self.parent_output_max_chars = int(
            self.cfg.get("parent_output_max_chars", PARENT_OUTPUT_MAX_CHARS))
        if self.parent_output_max_chars < 0:
            raise ValueError("parent_output_max_chars must be >= 0")
        self.require_final_code_marker = bool(
            self.cfg.get("coder_model_profile") == "qwen3.8-27b"
            and self.cfg.get("coder_preserve_thinking", False)
        )
        if self.eval_cpus < 1:
            raise ValueError("eval_cpus must be >= 1")

    # ---- prompt / sandbox program / scoring (subclasses implement) ----
    @abstractmethod
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        """Build the chat messages for one parent state."""
        ...

    def build_strategy_messages(
            self, messages: List[dict],
            previous_strategies: Optional[List[str]] = None) -> List[dict]:
        if self.cfg.get("api_strategy"):
            from generation.strategy_prompts_openai import build_openai_strategy_messages
            return build_openai_strategy_messages(messages, previous_strategies)
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
        instruction = LOCAL_STRATEGY_PROMPT.replace(
            "{previous_strategies}", history, 1)
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
Read the strategy critically and extract useful mathematical insights,
algorithmic ideas, and implementation details. Build on the parts you judge
helpful. Treat its proposals as suggestions, not requirements: you are free to
adapt, simplify, combine, replace, or discard any part, including choosing a
different overall approach. Use your own judgment to produce the best valid
solution, while following the original task, evaluator, required interface,
and execution constraints.
Do not output analysis, reasoning, a strategy, notes, or example
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
                       timeout_s: float, *, cpu_id: Optional[int] = None,
                       memory_limit_bytes: Optional[int] = None,
                       final_answer_text: Optional[str] = None
                       ) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        code = extract_python_code(
            response_text,
            require_final_marker=self.require_final_code_marker,
            final_answer_text=final_answer_text,
        )
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
            memory_limit_bytes=memory_limit_bytes,
        )
        diagnostics = [out.get("stdout", ""), out.get("traceback", ""),
                       out.get("stderr", "")]
        res.stdout = "\n".join(str(x).strip() for x in diagnostics if x).strip()
        res.observation = observation_text(
            out.get("stdout", ""), self.parent_output_max_chars)
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
        scored.observation = res.observation
        if not scored.valid and not scored.msg:
            scored.msg = "invalid"
        if not scored.valid and not scored.failure_kind:
            scored.failure_kind = "constraint"
        return scored
