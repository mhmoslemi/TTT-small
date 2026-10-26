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
from strategy_prompts import LOCAL_STRATEGY_PROMPT


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
            from strategy_prompts_openai import build_openai_strategy_messages
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
                       timeout_s: float, *, cpu_id: Optional[int] = None,
                       memory_limit_bytes: Optional[int] = None
                       ) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        code = extract_python_code(
            response_text,
            require_final_marker=self.require_final_code_marker,
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
