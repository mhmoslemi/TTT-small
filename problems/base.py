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
                  "parameter changes.\n\n"
            )
        instruction = (
            "## Strategy-stage output\n\n"
            + history
            + "Develop a detailed, concrete, step-by-step strategy for solving "
            "the task above. Think through the mathematics, algorithm, "
            "implementation structure, numerical choices, and likely failure "
            "modes. Do not write Python code or a code fence in this stage. "
            "Do not restate the task or these output instructions. Keep the "
            "final strategy concise (at most 1,200 words), and finish it well "
            "before the response-token limit. Your final answer must contain "
            "exactly one complete <strategy>...</strategy> block and nothing "
            "else; an unclosed block is unusable."
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
