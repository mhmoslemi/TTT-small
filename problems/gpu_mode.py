"""
Kernel Engineering (GPU Mode: trimul / mla_decode_nvidia).

  - problem_type "trimul":  score_scale 1500, target ~1000 us
  - problem_type "mla_decode_nvidia": score_scale 5000, target ~1700 us
  - reward = score_scale / runtime_us   (minimize runtime, reward higher=better)

Needs: 
  * a CUDA GPU + triton (and the packages in requirements/requirements-gpumode.txt)
  * the examples/gpu_mode/lib tree present (task.yml, reference.py, eval.py, utils.py)
  * run from the repo root so `examples` and `libkernelbot` resolve
"""

from __future__ import annotations
import os
import sys
from typing import Any, List
from problems.base import Problem, ParentContext, RewardResult, SeedState, render_state_context

from examples.gpu_mode.prompt import (
    TRIMUL_PROMPT,
    MLA_DECODE_PROMPT,
    MLA_DECODE_PROMPT_END,
    MLA_DECODE_INITIAL_STATE,
    MLA_DECODE_INITIAL_VALUE,
)

_DEFAULTS = {
    "trimul": {
        "score_scale": 1500.0,
        "target": 1000.0,
        "gpu_type": "H100",
        "task_yaml": "examples/gpu_mode/lib/bioml/trimul/task.yml",
    },
    "mla_decode_nvidia": {
        "score_scale": 5000.0,
        "target": 1700.0,
        "gpu_type": "H200",
        "task_yaml": "examples/gpu_mode/lib/mla-decode/task.yml",
    },
}


class GpuMode(Problem):
    name = "gpu_mode"
    metric_name = "runtime (microseconds)"
    maximize = False

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.problem_type = str(cfg.get("problem_type", "trimul")).lower()
        if self.problem_type not in _DEFAULTS:
            raise ValueError(f"problem_type must be one of {list(_DEFAULTS)}, got {self.problem_type}")
        d = _DEFAULTS[self.problem_type]
        self.score_scale = float(cfg.get("score_scale", d["score_scale"]))
        self.gpu_type = str(cfg.get("gpu_type", d["gpu_type"]))
        self.task_yaml = str(cfg.get("task_yaml", d["task_yaml"]))
        self.lib_dir = str(cfg.get(
            "kernel_lib_dir", "examples/gpu_mode/lib"))
        if self.target is None:
            self.target = d["target"]
        # entrypoint is implicit (custom_kernel inside submission.py); not used by sandbox.
        self.entrypoint = "custom_kernel"

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)
        if self.problem_type == "trimul":
            user = f"""{TRIMUL_PROMPT}

{state_ctx}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""
        else:
            user = f"""{MLA_DECODE_PROMPT}

{state_ctx}

{MLA_DECODE_PROMPT_END}
"""
        return [{"role": "user", "content": user}]

    def preprocess(self, code: str, parent: ParentContext) -> str:  
        return code

    def score(self, output: Any, stdout: str) -> RewardResult:  
        return RewardResult(reward=self.fail_score, msg="unused")

    # ------------------------------------------------------------------
    def _fail(self, msg: str) -> RewardResult:
        return RewardResult(reward=self.fail_score, raw_score=None, valid=False,
                            parsed=True, ran=False, msg=msg, construction=[])

    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float) -> RewardResult:
        from reward import extract_python_code
        code = extract_python_code(response_text)
        if code is None:
            return RewardResult(reward=self.fail_score, parsed=False, msg="no_code_block")

        if "@triton.jit" not in code:
            return self._fail("Code must contain @triton.jit.")
        if self.problem_type == "trimul" and "identity" in code:
            return self._fail("Identity kernel is not allowed.")

        if self.lib_dir not in sys.path:
            sys.path.insert(0, self.lib_dir)
        try:
            from libkernelbot.task import make_task_definition, build_task_config
            from libkernelbot.run_eval import run_config
            from libkernelbot.submission import compute_score
            from libkernelbot.consts import SubmissionMode
        except Exception as e:  
            return self._fail(
                f"Could not import libkernelbot from '{self.lib_dir}' ({e}). "
                f"Run from the repo root with the GPU-mode requirements installed."
            )

        try:
            task = make_task_definition(self.task_yaml).task
        except Exception as e:
            return self._fail(f"Could not load task '{self.task_yaml}': {e}")

        try:
            config = build_task_config(
                task=task,
                submission_content=code,
                arch=None,
                mode=SubmissionMode.LEADERBOARD,
            )
            result = run_config(config)
        except Exception as e:
            return self._fail(f"Local kernel run failed: {e}")

        if not getattr(result, "success", False):
            return self._fail(f"Error: {getattr(result, 'error', 'run failed')}")
        runs = result.runs
        if "test" in runs and (not runs["test"].run or not runs["test"].run.passed):
            return self._fail("Failed to pass test cases.")
        if "leaderboard" not in runs or not runs["leaderboard"].run or not runs["leaderboard"].run.passed:
            return self._fail("No passing leaderboard run.")

        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000.0
        except Exception as e:
            return self._fail(f"Could not compute leaderboard score: {e}")

        res = RewardResult()
        res.valid = True
        res.parsed = True
        res.ran = True
        res.code = code
        res.raw_score = float(score_us)
        res.reward = float(self.score_scale / score_us) if score_us > 0 else self.fail_score
        res.construction = []
        res.msg = f"runtime_us={score_us}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        if self.problem_type == "mla_decode_nvidia":
            us = abs(float(MLA_DECODE_INITIAL_VALUE))  
            reward = float(self.score_scale / us) if us > 0 else 0.0
            return [SeedState(code=MLA_DECODE_INITIAL_STATE, value=reward,
                              raw_score=us, construction=[])
                    for _ in range(self.num_seed_states)]
        return [SeedState(code="", value=0.0, raw_score=None, construction=[])
                for _ in range(self.num_seed_states)]
