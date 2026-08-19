"""
Kernel Engineering (GPU Mode: trimul / mla_decode_nvidia).

  - problem_type "trimul":  score_scale 1500, target ~1000 us
  - problem_type "mla_decode_nvidia": score_scale 5000, target ~1700 us
  - reward = score_scale / runtime_us   (minimize runtime, reward higher=better)

Needs:
  * a CUDA GPU + triton (and the packages in requirements/requirements-gpumode.txt)
  * the examples/gpu_mode/lib tree present (task.yml, reference.py, eval.py, utils.py)
  * run from the repo root so `examples` and `libkernelbot` resolve

Two changes from the original, both for the memory and feedback components:

  build_prompt takes `memory` and places the retrieved lessons between the
  parent kernel and the rules, adapting the instruction when they are present.

  compute_reward now captures compiler and test output into res.stdout. The
  original discarded it and returned only a one-line msg, which meant the
  feedback signal's f_i was "Failed to pass test cases." with nothing in it.
  Triton compile errors and correctness mismatches are the richest textual
  feedback this problem produces, and they were being thrown away.

  compute_reward now also honours a timeout. The original ignored
  sandbox_timeout_s entirely and called run_config inline, so a kernel that hung
  (an unterminated loop, a deadlocked launch) blocked forever. With
  reward_workers=1, which this problem requires, that stalls the whole step. The
  evaluation now runs in a spawned child that is terminated on timeout, and the
  child returns a plain dict so nothing has to pickle across.
"""

from __future__ import annotations
import multiprocessing as mp
import os
import sys
from typing import Any, Dict, List, Optional
from problems.base import (Problem, ParentContext, RewardResult, SeedState,
                           render_state_context)

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

_MEMORY_HEADER = """## Lessons from earlier kernels in this search

Extracted from kernels already generated and benchmarked here. Empirical
findings, not part of the specification above, and they do not override any rule
stated in it."""

_ANALYSIS_WITH_MEMORY = """## Analysis

Work through the lessons above before writing anything:
- Which apply to the kernel you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. They are evidence from
  earlier attempts, and some will be wrong or irrelevant for this kernel.
- Is anything they recommend already present above and still not fast enough?
  Then that avenue is exhausted and the win is somewhere they do not cover.

A lesson gives you an idea; you choose the implementation. Do not copy a kernel
body or an autotune configuration verbatim, and do not let a lesson fix your
tiling or memory-access strategy for you."""


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n // 3] + "\n...[truncated]...\n" + s[-(2 * n // 3):]


def collect_logs(result, limit: int = 4000) -> str:
    """
    Everything the runner said, as f_i for the feedback signal and as failure
    evidence for the memory maker.

    Ordered worst-first on purpose: a compile error explains a failure and a
    benchmark log does not, and the feedback reprompt keeps the tail.
    """
    parts = []
    err = getattr(result, "error", "") or ""
    if err:
        parts.append("runner error:\n" + _clip(err, limit))

    for name, ev in (getattr(result, "runs", {}) or {}).items():
        comp = getattr(ev, "compilation", None)
        if comp is not None and not getattr(comp, "success", True):
            parts.append(f"[{name}] COMPILE FAILED\n"
                         + _clip(getattr(comp, "stderr", "") or
                                 getattr(comp, "stdout", ""), limit))
        run = getattr(ev, "run", None)
        if run is None:
            continue
        if not getattr(run, "passed", False):
            body = (getattr(run, "stderr", "") or "") + "\n" + (getattr(run, "stdout", "") or "")
            parts.append(f"[{name}] RUN FAILED\n" + _clip(body.strip(), limit))
        elif getattr(run, "stderr", ""):
            parts.append(f"[{name}] warnings\n" + _clip(run.stderr, limit // 2))

    return "\n\n".join(p for p in parts if p.strip())[: limit * 2]


def _eval_child(conn, code: str, lib_dir: str, task_yaml: str,
                problem_type: str, log_chars: int,
                gpu_id: Optional[int] = None) -> None:
    """
    Runs in a spawned process. Does the whole evaluation and sends back a plain
    dict: no dataclasses, no datetimes, nothing that has to pickle.

    gpu_id pins the benchmark to one device, set BEFORE torch is imported so the
    driver honours it. This matters more than it looks: the reward here IS a
    measured runtime, so benchmarking on a device that is concurrently running
    generation or a training backward pass produces a number that reflects the
    contention rather than the kernel. Reserve a device that nothing else uses.
    """
    out = {"ok": False, "msg": "unknown", "logs": "", "score_us": None}
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        from libkernelbot.task import make_task_definition, build_task_config
        from libkernelbot.run_eval import run_config
        from libkernelbot.submission import compute_score
        from libkernelbot.consts import SubmissionMode

        task = make_task_definition(task_yaml).task
        config = build_task_config(task=task, submission_content=code,
                                   arch=None, mode=SubmissionMode.LEADERBOARD)
        result = run_config(config)
        out["logs"] = collect_logs(result, log_chars)

        if not getattr(result, "success", False):
            out["msg"] = f"Error: {getattr(result, 'error', 'run failed')}"
        else:
            runs = result.runs
            if "test" in runs and (not runs["test"].run or not runs["test"].run.passed):
                out["msg"] = "Failed to pass test cases."
            elif ("leaderboard" not in runs or not runs["leaderboard"].run
                  or not runs["leaderboard"].run.passed):
                out["msg"] = "No passing leaderboard run."
            else:
                score_seconds = compute_score(result, task, submission_id=-1)
                out["score_us"] = float(score_seconds) * 1_000_000.0
                out["ok"] = True
                out["msg"] = f"runtime_us={out['score_us']}"
    except Exception as e:
        out["msg"] = f"Local kernel run failed: {e!r}"
    try:
        conn.send(out)
    except Exception:
        pass
    finally:
        conn.close()


def run_eval_with_timeout(code: str, lib_dir: str, task_yaml: str,
                          problem_type: str, log_chars: int,
                          timeout_s: float, gpu_id: Optional[int] = None) -> Dict:
    """
    Spawn, wait, and kill on timeout.

    A thread cannot be killed, so a thread-based timeout would leave the hung
    kernel running in the background and leak a GPU context every time. A child
    process can be terminated, which is the point.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_eval_child,
                       args=(child_conn, code, lib_dir, task_yaml,
                             problem_type, log_chars, gpu_id), daemon=True)
    proc.start()
    child_conn.close()

    out = None
    try:
        if parent_conn.poll(timeout_s):
            out = parent_conn.recv()
    except Exception:
        out = None
    finally:
        parent_conn.close()

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
    else:
        proc.join(5)

    if out is None:
        return {"ok": False, "score_us": None, "logs": "",
                "msg": f"kernel_eval_timeout after {timeout_s:.0f}s "
                       f"(process killed)"}
    return out


class GpuMode(Problem):
    name = "gpu_mode"
    metric_name = "runtime (microseconds)"
    maximize = False

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.problem_type = str(cfg.get("problem_type", "trimul")).lower()
        if self.problem_type not in _DEFAULTS:
            raise ValueError(f"problem_type must be one of {list(_DEFAULTS)}, "
                             f"got {self.problem_type}")
        d = _DEFAULTS[self.problem_type]
        self.score_scale = float(cfg.get("score_scale", d["score_scale"]))
        self.gpu_type = str(cfg.get("gpu_type", d["gpu_type"]))
        self.task_yaml = str(cfg.get("task_yaml", d["task_yaml"]))
        self.lib_dir = str(cfg.get("kernel_lib_dir",
                                   cfg.get("lib_dir", "examples/gpu_mode/lib")))
        if self.target is None:
            self.target = d["target"]
        self.log_chars = int(cfg.get("kernel_log_chars", 4000))
        # 0 falls back to sandbox_timeout_s; set explicitly to override it.
        self.kernel_timeout_s = float(cfg.get("kernel_timeout_s", 0.0))
        # Physical device the benchmark runs on. None inherits the parent's,
        # which is the training GPU: correct only if nothing else is on it.
        gid = cfg.get("kernel_gpu_id", None)
        self.kernel_gpu_id = None if gid is None else int(gid)
        # entrypoint is implicit (custom_kernel inside submission.py)
        self.entrypoint = "custom_kernel"

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)

        memory_section = ""
        analysis = ""
        if memory and memory.strip():
            memory_section = f"\n{_MEMORY_HEADER}\n\n{memory.strip()}\n"
            analysis = f"\n{_ANALYSIS_WITH_MEMORY}\n"

        if self.problem_type == "trimul":
            user = f"""{TRIMUL_PROMPT}

{state_ctx}
{memory_section}{analysis}
Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
- Do not wrap the kernel in a try/except that falls back to a slow reference path.
  A fallback that silently produces correct-but-slow output hides the failure and
  scores worse than letting the kernel fail loudly.
"""
        else:
            user = f"""{MLA_DECODE_PROMPT}

{state_ctx}
{memory_section}{analysis}
{MLA_DECODE_PROMPT_END}
"""
        return [{"role": "user", "content": user}]

    def preprocess(self, code: str, parent: ParentContext) -> str:
        return code

    def score(self, output: Any, stdout: str) -> RewardResult:
        return RewardResult(reward=self.fail_score, msg="unused")

    # ------------------------------------------------------------------
    def _fail(self, msg: str, logs: str = "") -> RewardResult:
        return RewardResult(reward=self.fail_score, raw_score=None, valid=False,
                            parsed=True, ran=False, msg=msg, stdout=logs,
                            construction=[])

    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float) -> RewardResult:
        from reward import extract_python_code
        code = extract_python_code(response_text)
        if code is None:
            return RewardResult(reward=self.fail_score, parsed=False,
                                msg="no_code_block")

        if "@triton.jit" not in code:
            return self._fail("Code must contain @triton.jit.")
        if self.problem_type == "trimul" and "identity" in code:
            return self._fail("Identity kernel is not allowed.")

        if not os.path.exists(self.task_yaml):
            return self._fail(f"task_yaml not found: {self.task_yaml} "
                              f"(run from the repo root)")

        timeout = float(self.kernel_timeout_s or 0.0) or float(timeout_s or 0.0)
        if timeout > 0:
            out = run_eval_with_timeout(code, self.lib_dir, self.task_yaml,
                                        self.problem_type, self.log_chars,
                                        timeout, self.kernel_gpu_id)
        else:
            # timeout disabled: run inline, the original behaviour
            import multiprocessing.connection as _c

            class _Sink:
                def send(self, v): self.v = v
                def close(self): pass
            sink = _Sink()
            _eval_child(sink, code, self.lib_dir, self.task_yaml,
                        self.problem_type, self.log_chars, self.kernel_gpu_id)
            out = getattr(sink, "v", {"ok": False, "msg": "no result",
                                      "logs": "", "score_us": None})

        if not out.get("ok"):
            return self._fail(out.get("msg", "run failed"), out.get("logs", ""))

        score_us = float(out["score_us"])
        res = RewardResult()
        res.valid = True
        res.parsed = True
        res.ran = True
        res.code = code
        res.stdout = out.get("logs", "")      # warnings survive on success too
        res.raw_score = score_us
        res.reward = (float(self.score_scale / score_us) if score_us > 0
                      else self.fail_score)
        res.construction = []
        res.msg = out.get("msg", f"runtime_us={score_us}")
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