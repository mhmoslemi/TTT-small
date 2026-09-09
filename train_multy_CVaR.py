"""
TTT-Discover — multi-problem local runner.

Configuration: self-contained problem YAML < resumed config < CLI flags

Rank-selection mode (the proposed local surrogate):
    python train_multy_CVaR.py --problem erdos --advantage-mode rank --no-feedback

It uses exact reward midranks, KL-budgeted selection weights, A_i = G*q_i - 1,
token-level clipped ratios, a separate sampled reference KL, and sampled
trajectory entropy on completely tied groups. Advantages and old/reference
log-probabilities are frozen across --rank-update-epochs (default 1). No reward
standardization, EVT fitting, or quantile cutoff is used in this mode.

Rank mode sets rollout temperature=1 and top_p=1. Generation workers must use
the synchronized policy with no additional top-k/repetition/logit filtering;
vLLM workers return sampled-token behavior probabilities with the rollout.
Missing values retain a compatibility fallback through the trainer.
The built-in local generation path explicitly disables top-k sampling.
--feedback remains an optional auxiliary repair loss; --no-feedback implements
the standalone rank objective. Existing advantage modes retain their losses.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import sys
import argparse
import json
import logging
import math
import random
import time
from contextlib import (contextmanager, nullcontext, redirect_stderr,
                        redirect_stdout)
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml


_ROUTED_DEPENDENCY_NOTICES = (
    "Skipping import of cpp extensions due to incompatible torch version.",
    "No prebuilt binary for CUDA",
    "You are sending unauthenticated requests to the HF Hub.",
    "is an Enum subclass and is now natively supported by torch.compile",
    "`torch_dtype` is deprecated! Use `dtype` instead!",
)


class _NoticeRoutingStream:
    """Keep ordinary model-loading output visible and route known noise."""

    def __init__(self, visible, diagnostic, label):
        self.visible = visible
        self.diagnostic = diagnostic
        self.label = label
        self._routing_line = False

    def _diagnostic_is_open(self):
        return (self.diagnostic is not None
                and not bool(getattr(self.diagnostic, "closed", False)))

    def write(self, value):
        for piece in str(value).splitlines(keepends=True):
            route = (self._diagnostic_is_open()
                     and (self._routing_line
                          or any(marker in piece
                                 for marker in _ROUTED_DEPENDENCY_NOTICES)))
            if route:
                if not self._routing_line:
                    self.diagnostic.write(f"[{self.label}] ")
                self.diagnostic.write(piece)
            else:
                self.visible.write(piece)
            self._routing_line = bool(
                route and not piece.endswith(("\n", "\r")))
        return len(value)

    def flush(self):
        self.visible.flush()
        if self._diagnostic_is_open():
            self.diagnostic.flush()

    def detach_diagnostic(self):
        """Make a handler-retained wrapper safe after its file is closed."""
        self._routing_line = False
        self.diagnostic = None

    def __getattr__(self, name):
        return getattr(self.visible, name)


def _restore_logging_stream(routed_stream, visible_stream):
    """Detach temporary routing streams retained by logging handlers."""
    root = logging.getLogger()
    loggers = [root]
    loggers.extend(
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    handlers = [logging.lastResort]
    for logger in loggers:
        handlers.extend(logger.handlers)

    seen = set()
    for handler in handlers:
        if handler is None or id(handler) in seen:
            continue
        seen.add(id(handler))
        if getattr(handler, "stream", None) is not routed_stream:
            continue
        # setStream flushes the old stream first. This runs while the
        # diagnostic file is still open, then points future warnings back to
        # the original console stream.
        handler.setStream(visible_stream)


@contextmanager
def _route_dependency_notices(log_path):
    if not log_path:
        yield
        return
    with open(log_path, "a", buffering=1, encoding="utf-8") as diagnostic:
        stdout = _NoticeRoutingStream(
            sys.stdout, diagnostic, "trainer dependency stdout")
        stderr = _NoticeRoutingStream(
            sys.stderr, diagnostic, "trainer dependency stderr")
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
        finally:
            _restore_logging_stream(stdout, stdout.visible)
            _restore_logging_stream(stderr, stderr.visible)
            # Some libraries cache a handler before attaching it to a logger,
            # so it cannot always be found by _restore_logging_stream. Such a
            # handler may retain this wrapper after the context closes. Leave
            # that wrapper usable, but make every future write console-only.
            stdout.detach_diagnostic()
            stderr.detach_diagnostic()


# ======================================================================
# Every problem YAML is complete. There is no shared default-value file.
# ======================================================================


# ======================================================================
# CLI parsing + config loading (problem YAML < resumed config < CLI)
# ======================================================================
ADVANTAGE_MODES = ("entropic", "grpo", "cvar", "rank")
CVAR_ALPHA_DEFAULT = 0.2     # tail mass: cutoff at the 80th percentile
CVAR_LAMBDA_DEFAULT = 0.5    # weight of the upper-tail term vs plain GRPO
RANK_GAMMA_DEFAULT = math.log(2)
RANK_CLIP_EPSILON_DEFAULT = 0.2
RANK_ENTROPY_COEF_DEFAULT = 0.001
RANK_UPDATE_EPOCHS_DEFAULT = 1


def compute_group_advantages(rewards_np, mode: str, cvar_alpha=None,
                             cvar_lambda=None, *, rank_gamma=None,
                             return_info=False):
    """
    Dispatch the per-group advantage estimator.

    Returns (advantages, scale, scale_label). `scale` is the estimator's
    scalar for logging: beta for entropic, reward std for grpo, and the
    upper-VaR threshold q_{1-alpha} for cvar, or eta for rank (possibly +inf).
    return_info=True appends a diagnostics dict (empty for legacy estimators).
    """
    mode = str(mode or "entropic").lower()
    if mode == "entropic":
        from advantage import entropic_adaptive_advantages
        adv, beta = entropic_adaptive_advantages(rewards_np)
        result, info = (adv, beta, "beta"), {}
    elif mode == "grpo":
        from advantage import grpo_advantages
        adv, std = grpo_advantages(rewards_np)
        result, info = (adv, std, "std"), {}
    elif mode == "cvar":
        from advantage import upper_tail_advantages
        alpha = CVAR_ALPHA_DEFAULT if cvar_alpha is None else float(cvar_alpha)
        lam = CVAR_LAMBDA_DEFAULT if cvar_lambda is None else float(cvar_lambda)
        adv, q = upper_tail_advantages(rewards_np, alpha=alpha, lam=lam)
        result, info = (adv, q, "var"), {}
    elif mode == "rank":
        from advantage import rank_adaptive_advantages
        gamma = RANK_GAMMA_DEFAULT if rank_gamma is None else float(rank_gamma)
        adv, eta, info = rank_adaptive_advantages(
            rewards_np, gamma=gamma, return_info=True)
        result = (adv, eta, "eta")
    else:
        raise ValueError(f"unknown advantage_mode {mode!r}; expected one of {ADVANTAGE_MODES}")
    return (*result, info) if return_info else result


def _resolve_rank_options(merged):
    """Resolve trainer-owned rank options after YAML/resume/CLI overlays."""
    defaults = {
        "rank_gamma": RANK_GAMMA_DEFAULT,
        "rank_clip_epsilon": RANK_CLIP_EPSILON_DEFAULT,
        "rank_entropy_coef": RANK_ENTROPY_COEF_DEFAULT,
    }
    for name, default in defaults.items():
        value = merged.get(name)
        value = default if value is None else float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        merged[name] = value
    if merged["rank_gamma"] <= 0.0:
        raise ValueError("rank_gamma must be positive")
    if not 0.0 < merged["rank_clip_epsilon"] < 1.0:
        raise ValueError("rank_clip_epsilon must be in (0, 1)")
    for side in ("low", "high"):
        name = f"rank_clip_epsilon_{side}"
        value = merged.get(name)
        value = (merged["rank_clip_epsilon"] if value is None
                 else float(value))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")
        merged[name] = value
    if merged["rank_entropy_coef"] < 0.0:
        raise ValueError("rank_entropy_coef must be nonnegative")
    epochs = merged.get("rank_update_epochs")
    epochs = RANK_UPDATE_EPOCHS_DEFAULT if epochs is None else epochs
    if (isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer))
            or epochs < 1):
        raise ValueError("rank_update_epochs must be a positive integer")
    merged["rank_update_epochs"] = int(epochs)
    if merged.get("advantage_mode") == "rank":
        if int(merged["group_size"]) < 2:
            raise ValueError("rank mode requires group_size >= 2")
        kl_coef = float(merged["kl_penalty_coef"])
        if not math.isfinite(kl_coef) or kl_coef < 0.0:
            raise ValueError("rank mode requires a finite nonnegative kl_penalty_coef")
        if (float(merged["temperature"]) != 1.0
                or float(merged["top_p"]) != 1.0):
            print("[config] rank mode sets temperature=1 and top_p=1 "
                  "to match the policy likelihood used by the loss")
        merged["temperature"] = 1.0
        merged["top_p"] = 1.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTT-Discover multi-problem runner")
    # Problem selection
    p.add_argument("--problem", default=None,
                   help="Problem name. Loads configs/<problem>.yaml unless --config "
                        "is given. Defaults to erdos. "
                        "One of: circle_packing, "
                        "erdos, ac1, ac2, denoising, gpu_mode.")
    p.add_argument("--config", default=None,
                   help="Explicit path to a YAML config (overrides the --problem lookup).")
    p.add_argument("--resume", "--resume-from", dest="resume", default=None,
                   metavar="RUN_DIR",
                   help="Continue an existing run directory from its latest "
                        "completed checkpoint.")
    p.add_argument("--gpu-type", default=None,
                   help="Target hardware for a kernel problem: L40S, A100, H100, "
                        "H200, ... Sets the prompt's arch notes and rules line, "
                        "and scales target/score_scale from the H100 defaults.")
    p.add_argument("--kernel-gpu-id", type=int, default=None,
                   help="Physical device the kernel benchmark owns, exclusively.")
    p.add_argument("--kernel-timeout-s", type=float, default=None)
    p.add_argument("--problem-type", default=None,
                   help="Sub-type for multi-mode problems (ac1/ac2, trimul/mla_decode_nvidia).")

    # CLI overrides — all default None so we can tell 'not given' from 'given'.
    p.add_argument(
        "--backend", choices=["auto", "unsloth", "hf", "vllm"], default=None,
        help="Training backend. 'vllm' is shorthand for HF+PEFT training plus "
             "vLLM rollout generation (vLLM itself does not backpropagate).")
    p.add_argument(
        "--fast", action="store_const", const=True, default=None,
        help="Use the opt-in process-per-GPU rank trainer: work is balanced "
             "by sequence cost and every GPU independently learns a padded-"
             "token budget capped at 80%% GPU memory use. "
             "Without this flag the existing trainer is unchanged.")
    p.add_argument(
        "--isolate-eval", action="store_const", const=True, default=None,
        help="Run every CPU candidate in its own one-CPU-affined sandbox. "
             "In this mode reward_workers is concurrent processes per CPU, "
             "and excess candidates wait in the executor queue. Without this "
             "flag the existing evaluation path is unchanged.")
    p.add_argument("--generation-backend", choices=["hf", "vllm"], default=None,
                   help="Engine used by generation workers. Independent of the "
                        "differentiable training backend.")
    p.add_argument("--model-name", default=None)
    p.add_argument(
        "--training-model-name", default=None,
        help="Optional trainable checkpoint override. Rollout generation still "
             "uses --model-name. GPT-OSS QLoRA resolves this automatically to "
             "a trainable BitsAndBytes checkpoint.")
    p.add_argument("--load-in-4bit", action="store_const", const=True, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--num-circles", type=int, default=None)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--sandbox-timeout-s", type=float, default=None)
    p.add_argument("--num-steps", type=int, default=None,
                   help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=None,
                   help="Number of parent states sampled per step (paper: 8)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Rollouts per parent per step (paper: 64)")
    p.add_argument("--num-seed-states", type=int, default=None)
    # ---- adaptive batch growth (groups-per-step / group-size are the START) --
    p.add_argument("--max-groups-per-step", type=int, default=None,
                   help="Cap that G ratchets up to. Omit to use the starting G.")
    p.add_argument("--max-group-size", type=int, default=None,
                   help="Cap that K ratchets up to. Omit to use the starting K.")
    p.add_argument("--growth-force-step", type=int, default=None,
                   help="From this step on, run at (max G, max K) no matter what.")
    p.add_argument("--growth-valid-yield", type=float, default=None,
                   help="Best group's valid fraction must reach this to grow.")
    p.add_argument("--growth-distinct-min", type=int, default=None,
                   help="Distinct improved children needed to grow.")
    p.add_argument("--growth-factor", type=float, default=None,
                   help="Multiply G and K by this when both signals clear.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--adam-beta1", type=float, default=None)
    p.add_argument("--adam-beta2", type=float, default=None)
    p.add_argument("--adam-epsilon", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--kl-penalty-coef", type=float, default=None)
    p.add_argument("--advantage-mode", choices=list(ADVANTAGE_MODES), default=None,
                   help="Group advantage estimator: 'entropic' (adaptive-beta "
                        "entropic objective, default), 'grpo' (mean/std "
                        "normalized), or 'cvar' (grpo blended with an "
                        "upper-tail term above the (1-alpha)-quantile), or "
                        "'rank' (KL-budgeted midranks and clipped GRPO).")
    p.add_argument("--rank-gamma", type=float, default=None,
                   help="rank mode: selection KL budget (default log(2)); "
                        "distinct from --kl-penalty-coef.")
    p.add_argument("--rank-clip-epsilon", type=float, default=None,
                   help="rank mode: symmetric token-ratio clipping width used "
                        "for both sides unless a side-specific value is given "
                        "(default 0.2).")
    p.add_argument("--rank-clip-epsilon-low", type=float, default=None,
                   help="rank mode: lower clipping distance, giving the bound "
                        "1-epsilon-low (defaults to --rank-clip-epsilon).")
    p.add_argument("--rank-clip-epsilon-high", type=float, default=None,
                   help="rank mode: upper clipping distance, giving the bound "
                        "1+epsilon-high (defaults to --rank-clip-epsilon).")
    p.add_argument("--rank-entropy-coef", type=float, default=None,
                   help="rank mode: entropy coefficient for fully tied groups "
                        "(default 0.01; 0 disables entropy).")
    p.add_argument("--rank-update-epochs", type=int, default=None,
                   help="rank mode: updates per rollout batch using a frozen "
                        "old policy (default 1); clipping becomes active after "
                        "the first update when this is greater than 1.")
    p.add_argument("--cvar-alpha", type=float, default=None,
                   help="cvar mode: tail mass alpha in (0,1); cutoff is the "
                        f"(1-alpha)-quantile (default {CVAR_ALPHA_DEFAULT}).")
    p.add_argument("--cvar-lambda", type=float, default=None,
                   help="cvar mode: mixing weight in [0,1] of the upper-tail "
                        f"term; 0 = plain grpo (default {CVAR_LAMBDA_DEFAULT}).")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument(
        "--train-examples-per-microbatch", type=int, default=None,
        help="Maximum examples in one LoRA forward/backward call per GPU. "
             "Examples are length-bucketed and may be split further to keep "
             "the padded-token total within max_seq_length.")
    p.add_argument("--logprob-chunk", type=int, default=None,
                   help="Slice compute_token_logprobs over response positions "
                        "into chunks of at most this many tokens, bounding the "
                        "float32 log_softmax spike. Exact (no precision loss). "
                        "0 = single shot. Use when the feedback teacher forward "
                        "OOMs on a large-vocab model.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--thinking", dest="thinking",
                   action="store_const", const=True, default=None,
                   help="Enable thinking mode in chat templates that support it "
                        "(for example hybrid Qwen3 checkpoints).")
    p.add_argument("--no-thinking", dest="thinking",
                   action="store_const", const=False,
                   help="Disable thinking mode, overriding the YAML.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", dest="deterministic",
                   action="store_const", const=True, default=None,
                   help="Seed every generation stream from --seed so runs are "
                        "reproducible. Off by default.")
    p.add_argument("--no-deterministic", dest="deterministic",
                   action="store_const", const=False,
                   help="Force determinism off, overriding the YAML.")
    p.add_argument("--print-responses", type=int, default=None)
    p.add_argument("--max-saved-construction", type=int, default=None,
                   help="Max construction length stored per rollout meta. "
                        "0 disables saving it.")
    p.add_argument("--reward-workers", type=int, default=None,
                   help="Normally, total reward-launcher threads (0 = auto). "
                        "With --isolate-eval, concurrent sandbox processes per "
                        "CPU (must be >= 1). Use 1 for any problem whose reward "
                        "is a measured runtime.")
    p.add_argument("--eval-cpus", type=int, default=None,
                   help="CPU threads available to each candidate evaluation. "
                        "Auto reward-worker counts account for this value; "
                        "--isolate-eval overrides it to one CPU per candidate.")
    p.add_argument("--training-gpu-id", type=int, default=None,
                   help="Assert the training id derived from AVAILABLE_GPUS.")
    p.add_argument("--available-gpu-ids", type=str, default=None,
                   help="Legacy direct-Python fallback when AVAILABLE_GPUS is "
                        "unset; run.sh's AVAILABLE_GPUS is authoritative.")
    p.add_argument("--reserve-last-gpu-for-evaluation",
                   dest="reserve_last_gpu_for_evaluation",
                   action="store_const", const=True, default=None,
                   help="Compatibility flag; gpu_mode reservation is derived "
                        "from AVAILABLE_GPUS.")
    p.add_argument("--no-reserve-last-gpu-for-evaluation",
                   dest="reserve_last_gpu_for_evaluation",
                   action="store_const", const=False)
    p.add_argument("--evaluation-gpu-id", type=int, default=None,
                   help="Assert the evaluation id derived from AVAILABLE_GPUS.")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Assert the generation GPU count derived from "
                        "AVAILABLE_GPUS.")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Assert the generation group derived from "
                        "AVAILABLE_GPUS.")
    p.add_argument("--gen-micro-batch", type=int, default=None,
                   help="Max sequences each GPU holds per generate() call. The "
                        "worker loops in chunks of this size until the group's "
                        "rollouts are done, so group_size can be anything while "
                        "per-GPU KV memory stays bounded by this. With vLLM this "
                        "sets max_num_seqs. 0 lets the backend choose.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="Fraction of each generation GPU reserved by vLLM's "
                        "weights and KV cache (default: 0.9).")
    p.add_argument("--vllm-enforce-eager", dest="vllm_enforce_eager",
                   action="store_const", const=True, default=None,
                   help="Disable CUDA graphs in vLLM.")
    p.add_argument("--no-vllm-enforce-eager", dest="vllm_enforce_eager",
                   action="store_const", const=False)
    p.add_argument("--vllm-prefix-caching", dest="vllm_enable_prefix_caching",
                   action="store_const", const=True, default=None)
    p.add_argument("--no-vllm-prefix-caching",
                   dest="vllm_enable_prefix_caching",
                   action="store_const", const=False)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    p.add_argument("--vllm-pipeline-parallel-size", type=int, default=None)
    p.add_argument("--vllm-quantization", type=str, default=None,
                   help="Explicit vLLM quantizer; omit/auto for checkpoint-native.")
    p.add_argument("--vllm-max-num-batched-tokens", type=int, default=None)
    p.add_argument("--vllm-enable-expert-parallel",
                   dest="vllm_enable_expert_parallel",
                   action="store_const", const=True, default=None)
    p.add_argument("--no-vllm-enable-expert-parallel",
                   dest="vllm_enable_expert_parallel",
                   action="store_const", const=False)

    # ---- memory (Sec. 2.2) ----
    p.add_argument("--memory-version", type=lambda value: value.upper(),
                   choices=["V1", "V2"], default=None,
                   help="Memory implementation: V1 preserves historical "
                        "behavior; V2 enables corrected causal memory.")
    p.add_argument("--memory", dest="memory", action="store_const",
                   const=True, default=None,
                   help="Master switch for the memory module. Every other "
                        "--memory-* flag is ignored unless this is set.")
    p.add_argument("--no-memory", dest="memory", action="store_const",
                   const=False,
                   help="Force memory off, overriding the YAML.")
    p.add_argument("--memory-lookup-mode",
                   choices=["select", "all", "none"], default=None,
                   help="select = the model picks ids from the index (one extra "
                        "call per step); all = inject the whole bank; none = "
                        "never inject.")
    p.add_argument("--memory-lookup-max-select", type=int, default=None)
    p.add_argument("--memory-lookup-fallback",
                   choices=["none", "recent", "importance"], default=None)
    p.add_argument("--memory-catalog-max-lessons", type=int, default=None)
    p.add_argument("--memory-token-budget", type=int, default=None)
    p.add_argument("--memory-arm-control-fraction", type=float, default=None,
                   help="Share of each existing group generated without memory.")
    p.add_argument("--memory-arm-explore-fraction", type=float, default=None,
                   help="Share of each group assigned to an under-tested lesson.")
    p.add_argument("--memory-arm-max-lessons", type=int, default=None,
                   help="Maximum lessons placed together in one causal arm.")
    p.add_argument("--memory-arm-exploration-c", type=float, default=None,
                   help="UCB uncertainty weight for the exploratory memory arm.")
    p.add_argument("--memory-arm-comparison-n", type=int, default=None,
                   help="V2 fixed best-of-n comparison budget. 0 derives it "
                        "from the initial group size and arm fractions.")
    p.add_argument("--memory-outcome-credit", action="store_const",
                   const=True, default=None,
                   help="Credit lessons from matched best@K uplift vs null arms.")
    p.add_argument("--memory-no-text-reinforce", dest="memory_text_reinforce",
                   action="store_const", const=False, default=None,
                   help="Do not treat LLM paraphrase/confirmation as evidence.")
    p.add_argument("--memory-extract-mode",
                   choices=["contrast", "split"], default=None,
                   help="contrast = one call over successes and failures "
                        "together, asked why some worked and others did not.")
    p.add_argument("--memory-curate-every", type=int, default=None,
                   help="Rewrite the whole bank every N steps. 0 disables.")
    p.add_argument("--memory-curate-max-items", type=int, default=None)
    p.add_argument("--memory-extract-from",
                   choices=["both", "failure", "success"], default=None,
                   help="Which side of the batch produces lessons. 'failure' "
                        "skips the positive call entirely: one extraction call "
                        "per step instead of two.")
    p.add_argument("--memory-failures-only", dest="memory_extract_from",
                   action="store_const", const="failure", default=None,
                   help="Shorthand for --memory-extract-from failure.")
    p.add_argument("--memory-lessons-per-call", type=int, default=None)
    p.add_argument("--memory-require-full-lessons", action="store_const",
                   const=True, default=None)
    p.add_argument("--memory-max-examples-per-call", type=int, default=None)
    p.add_argument("--memory-reinforce-delta", type=float, default=None)
    p.add_argument("--memory-max-new-tokens", type=int, default=None)
    p.add_argument("--memory-max-code-lines", type=int, default=None)
    p.add_argument("--memory-allow-constructions", dest="memory_forbid_constructions",
                   action="store_const", const=False, default=None,
                   help="Disable the construction guard. Not recommended: this "
                        "is what let one coordinate formula reach 99%% of "
                        "programs and cap the run.")
    p.add_argument("--memory-dedup-jaccard", type=float, default=None)
    p.add_argument("--memory-inject-mode",
                   choices=["append", "system"], default=None)

    # ---- feedback signal (Sec. 2.3) ----
    p.add_argument("--feedback", dest="feedback", action="store_const",
                   const=True, default=None,
                   help="Master switch for the feedback-based failure signal. "
                        "Every other --feedback-* flag is ignored unless this is set.")
    p.add_argument("--no-feedback", dest="feedback", action="store_const",
                   const=False,
                   help="Force the feedback signal off, overriding the YAML.")
    p.add_argument("--feedback-lambda", type=float, default=None)
    p.add_argument("--feedback-anneal-steps", type=int, default=None,
                   help="Anneal lambda_f to feedback_lambda_final over this many "
                        "steps. 0 keeps it constant. Once the coefficient hits "
                        "zero the teacher forward is skipped entirely.")
    p.add_argument("--feedback-anneal-shape",
                   choices=["linear", "cosine"], default=None)
    p.add_argument("--feedback-lambda-final", type=float, default=None)
    p.add_argument("--feedback-clip", type=float, default=None)
    p.add_argument("--feedback-chars", type=int, default=None)
    p.add_argument("--feedback-max-per-step", type=int, default=None,
                   help="Code-failure teacher cap: 0 = auto from G*K, "
                        "-1 = all, >0 = fixed override.")
    p.add_argument("--feedback-auto-fraction", type=float, default=None,
                   help="Automatic teacher budget as a fraction of current G*K "
                        "(default: 0.20).")
    p.add_argument("--feedback-inject-mode",
                   choices=["append", "user_turn"], default=None)
    p.add_argument("--feedback-normalize", action="store_const",
                   const=True, default=None)
    p.add_argument("--feedback-adaptive", action="store_const",
                   const=True, default=None,
                   help="Gate feedback by the observed code-valid rate.")
    p.add_argument("--feedback-validity-floor", type=float, default=None)
    p.add_argument("--feedback-validity-target", type=float, default=None)
    p.add_argument("--feedback-max-reward-ratio", type=float, default=None,
                   help="Bound mean feedback advantage relative to reward advantage.")
    p.add_argument("--feedback-reward-scale-floor", type=float, default=None,
                   help="Nonzero reward scale used to repair constant-failure groups.")
    p.add_argument("--feedback-max-per-signature", type=int, default=None,
                   help="Per-failure-class cap: 0 = auto from the step cap, "
                        "-1 = unlimited, >0 = fixed override.")
    p.add_argument("--feedback-auto-signature-fraction", type=float, default=None,
                   help="Automatic per-signature cap as a fraction of the step "
                        "cap (default: 0.25).")

    return p


# CLI arg name -> config key (only where they differ)
_CLI_TO_CFG = {"lr": "learning_rate"}


def _parse_gpu_ids(value) -> list:
    """Parse and validate the physical GPU list without importing CUDA."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"gpu_ids must be comma-separated integers: {value!r}") from exc
    if any(gpu_id < 0 for gpu_id in ids):
        raise ValueError("gpu_ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError("gpu_ids must not contain duplicates")
    return ids


def _resolve_training_model_name(model_name, requested_name, load_in_4bit):
    """Keep inference and trainable checkpoint choices independent.

    OpenAI GPT-OSS is distributed in inference-oriented MXFP4. Transformers
    cannot train that quantizer and dequantizes it to BF16 when its kernels are
    unavailable. For QLoRA, use Unsloth's trainable BitsAndBytes conversion
    while leaving vLLM pointed at the configured original checkpoint.
    """
    if str(requested_name or "").strip():
        return str(requested_name).strip()
    generation_name = str(model_name).strip()
    if not bool(load_in_4bit):
        return generation_name
    lowered = generation_name.rstrip("/").lower()
    basename = lowered.rsplit("/", 1)[-1]
    if "unsloth-bnb-4bit" in lowered:
        return generation_name
    aliases = {
        "gpt-oss-20b": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "gpt-oss-120b": "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    }
    return aliases.get(basename, generation_name)


def _resolve_training_backend(backend, training_model_name):
    """Use the loader that matches GPT-OSS's Unsloth BNB tensor layout.

    The ``*-unsloth-bnb-4bit`` GPT-OSS conversions store every expert as a
    separate quantized module.  Vanilla Transformers expects fused expert
    tensors instead; it reports every checkpoint expert as unexpected,
    initializes a second BF16 expert bank, and can OOM while PEFT casts those
    replacement tensors.  Unsloth installs the matching model implementation
    before loading the checkpoint.
    """
    resolved = str(backend).strip().lower()
    checkpoint = str(training_model_name).strip().lower()
    requires_unsloth = (
        "gpt-oss" in checkpoint
        and "unsloth-bnb-4bit" in checkpoint
    )
    if requires_unsloth and resolved in ("auto", "hf"):
        return "unsloth"
    return resolved


def _resolve_training_memory_budgets(training_gpu_ids, memory, *,
                                     max_fraction=0.90):
    """Reserve host/runtime headroom and cap checkpoint placement per card."""
    if not memory or any(gpu_id not in memory for gpu_id in training_gpu_ids):
        return []
    budgets = []
    for gpu_id in training_gpu_ids:
        gpu = memory[gpu_id]
        # Keep at least 6 GiB outside Accelerate's weight placement for CUDA,
        # activations, temporary quantization buffers, and LoRA optimizer state.
        usable = min(
            float(gpu.total_gib) * float(max_fraction),
            float(gpu.free_gib) - 6.0)
        budgets.append(round(max(1.0, usable), 1))
    return budgets


def _validate_known_training_capacity(training_model_name, budgets):
    """Fail before loading a known giant checkpoint that cannot fit at all."""
    if not budgets:
        return
    name = str(training_model_name).lower()
    required_gib = None
    precision = ""
    if "gpt-oss-120b" in name and "bnb-4bit" in name:
        required_gib, precision = 61.0, "trainable 4-bit"
    elif "gpt-oss-20b" in name and "bnb-4bit" in name:
        required_gib, precision = 14.0, "trainable 4-bit"
    elif "gpt-oss-120b" in name:
        required_gib, precision = 196.0, "BF16 LoRA"
    if required_gib is not None and sum(budgets) < required_gib:
        raise ValueError(
            f"{training_model_name} needs about {required_gib:.0f} GiB for "
            f"{precision}, but the selected training GPUs have only "
            f"{sum(budgets):.1f} GiB of safe aggregate placement budget "
            f"{budgets}. Add GPUs or use load_in_4bit with the trainable "
            "GPT-OSS BitsAndBytes checkpoint.")


def load_config():
    """
    Load one complete problem YAML, then overlay resume state and explicit CLI.

    Returns (cfg, merged) where:
      cfg    is the attribute-style view of the fully merged YAML, and
      merged is the full dict (including problem-only keys like num_circles,
             problem_type, budget_s, score_scale, gpu_type, task_yaml, lib_dir),
             which is what the problem registry consumes.
    """
    args = _build_arg_parser().parse_args()

    # Read the saved run identity early enough to select its YAML. New runs
    # persist the complete merged config; older ones can recover standard
    # problem YAMLs by name.
    resume_dir = None
    saved = {}
    if args.resume is not None:
        resume_dir = Path(args.resume).expanduser().resolve()
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"--resume directory not found: {resume_dir}")
        saved_config_path = resume_dir / "config.json"
        if not saved_config_path.is_file():
            raise FileNotFoundError(
                f"--resume directory has no config.json: {resume_dir}"
            )
        saved = json.loads(saved_config_path.read_text())

    # The launcher defaults to erdos. Direct Python calls use the same selection
    # when neither --config nor --problem is provided.
    problem_name = (saved.get("problem") if saved
                    else (args.problem if args.problem is not None
                          else "erdos"))

    # 1) Complete, self-contained problem YAML.
    config_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = args.config
    if cfg_path is None and saved.get("problem_type"):
        typed = config_dir / f"{problem_name}_{saved['problem_type']}.yaml"
        if typed.exists():
            cfg_path = str(typed)
    cfg_path = cfg_path or str(config_dir / f"{problem_name}.yaml")
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"complete problem config not found: {cfg_path}")
    with open(cfg_path) as f:
        ydict = yaml.safe_load(f) or {}
    if not isinstance(ydict, dict) or not ydict:
        raise ValueError(f"problem config must be a non-empty mapping: {cfg_path}")
    from config_validation import validate_problem_config
    validate_problem_config(
        ydict,
        source=cfg_path,
        require_complete=(
            Path(cfg_path).expanduser().resolve().parent == config_dir.resolve()
        ),
    )
    merged = dict(ydict)
    print(f"[config] loaded {cfg_path}")

    # The registry routing key is the YAML's `problem` field when present
    # (this lets e.g. configs/gpu_mode_trimul.yaml declare `problem: gpu_mode`
    # while --problem just selects the file). With no YAML, --problem is the key.
    merged["problem"] = ydict.get("problem", problem_name)

    # 2) Saved config overlay. Older code wrote max_seq_length after adding the
    # memory allowance; undo that convention before main() adds it again.
    if saved:
        marker = saved.pop("_max_seq_length_includes_memory_topup", None)
        if (marker is None and saved.get("memory")
                and saved.get("memory_grant_context", True)
                and saved.get("memory_token_budget", 0)):
            saved["max_seq_length"] = max(
                1,
                int(saved.get("max_seq_length", 0))
                - int(saved.get("memory_token_budget", 0)),
            )
        merged.update(saved)
        print(f"[config] resuming original configuration from "
              f"{resume_dir / 'config.json'}")

    # 3) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type", "resume"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
    if args.rank_clip_epsilon is not None:
        if args.rank_clip_epsilon_low is None:
            merged["rank_clip_epsilon_low"] = args.rank_clip_epsilon
        if args.rank_clip_epsilon_high is None:
            merged["rank_clip_epsilon_high"] = args.rank_clip_epsilon
    # These modes are deliberately launch-scoped. Saved/YAML values must not
    # silently change a later invocation's trainer or evaluator.
    merged["fast"] = bool(args.fast)
    merged["isolate_eval"] = bool(args.isolate_eval)
    if args.problem_type is not None:
        merged["problem_type"] = args.problem_type

    # vLLM is an inference engine, not a differentiable trainer. Treat the
    # convenient `--backend vllm` spelling as the complete no-Unsloth mode.
    if str(merged["backend"]).lower() == "vllm":
        merged["backend"] = "hf"
        merged["generation_backend"] = "vllm"
        print("[config] vLLM mode: HF+PEFT training + vLLM generation")

    generation_backend = str(merged["generation_backend"]).lower()
    if generation_backend not in ("hf", "vllm"):
        raise ValueError("generation_backend must be 'hf' or 'vllm'")
    merged["generation_backend"] = generation_backend
    advantage_mode = str(merged.get("advantage_mode") or "entropic").lower()
    if advantage_mode not in ADVANTAGE_MODES:
        raise ValueError(f"advantage_mode must be one of {ADVANTAGE_MODES}")
    merged["advantage_mode"] = advantage_mode
    cvar_alpha = merged.get("cvar_alpha")
    cvar_alpha = CVAR_ALPHA_DEFAULT if cvar_alpha is None else float(cvar_alpha)
    if not (0.0 < cvar_alpha < 1.0):
        raise ValueError("cvar_alpha must be in (0, 1)")
    merged["cvar_alpha"] = cvar_alpha
    cvar_lambda = merged.get("cvar_lambda")
    cvar_lambda = CVAR_LAMBDA_DEFAULT if cvar_lambda is None else float(cvar_lambda)
    if not (0.0 <= cvar_lambda <= 1.0):
        raise ValueError("cvar_lambda must be in [0, 1]")
    merged["cvar_lambda"] = cvar_lambda
    _resolve_rank_options(merged)
    train_microbatch = int(merged["train_examples_per_microbatch"])
    if train_microbatch < 1:
        raise ValueError("train_examples_per_microbatch must be >= 1")
    merged["train_examples_per_microbatch"] = train_microbatch
    raw_tp = merged["vllm_tensor_parallel_size"]
    tp = (0 if str(raw_tp or "").strip().lower() in ("", "auto")
          else int(raw_tp))
    pp = int(merged["vllm_pipeline_parallel_size"] or 1)
    if tp < 0 or pp < 1:
        raise ValueError("vLLM TP must be >= 0 and PP must be >= 1")

    # Omitted growth caps mean fixed batch size. Resolve after YAML and CLI so
    # `--groups-per-step 5 --group-size 16` becomes max G=5, max K=16 even if
    # the selected problem YAML has different starting values.
    if args.groups_per_step is not None and args.max_groups_per_step is None:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if args.group_size is not None and args.max_group_size is None:
        merged["max_group_size"] = int(merged["group_size"])
    if merged["max_groups_per_step"] is None:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if merged["max_group_size"] is None:
        merged["max_group_size"] = int(merged["group_size"])
    if int(merged["max_groups_per_step"]) < int(merged["groups_per_step"]):
        raise ValueError("max_groups_per_step cannot be below groups_per_step")
    if int(merged["max_group_size"]) < int(merged["group_size"]):
        raise ValueError("max_group_size cannot be below group_size")

    # Resolve V2's fixed comparison budget before config.json is written. This
    # makes resume semantics explicit and fails impossible arm designs before
    # any GPU discovery or model loading.
    from memory import MemoryConfig
    memory_preview = MemoryConfig.from_dict(merged, verbose=False)
    merged["memory_version"] = memory_preview.version
    merged["memory_arm_comparison_n"] = memory_preview.arm_comparison_n

    # Resolve every physical role from one ordered inventory. run.sh exports
    # AVAILABLE_GPUS and that environment value is authoritative over old YAML
    # and resumed role fields. Direct Python invocations fall back to the legacy
    # inventory key, CUDA visibility, then one training device.
    from gpu_runtime import (allocate_gpu_roles,
                             derive_vllm_parallel_layout,
                             detect_attention_heads, parse_gpu_ids,
                             query_gpu_memory, resolve_memory_settings,
                             validate_attention_heads, validate_selected_gpus)

    inventory_source = "AVAILABLE_GPUS"
    inventory_value = os.environ.get("AVAILABLE_GPUS")
    if inventory_value is None:
        inventory_source = "legacy fallback"
        inventory_value = (args.available_gpu_ids
                           if args.available_gpu_ids is not None
                           else merged.get("available_gpu_ids"))
        if not str(inventory_value or "").strip():
            inventory_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not str(inventory_value or "").strip():
            inventory_value = str(int(merged.get("training_gpu_id", 0)))
        print("[config] warning: AVAILABLE_GPUS is unset; use run.sh so the "
              "ordered physical inventory is authoritative")

    available_gpu_ids = parse_gpu_ids(
        inventory_value, field=inventory_source)
    roles = allocate_gpu_roles(available_gpu_ids, merged["problem"])
    if merged["isolate_eval"]:
        if roles.gpu_problem:
            raise ValueError(
                "--isolate-eval is for CPU sandbox problems; gpu_mode uses "
                "its dedicated serialized GPU benchmark evaluator")
        if int(merged.get("reward_workers", 0) or 0) < 1:
            raise ValueError(
                "--isolate-eval requires reward_workers >= 1; in this mode "
                "reward_workers is the concurrent process count per CPU")
        # Fail before model loading when the host cannot enforce affinity.
        _isolated_evaluation_cpu_ids()
    training_gpu_id = roles.training
    gpu_ids = roles.generation
    # Training and rollout phases alternate residency, so every rollout card
    # can also hold a shard of the trainable model. In gpu_mode the exclusive
    # evaluation card is absent from roles.generation and remains untouched.
    training_gpu_ids = list(gpu_ids)
    evaluation_gpu_id = roles.evaluation

    # Explicit CLI role flags are accepted only as assertions. They may not
    # silently override the launcher inventory.
    assertions = [
        ("--training-gpu-id", args.training_gpu_id, training_gpu_id),
        ("--evaluation-gpu-id", args.evaluation_gpu_id, evaluation_gpu_id),
        ("--num-gpus", args.num_gpus, len(gpu_ids)),
    ]
    for flag, actual, expected in assertions:
        if actual is not None and actual != expected:
            raise ValueError(
                f"{flag}={actual} conflicts with {inventory_source}; "
                f"derived value is {expected}")
    if args.gpu_ids is not None:
        asserted = parse_gpu_ids(args.gpu_ids, field="--gpu-ids")
        if asserted != gpu_ids:
            raise ValueError(
                f"--gpu-ids={asserted} conflicts with {inventory_source}; "
                f"derived generation group is {gpu_ids}")
    if args.kernel_gpu_id is not None and args.kernel_gpu_id != evaluation_gpu_id:
        raise ValueError(
            f"--kernel-gpu-id={args.kernel_gpu_id} conflicts with "
            f"{inventory_source}; derived evaluation GPU is {evaluation_gpu_id}")

    merged["training_gpu_id"] = training_gpu_id
    merged["training_gpu_ids"] = ",".join(
        str(x) for x in training_gpu_ids)
    merged["num_training_gpus"] = len(training_gpu_ids)
    merged["available_gpu_ids"] = ",".join(str(x) for x in roles.available)
    merged["gpu_ids"] = ",".join(str(x) for x in gpu_ids)
    merged["num_gpus"] = len(gpu_ids)
    merged["evaluation_gpu_id"] = evaluation_gpu_id
    merged["sequential_generation"] = roles.sequential_generation
    merged["evaluation_shares_generation"] = roles.evaluation_shares_generation
    merged["reserve_last_gpu_for_evaluation"] = bool(
        roles.gpu_problem and evaluation_gpu_id != training_gpu_id)

    if roles.gpu_problem:
        if not str(merged.get("gpu_type") or "").strip():
            merged["gpu_type"] = "H100"
            print("[config] gpu_mode gpu_type not set; defaulting explicitly to H100")
        merged["kernel_gpu_id"] = evaluation_gpu_id
        if int(merged.get("reward_workers") or 0) != 1:
            print("[config] gpu_mode forces reward_workers=1 for serialized, "
                  "stable benchmark measurements")
        merged["reward_workers"] = 1

    memory = query_gpu_memory()
    validate_selected_gpus(roles, memory)

    merged["training_model_name"] = _resolve_training_model_name(
        merged.get("model_name", ""), merged.get("training_model_name"),
        merged.get("load_in_4bit", False),
    )
    if merged["training_model_name"] != merged.get("model_name"):
        print(f"[precision] trainable checkpoint: "
              f"{merged['training_model_name']} (generation remains "
              f"{merged.get('model_name')})")
    requested_backend = str(merged["backend"]).strip().lower()
    merged["backend"] = _resolve_training_backend(
        requested_backend, merged["training_model_name"])
    if merged["backend"] != requested_backend:
        print("[backend] GPT-OSS Unsloth BNB checkpoints require the patched "
              f"expert loader; switching {requested_backend} -> "
              f"{merged['backend']}")
    training_budgets = _resolve_training_memory_budgets(
        training_gpu_ids, memory,
        max_fraction=(0.80 if merged["fast"] else 0.90))
    _validate_known_training_capacity(
        merged["training_model_name"], training_budgets)
    merged["training_max_memory_gib"] = training_budgets
    if training_budgets:
        print(f"[memory] training weight budgets by logical GPU: "
              f"{training_budgets} GiB")

    if merged["fast"]:
        if merged["advantage_mode"] != "rank":
            raise ValueError("--fast currently requires --advantage-mode rank")
        if int(merged["rank_update_epochs"]) != 1:
            raise ValueError("--fast requires rank_update_epochs=1")
        if not (int(merged["num_training_gpus"]) > 1
                and merged["backend"] == "hf"
                and merged["generation_backend"] == "vllm"):
            raise ValueError(
                "--fast requires replicated HF LoRA training with vLLM "
                "generation on at least two training GPUs")

    # Consume every rollout GPU. Prefer compatible TP replicas for throughput.
    # If a complete model plus one full-context request would not fit per
    # replica, turn the replica factor into PP so the weights and KV cache are
    # sharded across the exact same GPU inventory.
    if generation_backend == "vllm":
        known_heads = detect_attention_heads(merged.get("model_name", ""))
        layout = derive_vllm_parallel_layout(
            merged, roles, memory, known_heads)
        merged["vllm_tensor_parallel_size"] = layout.tensor_parallel_size
        merged["vllm_pipeline_parallel_size"] = layout.pipeline_parallel_size
        validate_attention_heads(
            known_heads,
            merged["vllm_tensor_parallel_size"],
            merged.get("model_name", ""),
        )
        if layout.pipeline_parallel_size > 1:
            print(f"[config] compatible TP={layout.tensor_parallel_size} "
                  f"replicas need about "
                  f"{layout.unsharded_stage_required_gib:.1f} GiB/GPU, above "
                  f"the {layout.budget_gib:.1f} GiB vLLM budget; using one "
                  f"sharded engine with TP={layout.tensor_parallel_size}, "
                  f"PP={layout.pipeline_parallel_size}")
        elif layout.replicas > 1:
            print(f"[config] {len(gpu_ids)} rollout GPUs form {layout.replicas} "
                  f"parallel vLLM replicas at compatible "
                  f"TP={merged['vllm_tensor_parallel_size']}")
        elif layout.tensor_parallel_size > 1:
            print(f"[config] using one vLLM engine with "
                  f"TP={layout.tensor_parallel_size}; selected layout needs "
                  f"about {layout.unsharded_stage_required_gib:.1f} GiB/GPU "
                  f"within the {layout.budget_gib:.1f} GiB budget")

    for note in resolve_memory_settings(merged, roles, memory):
        print(f"[memory] auto: {note}")

    print(f"[config] AVAILABLE_GPUS={roles.available} ({inventory_source})")
    print(f"[config] GPU roles: train={training_gpu_ids}, "
          f"generation={gpu_ids}, evaluation={evaluation_gpu_id}; "
          f"sharing={'sequential' if roles.sequential_generation else 'isolated'}")

    # 4) Provide attribute access after the YAML's shared and problem-specific
    # keys have passed their ownership contract.
    merged["target_modules"] = tuple(merged["target_modules"])
    cfg = SimpleNamespace(**merged)
    if cfg.generation_backend == "vllm" and int(cfg.num_gpus or 0) < 1:
        raise ValueError("vLLM generation requires num_gpus >= 1")
    if cfg.generation_backend == "vllm":
        resolved_tp = int(cfg.vllm_tensor_parallel_size or 0)
        pp = int(cfg.vllm_pipeline_parallel_size)
        if resolved_tp == 0:
            if int(cfg.num_gpus) % pp:
                raise ValueError("vLLM generation GPU count must be divisible by PP")
            resolved_tp = int(cfg.num_gpus) // pp
        engine_gpus = resolved_tp * pp
        if engine_gpus < 1 or int(cfg.num_gpus) % engine_gpus:
            raise ValueError(
                f"num_gpus={cfg.num_gpus} cannot form complete vLLM groups of "
                f"TP={resolved_tp} * PP={pp}")
        print(f"[config] vLLM engines: {int(cfg.num_gpus) // engine_gpus} "
              f"replica(s), TP={resolved_tp}, PP={pp}, "
              f"{engine_gpus} GPU(s)/engine")
    print(f"[config] rollout thinking: "
          f"{'enabled' if cfg.thinking else 'disabled'}")
    merged["_resume_dir"] = str(resume_dir) if resume_dir is not None else None
    return cfg, merged


def _pin_training_process(training_gpu_ids) -> None:
    """Expose the ordered training/model-parallel group before CUDA imports."""
    ids = _parse_gpu_ids(training_gpu_ids)
    if not ids:
        raise ValueError("training_gpu_ids must contain at least one GPU")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    visible = ",".join(str(gpu_id) for gpu_id in ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["TTT_TRAINING_GPU_ID"] = str(ids[0])
    os.environ["TTT_TRAINING_GPU_IDS"] = visible
    logical = ",".join(f"cuda:{idx}" for idx in range(len(ids)))
    print(f"[gpu] trainer model-parallel group: physical {ids} "
          f"(logical {logical})")


def _move_optimizer_state(optimizer, device) -> None:
    """Move Adam state recursively without replacing Parameter identities."""
    import torch

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def _restore_optimizer_state_to_parameters(optimizer) -> None:
    """Return each optimizer tensor to the device of its sharded parameter."""
    import torch

    def move(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state, parameter.device)


def _attention_head_count(model) -> int:
    cfg = getattr(model, "config", None)
    for _ in range(3):
        if cfg is None:
            return 0
        for name in ("num_attention_heads", "n_head", "num_heads"):
            value = getattr(cfg, name, None)
            if value:
                return int(value)
        cfg = getattr(cfg, "text_config", None)
    return 0


# ======================================================================
# Generation
# ======================================================================
def _generate_batch(model, tokenizer, inputs, input_len, n_samples, cfg):
    """
    Generate n_samples completions for a SINGLE prompt in ONE batched
    model.generate() call (via num_return_sequences). Returns a list of
    (text, gen_token_ids).
    """
    import torch
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    max_new_tokens = min(
        int(cfg.max_new_tokens), int(cfg.max_seq_length) - int(input_len))
    if max_new_tokens < 1:
        return [("", []) for _ in range(int(n_samples))]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            **({"top_k": 0, "repetition_penalty": 1.0}
               if getattr(cfg, "advantage_mode", "entropic") == "rank" else {}),
            pad_token_id=pad_id,
            num_return_sequences=n_samples,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((text, gen_ids))
    return results


def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg):
    """
    Generate `group_size` responses from a single prompt, batched.

    Try to generate all `group_size` at once. If OOMs, halve the
    per-call batch size and retry, accumulating until we have group_size
    responses. This keeps the algorithm identical (still group_size IID
    samples from the same policy) while using the GPU in parallel.

    Returns (list of (text, gen_token_ids), prompt_len_in_tokens).
    """
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    responses = []
    remaining = group_size
    # Start by trying the whole group in one call. Cap the first attempt at the
    # micro-batch size when set, so the single-GPU path honors the same per-call
    # ceiling as the multi-GPU workers; OOM halving still applies below it.
    mb = int(getattr(cfg, "gen_micro_batch", 0) or 0)
    batch = group_size if (mb <= 0 or mb > group_size) else mb

    while remaining > 0:
        n = min(batch, remaining)
        try:
            chunk = _generate_batch(model, tokenizer, inputs, input_len, n, cfg)
            responses.extend(chunk)
            remaining -= n
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                # Can't even do one — re-raise, nothing we can do
                raise
            batch = max(1, batch // 2)
            print(f"  [oom] halving generation batch size to {batch}")

    return responses, input_len


def generate_prompt_jobs(model, tokenizer, prompts_by_group, counts_by_group,
                         cfg, *, max_new_tokens=None, temperature=None,
                         top_p=None, cap_state=None):
    """Stream locally generated rollouts from cross-prompt HF batches."""
    if len(prompts_by_group) != len(counts_by_group):
        raise ValueError("counts_by_group must align with prompts_by_group")
    if getattr(cfg, "advantage_mode", "entropic") == "rank":
        # This local path owns its sampling arguments, including top_k=0.
        # Use the existing per-prompt OOM-aware micro-batcher instead of an
        # external HF helper that may inherit a top-k generation default.
        rank_cfg = SimpleNamespace(**vars(cfg))
        rank_cfg.temperature = 1.0
        rank_cfg.top_p = 1.0
        if max_new_tokens is not None:
            rank_cfg.max_new_tokens = int(max_new_tokens)
        for group_idx, (prompt, count) in enumerate(zip(prompts_by_group, counts_by_group)):
            if int(count) > 0:
                responses, _ = generate_responses(
                    model, tokenizer, prompt, int(count), rank_cfg)
                yield group_idx, responses
        return

    from gen_workers import _iter_hf_job_batches
    jobs = [
        (group_idx, prompt, int(count))
        for group_idx, (prompt, count)
        in enumerate(zip(prompts_by_group, counts_by_group))
        if int(count) > 0
    ]
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens if max_new_tokens is not None
                              else cfg.max_new_tokens),
        "temperature": float(temperature if temperature is not None
                             else cfg.temperature),
        "top_p": float(top_p if top_p is not None else cfg.top_p),
        "micro_batch": int(getattr(cfg, "gen_micro_batch", 0) or 0),
    }
    yield from _iter_hf_job_batches(
        model, tokenizer, jobs, model.device, int(cfg.max_seq_length),
        gen_kwargs, cap_state=cap_state, log_prefix="trainer rollout")


# ======================================================================
# Logprob computation
# ======================================================================
def _causal_lm_decoder_and_head(model):
    """Return the decoder and output head without bypassing injected LoRA."""
    causal_lm = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            causal_lm = getter()
        except (AttributeError, TypeError):
            return None
    head_getter = getattr(causal_lm, "get_output_embeddings", None)
    if not callable(head_getter):
        return None
    head = head_getter()
    prefix = str(getattr(causal_lm, "base_model_prefix", "") or "")
    decoder = getattr(causal_lm, prefix, None) if prefix else None
    if decoder is None:
        for name in ("model", "transformer", "backbone"):
            candidate = getattr(causal_lm, name, None)
            if candidate is not None and candidate is not causal_lm:
                decoder = candidate
                break
    if decoder is None or decoder is causal_lm or head is None:
        return None
    return decoder, head


def _score_hidden_token_chunks(hidden_states, targets, output_head, *,
                               chunk, with_grad, return_entropy):
    """Project hidden states in checkpointed vocabulary-sized chunks.

    CausalLM.forward normally materializes logits for every token at once.
    For a 32k sequence and a 150k vocabulary that tensor alone is enormous.
    This exact path retains only the selected logprob/entropy vectors and
    recomputes each short output-head chunk during backward.
    """
    import torch
    import torch.nn.functional as F

    length = int(targets.shape[0])
    step = int(chunk or 0)
    if step < 1:
        step = min(length, 256)

    def project(hidden_chunk, target_chunk):
        logits = output_head(hidden_chunk).float()
        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(
            1, target_chunk.unsqueeze(-1)).squeeze(-1)
        if not return_entropy:
            return chosen
        safe = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
        entropy = -(log_probs.exp() * safe).sum(dim=-1)
        return chosen, entropy

    chosen_parts = []
    entropy_parts = []
    for start in range(0, length, step):
        end = min(start + step, length)
        hidden_chunk = hidden_states[start:end]
        target_chunk = targets[start:end]
        if with_grad and hidden_chunk.requires_grad:
            from torch.utils.checkpoint import checkpoint
            result = checkpoint(
                project, hidden_chunk, target_chunk, use_reentrant=False)
        else:
            result = project(hidden_chunk, target_chunk)
        if return_entropy:
            chosen, entropy = result
            chosen_parts.append(chosen)
            entropy_parts.append(entropy)
        else:
            chosen_parts.append(result)
    chosen = torch.cat(chosen_parts, dim=0)
    if return_entropy:
        return chosen, torch.cat(entropy_parts, dim=0)
    return chosen


def _decoder_token_logprobs(model, input_ids, attention_mask,
                            response_ranges, response_targets, *, chunk,
                            with_grad, return_entropy):
    """Memory-bounded exact scoring, or None for an unsupported model shape."""
    components = _causal_lm_decoder_and_head(model)
    if components is None:
        return None
    decoder, output_head = components
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        try:
            hidden = outputs[0]
        except (IndexError, KeyError, TypeError):
            return None
    scored = []
    entropies = []
    for row, ((start, end), targets) in enumerate(
            zip(response_ranges, response_targets)):
        result = _score_hidden_token_chunks(
            hidden[row, start:end, :], targets, output_head,
            chunk=chunk, with_grad=with_grad,
            return_entropy=return_entropy)
        if return_entropy:
            chosen, entropy = result
            scored.append(chosen)
            entropies.append(entropy)
        else:
            scored.append(result)
    return (scored, entropies) if return_entropy else scored


def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool,
                           chunk: int = 0, *, return_entropy: bool = False):
    """
    Per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs
    return_entropy=True returns (logprobs, token_entropies), both shape (R,).
    Entropies integrate over the full vocabulary, not just sampled tokens.

    The decoder runs once for exact causal attention. Its output head and
    log-softmax run in checkpointed `chunk` slices so full-sequence vocabulary
    logits are never retained. Unsupported model wrappers use the compatible
    full-forward fallback.
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    P = prompt_ids.shape[1]
    R = response_ids.shape[1]
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        bounded = _decoder_token_logprobs(
            model, full_ids, None,
            [(int(P) - 1, int(P) - 1 + int(R))],
            [response_ids[0]], chunk=chunk, with_grad=with_grad,
            return_entropy=return_entropy)
        if bounded is not None:
            if return_entropy:
                logprobs, entropies = bounded
                return logprobs[0], entropies[0]
            return bounded[0]

        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        # Predict response token at position P+k from logits at position P+k-1.
        pred_logits = logits[:, P - 1 : P - 1 + R, :]  # (1, R, V)

        if chunk and 0 < chunk < R:
            parts = []
            entropies = []
            for s in range(0, R, chunk):
                e = min(s + chunk, R)
                lp = F.log_softmax(pred_logits[:, s:e, :].float(), dim=-1)
                g = lp.gather(2, response_ids[:, s:e].unsqueeze(-1)).squeeze(-1)
                parts.append(g)          # keep only (1, e-s); lp freed next iter
                if return_entropy:
                    safe_lp = lp.masked_fill(~torch.isfinite(lp), 0.0)
                    entropies.append(-(lp.exp() * safe_lp).sum(dim=-1))
            gathered = torch.cat(parts, dim=1)  # (1, R)
            if return_entropy:
                entropy = torch.cat(entropies, dim=1)
        else:
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            gathered = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (1, R)
            if return_entropy:
                safe_lp = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
                entropy = -(log_probs.exp() * safe_lp).sum(dim=-1)
    if return_entropy:
        return gathered.squeeze(0), entropy.squeeze(0)
    return gathered.squeeze(0)


def compute_batched_token_logprobs(
        model, examples, with_grad: bool, chunk: int = 0, *,
        pad_token_id: int = 0, return_entropy: bool = False):
    """Compute response log-probabilities for a padded example microbatch.

    Padding is appended after each complete prompt/response sequence and masked,
    so every real token retains the same causal positions as the old one-example
    path.  A singleton deliberately uses compute_token_logprobs() directly,
    preserving the previous behavior for long examples that the planner cannot
    safely combine with another sequence.
    """
    import torch
    import torch.nn.functional as F

    examples = list(examples)
    if not examples:
        return ([], []) if return_entropy else []
    if len(examples) == 1:
        example = examples[0]
        result = compute_token_logprobs(
            model, example["prompt_ids"], example["response_ids"],
            with_grad=with_grad, chunk=chunk, return_entropy=return_entropy)
        if return_entropy:
            logprobs, entropies = result
            return [logprobs], [entropies]
        return [result]

    prompts = [example["prompt_ids"] for example in examples]
    responses = [example["response_ids"] for example in examples]
    for prompt_ids, response_ids in zip(prompts, responses):
        if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1
                or response_ids.ndim != 2 or response_ids.shape[0] != 1):
            raise ValueError(
                "training examples must contain (1, length) token tensors")
        if prompt_ids.shape[1] < 1 or response_ids.shape[1] < 1:
            raise ValueError("training prompt/response tensors must be nonempty")

    device = prompts[0].device
    dtype = prompts[0].dtype
    totals = [int(prompt.shape[1] + response.shape[1])
              for prompt, response in zip(prompts, responses)]
    max_total = max(totals)
    input_ids = torch.full(
        (len(examples), max_total), int(pad_token_id),
        dtype=dtype, device=device)
    attention_mask = torch.zeros(
        (len(examples), max_total), dtype=torch.long, device=device)
    for row, (prompt_ids, response_ids, total) in enumerate(
            zip(prompts, responses, totals)):
        if prompt_ids.device != device or response_ids.device != device:
            raise ValueError("one training microbatch spans multiple devices")
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        input_ids[row, :total] = full_ids[0]
        attention_mask[row, :total] = 1

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        response_ranges = []
        response_targets = []
        for prompt_ids, response_ids in zip(prompts, responses):
            prompt_len = int(prompt_ids.shape[1])
            response_len = int(response_ids.shape[1])
            response_ranges.append(
                (prompt_len - 1, prompt_len - 1 + response_len))
            response_targets.append(response_ids[0])
        bounded = _decoder_token_logprobs(
            model, input_ids, attention_mask, response_ranges,
            response_targets, chunk=chunk, with_grad=with_grad,
            return_entropy=return_entropy)
        if bounded is not None:
            return bounded

        gathered_examples = []
        entropy_examples = []
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        for row, (prompt_ids, response_ids) in enumerate(
                zip(prompts, responses)):
            prompt_len = int(prompt_ids.shape[1])
            response_len = int(response_ids.shape[1])
            pred_logits = logits[
                row, prompt_len - 1:prompt_len - 1 + response_len, :]
            targets = response_ids[0]
            step = (int(chunk) if chunk and 0 < int(chunk) < response_len
                    else response_len)
            parts = []
            entropies = []
            for start in range(0, response_len, step):
                end = min(start + step, response_len)
                log_probs = F.log_softmax(
                    pred_logits[start:end, :].float(), dim=-1)
                parts.append(log_probs.gather(
                    1, targets[start:end].unsqueeze(-1)).squeeze(-1))
                if return_entropy:
                    safe = log_probs.masked_fill(
                        ~torch.isfinite(log_probs), 0.0)
                    entropies.append(
                        -(log_probs.exp() * safe).sum(dim=-1))
            gathered_examples.append(torch.cat(parts, dim=0))
            if return_entropy:
                entropy_examples.append(torch.cat(entropies, dim=0))
    if return_entropy:
        return gathered_examples, entropy_examples
    return gathered_examples


def _training_microbatches(examples, cfg, *, partition_key=None):
    """Length-bucket examples into real model batches.

    train_examples_per_microbatch is an example-count ceiling.  The padded-token
    ceiling keeps one batch at or below max_seq_length total padded tokens, so a
    configured value such as 64 is useful for short responses without forcing
    64 long contexts into memory together.
    """
    example_cap = int(getattr(cfg, "train_examples_per_microbatch", 1) or 0)
    if example_cap < 1:
        raise ValueError("train_examples_per_microbatch must be >= 1")
    token_cap = max(1, int(getattr(cfg, "max_seq_length", 1) or 1))

    partitions = {}
    for example in examples:
        key = partition_key(example) if partition_key is not None else None
        partitions.setdefault(key, []).append(example)

    batches = []
    for partition in partitions.values():
        ordered = sorted(
            partition,
            key=lambda example: int(
                example["prompt_ids"].shape[1]
                + example["response_ids"].shape[1]),
            reverse=True,
        )
        current = []
        current_max = 0
        for example in ordered:
            length = int(example["prompt_ids"].shape[1]
                         + example["response_ids"].shape[1])
            next_max = max(current_max, length)
            exceeds_tokens = bool(
                current and next_max * (len(current) + 1) > token_cap)
            if current and (len(current) >= example_cap or exceeds_tokens):
                batches.append(current)
                current = []
                current_max = 0
            current.append(example)
            current_max = max(current_max, length)
        if current:
            batches.append(current)
    return batches


def _is_cuda_oom(error):
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return (isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
            and "cuda" in str(error).lower())


def _is_attention_kernel_unavailable(error):
    """Recognize SDPA failures caused by excluding quadratic math attention."""
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return (
        "no available kernel" in message
        or "no viable backend for scaled_dot_product_attention" in message
        or "no suitable kernel" in message
        or message.strip() == "invalid backend"
    )


def _split_training_batches(batches):
    """Halve every non-singleton while preserving order and partitioning."""
    split = []
    changed = False
    for batch in batches:
        batch = list(batch)
        if len(batch) <= 1:
            split.append(batch)
            continue
        middle = (len(batch) + 1) // 2
        split.extend((batch[:middle], batch[middle:]))
        changed = True
    return split, changed


def _run_oom_resilient_backward(model, batches, attempt, *, device_label):
    """Retry an update with smaller batches and CPU-saved activations.

    Each attempt starts the local replica's gradients from zero, so an OOM
    during backward cannot double-count a partially accumulated microbatch.
    Once batches reach one example, saved-tensor CPU offload is tried. A truly
    untrainable outlier is quarantined from this update rather than terminating
    the entire multi-hour run; all generated artifacts remain saved.
    """
    import gc
    import torch

    active = [list(batch) for batch in batches if batch]
    quarantined = []
    activation_offload = False
    cuda_devices = sorted({
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda" and parameter.device.index is not None
    })
    while active:
        model.zero_grad(set_to_none=True)
        try:
            offload_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if activation_offload
                and hasattr(torch.autograd.graph, "save_on_cpu")
                else nullcontext()
            )
            with offload_context:
                result = attempt(active)
            return result, active, quarantined
        except BaseException as error:
            cuda_oom = _is_cuda_oom(error)
            kernel_unavailable = _is_attention_kernel_unavailable(error)
            if not cuda_oom and not kernel_unavailable:
                raise
            model.zero_grad(set_to_none=True)
            gc.collect()
            for cuda_device in cuda_devices:
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()
            smaller, changed = _split_training_batches(active)
            if changed:
                active = smaller
                reason = ("fused attention rejected the padded batch"
                          if kernel_unavailable else "CUDA OOM")
                print(f"[train-oom] {device_label}: {reason}; retrying with maximum "
                      f"microbatch {max(len(batch) for batch in active)}",
                      flush=True)
                continue
            if cuda_oom and not activation_offload and hasattr(
                    torch.autograd.graph, "save_on_cpu"):
                activation_offload = True
                print(f"[train-oom] {device_label}: singleton OOM; retrying "
                      "with saved activations offloaded to CPU", flush=True)
                continue

            victim_index = max(
                range(len(active)),
                key=lambda index: int(
                    active[index][0]["prompt_ids"].shape[1]
                    + active[index][0]["response_ids"].shape[1]),
            )
            victim_batch = active.pop(victim_index)
            victim = victim_batch[0]
            victim_tokens = int(
                victim["prompt_ids"].shape[1]
                + victim["response_ids"].shape[1])
            quarantined.append(victim)
            activation_offload = False
            if kernel_unavailable:
                reason = "has no compatible linear-memory attention kernel"
            else:
                reason = "cannot fit even with activation offload"
            print(f"[train-oom] {device_label}: one {victim_tokens}-token "
                  f"rollout {reason}; excluding only that rollout from this "
                  "adapter update", flush=True)
    model.zero_grad(set_to_none=True)
    return None, [], quarantined


def _valid_example_token_logprobs(example, key):
    """Whether a cached vector is finite and aligned to the response."""
    import torch

    value = example.get(key)
    response_len = int(example["response_ids"].shape[1])
    return bool(
        torch.is_tensor(value)
        and value.ndim == 1
        and value.shape[0] == response_len
        and torch.isfinite(value).all()
    )


def _initialize_rank_logprob_caches(examples):
    """Use vLLM rollout/reference scores and identify any HF fallbacks."""
    missing_old = []
    missing_reference = []
    for example in examples:
        old = example.get("behavior_logprobs")
        if _valid_example_token_logprobs(example, "behavior_logprobs"):
            example["rank_old_logprobs"] = old.detach()
        else:
            example["rank_old_logprobs"] = None
            missing_old.append(example)

        reference = example.get("reference_logprobs")
        if _valid_example_token_logprobs(example, "reference_logprobs"):
            example["rank_reference_logprob"] = reference.detach()
        else:
            example["rank_reference_logprob"] = None
            missing_reference.append(example)
        example["rank_feedback_advantage"] = None
    return missing_old, missing_reference


def rank_grpo_loss(current_logprobs, old_logprobs, reference_logprob,
                   advantage, *, clip_epsilon=RANK_CLIP_EPSILON_DEFAULT,
                   clip_epsilon_low=None, clip_epsilon_high=None,
                   kl_coef=0.0, entropy_coef=0.0, token_entropies=None,
                   return_tensor_metrics=False):
    """One trajectory's loss; callers average equally within each parent.

    Per-token rho = exp(current_lp - old_lp) is differentiable; old_lp and the
    scalar group advantage are frozen. The sampled per-token reference-KL
    estimator is exp(log pi_ref - log pi_theta)
                 - (log pi_ref - log pi_theta) - 1.
    Clipping is asymmetric when requested: [1 - epsilon_low,
    1 + epsilon_high]. The legacy clip_epsilon remains the fallback for both.

    If enabled, trajectory entropy is estimated by the chain rule:
        H_hat = sum_l rho_prefix(l) * H(pi_theta(. | x, y_<l)).
    Prefix ratios exclude token l and remain differentiable, accounting for
    policy-dependent prefix probabilities. Thus entropy is not implemented as
    -mean(sampled logprobs) on fixed old-policy samples. The vocabulary entropy
    also supplies a gradient when all sampled completions are identical.

    Reward clipping does not clip the separate KL/entropy terms. Accumulation
    and exponentiation use float64, and nonfinite ratios fail explicitly rather
    than silently changing the objective with a log-ratio clamp.
    """
    import torch

    symmetric_epsilon = float(clip_epsilon)
    epsilon_low = (symmetric_epsilon if clip_epsilon_low is None
                   else float(clip_epsilon_low))
    epsilon_high = (symmetric_epsilon if clip_epsilon_high is None
                    else float(clip_epsilon_high))
    if (not math.isfinite(epsilon_low) or not 0.0 < epsilon_low < 1.0
            or not math.isfinite(epsilon_high)
            or not 0.0 < epsilon_high < 1.0):
        raise ValueError("clip epsilon low/high must be finite and in (0, 1)")
    if (not math.isfinite(float(kl_coef)) or kl_coef < 0.0
            or not math.isfinite(float(entropy_coef)) or entropy_coef < 0.0):
        raise ValueError("KL and entropy coefficients must be finite and nonnegative")
    cur = current_logprobs.to(dtype=torch.float64)
    old = torch.as_tensor(old_logprobs, dtype=cur.dtype, device=cur.device).detach()
    if cur.ndim != 1 or old.shape != cur.shape or cur.numel() == 0:
        raise ValueError("current and old logprobs must be matching nonempty vectors")
    adv = torch.as_tensor(advantage, dtype=cur.dtype, device=cur.device).detach()
    if adv.numel() != 1 or not torch.isfinite(adv).all():
        raise ValueError("rank advantage must be a finite scalar")
    if not torch.isfinite(cur).all() or not torch.isfinite(old).all():
        raise ValueError("rank policy logprobs must be finite")

    log_ratios = cur - old
    token_ratios = log_ratios.exp()
    clipped_token_ratios = token_ratios.clamp(
        1.0 - epsilon_low, 1.0 + epsilon_high)
    policy_loss = -torch.minimum(
        token_ratios * adv, clipped_token_ratios * adv).mean()
    ratio = token_ratios.mean()
    clipped_fraction = (
        token_ratios.detach() != clipped_token_ratios.detach()
    ).double().mean()
    kl_estimate = cur.new_zeros(())
    if kl_coef:
        if reference_logprob is None:
            raise ValueError("reference logprobs are required for nonzero KL coefficient")
        ref = torch.as_tensor(reference_logprob, dtype=cur.dtype,
                              device=cur.device).detach()
        if ref.shape != cur.shape or not torch.isfinite(ref).all():
            raise ValueError(
                "reference logprobs must be a finite vector matching current logprobs")
        log_ref_over_current = ref - cur
        kl_estimate = (
            log_ref_over_current.exp() - log_ref_over_current - 1.0
        ).mean()

    entropy_estimate = cur.new_zeros(())
    prefix_ratio_max = cur.new_ones(())
    if entropy_coef:
        if token_entropies is None or token_entropies.shape != cur.shape:
            raise ValueError("matching token entropies are required for entropy updates")
        prefix_log_ratios = torch.cat((cur.new_zeros(1), log_ratios.cumsum(0)[:-1]))
        prefix_ratios = prefix_log_ratios.exp()
        entropy_estimate = (prefix_ratios * token_entropies.to(cur.dtype)).sum()
        prefix_ratio_max = prefix_ratios.max()

    loss = policy_loss + kl_coef * kl_estimate - entropy_coef * entropy_estimate
    if not torch.isfinite(loss).all() or not torch.isfinite(ratio).all():
        raise FloatingPointError(
            "nonfinite rank objective/trajectory ratio; reduce learning rate or "
            "rank_update_epochs and inspect model logprobs")
    tensor_metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_estimate": kl_estimate.detach(),
        "entropy_estimate": entropy_estimate.detach(),
        "ratio": ratio.detach(),
        "prefix_ratio_max": prefix_ratio_max.detach(),
        "clipped": clipped_fraction.detach(),
    }
    if return_tensor_metrics:
        return loss, tensor_metrics
    return loss, {
        key: float(value.item()) for key, value in tensor_metrics.items()
    }


@contextmanager
def _rank_dropout_disabled(model):
    """Disable dropout while retaining training/checkpointing mode."""
    import torch
    changed = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            changed.append((module, "p", module.p))
            module.p = 0.0
        # Common attention implementations use functional dropout instead of
        # an nn.Dropout module. Generation is deterministic conditional on
        # sampled tokens; old/current likelihood forwards must match it.
        for name in ("attention_dropout", "attn_dropout", "dropout"):
            value = getattr(module, name, None)
            if isinstance(value, float) and value != 0.0:
                changed.append((module, name, value))
                setattr(module, name, 0.0)
    try:
        yield
    finally:
        for module, name, value in reversed(changed):
            setattr(module, name, value)


def _train_rank_examples(backend, model, tokenizer, optimizer, examples,
                         cfg, step_idx, *, fb_cfg=None, fb_on=False,
                         fb_lambda=0.0):
    """Cache the old/reference policies, then update the rank surrogate."""
    import torch

    epochs = int(getattr(cfg, "rank_update_epochs", RANK_UPDATE_EPOCHS_DEFAULT))
    epsilon = float(getattr(cfg, "rank_clip_epsilon", RANK_CLIP_EPSILON_DEFAULT))
    epsilon_low = float(getattr(cfg, "rank_clip_epsilon_low", epsilon))
    epsilon_high = float(getattr(cfg, "rank_clip_epsilon_high", epsilon))
    entropy_coef = float(getattr(cfg, "rank_entropy_coef", RANK_ENTROPY_COEF_DEFAULT))
    kl_coef = float(cfg.kl_penalty_coef)
    fb_stats = None
    if fb_on:
        from feedback import (FeedbackStats, bound_feedback_advantage,
                              feedback_advantage)
        fb_stats = FeedbackStats()
    backend.set_training_mode()
    started = time.time()
    update_batches = _training_microbatches(
        examples, cfg,
        partition_key=lambda example: bool(
            example["rank_entropy_gate"] and entropy_coef > 0.0))
    largest_batch = max((len(batch) for batch in update_batches), default=0)
    print(f"[train] LoRA microbatches={len(update_batches)}; "
          f"configured max={int(cfg.train_examples_per_microbatch)}, "
          f"effective max={largest_batch}, "
          f"padded-token cap={int(cfg.max_seq_length)}", flush=True)
    missing_old, missing_reference = _initialize_rank_logprob_caches(examples)
    reference_fallback = missing_reference if kl_coef else []
    print(f"[step {step_idx}] rank logprobs: vLLM supplied old="
          f"{len(examples) - len(missing_old)}/{len(examples)}, reference="
          f"{len(examples) - len(reference_fallback)}/{len(examples)}; "
          f"HF fallback old={len(missing_old)}, "
          f"reference={len(reference_fallback)} before {epochs} update(s)",
          flush=True)

    with _rank_dropout_disabled(model):
        # Usually both loops are empty because vLLM supplied the frozen values.
        # They retain exact compatibility with HF generation and with any vLLM
        # sequence whose logprob payload was incomplete.
        # Missing vLLM payloads are exceptional. Score them singly so one
        # padded fallback batch cannot select an incompatible attention path
        # or multiply the activation footprint of a near-limit sequence.
        for batch in ([example] for example in missing_old):
            old_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=False, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)
            if any(not torch.isfinite(value).all()
                   for value in old_logprobs):
                raise FloatingPointError("nonfinite old-policy logprobs")
            for ex, old_lp in zip(batch, old_logprobs):
                ex["rank_old_logprobs"] = old_lp.detach()
        if reference_fallback:
            with backend.disable_adapter(), torch.no_grad():
                for batch in ([example] for example in reference_fallback):
                    reference_logprobs = compute_batched_token_logprobs(
                        model, batch, with_grad=False,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id)
                    if any(not torch.isfinite(value).all()
                           for value in reference_logprobs):
                        raise FloatingPointError(
                            "nonfinite reference-policy logprobs")
                    for ex, ref_lp in zip(batch, reference_logprobs):
                        ex["rank_reference_logprob"] = ref_lp.detach()
        if fb_on:
            for ex in examples:
                if not ex.get("reprompt_text"):
                    continue
                old_lp = ex["rank_old_logprobs"].to(ex["response_ids"].device)
                fb_adv = feedback_advantage(
                    compute_token_logprobs, model, tokenizer,
                    ex["reprompt_text"], ex["response_ids"], old_lp,
                    fb_cfg, lam=fb_lambda, chunk=cfg.logprob_chunk)
                if fb_adv is None:
                    fb_stats.skipped += 1
                else:
                    fb_adv, _ = bound_feedback_advantage(
                        fb_adv, reward_advantage=ex["advantage"], cfg=fb_cfg)
                    fb_stats.add(fb_adv)
                    ex["rank_feedback_advantage"] = fb_adv.detach().cpu()

        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            def attempt(active_batches):
                totals = {key: 0.0 for key in (
                    "loss", "policy_loss", "kl_estimate",
                    "entropy_estimate", "ratio", "clipped_fraction")}
                max_ratio = 0.0
                max_prefix_ratio = 0.0
                entropy_examples = 0
                for batch in active_batches:
                    gate = bool(batch[0]["rank_entropy_gate"]
                                and entropy_coef > 0.0)
                    result = compute_batched_token_logprobs(
                        model, batch, with_grad=True,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id,
                        return_entropy=gate)
                    if gate:
                        current_logprobs, token_entropies = result
                    else:
                        current_logprobs = result
                        token_entropies = [None] * len(batch)
                    weighted_losses = []
                    for ex, cur_lp, token_entropy in zip(
                            batch, current_logprobs, token_entropies):
                        loss, metrics = rank_grpo_loss(
                            cur_lp, ex["rank_old_logprobs"],
                            ex["rank_reference_logprob"], ex["advantage"],
                            clip_epsilon=epsilon,
                            clip_epsilon_low=epsilon_low,
                            clip_epsilon_high=epsilon_high,
                            kl_coef=kl_coef,
                            entropy_coef=(entropy_coef if gate else 0.0),
                            token_entropies=token_entropy)
                        fb_adv = ex["rank_feedback_advantage"]
                        if fb_adv is not None:
                            loss = loss - (
                                fb_adv.to(cur_lp.device) * cur_lp).mean()
                        weight = float(ex["sample_weight"])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError(
                                "nonfinite rank/feedback loss")
                        weighted_losses.append(weight * loss)
                        totals["loss"] += (
                            weight * float(loss.detach().item()))
                        for key in ("policy_loss", "kl_estimate",
                                    "entropy_estimate", "ratio"):
                            totals[key] += weight * metrics[key]
                        totals["clipped_fraction"] += (
                            weight * float(metrics["clipped"]))
                        max_ratio = max(max_ratio, metrics["ratio"])
                        max_prefix_ratio = max(
                            max_prefix_ratio, metrics["prefix_ratio_max"])
                        entropy_examples += int(gate)
                    if weighted_losses:
                        sum(weighted_losses[1:],
                            weighted_losses[0]).backward()
                return (totals, max_ratio, max_prefix_ratio,
                        entropy_examples)

            result, _effective, quarantined = _run_oom_resilient_backward(
                model, update_batches, attempt,
                device_label=str(model.device))
            if result is None:
                totals = {key: 0.0 for key in (
                    "loss", "policy_loss", "kl_estimate",
                    "entropy_estimate", "ratio", "clipped_fraction")}
                max_ratio = max_prefix_ratio = 0.0
                entropy_examples = 0
            else:
                (totals, max_ratio, max_prefix_ratio,
                 entropy_examples) = result

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            history.append({"epoch": epoch + 1, **totals,
                            "ratio_max": max_ratio,
                            "prefix_ratio_max": max_prefix_ratio,
                            "entropy_examples": entropy_examples,
                            "oom_quarantined_examples": len(quarantined),
                            "grad_norm": float(grad_norm.item())})
            print(f"[step {step_idx}] rank epoch {epoch + 1}/{epochs}: "
                  f"loss={totals['loss']:.6f} "
                  f"KL estimate={totals['kl_estimate']:.6f} "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={len(quarantined)}", flush=True)
    if fb_stats is not None:
        print(fb_stats.line(step_idx, fb_lambda))
    for ex in examples:
        for key in ("rank_old_logprobs", "rank_reference_logprob", "rank_feedback_advantage"):
            ex.pop(key, None)
    return {
        "rank_updates": history,
        "rank_train_seconds": time.time() - started,
        "oom_quarantined_examples": sum(
            item["oom_quarantined_examples"] for item in history),
    }


class ReplicatedDataParallelTrainer:
    """Run each step's LoRA update concurrently on one model replica per GPU.

    Rollout orchestration remains in the main process.  Only the differentiable
    work is replicated: examples are load-balanced across devices, each replica
    accumulates its local gradients, and the (small) LoRA gradients are summed
    onto GPU 0 before one optimizer update.  Updated adapter weights are then
    broadcast back to every replica.  This preserves the existing global loss
    exactly without launching duplicate search/evaluation processes.
    """

    def __init__(self, primary_backend, primary_model, primary_tokenizer,
                 optimizer, cfg, dependency_log_path=None):
        import torch
        from model_backend import load_backend

        self.cfg = cfg
        self.optimizer = optimizer
        self.replicas = [
            (primary_backend, primary_model, primary_tokenizer, 0)
        ]
        self._offloaded = False
        world_size = int(cfg.num_training_gpus)
        physical_ids = _parse_gpu_ids(cfg.training_gpu_ids)
        if world_size != len(physical_ids):
            raise ValueError("num_training_gpus does not match training_gpu_ids")

        print(f"[train-parallel] loading {world_size - 1} additional trainer "
              f"replica(s) for data parallelism", flush=True)
        for logical_id in range(1, world_size):
            replica_cfg = SimpleNamespace(**vars(cfg))
            replica_cfg.training_replica_device = logical_id
            # The explicit replica device controls placement. Keep the complete
            # budget vector so model_backend can select this device's budget.
            replica_cfg.num_training_gpus = 1
            with _route_dependency_notices(dependency_log_path):
                replica_backend = load_backend(cfg.backend, replica_cfg)
                replica_model, replica_tokenizer = replica_backend.load()
            self.replicas.append(
                (replica_backend, replica_model, replica_tokenizer, logical_id)
            )
            print(f"[train-parallel] replica {logical_id}/{world_size - 1} "
                  f"ready on physical GPU {physical_ids[logical_id]}", flush=True)

        self._validate_parameter_layouts()
        self._validate_replica_devices()
        self._broadcast_trainable_parameters()
        for logical_id in range(world_size):
            torch.cuda.synchronize(logical_id)
        print(f"[train-parallel] active on all {world_size} training GPUs "
              f"{physical_ids}", flush=True)

    @property
    def world_size(self):
        return len(self.replicas)

    @property
    def is_offloaded(self):
        return self._offloaded

    @staticmethod
    def _trainable_parameters(model):
        return [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad]

    def _validate_parameter_layouts(self):
        expected = [
            (name, tuple(parameter.shape))
            for name, parameter in self._trainable_parameters(self.replicas[0][1])
        ]
        for _, model, _, logical_id in self.replicas[1:]:
            actual = [
                (name, tuple(parameter.shape))
                for name, parameter in self._trainable_parameters(model)
            ]
            if actual != expected:
                raise RuntimeError(
                    f"trainer replica {logical_id} has a different trainable "
                    "parameter layout")

    def _validate_replica_devices(self):
        import torch
        for _, model, _, logical_id in self.replicas:
            expected = torch.device(f"cuda:{logical_id}")
            wrong_devices = {
                str(parameter.device) for parameter in model.parameters()
                if parameter.device != expected
            }
            if wrong_devices:
                raise RuntimeError(
                    f"trainer replica {logical_id} expected every parameter on "
                    f"{expected}, but also found {sorted(wrong_devices)}")

    def _broadcast_trainable_parameters(self):
        import torch
        source = self._trainable_parameters(self.replicas[0][1])

        def copy_replica(replica):
            _, model, _, logical_id = replica
            target = self._trainable_parameters(model)
            with torch.cuda.device(logical_id), torch.no_grad():
                for (_, source_parameter), (_, target_parameter) in zip(source, target):
                    target_parameter.copy_(
                        source_parameter.detach().to(
                            target_parameter.device, non_blocking=True))
                torch.cuda.synchronize(logical_id)

        self._run_replicas(copy_replica, self.replicas[1:])

    @staticmethod
    def _run_replicas(fn, items):
        from concurrent.futures import ThreadPoolExecutor
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(fn, item) for item in items]
            return [future.result() for future in futures]

    @staticmethod
    def _cpu_examples(examples):
        import torch
        tensor_cache = {}
        out = []
        for example in examples:
            copied = {}
            for key, value in example.items():
                if torch.is_tensor(value):
                    cache_key = id(value)
                    if cache_key not in tensor_cache:
                        tensor_cache[cache_key] = value.detach().cpu()
                    copied[key] = tensor_cache[cache_key]
                else:
                    copied[key] = value
            out.append(copied)
        return out

    def _shard_examples(self, examples):
        """Longest-first scheduling keeps uneven sequence lengths balanced."""
        shards = [[] for _ in self.replicas]
        loads = [0 for _ in self.replicas]
        ordered = sorted(
            self._cpu_examples(examples),
            key=lambda ex: int(ex["prompt_ids"].numel()
                               + ex["response_ids"].numel()),
            reverse=True,
        )
        for example in ordered:
            rank = min(range(self.world_size), key=lambda idx: loads[idx])
            shards[rank].append(example)
            loads[rank] += int(example["prompt_ids"].numel()
                               + example["response_ids"].numel())
        return shards, loads

    @staticmethod
    def _move_shard(shard, logical_id):
        import torch
        device = torch.device(f"cuda:{logical_id}")
        tensor_cache = {}
        moved = []
        for example in shard:
            local = {}
            for key, value in example.items():
                if torch.is_tensor(value):
                    cache_key = id(value)
                    if cache_key not in tensor_cache:
                        tensor_cache[cache_key] = value.to(
                            device, non_blocking=True)
                    local[key] = tensor_cache[cache_key]
                else:
                    local[key] = value
            moved.append(local)
        return moved

    def _prepare_shards(self, examples):
        import torch
        shards, token_loads = self._shard_examples(examples)

        def move(item):
            replica, shard = item
            logical_id = replica[3]
            with torch.cuda.device(logical_id):
                return self._move_shard(shard, logical_id)

        local_shards = self._run_replicas(
            move, list(zip(self.replicas, shards)))
        counts = [len(shard) for shard in local_shards]
        print(f"[train-parallel] examples/GPU={counts}; "
              f"tokens/GPU={token_loads}", flush=True)
        return local_shards

    def _zero_gradients(self):
        for _, model, _, _ in self.replicas:
            model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

    def _sum_gradients_to_primary(self):
        """Sum replica gradients without DDP's implicit world-size average."""
        import torch
        primary = self._trainable_parameters(self.replicas[0][1])
        others = [self._trainable_parameters(replica[1])
                  for replica in self.replicas[1:]]
        with torch.cuda.device(0), torch.no_grad():
            for parameter_index, (_, destination) in enumerate(primary):
                gradient = destination.grad
                if gradient is None:
                    gradient = torch.zeros_like(destination)
                    destination.grad = gradient
                for replica_parameters in others:
                    source_gradient = replica_parameters[parameter_index][1].grad
                    if source_gradient is not None:
                        gradient.add_(source_gradient.to(
                            destination.device, non_blocking=True))
            torch.cuda.synchronize(0)

    @staticmethod
    def _merge_feedback_stats(stats):
        from feedback import FeedbackStats
        merged = FeedbackStats()
        for item in stats:
            if item is None:
                continue
            merged.n += item.n
            merged.skipped += item.skipped
            merged.sum_abs += item.sum_abs
            merged.sum_pos += item.sum_pos
            merged.max_abs = max(merged.max_abs, item.max_abs)
        return merged

    def _clip_step_and_sync(self, cfg):
        import torch
        self._sum_gradients_to_primary()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in
             self._trainable_parameters(self.replicas[0][1])],
            max_norm=cfg.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self._broadcast_trainable_parameters()
        value = float(grad_norm.item())
        for _, model, _, _ in self.replicas:
            model.zero_grad(set_to_none=True)
        return value

    @staticmethod
    def _padded_token_count(batch):
        if not batch:
            return 0
        return max(
            int(example["prompt_ids"].shape[1]
                + example["response_ids"].shape[1])
            for example in batch
        ) * len(batch)

    def _train_rank_fast(self, examples, cfg, step_idx, *, fb_cfg=None,
                         fb_on=False, fb_lambda=0.0):
        """Adaptive, work-stealing rank update selected only by --fast.

        Each replica repeatedly claims a length-bucketed batch from one shared
        CPU queue. Its padded-token budget is learned from real peak allocator
        use on that GPU and persists across steps. The controller aims below
        80% total device occupancy, while the normal OOM splitter remains the
        final guard for nonlinear attention/output-head memory spikes.
        """
        import threading
        import torch
        from feedback import (FeedbackStats, bound_feedback_advantage,
                              feedback_advantage)

        started = time.time()
        if self._offloaded:
            print(f"[train-fast] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()

        epochs = int(getattr(cfg, "rank_update_epochs",
                             RANK_UPDATE_EPOCHS_DEFAULT))
        if epochs != 1:
            raise ValueError("--fast requires rank_update_epochs=1")
        epsilon = float(getattr(cfg, "rank_clip_epsilon",
                                RANK_CLIP_EPSILON_DEFAULT))
        epsilon_low = float(getattr(cfg, "rank_clip_epsilon_low", epsilon))
        epsilon_high = float(getattr(cfg, "rank_clip_epsilon_high", epsilon))
        entropy_coef = float(getattr(cfg, "rank_entropy_coef",
                                     RANK_ENTROPY_COEF_DEFAULT))
        kl_coef = float(cfg.kl_penalty_coef)

        cpu_examples = self._cpu_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in cpu_examples)
        supplied_reference = sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in cpu_examples) if kl_coef else len(cpu_examples)
        print(f"[step {step_idx}] rank logprobs: vLLM supplied old="
              f"{supplied_old}/{len(cpu_examples)}, reference="
              f"{supplied_reference}/{len(cpu_examples)}; HF fallback runs only "
              "for missing values", flush=True)

        partitions = {False: [], True: []}
        for example in cpu_examples:
            gate = bool(example["rank_entropy_gate"] and entropy_coef > 0.0)
            partitions[gate].append(example)
        for partition in partitions.values():
            partition.sort(
                key=lambda example: int(
                    example["prompt_ids"].shape[1]
                    + example["response_ids"].shape[1]))

        queue_lock = threading.Lock()
        # A token budget is the real memory controller. This loose count guard
        # only prevents pathological thousands-of-tiny-sequences batches.
        max_examples_per_batch = min(64, max(1, len(cpu_examples)))

        def take_batch(token_budget):
            with queue_lock:
                available = [key for key, value in partitions.items() if value]
                if not available:
                    return []
                partition_key = max(
                    available,
                    key=lambda key: int(
                        partitions[key][-1]["prompt_ids"].shape[1]
                        + partitions[key][-1]["response_ids"].shape[1]))
                pending = partitions[partition_key]
                batch = [pending.pop()]
                maximum = int(
                    batch[0]["prompt_ids"].shape[1]
                    + batch[0]["response_ids"].shape[1])
                while pending and len(batch) < max_examples_per_batch:
                    candidate = pending[-1]
                    candidate_length = int(
                        candidate["prompt_ids"].shape[1]
                        + candidate["response_ids"].shape[1])
                    next_maximum = max(maximum, candidate_length)
                    if next_maximum * (len(batch) + 1) > token_budget:
                        break
                    batch.append(pending.pop())
                    maximum = next_maximum
                return batch

        initial_budget = max(1, int(cfg.max_seq_length))
        budgets = getattr(self, "_fast_token_budgets", None)
        if not isinstance(budgets, list) or len(budgets) != self.world_size:
            budgets = [initial_budget for _ in self.replicas]
            self._fast_token_budgets = budgets
        else:
            for replica_index in range(self.world_size):
                budgets[replica_index] = max(
                    1, int(budgets[replica_index] or initial_budget))

        print(f"[train-fast] shared adaptive queue on {self.world_size} GPUs; "
              "memory target=80%; initial padded-token budgets/GPU="
              f"{list(budgets)}", flush=True)
        self._zero_gradients()

        metric_keys = (
            "loss", "policy_loss", "kl_estimate", "entropy_estimate",
            "ratio", "clipped_fraction",
        )

        def work(replica_index):
            backend, model, tokenizer, logical_id = self.replicas[replica_index]
            backend.set_training_mode()
            parameters = [parameter for _, parameter in
                          self._trainable_parameters(model)]
            # Keep already-computed LoRA gradients outside model.grad so the
            # existing OOM retry helper can safely clear a partial next batch.
            gradient_accumulators = [None for _ in parameters]
            totals = {key: 0.0 for key in metric_keys}
            max_ratio = 0.0
            max_prefix_ratio = 0.0
            entropy_examples = 0
            quarantined_examples = 0
            scheduled_batches = 0
            backward_batches = 0
            trained_examples = 0
            peak_memory_fraction = 0.0
            feedback_parts = []
            budget = max(1, int(budgets[replica_index]))

            while True:
                cpu_batch = take_batch(budget)
                if not cpu_batch:
                    break
                scheduled_batches += 1
                requested_padded_tokens = self._padded_token_count(cpu_batch)

                with torch.cuda.device(logical_id):
                    base_allocated = int(torch.cuda.memory_allocated())
                    base_reserved = int(torch.cuda.memory_reserved())
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    external_bytes = max(
                        0, int(total_bytes) - int(free_bytes) - base_reserved)
                    torch.cuda.reset_peak_memory_stats()
                    local_batch = self._move_shard(cpu_batch, logical_id)

                    def attempt(active_batches):
                        attempt_values = {key: [] for key in metric_keys}
                        attempt_ratio_values = []
                        attempt_prefix_ratio_values = []
                        attempt_entropy_examples = 0
                        attempt_feedback = FeedbackStats()

                        with _rank_dropout_disabled(model):
                            for batch in active_batches:
                                missing_old, missing_reference = (
                                    _initialize_rank_logprob_caches(batch))
                                for fallback_batch in (
                                        [example] for example in missing_old):
                                    old_logprobs = (
                                        compute_batched_token_logprobs(
                                            model, fallback_batch,
                                            with_grad=False,
                                            chunk=cfg.logprob_chunk,
                                            pad_token_id=(
                                                tokenizer.pad_token_id)))
                                    if any(not torch.isfinite(value).all()
                                           for value in old_logprobs):
                                        raise FloatingPointError(
                                            "nonfinite old-policy logprobs")
                                    for example, old_lp in zip(
                                            fallback_batch, old_logprobs):
                                        example["rank_old_logprobs"] = (
                                            old_lp.detach())

                                if kl_coef and missing_reference:
                                    with backend.disable_adapter(), torch.no_grad():
                                        for fallback_batch in (
                                                [example] for example in
                                                missing_reference):
                                            reference_logprobs = (
                                                compute_batched_token_logprobs(
                                                    model, fallback_batch,
                                                    with_grad=False,
                                                    chunk=cfg.logprob_chunk,
                                                    pad_token_id=(
                                                        tokenizer.pad_token_id)))
                                            if any(
                                                    not torch.isfinite(value).all()
                                                    for value in
                                                    reference_logprobs):
                                                raise FloatingPointError(
                                                    "nonfinite reference-policy "
                                                    "logprobs")
                                            for example, reference_lp in zip(
                                                    fallback_batch,
                                                    reference_logprobs):
                                                example[
                                                    "rank_reference_logprob"
                                                ] = reference_lp.detach()

                                if fb_on:
                                    for example in batch:
                                        if not example.get("reprompt_text"):
                                            continue
                                        old_lp = example[
                                            "rank_old_logprobs"].to(
                                                example["response_ids"].device)
                                        fb_advantage = feedback_advantage(
                                            compute_token_logprobs, model,
                                            tokenizer,
                                            example["reprompt_text"],
                                            example["response_ids"], old_lp,
                                            fb_cfg, lam=fb_lambda,
                                            chunk=cfg.logprob_chunk)
                                        if fb_advantage is None:
                                            attempt_feedback.skipped += 1
                                        else:
                                            fb_advantage, _ = (
                                                bound_feedback_advantage(
                                                    fb_advantage,
                                                    reward_advantage=example[
                                                        "advantage"],
                                                    cfg=fb_cfg))
                                            attempt_feedback.add(fb_advantage)
                                            example[
                                                "rank_feedback_advantage"
                                            ] = fb_advantage.detach().cpu()

                                gate = bool(
                                    batch[0]["rank_entropy_gate"]
                                    and entropy_coef > 0.0)
                                result = compute_batched_token_logprobs(
                                    model, batch, with_grad=True,
                                    chunk=cfg.logprob_chunk,
                                    pad_token_id=tokenizer.pad_token_id,
                                    return_entropy=gate)
                                if gate:
                                    current_logprobs, token_entropies = result
                                else:
                                    current_logprobs = result
                                    token_entropies = [None] * len(batch)

                                weighted_losses = []
                                for example, current_lp, token_entropy in zip(
                                        batch, current_logprobs,
                                        token_entropies):
                                    loss, metrics = rank_grpo_loss(
                                        current_lp,
                                        example["rank_old_logprobs"],
                                        example["rank_reference_logprob"],
                                        example["advantage"],
                                        clip_epsilon=epsilon,
                                        clip_epsilon_low=epsilon_low,
                                        clip_epsilon_high=epsilon_high,
                                        kl_coef=kl_coef,
                                        entropy_coef=(entropy_coef
                                                      if gate else 0.0),
                                        token_entropies=token_entropy,
                                        return_tensor_metrics=True)
                                    fb_advantage = example[
                                        "rank_feedback_advantage"]
                                    if fb_advantage is not None:
                                        loss = loss - (
                                            fb_advantage.to(current_lp.device)
                                            * current_lp).mean()
                                    weight = float(example["sample_weight"])
                                    if not torch.isfinite(loss).all():
                                        raise FloatingPointError(
                                            "nonfinite rank/feedback loss")
                                    weighted_losses.append(weight * loss)
                                    attempt_values["loss"].append(
                                        weight * loss.detach())
                                    for key in (
                                            "policy_loss", "kl_estimate",
                                            "entropy_estimate", "ratio"):
                                        attempt_values[key].append(
                                            weight * metrics[key])
                                    attempt_values[
                                        "clipped_fraction"].append(
                                            weight * metrics["clipped"])
                                    attempt_ratio_values.append(
                                        metrics["ratio"])
                                    attempt_prefix_ratio_values.append(
                                        metrics["prefix_ratio_max"])
                                    attempt_entropy_examples += int(gate)
                                if weighted_losses:
                                    sum(weighted_losses[1:],
                                        weighted_losses[0]).backward()

                        zero = torch.zeros(
                            (), dtype=torch.float64,
                            device=torch.device(f"cuda:{logical_id}"))
                        reduced = [
                            (torch.stack(attempt_values[key]).sum()
                             if attempt_values[key] else zero)
                            for key in metric_keys
                        ]
                        reduced.extend((
                            (torch.stack(attempt_ratio_values).max()
                             if attempt_ratio_values else zero),
                            (torch.stack(attempt_prefix_ratio_values).max()
                             if attempt_prefix_ratio_values else zero),
                        ))
                        packed = torch.stack([
                            value.to(dtype=torch.float64)
                            for value in reduced
                        ]).detach().cpu().tolist()
                        attempt_totals = dict(zip(
                            metric_keys, packed[:len(metric_keys)]))
                        attempt_max_ratio = packed[len(metric_keys)]
                        attempt_max_prefix_ratio = packed[
                            len(metric_keys) + 1]
                        return (
                            attempt_totals, attempt_max_ratio,
                            attempt_max_prefix_ratio,
                            attempt_entropy_examples, attempt_feedback,
                        )

                    result, effective_batches, quarantined = (
                        _run_oom_resilient_backward(
                            model, [local_batch], attempt,
                            device_label=f"cuda:{logical_id}"))

                    peak_allocated = int(
                        torch.cuda.max_memory_allocated())
                    peak_reserved = int(torch.cuda.max_memory_reserved())
                    used_at_peak = min(
                        int(total_bytes), external_bytes + peak_reserved)
                    peak_memory_fraction = max(
                        peak_memory_fraction,
                        used_at_peak / max(1, int(total_bytes)))

                    if result is not None:
                        with torch.no_grad():
                            for parameter_index, parameter in enumerate(
                                    parameters):
                                gradient = parameter.grad
                                if gradient is None:
                                    continue
                                accumulator = gradient_accumulators[
                                    parameter_index]
                                if accumulator is None:
                                    gradient_accumulators[
                                        parameter_index] = gradient.detach()
                                else:
                                    accumulator.add_(gradient)
                                parameter.grad = None
                        for key in metric_keys:
                            totals[key] += result[0][key]
                        max_ratio = max(max_ratio, result[1])
                        max_prefix_ratio = max(
                            max_prefix_ratio, result[2])
                        entropy_examples += result[3]
                        feedback_parts.append(result[4])

                    model.zero_grad(set_to_none=True)
                    quarantined_examples += len(quarantined)
                    backward_batches += len(effective_batches)
                    trained_examples += sum(
                        len(batch) for batch in effective_batches)

                    successful_padded_tokens = max(
                        (self._padded_token_count(batch)
                         for batch in effective_batches),
                        default=0)
                    longest_successful = max(
                        (int(example["prompt_ids"].shape[1]
                             + example["response_ids"].shape[1])
                         for batch in effective_batches
                         for example in batch),
                        default=1)
                    backed_off = bool(
                        len(effective_batches) > 1 or quarantined)
                    if not effective_batches:
                        budget = max(1, budget // 2)
                    elif backed_off:
                        budget = max(
                            longest_successful,
                            min(budget, successful_padded_tokens))
                    else:
                        active_bytes = max(
                            1, peak_allocated - base_allocated)
                        target_process_bytes = max(
                            base_allocated + 1,
                            int(0.80 * int(total_bytes)) - external_bytes)
                        available_for_batch = max(
                            1, target_process_bytes - base_allocated)
                        desired_budget = int(
                            requested_padded_tokens
                            * available_for_batch / active_bytes * 0.96)
                        lower = max(
                            longest_successful, int(budget * 0.70))
                        upper = max(lower, int(budget * 1.35))
                        desired_budget = min(
                            upper, max(lower, desired_budget))
                        budget = max(
                            longest_successful,
                            int(round(0.5 * budget
                                      + 0.5 * desired_budget)))
                    budgets[replica_index] = budget

                del local_batch, effective_batches, quarantined

            with torch.cuda.device(logical_id), torch.no_grad():
                for parameter, accumulator in zip(
                        parameters, gradient_accumulators):
                    parameter.grad = accumulator
                torch.cuda.synchronize(logical_id)

            return {
                "totals": totals,
                "max_ratio": max_ratio,
                "max_prefix_ratio": max_prefix_ratio,
                "entropy_examples": entropy_examples,
                "quarantined_examples": quarantined_examples,
                "scheduled_batches": scheduled_batches,
                "backward_batches": backward_batches,
                "trained_examples": trained_examples,
                "peak_memory_fraction": peak_memory_fraction,
                "feedback_stats": self._merge_feedback_stats(feedback_parts),
            }

        results = self._run_replicas(work, range(self.world_size))
        totals = {
            key: sum(result["totals"][key] for result in results)
            for key in metric_keys
        }
        max_ratio = max(result["max_ratio"] for result in results)
        max_prefix_ratio = max(
            result["max_prefix_ratio"] for result in results)
        entropy_examples = sum(
            result["entropy_examples"] for result in results)
        quarantined_examples = sum(
            result["quarantined_examples"] for result in results)
        feedback_stats = self._merge_feedback_stats(
            [result["feedback_stats"] for result in results])
        grad_norm = self._clip_step_and_sync(cfg)
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "entropy_examples": entropy_examples,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        trained_per_gpu = [
            result["trained_examples"] for result in results]
        batches_per_gpu = [
            result["backward_batches"] for result in results]
        peak_percentages = [
            round(100.0 * result["peak_memory_fraction"], 1)
            for result in results]
        print(f"[train-fast] trained examples/GPU={trained_per_gpu}; "
              f"backward batches/GPU={batches_per_gpu}; final padded-token "
              f"budgets/GPU={list(budgets)}; peak memory/GPU="
              f"{peak_percentages}%", flush=True)
        print(f"[step {step_idx}] rank epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"KL estimate={totals['kl_estimate']:.6f} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        if fb_on:
            print(feedback_stats.line(step_idx, fb_lambda))
        return {
            "rank_updates": history,
            "rank_train_seconds": time.time() - started,
            "training_parallel_gpus": self.world_size,
            "training_fast": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(budgets),
            "fast_peak_memory_fraction": [
                result["peak_memory_fraction"] for result in results],
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_rank(self, examples, cfg, step_idx, *, fb_cfg=None,
                   fb_on=False, fb_lambda=0.0):
        if bool(getattr(cfg, "fast", False)):
            return self._train_rank_fast(
                examples, cfg, step_idx, fb_cfg=fb_cfg, fb_on=fb_on,
                fb_lambda=fb_lambda)

        import torch
        from feedback import (FeedbackStats, bound_feedback_advantage,
                              feedback_advantage)

        started = time.time()
        if self._offloaded:
            print(f"[train-parallel] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()
        epochs = int(getattr(cfg, "rank_update_epochs",
                             RANK_UPDATE_EPOCHS_DEFAULT))
        epsilon = float(getattr(cfg, "rank_clip_epsilon",
                                RANK_CLIP_EPSILON_DEFAULT))
        epsilon_low = float(getattr(cfg, "rank_clip_epsilon_low", epsilon))
        epsilon_high = float(getattr(cfg, "rank_clip_epsilon_high", epsilon))
        entropy_coef = float(getattr(cfg, "rank_entropy_coef",
                                     RANK_ENTROPY_COEF_DEFAULT))
        kl_coef = float(cfg.kl_penalty_coef)
        local_shards = self._prepare_shards(examples)
        update_batches = [
            _training_microbatches(
                shard, cfg,
                partition_key=lambda example: bool(
                    example["rank_entropy_gate"] and entropy_coef > 0.0))
            for shard in local_shards
        ]
        largest_batch = max(
            (len(batch) for batches in update_batches for batch in batches),
            default=0)
        print(f"[train-parallel] LoRA microbatches/GPU="
              f"{[len(batches) for batches in update_batches]}; "
              f"configured max={int(cfg.train_examples_per_microbatch)}, "
              f"effective max={largest_batch}, "
              f"padded-token cap={int(cfg.max_seq_length)}", flush=True)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in examples)
        supplied_reference = sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in examples) if kl_coef else len(examples)
        print(f"[step {step_idx}] rank logprobs: vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"{supplied_reference}/{len(examples)}; HF fallback runs only "
              f"for missing values before {epochs} update(s)", flush=True)

        def cache(item):
            replica_index, shard = item
            backend, model, tokenizer, logical_id = self.replicas[replica_index]
            stats = FeedbackStats() if fb_on else None
            backend.set_training_mode()
            with torch.cuda.device(logical_id), _rank_dropout_disabled(model):
                missing_old, missing_reference = (
                    _initialize_rank_logprob_caches(shard))
                # vLLM normally fills these caches. Singleton fallback keeps a
                # rare incomplete payload from destabilizing all eight replicas.
                for batch in ([example] for example in missing_old):
                    old_logprobs = compute_batched_token_logprobs(
                        model, batch, with_grad=False,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id)
                    if any(not torch.isfinite(value).all()
                           for value in old_logprobs):
                        raise FloatingPointError(
                            "nonfinite old-policy logprobs")
                    for example, old_lp in zip(batch, old_logprobs):
                        example["rank_old_logprobs"] = old_lp.detach()
                if kl_coef and missing_reference:
                    with backend.disable_adapter(), torch.no_grad():
                        for batch in (
                                [example] for example in missing_reference):
                            reference_logprobs = compute_batched_token_logprobs(
                                model, batch, with_grad=False,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id)
                            if any(not torch.isfinite(value).all()
                                   for value in reference_logprobs):
                                raise FloatingPointError(
                                    "nonfinite reference-policy logprobs")
                            for example, reference_lp in zip(
                                    batch, reference_logprobs):
                                example["rank_reference_logprob"] = (
                                    reference_lp.detach())
                if fb_on:
                    for example in shard:
                        if example.get("reprompt_text"):
                            old_lp = example["rank_old_logprobs"].to(
                                example["response_ids"].device)
                            fb_advantage = feedback_advantage(
                                compute_token_logprobs, model, tokenizer,
                                example["reprompt_text"],
                                example["response_ids"], old_lp, fb_cfg,
                                lam=fb_lambda,
                                chunk=cfg.logprob_chunk)
                            if fb_advantage is None:
                                stats.skipped += 1
                            else:
                                fb_advantage, _ = bound_feedback_advantage(
                                    fb_advantage,
                                    reward_advantage=example["advantage"],
                                    cfg=fb_cfg)
                                stats.add(fb_advantage)
                                example["rank_feedback_advantage"] = (
                                    fb_advantage.detach().cpu())
                torch.cuda.synchronize(logical_id)
            return stats

        cache_stats = self._run_replicas(
            cache, list(enumerate(local_shards)))
        feedback_stats = self._merge_feedback_stats(cache_stats)
        history = []
        for epoch in range(epochs):
            self._zero_gradients()

            def accumulate(item):
                replica_index, batches = item
                _, model, tokenizer, logical_id = self.replicas[replica_index]
                def attempt(active_batches):
                    totals = {key: 0.0 for key in (
                        "loss", "policy_loss", "kl_estimate",
                        "entropy_estimate", "ratio", "clipped_fraction")}
                    max_ratio = 0.0
                    max_prefix_ratio = 0.0
                    entropy_examples = 0
                    with _rank_dropout_disabled(model):
                        for batch in active_batches:
                            gate = bool(batch[0]["rank_entropy_gate"]
                                        and entropy_coef > 0.0)
                            result = compute_batched_token_logprobs(
                                model, batch, with_grad=True,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id,
                                return_entropy=gate)
                            if gate:
                                current_logprobs, token_entropies = result
                            else:
                                current_logprobs = result
                                token_entropies = [None] * len(batch)
                            weighted_losses = []
                            for example, current_lp, token_entropy in zip(
                                    batch, current_logprobs,
                                    token_entropies):
                                loss, metrics = rank_grpo_loss(
                                    current_lp,
                                    example["rank_old_logprobs"],
                                    example["rank_reference_logprob"],
                                    example["advantage"],
                                    clip_epsilon=epsilon,
                                    clip_epsilon_low=epsilon_low,
                                    clip_epsilon_high=epsilon_high,
                                    kl_coef=kl_coef,
                                    entropy_coef=(entropy_coef
                                                  if gate else 0.0),
                                    token_entropies=token_entropy)
                                fb_advantage = example[
                                    "rank_feedback_advantage"]
                                if fb_advantage is not None:
                                    loss = loss - (
                                        fb_advantage.to(current_lp.device)
                                        * current_lp).mean()
                                weight = float(example["sample_weight"])
                                if not torch.isfinite(loss).all():
                                    raise FloatingPointError(
                                        "nonfinite rank/feedback loss")
                                weighted_losses.append(weight * loss)
                                totals["loss"] += (
                                    weight * float(loss.detach().item()))
                                for key in ("policy_loss", "kl_estimate",
                                            "entropy_estimate", "ratio"):
                                    totals[key] += weight * metrics[key]
                                totals["clipped_fraction"] += (
                                    weight * float(metrics["clipped"]))
                                max_ratio = max(
                                    max_ratio, metrics["ratio"])
                                max_prefix_ratio = max(
                                    max_prefix_ratio,
                                    metrics["prefix_ratio_max"])
                                entropy_examples += int(gate)
                            if weighted_losses:
                                sum(weighted_losses[1:],
                                    weighted_losses[0]).backward()
                    torch.cuda.synchronize(logical_id)
                    return (totals, max_ratio, max_prefix_ratio,
                            entropy_examples)

                with torch.cuda.device(logical_id):
                    result, _effective, quarantined = (
                        _run_oom_resilient_backward(
                            model, batches, attempt,
                            device_label=f"cuda:{logical_id}"))
                if result is None:
                    result = (
                        {key: 0.0 for key in (
                            "loss", "policy_loss", "kl_estimate",
                            "entropy_estimate", "ratio",
                            "clipped_fraction")},
                        0.0, 0.0, 0,
                    )
                return (*result, len(quarantined))

            results = self._run_replicas(
                accumulate, list(enumerate(update_batches)))
            totals = {key: sum(result[0][key] for result in results)
                      for key in results[0][0]}
            max_ratio = max(result[1] for result in results)
            max_prefix_ratio = max(result[2] for result in results)
            entropy_examples = sum(result[3] for result in results)
            quarantined_examples = sum(result[4] for result in results)
            grad_norm = self._clip_step_and_sync(cfg)
            history.append({
                "epoch": epoch + 1, **totals, "ratio_max": max_ratio,
                "prefix_ratio_max": max_prefix_ratio,
                "entropy_examples": entropy_examples,
                "oom_quarantined_examples": quarantined_examples,
                "grad_norm": grad_norm,
            })
            print(f"[step {step_idx}] rank epoch {epoch + 1}/{epochs}: "
                  f"loss={totals['loss']:.6f} "
                  f"KL estimate={totals['kl_estimate']:.6f} "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={quarantined_examples}", flush=True)

        if fb_on:
            print(feedback_stats.line(step_idx, fb_lambda))
        return {
            "rank_updates": history,
            "rank_train_seconds": time.time() - started,
            "training_parallel_gpus": self.world_size,
            "oom_quarantined_examples": sum(
                item["oom_quarantined_examples"] for item in history),
        }

    def train_policy(self, examples, cfg, step_idx, *, fb_cfg=None,
                     fb_on=False, fb_lambda=0.0):
        import torch
        from feedback import (FeedbackStats, bound_feedback_advantage,
                              feedback_advantage)

        started = time.time()
        if self._offloaded:
            print(f"[train-parallel] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()
        local_shards = self._prepare_shards(examples)
        n_examples = len(examples)
        microbatches = [
            _training_microbatches(shard, cfg) for shard in local_shards
        ]
        largest_batch = max(
            (len(batch) for batches in microbatches for batch in batches),
            default=0)
        print(f"[train-parallel] LoRA microbatches/GPU="
              f"{[len(batches) for batches in microbatches]}; "
              f"configured max={int(cfg.train_examples_per_microbatch)}, "
              f"effective max={largest_batch}, "
              f"padded-token cap={int(cfg.max_seq_length)}", flush=True)
        self._zero_gradients()

        def accumulate(item):
            replica_index, batches = item
            backend, model, tokenizer, logical_id = self.replicas[replica_index]
            backend.set_training_mode()
            def attempt(active_batches):
                total_loss = 0.0
                total_logp_delta = 0.0
                ratio_sum = 0.0
                ratio_max = 0.0
                ratio_count = 0
                stats = FeedbackStats()
                kl_error = None
                for batch in active_batches:
                    current_logprobs = compute_batched_token_logprobs(
                        model, batch, with_grad=True,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id)
                    base_logprobs = [
                        example.get("reference_logprobs")
                        for example in batch
                    ]
                    supplied_reference = all(
                        _valid_example_token_logprobs(
                            example, "reference_logprobs")
                        for example in batch)
                    if not supplied_reference:
                        try:
                            with backend.disable_adapter(), torch.no_grad():
                                base_logprobs = compute_batched_token_logprobs(
                                    model, batch, with_grad=False,
                                    chunk=cfg.logprob_chunk,
                                    pad_token_id=tokenizer.pad_token_id)
                        except Exception as error:
                            kl_error = repr(error)
                            base_logprobs = [
                                current_lp.detach()
                                for current_lp in current_logprobs]
                    batch_losses = []
                    for example, current_lp, base_lp in zip(
                            batch, current_logprobs, base_logprobs):
                        base_lp = base_lp.to(current_lp.device)
                        response_ids = example["response_ids"]
                        advantage = example["advantage"]
                        logp_difference = (current_lp - base_lp).detach()
                        average_difference = logp_difference.mean()
                        kl_advantage = cfg.kl_penalty_coef * (
                            average_difference - (current_lp - base_lp))
                        effective_advantage = advantage + kl_advantage
                        if fb_on and example.get("reprompt_text"):
                            fb_advantage = feedback_advantage(
                                compute_token_logprobs, model, tokenizer,
                                example["reprompt_text"], response_ids,
                                current_lp.detach(), fb_cfg, lam=fb_lambda,
                                chunk=cfg.logprob_chunk)
                            if fb_advantage is None:
                                stats.skipped += 1
                            else:
                                fb_advantage, _ = bound_feedback_advantage(
                                    fb_advantage,
                                    reward_advantage=advantage, cfg=fb_cfg)
                                stats.add(fb_advantage)
                                effective_advantage = (
                                    effective_advantage + fb_advantage)
                        behavior_lp = example.get("behavior_logprobs")
                        if _valid_example_token_logprobs(
                                example, "behavior_logprobs"):
                            importance_ratio = torch.exp(
                                current_lp.detach() - behavior_lp)
                            ratio_sum += float(
                                importance_ratio.mean().item())
                            ratio_max = max(
                                ratio_max,
                                float(importance_ratio.max().item()))
                            ratio_count += 1
                        else:
                            importance_ratio = 1.0
                        loss = -(
                            importance_ratio * effective_advantage.detach()
                            * current_lp).mean()
                        batch_losses.append(loss / n_examples)
                        total_loss += float(loss.detach().item())
                        total_logp_delta += float(
                            logp_difference.mean().item())
                    if batch_losses:
                        sum(batch_losses[1:], batch_losses[0]).backward()
                torch.cuda.synchronize(logical_id)
                return (total_loss, total_logp_delta, ratio_sum, ratio_max,
                        ratio_count, stats, kl_error)

            with torch.cuda.device(logical_id):
                result, _effective, quarantined = (
                    _run_oom_resilient_backward(
                        model, batches, attempt,
                        device_label=f"cuda:{logical_id}"))
            if result is None:
                result = (0.0, 0.0, 0.0, 0.0, 0, FeedbackStats(), None)
            return (*result, len(quarantined))

        results = self._run_replicas(
            accumulate, list(enumerate(microbatches)))
        grad_norm = self._clip_step_and_sync(cfg)
        total_loss = sum(result[0] for result in results)
        total_logp_delta = sum(result[1] for result in results)
        ratio_sum = sum(result[2] for result in results)
        ratio_max = max(result[3] for result in results)
        ratio_count = sum(result[4] for result in results)
        feedback_stats = self._merge_feedback_stats(
            [result[5] for result in results])
        kl_errors = [result[6] for result in results if result[6] is not None]
        quarantined_examples = sum(result[7] for result in results)
        if kl_errors:
            print(f"[warn] disable_adapter failed ({kl_errors[0]}); "
                  "training without KL penalty on affected examples")
        elapsed = time.time() - started
        ratio_message = ""
        if ratio_count:
            ratio_message = (f"  IS ratio mean={ratio_sum / ratio_count:.9f} "
                             f"max={ratio_max:.3f}")
        print(f"[step {step_idx}] train time: {elapsed:.1f}s  "
              f"avg loss: {total_loss / n_examples:.9f}  "
              f"avg logpi_theta - logpi_base: "
              f"{total_logp_delta / n_examples:.9f}{ratio_message}  "
              f"OOM-quarantined={quarantined_examples}")
        if fb_on:
            print(feedback_stats.line(step_idx, fb_lambda))
        return {
            "training_seconds": elapsed,
            "training_parallel_gpus": self.world_size,
            "training_grad_norm": grad_norm,
            "oom_quarantined_examples": quarantined_examples,
        }

    def offload_for_generation(self):
        """Release every replica before an all-GPU vLLM phase."""
        if self._offloaded:
            return
        import gc
        import torch
        # Mark first so a partially completed move is recoverable if any one
        # quantized replica raises while transferring to host memory.
        self._offloaded = True
        try:
            for logical_id in range(self.world_size):
                torch.cuda.synchronize(logical_id)
            _move_optimizer_state(self.optimizer, "cpu")
            for backend, _, _, _ in self.replicas:
                backend.offload_for_generation()
            gc.collect()
            for logical_id in range(self.world_size):
                with torch.cuda.device(logical_id):
                    torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            self.restore_after_generation()
            raise

    def restore_after_generation(self):
        if not self._offloaded:
            return
        import gc
        import torch
        gc.collect()

        def restore(replica):
            backend, _, _, logical_id = replica
            with torch.cuda.device(logical_id):
                torch.cuda.empty_cache()
                backend.restore_after_generation()
                backend.set_training_mode()
                torch.cuda.synchronize(logical_id)

        self._run_replicas(restore, self.replicas)
        self._validate_replica_devices()
        _restore_optimizer_state_to_parameters(self.optimizer)
        self._offloaded = False


class ProcessDistributedTrainer:
    """Fast rank trainer with one persistent Python process per GPU."""

    def __init__(self, primary_backend, primary_model, primary_tokenizer,
                 optimizer, cfg, exp_dir, dependency_log_path=None):
        import queue
        import uuid
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        from fast_distributed import (
            broadcast_trainable_parameters, set_total_memory_ceiling,
            trainable_parameter_signature, validate_model_device, worker_main,
        )

        if dist.is_initialized():
            raise RuntimeError(
                "--fast cannot start because torch.distributed is already "
                "initialized in the main process")
        self.backend = primary_backend
        self.model = primary_model
        self.tokenizer = primary_tokenizer
        self.optimizer = optimizer
        self.cfg = cfg
        self._offloaded = False
        self._closed = False
        self._workers_idle = True
        self._dist_initialized = False
        self._queue_module = queue
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._work_queue = self._context.Queue()
        self._command_queues = {}
        self._processes = {}
        self._physical_ids = _parse_gpu_ids(cfg.training_gpu_ids)
        self._world_size = int(cfg.num_training_gpus)
        self._token_budgets = [
            max(1, int(cfg.max_seq_length))
            for _ in range(self._world_size)
        ]
        if self._world_size != len(self._physical_ids):
            raise ValueError(
                "num_training_gpus does not match training_gpu_ids")
        if self._world_size < 2:
            raise ValueError("process-distributed --fast requires two GPUs")

        # Eight independent Python processes must not each create a full-sized
        # host thread pool. GPU 0 was loaded before this class is constructed;
        # the allocator limit still constrains every subsequent allocation.
        torch.set_num_threads(max(
            1, int(os.cpu_count() or self._world_size) // self._world_size))
        torch.cuda.set_device(0)
        set_total_memory_ceiling(0, 0.80)
        validate_model_device(self.model, 0)
        expected_signature = trainable_parameter_signature(self.model)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        sync_dir = Path(exp_dir).resolve() / ".fast_trainer"
        sync_dir.mkdir(parents=True, exist_ok=True)
        rendezvous_path = sync_dir / (
            f"nccl-{os.getpid()}-{uuid.uuid4().hex}")
        self._init_method = rendezvous_path.as_uri()
        cfg_dict = dict(vars(cfg))

        print(f"[train-fast] starting {self._world_size - 1} persistent "
              "worker processes; main process is rank 0", flush=True)
        try:
            for rank in range(1, self._world_size):
                command_queue = self._context.Queue(maxsize=2)
                process = self._context.Process(
                    target=worker_main,
                    args=(
                        rank, self._world_size, cfg_dict,
                        self._init_method, self._work_queue, command_queue,
                        self._result_queue, dependency_log_path,
                    ),
                    name=f"ttt-fast-trainer-rank-{rank}",
                )
                process.start()
                self._command_queues[rank] = command_queue
                self._processes[rank] = process

            loaded = self._collect_event(
                "loaded", range(1, self._world_size))
            for rank, message in loaded.items():
                if message.get("parameter_signature") != expected_signature:
                    raise RuntimeError(
                        f"fast trainer rank {rank} has a different trainable "
                        "parameter layout")
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "init_distributed"})
            dist.init_process_group(
                backend="nccl", init_method=self._init_method,
                rank=0, world_size=self._world_size,
                timeout=timedelta(minutes=30))
            self._dist_initialized = True
            broadcast_trainable_parameters(self.model, source_rank=0)
            self._collect_event("ready", range(1, self._world_size))
        except BaseException:
            self._abort_workers()
            raise

        print(f"[train-fast] process-distributed trainer active on physical "
              f"GPUs {self._physical_ids}; adaptive memory ceiling=80%",
              flush=True)

    @property
    def world_size(self):
        return self._world_size

    @property
    def is_offloaded(self):
        return self._offloaded

    def _dead_workers(self, pending):
        return [
            rank for rank in pending
            if rank in self._processes
            and not self._processes[rank].is_alive()
        ]

    def _collect_event(self, event, ranks):
        pending = set(int(rank) for rank in ranks)
        results = {}
        while pending:
            try:
                message = self._result_queue.get(timeout=5.0)
            except self._queue_module.Empty:
                dead = self._dead_workers(pending)
                if dead:
                    codes = {
                        rank: self._processes[rank].exitcode for rank in dead
                    }
                    raise RuntimeError(
                        f"fast trainer worker(s) exited while waiting for "
                        f"{event}: {codes}")
                continue
            message_event = message.get("event")
            rank = int(message.get("rank", -1))
            if message_event == "error":
                raise RuntimeError(
                    f"fast trainer rank {rank} failed:\n"
                    f"{message.get('traceback', 'no traceback')}")
            if message_event != event or rank not in pending:
                raise RuntimeError(
                    f"unexpected fast trainer message while waiting for "
                    f"{event}: {message}")
            pending.remove(rank)
            results[rank] = message
        return results

    @staticmethod
    def _cpu_examples(examples):
        import torch

        tensor_cache = {}
        copied_examples = []
        for example in examples:
            copied = {}
            for key, value in example.items():
                if torch.is_tensor(value):
                    cache_key = id(value)
                    if cache_key not in tensor_cache:
                        tensor_cache[cache_key] = value.detach().cpu()
                    copied[key] = tensor_cache[cache_key]
                else:
                    copied[key] = value
            copied_examples.append(copied)
        return copied_examples

    def _queue_examples(self, examples):
        """Queue expensive examples first; idle ranks steal the next work."""
        maximum_length = max(1, int(self.cfg.max_seq_length))

        def estimated_cost(example):
            prompt = int(example["prompt_ids"].shape[1])
            response = int(example["response_ids"].shape[1])
            total = prompt + response
            return total * total + response * maximum_length

        ordered = sorted(
            self._cpu_examples(examples),
            key=estimated_cost, reverse=True)
        for example in ordered:
            self._work_queue.put(example)
        for _ in range(self._world_size):
            self._work_queue.put(None)
        print(f"[train-fast] queued {len(ordered)} examples longest-first; "
              "each free GPU claims the next adaptive batch", flush=True)
        return ordered

    @staticmethod
    def _merge_feedback_dicts(items):
        from feedback import FeedbackStats

        merged = FeedbackStats()
        for item in items:
            merged.n += int(item.get("n", 0))
            merged.skipped += int(item.get("skipped", 0))
            merged.sum_abs += float(item.get("sum_abs", 0.0))
            merged.sum_pos += float(item.get("sum_pos", 0.0))
            merged.max_abs = max(
                merged.max_abs, float(item.get("max_abs", 0.0)))
        return merged

    def train_rank(self, examples, cfg, step_idx, *, fb_cfg=None,
                   fb_on=False, fb_lambda=0.0):
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      local_rank_update,
                                      reduce_trainable_gradients)

        if self._offloaded:
            print(f"[train-fast] restoring {self._world_size} process "
                  "trainers for the update", flush=True)
            self.restore_after_generation()
        if int(getattr(cfg, "rank_update_epochs", 1)) != 1:
            raise ValueError("--fast requires rank_update_epochs=1")

        started = time.time()
        queued_examples = self._queue_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued_examples)
        supplied_reference = (sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued_examples)
            if float(cfg.kl_penalty_coef) else len(examples))
        print(f"[step {step_idx}] rank logprobs: vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"{supplied_reference}/{len(examples)}", flush=True)
        print(f"[train-fast] one process/GPU; memory ceiling=80%; "
              f"initial padded-token budgets/GPU={self._token_budgets}",
              flush=True)

        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_rank",
                "token_budget": self._token_budgets[rank],
                "memory_fraction": 0.80,
                "fb_cfg": fb_cfg,
                "fb_on": bool(fb_on),
                "fb_lambda": float(fb_lambda),
            })

        local_stats = local_rank_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0], memory_fraction=0.80,
            fb_cfg=fb_cfg, fb_on=fb_on, fb_lambda=fb_lambda,
            work_queue=self._work_queue)
        child_messages = self._collect_event(
            "computed", range(1, self._world_size))
        rank_stats = [local_stats] + [
            child_messages[rank]["stats"]
            for rank in range(1, self._world_size)
        ]
        accounted_examples = sum(
            int(stats["trained_examples"])
            + int(stats["quarantined_examples"])
            for stats in rank_stats)
        if accounted_examples != len(queued_examples):
            raise RuntimeError(
                "fast trainer shared queue lost or duplicated work: "
                f"accounted for {accounted_examples}/{len(queued_examples)} "
                "examples")

        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "finish_update"})
        reduce_trainable_gradients(self.model, destination_rank=0)

        update_error = None
        grad_norm = None
        try:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters()
                 if parameter.requires_grad],
                max_norm=cfg.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            grad_norm = float(grad_norm_tensor.item())
        except BaseException as error:
            update_error = error
        finally:
            self.model.zero_grad(set_to_none=True)
            broadcast_trainable_parameters(self.model, source_rank=0)
            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
        if update_error is not None:
            raise update_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(
                1, int(stats["token_budget"]))
        metric_keys = (
            "loss", "policy_loss", "kl_estimate", "entropy_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {
            key: sum(stats["totals"][key] for stats in rank_stats)
            for key in metric_keys
        }
        max_ratio = max(stats["max_ratio"] for stats in rank_stats)
        max_prefix_ratio = max(
            stats["max_prefix_ratio"] for stats in rank_stats)
        entropy_examples = sum(
            stats["entropy_examples"] for stats in rank_stats)
        quarantined_examples = sum(
            stats["quarantined_examples"] for stats in rank_stats)
        trained_per_gpu = [
            stats["trained_examples"] for stats in rank_stats]
        batches_per_gpu = [
            stats["backward_batches"] for stats in rank_stats]
        peak_fractions = [
            stats["peak_memory_fraction"] for stats in rank_stats]
        allocator_fractions = [
            stats["allocator_memory_fraction"] for stats in rank_stats]
        peak_percentages = [
            round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [
            round(100.0 * fraction, 1) for fraction in allocator_fractions]
        feedback_stats = self._merge_feedback_dicts([
            stats["feedback_stats"] for stats in rank_stats])
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "entropy_examples": entropy_examples,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        print(f"[train-fast] trained examples/GPU={trained_per_gpu}; "
              f"backward batches/GPU={batches_per_gpu}; final padded-token "
              f"budgets/GPU={self._token_budgets}; peak memory/GPU="
              f"{peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        print(f"[step {step_idx}] rank epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"KL estimate={totals['kl_estimate']:.6f} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        if fb_on:
            print(feedback_stats.line(step_idx, fb_lambda))
        return {
            "rank_updates": history,
            "rank_train_seconds": time.time() - started,
            "training_parallel_gpus": self._world_size,
            "training_fast": True,
            "training_fast_processes": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_policy(self, *args, **kwargs):
        raise RuntimeError("process-distributed --fast supports rank mode only")

    def offload_for_generation(self):
        if self._offloaded:
            return
        import gc
        import torch

        self._offloaded = True
        try:
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "offload"})
            torch.cuda.synchronize(0)
            _move_optimizer_state(self.optimizer, "cpu")
            self.backend.offload_for_generation()
            gc.collect()
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
            self._collect_event("offloaded", range(1, self._world_size))
        except BaseException:
            # A failed worker has already left the NCCL process group. Do not
            # attempt a graceful barrier/restore against a partial group.
            self._workers_idle = False
            self._abort_workers()
            raise

    def restore_after_generation(self):
        if not self._offloaded:
            return
        if not self._dist_initialized:
            raise RuntimeError(
                "process-distributed trainer is unavailable after a worker "
                "failure")
        import gc
        import torch
        from fast_distributed import (set_total_memory_ceiling,
                                      validate_model_device)

        try:
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "restore"})
            gc.collect()
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
                set_total_memory_ceiling(0, 0.80)
                self.backend.restore_after_generation()
                self.backend.set_training_mode()
                torch.cuda.synchronize(0)
            validate_model_device(self.model, 0)
            _restore_optimizer_state_to_parameters(self.optimizer)
            self._collect_event("restored", range(1, self._world_size))
            self._offloaded = False
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise

    def _abort_workers(self):
        import torch.distributed as dist

        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
        for process in self._processes.values():
            process.join(timeout=10.0)
        if self._dist_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        self._dist_initialized = False

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        import torch.distributed as dist

        if (not self._workers_idle or not self._dist_initialized
                or self._dead_workers(range(1, self._world_size))):
            self._abort_workers()
            return
        try:
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "stop"})
            dist.barrier()
            self._collect_event("stopped", range(1, self._world_size))
        finally:
            if self._dist_initialized and dist.is_initialized():
                dist.destroy_process_group()
            self._dist_initialized = False
            for process in self._processes.values():
                process.join(timeout=10.0)
            for process in self._processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10.0)


# ======================================================================
# LoRA adapter sync (main process -> generation workers)
# ======================================================================
def _as_float_list(seq, max_len: int = 4096):
    """
    Coerce a construction to a plain list of floats for the rollout meta.

    Returns None when absent, and refuses anything longer than max_len so a
    problem with a huge construction cannot bloat every meta file. 0 disables
    saving entirely.
    """
    if seq is None or max_len == 0:
        return None
    try:
        out = [float(x) for x in seq]
    except (TypeError, ValueError):
        return None
    if not out or (max_len > 0 and len(out) > max_len):
        return None
    return out


def _adapter_dir(exp_dir, step_idx):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}")


def _adapter_exists(exp_dir):
    from pathlib import Path
    p = Path(exp_dir)
    return any(p.glob("adapter_step*"))


def _save_adapter(model, exp_dir, step_idx, generation_model_name=None):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    Adapters are retained because completed checkpoints refer to their matching
    step directory; an interrupted write can therefore never invalidate the
    previous resumable checkpoint.
    """
    out_dir = _adapter_dir(exp_dir, step_idx)
    # PEFT/Unsloth models support save_pretrained, which writes just the adapter
    model.save_pretrained(out_dir)
    # The trainable GPT-OSS copy may be a BnB conversion while vLLM hosts the
    # original MXFP4 checkpoint. Their module names are identical; advertise
    # the actual rollout base so vLLM does not reject the adapter as mismatched.
    config_path = Path(out_dir) / "adapter_config.json"
    if generation_model_name and config_path.is_file():
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        if adapter_config.get("base_model_name_or_path") != generation_model_name:
            adapter_config["base_model_name_or_path"] = generation_model_name
            config_path.write_text(
                json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    return out_dir


def _load_adapter(model, adapter_dir):
    """Load saved LoRA weights into the already-created trainable adapter."""
    import torch
    from peft import set_peft_model_state_dict

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"checkpoint adapter not found: {adapter_dir}")

    try:
        from peft.utils.save_and_load import load_peft_weights
        weights = load_peft_weights(
            str(adapter_dir), device=str(next(model.parameters()).device)
        )
    except (ImportError, TypeError):
        safe_path = adapter_dir / "adapter_model.safetensors"
        bin_path = adapter_dir / "adapter_model.bin"
        if safe_path.is_file():
            from safetensors.torch import load_file
            weights = load_file(
                str(safe_path), device=str(next(model.parameters()).device)
            )
        elif bin_path.is_file():
            try:
                weights = torch.load(
                    str(bin_path), map_location="cpu", weights_only=True
                )
            except TypeError:
                weights = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"no adapter weights found under {adapter_dir}")

    set_peft_model_state_dict(model, weights)
    print(f"[resume] loaded LoRA adapter from {adapter_dir}")


def _save_training_checkpoint(exp_dir, next_step, adapter_path, sampler,
                              optimizer, next_g, next_k, memory_path=None):
    """Atomically save the state required for an exact next-step resume."""
    import torch

    target = Path(exp_dir) / "training_state.pt"
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "version": 1,
        "next_step": int(next_step),
        "adapter_dir": Path(adapter_path).name,
        "sampler": sampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_groups_per_step": int(next_g),
        "next_group_size": int(next_k),
        "memory_file": Path(memory_path).name if memory_path is not None else None,
    }
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def _load_training_checkpoint(exp_dir):
    import torch

    path = Path(exp_dir) / "training_state.pt"
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"unsupported training checkpoint: {path}")
    required = ("next_step", "adapter_dir", "sampler", "optimizer",
                "next_groups_per_step", "next_group_size")
    for key in required:
        if key not in payload:
            raise ValueError(f"training checkpoint is missing {key!r}: {path}")
    return payload


def _legacy_resume_info(exp_dir):
    """Locate the safe restart point for a pre-checkpoint run directory."""
    matches = []
    for path in Path(exp_dir).glob("adapter_step*"):
        try:
            matches.append((int(path.name.removeprefix("adapter_step")), path))
        except ValueError:
            continue
    if not matches:
        raise FileNotFoundError(
            "this older run has no training_state.pt and no adapter_step* "
            "directory; the trained policy cannot be resumed"
        )
    # Legacy adapters were written immediately before their numbered step.
    return max(matches, key=lambda item: item[0])


def _restore_legacy_archive(sampler, exp_dir, before_step):
    """Recover valid candidates from rollout files that predate checkpoints."""
    from reward import extract_python_code
    from sampler import State

    states = []
    total_rollouts = 0
    pattern = "step*/step*_group*_rollout*.meta.json"
    for meta_path in sorted(Path(exp_dir).glob(pattern)):
        try:
            meta = json.loads(meta_path.read_text())
            step = int(meta.get("step", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if step < 0 or step >= before_step:
            continue
        total_rollouts += 1
        if not meta.get("valid"):
            continue
        text_path = meta_path.with_name(
            meta_path.name.removesuffix(".meta.json") + ".txt"
        )
        try:
            code = extract_python_code(text_path.read_text(errors="replace"))
        except OSError:
            code = None
        if not code:
            continue
        try:
            reward = float(meta.get("reward", 0.0))
            raw = meta.get("raw_score")
            raw = float(raw) if raw is not None else None
            construction = meta.get("construction")
        except (TypeError, ValueError):
            continue
        states.append(State.make(
            timestep=step, value=reward, code=code, raw_score=raw,
            construction=construction,
        ))
    sampler.import_legacy_states(states, total_expansions=total_rollouts)
    return len(states), total_rollouts


# ======================================================================
# One training step
#
# Generation is streamed and each rollout's program is evaluated on a CPU
# thread pool WHILE the GPUs keep generating.
#
# `reward_workers` is configured in YAML (0 = auto). Auto
# leaves ~one CPU core per GPU worker for the generation loop and divides the
# remainder by the CPU allocation of each candidate evaluation.
# In --isolate-eval mode it instead means processes per CPU and must be >= 1.
# ======================================================================
def _isolated_evaluation_cpu_ids():
    """Return the exact Linux cpuset available to this training process."""
    if not (hasattr(os, "sched_getaffinity")
            and hasattr(os, "sched_setaffinity")):
        raise RuntimeError(
            "--isolate-eval requires Linux CPU affinity support")
    cpu_ids = sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    if not cpu_ids:
        raise RuntimeError("--isolate-eval found no CPUs in the allowed cpuset")
    return cpu_ids


def _resolve_reward_workers(cfg, problem, cpu_count=None) -> int:
    requested = int(getattr(cfg, "reward_workers", 0) or 0)
    if requested < 0:
        raise ValueError("reward_workers must be >= 0")
    if requested:
        return requested

    available = int(cpu_count if cpu_count is not None else (os.cpu_count() or 8))
    generation_workers = max(0, int(getattr(cfg, "num_gpus", 0) or 0))
    per_evaluation = max(1, int(getattr(problem, "eval_cpus", 1)))
    cpu_budget = max(1, available - generation_workers)
    return max(1, cpu_budget // per_evaluation)


def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg, exp_dir, problem, gen_pool=None,
               memory=None, extractor=None, mem_cfg=None, lookup=None,
               curator=None, fb_cfg=None, parallel_trainer=None):
    import os
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from queue import Queue

    from sampler import State
    from experiment_io import save_parent_selections, save_rollout
    from problems.base import ParentContext
    from gen_workers import make_progress_bar

    from memory import (MemoryArm, RolloutRecord, allocate_memory_arms,
                        build_injection, credit_memory_arms, inject_block,
                        memory_protocol_block)
    from feedback import (FeedbackStats, bound_feedback_advantage,
                          build_reprompt, feedback_advantage, format_feedback,
                          is_code_failure, render_chat, select_balanced)

    step_t0 = time.time()
    sampler.set_current_step(step_idx)
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.9f}  n={info['n']}  "
              f"Q={info['Q']:.9f}  P={info['P']:.9f}  bonus={info['bonus']:.9f}  "
              f"score={info['score']:.9f}")

    # Save the selection event immediately. Unlike the sampler checkpoint,
    # this survives later archive pruning and also exists if generation or
    # adapter training is interrupted.
    sampler_type = type(sampler).__name__
    save_parent_selections(
        exp_dir, step_idx, sampler_type, parents, sampler.last_picks_info)
    print(f"[step {step_idx}] saved {len(parents)} selected parent(s) before "
          f"generation/training", flush=True)

    # The coefficient in force this step. When it reaches zero the whole
    # feedback path is skipped: no reprompts built, no teacher forwards, so the
    # annealed tail costs exactly what a no-feedback run costs.
    import inspect as _inspect
    prompt_parameters = _inspect.signature(problem.build_prompt).parameters
    memory_aware_prompt = "memory" in prompt_parameters
    memory_protocol_aware = "memory_protocol" in prompt_parameters
    memory_v2 = bool(mem_cfg is not None and getattr(mem_cfg, "is_v2", False))
    # Only problems that declare it get their construction written to disk.
    save_ctor = (bool(getattr(problem, "saves_construction", False))
                 and int(getattr(cfg, "max_saved_construction", 0)) != 0)

    fb_base_lambda = fb_cfg.lambda_at(step_idx) if fb_cfg is not None else 0.0
    fb_candidate_on = bool(
        fb_cfg is not None and fb_cfg.enabled and fb_base_lambda > 0.0)
    if (fb_cfg is not None and fb_cfg.enabled and not fb_candidate_on):
        print(f"[step {step_idx}] feedback: lambda annealed to 0, term disabled")
    reprompt_by_key = {}    # (group, rollout) -> reprompt for a code failure

    all_examples = []
    rank_mode = getattr(cfg, "advantage_mode", "entropic") == "rank"
    rank_group_stats = []
    all_children = []
    saved_rollouts = 0
    mem_records = []            # RolloutRecord per rollout, for the memory maker
    mem_arm_updates = []        # matched treatment-vs-null outcome diagnostics
    mem_arm_rollouts = {}       # arm -> number of generated programs this step

    # ----- BUILD PROMPTS (one per parent/group) -----
    # Three passes now, because memory lookup is a batched LLM call rather than
    # a vector query: collect the parent contexts, ask the model once which
    # lessons it wants for all of them, then render.
    parent_ctxs = []
    base_messages = []

    for g, parent in enumerate(parents):
        sampler.record_expansion(parent, count=cfg.group_size)
        pc = ParentContext(
            code=parent.code,
            value=parent.value if parent.value is not None else 0.0,
            raw_score=parent.raw_score,
            construction=parent.construction,
        )
        parent_ctxs.append(pc)
        base_messages.append(problem.build_prompt(pc))

    # The adapter is saved BEFORE the lookup, not just before generation, so the
    # selection call runs on the same policy the rollouts will. Same file either
    # way, so this only moves the write earlier.
    adapter_path = None
    if gen_pool is not None:
        adapter_path = _save_adapter(
            model, exp_dir, step_idx, cfg.model_name)

    # ---- memory lookup (replaces the Eq. 7 retrieval) ----------------
    # One call covering every parent. An empty bank makes no call at all, so
    # step 0 is byte-identical to a --no-memory run.
    chosen_by_group = {}
    if memory is not None and lookup is not None:
        chosen_by_group = lookup.select_batch(
            parent_ctxs, step_idx=step_idx, adapter_path=adapter_path)

    # V2 accounts for every treatment already scheduled in this batch before
    # assigning exploratory arms. This prevents stale UCB statistics from
    # sending every parent to the same untested lesson.
    memory_reservations = {}
    if memory_v2:
        for chosen in chosen_by_group.values():
            for lesson in chosen:
                memory_reservations[lesson.id] = (
                    int(memory_reservations.get(lesson.id, 0)) + 1)

    def _render(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(cfg.thinking),
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    # One parent can now own several prompt arms, but their counts still sum to
    # exactly cfg.group_size. Advantages are computed over the parent union;
    # each example's likelihood still conditions on its actual prompt arm.
    prompt_jobs = []
    for g, _parent in enumerate(parents):
        pc = parent_ctxs[g]
        chosen = chosen_by_group.get(g, [])
        if memory is not None and mem_cfg is not None:
            if memory_v2:
                arms = allocate_memory_arms(
                    cfg.group_size, chosen, memory, mem_cfg, step_idx,
                    reservations=memory_reservations)
            else:
                arms = allocate_memory_arms(
                    cfg.group_size, chosen, memory, mem_cfg, step_idx)
        else:
            arms = [MemoryArm("no_memory", [], int(cfg.group_size))]

        # Rotate prompt order across parents and steps. This prevents a memory
        # arm from always receiving the first or last segment of the sampler's
        # RNG stream while remaining deterministic and resume-safe.
        if len(arms) > 1:
            shift = (int(step_idx) + int(g)) % len(arms)
            arms = arms[shift:] + arms[:shift]

        causal_protocol = bool(
            memory_v2 and len(arms) > 1
            and any(arm.lessons for arm in arms))

        for arm in arms:
            messages = base_messages[g]
            kept, n_tok = [], 0
            block = ""
            if arm.lessons:
                block, n_tok, kept = build_injection(
                    arm.lessons, tokenizer,
                    getattr(mem_cfg, "token_budget", 0),
                    version=getattr(mem_cfg, "version", "V1"))
            if causal_protocol:
                if memory_aware_prompt and memory_protocol_aware:
                    messages = problem.build_prompt(
                        pc, memory=block, memory_protocol=True)
                else:
                    messages = inject_block(
                        messages, memory_protocol_block(block),
                        mode=getattr(mem_cfg, "inject_mode", "append"))
            elif arm.lessons:
                if memory_aware_prompt:
                    messages = problem.build_prompt(pc, memory=block)
                else:
                    messages = inject_block(
                        messages, block,
                        mode=getattr(mem_cfg, "inject_mode", "append"))
            prompt_jobs.append({
                "parent_group": g,
                "arm": arm.name,
                "memory_ids": [lesson.id for lesson in kept],
                "memory_tokens": int(n_tok),
                "messages": messages,
                "prompt_text": _render(messages),
                "count": int(arm.count),
            })
            mem_arm_rollouts[arm.name] = (
                mem_arm_rollouts.get(arm.name, 0) + int(arm.count))

    if memory is not None and prompt_jobs:
        vals = [job["memory_tokens"] for job in prompt_jobs]
        print(f"[step {step_idx}] memory arms {mem_arm_rollouts}; injected "
              f"{sum(v > 0 for v in vals)}/{len(vals)} prompt variants, "
              f"{min(vals)}-{max(vals)} tokens; total rollout budget unchanged")

    num_groups = len(parents)
    total_rollouts = num_groups * cfg.group_size

    # ----- REWARD POOL (CPU), runs concurrently with generation -----
    # compute_reward delegates the heavy work to a subprocess sandbox, so the
    # launching thread mostly waits (GIL released) and many sandboxes run in
    # parallel across cores. THREAD-SAFETY REQUIREMENT: each compute_reward call
    # must use a unique temp file/dir and must not os.chdir or mutate shared
    # state; otherwise concurrent runs corrupt each other's rewards.
    isolated_eval = bool(getattr(cfg, "isolate_eval", False))
    isolated_cpu_slots = None
    isolated_cpu_count = 0
    isolated_processes_per_cpu = 0
    if isolated_eval:
        isolated_cpu_ids = _isolated_evaluation_cpu_ids()
        isolated_cpu_count = len(isolated_cpu_ids)
        isolated_processes_per_cpu = int(cfg.reward_workers)
        isolated_cpu_slots = Queue()
        # Layered admission is intentional: put one candidate on every CPU
        # before allowing a second on each CPU, and so on.
        for _layer in range(isolated_processes_per_cpu):
            for cpu_id in isolated_cpu_ids:
                isolated_cpu_slots.put(cpu_id)
        isolated_capacity = isolated_cpu_count * isolated_processes_per_cpu
        n_reward_workers = max(1, min(total_rollouts, isolated_capacity))
        print(f"[step {step_idx}] isolated evaluation pool: "
              f"{n_reward_workers} sandbox process(es) across "
              f"{isolated_cpu_count} CPU(s), up to "
              f"{isolated_processes_per_cpu}/CPU from reward_workers; each "
              "process tree is pinned to one CPU and excess candidates remain "
              "queued")
    else:
        n_reward_workers = _resolve_reward_workers(cfg, problem)
        print(f"[step {step_idx}] evaluation pool: {n_reward_workers} worker(s), "
              f"{getattr(problem, 'eval_cpus', 1)} CPU(s) per candidate")
    reward_pool = ThreadPoolExecutor(max_workers=n_reward_workers)

    # Mutable records preserve streamed arrival order while vLLM reference
    # scoring fills its result concurrently with the CPU reward futures.
    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}    # aligned RewardResult futures
    deferred_rollouts = []
    vllm_logprob_records = []
    defer_gpu_evaluation = bool(
        getattr(cfg, "evaluation_shares_generation", False))

    def _run_isolated_reward(response_text, parent_ctx):
        cpu_id = isolated_cpu_slots.get()
        try:
            return problem.compute_reward(
                response_text, parent_ctx, cfg.sandbox_timeout_s,
                cpu_id=cpu_id)
        finally:
            isolated_cpu_slots.put(cpu_id)

    def _submit_rollout(record):
        job = prompt_jobs[record["job_idx"]]
        g = job["parent_group"]
        group_responses[g].append(record)
        if isolated_eval:
            fut = reward_pool.submit(
                _run_isolated_reward, record["text"], parent_ctxs[g])
        else:
            fut = reward_pool.submit(
                problem.compute_reward, record["text"], parent_ctxs[g],
                cfg.sandbox_timeout_s
            )
        reward_futures[g].append(fut)

    def _queue_rollout(job_idx, text, token_ids, behavior_logprobs=None):
        record = {
            "job_idx": int(job_idx),
            "text": text,
            "token_ids": list(token_ids),
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": None,
        }
        if cfg.generation_backend == "vllm" and token_ids:
            vllm_logprob_records.append(record)
        if defer_gpu_evaluation:
            deferred_rollouts.append(record)
        else:
            _submit_rollout(record)

    # ----- ROLLOUTS (streamed) + dispatch rewards as each rollout lands -----
    rollout_t0 = time.time()
    evaluation_trainer_offloaded = False
    try:
        try:
            if gen_pool is not None:
                # Consume the generation stream (the adapter was already saved
                # above). CPU rewards and rewards on a distinct evaluation GPU
                # start immediately. A one-card GPU problem defers evaluation
                # until generation releases that same physical card.
                generation_options = {
                    "prompts_by_group": [
                        job["prompt_text"] for job in prompt_jobs],
                    "group_size": cfg.group_size,
                    "counts_by_group": [
                        job["count"] for job in prompt_jobs],
                    "adapter_path": adapter_path,
                    "max_new_tokens": cfg.max_new_tokens,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "step_idx": step_idx,
                }
                if cfg.generation_backend == "vllm":
                    generation_options["return_logprobs"] = True
                for job_idx, job_results in gen_pool.iter_group_jobs(
                        **generation_options):
                    for result in job_results:
                        text, token_ids = result[:2]
                        behavior_logprobs = (
                            result[2] if len(result) > 2 else None)
                        _queue_rollout(
                            job_idx, text, token_ids, behavior_logprobs)
            else:
                # In-process generation uses cross-prompt micro-batches rather
                # than draining one parent at a time. Ordinary CPU verification
                # overlaps later batches; one-card GPU-mode verification waits.
                backend.set_inference_mode()
                if cfg.deterministic:
                    torch.manual_seed((int(cfg.seed) * 1_000_003
                                       + step_idx * 1009 + 13) % (2**31 - 1))
                gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
                try:
                    cap_state = {"value": int(
                        getattr(cfg, "_local_generation_cap", 0) or 0)}
                    for job_idx, responses in generate_prompt_jobs(
                            model, tokenizer,
                            [job["prompt_text"] for job in prompt_jobs],
                            [job["count"] for job in prompt_jobs], cfg,
                            cap_state=cap_state):
                        for (text, token_ids) in responses:
                            _queue_rollout(job_idx, text, token_ids)
                        gen_bar.update(len(responses))
                    cfg._local_generation_cap = int(cap_state["value"])
                finally:
                    gen_bar.close()

            # CPU reward processes were submitted as each rollout arrived and
            # continue running here. Keep vLLM awake and use that same interval
            # to score fixed base/reference token logprobs without the adapter.
            if cfg.generation_backend == "vllm" and vllm_logprob_records:
                scoring_started = time.time()
                eval_done_before = sum(
                    future.done() for futures in reward_futures.values()
                    for future in futures)
                prompt_token_ids = {}
                score_pairs = []
                for record in vllm_logprob_records:
                    job_idx = record["job_idx"]
                    if job_idx not in prompt_token_ids:
                        prompt_token_ids[job_idx] = tokenizer(
                            prompt_jobs[job_idx]["prompt_text"]).input_ids
                    score_pairs.append((
                        prompt_token_ids[job_idx], record["token_ids"]))
                try:
                    reference_scores = gen_pool.score_token_logprobs(
                        score_pairs, show_progress=True)
                except Exception as error:
                    reference_scores = [None] * len(score_pairs)
                    print(f"[warn] vLLM reference scoring failed ({error!r}); "
                          "missing values will use their frozen rollout-policy "
                          "scores", flush=True)
                reference_behavior_fallbacks = 0
                for record, values in zip(
                        vllm_logprob_records, reference_scores):
                    behavior_values = record["behavior_logprobs"]
                    if (values is None and behavior_values is not None
                            and len(behavior_values)
                            == len(record["token_ids"])):
                        # Never send an extreme context back through the HF
                        # trainer merely because a vLLM score payload was
                        # incomplete. The rollout policy is frozen for this
                        # update, so it is the stable conservative reference.
                        values = list(behavior_values)
                        reference_behavior_fallbacks += 1
                    record["reference_logprobs"] = values
                behavior_count = sum(
                    record["behavior_logprobs"] is not None
                    for record in vllm_logprob_records)
                reference_count = sum(
                    record["reference_logprobs"] is not None
                    for record in vllm_logprob_records)
                eval_done_after = sum(
                    future.done() for futures in reward_futures.values()
                    for future in futures)
                print(f"[step {step_idx}] vLLM logprobs: rollout "
                      f"{behavior_count}/{len(vllm_logprob_records)}, "
                      f"reference {reference_count}/"
                      f"{len(vllm_logprob_records)} in "
                      f"{time.time() - scoring_started:.1f}s "
                      f"(rollout-policy fallbacks: "
                      f"{reference_behavior_fallbacks}; "
                      f"CPU evaluations completed during scoring: "
                      f"{eval_done_before}->{eval_done_after})", flush=True)
        finally:
            if gen_pool is not None and getattr(gen_pool, "sequential", False):
                gen_pool.release()

        if deferred_rollouts:
            print(f"[step {step_idx}] releasing trainer memory on the shared "
                  f"evaluation GPU before {len(deferred_rollouts)} serialized "
                  "benchmark(s)")
            try:
                torch.cuda.synchronize()
                _move_optimizer_state(optimizer, "cpu")
                backend.offload_for_generation()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                evaluation_trainer_offloaded = True
            except Exception as exc:
                try:
                    backend.restore_after_generation()
                    _restore_optimizer_state_to_parameters(optimizer)
                except Exception:
                    pass
                raise RuntimeError(
                    "one-GPU gpu_mode requires the trainer to offload before "
                    "candidate benchmark evaluation, but this model/runtime "
                    "cannot move the training state to CPU") from exc
            for record in deferred_rollouts:
                _submit_rollout(record)

        # Wait for whatever rewards are still running (a small tail if overlap
        # worked); shows how many were already done when generation finished.
        all_futs = [f for g in range(num_groups) for f in reward_futures[g]]
        eval_bar = make_progress_bar(len(all_futs), desc="evaluating")
        try:
            for _ in as_completed(all_futs):
                eval_bar.update(1)
        finally:
            eval_bar.close()
    finally:
        reward_pool.shutdown(wait=True)
        if evaluation_trainer_offloaded:
            print(f"[step {step_idx}] restoring trainer after shared-GPU "
                  "benchmark evaluation")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            backend.restore_after_generation()
            _restore_optimizer_state_to_parameters(optimizer)
            backend.set_training_mode()

    # ----- SCORE + ADVANTAGE + SAVE + COLLECT TRAINING EXAMPLES -----
    # ---- signals for adaptive batch growth ----
    # best_valid_yield: the single best group's valid fraction this step.
    # distinct_good: how many UNIQUE valid children beat their parent, deduped
    # by code so a collapsed group (same program N times) counts once.
    best_valid_yield = 0.0
    distinct_good_hashes = set()
    step_valid_count = 0
    step_rollout_count = 0
    step_code_failure_count = 0
    prompt_ids_by_job = {}

    for g, parent in enumerate(parents):
        responses = group_responses[g]          # streamed mutable rollout records
        futs = reward_futures[g]                 # aligned RewardResult futures

        rewards = []
        codes = []
        valids = []
        outs = []        # list of RewardResult
        for r_idx, record in enumerate(responses):
            res = futs[r_idx].result()           # already computed (or finishes now)
            rewards.append(res.reward)
            codes.append(res.code or "")
            valids.append(res.valid)
            outs.append(res)

        rewards_np = np.array(rewards, dtype=np.float64)
        if rewards_np.size == 0:
            print(f"  group {g}: no rollouts returned; no update for this group")
            continue
        adv_mode = getattr(cfg, "advantage_mode", "entropic")
        advantages, adv_scale, adv_scale_label, adv_info = compute_group_advantages(
            rewards_np, adv_mode,
            cvar_alpha=getattr(cfg, "cvar_alpha", None),
            cvar_lambda=getattr(cfg, "cvar_lambda", None),
            rank_gamma=getattr(cfg, "rank_gamma", None), return_info=True)
        constant = (bool(adv_info["all_tied"]) if rank_mode else
                    float(rewards_np.max() - rewards_np.min()) < 1e-12)
        rank_entropy_gate = bool(
            rank_mode
            and constant
            and any(valids)
            and float(rewards_np.max()) > float(cfg.fail_score)
        )
        if rank_mode:
            rank_diagnostics = {
                k: v for k, v in adv_info.items() if k not in ("ranks", "weights")
            }
            rank_group_stats.append({
                "group": g, **rank_diagnostics,
                "entropy_gate": rank_entropy_gate,
            })

        # growth signals for this group
        if len(valids):
            best_valid_yield = max(best_valid_yield, sum(valids) / len(valids))
        step_valid_count += sum(valids)
        step_rollout_count += len(valids)
        step_code_failure_count += sum(is_code_failure(res) for res in outs)
        parent_val = float(parent.value) if parent.value is not None else 0.0
        for r_idx in range(len(responses)):
            if valids[r_idx] and codes[r_idx] and rewards[r_idx] > parent_val:
                distinct_good_hashes.add(hash(codes[r_idx].strip()))

        print(f"  group {g}: rewards min={rewards_np.min():.9f} "
              f"mean={rewards_np.mean():.9f} max={rewards_np.max():.9f}  "
              f"valid={sum(valids)}/{len(valids)}  "
              f"{adv_scale_label}={adv_scale:.9f}")
        if rank_mode:
            print(f"    rank KL={adv_info['kl']:.6f} "
                  f"ESS={adv_info['ess']:.2f}/{len(rewards)} "
                  f"top ties={adv_info['top_count']} "
                  f"saturated={adv_info['saturated']} "
                  f"entropy gate={rank_entropy_gate}")

        # Outcome-based memory credit. All arms share this parent and the same
        # total K budget; expected_subsample_max corrects unequal arm sizes.
        arm_observations = {}
        for r_idx, record in enumerate(responses):
            job_idx = record["job_idx"]
            job = prompt_jobs[job_idx]
            obs = arm_observations.setdefault(job["arm"], {
                "memory_ids": job["memory_ids"], "rewards": [], "valids": [],
                "codes": [],
            })
            obs["rewards"].append(float(rewards[r_idx]))
            obs["valids"].append(bool(valids[r_idx]))
            obs["codes"].append(codes[r_idx])
        if (memory is not None and mem_cfg is not None
                and bool(getattr(mem_cfg, "outcome_credit", False))):
            updates = credit_memory_arms(
                memory, arm_observations, parent_val, step_idx,
                parent_id=parent.id)
            mem_arm_updates.extend({"group": g, **update} for update in updates)
            for update in updates:
                print(f"    memory {update['arm']}: n={update['n']} "
                      f"tail uplift={update['tail_uplift']:+.9f} "
                      f"valid={update['valid']}/{update['rollouts']}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, record in enumerate(responses):
            text = record["text"]
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            res = outs[r_idx]
            job = prompt_jobs[job_idx]
            # Allocate a durable ID even for invalid/duplicate candidates. A
            # valid candidate uses this exact State object in sampler.update,
            # so a child selected in a later step links back to this rollout.
            child = State.make(
                timestep=step_idx,
                value=rewards[r_idx],
                code=res.code or "",
                raw_score=res.raw_score,
                construction=res.construction,
            )
            archive_eligible = bool(valids[r_idx] and codes[r_idx])
            if archive_eligible:
                all_children.append((child, parent))
            pick_info = (sampler.last_picks_info[g]
                         if g < len(sampler.last_picks_info) else {})
            meta = {
                "step": step_idx,
                "group": g,
                "rollout": r_idx,
                "node_id": child.id,
                "parent_id": parent.id,
                "parent_timestep": int(parent.timestep),
                "sampler_type": sampler_type,
                "reward": float(rewards[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "valid": bool(valids[r_idx]),
                "parsed": bool(res.parsed),
                "ran": bool(res.ran),
                "msg": res.msg,
                "failure_kind": res.failure_kind,
                "advantage": float(advantages[r_idx]) if hasattr(advantages, "__len__") else 0.0,
                "beta": float(adv_scale) if adv_mode == "entropic" else 0.0,
                "advantage_mode": adv_mode,
                "advantage_scale": (float(adv_scale)
                                    if math.isfinite(adv_scale) else None),
                "n_response_tokens": len(token_ids),
                "sandbox_stdout": (res.stdout or "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_raw_score": (float(parent.raw_score)
                                     if parent.raw_score is not None else None),
                "parent_is_seed": parent.id in sampler._seed_ids,
                "parent_visit_count": int(pick_info.get("n", 0)),
                "parent_q_value": (float(pick_info["Q"])
                                   if pick_info.get("Q") is not None else None),
                "parent_prior": (float(pick_info["P"])
                                 if pick_info.get("P") is not None else None),
                "parent_exploration_bonus": (float(pick_info["bonus"])
                                             if pick_info.get("bonus") is not None else None),
                "parent_selection_score": (float(pick_info["score"])
                                           if pick_info.get("score") is not None else None),
                "archive_eligible": archive_eligible,
                "memory_arm": job["arm"],
                "memory_ids": job["memory_ids"],
                "memory_tokens": job["memory_tokens"],
                # The solution itself, and the one it started from. Neither is
                # recoverable afterwards: `construction` lives only in the
                # in-memory sampler State, and a mid-run rollout's parent array
                # is gone by the time anyone wants to plot it. Saving the result
                # means reproducing a figure needs no re-execution at all, which
                # also sidesteps programs that are stochastic or wall-clock
                # bounded and therefore cannot replay identically.
                "construction": (_as_float_list(getattr(res, "construction", None),
                                                cfg.max_saved_construction)
                                 if save_ctor else None),
                "parent_construction": (_as_float_list(
                    getattr(parent, "construction", None),
                    cfg.max_saved_construction) if save_ctor else None),
                "seed": int(cfg.seed),
            }
            if rank_mode:
                meta["rank_selection"] = {
                    **rank_diagnostics,
                    "score": float(adv_info["ranks"][r_idx]),
                    "weight": float(adv_info["weights"][r_idx]),
                }
            if memory_v2:
                meta["memory_version"] = "V2"
                meta["memory_comparison_n"] = int(
                    getattr(mem_cfg, "arm_comparison_n", 0) or 0)
            save_rollout(exp_dir, step_idx, g, r_idx, text, meta,
                         prompt_text=job["prompt_text"])
            saved_rollouts += 1
            if memory is not None:
                mem_records.append(RolloutRecord(
                    step=step_idx, group=g, rollout=r_idx,
                    parent_summary=(
                        f"parent reward="
                        f"{(parent.value if parent.value is not None else 0.0):.9f}"),
                    parent_code=parent.code or "",
                    parent_reward=(float(parent.value)
                                   if parent.value is not None else None),
                    response=text,
                    code=res.code or "",
                    reward=float(rewards[r_idx]),
                    raw_score=res.raw_score,
                    valid=bool(valids[r_idx]),
                    parsed=bool(res.parsed),
                    ran=bool(res.ran),
                    msg=res.msg or "",
                    stdout=res.stdout or "",
                    memory_arm=job["arm"],
                    memory_ids=list(job["memory_ids"]),
                ))

        # ---- reprompt(x_p, f_i) for code failures only (Sec. 2.3) --------
        # Built here, while the RewardResult is in hand. The teacher forward
        # itself happens in the train loop, where log pi_thetabar is already
        # available from the existing forward pass.
        if fb_candidate_on:
            for r_idx, record in enumerate(responses):
                job_idx = record["job_idx"]
                res = outs[r_idx]
                if not is_code_failure(res):
                    continue
                f_i = format_feedback(res.msg or "", res.stdout or "",
                                      int(fb_cfg.chars))
                rp_messages = build_reprompt(prompt_jobs[job_idx]["messages"], f_i,
                                             mode=fb_cfg.inject_mode)
                reprompt_by_key[(g, r_idx)] = render_chat(
                    tokenizer, rp_messages,
                    enable_thinking=bool(cfg.thinking),
                )

        # If reward is constant in this group there is no A^rew signal. With
        # the feedback signal on, those rollouts are still worth training on:
        # A^rew_i = 0 but A^fb is not, which is the whole point of Eq. 9. This
        # is where it pays most, since an all-failed group is exactly the case
        # the reward channel cannot score at all.
        # Rank mode uses exact equality and retains fully tied groups for
        # reference updates. Entropy is gated separately above: an all-failure
        # tie at fail_score must not receive an entropy update.
        if (constant and not rank_mode
                and not (fb_candidate_on and fb_cfg.include_constant_groups)):
            continue

        for r_idx, (record, adv) in enumerate(
                zip(responses, advantages)):
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            if len(token_ids) == 0:
                continue
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.to(model.device)
            response_ids = torch.tensor([token_ids], device=model.device)
            behavior_values = record["behavior_logprobs"]
            reference_values = record["reference_logprobs"]
            behavior_logprobs = (
                torch.tensor(behavior_values, dtype=torch.float32,
                             device=model.device)
                if behavior_values is not None
                and len(behavior_values) == len(token_ids) else None)
            reference_logprobs = (
                torch.tensor(reference_values, dtype=torch.float32,
                             device=model.device)
                if reference_values is not None
                and len(reference_values) == len(token_ids) else None)
            res = outs[r_idx]
            all_examples.append({
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": response_ids,
                "advantage": float(adv),
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": reference_logprobs,
                "reprompt_text": reprompt_by_key.get((g, r_idx)),
                "failure_signature": RolloutRecord(
                    msg=res.msg or "").failure_signature(),
                "reward_constant": constant,
                "rank_entropy_gate": rank_entropy_gate,
                "group_id": g,
                "prompt_job_id": job_idx,
                # Implements mean_over_parents(mean_over_rollouts(loss)).
                "sample_weight": 1.0 / (len(parents) * len(responses)),
            })

    # Persistence barrier: every response/prompt/meta file is on disk before
    # memory work or any adapter forward/backward/update begins below. This is
    # intentionally separate from stepXX.summary.json, which is the completion
    # marker and therefore can only be written after the trained adapter and
    # resumable checkpoint have both been saved by the caller.
    print(f"[step {step_idx}] saved {saved_rollouts} rollout .txt/.meta.json "
          f"pairs before adapter training", flush=True)

    valid_fraction = (step_valid_count / step_rollout_count
                      if step_rollout_count else 0.0)
    code_valid_fraction = (1.0 - step_code_failure_count / step_rollout_count
                           if step_rollout_count else 1.0)
    fb_lambda = (fb_cfg.effective_lambda(step_idx, code_valid_fraction)
                 if fb_cfg is not None and fb_cfg.enabled else 0.0)
    fb_on = bool(fb_candidate_on and fb_lambda > 0.0)
    if fb_candidate_on:
        print(f"[step {step_idx}] feedback: code-validity="
              f"{code_valid_fraction:.1%} "
              f"({step_code_failure_count} code failures), "
              f"scheduled lambda={fb_base_lambda:.4f}, "
              f"effective lambda={fb_lambda:.4f}")

    # Constant groups only carry a feedback signal. Drop them when the adaptive
    # controller turns feedback off after seeing this step's code validity.
    if not fb_on and not rank_mode:
        all_examples = [ex for ex in all_examples if not ex["reward_constant"]]

    # Cap the teacher forwards. Applied to all_examples rather than to
    # reprompt_by_key, because the examples were built during the scoring loop
    # above and already hold their reprompt text; shrinking the dict now would
    # change nothing. Selection is balanced across failure signatures and
    # spread across the batch rather than restricted to the first groups.
    feedback_teacher_rollouts = 0
    feedback_step_cap = 0
    feedback_signature_cap = 0
    if fb_on:
        feedback_step_cap, feedback_signature_cap = fb_cfg.resolve_caps(
            cfg.groups_per_step, cfg.group_size)
        total_label = feedback_step_cap or "all"
        signature_label = feedback_signature_cap or "all"
        print(f"[step {step_idx}] feedback budget: total={total_label}, "
              f"per-signature={signature_label} for "
              f"G={cfg.groups_per_step}, K={cfg.group_size}")
        with_fb = [i for i, ex in enumerate(all_examples) if ex.get("reprompt_text")]
        signatures = [ex.get("failure_signature", "unknown") for ex in all_examples]
        keep = set(select_balanced(
            with_fb, signatures, total_cap=feedback_step_cap,
            per_signature_cap=feedback_signature_cap))
        if len(keep) < len(with_fb):
            for i in with_fb:
                if i not in keep:
                    all_examples[i]["reprompt_text"] = None
            print(f"[step {step_idx}] feedback: balanced/capped to {len(keep)} "
                  f"of {len(with_fb)} code-failed rollouts")
        feedback_teacher_rollouts = len(keep)

        # A constant-reward example with no retained repair prompt has neither
        # a reward nor a feedback signal. Keeping it would only run the policy
        # and reference forwards for a KL-only update and dilute the batch.
        all_examples = [
            ex for ex in all_examples
            if rank_mode or not (ex["reward_constant"] and not ex.get("reprompt_text"))
        ]

    if rank_mode and cfg.generation_backend == "vllm":
        # Rank/PPO requires the frozen behavior probability. A malformed vLLM
        # payload cannot be reconstructed safely after the adapter changes and
        # must never force a full-context HF cache pass. Rollouts remain saved;
        # only an unusable training record is omitted.
        before = len(all_examples)
        all_examples = [
            example for example in all_examples
            if _valid_example_token_logprobs(
                example, "behavior_logprobs")
        ]
        unusable = before - len(all_examples)
        if unusable:
            print(f"[step {step_idx}] vLLM omitted behavior logprobs for "
                  f"{unusable} rollout(s); saved them but excluded only those "
                  "records from the rank update", flush=True)

    rollout_time = time.time() - rollout_t0
    print(f"[step {step_idx}] rollout+eval time: {rollout_time:.1f}s  "
          f"training examples: {len(all_examples)}  new children: {len(all_children)}")

    # Update archive
    sampler.update(all_children)

    # Report the problem-native metric before any memory or gradient work. This
    # is deliberately separate from reward: some problems maximize the raw
    # quantity, while others (Erdos bounds, runtime, MSE) minimize it.
    best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    if best_raw is not None:
        direction = "higher is better" if problem.maximize else "lower is better"
        print(
            f"[step {step_idx}] best-ever raw {problem.metric_name}: "
            f"{float(best_raw.raw_score):.9f} ({direction}; "
            f"reward={float(best_raw.value):.9f}, found step={best_raw.timestep})",
            flush=True,
        )
    else:
        print(f"[step {step_idx}] best-ever raw {problem.metric_name}: unavailable",
              flush=True)

    # ----- MEMORY (Sec. 2.2) ---------------------------------------------
    # Deliberately above the early return below. A step where every group had
    # constant reward carries no RL signal but plenty of evidence, and that is
    # exactly the step where the search is stuck and needs the lessons.
    #
    # update() extracts, applies the reinforcements the maker asked for, and
    # inserts whatever is genuinely new, printing its own summary line.
    if memory is not None and extractor is not None:
        extractor.update(mem_records, step_idx, adapter_path=adapter_path)
        # Curation runs after insertion, so it sees this step's lessons too.
        if curator is not None and curator.due(step_idx):
            curator.run(step_idx, adapter_path=adapter_path)
        memory.save()

    step_stats = {
        "best_valid_yield": float(best_valid_yield),
        "distinct_good": int(len(distinct_good_hashes)),
        "valid_fraction": float(valid_fraction),
        "feedback_code_valid_fraction": float(code_valid_fraction),
        "feedback_lambda_effective": float(fb_lambda),
        "feedback_teacher_rollouts": int(feedback_teacher_rollouts),
        "feedback_step_cap": int(feedback_step_cap),
        "feedback_signature_cap": int(feedback_signature_cap),
        "memory_arm_rollouts": mem_arm_rollouts,
        "memory_arm_updates": mem_arm_updates,
        "evaluation_isolated": isolated_eval,
        "evaluation_workers": int(n_reward_workers),
        "evaluation_cpu_count": int(isolated_cpu_count),
        "evaluation_processes_per_cpu": int(isolated_processes_per_cpu),
    }
    if memory_v2:
        step_stats["memory_version"] = "V2"
        step_stats["memory_comparison_n"] = int(
            getattr(mem_cfg, "arm_comparison_n", 0) or 0)

    if rank_mode:
        step_stats["rank_groups"] = rank_group_stats

    if not all_examples:
        print(f"[step {step_idx}] no training signal (all groups had constant reward)")
        return step_stats

    if rank_mode:
        if parallel_trainer is not None:
            step_stats.update(parallel_trainer.train_rank(
                all_examples, cfg, step_idx, fb_cfg=fb_cfg, fb_on=fb_on,
                fb_lambda=fb_lambda))
        else:
            step_stats.update(_train_rank_examples(
                backend, model, tokenizer, optimizer, all_examples, cfg,
                step_idx, fb_cfg=fb_cfg, fb_on=fb_on,
                fb_lambda=fb_lambda))
        return step_stats

    # ----- TRAIN STEP -----
    print(f"[step {step_idx}] starting adapter training; rollout artifacts "
          f"are already on disk", flush=True)
    if parallel_trainer is not None:
        step_stats.update(parallel_trainer.train_policy(
            all_examples, cfg, step_idx, fb_cfg=fb_cfg, fb_on=fb_on,
            fb_lambda=fb_lambda))
        best = sampler.best_state()
        if best is not None:
            raw = f" raw={best.raw_score:.9f}" if best.raw_score is not None else ""
            print(f"[step {step_idx}] best so far: value={best.value:.9f}{raw}  "
                  f"(step total {time.time() - step_t0:.1f}s, "
                  f"archive={sampler.archive_size()})")
        return step_stats

    backend.set_training_mode()
    optimizer.zero_grad()

    train_t0 = time.time()
    n_examples = len(all_examples)
    microbatches = _training_microbatches(all_examples, cfg)
    largest_batch = max((len(batch) for batch in microbatches), default=0)
    print(f"[train] LoRA microbatches={len(microbatches)}; "
          f"configured max={int(cfg.train_examples_per_microbatch)}, "
          f"effective max={largest_batch}, "
          f"padded-token cap={int(cfg.max_seq_length)}", flush=True)

    def attempt(active_batches):
        total_loss = 0.0
        total_logp_delta = 0.0
        is_ratio_sum = 0.0
        is_ratio_max = 0.0
        is_ratio_count = 0
        fb_stats = FeedbackStats()
        for batch in active_batches:
            current_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=True, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)
            base_logprobs = [
                example.get("reference_logprobs") for example in batch]
            supplied_reference = all(
                _valid_example_token_logprobs(
                    example, "reference_logprobs")
                for example in batch)
            if not supplied_reference:
                try:
                    with backend.disable_adapter(), torch.no_grad():
                        base_logprobs = compute_batched_token_logprobs(
                            model, batch, with_grad=False,
                            chunk=cfg.logprob_chunk,
                            pad_token_id=tokenizer.pad_token_id)
                except Exception as error:
                    if not hasattr(train_step, "_kl_warned"):
                        print(f"[warn] disable_adapter failed ({error}); "
                              "training without KL penalty")
                        train_step._kl_warned = True
                    base_logprobs = [
                        current_lp.detach()
                        for current_lp in current_logprobs]

            batch_losses = []
            for ex, cur_lp, base_lp in zip(
                    batch, current_logprobs, base_logprobs):
                base_lp = base_lp.to(cur_lp.device)
                rid = ex["response_ids"]
                adv = ex["advantage"]
                logp_diff = (cur_lp - base_lp).detach()
                avg_logp_diff = logp_diff.mean()
                kl_adv = cfg.kl_penalty_coef * (
                    avg_logp_diff - (cur_lp - base_lp))
                eff_adv = adv + kl_adv

                if fb_on and ex.get("reprompt_text"):
                    fb_adv = feedback_advantage(
                        compute_token_logprobs, model, tokenizer,
                        ex["reprompt_text"], rid, cur_lp.detach(), fb_cfg,
                        lam=fb_lambda, chunk=cfg.logprob_chunk)
                    if fb_adv is None:
                        fb_stats.skipped += 1
                    else:
                        fb_adv, _fb_scale = bound_feedback_advantage(
                            fb_adv, reward_advantage=adv, cfg=fb_cfg)
                        fb_stats.add(fb_adv)
                        eff_adv = eff_adv + fb_adv

                behavior_lp = ex.get("behavior_logprobs")
                if _valid_example_token_logprobs(
                        ex, "behavior_logprobs"):
                    is_ratio = torch.exp(cur_lp.detach() - behavior_lp)
                    is_ratio_sum += float(is_ratio.mean().item())
                    is_ratio_max = max(
                        is_ratio_max, float(is_ratio.max().item()))
                    is_ratio_count += 1
                else:
                    if (behavior_lp is not None
                            and not hasattr(train_step, "_is_len_warned")):
                        print("[warn] invalid behavior logprobs; skipping IS "
                              "for affected examples")
                        train_step._is_len_warned = True
                    is_ratio = 1.0

                loss = -(is_ratio * eff_adv.detach() * cur_lp).mean()
                batch_losses.append(loss / n_examples)
                total_loss += float(loss.detach().item())
                total_logp_delta += float(logp_diff.mean().item())
            if batch_losses:
                sum(batch_losses[1:], batch_losses[0]).backward()
        return (total_loss, total_logp_delta, is_ratio_sum,
                is_ratio_max, is_ratio_count, fb_stats)

    result, _effective, quarantined = _run_oom_resilient_backward(
        model, microbatches, attempt, device_label=str(model.device))
    if result is None:
        result = (0.0, 0.0, 0.0, 0.0, 0, FeedbackStats())
    (total_loss, total_logp_delta, is_ratio_sum, is_ratio_max,
     is_ratio_count, fb_stats) = result

    import torch as _torch
    _torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=cfg.grad_clip,
    )
    optimizer.step()

    train_time = time.time() - train_t0
    step_stats["training_seconds"] = train_time
    step_stats["training_parallel_gpus"] = 1
    step_stats["oom_quarantined_examples"] = len(quarantined)
    is_msg = ""
    if is_ratio_count > 0:
        is_msg = (f"  IS ratio mean={is_ratio_sum / is_ratio_count:.9f} "
                  f"max={is_ratio_max:.3f}")
    print(f"[step {step_idx}] train time: {train_time:.1f}s  "
          f"avg loss: {total_loss / n_examples:.9f}  "
          f"avg logpi_theta - logpi_base: "
          f"{total_logp_delta / n_examples:.9f}{is_msg}  "
          f"OOM-quarantined={len(quarantined)}")
    if fb_on:
        print(fb_stats.line(step_idx, fb_lambda))

    best = sampler.best_state()
    if best is not None:
        raw = f" raw={best.raw_score:.9f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.9f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")

    return step_stats


# ======================================================================
# Batch-size growth controller
# ======================================================================
def grow_batch(cur_g, cur_k, stats, cfg):
    """
    Monotonic ratchet. Grow (G, K) toward (max_groups_per_step, max_group_size)
    only when BOTH signals from the step just finished clear their thresholds:
      - best_valid_yield: the best group's valid fraction, and
      - distinct_good: the count of unique children that beat their parent.
    Otherwise hold. Never shrinks. The step >= growth_force_step override lives
    in the caller, not here.
    """
    g_max = int(cfg.max_groups_per_step)
    k_max = int(cfg.max_group_size)
    stats = stats or {}
    grow = (float(stats.get("best_valid_yield", 0.0)) >= cfg.growth_valid_yield
            and int(stats.get("distinct_good", 0)) >= cfg.growth_distinct_min)
    if grow:
        cur_g = min(g_max, int(round(cur_g * cfg.growth_factor)))
        cur_k = min(k_max, int(round(cur_k * cfg.growth_factor)))
    return cur_g, cur_k


# ======================================================================
# Main
# ======================================================================
def main():
    cfg, merged = load_config()
    # This must precede every import path that can initialize CUDA. Worker and
    # evaluation children replace CUDA_VISIBLE_DEVICES with their own physical
    # groups before importing their CUDA stacks.
    _pin_training_process(cfg.training_gpu_ids)
    if cfg.evaluation_gpu_id is not None:
        os.environ["TTT_EVALUATION_GPU_ID"] = str(cfg.evaluation_gpu_id)

    # Select the run directory before runtime-only context adjustments so a
    # fresh config.json stores the reusable base configuration.
    from experiment_io import (make_experiment_dir, save_final_summary,
                               save_step_summary)
    resume_dir = merged.pop("_resume_dir", None)
    exp_dir = make_experiment_dir(
        cfg, resume_dir=resume_dir, config_dict=merged
    )
    vllm_log_path = None
    dependency_log_path = None
    if cfg.generation_backend == "vllm":
        vllm_log_path = str((Path(exp_dir).resolve() / "vllm.log"))
        with open(vllm_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== TTT vLLM log parent_pid={os.getpid()} "
                f"run_dir={Path(exp_dir).resolve()} ===\n")
        print(f"[logs] vLLM output: {vllm_log_path}", flush=True)
        dependency_log_path = vllm_log_path
    else:
        dependency_log_path = str(
            Path(exp_dir).resolve() / "dependency_warnings.log")
        with open(dependency_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== TTT dependency warnings parent_pid={os.getpid()} "
                f"run_dir={Path(exp_dir).resolve()} ===\n")
        print(f"[logs] dependency warnings: {dependency_log_path}", flush=True)

    # One effective seed for every generation stream. None => not seeded, which
    # is the original behaviour. Set once here and threaded through unchanged.
    run_seed = cfg.seed if cfg.deterministic else None
    print(f"[init] deterministic = {cfg.deterministic}"
          + (f" (seed {cfg.seed})" if cfg.deterministic else ""))

    # ---- memory context top-up (must happen before the model loads) ----
    # The injected block is granted context ON TOP of the no-memory setting,
    # so max_new_tokens and the room available to the response are identical in
    # both modes. Give the no-memory baseline the SAME final max_seq_length if
    # you want step 0 to be bit-identical, since the backend reads it at load.
    from memory import MemoryConfig
    mem_cfg = MemoryConfig.from_dict(merged)
    if mem_cfg.enabled and mem_cfg.grant_context and mem_cfg.token_budget > 0:
        cfg.max_seq_length += mem_cfg.token_budget
        merged["max_seq_length"] = cfg.max_seq_length
        print(f"[memory] context raised by {mem_cfg.token_budget} tokens for the "
              f"injected block: max_seq_length = {cfg.max_seq_length}. "
              f"Use the same value for the no-memory baseline.")

    # Replicate the compact QLoRA trainer rather than layer-sharding it. The
    # main process still owns search/evaluation; only per-example gradient work
    # runs concurrently. vLLM phase sharing gives these replicas exclusive use
    # of the cards during training and requires them to offload for generation.
    use_replicated_training = bool(
        int(cfg.num_training_gpus) > 1
        and cfg.backend == "hf"
        and cfg.generation_backend == "vllm"
    )
    if use_replicated_training:
        cfg.training_replica_device = 0

    # Build the problem from the merged config (the registry reads problem-only
    # knobs like num_circles / problem_type / budget_s / score_scale from here).
    from problems.registry import get_problem
    problem_config = merged
    if cfg.isolate_eval:
        # Prompts must describe the sandbox's real allocation. Keep the saved
        # YAML value intact for ordinary launches, but tell this problem
        # instance that every candidate owns exactly one CPU.
        problem_config = dict(merged)
        problem_config["eval_cpus"] = 1
    problem = get_problem(cfg.problem, problem_config)

    print("=" * 70)
    print("TTT-Discover — local multi-problem implementation")
    print("=" * 70)
    problem_type = getattr(cfg, "problem_type", "")
    print(f"Problem:            {cfg.problem}"
          + (f" ({problem_type})" if problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Model:              {cfg.model_name}")
    if cfg.training_model_name != cfg.model_name:
        print(f"Training model:     {cfg.training_model_name}")
    print(f"Training backend:   {cfg.backend}")
    print(f"Generation backend: {cfg.generation_backend}")
    print(f"Training GPUs:      physical {cfg.training_gpu_ids}")
    print(f"Training layout:    "
          f"{'process-distributed LoRA' if cfg.fast else ('replicated data parallel' if use_replicated_training else 'model parallel/single GPU')}")
    print(f"Training scheduler: "
          f"{'adaptive process-per-GPU (--fast)' if cfg.fast else 'default'}")
    print(f"Generation GPUs:    {cfg.gpu_ids or 'in-process'}")
    print(f"Evaluation GPU:     "
          f"{cfg.evaluation_gpu_id if cfg.evaluation_gpu_id is not None else 'none'}")
    evaluation_mode = (
        f"isolated, {cfg.reward_workers} processes/CPU (--isolate-eval)"
        if cfg.isolate_eval else "default")
    print(f"CPU evaluation:    {evaluation_mode}")
    if cfg.generation_backend == "vllm":
        print(f"vLLM parallelism:   TP={cfg.vllm_tensor_parallel_size or 'auto'} "
              f"PP={cfg.vllm_pipeline_parallel_size}")
    print(f"Target:             {cfg.target}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Advantage mode:     {getattr(cfg, 'advantage_mode', 'entropic')}")
    if getattr(cfg, "advantage_mode", "entropic") == "cvar":
        print(f"CVaR alpha/lambda:  {cfg.cvar_alpha} / {cfg.cvar_lambda}")
    if getattr(cfg, "advantage_mode", "entropic") == "rank":
        print(f"Rank gamma:         {cfg.rank_gamma}")
        print(f"Rank clip eps low/high: "
              f"{cfg.rank_clip_epsilon_low} / {cfg.rank_clip_epsilon_high} "
              f"(bounds {1.0 - cfg.rank_clip_epsilon_low:.4f} / "
              f"{1.0 + cfg.rank_clip_epsilon_high:.4f})")
        print(f"Rank entropy coef:  {cfg.rank_entropy_coef} (fully tied groups)")
        print(f"Rank update epochs: {cfg.rank_update_epochs}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    print(f"Max seq length:     {cfg.max_seq_length}")
    print(f"Train microbatch:   up to "
          f"{cfg.train_examples_per_microbatch} examples/GPU")
    print(f"Logprob chunk:      {cfg.logprob_chunk or 'off (single shot)'}")
    print(f"Seed:               {cfg.seed}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print(f"Memory:             {'on' if mem_cfg.enabled else 'off'}")
    print(f"Feedback signal:    "
          f"{'on' if bool(merged['feedback']) else 'off'}")
    print("=" * 70)

    # ---- experiment dir ----
    action = "resuming in" if resume_dir else "writing all rollouts to"
    print(f"[init] {action}: {exp_dir}")

    # ---- seed states (problem-defined) ----
    seeds = problem.seed_states()
    print(f"[init] problem produced {len(seeds)} seed state(s)")

    # ---- backend + model ----
    # Load backend FIRST so Unsloth can patch transformers if used.
    with _route_dependency_notices(dependency_log_path):
        from model_backend import load_backend
        backend = load_backend(cfg.backend, cfg)
        model, tokenizer = backend.load()
    effective_4bit = bool(getattr(cfg, "effective_load_in_4bit",
                                  cfg.load_in_4bit))
    print(f"[precision] training copy: "
          f"{'BitsAndBytes 4-bit' if effective_4bit else 'checkpoint/default precision'}")

    if cfg.generation_backend == "vllm":
        from gpu_runtime import validate_attention_heads
        validate_attention_heads(
            _attention_head_count(model),
            int(cfg.vllm_tensor_parallel_size),
            cfg.model_name,
        )

    import torch  # safe to import now
    import random
    random.seed(cfg.seed)
    if run_seed is not None:
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

    # Load policy weights before constructing the optimizer, so both describe
    # the same completed step.
    resume_payload = None
    legacy_resume = None
    start_step = 0
    if resume_dir:
        resume_payload = _load_training_checkpoint(exp_dir)
        if resume_payload is not None:
            start_step = int(resume_payload["next_step"])
            adapter_path = Path(exp_dir) / resume_payload["adapter_dir"]
            _load_adapter(model, adapter_path)
            print(f"[resume] exact checkpoint found; next step is {start_step}")
        else:
            legacy_resume = _legacy_resume_info(exp_dir)
            start_step, adapter_path = legacy_resume
            _load_adapter(model, adapter_path)
            print(
                f"[resume] legacy run (no training_state.pt): restarting step "
                f"{start_step} from {adapter_path.name}. The archive will be "
                "reconstructed, but old PUCT/optimizer/growth statistics are "
                "unavailable."
            )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        print("[resume] restored optimizer state")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,} / total {total:,} "
          f"({100 * trainable / total:.2f}%)")

    parallel_trainer = None
    if use_replicated_training:
        if cfg.fast:
            parallel_trainer = ProcessDistributedTrainer(
                backend, model, tokenizer, optimizer, cfg, exp_dir,
                dependency_log_path=dependency_log_path,
            )
        else:
            parallel_trainer = ReplicatedDataParallelTrainer(
                backend, model, tokenizer, optimizer, cfg,
                dependency_log_path=dependency_log_path,
            )

    # ---- sampler ----
    from sampler import PUCTSampler
    sampler = PUCTSampler(
        num_seeds=len(seeds) if seeds else cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
        seed_states=seeds,
    )
    if resume_payload is not None:
        sampler.load_state_dict(resume_payload["sampler"])
        print("[resume] restored exact PUCT archive and visit statistics")
    elif legacy_resume is not None:
        n_states, n_rollouts = _restore_legacy_archive(
            sampler, exp_dir, before_step=start_step
        )
        print(f"[resume] reconstructed {n_states} valid archived candidates "
              f"from {n_rollouts} earlier rollouts")
    print(f"[init] sampler archive size = {sampler.archive_size()}")

    # ---- generation pool ----
    gen_pool = None
    # vLLM forms exact TP/PP engines over every rollout card. Because those
    # cards also hold either training replicas or model shards, the two runtimes
    # alternate residency.
    use_gen_pool = bool(cfg.num_gpus) and (
        cfg.generation_backend == "vllm"
        or (cfg.num_gpus > 1 and cfg.num_training_gpus == 1))
    if use_gen_pool:
        from gen_workers import (GenerationPool, HybridHFGenerationPool,
                                 PhasedVLLMGenerationPool, worker_seed)
        gpu_ids = _parse_gpu_ids(cfg.gpu_ids)
        pool_options = dict(
            model_name=cfg.model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=(effective_4bit
                          and cfg.training_model_name == cfg.model_name),
            seed=run_seed,
            gen_micro_batch=cfg.gen_micro_batch,
            backend=cfg.generation_backend,
            lora_rank=cfg.lora_rank,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            vllm_enforce_eager=cfg.vllm_enforce_eager,
            vllm_enable_prefix_caching=cfg.vllm_enable_prefix_caching,
            vllm_quantization=cfg.vllm_quantization,
            vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
            vllm_pipeline_parallel_size=cfg.vllm_pipeline_parallel_size,
            vllm_max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
            vllm_enable_expert_parallel=cfg.vllm_enable_expert_parallel,
            vllm_log_path=vllm_log_path,
        )
        if cfg.generation_backend == "vllm":
            trainer_offloaded = False

            def _offload_trainer_for_generation():
                nonlocal trainer_offloaded
                if (parallel_trainer is not None
                        and parallel_trainer.is_offloaded):
                    return
                if parallel_trainer is None and trainer_offloaded:
                    return
                replica_label = (f"all {parallel_trainer.world_size} trainer "
                                 "replicas" if parallel_trainer is not None
                                 else "trainer")
                print(f"[gpu] offloading {replica_label} to CPU before "
                      "shared-GPU vLLM generation", flush=True)
                try:
                    if parallel_trainer is not None:
                        parallel_trainer.offload_for_generation()
                    else:
                        torch.cuda.synchronize()
                        _move_optimizer_state(optimizer, "cpu")
                        backend.offload_for_generation()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    trainer_offloaded = True
                except Exception as exc:
                    try:
                        if parallel_trainer is not None:
                            parallel_trainer.restore_after_generation()
                        else:
                            backend.restore_after_generation()
                            _restore_optimizer_state_to_parameters(optimizer)
                    except Exception:
                        pass
                    raise RuntimeError(
                        "the training model could not be offloaded for the "
                        "shared vLLM rollout phase; use generation_backend=hf "
                        "for live-model generation on this runtime") from exc

            def _restore_trainer_after_generation():
                nonlocal trainer_offloaded
                if parallel_trainer is not None:
                    # Memory lookup, rollouts, extraction, and curation may each
                    # open a separate vLLM phase in one step. Keep replicas on
                    # CPU between those phases and restore them once, lazily,
                    # when the actual gradient update begins.
                    print("[gpu] keeping trainer replicas offloaded until the "
                          "adapter update", flush=True)
                    return
                if not trainer_offloaded:
                    return
                print("[gpu] restoring trainer after shared-GPU vLLM "
                      "generation", flush=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                backend.restore_after_generation()
                _restore_optimizer_state_to_parameters(optimizer)
                backend.set_training_mode()
                trainer_offloaded = False

            gen_pool = PhasedVLLMGenerationPool(
                before_start=_offload_trainer_for_generation,
                after_stop=_restore_trainer_after_generation,
                **pool_options,
            )
            print(f"[init] phase-shared vLLM pool configured across all "
                  f"rollout GPUs {gpu_ids}")
        else:
            # Do not load a duplicate base model beside the trainer. Rank zero
            # is the live HF/Unsloth training model; only the remaining cards
            # need worker processes.
            remote_options = dict(pool_options)
            remote_options["num_workers"] = len(gpu_ids) - 1
            remote_options["gpu_ids"] = gpu_ids[1:]
            remote_pool = GenerationPool(**remote_options)
            local_cap = {"value": 0}

            def _local_hf_rollouts(prompts_by_group, counts_by_group,
                                   max_new_tokens, temperature, top_p,
                                   step_idx):
                backend.set_inference_mode()
                if run_seed is not None:
                    local_seed = worker_seed(run_seed, step_idx, 0)
                    torch.manual_seed(local_seed)
                    torch.cuda.manual_seed_all(local_seed)
                try:
                    yield from generate_prompt_jobs(
                        model, tokenizer, prompts_by_group, counts_by_group,
                        cfg, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p,
                        cap_state=local_cap)
                finally:
                    backend.set_training_mode()

            gen_pool = HybridHFGenerationPool(
                remote_pool=remote_pool, local_iter=_local_hf_rollouts)
            print(f"[init] hybrid HF rollout pool: live trainer on "
                  f"physical GPU {gpu_ids[0]} plus persistent workers "
                  f"{gpu_ids[1:]}")
        if cfg.gen_micro_batch and cfg.gen_micro_batch > 0:
            limit_name = ("max_num_seqs" if cfg.generation_backend == "vllm"
                          else "micro-batch")
            limit_scope = ("/engine" if cfg.generation_backend == "vllm"
                           else "/GPU")
            print(f"[init] generation pool ready "
                  f"({limit_name} {cfg.gen_micro_batch}{limit_scope})")
        else:
            print("[init] generation pool ready")
    else:
        if cfg.num_training_gpus > 1:
            print(f"[init] model-parallel HF generation across training GPUs "
                  f"{cfg.training_gpu_ids}; prompts remain cross-batched")
        else:
            print("[init] single-GPU generation (no worker pool)")

    # ---- memory (Sec. 2.2) ----
    from memory import setup_memory
    mem_cfg, memory, extractor, lookup, curator = setup_memory(
        merged, problem, cfg, mem_cfg=mem_cfg,
        backend=backend, model=model, tokenizer=tokenizer,
        # PoolMemoryLLM treats a shared-card vLLM call as its own phase: it
        # offloads the trainer through the pool callback, generates, then
        # releases the pool and restores training placement in a finally block.
        gen_pool=gen_pool,
        exp_dir=exp_dir, seed=run_seed,
    )
    if resume_payload is not None and memory is not None:
        memory_file = resume_payload.get("memory_file")
        if memory_file:
            memory_path = Path(exp_dir) / memory_file
            if not memory_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint memory snapshot not found: {memory_path}"
                )
            n_lessons = memory.load(memory_path)
            memory.save()  # undo a partially completed later step, if present
            print(f"[resume] restored {n_lessons} memory lessons from "
                  f"{memory_path.name}")
    elif legacy_resume is not None and memory is not None:
        print("[resume] warning: legacy memory.json cannot be rolled back to the "
              "restarted step; it will be reused as-is")
    # ---- feedback-based program-repair signal (Sec. 2.3) ----
    from feedback import FeedbackConfig
    fb_cfg = FeedbackConfig.from_dict(merged)
    print(f"[init] {fb_cfg.describe()}")
    if fb_cfg.enabled and fb_cfg.anneal_steps > 0:
        print(f"[init] lambda schedule: {fb_cfg.schedule_preview()}")

    # ---- Elo re-ranker (retired and explicitly disabled) ----
    reranker = None
    if bool(merged.get("reranker_enabled", False)):
        print("[init] saved config requested the retired Elo re-ranker; "
              "forcing it off")
    else:
        print("[init] Elo re-ranker disabled")

    # ---- adaptive batch growth: start from the configured (G, K) ----
    if resume_payload is not None:
        cur_g = int(resume_payload["next_groups_per_step"])
        cur_k = int(resume_payload["next_group_size"])
    else:
        cur_g = int(cfg.groups_per_step)
        cur_k = int(cfg.group_size)
    print(f"[init] batch growth: start G={cur_g} K={cur_k} -> "
          f"max G={cfg.max_groups_per_step} K={cfg.max_group_size}; "
          f"grow when best-valid-yield>={cfg.growth_valid_yield} and "
          f"distinct-good>={cfg.growth_distinct_min} (x{cfg.growth_factor}); "
          f"forced to max at step {cfg.growth_force_step}")

    # ---- main loop ----
    cumulative_training_seconds = 0.0
    for completed_step in range(start_step):
        summary_path = (Path(exp_dir) / f"step{completed_step:02d}"
                        / f"step{completed_step:02d}.summary.json")
        try:
            completed_summary = json.loads(summary_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        cumulative_training_seconds += float(
            completed_summary.get(
                "training_seconds",
                completed_summary.get("rank_train_seconds", 0.0),
            ) or 0.0)
    if cumulative_training_seconds:
        print(f"[resume] prior cumulative training time: "
              f"{cumulative_training_seconds:.1f}s")
    try:
        if start_step >= cfg.num_steps:
            print(f"[resume] run already reached requested num_steps={cfg.num_steps}")
        for step in range(start_step, cfg.num_steps):
            # Hard convergence: from growth_force_step on, run at the cap no
            # matter what the signals say.
            if step >= cfg.growth_force_step:
                cur_g, cur_k = int(cfg.max_groups_per_step), int(cfg.max_group_size)
            cfg.groups_per_step = cur_g
            cfg.group_size = cur_k
            print(f"[step {step}] batch: G={cur_g} K={cur_k} "
                  f"({cur_g * cur_k} rollouts)")

            stats = train_step(backend, model, tokenizer, sampler, optimizer, step,
                               cfg, exp_dir, problem, gen_pool,
                               memory=memory, extractor=extractor, mem_cfg=mem_cfg,
                               lookup=lookup, curator=curator, fb_cfg=fb_cfg,
                               parallel_trainer=parallel_trainer)

            stats = stats or {}
            step_training_seconds = float(
                stats.get("training_seconds",
                          stats.get("rank_train_seconds", 0.0)) or 0.0)
            cumulative_training_seconds += step_training_seconds
            stats["training_seconds"] = step_training_seconds
            stats["cumulative_training_seconds"] = cumulative_training_seconds
            print(f"[step {step}] TOTAL TRAINING TIME: "
                  f"{step_training_seconds:.1f}s  "
                  f"(run cumulative {cumulative_training_seconds:.1f}s)",
                  flush=True)

            # Ratchet up for the next step (skipped once we are in the forced
            # region, since we are already pinned to the max there).
            if step < cfg.growth_force_step:
                cur_g, cur_k = grow_batch(cur_g, cur_k, stats, cfg)

            # Version the memory and adapter first, then atomically advance the
            # state pointer. A crash during any write leaves the previous
            # adapter/checkpoint pair valid.
            memory_path = None
            if memory is not None:
                memory_path = Path(exp_dir) / f"memory_step{step:03d}.json"
                memory.save(memory_path)
            adapter_path = _save_adapter(
                model, exp_dir, step, cfg.model_name)
            checkpoint_path = _save_training_checkpoint(
                exp_dir, step + 1, adapter_path, sampler, optimizer,
                next_g=cur_g, next_k=cur_k, memory_path=memory_path,
            )
            save_step_summary(exp_dir, step, {
                "step": step,
                "completed": True,
                "next_step": step + 1,
                "adapter_dir": Path(adapter_path).name,
                "checkpoint": Path(checkpoint_path).name,
                "archive_size": sampler.archive_size(),
                "next_groups_per_step": cur_g,
                "next_group_size": cur_k,
                **(stats or {}),
            })
            print(f"[checkpoint] completed step {step}; resume at step {step + 1}")
    finally:
        if reranker is not None:
            print("[shutdown] stopping Elo re-ranker ...")
            reranker.stop()
        if gen_pool is not None:
            print("[shutdown] stopping generation pool ...")
            gen_pool.shutdown()
        if (parallel_trainer is not None
                and hasattr(parallel_trainer, "shutdown")):
            print("[shutdown] stopping process-distributed trainer ...")
            parallel_trainer.shutdown()

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    best = sampler.best_state()
    if best is not None:
        raw = f"  (raw {getattr(problem, 'metric_name', 'metric')} = {best.raw_score:.9f})" \
            if best.raw_score is not None else ""
        print(f"Best reward (higher=better): {best.value:.9f}{raw}")
        print(f"Found at step:     {best.timestep}")
        print(f"\n--- best code ---\n{best.code}\n--- end ---")
        save_final_summary(exp_dir, best.value, best.code, best.timestep,
                           best_construction=(_as_float_list(
                               getattr(best, "construction", None),
                               cfg.max_saved_construction)
                               if getattr(problem, "saves_construction", False)
                               else None),
                           best_raw_score=(float(best.raw_score)
                                           if best.raw_score is not None else None))
    else:
        print("No valid solution was ever produced.")
        save_final_summary(exp_dir, None, None, None)
    if memory is not None:
        c = memory.counts()
        print(f"\nMemory: {c['total']} lessons "
              f"({c['success']}+/{c['failure']}-, "
              f"{c['local']} local/{c['global']} global)")
        print(f"        {memory.usage_summary()}")
        print(f"        {memory.stats}")
        memory.save()

    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()
