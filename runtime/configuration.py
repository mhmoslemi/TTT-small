"""CLI/YAML/resume resolution, validation, and automatic resource placement."""

import argparse
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import yaml

from models.strategist_models import (
    resolve_strategy_effort, resolve_strategy_model, resolve_strategy_token_limits, strategist_profile,
)
from models.coder_profiles import (
    _apply_coder_model_profile, _apply_qwen3_8b_thinking_sampling, _apply_strategy_model_profile,
    _dual_resident_qwen3_8b_strategy_coder_pools, _is_exact_qwen3_30b_a3b, _resolve_coder_token_limits,
)
from training.exact_overflow import (
    _clear_exact_sharded_recovery, _infer_exact_sharded_recovery, _read_exact_sharded_recovery,
    _write_exact_sharded_recovery,
)


ADVANTAGE_MODES = ("entropic", "spo-rs")
SEQUENCE_CLIP_EPSILON_DEFAULT = 0.2
SPO_RS_UPDATE_EPOCHS_DEFAULT = 1
SPO_RS_BETA_DEFAULT = 2.0
SPO_RS_D_HALF_DEFAULT = 0.06
SPO_RS_RHO_MIN_DEFAULT = 0.875
SPO_RS_RHO_MAX_DEFAULT = 0.96
SPO_RS_CLIP_EPSILON_DEFAULT = 0.2
SPO_RS_CLIP_EPSILON_LOW_DEFAULT = 0.2
SPO_RS_CLIP_EPSILON_HIGH_DEFAULT = 0.35


def _uses_clipped_policy_loss(mode):
    return str(mode or "entropic").lower() == "spo-rs"


def _normalize_saved_advantage_config(saved):
    """Migrate retained policy controls without reactivating removed objectives."""
    saved = dict(saved)
    mode = str(saved.get("advantage_mode") or "entropic").lower()
    if mode not in ADVANTAGE_MODES or saved.get("binary_coder_training", False):
        raise ValueError('saved run uses an unsupported training objective; only entropic and spo-rs can be resumed')
    if mode == "spo-rs" and "rank_update_epochs" in saved:
        saved.setdefault("spo_rs_update_epochs", saved["rank_update_epochs"])
    if mode == "entropic":
        for side in ("low", "high"):
            value = saved.get(f"rank_clip_epsilon_{side}")
            if value is None:
                value = saved.get("rank_clip_epsilon")
            if value is not None:
                saved.setdefault(f"sequence_clip_epsilon_{side}", value)
    for key in list(saved):
        if key.startswith(("rank_", "x_grpo_", "cvar_", "binary_coder_")):
            saved.pop(key)
    return saved


def _resolve_policy_options(merged):
    """Resolve full-policy sampling and the entropic MoE clipping bounds."""
    for side in ("low", "high"):
        name = f"sequence_clip_epsilon_{side}"
        value = merged.get(name)
        value = SEQUENCE_CLIP_EPSILON_DEFAULT if value is None else float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")
        merged[name] = value
    sequence_policy = _is_exact_qwen3_30b_a3b(merged.get("model_name"))
    if _uses_clipped_policy_loss(merged.get("advantage_mode")) or sequence_policy:
        if int(merged["group_size"]) < 2:
            raise ValueError("sequence/clipped policy modes require group_size >= 2")
        kl_coef = float(merged["kl_penalty_coef"])
        if not math.isfinite(kl_coef) or kl_coef < 0.0:
            raise ValueError("kl_penalty_coef must be finite and nonnegative")
        if float(merged["temperature"]) != 1.0 or float(merged["top_p"]) != 1.0:
            print("[config] sequence/clipped policy training sets temperature=1 "
                  "and top_p=1 to match its policy likelihood")
        merged["temperature"] = 1.0
        merged["top_p"] = 1.0
        merged["sampling_top_k"] = 0
        merged["sampling_min_p"] = 0.0
        merged["qwen3_8b_thinking_sampling"] = False


def _resolve_spo_rs_options(merged):
    if merged.get("advantage_mode") != "spo-rs":
        return

    epochs = merged.get("spo_rs_update_epochs")
    epochs = SPO_RS_UPDATE_EPOCHS_DEFAULT if epochs is None else epochs
    if (isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1):
        raise ValueError("spo_rs_update_epochs must be a positive integer")
    merged["spo_rs_update_epochs"] = int(epochs)
    raw_clip_epsilon = merged.get("spo_rs_clip_epsilon")
    defaults = {
        "spo_rs_beta": SPO_RS_BETA_DEFAULT,
        "spo_rs_d_half": SPO_RS_D_HALF_DEFAULT,
        "spo_rs_rho_min": SPO_RS_RHO_MIN_DEFAULT,
        "spo_rs_rho_max": SPO_RS_RHO_MAX_DEFAULT,
        "spo_rs_clip_epsilon": SPO_RS_CLIP_EPSILON_DEFAULT,
    }
    for name, default in defaults.items():
        raw = merged.get(name)
        value = default if raw is None else float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        merged[name] = value
    if merged["spo_rs_beta"] <= 0.0:
        raise ValueError("spo_rs_beta must be positive")
    adaptive_beta = merged.get("spo_rs_adaptive_beta", False)
    if not isinstance(adaptive_beta, bool):
        raise ValueError("spo_rs_adaptive_beta must be true or false")
    merged["spo_rs_adaptive_beta"] = bool(adaptive_beta)
    from training.spo_rs import BOUNDED_SHAPING_VERSION
    # The old adaptive-beta estimator did not have a historical baseline.
    merged["spo_rs_reward_shaping"] = (BOUNDED_SHAPING_VERSION if adaptive_beta else "exponential")
    if merged["spo_rs_d_half"] <= 0.0:
        raise ValueError("spo_rs_d_half must be positive")
    if not (0.0 < merged["spo_rs_rho_min"] <= merged["spo_rs_rho_max"] < 1.0):
        raise ValueError("SPO-RS rho bounds must satisfy 0 < rho_min <= rho_max < 1")
    if not 0.0 < merged["spo_rs_clip_epsilon"] < 1.0:
        raise ValueError("spo_rs_clip_epsilon must be in (0, 1)")
    side_defaults = {
        "low": SPO_RS_CLIP_EPSILON_LOW_DEFAULT,
        "high": SPO_RS_CLIP_EPSILON_HIGH_DEFAULT,
    }
    for side, default in side_defaults.items():
        name = f"spo_rs_clip_epsilon_{side}"
        raw = merged.get(name)
        if raw is None:
            value = (merged["spo_rs_clip_epsilon"] if raw_clip_epsilon is not None else default)
        else:
            value = float(raw)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")
        merged[name] = value


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTT-Discover multi-problem runner")
    # Problem selection
    p.add_argument("--problem", default=None,
                   help="Problem name. Loads configs/<problem>.yaml unless --config "
                        "is given. Defaults to erdos. "
                        "One of: circle_packing, "
                        "erdos, erdos-c4, ac1, ac2, denoising, gpu_mode.")
    p.add_argument("--config", default=None, help="Explicit path to a YAML config (overrides the --problem lookup).")
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
        help="Use the opt-in process-per-GPU LoRA trainer: work is balanced "
             "by sequence cost and every GPU independently learns a padded-"
             "token budget capped at 80%% GPU memory use. "
             "Without this flag the existing trainer is unchanged.")
    p.add_argument(
        "--strategies", action="store_const", const=True, default=None,
        help="Use the opt-in two-stage rollout path: a frozen reasoning model "
             "first produces sequential strategies, then the trainable coder "
             "policy produces programs conditioned on those strategies. "
             "Without this flag ordinary one-stage rollouts are unchanged.")
    p.add_argument(
        "--measure-entropy", action="store_const", const=True, default=None,
        help="Measure detached full-vocabulary coder entropy from existing "
             "pre-update training forwards, plus code diversity and validity. "
             "Save entropy.jsonl and a combined entropy PDF each completed step. "
             "No extra model forward; unscored rollouts have explicit missing "
             "coverage (--no-train records diversity only).")
    p.add_argument(
        "--fused-long-attention", action="store_const", const=True,
        default=None,
        help="Use exact fused causal SDPA for eligible unpadded long training "
             "sequences, with the existing exact blockwise attention as the "
             "automatic fallback. Without this flag attention is unchanged.")
    p.add_argument(
        "--no-train", action="store_const", const=True, default=None,
        help="Run rollout, evaluation, and search/archive updates but "
             "skip training-only logprob scoring, backward passes, and "
             "optimizer updates entirely.")
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
        "--coder-model-name", default=None,
        help="Policy checkpoint selected as --model-name when --strategies is "
             "active and --model-name was not explicitly supplied.")
    p.add_argument(
        "--coder-training-model-name", default=None,
        help="Optional trainable checkpoint paired with --coder-model-name. "
             "Used only by --strategies unless --training-model-name is explicit.")
    p.add_argument("--strategy-model-name", default=None,
                   help="Base reasoning checkpoint used only to generate "
                        "strategies; it never receives or trains the LoRA.")
    p.add_argument(
        "--api-strategy", default=None,
        help="OpenAI strategy model (gpt-6-luna). A nonempty value enables "
             "strategy rounds through the API; '' keeps local strategy selection.")
    p.add_argument(
        "--strategy-backend", choices=["local", "api"], default=None,
        help="Run the frozen strategist through a local generation pool or "
             "an OpenAI-compatible remote API.")
    p.add_argument(
        "--strategy-api", dest="strategy_backend", action="store_const",
        const="api",
        help="Shorthand for --strategy-backend api. No strategist weights or "
             "tokenizer are loaded locally.")
    p.add_argument("--strategy-api-base-url", default=None)
    p.add_argument("--strategy-api-key-env", default=None,
                   help="Environment-variable name containing the API key; "
                        "the key itself is never stored in run configuration.")
    p.add_argument("--strategy-api-concurrency", type=int, default=None)
    p.add_argument("--strategy-api-timeout-s", type=float, default=None)
    p.add_argument("--strategy-api-max-retries", type=int, default=None)
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
    p.add_argument("--num-steps", type=int, default=None, help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=None, help='Parent states per step.')
    p.add_argument("--group-size", type=int, default=None, help='Rollouts per parent group.')
    p.add_argument("--strategies-per-parent", type=int, default=None,
                   help="Sequential diverse strategies generated per parent.")
    p.add_argument("--programs-per-strategy", type=int, default=None,
                   help="LoRA-policy code rollouts sampled from each strategy.")
    p.add_argument(
        "--pilot-programs-per-strategy", type=int, default=None,
        help="Two-phase rollout allocation: first sample and evaluate this "
             "many programs per strategy, then dynamically redistribute the "
             "unchanged remaining per-parent rollout budget from pilot reward "
             "mean/variance. -1 preserves the existing one-pass generation "
             "path exactly.")
    p.add_argument(
        "--strategy-archive-top-r", type=int, default=None,
        help="With --strategies, admit only the top-r valid unique programs "
             "from each generated strategy before applying the existing "
             "topk_children_per_parent archive limit.")
    p.add_argument("--num-seed-states", type=int, default=None)
    p.add_argument("--parent-selection", choices=("puct", "uct", "llm"),
                   default=None, help="Override the YAML picker; on resume, "
                   "a switch to llm starts at the next fresh step. "
                   "llm uses GPT-6 Luna with maximum reasoning.")
    p.add_argument("--uct", dest="parent_selection", action="store_const",
                   const="uct", default=None,
                   help="Compatibility alias for --parent-selection uct.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--coder-retry", action="store_const", const=True, default=None,
        help="If fewer than 50%% of a coder rollout batch contain complete "
             "final code blocks, retry only enough incomplete responses once "
             "to target 60%% valid output. "
             "Disabled unless this flag is passed; originals are retained "
             "and retries are extra training examples.")
    p.add_argument("--strategy-max-new-tokens", type=int, default=None)
    p.add_argument("--strategy-format-max-retries", type=int, default=None,
                   help="Additional attempts for missing strategy blocks; "
                        "stop if exhausted (default: 3).")
    p.add_argument("--strategy-max-seq-length", type=int, default=None)
    p.add_argument("--strategy-temperature", type=float, default=None)
    p.add_argument("--strategy-top-p", type=float, default=None)
    p.add_argument("--strategy-thinking", dest="strategy_thinking", action="store_const", const=True, default=None)
    p.add_argument("--no-strategy-thinking", dest="strategy_thinking", action="store_const", const=False)
    p.add_argument("--strategy-reasoning-effort",
                   choices=["auto", "none", "low", "medium", "high", "xhigh", "max"],
                   default=None,
                   help="Model-native reasoning level; auto selects its "
                        "profile default. Qwen Flash-Next: low/medium/xhigh; "
                        "DeepSeek V4 0731: low/high/max.")
    p.add_argument("--strategy-vllm-quantization", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--adam-beta1", type=float, default=None)
    p.add_argument("--adam-beta2", type=float, default=None)
    p.add_argument("--adam-epsilon", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--kl-penalty-coef", type=float, default=None)
    p.add_argument("--advantage-mode", type=str.lower,
                   choices=list(ADVANTAGE_MODES), default=None,
                   help="Objective: entropic (adaptive beta within each group, "
                        "default) or spo-rs (historical entropic value tracking "
                        "with asymmetric PPO clipping).")
    p.add_argument("--spo-rs-update-epochs", type=int, default=None,
                   help="SPO-RS updates per rollout batch with a frozen old "
                        "policy (default 1; --fast requires 1).")
    p.add_argument("--sequence-clip-epsilon-low", type=float, default=None,
                   help="Entropic Qwen3-30B-A3B sequence-ratio clipping distance "
                        "below 1 (default 0.2).")
    p.add_argument("--sequence-clip-epsilon-high", type=float, default=None,
                   help="Entropic Qwen3-30B-A3B sequence-ratio clipping distance "
                        "above 1 (default 0.2).")
    beta_mode = p.add_mutually_exclusive_group()
    beta_mode.add_argument(
        "--adaptive-beta-spo", "--spo-rs-adaptive-beta",
        dest="spo_rs_adaptive_beta", action="store_const", const=True,
        default=None,
        help="SPO-RS: adaptive historical arctan reward shaping (25%% wider), "
             "with the run-wide moving-average baseline and all rollouts.")
    beta_mode.add_argument(
        "--constant-beta-spo", "--spo-rs-fixed-beta",
        dest="spo_rs_adaptive_beta", action="store_const", const=False,
        default=None,
        help="SPO-RS: use the fixed --spo-rs-beta/YAML value (default mode).")
    p.add_argument("--spo-rs-beta", type=float, default=None,
                   help="SPO-RS: exponential coefficient and advantage divisor "
                        f"(default {SPO_RS_BETA_DEFAULT}).")
    p.add_argument("--spo-rs-d-half", type=float, default=None,
                   help="SPO-RS: policy-KL half-life D_half "
                        f"(default {SPO_RS_D_HALF_DEFAULT}).")
    p.add_argument("--spo-rs-rho-min", type=float, default=None,
                   help="SPO-RS: minimum retained-history factor "
                        f"(default {SPO_RS_RHO_MIN_DEFAULT}).")
    p.add_argument("--spo-rs-rho-max", type=float, default=None,
                   help="SPO-RS: maximum retained-history factor "
                        f"(default {SPO_RS_RHO_MAX_DEFAULT}).")
    p.add_argument("--spo-rs-clip-epsilon", type=float, default=None,
                   help="SPO-RS: symmetric shorthand that sets both clipping "
                        "distances unless a side is specified explicitly.")
    p.add_argument("--spo-rs-clip-epsilon-low", type=float, default=None,
                   help="SPO-RS: distance below ratio 1 "
                        f"(default {SPO_RS_CLIP_EPSILON_LOW_DEFAULT}).")
    p.add_argument("--spo-rs-clip-epsilon-high", type=float, default=None,
                   help="SPO-RS: distance above ratio 1 "
                        f"(default {SPO_RS_CLIP_EPSILON_HIGH_DEFAULT}).")
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
                        "0 = single shot. Bounds projection memory on "
                        "large-vocabulary models.")
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
    p.add_argument("--no-vllm-enforce-eager", dest="vllm_enforce_eager", action="store_const", const=False)
    p.add_argument("--vllm-prefix-caching", dest="vllm_enable_prefix_caching",
                   action="store_const", const=True, default=None)
    p.add_argument("--no-vllm-prefix-caching", dest="vllm_enable_prefix_caching", action="store_const", const=False)
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


    return p


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

    Explicit legacy QLoRA configurations may still resolve a trainable
    BitsAndBytes conversion. The supported GPT-OSS coder profile rejects that
    split-policy setup and trains the original checkpoint's packed MXFP4 base.
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
    """Select the training loader from the checkpoint when backend is auto.

    GPT-OSS needs Unsloth for its packed/split expert layouts. All other
    checkpoints keep the established HF path. An explicit backend remains an
    override, except that an incompatible GPT-OSS BNB/HF pairing is corrected.
    """
    resolved = str(backend).strip().lower()
    checkpoint = str(training_model_name).strip().lower()
    if resolved == "auto":
        return "unsloth" if "gpt-oss" in checkpoint else "hf"
    requires_unsloth = ("gpt-oss" in checkpoint and "unsloth-bnb-4bit" in checkpoint)
    if requires_unsloth and resolved in ("auto", "hf"):
        return "unsloth"
    return resolved


def _resolve_training_memory_budgets(training_gpu_ids, memory, *, max_fraction=0.90):
    """Reserve host/runtime headroom and cap checkpoint placement per card."""
    if not memory or any(gpu_id not in memory for gpu_id in training_gpu_ids):
        return []
    budgets = []
    for gpu_id in training_gpu_ids:
        gpu = memory[gpu_id]
        # Keep at least 6 GiB outside Accelerate's weight placement for CUDA,
        # activations, temporary quantization buffers, and LoRA optimizer state.
        usable = min(float(gpu.total_gib) * float(max_fraction), float(gpu.free_gib) - 6.0)
        budgets.append(round(max(1.0, usable), 1))
    return budgets


def _validate_known_training_capacity(training_model_name, budgets, *, replicated=False):
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
        required_gib, precision = 65.0, "packed MXFP4 LoRA"
    available_gib = min(budgets) if replicated else sum(budgets)
    if required_gib is not None and available_gib < required_gib:
        placement = "on every replica GPU" if replicated else "in aggregate"
        remedy = (
            "Use larger cards or a smaller trainable checkpoint."
            if replicated else
            "Add GPUs or use a smaller trainable checkpoint."
        )
        raise ValueError(
            f"{training_model_name} needs about {required_gib:.0f} GiB for "
            f"{precision}, but the selected training GPUs provide only "
            f"{available_gib:.1f} GiB of safe placement budget {placement} "
            f"{budgets}. {remedy}")


def load_config():
    """
    Load one complete problem YAML, then overlay resume state and explicit CLI.

    Returns (cfg, merged) where:
      cfg    is the attribute-style view of the fully merged YAML, and
      merged is the full dict (including problem-only keys like num_circles,
             problem_type, budget_s, score_scale, gpu_type, task_yaml, lib_dir),
             which is what the problem registry consumes.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

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
            raise FileNotFoundError(f"--resume directory has no config.json: {resume_dir}")
        saved = _normalize_saved_advantage_config(json.loads(saved_config_path.read_text()))

    # The launcher defaults to erdos. Direct Python calls use the same selection
    # when neither --config nor --problem is provided.
    problem_name = (saved.get("problem") if saved else (args.problem if args.problem is not None else "erdos"))

    # 1) Complete, self-contained problem YAML.
    config_dir = Path(__file__).resolve().parents[1] / "configs"
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
    from config_validation import (RETIRED_BATCH_GROWTH_KEYS, validate_problem_config)
    for key in RETIRED_BATCH_GROWTH_KEYS:
        ydict.pop(key, None)
    # Old external configs may retain the retired repair-feature settings.
    for key in list(ydict):
        if key == "feedback" or key.startswith("feedback_"):
            ydict.pop(key)
    validate_problem_config(
        ydict,
        source=cfg_path,
        require_complete=(Path(cfg_path).expanduser().resolve().parent == config_dir.resolve()),
    )
    merged = dict(ydict)
    # Automatic model profiles provide defaults only. Track every value that
    # came from an operator-controlled source so a profile cannot silently
    # replace it later. Explicit CLI values remain stronger than YAML, and a
    # resumed run's saved configuration remains stronger than a fresh profile.
    profile_explicit_keys = set(ydict)
    print(f"[config] loaded {cfg_path}")

    # The registry routing key is the YAML's `problem` field when present
    # (this lets e.g. configs/gpu_mode_trimul.yaml declare `problem: gpu_mode`
    # while --problem just selects the file). With no YAML, --problem is the key.
    merged["problem"] = ydict.get("problem", problem_name)

    # 2) Saved config overlay. Retired feature keys are ignored so current runs
    # can resume checkpoints written before those optional systems were removed.
    if saved:
        saved.pop("_max_seq_length_includes_memory_topup", None)
        saved.pop("model_native_context_length", None)  # Retired diagnostic metadata.
        for key in list(saved):
            if (key == "memory" or key.startswith("memory_")
                    or key == "feedback" or key.startswith("feedback_")
                    or key.startswith("reranker_")
                    or key in RETIRED_BATCH_GROWTH_KEYS):
                saved.pop(key, None)
        profile_explicit_keys.update(saved)
        merged.update(saved)
        # Older checkpoints must not inherit a new preset's seed resolutions.
        if merged.get("problem") == "erdos":
            merged["seed_min_points"] = saved.get("seed_min_points", 40)
            merged["seed_max_points"] = saved.get("seed_max_points", 99)
        # A changed rollout budget restarts the unfinished step; saved
        # policy/optimizer/search state still comes from the last checkpoint.
        from experiment_io import ROLLOUT_CONFIG_KEYS
        for key in ROLLOUT_CONFIG_KEYS:
            if key in ydict:
                merged[key] = ydict[key]
        # Current strategist controls must not inherit stale resolved limits
        # or sampling defaults from a previous checkpoint's config.json.
        for key in ("strategy_max_seq_length", "strategy_max_new_tokens",
                    "strategy_temperature", "strategy_top_p",
                    "strategy_top_k", "strategy_min_p"):
            if key in ydict:
                merged[key] = ydict[key]
        # Preserve legacy one-pass generation only when neither source
        # specifies a pilot budget.
        if ("pilot_programs_per_strategy" not in saved and "pilot_programs_per_strategy" not in ydict):
            merged["pilot_programs_per_strategy"] = -1
        # Runs created before selectable phase-2 allocation used the original
        # median-quadrant rule.  Do not silently switch a resumed run to a new
        # allocator merely because its current problem YAML selects one.
        if "phase2_allocation_method" not in saved:
            merged["phase2_allocation_method"] = "rule_based"
        merged["spo_rs_adaptive_beta"] = bool(saved.get("spo_rs_adaptive_beta", False))
        # A new YAML selector must not switch an old resumed run's search.
        from search.sampler import parent_selection_mode
        merged["parent_selection"] = parent_selection_mode(saved)
        # Introducing an API-enabled preset must not switch an older resumed
        # run to a paid service. An explicit CLI selector can still opt in.
        merged["api_strategy"] = saved.get("api_strategy", "")
        print(f"[config] resuming saved configuration from "
              f"{resume_dir / 'config.json'}; current YAML rollout counts "
              "and strategy limits/sampling override saved values")

    # 3) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type", "resume"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
        if value != parser.get_default(arg_name):
            profile_explicit_keys.add(key)
    from search.sampler import can_migrate_parent_selection, parent_selection_mode
    merged["parent_selection"] = parent_selection_mode(merged)
    merged["uct"] = merged["parent_selection"] == "uct"
    if saved and merged["parent_selection"] != parent_selection_mode(saved):
        if not can_migrate_parent_selection(parent_selection_mode(saved), merged["parent_selection"]):
            raise ValueError('cannot change parent_selection while resuming except PUCT/UCT to LLM')
        if not (resume_dir / "training_state.pt").is_file():
            raise ValueError('parent-picker migration requires training_state.pt; refusing to reset the trained run')
    if (saved and args.spo_rs_adaptive_beta is not None
            and bool(saved.get("spo_rs_adaptive_beta", False))
            != bool(args.spo_rs_adaptive_beta)):
        raise ValueError(
            "cannot change fixed/adaptive SPO beta mode while resuming; "
            "start a fresh run for a different objective")
    if saved:
        from training.spo_rs import validate_shaping_resume
        validate_shaping_resume(saved)
    if args.spo_rs_clip_epsilon is not None:
        if args.spo_rs_clip_epsilon_low is None:
            merged["spo_rs_clip_epsilon_low"] = args.spo_rs_clip_epsilon
        if args.spo_rs_clip_epsilon_high is None:
            merged["spo_rs_clip_epsilon_high"] = args.spo_rs_clip_epsilon
    # Trainer/evaluator modes are launch-scoped. A nonempty API selector is
    # also an explicit opt-in to the strategy hierarchy.
    merged["fast"] = bool(args.fast)
    api_strategy = merged.get("api_strategy", "")
    if not isinstance(api_strategy, str):
        raise ValueError("api_strategy must be a model name or ''")
    merged["api_strategy"] = api_strategy.strip()
    if merged["api_strategy"] not in {"", "gpt-6-luna"}:
        raise ValueError("api_strategy supports gpt-6-luna or ''")
    merged["strategies"] = bool(args.strategies) or bool(merged["api_strategy"])
    if (saved.get("api_strategy") and args.api_strategy == "" and args.strategy_backend is None):
        merged["strategy_backend"] = ydict.get("strategy_backend", "local")
    merged["fused_long_attention"] = bool(args.fused_long_attention)
    merged["measure_entropy"] = bool(args.measure_entropy)
    merged["no_train"] = bool(args.no_train)
    merged["isolate_eval"] = bool(args.isolate_eval)
    # Coder recovery is deliberately launch-scoped. A saved run or external
    # YAML must not silently turn it back on when --coder-retry is absent.
    merged["coder_retry"] = bool(args.coder_retry)
    if merged["strategies"]:
        # Keep the ordinary policy checkpoint untouched unless the opt-in
        # hierarchy is requested. Explicit CLI model choices remain strongest.
        if args.model_name is None:
            merged["model_name"] = str(merged["coder_model_name"]).strip()
            if args.training_model_name is None:
                merged["training_model_name"] = str(merged.get('coder_training_model_name') or '').strip()

    # Resolve after the hierarchy chooses its coder and before validating the
    # resulting training layout and precision.
    _apply_coder_model_profile(merged, profile_explicit_keys)
    _resolve_coder_token_limits(merged)

    exact_sharded_recovery = _read_exact_sharded_recovery(resume_dir)
    if exact_sharded_recovery is not None:
        recovery_step = int(exact_sharded_recovery["step"])
        completed_summary = (Path(resume_dir) / f"step{recovery_step:02d}" / f"step{recovery_step:02d}.summary.json")
        if completed_summary.is_file():
            _clear_exact_sharded_recovery(resume_dir)
            exact_sharded_recovery = None
            print('[recovery] removed a stale exact sharded marker for an already checkpointed step')
    if exact_sharded_recovery is None:
        exact_sharded_recovery = _infer_exact_sharded_recovery(resume_dir)
        if exact_sharded_recovery is not None:
            marker = _write_exact_sharded_recovery(
                resume_dir,
                exact_sharded_recovery["step"],
                exact_sharded_recovery["reason"],
            )
            exact_sharded_recovery = _read_exact_sharded_recovery(resume_dir)
            print(
                "[recovery] recognized the previous exact single-GPU OOM; "
                f"created {marker.name}; generation and evaluation will not "
                "be repeated")
    if exact_sharded_recovery is not None:
        # This marker is written only after a fully durable SPO-RS rollout
        # batch proves that an exact trajectory cannot fit in ordinary GPU
        # memory. New markers retain the already-computed gradients and first
        # run only the deferred trajectories on RAM-bounded replicas. The
        # serial model-sharded lane is reserved for a trajectory that also
        # exhausts that bounded fallback. Legacy markers first replay only
        # backward on the fast replicas so they can be upgraded likewise.
        selective_overflow = bool(exact_sharded_recovery.get("overflow_gradient_file"))
        replicated_layout = bool(
            merged.get("fast", False)
            and str(merged.get("training_layout", "")).lower()
            == "replicated")
        from training.exact_overflow import supports_warm_overflow
        warm_recovery = bool(
            selective_overflow and replicated_layout
            and supports_warm_overflow(SimpleNamespace(**{
                **merged,
                "num_training_gpus": int(merged.get("num_training_gpus") or
                    len(_parse_gpu_ids(os.environ.get("AVAILABLE_GPUS", "0")))),
            })))
        can_recover_on_replicas = bool(
            selective_overflow
            and replicated_layout
            and not exact_sharded_recovery.get("replica_offload_exhausted", False))
        can_capture_selectively = bool(not selective_overflow and replicated_layout)
        if warm_recovery:
            merged["_exact_inprocess_recovery_step"] = int(exact_sharded_recovery["step"])
            print("[recovery] saved overflow gradients will finish in-process; "
                  "Qwen checkpoint groups try all GPU replicas before "
                  "cooperative overflow when the saved work predates that mode")
        elif can_recover_on_replicas:
            merged["_exact_replica_recovery_step"] = int(exact_sharded_recovery["step"])
            print(
                "[recovery] selective exact overflow will reuse completed "
                "gradients and dispatch only the remaining trajectories "
                "across the fast GPU replicas with bounded host-RAM "
                "activation offload")
        elif can_capture_selectively:
            # A marker written by an older build has no saved partial
            # gradients. Recompute its already-evaluated update once on the
            # fast replicas; the new overflow path will retain all successful
            # gradients and send only the oversized tail to sharded recovery.
            merged["_exact_replica_capture_step"] = int(exact_sharded_recovery["step"])
            failed_length = re.search(r"(\d+)-token rollout", str(exact_sharded_recovery.get("reason", "")))
            if (failed_length is not None and merged.get("coder_model_profile") != "qwen3.8-27b"):
                # Token sidecars are exact for current runs; the tiny margin
                # also covers legacy text recovery restoring an EOS/control
                # token without replaying the already-proven OOM workload.
                merged["_exact_replica_overflow_min_tokens"] = max(1, int(failed_length.group(1)) - 16)
            print(
                "[recovery] legacy exact-OOM step will replay only adapter "
                "backward on the fast replicas; generation and evaluation "
                "remain skipped, and any oversized trajectories will be "
                "captured for selective model-sharded overflow")
        else:
            merged["training_layout"] = "sharded"
            merged["fast"] = False
            merged["_exact_sharded_recovery_step"] = int(exact_sharded_recovery["step"])
        if selective_overflow and not can_recover_on_replicas and not warm_recovery:
            print(
                "[recovery] selective exact overflow will reuse completed "
                "replica gradients and run only deferred trajectories with "
                "one model sharded across all training GPUs; ordinary fast "
                "replicated training resumes after this step")
        elif not selective_overflow and not can_capture_selectively:
            print(
                "[recovery] exact long-context adapter update will use one "
                "model sharded across all training GPUs; ordinary fast "
                "replicated training resumes after this step")

    training_layout = str(merged.get("training_layout") or "auto").strip().lower()
    if training_layout not in {"auto", "replicated", "sharded"}:
        raise ValueError("training_layout must be 'auto', 'replicated', or 'sharded'")
    merged["training_layout"] = training_layout
    if training_layout == "sharded" and merged["fast"]:
        # --fast is process-per-GPU data parallelism: every process needs one
        # complete base model. Large MoE checkpoints instead need one trainer
        # whose frozen base weights are distributed across all selected GPUs.
        print("[config] training_layout=sharded overrides --fast; using one "
              "model-parallel LoRA trainer across all training GPUs")
        merged["fast"] = False
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
    merged["uct"] = bool(merged.get("uct", False))
    pilot_effort_counts = {}
    for effort in ("xhigh", "medium"):
        key = f"pilot_{effort}_programs_per_strategy"
        count = merged.get(key, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        merged[key] = count
        pilot_effort_counts[effort] = count
    strategy_format_max_retries = merged.get("strategy_format_max_retries", 3)
    if (isinstance(strategy_format_max_retries, bool)
            or not isinstance(strategy_format_max_retries, int)
            or strategy_format_max_retries < 0):
        raise ValueError("strategy_format_max_retries must be a non-negative integer")
    merged["strategy_format_max_retries"] = strategy_format_max_retries
    strict_strategy_validation = merged.get("strict_strategy_validation", False)
    if not isinstance(strict_strategy_validation, bool):
        raise ValueError("strict_strategy_validation must be a boolean")
    merged["strict_strategy_validation"] = strict_strategy_validation
    if merged["strategies"]:
        merged.setdefault("strategy_temperature", 1.0)
        merged.setdefault("strategy_top_p", 1.0)
        strategy_model_name = str(merged.get("strategy_model_name") or merged["model_name"]).strip()
        if not merged["api_strategy"]:
            strategy_model_name = resolve_strategy_model(strategy_model_name)
            _apply_strategy_model_profile(merged, strategy_model_name, profile_explicit_keys)
        strategy_backend = "api" if merged["api_strategy"] else str(
            merged.get("strategy_backend") or "local"
        ).strip().lower()
        if strategy_backend not in {"local", "api"}:
            raise ValueError("strategy_backend must be 'local' or 'api'")
        if (strategy_backend == "local" and strategist_profile(strategy_model_name) and generation_backend != "vllm"):
            raise ValueError('Flash-Next / DeepSeek V4 local strategists require generation_backend=vllm')
        strategies_per_parent = int(merged["strategies_per_parent"])
        programs_per_strategy = int(merged["programs_per_strategy"])
        pilot_programs_per_strategy = int(merged.get("pilot_programs_per_strategy", -1))
        phase2_allocation_method = str(merged.get("phase2_allocation_method") or "rule_based").strip().lower()
        strategy_archive_top_r = int(merged["strategy_archive_top_r"])
        native_strategy_context = None
        if merged["api_strategy"]:
            from generation.strategy_api import OPENAI_MAX_OUTPUT_TOKENS
            native_strategy_context = OPENAI_MAX_OUTPUT_TOKENS
        strategy_max_seq_length, strategy_max_new_tokens = resolve_strategy_token_limits(
            strategy_model_name, merged.get("strategy_max_seq_length", "auto"),
            merged.get("strategy_max_new_tokens", "auto"),
            native_context=native_strategy_context)
        if merged["api_strategy"]:
            if strategy_max_new_tokens > OPENAI_MAX_OUTPUT_TOKENS:
                print(f"[config] OpenAI strategy output budget capped at "
                      f"{OPENAI_MAX_OUTPUT_TOKENS} tokens (including reasoning)")
                strategy_max_new_tokens = OPENAI_MAX_OUTPUT_TOKENS
        strategy_temperature = float(merged["strategy_temperature"])
        strategy_top_p = float(merged["strategy_top_p"])
        if (strategies_per_parent < 1 or programs_per_strategy < 1
                or strategy_archive_top_r < 1
                or strategy_max_new_tokens < 1
                or strategy_max_seq_length < 1):
            raise ValueError("strategy counts and token limits must be positive")
        if (pilot_programs_per_strategy == 0 or pilot_programs_per_strategy < -1):
            raise ValueError('pilot_programs_per_strategy must be -1 (disabled) or a positive integer')
        if pilot_programs_per_strategy > programs_per_strategy:
            raise ValueError('pilot_programs_per_strategy cannot exceed programs_per_strategy')
        if (sum(pilot_effort_counts.values()) > 0 and pilot_programs_per_strategy < 1):
            raise ValueError('positive pilot medium/xhigh counts require adaptive pilots to be enabled')
        if (pilot_programs_per_strategy > 0 and sum(pilot_effort_counts.values()) > pilot_programs_per_strategy):
            raise ValueError('pilot medium + xhigh counts cannot exceed pilot_programs_per_strategy')
        if phase2_allocation_method not in {"rule_based", "bandit", "hurdle"}:
            raise ValueError("phase2_allocation_method must be 'rule_based', 'bandit', or 'hurdle'")
        if strategy_temperature <= 0.0:
            raise ValueError("strategy_temperature must be positive")
        if not 0.0 < strategy_top_p <= 1.0:
            raise ValueError("strategy_top_p must be in (0, 1]")
        effective_group_size = (strategies_per_parent * programs_per_strategy)
        if (int(merged.get("group_size", effective_group_size)) != effective_group_size):
            print(f"[config] hierarchical rollout group size derived as "
                  f"{strategies_per_parent} strategies x "
                  f"{programs_per_strategy} programs = "
                  f"{effective_group_size}; replacing configured group_size="
                  f"{merged.get('group_size')}")
        merged["strategy_model_name"] = strategy_model_name
        merged["strategy_backend"] = strategy_backend
        merged["strategies_per_parent"] = strategies_per_parent
        merged["programs_per_strategy"] = programs_per_strategy
        merged["pilot_programs_per_strategy"] = (pilot_programs_per_strategy)
        merged["phase2_allocation_method"] = phase2_allocation_method
        merged["strategy_archive_top_r"] = strategy_archive_top_r
        merged["strategy_max_new_tokens"] = strategy_max_new_tokens
        merged["strategy_max_seq_length"] = strategy_max_seq_length
        merged["strategy_temperature"] = strategy_temperature
        merged["strategy_top_p"] = strategy_top_p
        merged["strategy_thinking"] = bool(merged["strategy_thinking"])
        strategy_reasoning_effort = resolve_strategy_effort(
            strategy_model_name if strategy_backend == "local" else "",
            merged.get("strategy_reasoning_effort") or "high")
        if strategy_reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("strategy_reasoning_effort must be none, low, medium, high, xhigh, or max")
        if (strategy_backend == "local" and "gpt-oss" in strategy_model_name.lower()
                and strategy_reasoning_effort not in {"low", "medium", "high"}):
            raise ValueError("local GPT-OSS strategy reasoning supports low, medium, or high")
        merged["strategy_reasoning_effort"] = strategy_reasoning_effort
        if (strategist_profile(strategy_model_name)
                and "deepseek-v4-flash-0731" in strategy_model_name.lower()
                and strategy_reasoning_effort == "max"
                and strategy_max_seq_length < 393_216):
            print("[config] DeepSeek Think Max recommends at least 393216 "
                  "context tokens; keeping the configured strategy "
                  f"limit {strategy_max_seq_length} unchanged")
        merged["strategy_vllm_quantization"] = str(merged.get('strategy_vllm_quantization') or '').strip()
        merged["strategy_api_base_url"] = str(
            merged.get("strategy_api_base_url")
            or "https://api.deepseek.com").strip().rstrip("/")
        merged["strategy_api_key_env"] = str(merged.get("strategy_api_key_env") or "DEEPSEEK_API_KEY").strip()
        if merged["api_strategy"]:
            merged["strategy_api_base_url"] = "https://api.openai.com/v1"
            merged["strategy_api_key_env"] = "OPENAI_API_KEY"
        merged["strategy_api_concurrency"] = int(merged.get("strategy_api_concurrency") or merged["groups_per_step"])
        merged["strategy_api_timeout_s"] = float(merged.get("strategy_api_timeout_s") or 1800.0)
        merged["strategy_api_max_retries"] = int(
            merged.get("strategy_api_max_retries")
            if merged.get("strategy_api_max_retries") is not None else 5)
        if not merged["strategy_api_base_url"]:
            raise ValueError("strategy_api_base_url cannot be empty")
        if not merged["strategy_api_key_env"]:
            raise ValueError("strategy_api_key_env cannot be empty")
        if merged["strategy_api_concurrency"] < 1:
            raise ValueError("strategy_api_concurrency must be >= 1")
        if merged["strategy_api_timeout_s"] <= 0.0:
            raise ValueError("strategy_api_timeout_s must be positive")
        if merged["strategy_api_max_retries"] < 0:
            raise ValueError("strategy_api_max_retries must be >= 0")
        if strategy_backend == "api":
            from generation.strategy_api import strategy_api_key
            strategy_api_key(merged["strategy_api_key_env"])
        merged["group_size"] = effective_group_size
    # This ablation intentionally keeps two independent Qwen3-8B engines per
    # card resident together: a permanently base/no-LoRA strategist and a
    # coder that receives the current LoRA adapter. Keep the switch derived
    # rather than user-configurable so every other model pairing retains the
    # existing alternating-pool behavior.
    merged["dual_resident_qwen3_8b_strategy_coder_pools"] = (_dual_resident_qwen3_8b_strategy_coder_pools(merged))
    # This profile is derived after coder/strategist selection. Clipped-policy
    # resolvers below remain authoritative when they require full-policy
    # sampling for an exact likelihood ratio.
    _apply_qwen3_8b_thinking_sampling(merged, profile_explicit_keys)
    _resolve_policy_options(merged)
    _resolve_spo_rs_options(merged)
    train_microbatch = int(merged["train_examples_per_microbatch"])
    if train_microbatch < 1:
        raise ValueError("train_examples_per_microbatch must be >= 1")
    merged["train_examples_per_microbatch"] = train_microbatch
    raw_tp = merged["vllm_tensor_parallel_size"]
    tp = (0 if str(raw_tp or '').strip().lower() in ('', 'auto') else int(raw_tp))
    pp = int(merged["vllm_pipeline_parallel_size"] or 1)
    if tp < 0 or pp < 1:
        raise ValueError("vLLM TP must be >= 0 and PP must be >= 1")
    for key in ("vllm_sleep_level", "strategy_vllm_sleep_level"):
        value = int(merged.get(key, 2))
        if value not in (1, 2):
            raise ValueError(f"{key} must be 1 or 2")
        merged[key] = value
    merged["vllm_staged_loading"] = bool(merged.get("vllm_staged_loading", True))
    merged["strategy_vllm_staged_loading"] = bool(merged.get("strategy_vllm_staged_loading", True))
    strategy_persistent_workers = merged.get("strategy_vllm_persistent_workers")
    if strategy_persistent_workers is not None:
        strategy_persistent_workers = int(strategy_persistent_workers)
        if strategy_persistent_workers < 0:
            raise ValueError("strategy_vllm_persistent_workers must be nonnegative")
    merged["strategy_vllm_persistent_workers"] = (strategy_persistent_workers)

    # Resolve every physical role from one ordered inventory. run.sh exports
    # AVAILABLE_GPUS and that environment value is authoritative over old YAML
    # and resumed role fields. Direct Python invocations fall back to the legacy
    # inventory key, CUDA visibility, then one training device.
    from runtime.gpu_runtime import (align_vllm_layout_to_concurrency,
                             allocate_gpu_roles,
                             derive_vllm_parallel_layout,
                             detect_attention_heads, parse_gpu_ids,
                             query_gpu_memory, resolve_memory_settings,
                             validate_attention_heads, validate_selected_gpus,
                             vllm_runtime_reserve_gib)

    inventory_source = "AVAILABLE_GPUS"
    inventory_value = os.environ.get("AVAILABLE_GPUS")
    if inventory_value is None:
        inventory_source = "legacy fallback"
        inventory_value = (args.available_gpu_ids
                           if args.available_gpu_ids is not None
                           else merged.get("available_gpu_ids"))
        if not str(inventory_value or '').strip():
            inventory_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not str(inventory_value or '').strip():
            inventory_value = str(int(merged.get("training_gpu_id", 0)))
        print("[config] warning: AVAILABLE_GPUS is unset; use run.sh so the "
              "ordered physical inventory is authoritative")

    available_gpu_ids = parse_gpu_ids(inventory_value, field=inventory_source)
    roles = allocate_gpu_roles(available_gpu_ids, merged["problem"])
    if merged["isolate_eval"]:
        if roles.gpu_problem:
            raise ValueError(
                "--isolate-eval is for CPU sandbox problems; gpu_mode uses "
                "its dedicated serialized GPU benchmark evaluator")
        if int(merged.get('reward_workers', 0) or 0) < 1:
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
            raise ValueError(f'{flag}={actual} conflicts with {inventory_source}; derived value is {expected}')
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
    merged["training_gpu_ids"] = ",".join(str(x) for x in training_gpu_ids)
    merged["num_training_gpus"] = len(training_gpu_ids)
    merged["available_gpu_ids"] = ",".join(str(x) for x in roles.available)
    merged["gpu_ids"] = ",".join(str(x) for x in gpu_ids)
    merged["num_gpus"] = len(gpu_ids)
    merged["evaluation_gpu_id"] = evaluation_gpu_id
    merged["sequential_generation"] = roles.sequential_generation
    merged["evaluation_shares_generation"] = roles.evaluation_shares_generation
    merged["reserve_last_gpu_for_evaluation"] = bool(roles.gpu_problem and evaluation_gpu_id != training_gpu_id)

    if roles.gpu_problem:
        if not str(merged.get('gpu_type') or '').strip():
            merged["gpu_type"] = "H100"
            print("[config] gpu_mode gpu_type not set; defaulting explicitly to H100")
        merged["kernel_gpu_id"] = evaluation_gpu_id
        if int(merged.get('reward_workers') or 0) != 1:
            print('[config] gpu_mode forces reward_workers=1 for serialized, stable benchmark measurements')
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
    merged["backend"] = _resolve_training_backend(requested_backend, merged["training_model_name"])
    if merged["backend"] != requested_backend:
        print("[backend] GPT-OSS Unsloth BNB checkpoints require the patched "
              f"expert loader; switching {requested_backend} -> "
              f"{merged['backend']}")
    if merged.get("coder_model_profile") == "gpt-oss-120b":
        if (merged["training_model_name"] != merged.get("model_name") or bool(merged.get("load_in_4bit", False))):
            raise ValueError(
                "GPT-OSS-120B rollout and training must use the same original "
                "checkpoint. Remove coder_training_model_name and set "
                "load_in_4bit: false; modern Unsloth keeps the checkpoint's "
                "native MXFP4 experts packed during LoRA training.")
        if merged["backend"] != "unsloth":
            raise ValueError("GPT-OSS-120B aligned training requires backend: unsloth")
        if merged["training_layout"] != "replicated":
            raise ValueError('GPT-OSS-120B packed-MXFP4 training requires training_layout: replicated')
    if merged["training_layout"] == "sharded":
        if merged["backend"] != "hf":
            raise ValueError("training_layout=sharded currently requires backend=hf")
        if int(merged["num_training_gpus"]) < 2:
            raise ValueError("training_layout=sharded requires at least two training GPUs")
        if bool(merged.get("load_in_4bit", False)):
            raise ValueError(
                "training_layout=sharded expects an unquantized trainable "
                "checkpoint; set load_in_4bit: false")
    training_memory_fraction = float(merged.get("training_memory_fraction", 0.80))
    if not 0.0 < training_memory_fraction <= 0.90:
        raise ValueError("training_memory_fraction must be in (0, 0.90]")
    merged["training_memory_fraction"] = training_memory_fraction
    replicated_weight_budget = bool(merged["fast"] or merged["training_layout"] == "replicated")
    training_budgets = _resolve_training_memory_budgets(
        training_gpu_ids, memory,
        max_fraction=(training_memory_fraction if replicated_weight_budget else 0.90))
    _validate_known_training_capacity(
        merged["training_model_name"], training_budgets,
        replicated=replicated_weight_budget)
    merged["training_max_memory_gib"] = training_budgets
    if training_budgets:
        print(f'[memory] training weight budgets by logical GPU: {training_budgets} GiB')

    if merged["fast"] and not merged["no_train"]:
        fast_clipped_policy = bool(_uses_clipped_policy_loss(merged['advantage_mode']))
        if (fast_clipped_policy and int(merged["spo_rs_update_epochs"]) != 1):
            raise ValueError("--fast requires one clipped-policy update epoch")
        fast_backend_supported = bool(
            merged["backend"] == "hf"
            or (merged["backend"] == "unsloth" and merged.get("coder_model_profile") == "gpt-oss-120b")
        )
        if not (int(merged["num_training_gpus"]) > 1
                and fast_backend_supported
                and merged["generation_backend"] == "vllm"):
            raise ValueError(
                "--fast requires replicated HF LoRA training (or the "
                "profiled GPT-OSS-120B Unsloth QLoRA trainer) with vLLM "
                "generation on at least two training GPUs")

    # Consume every rollout GPU. Prefer compatible TP replicas for throughput.
    # If a complete model plus one full-context request would not fit per
    # replica, turn the replica factor into PP so the weights and KV cache are
    # sharded across the exact same GPU inventory.
    if generation_backend == "vllm":
        separate_local_strategy_pool = bool(
            merged["strategies"]
            and merged.get("strategy_backend", "local") == "local"
            and (str(merged.get("strategy_model_name") or merged.get("model_name"))
                 != str(merged.get("model_name"))
                 or merged["dual_resident_qwen3_8b_strategy_coder_pools"]))
        # This must match GenerationPool's two runtime flags.  The old layout
        # used the uncapped 90% allocator budget, then process startup lowered
        # it for scoring/sleep residency; Qwen3-32B consequently selected
        # TP=1 even though its 32K KV cache could not fit after that cap.
        automatic_runtime_reserve = vllm_runtime_reserve_gib(
            co_resident_sleep=separate_local_strategy_pool,
            token_scoring=not bool(merged["no_train"]),
        )
        profile_runtime_reserve = merged.get("vllm_runtime_reserve_override_gib")
        merged["vllm_runtime_reserve_gib"] = (
            automatic_runtime_reserve
            if profile_runtime_reserve is None else
            float(profile_runtime_reserve)
        )
        known_heads = detect_attention_heads(merged.get("model_name", ""))
        layout = derive_vllm_parallel_layout(merged, roles, memory, known_heads)
        if merged["dual_resident_qwen3_8b_strategy_coder_pools"]:
            # The coder needs one independent TP=1 engine per selected card;
            # the strategy topology is derived separately below from its own
            # model and context memory needs. Refuse an unsafe host rather than silently
            # weakening coder rollout parallelism.
            if (layout.tensor_parallel_size != 1 or layout.pipeline_parallel_size != 1):
                raise ValueError(
                    "the dual-resident Qwen3-8B strategy/coder ablation "
                    "requires one coder engine per GPU (TP=1, PP=1), but "
                    "the available memory cannot fit that layout "
                    "at the configured context")
        merged["vllm_tensor_parallel_size"] = layout.tensor_parallel_size
        merged["vllm_pipeline_parallel_size"] = layout.pipeline_parallel_size
        validate_attention_heads(known_heads, merged["vllm_tensor_parallel_size"], merged.get("model_name", ""))
        if merged["dual_resident_qwen3_8b_strategy_coder_pools"]:
            print(f"[config] Qwen3-8B strategist/coder ablation: two "
                  "independent co-resident pools; coder keeps "
                  f"{len(gpu_ids)} TP=1 engines and strategist topology is "
                  "set to maximum memory-safe replica parallelism")
        elif layout.pipeline_parallel_size > 1:
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

        if (merged["strategies"] and merged.get("strategy_backend", "local") == "local"):
            strategy_name = str(merged.get("strategy_model_name") or merged.get("model_name"))
            strategy_layout_cfg = dict(merged)
            strategy_layout_cfg.update({
                "model_name": strategy_name,
                "strategy_inference_profile": bool(strategist_profile(strategy_name)),
                "max_seq_length": int(merged["strategy_max_seq_length"]),
                "load_in_4bit": False,
                "vllm_quantization": merged.get("strategy_vllm_quantization", ""),
                "vllm_runtime_reserve_gib": vllm_runtime_reserve_gib(co_resident_sleep=True, token_scoring=False),
            })
            strategy_heads = detect_attention_heads(strategy_name)
            memory_strategy_layout = derive_vllm_parallel_layout(strategy_layout_cfg, roles, memory, strategy_heads)
            # Strategies inside one parent are sequential, so the number of
            # parents, rather than strategies_per_parent, is the largest live
            # request frontier. Fold replicas above it into TP/PP ranks rather
            # than leaving their GPUs idle for every dependent stage.
            strategy_frontier = int(merged["groups_per_step"])
            strategy_layout = align_vllm_layout_to_concurrency(
                strategy_layout_cfg,
                memory_strategy_layout,
                len(gpu_ids),
                strategy_heads,
                strategy_frontier,
            )
            strategy_layout_cfg["vllm_tensor_parallel_size"] = (strategy_layout.tensor_parallel_size)
            strategy_layout_cfg["vllm_pipeline_parallel_size"] = (strategy_layout.pipeline_parallel_size)
            for note in resolve_memory_settings(strategy_layout_cfg, roles, memory):
                print(f"[memory] auto strategy: {note}")
            merged["strategy_vllm_tensor_parallel_size"] = (strategy_layout.tensor_parallel_size)
            merged["strategy_vllm_pipeline_parallel_size"] = (strategy_layout.pipeline_parallel_size)
            merged["strategy_gen_micro_batch"] = int(strategy_layout_cfg["gen_micro_batch"])
            merged["strategy_vllm_max_num_batched_tokens"] = int(strategy_layout_cfg["vllm_max_num_batched_tokens"])
            merged["strategy_vllm_runtime_reserve_gib"] = float(strategy_layout_cfg["vllm_runtime_reserve_gib"])
            merged["strategy_generation_frontier"] = strategy_frontier
            merged["separate_strategy_inference_pool"] = bool(
                strategy_name != str(merged.get("model_name"))
                or merged["dual_resident_qwen3_8b_strategy_coder_pools"]
                or strategy_layout.tensor_parallel_size
                != layout.tensor_parallel_size
                or strategy_layout.pipeline_parallel_size
                != layout.pipeline_parallel_size
            )
            validate_attention_heads(strategy_heads, strategy_layout.tensor_parallel_size, strategy_name)
            frontier_note = ""
            if strategy_layout != memory_strategy_layout:
                frontier_note = (
                    f"; live frontier={strategy_frontier}, reshaped from "
                    f"{memory_strategy_layout.replicas}xTP"
                    f"{memory_strategy_layout.tensor_parallel_size} so every "
                    "rollout GPU participates")
            print(f"[config] strategy model {strategy_name}: "
                  f"{strategy_layout.replicas} vLLM replica(s), "
                  f"TP={strategy_layout.tensor_parallel_size}, "
                  f"PP={strategy_layout.pipeline_parallel_size}"
                  f"{frontier_note}")
    elif (merged["strategies"]
          and merged.get("strategy_backend", "local") == "local"
          and str(merged.get("strategy_model_name") or merged.get("model_name"))
          != str(merged.get("model_name"))):
        raise ValueError('a distinct strategy_model_name currently requires generation_backend=vllm')

    for note in resolve_memory_settings(merged, roles, memory):
        print(f"[memory] auto: {note}")

    if (merged.get("problem") in {"ac1", "ac2"}
            and merged.get("coder_model_profile") == "gpt-oss-120b"
            and merged.get("generation_backend") == "vllm"):
        # AC responses can exceed 64K tokens; bound vLLM's exact FP32
        # prompt-logprob projection without shortening or approximating them.
        merged["vllm_max_num_batched_tokens"] = min(int(merged["vllm_max_num_batched_tokens"]), 2048)
        print('[memory] AC GPT-OSS exact scoring: vLLM max batched tokens=2048')

    print(f"[config] AVAILABLE_GPUS={roles.available} ({inventory_source})")
    print(f"[config] GPU roles: train={training_gpu_ids}, "
          f"generation={gpu_ids}, evaluation={evaluation_gpu_id}; "
          f"sharing={'sequential' if roles.sequential_generation else 'isolated'}")

    # 4) Provide attribute access after the YAML's shared and problem-specific
    # keys have passed their ownership contract.
    merged["target_modules"] = tuple(merged["target_modules"])
    cfg = SimpleNamespace(**merged)
    if cfg.generation_backend == 'vllm' and int(cfg.num_gpus or 0) < 1:
        raise ValueError("vLLM generation requires num_gpus >= 1")
    if cfg.generation_backend == "vllm":
        resolved_tp = int(cfg.vllm_tensor_parallel_size or 0)
        pp = int(cfg.vllm_pipeline_parallel_size)
        if resolved_tp == 0:
            if int(cfg.num_gpus or 0) % pp:
                raise ValueError("vLLM generation GPU count must be divisible by PP")
            resolved_tp = int(cfg.num_gpus or 0) // pp
        engine_gpus = resolved_tp * pp
        if engine_gpus < 1 or int(cfg.num_gpus or 0) % engine_gpus:
            raise ValueError(
                f"num_gpus={cfg.num_gpus} cannot form complete vLLM groups of "
                f"TP={resolved_tp} * PP={pp}")
        print(f"[config] vLLM engines: {int(cfg.num_gpus or 0) // engine_gpus} "
              f"replica(s), TP={resolved_tp}, PP={pp}, "
              f"{engine_gpus} GPU(s)/engine")
    print(f"[config] rollout thinking: {('enabled' if cfg.thinking else 'disabled')}")
    merged["_resume_dir"] = str(resume_dir) if resume_dir is not None else None
    return cfg, merged


def _isolated_evaluation_cpu_ids():
    """Return the exact Linux cpuset available to this training process."""
    if not (hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")):
        raise RuntimeError("--isolate-eval requires Linux CPU affinity support")
    cpu_ids = sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    if not cpu_ids:
        raise RuntimeError("--isolate-eval found no CPUs in the allowed cpuset")
    return cpu_ids
