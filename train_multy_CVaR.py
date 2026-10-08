"""
TTT-Discover — multi-problem local runner.

Configuration: self-contained problem YAML < resumed config < CLI flags

Rank-selection mode (the proposed local surrogate):
    python train_multy_CVaR.py --problem erdos --advantage-mode rank

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
Existing advantage modes retain their losses.
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
import re
import shutil
import threading
import time
from contextlib import (contextmanager, nullcontext, redirect_stderr,
                        redirect_stdout)
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from entropy_tools import measure_policy_entropy, token_entropy as _token_entropy
from output_retries import (coder_output_issue, coder_retry_prompt_job, final_answer_scope,
                            output_retry_messages, retry_metadata,
                            strategy_retry_needed)
import terminal_output as _terminal_output


# Logging is diagnostic and must never make an otherwise valid trainer update
# fail.  Accept the pre-setting.log helper module during a rolling/mixed
# deployment; a complete deployment still uses the direct setting.log sink.
setting_log_only = getattr(
    _terminal_output, "setting_log_only", _terminal_output.terminal_log_only)
_native_bind_setting_log = getattr(
    _terminal_output, "bind_setting_log", None)


def bind_setting_log(path, *, time_offset=0):
    if callable(_native_bind_setting_log):
        return _native_bind_setting_log(path, time_offset=time_offset)
    return None


# Downstream modules in an older deployment may still import these names
# directly after importing this module. Publish the compatibility aliases so
# those imports remain non-fatal as well.
if not hasattr(_terminal_output, "setting_log_only"):
    _terminal_output.setting_log_only = setting_log_only
if not hasattr(_terminal_output, "bind_setting_log"):
    _terminal_output.bind_setting_log = bind_setting_log


_STRATEGY_FALLBACK = (
    "No usable strategy was returned. Derive the implementation directly "
    "from the task specification and current parent construction."
)
_STRATEGY_BLOCK_RE = re.compile(
    # Do not let an earlier malformed opening tag consume the model's actual
    # final block. With ``<strategy>...<strategy>final</strategy>``, matching
    # starts at the inner opening tag and only ``final`` reaches the coder.
    r"<strategy\b[^>]*>\s*((?:(?!<strategy\b).)*?)\s*</strategy\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_STRATEGY_FINAL_MARKERS = (
    "<|channel|>final<|message|>",
    "assistantfinal",
)
_LOG_TIME_OFFSET_SECONDS = -4 * 60 * 60
_QWEN3_8B_MODEL_ID = "qwen/qwen3-8b"
_QWEN3_8B_MODEL_BASENAME = "qwen3-8b"
_QWEN3_30B_A3B_MODEL_BASENAME = "qwen3-30b-a3b"
_GPT_OSS_120B_MODEL_BASENAME = "gpt-oss-120b"
_QWEN3_30B_A3B_THINKING_MODEL_BASENAME = (
    "qwen3-30b-a3b-thinking-2507")
_QWEN38_27B_MODEL_BASENAME = "qwen3.8-27b"
_DEEPSEEK_R1_0528_QWEN3_8B_MODEL_BASENAME = (
    "deepseek-r1-0528-qwen3-8b")
_FUSED_LONG_SINGLETON_MIN_TOKENS = 8192
_ANSI_ORANGE = "\033[38;5;208m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"


_GPT_OSS_CODER_DEVELOPER_INSTRUCTIONS = '''Formatting re-enabled

You are the implementation-only Python coder in a two-stage scientific-
discovery pipeline. The user message supplies the authoritative task and may
also contain a previous program, empirical lessons, and a planner-produced
<strategy> block.

Follow this precedence:
1. Obey the task's evaluator, mathematical constraints, allowed resources,
   runtime budget, required function signature, and return contract.
2. Use the <strategy> as fallible implementation guidance. Preserve useful
   ideas, but independently correct any mathematical, algorithmic, interface,
   or feasibility error instead of blindly translating it.
3. Treat previous programs and empirical lessons as evidence, not authority.

Use the model's analysis channel for private reasoning and verification. The
final channel is only the code artifact: emit exactly one complete fenced
Python block beginning with ```python and ending with ```, with no prose,
strategy, analysis, notes, example usage, or second code block before or after
it. The program must be syntactically complete, define the requested top-level
entry point, obey every source-size/data constraint, and close the fence well
before the response limit. Before producing the final channel, check that the
code's dimensions, optimization direction, feasibility handling, time-limit
path, and return values agree with the task as written.'''


def _coder_messages_for_template(messages, template_kind):
    """Apply model-family prompt semantics without touching other coders.

    Transformers' GPT-OSS chat template maps an input ``system`` message to
    Harmony's developer-instruction role; Harmony itself supplies the actual
    system header containing reasoning effort and channel declarations. This
    keeps durable behavior/output rules above the task while leaving the full
    task, parent state, and strategy in the user message.
    """
    if str(template_kind) != "gpt-oss":
        return messages

    staged = [dict(message) for message in messages]
    if staged and staged[0].get("role") == "system":
        existing = str(staged[0].get("content", "")).strip()
        staged[0]["content"] = _GPT_OSS_CODER_DEVELOPER_INSTRUCTIONS
        if existing:
            staged[0]["content"] += (
                "\n\nAdditional task-specific instructions:\n" + existing)
    else:
        staged.insert(0, {
            "role": "system",
            "content": _GPT_OSS_CODER_DEVELOPER_INSTRUCTIONS,
        })
    return staged


def _is_exact_qwen3_8b(model_name):
    return (str(model_name or "").strip().rstrip("/").lower()
            == _QWEN3_8B_MODEL_ID)


def _is_qwen3_8b_sampling_model(model_name):
    """Recognize the canonical checkpoint or an exactly named local copy."""
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower() == "qwen3-8b"


def _profile_default(merged, key, value, explicit_keys):
    """Apply an automatic model default unless configuration set the key."""
    if key not in explicit_keys:
        merged[key] = value


def _apply_qwen3_8b_thinking_sampling(merged, explicit_keys=frozenset()):
    """Resolve unfiltered sampling while leaving coder temperature/p in YAML."""
    # Project-wide invariant: never truncate the candidate distribution by a
    # fixed token count. vLLM translates 0 to its native disabled value (-1),
    # while Transformers accepts 0 directly.
    merged["sampling_top_k"] = 0
    if "min_p" in merged:
        merged["sampling_min_p"] = float(merged["min_p"])
    else:
        merged.setdefault("sampling_min_p", None)
    merged["strategy_sampling_top_k"] = 0
    if "strategy_min_p" in merged:
        merged["strategy_sampling_min_p"] = float(
            merged["strategy_min_p"])
    else:
        merged.setdefault("strategy_sampling_min_p", None)
    merged["qwen3_8b_thinking_sampling"] = False
    merged["strategy_qwen3_8b_thinking_sampling"] = False

    if (_is_qwen3_8b_sampling_model(merged.get("model_name"))
            and bool(merged.get("thinking", False))):
        # Coder temperature and nucleus sampling are experiment controls and
        # therefore come straight from YAML/config.  Qwen3-8B's automatic
        # runtime contract only enforces the project's deliberately
        # unfiltered policy: no top-k and no minimum-probability cutoff.
        merged["sampling_min_p"] = 0.0
        merged["qwen3_8b_thinking_sampling"] = bool(
            float(merged["temperature"]) == 0.6
            and float(merged["top_p"]) == 0.95
            and merged.get("sampling_top_k") == 0
            and merged.get("sampling_min_p") == 0.0)
        print(
            "[model-profile] Qwen3-8B coder thinking sampling resolved: "
            f"temperature={merged['temperature']}, top_p={merged['top_p']}, "
            f"top_k={merged['sampling_top_k']}, "
            f"min_p={merged['sampling_min_p']} "
            "(temperature/top_p from YAML/config; filters disabled)")

    if (bool(merged.get("strategies", False))
            and _is_qwen3_8b_sampling_model(
                merged.get("strategy_model_name"))
            and bool(merged.get("strategy_thinking", False))):
        _profile_default(
            merged, "strategy_temperature", 0.6, explicit_keys)
        _profile_default(merged, "strategy_top_p", 0.95, explicit_keys)
        if ("strategy_min_p" not in explicit_keys
                and "strategy_sampling_min_p" not in explicit_keys):
            merged["strategy_sampling_min_p"] = 0.0
        merged["strategy_qwen3_8b_thinking_sampling"] = bool(
            float(merged["strategy_temperature"]) == 0.6
            and float(merged["strategy_top_p"]) == 0.95
            and merged.get("strategy_sampling_top_k") == 0
            and merged.get("strategy_sampling_min_p") == 0.0)
        print(
            "[model-profile] Qwen3-8B strategist thinking sampling resolved: "
            f"temperature={merged['strategy_temperature']}, "
            f"top_p={merged['strategy_top_p']}, "
            f"top_k={merged['strategy_sampling_top_k']}, "
            f"min_p={merged['strategy_sampling_min_p']} "
            "(explicit YAML/config values take precedence)")


def _is_exact_qwen3_30b_a3b(model_name):
    """Match only the original hybrid-thinking A3B coder checkpoint.

    Accept the canonical Hub id and an explicitly named local model directory,
    while excluding Instruct-2507, Thinking-2507, Base, and quantized variants.
    """
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower() == (
        _QWEN3_30B_A3B_MODEL_BASENAME)


def _model_basename(model_name):
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower()


def _apply_strategy_model_profile(
        merged, model_name, explicit_keys=frozenset()):
    """Apply model-native strategist defaults below explicit configuration."""
    if (_model_basename(model_name)
            != _DEEPSEEK_R1_0528_QWEN3_8B_MODEL_BASENAME):
        return

    # DeepSeek evaluates this checkpoint with 64K generations, but its native
    # context is 128K. Expose that full output ceiling here; generation already
    # subtracts the actual prompt length, so prompt + completion never exceeds
    # 131072. It is a native reasoning model: thinking stays enabled and
    # `high` is the strongest effort label exposed by this runner.
    _profile_default(
        merged, "strategy_max_new_tokens", 131_072, explicit_keys)
    _profile_default(
        merged, "strategy_max_seq_length", 131_072, explicit_keys)
    _profile_default(merged, "strategy_temperature", 0.6, explicit_keys)
    _profile_default(merged, "strategy_top_p", 0.95, explicit_keys)
    _profile_default(merged, "strategy_thinking", True, explicit_keys)
    _profile_default(
        merged, "strategy_reasoning_effort", "high", explicit_keys)
    _profile_default(
        merged, "strategy_vllm_quantization", "", explicit_keys)

    # At 8B, the exact model fits without quantization. The topology planner
    # keeps independent TP=1 replicas while the live parent-chain frontier can
    # feed them, and folds otherwise-idle replicas into TP ranks when it cannot.
    # Retain every selected engine's level-1 backup between phases.
    _profile_default(merged, "strategy_vllm_sleep_level", 1, explicit_keys)
    _profile_default(
        merged, "strategy_vllm_staged_loading", False, explicit_keys)
    _profile_default(
        merged, "strategy_vllm_persistent_workers", None, explicit_keys)
    print("[model-profile] DeepSeek-R1-0528-Qwen3-8B strategist: "
          f"native reasoning, max_new={merged['strategy_max_new_tokens']}, "
          f"max_seq={merged['strategy_max_seq_length']}, "
          f"temperature={merged['strategy_temperature']}, "
          f"top_p={merged['strategy_top_p']}; explicit YAML/config values "
          "take precedence")


def _coder_model_profile(model_name):
    """Return the automatic runtime contract for explicitly supported coders."""
    basename = _model_basename(model_name)
    if basename == _QWEN3_8B_MODEL_BASENAME:
        return {
            "name": "qwen3-8b",
            # The checkpoint's exact native rotary-position ceiling.  Longer
            # 131K operation requires YaRN and is intentionally not enabled
            # implicitly because it changes positional scaling/model behavior.
            "native_context": 40_960,
            # Generation code subtracts the actual prompt length, so this
            # exposes every native token still available to the completion.
            "max_output": 40_960,
            "template_kind": "qwen3",
            "training_4bit": False,
            "training_layout": "replicated",
            "training_memory_fraction": 0.88,
            "vllm_runtime_reserve_gib": 5.0,
            "force_thinking": True,
        }
    if basename == _GPT_OSS_120B_MODEL_BASENAME:
        return {
            "name": "gpt-oss-120b",
            "native_context": 131_072,
            "max_output": 131_072,
            "reasoning_effort": "Medium",
            "template_kind": "gpt-oss",
            "training_4bit": True,
            # The trainable Unsloth BitsAndBytes copy fits on one 95+ GiB
            # card. Keep one complete QLoRA trainer per GPU; the native MXFP4
            # checkpoint remains exclusive to vLLM rollout generation.
            "training_layout": "replicated",
            "training_memory_fraction": 0.88,
            "vllm_runtime_reserve_gib": 5.0,
        }
    if basename == _QWEN3_30B_A3B_THINKING_MODEL_BASENAME:
        return {
            "name": "qwen3-30b-a3b-thinking-2507",
            "native_context": 262_144,
            "max_output": 262_144,
            "reasoning_effort": "thinking-only",
            "template_kind": "qwen-thinking",
            "training_4bit": False,
            "training_layout": "sharded",
            "vllm_runtime_reserve_gib": 5.0,
        }
    if basename == _QWEN38_27B_MODEL_BASENAME:
        return {
            "name": "qwen3.8-27b",
            "native_context": 262_144,
            "max_output": 262_144,
            "reasoning_effort": "medium", #xhigh
            "template_kind": "qwen3.8",
            "training_4bit": False,
            # A BF16 copy plus the LoRA state fits on each 95+ GiB card.  Use
            # one complete trainer per GPU so independent rollouts execute in
            # parallel; layer-sharding a single copy serializes the forward
            # and backward passes across cards.
            "training_layout": "replicated",
            "training_memory_fraction": 0.88,
            # Qwen3.8 has a roughly 248K-token vocabulary. vLLM's exact
            # prompt-logprob path materializes an FP32 softmax for an entire
            # prefill chunk, so its transient projection is much larger than
            # the ordinary generation activation. Keep enough actual card
            # headroom instead of assigning that memory to the KV cache.
            "vllm_runtime_reserve_gib": 15.0,
        }
    return None


def _resolve_coder_token_limits(merged):
    """Fill missing coder limits without changing explicit limits or model behavior.

    Known coder profiles run first. Other checkpoints supply their context
    through config.json; reading that small file does not load any weights or
    initialize CUDA. No RoPE scaling or context extension is enabled here.
    """
    keys = ("max_seq_length", "max_new_tokens")
    if all(key in merged for key in keys):
        return
    if "max_seq_length" not in merged:
        model_name = str(merged["model_name"])
        try:
            model_path = Path(model_name).expanduser()
            if model_path.is_dir():
                config_path = model_path / "config.json"
            else:
                from huggingface_hub import hf_hub_download
                config_path = Path(hf_hub_download(model_name, "config.json"))
            metadata = json.loads(config_path.read_text())
            context = None
            for candidate in (metadata.get("text_config"), metadata):
                if not isinstance(candidate, dict):
                    continue
                for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
                    value = candidate.get(key)
                    if (isinstance(value, int) and not isinstance(value, bool)
                            and value > 0):
                        context = value
                        break
                if context is not None:
                    break
            if context is None:
                raise ValueError("checkpoint config has no positive context length")
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve the coder context limit for {model_name!r}; "
                "make its config.json available or explicitly set max_seq_length. "
                f"Details: {exc}") from exc
        merged["max_seq_length"] = context
    # With no separate output cap, allow all context remaining after the prompt.
    merged.setdefault("max_new_tokens", int(merged["max_seq_length"]))
    print(f"[config] coder token limits: max_seq={merged['max_seq_length']}, "
          f"max_new={merged['max_new_tokens']}; output is capped by the "
          "remaining context after the prompt")


def _coder_effort_for_rollout_phase(cfg, phase=None):
    """Apply Qwen3.8's mixed pilot efforts without changing other coders."""
    if getattr(cfg, "coder_template_kind", "generic") == "qwen3.8":
        if phase == "pilot_xhigh":
            return "xhigh"
        if phase in {"pilot", "adaptive"}:
            return "medium"
    return getattr(cfg, "coder_reasoning_effort", None)


def _coder_phase_generation_jobs(source_job_indices, counts, phase, cfg):
    """Split Qwen3.8 pilots into one xhigh and N-1 medium samples per source.

    Both prompt variants enter the same scheduler call, retaining the original
    parent/strategy/fold allocation ID. The internal pilot_xhigh phase controls
    rendering only; all these records remain pilots for evaluation/allocation.
    """
    high_effort = []
    active = []
    for source_idx, count in zip(source_job_indices, counts):
        count = int(count)
        if count <= 0:
            continue
        if (phase == "pilot"
                and getattr(cfg, "coder_template_kind", "generic") == "qwen3.8"):
            # Put all xhigh prompt groups first in the SAME scheduler call.
            # Consecutive IDs let the existing round-robin dispatcher spread
            # them across every worker instead of only the even-numbered ones.
            high_effort.append((int(source_idx), 1, "pilot_xhigh"))
            if count > 1:
                active.append((int(source_idx), count - 1, "pilot"))
        else:
            active.append((int(source_idx), count, phase))
    return high_effort + active


def _phase_coder_prompt_job(prompt_jobs, source_idx, phase, cfg, render, cache):
    """Return an immutable phase-specific prompt ID without changing pilots.

    Records must retain the exact prompt used at generation for old/reference
    logprobs, training, and artifact saving. Appending a prompt variant instead
    of replacing the source job preserves pilot histories and tensor caches.
    Allocation still uses the original source ID, so variants add no budget.
    """
    effort = _coder_effort_for_rollout_phase(cfg, phase)
    # The xhigh pilot needs an alias even when xhigh is the configured base:
    # a retry is tagged "pilot" for allocation, so its render phase must be
    # retained explicitly to avoid silently retrying it at medium effort.
    if (effort == _coder_effort_for_rollout_phase(cfg)
            and phase != "pilot_xhigh"):
        return int(source_idx)
    key = (int(source_idx), effort)
    if key not in cache:
        source = prompt_jobs[int(source_idx)]
        variant = {
            **source,
            "prompt_text": render(source["messages"], rollout_phase=phase),
            "coder_reasoning_effort": effort,
            "coder_prompt_phase": phase,
            # This is a prompt alias, not a new allocation. Phase generation
            # supplies the actual counts explicitly to the existing scheduler.
            "count": 0,
        }
        cache[key] = len(prompt_jobs)
        prompt_jobs.append(variant)
    return cache[key]


def _apply_coder_model_profile(merged, explicit_keys=frozenset()):
    """Wire reasoning, context, rollout, and exact-training behavior by model.

    These are safe defaults derived from ``coder_model_name`` (which is copied
    to ``model_name`` by ``--strategies``). Any operator-controlled YAML,
    resumed-config, or explicit CLI value remains authoritative.
    """
    profile = _coder_model_profile(merged.get("model_name"))
    if profile is None:
        merged.pop("coder_model_profile", None)
        return

    native_context = int(profile["native_context"])
    merged["coder_model_profile"] = str(profile["name"])
    merged["coder_template_kind"] = str(profile["template_kind"])
    if "reasoning_effort" in profile:
        _profile_default(
            merged, "coder_reasoning_effort",
            str(profile["reasoning_effort"]), explicit_keys)
    else:
        # Qwen3-8B exposes only the real enable_thinking switch; it has no
        # native low/medium/high reasoning-effort control.
        merged.pop("coder_reasoning_effort", None)
    merged["coder_preserve_thinking"] = bool(
        profile["template_kind"] == "qwen3.8")
    _profile_default(merged, "max_seq_length", native_context, explicit_keys)
    _profile_default(
        merged, "max_new_tokens", int(profile["max_output"]), explicit_keys)
    if bool(profile.get("force_thinking", False)):
        # This project uses Qwen3-8B only in its thinking mode.  Unlike
        # temperature/top-p, this is part of the selected model contract.
        merged["thinking"] = True
    else:
        _profile_default(merged, "thinking", True, explicit_keys)

    # Native GPT-OSS MXFP4 remains the vLLM checkpoint. Its trainable copy is
    # selected independently by _resolve_training_model_name below.
    _profile_default(
        merged, "load_in_4bit", bool(profile["training_4bit"]), explicit_keys)
    _profile_default(merged, "training_model_name", "", explicit_keys)
    _profile_default(
        merged, "training_layout", str(profile["training_layout"]),
        explicit_keys)
    _profile_default(
        merged, "training_memory_fraction",
        float(profile.get("training_memory_fraction", 0.80)), explicit_keys)

    # On the 95+ GiB cards used by this project, the exact checkpoint/KV model
    # admits one full-context engine per card. The model profile leaves the
    # measured amount of memory outside vLLM's KV cache for CUDA/NCCL and the
    # bounded exact-token projection. Qwen3.8 needs more than the other
    # profiles because its vocabulary projection is materially larger.
    _profile_default(
        merged, "vllm_gpu_memory_utilization", 0.965, explicit_keys)
    _profile_default(merged, "vllm_quantization", "", explicit_keys)
    _profile_default(merged, "gen_micro_batch", "auto", explicit_keys)
    _profile_default(
        merged, "vllm_max_num_batched_tokens", "auto", explicit_keys)
    merged["vllm_runtime_reserve_override_gib"] = float(
        profile["vllm_runtime_reserve_gib"])
    _profile_default(merged, "vllm_staged_loading", True, explicit_keys)
    # Final TP=1 admission is decided after nvidia-smi identifies the cards.
    # Blackwell can consume GPT-OSS's native MXFP4 layout directly; older
    # architectures may need a large transient Marlin repack and must retain
    # the conservative TP>=2 startup guard in gpu_runtime.py.
    merged["allow_gpt_oss_tp1"] = False

    # Exact long-context training policy. This does not shorten, approximate,
    # or discard a trajectory. It selects exact memory-saving execution modes
    # before known-failing ordinary attempts and raises if all exact modes are
    # exhausted.
    merged["strict_exact_long_training"] = True
    _profile_default(merged, "fused_long_attention", True, explicit_keys)
    merged["long_training_checkpoint_min_tokens"] = 16_384
    merged["long_training_headroom_min_tokens"] = 32_768
    # CPU activation traffic is much slower than recomputation. Use it only
    # after an actual OOM proves it necessary; the persistent OOM profile then
    # starts comparable later trajectories in that exact fallback directly.
    merged["long_training_cpu_offload_min_tokens"] = None

    precedence_note = (
        "temperature/top_p remain YAML-controlled; thinking on and "
        "top-k/min-p off are enforced"
        if profile["name"] == "qwen3-8b"
        else "explicit YAML/config values take precedence"
    )
    reasoning_label = merged.get(
        "coder_reasoning_effort", "thinking on/off only")
    print(
        f"[model-profile] {profile['name']}: context limit="
        f"{merged['max_seq_length']}, max generated tokens="
        f"{merged['max_new_tokens']}, reasoning="
        f"{reasoning_label}, rollout target=one "
        "TP=1 engine/GPU; training layout="
        f"{merged['training_layout']}; {precedence_note}; exact "
        "long-context training enabled",
        flush=True,
    )


def _uses_sequence_level_policy_ratio(cfg):
    """Use the MoE-stable sequence ratio only for plain Qwen3-30B-A3B."""
    return _is_exact_qwen3_30b_a3b(getattr(cfg, "model_name", ""))


def _detached_behavior_importance_ratio(cfg, current_logprobs,
                                        behavior_logprobs):
    """Return the existing token ratios or the A3B trajectory ratio.

    Standard entropic/GRPO/CVaR updates use a detached importance weight rather
    than the differentiable clipped surrogate. Keep that objective unchanged
    except that plain Qwen3-30B-A3B receives one geometric-mean response weight
    instead of volatile individual-token MoE weights.
    """
    import torch

    current = current_logprobs.detach()
    behavior = torch.as_tensor(
        behavior_logprobs, dtype=current.dtype, device=current.device)
    log_ratios = current - behavior
    if _uses_sequence_level_policy_ratio(cfg):
        return log_ratios.mean().exp()
    return log_ratios.exp()


def _dual_resident_qwen3_8b_strategy_coder_pools(config):
    """Select independent co-resident strategist and coder Qwen3-8B pools."""
    return bool(
        config.get("strategies", False)
        and str(config.get("strategy_backend") or "local").lower() == "local"
        and str(config.get("generation_backend") or "hf").lower() == "vllm"
        and _is_exact_qwen3_8b(config.get("coder_model_name"))
        and _is_exact_qwen3_8b(config.get("strategy_model_name"))
        and _is_exact_qwen3_8b(config.get("model_name"))
    )


def _extract_final_strategy(response_text):
    """Extract only the final, complete strategy block.

    GPT-OSS may expose a long analysis channel before its final answer. None of
    that text is allowed into a later strategist or coder prompt. We retain
    only the body of the last complete ``<strategy>...</strategy>`` block at
    the end of the response. An explicit final channel is preferred when the
    template exposes one, but is not required: preceding reasoning is harmless
    because it is never returned. A missing, empty, or unclosed final block is
    retried by the caller; exhausting the configured limit stops the run.
    The fallback string is only an extraction-error sentinel, never handed off.
    """
    raw = final_answer_scope(response_text).strip()
    if not raw:
        return _STRATEGY_FALLBACK, "empty response"

    lowered = raw.lower()
    final_position = -1
    final_marker = ""
    for marker in _STRATEGY_FINAL_MARKERS:
        position = lowered.rfind(marker.lower())
        if position > final_position:
            final_position = position
            final_marker = marker
    if final_position >= 0:
        final_text = raw[final_position + len(final_marker):].strip()
    else:
        # Non-Harmony templates can include reasoning without labeling their
        # channels. Only the final block below is retained, so the prefix is
        # never exposed to a later strategist or to the coder.
        final_text = raw

    matches = list(_STRATEGY_BLOCK_RE.finditer(final_text))
    if not matches:
        return _STRATEGY_FALLBACK, "missing complete final strategy"

    match = matches[-1]
    trailing = final_text[match.end():]
    if re.search(r"<strategy\b", trailing, flags=re.IGNORECASE):
        return _STRATEGY_FALLBACK, "unclosed final strategy block"

    body = match.group(1).strip()
    if not body:
        return _STRATEGY_FALLBACK, "empty final strategy"
    return body, None


class _TimestampedLineStream:
    def __init__(self, stream):
        self.stream = stream
        self._line_start = True
        self._dynamic_progress = False
        self._lock = threading.Lock()

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        progress_stream = getattr(self.stream, "progress_stream", None)
        route_progress = bool(
            progress_stream is not None
            and ("\r" in value
                 or (self._dynamic_progress
                     and "\n" in value
                     and not value.strip())))
        if route_progress:
            with self._lock:
                progress_stream.write(value)
                if "\r" in value and "\n" not in value:
                    self._dynamic_progress = True
                if "\n" in value:
                    self._dynamic_progress = False
                    self._line_start = True
            return len(value)
        parts = value.split("\n")
        with self._lock:
            for index, part in enumerate(parts):
                if part and self._line_start:
                    timestamp = time.strftime(
                        "[%H:%M:%S] ",
                        time.localtime(
                            time.time() + _LOG_TIME_OFFSET_SECONDS),
                    )
                    self.stream.write(timestamp)
                if part:
                    self.stream.write(part)
                    self._line_start = False
                if index + 1 < len(parts):
                    self.stream.write("\n")
                    self._line_start = True
        return len(value)

    def flush(self):
        self.stream.flush()

    def write_log_only(self, value):
        """Timestamp complete diagnostic lines without touching the TTY."""
        value = str(value)
        if not value:
            return 0
        writer = getattr(self.stream, "write_log_only", None)
        if writer is None:
            return self.write(value)
        parts = value.split("\n")
        rendered = []
        for index, part in enumerate(parts):
            if part:
                timestamp = time.strftime(
                    "[%H:%M:%S] ",
                    time.localtime(time.time() + _LOG_TIME_OFFSET_SECONDS),
                )
                rendered.extend((timestamp, part))
            if index + 1 < len(parts):
                rendered.append("\n")
        with self._lock:
            writer("".join(rendered))
        return len(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _install_console_timestamps():
    if not isinstance(sys.stdout, _TimestampedLineStream):
        sys.stdout = _TimestampedLineStream(sys.stdout)
    if not isinstance(sys.stderr, _TimestampedLineStream):
        sys.stderr = _TimestampedLineStream(sys.stderr)


class _TerminalTeeStream:
    """Mirror one console stream to the run log without changing its TTY."""

    def __init__(self, console, sink):
        self.console = console
        self.sink = sink
        self._progress_stream = _TerminalProgressTeeStream(console, sink)

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.console.write(value)
            self.sink.write(value)
        return len(value)

    def flush(self):
        with self.sink.lock:
            self.console.flush()
            self.sink.flush()

    def write_log_only(self, value):
        """Append to temirnal.log while deliberately bypassing the console."""
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.sink.write(value)
        return len(value)

    @property
    def progress_stream(self):
        """TTY-preserving stream whose saved progress row updates in place."""
        return self._progress_stream

    def __getattr__(self, name):
        # Preserve fileno(), isatty(), encoding, etc. from the real terminal so
        # tqdm and libraries keep exactly their existing console behaviour.
        return getattr(self.console, name)


class _TerminalProgressTeeStream:
    """Show raw dynamic progress on the TTY and one live row in the log."""

    def __init__(self, console, sink):
        self.console = console
        self.sink = sink

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.console.write(value)
            self.sink.write_progress(value)
        return len(value)

    def flush(self):
        with self.sink.lock:
            self.console.flush()
            self.sink.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class _TerminalLogSink:
    """Append ordinary output while rewriting one dynamic progress row."""

    def __init__(self):
        self.lock = threading.RLock()
        self.log_file = None
        self.pending = []
        self._progress_offset = None
        self._progress_text = None

    @staticmethod
    def _clean_progress_text(value):
        # tqdm uses carriage returns for in-place updates and may include ANSI
        # cursor/colour controls. Keep the visible text, not terminal control
        # bytes, in the persistent plain-text log.
        value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
        candidates = []
        for line in value.replace("\n", "\r\n").split("\r"):
            cleaned = line.rstrip()
            if cleaned.strip():
                candidates.append(cleaned)
        return candidates[-1] if candidates else None

    def _remove_visible_progress(self):
        if self.log_file is None or self._progress_offset is None:
            return
        self.log_file.seek(self._progress_offset)
        self.log_file.truncate()
        self.log_file.seek(0, os.SEEK_END)
        self._progress_offset = None

    def _show_progress(self):
        if self.log_file is None or not self._progress_text:
            return
        self._remove_visible_progress()
        self.log_file.seek(0, os.SEEK_END)
        self._progress_offset = self.log_file.tell()
        timestamp = time.strftime(
            "[%H:%M:%S] ",
            time.localtime(time.time() + _LOG_TIME_OFFSET_SECONDS),
        )
        self.log_file.write(timestamp + self._progress_text)
        self.log_file.truncate()
        self.log_file.seek(0, os.SEEK_END)

    def _finish_progress(self):
        if self.log_file is None or self._progress_text is None:
            return
        if self._progress_offset is None:
            self._show_progress()
        self.log_file.seek(0, os.SEEK_END)
        self.log_file.write("\n")
        self._progress_offset = None
        self._progress_text = None

    def write(self, value):
        value = str(value)
        if self.log_file is None:
            self.pending.append(("ordinary", value))
            return
        # A regular status line takes precedence over a transient bar. Remove
        # the visible bar, append the status, and retain its latest text so a
        # later refresh/close can redraw or finalize it after the status line.
        self._remove_visible_progress()
        self.log_file.seek(0, os.SEEK_END)
        self.log_file.write(value)
        self.log_file.flush()

    def write_progress(self, value):
        if self.log_file is None:
            self.pending.append(("progress", str(value)))
            return
        cleaned = self._clean_progress_text(value)
        if cleaned is not None:
            self._progress_text = cleaned
            self._show_progress()
        if "\n" in str(value):
            self._finish_progress()
        self.log_file.flush()

    def flush(self):
        if self.log_file is not None:
            self.log_file.flush()

    def bind(self, log_path):
        with self.lock:
            if self.log_file is not None:
                return
            mode = "r+" if os.path.exists(log_path) else "w+"
            self.log_file = open(
                log_path, mode, buffering=1, encoding="utf-8")
            self.log_file.seek(0, os.SEEK_END)
            if self.pending:
                pending = list(self.pending)
                self.pending.clear()
                for kind, value in pending:
                    if kind == "progress":
                        self.write_progress(value)
                    else:
                        self.write(value)
            self.log_file.flush()


def _install_terminal_log():
    """Start capturing timestamped stdout/stderr before the run dir exists."""
    sink = _TerminalLogSink()

    def attach(stream):
        # Ordinary-line timestamps are added before the tee. Dynamic progress
        # bypasses that wrapper and receives its saved timestamp from the sink
        # each time the single persistent progress row is replaced.
        if isinstance(stream, _TimestampedLineStream):
            stream.stream = _TerminalTeeStream(stream.stream, sink)
            return stream
        return _TerminalTeeStream(stream, sink)

    sys.stdout = attach(sys.stdout)
    sys.stderr = attach(sys.stderr)
    return sink


_ROUTED_DEPENDENCY_NOTICES = (
    "Skipping import of cpp extensions due to incompatible torch version.",
    "No prebuilt binary for CUDA",
    "You are sending unauthenticated requests to the HF Hub.",
    "is an Enum subclass and is now natively supported by torch.compile",
    "`torch_dtype` is deprecated! Use `dtype` instead!",
)

_SETTING_LOG_ONLY_NOTICES = (
    "Unrecognized keys in `rope_parameters`",
)


class _NoticeRoutingStream:
    """Keep ordinary model-loading output visible and route known noise."""

    def __init__(self, visible, diagnostic, label):
        self.visible = visible
        self.diagnostic = diagnostic
        self.label = label
        self._routing_line = False
        self._setting_log_line = False

    def _diagnostic_is_open(self):
        return (self.diagnostic is not None
                and not bool(getattr(self.diagnostic, "closed", False)))

    def write(self, value):
        for piece in str(value).splitlines(keepends=True):
            setting_only = (
                self._setting_log_line
                or any(marker in piece
                       for marker in _SETTING_LOG_ONLY_NOTICES)
            )
            route = (self._diagnostic_is_open()
                     and not setting_only
                     and (self._routing_line
                          or any(marker in piece
                                 for marker in _ROUTED_DEPENDENCY_NOTICES)))
            if setting_only:
                setting_log_only(piece, end="")
            elif route:
                if not self._routing_line:
                    self.diagnostic.write(f"[{self.label}] ")
                self.diagnostic.write(piece)
            else:
                self.visible.write(piece)
            self._setting_log_line = bool(
                setting_only and not piece.endswith(("\n", "\r")))
            self._routing_line = bool(
                route and not piece.endswith(("\n", "\r")))
        return len(value)

    def flush(self):
        self.visible.flush()
        if self._diagnostic_is_open():
            self.diagnostic.flush()

    def write_log_only(self, value):
        writer = getattr(self.visible, "write_log_only", None)
        if writer is not None:
            return writer(value)
        return self.diagnostic.write(value)

    def detach_diagnostic(self):
        """Make a handler-retained wrapper safe after its file is closed."""
        self._routing_line = False
        self._setting_log_line = False
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
ADVANTAGE_MODES = ("entropic", "grpo", "cvar", "rank", "x-grpo", "spo-rs")
CLIPPED_POLICY_MODES = frozenset(
    ("rank", "x-grpo", "spo-rs", "binary-coder"))
CVAR_ALPHA_DEFAULT = 0.2     # tail mass: cutoff at the 80th percentile
CVAR_LAMBDA_DEFAULT = 0.5    # weight of the upper-tail term vs plain GRPO
# RANK_GAMMA_DEFAULT = math.log(4) # cirlce pack
RANK_GAMMA_DEFAULT = math.log(8) # erdos
RANK_CLIP_EPSILON_DEFAULT = 0.2
RANK_ENTROPY_COEF_DEFAULT = 0.001
RANK_UPDATE_EPOCHS_DEFAULT = 1
X_GRPO_RELATIVE_ERROR_DEFAULT = 0.99
X_GRPO_ENTROPY_COEF_DEFAULT = 0.001
# X_GRPO_CONTEXTS_PER_STEP_DEFAULT = 6
X_GRPO_CONTEXTS_PER_STEP_DEFAULT = 3
X_GRPO_BASE_BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# X_GRPO_BASE_BUDGETS = (0.25, 0.5, 1.0, 2.0, 4.0)
SPO_RS_BETA_DEFAULT = 2.0
SPO_RS_D_HALF_DEFAULT = 0.06
SPO_RS_RHO_MIN_DEFAULT = 0.875
SPO_RS_RHO_MAX_DEFAULT = 0.96
SPO_RS_CLIP_EPSILON_DEFAULT = 0.2
SPO_RS_CLIP_EPSILON_LOW_DEFAULT = 0.2
SPO_RS_CLIP_EPSILON_HIGH_DEFAULT = 0.35


def _uses_clipped_policy_loss(mode):
    return str(mode or "entropic").lower() in CLIPPED_POLICY_MODES


def _resolve_binary_coder_options(merged):
    from problems.binary_coder import BinaryCoderConfig

    resolved = BinaryCoderConfig.from_mapping(merged)
    merged.update(resolved.as_dict())
    if resolved.enabled:
        if (float(merged["temperature"]) != 1.0
                or float(merged["top_p"]) != 1.0):
            print("[config] binary coder PPO sets temperature=1 and top_p=1 "
                  "to match the policy likelihood used by the loss")
        merged["temperature"] = 1.0
        merged["top_p"] = 1.0
        merged["sampling_top_k"] = 0
        merged["sampling_min_p"] = 0.0
        merged["qwen3_8b_thinking_sampling"] = False


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
    sequence_policy = _is_exact_qwen3_30b_a3b(merged.get("model_name"))
    if (_uses_clipped_policy_loss(merged.get("advantage_mode"))
            or sequence_policy):
        if int(merged["group_size"]) < 2:
            raise ValueError(
                "sequence/clipped policy modes require group_size >= 2")
        kl_coef = float(merged["kl_penalty_coef"])
        if not math.isfinite(kl_coef) or kl_coef < 0.0:
            raise ValueError(
                "sequence/clipped policy modes require a finite nonnegative "
                "kl_penalty_coef")
        if (float(merged["temperature"]) != 1.0
                or float(merged["top_p"]) != 1.0):
            label = ("Qwen3-30B-A3B sequence-policy mode"
                     if sequence_policy else "clipped-policy mode")
            print(f"[config] {label} sets temperature=1 and top_p=1 "
                  "to match the policy likelihood used by the loss")
        merged["temperature"] = 1.0
        merged["top_p"] = 1.0
        merged["sampling_top_k"] = 0
        merged["sampling_min_p"] = 0.0
        merged["qwen3_8b_thinking_sampling"] = False


def _resolve_x_grpo_options(merged):
    if merged.get("advantage_mode") != "x-grpo":
        return

    group_size = int(merged["group_size"])
    contexts = merged.get("x_grpo_contexts_per_step")
    contexts = (X_GRPO_CONTEXTS_PER_STEP_DEFAULT
                if contexts is None else contexts)
    if (isinstance(contexts, bool)
            or not isinstance(contexts, (int, np.integer))
            or contexts < 1):
        raise ValueError("x_grpo_contexts_per_step must be a positive integer")
    merged["x_grpo_contexts_per_step"] = int(contexts)

    raw_budgets = merged.get("x_grpo_budgets")
    if raw_budgets is None:
        maximum = float(group_size - 1)
        budgets = [value for value in X_GRPO_BASE_BUDGETS
                   if value <= maximum]
        if not budgets or budgets[-1] != maximum:
            budgets.append(maximum)
    elif isinstance(raw_budgets, str):
        pieces = [piece.strip() for piece in raw_budgets.split(",")]
        if not pieces or any(not piece for piece in pieces):
            raise ValueError(
                "x_grpo_budgets must be a comma-separated list of numbers")
        budgets = [float(piece) for piece in pieces]
    elif isinstance(raw_budgets, (list, tuple)):
        budgets = [float(value) for value in raw_budgets]
    else:
        raise ValueError("x_grpo_budgets must be a sequence or comma-separated string")

    budgets = sorted(set(budgets))
    if not budgets:
        raise ValueError("x_grpo_budgets must not be empty")
    maximum = float(group_size - 1)
    if any(not math.isfinite(value) or value <= 0.0 or value > maximum
           for value in budgets):
        raise ValueError(
            f"x_grpo_budgets must be finite and in (0, {maximum}]")
    merged["x_grpo_budgets"] = tuple(budgets)

    relative_error = merged.get("x_grpo_relative_error")
    relative_error = (X_GRPO_RELATIVE_ERROR_DEFAULT
                      if relative_error is None else float(relative_error))
    if not math.isfinite(relative_error) or not 0.0 < relative_error < 1.0:
        raise ValueError("x_grpo_relative_error must be finite and in (0, 1)")
    merged["x_grpo_relative_error"] = relative_error

    entropy_coef = merged.get("x_grpo_entropy_coef")
    entropy_coef = (X_GRPO_ENTROPY_COEF_DEFAULT
                    if entropy_coef is None else float(entropy_coef))
    if not math.isfinite(entropy_coef) or entropy_coef < 0.0:
        raise ValueError("x_grpo_entropy_coef must be finite and nonnegative")
    if merged.get("advantage_mode") == "x-grpo":
        if int(merged["groups_per_step"]) < 3:
            raise ValueError("X-GRPO requires groups_per_step >= 3")
        if entropy_coef <= 0.0:
            raise ValueError("X-GRPO requires x_grpo_entropy_coef > 0")
    merged["x_grpo_entropy_coef"] = entropy_coef


def _resolve_spo_rs_options(merged):
    if merged.get("advantage_mode") != "spo-rs":
        return

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
    if merged["spo_rs_d_half"] <= 0.0:
        raise ValueError("spo_rs_d_half must be positive")
    if not (0.0 < merged["spo_rs_rho_min"]
            <= merged["spo_rs_rho_max"] < 1.0):
        raise ValueError(
            "SPO-RS rho bounds must satisfy 0 < rho_min <= rho_max < 1")
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
            value = (merged["spo_rs_clip_epsilon"]
                     if raw_clip_epsilon is not None else default)
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
    p.add_argument("--num-steps", type=int, default=None,
                   help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=None,
                   help="Parent states per step; in X-GRPO, K independent "
                        "groups per fixed context (paper default: 8)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Rollouts per parent group; in X-GRPO, group size G "
                        "(paper default: 64)")
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
    p.add_argument("--uct", action="store_const", const=True, default=None,
                   help="Use UCT parent selection instead of PUCT. Both use "
                        "the same puct_c configuration value as c.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--coder-retry", action="store_const", const=True, default=None,
        help="Retry each coder response missing its required final code block "
             "once. Disabled unless this flag is passed; the original failed "
             "rollout is retained and the retry is an extra training example.")
    p.add_argument("--strategy-max-new-tokens", type=int, default=None)
    p.add_argument("--strategy-format-max-retries", type=int, default=None,
                   help="Additional attempts for missing strategy blocks; "
                        "stop if exhausted (default: 3).")
    p.add_argument("--strategy-max-seq-length", type=int, default=None)
    p.add_argument("--strategy-temperature", type=float, default=None)
    p.add_argument("--strategy-top-p", type=float, default=None)
    p.add_argument("--strategy-thinking", dest="strategy_thinking",
                   action="store_const", const=True, default=None)
    p.add_argument("--no-strategy-thinking", dest="strategy_thinking",
                   action="store_const", const=False)
    p.add_argument("--strategy-reasoning-effort",
                   choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort passed to strategy-model chat "
                        "templates that support it, including GPT-OSS.")
    p.add_argument("--strategy-vllm-quantization", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--adam-beta1", type=float, default=None)
    p.add_argument("--adam-beta2", type=float, default=None)
    p.add_argument("--adam-epsilon", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--kl-penalty-coef", type=float, default=None)
    p.add_argument("--advantage-mode", type=str.lower,
                   choices=list(ADVANTAGE_MODES), default=None,
                   help="Group advantage estimator: 'entropic' (adaptive-beta "
                        "entropic objective, default), 'grpo' (mean/std "
                        "normalized), or 'cvar' (grpo blended with an "
                        "upper-tail term above the (1-alpha)-quantile), or "
                        "'rank'/'x-grpo' (midrank objectives with clipped GRPO), "
                        "or 'spo-rs' (KL-adaptive entropic value tracking with "
                        "configurable asymmetric PPO clipping).")
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
    p.add_argument("--x-grpo-budgets", default=None,
                   help="X-GRPO: comma-separated concentration-budget grid. "
                        "Defaults to a geometric grid capped by group_size-1.")
    p.add_argument("--x-grpo-contexts-per-step", type=int, default=None,
                   help="X-GRPO: number P of independently conditioned parent "
                        "contexts per step (default 1). groups_per_step is K "
                        "cross-fit groups per context and group_size is G.")
    p.add_argument("--x-grpo-relative-error", type=float, default=None,
                   help="X-GRPO: tolerated leave-one-group-out relative "
                        "gradient-estimation error c (default 0.5).")
    p.add_argument("--x-grpo-entropy-coef", type=float, default=None,
                   help="X-GRPO: exploration coefficient for rejected-budget "
                        "or completely tied groups (default 0.001).")
    p.add_argument("--spo-rs-beta", type=float, default=None,
                   help="SPO-RS: fixed entropic reward coefficient beta "
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


def _validate_known_training_capacity(
        training_model_name, budgets, *, replicated=False):
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
    from config_validation import (RETIRED_BATCH_GROWTH_KEYS,
                                   validate_problem_config)
    for key in RETIRED_BATCH_GROWTH_KEYS:
        ydict.pop(key, None)
    # Old external configs may retain the retired repair-feature settings.
    for key in list(ydict):
        if key == "feedback" or key.startswith("feedback_"):
            ydict.pop(key)
    validate_problem_config(
        ydict,
        source=cfg_path,
        require_complete=(
            Path(cfg_path).expanduser().resolve().parent == config_dir.resolve()
        ),
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
        # Runs created before the binary coder phase existed must retain their
        # original adapter rank/objective when resumed, even if the current
        # problem preset now enables the new phase for fresh runs.
        if "binary_coder_training" not in saved:
            merged["binary_coder_training"] = False
        # A run saved before adaptive pilots existed must resume its original
        # one-pass rollout schedule even if today's YAML enables pilots.
        if "pilot_programs_per_strategy" not in saved:
            merged["pilot_programs_per_strategy"] = -1
        # Runs created before selectable phase-2 allocation used the original
        # median-quadrant rule.  Do not silently switch a resumed run to a new
        # allocator merely because its current problem YAML selects one.
        if "phase2_allocation_method" not in saved:
            merged["phase2_allocation_method"] = "rule_based"
        print(f"[config] resuming original configuration from "
              f"{resume_dir / 'config.json'}")

    # 3) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type", "resume"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
        if value != parser.get_default(arg_name):
            profile_explicit_keys.add(key)
    if args.rank_clip_epsilon is not None:
        if args.rank_clip_epsilon_low is None:
            merged["rank_clip_epsilon_low"] = args.rank_clip_epsilon
        if args.rank_clip_epsilon_high is None:
            merged["rank_clip_epsilon_high"] = args.rank_clip_epsilon
    if args.spo_rs_clip_epsilon is not None:
        if args.spo_rs_clip_epsilon_low is None:
            merged["spo_rs_clip_epsilon_low"] = args.spo_rs_clip_epsilon
        if args.spo_rs_clip_epsilon_high is None:
            merged["spo_rs_clip_epsilon_high"] = args.spo_rs_clip_epsilon
    # These modes are deliberately launch-scoped. Saved/YAML values must not
    # silently change a later invocation's trainer or evaluator.
    merged["fast"] = bool(args.fast)
    merged["strategies"] = bool(args.strategies)
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
                merged["training_model_name"] = str(
                    merged.get("coder_training_model_name") or "").strip()

    # Resolve after the hierarchy chooses its coder and before validating the
    # resulting training layout and precision.
    _apply_coder_model_profile(merged, profile_explicit_keys)
    _resolve_coder_token_limits(merged)

    training_layout = str(
        merged.get("training_layout") or "auto").strip().lower()
    if training_layout not in {"auto", "replicated", "sharded"}:
        raise ValueError(
            "training_layout must be 'auto', 'replicated', or 'sharded'")
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
    strategy_format_max_retries = merged.get("strategy_format_max_retries", 3)
    if (isinstance(strategy_format_max_retries, bool)
            or not isinstance(strategy_format_max_retries, int)
            or strategy_format_max_retries < 0):
        raise ValueError("strategy_format_max_retries must be a non-negative integer")
    merged["strategy_format_max_retries"] = strategy_format_max_retries
    if merged["strategies"]:
        strategy_model_name = str(
            merged.get("strategy_model_name") or merged["model_name"]
        ).strip()
        _apply_strategy_model_profile(
            merged, strategy_model_name, profile_explicit_keys)
        strategy_backend = str(
            merged.get("strategy_backend") or "local"
        ).strip().lower()
        if strategy_backend not in {"local", "api"}:
            raise ValueError("strategy_backend must be 'local' or 'api'")
        strategies_per_parent = int(merged["strategies_per_parent"])
        programs_per_strategy = int(merged["programs_per_strategy"])
        pilot_programs_per_strategy = int(
            merged.get("pilot_programs_per_strategy", -1))
        phase2_allocation_method = str(
            merged.get("phase2_allocation_method") or "rule_based"
        ).strip().lower()
        strategy_archive_top_r = int(merged["strategy_archive_top_r"])
        strategy_max_new_tokens = int(merged["strategy_max_new_tokens"])
        strategy_max_seq_length = int(merged["strategy_max_seq_length"])
        strategy_temperature = float(merged["strategy_temperature"])
        strategy_top_p = float(merged["strategy_top_p"])
        if (strategies_per_parent < 1 or programs_per_strategy < 1
                or strategy_archive_top_r < 1
                or strategy_max_new_tokens < 1
                or strategy_max_seq_length < 1):
            raise ValueError(
                "strategy counts and token limits must be positive")
        if (pilot_programs_per_strategy == 0
                or pilot_programs_per_strategy < -1):
            raise ValueError(
                "pilot_programs_per_strategy must be -1 (disabled) or a "
                "positive integer")
        if pilot_programs_per_strategy > programs_per_strategy:
            raise ValueError(
                "pilot_programs_per_strategy cannot exceed "
                "programs_per_strategy")
        if phase2_allocation_method not in {"rule_based", "bandit", "hurdle"}:
            raise ValueError(
                "phase2_allocation_method must be 'rule_based', 'bandit', or 'hurdle'")
        if strategy_temperature <= 0.0:
            raise ValueError("strategy_temperature must be positive")
        if not 0.0 < strategy_top_p <= 1.0:
            raise ValueError("strategy_top_p must be in (0, 1]")
        effective_group_size = (
            strategies_per_parent * programs_per_strategy)
        if (int(merged.get("group_size", effective_group_size))
                != effective_group_size):
            print(f"[config] hierarchical rollout group size derived as "
                  f"{strategies_per_parent} strategies x "
                  f"{programs_per_strategy} programs = "
                  f"{effective_group_size}; replacing configured group_size="
                  f"{merged.get('group_size')}")
        merged["strategy_model_name"] = strategy_model_name
        merged["strategy_backend"] = strategy_backend
        merged["strategies_per_parent"] = strategies_per_parent
        merged["programs_per_strategy"] = programs_per_strategy
        merged["pilot_programs_per_strategy"] = (
            pilot_programs_per_strategy)
        merged["phase2_allocation_method"] = phase2_allocation_method
        merged["strategy_archive_top_r"] = strategy_archive_top_r
        merged["strategy_max_new_tokens"] = strategy_max_new_tokens
        merged["strategy_max_seq_length"] = strategy_max_seq_length
        merged["strategy_temperature"] = strategy_temperature
        merged["strategy_top_p"] = strategy_top_p
        merged["strategy_thinking"] = bool(merged["strategy_thinking"])
        strategy_reasoning_effort = str(
            merged.get("strategy_reasoning_effort") or "high"
        ).strip().lower()
        if strategy_reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "strategy_reasoning_effort must be low, medium, or high")
        merged["strategy_reasoning_effort"] = strategy_reasoning_effort
        merged["strategy_vllm_quantization"] = str(
            merged.get("strategy_vllm_quantization") or "").strip()
        merged["strategy_api_base_url"] = str(
            merged.get("strategy_api_base_url")
            or "https://api.deepseek.com").strip().rstrip("/")
        merged["strategy_api_key_env"] = str(
            merged.get("strategy_api_key_env")
            or "DEEPSEEK_API_KEY").strip()
        merged["strategy_api_concurrency"] = int(
            merged.get("strategy_api_concurrency") or 8)
        merged["strategy_api_timeout_s"] = float(
            merged.get("strategy_api_timeout_s") or 1800.0)
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
        if (strategy_backend == "api"
                and not os.environ.get(merged["strategy_api_key_env"], "")):
            raise ValueError(
                "strategy API mode requires environment variable "
                f"{merged['strategy_api_key_env']} to be set")
        merged["group_size"] = effective_group_size
    # This ablation intentionally keeps two independent Qwen3-8B engines per
    # card resident together: a permanently base/no-LoRA strategist and a
    # coder that receives the current LoRA adapter. Keep the switch derived
    # rather than user-configurable so every other model pairing retains the
    # existing alternating-pool behavior.
    merged["dual_resident_qwen3_8b_strategy_coder_pools"] = (
        _dual_resident_qwen3_8b_strategy_coder_pools(merged))
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
    # This profile is derived after coder/strategist selection. Clipped-policy
    # resolvers below remain authoritative when they require full-policy
    # sampling for an exact likelihood ratio.
    _apply_qwen3_8b_thinking_sampling(merged, profile_explicit_keys)
    _resolve_binary_coder_options(merged)
    _resolve_rank_options(merged)
    _resolve_x_grpo_options(merged)
    _resolve_spo_rs_options(merged)
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
    for key in ("vllm_sleep_level", "strategy_vllm_sleep_level"):
        value = int(merged.get(key, 2))
        if value not in (1, 2):
            raise ValueError(f"{key} must be 1 or 2")
        merged[key] = value
    merged["vllm_staged_loading"] = bool(
        merged.get("vllm_staged_loading", True))
    merged["strategy_vllm_staged_loading"] = bool(
        merged.get("strategy_vllm_staged_loading", True))
    strategy_persistent_workers = merged.get(
        "strategy_vllm_persistent_workers")
    if strategy_persistent_workers is not None:
        strategy_persistent_workers = int(strategy_persistent_workers)
        if strategy_persistent_workers < 0:
            raise ValueError(
                "strategy_vllm_persistent_workers must be nonnegative")
    merged["strategy_vllm_persistent_workers"] = (
        strategy_persistent_workers)

    # Resolve every physical role from one ordered inventory. run.sh exports
    # AVAILABLE_GPUS and that environment value is authoritative over old YAML
    # and resumed role fields. Direct Python invocations fall back to the legacy
    # inventory key, CUDA visibility, then one training device.
    from gpu_runtime import (align_vllm_layout_to_concurrency,
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

    if merged.get("coder_model_profile") == "gpt-oss-120b":
        generation_cards = [
            memory[gpu_id] for gpu_id in gpu_ids if gpu_id in memory]
        native_mxfp4_cards = bool(generation_cards) and all(
            ("blackwell" in card.name.lower()
             or "rtx pro 6000" in card.name.lower())
            for card in generation_cards
        )
        # A CPU-only configuration inspection has no device evidence, so keep
        # the conservative guard. On the target RTX PRO 6000 Blackwell host,
        # this admits eight independent TP=1 engines over eight cards.
        merged["allow_gpt_oss_tp1"] = native_mxfp4_cards
        if native_mxfp4_cards:
            print("[model-profile] GPT-OSS native MXFP4 Blackwell path: "
                  "TP=1 rollout engines admitted")
        else:
            print("[model-profile] GPT-OSS TP=1 native-MXFP4 support was not "
                  "confirmed on every rollout GPU; retaining automatic "
                  "TP sharding")

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
    if (merged.get("coder_model_profile") == "gpt-oss-120b"
            and merged["backend"] != "unsloth"):
        raise ValueError(
            "the automatic GPT-OSS-120B coder profile requires the Unsloth "
            "BitsAndBytes training backend; native MXFP4 remains the vLLM "
            "rollout checkpoint but is not differentiable")
    if merged["training_layout"] == "sharded":
        if merged["backend"] != "hf":
            raise ValueError(
                "training_layout=sharded currently requires backend=hf")
        if int(merged["num_training_gpus"]) < 2:
            raise ValueError(
                "training_layout=sharded requires at least two training GPUs")
        if bool(merged.get("load_in_4bit", False)):
            raise ValueError(
                "training_layout=sharded expects an unquantized trainable "
                "checkpoint; set load_in_4bit: false")
    training_memory_fraction = float(
        merged.get("training_memory_fraction", 0.80))
    if not 0.0 < training_memory_fraction <= 0.90:
        raise ValueError(
            "training_memory_fraction must be in (0, 0.90]")
    merged["training_memory_fraction"] = training_memory_fraction
    replicated_weight_budget = bool(
        merged["fast"] or merged["training_layout"] == "replicated")
    training_budgets = _resolve_training_memory_budgets(
        training_gpu_ids, memory,
        max_fraction=(
            training_memory_fraction if replicated_weight_budget else 0.90))
    _validate_known_training_capacity(
        merged["training_model_name"], training_budgets,
        replicated=replicated_weight_budget)
    merged["training_max_memory_gib"] = training_budgets
    if training_budgets:
        print(f"[memory] training weight budgets by logical GPU: "
              f"{training_budgets} GiB")

    if merged["fast"] and not merged["no_train"]:
        fast_clipped_policy = bool(
            merged["binary_coder_training"]
            or _uses_clipped_policy_loss(merged["advantage_mode"])
        )
        if (fast_clipped_policy
                and int(merged["rank_update_epochs"]) != 1):
            raise ValueError("--fast requires one clipped-policy update epoch")
        fast_backend_supported = bool(
            merged["backend"] == "hf"
            or (merged["backend"] == "unsloth"
                and merged.get("coder_model_profile") == "gpt-oss-120b")
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
            and (str(merged.get("strategy_model_name")
                     or merged.get("model_name"))
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
        profile_runtime_reserve = merged.get(
            "vllm_runtime_reserve_override_gib")
        merged["vllm_runtime_reserve_gib"] = (
            automatic_runtime_reserve
            if profile_runtime_reserve is None else
            float(profile_runtime_reserve)
        )
        known_heads = detect_attention_heads(merged.get("model_name", ""))
        layout = derive_vllm_parallel_layout(
            merged, roles, memory, known_heads)
        if merged["dual_resident_qwen3_8b_strategy_coder_pools"]:
            # The coder needs one independent TP=1 engine per selected card;
            # the strategy topology is derived separately below from its own
            # model and context memory needs. Refuse an unsafe host rather than silently
            # weakening coder rollout parallelism.
            if (layout.tensor_parallel_size != 1
                    or layout.pipeline_parallel_size != 1):
                raise ValueError(
                    "the dual-resident Qwen3-8B strategy/coder ablation "
                    "requires one coder engine per GPU (TP=1, PP=1), but "
                    "the available memory cannot fit that layout "
                    "at the configured context")
        merged["vllm_tensor_parallel_size"] = layout.tensor_parallel_size
        merged["vllm_pipeline_parallel_size"] = layout.pipeline_parallel_size
        validate_attention_heads(
            known_heads,
            merged["vllm_tensor_parallel_size"],
            merged.get("model_name", ""),
        )
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

        if (merged["strategies"]
                and merged.get("strategy_backend", "local") == "local"):
            strategy_name = str(
                merged.get("strategy_model_name") or merged.get("model_name"))
            strategy_layout_cfg = dict(merged)
            strategy_layout_cfg.update({
                "model_name": strategy_name,
                "max_seq_length": int(merged["strategy_max_seq_length"]),
                "load_in_4bit": False,
                "vllm_quantization": merged.get(
                    "strategy_vllm_quantization", ""),
                "vllm_runtime_reserve_gib": vllm_runtime_reserve_gib(
                    co_resident_sleep=True,
                    token_scoring=False,
                ),
            })
            strategy_heads = detect_attention_heads(strategy_name)
            memory_strategy_layout = derive_vllm_parallel_layout(
                strategy_layout_cfg, roles, memory, strategy_heads)
            # Strategies inside one parent are sequential, so the number of
            # parents/folds—not strategies_per_parent—is the largest live
            # request frontier. Fold replicas above it into TP/PP ranks rather
            # than leaving their GPUs idle for every dependent stage.
            strategy_frontier = int(merged["groups_per_step"])
            if str(merged.get("advantage_mode", "")).lower() == "x-grpo":
                strategy_frontier *= int(
                    merged.get("x_grpo_contexts_per_step", 1))
            strategy_layout = align_vllm_layout_to_concurrency(
                strategy_layout_cfg,
                memory_strategy_layout,
                len(gpu_ids),
                strategy_heads,
                strategy_frontier,
            )
            strategy_layout_cfg["vllm_tensor_parallel_size"] = (
                strategy_layout.tensor_parallel_size)
            strategy_layout_cfg["vllm_pipeline_parallel_size"] = (
                strategy_layout.pipeline_parallel_size)
            for note in resolve_memory_settings(
                    strategy_layout_cfg, roles, memory):
                print(f"[memory] auto strategy: {note}")
            merged["strategy_vllm_tensor_parallel_size"] = (
                strategy_layout.tensor_parallel_size)
            merged["strategy_vllm_pipeline_parallel_size"] = (
                strategy_layout.pipeline_parallel_size)
            merged["strategy_gen_micro_batch"] = int(
                strategy_layout_cfg["gen_micro_batch"])
            merged["strategy_vllm_max_num_batched_tokens"] = int(
                strategy_layout_cfg["vllm_max_num_batched_tokens"])
            merged["strategy_vllm_runtime_reserve_gib"] = float(
                strategy_layout_cfg["vllm_runtime_reserve_gib"])
            merged["strategy_generation_frontier"] = strategy_frontier
            merged["separate_strategy_inference_pool"] = bool(
                strategy_name != str(merged.get("model_name"))
                or merged["dual_resident_qwen3_8b_strategy_coder_pools"]
                or strategy_layout.tensor_parallel_size
                != layout.tensor_parallel_size
                or strategy_layout.pipeline_parallel_size
                != layout.pipeline_parallel_size
            )
            validate_attention_heads(
                strategy_heads,
                strategy_layout.tensor_parallel_size,
                strategy_name,
            )
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
          and str(merged.get("strategy_model_name")
                  or merged.get("model_name"))
          != str(merged.get("model_name"))):
        raise ValueError(
            "a distinct strategy_model_name currently requires "
            "generation_backend=vllm")

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
    from gen_workers import _canonical_generated_text

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    max_new_tokens = min(
        int(cfg.max_new_tokens), int(cfg.max_seq_length) - int(input_len))
    if max_new_tokens < 1:
        return [("", []) for _ in range(int(n_samples))]

    policy_sampling = {}
    sampling_top_k = getattr(cfg, "sampling_top_k", None)
    sampling_min_p = getattr(cfg, "sampling_min_p", None)
    if sampling_top_k is not None:
        policy_sampling["top_k"] = int(sampling_top_k)
    if sampling_min_p is not None:
        policy_sampling["min_p"] = float(sampling_min_p)
    if (_uses_clipped_policy_loss(
            getattr(cfg, "advantage_mode", "entropic"))
            or _uses_sequence_level_policy_ratio(cfg)):
        policy_sampling.update(top_k=0, repetition_penalty=1.0)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            **policy_sampling,
            pad_token_id=pad_id,
            num_return_sequences=n_samples,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = _canonical_generated_text(tokenizer, gen_ids, decoded)
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
                         top_p=None, top_k=None, min_p=None, cap_state=None):
    """Stream locally generated rollouts from cross-prompt HF batches."""
    if len(prompts_by_group) != len(counts_by_group):
        raise ValueError("counts_by_group must align with prompts_by_group")
    if _uses_clipped_policy_loss(getattr(cfg, "advantage_mode", "entropic")):
        # This local path owns its sampling arguments, including top_k=0.
        # Use the existing per-prompt OOM-aware micro-batcher instead of an
        # external HF helper that may inherit a top-k generation default.
        rank_cfg = SimpleNamespace(**vars(cfg))
        rank_cfg.temperature = 1.0
        rank_cfg.top_p = 1.0
        rank_cfg.sampling_top_k = 0
        rank_cfg.sampling_min_p = 0.0
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
    resolved_top_k = (top_k if top_k is not None
                      else getattr(cfg, "sampling_top_k", None))
    resolved_min_p = (min_p if min_p is not None
                      else getattr(cfg, "sampling_min_p", None))
    if resolved_top_k is not None:
        gen_kwargs["top_k"] = int(resolved_top_k)
    if resolved_min_p is not None:
        gen_kwargs["min_p"] = float(resolved_min_p)
    if _uses_sequence_level_policy_ratio(cfg):
        gen_kwargs.update(top_k=0, repetition_penalty=1.0)
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
        # Examples may be built while the trainer is offloaded to CPU. With
        # model sharding, the output head can also run on a different GPU from
        # the input embeddings. Align indices with the actual projected scores
        # here, including when checkpoint recomputes this chunk in backward.
        target_chunk = target_chunk.to(device=log_probs.device)
        chosen = log_probs.gather(
            1, target_chunk.unsqueeze(-1)).squeeze(-1)
        if not return_entropy:
            return chosen
        entropy = _token_entropy(
            log_probs, detached=(return_entropy == "measure"))
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


def _shared_prefix_token_logprobs(model, examples, *, chunk, with_grad,
                                  return_entropy):
    """Score response branches while evaluating their common prompt once.

    The packed sequence contains one prompt followed by isolated response
    branches.  Each branch resets its logical positions after the prompt and
    the registered attention backend lets it see only the prompt and its own
    causal history.  The prompt remains in the autograd graph, so gradients
    from every response accumulate through it exactly; no detached-KV
    approximation is used.
    """
    import torch

    examples = list(examples)
    if len(examples) < 2:
        return None
    components = _causal_lm_decoder_and_head(model)
    if components is None:
        return None
    decoder, output_head = components
    decoder_config = getattr(decoder, "config", None)
    if str(getattr(decoder_config, "_attn_implementation", "")) != (
            "ttt_blockwise_attention"):
        return None
    # Hybrid Gated-DeltaNet layers carry a recurrent state along the sequence.
    # A custom mask isolates the full-attention layers, but cannot reset that
    # recurrent state between concatenated response branches. Score hybrid
    # models as ordinary padded batch rows so every rollout has independent
    # state, matching the original one-rollout forward exactly.
    layer_types = tuple(getattr(decoder_config, "layer_types", ()) or ())
    if "linear_attention" in layer_types:
        return None

    prompt = examples[0]["prompt_ids"]
    if prompt.ndim != 2 or int(prompt.shape[0]) != 1:
        return None
    for example in examples[1:]:
        candidate = example["prompt_ids"]
        if (candidate.shape != prompt.shape
                or candidate.device != prompt.device
                or candidate.dtype != prompt.dtype
                or not torch.equal(candidate, prompt)):
            return None

    prompt_length = int(prompt.shape[1])
    if prompt_length < 1:
        return None
    device = prompt.device
    input_parts = [prompt[0]]
    segment_parts = [torch.zeros(
        prompt_length, dtype=torch.long, device=device)]
    position_parts = [torch.arange(
        prompt_length, dtype=torch.long, device=device)]
    response_ranges = []
    attention_branches = []
    cursor = prompt_length
    for branch_index, example in enumerate(examples, start=1):
        response = example["response_ids"]
        if (response.ndim != 2 or int(response.shape[0]) != 1
                or int(response.shape[1]) < 1):
            return None
        branch_input = response[0, :-1]
        branch_length = int(branch_input.shape[0])
        start = cursor
        end = start + branch_length
        response_ranges.append((start, end))
        if branch_length:
            input_parts.append(branch_input)
            segment_parts.append(torch.full(
                (branch_length,), branch_index,
                dtype=torch.long, device=device))
            position_parts.append(torch.arange(
                prompt_length, prompt_length + branch_length,
                dtype=torch.long, device=device))
            attention_branches.append((start, end))
            cursor = end

    packed_ids = torch.cat(input_parts, dim=0).unsqueeze(0)
    segments = torch.cat(segment_parts, dim=0)
    positions = torch.cat(position_parts, dim=0)
    compact_mask = torch.stack((segments, positions), dim=0).unsqueeze(0)
    from model_backend import register_shared_prefix_attention_mask
    register_shared_prefix_attention_mask(
        compact_mask, prompt_length, attention_branches)

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = decoder(
            input_ids=packed_ids,
            attention_mask=compact_mask,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            try:
                hidden = outputs[0]
            except (IndexError, KeyError, TypeError):
                return None
        prompt_last = hidden[0, prompt_length - 1:prompt_length, :]
        scored = []
        entropies = []
        for example, (start, end) in zip(examples, response_ranges):
            prediction_hidden = torch.cat(
                (prompt_last, hidden[0, start:end, :]), dim=0)
            targets = example["response_ids"][0]
            result = _score_hidden_token_chunks(
                prediction_hidden, targets, output_head,
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
        targets = response_ids.to(device=pred_logits.device)

        if chunk and 0 < chunk < R:
            parts = []
            entropies = []
            for s in range(0, R, chunk):
                e = min(s + chunk, R)
                lp = F.log_softmax(pred_logits[:, s:e, :].float(), dim=-1)
                g = lp.gather(2, targets[:, s:e].unsqueeze(-1)).squeeze(-1)
                parts.append(g)          # keep only (1, e-s); lp freed next iter
                if return_entropy:
                    entropies.append(_token_entropy(
                        lp, detached=(return_entropy == "measure")))
            gathered = torch.cat(parts, dim=1)  # (1, R)
            if return_entropy:
                entropy = torch.cat(entropies, dim=1)
        else:
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, R)
            if return_entropy:
                entropy = _token_entropy(
                    log_probs, detached=(return_entropy == "measure"))
    if return_entropy:
        return gathered.squeeze(0), entropy.squeeze(0)
    return gathered.squeeze(0)


@measure_policy_entropy
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

    shared = _shared_prefix_token_logprobs(
        model, examples, chunk=chunk, with_grad=with_grad,
        return_entropy=return_entropy)
    if shared is not None:
        return shared

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
            targets = response_ids[0].to(device=pred_logits.device)
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
                    entropies.append(_token_entropy(
                        log_probs, detached=(return_entropy == "measure")))
            gathered_examples.append(torch.cat(parts, dim=0))
            if return_entropy:
                entropy_examples.append(torch.cat(entropies, dim=0))
    if return_entropy:
        return gathered_examples, entropy_examples
    return gathered_examples


def _requires_fused_long_singleton(cfg, sequence_tokens):
    """Whether one long trajectory should use unmasked fused attention.

    Multi-example batches require padding or the custom shared-prefix mask and
    therefore cannot enter the exact full-sequence Flash-SDPA path.  Keep the
    configured microbatch cap for ordinary trajectories, but isolate genuinely
    long ones when the user explicitly enabled fused long attention.
    """
    return bool(
        getattr(cfg, "fused_long_attention", False)
        and int(sequence_tokens) >= _FUSED_LONG_SINGLETON_MIN_TOKENS
    )


def _exceeds_fused_long_pack(cfg, packed_tokens):
    """Prevent a masked multi-example pack from becoming a long sequence."""
    return bool(
        getattr(cfg, "fused_long_attention", False)
        and int(packed_tokens) >= _FUSED_LONG_SINGLETON_MIN_TOKENS
    )


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
            if _requires_fused_long_singleton(cfg, length):
                if current:
                    batches.append(current)
                    current = []
                    current_max = 0
                batches.append([example])
                continue
            next_max = max(current_max, length)
            exceeds_tokens = bool(
                current and next_max * (len(current) + 1) > token_cap)
            exceeds_fused_pack = bool(
                current and _exceeds_fused_long_pack(
                    cfg, next_max * (len(current) + 1)))
            if current and (len(current) >= example_cap or exceeds_tokens
                            or exceeds_fused_pack):
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


def _split_training_batches_to_work_cap(batches, work_cap):
    """Reuse a learned safe workload size without hard-coding batch counts.

    The OOM controller's work estimate accounts for ordinary padded batches
    and packed shared-prefix batches.  Recursively halving only batches above
    the learned cap preserves every example and the original order while
    avoiding a known-to-fail large attempt on later, similarly sized work.
    """
    work_cap = max(1, int(work_cap))
    split = []
    changed = False

    def append(batch):
        nonlocal changed
        batch = list(batch)
        if (len(batch) <= 1
                or _gradient_checkpointing_work_units([batch]) <= work_cap):
            split.append(batch)
            return
        middle = (len(batch) + 1) // 2
        changed = True
        append(batch[:middle])
        append(batch[middle:])

    for batch in batches:
        if batch:
            append(batch)
    return split, changed


def _oom_profile_work_floor(work):
    """Lower edge of the workload neighborhood covered by an OOM result."""
    work = max(1, int(work))
    # Lengths of sibling rollouts are rarely identical.  A narrow 12.5%
    # neighborhood lets the next comparable branch reuse the proven mode,
    # while materially shorter work still gets a chance to use the fast path.
    return max(1, work - max(1, work // 8))


def _remember_oom_profile_minimum(profile, key, value):
    value = max(1, int(value))
    previous = profile.get(key)
    profile[key] = value if previous is None else min(int(previous), value)


def _gradient_checkpointing_work_units(batches):
    """Estimate the largest independently executed batch's activation load."""
    largest = 0
    for batch in batches:
        batch = list(batch)
        if not batch:
            continue
        prompt_job_id = batch[0].get("prompt_job_id")
        shared_prefix = bool(
            len(batch) > 1
            and prompt_job_id is not None
            and all(bool(example.get(
                "_shared_prefix_packing_allowed", True))
                    for example in batch)
            and all(
                example.get("prompt_job_id") == prompt_job_id
                and tuple(example["prompt_ids"].shape)
                == tuple(batch[0]["prompt_ids"].shape)
                for example in batch[1:]
            )
        )
        if shared_prefix:
            work = int(batch[0]["prompt_ids"].shape[1]) + sum(
                max(0, int(example["response_ids"].shape[1]) - 1)
                for example in batch
            )
        else:
            work = max(
                int(example["prompt_ids"].shape[1]
                    + example["response_ids"].shape[1])
                for example in batch
            ) * len(batch)
        largest = max(largest, work)
    return largest


def _run_oom_resilient_backward(
        model, batches, attempt, *, device_label, expand_memory=None,
        start_checkpointed=False, start_with_headroom=False):
    """Retry with checkpointing, smaller batches, and memory fallbacks.

    Each attempt starts the local replica's gradients from zero, so an OOM
    during backward cannot double-count a partially accumulated microbatch.
    Ordinary workloads first run without transformer gradient checkpointing.
    A caller that has identified an exact fused long singleton can start with
    checkpointing and reserved GPU headroom, avoiding known-failing attempts.
    Otherwise an OOM retries with checkpointing before batches are split.
    The learned safe work cap, checkpointing threshold, CPU-offload threshold,
    and reserved-headroom threshold are retained on the persistent model, so
    comparable later batches start directly in the proven execution mode.
    Once batches reach one example, fast training uses reserved card headroom
    before the much slower saved-tensor CPU offload. A truly untrainable
    outlier is quarantined only after every exact fallback has failed.
    """
    import gc
    import torch
    from model_backend import set_gradient_checkpointing

    active = [list(batch) for batch in batches if batch]
    quarantined = []
    profile = getattr(model, "_ttt_oom_execution_profile", None)
    if not isinstance(profile, dict):
        profile = {}
        setattr(model, "_ttt_oom_execution_profile", profile)

    cached_work_cap = profile.get("safe_work_cap")
    reused_split = False
    if cached_work_cap is not None:
        active, reused_split = _split_training_batches_to_work_cap(
            active, cached_work_cap)

    strict_exact_training = bool(
        getattr(model, "_ttt_strict_exact_long_training", False))
    activation_offload = False
    memory_expanded = False
    split_backoff = False
    offload_trigger_work = None
    headroom_trigger_work = None
    checkpoint_threshold = getattr(
        model, "_ttt_gradient_checkpointing_min_work", None)
    checkpointing_available = set_gradient_checkpointing(model, False)
    checkpointing = False

    def select_checkpointing():
        nonlocal checkpointing, checkpointing_available
        work = _gradient_checkpointing_work_units(active)
        wanted = bool(
            checkpointing_available
            and (
                bool(start_checkpointed)
                or (checkpoint_threshold is not None
                    and work >= int(checkpoint_threshold))
            )
        )
        if wanted != checkpointing:
            if not set_gradient_checkpointing(model, wanted):
                checkpointing_available = False
                wanted = False
            checkpointing = wanted
        return work

    active_work = select_checkpointing()
    configured_headroom_min_work = getattr(
        model, "_ttt_reserved_headroom_min_work", None)
    headroom_min_work = profile.get(
        "headroom_min_work", configured_headroom_min_work)
    if ((bool(start_with_headroom)
         or (headroom_min_work is not None
             and active_work >= int(headroom_min_work)))
            and expand_memory is not None):
        expanded_fraction = expand_memory()
        memory_expanded = expanded_fraction is not None
    configured_offload_min_work = getattr(
        model, "_ttt_activation_offload_min_work", None)
    offload_min_work = profile.get(
        "activation_offload_min_work", configured_offload_min_work)
    if (offload_min_work is not None
            and active_work >= int(offload_min_work)
            and hasattr(torch.autograd.graph, "save_on_cpu")):
        activation_offload = True

    reused_modes = []
    if reused_split:
        reused_modes.append(
            f"work cap {int(cached_work_cap)}")
    if checkpointing:
        reused_modes.append("gradient checkpointing")
    if activation_offload:
        reused_modes.append("CPU activation offload")
    if memory_expanded:
        reused_modes.append("reserved GPU headroom")
    if reused_modes:
        signature = tuple(reused_modes)
        if getattr(model, "_ttt_reported_oom_profile", None) != signature:
            mode_source = (
                "starting long-sequence mode"
                if start_checkpointed or start_with_headroom else
                "reusing learned safe mode"
            )
            log_prefix = (
                "[train-memory]"
                if start_checkpointed or start_with_headroom else
                "[train-oom]"
            )
            print(
                f"{log_prefix} {device_label}: {mode_source} "
                f"({', '.join(reused_modes)})",
                flush=True,
            )
            setattr(model, "_ttt_reported_oom_profile", signature)

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
            if split_backoff and active:
                successful_work = _gradient_checkpointing_work_units(active)
                _remember_oom_profile_minimum(
                    profile, "safe_work_cap", successful_work)
            if offload_trigger_work is not None:
                _remember_oom_profile_minimum(
                    profile, "activation_offload_min_work",
                    _oom_profile_work_floor(offload_trigger_work))
            if headroom_trigger_work is not None:
                _remember_oom_profile_minimum(
                    profile, "headroom_min_work",
                    _oom_profile_work_floor(headroom_trigger_work))
            set_gradient_checkpointing(model, False)
            return result, active, quarantined
        except BaseException as error:
            cuda_oom = _is_cuda_oom(error)
            kernel_unavailable = _is_attention_kernel_unavailable(error)
            if not cuda_oom and not kernel_unavailable:
                set_gradient_checkpointing(model, False)
                raise
            model.zero_grad(set_to_none=True)
            gc.collect()
            for cuda_device in cuda_devices:
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()
            if cuda_oom and not checkpointing and checkpointing_available:
                learned_checkpoint_floor = _oom_profile_work_floor(
                    active_work)
                checkpoint_threshold = (
                    learned_checkpoint_floor
                    if checkpoint_threshold is None else
                    min(int(checkpoint_threshold),
                        learned_checkpoint_floor)
                )
                setattr(
                    model, "_ttt_gradient_checkpointing_min_work",
                    int(checkpoint_threshold))
                if set_gradient_checkpointing(model, True):
                    checkpointing = True
                    print(
                        f"[train-oom] {device_label}: retrying the same "
                        "workload with gradient checkpointing",
                        flush=True,
                    )
                    continue
                checkpointing_available = False
            smaller, changed = _split_training_batches(active)
            if changed:
                active = smaller
                split_backoff = True
                active_work = select_checkpointing()
                reason = ("fused attention rejected the padded batch"
                          if kernel_unavailable else "CUDA OOM")
                print(f"[train-oom] {device_label}: {reason}; retrying with maximum "
                      f"microbatch {max(len(batch) for batch in active)}",
                      flush=True)
                continue
            if (cuda_oom and not memory_expanded
                    and expand_memory is not None):
                expanded_fraction = expand_memory()
                if expanded_fraction is not None:
                    memory_expanded = True
                    headroom_trigger_work = active_work
                    print(
                        f"[train-oom] {device_label}: singleton OOM; retrying "
                        "the exact update with "
                        f"reserved GPU headroom (allocator cap "
                        f"{100.0 * float(expanded_fraction):.1f}%)",
                        flush=True,
                    )
                    continue
            if cuda_oom and not activation_offload and hasattr(
                    torch.autograd.graph, "save_on_cpu"):
                activation_offload = True
                offload_trigger_work = active_work
                setting_log_only(
                    f"[train-oom] {device_label}: singleton still OOM after "
                    "GPU headroom; retrying with saved activations offloaded "
                    "to CPU", flush=True)
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
            if strict_exact_training:
                set_gradient_checkpointing(model, False)
                model.zero_grad(set_to_none=True)
                failure = (
                    "no compatible exact linear-memory attention kernel"
                    if kernel_unavailable else
                    "insufficient memory after checkpointing, reserved "
                    "headroom, and CPU activation offload"
                )
                raise RuntimeError(
                    f"exact long-context training failed for a "
                    f"{victim_tokens}-token rollout on {device_label}: "
                    f"{failure}. The configured coder profile forbids "
                    "truncating or silently excluding this trajectory.") from error
            quarantined.append(victim)
            # Once the reserved-headroom path was required, keep activations
            # offloaded while checking the remaining long singletons. This
            # avoids repeating one guaranteed-to-fail ordinary attempt for
            # every neighboring rollout.
            activation_offload = memory_expanded
            if kernel_unavailable:
                reason = "has no compatible linear-memory attention kernel"
            else:
                reason = "cannot fit even with activation offload"
            print(f"[train-oom] {device_label}: one {victim_tokens}-token "
                  f"rollout {reason}; excluding only that rollout from this "
                  "adapter update", flush=True)
            active_work = select_checkpointing()
    model.zero_grad(set_to_none=True)
    set_gradient_checkpointing(model, False)
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
    return missing_old, missing_reference


def rank_grpo_loss(current_logprobs, old_logprobs, reference_logprob,
                   advantage, *, clip_epsilon=RANK_CLIP_EPSILON_DEFAULT,
                   clip_epsilon_low=None, clip_epsilon_high=None,
                   kl_coef=0.0, entropy_coef=0.0, token_entropies=None,
                   sequence_level_ratio=False,
                   return_tensor_metrics=False):
    """One trajectory's loss; callers average equally within each parent.

    Normally, per-token rho = exp(current_lp - old_lp) is differentiable. For
    the plain Qwen3-30B-A3B MoE coder, callers select the GSPO-style trajectory
    ratio exp(mean(current_lp - old_lp)); the same scalar ratio and asymmetric
    bounds are then used for the complete response. Old log-probabilities and
    the scalar group advantage are frozen. Policy/reference regularization uses
    the same centered log-probability correction as entropic mode:
        d = log pi_theta - log pi_ref
        A_KL = kl_coef * (mean(d) - d).
    Its policy-gradient surrogate is kept separate from the rank PPO surrogate
    so rank advantages and asymmetric clipping retain their existing behavior.
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
        raise ValueError("policy advantage must be a finite scalar")
    if not torch.isfinite(cur).all() or not torch.isfinite(old).all():
        raise ValueError("rank policy logprobs must be finite")

    log_ratios = cur - old
    if sequence_level_ratio:
        policy_ratios = log_ratios.mean().exp()
    else:
        policy_ratios = log_ratios.exp()
    clipped_policy_ratios = policy_ratios.clamp(
        1.0 - epsilon_low, 1.0 + epsilon_high)
    policy_loss = -torch.minimum(
        policy_ratios * adv, clipped_policy_ratios * adv).mean()
    ratio = policy_ratios.mean()
    clipped_fraction = (
        policy_ratios.detach() != clipped_policy_ratios.detach()
    ).double().mean()
    policy_reference_delta = cur.new_zeros(())
    kl_policy_loss = cur.new_zeros(())
    if kl_coef:
        if reference_logprob is None:
            raise ValueError("reference logprobs are required for nonzero KL coefficient")
        ref = torch.as_tensor(reference_logprob, dtype=cur.dtype,
                              device=cur.device).detach()
        if ref.shape != cur.shape or not torch.isfinite(ref).all():
            raise ValueError(
                "reference logprobs must be a finite vector matching current logprobs")
        logp_difference = (cur - ref).detach()
        policy_reference_delta = logp_difference.mean()
        kl_advantage = kl_coef * (
            policy_reference_delta - (cur - ref)
        )
        # This is the KL component of entropic mode's policy-gradient loss.
        # Keep the current/reference centering unchanged. In sequence-ratio
        # mode, one detached trajectory weight replaces volatile per-token MoE
        # weights without changing how pi_theta and pi_ref are compared.
        kl_policy_loss = -(
            policy_ratios.detach() * kl_advantage.detach() * cur
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

    loss = policy_loss + kl_policy_loss - entropy_coef * entropy_estimate
    if not torch.isfinite(loss).all() or not torch.isfinite(ratio).all():
        raise FloatingPointError(
            "nonfinite clipped objective/trajectory ratio; reduce learning "
            "rate or update epochs and inspect model logprobs")
    tensor_metrics = {
        "policy_loss": policy_loss.detach(),
        # Retain the metric key for checkpoint/log compatibility. Its value now
        # matches entropic mode's mean log pi_theta - log pi_base diagnostic.
        "kl_estimate": policy_reference_delta.detach(),
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


def _rank_entropy_coefficient(cfg):
    if getattr(cfg, "advantage_mode", "entropic") == "binary-coder":
        return 0.0
    if getattr(cfg, "advantage_mode", "entropic") == "spo-rs":
        return 0.0
    if getattr(cfg, "advantage_mode", "entropic") == "x-grpo":
        return float(getattr(
            cfg, "x_grpo_entropy_coef", X_GRPO_ENTROPY_COEF_DEFAULT))
    return float(getattr(
        cfg, "rank_entropy_coef", RANK_ENTROPY_COEF_DEFAULT))


def _clipped_policy_options(cfg):
    """Return PPO epsilon, lower epsilon, upper epsilon, and reference-KL coef."""
    if getattr(cfg, "advantage_mode", "entropic") == "binary-coder":
        low = float(cfg.binary_coder_clip_epsilon_low)
        high = float(cfg.binary_coder_clip_epsilon_high)
        return low, low, high, 0.0
    if getattr(cfg, "advantage_mode", "entropic") == "spo-rs":
        epsilon = float(getattr(
            cfg, "spo_rs_clip_epsilon", SPO_RS_CLIP_EPSILON_DEFAULT))
        epsilon_low = float(getattr(
            cfg, "spo_rs_clip_epsilon_low",
            SPO_RS_CLIP_EPSILON_LOW_DEFAULT))
        epsilon_high = float(getattr(
            cfg, "spo_rs_clip_epsilon_high",
            SPO_RS_CLIP_EPSILON_HIGH_DEFAULT))
        return epsilon, epsilon_low, epsilon_high, 0.0
    epsilon = float(getattr(
        cfg, "rank_clip_epsilon", RANK_CLIP_EPSILON_DEFAULT))
    epsilon_low = float(getattr(cfg, "rank_clip_epsilon_low", epsilon))
    epsilon_high = float(getattr(cfg, "rank_clip_epsilon_high", epsilon))
    return epsilon, epsilon_low, epsilon_high, float(cfg.kl_penalty_coef)


def _clipped_reference_count(cfg, supplied, total):
    if getattr(cfg, "advantage_mode", "entropic") in (
            "spo-rs", "binary-coder"):
        return "off"
    return f"{int(supplied)}/{int(total)}"


def _clipped_reference_metric(cfg, value):
    if getattr(cfg, "advantage_mode", "entropic") in (
            "spo-rs", "binary-coder"):
        return "reference=off"
    return f"avg logpi_theta-logpi_base={float(value):.6f}"


def clipped_policy_loss(cfg, current_logprobs, old_logprobs,
                        reference_logprob, advantage, *,
                        clip_epsilon, clip_epsilon_low,
                        clip_epsilon_high, kl_coef=0.0,
                        entropy_coef=0.0, token_entropies=None,
                        return_tensor_metrics=False):
    """Dispatch the shared fast trainer to the active clipped objective."""
    if getattr(cfg, "advantage_mode", "entropic") == "binary-coder":
        from problems.binary_coder import binary_coder_clipped_loss
        return binary_coder_clipped_loss(
            current_logprobs, old_logprobs, advantage,
            clip_epsilon_low=clip_epsilon_low,
            clip_epsilon_high=clip_epsilon_high,
            sequence_level_ratio=_uses_sequence_level_policy_ratio(cfg),
            return_tensor_metrics=return_tensor_metrics)
    return rank_grpo_loss(
        current_logprobs, old_logprobs, reference_logprob, advantage,
        clip_epsilon=clip_epsilon,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
        token_entropies=token_entropies,
        sequence_level_ratio=_uses_sequence_level_policy_ratio(cfg),
        return_tensor_metrics=return_tensor_metrics)


def _a3b_sequence_clipped_standard_loss(
        cfg, current_logprobs, behavior_logprobs, reference_logprobs,
        advantage):
    """Apply the A3B sequence surrogate inside standard advantage modes.

    Entropic, GRPO, and CVaR retain their existing advantages, centered
    current/reference correction and sample normalization. Only the
    reward-policy importance ratio and clipping unit become one
    length-normalized response ratio. The helper is deliberately unavailable
    to every other coder model.
    """
    if not _uses_sequence_level_policy_ratio(cfg):
        raise ValueError(
            "the sequence-clipped standard loss is Qwen3-30B-A3B only")
    epsilon, epsilon_low, epsilon_high, kl_coef = (
        _clipped_policy_options(cfg))
    old = (current_logprobs.detach()
           if behavior_logprobs is None else behavior_logprobs)
    loss, metrics = clipped_policy_loss(
        cfg, current_logprobs, old, reference_logprobs, advantage,
        clip_epsilon=epsilon,
        clip_epsilon_low=epsilon_low,
        clip_epsilon_high=epsilon_high,
        kl_coef=kl_coef,
        return_tensor_metrics=True)
    return loss, metrics


def _x_grpo_batched_autograd_unavailable(error):
    """Whether PyTorch cannot vectorize per-example VJPs on this model."""
    message = str(error).lower()
    markers = (
        "is_grads_batched",
        "batched grad",
        "batching rule",
        "batchingrule",
        "inside of vmap",
        "functorch",
        "vmap",
        "vmap-incompatible",
        "vmap incompatible",
    )
    return any(marker in message for marker in markers)


def _x_grpo_scalar_parameter_gradients(scores, parameters, *, vectorized):
    """Differentiate scalar score aggregates through one shared graph."""
    import torch

    if scores.ndim != 1 or scores.numel() < 1:
        raise ValueError("X-GRPO rollout scores must be a nonempty vector")
    count = int(scores.numel())
    if count == 1:
        gradients = torch.autograd.grad(
            scores[0], parameters, allow_unused=True)
        return tuple(
            None if gradient is None else gradient.unsqueeze(0)
            for gradient in gradients)

    if vectorized:
        basis = torch.eye(count, dtype=scores.dtype, device=scores.device)
        return torch.autograd.grad(
            scores, parameters, grad_outputs=basis,
            is_grads_batched=True, allow_unused=True)

    rows = [[] for _ in parameters]
    for row_index in range(count):
        gradients = torch.autograd.grad(
            scores[row_index], parameters, allow_unused=True,
            retain_graph=row_index + 1 < count)
        for parameter_index, (parameter, gradient) in enumerate(
                zip(parameters, gradients)):
            rows[parameter_index].append(
                torch.zeros_like(parameter) if gradient is None else gradient)
    return tuple(torch.stack(parameter_rows, dim=0)
                 for parameter_rows in rows)


def _x_grpo_cache_group_gradients(model, tokenizer, examples, cfg,
                                  budget_indices, *, group_id, cache_dir,
                                  device_label, batches=None,
                                  parameters=None):
    """Cache budget gradients from shared advantage-stratum derivatives."""
    import torch

    group_examples = list(examples)
    indices = tuple(int(index) for index in budget_indices)
    if not indices:
        raise ValueError("X-GRPO diagnostic budgets must not be empty")
    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("X-GRPO diagnostic budget indices must be unique and nonnegative")
    parameters = (
        tuple(parameter for parameter in model.parameters()
              if parameter.requires_grad)
        if parameters is None else tuple(parameters))
    flat_count = sum(parameter.numel() for parameter in parameters)
    if flat_count < 1:
        raise RuntimeError("X-GRPO found no trainable parameters")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    empty_cache = {index: None for index in indices}
    if not group_examples:
        model.zero_grad(set_to_none=True)
        return empty_cache, 0, []
    group_sizes = {int(example["x_grpo_group_size"])
                   for example in group_examples}
    if len(group_sizes) != 1:
        raise ValueError("X-GRPO examples disagree about their sampled group size")
    sampled_group_size = group_sizes.pop()
    if sampled_group_size < 2:
        raise ValueError("X-GRPO sampled group size must be at least two")

    advantage_vectors = {}
    representative_for = {}
    representatives = []
    for index in indices:
        vector = tuple(
            float(example["x_grpo_trial_advantages"][index])
            for example in group_examples)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("X-GRPO diagnostic advantages must be finite")
        if not any(value != 0.0 for value in vector):
            representative_for[index] = None
            continue
        representative = advantage_vectors.get(vector)
        if representative is None:
            representative = index
            advantage_vectors[vector] = index
            representatives.append(index)
        representative_for[index] = representative

    if not representatives:
        model.zero_grad(set_to_none=True)
        planned = (_training_microbatches(group_examples, cfg)
                   if batches is None else [list(batch) for batch in batches])
        return empty_cache, 0, planned

    planned_batches = (
        _training_microbatches(group_examples, cfg)
        if batches is None else [list(batch) for batch in batches])
    representative_rows = {
        index: row for row, index in enumerate(representatives)}
    vectorized = True

    def attempt(active_batches):
        accumulator_count = len(representatives)
        accumulators = [
            torch.zeros(
                (accumulator_count, *parameter.shape),
                dtype=parameter.dtype, device=parameter.device)
            for parameter in parameters
        ]
        for batch in active_batches:
            current_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=True, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)
            scores = torch.stack(
                [current_lp.mean() for current_lp in current_logprobs])
            strata = {}
            for row, example in enumerate(batch):
                signature = tuple(
                    float(example["x_grpo_trial_advantages"][index])
                    for index in representatives)
                if any(value != 0.0 for value in signature):
                    strata.setdefault(signature, []).append(row)
            if not strata:
                model.zero_grad(set_to_none=True)
                del current_logprobs, scores
                continue

            signatures = tuple(strata)
            stratum_scores = torch.stack([
                scores[rows].sum() for rows in strata.values()
            ])
            stratum_gradients = _x_grpo_scalar_parameter_gradients(
                stratum_scores, parameters, vectorized=vectorized)
            weights = scores.new_tensor(signatures).transpose(0, 1)
            weights.div_(sampled_group_size)
            with torch.no_grad():
                weights_by_device_dtype = {}
                for accumulator, gradient in zip(
                        accumulators, stratum_gradients):
                    if gradient is None:
                        continue
                    weight_key = (gradient.device, gradient.dtype)
                    dtype_weights = weights_by_device_dtype.get(weight_key)
                    if dtype_weights is None:
                        dtype_weights = weights.to(
                            device=gradient.device, dtype=gradient.dtype)
                        weights_by_device_dtype[weight_key] = dtype_weights
                    accumulator.reshape(accumulator_count, -1).addmm_(
                        dtype_weights, gradient.detach().reshape(
                            len(signatures), -1))
            model.zero_grad(set_to_none=True)
            del accumulator, gradient
            del current_logprobs, scores, stratum_scores, stratum_gradients
            del signatures, strata, weights
            del weights_by_device_dtype

        cached = {}
        for index in representatives:
            row = representative_rows[index]
            flat = torch.cat([
                accumulator[row].detach().reshape(-1).float().cpu()
                for accumulator in accumulators
            ])
            if flat.numel() != flat_count or not torch.isfinite(flat).all():
                raise FloatingPointError(
                    "nonfinite X-GRPO diagnostic gradient")
            cache_path = cache_root / (
                f"group-{int(group_id)}-budget-{int(index)}.pt")
            torch.save(flat, cache_path)
            cached[index] = str(cache_path)
            del flat
        return cached

    retry_without_vectorized_autograd = False
    try:
        result, effective, quarantined = _run_oom_resilient_backward(
            model, planned_batches, attempt, device_label=device_label)
    except (RuntimeError, TypeError) as error:
        if not vectorized or not _x_grpo_batched_autograd_unavailable(error):
            raise
        vectorized = False
        retry_without_vectorized_autograd = True
    if retry_without_vectorized_autograd:
        model.zero_grad(set_to_none=True)
        print(f"[X-GRPO] {device_label}: batched diagnostic autograd is "
              "unavailable; retrying with one backward per distinct "
              "advantage stratum within each shared forward", flush=True)
        result, effective, quarantined = _run_oom_resilient_backward(
            model, planned_batches, attempt, device_label=device_label)
    model.zero_grad(set_to_none=True)
    if result is None:
        result = {index: None for index in representatives}
    cached = {}
    for index in indices:
        representative = representative_for[index]
        cached[index] = (None if representative is None
                         else result[representative])
    return cached, len(quarantined), effective


def _x_grpo_load_cached_gradient(cache_path, zero_gradient):
    """Load one self-generated aggregate and validate its exact layout."""
    import torch

    if cache_path is None:
        return zero_gradient
    try:
        gradient = torch.load(
            cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        gradient = torch.load(cache_path, map_location="cpu")
    if (not torch.is_tensor(gradient)
            or gradient.device.type != "cpu"
            or gradient.dtype != torch.float32
            or gradient.ndim != 1
            or gradient.numel() != zero_gradient.numel()
            or not torch.isfinite(gradient).all()):
        raise RuntimeError("invalid cached X-GRPO diagnostic gradient")
    return gradient


def _x_grpo_delete_consumed_cache(cached_by_group, budget_index):
    """Unlink aggregates once no later budget aliases the same cache file."""
    current_paths = {
        cache[budget_index]
        for cache in cached_by_group.values()
        if cache.get(budget_index) is not None
    }
    future_paths = {
        cache[index]
        for cache in cached_by_group.values()
        for index in cache
        if index > budget_index and cache[index] is not None
    }
    for cache_path in current_paths - future_paths:
        Path(cache_path).unlink(missing_ok=True)


def _x_grpo_crossfit_metrics(gradients, relative_error):
    """Return leave-one-group-out precision diagnostics for one trial budget."""
    import torch

    group_ids = sorted(gradients)
    group_count = len(group_ids)
    if group_count < 3:
        raise ValueError("X-GRPO calibration requires at least three groups")
    total = gradients[group_ids[0]].clone()
    gradient_squared_norms = {
        group_id: float(torch.dot(gradient, gradient).item())
        for group_id, gradient in gradients.items()
    }
    total_squared_norms = gradient_squared_norms[group_ids[0]]
    for group_id in group_ids[1:]:
        gradient = gradients[group_id]
        total.add_(gradient)
        total_squared_norms += gradient_squared_norms[group_id]
    total_norm_squared = float(torch.dot(total, total).item())

    heldout_count = group_count - 1
    denominator = heldout_count * (heldout_count - 1)
    results = {}
    for group_id in group_ids:
        gradient = gradients[group_id]
        gradient_norm_squared = gradient_squared_norms[group_id]
        heldout_sum = None
        if gradient_norm_squared == 0.0:
            heldout_sum_norm_squared = total_norm_squared
        else:
            heldout_sum = total - gradient
            heldout_sum_norm_squared = float(
                torch.dot(heldout_sum, heldout_sum).item())
        heldout_squared_norms = max(
            0.0, total_squared_norms - gradient_norm_squared)
        centered_sum = max(
            0.0,
            heldout_squared_norms
            - heldout_sum_norm_squared / heldout_count,
        )
        signal_squared = heldout_sum_norm_squared / (heldout_count ** 2)
        error_squared = centered_sum / denominator
        accepted = bool(
            signal_squared > 0.0
            and error_squared
            <= float(relative_error) ** 2 * signal_squared)
        signal_norm = math.sqrt(signal_squared)
        standard_error = math.sqrt(error_squared)
        results[group_id] = {
            "gradient_norm": signal_norm,
            "standard_error": standard_error,
            "relative_error": (
                standard_error / signal_norm if signal_norm > 0.0
                else None),
            "accepted": accepted,
        }
        if heldout_sum is not None:
            del heldout_sum
    return results


def _x_grpo_context_group_map(examples, *, context_group_ids=None,
                              group_ids=None):
    if context_group_ids is not None and group_ids is not None:
        raise ValueError(
            "pass X-GRPO context groups or flat group ids, not both")

    examples = list(examples)
    if context_group_ids is None:
        allowed = (
            {int(group_id) for group_id in group_ids}
            if group_ids is not None else
            {int(example["group_id"]) for example in examples}
        )
        inferred = {}
        for example in examples:
            group_id = int(example["group_id"])
            if group_id not in allowed:
                raise ValueError(
                    f"unexpected X-GRPO diagnostic group {group_id}")
            context_id = int(example.get("x_grpo_context_id", 0))
            inferred.setdefault(context_id, set()).add(group_id)
        missing = allowed - {
            group_id for ids in inferred.values() for group_id in ids
        }
        if missing:
            if len(inferred) > 1:
                raise ValueError(
                    "cannot infer the context of empty X-GRPO groups; pass "
                    "context_group_ids")
            context_id = next(iter(inferred), 0)
            inferred.setdefault(context_id, set()).update(missing)
        context_group_ids = inferred

    normalized = {}
    owners = {}
    for context_id, ids in context_group_ids.items():
        context_id = int(context_id)
        ordered = tuple(sorted({int(group_id) for group_id in ids}))
        if len(ordered) < 3:
            raise ValueError(
                f"X-GRPO context {context_id} requires at least three groups")
        for group_id in ordered:
            previous = owners.get(group_id)
            if previous is not None:
                raise ValueError(
                    f"X-GRPO group {group_id} belongs to contexts "
                    f"{previous} and {context_id}")
            owners[group_id] = context_id
        normalized[context_id] = ordered
    if not normalized:
        raise ValueError("X-GRPO calibration requires at least one context")
    return dict(sorted(normalized.items()))


def _calibrate_x_grpo_local(backend, model, tokenizer, examples, cfg, step_idx,
                            *, context_group_ids=None, group_ids=None):
    """Serial calibration path for a single/model-parallel training model."""
    from tempfile import TemporaryDirectory
    import torch

    examples = list(examples)
    contexts = _x_grpo_context_group_map(
        examples, context_group_ids=context_group_ids, group_ids=group_ids)
    expected_group_ids = sorted(
        group_id for ids in contexts.values() for group_id in ids
    )
    groups = {group_id: [] for group_id in expected_group_ids}
    context_for_group = {
        group_id: context_id
        for context_id, ids in contexts.items()
        for group_id in ids
    }
    for example in examples:
        group_id = int(example["group_id"])
        if group_id not in groups:
            raise ValueError(
                f"unexpected X-GRPO diagnostic group {group_id}")
        example_context = int(example.get(
            "x_grpo_context_id", context_for_group[group_id]))
        if example_context != context_for_group[group_id]:
            raise ValueError(
                f"X-GRPO group {group_id} was assigned to context "
                f"{context_for_group[group_id]} but contains context "
                f"{example_context}")
        groups[group_id].append(example)

    budgets = tuple(float(value) for value in cfg.x_grpo_budgets)
    selected = {group_id: 0.0 for group_id in groups}
    diagnostics = {group_id: [] for group_id in groups}
    quarantined_total = 0
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    flat_count = sum(parameter.numel() for parameter in parameters)
    if flat_count < 1:
        raise RuntimeError("X-GRPO found no trainable parameters")
    zero_gradient = torch.zeros(
        flat_count, dtype=torch.float32, device="cpu")
    batch_plans = {
        group_id: _training_microbatches(group_examples, cfg)
        for group_id, group_examples in groups.items()
    }
    budget_indices = tuple(range(len(budgets)))
    backend.set_training_mode()
    print(f"[step {step_idx}] X-GRPO calibration: {len(contexts)} contexts, "
          f"{len(groups)} groups x {len(budgets)} budgets on the primary "
          "trainer; identical advantage strata share one aggregate backward",
          flush=True)
    try:
        with TemporaryDirectory(prefix="x-grpo-gradients-") as cache_dir:
            cached_by_group = {}
            with _rank_dropout_disabled(model):
                for group_id in sorted(groups):
                    cached, quarantined, effective = (
                        _x_grpo_cache_group_gradients(
                            model, tokenizer, groups[group_id], cfg,
                            budget_indices, group_id=group_id,
                            cache_dir=cache_dir,
                            device_label=str(model.device),
                            batches=batch_plans[group_id],
                            parameters=parameters))
                    batch_plans[group_id] = effective
                    quarantined_total += quarantined
                    cached_by_group[group_id] = cached

            for context_id, context_ids in contexts.items():
                context_caches = {
                    group_id: cached_by_group[group_id]
                    for group_id in context_ids
                }
                for budget_index, budget in enumerate(budgets):
                    gradients = {
                        group_id: _x_grpo_load_cached_gradient(
                            cached_by_group[group_id][budget_index],
                            zero_gradient)
                        for group_id in context_ids
                    }
                    trial = _x_grpo_crossfit_metrics(
                        gradients, cfg.x_grpo_relative_error)
                    for group_id, metrics in trial.items():
                        diagnostics[group_id].append({
                            "budget": budget, **metrics})
                        if metrics["accepted"]:
                            selected[group_id] = max(
                                selected[group_id], budget)
                    accepted_count = sum(
                        item["accepted"] for item in trial.values())
                    print(f"[step {step_idx}] X-GRPO context {context_id} "
                          f"budget {budget:g}: accepted for "
                          f"{accepted_count}/{len(context_ids)} held-out "
                          "groups", flush=True)
                    del gradients
                    _x_grpo_delete_consumed_cache(
                        context_caches, budget_index)
    finally:
        model.zero_grad(set_to_none=True)
    return {
        "selected_budgets": selected,
        "groups": diagnostics,
        "diagnostic_quarantined_examples": quarantined_total,
        "distributed": False,
    }


def _apply_x_grpo_calibration(examples, group_data, calibration, cfg):
    from advantage import x_grpo_advantages

    budgets = tuple(float(value) for value in cfg.x_grpo_budgets)
    budget_indices = {value: index for index, value in enumerate(budgets)}
    selected = {
        int(group_id): float(value)
        for group_id, value in calibration["selected_budgets"].items()
    }
    expected_groups = set(group_data)
    if set(selected) != expected_groups:
        raise RuntimeError(
            "X-GRPO calibration did not return every sampled group: "
            f"expected {sorted(expected_groups)}, got {sorted(selected)}")

    final_by_group = {}
    summaries = []
    for group_id in sorted(group_data):
        data = group_data[group_id]
        selected_budget = selected[group_id]
        if selected_budget == 0.0:
            final_advantages, _cutoff, final_info = x_grpo_advantages(
                data["rewards"], 0.0, return_info=True)
        else:
            budget_index = budget_indices[selected_budget]
            final_advantages = data["trial_advantages"][budget_index]
            final_info = data["trial_info"][budget_index]
        entropy_gate = bool(
            (selected_budget == 0.0 or final_info["all_tied"])
            and data["entropy_eligible"])
        final_by_group[group_id] = (
            final_advantages, entropy_gate, final_info)
        serializable_info = {
            key: value for key, value in final_info.items()
            if key not in ("ranks", "weights")
        }
        summaries.append({
            "group": group_id,
            "context": int(data["context"]),
            "fold": int(data["fold"]),
            "selected_budget": selected_budget,
            "entropy_gate": entropy_gate,
            **serializable_info,
            "calibration": calibration["groups"][group_id],
        })

    for example in examples:
        group_id = int(example["group_id"])
        rollout_index = int(example["rollout_index"])
        advantages, entropy_gate, _info = final_by_group[group_id]
        example["advantage"] = float(advantages[rollout_index])
        example["rank_entropy_gate"] = entropy_gate

    return summaries


def _prepare_binary_overlap_examples(
        model, tokenizer, group_responses, reward_futures, prompt_jobs,
        parents):
    """Build the binary-coder tensors before the CPU rewards are available.

    Generation already supplied the frozen behavior log-probabilities.  The
    prompt/response tensors and the denominator of the step-wide mean therefore
    do not depend on reward.  Each prepared example retains only its aligned
    reward future; the future is resolved *after* that microbatch's policy
    forward, keeping at most one autograd graph alive while CPU verification
    continues.
    """
    import torch

    prompt_ids_by_job = {}
    prepared = []
    submission_index = 0
    for group_id in sorted(group_responses):
        responses = group_responses[group_id]
        futures = reward_futures[group_id]
        if len(responses) != len(futures):
            raise RuntimeError(
                f"binary overlap group {group_id} has {len(responses)} "
                f"responses but {len(futures)} reward futures")
        for rollout_index, (record, reward_future) in enumerate(
                zip(responses, futures)):
            token_ids = list(record.get("token_ids") or ())
            behavior_values = record.get("behavior_logprobs")
            if (not token_ids or behavior_values is None
                    or len(behavior_values) != len(token_ids)):
                submission_index += 1
                continue
            job_idx = int(record["job_idx"])
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.to(model.device)
            response_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=model.device)
            behavior_logprobs = torch.as_tensor(
                behavior_values, dtype=torch.float32,
                device=model.device)
            if not torch.isfinite(behavior_logprobs).all():
                submission_index += 1
                continue
            parent_group = int(record["parent_group"])
            example = {
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": response_ids,
                # Filled from the verified RewardResult after the forward.
                "_entropy_measurement": record.get("_entropy_measurement"),
                "advantage": None,
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": None,
                "reward_constant": False,
                "rank_entropy_gate": False,
                "group_id": int(group_id),
                "x_grpo_context_id": int(record["context_id"]),
                "x_grpo_fold_index": int(record["fold_index"]),
                "rollout_index": int(rollout_index),
                "prompt_job_id": job_idx,
                "x_grpo_group_size": len(responses),
                "x_grpo_all_tied": False,
                "x_grpo_trial_advantages": (),
                "sample_weight": 0.0,
                "_overlap_key": (int(group_id), int(rollout_index)),
                "_overlap_submission_index": int(submission_index),
                "_overlap_reward_future": reward_future,
                "_overlap_parent": parents[parent_group],
            }
            # Reuse these exact tensors in the later save/accounting pass.  This
            # avoids tokenizing and allocating every long prompt a second time.
            record["_binary_overlap_example"] = example
            prepared.append(example)
            submission_index += 1

    if prepared:
        weight = 1.0 / len(prepared)
        for example in prepared:
            example["sample_weight"] = weight
    return prepared


def _prepare_entropic_overlap_group(
        tokenizer, group_id, responses, reward_futures, prompt_jobs,
        prompt_ids_by_job, num_groups):
    """Materialize one reward-complete entropic group on CPU.

    Entropic advantages depend on every reward in their group, but they do not
    depend on rewards from any other group.  As soon as one group is complete,
    this builds the exact examples used by the ordinary post-evaluation path so
    process-distributed forward/backward can begin while other groups remain in
    the CPU evaluator.  Parameters are not updated here.
    """
    import torch

    if len(responses) != len(reward_futures):
        raise RuntimeError(
            f"entropic overlap group {group_id} has {len(responses)} "
            f"responses but {len(reward_futures)} reward futures")
    if any(not future.done() for future in reward_futures):
        raise RuntimeError(
            f"entropic overlap group {group_id} was prepared before all of "
            "its rewards completed")

    rewards = np.asarray(
        [float(future.result().reward) for future in reward_futures],
        dtype=np.float64,
    )
    if rewards.size == 0:
        return []
    advantages, _beta, _label, _info = compute_group_advantages(
        rewards, "entropic", return_info=True)
    constant = float(rewards.max() - rewards.min()) < 1e-12
    if constant:
        return []

    prepared = []
    for rollout_index, (record, advantage) in enumerate(
            zip(responses, advantages)):
        token_ids = list(record.get("token_ids") or ())
        if not token_ids:
            continue
        job_idx = int(record["job_idx"])
        if job_idx not in prompt_ids_by_job:
            prompt_ids_by_job[job_idx] = tokenizer(
                prompt_jobs[job_idx]["prompt_text"],
                return_tensors="pt").input_ids.cpu()

        behavior_values = record.get("behavior_logprobs")
        reference_values = record.get("reference_logprobs")
        behavior_logprobs = (
            torch.as_tensor(behavior_values, dtype=torch.float32).cpu()
            if behavior_values is not None
            and len(behavior_values) == len(token_ids) else None)
        reference_logprobs = (
            torch.as_tensor(reference_values, dtype=torch.float32).cpu()
            if reference_values is not None
            and len(reference_values) == len(token_ids) else None)
        example = {
            "prompt_ids": prompt_ids_by_job[job_idx],
            "response_ids": torch.tensor(
                [token_ids], dtype=torch.long, device="cpu"),
            "_entropy_measurement": record.get("_entropy_measurement"),
            "advantage": float(advantage),
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": reference_logprobs,
            "reward_constant": False,
            "rank_entropy_gate": False,
            "group_id": int(group_id),
            "x_grpo_context_id": int(record["context_id"]),
            "x_grpo_fold_index": int(record["fold_index"]),
            "rollout_index": int(rollout_index),
            "prompt_job_id": job_idx,
            "x_grpo_group_size": len(responses),
            "x_grpo_all_tied": False,
            "x_grpo_trial_advantages": (),
            "sample_weight": 1.0 / (int(num_groups) * len(responses)),
            "_overlap_key": (int(group_id), int(rollout_index)),
        }
        record["_entropic_overlap_example"] = example
        prepared.append(example)
    return prepared


def _precompute_binary_coder_backward(
        backend, model, tokenizer, optimizer, examples, cfg, step_idx):
    """Run the exact one-epoch binary backward while rewards finish on CPU.

    No parameter is updated here.  Every forward sees the same frozen policy as
    the ordinary post-evaluation implementation.  A microbatch's reward futures
    are read only after its differentiable forward has been launched; its fixed
    +/-1 labels are then applied and the loss is backpropagated immediately.
    The caller performs the single optimizer step only after all evaluations and
    rollout artifacts have completed.
    """
    import torch
    from problems.binary_coder import verified_usability

    if str(getattr(cfg, "advantage_mode", "")) != "binary-coder":
        raise ValueError("evaluation-overlapped backward is binary-coder only")
    epochs = int(getattr(cfg, "rank_update_epochs", RANK_UPDATE_EPOCHS_DEFAULT))
    if epochs != 1:
        raise ValueError(
            "evaluation-overlapped binary training requires exactly one "
            "clipped-policy update epoch")
    if not examples:
        return None

    backend.set_training_mode()
    started = time.time()
    update_batches = _training_microbatches(examples, cfg)
    largest_batch = max((len(batch) for batch in update_batches), default=0)
    print(f"[train-overlap] binary coder LoRA microbatches="
          f"{len(update_batches)}; configured max="
          f"{int(cfg.train_examples_per_microbatch)}, effective max="
          f"{largest_batch}, padded-token cap={int(cfg.max_seq_length)}",
          flush=True)

    missing_old, _missing_reference = _initialize_rank_logprob_caches(examples)
    if missing_old:
        raise RuntimeError(
            "binary overlap received examples without complete frozen "
            "behavior log-probabilities")
    print(f"[step {step_idx}] binary-coder logprobs: vLLM supplied old="
          f"{len(examples)}/{len(examples)}, reference=off; starting policy "
          "forward/backward while CPU evaluation is active", flush=True)

    epsilon, epsilon_low, epsilon_high, kl_coef = (
        _clipped_policy_options(cfg))
    if kl_coef != 0.0:
        raise RuntimeError("binary coder overlap unexpectedly enabled KL")
    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    reward_wait_seconds = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    def attempt(active_batches):
        nonlocal reward_wait_seconds
        totals = {key: 0.0 for key in metric_keys}
        max_ratio = 0.0
        max_prefix_ratio = 0.0
        entropy_examples = 0
        pending_batches = list(active_batches)
        while pending_batches:
            # Re-evaluate readiness after every backward.  Evaluation continues
            # concurrently, so a batch that was pending when this attempt began
            # may be ready now.  If none is ready, choose the batch whose latest
            # submitted rollout is earliest in the executor's FIFO queue.
            ready_indices = [
                index for index, candidate in enumerate(pending_batches)
                if all(ex["_overlap_reward_future"].done()
                       for ex in candidate)
            ]
            if ready_indices:
                batch_index = ready_indices[0]
            else:
                batch_index = min(
                    range(len(pending_batches)),
                    key=lambda index: max(
                        ex["_overlap_submission_index"]
                        for ex in pending_batches[index]),
                )
            batch = pending_batches.pop(batch_index)
            # This is the expensive reward-independent part.  CUDA kernels can
            # continue executing while the host waits below for this batch's
            # CPU verifier results.
            current_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=True, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)
            weighted_losses = []
            for example, current_lp in zip(batch, current_logprobs):
                wait_started = time.time()
                result = example["_overlap_reward_future"].result()
                reward_wait_seconds += time.time() - wait_started
                usable, _reason = verified_usability(
                    result, example["_overlap_parent"],
                    fail_score=cfg.fail_score)
                advantage = 1.0 if usable else -1.0
                example["advantage"] = advantage
                loss, metrics = clipped_policy_loss(
                    cfg, current_lp, example["rank_old_logprobs"], None,
                    advantage, clip_epsilon=epsilon,
                    clip_epsilon_low=epsilon_low,
                    clip_epsilon_high=epsilon_high, kl_coef=0.0,
                    entropy_coef=0.0, token_entropies=None)
                weight = float(example["sample_weight"])
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        "nonfinite evaluation-overlapped binary coder loss")
                weighted_losses.append(weight * loss)
                totals["loss"] += weight * float(loss.detach().item())
                for key in ("policy_loss", "kl_estimate",
                            "entropy_estimate", "ratio"):
                    totals[key] += weight * metrics[key]
                totals["clipped_fraction"] += (
                    weight * float(metrics["clipped"]))
                max_ratio = max(max_ratio, metrics["ratio"])
                max_prefix_ratio = max(
                    max_prefix_ratio, metrics["prefix_ratio_max"])
            if weighted_losses:
                sum(weighted_losses[1:], weighted_losses[0]).backward()
        return totals, max_ratio, max_prefix_ratio, entropy_examples

    with _rank_dropout_disabled(model):
        result, effective_batches, quarantined = (
            _run_oom_resilient_backward(
                model, update_batches, attempt,
                device_label=str(model.device)))
    if result is None:
        result = (
            {key: 0.0 for key in metric_keys}, 0.0, 0.0, 0)
    totals, max_ratio, max_prefix_ratio, entropy_examples = result
    trained_keys = {
        example["_overlap_key"]
        for batch in effective_batches for example in batch
    }
    scheduled_keys = {example["_overlap_key"] for example in examples}
    wall_seconds = time.time() - started
    return {
        "examples": examples,
        "scheduled_keys": scheduled_keys,
        "trained_keys": trained_keys,
        "totals": totals,
        "max_ratio": max_ratio,
        "max_prefix_ratio": max_prefix_ratio,
        "entropy_examples": entropy_examples,
        "quarantined_examples": len(quarantined),
        "wall_seconds": wall_seconds,
        "reward_wait_seconds": reward_wait_seconds,
        "active_seconds": max(0.0, wall_seconds - reward_wait_seconds),
    }


def _finish_binary_coder_overlap_update(
        model, optimizer, cfg, step_idx, state):
    """Apply the one optimizer step after overlapped gradients are complete."""
    import torch

    finish_started = time.time()
    update_error = None
    grad_norm = None
    try:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            max_norm=cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        grad_norm = float(grad_norm_tensor.item())
    except BaseException as error:
        update_error = error
    finally:
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
    if update_error is not None:
        raise update_error

    totals = state["totals"]
    history = [{
        "epoch": 1,
        **totals,
        "ratio_max": state["max_ratio"],
        "prefix_ratio_max": state["max_prefix_ratio"],
        "entropy_examples": state["entropy_examples"],
        "oom_quarantined_examples": state["quarantined_examples"],
        "grad_norm": grad_norm,
    }]
    print(f"[step {step_idx}] binary-coder epoch 1/1: "
          f"loss={totals['loss']:.6f} reference=off "
          f"ratio={totals['ratio']:.6f} max={state['max_ratio']:.6f} "
          f"clipped={totals['clipped_fraction']:.1%} entropy examples=0 "
          f"OOM-quarantined={state['quarantined_examples']} "
          f"(forward/backward overlapped CPU evaluation)", flush=True)
    training_seconds = (
        float(state.get("restore_seconds", 0.0))
        + float(state["active_seconds"])
        + (time.time() - finish_started)
    )
    for example in state["examples"]:
        for key in (
                "rank_old_logprobs", "rank_reference_logprob",
                "_overlap_reward_future",
                "_overlap_parent"):
            example.pop(key, None)
    return {
        "rank_updates": history,
        "rank_train_seconds": training_seconds,
        "binary_train_overlap_wall_seconds": float(state["wall_seconds"]),
        "binary_train_overlap_reward_wait_seconds": float(
            state["reward_wait_seconds"]),
        "binary_train_overlap_active_seconds": float(state["active_seconds"]),
        "oom_quarantined_examples": int(state["quarantined_examples"]),
    }


def _train_rank_examples(backend, model, tokenizer, optimizer, examples,
                         cfg, step_idx):
    """Cache the old/reference policies, then update the rank surrogate."""
    import torch

    epochs = int(getattr(cfg, "rank_update_epochs", RANK_UPDATE_EPOCHS_DEFAULT))
    epsilon, epsilon_low, epsilon_high, kl_coef = (
        _clipped_policy_options(cfg))
    entropy_coef = _rank_entropy_coefficient(cfg)
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
    supplied_reference = len(examples) - len(reference_fallback)
    reference_label = _clipped_reference_count(
        cfg, supplied_reference, len(examples))
    fallback_reference_label = (
        "off" if cfg.advantage_mode in ("spo-rs", "binary-coder")
        else str(len(reference_fallback)))
    print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: vLLM supplied old="
          f"{len(examples) - len(missing_old)}/{len(examples)}, reference="
          f"{reference_label}; HF fallback old={len(missing_old)}, "
          f"reference={fallback_reference_label} before {epochs} update(s)",
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
                        measure_entropy=(epoch == 0),
                        return_entropy=gate)
                    if gate:
                        current_logprobs, token_entropies = result
                    else:
                        current_logprobs = result
                        token_entropies = [None] * len(batch)
                    weighted_losses = []
                    for ex, cur_lp, token_entropy in zip(
                            batch, current_logprobs, token_entropies):
                        loss, metrics = clipped_policy_loss(
                            cfg, cur_lp, ex["rank_old_logprobs"],
                            ex["rank_reference_logprob"], ex["advantage"],
                            clip_epsilon=epsilon,
                            clip_epsilon_low=epsilon_low,
                            clip_epsilon_high=epsilon_high,
                            kl_coef=kl_coef,
                            entropy_coef=(entropy_coef if gate else 0.0),
                            token_entropies=token_entropy)
                        weight = float(ex["sample_weight"])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError(
                                "nonfinite rank loss")
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
            print(f"[step {step_idx}] {cfg.advantage_mode} epoch "
                  f"{epoch + 1}/{epochs}: "
                  f"loss={totals['loss']:.6f} "
                  f"{_clipped_reference_metric(cfg, totals['kl_estimate'])} "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={len(quarantined)}", flush=True)
    for ex in examples:
        for key in ("rank_old_logprobs", "rank_reference_logprob"):
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
        from loading_logs import quiet_replica_load

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
            replica_log = (f"{dependency_log_path}.trainer-rank{logical_id}"
                           if dependency_log_path else None)
            with quiet_replica_load(replica_log):
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

    def _train_rank_fast(self, examples, cfg, step_idx):
        """Adaptive, work-stealing rank update selected only by --fast.

        Each replica repeatedly claims a length-bucketed batch from one shared
        CPU queue. Its padded-token budget is learned from real peak allocator
        use on that GPU and persists across steps. The controller aims below
        80% total device occupancy, while the normal OOM splitter remains the
        final guard for nonlinear attention/output-head memory spikes.
        """
        import threading
        import torch

        started = time.time()
        if self._offloaded:
            print(f"[train-fast] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()

        epochs = int(getattr(cfg, "rank_update_epochs",
                             RANK_UPDATE_EPOCHS_DEFAULT))
        if epochs != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")
        epsilon, epsilon_low, epsilon_high, kl_coef = (
            _clipped_policy_options(cfg))
        entropy_coef = _rank_entropy_coefficient(cfg)

        cpu_examples = self._cpu_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in cpu_examples)
        supplied_reference = sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in cpu_examples) if kl_coef else len(cpu_examples)
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(cpu_examples)}, reference="
              f"{_clipped_reference_count(cfg, supplied_reference, len(cpu_examples))}; "
              "HF fallback runs only "
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
                                    loss, metrics = clipped_policy_loss(
                                        cfg, current_lp,
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
                                    weight = float(example["sample_weight"])
                                    if not torch.isfinite(loss).all():
                                        raise FloatingPointError(
                                            "nonfinite rank loss")
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
                            attempt_entropy_examples,
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
        print(f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"{_clipped_reference_metric(cfg, totals['kl_estimate'])} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
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

    def train_rank(self, examples, cfg, step_idx):
        if bool(getattr(cfg, "fast", False)):
            return self._train_rank_fast(
                examples, cfg, step_idx)

        import torch

        started = time.time()
        if self._offloaded:
            print(f"[train-parallel] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()
        epochs = int(getattr(cfg, "rank_update_epochs",
                             RANK_UPDATE_EPOCHS_DEFAULT))
        epsilon, epsilon_low, epsilon_high, kl_coef = (
            _clipped_policy_options(cfg))
        entropy_coef = _rank_entropy_coefficient(cfg)
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
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"{_clipped_reference_count(cfg, supplied_reference, len(examples))}; "
              "HF fallback runs only "
              f"for missing values before {epochs} update(s)", flush=True)

        def cache(item):
            replica_index, shard = item
            backend, model, tokenizer, logical_id = self.replicas[replica_index]
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
                torch.cuda.synchronize(logical_id)

        self._run_replicas(
            cache, list(enumerate(local_shards)))
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
                                measure_entropy=(epoch == 0),
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
                                loss, metrics = clipped_policy_loss(
                                    cfg, current_lp,
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
                                weight = float(example["sample_weight"])
                                if not torch.isfinite(loss).all():
                                    raise FloatingPointError(
                                        "nonfinite rank loss")
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
            print(f"[step {step_idx}] {cfg.advantage_mode} epoch "
                  f"{epoch + 1}/{epochs}: "
                  f"loss={totals['loss']:.6f} "
                  f"{_clipped_reference_metric(cfg, totals['kl_estimate'])} "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={quarantined_examples}", flush=True)

        return {
            "rank_updates": history,
            "rank_train_seconds": time.time() - started,
            "training_parallel_gpus": self.world_size,
            "oom_quarantined_examples": sum(
                item["oom_quarantined_examples"] for item in history),
        }

    def train_policy(self, examples, cfg, step_idx):
        import torch

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
        sequence_policy = _uses_sequence_level_policy_ratio(cfg)
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
                kl_error = None
                for batch in active_batches:
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
                            raise RuntimeError(
                                "exact base-policy logprob fallback failed; "
                                "refusing to replace the configured KL "
                                "penalty with the current policy") from error
                    current_logprobs = compute_batched_token_logprobs(
                        model, batch, with_grad=True,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id)
                    batch_losses = []
                    for example, current_lp, base_lp in zip(
                            batch, current_logprobs, base_logprobs):
                        base_lp = base_lp.to(current_lp.device)
                        advantage = example["advantage"]
                        logp_difference = (current_lp - base_lp).detach()
                        average_difference = logp_difference.mean()
                        kl_advantage = cfg.kl_penalty_coef * (
                            average_difference - (current_lp - base_lp))
                        effective_advantage = advantage + kl_advantage
                        behavior_lp = example.get("behavior_logprobs")
                        has_behavior = _valid_example_token_logprobs(
                            example, "behavior_logprobs")
                        if sequence_policy:
                            loss, policy_metrics = (
                                _a3b_sequence_clipped_standard_loss(
                                    cfg, current_lp,
                                    behavior_lp if has_behavior else None,
                                    base_lp, advantage))
                            importance_ratio = policy_metrics["ratio"]
                        elif has_behavior:
                            importance_ratio = (
                                _detached_behavior_importance_ratio(
                                    cfg, current_lp, behavior_lp))
                            loss = -(
                                importance_ratio
                                * effective_advantage.detach()
                                * current_lp).mean()
                        else:
                            importance_ratio = 1.0
                            loss = -(
                                effective_advantage.detach()
                                * current_lp).mean()
                        if has_behavior:
                            ratio_sum += float(
                                importance_ratio.mean().item())
                            ratio_max = max(
                                ratio_max,
                                float(importance_ratio.max().item()))
                            ratio_count += 1
                        batch_losses.append(loss / n_examples)
                        total_loss += float(loss.detach().item())
                        total_logp_delta += float(
                            logp_difference.mean().item())
                    if batch_losses:
                        sum(batch_losses[1:], batch_losses[0]).backward()
                torch.cuda.synchronize(logical_id)
                return (total_loss, total_logp_delta, ratio_sum, ratio_max,
                        ratio_count, kl_error)

            with torch.cuda.device(logical_id):
                result, _effective, quarantined = (
                    _run_oom_resilient_backward(
                        model, batches, attempt,
                        device_label=f"cuda:{logical_id}"))
            if result is None:
                result = (0.0, 0.0, 0.0, 0.0, 0, None)
            return (*result, len(quarantined))

        results = self._run_replicas(
            accumulate, list(enumerate(microbatches)))
        grad_norm = self._clip_step_and_sync(cfg)
        total_loss = sum(result[0] for result in results)
        total_logp_delta = sum(result[1] for result in results)
        ratio_sum = sum(result[2] for result in results)
        ratio_max = max(result[3] for result in results)
        ratio_count = sum(result[4] for result in results)
        kl_errors = [result[5] for result in results if result[5] is not None]
        quarantined_examples = sum(result[6] for result in results)
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
    """Fast LoRA trainer with one persistent Python process per GPU."""

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
                "the process trainer cannot start because torch.distributed is already "
                "initialized in the main process")
        self.backend = primary_backend
        self.model = primary_model
        self.tokenizer = primary_tokenizer
        self.optimizer = optimizer
        self.cfg = cfg
        self._process_label = (
            "train-fast" if bool(cfg.fast) else "train-process")
        self._offloaded = False
        self._closed = False
        self._workers_idle = True
        self._entropic_overlap = None
        self._dist_initialized = False
        self._queue_module = queue
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._work_queue = self._context.Queue()
        self._command_queues = {}
        self._processes = {}
        self._physical_ids = _parse_gpu_ids(cfg.training_gpu_ids)
        self._world_size = int(cfg.num_training_gpus)
        self._memory_fraction = float(
            getattr(cfg, "training_memory_fraction", 0.80))
        self._memory_percentage = round(
            100.0 * self._memory_fraction, 1)
        self._token_budgets = [
            max(1, int(cfg.max_seq_length))
            for _ in range(self._world_size)
        ]
        if self._world_size != len(self._physical_ids):
            raise ValueError(
                "num_training_gpus does not match training_gpu_ids")
        if self._world_size < 2:
            raise ValueError("process-distributed training requires two GPUs")

        # Eight independent Python processes must not each create a full-sized
        # host thread pool. GPU 0 was loaded before this class is constructed;
        # the allocator limit still constrains every subsequent allocation.
        torch.set_num_threads(max(
            1, int(os.cpu_count() or self._world_size) // self._world_size))
        torch.cuda.set_device(0)
        set_total_memory_ceiling(0, self._memory_fraction)
        validate_model_device(self.model, 0)
        expected_signature = trainable_parameter_signature(self.model)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        sync_dir = Path(exp_dir).resolve() / ".fast_trainer"
        sync_dir.mkdir(parents=True, exist_ok=True)
        rendezvous_path = sync_dir / (
            f"nccl-{os.getpid()}-{uuid.uuid4().hex}")
        self._init_method = rendezvous_path.as_uri()
        cfg_dict = dict(vars(cfg))
        cfg_dict["_terminal_log_path"] = str(
            Path(exp_dir).resolve() / "temirnal.log")
        cfg_dict["_setting_log_path"] = str(
            Path(exp_dir).resolve() / "setting.log")
        cfg_dict["_log_time_offset_seconds"] = _LOG_TIME_OFFSET_SECONDS

        setting_log_only(
            f"[{self._process_label}] starting "
            f"{self._world_size - 1} persistent "
            "worker processes; main process is rank 0", flush=True)
        setting_log_only(
            f"[{self._process_label}] loading trainer replicas one at a time "
            "to cap host/GPU initialization peaks", flush=True)
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
                message = self._collect_event("loaded", [rank])[rank]
                if message.get("parameter_signature") != expected_signature:
                    raise RuntimeError(
                        f"fast trainer rank {rank} has a different trainable "
                        "parameter layout")
                print(f"[{self._process_label}] trainer replica {rank + 1}/"
                      f"{self._world_size} loaded on logical GPU {rank}",
                      flush=True)
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

        setting_log_only(
            f"[{self._process_label}] process-distributed trainer active on "
            f"physical GPUs {self._physical_ids}; adaptive memory ceiling="
            f"{self._memory_percentage:g}%, "
            "exact long-rollout rescue up to 96%",
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

    def _queue_examples(self, examples, *, seal=True, announce=True):
        """Queue expensive examples first; idle ranks steal the next work."""
        import torch

        maximum_length = max(1, int(self.cfg.max_seq_length))

        def estimated_cost(example):
            prompt = int(example["prompt_ids"].shape[1])
            response = int(example["response_ids"].shape[1])
            total = prompt + response
            return total * total + response * maximum_length

        copied = self._cpu_examples(examples)
        # Packing response branches behind one shared prompt is exact for
        # ordinary transformer attention because the registered branch mask
        # isolates every response.  It is not valid for GPT-OSS (learned
        # attention sinks and alternating sliding attention), nor for
        # Qwen3.8's recurrent Gated-DeltaNet layers, whose state cannot be
        # reset by an attention mask.  Keep those profiles as independent
        # batch rows throughout scheduling and scoring.
        shared_prefix_packing = str(getattr(
            self.cfg, "coder_model_profile", "")) not in {
                "gpt-oss-120b", "qwen3.8-27b",
            }
        for example in copied:
            example["_shared_prefix_packing_allowed"] = shared_prefix_packing

        if (bool(getattr(self.cfg, "strategies", False))
                and shared_prefix_packing):
            from fast_distributed import SHARED_PREFIX_WORK_KEY

            grouped = {}
            for example in copied:
                prompt_job_id = example.get("prompt_job_id")
                if prompt_job_id is None:
                    raise RuntimeError(
                        "strategy training example has no prompt_job_id")
                grouped.setdefault(int(prompt_job_id), []).append(example)

            def group_cost(group):
                prompt = int(group[0]["prompt_ids"].shape[1])
                responses = [
                    int(example["response_ids"].shape[1])
                    for example in group
                ]
                # The prompt is evaluated once. Each isolated response branch
                # attends to that prompt and its own causal prefix.
                return prompt * prompt + sum(
                    response * (prompt + response)
                    for response in responses)

            groups = []
            for prompt_job_id, group in grouped.items():
                prompt = group[0]["prompt_ids"]
                if any(
                        example["prompt_ids"].shape != prompt.shape
                        or not torch.equal(example["prompt_ids"], prompt)
                        for example in group[1:]):
                    raise RuntimeError(
                        "one strategy prompt_job_id contains different prompts: "
                        f"{prompt_job_id}")
                # Longest branches first makes any budget-driven split stable
                # and avoids leaving one pathological response until last.
                group.sort(
                    key=lambda example: int(
                        example["response_ids"].shape[1]),
                    reverse=True)
                groups.append(group)
            groups.sort(key=group_cost, reverse=True)
            for group in groups:
                self._work_queue.put({SHARED_PREFIX_WORK_KEY: group})
            ordered = copied
            if announce:
                print(
                    f"[{self._process_label}] queued {len(groups)} shared strategy prompts / "
                    f"{len(ordered)} examples by packed cost; each free GPU "
                    "claims one complete prompt group",
                    flush=True,
                )
        else:
            ordered = sorted(copied, key=estimated_cost, reverse=True)
            for example in ordered:
                self._work_queue.put(example)
            if announce:
                print(f"[{self._process_label}] queued {len(ordered)} examples longest-first; "
                      "each free GPU claims the next adaptive batch", flush=True)
        if seal:
            for _ in range(self._world_size):
                self._work_queue.put(None)
        return ordered


    def calibrate_x_grpo(self, examples, cfg, step_idx, *,
                         context_group_ids=None, group_ids=None):
        from fast_distributed import local_x_grpo_calibration

        if self._offloaded:
            print(f"[train-fast] restoring {self._world_size} process "
                  "trainers for X-GRPO calibration", flush=True)
            self.restore_after_generation()

        cpu_examples = self._cpu_examples(examples)
        contexts = _x_grpo_context_group_map(
            cpu_examples, context_group_ids=context_group_ids,
            group_ids=group_ids)
        expected_group_ids = sorted(
            group_id for ids in contexts.values() for group_id in ids
        )
        groups = {group_id: [] for group_id in expected_group_ids}
        context_for_group = {
            group_id: context_id
            for context_id, ids in contexts.items()
            for group_id in ids
        }
        for example in cpu_examples:
            group_id = int(example["group_id"])
            if group_id not in groups:
                raise ValueError(
                    f"unexpected X-GRPO diagnostic group {group_id}")
            example_context = int(example.get(
                "x_grpo_context_id", context_for_group[group_id]))
            if example_context != context_for_group[group_id]:
                raise ValueError(
                    f"X-GRPO group {group_id} was assigned to context "
                    f"{context_for_group[group_id]} but contains context "
                    f"{example_context}")
            groups[group_id].append(example)

        shards = [[] for _ in range(self._world_size)]
        shard_group_ids = [[] for _ in range(self._world_size)]
        loads = [0 for _ in range(self._world_size)]
        group_items = []
        for group_id, group_examples in groups.items():
            cost = sum(
                int(example["prompt_ids"].shape[1]
                    + example["response_ids"].shape[1])
                for example in group_examples)
            group_items.append((cost, group_id, group_examples))
        for cost, group_id, group_examples in sorted(
                group_items, reverse=True):
            rank = min(range(self._world_size), key=lambda item: loads[item])
            shards[rank].extend(group_examples)
            shard_group_ids[rank].append(group_id)
            loads[rank] += cost

        print(f"[step {step_idx}] X-GRPO distributed calibration: "
              f"{len(contexts)} contexts, {len(groups)} groups x "
              f"{len(cfg.x_grpo_budgets)} budgets in one dispatch; "
              f"group-token loads/GPU={loads}; rollout LoRA gradients are "
              "collapsed by identical advantage stratum and temporary "
              "aggregates are deleted after use", flush=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "calibrate_x_grpo",
                "examples": shards[rank],
                "group_ids": shard_group_ids[rank],
                "context_group_ids": contexts,
            })

        local = local_x_grpo_calibration(
            self.backend, self.model, self.tokenizer, shards[0], cfg, 0,
            context_group_ids=contexts, group_ids=shard_group_ids[0])
        child_messages = self._collect_event(
            "calibrated", range(1, self._world_size))
        self._workers_idle = True
        parts = [local] + [
            child_messages[rank]["calibration"]
            for rank in range(1, self._world_size)
        ]
        selected = {}
        diagnostics = {}
        quarantined = 0
        for part in parts:
            selected.update({
                int(group_id): float(value)
                for group_id, value in part["selected_budgets"].items()
            })
            diagnostics.update({
                int(group_id): values
                for group_id, values in part["groups"].items()
            })
            quarantined += int(
                part.get("diagnostic_quarantined_examples", 0))
        if set(selected) != set(groups):
            raise RuntimeError(
                "distributed X-GRPO calibration lost a group: "
                f"expected {sorted(groups)}, got {sorted(selected)}")
        for context_id, context_ids in contexts.items():
            for budget_index, budget in enumerate(cfg.x_grpo_budgets):
                accepted = sum(
                    bool(diagnostics[group_id][budget_index]["accepted"])
                    for group_id in context_ids)
                print(f"[step {step_idx}] X-GRPO context {context_id} "
                      f"budget {float(budget):g}: accepted for "
                      f"{accepted}/{len(context_ids)} held-out groups",
                      flush=True)
        return {
            "selected_budgets": selected,
            "groups": diagnostics,
            "diagnostic_quarantined_examples": quarantined,
            "distributed": True,
        }

    def train_rank(self, examples, cfg, step_idx):
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      local_rank_update,
                                      reduce_trainable_gradients)

        if self._offloaded:
            print(f"[train-fast] restoring {self._world_size} process "
                  "trainers for the update", flush=True)
            self.restore_after_generation()
        if int(getattr(cfg, "rank_update_epochs", 1)) != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")

        started = time.time()
        queued_examples = self._queue_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued_examples)
        _epsilon, _epsilon_low, _epsilon_high, policy_kl_coef = (
            _clipped_policy_options(cfg))
        supplied_reference = (sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued_examples)
            if policy_kl_coef else len(examples))
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"{_clipped_reference_count(cfg, supplied_reference, len(examples))}",
              flush=True)
        print(f"[train-fast] one process/GPU; memory ceiling="
              f"{self._memory_percentage:g}%; "
              "exact singleton rescue=96%; "
              f"initial padded-token budgets/GPU={self._token_budgets}",
              flush=True)

        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_rank",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
            })

        local_stats = local_rank_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
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
        print(f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"{_clipped_reference_metric(cfg, totals['kl_estimate'])} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
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

    def begin_entropic_overlap(self, cfg, step_idx,
                               normalization_examples):
        """Start a streaming entropic backward before CPU evaluation ends."""
        from concurrent.futures import ThreadPoolExecutor
        from fast_distributed import local_policy_update

        if str(getattr(cfg, "advantage_mode", "")).lower() != "entropic":
            raise ValueError("streaming policy overlap is entropic-only")
        if self._closed or not self._workers_idle:
            raise RuntimeError(
                "process trainer is not idle for an entropic overlap update")
        if self._entropic_overlap is not None:
            raise RuntimeError("an entropic overlap update is already active")
        normalization_examples = int(normalization_examples)
        if normalization_examples < 1:
            raise ValueError(
                "entropic overlap needs a positive normalization count")
        if self._offloaded:
            print(f"[{self._process_label}] restoring {self._world_size} "
                  "process trainers while CPU evaluation continues",
                  flush=True)
            self.restore_after_generation()

        adaptive_batches = bool(cfg.fast)
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_policy",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                # The exact post-filter count is reward-dependent. Divide by
                # the known pre-reward upper bound during backward, then apply
                # the small exact correction on every rank before reduction.
                "total_examples": normalization_examples,
                "memory_fraction": self._memory_fraction,
                "adaptive_batches": adaptive_batches,
            })

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="entropic-rank0-overlap")
        state = {
            "step_idx": int(step_idx),
            "started_at": time.time(),
            "computed_at": None,
            "executor": executor,
            "future": None,
            "queued_examples": 0,
            "normalization_examples": normalization_examples,
            "scheduled_keys": set(),
            "supplied_old": 0,
            "supplied_reference": 0,
            "sealed": False,
        }
        future = executor.submit(
            local_policy_update,
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0], normalization_examples,
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            adaptive_batches=adaptive_batches)
        state["future"] = future

        def _record_completion(_future):
            state["computed_at"] = time.time()

        future.add_done_callback(_record_completion)
        self._entropic_overlap = state
        return state

    def queue_entropic_overlap(self, examples):
        """Feed one reward-complete group to the active GPU consumers."""
        state = self._entropic_overlap
        if state is None or state["sealed"]:
            raise RuntimeError("entropic overlap queue is not open")
        queued = self._queue_examples(
            examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued)
        state["supplied_reference"] += sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued)
        state["scheduled_keys"].update(
            example["_overlap_key"] for example in queued)
        return len(queued)

    def seal_entropic_overlap(self):
        """Tell every GPU consumer that no more completed groups remain."""
        state = self._entropic_overlap
        if state is None:
            return
        if not state["sealed"]:
            for _ in range(self._world_size):
                self._work_queue.put(None)
            state["sealed"] = True

    def finish_entropic_overlap(self, cfg, step_idx, expected_examples, *,
                                apply_update=True):
        """Finish, globally normalize, and optionally apply streamed gradients."""
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      reduce_trainable_gradients)

        state = self._entropic_overlap
        if state is None:
            raise RuntimeError("no entropic overlap update is active")
        self.seal_entropic_overlap()
        expected_examples = int(expected_examples)
        queued_examples = int(state["queued_examples"])
        if apply_update and expected_examples != queued_examples:
            raise RuntimeError(
                "entropic overlap final example count changed: "
                f"queued={queued_examples}, final={expected_examples}")

        try:
            local_stats = state["future"].result()
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
            if accounted_examples != queued_examples:
                raise RuntimeError(
                    "streaming entropic queue lost or duplicated work: "
                    f"accounted for {accounted_examples}/{queued_examples} "
                    "examples")

            gradient_scale = (
                float(state["normalization_examples"]) / expected_examples
                if apply_update and expected_examples > 0 else 1.0)
            for command_queue in self._command_queues.values():
                command_queue.put({
                    "kind": "finish_update",
                    "gradient_scale": gradient_scale,
                    "apply_update": bool(apply_update),
                })

            grad_norm = None
            update_error = None
            if apply_update:
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                reduce_trainable_gradients(
                    self.model, destination_rank=0)
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
                    broadcast_trainable_parameters(
                        self.model, source_rank=0)
            else:
                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)

            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
            if update_error is not None:
                raise update_error
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise
        finally:
            state["executor"].shutdown(wait=True, cancel_futures=False)
            self._entropic_overlap = None

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(
                1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        total_loss = sum(
            float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(
            float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(
            float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(
            float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(
            int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(
            int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [
            int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [
            int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [
            float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [
            float(stats["allocator_memory_fraction"])
            for stats in rank_stats]
        kl_errors = [
            error for stats in rank_stats for error in stats["kl_errors"]]
        if kl_errors:
            print(f"[warn] disable_adapter failed ({kl_errors[0]}); "
                  "training without KL penalty on affected examples",
                  flush=True)
        peak_percentages = [
            round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [
            round(100.0 * fraction, 1)
            for fraction in allocator_fractions]
        print(f"[{self._process_label}] trained examples/GPU="
              f"{trained_per_gpu}; backward batches/GPU={batches_per_gpu}; "
              f"final padded-token budgets/GPU={self._token_budgets}; "
              f"peak memory/GPU={peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        completed_at = state["computed_at"] or time.time()
        elapsed = max(0.0, completed_at - state["started_at"])
        ratio_message = ""
        if ratio_count:
            ratio_message = (
                f"  IS ratio mean={ratio_sum / ratio_count:.9f} "
                f"max={ratio_max:.3f}")
        print(f"[step {step_idx}] train time: {elapsed:.1f}s  "
              f"avg loss: {total_loss / expected_examples:.9f}  "
              "avg logpi_theta - logpi_base: "
              f"{total_logp_delta / expected_examples:.9f}{ratio_message}  "
              f"OOM-quarantined={quarantined_examples} "
              "(forward/backward overlapped CPU evaluation)", flush=True)
        return {
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_grad_norm": grad_norm,
            "training_fast": bool(cfg.fast),
            "training_fast_processes": True,
            "training_overlapped_evaluation": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_policy(self, examples, cfg, step_idx):
        """Run the exact standard policy loss on the process-per-GPU path."""
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      local_policy_update,
                                      reduce_trainable_gradients)

        process_label = self._process_label
        adaptive_batches = bool(cfg.fast)
        if self._offloaded:
            print(f"[{process_label}] restoring {self._world_size} process "
                  "trainers for the update", flush=True)
            self.restore_after_generation()

        started = time.time()
        if not examples:
            return {
                "training_seconds": 0.0,
                "training_parallel_gpus": self._world_size,
                "training_fast": bool(cfg.fast),
                "training_fast_processes": True,
                "oom_quarantined_examples": 0,
            }
        queued_examples = self._queue_examples(examples)
        total_examples = len(queued_examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued_examples)
        supplied_reference = sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued_examples)
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              f"vLLM supplied old={supplied_old}/{total_examples}, "
              f"reference={supplied_reference}/{total_examples}",
              flush=True)
        scheduler_label = (
            "adaptive batches" if adaptive_batches else
            f"configured microbatches up to "
            f"{int(cfg.train_examples_per_microbatch)} examples")
        print(f"[{process_label}] one process/GPU; {scheduler_label}; "
              f"memory ceiling={self._memory_percentage:g}%; "
              "exact singleton rescue=96%; "
              f"initial padded-token budgets/GPU={self._token_budgets}",
              flush=True)

        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_policy",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "total_examples": total_examples,
                "memory_fraction": self._memory_fraction,
                "adaptive_batches": adaptive_batches,
            })

        local_stats = local_policy_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0], total_examples,
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            adaptive_batches=adaptive_batches)
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
        if accounted_examples != total_examples:
            raise RuntimeError(
                "fast trainer shared queue lost or duplicated work: "
                f"accounted for {accounted_examples}/{total_examples} "
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
        total_loss = sum(
            float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(
            float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(
            float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(
            float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(
            int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(
            int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [
            int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [
            int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [
            float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [
            float(stats["allocator_memory_fraction"])
            for stats in rank_stats]
        kl_errors = [
            error for stats in rank_stats for error in stats["kl_errors"]
        ]
        if kl_errors:
            print(f"[warn] disable_adapter failed ({kl_errors[0]}); "
                  "training without KL penalty on affected examples",
                  flush=True)
        peak_percentages = [
            round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [
            round(100.0 * fraction, 1)
            for fraction in allocator_fractions]
        print(f"[{process_label}] trained examples/GPU={trained_per_gpu}; "
              f"backward batches/GPU={batches_per_gpu}; final padded-token "
              f"budgets/GPU={self._token_budgets}; peak memory/GPU="
              f"{peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        elapsed = time.time() - started
        ratio_message = ""
        if ratio_count:
            ratio_message = (
                f"  IS ratio mean={ratio_sum / ratio_count:.9f} "
                f"max={ratio_max:.3f}")
        print(f"[step {step_idx}] train time: {elapsed:.1f}s  "
              f"avg loss: {total_loss / total_examples:.9f}  "
              "avg logpi_theta - logpi_base: "
              f"{total_logp_delta / total_examples:.9f}{ratio_message}  "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        return {
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_grad_norm": grad_norm,
            "training_fast": bool(cfg.fast),
            "training_fast_processes": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

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
                set_total_memory_ceiling(0, self._memory_fraction)
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

        # A downstream save/evaluation exception can bypass the normal overlap
        # finalizer. Unblock rank 0's queue consumer before aborting worker
        # processes; otherwise ThreadPoolExecutor's non-daemon thread could keep
        # the interpreter alive indefinitely during error shutdown.
        if self._entropic_overlap is not None:
            state = self._entropic_overlap
            self.seal_entropic_overlap()
            try:
                state["future"].result()
            except Exception:
                pass
            finally:
                state["executor"].shutdown(
                    wait=True, cancel_futures=False)
                self._entropic_overlap = None

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


def _save_adapter(model, exp_dir, step_idx, generation_model_name=None,
                  directory_name=None):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    The caller writes a new step directory before advancing the checkpoint,
    then prunes superseded snapshots. This preserves crash-safe resume and a
    fresh path for LoRA caches without retaining the full adapter history.
    """
    out_dir = (str(Path(exp_dir) / str(directory_name))
               if directory_name is not None
               else _adapter_dir(exp_dir, step_idx))
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


def _prune_adapter_snapshots(exp_dir, keep_paths):
    """Delete completed adapter snapshots not required by the live state.

    A normal run retains only the checkpoint's current adapter. SPO-RS passes
    its preceding sampling adapter as a second keep path because the exact
    consecutive-policy KL at the next step requires both policies.
    """
    root = Path(exp_dir)
    keep_names = {
        Path(path).name for path in keep_paths if path is not None
    }
    removed = []
    for path in root.iterdir():
        is_step_adapter = bool(
            re.fullmatch(r"adapter_step\d+", path.name))
        if (not is_step_adapter and path.name != "adapter_initial"):
            continue
        if path.name in keep_names:
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
        except OSError as error:
            print(f"[warn] could not remove superseded adapter {path}: "
                  f"{error}", flush=True)
            continue
        removed.append(path.name)
    return removed


def _live_adapter_snapshots(exp_dir, current_adapter, spo_rs_tracker=None):
    """Return the minimal adapter set required to continue this run."""
    candidates = []
    if current_adapter is not None:
        candidates.append(Path(current_adapter))
    if spo_rs_tracker is not None:
        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is not None:
            candidates.append(Path(exp_dir) / previous_name)
    keep = []
    seen = set()
    for path in candidates:
        if path.name not in seen:
            keep.append(path)
            seen.add(path.name)
    return keep


def _load_adapter(model, adapter_dir, *, announce=True):
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
    if announce:
        print(f"[resume] loaded LoRA adapter from {adapter_dir}")


def _save_training_checkpoint(exp_dir, next_step, adapter_path, sampler,
                              optimizer, spo_rs_tracker=None):
    """Atomically save the state required for an exact next-step resume."""
    import torch

    target = Path(exp_dir) / "training_state.pt"
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "version": 2,
        "next_step": int(next_step),
        "adapter_dir": Path(adapter_path).name,
        "sampler": sampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "spo_rs_tracker": (spo_rs_tracker.state_dict()
                           if spo_rs_tracker is not None else None),
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
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise ValueError(f"unsupported training checkpoint: {path}")
    # Version 1's next-batch fields are no longer required or consumed.
    # Fixed batch sizes come from the resolved saved config / CLI overrides.
    required = ("next_step", "adapter_dir", "sampler", "optimizer")
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


def _logged_step_rewards(exp_dir, step_idx):
    """Read one completed step's finite rewards for tracker migration."""
    step_dir = Path(exp_dir) / f"step{int(step_idx):02d}"
    rewards = []
    for meta_path in sorted(
            step_dir.glob(f"step{int(step_idx):02d}_group*_rollout*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
            reward = float(meta["reward"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            continue
        if math.isfinite(reward):
            rewards.append(reward)
    return rewards


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


def _automatic_sandbox_memory_limit_bytes(concurrent_sandboxes: int) -> int:
    """Return a conservative automatic per-process sandbox address-space cap.

    At most 40% of physical RAM can be committed by concurrently admitted
    generated programs. The remainder is reserved for trainer replicas,
    sleeping vLLM weight backups, the active rollout engines, and the parent
    process. A four-GiB per-program ceiling is already far above the working
    set required by the intended sub-1k-point discovery problem.
    """
    concurrent_sandboxes = max(1, int(concurrent_sandboxes))
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total_bytes = page_size * physical_pages
    except (AttributeError, OSError, TypeError, ValueError):
        total_bytes = 8 * 1024**3
    shared_budget = max(1, int(total_bytes * 0.40))
    per_process = shared_budget // concurrent_sandboxes
    return max(512 * 1024**2, min(4 * 1024**3, per_process))


def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg, exp_dir, problem, gen_pool=None,
               strategy_pool=None, strategy_tokenizer=None,
               parallel_trainer=None,
               spo_rs_tracker=None, sampling_adapter_path=None,
               ensure_trainer_ready=None):
    import os
    import torch
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
    from queue import Queue

    from sampler import State
    from experiment_io import (save_parent_selections, save_rollout,
                               save_rollout_artifacts,
                               save_strategy_response)
    from problems.base import ParentContext, RewardResult, is_code_failure
    from gen_workers import make_progress_bar


    step_t0 = time.time()
    entropy_directory = None
    entropy_observations = []
    if getattr(cfg, "measure_entropy", False):
        from entropy_tools import (begin_step, observation_descriptor,
                                   rollout_observation)
        entropy_directory = begin_step(exp_dir, step_idx)
    active_advantage_mode = str(
        getattr(cfg, "advantage_mode", "entropic")).lower()
    rank_mode = active_advantage_mode == "rank"
    x_grpo_mode = active_advantage_mode == "x-grpo"
    spo_rs_mode = active_advantage_mode == "spo-rs"
    binary_coder_mode = active_advantage_mode == "binary-coder"
    if spo_rs_mode and spo_rs_tracker is None:
        raise ValueError("SPO-RS mode requires its persistent value tracker")
    if spo_rs_mode and not spo_rs_tracker.initialized:
        raise ValueError(
            "SPO-RS tracker must be initialized from pre-step history before "
            "rollouts are sampled")
    clipped_policy_mode = _uses_clipped_policy_loss(active_advantage_mode)
    x_grpo_groups_per_context = (
        int(cfg.groups_per_step) if x_grpo_mode else 1)
    requested_parent_contexts = (
        int(cfg.x_grpo_contexts_per_step)
        if x_grpo_mode else int(cfg.groups_per_step))
    parents = sampler.sample_states(requested_parent_contexts)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        prior_label = ("" if info.get("selection_mode") == "uct"
                       else f"  P={info['P']:.9f}")
        print(f"  parent {i} [{tag}]  value={info['value']:.9f}  n={info['n']}  "
              f"Q={info['Q']:.9f}{prior_label}  "
              f"bonus={info['bonus']:.9f}  score={info['score']:.9f}")

    # Save the selection event immediately. Unlike the sampler checkpoint,
    # this survives later archive pruning and also exists if generation or
    # adapter training is interrupted.
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False)
                    else type(sampler).__name__)
    save_parent_selections(
        exp_dir, step_idx, sampler_type, parents, sampler.last_picks_info)
    setting_log_only(
        f"[step {step_idx}] saved {len(parents)} selected parent(s) before "
        f"generation/training", flush=True)

    # Only problems that declare it get their construction written to disk.
    save_ctor = (bool(getattr(problem, "saves_construction", False))
                 and int(getattr(cfg, "max_saved_construction", 0)) != 0)

    training_enabled = not bool(getattr(cfg, "no_train", False))

    all_examples = []
    rank_group_stats = []
    x_grpo_groups = {}
    spo_rs_updates = []
    binary_coder_diagnostics = []
    binary_overlap_state = None
    entropic_overlap_state = None
    all_children = []
    all_child_strategy_keys = []
    strategy_pilot_diagnostics = []
    saved_rollouts = 0
    # ----- BUILD PROMPTS (one per parent/group) -----
    parent_ctxs = []
    base_messages = []

    for g, parent in enumerate(parents):
        sampler.record_expansion(
            parent,
            count=int(cfg.group_size) * x_grpo_groups_per_context)
        pc = ParentContext(
            code=parent.code,
            value=parent.value if parent.value is not None else 0.0,
            raw_score=parent.raw_score,
            construction=parent.construction,
        )
        parent_ctxs.append(pc)
        base_messages.append(problem.build_prompt(pc))

    # SPO-RS needs immutable consecutive policy versions. Binary-coder mode
    # likewise reuses the adapter saved after the preceding update; once its
    # initialization window closes, this is the frozen rollout policy for the
    # rest of the discovery run. Only a fresh run needs an initial snapshot.
    adapter_path = None
    if spo_rs_mode:
        if sampling_adapter_path is None:
            adapter_path = _save_adapter(
                model, exp_dir, step_idx, cfg.model_name,
                directory_name="adapter_initial")
        else:
            adapter_path = str(Path(sampling_adapter_path))
            if not Path(adapter_path).is_dir():
                raise FileNotFoundError(
                    f"SPO-RS sampling adapter not found: {adapter_path}")
    elif binary_coder_mode and sampling_adapter_path is not None:
        adapter_path = str(Path(sampling_adapter_path))
        if not Path(adapter_path).is_dir():
            raise FileNotFoundError(
                f"binary coder sampling adapter not found: {adapter_path}")
    elif gen_pool is not None:
        adapter_path = _save_adapter(
            model, exp_dir, step_idx, cfg.model_name)

    def _render(messages, *, rollout_phase=None):
        template_kind = str(
            getattr(cfg, "coder_template_kind", "generic"))
        render_messages = _coder_messages_for_template(
            messages, template_kind)
        template_options = {}
        if template_kind == "gpt-oss":
            template_options["reasoning_effort"] = str(
                cfg.coder_reasoning_effort)
        elif template_kind == "qwen3.8":
            template_options.update({
                "enable_thinking": True,
                "reasoning_effort": str(
                    _coder_effort_for_rollout_phase(cfg, rollout_phase)),
                "preserve_thinking": bool(cfg.coder_preserve_thinking),
            })
        elif template_kind == "qwen-thinking":
            # Thinking-2507 is a thinking-only checkpoint. Its native template
            # always opens the reasoning channel and deliberately exposes no
            # lower/higher effort selector, so no template switch is needed.
            pass
        else:
            template_options["enable_thinking"] = bool(cfg.thinking)
        try:
            return tokenizer.apply_chat_template(
                render_messages, tokenize=False, add_generation_prompt=True,
                **template_options,
            )
        except TypeError:
            # Older Qwen templates may not expose preserve_thinking even when
            # they support enable_thinking/reasoning_effort. Remove only that
            # optional history-control argument before the generic fallback.
            if "preserve_thinking" in template_options:
                reduced = dict(template_options)
                reduced.pop("preserve_thinking", None)
                try:
                    return tokenizer.apply_chat_template(
                        render_messages, tokenize=False,
                        add_generation_prompt=True, **reduced)
                except TypeError:
                    pass
            if template_kind in {"gpt-oss", "qwen3.8"}:
                raise RuntimeError(
                    f"the installed tokenizer/chat template cannot apply "
                    f"the required {template_kind} reasoning controls "
                    f"{template_options}; update transformers and refresh "
                    "the model tokenizer files")
            return tokenizer.apply_chat_template(
                render_messages, tokenize=False, add_generation_prompt=True)

    strategy_tokenizer = strategy_tokenizer or tokenizer

    def _render_strategy(messages, *, reasoning_effort=None):
        if getattr(cfg, "strategy_backend", "local") == "api":
            # The remote client applies the provider's chat template. Passing
            # structured messages here avoids loading any strategist tokenizer
            # or weights in this process.
            return [dict(message) for message in messages]
        strategy_name = str(getattr(cfg, "strategy_model_name", "")).lower()
        template_options = {}
        if "gpt-oss" in strategy_name:
            template_options["reasoning_effort"] = str(
                cfg.strategy_reasoning_effort
                if reasoning_effort is None else reasoning_effort)
        else:
            template_options["enable_thinking"] = bool(
                cfg.strategy_thinking)
        try:
            return strategy_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_options,
            )
        except TypeError:
            return strategy_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    prompt_jobs = []
    phase_prompt_job_cache = {}
    for g, _parent in enumerate(parents):
        messages = base_messages[g]
        count = (int(cfg.group_size) * x_grpo_groups_per_context
                 if x_grpo_mode else int(cfg.group_size))
        prompt_jobs.append({
            "parent_group": g,
            "messages": messages,
            "prompt_text": _render(messages),
            "coder_reasoning_effort": _coder_effort_for_rollout_phase(cfg),
            "count": count,
        })

    two_stage_rollouts = bool(
        getattr(problem, "two_stage_rollouts", False))
    strategies_per_parent = (
        int(cfg.strategies_per_parent) if two_stage_rollouts else 0)
    programs_per_strategy = (
        int(cfg.programs_per_strategy) if two_stage_rollouts else 0)
    pilot_programs_per_strategy = (
        int(getattr(cfg, "pilot_programs_per_strategy", -1))
        if two_stage_rollouts else -1)
    phase2_allocation_method = str(getattr(
        cfg, "phase2_allocation_method", "rule_based"
    )).strip().lower()
    if phase2_allocation_method not in {"rule_based", "bandit", "hurdle"}:
        raise RuntimeError(
            "phase2_allocation_method must be 'rule_based', 'bandit', or 'hurdle'")
    adaptive_strategy_pilots = bool(
        two_stage_rollouts and pilot_programs_per_strategy > 0)
    if (adaptive_strategy_pilots
            and getattr(cfg, "coder_template_kind", "generic") == "qwen3.8"):
        print(f"[step {step_idx}] Qwen3.8 coder reasoning: "
              f"pilot per parent/strategy={pilot_programs_per_strategy - 1} medium "
              "+ 1 xhigh; "
              f"phase 2={_coder_effort_for_rollout_phase(cfg, 'adaptive')}",
              flush=True)
    if (two_stage_rollouts
            and int(cfg.group_size)
            != strategies_per_parent * programs_per_strategy):
        raise RuntimeError(
            "hierarchical rollout group size must equal "
            "strategies_per_parent * programs_per_strategy")

    def _source_schedule(parent_group):
        indices = [
            index for index, job in enumerate(prompt_jobs)
            if int(job["parent_group"]) == int(parent_group)
        ]
        if not indices:
            raise RuntimeError(
                f"parent {parent_group} has no rendered prompt job")
        weights = [max(0, int(prompt_jobs[index]["count"]))
                   for index in indices]
        total = sum(weights)
        if total < 1:
            return [indices[slot % len(indices)]
                    for slot in range(strategies_per_parent)]
        cumulative = []
        running = 0
        for index, weight in zip(indices, weights):
            running += weight
            cumulative.append((running, index))
        schedule = []
        for slot in range(strategies_per_parent):
            target = (slot + 0.5) * total / strategies_per_parent
            selected = cumulative[-1][1]
            for boundary, index in cumulative:
                if target <= boundary:
                    selected = index
                    break
            schedule.append(selected)
        return schedule

    strategy_chains = []
    strategy_prompt_cache = {}
    strategy_attempt_prompts = {}
    strategy_format_max_retries = int(getattr(cfg, "strategy_format_max_retries", 3))
    if two_stage_rollouts:
        for parent_group in range(len(parents)):
            schedule = _source_schedule(parent_group)
            for fold_index in range(x_grpo_groups_per_context):
                strategy_chains.append({
                    "chain_id": len(strategy_chains),
                    "parent_group": parent_group,
                    "fold_index": fold_index,
                    "source_indices": list(schedule),
                    "strategies": [],
                })
    def _record_strategy_response(chain, strategy_index, response_text):
        """Save and register a strategy at stream-arrival time."""
        if len(chain["strategies"]) != int(strategy_index):
            raise RuntimeError(
                f"strategy chain {chain['chain_id']} received strategy "
                f"{strategy_index} after {len(chain['strategies'])} records")
        raw_response = str(response_text or "")
        save_strategy_response(
            exp_dir,
            step_idx,
            int(chain["parent_group"]),
            int(chain["fold_index"]),
            int(strategy_index),
            raw_response,
        )
        strategy, extraction_issue = _extract_final_strategy(raw_response)
        if extraction_issue is not None:
            raise RuntimeError("cannot admit an invalid strategy into a dependent chain")
        chain["strategies"].append({
            "response": raw_response,
            "strategy": strategy,
            "extraction_issue": extraction_issue,
        })

    def _strategy_prompt(source_jobs, chain, strategy_index, *, retry=0):
        from copy import deepcopy
        key = (int(chain["chain_id"]), int(strategy_index))
        if key not in strategy_prompt_cache:
            source_idx = chain["source_indices"][strategy_index]
            strategy_prompt_cache[key] = deepcopy(problem.build_strategy_messages(
                source_jobs[source_idx]["messages"],
                previous_strategies=[record["strategy"]
                                     for record in chain["strategies"]]))
        messages = deepcopy(strategy_prompt_cache[key])
        if retry:
            messages = output_retry_messages(messages, "strategy")
        prompt = _render_strategy(messages, reasoning_effort=None)
        strategy_attempt_prompts[(*key, int(retry))] = prompt
        return prompt

    def _handle_strategy_attempt(chain_index, strategy_index, response, attempt):
        chain = strategy_chains[int(chain_index)]
        issue = _extract_final_strategy(response)[1]
        save_strategy_response(
            exp_dir, step_idx, int(chain["parent_group"]),
            int(chain["fold_index"]), int(strategy_index), response,
            attempt=int(attempt), extraction_issue=issue,
            prompt_text=strategy_attempt_prompts.pop(
                (int(chain["chain_id"]), int(strategy_index), int(attempt))))
        needs_retry = strategy_retry_needed(
            issue, attempt, strategy_format_max_retries,
            label=f"strategy {strategy_index + 1}, chain {chain_index + 1}")
        if needs_retry:
            print(f"[warn] strategy {strategy_index + 1}, chain {chain_index + 1}: "
                  f"{issue}; format retry {int(attempt) + 1}/"
                  f"{strategy_format_max_retries} with unchanged reasoning/sampling",
                  flush=True)
        else:
            _record_strategy_response(chain, strategy_index, response)
            strategy_prompt_cache.pop((int(chain["chain_id"]), int(strategy_index)), None)
        return needs_retry

    def _code_prompt_jobs(source_jobs, chains):
        extraction_failures = sum(
            record["extraction_issue"] is not None
            for chain in chains for record in chain["strategies"])
        if extraction_failures:
            raise RuntimeError("invalid strategies cannot be passed to the coder")
        code_jobs = []
        for chain in chains:
            if len(chain["strategies"]) != strategies_per_parent:
                raise RuntimeError(
                    f"strategy chain {chain['chain_id']} expected "
                    f"{strategies_per_parent} strategies, received "
                    f"{len(chain['strategies'])}")
            for strategy_index, strategy_record in enumerate(
                    chain["strategies"]):
                source_idx = chain["source_indices"][strategy_index]
                source_job = source_jobs[source_idx]
                strategy = strategy_record["strategy"]
                messages = problem.build_code_messages(
                    source_job["messages"], strategy)
                code_jobs.append({
                    **source_job,
                    "messages": messages,
                    "prompt_text": _render(messages),
                    "count": programs_per_strategy,
                    "strategy": strategy,
                    "strategy_response": strategy_record["response"],
                    "strategy_index": strategy_index,
                    "assigned_fold_index": int(chain["fold_index"]),
                })
        return code_jobs

    spo_rs_divergence = 0.0
    spo_rs_context_divergences = {}
    spo_rs_missing_policy_scores = 0
    spo_rs_previous_adapter_path = None
    if spo_rs_mode and spo_rs_tracker.initialized:
        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is not None:
            spo_rs_previous_adapter_path = Path(exp_dir) / previous_name

    group_specs = []
    for parent_group in range(len(parents)):
        for fold_index in range(x_grpo_groups_per_context):
            group_specs.append({
                "group_id": len(group_specs),
                "parent_group": parent_group,
                "context_id": parent_group,
                "fold_index": fold_index,
            })
    num_groups = len(group_specs)
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
    isolated_memory_limit_bytes = None
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
        isolated_memory_limit_bytes = (
            _automatic_sandbox_memory_limit_bytes(n_reward_workers))
        setting_log_only(
            f"[step {step_idx}] isolated evaluation pool: "
            f"{n_reward_workers} sandbox process(es) across "
            f"{isolated_cpu_count} CPU(s), up to "
            f"{isolated_processes_per_cpu}/CPU from reward_workers; each "
            "process tree is pinned to one CPU and excess candidates remain "
            "queued; sandbox launches are FD-safe and do not reduce this "
            f"parallelism; automatic memory ceiling="
            f"{isolated_memory_limit_bytes / 1024**3:.2f} GiB/process, "
            "stdout+stderr capture is bounded")
    else:
        n_reward_workers = _resolve_reward_workers(cfg, problem)
        print(f"[step {step_idx}] evaluation pool: {n_reward_workers} worker(s), "
              f"{getattr(problem, 'eval_cpus', 1)} CPU(s) per candidate")
    reward_pool = ThreadPoolExecutor(max_workers=n_reward_workers)
    # Large response/prompt files are persisted as soon as generation returns,
    # on a dedicated thread so disk I/O never blocks the generation scheduler.
    # The complete reward/advantage metadata replaces the provisional metadata
    # below after evaluation; the large text artifacts are written only once.
    rollout_io_pool = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"rollout-save-step{step_idx}")
    rollout_io_futures = []

    # This bar must exist before generation begins.  Reward futures are added
    # incrementally as rollout batches arrive, so constructing it from
    # ``all_futs`` after generation made correctly overlapped evaluation look
    # as though it started late at 0%.  Future callbacks keep it live from the
    # first completed sandbox while GPU generation is still running.
    eval_bar = make_progress_bar(total_rollouts, desc="evaluating")
    eval_progress_condition = threading.Condition()
    eval_progress_completed = 0
    eval_bar_closed = False

    def _record_evaluation_completion(_future):
        nonlocal eval_progress_completed
        with eval_progress_condition:
            eval_progress_completed += 1
            try:
                eval_bar.update(1)
            finally:
                eval_progress_condition.notify_all()

    # Mutable records preserve streamed arrival order while vLLM reference
    # scoring fills its result concurrently with the CPU reward futures.
    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}    # aligned RewardResult futures
    deferred_rollouts = []
    vllm_logprob_records = []
    queued_by_parent = {g: 0 for g in range(len(parents))}
    queued_by_group = {g: 0 for g in range(num_groups)}
    planned_by_group = {g: 0 for g in range(num_groups)}
    primary_rollout_records = []
    coder_retry_records = []
    coder_retry_prompt_cache = {}
    # Adaptive generation must never stop useful planned work to run a small
    # recovery-only batch. Keep each batch's original seed offset and drain it
    # after every pilot/phase-2 rollout has been generated. The adapter does
    # not change inside a step, so this is scheduling-only: prompts, policy,
    # sampling, token limits, rewards, and training membership stay identical.
    deferred_coder_retry_batches = []
    defer_gpu_evaluation = bool(
        getattr(cfg, "evaluation_shares_generation", False))
    if adaptive_strategy_pilots and defer_gpu_evaluation:
        raise ValueError(
            "adaptive strategy pilots require CPU evaluation concurrent with "
            "rollout generation; set pilot_programs_per_strategy=-1 for the "
            "shared-GPU evaluator")

    def _run_isolated_reward(response_text, parent_ctx):
        cpu_id = isolated_cpu_slots.get()
        try:
            return problem.compute_reward(
                response_text, parent_ctx, cfg.sandbox_timeout_s,
                cpu_id=cpu_id,
                memory_limit_bytes=isolated_memory_limit_bytes)
        finally:
            isolated_cpu_slots.put(cpu_id)

    def _submit_rollout(record):
        job = prompt_jobs[record["job_idx"]]
        g = int(record["group_id"])
        parent_group = int(job["parent_group"])
        group_responses[g].append(record)
        if record.get("output_format_issue") is not None:
            # A missing artifact is already a verified format failure. Never
            # execute an earlier reasoning snippet or evaluate it twice.
            fut = Future()
            fut.set_result(RewardResult(
                reward=float(problem.fail_score), msg=record["output_format_issue"],
                failure_kind="code"))
        elif isolated_eval:
            fut = reward_pool.submit(
                _run_isolated_reward, record["text"],
                parent_ctxs[parent_group])
        else:
            fut = reward_pool.submit(
                problem.compute_reward, record["text"],
                parent_ctxs[parent_group],
                cfg.sandbox_timeout_s
            )
        reward_futures[g].append(fut)
        record["_reward_future"] = fut
        fut.add_done_callback(_record_evaluation_completion)
        return fut

    def _queue_rollout(job_idx, text, token_ids, behavior_logprobs=None, *,
                       rollout_phase=None, strategy_source_job_idx=None,
                       retry_of=None):
        # Admit every sampled code response exactly once. Incomplete or
        # malformed originals are preserved with their failure reward. Format
        # retries are separate examples and never count toward allocation.
        job = prompt_jobs[int(job_idx)]
        parent_group = int(job["parent_group"])
        parent_ordinal = queued_by_parent[parent_group]
        if retry_of is not None:
            fold_index = int(retry_of["fold_index"])
            group_id = int(retry_of["group_id"])
        elif x_grpo_mode:
            maximum = x_grpo_groups_per_context * int(cfg.group_size)
            if parent_ordinal >= maximum:
                raise RuntimeError(
                    "X-GRPO generation returned more than K*G rollouts for "
                    f"context {parent_group}")
            fold_index = int(job.get(
                "assigned_fold_index",
                parent_ordinal // int(cfg.group_size),
            ))
            group_id = (
                parent_group * x_grpo_groups_per_context + fold_index)
            if planned_by_group[group_id] >= int(cfg.group_size):
                raise RuntimeError(
                    "X-GRPO generation returned more than G rollouts for "
                    f"context {parent_group} fold {fold_index}")
        else:
            fold_index = 0
            group_id = parent_group
        rollout_index = int(queued_by_group[group_id])
        if retry_of is None:
            queued_by_parent[parent_group] = parent_ordinal + 1
            planned_by_group[group_id] += 1
        queued_by_group[group_id] = rollout_index + 1
        record = {
            "job_idx": int(job_idx),
            "parent_group": parent_group,
            "group_id": int(group_id),
            "context_id": parent_group,
            "fold_index": int(fold_index),
            "rollout_index": rollout_index,
            "text": text,
            "token_ids": list(token_ids),
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": None,
            "output_format_issue": coder_output_issue(
                text, require_final_marker=bool(
                    getattr(problem, "require_final_code_marker", False))),
            "retry_attempt": 0 if retry_of is None else 1,
            "retry_of_group": None if retry_of is None else int(retry_of["group_id"]),
            "retry_of_rollout": None if retry_of is None else int(retry_of["rollout_index"]),
            "counts_toward_allocation": retry_of is None,
        }
        (primary_rollout_records if retry_of is None else coder_retry_records).append(record)
        if entropy_directory is not None:
            record["_entropy_measurement"] = observation_descriptor(
                entropy_directory, int(group_id), rollout_index)
        if rollout_phase is not None:
            record["strategy_rollout_phase"] = str(rollout_phase)
        if strategy_source_job_idx is not None:
            record["strategy_source_job_idx"] = int(
                strategy_source_job_idx)
        save_future = rollout_io_pool.submit(
            save_rollout_artifacts,
            exp_dir, step_idx, int(group_id), rollout_index, text,
            prompt_text=job["prompt_text"],
            strategy_text=(job.get("strategy_response")
                           if two_stage_rollouts else None),
            pending_meta={
                **retry_metadata(record),
                "status": "evaluation_pending",
                "step": int(step_idx),
                "group": int(group_id),
                "parent_group": parent_group,
                "rollout": rollout_index,
                "n_response_tokens": len(token_ids),
                "coder_reasoning_effort": job.get("coder_reasoning_effort"),
                "strategy_rollout_phase": (
                    str(rollout_phase) if rollout_phase is not None else None),
                "strategy_source_job_idx": (
                    int(strategy_source_job_idx)
                    if strategy_source_job_idx is not None else None),
            },
        )
        record["_artifact_save_future"] = save_future
        rollout_io_futures.append(save_future)
        if (training_enabled and cfg.generation_backend == "vllm"
                and token_ids):
            vllm_logprob_records.append(record)
        if defer_gpu_evaluation:
            deferred_rollouts.append(record)
        else:
            _submit_rollout(record)
        return record

    def _run_coder_format_retries(records, *, seed_offset):
        """One extra attempt per missing output; never returned to the allocator."""
        if not bool(getattr(cfg, "coder_retry", False)):
            return
        failed = [record for record in records
                  if record.get("output_format_issue") is not None
                  and not record.get("retry_attempt", 0)]
        if not failed:
            return
        prompt_indices = [coder_retry_prompt_job(
            prompt_jobs, record, _render, coder_retry_prompt_cache) for record in failed]
        prompts = [prompt_jobs[index]["prompt_text"] for index in prompt_indices]
        with eval_progress_condition:
            eval_bar.total += len(failed)
            eval_bar.refresh()
        print(f"[step {step_idx}] coder format recovery: {len(failed)} extra attempts "
              "(one per missing output); originals retained, allocation unchanged, "
              "original phase reasoning/sampling preserved", flush=True)
        retry_step = int(step_idx) + int(seed_offset) + 20_000_000
        if gen_pool is not None:
            options = dict(
                prompts_by_group=prompts, group_size=1,
                counts_by_group=[1] * len(prompts), adapter_path=adapter_path,
                max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                top_p=cfg.top_p, top_k=getattr(cfg, "sampling_top_k", None),
                min_p=getattr(cfg, "sampling_min_p", None), step_idx=retry_step,
                progress_desc="coder format retries",
                full_policy_sampling=_uses_sequence_level_policy_ratio(cfg))
            if cfg.generation_backend == "vllm":
                options["return_logprobs"] = training_enabled
            stream = gen_pool.iter_group_jobs(**options)
        else:
            _seed_local_generation(retry_step)
            stream = generate_prompt_jobs(
                model, tokenizer, prompts, [1] * len(prompts), cfg,
                cap_state=cap_state)
        seen = set()
        still_missing = 0
        for index, results in stream:
            index = int(index)
            if index not in range(len(failed)) or index in seen or len(results) != 1:
                raise RuntimeError("coder format retries returned invalid/duplicate sample IDs")
            original = failed[index]
            result = results[0]
            record = _queue_rollout(
                prompt_indices[index], result[0], result[1],
                result[2] if len(result) > 2 else None,
                rollout_phase=original.get("strategy_rollout_phase"),
                strategy_source_job_idx=original.get("strategy_source_job_idx"),
                retry_of=original)
            source = original.get("strategy_source_job_idx")
            if source is not None:
                _attach_strategy_plan(record, source, original["strategy_rollout_phase"])
            still_missing += record["output_format_issue"] is not None
            seen.add(index)
        if len(seen) != len(failed):
            raise RuntimeError("coder format retries did not return every extra attempt")
        print(f"[step {step_idx}] coder format recovery complete: "
              f"{len(failed) - still_missing}/{len(failed)} complete blocks; "
              f"{still_missing} still missing (failure reward, no further retries)", flush=True)

    def _defer_coder_format_retries(records, *, seed_offset):
        if not bool(getattr(cfg, "coder_retry", False)):
            return
        failed = [record for record in records
                  if record.get("output_format_issue") is not None
                  and not record.get("retry_attempt", 0)]
        if failed:
            deferred_coder_retry_batches.append(
                (failed, int(seed_offset)))

    def _drain_deferred_coder_format_retries():
        if not bool(getattr(cfg, "coder_retry", False)):
            deferred_coder_retry_batches.clear()
            return
        while deferred_coder_retry_batches:
            records, seed_offset = deferred_coder_retry_batches.pop(0)
            _run_coder_format_retries(
                records, seed_offset=int(seed_offset))

    def _strategy_plan_metadata(source_job_idx):
        source_job = prompt_jobs[int(source_job_idx)]
        return {
            "strategy_pilot_reward_mean": source_job.get(
                "strategy_pilot_reward_mean"),
            "strategy_pilot_reward_variance": source_job.get(
                "strategy_pilot_reward_variance"),
            "strategy_pilot_mean_threshold": source_job.get(
                "strategy_pilot_mean_threshold"),
            "strategy_pilot_variance_threshold": source_job.get(
                "strategy_pilot_variance_threshold"),
            "strategy_pilot_scenario": source_job.get(
                "strategy_pilot_scenario"),
            "strategy_phase2_allocation_method": source_job.get(
                "strategy_phase2_allocation_method"),
            "strategy_pilot_valid_count": source_job.get(
                "strategy_pilot_valid_count"),
            "strategy_posterior_valid_probability": source_job.get(
                "strategy_posterior_valid_probability"),
            "strategy_initial_expected_improvement": source_job.get(
                "strategy_initial_expected_improvement"),
            "strategy_hurdle": source_job.get("strategy_hurdle"),
            "strategy_pilot_followup_count": source_job.get(
                "strategy_pilot_followup_count"),
            "strategy_allocated_programs": source_job.get(
                "strategy_allocated_programs"),
        }

    def _attach_strategy_plan(record, source_job_idx, rollout_phase):
        record["strategy_rollout_phase"] = str(rollout_phase)
        record["strategy_source_job_idx"] = int(source_job_idx)
        record.update(_strategy_plan_metadata(source_job_idx))

    def _allocate_largest_remainder(total, weights, tie_values):
        total = int(total)
        if total < 0:
            raise RuntimeError("adaptive rollout budget became negative")
        if total == 0:
            return [0] * len(weights)
        normalized = [max(0.0, float(weight)) for weight in weights]
        fallback = not any(weight > 0.0 for weight in normalized)
        if fallback:
            normalized = [1.0] * len(normalized)
        weight_sum = sum(normalized)
        quotas = [total * weight / weight_sum for weight in normalized]
        allocations = [int(math.floor(quota)) for quota in quotas]
        leftover = total - sum(allocations)
        order = sorted(
            range(len(allocations)),
            key=lambda index: (
                -(quotas[index] - allocations[index]),
                -float(tie_values[index][0]),
                -float(tie_values[index][1]),
                int(tie_values[index][2]),
            ),
        )
        for index in order[:leftover]:
            allocations[index] += 1
        if sum(allocations) != total:
            raise RuntimeError("adaptive rollout allocation lost budget")
        return allocations

    def _finish_strategy_pilots(
            pilot_records, source_job_indices, *, rewards_already_ready=False):
        """Finish one or more ready parent/fold pilot sets exactly."""
        expected_pilots = (
            len(source_job_indices) * pilot_programs_per_strategy)
        if len(pilot_records) != expected_pilots:
            raise RuntimeError(
                "pilot generation returned "
                f"{len(pilot_records)}/{expected_pilots} rollouts")
        pilot_futures = [record.get("_reward_future")
                         for record in pilot_records]
        if any(future is None for future in pilot_futures):
            raise RuntimeError(
                "adaptive pilot allocation requires CPU rewards to run "
                "concurrently with generation")
        if rewards_already_ready:
            if not all(future.done() for future in pilot_futures):
                raise RuntimeError(
                    "pilot allocation was requested before its rewards were "
                    "ready")
        else:
            wait(pilot_futures)

        rewards_by_source = {
            int(source_idx): [] for source_idx in source_job_indices}
        raw_scores_by_source = {
            int(source_idx): [] for source_idx in source_job_indices}
        valid_flags_by_source = {
            int(source_idx): [] for source_idx in source_job_indices}
        records_by_source = {
            int(source_idx): [] for source_idx in source_job_indices}
        for record in pilot_records:
            source_idx = int(record["strategy_source_job_idx"])
            result = record["_reward_future"].result()
            # Only a short prefix is persisted in rollout metadata. Drop the
            # remainder as soon as the pilot statistic has consumed the result
            # so completed futures cannot retain hundreds of MiB of diagnostic
            # output while phase 2 is generated.
            if len(result.stdout or "") > 4096:
                result.stdout = (
                    (result.stdout or "")[:4096]
                    + "\n[... sandbox output truncated ...]")
            reward = float(result.reward)
            if not math.isfinite(reward):
                raise RuntimeError(
                    f"pilot reward for strategy job {source_idx} is not finite")
            rewards_by_source[source_idx].append(reward)
            valid_flags_by_source[source_idx].append(bool(
                getattr(result, "valid", False)))
            raw_score = getattr(result, "raw_score", None)
            if bool(getattr(result, "valid", False)) and raw_score is not None:
                try:
                    raw_score = float(raw_score)
                except (TypeError, ValueError):
                    raw_score = None
                if raw_score is not None and math.isfinite(raw_score):
                    raw_scores_by_source[source_idx].append(raw_score)
            records_by_source[source_idx].append(record)

        chains = {}
        for source_idx in source_job_indices:
            job = prompt_jobs[int(source_idx)]
            key = (int(job["parent_group"]),
                   int(job.get("assigned_fold_index", 0)))
            chains.setdefault(key, []).append(int(source_idx))

        followup_counts = {int(source_idx): 0
                           for source_idx in source_job_indices}
        for (parent_group, fold_index), chain_indices in sorted(
                chains.items()):
            chain_indices.sort(
                key=lambda index: int(prompt_jobs[index]["strategy_index"]))
            if len(chain_indices) != strategies_per_parent:
                raise RuntimeError(
                    f"parent {parent_group} fold {fold_index} has "
                    f"{len(chain_indices)} strategy jobs; expected "
                    f"{strategies_per_parent}")

            means = []
            variances = []
            for source_idx in chain_indices:
                rewards = rewards_by_source[source_idx]
                if len(rewards) != pilot_programs_per_strategy:
                    raise RuntimeError(
                        f"strategy job {source_idx} has {len(rewards)} pilot "
                        f"rewards; expected {pilot_programs_per_strategy}")
                values = np.asarray(rewards, dtype=np.float64)
                means.append(float(values.mean()))
                variances.append(float(values.var(ddof=0)))

            remaining = (
                strategies_per_parent
                * (programs_per_strategy - pilot_programs_per_strategy))
            bandit_diagnostics = [None] * len(chain_indices)
            bandit_summary = None

            if phase2_allocation_method in {"bandit", "hurdle"}:
                from rollout_allocation import (
                    allocate_hurdle_expected_best,
                    allocate_posterior_expected_best,
                )
                allocation_seed = (
                    int(cfg.seed) * 1_000_003
                    + (int(step_idx) + 1) * 1009
                    + (int(parent_group) + 1) * 9176
                    + (int(fold_index) + 1) * 131
                ) % (2 ** 32)
                reward_groups = [rewards_by_source[index]
                                 for index in chain_indices]
                valid_groups = [valid_flags_by_source[index]
                                for index in chain_indices]
                if phase2_allocation_method == "hurdle":
                    saved_parent_reward = parents[parent_group].value
                    if saved_parent_reward is None:
                        saved_parent_reward = cfg.fail_score
                    followups, bandit_diagnostics, bandit_summary = (
                        allocate_hurdle_expected_best(
                            reward_groups, valid_groups, remaining,
                            parent_reward=float(saved_parent_reward),
                        )
                    )
                    allocation_label = "hurdle expected-best"
                    baseline_label = (
                        f"parent reward={float(saved_parent_reward):.9f}, ")
                    fallback_label = " equally (no pilot improved the saved parent)"
                else:
                    followups, bandit_diagnostics, bandit_summary = (
                        allocate_posterior_expected_best(
                            reward_groups, valid_groups, remaining,
                            fail_reward=float(cfg.fail_score),
                            seed=allocation_seed,
                        )
                    )
                    allocation_label = "posterior expected-best bandit"
                    baseline_label = ""
                    fallback_label = " equally (all pilots invalid)"
                mean_threshold = None
                variance_threshold = None
                scenarios = [f"{phase2_allocation_method}-expected-best"] * len(chain_indices)
                projected_gain = float(bandit_summary['projected_expected_gain'])
                gain_text = f"{projected_gain:.9f}"
                if phase2_allocation_method == "hurdle":
                    gain_text = (
                        f"{projected_gain:.9g}"
                        if bandit_summary['gain_estimate_available'] else "unavailable")
                print(
                    f"[step {step_idx}] rollout pilot parent {parent_group}"
                    f"/fold {fold_index}: {allocation_label}; "
                    f"{baseline_label}"
                    f"pilot best reward="
                    f"{float(bandit_summary['pilot_best_reward']):.9f}, "
                    f"projected phase-2 gain="
                    f"{gain_text}; "
                    f"allocating {remaining} phase-2 rollouts"
                    + (fallback_label
                       if bandit_summary.get("fallback_equal") else ""),
                    flush=True,
                )
            else:
                # Original rule-based allocator. Keep this path byte-for-byte
                # equivalent in behavior so selecting the new bandit cannot
                # change or retire existing experiments.
                mean_threshold = float(np.median(
                    np.asarray(means, dtype=np.float64)))
                variance_threshold = float(np.median(
                    np.asarray(variances, dtype=np.float64)))
                scenarios = []
                weights = []
                for mean, variance in zip(means, variances):
                    # Exact threshold ties stay on the low side. If every
                    # strategy ties on both statistics, the zero-weight
                    # fallback restores equal per-strategy allocation.
                    high_mean = mean > mean_threshold
                    high_variance = variance > variance_threshold
                    if high_mean and high_variance:
                        scenario, weight = "high-mean/high-variance", 4.0
                    elif high_mean:
                        scenario, weight = "high-mean/low-variance", 3.0
                    elif high_variance:
                        scenario, weight = "low-mean/high-variance", 2.0
                    else:
                        scenario, weight = "low-mean/low-variance", 0.0
                    scenarios.append(scenario)
                    weights.append(weight)

                tie_values = [
                    (mean, variance,
                     int(prompt_jobs[source_idx]["strategy_index"]))
                    for source_idx, mean, variance in zip(
                        chain_indices, means, variances)
                ]

                # The original rule-based zero-signal exploration floor.
                zero_signal = [
                    abs(mean) <= 1e-12 and abs(variance) <= 1e-12
                    for mean, variance in zip(means, variances)
                ]
                requested_floor = max(
                    0, int(pilot_programs_per_strategy) // 3)
                floor_allocations = [0] * len(chain_indices)
                floor_demand = requested_floor * sum(zero_signal)
                if floor_demand > 0:
                    floor_budget = min(remaining, floor_demand)
                    if floor_budget == floor_demand:
                        floor_allocations = [
                            requested_floor if is_zero else 0
                            for is_zero in zero_signal
                        ]
                    else:
                        floor_allocations = _allocate_largest_remainder(
                            floor_budget,
                            [1.0 if is_zero else 0.0
                             for is_zero in zero_signal],
                            tie_values,
                        )
                weighted_budget = remaining - sum(floor_allocations)
                weighted_allocations = _allocate_largest_remainder(
                    weighted_budget, weights, tie_values)
                followups = [
                    int(floor_count) + int(weighted_count)
                    for floor_count, weighted_count in zip(
                        floor_allocations, weighted_allocations)
                ]
                if sum(followups) != remaining:
                    raise RuntimeError(
                        "adaptive rollout exploration floor lost budget")
                fallback_equal = not any(weight > 0.0 for weight in weights)
                floor_total = sum(floor_allocations)
                if requested_floor > 0 and any(zero_signal):
                    if floor_total == floor_demand:
                        floor_label = (
                            f"; zero-signal exploration floor="
                            f"{requested_floor} for {sum(zero_signal)} "
                            "strategy(s)")
                    else:
                        floor_label = (
                            f"; zero-signal exploration reserved="
                            f"{floor_total} across {sum(zero_signal)} "
                            f"strategy(s) (requested floor={requested_floor} "
                            "each; phase-2 budget constrained)")
                else:
                    floor_label = ""
                print(
                    f"[step {step_idx}] rollout pilot parent {parent_group}"
                    f"/fold {fold_index}: mean threshold="
                    f"{mean_threshold:.9f}, variance threshold="
                    f"{variance_threshold:.9f}; allocating {remaining} "
                    "phase-2 rollouts"
                    + (" equally (all pilot statistics tied)"
                       if fallback_equal else "")
                    + floor_label,
                    flush=True,
                )

            for (source_idx, mean, variance, scenario, followup,
                 bandit_arm) in zip(
                    chain_indices, means, variances, scenarios, followups,
                    bandit_diagnostics):
                job = prompt_jobs[source_idx]
                allocated = pilot_programs_per_strategy + int(followup)
                raw_scores = raw_scores_by_source[source_idx]
                best_raw_score = (
                    (max(raw_scores) if problem.maximize else min(raw_scores))
                    if raw_scores else None
                )
                diagnostic = {
                    "parent_group": int(parent_group),
                    "fold_index": int(fold_index),
                    "strategy_index": int(job["strategy_index"]),
                    "pilot_count": int(pilot_programs_per_strategy),
                    "pilot_reward_mean": float(mean),
                    "pilot_reward_variance": float(variance),
                    "mean_threshold": (
                        float(mean_threshold)
                        if mean_threshold is not None else None),
                    "variance_threshold": (
                        float(variance_threshold)
                        if variance_threshold is not None else None),
                    "scenario": str(scenario),
                    "allocation_method": phase2_allocation_method,
                    "followup_count": int(followup),
                    "allocated_programs": int(allocated),
                    "source_job_idx": int(source_idx),
                    "pilot_best_raw_score": (
                        float(best_raw_score)
                        if best_raw_score is not None else None),
                }
                if bandit_arm is not None:
                    diagnostic.update({
                        "pilot_valid_count": int(
                            bandit_arm["valid_count"]),
                        "posterior_valid_probability": float(
                            bandit_arm[
                                "posterior_valid_probability"]),
                        "initial_expected_improvement": float(
                            bandit_arm[
                                "initial_expected_improvement"]),
                        "final_marginal_expected_improvement": float(
                            bandit_arm[
                                "final_marginal_expected_improvement"]),
                        "projected_expected_gain": float(
                            bandit_summary[
                                "projected_expected_gain"]),
                    })
                    if phase2_allocation_method == "hurdle":
                        diagnostic["hurdle"] = {
                            key: bandit_arm[key] for key in (
                                "parent_reward", "improvement_count",
                                "posterior_improvement_probability",
                                "mean_positive_improvement",
                                "best_positive_improvement",
                            )
                        }
                        diagnostic["hurdle"].update({
                            key: bandit_summary[key] for key in (
                                "incumbent_reward", "positive_gain_bandwidth",
                                "gain_estimate_available", "fallback_equal",
                            )
                        })
                strategy_pilot_diagnostics.append(diagnostic)
                job.update({
                    "strategy_pilot_reward_mean": float(mean),
                    "strategy_pilot_reward_variance": float(variance),
                    "strategy_pilot_mean_threshold": (
                        float(mean_threshold)
                        if mean_threshold is not None else None),
                    "strategy_pilot_variance_threshold": (
                        float(variance_threshold)
                        if variance_threshold is not None else None),
                    "strategy_pilot_scenario": str(scenario),
                    "strategy_phase2_allocation_method": (
                        phase2_allocation_method),
                    "strategy_pilot_valid_count": diagnostic.get(
                        "pilot_valid_count"),
                    "strategy_posterior_valid_probability": diagnostic.get(
                        "posterior_valid_probability"),
                    "strategy_initial_expected_improvement": diagnostic.get(
                        "initial_expected_improvement"),
                    "strategy_hurdle": diagnostic.get("hurdle"),
                    "strategy_pilot_followup_count": int(followup),
                    "strategy_allocated_programs": int(allocated),
                })
                followup_counts[source_idx] = int(followup)
                for record in records_by_source[source_idx]:
                    _attach_strategy_plan(record, source_idx, "pilot")
                best_raw_text = (
                    f"{best_raw_score:.9f}"
                    if best_raw_score is not None else "unavailable"
                )
                if phase2_allocation_method == "hurdle":
                    allocation_text = (
                        f"valid={int(bandit_arm['valid_count'])}/"
                        f"{pilot_programs_per_strategy}, improved="
                        f"{int(bandit_arm['improvement_count'])}/"
                        f"{pilot_programs_per_strategy}, posterior-improve="
                        f"{float(bandit_arm['posterior_improvement_probability']):.6f}, "
                        f"initial-EI={float(bandit_arm['initial_expected_improvement']):.9g}"
                    )
                elif bandit_arm is not None:
                    allocation_text = (
                        f"valid={int(bandit_arm['valid_count'])}/"
                        f"{pilot_programs_per_strategy}, posterior-valid="
                        f"{float(bandit_arm['posterior_valid_probability']):.6f}, "
                        f"initial-EI="
                        f"{float(bandit_arm['initial_expected_improvement']):.9f}"
                    )
                else:
                    allocation_text = str(scenario)
                print(
                    f"[step {step_idx}]   strategy "
                    f"{int(job['strategy_index'])}: mean={mean:.9f}, "
                    f"variance={variance:.9f}, {allocation_text}, "
                    f"phase2={int(followup)}, total={allocated}, "
                    f"{_ANSI_ORANGE}pilot best raw "
                    f"{problem.metric_name}={best_raw_text}{_ANSI_RESET}",
                    flush=True,
                )

            if (sum(pilot_programs_per_strategy + count
                    for count in followups) != int(cfg.group_size)):
                raise RuntimeError(
                    "adaptive strategy allocation changed the fixed group "
                    "rollout budget")
        return [followup_counts[int(source_idx)]
                for source_idx in source_job_indices]

    def _print_adaptive_phase2_results():
        """Repeat pilot diagnostics with each strategy's final best result."""
        if not strategy_pilot_diagnostics:
            return

        records_by_source = {}
        for responses in group_responses.values():
            for record in responses:
                if record.get("retry_attempt", 0):
                    continue
                source_idx = record.get("strategy_source_job_idx")
                if source_idx is None:
                    continue
                records_by_source.setdefault(int(source_idx), []).append(
                    record)

        diagnostics_by_chain = {}
        for diagnostic in strategy_pilot_diagnostics:
            key = (int(diagnostic["parent_group"]),
                   int(diagnostic["fold_index"]))
            diagnostics_by_chain.setdefault(key, []).append(diagnostic)

        for (parent_group, fold_index), diagnostics in sorted(
                diagnostics_by_chain.items()):
            diagnostics.sort(key=lambda item: int(item["strategy_index"]))
            allocation_method = str(diagnostics[0].get(
                "allocation_method") or "rule_based")
            phase2_total = sum(
                int(item["followup_count"]) for item in diagnostics)
            if allocation_method in {"bandit", "hurdle"}:
                allocation_label = (
                    "hurdle expected-best" if allocation_method == "hurdle"
                    else "posterior expected-best bandit")
                print(
                    f"[step {step_idx}] rollout phase 2 complete parent "
                    f"{parent_group}/fold {fold_index}: {allocation_label}; "
                    f"evaluated {phase2_total} "
                    "phase-2 rollouts",
                    flush=True,
                )
            else:
                mean_threshold = float(diagnostics[0]["mean_threshold"])
                variance_threshold = float(
                    diagnostics[0]["variance_threshold"])
                print(
                    f"[step {step_idx}] rollout phase 2 complete parent "
                    f"{parent_group}/fold {fold_index}: mean threshold="
                    f"{mean_threshold:.9f}, variance threshold="
                    f"{variance_threshold:.9f}; evaluated {phase2_total} "
                    "phase-2 rollouts",
                    flush=True,
                )

            for diagnostic in diagnostics:
                source_idx = int(diagnostic["source_job_idx"])
                followup_count = int(diagnostic["followup_count"])
                pilot_best = diagnostic.get("pilot_best_raw_score")
                pilot_best_text = (
                    f"{float(pilot_best):.9f}"
                    if pilot_best is not None else "unavailable"
                )
                if allocation_method == "hurdle":
                    hurdle = diagnostic["hurdle"]
                    allocation_text = (
                        f"valid={int(diagnostic['pilot_valid_count'])}/"
                        f"{int(diagnostic['pilot_count'])}, improved="
                        f"{int(hurdle['improvement_count'])}/"
                        f"{int(diagnostic['pilot_count'])}, posterior-improve="
                        f"{float(hurdle['posterior_improvement_probability']):.6f}, "
                        f"initial-EI={float(diagnostic['initial_expected_improvement']):.9g}"
                    )
                elif allocation_method == "bandit":
                    allocation_text = (
                        f"valid={int(diagnostic['pilot_valid_count'])}/"
                        f"{int(diagnostic['pilot_count'])}, posterior-valid="
                        f"{float(diagnostic['posterior_valid_probability']):.6f}, "
                        f"initial-EI="
                        f"{float(diagnostic['initial_expected_improvement']):.9f}"
                    )
                else:
                    allocation_text = str(diagnostic["scenario"])
                line = (
                    f"[step {step_idx}]   strategy "
                    f"{int(diagnostic['strategy_index'])}: mean="
                    f"{float(diagnostic['pilot_reward_mean']):.9f}, "
                    f"variance="
                    f"{float(diagnostic['pilot_reward_variance']):.9f}, "
                    f"{allocation_text}, phase2={followup_count}, "
                    f"total={int(diagnostic['allocated_programs'])}, "
                    f"{_ANSI_ORANGE}pilot best raw "
                    f"{problem.metric_name}={pilot_best_text}"
                    f"{_ANSI_RESET}"
                )
                if followup_count > 0:
                    raw_scores = []
                    for record in records_by_source.get(source_idx, []):
                        future = record.get("_reward_future")
                        if future is None or not future.done():
                            raise RuntimeError(
                                "phase-2 report requested before all strategy "
                                "rewards were ready")
                        result = future.result()
                        raw_score = getattr(result, "raw_score", None)
                        if (not bool(getattr(result, "valid", False))
                                or raw_score is None):
                            continue
                        try:
                            raw_score = float(raw_score)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(raw_score):
                            raw_scores.append(raw_score)
                    final_best = (
                        (max(raw_scores) if problem.maximize
                         else min(raw_scores))
                        if raw_scores else None
                    )
                    diagnostic["final_best_raw_score"] = (
                        float(final_best) if final_best is not None else None)
                    final_best_text = (
                        f"{final_best:.9f}"
                        if final_best is not None else "unavailable"
                    )
                    line += (
                        f" {_ANSI_YELLOW}--> best raw after new rollouts "
                        f"{problem.metric_name}={final_best_text}"
                        f"{_ANSI_RESET}"
                    )
                print(line, flush=True)

    def _run_adaptive_followups_as_ready(
            pilot_records, source_job_indices, run_phase):
        """Allocate and generate each parent as soon as its pilots finish.

        The selected allocator sees all strategies and every pilot for that
        same parent/fold. There is no global all-parent reward barrier: a ready
        parent keeps the rollout GPUs occupied while slow CPU sandboxes
        belonging to other parents continue in parallel.
        """
        source_job_indices = [int(index) for index in source_job_indices]
        source_position = {
            source_idx: position
            for position, source_idx in enumerate(source_job_indices)
        }
        records_by_source = {source_idx: []
                             for source_idx in source_job_indices}
        for record in pilot_records:
            source_idx = int(record["strategy_source_job_idx"])
            if source_idx not in records_by_source:
                raise RuntimeError(
                    f"pilot record references unknown strategy {source_idx}")
            records_by_source[source_idx].append(record)
        for source_idx, records in records_by_source.items():
            if len(records) != pilot_programs_per_strategy:
                raise RuntimeError(
                    f"strategy job {source_idx} has {len(records)} pilot "
                    f"records; expected {pilot_programs_per_strategy}")

        pending_chains = {}
        for source_idx in source_job_indices:
            job = prompt_jobs[source_idx]
            key = (int(job["parent_group"]),
                   int(job.get("assigned_fold_index", 0)))
            pending_chains.setdefault(key, []).append(source_idx)
        for key, chain_indices in pending_chains.items():
            chain_indices.sort(
                key=lambda index: int(
                    prompt_jobs[index]["strategy_index"]))
            if len(chain_indices) != strategies_per_parent:
                raise RuntimeError(
                    f"parent {key[0]} fold {key[1]} has "
                    f"{len(chain_indices)} strategy jobs; expected "
                    f"{strategies_per_parent}")

        remaining_per_chain = (
            strategies_per_parent
            * (programs_per_strategy - pilot_programs_per_strategy))
        expected_followups = len(pending_chains) * remaining_per_chain
        print(
            f"[step {step_idx}] adaptive phase 2: "
            f"{expected_followups} programs total; each parent is dispatched "
            "as soon as its complete pilot set is evaluated",
            flush=True,
        )

        all_followup_counts = [0] * len(source_job_indices)
        dispatched = 0
        dispatch_index = 0
        while pending_chains:
            ready_keys = []
            for key, chain_indices in pending_chains.items():
                chain_futures = [
                    record.get("_reward_future")
                    for source_idx in chain_indices
                    for record in records_by_source[source_idx]
                ]
                if any(future is None for future in chain_futures):
                    raise RuntimeError(
                        "adaptive pilot allocation requires CPU rewards to "
                        "run concurrently with generation")
                if all(future.done() for future in chain_futures):
                    ready_keys.append(key)

            if not ready_keys:
                incomplete = {
                    record["_reward_future"]
                    for chain_indices in pending_chains.values()
                    for source_idx in chain_indices
                    for record in records_by_source[source_idx]
                    if not record["_reward_future"].done()
                }
                if not incomplete:
                    raise RuntimeError(
                        "adaptive pilot scheduler has no ready chain or "
                        "pending reward")
                wait(incomplete, return_when=FIRST_COMPLETED)
                continue

            ready_keys.sort()
            ready_sources = [
                source_idx
                for key in ready_keys
                for source_idx in pending_chains[key]
            ]
            ready_records = [
                record
                for source_idx in ready_sources
                for record in records_by_source[source_idx]
            ]
            ready_followups = _finish_strategy_pilots(
                ready_records, ready_sources, rewards_already_ready=True)
            batch_counts = [0] * len(source_job_indices)
            for source_idx, count in zip(
                    ready_sources, ready_followups):
                position = source_position[source_idx]
                batch_counts[position] = int(count)
                all_followup_counts[position] = int(count)
            batch_total = sum(batch_counts)
            if batch_total:
                run_phase(
                    batch_counts, dispatch_index, batch_total,
                    dispatched + batch_total, expected_followups)
                dispatched += batch_total
                dispatch_index += 1
            for key in ready_keys:
                del pending_chains[key]

        if dispatched != expected_followups:
            raise RuntimeError(
                "adaptive phase-2 dispatch changed the fixed rollout budget: "
                f"{dispatched}/{expected_followups}")
        return all_followup_counts

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
                if two_stage_rollouts:
                    source_prompt_jobs = prompt_jobs
                    planning_pool = strategy_pool or gen_pool
                    setting_log_only(f"[step {step_idx}] hierarchical planning: "
                          f"{len(strategy_chains)} chain(s) x "
                          f"{strategies_per_parent} sequential strategies with "
                          f"{cfg.strategy_model_name}; LoRA disabled "
                          f"(max {cfg.strategy_max_new_tokens} tokens each)",
                          flush=True)
                    try:
                        def _generate_strategy_batch(
                                prompts, chain_indices, strategy_index,
                                *, retry=0):
                            responses = {}
                            completed = set()
                            suffix = " retry" if retry else ""
                            for prompt_index, job_results in (
                                    planning_pool.iter_group_jobs(
                                        prompts_by_group=prompts,
                                        group_size=1,
                                        counts_by_group=[1] * len(prompts),
                                        adapter_path=None,
                                        max_new_tokens=int(
                                            cfg.strategy_max_new_tokens),
                                        temperature=float(
                                            cfg.strategy_temperature),
                                        top_p=float(cfg.strategy_top_p),
                                        top_k=getattr(
                                            cfg,
                                            "strategy_sampling_top_k",
                                            None),
                                        min_p=getattr(
                                            cfg,
                                            "strategy_sampling_min_p",
                                            None),
                                        step_idx=(
                                            int(step_idx) + 2_000_000
                                            + strategy_index * 10_000
                                            + int(retry) * 1_000_000),
                                        progress_desc=(
                                            f"strategy "
                                            f"{strategy_index + 1}/"
                                            f"{strategies_per_parent}"
                                            f"{suffix}"))):
                                if len(job_results) != 1:
                                    raise RuntimeError(
                                        "strategy generation must return "
                                        "exactly one response per chain")
                                prompt_index = int(prompt_index)
                                if (prompt_index < 0
                                        or prompt_index
                                        >= len(chain_indices)):
                                    raise RuntimeError(
                                        "strategy generation returned an "
                                        "out-of-range prompt index")
                                chain_index = int(
                                    chain_indices[prompt_index])
                                if chain_index in completed:
                                    raise RuntimeError(
                                        "strategy generation returned chain "
                                        f"{chain_index} more than once")
                                responses[chain_index] = _handle_strategy_attempt(
                                    chain_index, strategy_index, job_results[0][0], retry)
                                completed.add(chain_index)
                            if completed != set(chain_indices):
                                raise RuntimeError(
                                    "strategy generation did not return every "
                                    "parent/fold chain")
                            return responses

                        if hasattr(planning_pool, "run_sequential_chains"):
                            def _build_ready_strategy_prompt(
                                    chain_index, strategy_index, retry):
                                return _strategy_prompt(
                                    source_prompt_jobs,
                                    strategy_chains[int(chain_index)],
                                    int(strategy_index),
                                    retry=int(retry),
                                )

                            planning_pool.run_sequential_chains(
                                num_chains=len(strategy_chains),
                                num_stages=strategies_per_parent,
                                prompt_builder=_build_ready_strategy_prompt,
                                result_handler=_handle_strategy_attempt,
                                max_retries=strategy_format_max_retries,
                                adapter_path=None,
                                max_new_tokens=int(
                                    cfg.strategy_max_new_tokens),
                                temperature=float(
                                    cfg.strategy_temperature),
                                top_p=float(cfg.strategy_top_p),
                                top_k=getattr(
                                    cfg, "strategy_sampling_top_k", None),
                                min_p=getattr(
                                    cfg, "strategy_sampling_min_p", None),
                                step_idx=int(step_idx) + 2_000_000,
                                progress_desc="strategies pipelined",
                            )
                        else:
                            # API and legacy pools retain the compatible
                            # depth-batched path. Local vLLM pools use the
                            # dependency-pipelined scheduler above.
                            for strategy_index in range(
                                    strategies_per_parent):
                                chain_indices = list(range(
                                    len(strategy_chains)))
                                for attempt in range(strategy_format_max_retries + 1):
                                    strategy_prompts = [
                                        _strategy_prompt(
                                            source_prompt_jobs,
                                            strategy_chains[chain_index],
                                            strategy_index, retry=attempt)
                                        for chain_index in chain_indices]
                                    needs_retry = _generate_strategy_batch(
                                        strategy_prompts, chain_indices,
                                        strategy_index, retry=attempt)
                                    chain_indices = [index for index in chain_indices
                                                     if needs_retry[index]]
                                    if not chain_indices:
                                        break
                    finally:
                        if (planning_pool is not gen_pool
                                and not getattr(
                                    cfg,
                                    "dual_resident_qwen3_8b_strategy_coder_pools",
                                    False)):
                            planning_pool.release()
                    prompt_jobs = _code_prompt_jobs(
                        source_prompt_jobs, strategy_chains)
                    setting_log_only(f"[step {step_idx}] two-stage coding: generating "
                          f"{sum(job['count'] for job in prompt_jobs)} "
                          f"LoRA-policy programs from {len(prompt_jobs)} "
                          "strategy-conditioned prompts", flush=True)

                if adaptive_strategy_pilots:
                    source_job_indices = list(range(len(prompt_jobs)))

                    def _run_vllm_code_phase(
                            counts, *, phase, seed_offset, progress_desc,
                            defer_format_retries=False):
                        active = _coder_phase_generation_jobs(
                            source_job_indices, counts, phase, cfg)
                        if not active:
                            return []
                        active_indices = [item[0] for item in active]
                        active_counts = [item[1] for item in active]
                        active_prompt_indices = [
                            _phase_coder_prompt_job(
                                prompt_jobs, index, prompt_phase, cfg, _render,
                                phase_prompt_job_cache)
                            for index, _count, prompt_phase in active]
                        options = {
                            "prompts_by_group": [
                                prompt_jobs[index]["prompt_text"]
                                for index in active_prompt_indices],
                            "group_size": max(active_counts),
                            "counts_by_group": active_counts,
                            "adapter_path": adapter_path,
                            "max_new_tokens": cfg.max_new_tokens,
                            "temperature": cfg.temperature,
                            "top_p": cfg.top_p,
                            "top_k": getattr(
                                cfg, "sampling_top_k", None),
                            "min_p": getattr(
                                cfg, "sampling_min_p", None),
                            "step_idx": int(step_idx) + int(seed_offset),
                            "progress_desc": progress_desc,
                            "full_policy_sampling": (
                                _uses_sequence_level_policy_ratio(cfg)),
                        }
                        if cfg.generation_backend == "vllm":
                            options["return_logprobs"] = training_enabled
                        phase_records = []
                        for local_idx, job_results in (
                                gen_pool.iter_group_jobs(**options)):
                            source_idx = active_indices[int(local_idx)]
                            for result in job_results:
                                text, token_ids = result[:2]
                                behavior_logprobs = (
                                    result[2] if len(result) > 2 else None)
                                record = _queue_rollout(
                                    active_prompt_indices[int(local_idx)],
                                    text, token_ids,
                                    behavior_logprobs,
                                    rollout_phase=phase,
                                    strategy_source_job_idx=source_idx)
                                _attach_strategy_plan(
                                    record, source_idx, phase)
                                phase_records.append(record)
                        expected = sum(active_counts)
                        if len(phase_records) != expected:
                            raise RuntimeError(
                                f"{phase} generation returned "
                                f"{len(phase_records)}/{expected} rollouts")
                        if defer_format_retries:
                            _defer_coder_format_retries(
                                phase_records, seed_offset=seed_offset)
                        else:
                            _run_coder_format_retries(
                                phase_records, seed_offset=seed_offset)
                        return phase_records

                    setting_log_only(
                        f"[step {step_idx}] rollout pilot: "
                        f"{pilot_programs_per_strategy} programs x "
                        f"{len(source_job_indices)} strategies; CPU evaluation "
                        "starts as each program arrives",
                        flush=True,
                    )
                    pilot_records = _run_vllm_code_phase(
                        [pilot_programs_per_strategy]
                        * len(source_job_indices),
                        phase="pilot", seed_offset=0,
                        progress_desc="pilot rollouts",
                        defer_format_retries=True)

                    def _run_ready_vllm_followups(
                            counts, dispatch_index, _batch_total,
                            dispatched_after, expected_total):
                        _run_vllm_code_phase(
                            counts, phase="adaptive",
                            seed_offset=(
                                4_000_000 + dispatch_index * 100_000),
                            progress_desc=(
                                "adaptive rollouts "
                                f"{dispatched_after}/{expected_total}"),
                            defer_format_retries=True)

                    _run_adaptive_followups_as_ready(
                        pilot_records, source_job_indices,
                        _run_ready_vllm_followups)
                    _drain_deferred_coder_format_retries()
                else:
                    # Keep the disabled path structurally identical to the
                    # original one-pass rollout scheduler.
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
                        "top_k": getattr(cfg, "sampling_top_k", None),
                        "min_p": getattr(cfg, "sampling_min_p", None),
                        "step_idx": step_idx,
                        "full_policy_sampling": (
                            _uses_sequence_level_policy_ratio(cfg)),
                    }
                    if cfg.generation_backend == "vllm":
                        generation_options["return_logprobs"] = training_enabled
                    for job_idx, job_results in gen_pool.iter_group_jobs(
                            **generation_options):
                        for result in job_results:
                            text, token_ids = result[:2]
                            behavior_logprobs = (
                                result[2] if len(result) > 2 else None)
                            _queue_rollout(
                                job_idx, text, token_ids, behavior_logprobs)
                    _run_coder_format_retries(primary_rollout_records, seed_offset=0)
            else:
                # In-process generation uses cross-prompt micro-batches rather
                # than draining one parent at a time. Ordinary CPU verification
                # overlaps later batches; one-card GPU-mode verification waits.
                backend.set_inference_mode()
                cap_state = {"value": int(
                    getattr(cfg, "_local_generation_cap", 0) or 0)}

                def _seed_local_generation(seed_step):
                    if cfg.deterministic:
                        torch.manual_seed((int(cfg.seed) * 1_000_003
                                           + int(seed_step) * 1009 + 13)
                                          % (2**31 - 1))

                if two_stage_rollouts:
                    source_prompt_jobs = prompt_jobs
                    setting_log_only(f"[step {step_idx}] hierarchical planning: "
                          f"{len(strategy_chains)} chain(s) x "
                          f"{strategies_per_parent} sequential strategies with "
                          f"{cfg.strategy_model_name}; LoRA disabled "
                          f"(max {cfg.strategy_max_new_tokens} tokens each)",
                          flush=True)
                    strategy_cfg = SimpleNamespace(**vars(cfg))
                    strategy_cfg.advantage_mode = "entropic"
                    strategy_cfg.temperature = float(
                        cfg.strategy_temperature)
                    strategy_cfg.top_p = float(cfg.strategy_top_p)
                    strategy_cfg.sampling_top_k = getattr(
                        cfg, "strategy_sampling_top_k", None)
                    strategy_cfg.sampling_min_p = getattr(
                        cfg, "strategy_sampling_min_p", None)
                    with backend.disable_adapter():
                        for strategy_index in range(strategies_per_parent):
                            pending_chains = list(range(len(strategy_chains)))
                            for attempt in range(strategy_format_max_retries + 1):
                                strategy_prompts = [
                                    _strategy_prompt(source_prompt_jobs,
                                                     strategy_chains[index],
                                                     strategy_index, retry=attempt)
                                    for index in pending_chains]
                                _seed_local_generation(
                                    int(step_idx) + 2_000_000
                                    + strategy_index * 10_000 + attempt * 1_000_000)
                                completed_chains = set()
                                retry_chains = []
                                strategy_bar = make_progress_bar(
                                    len(pending_chains),
                                    desc=f"strategy {strategy_index + 1} attempt {attempt}")
                                try:
                                    for local_index, responses in generate_prompt_jobs(
                                            model, tokenizer, strategy_prompts,
                                            [1] * len(strategy_prompts), strategy_cfg,
                                            max_new_tokens=int(cfg.strategy_max_new_tokens),
                                            temperature=float(cfg.strategy_temperature),
                                            top_p=float(cfg.strategy_top_p),
                                            top_k=getattr(cfg, "strategy_sampling_top_k", None),
                                            min_p=getattr(cfg, "strategy_sampling_min_p", None),
                                            cap_state=cap_state):
                                        chain_index = pending_chains[int(local_index)]
                                        if len(responses) != 1 or chain_index in completed_chains:
                                            raise RuntimeError("invalid strategy generation result count")
                                        if _handle_strategy_attempt(
                                                chain_index, strategy_index,
                                                responses[0][0], attempt):
                                            retry_chains.append(chain_index)
                                        completed_chains.add(chain_index)
                                        strategy_bar.update(1)
                                finally:
                                    strategy_bar.close()
                                if completed_chains != set(pending_chains):
                                    raise RuntimeError("strategy generation omitted a parent/fold chain")
                                pending_chains = sorted(retry_chains)
                                if not pending_chains:
                                    break
                    prompt_jobs = _code_prompt_jobs(
                        source_prompt_jobs, strategy_chains)
                    setting_log_only(f"[step {step_idx}] two-stage coding: generating "
                          f"{sum(job['count'] for job in prompt_jobs)} "
                          f"LoRA-policy programs from {len(prompt_jobs)} "
                          "strategy-conditioned prompts", flush=True)

                if adaptive_strategy_pilots:
                    source_job_indices = list(range(len(prompt_jobs)))

                    def _run_local_code_phase(
                            counts, *, phase, seed_offset, progress_desc,
                            defer_format_retries=False):
                        active = _coder_phase_generation_jobs(
                            source_job_indices, counts, phase, cfg)
                        if not active:
                            return []
                        active_indices = [item[0] for item in active]
                        active_counts = [item[1] for item in active]
                        active_prompt_indices = [
                            _phase_coder_prompt_job(
                                prompt_jobs, index, prompt_phase, cfg, _render,
                                phase_prompt_job_cache)
                            for index, _count, prompt_phase in active]
                        _seed_local_generation(
                            int(step_idx) + int(seed_offset))
                        phase_records = []
                        gen_bar = make_progress_bar(
                            sum(active_counts), desc=progress_desc)
                        try:
                            for local_idx, responses in generate_prompt_jobs(
                                    model, tokenizer,
                                    [prompt_jobs[index]["prompt_text"]
                                     for index in active_prompt_indices],
                                    active_counts, cfg,
                                    cap_state=cap_state):
                                source_idx = active_indices[int(local_idx)]
                                for text, token_ids in responses:
                                    record = _queue_rollout(
                                        active_prompt_indices[int(local_idx)],
                                        text, token_ids,
                                        rollout_phase=phase,
                                        strategy_source_job_idx=source_idx)
                                    _attach_strategy_plan(
                                        record, source_idx, phase)
                                    phase_records.append(record)
                                gen_bar.update(len(responses))
                        finally:
                            gen_bar.close()
                        expected = sum(active_counts)
                        if len(phase_records) != expected:
                            raise RuntimeError(
                                f"{phase} generation returned "
                                f"{len(phase_records)}/{expected} rollouts")
                        if defer_format_retries:
                            _defer_coder_format_retries(
                                phase_records, seed_offset=seed_offset)
                        else:
                            _run_coder_format_retries(
                                phase_records, seed_offset=seed_offset)
                        return phase_records

                    setting_log_only(
                        f"[step {step_idx}] rollout pilot: "
                        f"{pilot_programs_per_strategy} programs x "
                        f"{len(source_job_indices)} strategies; CPU evaluation "
                        "starts as each program arrives",
                        flush=True,
                    )
                    pilot_records = _run_local_code_phase(
                        [pilot_programs_per_strategy]
                        * len(source_job_indices),
                        phase="pilot", seed_offset=0,
                        progress_desc="pilot rollouts",
                        defer_format_retries=True)

                    def _run_ready_local_followups(
                            counts, dispatch_index, _batch_total,
                            dispatched_after, expected_total):
                        _run_local_code_phase(
                            counts, phase="adaptive",
                            seed_offset=(
                                4_000_000 + dispatch_index * 100_000),
                            progress_desc=(
                                "adaptive rollouts "
                                f"{dispatched_after}/{expected_total}"),
                            defer_format_retries=True)

                    _run_adaptive_followups_as_ready(
                        pilot_records, source_job_indices,
                        _run_ready_local_followups)
                    _drain_deferred_coder_format_retries()
                    cfg._local_generation_cap = int(cap_state["value"])
                else:
                    # Keep the disabled path structurally identical to the
                    # original one-pass rollout scheduler.
                    _seed_local_generation(step_idx)
                    gen_bar = make_progress_bar(
                        total_rollouts, desc="rollouts")
                    try:
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

                    _run_coder_format_retries(primary_rollout_records, seed_offset=0)
                    cfg._local_generation_cap = int(cap_state["value"])

            if x_grpo_mode:
                expected_per_context = (
                    x_grpo_groups_per_context * int(cfg.group_size))
                incomplete = {
                    context_id: count
                    for context_id, count in queued_by_parent.items()
                    if count != expected_per_context
                }
                if incomplete:
                    raise RuntimeError(
                        "X-GRPO requires exactly K*G rollouts from every fixed "
                        f"context; expected {expected_per_context}, got "
                        f"{incomplete}")
            elif adaptive_strategy_pilots:
                incomplete = {
                    parent_group: count
                    for parent_group, count in queued_by_parent.items()
                    if count != int(cfg.group_size)
                }
                if incomplete:
                    raise RuntimeError(
                        "adaptive strategy generation must preserve exactly "
                        f"{int(cfg.group_size)} rollouts per parent; got "
                        f"{incomplete}")

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
                if spo_rs_mode:
                    reference_scores = [None] * len(score_pairs)
                    current_policy_scores = [None] * len(score_pairs)
                    previous_policy_scores = [None] * len(score_pairs)
                    same_policy = bool(
                        spo_rs_tracker.initialized
                        and spo_rs_previous_adapter_path is not None
                        and Path(adapter_path).resolve()
                        == spo_rs_previous_adapter_path.resolve())
                    if same_policy:
                        spo_rs_divergence = 0.0
                        spo_rs_context_divergences = {
                            int(record["context_id"]): 0.0
                            for record in vllm_logprob_records
                        }
                    elif spo_rs_tracker.initialized:
                        if (spo_rs_previous_adapter_path is None
                                or not spo_rs_previous_adapter_path.is_dir()):
                            raise FileNotFoundError(
                                "SPO-RS cannot compute the consecutive-policy "
                                "KL because its previous sampling adapter is "
                                f"missing: {spo_rs_previous_adapter_path}")
                        else:
                            # These responses were sampled by adapter_path
                            # moments ago, and vLLM already returned the exact
                            # chosen-token logprobs. Re-score only if an engine
                            # produced an incomplete rollout payload.
                            current_policy_scores = [
                                (list(record["behavior_logprobs"])
                                 if record["behavior_logprobs"] is not None
                                 and len(record["behavior_logprobs"])
                                 == len(record["token_ids"])
                                 else None)
                                for record in vllm_logprob_records
                            ]
                            if any(value is None
                                   for value in current_policy_scores):
                                rescored_current = (
                                    gen_pool.score_token_logprobs(
                                        score_pairs, show_progress=True,
                                        adapter_path=str(adapter_path),
                                        require_complete=True))
                                current_policy_scores = [
                                    existing if existing is not None else retry
                                    for existing, retry in zip(
                                        current_policy_scores,
                                        rescored_current)
                                ]
                            previous_policy_scores = (
                                gen_pool.score_token_logprobs(
                                    score_pairs, show_progress=True,
                                    adapter_path=str(
                                        spo_rs_previous_adapter_path),
                                    require_complete=True))
                            (spo_rs_divergence,
                             spo_rs_missing_policy_scores,
                             spo_rs_context_divergences) = (
                                spo_rs_tracker.consecutive_policy_divergence(
                                    current_policy_scores,
                                    previous_policy_scores,
                                    [record["context_id"] for record
                                     in vllm_logprob_records],
                                    required_context_ids=[
                                        spec["context_id"]
                                        for spec in group_specs]))
                            if (spo_rs_missing_policy_scores
                                    or not math.isfinite(spo_rs_divergence)):
                                raise RuntimeError(
                                    "SPO-RS exact consecutive-policy scoring "
                                    "completed without a usable score for "
                                    f"{spo_rs_missing_policy_scores} rollout(s)")
                elif binary_coder_mode:
                    reference_scores = [None] * len(score_pairs)
                    current_policy_scores = [None] * len(score_pairs)
                else:
                    current_policy_scores = [None] * len(score_pairs)
                    try:
                        reference_scores = gen_pool.score_token_logprobs(
                            score_pairs, show_progress=True)
                    except Exception as error:
                        reference_scores = [None] * len(score_pairs)
                        print(f"[warn] vLLM reference scoring failed ({error!r}); "
                              "missing values will be recomputed exactly by "
                              "the HF trainer", flush=True)
                for record, values in zip(
                        vllm_logprob_records, reference_scores):
                    record["reference_logprobs"] = values
                behavior_count = sum(
                    record["behavior_logprobs"] is not None
                    for record in vllm_logprob_records)
                reference_count = sum(
                    record["reference_logprobs"] is not None
                    for record in vllm_logprob_records)
                reference_hf_fallbacks = (
                    0 if spo_rs_mode or binary_coder_mode else
                    len(vllm_logprob_records) - reference_count)
                eval_done_after = sum(
                    future.done() for futures in reward_futures.values()
                    for future in futures)
                reference_label = (
                    "off for SPO-RS" if spo_rs_mode else
                    "off for binary coder" if binary_coder_mode else
                    f"{reference_count}/{len(vllm_logprob_records)}")
                if spo_rs_mode and not spo_rs_tracker.initialized:
                    policy_label = ", SPO-RS policy KL=initialization"
                elif spo_rs_mode and same_policy:
                    policy_label = ", SPO-RS policy KL=0 (same adapter)"
                elif spo_rs_mode:
                    divergence_label = (
                        "missing" if not math.isfinite(spo_rs_divergence)
                        else f"{spo_rs_divergence:.6f}")
                    policy_label = (
                        f", SPO-RS policy KL={divergence_label} "
                        f"on {len(spo_rs_context_divergences)} contexts, "
                        f"missing={spo_rs_missing_policy_scores}")
                else:
                    policy_label = ""
                print(f"[step {step_idx}] vLLM logprobs: rollout "
                      f"{behavior_count}/{len(vllm_logprob_records)}, "
                      f"reference {reference_label}{policy_label} in "
                      f"{time.time() - scoring_started:.1f}s "
                      f"(HF exact reference fallbacks: "
                      f"{reference_hf_fallbacks}; "
                      f"CPU evaluations completed during scoring: "
                      f"{eval_done_before}->{eval_done_after})", flush=True)

        finally:
            if gen_pool is not None and getattr(gen_pool, "sequential", False):
                gen_pool.release()
            if (getattr(
                    cfg,
                    "dual_resident_qwen3_8b_strategy_coder_pools",
                    False)
                    and strategy_pool is not None
                    and strategy_pool is not gen_pool
                    and getattr(strategy_pool, "sequential", False)):
                strategy_pool.release()

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

        all_futs = [f for g in range(num_groups) for f in reward_futures[g]]
        try:
            # Entropic advantages need every reward in their own group, not the
            # rewards of later groups. Stream each completed group into the
            # process-per-GPU trainer while the remaining sandbox jobs continue
            # on CPU. All forwards see the frozen pre-update policy; gradients
            # are accumulated only, globally normalized after the final group,
            # and applied below the rollout persistence barrier.
            entropic_overlap_eligible = bool(
                active_advantage_mode == "entropic"
                and training_enabled
                and parallel_trainer is not None
                and hasattr(parallel_trainer, "begin_entropic_overlap")
                and cfg.generation_backend == "vllm"
                and not deferred_rollouts
                and not evaluation_trainer_offloaded
            )
            if entropic_overlap_eligible:
                pending_groups = {
                    int(group_id) for group_id in range(num_groups)
                    if reward_futures[group_id]
                }
                overlap_normalization_examples = sum(
                    bool(record.get("token_ids"))
                    for responses in group_responses.values()
                    for record in responses)
                overlap_prompt_ids = {}
                try:
                    while pending_groups:
                        ready_groups = [
                            group_id for group_id in sorted(pending_groups)
                            if all(future.done() for future in
                                   reward_futures[group_id])
                        ]
                        if not ready_groups:
                            unfinished = [
                                future for group_id in pending_groups
                                for future in reward_futures[group_id]
                                if not future.done()
                            ]
                            if not unfinished:
                                continue
                            wait(unfinished, return_when=FIRST_COMPLETED)
                            continue

                        for group_id in ready_groups:
                            prepared = _prepare_entropic_overlap_group(
                                tokenizer, group_id,
                                group_responses[group_id],
                                reward_futures[group_id], prompt_jobs,
                                overlap_prompt_ids, num_groups)
                            if prepared:
                                if entropic_overlap_state is None:
                                    pending_before = sum(
                                        not future.done()
                                        for future in all_futs)
                                    entropic_overlap_state = (
                                        parallel_trainer.
                                        begin_entropic_overlap(
                                            cfg, step_idx,
                                            overlap_normalization_examples))
                                    print(
                                        f"[step {step_idx}] entropic "
                                        "evaluation/training overlap started; "
                                        f"{pending_before}/{len(all_futs)} CPU "
                                        "evaluations still pending",
                                        flush=True,
                                    )
                                queued = (
                                    parallel_trainer.queue_entropic_overlap(
                                        prepared))
                                remaining = sum(
                                    not future.done() for future in all_futs)
                                print(
                                    f"[step {step_idx}] entropic overlap: "
                                    f"group {group_id} reward-complete; queued "
                                    f"{queued} examples to all GPUs; "
                                    f"{remaining}/{len(all_futs)} CPU "
                                    "evaluations still pending",
                                    flush=True,
                                )
                            pending_groups.remove(group_id)

                    if entropic_overlap_state is not None:
                        parallel_trainer.seal_entropic_overlap()
                        print(
                            f"[step {step_idx}] entropic overlap queued "
                            f"{entropic_overlap_state['queued_examples']} "
                            "examples "
                            f"(old logprobs "
                            f"{entropic_overlap_state['supplied_old']}/"
                            f"{entropic_overlap_state['queued_examples']}, "
                            f"reference "
                            f"{entropic_overlap_state['supplied_reference']}/"
                            f"{entropic_overlap_state['queued_examples']}); "
                            "backward continues while rollout "
                            "artifacts are processed; optimizer step deferred",
                            flush=True,
                        )
                except Exception as error:
                    if entropic_overlap_state is not None:
                        parallel_trainer.finish_entropic_overlap(
                            cfg, step_idx,
                            entropic_overlap_state["queued_examples"],
                            apply_update=False)
                    elif not parallel_trainer._workers_idle:
                        # A restore/start failure can leave the distributed
                        # process group unusable; an ordinary retry on that same
                        # trainer would deadlock rather than provide a fallback.
                        raise
                    for responses in group_responses.values():
                        for record in responses:
                            record.pop("_entropic_overlap_example", None)
                    entropic_overlap_state = None
                    print(
                        f"[warn] entropic evaluation/training overlap was "
                        f"disabled for this step ({type(error).__name__}: "
                        f"{error}); falling back to the unchanged "
                        "post-evaluation update",
                        flush=True,
                    )

            # Binary usability is a per-rollout label and the fast objective
            # has one update epoch, no KL, and no entropy regularizer. On the
            # sharded trainer we can therefore launch each policy forward while
            # CPU rewards are outstanding, attach the +/-1 label when that
            # microbatch's future resolves, and backpropagate without changing
            # the policy. The single optimizer step remains below the artifact-
            # persistence barrier.
            overlap_eligible = bool(
                binary_coder_mode
                and training_enabled
                and parallel_trainer is None
                and ensure_trainer_ready is not None
                and cfg.generation_backend == "vllm"
                and int(getattr(cfg, "rank_update_epochs", 1)) == 1
                and not deferred_rollouts
                and not evaluation_trainer_offloaded
            )
            if overlap_eligible:
                pending_before = sum(not future.done() for future in all_futs)
                restore_started = time.time()
                overlap_examples = None
                try:
                    ensure_trainer_ready()
                    restore_seconds = time.time() - restore_started
                    overlap_examples = _prepare_binary_overlap_examples(
                        model, tokenizer, group_responses, reward_futures,
                        prompt_jobs, parents)
                    if overlap_examples:
                        print(
                            f"[step {step_idx}] binary coder "
                            "evaluation/training overlap: "
                            f"{len(overlap_examples)} examples; "
                            f"{pending_before}/{len(all_futs)} CPU "
                            "evaluations were still pending when trainer "
                            "restore began",
                            flush=True,
                        )
                        binary_overlap_state = (
                            _precompute_binary_coder_backward(
                                backend, model, tokenizer, optimizer,
                                overlap_examples, cfg, step_idx))
                        if binary_overlap_state is not None:
                            binary_overlap_state["restore_seconds"] = (
                                restore_seconds)
                            pending_after = sum(
                                not future.done() for future in all_futs)
                            print(
                                f"[step {step_idx}] binary coder backward "
                                "ready; optimizer step deferred until rollout "
                                "artifacts are saved; CPU evaluations still "
                                f"pending: {pending_after}/{len(all_futs)}",
                                flush=True,
                            )
                except Exception as error:
                    # The ordinary post-evaluation path is the correctness
                    # fallback. Clear every partial gradient and discard only
                    # temporary tensor records; completed rewards are reused.
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    for responses in group_responses.values():
                        for record in responses:
                            record.pop("_binary_overlap_example", None)
                    overlap_examples = None
                    binary_overlap_state = None
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        f"[warn] binary evaluation/training overlap was "
                        f"disabled for this step ({type(error).__name__}: "
                        f"{error}); falling back to the unchanged "
                        "post-evaluation update",
                        flush=True,
                    )

            # If training finished first, wait only for the remaining reward
            # tail. If rewards finished first, this returns immediately.  Do
            # not consume exceptions here: the aligned per-group result() call
            # below remains the single place that reports reward failures.
            wait(all_futs)
            with eval_progress_condition:
                eval_progress_condition.wait_for(
                    lambda: eval_progress_completed >= len(all_futs))
        finally:
            eval_bar.close()
            eval_bar_closed = True
    finally:
        reward_pool.shutdown(wait=True)
        rollout_io_pool.shutdown(wait=True)
        if not eval_bar_closed:
            eval_bar.close()
        if evaluation_trainer_offloaded:
            print(f"[step {step_idx}] restoring trainer after shared-GPU "
                  "benchmark evaluation")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            backend.restore_after_generation()
            _restore_optimizer_state_to_parameters(optimizer)
            backend.set_training_mode()

    # Surface an unlikely disk failure before constructing the checkpoint.
    # In the normal path every write completed concurrently with generation or
    # evaluation, so this is only a non-blocking correctness barrier.
    for save_future in rollout_io_futures:
        save_future.result()

    # Reuse the completed verifier results for the final adaptive report. No
    # rollout is evaluated a second time here.
    if adaptive_strategy_pilots:
        _print_adaptive_phase2_results()

    if spo_rs_mode and not training_enabled:
        # With policy optimization disabled, consecutive policies are exactly
        # identical even though the selected parents may be new.
        spo_rs_divergence = 0.0
        spo_rs_context_divergences = {
            int(spec["context_id"]): 0.0 for spec in group_specs
        }

    if (spo_rs_mode and training_enabled
            and cfg.generation_backend != "vllm"):
        drift_t0 = time.time()
        current_records = [
            record for responses in group_responses.values()
            for record in responses if record["token_ids"]
        ]
        prompt_cache = {}
        score_examples = []
        for record in current_records:
            job_idx = int(record["job_idx"])
            if job_idx not in prompt_cache:
                prompt_cache[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.to(model.device)
            score_examples.append({
                "prompt_ids": prompt_cache[job_idx],
                "response_ids": torch.tensor(
                    [record["token_ids"]], device=model.device),
                "_spo_rs_record": record,
            })

        def _score_local_policy(field_name):
            for batch in _training_microbatches(score_examples, cfg):
                scores = compute_batched_token_logprobs(
                    model, batch, with_grad=False,
                    chunk=cfg.logprob_chunk,
                    pad_token_id=tokenizer.pad_token_id)
                for example, score in zip(batch, scores):
                    example["_spo_rs_record"][field_name] = (
                        score.detach().cpu())

        with _rank_dropout_disabled(model):
            try:
                _score_local_policy("_spo_rs_current_score")
                for record in current_records:
                    record["behavior_logprobs"] = record.get(
                        "_spo_rs_current_score")

                same_policy = bool(
                    spo_rs_tracker.initialized
                    and spo_rs_previous_adapter_path is not None
                    and Path(adapter_path).resolve()
                    == spo_rs_previous_adapter_path.resolve())
                if same_policy:
                    spo_rs_divergence = 0.0
                    spo_rs_context_divergences = {
                        int(record["context_id"]): 0.0
                        for record in current_records
                    }
                elif spo_rs_tracker.initialized:
                    if (spo_rs_previous_adapter_path is None
                            or not spo_rs_previous_adapter_path.is_dir()):
                        spo_rs_divergence = math.inf
                        spo_rs_missing_policy_scores = len(current_records)
                    else:
                        try:
                            _load_adapter(
                                model, spo_rs_previous_adapter_path,
                                announce=False)
                            _score_local_policy("_spo_rs_previous_score")
                        finally:
                            _load_adapter(model, adapter_path, announce=False)
                        (spo_rs_divergence,
                         spo_rs_missing_policy_scores,
                         spo_rs_context_divergences) = (
                            spo_rs_tracker.consecutive_policy_divergence(
                                [record.get("_spo_rs_current_score")
                                 for record in current_records],
                                [record.get("_spo_rs_previous_score")
                                 for record in current_records],
                                [record["context_id"]
                                 for record in current_records],
                                required_context_ids=[
                                    spec["context_id"]
                                    for spec in group_specs]))
            except Exception as error:
                spo_rs_divergence = math.inf
                spo_rs_missing_policy_scores = len(current_records)
                print(f"[warn] SPO-RS consecutive-policy scoring failed "
                      f"({error!r}); this step uses rho_min", flush=True)
            finally:
                for record in current_records:
                    record.pop("_spo_rs_current_score", None)
                    record.pop("_spo_rs_previous_score", None)
        divergence_label = (
            "initialization" if not spo_rs_tracker.initialized
            else ("missing" if not math.isfinite(spo_rs_divergence)
                  else f"{spo_rs_divergence:.6f}"))
        print(f"[step {step_idx}] SPO-RS local policy scoring: "
              f"current={sum(record['behavior_logprobs'] is not None for record in current_records)}/"
              f"{len(current_records)}, KL={divergence_label} on "
              f"{len(spo_rs_context_divergences)} contexts, "
              f"missing={spo_rs_missing_policy_scores} in "
              f"{time.time() - drift_t0:.1f}s", flush=True)

    # ----- SCORE + ADVANTAGE + SAVE + COLLECT TRAINING EXAMPLES -----
    step_valid_count = 0
    step_rollout_count = 0
    step_code_failure_count = 0
    prompt_ids_by_job = {}

    # Resolve every rollout against one immutable, run-wide pre-step tracker
    # value. Only after all advantages are fixed does this step enter history.
    spo_rs_prepared = {}
    if spo_rs_mode:
        samples = []
        for group_spec in group_specs:
            group_id = int(group_spec["group_id"])
            responses = group_responses[group_id]
            futures = reward_futures[group_id]
            if len(responses) != len(futures):
                raise RuntimeError(
                    f"SPO-RS group {group_id} has {len(responses)} responses "
                    f"but {len(futures)} reward results")
            for rollout_index, record in enumerate(responses):
                job_idx = int(record["job_idx"])
                samples.append({
                    "group": group_id,
                    "rollout": rollout_index,
                    "job": job_idx,
                    "record": record,
                    "reward": float(futures[rollout_index].result().reward),
                })
        if samples:
            transformed = spo_rs_tracker.transformed_rewards(
                [sample["reward"] for sample in samples])
            normalized = spo_rs_tracker.normalized_advantages(transformed)
            if normalized is None:
                raise RuntimeError(
                    "SPO-RS lost its pre-step historical baseline")
            update = spo_rs_tracker.update(
                transformed,
                divergence=spo_rs_divergence,
                policy_adapter=adapter_path,
                step=step_idx,
            )
            update.update({
                "groups": sorted({sample["group"] for sample in samples}),
                "prompt_jobs": sorted({sample["job"] for sample in samples}),
                "contexts": sorted({
                    int(sample["record"]["context_id"])
                    for sample in samples}),
                "context_divergences": spo_rs_context_divergences,
                "missing_policy_scores": int(
                    spo_rs_missing_policy_scores),
            })
            spo_rs_updates.append(update)

            for sample_index, sample in enumerate(samples):
                rollout_key = (sample["group"], sample["rollout"])
                spo_rs_prepared[rollout_key] = {
                    "advantage": float(normalized[sample_index]),
                    "trainable": True,
                    "info": {
                        **update,
                        "transformed_reward": float(transformed[sample_index]),
                        "used_for_policy_update": True,
                    },
                }

            divergence_label = (
                "missing" if update["divergence"] is None
                else f"{update['divergence']:.6f}")
            print(f"[step {step_idx}] SPO-RS global tracker: "
                  f"v={update['value_before']:.9f}->"
                  f"{update['value_after']:.9f} "
                  f"M={update['group_size']} D={divergence_label} "
                  f"rho={update['rho']:.6f} "
                  f"eta={update['eta']:.6f}", flush=True)

    for group_spec in group_specs:
        g = int(group_spec["group_id"])
        parent_group = int(group_spec["parent_group"])
        context_id = int(group_spec["context_id"])
        fold_index = int(group_spec["fold_index"])
        parent = parents[parent_group]
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
        if x_grpo_mode and planned_by_group[g] != int(cfg.group_size):
            raise RuntimeError(
                f"X-GRPO context {context_id} fold {fold_index} returned "
                f"{planned_by_group[g]} planned rollouts; expected G={int(cfg.group_size)}")
        adv_mode = active_advantage_mode
        x_trial_advantages = None
        x_trial_info = None
        spo_rs_rollout_info = [None] * len(responses)
        spo_rs_trainable = [True] * len(responses)
        if x_grpo_mode:
            from advantage import x_grpo_advantages
            x_trial_advantages = []
            x_trial_info = []
            for budget in cfg.x_grpo_budgets:
                trial_advantages, _cutoff, trial_info = x_grpo_advantages(
                    rewards_np, budget, return_info=True)
                x_trial_advantages.append(trial_advantages)
                x_trial_info.append(trial_info)
            advantages = np.zeros_like(rewards_np)
            adv_scale = float(len(cfg.x_grpo_budgets))
            adv_scale_label = "trial_budgets"
            adv_info = x_trial_info[0]
            x_grpo_groups[g] = {
                "group": g,
                "context": context_id,
                "fold": fold_index,
                "rewards": rewards_np,
                "trial_advantages": x_trial_advantages,
                "trial_info": x_trial_info,
                "entropy_eligible": bool(
                    any(valids)
                    and float(rewards_np.max()) > float(cfg.fail_score)),
            }
        elif spo_rs_mode:
            prepared = [spo_rs_prepared[(g, r_idx)]
                        for r_idx in range(len(responses))]
            advantages = np.asarray(
                [item["advantage"] for item in prepared],
                dtype=np.float64)
            spo_rs_trainable = [item["trainable"] for item in prepared]
            spo_rs_rollout_info = [item["info"] for item in prepared]
            adv_scale = float(cfg.spo_rs_beta)
            adv_scale_label = "beta"
            adv_info = {}
        elif binary_coder_mode:
            from problems.binary_coder import binary_coder_advantages
            advantages, group_binary_diagnostics = binary_coder_advantages(
                outs, parent, fail_score=cfg.fail_score)
            binary_coder_diagnostics.extend({
                "group": g,
                "rollout": r_idx,
                **diagnostic,
            } for r_idx, diagnostic in enumerate(
                group_binary_diagnostics))
            adv_scale = 1.0
            adv_scale_label = "binary"
            adv_info = {}
        else:
            advantages, adv_scale, adv_scale_label, adv_info = (
                compute_group_advantages(
                    rewards_np, adv_mode,
                    cvar_alpha=getattr(cfg, "cvar_alpha", None),
                    cvar_lambda=getattr(cfg, "cvar_lambda", None),
                    rank_gamma=getattr(cfg, "rank_gamma", None),
                    return_info=True))
        constant = (
            bool(adv_info["all_tied"])
            if rank_mode or x_grpo_mode else
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

        step_valid_count += sum(valids)
        step_rollout_count += len(valids)
        step_code_failure_count += sum(is_code_failure(res) for res in outs)

        group_label = (f"context {context_id} fold {fold_index}"
                       if x_grpo_mode else f"group {g}")
        print(f"  {group_label}: rewards min={rewards_np.min():.9f} "
              f"mean={rewards_np.mean():.9f} max={rewards_np.max():.9f}  "
              f"valid={sum(valids)}/{len(valids)}  "
              f"{adv_scale_label}={adv_scale:.9f}")
        if rank_mode:
            print(f"    rank KL={adv_info['kl']:.6f} "
                  f"ESS={adv_info['ess']:.2f}/{len(rewards)} "
                  f"top ties={adv_info['top_count']} "
                  f"saturated={adv_info['saturated']} "
                  f"entropy gate={rank_entropy_gate}")
        elif x_grpo_mode:
            print(f"    X-GRPO candidates={list(cfg.x_grpo_budgets)} "
                  f"top ties={adv_info['top_count']} "
                  f"all tied={adv_info['all_tied']}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, record in enumerate(responses):
            text = record["text"]
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            res = outs[r_idx]
            job = prompt_jobs[job_idx]
            artifact_rollout_index = int(record.get(
                "rollout_index", r_idx))
            if artifact_rollout_index != r_idx:
                raise RuntimeError(
                    f"rollout persistence order changed for group {g}: "
                    f"generated index {artifact_rollout_index}, final index "
                    f"{r_idx}")
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
                all_child_strategy_keys.append(
                    int(job["strategy_index"])
                    if two_stage_rollouts else None)
            pick_info = (sampler.last_picks_info[parent_group]
                         if parent_group < len(sampler.last_picks_info) else {})
            meta = {
                **retry_metadata(record),
                "step": step_idx,
                "group": g,
                "parent_group": parent_group,
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
                "beta": (float(adv_scale)
                         if adv_mode in ("entropic", "spo-rs") else 0.0),
                "advantage_mode": adv_mode,
                "advantage_scale": (float(adv_scale)
                                    if math.isfinite(adv_scale) else None),
                "n_response_tokens": len(token_ids),
                "coder_reasoning_effort": job.get("coder_reasoning_effort"),
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
                "two_stage_rollout": bool(two_stage_rollouts),
                "strategy": (job.get("strategy")
                             if two_stage_rollouts else None),
                "strategy_index": (job.get("strategy_index")
                                   if two_stage_rollouts else None),
                "strategy_model_name": (cfg.strategy_model_name
                                        if two_stage_rollouts else None),
                "programs_per_strategy": (
                    programs_per_strategy if two_stage_rollouts else None),
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
            if adaptive_strategy_pilots:
                meta.update({
                    "pilot_programs_per_strategy": (
                        pilot_programs_per_strategy),
                    "strategy_rollout_phase": record.get(
                        "strategy_rollout_phase"),
                    "strategy_source_job_idx": record.get(
                        "strategy_source_job_idx"),
                    "strategy_pilot_reward_mean": record.get(
                        "strategy_pilot_reward_mean"),
                    "strategy_pilot_reward_variance": record.get(
                        "strategy_pilot_reward_variance"),
                    "strategy_pilot_mean_threshold": record.get(
                        "strategy_pilot_mean_threshold"),
                    "strategy_pilot_variance_threshold": record.get(
                        "strategy_pilot_variance_threshold"),
                    "strategy_pilot_scenario": record.get(
                        "strategy_pilot_scenario"),
                    "strategy_phase2_allocation_method": record.get(
                        "strategy_phase2_allocation_method"),
                    "strategy_pilot_valid_count": record.get(
                        "strategy_pilot_valid_count"),
                    "strategy_posterior_valid_probability": record.get(
                        "strategy_posterior_valid_probability"),
                    "strategy_initial_expected_improvement": record.get(
                        "strategy_initial_expected_improvement"),
                    "strategy_hurdle": record.get("strategy_hurdle"),
                    "strategy_pilot_followup_count": record.get(
                        "strategy_pilot_followup_count"),
                    "strategy_allocated_programs": record.get(
                        "strategy_allocated_programs"),
                })
            if rank_mode:
                meta["rank_selection"] = {
                    **rank_diagnostics,
                    "score": float(adv_info["ranks"][r_idx]),
                    "weight": float(adv_info["weights"][r_idx]),
                }
            elif x_grpo_mode:
                meta["x_grpo_context"] = context_id
                meta["x_grpo_fold"] = fold_index
                meta["x_grpo_rank"] = float(adv_info["ranks"][r_idx])
                meta["x_grpo_trial_advantages"] = [
                    float(values[r_idx]) for values in x_trial_advantages]
            elif spo_rs_mode:
                meta["spo_rs"] = spo_rs_rollout_info[r_idx]
            elif binary_coder_mode:
                meta["binary_coder"] = group_binary_diagnostics[r_idx]
            if entropy_directory is not None:
                entropy_observations.append(rollout_observation(
                    record.get("_entropy_measurement"), meta, res.code))
            save_rollout(exp_dir, step_idx, g, artifact_rollout_index,
                         text, meta,
                         prompt_text=job["prompt_text"],
                         strategy_text=(job.get("strategy_response")
                                        if two_stage_rollouts else None),
                         artifacts_already_saved=True)
            saved_rollouts += 1

        # Standard group-centered objectives skip constant-reward groups.
        # Rank mode uses exact equality and retains fully tied groups for
        # reference updates. Entropy is gated separately above: an all-failure
        # tie at fail_score must not receive an entropy update.
        if constant and not clipped_policy_mode:
            continue

        for r_idx, (record, adv) in enumerate(
                zip(responses, advantages)):
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            if not training_enabled:
                continue
            if spo_rs_mode and not spo_rs_trainable[r_idx]:
                continue
            if len(token_ids) == 0:
                continue
            prepared = (
                record.get("_binary_overlap_example")
                or record.get("_entropic_overlap_example"))
            res = outs[r_idx]
            if prepared is not None:
                prepared_advantage = prepared.get("advantage")
                if (prepared_advantage is not None
                        and float(prepared_advantage) != float(adv)):
                    prepared["_overlap_label_mismatch"] = True
                prepared.update({
                    "advantage": float(adv),
                    "reward_constant": constant,
                    "rank_entropy_gate": rank_entropy_gate,
                    "x_grpo_group_size": len(responses),
                    "sample_weight": 1.0 / (num_groups * len(responses)),
                })
                all_examples.append(prepared)
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
            all_examples.append({
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": response_ids,
                "_entropy_measurement": record.get("_entropy_measurement"),
                "advantage": float(adv),
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": reference_logprobs,
                "reward_constant": constant,
                "rank_entropy_gate": rank_entropy_gate,
                "group_id": g,
                "x_grpo_context_id": context_id,
                "x_grpo_fold_index": fold_index,
                "rollout_index": r_idx,
                "prompt_job_id": job_idx,
                "x_grpo_group_size": len(responses),
                "x_grpo_all_tied": bool(constant) if x_grpo_mode else False,
                "x_grpo_trial_advantages": (
                    tuple(float(values[r_idx])
                          for values in x_trial_advantages)
                    if x_grpo_mode else ()),
                "sample_weight": 1.0 / (num_groups * len(responses)),
            })

    # Persistence barrier: every response/prompt/meta file is on disk before the
    # optimizer can mutate the adapter. Binary coder and streamed entropic mode
    # may already have accumulated exact gradients during CPU evaluation, but
    # their single update is deliberately deferred until this barrier.
    # stepXX.summary.json remains the completion marker and is written only
    # after the adapter/checkpoint.
    save_suffix = ("with training disabled"
                   if not training_enabled else "before adapter update")
    print(f"[step {step_idx}] saved {saved_rollouts} rollout .txt/.meta.json "
          f"pairs {save_suffix}", flush=True)

    valid_fraction = (step_valid_count / step_rollout_count
                      if step_rollout_count else 0.0)
    code_valid_fraction = (1.0 - step_code_failure_count / step_rollout_count
                           if step_rollout_count else 1.0)

    # Preserve the standard objectives' constant-reward-group filtering.
    if not clipped_policy_mode:
        all_examples = [ex for ex in all_examples if not ex["reward_constant"]]

    if clipped_policy_mode and cfg.generation_backend == "vllm":
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
                  "records from the clipped-policy update", flush=True)

    if (spo_rs_mode or binary_coder_mode) and all_examples:
        trajectory_weight = 1.0 / len(all_examples)
        for example in all_examples:
            example["sample_weight"] = trajectory_weight

    if binary_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        labels_complete = all(
            example.get("advantage") in (-1.0, 1.0)
            for example in all_examples
        )
        labels_match = not any(
            example.get("_overlap_label_mismatch", False)
            for example in all_examples
        )
        if (actual_keys != binary_overlap_state["scheduled_keys"]
                or not labels_complete or not labels_match):
            # Never apply a partial or differently normalized gradient.  The
            # complete, ordinary training path below remains exact.
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[warn] discarding precomputed binary gradients because the "
                f"final training set changed "
                f"({len(actual_keys)}/"
                f"{len(binary_overlap_state['scheduled_keys'])} records); "
                "using the unchanged post-evaluation update",
                flush=True,
            )
            binary_overlap_state = None

    if entropic_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(
            math.isfinite(float(example["advantage"]))
            for example in all_examples)
        advantages_match = not any(
            example.get("_overlap_label_mismatch", False)
            for example in all_examples)
        expected_keys = entropic_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys
                or not advantages_complete or not advantages_match):
            parallel_trainer.finish_entropic_overlap(
                cfg, step_idx,
                entropic_overlap_state["queued_examples"],
                apply_update=False)
            print(
                f"[warn] discarded streamed entropic gradients because the "
                f"final training set changed ({len(actual_keys)}/"
                f"{len(expected_keys)} records); using the unchanged "
                "post-evaluation update",
                flush=True,
            )
            entropic_overlap_state = None

    rollout_time = time.time() - rollout_t0
    training_label = (str(len(all_examples))
                      if training_enabled else "disabled")
    rollout_phase_label = (
        "rollout+eval+overlapped-backward wall time"
        if (binary_overlap_state is not None
            or entropic_overlap_state is not None)
        else "rollout+eval time")
    print(f"[step {step_idx}] {rollout_phase_label}: {rollout_time:.1f}s  "
          f"training examples: {training_label}  "
          f"new children: {len(all_children)}")

    # Keep the previous raw optimum so the status line can distinguish a new
    # run best from a step that merely repeats the best-ever value.
    previous_best_raw = sampler.best_raw_state(
        maximize=bool(problem.maximize))
    previous_best_value = (
        float(previous_best_raw.raw_score)
        if previous_best_raw is not None else None)

    # Update archive
    archive_stats = sampler.update(
        all_children,
        strategy_keys=(all_child_strategy_keys
                       if two_stage_rollouts else None),
        strategy_top_r=(int(cfg.strategy_archive_top_r)
                        if two_stage_rollouts else 0),
    )
    if two_stage_rollouts:
        print(
            f"[step {step_idx}] strategy archive: "
            f"valid={archive_stats['submitted']}, "
            f"unique-new={archive_stats['unique_new']}, "
            f"top-{archive_stats['strategy_top_r']}/strategy="
            f"{archive_stats['strategy_admitted']}, "
            f"survived parent-top-{cfg.topk_children_per_parent}/global-cap="
            f"{archive_stats['retained_new']}",
            flush=True,
        )

    # Keep the best valid candidate produced by this exact step separate from
    # the cumulative best. ``all_children`` still contains candidates removed
    # later by deduplication or archive caps, so the per-step result reflects
    # what was actually evaluated rather than only what survived the archive.
    step_states = [child for child, _parent in all_children]
    step_raw_states = []
    for child in step_states:
        try:
            raw_score = float(child.raw_score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(raw_score):
            step_raw_states.append(child)
    if step_raw_states:
        step_best_result = (
            max(step_raw_states, key=lambda state: float(state.raw_score))
            if problem.maximize else
            min(step_raw_states, key=lambda state: float(state.raw_score))
        )
        step_best_raw_score = float(step_best_result.raw_score)
    elif step_states:
        # Generic fallback for a problem that exposes only the normalized
        # higher-is-better reward and no native raw metric.
        step_best_result = max(
            step_states,
            key=lambda state: (float(state.value)
                               if state.value is not None else -math.inf),
        )
        step_best_raw_score = None
    else:
        step_best_result = None
        step_best_raw_score = None

    # Report the problem-native metric before any gradient work. This
    # is deliberately separate from reward: some problems maximize the raw
    # quantity, while others (Erdos bounds, runtime, MSE) minimize it.
    best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    best_seen_result = best_raw if best_raw is not None else sampler.best_state()
    if best_raw is not None:
        direction = "higher is better" if problem.maximize else "lower is better"
        best_value = float(best_raw.raw_score)
        improved = (
            previous_best_value is None
            or (best_value > previous_best_value if problem.maximize
                else best_value < previous_best_value)
        )
        target = getattr(problem, "target", None)
        try:
            target = float(target) if target is not None else None
        except (TypeError, ValueError):
            target = None
        beats_sota = bool(
            improved and target is not None and math.isfinite(target)
            and (best_value > target if problem.maximize
                 else best_value < target)
        )
        message = (
            f"[step {step_idx}] best-ever raw {problem.metric_name}: "
            f"{best_value:.9f} ({direction}; "
            f"reward={float(best_raw.value):.9f}, found step={best_raw.timestep})"
        )
        if beats_sota:
            margin = (best_value - target if problem.maximize
                      else target - best_value)
            message += (
                f"  🏆 CONGRATULATIONS — NEW SOTA! "
                f"target={target:.9f}, beaten by {margin:.9f} 🏆")
            color = "\033[32m"       # dark green
        elif improved:
            message += "  ★ NEW RUN BEST"
            color = "\033[92m"       # light green
        else:
            message += "  no improvement this step"
            color = "\033[93m"       # yellow
        print(f"{color}{message}\033[0m", flush=True)
    else:
        message = (f"[step {step_idx}] best-ever raw "
                   f"{problem.metric_name}: unavailable")
        print(f"\033[93m{message}\033[0m", flush=True)

    step_stats = {
        "planned_rollouts": len(primary_rollout_records),
        "coder_format_retries": len(coder_retry_records),
        "coder_retry_enabled": bool(getattr(cfg, "coder_retry", False)),
        "coder_format_retries_still_missing": sum(
            record["output_format_issue"] is not None for record in coder_retry_records),
        "result_metric_name": str(problem.metric_name),
        "result_maximize": bool(problem.maximize),
        "step_best_raw_score": step_best_raw_score,
        "step_best_reward": (
            float(step_best_result.value)
            if (step_best_result is not None
                and step_best_result.value is not None) else None),
        "best_seen_raw_score": (
            float(best_seen_result.raw_score)
            if (best_seen_result is not None
                and best_seen_result.raw_score is not None)
            else None),
        "best_seen_reward": (
            float(best_seen_result.value)
            if (best_seen_result is not None
                and best_seen_result.value is not None)
            else None),
        "best_seen_step": (
            int(best_seen_result.timestep)
            if best_seen_result is not None else None),
        "valid_fraction": float(valid_fraction),
        "code_valid_fraction": float(code_valid_fraction),
        "evaluation_isolated": isolated_eval,
        "evaluation_workers": int(n_reward_workers),
        "evaluation_cpu_count": int(isolated_cpu_count),
        "evaluation_processes_per_cpu": int(isolated_processes_per_cpu),
    }
    if entropy_directory is not None:
        step_stats["_entropy_observations"] = entropy_observations
    if adaptive_strategy_pilots:
        step_stats["strategy_pilot_allocation"] = (
            strategy_pilot_diagnostics)
    if rank_mode:
        step_stats["rank_groups"] = rank_group_stats
    elif binary_coder_mode:
        usable_count = sum(
            bool(item["usable"]) for item in binary_coder_diagnostics)
        reason_counts = {}
        for item in binary_coder_diagnostics:
            reason = str(item["reason"])
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        step_stats["binary_coder"] = {
            "usable": int(usable_count),
            "failed": int(len(binary_coder_diagnostics) - usable_count),
            "total": int(len(binary_coder_diagnostics)),
            "reasons": reason_counts,
            "usable_fraction": (
                float(usable_count / len(binary_coder_diagnostics))
                if binary_coder_diagnostics else 0.0),
        }
        print(f"[step {step_idx}] binary coder verified usability: "
              f"{usable_count}/{len(binary_coder_diagnostics)} "
              f"({step_stats['binary_coder']['usable_fraction']:.1%}); "
              "usable=+1, every rejection=-1", flush=True)
    elif spo_rs_mode:
        step_stats["spo_rs_updates"] = spo_rs_updates
        step_stats["spo_rs_tracker_size"] = len(spo_rs_tracker)
        step_stats["spo_rs_divergence"] = (
            None if not math.isfinite(spo_rs_divergence)
            else float(spo_rs_divergence))
        step_stats["spo_rs_context_divergences"] = (
            spo_rs_context_divergences)
        step_stats["spo_rs_missing_policy_scores"] = int(
            spo_rs_missing_policy_scores)
        step_stats["spo_rs_sampling_adapter"] = Path(adapter_path).name
        step_stats["spo_rs_policy_updated"] = False

    if not training_enabled:
        step_stats["training_disabled"] = True
        step_stats["training_seconds"] = 0.0
        disabled_reason = str(getattr(
            cfg, "_training_disabled_reason", "--no-train"))
        print(f"[step {step_idx}] training disabled ({disabled_reason}); "
              "skipping training-only logprobs, backward, and optimizer update",
              flush=True)
        return step_stats

    if not all_examples:
        print(f"[step {step_idx}] no usable training examples")
        return step_stats

    # A model-parallel trainer can remain on CPU throughout every vLLM phase
    # and all reward processing. Restore its shards only when this step has a
    # real calibration/backward update to perform.
    if ensure_trainer_ready is not None:
        ensure_trainer_ready()

    if x_grpo_mode:
        calibration_t0 = time.time()
        group_ids_by_context = {}
        for group_id, group_data in x_grpo_groups.items():
            group_ids_by_context.setdefault(
                int(group_data["context"]), []).append(int(group_id))
        for context_id in sorted(group_ids_by_context):
            group_ids_by_context[context_id] = tuple(sorted(
                group_ids_by_context[context_id]))
            if (len(group_ids_by_context[context_id])
                    != x_grpo_groups_per_context):
                raise RuntimeError(
                    f"X-GRPO context {context_id} has "
                    f"{len(group_ids_by_context[context_id])} groups; expected "
                    f"K={x_grpo_groups_per_context}")
        if (parallel_trainer is not None
                and hasattr(parallel_trainer, "calibrate_x_grpo")):
            calibration = parallel_trainer.calibrate_x_grpo(
                all_examples, cfg, step_idx,
                context_group_ids=group_ids_by_context)
        else:
            if (parallel_trainer is not None
                    and parallel_trainer.is_offloaded):
                parallel_trainer.restore_after_generation()
            calibration = _calibrate_x_grpo_local(
                backend, model, tokenizer, all_examples, cfg, step_idx,
                context_group_ids=group_ids_by_context)
        x_grpo_group_stats = _apply_x_grpo_calibration(
            all_examples, x_grpo_groups, calibration, cfg)
        selected_label = ", ".join(
            f"p{item['context']}/k{item['fold']}="
            f"{item['selected_budget']:g}"
            for item in x_grpo_group_stats)
        calibration_seconds = time.time() - calibration_t0
        print(f"[step {step_idx}] X-GRPO selected budgets: "
              f"{selected_label}  (calibration: {calibration_seconds:.1f}s)",
              flush=True)
        step_stats["x_grpo_groups"] = x_grpo_group_stats
        step_stats["x_grpo_calibration_distributed"] = bool(
            calibration.get("distributed", False))
        step_stats["x_grpo_diagnostic_quarantined_examples"] = int(
            calibration.get("diagnostic_quarantined_examples", 0))
        step_stats["x_grpo_contexts"] = len(group_ids_by_context)
        step_stats["x_grpo_groups_per_context"] = (
            x_grpo_groups_per_context)

    if clipped_policy_mode:
        if binary_overlap_state is not None:
            step_stats.update(_finish_binary_coder_overlap_update(
                model, optimizer, cfg, step_idx, binary_overlap_state))
        elif parallel_trainer is not None:
            step_stats.update(parallel_trainer.train_rank(
                all_examples, cfg, step_idx))
        else:
            step_stats.update(_train_rank_examples(
                backend, model, tokenizer, optimizer, all_examples, cfg,
                step_idx))
        if spo_rs_mode:
            step_stats["spo_rs_policy_updated"] = True
        return step_stats

    # ----- TRAIN STEP -----
    print(f"[step {step_idx}] starting adapter training; rollout artifacts "
          f"are already on disk", flush=True)
    if parallel_trainer is not None:
        if entropic_overlap_state is not None:
            step_stats.update(parallel_trainer.finish_entropic_overlap(
                cfg, step_idx, len(all_examples), apply_update=True))
            entropic_overlap_state = None
        else:
            step_stats.update(parallel_trainer.train_policy(
                all_examples, cfg, step_idx))
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
    sequence_policy = _uses_sequence_level_policy_ratio(cfg)
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
        for batch in active_batches:
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
                    raise RuntimeError(
                        "exact base-policy logprob fallback failed; "
                        "refusing to replace the configured KL penalty "
                        "with the current policy") from error
            current_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=True, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)

            batch_losses = []
            for ex, cur_lp, base_lp in zip(
                    batch, current_logprobs, base_logprobs):
                base_lp = base_lp.to(cur_lp.device)
                adv = ex["advantage"]
                logp_diff = (cur_lp - base_lp).detach()
                avg_logp_diff = logp_diff.mean()
                kl_adv = cfg.kl_penalty_coef * (
                    avg_logp_diff - (cur_lp - base_lp))
                eff_adv = adv + kl_adv


                behavior_lp = ex.get("behavior_logprobs")
                has_behavior = _valid_example_token_logprobs(
                    ex, "behavior_logprobs")
                if sequence_policy:
                    loss, policy_metrics = (
                        _a3b_sequence_clipped_standard_loss(
                            cfg, cur_lp,
                            behavior_lp if has_behavior else None,
                            base_lp, adv))
                    is_ratio = policy_metrics["ratio"]
                elif has_behavior:
                    is_ratio = _detached_behavior_importance_ratio(
                        cfg, cur_lp, behavior_lp)
                    loss = -(is_ratio * eff_adv.detach() * cur_lp).mean()
                else:
                    if (behavior_lp is not None
                            and not hasattr(train_step, "_is_len_warned")):
                        print("[warn] invalid behavior logprobs; skipping IS "
                              "for affected examples")
                        train_step._is_len_warned = True
                    is_ratio = 1.0
                    loss = -(eff_adv.detach() * cur_lp).mean()

                if has_behavior:
                    is_ratio_sum += float(is_ratio.mean().item())
                    is_ratio_max = max(
                        is_ratio_max, float(is_ratio.max().item()))
                    is_ratio_count += 1
                batch_losses.append(loss / n_examples)
                total_loss += float(loss.detach().item())
                total_logp_delta += float(logp_diff.mean().item())
            if batch_losses:
                sum(batch_losses[1:], batch_losses[0]).backward()
        return (total_loss, total_logp_delta, is_ratio_sum,
                is_ratio_max, is_ratio_count)

    result, _effective, quarantined = _run_oom_resilient_backward(
        model, microbatches, attempt, device_label=str(model.device))
    if result is None:
        result = (0.0, 0.0, 0.0, 0.0, 0)
    (total_loss, total_logp_delta, is_ratio_sum, is_ratio_max,
     is_ratio_count) = result

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

    best = sampler.best_state()
    if best is not None:
        raw = f" raw={best.raw_score:.9f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.9f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")

    return step_stats


# ======================================================================
# Main
# ======================================================================
def main():
    _install_console_timestamps()
    terminal_log = _install_terminal_log()
    from builtins import print as console_print
    from startup_logs import StartupLog, print_dashboard
    startup_log = StartupLog(time_offset=_LOG_TIME_OFFSET_SECONDS)
    with startup_log.capture():
        cfg, merged = load_config()
        from problems.binary_coder import (
            BINARY_CODER_MODE, BinaryCoderConfig)
        binary_coder_cfg = BinaryCoderConfig.from_mapping(merged)
        # This must precede every import path that can initialize CUDA. Worker
        # and evaluation children set their own physical groups before import.
        _pin_training_process(cfg.training_gpu_ids)
        if cfg.evaluation_gpu_id is not None:
            os.environ["TTT_EVALUATION_GPU_ID"] = str(cfg.evaluation_gpu_id)

        # Save the reusable base configuration before runtime-only adjustments.
        from experiment_io import (append_step_result, make_experiment_dir,
                                   save_final_summary, save_step_summary)
        resume_dir = merged.pop("_resume_dir", None)
        exp_dir = make_experiment_dir(
            cfg, resume_dir=resume_dir, config_dict=merged
        )
        setting_log_path = Path(exp_dir).resolve() / "setting.log"
        startup_log.title = (
            "Strategist Bandit" if cfg.strategies
            else "Single-Model Discovery")
        startup_log.bind(setting_log_path)
        bind_setting_log(
            setting_log_path, time_offset=_LOG_TIME_OFFSET_SECONDS)
    # Keep the full resolved settings in the file; show their grouped dashboard
    # once below. Restore normal print before loading/training begins.
    print = startup_log.print
    terminal_log_path = str(Path(exp_dir).resolve() / "temirnal.log")
    terminal_log.bind(terminal_log_path)
    print(f"[logs] terminal output: {terminal_log_path}", flush=True)
    result_log_path = Path(exp_dir).resolve() / "result.txt"
    result_log_path.touch(exist_ok=True)
    print(f"[logs] step results: {result_log_path}", flush=True)
    vllm_log_path = None
    strategy_vllm_log_path = None
    dependency_log_path = None
    if cfg.generation_backend == "vllm":
        vllm_log_path = str((Path(exp_dir).resolve() / "vllm.log"))
        with open(vllm_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== TTT vLLM log parent_pid={os.getpid()} "
                f"run_dir={Path(exp_dir).resolve()} ===\n")
        print(f"[logs] vLLM output: {vllm_log_path}", flush=True)
        if (cfg.strategies
                and getattr(cfg, "strategy_backend", "local") == "local"
                and (getattr(cfg, "strategy_model_name", cfg.model_name)
                     != cfg.model_name
                     or getattr(
                         cfg,
                         "dual_resident_qwen3_8b_strategy_coder_pools",
                         False))):
            strategy_vllm_log_path = str(
                Path(exp_dir).resolve() / "strategy_vllm.log")
            with open(strategy_vllm_log_path, "a",
                      encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n=== TTT strategy vLLM log parent_pid={os.getpid()} "
                    f"run_dir={Path(exp_dir).resolve()} ===\n")
            print(f"[logs] strategy vLLM output: "
                  f"{strategy_vllm_log_path}", flush=True)
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

    # Replicate the complete trainer whenever the profiled trainable copy fits
    # on one card, rather than layer-sharding one update across every card. The
    # main process still owns search/evaluation; only per-example gradient work
    # runs concurrently. vLLM phase sharing gives these replicas exclusive use
    # of the cards during training and requires them to offload for generation.
    replicated_backend_supported = bool(
        cfg.backend == "hf"
        or (cfg.backend == "unsloth"
            and getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b")
    )
    use_replicated_training = bool(
        not cfg.no_train
        and int(cfg.num_training_gpus) > 1
        and replicated_backend_supported
        and cfg.generation_backend == "vllm"
        and cfg.training_layout != "sharded"
    )
    # Standard entropic/GRPO/CVaR updates use isolated GPU processes even
    # without --fast.  The former same-interpreter threaded replicas are not a
    # safe execution boundary for concurrent autograd.  --fast remains an
    # optional scheduling choice: it enables adaptive batches; without it the
    # configured microbatch ceiling is honored.  Clipped-policy modes retain
    # their existing default trainer unless --fast is explicitly requested.
    standard_policy_objective = not bool(
        binary_coder_cfg.enabled
        or _uses_clipped_policy_loss(cfg.advantage_mode))
    use_process_distributed_training = bool(
        use_replicated_training
        and (cfg.fast or standard_policy_objective
             or getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b")
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
    # Strategy generation is a launch mode, not a permanent property of a
    # problem class. Without --strategies every problem follows its direct
    # one-stage policy rollout path.
    problem.two_stage_rollouts = bool(cfg.strategies)

    startup_log.begin_summary()
    print("=" * 70)
    print("Strategist Bandit" if cfg.strategies
          else "Single-Model Discovery")
    print("=" * 70)
    problem_type = getattr(cfg, "problem_type", "")
    print(f"Problem:            {cfg.problem}"
          + (f" ({problem_type})" if problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Model:              {cfg.model_name}")
    if getattr(problem, "two_stage_rollouts", False):
        strategy_location = (
            "remote API; no local weights"
            if cfg.strategy_backend == "api" else "local; no LoRA")
        print(f"Strategy model:     {cfg.strategy_model_name} "
              f"({strategy_location})")
    if cfg.training_model_name != cfg.model_name:
        print(f"Training model:     {cfg.training_model_name}")
    print(f"Training backend:   {cfg.backend}")
    print(f"Generation backend: {cfg.generation_backend}")
    print(f"Training GPUs:      physical {cfg.training_gpu_ids}")
    training_layout = (
        "disabled (--no-train)" if cfg.no_train
        else ("process-distributed LoRA"
              if use_process_distributed_training
              else ("replicated data parallel" if use_replicated_training
                    else "model parallel/single GPU"))
    )
    training_scheduler = (
        "disabled (--no-train)" if cfg.no_train
        else ("adaptive process-per-GPU (--fast)"
              if use_process_distributed_training and cfg.fast
              else ("configured process-per-GPU"
                    if use_process_distributed_training
              else ("concurrent length-balanced replicas (--fast)"
                    if cfg.fast and use_replicated_training else "default")))
    )
    print(f"Training layout:    {training_layout}")
    print(f"Training scheduler: {training_scheduler}")
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
    print(f"Search selection:   {'UCT' if cfg.uct else 'PUCT'} "
          f"(c={cfg.puct_c})")
    configured_advantage_mode = getattr(cfg, "advantage_mode", "entropic")
    if configured_advantage_mode == "x-grpo":
        print(f"X-GRPO contexts P:  {cfg.x_grpo_contexts_per_step}")
        print(f"X-GRPO groups K:    {cfg.groups_per_step} per context")
        print(f"X-GRPO group size G: {cfg.group_size}")
        print(f"Total rollouts/step: "
              f"{cfg.x_grpo_contexts_per_step * cfg.groups_per_step * cfg.group_size}")
    else:
        print(f"Groups per step:    {cfg.groups_per_step}")
        print(f"Group size:         {cfg.group_size}")
        print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    if binary_coder_cfg.enabled:
        print("Reference KL:        off (binary coder PPO objective)")
    elif configured_advantage_mode == "spo-rs":
        print("Reference KL:        off (SPO-RS PPO objective)")
    else:
        print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Advantage mode:     "
          f"{'binary-coder' if binary_coder_cfg.enabled else getattr(cfg, 'advantage_mode', 'entropic')}")
    if _uses_sequence_level_policy_ratio(cfg):
        print("MoE policy ratio:    sequence-level geometric mean "
              "(Qwen3-30B-A3B only; existing asymmetric low/high bounds)")
    if binary_coder_cfg.enabled:
        print(f"Binary coder phase: steps 0-{binary_coder_cfg.init_steps - 1}; "
              f"verified +/-1 advantages, LoRA rank "
              f"{binary_coder_cfg.lora_rank}")
        print(f"Binary coder clip:  low/high "
              f"{binary_coder_cfg.clip_epsilon_low} / "
              f"{binary_coder_cfg.clip_epsilon_high} "
              f"(bounds {1.0 - binary_coder_cfg.clip_epsilon_low:.4f} / "
              f"{1.0 + binary_coder_cfg.clip_epsilon_high:.4f}); "
              "adapter remains active and freezes afterward")
    if (not binary_coder_cfg.enabled
            and getattr(cfg, "advantage_mode", "entropic") == "cvar"):
        print(f"CVaR alpha/lambda:  {cfg.cvar_alpha} / {cfg.cvar_lambda}")
    if not binary_coder_cfg.enabled:
        if configured_advantage_mode == "spo-rs":
            print(f"SPO-RS beta:        {cfg.spo_rs_beta}")
            print(f"SPO-RS D_half:      {cfg.spo_rs_d_half}")
            print(f"SPO-RS rho min/max: {cfg.spo_rs_rho_min} / {cfg.spo_rs_rho_max}")
            print(f"SPO-RS clip eps low/high: "
                  f"{cfg.spo_rs_clip_epsilon_low} / "
                  f"{cfg.spo_rs_clip_epsilon_high} "
                  f"(bounds {1.0 - cfg.spo_rs_clip_epsilon_low:.4f} / "
                  f"{1.0 + cfg.spo_rs_clip_epsilon_high:.4f})")
            print(f"SPO-RS update epochs: {cfg.rank_update_epochs}")
        elif _uses_clipped_policy_loss(configured_advantage_mode):
            print(f"Rank clip eps low/high: "
                  f"{cfg.rank_clip_epsilon_low} / {cfg.rank_clip_epsilon_high} "
                  f"(bounds {1.0 - cfg.rank_clip_epsilon_low:.4f} / "
                  f"{1.0 + cfg.rank_clip_epsilon_high:.4f})")
            print(f"Rank update epochs: {cfg.rank_update_epochs}")
    if not binary_coder_cfg.enabled and configured_advantage_mode == "rank":
        print(f"Rank gamma:         {cfg.rank_gamma}")
        print(f"Rank entropy coef:  {cfg.rank_entropy_coef} (fully tied groups)")
    elif (not binary_coder_cfg.enabled
          and configured_advantage_mode == "x-grpo"):
        print(f"X-GRPO budgets:     {list(cfg.x_grpo_budgets)}")
        print(f"X-GRPO rel. error:  {cfg.x_grpo_relative_error}")
        print(f"X-GRPO entropy:     {cfg.x_grpo_entropy_coef}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    if getattr(problem, "two_stage_rollouts", False):
        pilot_count = int(getattr(cfg, "pilot_programs_per_strategy", -1))
        if (getattr(cfg, "coder_template_kind", "generic") == "qwen3.8"
                and pilot_count > 0):
            print(f"Coder reasoning:    pilot={pilot_count - 1} medium + 1 xhigh "
                  "per parent/strategy; phase 2=medium")
        else:
            print(f"Coder reasoning:    pilot/base={_coder_effort_for_rollout_phase(cfg)}, "
                  f"phase 2={_coder_effort_for_rollout_phase(cfg, 'adaptive')}")
    else:
        direct_effort = _coder_effort_for_rollout_phase(cfg)
        if direct_effort is not None:
            print(f"Coder reasoning:    direct={direct_effort}")
    print(
        f"Coder sampling:     temperature={cfg.temperature}, "
        f"top_p={cfg.top_p}, "
        f"top_k={getattr(cfg, 'sampling_top_k', None)}, "
        f"min_p={getattr(cfg, 'sampling_min_p', None)}, "
        f"thinking={'on' if cfg.thinking else 'off'}")
    if getattr(problem, "two_stage_rollouts", False):
        print(f"Rollout hierarchy:  {cfg.strategies_per_parent} sequential "
              f"strategies/parent x {cfg.programs_per_strategy} "
              f"programs/strategy = {cfg.group_size}/parent")
        if int(getattr(cfg, "pilot_programs_per_strategy", -1)) == -1:
            print("Rollout pilot:      off (-1; unchanged one-pass generation)")
        else:
            allocation_method = str(getattr(
                cfg, "phase2_allocation_method", "rule_based"))
            allocation_label = (
                "hurdle expected-best allocation"
                if allocation_method == "hurdle" else
                "posterior expected-best bandit"
                if allocation_method == "bandit"
                else "dynamic per-parent median mean/variance rule"
            )
            print(
                f"Rollout pilot:      "
                f"{int(cfg.pilot_programs_per_strategy)}/strategy, then "
                f"{allocation_label}; fixed "
                f"total={cfg.group_size}/parent")
        print(f"Strategy archive:   top {cfg.strategy_archive_top_r}/strategy "
              f"then existing top {cfg.topk_children_per_parent}/parent")
        coder_retry_label = (
            "once (extra training example)"
            if bool(getattr(cfg, "coder_retry", False)) else
            "off (enable with --coder-retry)"
        )
        print(f"Output retries:     strategy up to {cfg.strategy_format_max_retries} "
              f"(stop on exhaustion); coder {coder_retry_label}")
        print(f"Strategy sampling:  max_new={cfg.strategy_max_new_tokens}, "
              f"max_seq={cfg.strategy_max_seq_length}, "
              f"temperature={cfg.strategy_temperature}, "
              f"top_p={cfg.strategy_top_p}, "
              f"top_k={getattr(cfg, 'strategy_sampling_top_k', None)}, "
              f"min_p={getattr(cfg, 'strategy_sampling_min_p', None)}, "
              f"thinking={'on' if cfg.strategy_thinking else 'off'}, "
              f"reasoning_effort={cfg.strategy_reasoning_effort}")
    print(f"Max seq length:     {cfg.max_seq_length}")
    if getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b":
        fused_attention_label = (
            "GPT-OSS sink-aware Unsloth path "
            "(generic Flash-SDPA prohibited)")
    elif getattr(cfg, "coder_model_profile", "") == "qwen3.8-27b":
        fused_attention_label = (
            "on for full-attention layers; exact FLA for Gated DeltaNet")
    elif (getattr(cfg, "coder_model_profile", "")
          == "qwen3-30b-a3b-thinking-2507"):
        fused_attention_label = "on (exact Qwen full-attention path)"
    else:
        fused_attention_label = (
            "on" if cfg.fused_long_attention else "off")
    print(f"Fused long attention: {fused_attention_label}")
    print(f"Train microbatch:   up to "
          f"{cfg.train_examples_per_microbatch} examples/GPU")
    print(f"LoRA:               rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, "
          f"dropout={cfg.lora_dropout}")
    print(f"Deterministic:      {cfg.deterministic}")
    if use_process_distributed_training:
        print(f"Training memory cap: {100.0 * float(cfg.training_memory_fraction):g}% "
              "per GPU (96% exact long-rollout rescue)")
    print(f"Logprob chunk:      {cfg.logprob_chunk or 'off (single shot)'}")
    print(f"Seed:               {cfg.seed}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print("=" * 70)

    # ---- experiment dir ----
    startup_log.end_summary()
    action = "resuming in" if resume_dir else "writing all rollouts to"
    print(f"[init] {action}: {exp_dir}")

    # ---- seed states (problem-defined) ----
    seeds = problem.seed_states()
    print(f"[init] problem produced {len(seeds)} seed state(s)")
    print = console_print
    print_dashboard(startup_log.fields, exp_dir, resuming=bool(resume_dir))

    # ---- backend + model ----
    # Load backend FIRST so Unsloth can patch transformers if used.
    with startup_log.capture(loading=True), _route_dependency_notices(dependency_log_path):
        from model_backend import load_backend
        backend = load_backend(cfg.backend, cfg)
        model, tokenizer = backend.load()
    strategy_tokenizer = tokenizer
    if (getattr(problem, "two_stage_rollouts", False)
            and cfg.strategy_backend == "local"
            and cfg.strategy_model_name != cfg.model_name):
        from transformers import AutoTokenizer
        with _route_dependency_notices(dependency_log_path):
            strategy_tokenizer = AutoTokenizer.from_pretrained(
                cfg.strategy_model_name, trust_remote_code=True)
        if strategy_tokenizer.pad_token_id is None:
            strategy_tokenizer.pad_token = strategy_tokenizer.eos_token
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
    current_policy_adapter_path = None
    if resume_dir:
        resume_payload = _load_training_checkpoint(exp_dir)
        if resume_payload is not None:
            start_step = int(resume_payload["next_step"])
            adapter_path = Path(exp_dir) / resume_payload["adapter_dir"]
            _load_adapter(model, adapter_path)
            current_policy_adapter_path = adapter_path
            print(f"[resume] training checkpoint found; next step is {start_step}")
            if resume_payload["version"] == 1:
                print("[resume] legacy batch-growth state ignored; "
                      "using fixed configured batch sizes")
        else:
            legacy_resume = _legacy_resume_info(exp_dir)
            start_step, adapter_path = legacy_resume
            _load_adapter(model, adapter_path)
            current_policy_adapter_path = adapter_path
            print(
                f"[resume] legacy run (no training_state.pt): restarting step "
                f"{start_step} from {adapter_path.name}. The archive will be "
                "reconstructed, but old PUCT/optimizer statistics are "
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
        if use_process_distributed_training:
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
        use_uct=cfg.uct,
    )
    if resume_payload is not None:
        sampler.load_state_dict(resume_payload["sampler"])
        print(f"[resume] restored exact "
              f"{'UCT' if cfg.uct else 'PUCT'} archive and visit statistics")
    elif legacy_resume is not None:
        n_states, n_rollouts = _restore_legacy_archive(
            sampler, exp_dir, before_step=start_step
        )
        print(f"[resume] reconstructed {n_states} valid archived candidates "
              f"from {n_rollouts} earlier rollouts")
    setting_log_only(
        f"[init] sampler archive size = {sampler.archive_size()}", flush=True)

    spo_rs_tracker = None
    if configured_advantage_mode == "spo-rs" and not binary_coder_cfg.enabled:
        from spo_rs import SPORSTracker
        spo_rs_tracker = SPORSTracker(
            entropic_beta=cfg.spo_rs_beta,
            d_half=cfg.spo_rs_d_half,
            rho_min=cfg.spo_rs_rho_min,
            rho_max=cfg.spo_rs_rho_max,
        )
        tracker_state = (resume_payload.get("spo_rs_tracker")
                         if resume_payload is not None else None)
        if tracker_state is not None and tracker_state.get("version") == 2:
            spo_rs_tracker.load_state_dict(tracker_state)
            print("[resume] restored the run-wide SPO-RS tracker")
        elif start_step:
            # Version 1 attached history to rendered prompts. It cannot be
            # merged into the requested global statistic, so initialize the
            # new tracker once from the most recent completed rollout batch.
            last_step = start_step - 1
            rewards = _logged_step_rewards(exp_dir, last_step)
            if not rewards:
                raise ValueError(
                    "cannot initialize the run-wide SPO-RS tracker: no finite "
                    f"rewards were logged for completed step {last_step}")
            prior_sampling_adapter = Path(exp_dir) / (
                "adapter_step000" if last_step == 0
                else f"adapter_step{last_step - 1:03d}")
            if not prior_sampling_adapter.is_dir():
                raise FileNotFoundError(
                    "cannot initialize the run-wide SPO-RS tracker: the "
                    f"sampling adapter for step {last_step} is missing at "
                    f"{prior_sampling_adapter}")
            transformed = spo_rs_tracker.transformed_rewards(rewards)
            spo_rs_tracker.update(
                transformed, divergence=0.0,
                policy_adapter=prior_sampling_adapter,
                step=last_step)
            source = ("legacy prompt trackers" if tracker_state is not None
                      else "logged rollout history")
            print(f"[resume] replaced {source} with one run-wide SPO-RS "
                  f"tracker initialized from {len(rewards)} rewards in step "
                  f"{last_step}")
        else:
            # Step 0 needs a baseline that exists before its responses are
            # sampled. The verified seed archive is already available history
            # on exactly the problem reward scale, so use it once rather than
            # leaking step-0 rewards into their own advantages or suppressing
            # the first policy update.
            seed_rewards = []
            for seed in seeds:
                try:
                    reward = float(seed.value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(reward):
                    seed_rewards.append(reward)
            if not seed_rewards:
                raise ValueError(
                    "a fresh SPO-RS run requires at least one finite verified "
                    "seed reward to initialize its pre-step baseline")
            initial_adapter = Path(_save_adapter(
                model, exp_dir, -1, cfg.model_name,
                directory_name="adapter_initial"))
            transformed = spo_rs_tracker.transformed_rewards(seed_rewards)
            initialization = spo_rs_tracker.update(
                transformed, divergence=0.0,
                policy_adapter=initial_adapter, step=-1)
            current_policy_adapter_path = initial_adapter
            print("[init] SPO-RS global tracker initialized before step 0 "
                  f"from {len(seed_rewards)} verified seed rewards: "
                  f"v={initialization['value_after']:.9f}, "
                  f"N_eff={initialization['effective_count_after']:.2f}; "
                  "step 0 policy update enabled", flush=True)

    removed_adapters = _prune_adapter_snapshots(
        exp_dir,
        _live_adapter_snapshots(
            exp_dir, current_policy_adapter_path, spo_rs_tracker),
    )
    if removed_adapters:
        print(f"[checkpoint] removed {len(removed_adapters)} superseded "
              "adapter snapshot(s) while restoring the rolling checkpoint",
              flush=True)

    # ---- generation pool ----
    gen_pool = None
    strategy_pool = None
    ensure_trainer_ready = None
    # Unlike cfg.no_train, this can change after the binary-coder bootstrap.
    # Generation callbacks consult it so all later steps follow the genuine
    # no-training residency path instead of waiting for a nonexistent update.
    runtime_training_active = {"value": not bool(cfg.no_train)}
    if (getattr(problem, "two_stage_rollouts", False)
            and cfg.strategy_backend == "api"):
        from strategy_api import StrategyAPIGenerationPool
        strategy_pool = StrategyAPIGenerationPool(
            model_name=cfg.strategy_model_name,
            base_url=cfg.strategy_api_base_url,
            api_key_env=cfg.strategy_api_key_env,
            concurrency=cfg.strategy_api_concurrency,
            timeout_s=cfg.strategy_api_timeout_s,
            max_retries=cfg.strategy_api_max_retries,
            thinking=cfg.strategy_thinking,
            reasoning_effort=cfg.strategy_reasoning_effort,
        )
        print(f"[init] remote strategy API configured for "
              f"{cfg.strategy_model_name}; no strategist tokenizer, weights, "
              "or local generation pool loaded", flush=True)
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
        dual_resident_qwen3_8b_pools = bool(getattr(
            cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False))
        separate_strategy_pool = bool(
            cfg.generation_backend == "vllm"
            and getattr(problem, "two_stage_rollouts", False)
            and cfg.strategy_backend == "local"
            and bool(getattr(
                cfg, "separate_strategy_inference_pool",
                cfg.strategy_model_name != cfg.model_name
                or dual_resident_qwen3_8b_pools)))
        generation_utilization = float(cfg.vllm_gpu_memory_utilization)
        if dual_resident_qwen3_8b_pools:
            # Each physical card hosts two independent Qwen3-8B vLLM
            # processes. Bound each allocator so strategist + coder + CUDA
            # runtime fit together while the differentiable trainer is off.
            generation_utilization = min(generation_utilization, 0.40)
            print("[memory] dual Qwen3-8B pools: limiting each vLLM engine "
                  "to 40% GPU memory for concurrent residency", flush=True)
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
            lora_rank=(binary_coder_cfg.lora_rank
                       if binary_coder_cfg.enabled else cfg.lora_rank),
            vllm_gpu_memory_utilization=generation_utilization,
            vllm_enforce_eager=cfg.vllm_enforce_eager,
            vllm_enable_prefix_caching=cfg.vllm_enable_prefix_caching,
            vllm_quantization=cfg.vllm_quantization,
            vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
            vllm_pipeline_parallel_size=cfg.vllm_pipeline_parallel_size,
            vllm_max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
            vllm_enable_expert_parallel=cfg.vllm_enable_expert_parallel,
            vllm_sleep_level=cfg.vllm_sleep_level,
            # Distinct strategist/coder pools share the same cards. Account
            # for either a sleeping pool's residual CUDA/NCCL state or the
            # Qwen3-8B ablation's concurrently awake peer pool.
            vllm_co_resident_sleep=separate_strategy_pool,
            # Training modes consume exact chosen-token prompt logprobs after
            # generation. Give that vocabulary projection explicit memory
            # headroom; this changes scheduling only, never the scores.
            vllm_token_scoring=not bool(cfg.no_train),
            vllm_runtime_reserve_gib=float(
                cfg.vllm_runtime_reserve_gib),
            vllm_staged_loading=cfg.vllm_staged_loading,
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
                replica_label = (
                    "inactive training model"
                    if not runtime_training_active["value"]
                    else (f"all {parallel_trainer.world_size} trainer replicas"
                          if parallel_trainer is not None else "trainer")
                )
                setting_log_only(
                    f"[gpu] offloading {replica_label} to CPU before "
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
                if not runtime_training_active["value"]:
                    print("[gpu] keeping inactive training model offloaded "
                          "(no-training phase)", flush=True)
                    return
                if dual_resident_qwen3_8b_pools and any(
                        pool is not None and getattr(pool, "active", False)
                        for pool in (gen_pool, strategy_pool)):
                    print("[gpu] keeping trainer offloaded while the other "
                          "Qwen3-8B generation pool remains active",
                          flush=True)
                    return
                if (parallel_trainer is not None
                        or cfg.training_layout == "sharded"):
                    # Keep the trainer on CPU through generation and restore it
                    # once, lazily, when the actual gradient update begins.
                    trainer_label = ("trainer replicas"
                                     if parallel_trainer is not None
                                     else "sharded trainer")
                    print(f"[gpu] keeping {trainer_label} offloaded until the "
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

            def _ensure_sharded_trainer_ready():
                nonlocal trainer_offloaded
                if not runtime_training_active["value"]:
                    raise RuntimeError(
                        "trainer restore requested during a no-training phase")
                if not trainer_offloaded:
                    return
                print("[gpu] restoring sharded trainer for adapter update",
                      flush=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                backend.restore_after_generation()
                _restore_optimizer_state_to_parameters(optimizer)
                backend.set_training_mode()
                trainer_offloaded = False

            if cfg.training_layout == "sharded":
                ensure_trainer_ready = _ensure_sharded_trainer_ready

            gen_pool = PhasedVLLMGenerationPool(
                before_start=_offload_trainer_for_generation,
                after_stop=_restore_trainer_after_generation,
                **pool_options,
            )
            setting_log_only(
                f"[init] phase-shared vLLM pool configured across all "
                f"rollout GPUs {gpu_ids}", flush=True)
            if separate_strategy_pool:
                strategy_pool_options = dict(pool_options)
                strategy_pool_options.update({
                    "model_name": cfg.strategy_model_name,
                    "max_seq_length": cfg.strategy_max_seq_length,
                    "load_in_4bit": False,
                    "gen_micro_batch": getattr(
                        cfg, "strategy_gen_micro_batch",
                        cfg.gen_micro_batch),
                    "vllm_quantization": (
                        cfg.strategy_vllm_quantization),
                    "vllm_tensor_parallel_size": (
                        cfg.strategy_vllm_tensor_parallel_size),
                    "vllm_pipeline_parallel_size": (
                        cfg.strategy_vllm_pipeline_parallel_size),
                    "vllm_max_num_batched_tokens": getattr(
                        cfg, "strategy_vllm_max_num_batched_tokens",
                        cfg.vllm_max_num_batched_tokens),
                    "vllm_sleep_level": cfg.strategy_vllm_sleep_level,
                    "vllm_token_scoring": False,
                    "vllm_runtime_reserve_gib": float(getattr(
                        cfg, "strategy_vllm_runtime_reserve_gib",
                        16.0)),
                    "vllm_persistent_workers": getattr(
                        cfg, "strategy_vllm_persistent_workers", None),
                    "vllm_staged_loading": (
                        cfg.strategy_vllm_staged_loading),
                    "vllm_log_path": strategy_vllm_log_path,
                })
                strategy_pool = PhasedVLLMGenerationPool(
                    before_start=_offload_trainer_for_generation,
                    after_stop=_restore_trainer_after_generation,
                    **strategy_pool_options,
                )
                setting_log_only(
                    f"[init] separate no-LoRA strategy vLLM pool "
                    f"configured for {cfg.strategy_model_name}", flush=True)
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
                                   adapter_path,
                                   max_new_tokens, temperature, top_p,
                                   step_idx, top_k=None, min_p=None):
                backend.set_inference_mode()
                if run_seed is not None:
                    local_seed = worker_seed(run_seed, step_idx, 0)
                    torch.manual_seed(local_seed)
                    torch.cuda.manual_seed_all(local_seed)
                try:
                    adapter_context = (
                        backend.disable_adapter()
                        if adapter_path is None else nullcontext()
                    )
                    with adapter_context:
                        yield from generate_prompt_jobs(
                            model, tokenizer, prompts_by_group,
                            counts_by_group, cfg,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            top_k=top_k, min_p=min_p,
                            cap_state=local_cap)
                finally:
                    backend.set_training_mode()

            gen_pool = HybridHFGenerationPool(
                remote_pool=remote_pool, local_iter=_local_hf_rollouts)
            print(f"[init] hybrid HF rollout pool: live trainer on "
                  f"physical GPU {gpu_ids[0]} plus persistent workers "
                  f"{gpu_ids[1:]}")
        if (getattr(problem, "two_stage_rollouts", False)
                and strategy_pool is None):
            strategy_pool = gen_pool
        elif dual_resident_qwen3_8b_pools:
            print(f"[init] dual-resident Qwen3-8B pools: strategist and coder "
                  f"use independent GPU-complete inference topologies; "
                  "strategist is always base/no-LoRA, coder receives the "
                  "current LoRA", flush=True)
        if cfg.gen_micro_batch and cfg.gen_micro_batch > 0:
            limit_name = ("max_num_seqs" if cfg.generation_backend == "vllm"
                          else "micro-batch")
            limit_scope = ("/engine" if cfg.generation_backend == "vllm"
                           else "/GPU")
            setting_log_only(
                f"[init] generation pool ready "
                f"({limit_name} {cfg.gen_micro_batch}{limit_scope})",
                flush=True)
        else:
            setting_log_only("[init] generation pool ready", flush=True)
    else:
        if cfg.num_training_gpus > 1:
            print(f"[init] model-parallel HF generation across training GPUs "
                  f"{cfg.training_gpu_ids}; prompts remain cross-batched")
        else:
            print("[init] single-GPU generation (no worker pool)")


    # ---- main loop ----
    from experiment_io import StepPlotter
    step_plotter = StepPlotter(exp_dir, cfg.problem)
    cumulative_training_seconds = 0.0
    for completed_step in range(start_step):
        summary_path = (Path(exp_dir) / f"step{completed_step:02d}"
                        / f"step{completed_step:02d}.summary.json")
        try:
            completed_summary = json.loads(summary_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if "result_metric_name" in completed_summary:
            append_step_result(
                exp_dir, completed_step, completed_summary)
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
            step_cfg = cfg
            if binary_coder_cfg.enabled:
                step_cfg = SimpleNamespace(**vars(cfg))
                step_cfg.advantage_mode = BINARY_CODER_MODE
                if binary_coder_cfg.active(step):
                    step_cfg.no_train = bool(cfg.no_train)
                    print(f"[step {step}] binary coder initialization "
                          f"{step + 1}/{binary_coder_cfg.init_steps}: "
                          "training the verified-usability adapter", flush=True)
                else:
                    step_cfg.no_train = True
                    step_cfg._training_disabled_reason = (
                        "binary coder initialization complete; full "
                        "--no-train execution with frozen adapter")
                    print(f"[step {step}] binary coder initialization is "
                          "complete; using --no-train execution with the "
                          "frozen coder adapter", flush=True)
            runtime_training_active["value"] = not bool(step_cfg.no_train)
            if configured_advantage_mode == "x-grpo":
                rollout_count = (
                    int(cfg.x_grpo_contexts_per_step)
                    * cfg.groups_per_step * cfg.group_size)
                hierarchy = (
                    f"{cfg.strategies_per_parent} strategies x "
                    f"{cfg.programs_per_strategy} programs/group; "
                    if getattr(problem, "two_stage_rollouts", False) else "")
                setting_log_only(
                    f"[step {step}] batch: "
                    f"P={cfg.x_grpo_contexts_per_step} "
                    f"K={cfg.groups_per_step} G={cfg.group_size} "
                    f"({hierarchy}{rollout_count} rollouts)", flush=True)
            elif getattr(problem, "two_stage_rollouts", False):
                setting_log_only(
                    f"[step {step}] batch: parents={cfg.groups_per_step} "
                    f"strategies/parent={cfg.strategies_per_parent} "
                    f"programs/strategy={cfg.programs_per_strategy} "
                    f"({cfg.groups_per_step * cfg.group_size} code rollouts)",
                    flush=True)
            else:
                setting_log_only(
                    f"[step {step}] batch: G={cfg.groups_per_step} "
                    f"K={cfg.group_size} "
                    f"({cfg.groups_per_step * cfg.group_size} rollouts)",
                    flush=True)

            stats = train_step(backend, model, tokenizer, sampler, optimizer, step,
                               step_cfg, exp_dir, problem, gen_pool,
                               strategy_pool=strategy_pool,
                               strategy_tokenizer=strategy_tokenizer,
                               parallel_trainer=parallel_trainer,
                               spo_rs_tracker=spo_rs_tracker,
                               sampling_adapter_path=(
                                   current_policy_adapter_path
                                   if (binary_coder_cfg.enabled
                                       or configured_advantage_mode == "spo-rs")
                                   else None),
                               ensure_trainer_ready=ensure_trainer_ready)

            stats = stats or {}
            entropy_observations = stats.pop("_entropy_observations", [])
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

            # Version the adapter first, then atomically advance the state
            # pointer. A crash during either write leaves the previous pair
            # valid.
            if binary_coder_cfg.enabled:
                if (runtime_training_active["value"]
                        or current_policy_adapter_path is None):
                    adapter_path = _save_adapter(
                        model, exp_dir, step, cfg.model_name)
                else:
                    # No weights change after the bootstrap. Reuse its final
                    # adapter rather than writing an identical copy per step.
                    adapter_path = Path(current_policy_adapter_path)
            elif configured_advantage_mode == "spo-rs":
                if stats.get("spo_rs_policy_updated", False):
                    adapter_path = _save_adapter(
                        model, exp_dir, step, cfg.model_name)
                else:
                    adapter_path = (Path(exp_dir)
                                    / stats["spo_rs_sampling_adapter"])
            else:
                adapter_path = _save_adapter(
                    model, exp_dir, step, cfg.model_name)
            checkpoint_path = _save_training_checkpoint(
                exp_dir, step + 1, adapter_path, sampler, optimizer,
                spo_rs_tracker=spo_rs_tracker,
            )
            step_summary = {
                "step": step,
                "completed": True,
                "next_step": step + 1,
                "adapter_dir": Path(adapter_path).name,
                "checkpoint": Path(checkpoint_path).name,
                "archive_size": sampler.archive_size(),
                "groups_per_step": int(cfg.groups_per_step),
                "group_size": int(cfg.group_size),
                **(stats or {}),
            }
            save_step_summary(exp_dir, step, step_summary)
            append_step_result(exp_dir, step, step_summary)
            if getattr(step_cfg, "measure_entropy", False):
                from entropy_tools import save_step as save_entropy_step
                try:
                    entropy_summary = save_entropy_step(
                        exp_dir, step, entropy_observations, step_cfg, stats)
                    measured = entropy_summary["all"]
                    value = measured["entropy_nats"]
                    label = f"{value:.6f} nats" if value is not None else "unavailable"
                    print(f"[step {step}] coder entropy: {label}; "
                          f"measured {measured['measured_rollouts']}/"
                          f"{measured['rollouts']} rollouts "
                          f"({measured['measured_tokens']}/"
                          f"{measured['response_tokens']} response tokens); "
                          + ("updated entropy.jsonl, entropy.pdf, "
                             "strategy_diversity.svg"
                             if bool(getattr(step_cfg, "strategies", False))
                             else "updated entropy.jsonl, entropy.pdf"),
                          flush=True)
                except (OSError, ValueError, TypeError) as error:
                    print(f"[warn] entropy diagnostics could not be saved: "
                          f"{error}; training checkpoint is already saved",
                          flush=True)
            current_policy_adapter_path = Path(adapter_path)
            removed_adapters = _prune_adapter_snapshots(
                exp_dir,
                _live_adapter_snapshots(
                    exp_dir, current_policy_adapter_path, spo_rs_tracker),
            )
            if removed_adapters:
                retained = (
                    "two rolling adapters for consecutive-policy KL"
                    if (spo_rs_tracker is not None
                        and len(_live_adapter_snapshots(
                            exp_dir, current_policy_adapter_path,
                            spo_rs_tracker)) > 1)
                    else Path(adapter_path).name
                )
                print(f"[checkpoint] removed {len(removed_adapters)} "
                      f"superseded adapter snapshot(s); retained {retained}",
                      flush=True)
            print(f"[checkpoint] completed step {step}; resume at step {step + 1}")
            step_plotter.submit(step)
    finally:
        # Plotting uses no GPUs and never delays generation of the next step.
        # Finish queued figures on normal completion; cancel pending work when
        # the training loop exits with an exception or interruption.
        step_plotter.close(wait=sys.exc_info()[0] is None)
        if strategy_pool is not None and strategy_pool is not gen_pool:
            print("[shutdown] stopping strategy generation pool ...")
            strategy_pool.shutdown()
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
    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()
