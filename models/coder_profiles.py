"""Coder/strategist defaults and exact per-phase prompt controls."""

import json
from pathlib import Path

from models.strategist_models import strategist_profile


_QWEN3_8B_MODEL_ID = "qwen/qwen3-8b"
_QWEN3_8B_MODEL_BASENAME = "qwen3-8b"
_QWEN3_30B_A3B_MODEL_BASENAME = "qwen3-30b-a3b"
_GPT_OSS_120B_MODEL_BASENAME = "gpt-oss-120b"
_QWEN3_30B_A3B_THINKING_MODEL_BASENAME = ("qwen3-30b-a3b-thinking-2507")
_QWEN38_27B_MODEL_BASENAME = "qwen3.8-27b"
_DEEPSEEK_R1_0528_QWEN3_8B_MODEL_BASENAME = ("deepseek-r1-0528-qwen3-8b")


def _is_exact_qwen3_8b(model_name):
    return (str(model_name or "").strip().rstrip("/").lower() == _QWEN3_8B_MODEL_ID)


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
        merged["strategy_sampling_min_p"] = float(merged["strategy_min_p"])
    else:
        merged["strategy_sampling_min_p"] = 0.0
    merged["qwen3_8b_thinking_sampling"] = False
    merged["strategy_qwen3_8b_thinking_sampling"] = False

    if (_is_qwen3_8b_sampling_model(merged.get("model_name")) and bool(merged.get("thinking", False))):
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
            and _is_qwen3_8b_sampling_model(merged.get("strategy_model_name"))
            and bool(merged.get("strategy_thinking", False))):
        _profile_default(merged, "strategy_temperature", 1.0, explicit_keys)
        _profile_default(merged, "strategy_top_p", 1.0, explicit_keys)
        if ("strategy_min_p" not in explicit_keys and "strategy_sampling_min_p" not in explicit_keys):
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
    return normalized.rsplit("/", 1)[-1].lower() == (_QWEN3_30B_A3B_MODEL_BASENAME)


def _model_basename(model_name):
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower()


def _strategy_model_name(cfg):
    """Keep the configured local checkpoint available when the API is off."""
    return getattr(cfg, "api_strategy", "") or cfg.strategy_model_name


def _apply_strategy_model_profile(merged, model_name, explicit_keys=frozenset()):
    """Apply model-native strategist defaults below explicit configuration."""
    profile = strategist_profile(model_name)
    if profile:
        for key in ("strategy_max_new_tokens", "strategy_max_seq_length"):
            _profile_default(merged, key, profile["context"], explicit_keys)
        for key, value in (("strategy_reasoning_effort", "auto"),
                           ("strategy_thinking", True),
                           ("strategy_vllm_sleep_level", 1),
                           ("strategy_vllm_staged_loading", False)):
            _profile_default(merged, key, value, explicit_keys)
        print(f"[model-profile] frozen {profile['family']} strategist: "
              f"{model_name}; official efforts={','.join(profile['efforts'])}; "
              "native checkpoint quantization; YAML sampling/token limits "
              "take precedence")
        return
    if (_model_basename(model_name) != _DEEPSEEK_R1_0528_QWEN3_8B_MODEL_BASENAME):
        return

    # DeepSeek evaluates this checkpoint with 64K generations, but its native
    # context is 128K. Expose that full output ceiling here; generation already
    # subtracts the actual prompt length, so prompt + completion never exceeds
    # 131072. It is a native reasoning model: thinking stays enabled and
    # `high` is the strongest effort label exposed by this runner.
    _profile_default(merged, "strategy_max_new_tokens", 131_072, explicit_keys)
    _profile_default(merged, "strategy_max_seq_length", 131_072, explicit_keys)
    _profile_default(merged, "strategy_thinking", True, explicit_keys)
    _profile_default(merged, "strategy_reasoning_effort", "high", explicit_keys)
    _profile_default(merged, "strategy_vllm_quantization", "", explicit_keys)

    # At 8B, the exact model fits without quantization. The topology planner
    # keeps independent TP=1 replicas while the live parent-chain frontier can
    # feed them, and folds otherwise-idle replicas into TP ranks when it cannot.
    # Retain every selected engine's level-1 backup between phases.
    _profile_default(merged, "strategy_vllm_sleep_level", 1, explicit_keys)
    _profile_default(merged, "strategy_vllm_staged_loading", False, explicit_keys)
    _profile_default(merged, "strategy_vllm_persistent_workers", None, explicit_keys)
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
            "reasoning_effort": "high",
            "template_kind": "gpt-oss",
            # Modern Unsloth trains the published MXFP4 experts while keeping
            # them packed. This is the same base used by vLLM, without NF4
            # re-quantization or a second full-precision model copy.
            "training_4bit": False,
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
            "reasoning_effort": "low",
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


def _coder_vllm_persistent_workers(cfg):
    """Release GPT-OSS coder engines completely during adapter training."""
    if (str(getattr(cfg, "coder_model_profile", "")) == "gpt-oss-120b" and not bool(getattr(cfg, "no_train", False))):
        return 0
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
                    if (isinstance(value, int) and not isinstance(value, bool) and value > 0):
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
    """Keep the legacy phase-2 fallback; explicit splits use prompt variants."""
    if (getattr(cfg, "coder_template_kind", "generic") == "qwen3.8" and phase == "adaptive"):
        return "medium"
    return getattr(cfg, "coder_reasoning_effort", None)


def _coder_effort_entries(prompt_jobs, source_job_indices, counts, cfg, step_idx, phase):
    """Route Qwen3.8 rollouts through the configured per-strategy counts.

    The split changes only the chat-template reasoning control. Every source
    keeps exactly its requested sample count, allocation identity, and pilot
    status. Unassigned rollouts retain the base effort; other models are unchanged.
    """
    del prompt_jobs, step_idx
    base_effort = _coder_effort_for_rollout_phase(cfg, phase)
    entries = [
        (int(source_idx), int(count), base_effort)
        for source_idx, count in zip(source_job_indices, counts)
        if int(count) > 0
    ]
    if (phase == "adaptive" and getattr(cfg, "coder_template_kind", "generic") == "qwen3.8"):
        requested = {
            effort: int(getattr(cfg, f"phase2_{effort}_programs_per_strategy", default))
            for effort, default in (("xhigh", 0), ("medium", -1), ("low", 0))
        }
        remainder_effort = next(
            (effort for effort, count in requested.items() if count == -1),
            _coder_effort_for_rollout_phase(cfg))
        grouped = {effort: [] for effort in requested}
        for source_idx, count, _effort in entries:
            remaining = count
            split = {}
            # Match pilot priority when a strategy receives fewer than requested.
            for effort, wanted in requested.items():
                if wanted < 0:
                    continue
                allocated = min(wanted, remaining)
                split[effort] = allocated
                remaining -= allocated
            if remaining:
                split[remainder_effort] = split.get(remainder_effort, 0) + remaining
            for effort, allocated in split.items():
                if allocated:
                    grouped.setdefault(effort, []).append((source_idx, allocated, effort))
        return [entry for group in grouped.values() for entry in group]
    xhigh_per_strategy = int(getattr(cfg, "pilot_xhigh_programs_per_strategy", 0))
    medium_per_strategy = int(getattr(cfg, "pilot_medium_programs_per_strategy", 0))
    if (str(phase) != "pilot" or getattr(cfg, "coder_template_kind", "generic") != "qwen3.8"):
        return entries

    xhigh_entries = []
    medium_entries = []
    base_entries = []
    for source_idx, count, _effort in entries:
        xhigh_count = min(xhigh_per_strategy, count)
        medium_count = min(medium_per_strategy, count - xhigh_count)
        if xhigh_count:
            xhigh_entries.append((source_idx, xhigh_count, "xhigh"))
        if medium_count:
            medium_entries.append((source_idx, medium_count, "medium"))
        if count > xhigh_count + medium_count:
            base_entries.append((source_idx, count - xhigh_count - medium_count, base_effort))
    # Keep all xhigh variants in the same scheduler call and place those groups
    # first so round-robin assignment spreads the longer requests across the
    # available engines. This adds no rollout or allocation budget.
    return xhigh_entries + medium_entries + base_entries


def _phase_coder_prompt_job(prompt_jobs, source_idx, phase, cfg, render, cache, *, effort=None):
    """Return an immutable phase-specific prompt ID without changing pilots.

    Records must retain the exact prompt used at generation for old/reference
    logprobs, training, and artifact saving. Appending a prompt variant instead
    of replacing the source job preserves pilot histories and tensor caches.
    Allocation still uses the original source ID, so variants add no budget.
    """
    if effort is None:
        effort = _coder_effort_for_rollout_phase(cfg, phase)
    if effort == _coder_effort_for_rollout_phase(cfg):
        return int(source_idx)
    key = (int(source_idx), effort)
    if key not in cache:
        source = prompt_jobs[int(source_idx)]
        variant = {
            **source,
            "prompt_text": render(source["messages"], rollout_phase=phase, reasoning_effort=effort),
            "coder_reasoning_effort": effort,
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
        _profile_default(merged, "coder_reasoning_effort", str(profile["reasoning_effort"]), explicit_keys)
    else:
        # Qwen3-8B exposes only the real enable_thinking switch; it has no
        # native low/medium/high reasoning-effort control.
        merged.pop("coder_reasoning_effort", None)
    merged["coder_preserve_thinking"] = bool(profile["template_kind"] == "qwen3.8")
    _profile_default(merged, "max_seq_length", native_context, explicit_keys)
    _profile_default(merged, "max_new_tokens", int(profile["max_output"]), explicit_keys)
    if bool(profile.get("force_thinking", False)):
        # This project uses Qwen3-8B only in its thinking mode.  Unlike
        # temperature/top-p, this is part of the selected model contract.
        merged["thinking"] = True
    else:
        _profile_default(merged, "thinking", True, explicit_keys)

    # Quantization and placement describe only the differentiable copy. The
    # aligned GPT-OSS profile keeps the original checkpoint name on both sides.
    _profile_default(merged, "load_in_4bit", bool(profile["training_4bit"]), explicit_keys)
    _profile_default(merged, "training_model_name", "", explicit_keys)
    _profile_default(merged, "training_layout", str(profile["training_layout"]), explicit_keys)
    _profile_default(
        merged, "training_memory_fraction",
        float(profile.get("training_memory_fraction", 0.80)), explicit_keys)

    # The model profile leaves memory outside vLLM's KV cache for CUDA/NCCL
    # and the bounded exact-token projection. The layout planner also accounts
    # for startup weight conversion before choosing the smallest fitting TP
    # group. Qwen3.8 needs a larger projection reserve than the other profiles.
    _profile_default(merged, "vllm_gpu_memory_utilization", 0.965, explicit_keys)
    _profile_default(merged, "vllm_quantization", "", explicit_keys)
    _profile_default(merged, "gen_micro_batch", "auto", explicit_keys)
    _profile_default(merged, "vllm_max_num_batched_tokens", "auto", explicit_keys)
    merged["vllm_runtime_reserve_override_gib"] = float(profile["vllm_runtime_reserve_gib"])
    _profile_default(merged, "vllm_staged_loading", True, explicit_keys)
    # Discard the obsolete hardware-name bypass, including resumed runs.
    # Admission now depends on the bounded-conversion worker contract in
    # gpu_runtime.py, rather than assuming a backend from the GPU's name.
    merged.pop("allow_gpt_oss_tp1", None)

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
    reasoning_label = merged.get("coder_reasoning_effort", "thinking on/off only")
    print(
        f"[model-profile] {profile['name']}: context limit="
        f"{merged['max_seq_length']}, max generated tokens="
        f"{merged['max_new_tokens']}, reasoning="
        f"{reasoning_label}, rollout target=maximum memory-safe replicas; "
        "training layout="
        f"{merged['training_layout']}; {precedence_note}; exact "
        "long-context training enabled",
        flush=True,
    )


def _uses_sequence_level_policy_ratio(cfg):
    """Use the MoE-stable sequence ratio only for plain Qwen3-30B-A3B."""
    return _is_exact_qwen3_30b_a3b(getattr(cfg, "model_name", ""))


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
