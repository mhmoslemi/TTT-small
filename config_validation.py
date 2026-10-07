"""Validation for the self-contained problem YAML presets."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path


# These settings are consumed by the shared training, generation, search,
# runtimes. Every checked-in preset carries them so a
# selected YAML is self-contained, but that does not make problem-specific
# fields interchangeable.
COMMON_REQUIRED_KEYS = frozenset("""
problem target
fail_score model_name training_model_name coder_model_name
coder_training_model_name strategy_model_name backend load_in_4bit
lora_rank lora_alpha
lora_dropout target_modules
binary_coder_training binary_coder_init_steps binary_coder_lora_rank
binary_coder_clip_epsilon_low binary_coder_clip_epsilon_high
training_gpu_id available_gpu_ids reserve_last_gpu_for_evaluation
evaluation_gpu_id num_gpus gpu_ids sequential_generation
evaluation_shares_generation
generation_backend gen_micro_batch vllm_gpu_memory_utilization
vllm_enforce_eager vllm_enable_prefix_caching vllm_tensor_parallel_size
vllm_pipeline_parallel_size vllm_quantization
vllm_max_num_batched_tokens vllm_enable_expert_parallel vllm_sleep_level
vllm_staged_loading strategy_vllm_sleep_level strategy_vllm_staged_loading
num_steps groups_per_step group_size strategies_per_parent
programs_per_strategy pilot_programs_per_strategy strategy_archive_top_r
num_seed_states learning_rate adam_beta1 adam_beta2 adam_epsilon weight_decay
kl_penalty_coef grad_clip
train_examples_per_microbatch logprob_chunk
puct_c max_buffer_size topk_children_per_parent
temperature top_p top_k thinking strategy_max_new_tokens
strategy_max_seq_length strategy_temperature strategy_top_p strategy_top_k
strategy_thinking strategy_reasoning_effort strategy_vllm_quantization
deterministic seed
sandbox_timeout_s reward_workers print_responses max_saved_construction
""".split())

COMMON_OPTIONAL_KEYS = frozenset({
    "max_seq_length", "max_new_tokens",  # Otherwise derived from the coder.
    "advantage_mode", "cvar_alpha", "cvar_lambda", "fast", "isolate_eval",
    "fused_long_attention",
    "training_layout",
    "phase2_allocation_method",
    "min_p", "strategy_min_p",
    "strategy_vllm_persistent_workers",
    "strategy_backend", "strategy_api_base_url", "strategy_api_key_env",
    "strategy_api_concurrency", "strategy_api_timeout_s",
    "strategy_api_max_retries",
    "strategy_format_max_retries",
    "uct",
    "x_grpo_budgets", "x_grpo_relative_error", "x_grpo_entropy_coef",
    "x_grpo_contexts_per_step",
    "spo_rs_beta", "spo_rs_d_half", "spo_rs_rho_min", "spo_rs_rho_max",
    "spo_rs_clip_epsilon", "spo_rs_clip_epsilon_low",
    "spo_rs_clip_epsilon_high",
})

# Compatibility filter only: these keys cannot activate any runtime behavior.
# Archived configs may still contain them; fresh presets no longer do.
RETIRED_BATCH_GROWTH_KEYS = frozenset({
    "max_groups_per_step", "max_group_size", "growth_force_step",
    "growth_valid_yield", "growth_distinct_min", "growth_factor",
})

CPU_PROBLEMS = frozenset({
    "circle_packing", "erdos", "erdos-c4", "ac1", "ac2", "denoising",
})

PROBLEM_REQUIRED_KEYS = {
    "circle_packing": frozenset({
        "num_circles", "degenerate_threshold", "eval_cpus",
    }),
    "erdos": frozenset({
        "budget_s", "eval_cpus",
    }),
    "erdos-c4": frozenset({
        "budget_s", "eval_cpus", "max_coeff_count", "benchmark_c4",
    }),
    "ac1": frozenset({"problem_type", "budget_s", "eval_cpus"}),
    "ac2": frozenset({"problem_type", "budget_s", "eval_cpus"}),
    "denoising": frozenset({"eval_seed", "eval_cpus"}),
    "gpu_mode": frozenset({
        "problem_type", "score_scale", "gpu_type", "gpu_lease_timeout_s",
        "triton_version", "task_yaml", "lib_dir", "kernel_log_chars",
        "kernel_timeout_s", "show_launch_note", "seed_from_reference",
    }),
}

PROBLEM_OPTIONAL_KEYS = {
    "circle_packing": frozenset(),
    "erdos": frozenset(),
    "erdos-c4": frozenset(),
    "ac1": frozenset(),
    "ac2": frozenset(),
    "denoising": frozenset(),
    "gpu_mode": frozenset({"kernel_lib_dir", "mla_seed_runtime_us"}),
}

# A present field from this table is invalid outside its owner set, even for a
# partial external config. This catches exactly the copy/paste failure that put
# num_circles in every preset.
EXCLUSIVE_KEY_OWNERS = {
    "num_circles": frozenset({"circle_packing"}),
    "degenerate_threshold": frozenset({"circle_packing"}),
    "eval_seed": frozenset({"denoising"}),
    "budget_s": frozenset({"erdos", "erdos-c4", "ac1", "ac2"}),
    "eval_cpus": CPU_PROBLEMS,
    "max_coeff_count": frozenset({"erdos-c4"}),
    "benchmark_c4": frozenset({"erdos-c4"}),
    "problem_type": frozenset({"ac1", "ac2", "gpu_mode"}),
    "score_scale": frozenset({"gpu_mode"}),
    "gpu_type": frozenset({"gpu_mode"}),
    "gpu_lease_timeout_s": frozenset({"gpu_mode"}),
    "triton_version": frozenset({"gpu_mode"}),
    "task_yaml": frozenset({"gpu_mode"}),
    "lib_dir": frozenset({"gpu_mode"}),
    "kernel_lib_dir": frozenset({"gpu_mode"}),
    "kernel_log_chars": frozenset({"gpu_mode"}),
    "kernel_timeout_s": frozenset({"gpu_mode"}),
    "show_launch_note": frozenset({"gpu_mode"}),
    "seed_from_reference": frozenset({"gpu_mode"}),
    "mla_seed_runtime_us": frozenset({"gpu_mode"}),
}


def _label(source) -> str:
    return str(source) if source is not None else "configuration"


def _positive_number(data: Mapping, key: str, source) -> None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{_label(source)}: {key} must be a positive number")


def _positive_int(data: Mapping, key: str, source) -> None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{_label(source)}: {key} must be a positive integer")


def validate_problem_config(
    data: Mapping, *, source: str | Path | None = None,
    require_complete: bool = True,
) -> None:
    """Raise ValueError when a problem configuration violates its contract."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError(f"{_label(source)}: config must be a non-empty mapping")

    problem = str(data.get("problem", "")).strip().lower()
    if problem not in PROBLEM_REQUIRED_KEYS:
        raise ValueError(
            f"{_label(source)}: problem must be one of "
            f"{sorted(PROBLEM_REQUIRED_KEYS)}, got {problem!r}"
        )

    for key, owners in EXCLUSIVE_KEY_OWNERS.items():
        if key in data and problem not in owners:
            raise ValueError(
                f"{_label(source)}: {key} is not valid for problem={problem}"
            )

    required = COMMON_REQUIRED_KEYS | PROBLEM_REQUIRED_KEYS[problem]
    allowed = required | COMMON_OPTIONAL_KEYS | PROBLEM_OPTIONAL_KEYS[problem]
    if require_complete:
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"{_label(source)}: missing required keys: {sorted(missing)}"
            )
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"{_label(source)}: unknown or misplaced keys: {sorted(unknown)}"
            )

    if problem in ("ac1", "ac2") and data.get("problem_type") != problem:
        raise ValueError(
            f"{_label(source)}: {problem} requires problem_type={problem}"
        )
    if problem == "gpu_mode" and data.get("problem_type") not in {
        "trimul", "mla_decode_nvidia",
    }:
        raise ValueError(
            f"{_label(source)}: unsupported gpu_mode problem_type "
            f"{data.get('problem_type')!r}"
        )
    if (problem == "gpu_mode"
            and data.get("problem_type") == "mla_decode_nvidia"
            and "mla_seed_runtime_us" not in data):
        raise ValueError(
            f"{_label(source)}: mla_decode_nvidia requires the explicit "
            "mla_seed_runtime_us key (use null until measured on gpu_type)"
        )

    for key in ("num_steps", "groups_per_step", "group_size",
                "num_seed_states", "max_new_tokens", "max_seq_length",
                "train_examples_per_microbatch", "strategies_per_parent",
                "programs_per_strategy", "strategy_archive_top_r",
                "strategy_max_new_tokens",
                "strategy_max_seq_length", "binary_coder_lora_rank"):
        if key in data:
            _positive_int(data, key, source)
    if "strategy_format_max_retries" in data:
        retries = data["strategy_format_max_retries"]
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(
                f"{_label(source)}: strategy_format_max_retries must be a "
                "non-negative integer")
    if "pilot_programs_per_strategy" in data:
        pilot_count = data["pilot_programs_per_strategy"]
        if (isinstance(pilot_count, bool)
                or not isinstance(pilot_count, int)
                or pilot_count == 0 or pilot_count < -1):
            raise ValueError(
                f"{_label(source)}: pilot_programs_per_strategy must be -1 "
                "(disabled) or a positive integer")
        programs_per_strategy = data.get("programs_per_strategy")
        if (pilot_count > 0 and isinstance(programs_per_strategy, int)
                and pilot_count > programs_per_strategy):
            raise ValueError(
                f"{_label(source)}: pilot_programs_per_strategy cannot exceed "
                "programs_per_strategy")
    if "phase2_allocation_method" in data:
        allocation_method = str(data["phase2_allocation_method"]).strip().lower()
        if allocation_method not in {"rule_based", "bandit", "hurdle"}:
            raise ValueError(
                f"{_label(source)}: phase2_allocation_method must be "
                "'rule_based', 'bandit', or 'hurdle'")
    for key in ("top_k", "strategy_top_k"):
        if key in data:
            value = data[key]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value != 0):
                raise ValueError(
                    f"{_label(source)}: {key} must be 0 because top-k "
                    "filtering is disabled project-wide")
    for key in ("min_p", "strategy_min_p"):
        if key in data:
            value = data[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"{_label(source)}: {key} must be in [0, 1]")
    if "binary_coder_training" in data:
        if not isinstance(data["binary_coder_training"], bool):
            raise ValueError(
                f"{_label(source)}: binary_coder_training must be true or false")
        init_steps = data.get("binary_coder_init_steps")
        if (isinstance(init_steps, bool) or not isinstance(init_steps, int)
                or init_steps < 0):
            raise ValueError(
                f"{_label(source)}: binary_coder_init_steps must be a "
                "nonnegative integer")
        if data["binary_coder_training"] and init_steps < 1:
            raise ValueError(
                f"{_label(source)}: binary_coder_init_steps must be positive "
                "when binary_coder_training is enabled")
    for key in ("binary_coder_clip_epsilon_low",
                "binary_coder_clip_epsilon_high"):
        if key in data and not 0.0 < float(data[key]) < 1.0:
            raise ValueError(f"{_label(source)}: {key} must be in (0, 1)")
    if "eval_cpus" in data:
        _positive_int(data, "eval_cpus", source)
    if "num_circles" in data:
        _positive_int(data, "num_circles", source)
    if "max_coeff_count" in data:
        _positive_int(data, "max_coeff_count", source)
        if int(data["max_coeff_count"]) < 3:
            raise ValueError(
                f"{_label(source)}: max_coeff_count must be at least 3"
            )
    for key in ("sandbox_timeout_s", "budget_s", "score_scale",
                "kernel_timeout_s", "benchmark_c4"):
        if key in data:
            _positive_number(data, key, source)

    if "top_p" in data and not 0 < float(data["top_p"]) <= 1:
        raise ValueError(f"{_label(source)}: top_p must be in (0, 1]")
    if ("strategy_top_p" in data
            and not 0 < float(data["strategy_top_p"]) <= 1):
        raise ValueError(
            f"{_label(source)}: strategy_top_p must be in (0, 1]")
    if ("strategy_temperature" in data
            and float(data["strategy_temperature"]) <= 0):
        raise ValueError(
            f"{_label(source)}: strategy_temperature must be positive")
    if ("strategy_thinking" in data
            and not isinstance(data["strategy_thinking"], bool)):
        raise ValueError(
            f"{_label(source)}: strategy_thinking must be true or false")
    if ("strategy_reasoning_effort" in data
            and str(data["strategy_reasoning_effort"]).lower()
            not in {"low", "medium", "high"}):
        raise ValueError(
            f"{_label(source)}: strategy_reasoning_effort must be low, "
            "medium, or high")
    if ("strategy_backend" in data
            and str(data["strategy_backend"]).lower() not in {"local", "api"}):
        raise ValueError(
            f"{_label(source)}: strategy_backend must be local or api")
    if ("strategy_api_concurrency" in data
            and int(data["strategy_api_concurrency"]) < 1):
        raise ValueError(
            f"{_label(source)}: strategy_api_concurrency must be >= 1")
    if ("strategy_api_timeout_s" in data
            and float(data["strategy_api_timeout_s"]) <= 0):
        raise ValueError(
            f"{_label(source)}: strategy_api_timeout_s must be positive")
    if ("strategy_api_max_retries" in data
            and int(data["strategy_api_max_retries"]) < 0):
        raise ValueError(
            f"{_label(source)}: strategy_api_max_retries must be >= 0")
    for key in ("vllm_sleep_level", "strategy_vllm_sleep_level"):
        if key in data and int(data[key]) not in (1, 2):
            raise ValueError(f"{_label(source)}: {key} must be 1 or 2")
    if "strategy_vllm_persistent_workers" in data:
        value = data["strategy_vllm_persistent_workers"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{_label(source)}: strategy_vllm_persistent_workers must "
                "be a nonnegative integer")
    for key in ("vllm_staged_loading", "strategy_vllm_staged_loading"):
        if key in data and not isinstance(data[key], bool):
            raise ValueError(
                f"{_label(source)}: {key} must be true or false")
    for key in ("adam_beta1", "adam_beta2"):
        if key in data and not 0 <= float(data[key]) < 1:
            raise ValueError(f"{_label(source)}: {key} must be in [0, 1)")
    if "adam_epsilon" in data:
        _positive_number(data, "adam_epsilon", source)
    if "weight_decay" in data and float(data["weight_decay"]) < 0:
        raise ValueError(f"{_label(source)}: weight_decay must be >= 0")
    if "uct" in data and not isinstance(data["uct"], bool):
        raise ValueError(f"{_label(source)}: uct must be true or false")
    if "reward_workers" in data and int(data["reward_workers"]) < 0:
        raise ValueError(f"{_label(source)}: reward_workers must be >= 0")
    if problem == "gpu_mode" and data.get("reward_workers") != 1:
        raise ValueError(
            f"{_label(source)}: gpu_mode requires reward_workers=1"
        )
    if problem in {"erdos", "erdos-c4", "ac1", "ac2"} and all(
        key in data for key in ("budget_s", "sandbox_timeout_s")
    ) and float(data["sandbox_timeout_s"]) <= float(data["budget_s"]):
        raise ValueError(
            f"{_label(source)}: sandbox_timeout_s must exceed budget_s so "
            "the candidate can return its best result before the hard timeout"
        )
    if problem == "gpu_mode" and not str(data.get("gpu_type", "")).strip():
        raise ValueError(f"{_label(source)}: gpu_mode requires gpu_type")
