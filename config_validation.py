"""Validation for the self-contained problem YAML presets."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path


# These settings are consumed by the shared training, generation, search,
# memory, and feedback runtimes. Every checked-in preset carries them so a
# selected YAML is self-contained, but that does not make problem-specific
# fields interchangeable.
RERANKER_REQUIRED_KEYS = frozenset("""
reranker_enabled reranker_backend reranker_model reranker_base_url
reranker_api_key reranker_api_key_env reranker_temperature
reranker_max_tokens reranker_request_timeout_s reranker_judge_gpu
reranker_judge_batch_size reranker_judge_load_in_4bit reranker_top_p
reranker_max_seq_length reranker_top_k reranker_debate
reranker_tournament_mode reranker_num_random_matches reranker_both_orders
reranker_judge_concurrency reranker_max_code_chars reranker_elo_init
reranker_elo_k reranker_elo_softmax_temp reranker_prior_weight
reranker_poll_interval_s reranker_min_states_to_rank reranker_goal
""".split())


COMMON_REQUIRED_KEYS = frozenset("""
problem target
fail_score model_name training_model_name backend max_seq_length load_in_4bit
lora_rank lora_alpha
lora_dropout target_modules
training_gpu_id available_gpu_ids reserve_last_gpu_for_evaluation
evaluation_gpu_id num_gpus gpu_ids sequential_generation
evaluation_shares_generation
generation_backend gen_micro_batch vllm_gpu_memory_utilization
vllm_enforce_eager vllm_enable_prefix_caching vllm_tensor_parallel_size
vllm_pipeline_parallel_size vllm_quantization
vllm_max_num_batched_tokens vllm_enable_expert_parallel
num_steps groups_per_step group_size num_seed_states max_groups_per_step
max_group_size growth_force_step growth_valid_yield growth_distinct_min
growth_factor learning_rate adam_beta1 adam_beta2 adam_epsilon weight_decay
kl_penalty_coef grad_clip
train_examples_per_microbatch logprob_chunk
puct_c max_buffer_size topk_children_per_parent
max_new_tokens temperature top_p thinking deterministic seed
sandbox_timeout_s reward_workers print_responses max_saved_construction
memory memory_version memory_extract_mode memory_lessons_per_call memory_hygiene_profile
memory_max_examples_per_call memory_max_chars_per_example
memory_feedback_chars memory_max_new_tokens memory_forbid_constructions
memory_max_code_lines memory_global_scope_allows_code memory_lookup_mode
memory_lookup_max_select memory_lookup_fallback memory_lookup_max_new_tokens
memory_lookup_temperature memory_catalog_max_lessons memory_catalog_chars
memory_curate_every memory_curate_min_bank memory_curate_max_items
memory_curate_max_new_tokens
memory_curate_min_keep_frac memory_max_lessons memory_dedup_jaccard
memory_reinforce_delta memory_persist memory_inject_mode memory_token_budget
memory_grant_context memory_arm_control_fraction memory_arm_explore_fraction
memory_arm_max_lessons memory_arm_exploration_c memory_arm_comparison_n memory_outcome_credit
memory_text_reinforce memory_extract_from memory_require_full_lessons
memory_temperature memory_top_p memory_use_gen_pool
feedback feedback_lambda feedback_anneal_steps feedback_anneal_shape
feedback_lambda_final feedback_clip feedback_chars feedback_max_per_step
feedback_auto_fraction feedback_include_constant_groups feedback_inject_mode
feedback_normalize feedback_adaptive feedback_validity_floor
feedback_validity_target feedback_max_reward_ratio feedback_reward_scale_floor
feedback_max_per_signature feedback_auto_signature_fraction
""".split()) | RERANKER_REQUIRED_KEYS

COMMON_OPTIONAL_KEYS = frozenset()

CPU_PROBLEMS = frozenset({
    "circle_packing", "erdos", "ac1", "ac2", "denoising",
})

PROBLEM_REQUIRED_KEYS = {
    "circle_packing": frozenset({
        "num_circles", "degenerate_threshold", "eval_cpus",
    }),
    "erdos": frozenset({"budget_s", "eval_cpus"}),
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
    "budget_s": frozenset({"erdos", "ac1", "ac2"}),
    "eval_cpus": CPU_PROBLEMS,
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
                "num_seed_states", "max_new_tokens", "max_seq_length"):
        if key in data:
            _positive_int(data, key, source)
    if "eval_cpus" in data:
        _positive_int(data, "eval_cpus", source)
    if "num_circles" in data:
        _positive_int(data, "num_circles", source)
    for key in ("sandbox_timeout_s", "budget_s", "score_scale",
                "kernel_timeout_s"):
        if key in data:
            _positive_number(data, key, source)

    if "top_p" in data and not 0 < float(data["top_p"]) <= 1:
        raise ValueError(f"{_label(source)}: top_p must be in (0, 1]")
    if "memory_top_p" in data and not 0 < float(data["memory_top_p"]) <= 1:
        raise ValueError(f"{_label(source)}: memory_top_p must be in (0, 1]")
    if "memory_version" in data and str(data["memory_version"]).upper() not in {
            "V1", "V2"}:
        raise ValueError(f"{_label(source)}: memory_version must be V1 or V2")
    if ("memory_arm_comparison_n" in data
            and int(data["memory_arm_comparison_n"]) < 0):
        raise ValueError(
            f"{_label(source)}: memory_arm_comparison_n must be >= 0")
    for key in ("adam_beta1", "adam_beta2"):
        if key in data and not 0 <= float(data[key]) < 1:
            raise ValueError(f"{_label(source)}: {key} must be in [0, 1)")
    if "adam_epsilon" in data:
        _positive_number(data, "adam_epsilon", source)
    if "weight_decay" in data and float(data["weight_decay"]) < 0:
        raise ValueError(f"{_label(source)}: weight_decay must be >= 0")
    if "reranker_enabled" in data and data["reranker_enabled"] is not False:
        raise ValueError(
            f"{_label(source)}: reranker_enabled must be false; the Elo "
            "reranker implementation is not installed"
        )
    if "reward_workers" in data and int(data["reward_workers"]) < 0:
        raise ValueError(f"{_label(source)}: reward_workers must be >= 0")
    if problem == "gpu_mode" and data.get("reward_workers") != 1:
        raise ValueError(
            f"{_label(source)}: gpu_mode requires reward_workers=1"
        )
    if problem in {"erdos", "ac1", "ac2"} and all(
        key in data for key in ("budget_s", "sandbox_timeout_s")
    ) and float(data["sandbox_timeout_s"]) <= float(data["budget_s"]):
        raise ValueError(
            f"{_label(source)}: sandbox_timeout_s must exceed budget_s so "
            "the candidate can return its best result before the hard timeout"
        )
    if problem == "gpu_mode" and not str(data.get("gpu_type", "")).strip():
        raise ValueError(f"{_label(source)}: gpu_mode requires gpu_type")
