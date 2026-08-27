"""
TTT-Discover — multi-problem local runner.

Config():  defaults  <  YAML  <  CLI flags
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import argparse
import json
import random
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import yaml


# ======================================================================
# Config
# ======================================================================
@dataclass
class Config:
    # Problem selector
    problem: str = "circle_packing"

        # "circle_packing", "erdos", "ac1", "ac2",
        # "denoising", "gpu_mode", "ahc",

    problem_type: str = ""        # ac1/ac2, trimul/mla_decode_nvidia, etc.

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    # "vllm" is a convenience mode: HF+PEFT trains, vLLM generates rollouts.
    backend: str = "auto"        # "auto" | "unsloth" | "hf" | "vllm"
    generation_backend: str = "hf"   # "hf" | "vllm"
    max_seq_length: int = 32000
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    num_circles: int = 26
    # None means "let the problem decide". A concrete default here leaks circle
    # packing's target into every other problem, because Problem.__init__ only
    # applies its own when this is None.
    target: Optional[float] = None
    sandbox_timeout_s: float = 30.0

    # RL hyperparameters
    num_steps: int = 50
    groups_per_step: int = 8       # START value; ratchets up toward the max below
    group_size: int = 64           # START value; ratchets up toward the max below
    num_seed_states: int = 16

    # ---- adaptive batch growth ----
    # groups_per_step / group_size are the STARTING (G, K). Each step, if the
    # best group's valid fraction and the count of distinct improved children
    # both clear their thresholds, (G, K) are multiplied by growth_factor and
    # clamped to (max_groups_per_step, max_group_size). Growth is monotonic: it
    # never shrinks. At step >= growth_force_step, (G, K) are pinned to the max
    # no matter what the signals say.
    # None means "same as the starting value"; explicit larger values enable
    # adaptive growth. Resolved to concrete ints during config loading.
    max_groups_per_step: Optional[int] = None
    max_group_size: Optional[int] = None
    growth_force_step: int = 10    # from this step on, run at max unconditionally
    growth_valid_yield: float = 0.7   # best group's valid fraction must reach this
    growth_distinct_min: int = 4      # this many distinct improved children needed
    growth_factor: float = 2.0        # multiply G and K by this when both clear
    learning_rate: float = 4e-5
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 4200
    grad_clip: float = 1.0

    train_examples_per_microbatch: int = 1

    # Bound the transient log_softmax allocation in compute_token_logprobs by
    # slicing the response positions into chunks of at most this many tokens.
    # The forward pass is untouched; only the float32 log_softmax + gather is
    # chunked, so results are EXACT (log_softmax is per-position over the vocab).
    # This is what keeps the feedback teacher forward from OOMing when it stacks
    # on top of the still-alive training graph. 0 = single shot (identical to
    # before). Nothing here depends on the model or its vocab.
    logprob_chunk: int = 0

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0
    # Qwen hybrid models render a thinking preamble when this is enabled.
    # This controls solution rollouts (and their matched feedback reprompts),
    # not the separate memory extraction/selection calls.
    thinking: bool = False

    # PUCT
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children_per_parent: int = 2

    # Misc
    seed: int = 42
    deterministic: bool = False        # master switch for reproducible sampling
    print_responses: int = 0           # how many rollouts to print per step
    # Cap on the construction length written into each rollout's meta. Erdos
    # carries 40-100 floats; circle packing and gpu_mode carry none. 0 disables.
    max_saved_construction: int = 4096
    # Threads used to evaluate rollouts. 0 = auto (cpu_count - num_gpus), which
    # is right for a subprocess sandbox and WRONG for anything that benchmarks
    # on the GPU: concurrent timing runs contend and the reward IS the timing.
    # Set 1 for gpu_mode.
    reward_workers: int = 0

    # Multi-GPU generation
    # gpu_ids is authoritative when present; num_gpus is derived from its
    # length. Passing --num-gpus alone selects devices 0..N-1.
    num_gpus: Optional[int] = None
    gpu_ids: str = "0"
    # Max sequences each GPU holds per generate() call. The worker loops in
    # chunks of this size until the group's rollouts are done, so group_size can
    # be anything while per-GPU KV memory stays bounded by this. 0 = one shot
    # (byte-identical to no micro-batch; a deterministic run with 0 draws exactly
    # what it did before this feature existed).
    gen_micro_batch: int = 0
    # vLLM owns its own scheduler/KV cache. gen_micro_batch, when nonzero, is
    # passed as max_num_seqs; 0 lets vLLM choose. Generation GPUs should be
    # separate from the main training GPU unless the cards have ample headroom.
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enforce_eager: bool = False
    vllm_enable_prefix_caching: bool = True

    # ---- Memory (Sec. 2.2) ----
    # `memory` is the master switch. When it is False, every memory_* field
    # below is ignored. No embeddings anywhere: retrieval is the model reading
    # a one-line index of the whole bank and naming the ids it wants.
    memory: bool = False
    memory_lookup_mode: str = "select"      # select | all | none
    memory_lookup_max_select: int = 5
    memory_lookup_max_new_tokens: int = 256
    memory_lookup_temperature: float = 0.3
    memory_lookup_fallback: str = "none"    # none | recent | importance
    memory_catalog_max_lessons: int = 0     # 0 = show the whole bank
    memory_catalog_chars: int = 200
    memory_inject_mode: str = "append"      # append | system
    memory_token_budget: int = 1200
    memory_grant_context: bool = True
    memory_arm_control_fraction: float = 0.0
    memory_arm_explore_fraction: float = 0.0
    memory_arm_max_lessons: int = 1
    memory_arm_exploration_c: float = 0.5
    memory_outcome_credit: bool = False
    memory_text_reinforce: bool = True
    memory_extract_mode: str = "contrast"   # contrast | split
    memory_extract_from: str = "both"       # both | failure | success (split only)
    memory_curate_every: int = 0            # 0 = never; N = every N steps
    memory_curate_min_bank: int = 20
    memory_curate_max_items: int = 60
    memory_curate_min_keep_frac: float = 0.25
    memory_lessons_per_call: int = 3        # L, a ceiling
    memory_require_full_lessons: bool = False
    memory_max_examples_per_call: int = 8
    memory_max_chars_per_example: int = 1500
    memory_feedback_chars: int = 800
    memory_reinforce_delta: float = 0.15
    memory_max_new_tokens: int = 1024
    memory_temperature: float = 0.7
    memory_top_p: float = 0.95
    memory_use_gen_pool: bool = True
    memory_forbid_constructions: bool = True
    memory_max_code_lines: int = 4
    memory_global_scope_allows_code: bool = False
    memory_max_lessons: int = 500
    memory_dedup_jaccard: float = 0.6
    memory_persist: bool = True

    # ---- Feedback-based program-repair signal (Sec. 2.3, Eq. 9) ----
    # `feedback` is the master switch, same rule as `memory`: when it is False
    # every feedback_* field below is ignored.
    feedback: bool = False
    feedback_lambda: float = 0.2           # lambda_f at step 0
    feedback_anneal_steps: int = 0         # 0 = constant; 10 = off from step 10
    feedback_anneal_shape: str = "linear"  # linear | cosine
    feedback_lambda_final: float = 0.0
    feedback_clip: float = 5.0             # clamp |A^fb| per token; 0 = Eq. 9 as written
    feedback_chars: int = 1200             # verifier text budget in the reprompt
    # 0 = auto (20% of current G*K); -1 = all eligible code failures;
    # >0 = explicit cap.
    feedback_max_per_step: int = 0
    feedback_auto_fraction: float = 0.20
    feedback_include_constant_groups: bool = True
    feedback_inject_mode: str = "append"   # append | user_turn
    feedback_normalize: bool = False
    feedback_adaptive: bool = False
    feedback_validity_floor: float = 0.5
    feedback_validity_target: float = 0.9
    feedback_max_reward_ratio: float = 0.0
    feedback_reward_scale_floor: float = 0.25
    # 0 = auto (25% of step cap); -1 = unlimited; >0 = explicit cap.
    feedback_max_per_signature: int = 0
    feedback_auto_signature_fraction: float = 0.25


# ======================================================================
# CLI parsing + config loading (defaults < YAML < CLI)
# ======================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTT-Discover multi-problem runner")
    # Problem selection
    p.add_argument("--problem", default=None,
                   help="Problem name. Loads configs/<problem>.yaml unless --config "
                        "is given. Defaults to Config.problem, so changing that one "
                        "field is enough to switch problems. One of: circle_packing, "
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
    p.add_argument("--generation-backend", choices=["hf", "vllm"], default=None,
                   help="Engine used by generation workers. Independent of the "
                        "differentiable training backend.")
    p.add_argument("--model-name", default=None)
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
    p.add_argument("--kl-penalty-coef", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
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
                   help="Threads for reward evaluation. 0 = auto. Use 1 for any "
                        "problem whose reward is a measured runtime.")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of generation GPUs. Normally omit this: it is "
                        "derived from --gpu-ids. Passing it alone selects 0..N-1.")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated physical GPU ids for the workers, e.g. "
                        "'0,1,2,3,4,5,6,7'. Its length sets num_gpus.")
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

    # ---- memory (Sec. 2.2) ----
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


def load_config():
    """
    Merge Config() defaults < YAML < resumed config.json < CLI flags.

    Returns (cfg, merged) where:
      cfg    is a Config built from the engine-level fields, and
      merged is the full dict (including problem-only keys like num_circles,
             problem_type, budget_s, score_scale, gpu_type, task_yaml, lib_dir),
             which is what the problem registry consumes.
    """
    args = _build_arg_parser().parse_args()

    # Read the saved run identity early enough to select its YAML. New runs
    # persist the complete merged config; older ones at least contain the
    # dataclass fields and can recover standard problem YAMLs by name.
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

    # 1) defaults from the dataclass
    merged = {f.name: getattr(Config(), f.name) for f in fields(Config)}

    # The problem name comes from Config.problem unless --problem overrides it,
    # so editing that one field switches problems: it picks the YAML, and the
    # YAML then supplies everything else. Previously --problem carried its own
    # hardcoded default, which silently won over the dataclass.
    problem_name = (saved.get("problem") if saved
                    else (args.problem if args.problem is not None
                          else merged["problem"]))

    # 2) YAML overlay
    cfg_path = args.config
    if cfg_path is None and saved.get("problem_type"):
        typed = os.path.join(
            "configs", f"{problem_name}_{saved['problem_type']}.yaml"
        )
        if os.path.exists(typed):
            cfg_path = typed
    cfg_path = cfg_path or os.path.join("configs", f"{problem_name}.yaml")
    ydict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            ydict = yaml.safe_load(f) or {}
        merged.update(ydict)
        print(f"[config] loaded {cfg_path}")
    elif args.config is not None:
        raise FileNotFoundError(f"--config path not found: {cfg_path}")
    else:
        print(f"[config] no YAML at {cfg_path}; using Config() defaults + CLI. "
              f"This is usually a mistake: the problem's own knobs "
              f"(task paths, batch sizes, reward_workers) live in that file.")

    # The registry routing key is the YAML's `problem` field when present
    # (this lets e.g. configs/gpu_mode_trimul.yaml declare `problem: gpu_mode`
    # while --problem just selects the file). With no YAML, --problem is the key.
    merged["problem"] = ydict.get("problem", problem_name)

    # 3) Saved config overlay. Older code wrote max_seq_length after adding the
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

    # 4) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type", "resume"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
    if args.problem_type is not None:
        merged["problem_type"] = args.problem_type

    # vLLM is an inference engine, not a differentiable trainer. Treat the
    # convenient `--backend vllm` spelling as the complete no-Unsloth mode.
    if str(merged.get("backend", "")).lower() == "vllm":
        merged["backend"] = "hf"
        merged["generation_backend"] = "vllm"
        print("[config] vLLM mode: HF+PEFT training + vLLM generation")

    generation_backend = str(merged.get("generation_backend", "hf")).lower()
    if generation_backend not in ("hf", "vllm"):
        raise ValueError("generation_backend must be 'hf' or 'vllm'")
    merged["generation_backend"] = generation_backend
    util = float(merged.get("vllm_gpu_memory_utilization", 0.9))
    if not 0.0 < util <= 1.0:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")

    # Omitted growth caps mean fixed batch size. Resolve after YAML and CLI so
    # `--groups-per-step 5 --group-size 16` becomes max G=5, max K=16 even if
    # the selected problem YAML has different starting values.
    if args.groups_per_step is not None and args.max_groups_per_step is None:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if args.group_size is not None and args.max_group_size is None:
        merged["max_group_size"] = int(merged["group_size"])
    if merged.get("max_groups_per_step") is None:
        merged["max_groups_per_step"] = int(merged["groups_per_step"])
    if merged.get("max_group_size") is None:
        merged["max_group_size"] = int(merged["group_size"])
    if int(merged["max_groups_per_step"]) < int(merged["groups_per_step"]):
        raise ValueError("max_groups_per_step cannot be below groups_per_step")
    if int(merged["max_group_size"]) < int(merged["group_size"]):
        raise ValueError("max_group_size cannot be below group_size")

    # A GPU list is already an exact worker specification, so its length is the
    # single source of truth. `--num-gpus N` remains as a convenience when no
    # list is passed and expands to devices 0..N-1.
    gpu_ids = _parse_gpu_ids(merged.get("gpu_ids"))
    if args.gpu_ids is not None:
        if args.num_gpus is not None and int(args.num_gpus) != len(gpu_ids):
            raise ValueError(
                f"--num-gpus={args.num_gpus} disagrees with {len(gpu_ids)} "
                f"id(s) in --gpu-ids")
        merged["num_gpus"] = len(gpu_ids)
    elif args.num_gpus is not None:
        n_gpus = int(args.num_gpus)
        if n_gpus < 0:
            raise ValueError("num_gpus must be >= 0")
        gpu_ids = list(range(n_gpus))
        merged["num_gpus"] = n_gpus
    elif gpu_ids:
        merged["num_gpus"] = len(gpu_ids)
    else:
        n_gpus = int(merged.get("num_gpus") or 0)
        if n_gpus < 0:
            raise ValueError("num_gpus must be >= 0")
        gpu_ids = list(range(n_gpus))
        merged["num_gpus"] = n_gpus
    merged["gpu_ids"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    print(f"[config] generation GPUs: {gpu_ids} "
          f"(num_gpus={merged['num_gpus']}, derived from gpu_ids)")

    # 5) build the Config from the fields it knows; leave the rest in `merged`
    known = {f.name for f in fields(Config)}
    cfg_kwargs = {k: v for k, v in merged.items() if k in known}
    cfg = Config(**cfg_kwargs)
    if cfg.generation_backend == "vllm" and int(cfg.num_gpus or 0) < 1:
        raise ValueError("vLLM generation requires num_gpus >= 1")
    print(f"[config] rollout thinking: "
          f"{'enabled' if cfg.thinking else 'disabled'}")
    merged["_resume_dir"] = str(resume_dir) if resume_dir is not None else None
    return cfg, merged


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

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
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


def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg: Config):
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


# ======================================================================
# Logprob computation
# ======================================================================
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool,
                           chunk: int = 0):
    """
    Per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs

    `chunk` caps the transient log_softmax allocation. The forward runs once (a
    single model(full_ids) call, needed for correct causal attention), but the
    float32 log_softmax + gather over the response positions is done in slices
    of at most `chunk` tokens. The full (1, R, V) float32 tensor is what spikes
    memory on a large-vocab model; slicing caps the spike at (1, chunk, V) no
    matter how long the response is or how big the vocab is. log_softmax is
    per-position over the vocab dim, so this is EXACT, not an approximation.

    chunk <= 0 (or chunk >= R) takes the original single-shot path, byte-
    identical to before this argument existed, so a deterministic run with
    chunk=0 draws exactly what it did before.
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    P = prompt_ids.shape[1]
    R = response_ids.shape[1]
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        # Predict response token at position P+k from logits at position P+k-1.
        pred_logits = logits[:, P - 1 : P - 1 + R, :]  # (1, R, V)

        if chunk and 0 < chunk < R:
            parts = []
            for s in range(0, R, chunk):
                e = min(s + chunk, R)
                lp = F.log_softmax(pred_logits[:, s:e, :].float(), dim=-1)
                g = lp.gather(2, response_ids[:, s:e].unsqueeze(-1)).squeeze(-1)
                parts.append(g)          # keep only (1, e-s); lp freed next iter
            gathered = torch.cat(parts, dim=1)  # (1, R)
        else:
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            gathered = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return gathered.squeeze(0)


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


def _save_adapter(model, exp_dir, step_idx):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    Adapters are retained because completed checkpoints refer to their matching
    step directory; an interrupted write can therefore never invalidate the
    previous resumable checkpoint.
    """
    out_dir = _adapter_dir(exp_dir, step_idx)
    # PEFT/Unsloth models support save_pretrained, which writes just the adapter
    model.save_pretrained(out_dir)
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
# Optional tuning: add `reward_workers: int = 0` to Config (0 = auto). Auto
# leaves ~one CPU core per GPU worker for the generation loop.
# ======================================================================
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg: Config, exp_dir, problem, gen_pool=None,
               memory=None, extractor=None, mem_cfg=None, lookup=None,
               curator=None, fb_cfg=None):
    import os
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from advantage import entropic_adaptive_advantages
    from sampler import State
    from experiment_io import save_parent_selections, save_rollout
    from problems.base import ParentContext
    from gen_workers import make_progress_bar

    from memory import (MemoryArm, RolloutRecord, allocate_memory_arms,
                        build_injection, credit_memory_arms, inject_block)
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
    memory_aware_prompt = "memory" in _inspect.signature(
        problem.build_prompt).parameters
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
        adapter_path = _save_adapter(model, exp_dir, step_idx)

    # ---- memory lookup (replaces the Eq. 7 retrieval) ----------------
    # One call covering every parent. An empty bank makes no call at all, so
    # step 0 is byte-identical to a --no-memory run.
    chosen_by_group = {}
    if memory is not None and lookup is not None:
        chosen_by_group = lookup.select_batch(
            parent_ctxs, step_idx=step_idx, adapter_path=adapter_path)

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
    # exactly cfg.group_size. Entropic advantages are computed over the union.
    prompt_jobs = []
    for g, _parent in enumerate(parents):
        pc = parent_ctxs[g]
        chosen = chosen_by_group.get(g, [])
        if memory is not None and mem_cfg is not None:
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

        for arm in arms:
            messages = base_messages[g]
            kept, n_tok = [], 0
            if arm.lessons:
                block, n_tok, kept = build_injection(
                    arm.lessons, tokenizer, getattr(mem_cfg, "token_budget", 0))
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
    n_reward_workers = getattr(cfg, "reward_workers", 0)
    if not n_reward_workers:
        n_reward_workers = max(1, (os.cpu_count() or 8) - max(0, cfg.num_gpus))
    reward_pool = ThreadPoolExecutor(max_workers=n_reward_workers)

    # (text, token_ids, prompt_job_index), arrival order under each parent.
    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}    # aligned RewardResult futures

    def _submit_rollout(job_idx, text, token_ids):
        job = prompt_jobs[job_idx]
        g = job["parent_group"]
        group_responses[g].append((text, token_ids, job_idx))
        fut = reward_pool.submit(
            problem.compute_reward, text, parent_ctxs[g], cfg.sandbox_timeout_s
        )
        reward_futures[g].append(fut)

    # ----- ROLLOUTS (streamed) + dispatch rewards as each rollout lands -----
    rollout_t0 = time.time()
    try:
        if gen_pool is not None:
            # Multi-GPU: consume the generation stream (the adapter was already
            # saved above, before the memory lookup). Each (worker, group) job
            # that lands is queued for reward eval right away, so by the time
            # generation finishes most rewards are done.
            for job_idx, job_results in gen_pool.iter_group_jobs(
                    prompts_by_group=[job["prompt_text"] for job in prompt_jobs],
                    group_size=cfg.group_size,
                    counts_by_group=[job["count"] for job in prompt_jobs],
                    adapter_path=adapter_path,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    step_idx=step_idx,
                ):
                for (text, token_ids) in job_results:
                    _submit_rollout(job_idx, text, token_ids)
        else:
            # Single-GPU: generate group by group; submit each group's rewards
            # right after it's generated so eval overlaps the next group's gen.
            backend.set_inference_mode()
            if cfg.deterministic:
                torch.manual_seed((int(cfg.seed) * 1_000_003
                                   + step_idx * 1009 + 13) % (2**31 - 1))
            gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
            try:
                for job_idx, job in enumerate(prompt_jobs):
                    responses, _ = generate_responses(
                        model, tokenizer, job["prompt_text"], job["count"], cfg
                    )
                    for (text, token_ids) in responses:
                        _submit_rollout(job_idx, text, token_ids)
                    gen_bar.update(len(responses))
            finally:
                gen_bar.close()

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
        responses = group_responses[g]          # list of (text, token_ids, job)
        futs = reward_futures[g]                 # aligned RewardResult futures

        rewards = []
        codes = []
        valids = []
        outs = []        # list of RewardResult
        for r_idx, (text, token_ids, job_idx) in enumerate(responses):
            res = futs[r_idx].result()           # already computed (or finishes now)
            rewards.append(res.reward)
            codes.append(res.code or "")
            valids.append(res.valid)
            outs.append(res)

        rewards_np = np.array(rewards, dtype=np.float64)
        advantages, beta = entropic_adaptive_advantages(rewards_np)

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
              f"valid={sum(valids)}/{len(valids)}  beta={beta:.9f}")

        # Outcome-based memory credit. All arms share this parent and the same
        # total K budget; expected_subsample_max corrects unequal arm sizes.
        arm_observations = {}
        for r_idx, (_text, _token_ids, job_idx) in enumerate(responses):
            job = prompt_jobs[job_idx]
            obs = arm_observations.setdefault(job["arm"], {
                "memory_ids": job["memory_ids"], "rewards": [], "valids": [],
            })
            obs["rewards"].append(float(rewards[r_idx]))
            obs["valids"].append(bool(valids[r_idx]))
        if (memory is not None and mem_cfg is not None
                and bool(getattr(mem_cfg, "outcome_credit", False))):
            updates = credit_memory_arms(
                memory, arm_observations, parent_val, step_idx)
            mem_arm_updates.extend({"group": g, **update} for update in updates)
            for update in updates:
                print(f"    memory {update['arm']}: n={update['n']} "
                      f"tail uplift={update['tail_uplift']:+.9f} "
                      f"valid={update['valid']}/{update['rollouts']}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, (text, token_ids, job_idx) in enumerate(responses):
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
                "beta": float(beta),
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
            for r_idx, (text, token_ids, job_idx) in enumerate(responses):
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
        constant = float(rewards_np.max() - rewards_np.min()) < 1e-12
        if constant and not (fb_candidate_on and fb_cfg.include_constant_groups):
            continue

        for r_idx, ((text, token_ids, job_idx), adv) in enumerate(
                zip(responses, advantages)):
            if len(token_ids) == 0:
                continue
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.to(model.device)
            response_ids = torch.tensor([token_ids], device=model.device)
            res = outs[r_idx]
            all_examples.append({
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": response_ids,
                "advantage": float(adv),
                "behavior_logprobs": None,   # IS disabled (workers don't return logprobs)
                "reprompt_text": reprompt_by_key.get((g, r_idx)),
                "failure_signature": RolloutRecord(
                    msg=res.msg or "").failure_signature(),
                "reward_constant": constant,
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
    if not fb_on:
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
            if not (ex["reward_constant"] and not ex.get("reprompt_text"))
        ]

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
    }

    if not all_examples:
        print(f"[step {step_idx}] no training signal (all groups had constant reward)")
        return step_stats

    # ----- TRAIN STEP -----
    print(f"[step {step_idx}] starting adapter training; rollout artifacts "
          f"are already on disk", flush=True)
    backend.set_training_mode()
    optimizer.zero_grad()

    train_t0 = time.time()
    total_loss = 0.0
    total_logp_delta = 0.0
    n_examples = len(all_examples)

    is_ratio_sum = 0.0
    is_ratio_max = 0.0
    is_ratio_count = 0
    fb_stats = FeedbackStats()

    for ex in all_examples:
        pid = ex["prompt_ids"]
        rid = ex["response_ids"]
        adv = ex["advantage"]

        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True,
                                        chunk=cfg.logprob_chunk)  # (R,)

        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False,
                                                 chunk=cfg.logprob_chunk)
        except Exception as e:
            if not hasattr(train_step, "_kl_warned"):
                print(f"[warn] disable_adapter failed ({e}); training without KL penalty")
                train_step._kl_warned = True
            base_lp = cur_lp.detach()

        logp_diff = (cur_lp - base_lp).detach()
        avg_logp_diff = logp_diff.mean()
        kl_adv = cfg.kl_penalty_coef * (avg_logp_diff - (cur_lp - base_lp))
        eff_adv = adv + kl_adv

        # ---- feedback-based program repair (Sec. 2.3, Eq. 9) -------------
        # A_{i,l} = A^rew_i + lambda_f * d_i * A^fb_{i,l}, and d_i is implicit:
        # reprompt_text is only ever set for an eligible code failure.
        #
        # cur_lp.detach() IS log pi_thetabar here. Gradients accumulate across
        # every example and optimizer.step() runs once at the end of the loop,
        # so theta has not moved since the rollouts were sampled. If that ever
        # changes to more than one update per step, this term needs its own
        # forward pass at thetabar.
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
        if behavior_lp is not None and behavior_lp.shape[0] == cur_lp.shape[0]:
            is_ratio = torch.exp(cur_lp.detach() - behavior_lp)
            is_ratio_sum += float(is_ratio.mean().item())
            is_ratio_max = max(is_ratio_max, float(is_ratio.max().item()))
            is_ratio_count += 1
        else:
            if behavior_lp is not None and not hasattr(train_step, "_is_len_warned"):
                print(f"[warn] behavior/current logprob length mismatch "
                      f"({behavior_lp.shape[0]} vs {cur_lp.shape[0]}); "
                      f"skipping IS for affected examples")
                train_step._is_len_warned = True
            is_ratio = 1.0

        loss = -(is_ratio * eff_adv.detach() * cur_lp).mean()
        (loss / n_examples).backward()

        total_loss += float(loss.detach().item())
        total_logp_delta += float(logp_diff.mean().item())

    import torch as _torch
    _torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=cfg.grad_clip,
    )
    optimizer.step()

    train_time = time.time() - train_t0
    is_msg = ""
    if is_ratio_count > 0:
        is_msg = (f"  IS ratio mean={is_ratio_sum / is_ratio_count:.9f} "
                  f"max={is_ratio_max:.3f}")
    print(f"[step {step_idx}] train time: {train_time:.1f}s  "
          f"avg loss: {total_loss / n_examples:.9f}  "
          f"avg logpi_theta - logpi_base: {total_logp_delta / n_examples:.9f}{is_msg}")
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

    # Select the run directory before runtime-only context adjustments so a
    # fresh config.json stores the reusable base configuration.
    from experiment_io import (make_experiment_dir, save_final_summary,
                               save_step_summary)
    resume_dir = merged.pop("_resume_dir", None)
    exp_dir = make_experiment_dir(
        cfg, resume_dir=resume_dir, config_dict=merged
    )

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

    # Build the problem from the merged config (the registry reads problem-only
    # knobs like num_circles / problem_type / budget_s / score_scale from here).
    from problems.registry import get_problem
    problem = get_problem(cfg.problem, merged)

    print("=" * 70)
    print("TTT-Discover — local multi-problem implementation")
    print("=" * 70)
    print(f"Problem:            {cfg.problem}"
          + (f" ({cfg.problem_type})" if cfg.problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Model:              {cfg.model_name}")
    print(f"Training backend:   {cfg.backend}")
    print(f"Generation backend: {cfg.generation_backend}")
    print(f"Target:             {cfg.target}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    print(f"Max seq length:     {cfg.max_seq_length}")
    print(f"Logprob chunk:      {cfg.logprob_chunk or 'off (single shot)'}")
    print(f"Seed:               {cfg.seed}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print(f"Memory:             {'on' if mem_cfg.enabled else 'off'}")
    print(f"Feedback signal:    "
          f"{'on' if bool(merged.get('feedback', False)) else 'off'}")
    print("=" * 70)

    # ---- experiment dir ----
    action = "resuming in" if resume_dir else "writing all rollouts to"
    print(f"[init] {action}: {exp_dir}")

    # ---- seed states (problem-defined) ----
    seeds = problem.seed_states()
    print(f"[init] problem produced {len(seeds)} seed state(s)")

    # ---- backend + model ----
    # Load backend FIRST so Unsloth can patch transformers if used.
    from model_backend import load_backend
    backend = load_backend(cfg.backend, cfg)
    model, tokenizer = backend.load()

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
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        print("[resume] restored optimizer state")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,} / total {total:,} "
          f"({100 * trainable / total:.2f}%)")

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
    # The original HF path stays in-process for one GPU. vLLM always runs in a
    # worker so its CUDA/runtime state is isolated from the training model.
    use_gen_pool = bool(cfg.num_gpus) and (
        cfg.num_gpus > 1 or cfg.generation_backend == "vllm")
    if use_gen_pool:
        from gen_workers import GenerationPool
        gpu_ids = _parse_gpu_ids(cfg.gpu_ids)
        print(f"[init] starting {cfg.generation_backend} generation pool: "
              f"{cfg.num_gpus} GPUs {gpu_ids}")
        gen_pool = GenerationPool(
            model_name=cfg.model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            seed=run_seed,
            gen_micro_batch=cfg.gen_micro_batch,
            backend=cfg.generation_backend,
            lora_rank=cfg.lora_rank,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            vllm_enforce_eager=cfg.vllm_enforce_eager,
            vllm_enable_prefix_caching=cfg.vllm_enable_prefix_caching,
        )
        if cfg.gen_micro_batch and cfg.gen_micro_batch > 0:
            limit_name = ("max_num_seqs" if cfg.generation_backend == "vllm"
                          else "micro-batch")
            print(f"[init] generation pool ready "
                  f"({limit_name} {cfg.gen_micro_batch}/GPU)")
        else:
            print("[init] generation pool ready")
    else:
        print("[init] single-GPU generation (no worker pool)")

    # ---- memory (Sec. 2.2) ----
    from memory import setup_memory
    mem_cfg, memory, extractor, lookup, curator = setup_memory(
        merged, problem, cfg, mem_cfg=mem_cfg,
        backend=backend, model=model, tokenizer=tokenizer,
        gen_pool=gen_pool, exp_dir=exp_dir,  seed=run_seed,
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

    # ---- Elo re-ranker (optional, background thread) ----
    reranker = None
    try:
        from reranker.config import RerankerConfig
        rcfg = RerankerConfig.from_dict(merged)
        if rcfg.enabled:
            from reranker.judges import make_judge
            from reranker.reranker import MultiAgentReRanker
            judge = make_judge(rcfg)
            if judge is not None:
                reranker = MultiAgentReRanker(
                    sampler=sampler,
                    judge=judge,
                    cfg=rcfg,
                    metric_name=getattr(problem, "metric_name", "score"),
                    maximize=getattr(problem, "maximize", True),
                    target=getattr(problem, "target", None),
                    exp_dir=exp_dir,
                )
                reranker.start()
                print(f"[init] Elo re-ranker started "
                      f"(backend={rcfg.backend}, model={rcfg.model}, "
                      f"top_k={rcfg.top_k}, debate={rcfg.debate})")
            else:
                print("[init] Elo re-ranker enabled but judge unavailable; "
                      "continuing with rank-based prior")
        else:
            print("[init] Elo re-ranker disabled")
    except Exception as e:
        print(f"[init] Elo re-ranker setup failed ({e!r}); "
              f"continuing with rank-based prior")
        reranker = None

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
                               lookup=lookup, curator=curator, fb_cfg=fb_cfg)

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
            adapter_path = _save_adapter(model, exp_dir, step)
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
