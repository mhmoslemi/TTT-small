"""
TTT-Discover — multi-problem local runner.

Config():  defaults  <  YAML  <  CLI flags
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import argparse
import random
import time
from dataclasses import dataclass, field, fields
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
    backend: str = "auto"        # "auto" | "unsloth" | "hf"
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
    max_groups_per_step: int = 8
    max_group_size: int = 64
    growth_force_step: int = 10    # from this step on, run at max unconditionally
    growth_valid_yield: float = 0.7   # best group's valid fraction must reach this
    growth_distinct_min: int = 4      # this many distinct improved children needed
    growth_factor: float = 2.0        # multiply G and K by this when both clear
    learning_rate: float = 4e-5
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 4200
    grad_clip: float = 1.0

    train_examples_per_microbatch: int = 1

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0

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
    num_gpus: int = 4
    # num_gpus: int = 1
    # gpu_ids: str = "0,1,2,3,4,5,6,7"
    gpu_ids: str = "1,2,4,6"
    # gpu_ids: str = "1"

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

    # ---- Feedback-based failure signal (Sec. 2.3, Eq. 9) ----
    # `feedback` is the master switch, same rule as `memory`: when it is False
    # every feedback_* field below is ignored.
    feedback: bool = False
    feedback_lambda: float = 0.2           # lambda_f at step 0
    feedback_anneal_steps: int = 0         # 0 = constant; 10 = off from step 10
    feedback_anneal_shape: str = "linear"  # linear | cosine
    feedback_lambda_final: float = 0.0
    feedback_clip: float = 5.0             # clamp |A^fb| per token; 0 = Eq. 9 as written
    feedback_chars: int = 1200             # verifier text budget in the reprompt
    feedback_max_per_step: int = 0         # 0 = every failed rollout
    feedback_include_constant_groups: bool = True
    feedback_inject_mode: str = "append"   # append | user_turn
    feedback_normalize: bool = False


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
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default=None)
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
                   help="Cap that G (groups per step) ratchets up to.")
    p.add_argument("--max-group-size", type=int, default=None,
                   help="Cap that K (rollouts per group) ratchets up to.")
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
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
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
                   help="Number of GPUs for parallel generation. 1 = single-process "
                        "in-line generation (no worker pool). >1 spawns that many "
                        "plain-HF generation workers, one per GPU.")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated physical GPU ids for the workers, e.g. "
                        "'0,1,2,3,4,5,6,7'. Defaults to 0..num_gpus-1.")

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
    p.add_argument("--feedback-max-per-step", type=int, default=None)
    p.add_argument("--feedback-inject-mode",
                   choices=["append", "user_turn"], default=None)
    p.add_argument("--feedback-normalize", action="store_const",
                   const=True, default=None)

    return p


# CLI arg name -> config key (only where they differ)
_CLI_TO_CFG = {"lr": "learning_rate"}


def load_config():
    """
    Merge Config() defaults < YAML(configs/<problem>.yaml or --config) < CLI flags.

    Returns (cfg, merged) where:
      cfg    is a Config built from the engine-level fields, and
      merged is the full dict (including problem-only keys like num_circles,
             problem_type, budget_s, score_scale, gpu_type, task_yaml, lib_dir),
             which is what the problem registry consumes.
    """
    args = _build_arg_parser().parse_args()

    # 1) defaults from the dataclass
    merged = {f.name: getattr(Config(), f.name) for f in fields(Config)}

    # The problem name comes from Config.problem unless --problem overrides it,
    # so editing that one field switches problems: it picks the YAML, and the
    # YAML then supplies everything else. Previously --problem carried its own
    # hardcoded default, which silently won over the dataclass.
    problem_name = args.problem if args.problem is not None else merged["problem"]

    # 2) YAML overlay
    cfg_path = args.config or os.path.join("configs", f"{problem_name}.yaml")
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

    # 3) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
    if args.problem_type is not None:
        merged["problem_type"] = args.problem_type

    # 4) build the Config from the fields it knows; leave the rest in `merged`
    known = {f.name for f in fields(Config)}
    cfg_kwargs = {k: v for k, v in merged.items() if k in known}
    cfg = Config(**cfg_kwargs)
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
    # Start by trying the whole group in one call.
    batch = group_size

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
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool):
    """
    Returns the per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        # Predict response token at position P+k from logits at position P+k-1
        pred_logits = logits[:, P - 1 : P - 1 + R, :]
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
    Returns the directory path. Cleans up the previous step's adapter to
    avoid filling the disk (we only ever need the latest).
    """
    import shutil
    from pathlib import Path

    out_dir = _adapter_dir(exp_dir, step_idx)
    # PEFT/Unsloth models support save_pretrained, which writes just the adapter
    model.save_pretrained(out_dir)

    # Remove older adapter dirs (keep only the current one)
    for old in Path(exp_dir).glob("adapter_step*"):
        if str(old) != out_dir:
            try:
                shutil.rmtree(old)
            except Exception:
                pass

    return out_dir


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
    from experiment_io import save_rollout
    from problems.base import ParentContext
    from gen_workers import make_progress_bar

    from memory import RolloutRecord, build_injection, inject_block
    from feedback import (FeedbackStats, build_reprompt, feedback_advantage,
                          format_feedback, is_failure, render_chat,
                          select_capped)

    step_t0 = time.time()
    sampler.set_current_step(step_idx)
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.9f}  n={info['n']}  "
              f"Q={info['Q']:.9f}  P={info['P']:.9f}  bonus={info['bonus']:.9f}  "
              f"score={info['score']:.9f}")

    # The coefficient in force this step. When it reaches zero the whole
    # feedback path is skipped: no reprompts built, no teacher forwards, so the
    # annealed tail costs exactly what a no-feedback run costs.
    import inspect as _inspect
    memory_aware_prompt = "memory" in _inspect.signature(
        problem.build_prompt).parameters
    # Only problems that declare it get their construction written to disk.
    save_ctor = (bool(getattr(problem, "saves_construction", False))
                 and int(getattr(cfg, "max_saved_construction", 0)) != 0)

    fb_lambda = fb_cfg.lambda_at(step_idx) if fb_cfg is not None else 0.0
    fb_on = bool(fb_cfg is not None and fb_cfg.enabled and fb_lambda > 0.0)
    if (fb_cfg is not None and fb_cfg.enabled and not fb_on):
        print(f"[step {step_idx}] feedback: lambda annealed to 0, term disabled")
    fail_score = float(getattr(problem, "fail_score", 0.0))
    reprompt_by_key = {}        # (group, rollout) -> reprompt text for a failure

    all_examples = []
    all_children = []
    mem_records = []            # RolloutRecord per rollout, for the memory maker
    retrieved_by_group = {}     # group -> ids of the lessons put in its prompt
    mem_tokens_by_group = {}    # group -> tokens the injected block occupied
    messages_by_group = {}      # group -> the message list, for reprompt(x_p, f_i)

    # ----- BUILD PROMPTS (one per parent/group) -----
    # Three passes now, because memory lookup is a batched LLM call rather than
    # a vector query: collect the parent contexts, ask the model once which
    # lessons it wants for all of them, then render.
    prompts_by_group = []
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

    for g, parent in enumerate(parents):
        messages = base_messages[g]
        pc = parent_ctxs[g]

        chosen = chosen_by_group.get(g, [])
        if memory is not None and chosen:
            # Order is the order the model asked for, so build_injection trims
            # from the tail: its last pick is the one it wanted least.
            block, n_tok, kept = build_injection(
                chosen, tokenizer, getattr(mem_cfg, "token_budget", 0))
            retrieved_by_group[g] = [l.id for l in kept]
            mem_tokens_by_group[g] = n_tok
            # Preferred path: the problem places the block itself, between the
            # parent state and the instruction, and adapts its instruction to
            # it. Problems that predate the `memory` argument fall back to the
            # trainer appending the block after the instruction.
            if memory_aware_prompt:
                messages = problem.build_prompt(pc, memory=block)
            else:
                messages = inject_block(
                    messages, block,
                    mode=getattr(mem_cfg, "inject_mode", "append"))
        elif memory is not None:
            retrieved_by_group[g] = []
            mem_tokens_by_group[g] = 0

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        messages_by_group[g] = messages
        prompts_by_group.append(prompt_text)

    if memory is not None and mem_tokens_by_group:
        vals = list(mem_tokens_by_group.values())
        used = sum(1 for v in vals if v)
        print(f"[step {step_idx}] memory injected {used}/{len(vals)} prompts, "
              f"{min(vals)}-{max(vals)} tokens "
              f"(budget {getattr(mem_cfg, 'token_budget', 0)}); "
              f"max_new_tokens unchanged at {cfg.max_new_tokens}")

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

    group_responses = {g: [] for g in range(num_groups)}   # (text, token_ids), arrival order
    reward_futures = {g: [] for g in range(num_groups)}    # aligned RewardResult futures

    def _submit_rollout(g, text, token_ids):
        group_responses[g].append((text, token_ids))
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
            for group_idx, job_results in gen_pool.iter_group_jobs(
                    prompts_by_group=prompts_by_group,
                    group_size=cfg.group_size,
                    adapter_path=adapter_path,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    step_idx=step_idx,
            ):
                for (text, token_ids) in job_results:
                    _submit_rollout(group_idx, text, token_ids)
        else:
            # Single-GPU: generate group by group; submit each group's rewards
            # right after it's generated so eval overlaps the next group's gen.
            backend.set_inference_mode()
            if cfg.deterministic:
                torch.manual_seed((int(cfg.seed) * 1_000_003
                                   + step_idx * 1009 + 13) % (2**31 - 1))
            gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
            try:
                for g, prompt_text in enumerate(prompts_by_group):
                    responses, _ = generate_responses(
                        model, tokenizer, prompt_text, cfg.group_size, cfg
                    )
                    for (text, token_ids) in responses:
                        _submit_rollout(g, text, token_ids)
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

    for g, parent in enumerate(parents):
        prompt_text = prompts_by_group[g]
        responses = group_responses[g]          # list of (text, token_ids)
        futs = reward_futures[g]                 # aligned RewardResult futures
        pc = parent_ctxs[g]

        rewards = []
        codes = []
        valids = []
        outs = []        # list of RewardResult
        for r_idx, (text, token_ids) in enumerate(responses):
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
        parent_val = float(parent.value) if parent.value is not None else 0.0
        for r_idx in range(len(responses)):
            if valids[r_idx] and codes[r_idx] and rewards[r_idx] > parent_val:
                distinct_good_hashes.add(hash(codes[r_idx].strip()))

        print(f"  group {g}: rewards min={rewards_np.min():.9f} "
              f"mean={rewards_np.mean():.9f} max={rewards_np.max():.9f}  "
              f"valid={sum(valids)}/{len(valids)}  beta={beta:.9f}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            meta = {
                "step": step_idx,
                "group": g,
                "rollout": r_idx,
                "reward": float(rewards[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "valid": bool(valids[r_idx]),
                "parsed": bool(res.parsed),
                "ran": bool(res.ran),
                "msg": res.msg,
                "advantage": float(advantages[r_idx]) if hasattr(advantages, "__len__") else 0.0,
                "beta": float(beta),
                "n_response_tokens": len(token_ids),
                "sandbox_stdout": (res.stdout or "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_is_seed": parent.id in sampler._seed_ids,
                "memory_ids": retrieved_by_group.get(g, []),
                "memory_tokens": mem_tokens_by_group.get(g, 0),
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
                         prompt_text=prompt_text)
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
                ))

        # ---- reprompt(x_p, f_i) for every failed rollout (Sec. 2.3) ------
        # Built here, while the RewardResult is in hand. The teacher forward
        # itself happens in the train loop, where log pi_thetabar is already
        # available from the existing forward pass.
        if fb_on:
            for r_idx, (text, token_ids) in enumerate(responses):
                res = outs[r_idx]
                if not is_failure(res, fail_score):
                    continue
                f_i = format_feedback(res.msg or "", res.stdout or "",
                                      int(fb_cfg.chars))
                rp_messages = build_reprompt(messages_by_group[g], f_i,
                                             mode=fb_cfg.inject_mode)
                reprompt_by_key[(g, r_idx)] = render_chat(tokenizer, rp_messages)

        # Children for the sampler
        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            if valids[r_idx] and codes[r_idx]:
                child = State.make(
                    timestep=step_idx,
                    value=rewards[r_idx],
                    code=codes[r_idx],
                    raw_score=res.raw_score,
                    construction=res.construction,
                )
                all_children.append((child, parent))

        # If reward is constant in this group there is no A^rew signal. With
        # the feedback signal on, those rollouts are still worth training on:
        # A^rew_i = 0 but A^fb is not, which is the whole point of Eq. 9. This
        # is where it pays most, since an all-failed group is exactly the case
        # the reward channel cannot score at all.
        constant = float(rewards_np.max() - rewards_np.min()) < 1e-12
        if constant and not (fb_on and fb_cfg.include_constant_groups):
            continue

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        for r_idx, ((text, token_ids), adv) in enumerate(zip(responses, advantages)):
            if len(token_ids) == 0:
                continue
            response_ids = torch.tensor([token_ids], device=model.device)
            all_examples.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "advantage": float(adv),
                "behavior_logprobs": None,   # IS disabled (workers don't return logprobs)
                "reprompt_text": reprompt_by_key.get((g, r_idx)),
            })

    # Cap the teacher forwards. Applied to all_examples rather than to
    # reprompt_by_key, because the examples were built during the scoring loop
    # above and already hold their reprompt text; shrinking the dict now would
    # change nothing. Even stride, so the cap does not silently restrict the
    # signal to the first group or two.
    if fb_on and fb_cfg.max_per_step > 0:
        with_fb = [i for i, ex in enumerate(all_examples) if ex.get("reprompt_text")]
        if len(with_fb) > fb_cfg.max_per_step:
            keep = set(select_capped(with_fb, int(fb_cfg.max_per_step)))
            for i in with_fb:
                if i not in keep:
                    all_examples[i]["reprompt_text"] = None
            print(f"[step {step_idx}] feedback: capped to {fb_cfg.max_per_step} "
                  f"of {len(with_fb)} failed rollouts")

    rollout_time = time.time() - rollout_t0
    print(f"[step {step_idx}] rollout+eval time: {rollout_time:.1f}s  "
          f"training examples: {len(all_examples)}  new children: {len(all_children)}")

    # Update archive
    sampler.update(all_children)

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

    step_stats = {"best_valid_yield": float(best_valid_yield),
                  "distinct_good": int(len(distinct_good_hashes))}

    if not all_examples:
        print(f"[step {step_idx}] no training signal (all groups had constant reward)")
        return step_stats

    # ----- TRAIN STEP -----
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

        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)  # (R,)

        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False)
        except Exception as e:
            if not hasattr(train_step, "_kl_warned"):
                print(f"[warn] disable_adapter failed ({e}); training without KL penalty")
                train_step._kl_warned = True
            base_lp = cur_lp.detach()

        logp_diff = (cur_lp - base_lp).detach()
        avg_logp_diff = logp_diff.mean()
        kl_adv = cfg.kl_penalty_coef * (avg_logp_diff - (cur_lp - base_lp))
        eff_adv = adv + kl_adv

        # ---- feedback-based failure signal (Sec. 2.3, Eq. 9) -------------
        # A_{i,l} = A^rew_i + lambda_f * d_i * A^fb_{i,l}, and d_i is implicit:
        # reprompt_text is only ever set for a failed rollout.
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
                lam=fb_lambda)
            if fb_adv is None:
                fb_stats.skipped += 1
            else:
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
    print(f"Backend:            {cfg.backend}")
    print(f"Target:             {cfg.target}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    print(f"Max seq length:     {cfg.max_seq_length}")
    print(f"Seed:               {cfg.seed}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print(f"Memory:             {'on' if mem_cfg.enabled else 'off'}")
    print(f"Feedback signal:    "
          f"{'on' if bool(merged.get('feedback', False)) else 'off'}")
    print("=" * 70)

    # ---- experiment dir ----
    from experiment_io import make_experiment_dir, save_final_summary
    exp_dir = make_experiment_dir(cfg)
    print(f"[init] writing all rollouts to: {exp_dir}")

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


    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0
    )
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
    print(f"[init] sampler archive size = {sampler.archive_size()}")

    # ---- generation pool (multi-GPU) ----
    gen_pool = None
    if cfg.num_gpus and cfg.num_gpus > 1:
        from gen_workers import GenerationPool
        if cfg.gpu_ids:
            gpu_ids = [int(x) for x in cfg.gpu_ids.split(",")]
        else:
            gpu_ids = list(range(cfg.num_gpus))
        print(f"[init] starting generation pool: {cfg.num_gpus} GPUs {gpu_ids}")
        gen_pool = GenerationPool(
            model_name=cfg.model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            seed=run_seed,
        )
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
    # ---- feedback-based failure signal (Sec. 2.3) ----
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
    cur_g = int(cfg.groups_per_step)
    cur_k = int(cfg.group_size)
    print(f"[init] batch growth: start G={cur_g} K={cur_k} -> "
          f"max G={cfg.max_groups_per_step} K={cfg.max_group_size}; "
          f"grow when best-valid-yield>={cfg.growth_valid_yield} and "
          f"distinct-good>={cfg.growth_distinct_min} (x{cfg.growth_factor}); "
          f"forced to max at step {cfg.growth_force_step}")

    # ---- main loop ----
    try:
        for step in range(cfg.num_steps):
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