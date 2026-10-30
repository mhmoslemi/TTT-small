"""TTT-Discover multi-problem runner with entropic and SPO-RS training.

Configuration: saved settings, current YAML rollout/strategy controls, then CLI.
Entropic training uses adaptive-beta, leave-one-out group advantages.
SPO-RS uses a historical run-wide baseline and clipped policy updates.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import sys
import json
import math
import random
import re
import shutil
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from generation.output_retries import (
    coder_output_issue, coder_retry_gate, coder_retry_prompt_job, output_retry_messages,
    retry_metadata, strategy_retry_needed,
)
from models.strategist_models import (
    render_strategy_prompt, strategist_engine_options, strategist_profile, validate_strategist_runtime,
)
# Same-name aliases retain entrypoint exports used by existing tools/tests.
from diagnostics.console import (
    _LOG_TIME_OFFSET_SECONDS, _NoticeRoutingStream as _NoticeRoutingStream,
    _TerminalLogSink as _TerminalLogSink, _TerminalTeeStream as _TerminalTeeStream,
    _TimestampedLineStream as _TimestampedLineStream, _install_console_timestamps, _install_terminal_log,
    _route_dependency_notices, bind_setting_log, setting_log_only,
)
from models.coder_profiles import (
    _apply_qwen3_8b_thinking_sampling as _apply_qwen3_8b_thinking_sampling, _coder_effort_entries,
    _coder_effort_for_rollout_phase, _coder_model_profile as _coder_model_profile,
    _coder_vllm_persistent_workers, _model_basename as _model_basename, _phase_coder_prompt_job,
    _strategy_model_name, _uses_sequence_level_policy_ratio,
)
from runtime.configuration import (
    _build_arg_parser as _build_arg_parser, _isolated_evaluation_cpu_ids,
    _normalize_saved_advantage_config as _normalize_saved_advantage_config, _parse_gpu_ids,
    _resolve_training_backend as _resolve_training_backend,
    _resolve_training_memory_budgets as _resolve_training_memory_budgets,
    _resolve_training_model_name as _resolve_training_model_name, _uses_clipped_policy_loss,
    _validate_known_training_capacity as _validate_known_training_capacity, load_config,
)
from training.exact_overflow import (
    ExactLongContextTrainingOOM, _clear_exact_sharded_recovery,
    _clip_sharded_grad_norm_ as _clip_sharded_grad_norm_,
    _commit_exact_spo_rs_overflow_progress as _commit_exact_spo_rs_overflow_progress,
    _load_exact_spo_rs_overflow, _read_exact_sharded_recovery, _restart_for_exact_resume,
    _resume_process_argv as _resume_process_argv, _save_exact_spo_rs_overflow as _save_exact_spo_rs_overflow,
    _write_exact_sharded_recovery, _validate_precomputed_exact_gradients,
)
from training.fast_distributed import _move_optimizer_state, _restore_optimizer_state_to_parameters
from training.trainers import (
    ProcessDistributedTrainer, ReplicatedDataParallelTrainer, _a3b_sequence_clipped_standard_loss,
    _apply_precomputed_exact_gradients as _apply_precomputed_exact_gradients,
    _clipped_policy_options as _clipped_policy_options, _complete_spo_rs_overlap_examples,
    _detached_behavior_importance_ratio, _initialize_behavior_logprobs as _initialize_behavior_logprobs,
    _policy_dropout_disabled, _prepare_entropic_overlap_group, _prepare_spo_rs_overlap_inputs,
    _run_oom_resilient_backward, _train_spo_rs_examples, _training_microbatches,
    _valid_example_token_logprobs, clipped_policy_loss as clipped_policy_loss,
    clipped_surrogate_loss as clipped_surrogate_loss, compute_batched_token_logprobs,
    compute_group_advantages, compute_token_logprobs as compute_token_logprobs,
)


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
_ANSI_ORANGE = "\033[38;5;208m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"


def _extract_final_strategy(response_text, *, strict=False):
    """Extract only the final, complete strategy block.

    Explicit JSON plans and named plan fences are normalized before extraction.

    GPT-OSS may expose a long analysis channel before its final answer. None of
    that text is allowed into a later strategist or coder prompt. We retain
    only the body of the last complete ``<strategy>...</strategy>`` block at
    the end of the response. An explicit final channel is preferred when the
    template exposes one, but is not required: preceding reasoning is harmless
    because it is never returned. A missing, empty, or unclosed final block is
    retried by the caller. If every retry is malformed, the caller deliberately
    hands the coder the explicit no-strategy fallback instead of aborting the
    entire rollout step.
    """
    from generation.strategy_prompts_openai import (
        normalize_strategy_plan, strategy_answer_scope, strategy_boundary_view,
    )
    final_text = strategy_answer_scope(normalize_strategy_plan(response_text)).strip()
    if not final_text:
        return _STRATEGY_FALLBACK, "empty response"

    boundary_view = strategy_boundary_view(final_text)
    matches = list(_STRATEGY_BLOCK_RE.finditer(boundary_view))
    if not matches:
        return _STRATEGY_FALLBACK, "missing complete final strategy"

    match = matches[-1]
    if re.search(r"<strategy\b", boundary_view[match.end():], flags=re.IGNORECASE):
        return _STRATEGY_FALLBACK, "unclosed final strategy block"
    # Match masked boundaries, but hand off the untouched original plan.
    body = final_text[match.start(1):match.end(1)].strip()
    if not body:
        return _STRATEGY_FALLBACK, "empty final strategy"
    placeholder = re.sub(r"[\W\d_]+", "", body).casefold()
    if not placeholder or placeholder in {"todo", "tbd", "placeholder"}:
        return _STRATEGY_FALLBACK, "incomplete final strategy"
    if not strict:
        return body, None

    if len(body) < 80:
        return _STRATEGY_FALLBACK, "missing complete final strategy"
    # Extra strict checks remain opt-in; quoted boundaries are never delimiters.
    if (re.search(r"```\s*python\b", body, flags=re.IGNORECASE)
            or re.search(
                r"```[^`]*(?:^|\n)\s*(?:def |class |from \S+ import |import \S+)",
                body, flags=re.IGNORECASE | re.DOTALL)):
        return _STRATEGY_FALLBACK, "strategy contained a code fence"
    if re.search(
            r"<\s*\.\.\.\s*>|\brest of (?:the )?(?:long )?strategy\b|"
            r"\bcontinue(?:d)? (?:in|with) (?:the )?next response\b|"
            r"\bplaceholder implementation\b|"
            r"\bcannot (?:provide|deliver|write) (?:the )?(?:full|complete)\b",
            body, flags=re.IGNORECASE):
        return _STRATEGY_FALLBACK, "incomplete final strategy"
    return body, None


# CLI arg name -> config key (only where they differ)


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
    print(f'[gpu] trainer model-parallel group: physical {ids} (logical {logical})')


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
    from generation.gen_workers import _canonical_generated_text

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    max_new_tokens = min(int(cfg.max_new_tokens), int(cfg.max_seq_length) - int(input_len))
    if max_new_tokens < 1:
        return [("", []) for _ in range(int(n_samples))]

    policy_sampling = {}
    sampling_top_k = getattr(cfg, "sampling_top_k", None)
    sampling_min_p = getattr(cfg, "sampling_min_p", None)
    if sampling_top_k is not None:
        policy_sampling["top_k"] = int(sampling_top_k)
    if sampling_min_p is not None:
        policy_sampling["min_p"] = float(sampling_min_p)
    if (_uses_clipped_policy_loss(getattr(cfg, "advantage_mode", "entropic"))
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
        policy_cfg = SimpleNamespace(**vars(cfg))
        policy_cfg.temperature = 1.0
        policy_cfg.top_p = 1.0
        policy_cfg.sampling_top_k = 0
        policy_cfg.sampling_min_p = 0.0
        if max_new_tokens is not None:
            policy_cfg.max_new_tokens = int(max_new_tokens)
        for group_idx, (prompt, count) in enumerate(zip(prompts_by_group, counts_by_group)):
            if int(count) > 0:
                responses, _ = generate_responses(model, tokenizer, prompt, int(count), policy_cfg)
                yield group_idx, responses
        return

    from generation.gen_workers import _iter_hf_job_batches
    jobs = [
        (group_idx, prompt, int(count))
        for group_idx, (prompt, count)
        in enumerate(zip(prompts_by_group, counts_by_group))
        if int(count) > 0
    ]
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens),
        "temperature": float(temperature if temperature is not None else cfg.temperature),
        "top_p": float(top_p if top_p is not None else cfg.top_p),
        "micro_batch": int(getattr(cfg, 'gen_micro_batch', 0) or 0),
    }
    resolved_top_k = (top_k if top_k is not None else getattr(cfg, "sampling_top_k", None))
    resolved_min_p = (min_p if min_p is not None else getattr(cfg, "sampling_min_p", None))
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


def _save_adapter(model, exp_dir, step_idx, generation_model_name=None, directory_name=None):
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
    # Explicit legacy configurations may train against a differently named
    # compatible base; always advertise the rollout base to the vLLM loader.
    config_path = Path(out_dir) / "adapter_config.json"
    if generation_model_name and config_path.is_file():
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        if adapter_config.get("base_model_name_or_path") != generation_model_name:
            adapter_config["base_model_name_or_path"] = generation_model_name
            config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    return out_dir


def _prune_adapter_snapshots(exp_dir, keep_paths):
    """Delete completed adapter snapshots not required by the live state.

    A normal run retains only the checkpoint's current adapter. SPO-RS passes
    its preceding sampling adapter as a second keep path because the exact
    consecutive-policy KL at the next step requires both policies.
    """
    root = Path(exp_dir)
    keep_names = {Path(path).name for path in keep_paths if path is not None}
    removed = []
    for path in root.iterdir():
        is_step_adapter = bool(re.fullmatch(r"adapter_step\d+", path.name))
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
            print(f'[warn] could not remove superseded adapter {path}: {error}', flush=True)
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
        weights = load_peft_weights(str(adapter_dir), device=str(next(model.parameters()).device))
    except (ImportError, TypeError):
        safe_path = adapter_dir / "adapter_model.safetensors"
        bin_path = adapter_dir / "adapter_model.bin"
        if safe_path.is_file():
            from safetensors.torch import load_file
            weights = load_file(str(safe_path), device=str(next(model.parameters()).device))
        elif bin_path.is_file():
            try:
                weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            except TypeError:
                weights = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"no adapter weights found under {adapter_dir}")

    set_peft_model_state_dict(model, weights)
    if announce:
        print(f"[resume] loaded LoRA adapter from {adapter_dir}")


def _save_training_checkpoint(exp_dir, next_step, adapter_path, sampler, optimizer, spo_rs_tracker=None):
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
        "spo_rs_tracker": (spo_rs_tracker.state_dict() if spo_rs_tracker is not None else None),
    }
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def _load_training_checkpoint(exp_dir, *, mmap=False):
    import torch

    path = Path(exp_dir) / "training_state.pt"
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, **({"mmap": True} if mmap else {}))
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


def _migrate_parent_selection(sampler, cfg, exp_dir, next_step, adapter_path,
                              optimizer, spo_rs_tracker, *, recovery_pending,
                              problem=None):
    """Switch only before fresh selection, preserving the complete training state."""
    if sampler.selection_mode == cfg.parent_selection:
        return sampler
    from search.sampler import can_migrate_parent_selection
    if not can_migrate_parent_selection(sampler.selection_mode, cfg.parent_selection):
        raise ValueError("unsupported resume parent-selection change")
    if recovery_pending:
        print(f"[resume] finishing interrupted step {next_step} with "
              f"{sampler.selection_mode.upper()}; "
              f"{cfg.parent_selection.upper()} starts at the next fresh step",
              flush=True)
        return sampler
    from search.llm_sampler import LLMSampler
    migrated = LLMSampler(
        num_seeds=0, exp_dir=exp_dir, problem=problem,
        puct_c=sampler.puct_c, max_buffer_size=sampler.max_buffer_size,
        topk_children=sampler.topk_children)
    migrated.load_state_dict(dict(sampler.state_dict(), selection_mode="llm"))
    detail = "the full evaluated tree is available for LLM selection"
    # Persist intent first: a crash before checkpoint replacement repeats the
    # migration, never recovers new-picker decisions with the legacy picker.
    path = Path(exp_dir) / "config.json"
    saved = json.loads(path.read_text())
    saved.update(parent_selection=cfg.parent_selection, uct=False)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, indent=2) + "\n")
    tmp.replace(path)
    _save_training_checkpoint(exp_dir, next_step, adapter_path, migrated, optimizer, spo_rs_tracker)
    print(f"[resume] {sampler.selection_mode.upper()} -> "
          f"{cfg.parent_selection.upper()} at step {next_step}; {detail}; "
          "archive, adapter, optimizer and SPO tracker preserved", flush=True)
    return migrated


def _prefetch_resume_parents(sampler, cfg, exp_dir, next_step, adapter_path,
                             optimizer, spo_rs_tracker, *, problem,
                             interrupted_artifacts=None):
    """Start fresh-step selection before resumed trainer/generation startup."""
    if (cfg.parent_selection != "llm" or next_step >= cfg.num_steps or interrupted_artifacts is not None):
        return sampler, None
    if (cfg.advantage_mode == "spo-rs" and _recoverable_pending_spo_rs_artifacts(exp_dir, next_step, cfg) is not None):
        return sampler, None
    sampler = _migrate_parent_selection(
        sampler, cfg, exp_dir, next_step, adapter_path,
        optimizer, spo_rs_tracker, recovery_pending=False, problem=problem)
    future = sampler.prefetch_states(int(cfg.groups_per_step), selection_step=next_step)
    print(f"[resume] LLM parent request for step {next_step} overlaps "
          "trainer initialization and generation warm-up", flush=True)
    return sampler, future


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
    from evaluation.reward import extract_python_code
    from search.sampler import State

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
        text_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".txt")
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
        states.append(State.make(timestep=step, value=reward, code=code, raw_score=raw, construction=construction,))
    sampler.import_legacy_states(states, total_expansions=total_rollouts)
    return len(states), total_rollouts


def _logged_step_rewards(exp_dir, step_idx):
    """Read one completed step's finite rewards for tracker migration."""
    step_dir = Path(exp_dir) / f"step{int(step_idx):02d}"
    rewards = []
    for meta_path in sorted(step_dir.glob(f"step{int(step_idx):02d}_group*_rollout*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
            reward = float(meta["reward"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            continue
        if math.isfinite(reward):
            rewards.append(reward)
    return rewards


def _recoverable_spo_rs_artifacts(exp_dir, step_idx, cfg):
    """Return a complete evaluated-but-uncheckpointed SPO-RS step.

    Rollout metadata is written only after reward evaluation, while the step
    summary is written only after the adapter and training checkpoint.  Their
    combination therefore forms a crash-safe boundary for replaying only the
    interrupted optimizer update without regenerating or reevaluating code.
    """
    step_idx = int(step_idx)
    step_dir = Path(exp_dir) / f"step{step_idx:02d}"
    summary_path = step_dir / f"step{step_idx:02d}.summary.json"
    if summary_path.is_file() or not step_dir.is_dir():
        return None
    paths = sorted(step_dir.glob(f"step{step_idx:02d}_group*_rollout*.meta.json"))
    expected_primary = int(cfg.groups_per_step) * int(cfg.group_size)
    if len(paths) < expected_primary:
        return None

    artifacts = []
    for meta_path in paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        required = (
            "step", "group", "rollout", "reward", "valid", "advantage",
            "advantage_mode", "n_response_tokens",
        )
        if any(name not in meta for name in required):
            return None
        if (int(meta["step"]) != step_idx or str(meta["advantage_mode"]).lower() != "spo-rs"):
            return None
        try:
            reward = float(meta["reward"])
            advantage = float(meta["advantage"])
            response_tokens = int(meta["n_response_tokens"])
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(reward) or not math.isfinite(advantage) or response_tokens < 0):
            return None
        base = meta_path.name.removesuffix(".meta.json")
        response_path = meta_path.with_name(base + ".txt")
        prompt_path = meta_path.with_name(base + ".prompt.txt")
        token_path = meta_path.with_name(base + ".tokens.zlib")
        behavior_path = meta_path.with_name(base + ".behavior_logprobs.zlib")
        if not response_path.is_file() or not prompt_path.is_file():
            return None
        artifacts.append({
            "meta_path": meta_path,
            "response_path": response_path,
            "prompt_path": prompt_path,
            "token_path": token_path if token_path.is_file() else None,
            "behavior_logprobs_path": (behavior_path if behavior_path.is_file() else None),
            "meta": meta,
        })

    primary = [
        item for item in artifacts
        if bool(item["meta"].get("counts_toward_allocation", True))
        and int(item["meta"].get("retry_attempt", 0) or 0) == 0
    ]
    if len(primary) != expected_primary:
        return None
    per_group = {}
    for item in primary:
        group = int(item["meta"]["group"])
        per_group[group] = per_group.get(group, 0) + 1
    expected_groups = set(range(int(cfg.groups_per_step)))
    if (set(per_group) != expected_groups or any(count != int(cfg.group_size) for count in per_group.values())):
        return None
    artifacts.sort(key=lambda item: (
        int(item["meta"]["group"]), int(item["meta"]["rollout"])))
    return artifacts


def _recoverable_pending_spo_rs_artifacts(exp_dir, step_idx, cfg):
    """Return a completely generated SPO-RS step awaiting final evaluation.

    Generation writes response text, exact token IDs, rollout-policy
    logprobs, prompts, and provisional metadata atomically as each rollout
    arrives.  If exact consecutive-policy scoring later fails, those durable
    artifacts are sufficient to resume without sampling the policy again.
    Rewards are deliberately recomputed because provisional metadata never
    claims that an in-memory evaluation survived a process crash.
    """
    step_idx = int(step_idx)
    step_dir = Path(exp_dir) / f"step{step_idx:02d}"
    summary_path = step_dir / f"step{step_idx:02d}.summary.json"
    if summary_path.is_file() or not step_dir.is_dir():
        return None
    paths = sorted(step_dir.glob(f"step{step_idx:02d}_group*_rollout*.meta.json"))
    expected_primary = int(cfg.groups_per_step) * int(cfg.group_size)
    if len(paths) < expected_primary:
        return None

    artifacts = []
    for meta_path in paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if (int(meta.get("step", -1)) != step_idx or meta.get("status") != "evaluation_pending"):
            return None
        required = (
            "group", "parent_group", "rollout", "n_response_tokens",
            "counts_toward_allocation", "retry_attempt",
        )
        if any(name not in meta for name in required):
            return None
        base = meta_path.name.removesuffix(".meta.json")
        response_path = meta_path.with_name(base + ".txt")
        prompt_path = meta_path.with_name(base + ".prompt.txt")
        token_path = meta_path.with_name(base + ".tokens.zlib")
        behavior_path = meta_path.with_name(base + ".behavior_logprobs.zlib")
        if not all(path.is_file() for path in (
                response_path, prompt_path, token_path, behavior_path)):
            return None
        artifacts.append({
            "meta_path": meta_path,
            "response_path": response_path,
            "prompt_path": prompt_path,
            "token_path": token_path,
            "behavior_logprobs_path": behavior_path,
            "meta": meta,
        })

    primary = [
        item for item in artifacts
        if bool(item["meta"].get("counts_toward_allocation", True))
        and int(item["meta"].get("retry_attempt", 0) or 0) == 0
    ]
    if len(primary) != expected_primary:
        return None
    per_group = {}
    for item in primary:
        group = int(item["meta"]["group"])
        per_group[group] = per_group.get(group, 0) + 1
    if (set(per_group) != set(range(int(cfg.groups_per_step)))
            or any(count != int(cfg.group_size) for count in per_group.values())):
        return None
    artifacts.sort(key=lambda item: (
        int(item["meta"]["group"]), int(item["meta"]["rollout"])))
    return artifacts


def _literal_metadata_mapping(value):
    """Recover mappings stringified by the rollout JSON compatibility layer."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    import ast
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _literal_float_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        import ast
        try:
            values = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    else:
        return None
    try:
        result = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _metadata_is_code_failure(meta):
    """Mirror problems.base.is_code_failure from persisted reward fields."""
    kind = str(meta.get("failure_kind", "") or "").strip().lower()
    if kind:
        return kind == "code"
    if not bool(meta.get("parsed", False)):
        return True
    if bool(meta.get("ran", False)):
        return False
    return "timeout" not in str(meta.get("msg", "") or "").lower()


def _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_count):
    """Build a safe training sequence from a legacy text-only artifact.

    Old runs saved vLLM's decoded text but not ``candidate.token_ids``.  Such
    text is deliberately decoded with ``skip_special_tokens=True`` and BPE
    decode/encode is not generally invertible, so matching the recorded token
    count is neither necessary nor proof that the original IDs were recovered.
    Use the tokenizer's canonical representation of the *exact saved text* and
    restore one hidden terminal EOS when the recorded sequence was longer and
    an unambiguous, invisibly-decoded EOS is available.  Any remaining count
    difference is diagnostic only; padding it with guessed special tokens
    would fabricate a trajectory.

    New runs carry an exact ``.tokens.zlib`` sidecar and never use this path.
    """
    token_ids = list(tokenizer(response_text, add_special_tokens=False).input_ids)
    expected_count = int(expected_count)
    canonical_count = len(token_ids)
    restored_terminal = False

    configured = []

    def add_ids(value):
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        for item in values:
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
            if item >= 0 and item not in configured:
                configured.append(item)

    generation_config = getattr(model, "generation_config", None)
    add_ids(getattr(generation_config, "eos_token_id", None))
    add_ids(getattr(tokenizer, "eos_token_id", None))
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    try:
        tokenizer_eos = int(tokenizer_eos)
    except (TypeError, ValueError):
        tokenizer_eos = None
    chosen = (tokenizer_eos if tokenizer_eos in configured else configured[0] if len(configured) == 1 else None)
    if canonical_count < expected_count and chosen is not None:
        try:
            visible = tokenizer.decode([chosen], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except TypeError:
            visible = tokenizer.decode([chosen], skip_special_tokens=True)
        if not str(visible or ""):
            token_ids.append(chosen)
            restored_terminal = True

    recovered_count = len(token_ids)
    recovery = {
        "source": "canonical-text",
        "exact": False,
        "recorded_count": expected_count,
        "canonical_count": canonical_count,
        "recovered_count": recovered_count,
        "restored_terminal": restored_terminal,
        "unavailable_original_tokens": max(0, expected_count - recovered_count),
        "canonical_extra_tokens": max(0, recovered_count - expected_count),
    }
    return token_ids, recovery


def _prepare_recoverable_response_tokens(model, tokenizer, artifacts):
    """Validate/cache artifact token IDs before distributed replicas load."""
    from experiment_io import (load_rollout_logprobs, load_rollout_token_ids)
    from zlib import error as ZlibError

    summary = {
        "exact_sidecars": 0,
        "behavior_logprob_sidecars": 0,
        "canonical_text": 0,
        "restored_terminal_tokens": 0,
        "unavailable_original_tokens": 0,
        "canonical_extra_tokens": 0,
        "count_changed_responses": 0,
    }
    for item in artifacts:
        expected_count = int(item["meta"]["n_response_tokens"])
        token_path = item.get("token_path")
        if token_path is not None:
            try:
                token_ids = load_rollout_token_ids(token_path)
            except (OSError, ValueError, ZlibError) as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot read exact token sidecar "
                    f"{Path(token_path).name}: {error}") from error
            if len(token_ids) != expected_count:
                raise RuntimeError(
                    "evaluated-step recovery token sidecar length differs "
                    f"for {Path(token_path).name}: saved metadata "
                    f"{expected_count}, sidecar {len(token_ids)}")
            recovery = {
                "source": "exact-sidecar",
                "exact": True,
                "recorded_count": expected_count,
                "canonical_count": None,
                "recovered_count": len(token_ids),
                "restored_terminal": False,
                "unavailable_original_tokens": 0,
                "canonical_extra_tokens": 0,
            }
            summary["exact_sidecars"] += 1
        else:
            response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
            try:
                token_ids, recovery = (
                    _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_count))
            except ValueError as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot tokenize the saved "
                    "response text for "
                    f"{item['response_path'].name}: {error}") from error
            summary["canonical_text"] += 1
        item["response_token_ids"] = token_ids
        item["response_token_recovery"] = recovery
        behavior_path = item.get("behavior_logprobs_path")
        if behavior_path is not None:
            if not bool(recovery.get("exact", False)):
                raise RuntimeError(
                    "evaluated-step recovery found rollout-policy logprobs "
                    "without their exact token-ID sidecar for "
                    f"{Path(behavior_path).name}")
            try:
                behavior_logprobs = load_rollout_logprobs(behavior_path)
            except (OSError, ValueError, ZlibError) as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot read rollout-policy "
                    f"logprobs from {Path(behavior_path).name}: {error}") from error
            if len(behavior_logprobs) != len(token_ids):
                raise RuntimeError(
                    "evaluated-step recovery rollout-policy logprob length "
                    f"differs for {Path(behavior_path).name}: tokens "
                    f"{len(token_ids)}, logprobs {len(behavior_logprobs)}")
            item["behavior_logprobs"] = behavior_logprobs
            summary["behavior_logprob_sidecars"] += 1
        else:
            item["behavior_logprobs"] = None
        restored_terminal = bool(recovery["restored_terminal"])
        item["restored_terminal_token"] = restored_terminal
        summary["restored_terminal_tokens"] += int(restored_terminal)
        summary["unavailable_original_tokens"] += int(recovery["unavailable_original_tokens"])
        summary["canonical_extra_tokens"] += int(recovery["canonical_extra_tokens"])
        summary["count_changed_responses"] += int(len(token_ids) != expected_count)
    return summary


def _finalize_pending_spo_rs_artifacts(
        tokenizer, sampler, step_idx, cfg, exp_dir, problem, gen_pool,
        spo_rs_tracker, sampling_adapter_path, artifacts):
    """Evaluate and score a generated interrupted step, then seal its metadata.

    No policy sampling occurs here.  The exact generated token IDs and their
    rollout-policy logprobs come from the durable sidecars.  CPU rewards are
    recomputed while vLLM scores those same trajectories under the preceding
    policy, after which the ordinary evaluated-step recovery performs the
    unchanged fast distributed adapter update.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from copy import deepcopy
    from queue import Queue

    from experiment_io import save_rollout
    from generation.gen_workers import make_progress_bar, native_final_answer_scope
    from problems.base import ParentContext, RewardResult
    from search.sampler import State

    if cfg.generation_backend != "vllm":
        raise RuntimeError('generated-step SPO-RS recovery requires exact vLLM rollout artifacts')
    if gen_pool is None:
        raise RuntimeError(
            "generated-step SPO-RS recovery requires the "
            "configured vLLM pool for exact consecutive-policy scoring")
    if (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError('generated-step SPO-RS recovery requires the restored pre-step tracker')
    spo_rs_tracker.prepare_step()
    if sampling_adapter_path is None or not Path(sampling_adapter_path).is_dir():
        raise FileNotFoundError(
            "generated-step SPO-RS recovery cannot find its sampling adapter: "
            f"{sampling_adapter_path}")

    token_summary = _prepare_recoverable_response_tokens(None, tokenizer, artifacts)
    if token_summary["exact_sidecars"] != len(artifacts):
        raise RuntimeError("generated-step recovery refuses non-exact response token IDs")
    if token_summary["behavior_logprob_sidecars"] != len(artifacts):
        raise RuntimeError('generated-step recovery requires every rollout-policy logprob sidecar')

    parents = sampler.sample_states(int(cfg.groups_per_step), selection_step=step_idx)
    if len(parents) != int(cfg.groups_per_step):
        raise RuntimeError("generated-step recovery could not reconstruct every parent")
    parents_path = (Path(exp_dir) / f"step{int(step_idx):02d}" / f"step{int(step_idx):02d}.parents.json")
    try:
        saved_parent_rows = json.loads(parents_path.read_text(encoding="utf-8"))["parents"]
    except (OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        raise RuntimeError("generated-step recovery cannot read its saved parent selection") \
            from error
    saved_parents = {int(row["group"]): row for row in saved_parent_rows}
    for group, parent in enumerate(parents):
        row = saved_parents.get(group)
        if row is None:
            raise RuntimeError(f"generated-step recovery is missing parent group {group}")
        if (str(row.get("parent_id")) != str(parent.id)
                or int(row.get("parent_timestep", -1))
                != int(parent.timestep)
                or not math.isclose(float(row.get("parent_reward")), float(parent.value), rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(
                f"generated-step recovery parent {group} does not match the "
                "restored sampler checkpoint")

    parent_contexts = [
        ParentContext(
            code=parent.code,
            value=(parent.value if parent.value is not None else 0.0),
            raw_score=parent.raw_score,
            construction=parent.construction,
        )
        for parent in parents
    ]

    isolated_eval = bool(getattr(cfg, "isolate_eval", False))
    isolated_slots = None
    isolated_memory_limit = None
    if isolated_eval:
        cpu_ids = _isolated_evaluation_cpu_ids()
        processes_per_cpu = int(cfg.reward_workers)
        isolated_slots = Queue()
        for _layer in range(processes_per_cpu):
            for cpu_id in cpu_ids:
                isolated_slots.put(cpu_id)
        workers = max(1, min(len(artifacts), len(cpu_ids) * processes_per_cpu))
        isolated_memory_limit = _automatic_sandbox_memory_limit_bytes(workers)
    else:
        workers = max(1, min(len(artifacts), _resolve_reward_workers(cfg, problem)))

    from runtime.host_memory import HostMemoryAdmission
    import multiprocessing
    host_admission = (getattr(gen_pool, "host_memory_admission", None)
                      or HostMemoryAdmission(multiprocessing.get_context("spawn")))

    def evaluate(item):
        meta = item["meta"]
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        final_text = native_final_answer_scope(
            tokenizer, item["response_token_ids"],
            require_final_marker=bool(getattr(problem, "require_final_code_marker", False)),
            prompt_text=item["prompt_path"].read_text(encoding="utf-8", errors="replace"))
        issue = coder_output_issue(
            response_text, require_final_marker=bool(getattr(problem, "require_final_code_marker", False)),
            final_answer_text=final_text)
        meta["output_format_issue"] = issue
        if issue:
            return RewardResult(reward=float(problem.fail_score), msg=str(issue), failure_kind="code")
        answer_options = {} if final_text is None else {"final_answer_text": final_text}
        parent_context = parent_contexts[int(meta["parent_group"])]
        if not isolated_eval:
            return problem.compute_reward(response_text, parent_context, cfg.sandbox_timeout_s, **answer_options)
        with host_admission.reservation(isolated_memory_limit):
            cpu_id = isolated_slots.get()
            try:
                return problem.compute_reward(
                    response_text, parent_context, cfg.sandbox_timeout_s,
                    cpu_id=cpu_id,
                    memory_limit_bytes=isolated_memory_limit, **answer_options)
            finally:
                isolated_slots.put(cpu_id)

    evaluation_pool = ThreadPoolExecutor(max_workers=workers)
    evaluation_futures = {evaluation_pool.submit(evaluate, item): index for index, item in enumerate(artifacts)}
    results = [None] * len(artifacts)
    try:
        prompt_cache = {}
        score_pairs = []
        current_scores = []
        context_ids = []
        for item in artifacts:
            prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
            if prompt_text not in prompt_cache:
                prompt_cache[prompt_text] = tokenizer(prompt_text).input_ids
            score_pairs.append((
                prompt_cache[prompt_text], item["response_token_ids"]))
            current_scores.append(item["behavior_logprobs"])
            context_ids.append(int(item["meta"]["parent_group"]))

        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is None:
            raise RuntimeError("generated-step recovery has no preceding SPO-RS adapter")
        previous_adapter_path = Path(exp_dir) / previous_name
        if not previous_adapter_path.is_dir():
            raise FileNotFoundError(
                "generated-step recovery cannot find the preceding policy "
                f"adapter: {previous_adapter_path}")
        if (previous_adapter_path.resolve() == Path(sampling_adapter_path).resolve()):
            previous_scores = current_scores
        else:
            try:
                previous_scores = gen_pool.score_token_logprobs(
                    score_pairs, show_progress=True,
                    adapter_path=str(previous_adapter_path),
                    require_complete=True)
            finally:
                if getattr(gen_pool, "sequential", False):
                    gen_pool.release()

        divergence, missing_scores, context_divergences = (
            spo_rs_tracker.consecutive_policy_divergence(
                current_scores, previous_scores, context_ids,
                required_context_ids=range(int(cfg.groups_per_step))))
        if missing_scores or not math.isfinite(divergence):
            raise RuntimeError('generated-step recovery did not obtain every exact consecutive-policy score')

        eval_bar = make_progress_bar(len(evaluation_futures), desc="recovering evaluations")
        try:
            for future in as_completed(evaluation_futures):
                results[evaluation_futures[future]] = future.result()
                eval_bar.update(1)
        finally:
            eval_bar.close()
    finally:
        evaluation_pool.shutdown(wait=True, cancel_futures=False)

    rewards = [float(result.reward) for result in results]
    prompt_indices = {}
    for item in artifacts:
        prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
        if prompt_text not in prompt_indices:
            prompt_indices[prompt_text] = len(prompt_indices)
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    advantages = spo_rs_tracker.normalized_advantages(transformed)
    if advantages is None:
        raise RuntimeError("generated-step recovery lost the pre-step SPO-RS baseline")
    preview_tracker = deepcopy(spo_rs_tracker)
    update = preview_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx,
        raw_rewards=rewards)
    update.update({
        "groups": sorted({int(item["meta"]["group"]) for item in artifacts}),
        "prompt_jobs": list(range(len(prompt_indices))),
        "contexts": sorted(set(context_ids)),
        "context_divergences": context_divergences,
        "missing_policy_scores": int(missing_scores),
    })
    rollout_infos = [{
        **update,
        "transformed_reward": float(transformed[index]),
        "used_for_policy_update": True,
    } for index in range(len(artifacts))]
    beta_by_index = np.full(len(artifacts), float(cfg.spo_rs_beta), dtype=np.float64)

    save_ctor = bool(getattr(problem, "saves_construction", False))
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False) else type(sampler).__name__)
    step_dir = Path(exp_dir) / f"step{int(step_idx):02d}"
    for index, (item, result) in enumerate(zip(artifacts, results)):
        pending = item["meta"]
        group = int(pending["group"])
        parent_group = int(pending["parent_group"])
        rollout = int(pending["rollout"])
        parent = parents[parent_group]
        parent_row = saved_parents[parent_group]
        child = State.make(
            timestep=step_idx, value=float(result.reward),
            code=result.code or "", raw_score=result.raw_score,
            construction=result.construction)
        archive_eligible = bool(result.valid and result.code)
        strategy_source = pending.get("strategy_source_job_idx")
        strategy_index = (
            int(strategy_source) % int(cfg.strategies_per_parent)
            if bool(cfg.strategies) and strategy_source is not None else None)
        base = item["meta_path"].name.removesuffix(".meta.json")
        strategy_path = step_dir / f"{base}.strategy.txt"
        strategy = None
        if strategy_path.is_file():
            raw_strategy = strategy_path.read_text(encoding="utf-8", errors="replace")
            strategy, _issue = _extract_final_strategy(
                raw_strategy,
                strict=bool(getattr(cfg, "strict_strategy_validation", False)),
            )
        info = rollout_infos[index]
        meta = {key: value for key, value in pending.items() if key != "status"}
        meta.update({
            "node_id": child.id,
            "parent_id": parent.id,
            "parent_timestep": int(parent.timestep),
            "sampler_type": sampler_type,
            "reward": float(result.reward),
            "raw_score": (float(result.raw_score) if result.raw_score is not None else None),
            "valid": bool(result.valid),
            "parsed": bool(result.parsed),
            "ran": bool(result.ran),
            "msg": result.msg,
            "failure_kind": result.failure_kind,
            "advantage": float(advantages[index]),
            "beta": float(beta_by_index[index]),
            "advantage_mode": "spo-rs",
            "advantage_scale": float(beta_by_index[index]),
            "sandbox_stdout": (result.stdout or "")[:2000],
            "parent_value": (float(parent.value) if parent.value is not None else None),
            "parent_raw_score": (float(parent.raw_score) if parent.raw_score is not None else None),
            "parent_is_seed": parent.id in sampler._seed_ids,
            "parent_visit_count": int(parent_row.get("visit_count", 0)),
            "parent_q_value": parent_row.get("q_value"),
            "parent_prior": parent_row.get("prior"),
            "parent_exploration_bonus": parent_row.get("exploration_bonus"),
            "parent_selection_score": parent_row.get("selection_score"),
            "archive_eligible": archive_eligible,
            "two_stage_rollout": bool(cfg.strategies),
            "strategy": strategy,
            "strategy_index": strategy_index,
            "strategy_model_name": (cfg.strategy_model_name if bool(cfg.strategies) else None),
            "programs_per_strategy": (int(cfg.programs_per_strategy) if bool(cfg.strategies) else None),
            "construction": (_as_float_list(result.construction, cfg.max_saved_construction) if save_ctor else None),
            "parent_construction": (
                _as_float_list(parent.construction, cfg.max_saved_construction)
                if save_ctor else None),
            "seed": int(cfg.seed),
            "spo_rs": info,
        })
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        save_rollout(exp_dir, step_idx, group, rollout, response_text, meta, artifacts_already_saved=True)

    print(
        f"[resume] step {step_idx}: reused {len(artifacts)} exact generated "
        "rollouts, recomputed their CPU rewards, and completed exact "
        "consecutive-policy scoring; no policy sampling was repeated",
        flush=True,
    )
    completed = _recoverable_spo_rs_artifacts(exp_dir, step_idx, cfg)
    if completed is None:
        raise RuntimeError("generated-step recovery failed to seal complete rollout metadata")
    _prepare_recoverable_response_tokens(None, tokenizer, completed)
    return completed


def _resume_spo_rs_step_from_artifacts(
        backend, model, tokenizer, sampler, optimizer, step_idx, cfg, exp_dir,
        problem, parallel_trainer, spo_rs_tracker, sampling_adapter_path,
        artifacts):
    """Finish one interrupted SPO-RS step from its durable evaluations."""
    import torch
    from experiment_io import save_rollout
    from evaluation.reward import extract_python_code
    from generation.gen_workers import native_final_answer_scope
    from search.sampler import State

    adaptive_spo_beta = bool(getattr(cfg, "spo_rs_adaptive_beta", False))

    if (parallel_trainer is None and str(getattr(cfg, "training_layout", "")) != "sharded"):
        raise RuntimeError(
            "evaluated-step recovery requires either the configured fast "
            "distributed trainer or the automatic exact sharded trainer")
    if (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError("evaluated-step recovery requires the pre-step SPO-RS tracker")
    spo_rs_tracker.prepare_step()
    if sampling_adapter_path is None or not Path(sampling_adapter_path).is_dir():
        raise FileNotFoundError(f'evaluated-step recovery cannot find the sampling adapter: {sampling_adapter_path}')

    parents = sampler.sample_states(int(cfg.groups_per_step), selection_step=step_idx)
    if len(parents) != int(cfg.groups_per_step):
        raise RuntimeError(
            "evaluated-step recovery could not reconstruct every selected "
            f"parent ({len(parents)}/{int(cfg.groups_per_step)})")
    by_group = {}
    for item in artifacts:
        by_group.setdefault(int(item["meta"]["group"]), []).append(item)
    for group, parent in enumerate(parents):
        records = by_group.get(group, ())
        if not records:
            raise RuntimeError(f"evaluated-step recovery has no rollout group {group}")
        first = records[0]["meta"]
        try:
            saved_value = float(first["parent_value"])
            saved_timestep = int(first["parent_timestep"])
            saved_parent_id = str(first["parent_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f'evaluated-step recovery lacks parent identity for group {group}') from error
        current_value = float(parent.value)
        if (saved_timestep != int(parent.timestep)
                or not math.isclose(saved_value, current_value, rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(f'evaluated-step recovery parent {group} does not match the restored sampler state')
        if parent.id != saved_parent_id:
            if step_idx == 0 and parent.id in sampler._seed_ids:
                # Fresh seed State UUIDs are intentionally random. Reuse the
                # durable IDs already embedded in every rollout so the healed
                # checkpoint preserves the original lineage exactly.
                sampler.reassign_seed_id(parent, saved_parent_id)
            else:
                raise RuntimeError(
                    f"evaluated-step recovery parent ID {group} differs from "
                    "the restored sampler checkpoint")
        sampler.record_expansion(parent, count=int(cfg.group_size))

    rewards = [float(item["meta"]["reward"]) for item in artifacts]
    saved_advantages = np.asarray([float(item["meta"]["advantage"]) for item in artifacts], dtype=np.float64)
    spo_infos = [_literal_metadata_mapping(item["meta"].get("spo_rs")) for item in artifacts]
    if any(not info or not bool(info.get("used_for_policy_update", False)) for info in spo_infos):
        raise RuntimeError(
            "evaluated-step recovery found a rollout that was not part of "
            "the saved SPO-RS policy update")
    recovery_updates = []
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    recovered_advantages = spo_rs_tracker.normalized_advantages(transformed)
    spo_info = spo_infos[0]
    saved_value_before = spo_info.get("value_before")
    if (saved_value_before is None
            or not math.isclose(float(saved_value_before),
                                float(spo_rs_tracker.baseline()),
                                rel_tol=0.0, abs_tol=1e-10)):
        raise RuntimeError('evaluated-step recovery does not match the saved pre-step SPO-RS value')
    if int(spo_info.get("group_size", -1)) != len(artifacts):
        raise RuntimeError('evaluated-step recovery rollout count differs from the saved SPO-RS update')
    if Path(str(spo_info.get("policy_adapter", ""))).name != Path(sampling_adapter_path).name:
        raise RuntimeError('evaluated-step recovery sampling adapter differs from the saved SPO-RS update')
    for index, (info, transformed_reward) in enumerate(zip(spo_infos, transformed)):
        saved_transformed = info.get("transformed_reward")
        if (saved_transformed is None
                or not math.isclose(float(saved_transformed), float(transformed_reward), rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError(
                "evaluated-step recovery transformed reward differs from "
                f"the saved SPO-RS record at rollout {index}")
    divergence = spo_info.get("divergence")
    if divergence is None:
        if bool(spo_info.get("divergence_missing", False)):
            divergence = math.inf
        else:
            raise RuntimeError("evaluated-step recovery has no saved SPO-RS divergence")
    divergence = float(divergence)
    if math.isnan(divergence) or divergence < 0.0:
        raise RuntimeError("evaluated-step recovery found an invalid SPO-RS divergence")
    tracker_update = spo_rs_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx,
        raw_rewards=rewards)
    recovery_updates = [tracker_update]
    saved_after = spo_info.get("value_after")
    if (saved_after is not None
            and not math.isclose(float(saved_after), float(tracker_update["value_after"]), rel_tol=0.0, abs_tol=1e-10)):
        raise RuntimeError(
            "evaluated-step recovery produced a different SPO-RS tracker "
            "value; refusing an inexact update")

    if (recovered_advantages is None
            or recovered_advantages.shape != saved_advantages.shape
            or not np.allclose(recovered_advantages, saved_advantages, rtol=0.0, atol=1e-10)):
        raise RuntimeError(
            "evaluated-step recovery advantages differ from the saved SPO "
            "objective; refusing an inexact update")

    prompt_cache = {}
    prompt_job_ids = {}
    all_examples = []
    all_children = []
    strategy_keys = []
    entropy_directory = None
    entropy_observations = []
    if bool(getattr(cfg, "measure_entropy", False)):
        from analysis.entropy_tools import (begin_step, observation_descriptor, rollout_observation)
        entropy_directory = begin_step(exp_dir, step_idx)
    group_sizes = {group: len(records) for group, records in by_group.items()}
    restored_terminal_tokens = 0
    canonical_recovery_responses = 0
    unavailable_original_tokens = 0
    canonical_extra_tokens = 0
    for item in artifacts:
        meta = item["meta"]
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
        expected_tokens = int(meta["n_response_tokens"])
        response_tokens = item.get("response_token_ids")
        if response_tokens is None:
            try:
                response_tokens, recovery = (
                    _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_tokens))
            except ValueError as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot tokenize the saved "
                    "response text for "
                    f"{item['response_path'].name}: {error}") from error
        else:
            response_tokens = list(response_tokens)
            recovery = item.get("response_token_recovery") or {
                "source": "exact-sidecar",
                "exact": True,
                "recorded_count": expected_tokens,
                "canonical_count": None,
                "recovered_count": len(response_tokens),
                "restored_terminal": bool(item.get("restored_terminal_token")),
                "unavailable_original_tokens": 0,
                "canonical_extra_tokens": 0,
            }
            if (bool(recovery.get("exact", False)) and len(response_tokens) != expected_tokens):
                raise RuntimeError(
                    "evaluated-step recovery exact token cache changed for "
                    f"{item['response_path'].name}")
        restored_terminal = bool(recovery.get("restored_terminal", False))
        restored_terminal_tokens += int(restored_terminal)
        canonical_recovery_responses += int(recovery.get("source") == "canonical-text")
        unavailable_original_tokens += int(recovery.get('unavailable_original_tokens', 0) or 0)
        canonical_extra_tokens += int(recovery.get('canonical_extra_tokens', 0) or 0)
        if prompt_text not in prompt_cache:
            prompt_cache[prompt_text] = tokenizer(prompt_text, return_tensors="pt").input_ids
            prompt_job_ids[prompt_text] = len(prompt_job_ids)
        group = int(meta["group"])
        entropy_descriptor = None
        if entropy_directory is not None:
            entropy_descriptor = observation_descriptor(entropy_directory, group, int(meta["rollout"]))
        if response_tokens:
            behavior_values = item.get("behavior_logprobs")
            behavior_logprobs = (
                torch.tensor(behavior_values, dtype=torch.float32)
                if behavior_values is not None
                and len(behavior_values) == len(response_tokens)
                else None)
            all_examples.append({
                "prompt_ids": prompt_cache[prompt_text],
                "response_ids": torch.tensor([response_tokens], dtype=torch.long),
                "_entropy_measurement": entropy_descriptor,
                "advantage": float(meta["advantage"]),
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": None,
                "reward_constant": False,
                "group_id": group,
                "rollout_index": int(meta["rollout"]),
                "prompt_job_id": prompt_job_ids[prompt_text],
                "sample_weight": 0.0,
                "_overlap_key": (group, int(meta["rollout"])),
            })

        # Re-encoded legacy text cannot prove where a native token occurred.
        final_text = (native_final_answer_scope(
            tokenizer, response_tokens,
            require_final_marker=bool(getattr(problem, "require_final_code_marker", False)),
            prompt_text=prompt_text)
            if bool(recovery.get("exact", False)) else None)
        code = extract_python_code(response_text, final_answer_text=final_text)
        if entropy_descriptor is not None:
            entropy_meta = dict(meta)
            entropy_meta["n_response_tokens"] = len(response_tokens)
            entropy_observations.append(rollout_observation(entropy_descriptor, entropy_meta, code))

        if bool(meta.get("archive_eligible", False)):
            if not code:
                raise RuntimeError(
                    "evaluated-step recovery lost code from an archive-eligible "
                    f"rollout: {item['response_path'].name}")
            child = State.make(
                timestep=step_idx, value=float(meta["reward"]), code=code,
                raw_score=(float(meta["raw_score"]) if meta.get("raw_score") is not None else None),
                construction=_literal_float_list(meta.get("construction")),
            )
            if meta.get("node_id"):
                child.id = str(meta["node_id"])
            parent_group = int(meta.get("parent_group", group))
            all_children.append((child, parents[parent_group]))
            strategy_keys.append(int(meta["strategy_index"]) if meta.get("strategy_index") is not None else None)

    if not all_examples:
        raise RuntimeError("evaluated-step recovery found no nonempty responses to train")
    trajectory_weight = 1.0 / len(all_examples)
    for example in all_examples:
        example["sample_weight"] = trajectory_weight

    overflow_state = _load_exact_spo_rs_overflow(exp_dir, step_idx)
    recovery_examples = all_examples
    if overflow_state is not None:
        try:
            _validate_precomputed_exact_gradients(model, overflow_state, step_idx)
            deferred_keys = {(int(key[0]), int(key[1])) for key in overflow_state.get("deferred_keys", ())}
            remaining_keys = {
                (int(key[0]), int(key[1]))
                for key in overflow_state.get("remaining_keys", overflow_state.get("deferred_keys", ()))
            }
            marker_keys = {
                (int(key[0]), int(key[1]))
                for key in (_read_exact_sharded_recovery(exp_dir) or {}).get("deferred_keys", ())
            }
            if not deferred_keys or deferred_keys != marker_keys:
                raise RuntimeError('trajectory keys disagree between the marker and gradient checkpoint')
            if not remaining_keys.issubset(deferred_keys):
                raise RuntimeError(
                    "remaining overflow trajectories are not a subset of "
                    "the original deferred trajectories")
            example_by_key = {}
            for example in all_examples:
                key = tuple(example["_overlap_key"])
                if key in example_by_key:
                    raise RuntimeError(f"durable trajectory key {key} is duplicated")
                example_by_key[key] = example
            missing_keys = deferred_keys - set(example_by_key)
            if missing_keys:
                raise RuntimeError(f'deferred trajectories are absent from durable rollouts: {sorted(missing_keys)}')
            normal_count = int(overflow_state.get("normal_stats", {}).get("trained_examples", -1))
            if normal_count + len(deferred_keys) != len(all_examples):
                raise RuntimeError(
                    "trajectory accounting changed: retained="
                    f"{normal_count}, deferred={len(deferred_keys)}, "
                    f"recovered={len(all_examples)}")
            recovery_examples = [example_by_key[key] for key in sorted(remaining_keys)]
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            print(
                f"[warn] selective exact SPO-RS overflow state is not "
                f"reusable ({error}); falling back to the complete exact "
                "model-sharded update",
                flush=True,
            )
            overflow_state = None
            recovery_examples = all_examples

    archive_stats = sampler.update(
        all_children,
        strategy_keys=(strategy_keys if bool(cfg.strategies) else None),
        strategy_top_r=(int(cfg.strategy_archive_top_r) if bool(cfg.strategies) else 0),
    )
    print(
        f"[resume] step {step_idx}: reusing {len(artifacts)} saved rollout "
        "evaluations; generation and CPU evaluation are skipped",
        flush=True,
    )
    if restored_terminal_tokens:
        print(
            f"[resume] step {step_idx}: restored the omitted terminal EOS in "
            f"{restored_terminal_tokens}/{len(artifacts)} saved responses",
            flush=True,
        )
    if canonical_recovery_responses:
        print(
            f"[resume] step {step_idx}: {canonical_recovery_responses} "
            "legacy text-only response(s) use canonical tokenizer IDs; "
            f"{unavailable_original_tokens} hidden/noncanonical original "
            f"token(s) were unavailable and {canonical_extra_tokens} "
            "canonical token(s) exceeded recorded counts",
            flush=True,
        )
    print(
        f"[resume] step {step_idx}: restored {archive_stats['submitted']} "
        "valid children and the exact saved SPO-RS advantages; resuming at "
        "adapter training",
        flush=True,
    )
    if overflow_state is not None:
        from training.exact_overflow import supports_warm_overflow
        recovery_lane = (
            ("grouped-checkpoint GPU replicas, then cooperative groups if needed"
             if (not overflow_state.get("replica_checkpoint_groups_attempted", False)
                 and any(hasattr(module, "_ttt_group_checkpoint_min_tokens") for module in model.modules())) else
             "warm cooperative GPU groups (no process restart)"
             if supports_warm_overflow(cfg) else
             f"the {parallel_trainer.world_size}-GPU RAM-bounded replica lane")
            if parallel_trainer is not None
            and hasattr(parallel_trainer, "recover_spo_rs_overflow")
            else "the model-sharded overflow lane"
        )
        print(
            f"[recovery] step {step_idx}: reusing exact fast-path gradients "
            f"for {len(all_examples) - len(recovery_examples)}/"
            f"{len(all_examples)} trajectories; only "
            f"{len(recovery_examples)} oversized trajectory/trajectories "
            f"will run through {recovery_lane}",
            flush=True,
        )

    raw_states = [
        child for child, _parent in all_children
        if child.raw_score is not None and math.isfinite(float(child.raw_score))
    ]
    step_best = (
        (max(raw_states, key=lambda state: float(state.raw_score))
         if problem.maximize else
         min(raw_states, key=lambda state: float(state.raw_score)))
        if raw_states else None)
    best_seen = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
    stats = {
        "planned_rollouts": sum(bool(item["meta"].get("counts_toward_allocation", True)) for item in artifacts),
        "coder_format_retries": sum(int(item['meta'].get('retry_attempt', 0) or 0) > 0 for item in artifacts),
        "coder_retry_enabled": bool(getattr(cfg, "coder_retry", False)),
        "result_metric_name": str(problem.metric_name),
        "result_maximize": bool(problem.maximize),
        "step_best_raw_score": (float(step_best.raw_score) if step_best is not None else None),
        "step_best_reward": (float(step_best.value) if step_best is not None else None),
        "best_seen_raw_score": (
            float(best_seen.raw_score)
            if best_seen is not None and best_seen.raw_score is not None
            else None),
        "best_seen_reward": (float(best_seen.value) if best_seen is not None and best_seen.value is not None else None),
        "best_seen_step": (int(best_seen.timestep) if best_seen is not None else None),
        "valid_fraction": (sum(bool(item["meta"]["valid"]) for item in artifacts) / len(artifacts)),
        "code_valid_fraction": (
            1.0 - sum(_metadata_is_code_failure(item["meta"]) for item in artifacts) / len(artifacts)),
        "spo_rs_updates": recovery_updates,
        "spo_rs_adaptive_beta": adaptive_spo_beta,
        "spo_rs_tracker_size": (0 if spo_rs_tracker is None else len(spo_rs_tracker)),
        "spo_rs_divergence": (None if not math.isfinite(divergence) else divergence),
        "spo_rs_context_divergences": spo_info.get("context_divergences", {}),
        "spo_rs_missing_policy_scores": int(spo_info.get('missing_policy_scores', 0) or 0),
        "spo_rs_sampling_adapter": Path(sampling_adapter_path).name,
        "spo_rs_policy_updated": False,
        "recovered_evaluated_step": True,
        "recovered_rollouts": len(artifacts),
        "_entropy_observations": entropy_observations,
    }
    if (overflow_state is not None
            and parallel_trainer is not None
            and hasattr(parallel_trainer, "recover_spo_rs_overflow")):
        stats.update(parallel_trainer.recover_spo_rs_overflow(recovery_examples, cfg, step_idx, overflow_state))
    elif parallel_trainer is not None:
        stats.update(parallel_trainer.train_spo_rs(all_examples, cfg, step_idx))
    else:
        try:
            input_device = next(model.get_input_embeddings().parameters()).device
        except (AttributeError, StopIteration):
            input_device = model.device
        for example in recovery_examples:
            example["prompt_ids"] = example["prompt_ids"].to(input_device)
            example["response_ids"] = example["response_ids"].to(input_device)
        print(
            f"[recovery] step {step_idx}: applying the unchanged SPO-RS "
            "update with one exact model sharded across "
            f"{int(cfg.num_training_gpus)} GPUs",
            flush=True,
        )
        stats.update(_train_spo_rs_examples(
            backend, model, tokenizer, optimizer, recovery_examples,
            cfg, step_idx,
            precomputed_gradient_state=overflow_state,
            exact_overflow_run_dir=exp_dir))
    stats["spo_rs_policy_updated"] = True
    return stats


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

    This sets a per-program ceiling, not a promise that the whole evaluator
    pool fits beside resident models. HostMemoryAdmission separately reserves
    live RAM at each launch and shares that budget with transfers/activations.
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

    from search.sampler import State
    from experiment_io import (save_parent_selections, save_rollout,
                               save_rollout_logprobs,
                               save_rollout_artifacts,
                               save_strategy_response)
    from problems.base import ParentContext, RewardResult, is_code_failure
    from generation.gen_workers import make_progress_bar, native_final_answer_scope


    step_t0 = time.time()
    entropy_directory = None
    entropy_observations = []
    if getattr(cfg, "measure_entropy", False):
        from analysis.entropy_tools import (begin_step, observation_descriptor, rollout_observation)
        entropy_directory = begin_step(exp_dir, step_idx)
    active_advantage_mode = str(getattr(cfg, "advantage_mode", "entropic")).lower()
    spo_rs_mode = active_advantage_mode == "spo-rs"
    adaptive_spo_beta = bool(spo_rs_mode and getattr(cfg, "spo_rs_adaptive_beta", False))
    if spo_rs_mode and spo_rs_tracker is None:
        raise ValueError("SPO-RS mode requires its persistent value tracker")
    if (spo_rs_mode and (not spo_rs_tracker.initialized)):
        raise ValueError('SPO-RS tracker must be initialized from pre-step history before rollouts are sampled')
    if spo_rs_mode:
        spo_rs_tracker.prepare_step()
    clipped_policy_mode = _uses_clipped_policy_loss(active_advantage_mode)
    requested_parent_contexts = (int(cfg.groups_per_step))
    parents = sampler.sample_states(requested_parent_contexts, selection_step=step_idx)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        if info.get("selection_mode") == "llm":
            detail = info["llm"]
            print(f"  parent {i} [{tag}] raw={parents[i].raw_score!r} "
                  f"Luna rank={detail['rank']} ({detail['role']}): "
                  f"{detail['reason']}")
            continue
        prior_label = ("" if info.get("selection_mode") == "uct" else f"  P={info['P']:.9f}")
        print(f"  parent {i} [{tag}]  value={info['value']:.9f}  n={info['n']}  "
              f"Q={info['Q']:.9f}{prior_label}  "
              f"bonus={info['bonus']:.9f}  score={info['score']:.9f}")

    # Save the selection event immediately. Unlike the sampler checkpoint,
    # this survives later archive pruning and also exists if generation or
    # adapter training is interrupted.
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False) else type(sampler).__name__)
    save_parent_selections(exp_dir, step_idx, sampler_type, parents, sampler.last_picks_info, rollout_config=cfg)
    setting_log_only(
        f"[step {step_idx}] saved {len(parents)} selected parent(s) before "
        f"generation/training", flush=True)

    # Only problems that declare it get their construction written to disk.
    save_ctor = (bool(getattr(problem, "saves_construction", False))
                 and int(getattr(cfg, "max_saved_construction", 0)) != 0)

    training_enabled = not bool(getattr(cfg, "no_train", False))

    all_examples = []
    spo_rs_updates = []
    entropic_overlap_state = None
    spo_rs_overlap_state = None
    all_children = []
    all_child_strategy_keys = []
    strategy_pilot_diagnostics = []
    saved_rollouts = 0
    # ----- BUILD PROMPTS (one per parent/group) -----
    parent_ctxs = []
    base_messages = []

    for g, parent in enumerate(parents):
        sampler.record_expansion(parent, count=int(cfg.group_size))
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
            adapter_path = _save_adapter(model, exp_dir, step_idx, cfg.model_name, directory_name="adapter_initial")
        else:
            adapter_path = str(Path(sampling_adapter_path))
            if not Path(adapter_path).is_dir():
                raise FileNotFoundError(f"SPO-RS sampling adapter not found: {adapter_path}")
    elif gen_pool is not None:
        adapter_path = _save_adapter(model, exp_dir, step_idx, cfg.model_name)

    def _render(messages, *, rollout_phase=None, reasoning_effort=None):
        template_kind = str(getattr(cfg, "coder_template_kind", "generic"))
        template_options = {}
        if template_kind == "gpt-oss":
            template_options["reasoning_effort"] = str(cfg.coder_reasoning_effort)
        elif template_kind == "qwen3.8":
            template_options.update({
                "enable_thinking": True,
                "reasoning_effort": str(
                    _coder_effort_for_rollout_phase(cfg, rollout_phase)
                    if reasoning_effort is None else reasoning_effort),
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
                messages, tokenize=False, add_generation_prompt=True,
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
                        messages, tokenize=False,
                        add_generation_prompt=True, **reduced)
                except TypeError:
                    pass
            if template_kind in {"gpt-oss", "qwen3.8"}:
                raise RuntimeError(
                    f"the installed tokenizer/chat template cannot apply "
                    f"the required {template_kind} reasoning controls "
                    f"{template_options}; update transformers and refresh "
                    "the model tokenizer files")
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    strategy_tokenizer = strategy_tokenizer or tokenizer

    def _render_strategy(messages, *, reasoning_effort=None):
        if getattr(cfg, "strategy_backend", "local") == "api":
            # The remote client applies the provider's chat template. Passing
            # structured messages here avoids loading any strategist tokenizer
            # or weights in this process.
            return [dict(message) for message in messages]
        strategy_name = str(getattr(cfg, "strategy_model_name", "")).lower()
        if strategist_profile(strategy_name):
            return render_strategy_prompt(
                strategy_tokenizer, messages, cfg.strategy_model_name,
                thinking=cfg.strategy_thinking,
                effort=(cfg.strategy_reasoning_effort if reasoning_effort is None else reasoning_effort))
        template_options = {}
        if "gpt-oss" in strategy_name:
            template_options["reasoning_effort"] = str(
                cfg.strategy_reasoning_effort
                if reasoning_effort is None else reasoning_effort)
        else:
            template_options["enable_thinking"] = bool(cfg.strategy_thinking)
        try:
            return strategy_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_options,
            )
        except TypeError:
            return strategy_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_jobs = []
    phase_prompt_job_cache = {}
    for g, _parent in enumerate(parents):
        messages = base_messages[g]
        count = int(cfg.group_size)
        prompt_jobs.append({
            "parent_group": g,
            "messages": messages,
            "prompt_text": _render(messages),
            "coder_reasoning_effort": _coder_effort_for_rollout_phase(cfg),
            "count": count,
        })

    two_stage_rollouts = bool(getattr(problem, "two_stage_rollouts", False))
    strategies_per_parent = (int(cfg.strategies_per_parent) if two_stage_rollouts else 0)
    programs_per_strategy = (int(cfg.programs_per_strategy) if two_stage_rollouts else 0)
    pilot_programs_per_strategy = (int(getattr(cfg, "pilot_programs_per_strategy", -1)) if two_stage_rollouts else -1)
    phase2_allocation_method = str(getattr(cfg, "phase2_allocation_method", "rule_based")).strip().lower()
    if phase2_allocation_method not in {"rule_based", "bandit", "hurdle"}:
        raise RuntimeError("phase2_allocation_method must be 'rule_based', 'bandit', or 'hurdle'")
    adaptive_strategy_pilots = bool(two_stage_rollouts and pilot_programs_per_strategy > 0)
    if (two_stage_rollouts and int(cfg.group_size) != strategies_per_parent * programs_per_strategy):
        raise RuntimeError(
            "hierarchical rollout group size must equal "
            "strategies_per_parent * programs_per_strategy")

    def _source_schedule(parent_group):
        indices = [index for index, job in enumerate(prompt_jobs) if int(job["parent_group"]) == int(parent_group)]
        if not indices:
            raise RuntimeError(f"parent {parent_group} has no rendered prompt job")
        weights = [max(0, int(prompt_jobs[index]["count"])) for index in indices]
        total = sum(weights)
        if total < 1:
            return [indices[slot % len(indices)] for slot in range(strategies_per_parent)]
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
    strict_strategy_validation = bool(getattr(cfg, "strict_strategy_validation", False))
    if two_stage_rollouts:
        for parent_group in range(len(parents)):
            schedule = _source_schedule(parent_group)
            strategy_chains.append({
                "chain_id": len(strategy_chains),
                "parent_group": parent_group,
                "source_indices": list(schedule),
                "strategies": [],
            })
    def _record_strategy_response(chain, strategy_index, response_text):
        """Save and register a strategy at stream-arrival time."""
        if len(chain["strategies"]) != int(strategy_index):
            raise RuntimeError(
                f"strategy chain {chain['chain_id']} received strategy "
                f"{strategy_index} after {len(chain['strategies'])} records")
        raw_response = str(response_text or '')
        normalize_response = getattr(problem, "normalize_strategy_response", None)
        accepted_response = (str(normalize_response(raw_response)) if callable(normalize_response) else raw_response)
        save_strategy_response(exp_dir, step_idx, int(chain["parent_group"]), int(strategy_index), accepted_response)
        strategy, extraction_issue = _extract_final_strategy(accepted_response, strict=strict_strategy_validation)
        if extraction_issue is None and strict_strategy_validation:
            validate_strategy = getattr(problem, "strategy_output_issue", None)
            if callable(validate_strategy):
                extraction_issue = validate_strategy(strategy)
                if extraction_issue is not None:
                    strategy = _STRATEGY_FALLBACK
        if extraction_issue is not None:
            raise RuntimeError("cannot admit an invalid strategy into a dependent chain")
        chain["strategies"].append({
            "response": accepted_response,
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
                previous_strategies=[record["strategy"] for record in chain["strategies"]]))
        messages = deepcopy(strategy_prompt_cache[key])
        if retry:
            if getattr(cfg, "api_strategy", ""):
                from generation.strategy_prompts_openai import openai_strategy_retry_messages
                messages = openai_strategy_retry_messages(messages)
            else:
                messages = output_retry_messages(messages, "strategy")
        prompt = _render_strategy(messages, reasoning_effort=None)
        strategy_attempt_prompts[(*key, int(retry))] = prompt
        return prompt

    def _handle_strategy_attempt(chain_index, strategy_index, response, attempt):
        chain = strategy_chains[int(chain_index)]
        raw_response = str(response or '')
        normalize_response = getattr(problem, "normalize_strategy_response", None)
        checked_response = (str(normalize_response(raw_response)) if callable(normalize_response) else raw_response)
        strategy, issue = _extract_final_strategy(checked_response, strict=strict_strategy_validation)
        if issue is None and strict_strategy_validation:
            validate_strategy = getattr(problem, "strategy_output_issue", None)
            if callable(validate_strategy):
                issue = validate_strategy(strategy)
        save_strategy_response(
            exp_dir, step_idx, int(chain["parent_group"]),
            int(strategy_index), raw_response,
            attempt=int(attempt), extraction_issue=issue,
            prompt_text=strategy_attempt_prompts.pop((int(chain["chain_id"]), int(strategy_index), int(attempt))))
        needs_retry = strategy_retry_needed(
            issue, attempt, strategy_format_max_retries,
            label=f"strategy {strategy_index + 1}, chain {chain_index + 1}")
        if needs_retry:
            print(f"[warn] strategy {strategy_index + 1}, chain {chain_index + 1}: "
                  f"{issue}; format retry {int(attempt) + 1}/"
                  f"{strategy_format_max_retries} with unchanged reasoning/sampling",
                  flush=True)
        elif issue is not None:
            # Preserve every malformed raw attempt above, but make the
            # canonical strategy for this slot an explicit direct-solve
            # instruction.  This keeps dependent stages and coder rollout
            # counts intact without admitting malformed planner text.
            fallback_response = (f"<strategy>{_STRATEGY_FALLBACK}</strategy>")
            _record_strategy_response(chain, strategy_index, fallback_response)
            chain["strategies"][-1].update({
                "used_fallback": True,
                "fallback_issue": issue,
                "failed_attempts": int(attempt) + 1,
            })
            strategy_prompt_cache.pop((int(chain["chain_id"]), int(strategy_index)), None)
            print(
                f"[warn] strategy {strategy_index + 1}, chain "
                f"{chain_index + 1}: {issue} after "
                f"{int(attempt) + 1} attempts; continuing this slot without "
                "a strategist plan",
                flush=True,
            )
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
            for strategy_index, strategy_record in enumerate(chain["strategies"]):
                source_idx = chain["source_indices"][strategy_index]
                source_job = source_jobs[source_idx]
                strategy = strategy_record["strategy"]
                messages = problem.build_code_messages(source_job["messages"], strategy)
                code_jobs.append({
                    **source_job,
                    "messages": messages,
                    "prompt_text": _render(messages),
                    "count": programs_per_strategy,
                    "strategy": strategy,
                    "strategy_response": strategy_record["response"],
                    "strategy_index": strategy_index,
                })
        return code_jobs

    spo_rs_divergence = 0.0
    spo_rs_context_divergences = {}
    spo_rs_missing_policy_scores = 0
    spo_rs_previous_adapter_path = None
    if (spo_rs_mode and spo_rs_tracker.initialized):
        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is not None:
            spo_rs_previous_adapter_path = Path(exp_dir) / previous_name

    group_specs = []
    for parent_group in range(len(parents)):
        group_specs.append({
            "group_id": len(group_specs),
            "parent_group": parent_group,
            "context_id": parent_group,
        })
    num_groups = len(group_specs)
    total_rollouts = num_groups * cfg.group_size

    # ----- REWARD POOL (CPU), runs concurrently with generation -----
    # compute_reward delegates the heavy work to a subprocess sandbox, so the
    # launching thread mostly waits (GIL released) and many sandboxes run in
    # parallel across cores. THREAD-SAFETY REQUIREMENT: each compute_reward call
    # must use a unique temp file/dir and must not os.chdir or mutate shared
    # state; otherwise concurrent runs corrupt each other's rewards.
    from runtime.host_memory import HostMemoryAdmission
    host_admission = getattr(parallel_trainer, "host_memory_admission", None)
    if host_admission is None:
        host_admission = getattr(gen_pool, "host_memory_admission", None)
    if host_admission is None:
        import multiprocessing
        host_admission = HostMemoryAdmission(multiprocessing.get_context("spawn"))
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
        isolated_memory_limit_bytes = (_automatic_sandbox_memory_limit_bytes(n_reward_workers))
        setting_log_only(
            f"[step {step_idx}] isolated evaluation pool: "
            f"{n_reward_workers} sandbox process(es) across "
            f"{isolated_cpu_count} CPU(s), up to "
            f"{isolated_processes_per_cpu}/CPU from reward_workers; each "
            "process tree is pinned to one CPU and excess candidates remain "
            "queued; live RAM admission is shared with model transfers and "
            f"activation offload; automatic memory ceiling="
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
    rollout_io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rollout-save-step{step_idx}")
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
    defer_gpu_evaluation = bool(getattr(cfg, "evaluation_shares_generation", False))
    if adaptive_strategy_pilots and defer_gpu_evaluation:
        raise ValueError(
            "adaptive strategy pilots require CPU evaluation concurrent with "
            "rollout generation; set pilot_programs_per_strategy=-1 for the "
            "shared-GPU evaluator")

    def _run_isolated_reward(response_text, parent_ctx, **answer_options):
        with host_admission.reservation(isolated_memory_limit_bytes):
            cpu_id = isolated_cpu_slots.get()
            try:
                return problem.compute_reward(
                    response_text, parent_ctx, cfg.sandbox_timeout_s,
                    cpu_id=cpu_id,
                    memory_limit_bytes=isolated_memory_limit_bytes, **answer_options)
            finally:
                isolated_cpu_slots.put(cpu_id)

    def _submit_rollout(record):
        job = prompt_jobs[record["job_idx"]]
        g = int(record["group_id"])
        parent_group = int(job["parent_group"])
        group_responses[g].append(record)
        final_text = record["_final_answer_text"]
        answer_options = {} if final_text is None else {"final_answer_text": final_text}
        if record.get("output_format_issue") is not None:
            # A missing artifact is already a verified format failure. Never
            # execute an earlier reasoning snippet or evaluate it twice.
            fut = Future()
            fut.set_result(RewardResult(
                reward=float(problem.fail_score), msg=record["output_format_issue"],
                failure_kind="code"))
        elif isolated_eval:
            fut = reward_pool.submit(
                _run_isolated_reward, record["text"], parent_ctxs[parent_group], **answer_options)
        else:
            fut = reward_pool.submit(
                problem.compute_reward, record["text"],
                parent_ctxs[parent_group],
                cfg.sandbox_timeout_s, **answer_options
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
        token_ids = list(token_ids)
        if behavior_logprobs is not None:
            behavior_values = list(behavior_logprobs)
            behavior_logprobs = (
                behavior_values
                if len(behavior_values) == len(token_ids)
                and all(math.isfinite(float(value)) for value in behavior_values)
                else None)
        job = prompt_jobs[int(job_idx)]
        final_text = native_final_answer_scope(
            tokenizer, token_ids,
            require_final_marker=bool(getattr(problem, "require_final_code_marker", False)),
            prompt_text=job["prompt_text"])
        parent_group = int(job["parent_group"])
        parent_ordinal = queued_by_parent[parent_group]
        if retry_of is not None:
            group_id = int(retry_of["group_id"])
        else:
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
            "rollout_index": rollout_index,
            "text": text,
            "_final_answer_text": final_text,
            "token_ids": token_ids,
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": None,
            "output_format_issue": coder_output_issue(
                text, require_final_marker=bool(getattr(problem, "require_final_code_marker", False)),
                final_answer_text=final_text),
            "retry_attempt": 0 if retry_of is None else 1,
            "retry_of_group": None if retry_of is None else int(retry_of["group_id"]),
            "retry_of_rollout": None if retry_of is None else int(retry_of["rollout_index"]),
            "counts_toward_allocation": retry_of is None,
        }
        (primary_rollout_records if retry_of is None else coder_retry_records).append(record)
        if entropy_directory is not None:
            record["_entropy_measurement"] = observation_descriptor(entropy_directory, int(group_id), rollout_index)
        if rollout_phase is not None:
            record["strategy_rollout_phase"] = str(rollout_phase)
        if strategy_source_job_idx is not None:
            record["strategy_source_job_idx"] = int(strategy_source_job_idx)
        save_future = rollout_io_pool.submit(
            save_rollout_artifacts,
            exp_dir, step_idx, int(group_id), rollout_index, text,
            prompt_text=job["prompt_text"],
            strategy_text=(job.get("strategy_response") if two_stage_rollouts else None),
            response_token_ids=token_ids,
            behavior_logprobs=behavior_logprobs,
            pending_meta={
                **retry_metadata(record),
                "status": "evaluation_pending",
                "step": int(step_idx),
                "group": int(group_id),
                "parent_group": parent_group,
                "rollout": rollout_index,
                "n_response_tokens": len(token_ids),
                "coder_reasoning_effort": job.get("coder_reasoning_effort"),
                "strategy_rollout_phase": (str(rollout_phase) if rollout_phase is not None else None),
                "strategy_source_job_idx": (
                    int(strategy_source_job_idx)
                    if strategy_source_job_idx is not None else None),
            },
        )
        record["_artifact_save_future"] = save_future
        rollout_io_futures.append(save_future)
        if (training_enabled and cfg.generation_backend == "vllm" and token_ids):
            vllm_logprob_records.append(record)
        if defer_gpu_evaluation:
            deferred_rollouts.append(record)
        else:
            _submit_rollout(record)
        return record

    def _run_coder_format_retries(records, *, seed_offset):
        """Recover a low-validity batch; never return retries to the allocator."""
        if not bool(getattr(cfg, "coder_retry", False)):
            return
        triggered, total, valid, missing, retry_budget = coder_retry_gate(records)
        if not triggered:
            if missing:
                print(
                    f"[step {step_idx}] coder format recovery skipped: "
                    f"{valid}/{total} complete blocks (at least 50%); "
                    f"{missing} incomplete original(s) retain failure reward",
                    flush=True,
                )
            return
        failed = [record for record in records
                  if record.get("output_format_issue") is not None
                  and not record.get("retry_attempt", 0)]
        if not failed:
            return
        # Spread a limited recovery budget across the streamed failure order
        # instead of always favoring the first strategy/worker group.
        if retry_budget < len(failed):
            failed = [failed[(index * len(failed)) // retry_budget] for index in range(retry_budget)]
        prompt_indices = [coder_retry_prompt_job(
            prompt_jobs, record, _render, coder_retry_prompt_cache,
            reasoning_effort=prompt_jobs[int(record["job_idx"])].get(
                "coder_reasoning_effort", _coder_effort_for_rollout_phase(cfg, record.get("strategy_rollout_phase"))),
            continue_from_failure=bool(getattr(problem, "coder_retry_uses_failed_response", False)))
            for record in failed]
        prompts = [prompt_jobs[index]["prompt_text"] for index in prompt_indices]
        with eval_progress_condition:
            eval_bar.total += len(failed)
            eval_bar.refresh()
        print(f"[step {step_idx}] coder format recovery: "
              f"{valid}/{total} complete blocks (below 50%); "
              f"{len(failed)} extra attempts "
              "to target at least 60% valid; originals retained, allocation unchanged, "
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
            stream = generate_prompt_jobs(model, tokenizer, prompts, [1] * len(prompts), cfg, cap_state=cap_state)
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
        # Retain every planned record, including fully valid batches, because
        # the 50% trigger is defined over the complete pilot + phase-2 set.
        if records:
            deferred_coder_retry_batches.append((list(records), int(seed_offset)))

    def _drain_deferred_coder_format_retries():
        if not bool(getattr(cfg, "coder_retry", False)):
            deferred_coder_retry_batches.clear()
            return
        batches = list(deferred_coder_retry_batches)
        deferred_coder_retry_batches.clear()
        all_records = [record for records, _seed_offset in batches for record in records]
        _run_coder_format_retries(all_records, seed_offset=0)

    def _strategy_plan_metadata(source_job_idx):
        source_job = prompt_jobs[int(source_job_idx)]
        return {
            "strategy_pilot_reward_mean": source_job.get("strategy_pilot_reward_mean"),
            "strategy_pilot_reward_variance": source_job.get("strategy_pilot_reward_variance"),
            "strategy_pilot_mean_threshold": source_job.get("strategy_pilot_mean_threshold"),
            "strategy_pilot_variance_threshold": source_job.get("strategy_pilot_variance_threshold"),
            "strategy_pilot_scenario": source_job.get("strategy_pilot_scenario"),
            "strategy_phase2_allocation_method": source_job.get("strategy_phase2_allocation_method"),
            "strategy_pilot_valid_count": source_job.get("strategy_pilot_valid_count"),
            "strategy_posterior_valid_probability": source_job.get("strategy_posterior_valid_probability"),
            "strategy_initial_expected_improvement": source_job.get("strategy_initial_expected_improvement"),
            "strategy_hurdle": source_job.get("strategy_hurdle"),
            "strategy_pilot_followup_count": source_job.get("strategy_pilot_followup_count"),
            "strategy_allocated_programs": source_job.get("strategy_allocated_programs"),
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

    def _finish_strategy_pilots(pilot_records, source_job_indices, *, rewards_already_ready=False):
        """Finish one or more ready parent pilot sets exactly."""
        expected_pilots = (len(source_job_indices) * pilot_programs_per_strategy)
        if len(pilot_records) != expected_pilots:
            raise RuntimeError(f'pilot generation returned {len(pilot_records)}/{expected_pilots} rollouts')
        pilot_futures = [record.get("_reward_future") for record in pilot_records]
        if any(future is None for future in pilot_futures):
            raise RuntimeError('adaptive pilot allocation requires CPU rewards to run concurrently with generation')
        if rewards_already_ready:
            if not all(future.done() for future in pilot_futures):
                raise RuntimeError('pilot allocation was requested before its rewards were ready')
        else:
            wait(pilot_futures)

        rewards_by_source = {int(source_idx): [] for source_idx in source_job_indices}
        raw_scores_by_source = {int(source_idx): [] for source_idx in source_job_indices}
        valid_flags_by_source = {int(source_idx): [] for source_idx in source_job_indices}
        records_by_source = {int(source_idx): [] for source_idx in source_job_indices}
        for record in pilot_records:
            source_idx = int(record["strategy_source_job_idx"])
            result = record["_reward_future"].result()
            # Only a short prefix is persisted in rollout metadata. Drop the
            # remainder as soon as the pilot statistic has consumed the result
            # so completed futures cannot retain hundreds of MiB of diagnostic
            # output while phase 2 is generated.
            if len(result.stdout or '') > 4096:
                result.stdout = ((result.stdout or '')[:4096] + "\n[... sandbox output truncated ...]")
            reward = float(result.reward)
            if not math.isfinite(reward):
                raise RuntimeError(f"pilot reward for strategy job {source_idx} is not finite")
            rewards_by_source[source_idx].append(reward)
            valid_flags_by_source[source_idx].append(bool(getattr(result, "valid", False)))
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
            key = int(job["parent_group"])
            chains.setdefault(key, []).append(int(source_idx))

        followup_counts = {int(source_idx): 0 for source_idx in source_job_indices}
        for parent_group, chain_indices in sorted(chains.items()):
            chain_indices.sort(key=lambda index: int(prompt_jobs[index]["strategy_index"]))
            if len(chain_indices) != strategies_per_parent:
                raise RuntimeError(
                    f"parent {parent_group} has "
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

            remaining = (strategies_per_parent * (programs_per_strategy - pilot_programs_per_strategy))
            bandit_diagnostics = [None] * len(chain_indices)
            bandit_summary = None

            if phase2_allocation_method in {"bandit", "hurdle"}:
                from search.rollout_allocation import (allocate_hurdle_expected_best, allocate_posterior_expected_best)
                allocation_seed = (
                    int(cfg.seed) * 1_000_003
                    + (int(step_idx) + 1) * 1009
                    + (int(parent_group) + 1) * 9176
                    + 131
                ) % (2 ** 32)
                reward_groups = [rewards_by_source[index] for index in chain_indices]
                valid_groups = [valid_flags_by_source[index] for index in chain_indices]
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
                    baseline_label = (f"parent reward={float(saved_parent_reward):.9f}, ")
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
                    f": {allocation_label}; "
                    f"{baseline_label}"
                    f"pilot best reward="
                    f"{float(bandit_summary['pilot_best_reward']):.9f}, "
                    f"projected phase-2 gain="
                    f"{gain_text}; "
                    f"allocating {remaining} phase-2 rollouts"
                    + (fallback_label if bandit_summary.get("fallback_equal") else ""),
                    flush=True,
                )
            else:
                # Original rule-based allocator. Keep this path byte-for-byte
                # equivalent in behavior so selecting the new bandit cannot
                # change or retire existing experiments.
                mean_threshold = float(np.median(np.asarray(means, dtype=np.float64)))
                variance_threshold = float(np.median(np.asarray(variances, dtype=np.float64)))
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
                    for source_idx, mean, variance in zip(chain_indices, means, variances)
                ]

                # The original rule-based zero-signal exploration floor.
                zero_signal = [
                    abs(mean) <= 1e-12 and abs(variance) <= 1e-12
                    for mean, variance in zip(means, variances)
                ]
                requested_floor = max(0, int(pilot_programs_per_strategy) // 3)
                floor_allocations = [0] * len(chain_indices)
                floor_demand = requested_floor * sum(zero_signal)
                if floor_demand > 0:
                    floor_budget = min(remaining, floor_demand)
                    if floor_budget == floor_demand:
                        floor_allocations = [requested_floor if is_zero else 0 for is_zero in zero_signal]
                    else:
                        floor_allocations = _allocate_largest_remainder(
                            floor_budget,
                            [1.0 if is_zero else 0.0 for is_zero in zero_signal],
                            tie_values,
                        )
                weighted_budget = remaining - sum(floor_allocations)
                weighted_allocations = _allocate_largest_remainder(weighted_budget, weights, tie_values)
                followups = [
                    int(floor_count) + int(weighted_count)
                    for floor_count, weighted_count in zip(floor_allocations, weighted_allocations)
                ]
                if sum(followups) != remaining:
                    raise RuntimeError("adaptive rollout exploration floor lost budget")
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
                    f": mean threshold="
                    f"{mean_threshold:.9f}, variance threshold="
                    f"{variance_threshold:.9f}; allocating {remaining} "
                    "phase-2 rollouts"
                    + (" equally (all pilot statistics tied)" if fallback_equal else "")
                    + floor_label,
                    flush=True,
                )

            for (source_idx, mean, variance, scenario, followup,
                 bandit_arm) in zip(chain_indices, means, variances, scenarios, followups, bandit_diagnostics):
                job = prompt_jobs[source_idx]
                allocated = pilot_programs_per_strategy + int(followup)
                raw_scores = raw_scores_by_source[source_idx]
                best_raw_score = ((max(raw_scores) if problem.maximize else min(raw_scores)) if raw_scores else None)
                diagnostic = {
                    "parent_group": int(parent_group),
                    "strategy_index": int(job["strategy_index"]),
                    "pilot_count": int(pilot_programs_per_strategy),
                    "pilot_reward_mean": float(mean),
                    "pilot_reward_variance": float(variance),
                    "mean_threshold": (float(mean_threshold) if mean_threshold is not None else None),
                    "variance_threshold": (float(variance_threshold) if variance_threshold is not None else None),
                    "scenario": str(scenario),
                    "allocation_method": phase2_allocation_method,
                    "followup_count": int(followup),
                    "allocated_programs": int(allocated),
                    "source_job_idx": int(source_idx),
                    "pilot_best_raw_score": (float(best_raw_score) if best_raw_score is not None else None),
                }
                if bandit_arm is not None:
                    diagnostic.update({
                        "pilot_valid_count": int(bandit_arm["valid_count"]),
                        "posterior_valid_probability": float(bandit_arm["posterior_valid_probability"]),
                        "initial_expected_improvement": float(bandit_arm["initial_expected_improvement"]),
                        "final_marginal_expected_improvement": float(bandit_arm["final_marginal_expected_improvement"]),
                        "projected_expected_gain": float(bandit_summary["projected_expected_gain"]),
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
                    "strategy_pilot_mean_threshold": (float(mean_threshold) if mean_threshold is not None else None),
                    "strategy_pilot_variance_threshold": (
                        float(variance_threshold)
                        if variance_threshold is not None else None),
                    "strategy_pilot_scenario": str(scenario),
                    "strategy_phase2_allocation_method": (phase2_allocation_method),
                    "strategy_pilot_valid_count": diagnostic.get("pilot_valid_count"),
                    "strategy_posterior_valid_probability": diagnostic.get("posterior_valid_probability"),
                    "strategy_initial_expected_improvement": diagnostic.get("initial_expected_improvement"),
                    "strategy_hurdle": diagnostic.get("hurdle"),
                    "strategy_pilot_followup_count": int(followup),
                    "strategy_allocated_programs": int(allocated),
                })
                followup_counts[source_idx] = int(followup)
                for record in records_by_source[source_idx]:
                    _attach_strategy_plan(record, source_idx, "pilot")
                best_raw_text = (f"{best_raw_score:.9f}" if best_raw_score is not None else "unavailable")
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

            if (sum(pilot_programs_per_strategy + count for count in followups) != int(cfg.group_size)):
                raise RuntimeError('adaptive strategy allocation changed the fixed group rollout budget')
        return [followup_counts[int(source_idx)] for source_idx in source_job_indices]

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
                records_by_source.setdefault(int(source_idx), []).append(record)

        diagnostics_by_chain = {}
        for diagnostic in strategy_pilot_diagnostics:
            key = int(diagnostic["parent_group"])
            diagnostics_by_chain.setdefault(key, []).append(diagnostic)

        for parent_group, diagnostics in sorted(diagnostics_by_chain.items()):
            diagnostics.sort(key=lambda item: int(item["strategy_index"]))
            allocation_method = str(diagnostics[0].get("allocation_method") or "rule_based")
            phase2_total = sum(int(item["followup_count"]) for item in diagnostics)
            if allocation_method in {"bandit", "hurdle"}:
                allocation_label = (
                    "hurdle expected-best" if allocation_method == "hurdle"
                    else "posterior expected-best bandit")
                print(
                    f"[step {step_idx}] rollout phase 2 complete parent "
                    f"{parent_group}: {allocation_label}; "
                    f"evaluated {phase2_total} "
                    "phase-2 rollouts",
                    flush=True,
                )
            else:
                mean_threshold = float(diagnostics[0]["mean_threshold"])
                variance_threshold = float(diagnostics[0]["variance_threshold"])
                print(
                    f"[step {step_idx}] rollout phase 2 complete parent "
                    f"{parent_group}: mean threshold="
                    f"{mean_threshold:.9f}, variance threshold="
                    f"{variance_threshold:.9f}; evaluated {phase2_total} "
                    "phase-2 rollouts",
                    flush=True,
                )

            for diagnostic in diagnostics:
                source_idx = int(diagnostic["source_job_idx"])
                followup_count = int(diagnostic["followup_count"])
                pilot_best = diagnostic.get("pilot_best_raw_score")
                pilot_best_text = (f"{float(pilot_best):.9f}" if pilot_best is not None else "unavailable")
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
                            raise RuntimeError('phase-2 report requested before all strategy rewards were ready')
                        result = future.result()
                        raw_score = getattr(result, "raw_score", None)
                        if (not bool(getattr(result, "valid", False)) or raw_score is None):
                            continue
                        try:
                            raw_score = float(raw_score)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(raw_score):
                            raw_scores.append(raw_score)
                    final_best = ((max(raw_scores) if problem.maximize else min(raw_scores)) if raw_scores else None)
                    diagnostic["final_best_raw_score"] = (float(final_best) if final_best is not None else None)
                    final_best_text = (f"{final_best:.9f}" if final_best is not None else "unavailable")
                    line += (
                        f" {_ANSI_YELLOW}--> best raw after new rollouts "
                        f"{problem.metric_name}={final_best_text}"
                        f"{_ANSI_RESET}"
                    )
                print(line, flush=True)

    def _run_adaptive_followups_as_ready(pilot_records, source_job_indices, run_phase):
        """Allocate and generate each parent as soon as its pilots finish.

        The selected allocator sees all strategies and every pilot for that
        same parent. There is no global all-parent reward barrier: a ready
        parent keeps the rollout GPUs occupied while slow CPU sandboxes
        belonging to other parents continue in parallel.
        """
        source_job_indices = [int(index) for index in source_job_indices]
        source_position = {source_idx: position for position, source_idx in enumerate(source_job_indices)}
        records_by_source = {source_idx: [] for source_idx in source_job_indices}
        for record in pilot_records:
            source_idx = int(record["strategy_source_job_idx"])
            if source_idx not in records_by_source:
                raise RuntimeError(f"pilot record references unknown strategy {source_idx}")
            records_by_source[source_idx].append(record)
        for source_idx, records in records_by_source.items():
            if len(records) != pilot_programs_per_strategy:
                raise RuntimeError(
                    f"strategy job {source_idx} has {len(records)} pilot "
                    f"records; expected {pilot_programs_per_strategy}")

        pending_chains = {}
        for source_idx in source_job_indices:
            job = prompt_jobs[source_idx]
            key = int(job["parent_group"])
            pending_chains.setdefault(key, []).append(source_idx)
        for key, chain_indices in pending_chains.items():
            chain_indices.sort(key=lambda index: int(prompt_jobs[index]["strategy_index"]))
            if len(chain_indices) != strategies_per_parent:
                raise RuntimeError(
                    f"parent {key} has "
                    f"{len(chain_indices)} strategy jobs; expected "
                    f"{strategies_per_parent}")

        remaining_per_chain = (strategies_per_parent * (programs_per_strategy - pilot_programs_per_strategy))
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
                    raise RuntimeError('adaptive pilot scheduler has no ready chain or pending reward')
                wait(incomplete, return_when=FIRST_COMPLETED)
                continue

            ready_keys.sort()
            ready_sources = [source_idx for key in ready_keys for source_idx in pending_chains[key]]
            ready_records = [record for source_idx in ready_sources for record in records_by_source[source_idx]]
            ready_followups = _finish_strategy_pilots(ready_records, ready_sources, rewards_already_ready=True)
            batch_counts = [0] * len(source_job_indices)
            for source_idx, count in zip(ready_sources, ready_followups):
                position = source_position[source_idx]
                batch_counts[position] = int(count)
                all_followup_counts[position] = int(count)
            batch_total = sum(batch_counts)
            if batch_total:
                run_phase(batch_counts, dispatch_index, batch_total, dispatched + batch_total, expected_followups)
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
    coder_phase_complete = False
    try:
        try:
            api_planned = bool(two_stage_rollouts and getattr(cfg, "strategy_backend", "local") == "api")
            if api_planned:
                from generation.strategy_api import run_strategy_rounds
                if strategy_pool is None:
                    raise RuntimeError("API strategy mode requires its remote generation pool")
                source_prompt_jobs = prompt_jobs
                setting_log_only(
                    f"[step {step_idx}] API planning: {len(strategy_chains)} parents "
                    f"x {strategies_per_parent} sequential rounds with "
                    f"{_strategy_model_name(cfg)}", flush=True)
                run_strategy_rounds(
                    strategy_pool, num_parents=len(strategy_chains),
                    num_strategies=strategies_per_parent,
                    prompt_builder=lambda parent, stage, retry: _strategy_prompt(
                        source_prompt_jobs, strategy_chains[parent], stage, retry=retry),
                    result_handler=_handle_strategy_attempt,
                    max_retries=strategy_format_max_retries,
                    generation_options={
                        "max_new_tokens": int(cfg.strategy_max_new_tokens),
                        "temperature": float(cfg.strategy_temperature),
                        "top_p": float(cfg.strategy_top_p),
                        "top_k": getattr(cfg, "strategy_sampling_top_k", None),
                        "min_p": getattr(cfg, "strategy_sampling_min_p", None),
                        "step_idx": int(step_idx) + 2_000_000,
                    },
                )
                prompt_jobs = _code_prompt_jobs(source_prompt_jobs, strategy_chains)
            if gen_pool is not None:
                # Consume the generation stream (the adapter was already saved
                # above). CPU rewards and rewards on a distinct evaluation GPU
                # start immediately. A one-card GPU problem defers evaluation
                # until generation releases that same physical card.
                if two_stage_rollouts and not api_planned:
                    source_prompt_jobs = prompt_jobs
                    planning_pool = strategy_pool or gen_pool
                    setting_log_only(f"[step {step_idx}] hierarchical planning: "
                          f"{len(strategy_chains)} chain(s) x "
                          f"{strategies_per_parent} sequential strategies with "
                          f"{cfg.strategy_model_name}; LoRA disabled "
                          f"(max {cfg.strategy_max_new_tokens} tokens each)",
                          flush=True)
                    try:
                        def _generate_strategy_batch(prompts, chain_indices, strategy_index, *, retry=0):
                            responses = {}
                            completed = set()
                            suffix = " retry" if retry else ""
                            for prompt_index, job_results in (
                                    planning_pool.iter_group_jobs(
                                        prompts_by_group=prompts,
                                        group_size=1,
                                        counts_by_group=[1] * len(prompts),
                                        adapter_path=None,
                                        max_new_tokens=int(cfg.strategy_max_new_tokens),
                                        temperature=float(cfg.strategy_temperature),
                                        top_p=float(cfg.strategy_top_p),
                                        top_k=getattr(cfg, "strategy_sampling_top_k", None),
                                        min_p=getattr(cfg, "strategy_sampling_min_p", None),
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
                                if (prompt_index < 0 or prompt_index >= len(chain_indices)):
                                    raise RuntimeError('strategy generation returned an out-of-range prompt index')
                                chain_index = int(chain_indices[prompt_index])
                                if chain_index in completed:
                                    raise RuntimeError(
                                        "strategy generation returned chain "
                                        f"{chain_index} more than once")
                                responses[chain_index] = _handle_strategy_attempt(
                                    chain_index, strategy_index, job_results[0][0], retry)
                                completed.add(chain_index)
                            if completed != set(chain_indices):
                                raise RuntimeError('strategy generation did not return every parent chain')
                            return responses

                        if hasattr(planning_pool, "run_sequential_chains"):
                            def _build_ready_strategy_prompt(chain_index, strategy_index, retry):
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
                                max_new_tokens=int(cfg.strategy_max_new_tokens),
                                temperature=float(cfg.strategy_temperature),
                                top_p=float(cfg.strategy_top_p),
                                top_k=getattr(cfg, "strategy_sampling_top_k", None),
                                min_p=getattr(cfg, "strategy_sampling_min_p", None),
                                step_idx=int(step_idx) + 2_000_000,
                                progress_desc="strategies pipelined",
                            )
                        else:
                            # Legacy local pools retain the compatible
                            # depth-batched path. Local vLLM pools use the
                            # dependency-pipelined scheduler above.
                            for strategy_index in range(strategies_per_parent):
                                chain_indices = list(range(len(strategy_chains)))
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
                                    chain_indices = [index for index in chain_indices if needs_retry[index]]
                                    if not chain_indices:
                                        break
                    finally:
                        if (planning_pool is not gen_pool
                                and not getattr(cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False)):
                            planning_pool.release()
                    prompt_jobs = _code_prompt_jobs(source_prompt_jobs, strategy_chains)
                    setting_log_only(f"[step {step_idx}] two-stage coding: generating "
                          f"{sum(job['count'] for job in prompt_jobs)} "
                          f"LoRA-policy programs from {len(prompt_jobs)} "
                          "strategy-conditioned prompts", flush=True)

                if adaptive_strategy_pilots:
                    source_job_indices = list(range(len(prompt_jobs)))

                    def _run_vllm_code_phase(counts, *, phase, seed_offset, progress_desc, defer_format_retries=False):
                        active = _coder_effort_entries(prompt_jobs, source_job_indices, counts, cfg, step_idx, phase)
                        if not active:
                            return []
                        active_indices = [item[0] for item in active]
                        active_counts = [item[1] for item in active]
                        active_prompt_indices = [
                            _phase_coder_prompt_job(
                                prompt_jobs, index, phase, cfg, _render,
                                phase_prompt_job_cache, effort=effort)
                            for index, _count, effort in active]
                        options = {
                            "prompts_by_group": [prompt_jobs[index]["prompt_text"] for index in active_prompt_indices],
                            "group_size": max(active_counts),
                            "counts_by_group": active_counts,
                            "adapter_path": adapter_path,
                            "max_new_tokens": cfg.max_new_tokens,
                            "temperature": cfg.temperature,
                            "top_p": cfg.top_p,
                            "top_k": getattr(cfg, "sampling_top_k", None),
                            "min_p": getattr(cfg, "sampling_min_p", None),
                            "step_idx": int(step_idx) + int(seed_offset),
                            "progress_desc": progress_desc,
                            "full_policy_sampling": (_uses_sequence_level_policy_ratio(cfg)),
                        }
                        if cfg.generation_backend == "vllm":
                            options["return_logprobs"] = training_enabled
                        phase_records = []
                        for local_idx, job_results in (gen_pool.iter_group_jobs(**options)):
                            source_idx = active_indices[int(local_idx)]
                            for result in job_results:
                                text, token_ids = result[:2]
                                behavior_logprobs = (result[2] if len(result) > 2 else None)
                                record = _queue_rollout(
                                    active_prompt_indices[int(local_idx)],
                                    text, token_ids,
                                    behavior_logprobs,
                                    rollout_phase=phase,
                                    strategy_source_job_idx=source_idx)
                                _attach_strategy_plan(record, source_idx, phase)
                                phase_records.append(record)
                        expected = sum(active_counts)
                        if len(phase_records) != expected:
                            raise RuntimeError(
                                f"{phase} generation returned "
                                f"{len(phase_records)}/{expected} rollouts")
                        if defer_format_retries:
                            _defer_coder_format_retries(phase_records, seed_offset=seed_offset)
                        else:
                            _run_coder_format_retries(phase_records, seed_offset=seed_offset)
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
                            seed_offset=(4_000_000 + dispatch_index * 100_000),
                            progress_desc=(f'adaptive rollouts {dispatched_after}/{expected_total}'),
                            defer_format_retries=True)

                    _run_adaptive_followups_as_ready(pilot_records, source_job_indices, _run_ready_vllm_followups)
                    _drain_deferred_coder_format_retries()
                else:
                    # Keep the disabled path structurally identical to the
                    # original one-pass rollout scheduler.
                    generation_options = {
                        "prompts_by_group": [job["prompt_text"] for job in prompt_jobs],
                        "group_size": cfg.group_size,
                        "counts_by_group": [job["count"] for job in prompt_jobs],
                        "adapter_path": adapter_path,
                        "max_new_tokens": cfg.max_new_tokens,
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "top_k": getattr(cfg, "sampling_top_k", None),
                        "min_p": getattr(cfg, "sampling_min_p", None),
                        "step_idx": step_idx,
                        "full_policy_sampling": (_uses_sequence_level_policy_ratio(cfg)),
                    }
                    if cfg.generation_backend == "vllm":
                        generation_options["return_logprobs"] = training_enabled
                    for job_idx, job_results in gen_pool.iter_group_jobs(**generation_options):
                        for result in job_results:
                            text, token_ids = result[:2]
                            behavior_logprobs = (result[2] if len(result) > 2 else None)
                            _queue_rollout(job_idx, text, token_ids, behavior_logprobs)
                    _run_coder_format_retries(primary_rollout_records, seed_offset=0)
            else:
                # In-process generation uses cross-prompt micro-batches rather
                # than draining one parent at a time. Ordinary CPU verification
                # overlaps later batches; one-card GPU-mode verification waits.
                backend.set_inference_mode()
                cap_state = {"value": int(getattr(cfg, '_local_generation_cap', 0) or 0)}

                def _seed_local_generation(seed_step):
                    if cfg.deterministic:
                        torch.manual_seed((int(cfg.seed) * 1_000_003 + int(seed_step) * 1009 + 13) % (2**31 - 1))

                if two_stage_rollouts and not api_planned:
                    source_prompt_jobs = prompt_jobs
                    setting_log_only(f"[step {step_idx}] hierarchical planning: "
                          f"{len(strategy_chains)} chain(s) x "
                          f"{strategies_per_parent} sequential strategies with "
                          f"{cfg.strategy_model_name}; LoRA disabled "
                          f"(max {cfg.strategy_max_new_tokens} tokens each)",
                          flush=True)
                    strategy_cfg = SimpleNamespace(**vars(cfg))
                    strategy_cfg.advantage_mode = "entropic"
                    strategy_cfg.max_seq_length = int(cfg.strategy_max_seq_length)
                    strategy_cfg.temperature = float(cfg.strategy_temperature)
                    strategy_cfg.top_p = float(cfg.strategy_top_p)
                    strategy_cfg.sampling_top_k = getattr(cfg, "strategy_sampling_top_k", None)
                    strategy_cfg.sampling_min_p = getattr(cfg, "strategy_sampling_min_p", None)
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
                                    raise RuntimeError("strategy generation omitted a parent chain")
                                pending_chains = sorted(retry_chains)
                                if not pending_chains:
                                    break
                    prompt_jobs = _code_prompt_jobs(source_prompt_jobs, strategy_chains)
                    setting_log_only(f"[step {step_idx}] two-stage coding: generating "
                          f"{sum(job['count'] for job in prompt_jobs)} "
                          f"LoRA-policy programs from {len(prompt_jobs)} "
                          "strategy-conditioned prompts", flush=True)

                if adaptive_strategy_pilots:
                    source_job_indices = list(range(len(prompt_jobs)))

                    def _run_local_code_phase(counts, *, phase, seed_offset, progress_desc, defer_format_retries=False):
                        active = _coder_effort_entries(prompt_jobs, source_job_indices, counts, cfg, step_idx, phase)
                        if not active:
                            return []
                        active_indices = [item[0] for item in active]
                        active_counts = [item[1] for item in active]
                        active_prompt_indices = [
                            _phase_coder_prompt_job(
                                prompt_jobs, index, phase, cfg, _render,
                                phase_prompt_job_cache, effort=effort)
                            for index, _count, effort in active]
                        _seed_local_generation(int(step_idx) + int(seed_offset))
                        phase_records = []
                        gen_bar = make_progress_bar(sum(active_counts), desc=progress_desc)
                        try:
                            for local_idx, responses in generate_prompt_jobs(
                                    model, tokenizer,
                                    [prompt_jobs[index]["prompt_text"] for index in active_prompt_indices],
                                    active_counts, cfg,
                                    cap_state=cap_state):
                                source_idx = active_indices[int(local_idx)]
                                for text, token_ids in responses:
                                    record = _queue_rollout(
                                        active_prompt_indices[int(local_idx)],
                                        text, token_ids,
                                        rollout_phase=phase,
                                        strategy_source_job_idx=source_idx)
                                    _attach_strategy_plan(record, source_idx, phase)
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
                            _defer_coder_format_retries(phase_records, seed_offset=seed_offset)
                        else:
                            _run_coder_format_retries(phase_records, seed_offset=seed_offset)
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
                            seed_offset=(4_000_000 + dispatch_index * 100_000),
                            progress_desc=(f'adaptive rollouts {dispatched_after}/{expected_total}'),
                            defer_format_retries=True)

                    _run_adaptive_followups_as_ready(pilot_records, source_job_indices, _run_ready_local_followups)
                    _drain_deferred_coder_format_retries()
                    cfg._local_generation_cap = int(cap_state["value"])
                else:
                    # Keep the disabled path structurally identical to the
                    # original one-pass rollout scheduler.
                    _seed_local_generation(step_idx)
                    gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
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

            if adaptive_strategy_pilots:
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
                eval_done_before = sum(future.done() for futures in reward_futures.values() for future in futures)
                prompt_token_ids = {}
                score_pairs = []
                for record in vllm_logprob_records:
                    job_idx = record["job_idx"]
                    if job_idx not in prompt_token_ids:
                        prompt_token_ids[job_idx] = tokenizer(prompt_jobs[job_idx]["prompt_text"]).input_ids
                    score_pairs.append((
                        prompt_token_ids[job_idx], record["token_ids"]))
                if spo_rs_mode:
                    reference_scores = [None] * len(score_pairs)
                    current_policy_scores = [
                        (list(record["behavior_logprobs"])
                         if record["behavior_logprobs"] is not None
                         and len(record["behavior_logprobs"])
                         == len(record["token_ids"])
                         else None)
                        for record in vllm_logprob_records
                    ]
                    if any(value is None for value in current_policy_scores):
                        rescored_current = gen_pool.score_token_logprobs(
                            score_pairs, show_progress=True,
                            adapter_path=str(adapter_path),
                            require_complete=True)
                        current_policy_scores = [
                            existing if existing is not None else retry
                            for existing, retry in zip(current_policy_scores, rescored_current)
                        ]
                    for record, values in zip(vllm_logprob_records, current_policy_scores):
                        if values is None or len(values) != len(record["token_ids"]):
                            raise RuntimeError(
                                "SPO-RS could not obtain complete rollout-"
                                "policy logprobs before training")
                        if (record["behavior_logprobs"] is None
                                or len(record["behavior_logprobs"])
                                != len(record["token_ids"])):
                            record["behavior_logprobs"] = values
                            base = (
                                f"step{int(step_idx):02d}_"
                                f"group{int(record['group_id']):02d}_"
                                f"rollout{int(record['rollout_index']):03d}")
                            save_future = rollout_io_pool.submit(
                                save_rollout_logprobs,
                                Path(exp_dir) / f"step{int(step_idx):02d}"
                                / f"{base}.behavior_logprobs.zlib",
                                values,
                            )
                            rollout_io_futures.append(save_future)
                    previous_policy_scores = [None] * len(score_pairs)
                    same_policy = bool(
                        spo_rs_tracker.initialized
                        and spo_rs_previous_adapter_path is not None
                        and Path(adapter_path).resolve()
                        == spo_rs_previous_adapter_path.resolve())
                    if same_policy:
                        spo_rs_divergence = 0.0
                        spo_rs_context_divergences = {int(record["context_id"]): 0.0 for record in vllm_logprob_records}
                    elif (spo_rs_tracker.initialized):
                        if (spo_rs_previous_adapter_path is None or not spo_rs_previous_adapter_path.is_dir()):
                            raise FileNotFoundError(
                                "SPO-RS cannot compute the consecutive-policy "
                                "KL because its previous sampling adapter is "
                                f"missing: {spo_rs_previous_adapter_path}")
                        else:
                            previous_policy_scores = (
                                gen_pool.score_token_logprobs(
                                    score_pairs, show_progress=True,
                                    adapter_path=str(spo_rs_previous_adapter_path),
                                    require_complete=True))
                            (spo_rs_divergence,
                             spo_rs_missing_policy_scores,
                             spo_rs_context_divergences) = (
                                spo_rs_tracker.consecutive_policy_divergence(
                                    current_policy_scores,
                                    previous_policy_scores,
                                    [record["context_id"] for record in vllm_logprob_records],
                                    required_context_ids=[spec["context_id"] for spec in group_specs]))
                            if (spo_rs_missing_policy_scores or not math.isfinite(spo_rs_divergence)):
                                raise RuntimeError(
                                    "SPO-RS exact consecutive-policy scoring "
                                    "completed without a usable score for "
                                    f"{spo_rs_missing_policy_scores} rollout(s)")
                else:
                    current_policy_scores = [None] * len(score_pairs)
                    try:
                        reference_scores = gen_pool.score_token_logprobs(score_pairs, show_progress=True)
                    except Exception as error:
                        reference_scores = [None] * len(score_pairs)
                        print(f"[warn] vLLM reference scoring failed ({error!r}); "
                              "missing values will be recomputed exactly by "
                              "the HF trainer", flush=True)
                for record, values in zip(vllm_logprob_records, reference_scores):
                    record["reference_logprobs"] = values
                behavior_count = sum(record["behavior_logprobs"] is not None for record in vllm_logprob_records)
                reference_count = sum(record["reference_logprobs"] is not None for record in vllm_logprob_records)
                reference_hf_fallbacks = (0 if spo_rs_mode else len(vllm_logprob_records) - reference_count)
                eval_done_after = sum(future.done() for futures in reward_futures.values() for future in futures)
                reference_label = (
                    'off for SPO-RS' if spo_rs_mode else f'{reference_count}/{len(vllm_logprob_records)}')
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

            coder_phase_complete = True
        finally:
            if gen_pool is not None and getattr(gen_pool, "sequential", False):
                immediate_restore = bool(
                    coder_phase_complete and training_enabled
                    and not defer_gpu_evaluation
                    and hasattr(parallel_trainer, "restore_on_gpus")
                    and hasattr(gen_pool, "release_for_training")
                    and not getattr(cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False))
                if immediate_restore:
                    gen_pool.release_for_training(parallel_trainer)
                else:
                    gen_pool.release()
            if (getattr(cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False)
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
            # Both transforms freeze calibration and baseline from history,
            # so reward-complete examples can stream into the same fast
            # distributed backward while other CPU evaluations finish.
            spo_rs_overlap_eligible = bool(
                spo_rs_mode
                and training_enabled
                and parallel_trainer is not None
                and hasattr(parallel_trainer, "begin_spo_rs_overlap")
                and cfg.generation_backend == "vllm"
                and int(getattr(cfg, "spo_rs_update_epochs", 1)) == 1
                and not deferred_rollouts
                and not evaluation_trainer_offloaded
            )
            if spo_rs_overlap_eligible:
                staged_spo_rs_examples = []
                try:
                    staged_spo_rs_examples = (
                        _prepare_spo_rs_overlap_inputs(tokenizer, group_responses, reward_futures, prompt_jobs))
                    if staged_spo_rs_examples:
                        pending_before = sum(not future.done() for future in all_futs)
                        pending_by_future = {
                            example["_overlap_reward_future"]: example
                            for example in staged_spo_rs_examples
                        }
                        if (len(pending_by_future) != len(staged_spo_rs_examples)):
                            raise RuntimeError(
                                "SPO-RS overlap found a reward future shared "
                                "by multiple rollout records")
                        spo_rs_overlap_state = (parallel_trainer.begin_spo_rs_overlap(cfg, step_idx))
                        print(
                            f"[step {step_idx}] SPO-RS "
                            "evaluation/training overlap started; "
                            f"prepared {len(staged_spo_rs_examples)} exact "
                            "training inputs; "
                            f"{pending_before}/{len(all_futs)} CPU "
                            "evaluations were pending before trainer restore",
                            flush=True,
                        )
                        while pending_by_future:
                            ready_futures = [future for future in pending_by_future if future.done()]
                            if not ready_futures:
                                wait(tuple(pending_by_future), return_when=FIRST_COMPLETED)
                                continue
                            ready_examples = [pending_by_future.pop(future) for future in ready_futures]
                            _complete_spo_rs_overlap_examples(ready_examples, spo_rs_tracker)
                            parallel_trainer.queue_spo_rs_overlap(ready_examples)

                        if spo_rs_overlap_state is not None:
                            parallel_trainer.seal_spo_rs_overlap()
                        print(
                            f"[step {step_idx}] SPO-RS overlap queued "
                            f"{spo_rs_overlap_state['queued_examples']} "
                            "examples to the shared "
                            f"{parallel_trainer.world_size}-GPU work queue "
                            f"(old logprobs "
                            f"{spo_rs_overlap_state['supplied_old']}/"
                            f"{spo_rs_overlap_state['queued_examples']}); "
                            "backward continues while rollout artifacts are "
                            "processed; optimizer step deferred",
                            flush=True,
                        )
                except Exception as error:
                    if spo_rs_overlap_state is not None:
                        parallel_trainer.finish_spo_rs_overlap(
                            cfg, step_idx,
                            spo_rs_overlap_state["queued_examples"],
                            apply_update=False)
                    elif not parallel_trainer._workers_idle:
                        raise
                    for responses in group_responses.values():
                        for record in responses:
                            record.pop("_spo_rs_overlap_example", None)
                    spo_rs_overlap_state = None
                    print(
                        f"[warn] SPO-RS evaluation/training overlap was "
                        f"disabled for this step ({type(error).__name__}: "
                        f"{error}); falling back to the unchanged "
                        "post-evaluation update",
                        flush=True,
                    )

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
                pending_groups = {int(group_id) for group_id in range(num_groups) if reward_futures[group_id]}
                overlap_normalization_examples = sum(
                    bool(record.get("token_ids"))
                    for responses in group_responses.values()
                    for record in responses)
                overlap_prompt_ids = {}
                try:
                    while pending_groups:
                        ready_groups = [
                            group_id for group_id in sorted(pending_groups)
                            if all(future.done() for future in reward_futures[group_id])
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
                                    pending_before = sum(not future.done() for future in all_futs)
                                    entropic_overlap_state = (
                                        parallel_trainer.
                                        begin_entropic_overlap(cfg, step_idx, overlap_normalization_examples))
                                    print(
                                        f"[step {step_idx}] entropic "
                                        "evaluation/training overlap started; "
                                        f"{pending_before}/{len(all_futs)} CPU "
                                        "evaluations still pending",
                                        flush=True,
                                    )
                                queued = (parallel_trainer.queue_entropic_overlap(prepared))
                                remaining = sum(not future.done() for future in all_futs)
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

            # If training finished first, wait only for the remaining reward
            # tail. If rewards finished first, this returns immediately.  Do
            # not consume exceptions here: the aligned per-group result() call
            # below remains the single place that reports reward failures.
            wait(all_futs)
            with eval_progress_condition:
                eval_progress_condition.wait_for(lambda: eval_progress_completed >= len(all_futs))
        finally:
            eval_bar.close()
            eval_bar_closed = True
    finally:
        reward_pool.shutdown(wait=True)
        rollout_io_pool.shutdown(wait=True)
        if not eval_bar_closed:
            eval_bar.close()
        if evaluation_trainer_offloaded:
            print(f'[step {step_idx}] restoring trainer after shared-GPU benchmark evaluation')
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
        spo_rs_context_divergences = {int(spec["context_id"]): 0.0 for spec in group_specs}

    if (spo_rs_mode and training_enabled and cfg.generation_backend != "vllm"):
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
                "response_ids": torch.tensor([record["token_ids"]], device=model.device),
                "_spo_rs_record": record,
            })

        def _score_local_policy(field_name):
            for batch in _training_microbatches(score_examples, cfg):
                scores = compute_batched_token_logprobs(
                    model, batch, with_grad=False,
                    chunk=cfg.logprob_chunk,
                    pad_token_id=tokenizer.pad_token_id)
                for example, score in zip(batch, scores):
                    example["_spo_rs_record"][field_name] = (score.detach().cpu())

        with _policy_dropout_disabled(model):
            try:
                _score_local_policy("_spo_rs_current_score")
                for record in current_records:
                    record["behavior_logprobs"] = record.get("_spo_rs_current_score")

                same_policy = bool(
                    spo_rs_tracker.initialized
                    and spo_rs_previous_adapter_path is not None
                    and Path(adapter_path).resolve()
                    == spo_rs_previous_adapter_path.resolve())
                if same_policy:
                    spo_rs_divergence = 0.0
                    spo_rs_context_divergences = {int(record["context_id"]): 0.0 for record in current_records}
                elif spo_rs_tracker.initialized:
                    if (spo_rs_previous_adapter_path is None or not spo_rs_previous_adapter_path.is_dir()):
                        spo_rs_divergence = math.inf
                        spo_rs_missing_policy_scores = len(current_records)
                    else:
                        try:
                            _load_adapter(model, spo_rs_previous_adapter_path, announce=False)
                            _score_local_policy("_spo_rs_previous_score")
                        finally:
                            _load_adapter(model, adapter_path, announce=False)
                        (spo_rs_divergence,
                         spo_rs_missing_policy_scores,
                         spo_rs_context_divergences) = (
                            spo_rs_tracker.consecutive_policy_divergence(
                                [record.get("_spo_rs_current_score") for record in current_records],
                                [record.get("_spo_rs_previous_score") for record in current_records],
                                [record["context_id"] for record in current_records],
                                required_context_ids=[spec["context_id"] for spec in group_specs]))
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
            else "missing" if not math.isfinite(spo_rs_divergence)
            else f"{spo_rs_divergence:.6f}")
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

    # Both SPO transforms use one immutable, run-wide pre-step baseline.
    # Update history only after every advantage has been formed, including
    # those already used by the overlap lane.
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
            transformed = spo_rs_tracker.transformed_rewards([sample["reward"] for sample in samples])
            normalized = spo_rs_tracker.normalized_advantages(transformed)
            if normalized is None:
                raise RuntimeError("SPO-RS lost its pre-step historical baseline")
            update = spo_rs_tracker.update(
                transformed,
                divergence=spo_rs_divergence,
                policy_adapter=adapter_path,
                step=step_idx,
                raw_rewards=[sample["reward"] for sample in samples],
            )
            update.update({
                "groups": sorted({sample["group"] for sample in samples}),
                "prompt_jobs": sorted({sample["job"] for sample in samples}),
                "contexts": sorted({int(sample["record"]["context_id"]) for sample in samples}),
                "context_divergences": spo_rs_context_divergences,
                "missing_policy_scores": int(spo_rs_missing_policy_scores),
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

            divergence_label = ("missing" if update["divergence"] is None else f"{update['divergence']:.6f}")
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
            codes.append(res.code or '')
            valids.append(res.valid)
            outs.append(res)

        rewards_np = np.array(rewards, dtype=np.float64)
        if rewards_np.size == 0:
            print(f"  group {g}: no rollouts returned; no update for this group")
            continue
        adv_mode = active_advantage_mode
        spo_rs_rollout_info = [None] * len(responses)
        spo_rs_trainable = [True] * len(responses)
        if spo_rs_mode:
            prepared = [spo_rs_prepared[(g, r_idx)] for r_idx in range(len(responses))]
            advantages = np.asarray([item["advantage"] for item in prepared], dtype=np.float64)
            spo_rs_trainable = [item["trainable"] for item in prepared]
            spo_rs_rollout_info = [item["info"] for item in prepared]
            adv_scale = float(cfg.spo_rs_beta)
            adv_scale_label = "beta"
        else:
            advantages, adv_scale = compute_group_advantages(rewards_np)
            adv_scale_label = "beta"
        constant = (float(rewards_np.max() - rewards_np.min()) < 1e-12)

        step_valid_count += sum(valids)
        step_rollout_count += len(valids)
        step_code_failure_count += sum(is_code_failure(res) for res in outs)

        print(f"  group {g}: rewards min={rewards_np.min():.9f} "
              f"mean={rewards_np.mean():.9f} max={rewards_np.max():.9f}  "
              f"valid={sum(valids)}/{len(valids)}  "
              f"{adv_scale_label}={adv_scale:.9f}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, record in enumerate(responses):
            text = record["text"]
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            res = outs[r_idx]
            job = prompt_jobs[job_idx]
            artifact_rollout_index = int(record.get("rollout_index", r_idx))
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
                all_child_strategy_keys.append(int(job["strategy_index"]) if two_stage_rollouts else None)
            pick_info = (sampler.last_picks_info[parent_group] if parent_group < len(sampler.last_picks_info) else {})
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
                "beta": (float(adv_scale) if adv_mode in ("entropic", "spo-rs") else 0.0),
                "advantage_mode": adv_mode,
                "advantage_scale": (float(adv_scale) if math.isfinite(adv_scale) else None),
                "n_response_tokens": len(token_ids),
                "coder_reasoning_effort": job.get("coder_reasoning_effort"),
                "sandbox_stdout": (res.stdout or '')[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_raw_score": (float(parent.raw_score) if parent.raw_score is not None else None),
                "parent_is_seed": parent.id in sampler._seed_ids,
                "parent_visit_count": int(pick_info.get("n", 0)),
                "parent_q_value": (float(pick_info["Q"]) if pick_info.get("Q") is not None else None),
                "parent_prior": (float(pick_info["P"]) if pick_info.get("P") is not None else None),
                "parent_exploration_bonus": (float(pick_info["bonus"]) if pick_info.get("bonus") is not None else None),
                "parent_selection_score": (float(pick_info["score"]) if pick_info.get("score") is not None else None),
                "archive_eligible": archive_eligible,
                "two_stage_rollout": bool(two_stage_rollouts),
                "strategy": (job.get("strategy") if two_stage_rollouts else None),
                "strategy_index": (job.get("strategy_index") if two_stage_rollouts else None),
                "strategy_model_name": (cfg.strategy_model_name if two_stage_rollouts else None),
                "programs_per_strategy": (programs_per_strategy if two_stage_rollouts else None),
                # The solution itself, and the one it started from. Neither is
                # recoverable afterwards: `construction` lives only in the
                # in-memory sampler State, and a mid-run rollout's parent array
                # is gone by the time anyone wants to plot it. Saving the result
                # means reproducing a figure needs no re-execution at all, which
                # also sidesteps programs that are stochastic or wall-clock
                # bounded and therefore cannot replay identically.
                "construction": (_as_float_list(getattr(res, "construction", None), cfg.max_saved_construction)
                                 if save_ctor else None),
                "parent_construction": (_as_float_list(
                    getattr(parent, "construction", None),
                    cfg.max_saved_construction) if save_ctor else None),
                "seed": int(cfg.seed),
            }
            if adaptive_strategy_pilots:
                meta.update({
                    "pilot_programs_per_strategy": (pilot_programs_per_strategy),
                    "strategy_rollout_phase": record.get("strategy_rollout_phase"),
                    "strategy_source_job_idx": record.get("strategy_source_job_idx"),
                    "strategy_pilot_reward_mean": record.get("strategy_pilot_reward_mean"),
                    "strategy_pilot_reward_variance": record.get("strategy_pilot_reward_variance"),
                    "strategy_pilot_mean_threshold": record.get("strategy_pilot_mean_threshold"),
                    "strategy_pilot_variance_threshold": record.get("strategy_pilot_variance_threshold"),
                    "strategy_pilot_scenario": record.get("strategy_pilot_scenario"),
                    "strategy_phase2_allocation_method": record.get("strategy_phase2_allocation_method"),
                    "strategy_pilot_valid_count": record.get("strategy_pilot_valid_count"),
                    "strategy_posterior_valid_probability": record.get("strategy_posterior_valid_probability"),
                    "strategy_initial_expected_improvement": record.get("strategy_initial_expected_improvement"),
                    "strategy_hurdle": record.get("strategy_hurdle"),
                    "strategy_pilot_followup_count": record.get("strategy_pilot_followup_count"),
                    "strategy_allocated_programs": record.get("strategy_allocated_programs"),
                })
            if spo_rs_mode:
                meta["spo_rs"] = spo_rs_rollout_info[r_idx]
            if entropy_directory is not None:
                entropy_observations.append(rollout_observation(record.get("_entropy_measurement"), meta, res.code))
            save_rollout(exp_dir, step_idx, g, artifact_rollout_index,
                         text, meta,
                         prompt_text=job["prompt_text"],
                         strategy_text=(job.get("strategy_response") if two_stage_rollouts else None),
                         artifacts_already_saved=True)
            saved_rollouts += 1

        # Entropic training skips constant groups. SPO-RS compares every
        # rollout with its historical baseline, so tied groups can still learn.
        if constant and not clipped_policy_mode:
            continue

        for r_idx, (record, adv) in enumerate(zip(responses, advantages)):
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            if not training_enabled:
                continue
            if spo_rs_mode and not spo_rs_trainable[r_idx]:
                continue
            if len(token_ids) == 0:
                continue
            prepared = (record.get("_entropic_overlap_example") or record.get("_spo_rs_overlap_example"))
            res = outs[r_idx]
            if prepared is not None:
                prepared_advantage = prepared.get("advantage")
                if (prepared_advantage is not None and float(prepared_advantage) != float(adv)):
                    prepared["_overlap_label_mismatch"] = True
                prepared.update({
                    "advantage": float(adv),
                    "reward_constant": constant,
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
                torch.tensor(behavior_values, dtype=torch.float32, device=model.device)
                if behavior_values is not None
                and len(behavior_values) == len(token_ids) else None)
            reference_logprobs = (
                torch.tensor(reference_values, dtype=torch.float32, device=model.device)
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
                "group_id": g,
                "rollout_index": r_idx,
                "prompt_job_id": job_idx,
                "sample_weight": 1.0 / (num_groups * len(responses)),
            })

    # Persistence barrier: every response/prompt/meta file is on disk before the
    # optimizer can mutate the adapter. Streamed entropic and SPO-RS training
    # may already have accumulated exact gradients during
    # CPU evaluation, but their single update is deliberately deferred until
    # this barrier.
    # stepXX.summary.json remains the completion marker and is written only
    # after the adapter/checkpoint.
    save_suffix = ("with training disabled" if not training_enabled else "before adapter update")
    print(f'[step {step_idx}] saved {saved_rollouts} rollout .txt/.meta.json pairs {save_suffix}', flush=True)

    valid_fraction = (step_valid_count / step_rollout_count if step_rollout_count else 0.0)
    code_valid_fraction = (1.0 - step_code_failure_count / step_rollout_count if step_rollout_count else 1.0)

    # Preserve the standard objectives' constant-reward-group filtering.
    if not clipped_policy_mode:
        all_examples = [ex for ex in all_examples if not ex["reward_constant"]]

    if clipped_policy_mode and cfg.generation_backend == "vllm":
        # SPO-RS requires the frozen behavior probability. A malformed vLLM
        # payload cannot be reconstructed safely after the adapter changes and
        # must never force a full-context HF cache pass. Rollouts remain saved;
        # only an unusable training record is omitted.
        before = len(all_examples)
        all_examples = [
            example for example in all_examples
            if _valid_example_token_logprobs(example, "behavior_logprobs")
        ]
        unusable = before - len(all_examples)
        if unusable:
            print(f"[step {step_idx}] vLLM omitted behavior logprobs for "
                  f"{unusable} rollout(s); saved them but excluded only those "
                  "records from the clipped-policy update", flush=True)

    if spo_rs_mode and all_examples:
        trajectory_weight = 1.0 / len(all_examples)
        for example in all_examples:
            example["sample_weight"] = trajectory_weight


    if entropic_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(math.isfinite(float(example["advantage"])) for example in all_examples)
        advantages_match = not any(example.get("_overlap_label_mismatch", False) for example in all_examples)
        expected_keys = entropic_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys or not advantages_complete or not advantages_match):
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

    if spo_rs_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(
            example.get("advantage") is not None
            and math.isfinite(float(example["advantage"]))
            for example in all_examples)
        advantages_match = not any(example.get("_overlap_label_mismatch", False) for example in all_examples)
        expected_keys = spo_rs_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys or not advantages_complete or not advantages_match):
            parallel_trainer.finish_spo_rs_overlap(
                cfg, step_idx,
                spo_rs_overlap_state["queued_examples"],
                apply_update=False)
            print(
                f"[warn] discarded streamed SPO-RS gradients because the "
                f"final training set changed ({len(actual_keys)}/"
                f"{len(expected_keys)} records); using the unchanged "
                "post-evaluation update",
                flush=True,
            )
            spo_rs_overlap_state = None

    rollout_time = time.time() - rollout_t0
    training_label = (str(len(all_examples)) if training_enabled else "disabled")
    rollout_phase_label = (
        "rollout+eval+overlapped-backward wall time"
        if (entropic_overlap_state is not None or spo_rs_overlap_state is not None)
        else "rollout+eval time")
    print(f"[step {step_idx}] {rollout_phase_label}: {rollout_time:.1f}s  "
          f"training examples: {training_label}  "
          f"new children: {len(all_children)}")

    # Keep the previous raw optimum so the status line can distinguish a new
    # run best from a step that merely repeats the best-ever value.
    previous_best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    previous_best_value = (float(previous_best_raw.raw_score) if previous_best_raw is not None else None)

    # Update archive
    archive_stats = sampler.update(
        all_children,
        strategy_keys=(all_child_strategy_keys if two_stage_rollouts else None),
        strategy_top_r=(int(cfg.strategy_archive_top_r) if two_stage_rollouts else 0),
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
            key=lambda state: (float(state.value) if state.value is not None else -math.inf),
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
            or (best_value > previous_best_value if problem.maximize else best_value < previous_best_value)
        )
        # ``target`` is part of the problem context shown to the model.
        # ``target_print`` is deliberately display-only, so changing the SOTA
        # comparison printed here cannot leak a new target into the prompt.
        target = getattr(cfg, "target_print", getattr(problem, "target", None))
        try:
            target = float(target) if target is not None else None
        except (TypeError, ValueError):
            target = None
        beats_sota = bool(
            improved and target is not None and math.isfinite(target)
            and (best_value > target if problem.maximize else best_value < target)
        )
        message = (
            f"[step {step_idx}] best-ever raw {problem.metric_name}: "
            f"{best_value:.9f} ({direction}; "
            f"reward={float(best_raw.value):.9f}, found step={best_raw.timestep})"
        )
        if beats_sota:
            margin = (best_value - target if problem.maximize else target - best_value)
            message += (f'  🏆 CONGRATULATIONS — NEW SOTA! target={target:.12f}, beaten by {margin:.12f} 🏆')
            color = "\033[32m"       # dark green
        elif improved:
            message += "  ★ NEW RUN BEST"
            color = "\033[92m"       # light green
        else:
            message += "  no improvement this step"
            color = "\033[93m"       # yellow
        print(f"{color}{message}\033[0m", flush=True)
    else:
        message = (f'[step {step_idx}] best-ever raw {problem.metric_name}: unavailable')
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
            if (step_best_result is not None and step_best_result.value is not None) else None),
        "best_seen_raw_score": (
            float(best_seen_result.raw_score)
            if (best_seen_result is not None and best_seen_result.raw_score is not None)
            else None),
        "best_seen_reward": (
            float(best_seen_result.value)
            if (best_seen_result is not None and best_seen_result.value is not None)
            else None),
        "best_seen_step": (int(best_seen_result.timestep) if best_seen_result is not None else None),
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
        step_stats["strategy_pilot_allocation"] = (strategy_pilot_diagnostics)
    if spo_rs_mode:
        step_stats["spo_rs_updates"] = spo_rs_updates
        step_stats["spo_rs_adaptive_beta"] = adaptive_spo_beta
        step_stats["spo_rs_tracker_size"] = (0 if spo_rs_tracker is None else len(spo_rs_tracker))
        step_stats["spo_rs_divergence"] = (None if not math.isfinite(spo_rs_divergence) else float(spo_rs_divergence))
        step_stats["spo_rs_context_divergences"] = (spo_rs_context_divergences)
        step_stats["spo_rs_missing_policy_scores"] = int(spo_rs_missing_policy_scores)
        step_stats["spo_rs_sampling_adapter"] = Path(adapter_path).name
        step_stats["spo_rs_policy_updated"] = False

    if not training_enabled:
        step_stats["training_disabled"] = True
        step_stats["training_seconds"] = 0.0
        disabled_reason = str(getattr(cfg, "_training_disabled_reason", "--no-train"))
        print(f"[step {step_idx}] training disabled ({disabled_reason}); "
              "skipping training-only logprobs, backward, and optimizer update",
              flush=True)
        return step_stats

    if not all_examples:
        print(f"[step {step_idx}] no usable training examples")
        return step_stats

    if (sampler.selection_mode == "llm" and step_idx + 1 < int(cfg.num_steps)):
        sampler.prefetch_states(int(cfg.groups_per_step), selection_step=step_idx + 1)

    # A model-parallel trainer can remain on CPU throughout every vLLM phase
    # and all reward processing. Restore its shards only when this step has a
    # real backward update to perform.
    if ensure_trainer_ready is not None:
        ensure_trainer_ready()


    if clipped_policy_mode:
        if spo_rs_overlap_state is not None:
            step_stats.update(parallel_trainer.finish_spo_rs_overlap(
                cfg, step_idx, len(all_examples), apply_update=True))
            spo_rs_overlap_state = None
        elif parallel_trainer is not None:
            step_stats.update(parallel_trainer.train_spo_rs(all_examples, cfg, step_idx))
        else:
            step_stats.update(_train_spo_rs_examples(backend, model, tokenizer, optimizer, all_examples, cfg, step_idx))
        if spo_rs_mode:
            step_stats["spo_rs_policy_updated"] = True
        return step_stats

    # ----- TRAIN STEP -----
    print(f'[step {step_idx}] starting adapter training; rollout artifacts are already on disk', flush=True)
    if parallel_trainer is not None:
        if entropic_overlap_state is not None:
            step_stats.update(parallel_trainer.finish_entropic_overlap(
                cfg, step_idx, len(all_examples), apply_update=True))
            entropic_overlap_state = None
        else:
            step_stats.update(parallel_trainer.train_policy(all_examples, cfg, step_idx))
        best = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
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
            base_logprobs = [example.get("reference_logprobs") for example in batch]
            supplied_reference = all(_valid_example_token_logprobs(example, "reference_logprobs") for example in batch)
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
            for ex, cur_lp, base_lp in zip(batch, current_logprobs, base_logprobs):
                base_lp = base_lp.to(cur_lp.device)
                adv = ex["advantage"]
                logp_diff = (cur_lp - base_lp).detach()
                avg_logp_diff = logp_diff.mean()
                kl_adv = cfg.kl_penalty_coef * (avg_logp_diff - (cur_lp - base_lp))
                eff_adv = adv + kl_adv


                behavior_lp = ex.get("behavior_logprobs")
                has_behavior = _valid_example_token_logprobs(ex, "behavior_logprobs")
                if sequence_policy:
                    loss, policy_metrics = (
                        _a3b_sequence_clipped_standard_loss(
                            cfg, cur_lp,
                            behavior_lp if has_behavior else None,
                            base_lp, adv))
                    is_ratio = policy_metrics["ratio"]
                elif has_behavior:
                    is_ratio = _detached_behavior_importance_ratio(cfg, cur_lp, behavior_lp)
                    loss = -(is_ratio * eff_adv.detach() * cur_lp).mean()
                else:
                    if (behavior_lp is not None and not hasattr(train_step, "_is_len_warned")):
                        print('[warn] invalid behavior logprobs; skipping IS for affected examples')
                        train_step._is_len_warned = True
                    is_ratio = 1.0
                    loss = -(eff_adv.detach() * cur_lp).mean()

                if has_behavior:
                    is_ratio_sum += float(is_ratio.mean().item())
                    is_ratio_max = max(is_ratio_max, float(is_ratio.max().item()))
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
    _torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=cfg.grad_clip)
    optimizer.step()

    train_time = time.time() - train_t0
    step_stats["training_seconds"] = train_time
    step_stats["training_parallel_gpus"] = 1
    step_stats["oom_quarantined_examples"] = len(quarantined)
    is_msg = ""
    if is_ratio_count > 0:
        is_msg = (f'  IS ratio mean={is_ratio_sum / is_ratio_count:.9f} max={is_ratio_max:.3f}')
    print(f"[step {step_idx}] train time: {train_time:.1f}s  "
          f"avg loss: {total_loss / n_examples:.9f}  "
          f"avg logpi_theta - logpi_base: "
          f"{total_logp_delta / n_examples:.9f}{is_msg}  "
          f"OOM-quarantined={len(quarantined)}")

    # Save the same problem-native best-ever state reported after each step.
    # The independent raw record includes seeds and candidates removed by
    # exact-code deduplication/archive pruning; ``best_state()`` alone can
    # therefore save an older or worse stochastic realization.
    best = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
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
    from diagnostics.startup_logs import StartupLog, print_dashboard
    startup_log = StartupLog(time_offset=_LOG_TIME_OFFSET_SECONDS)
    with startup_log.capture():
        cfg, merged = load_config()
        if cfg.strategies and cfg.strategy_backend == "local":
            validate_strategist_runtime(cfg.strategy_model_name, prepare_encoder=True)
        # This must precede every import path that can initialize CUDA. Worker
        # and evaluation children set their own physical groups before import.
        _pin_training_process(cfg.training_gpu_ids)
        if cfg.evaluation_gpu_id is not None:
            os.environ["TTT_EVALUATION_GPU_ID"] = str(cfg.evaluation_gpu_id)

        # Save the reusable base configuration before runtime-only adjustments.
        from experiment_io import (append_step_result, make_experiment_dir,
                                   restart_changed_rollout_step,
                                   save_best_state_snapshot,
                                   save_final_summary, save_step_summary)
        resume_dir = merged.pop("_resume_dir", None)
        if resume_dir:
            # Read only the cursor via memory-mapped CPU storage before model
            # loading, so stale overflow recovery cannot consume a new batch.
            checkpoint = _load_training_checkpoint(resume_dir, mmap=True)
            if checkpoint is not None:
                restart_step = int(checkpoint["next_step"])
            else:
                try:
                    restart_step, _ = _legacy_resume_info(resume_dir)
                except FileNotFoundError:
                    restart_step = 0
            del checkpoint
            if restart_changed_rollout_step(cfg, resume_dir, restart_step):
                # Re-resolve layout without the old step's overflow marker.
                cfg, merged = load_config()
                merged.pop("_resume_dir", None)
        exp_dir = make_experiment_dir(cfg, resume_dir=resume_dir, config_dict=merged)
        setting_log_path = Path(exp_dir).resolve() / "setting.log"
        startup_log.title = ("Strategist Bandit" if cfg.strategies else "Single-Model Discovery")
        startup_log.bind(setting_log_path)
        bind_setting_log(setting_log_path, time_offset=_LOG_TIME_OFFSET_SECONDS)
    # Keep the full resolved settings in the file; show their grouped dashboard
    # once below. Restore normal print before loading/training begins.
    print = startup_log.print
    terminal_log_path = str(Path(exp_dir).resolve() / "temirnal.log")
    terminal_log.bind(terminal_log_path)
    if cfg.generation_backend == "vllm":
        from runtime.host_memory import validate_memory_handoff_runtime
        runtime = validate_memory_handoff_runtime(__file__)
        with (Path(exp_dir) / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(runtime, handle, indent=2)
        runtime_message = (f"[runtime] checkout={runtime['checkout']}; memory handoff: {runtime['memory_handoff']}")
        console_print(runtime_message, flush=True)
        setting_log_only(runtime_message, flush=True)
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
            log_handle.write(f'\n=== TTT vLLM log parent_pid={os.getpid()} run_dir={Path(exp_dir).resolve()} ===\n')
        print(f"[logs] vLLM output: {vllm_log_path}", flush=True)
        if (cfg.strategies
                and getattr(cfg, "strategy_backend", "local") == "local"
                and (getattr(cfg, "strategy_model_name", cfg.model_name)
                     != cfg.model_name
                     or getattr(cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False))):
            strategy_vllm_log_path = str(Path(exp_dir).resolve() / "strategy_vllm.log")
            with open(strategy_vllm_log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n=== TTT strategy vLLM log parent_pid={os.getpid()} "
                    f"run_dir={Path(exp_dir).resolve()} ===\n")
            print(f'[logs] strategy vLLM output: {strategy_vllm_log_path}', flush=True)
        dependency_log_path = vllm_log_path
    else:
        dependency_log_path = str(Path(exp_dir).resolve() / "dependency_warnings.log")
        with open(dependency_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== TTT dependency warnings parent_pid={os.getpid()} "
                f"run_dir={Path(exp_dir).resolve()} ===\n")
        print(f"[logs] dependency warnings: {dependency_log_path}", flush=True)

    # One effective seed for every generation stream. None => not seeded, which
    # is the original behaviour. Set once here and threaded through unchanged.
    run_seed = cfg.seed if cfg.deterministic else None
    print(f"[init] deterministic = {cfg.deterministic}" + (f" (seed {cfg.seed})" if cfg.deterministic else ""))

    # Replicate the complete trainer whenever the profiled trainable copy fits
    # on one card, rather than layer-sharding one update across every card. The
    # main process still owns search/evaluation; only per-example gradient work
    # runs concurrently. vLLM phase sharing gives these replicas exclusive use
    # of the cards during training and requires them to offload for generation.
    replicated_backend_supported = bool(
        cfg.backend == "hf"
        or (cfg.backend == "unsloth" and getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b")
    )
    use_replicated_training = bool(
        not cfg.no_train
        and int(cfg.num_training_gpus) > 1
        and replicated_backend_supported
        and cfg.generation_backend == "vllm"
        and cfg.training_layout != "sharded"
    )
    # Entropic updates use isolated GPU processes even
    # without --fast.  The former same-interpreter threaded replicas are not a
    # safe execution boundary for concurrent autograd.  --fast remains an
    # optional scheduling choice: it enables adaptive batches; without it the
    # configured microbatch ceiling is honored.  Clipped-policy modes retain
    # their existing default trainer unless --fast is explicitly requested.
    standard_policy_objective = not bool(_uses_clipped_policy_loss(cfg.advantage_mode))
    use_process_distributed_training = bool(
        use_replicated_training
        and (cfg.fast or standard_policy_objective or getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b")
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
    print("Strategist Bandit" if cfg.strategies else "Single-Model Discovery")
    print("=" * 70)
    problem_type = getattr(cfg, "problem_type", "")
    print(f"Problem:            {cfg.problem}" + (f" ({problem_type})" if problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Model:              {cfg.model_name}")
    if getattr(problem, "two_stage_rollouts", False):
        strategy_location = ("remote API; no local weights" if cfg.strategy_backend == "api" else "local; no LoRA")
        print(f'Strategy model:     {_strategy_model_name(cfg)} ({strategy_location})')
    if cfg.training_model_name != cfg.model_name:
        print(f"Training model:     {cfg.training_model_name}")
    print(f"Training backend:   {cfg.backend}")
    print(f"Generation backend: {cfg.generation_backend}")
    print(f"Training GPUs:      physical {cfg.training_gpu_ids}")
    training_layout = (
        "disabled (--no-train)" if cfg.no_train
        else ("process-distributed LoRA"
              if use_process_distributed_training
              else ("replicated data parallel" if use_replicated_training else "model parallel/single GPU"))
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
    print(f"Evaluation GPU:     {(cfg.evaluation_gpu_id if cfg.evaluation_gpu_id is not None else 'none')}")
    evaluation_mode = (
        f"isolated, {cfg.reward_workers} processes/CPU (--isolate-eval)"
        if cfg.isolate_eval else "default")
    print(f"CPU evaluation:    {evaluation_mode}")
    if cfg.generation_backend == "vllm":
        print(f"vLLM parallelism:   TP={cfg.vllm_tensor_parallel_size or 'auto'} "
              f"PP={cfg.vllm_pipeline_parallel_size}")
    print(f"Prompt target:      {cfg.target}")
    print(f"Print/SOTA target:  {getattr(cfg, 'target_print', cfg.target)}")
    print(f"Steps:              {cfg.num_steps}")
    selection = cfg.parent_selection
    selection_detail = ("gpt-6-luna, reasoning=max; first pick=best raw score"
                        if selection == "llm" else f"c={cfg.puct_c}")
    print(f"Search selection:   {selection.upper()} ({selection_detail})")
    configured_advantage_mode = getattr(cfg, "advantage_mode", "entropic")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    if configured_advantage_mode == "spo-rs":
        print("Reference KL:        off (SPO-RS PPO objective)")
    else:
        print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Advantage mode:     {getattr(cfg, 'advantage_mode', 'entropic')}")
    if _uses_sequence_level_policy_ratio(cfg):
        print("MoE policy ratio:    sequence-level geometric mean "
              "(Qwen3-30B-A3B only; existing asymmetric low/high bounds)")
    if configured_advantage_mode == "spo-rs":
        if bool(getattr(cfg, "spo_rs_adaptive_beta", False)):
            print(f'SPO-RS shaping:     historical arctan, width=1.25; beta={cfg.spo_rs_beta}; all rollouts train')
        else:
            print(f"SPO-RS beta:        fixed {cfg.spo_rs_beta}")
        print(f"SPO-RS D_half:      {cfg.spo_rs_d_half}")
        print(f"SPO-RS rho min/max: {cfg.spo_rs_rho_min} / {cfg.spo_rs_rho_max}")
        print(f"SPO-RS clip eps low/high: "
              f"{cfg.spo_rs_clip_epsilon_low} / "
              f"{cfg.spo_rs_clip_epsilon_high} "
              f"(bounds {1.0 - cfg.spo_rs_clip_epsilon_low:.4f} / "
              f"{1.0 + cfg.spo_rs_clip_epsilon_high:.4f})")
        print(f"SPO-RS update epochs: {cfg.spo_rs_update_epochs}")
    elif _uses_sequence_level_policy_ratio(cfg):
        print(f"Sequence clip eps low/high: "
              f"{cfg.sequence_clip_epsilon_low} / {cfg.sequence_clip_epsilon_high} "
              f"(bounds {1.0 - cfg.sequence_clip_epsilon_low:.4f} / "
              f"{1.0 + cfg.sequence_clip_epsilon_high:.4f})")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    if getattr(problem, "two_stage_rollouts", False):
        base_effort = _coder_effort_for_rollout_phase(cfg)
        pilot_count = int(getattr(cfg, "pilot_programs_per_strategy", -1))
        if (getattr(cfg, "coder_template_kind", "generic") == "qwen3.8" and pilot_count > 0):
            pilot_efforts = _coder_effort_entries([], [0], [pilot_count], cfg, 0, "pilot")
            pilot_description = " + ".join(f"{count} {effort}" for _, count, effort in pilot_efforts)
            phase2_description = ", ".join(
                f"{effort}={('remaining' if count == -1 else count)}"
                for effort in ("low", "medium", "xhigh")
                for count in [getattr(cfg, f"phase2_{effort}_programs_per_strategy", -1 if effort == "medium" else 0)])
            print(
                "Coder reasoning:    pilot/strategy="
                f"{pilot_description}; "
                f"phase 2/strategy: {phase2_description}; "
                f"unassigned={base_effort}")
        else:
            print(f"Coder reasoning:    all rollout phases={base_effort}")
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
            allocation_method = str(getattr(cfg, "phase2_allocation_method", "rule_based"))
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
            "only enough to target 60% when planned validity is below 50%"
            if bool(getattr(cfg, "coder_retry", False)) else
            "off (enable with --coder-retry)"
        )
        strategy_validation_label = (
            "strict" if bool(getattr(cfg, "strict_strategy_validation", False))
            else "basic format")
        print(f"Output retries:     strategy up to {cfg.strategy_format_max_retries} "
              f"({strategy_validation_label} validation; direct-solve fallback "
              f"on exhaustion); coder "
              f"{coder_retry_label}")
        print(f"Strategy sampling:  max_new={cfg.strategy_max_new_tokens}, "
              f"max_seq={cfg.strategy_max_seq_length}, "
              f"temperature={cfg.strategy_temperature}, "
              f"top_p={cfg.strategy_top_p}, "
              f"top_k={getattr(cfg, 'strategy_sampling_top_k', None)}, "
              f"min_p={getattr(cfg, 'strategy_sampling_min_p', None)}, "
              f"thinking={'on' if cfg.strategy_thinking else 'off'}, "
              f"reasoning_effort={cfg.strategy_reasoning_effort}")
        if getattr(cfg, "api_strategy", ""):
            print("Strategy API:       text only; tools disabled; local sampling "
                  "fields ignored; output budget includes reasoning")
    print(f"Max seq length:     {cfg.max_seq_length}")
    if getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b":
        fused_attention_label = ('GPT-OSS sink-aware Unsloth path (generic Flash-SDPA prohibited)')
    elif getattr(cfg, "coder_model_profile", "") == "qwen3.8-27b":
        fused_attention_label = ("on for full-attention layers; exact FLA for Gated DeltaNet")
    elif (getattr(cfg, "coder_model_profile", "") == "qwen3-30b-a3b-thinking-2507"):
        fused_attention_label = "on (exact Qwen full-attention path)"
    else:
        fused_attention_label = ("on" if cfg.fused_long_attention else "off")
    print(f"Fused long attention: {fused_attention_label}")
    print(f'Train microbatch:   up to {cfg.train_examples_per_microbatch} examples/GPU')
    print(f'LoRA:               rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}')
    print(f"Deterministic:      {cfg.deterministic}")
    if use_process_distributed_training:
        print(f"Training memory cap: {100.0 * float(cfg.training_memory_fraction):g}% "
              "per GPU (96%->98%->99% staged exact long-rollout rescue)")
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
        from models.model_backend import load_backend
        backend = load_backend(cfg.backend, cfg)
        model, tokenizer = backend.load()
    strategy_tokenizer = tokenizer
    if (getattr(problem, "two_stage_rollouts", False)
            and cfg.strategy_backend == "local"
            and cfg.strategy_model_name != cfg.model_name):
        from transformers import AutoTokenizer
        with _route_dependency_notices(dependency_log_path):
            strategy_tokenizer = AutoTokenizer.from_pretrained(cfg.strategy_model_name, trust_remote_code=True)
        if strategy_tokenizer.pad_token_id is None:
            strategy_tokenizer.pad_token = strategy_tokenizer.eos_token
    effective_4bit = bool(getattr(cfg, "effective_load_in_4bit", cfg.load_in_4bit))
    if getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b":
        training_precision = "original packed MXFP4 (no NF4 re-quantization)"
    else:
        training_precision = ("BitsAndBytes 4-bit" if effective_4bit else "checkpoint/default precision")
    print(f"[precision] training copy: {training_precision}")

    if cfg.generation_backend == "vllm":
        from runtime.gpu_runtime import validate_attention_heads
        validate_attention_heads(_attention_head_count(model), int(cfg.vllm_tensor_parallel_size), cfg.model_name)

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
    interrupted_spo_rs_artifacts = None
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
                print('[resume] legacy batch-growth state ignored; using fixed configured batch sizes')
            if configured_advantage_mode == "spo-rs":
                interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, start_step, cfg))
                if interrupted_spo_rs_artifacts is not None:
                    token_recovery = (
                        _prepare_recoverable_response_tokens(model, tokenizer, interrupted_spo_rs_artifacts))
                    print(
                        "[resume] preflight prepared the interrupted "
                        f"step-{start_step} responses: "
                        f"{token_recovery['exact_sidecars']} exact token "
                        "sidecar(s), "
                        f"{token_recovery['canonical_text']} legacy "
                        "text reconstruction(s), "
                        f"{token_recovery['behavior_logprob_sidecars']} "
                        "rollout-policy logprob sidecar(s), "
                        f"{token_recovery['restored_terminal_tokens']} "
                        "terminal EOS token(s) restored",
                        flush=True,
                    )
        else:
            if configured_advantage_mode == "spo-rs":
                interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, 0, cfg))
            initial_adapter = Path(exp_dir) / "adapter_initial"
            if (interrupted_spo_rs_artifacts is not None and initial_adapter.is_dir()):
                start_step = 0
                adapter_path = initial_adapter
                _load_adapter(model, adapter_path)
                token_recovery = _prepare_recoverable_response_tokens(model, tokenizer, interrupted_spo_rs_artifacts)
                current_policy_adapter_path = adapter_path
                print(
                    "[resume] found a fully evaluated interrupted SPO-RS "
                    "step 0; restored adapter_initial and will skip rollout "
                    "generation/evaluation",
                    flush=True,
                )
                print(
                    "[resume] preflight prepared all saved responses: "
                    f"{token_recovery['exact_sidecars']} exact token "
                    "sidecar(s), "
                    f"{token_recovery['canonical_text']} legacy text "
                    "reconstruction(s), "
                    f"{token_recovery['behavior_logprob_sidecars']} "
                    "rollout-policy logprob sidecar(s), "
                    f"{token_recovery['restored_terminal_tokens']} terminal "
                    "EOS token(s) restored",
                    flush=True,
                )
            else:
                legacy_resume = _legacy_resume_info(exp_dir)
                start_step, adapter_path = legacy_resume
                _load_adapter(model, adapter_path)
                current_policy_adapter_path = adapter_path
                print(
                    f"[resume] legacy run (no training_state.pt): restarting "
                    f"step {start_step} from {adapter_path.name}. The archive "
                    "will be reconstructed, but old PUCT/optimizer statistics "
                    "are unavailable."
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
    print(f'[init] trainable params: {trainable:,} / total {total:,} ({100 * trainable / total:.2f}%)')

    # ---- sampler ----
    from search.sampler import can_migrate_parent_selection, make_sampler
    restored_selection = (resume_payload["sampler"].get("selection_mode", cfg.parent_selection)
                          if resume_payload is not None else cfg.parent_selection)
    if (restored_selection != cfg.parent_selection
            and not can_migrate_parent_selection(restored_selection, cfg.parent_selection)):
        raise ValueError("saved sampler differs from the requested parent picker")
    sampler = make_sampler(
        parent_selection=restored_selection,
        group_size=int(cfg.group_size), seed=int(cfg.seed),
        exp_dir=exp_dir, problem=problem,
        num_seeds=len(seeds) if seeds else cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
        seed_states=seeds,
    )
    if resume_payload is not None:
        sampler.load_state_dict(resume_payload["sampler"])
        print(f'[resume] restored exact {sampler.selection_mode.upper()} archive and search statistics')
    elif legacy_resume is not None:
        n_states, n_rollouts = _restore_legacy_archive(sampler, exp_dir, before_step=start_step)
        print(f'[resume] reconstructed {n_states} valid archived candidates from {n_rollouts} earlier rollouts')
    setting_log_only(f"[init] sampler archive size = {sampler.archive_size()}", flush=True)

    spo_rs_tracker = None
    if (configured_advantage_mode == 'spo-rs'):
        from training.spo_rs import BoundedSPORSTracker, SPORSTracker
        bounded_shaping = bool(getattr(cfg, "spo_rs_adaptive_beta", False))
        tracker_class = BoundedSPORSTracker if bounded_shaping else SPORSTracker
        spo_rs_tracker = tracker_class(
            entropic_beta=cfg.spo_rs_beta,
            d_half=cfg.spo_rs_d_half,
            rho_min=cfg.spo_rs_rho_min,
            rho_max=cfg.spo_rs_rho_max,
            **({"fail_score": cfg.fail_score} if bounded_shaping else {}),
        )
        tracker_state = (resume_payload.get("spo_rs_tracker") if resume_payload is not None else None)
        if tracker_state is not None and (bounded_shaping or tracker_state.get("version") == 2):
            spo_rs_tracker.load_state_dict(tracker_state)
            print("[resume] restored the run-wide SPO-RS tracker")
        elif start_step:
            if bounded_shaping:
                raise ValueError(
                    "bounded SPO-RS requires its exact historical reward "
                    "checkpoint; refusing to approximate or reset its baseline")
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
                step=last_step, raw_rewards=rewards)
            source = ("legacy prompt trackers" if tracker_state is not None else "logged rollout history")
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
            initial_adapter = Path(_save_adapter(model, exp_dir, -1, cfg.model_name, directory_name="adapter_initial"))
            transformed = spo_rs_tracker.transformed_rewards(seed_rewards)
            initialization = spo_rs_tracker.update(
                transformed, divergence=0.0,
                policy_adapter=initial_adapter, step=-1,
                raw_rewards=seed_rewards)
            current_policy_adapter_path = initial_adapter
            print("[init] SPO-RS global tracker initialized before step 0 "
                  f"from {len(seed_rewards)} verified seed rewards: "
                  f"v={initialization['value_after']:.9f}, "
                  f"N_eff={initialization['effective_count_after']:.2f}; "
                  "step 0 policy update enabled", flush=True)

    removed_adapters = _prune_adapter_snapshots(
        exp_dir,
        _live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker),
    )
    if removed_adapters:
        print(f"[checkpoint] removed {len(removed_adapters)} superseded "
              "adapter snapshot(s) while restoring the rolling checkpoint",
              flush=True)

    resume_parent_prefetch = None
    if resume_dir:
        sampler, resume_parent_prefetch = _prefetch_resume_parents(
            sampler, cfg, exp_dir, start_step, current_policy_adapter_path,
            optimizer, spo_rs_tracker, problem=problem,
            interrupted_artifacts=interrupted_spo_rs_artifacts)

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

    # ---- generation pool ----
    gen_pool = None
    strategy_pool = None
    ensure_trainer_ready = None
    # Generation callbacks consult it so all later steps follow the genuine
    # no-training residency path instead of waiting for a nonexistent update.
    runtime_training_active = {"value": not bool(cfg.no_train)}
    if (getattr(problem, "two_stage_rollouts", False) and cfg.strategy_backend == "api"):
        from generation.strategy_api import StrategyAPIGenerationPool
        strategy_pool = StrategyAPIGenerationPool(
            model_name=_strategy_model_name(cfg),
            base_url=cfg.strategy_api_base_url,
            api_key_env=cfg.strategy_api_key_env,
            concurrency=cfg.strategy_api_concurrency,
            timeout_s=cfg.strategy_api_timeout_s,
            max_retries=cfg.strategy_api_max_retries,
            thinking=cfg.strategy_thinking,
            reasoning_effort=cfg.strategy_reasoning_effort,
            api_strategy=cfg.api_strategy,
        )
        print(f"[init] remote strategy API configured for "
              f"{_strategy_model_name(cfg)}; no strategist tokenizer, weights, "
              "or local generation pool loaded", flush=True)
    # vLLM forms exact TP/PP engines over every rollout card. Because those
    # cards also hold either training replicas or model shards, the two runtimes
    # alternate residency.
    use_gen_pool = bool(cfg.num_gpus) and (
        cfg.generation_backend == "vllm"
        or (cfg.num_gpus > 1 and cfg.num_training_gpus == 1))
    if use_gen_pool:
        from generation.gen_workers import (GenerationPool, HybridHFGenerationPool,
                                 PhasedVLLMGenerationPool, worker_seed)
        gpu_ids = _parse_gpu_ids(cfg.gpu_ids)
        dual_resident_qwen3_8b_pools = bool(getattr(cfg, "dual_resident_qwen3_8b_strategy_coder_pools", False))
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
            load_in_4bit=(effective_4bit and cfg.training_model_name == cfg.model_name),
            seed=run_seed,
            gen_micro_batch=cfg.gen_micro_batch,
            backend=cfg.generation_backend,
            lora_rank=(cfg.lora_rank),
            vllm_gpu_memory_utilization=generation_utilization,
            vllm_enforce_eager=cfg.vllm_enforce_eager,
            vllm_enable_prefix_caching=cfg.vllm_enable_prefix_caching,
            vllm_quantization=cfg.vllm_quantization,
            vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
            vllm_pipeline_parallel_size=cfg.vllm_pipeline_parallel_size,
            vllm_max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
            vllm_enable_expert_parallel=cfg.vllm_enable_expert_parallel,
            vllm_sleep_level=cfg.vllm_sleep_level,
            # A sleeping GPT-OSS EngineCore retains enough CUDA/NCCL state to
            # starve the packed-MXFP4 trainer's fused dequant workspace.  A
            # fresh rollout pool is faster overall than training every step
            # through the OOM-driven unfused fallback.
            vllm_persistent_workers=_coder_vllm_persistent_workers(cfg),
            # Distinct strategist/coder pools share the same cards. Account
            # for either a sleeping pool's residual CUDA/NCCL state or the
            # Qwen3-8B ablation's concurrently awake peer pool.
            vllm_co_resident_sleep=separate_strategy_pool,
            # Training modes consume exact chosen-token prompt logprobs after
            # generation. Give that vocabulary projection explicit memory
            # headroom; this changes scheduling only, never the scores.
            vllm_token_scoring=not bool(cfg.no_train),
            vllm_runtime_reserve_gib=float(cfg.vllm_runtime_reserve_gib),
            vllm_staged_loading=cfg.vllm_staged_loading,
            vllm_log_path=vllm_log_path,
        )
        if cfg.generation_backend == "vllm":
            from runtime.host_memory import HostMemoryAdmission
            import multiprocessing
            pool_options["host_memory_admission"] = (
                getattr(parallel_trainer, "host_memory_admission", None)
                or HostMemoryAdmission(multiprocessing.get_context("spawn")))
            trainer_offloaded = False

            def _offload_trainer_for_generation(phase="coder"):
                nonlocal trainer_offloaded
                if (parallel_trainer is not None and parallel_trainer.is_offloaded):
                    return
                if parallel_trainer is None and trainer_offloaded:
                    return
                from runtime.host_memory import (
                    GIB, estimated_module_bytes, estimated_optimizer_bytes,
                    host_memory_info, reclaim_unused_host_memory,
                )

                # Reclaim cached activation-offload blocks on *every* trainer
                # process before allocating complete CPU model replicas. This
                # is a phase-boundary operation: it leaves live tensors and
                # gradients untouched and runs only once per rollout step.
                if parallel_trainer is not None:
                    cleanups = parallel_trainer.reclaim_host_memory()
                    required_bytes = (parallel_trainer.estimated_host_offload_bytes())
                else:
                    torch.cuda.synchronize()
                    cleanups = [reclaim_unused_host_memory(
                        torch,
                        cuda_devices=tuple(range(torch.cuda.device_count())),
                        force=True,
                    )]
                    required_bytes = int(
                        1.05 * (estimated_module_bytes(model) + estimated_optimizer_bytes(optimizer))
                        + GIB
                    )

                reclaimed_bytes = sum(max(
                    0,
                    int(cleanup.get("pinned_cached_before", 0))
                    - int(cleanup.get("pinned_cached_after", 0)),
                ) for cleanup in cleanups)
                memory = host_memory_info()
                if reclaimed_bytes >= GIB:
                    available = ("unknown" if memory is None else f"{memory['available'] / GIB:.1f} GiB")
                    setting_log_only(
                        f"[host-memory] released "
                        f"{reclaimed_bytes / GIB:.1f} GiB of inactive pinned "
                        f"trainer cache; available RAM={available}",
                        flush=True,
                    )

                # Normally the cache purge above is sufficient and both vLLM
                # pools remain RAM-resident.  Under genuine pressure, discard
                # only sleeping level-1 backups, largest/unused peer first.
                # This trades one later checkpoint reload for survival instead
                # of permanently choosing a slower level-2/transient policy.
                if memory is not None:
                    reserve_bytes = max(32 * GIB, int(0.05 * memory["total"]))
                    target_available = required_bytes + reserve_bytes
                    if phase == "strategy":
                        pressure_candidates = (
                            ("coder", gen_pool),
                            ("strategist", strategy_pool),
                        )
                    else:
                        pressure_candidates = (
                            ("strategist", strategy_pool),
                            ("coder", gen_pool),
                        )
                    for pool_label, pool in pressure_candidates:
                        if memory["available"] >= target_available:
                            break
                        discard = getattr(pool, "discard_sleeping_host_residency", None)
                        if pool is None or not callable(discard) or not discard():
                            continue
                        memory = host_memory_info() or memory
                        print(
                            f"[host-memory] low-RAM guard discarded the "
                            f"sleeping {pool_label} vLLM weight backup; it "
                            "will reload on its next use",
                            flush=True,
                        )
                    if memory["available"] < target_available:
                        raise MemoryError(
                            "insufficient host RAM for exact trainer offload "
                            f"after cache reclamation: available="
                            f"{memory['available'] / GIB:.1f} GiB, estimated "
                            f"offload={required_bytes / GIB:.1f} GiB, safety "
                            f"reserve={reserve_bytes / GIB:.1f} GiB")
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
                        parallel_trainer.offload_for_generation(purge_host_cache=False)
                    else:
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
                if (parallel_trainer is not None and not parallel_trainer.is_offloaded):
                    trainer_offloaded = False
                    return
                if not runtime_training_active["value"]:
                    print('[gpu] keeping inactive training model offloaded (no-training phase)', flush=True)
                    return
                if dual_resident_qwen3_8b_pools and any(
                        pool is not None and getattr(pool, "active", False)
                        for pool in (gen_pool, strategy_pool)):
                    print("[gpu] keeping trainer offloaded while the other "
                          "Qwen3-8B generation pool remains active",
                          flush=True)
                    return
                if (parallel_trainer is not None or cfg.training_layout == "sharded"):
                    # Keep the trainer on CPU through generation and restore it
                    # once, lazily, when the actual gradient update begins.
                    trainer_label = ("trainer replicas" if parallel_trainer is not None else "sharded trainer")
                    print(f'[gpu] keeping {trainer_label} offloaded until the adapter update', flush=True)
                    return
                if not trainer_offloaded:
                    return
                print('[gpu] restoring trainer after shared-GPU vLLM generation', flush=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                backend.restore_after_generation()
                _restore_optimizer_state_to_parameters(optimizer)
                backend.set_training_mode()
                from runtime.host_memory import reclaim_unused_host_memory
                reclaim_unused_host_memory(torch, cuda_devices=tuple(range(torch.cuda.device_count())), force=True)
                trainer_offloaded = False

            def _ensure_sharded_trainer_ready():
                nonlocal trainer_offloaded
                if not runtime_training_active["value"]:
                    raise RuntimeError("trainer restore requested during a no-training phase")
                if not trainer_offloaded:
                    return
                print("[gpu] restoring sharded trainer for adapter update", flush=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                backend.restore_after_generation()
                _restore_optimizer_state_to_parameters(optimizer)
                backend.set_training_mode()
                from runtime.host_memory import reclaim_unused_host_memory
                reclaim_unused_host_memory(torch, cuda_devices=tuple(range(torch.cuda.device_count())), force=True)
                trainer_offloaded = False

            if cfg.training_layout == "sharded":
                ensure_trainer_ready = _ensure_sharded_trainer_ready

            gen_pool = PhasedVLLMGenerationPool(
                before_start=lambda: _offload_trainer_for_generation("coder"),
                after_stop=_restore_trainer_after_generation,
                **pool_options,
            )
            setting_log_only(
                f"[init] phase-shared vLLM pool configured across all "
                f"rollout GPUs {gpu_ids}", flush=True)
            if separate_strategy_pool:
                strategy_pool_options = dict(pool_options)
                # A sleeping local strategist still owns a CUDA context on
                # every rollout GPU.  That residue is small beside ordinary
                # coders, but it prevents a full-context GPT-OSS-120B coder
                # from restoring its level-1 allocation after training.  Keep
                # the expensive coder resident and make the much smaller
                # strategist pool transient so its processes are completely
                # gone before the coder wakes.
                strategy_persistent_workers = getattr(cfg, "strategy_vllm_persistent_workers", None)
                if str(getattr(cfg, "coder_model_profile", "")).lower() == ("gpt-oss-120b"):
                    strategy_persistent_workers = 0
                    setting_log_only(
                        "[memory] GPT-OSS-120B coder: local strategist "
                        "engines use transient residency so coder wake-up "
                        "retains the full 131k-context memory budget",
                        flush=True,
                    )
                strategy_pool_options.update({
                    "model_name": cfg.strategy_model_name,
                    "vllm_engine_options": strategist_engine_options(cfg.strategy_model_name),
                    "max_seq_length": cfg.strategy_max_seq_length,
                    "load_in_4bit": False,
                    "gen_micro_batch": getattr(cfg, "strategy_gen_micro_batch", cfg.gen_micro_batch),
                    "vllm_quantization": (cfg.strategy_vllm_quantization),
                    "vllm_tensor_parallel_size": (cfg.strategy_vllm_tensor_parallel_size),
                    "vllm_pipeline_parallel_size": (cfg.strategy_vllm_pipeline_parallel_size),
                    "vllm_max_num_batched_tokens": getattr(
                        cfg, "strategy_vllm_max_num_batched_tokens",
                        cfg.vllm_max_num_batched_tokens),
                    "vllm_sleep_level": cfg.strategy_vllm_sleep_level,
                    "vllm_token_scoring": False,
                    "vllm_runtime_reserve_gib": float(getattr(cfg, "strategy_vllm_runtime_reserve_gib", 16.0)),
                    "vllm_persistent_workers": strategy_persistent_workers,
                    "vllm_staged_loading": (cfg.strategy_vllm_staged_loading),
                    "vllm_log_path": strategy_vllm_log_path,
                })
                strategy_pool = PhasedVLLMGenerationPool(
                    before_start=lambda: _offload_trainer_for_generation("strategy"),
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
                    adapter_context = (backend.disable_adapter() if adapter_path is None else nullcontext())
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

            gen_pool = HybridHFGenerationPool(remote_pool=remote_pool, local_iter=_local_hf_rollouts)
            print(f"[init] hybrid HF rollout pool: live trainer on "
                  f"physical GPU {gpu_ids[0]} plus persistent workers "
                  f"{gpu_ids[1:]}")
        if (getattr(problem, "two_stage_rollouts", False) and strategy_pool is None):
            strategy_pool = gen_pool
        elif dual_resident_qwen3_8b_pools:
            print(f"[init] dual-resident Qwen3-8B pools: strategist and coder "
                  f"use independent GPU-complete inference topologies; "
                  "strategist is always base/no-LoRA, coder receives the "
                  "current LoRA", flush=True)
        if cfg.gen_micro_batch and cfg.gen_micro_batch > 0:
            limit_name = ("max_num_seqs" if cfg.generation_backend == "vllm" else "micro-batch")
            limit_scope = ("/engine" if cfg.generation_backend == "vllm" else "/GPU")
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

    if parallel_trainer is not None:
        # Exact overflow may need the same sleeping-backup reclamation as generation.
        parallel_trainer._generation_pools = (strategy_pool, gen_pool)

    # ---- main loop ----
    from experiment_io import StepPlotter
    step_plotter = StepPlotter(exp_dir, cfg.problem)
    cumulative_training_seconds = 0.0
    for completed_step in range(start_step):
        summary_path = (Path(exp_dir) / f"step{completed_step:02d}" / f"step{completed_step:02d}.summary.json")
        try:
            completed_summary = json.loads(summary_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if "result_metric_name" in completed_summary:
            append_step_result(exp_dir, completed_step, completed_summary)
        cumulative_training_seconds += float(completed_summary.get("training_seconds", 0.0) or 0.0)
    if cumulative_training_seconds:
        print(f'[resume] prior cumulative training time: {cumulative_training_seconds:.1f}s')
    restart_for_exact_recovery = False
    restart_after_exact_recovery = False
    active_step = None
    try:
        if (resume_parent_prefetch is not None and cfg.generation_backend == "vllm"):
            warmup_pool = (
                strategy_pool
                if (getattr(problem, "two_stage_rollouts", False) and cfg.strategy_backend == "local")
                else gen_pool)
            if warmup_pool is not None:
                warmup_pool.prepare()
        if start_step >= cfg.num_steps:
            print(f"[resume] run already reached requested num_steps={cfg.num_steps}")
        for step in range(start_step, cfg.num_steps):
            active_step = int(step)
            step_cfg = cfg
            runtime_training_active["value"] = not bool(step_cfg.no_train)
            if getattr(problem, "two_stage_rollouts", False):
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

            if (resume_dir and step == start_step and configured_advantage_mode == "spo-rs"):
                pending_spo_rs_artifacts = None
                if interrupted_spo_rs_artifacts is None:
                    interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, step, step_cfg))
                if interrupted_spo_rs_artifacts is None:
                    pending_spo_rs_artifacts = (_recoverable_pending_spo_rs_artifacts(exp_dir, step, step_cfg))
            else:
                interrupted_spo_rs_artifacts = None
                pending_spo_rs_artifacts = None

            sampler = _migrate_parent_selection(
                sampler, cfg, exp_dir, step, current_policy_adapter_path,
                optimizer, spo_rs_tracker,
                recovery_pending=(interrupted_spo_rs_artifacts is not None or pending_spo_rs_artifacts is not None),
                problem=problem)

            if interrupted_spo_rs_artifacts is not None:
                stats = _resume_spo_rs_step_from_artifacts(
                    backend, model, tokenizer, sampler, optimizer, step,
                    step_cfg, exp_dir, problem, parallel_trainer,
                    spo_rs_tracker, current_policy_adapter_path,
                    interrupted_spo_rs_artifacts)
            elif pending_spo_rs_artifacts is not None:
                print(
                    f"[resume] found {len(pending_spo_rs_artifacts)} exact "
                    f"saved trajectories for interrupted SPO-RS step {step}; "
                    "skipping strategist and coder generation",
                    flush=True,
                )
                completed_spo_rs_artifacts = (
                    _finalize_pending_spo_rs_artifacts(
                        tokenizer, sampler, step, step_cfg, exp_dir, problem,
                        gen_pool, spo_rs_tracker,
                        current_policy_adapter_path,
                        pending_spo_rs_artifacts))
                stats = _resume_spo_rs_step_from_artifacts(
                    backend, model, tokenizer, sampler, optimizer, step,
                    step_cfg, exp_dir, problem, parallel_trainer,
                    spo_rs_tracker, current_policy_adapter_path,
                    completed_spo_rs_artifacts)
            else:
                stats = train_step(
                    backend, model, tokenizer, sampler, optimizer, step,
                    step_cfg, exp_dir, problem, gen_pool,
                    strategy_pool=strategy_pool,
                    strategy_tokenizer=strategy_tokenizer,
                    parallel_trainer=parallel_trainer,
                    spo_rs_tracker=spo_rs_tracker,
                    sampling_adapter_path=(
                        current_policy_adapter_path if configured_advantage_mode == 'spo-rs' else None),
                    ensure_trainer_ready=ensure_trainer_ready)

            stats = stats or {}
            entropy_observations = stats.pop("_entropy_observations", [])
            step_training_seconds = float(stats.get("training_seconds", 0.0) or 0.0)
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
            if configured_advantage_mode == "spo-rs":
                if stats.get("spo_rs_policy_updated", False):
                    adapter_path = _save_adapter(model, exp_dir, step, cfg.model_name)
                else:
                    adapter_path = (Path(exp_dir) / stats["spo_rs_sampling_adapter"])
            else:
                adapter_path = _save_adapter(model, exp_dir, step, cfg.model_name)
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
            # Snapshot the sampler's independent native-metric optimum before
            # asynchronous plotting starts.  This includes unbeaten seeds and
            # evaluated children that no longer survive archive pruning.
            best_seen_state = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
            best_snapshot_path = save_best_state_snapshot(
                exp_dir, step, best_seen_state,
                metric_name=str(problem.metric_name),
                maximize=bool(problem.maximize),
                construction=(
                    _as_float_list(getattr(best_seen_state, "construction", None), cfg.max_saved_construction)
                    if (best_seen_state is not None and getattr(problem, "saves_construction", False))
                    else None),
            )
            step_summary["best_seen_snapshot"] = (
                str(Path(best_snapshot_path).relative_to(exp_dir))
                if best_snapshot_path is not None else None)
            save_step_summary(exp_dir, step, step_summary)
            append_step_result(exp_dir, step, step_summary)
            if getattr(step_cfg, "measure_entropy", False):
                from analysis.entropy_tools import save_step as save_entropy_step
                try:
                    entropy_summary = save_entropy_step(exp_dir, step, entropy_observations, step_cfg, stats)
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
                _live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker),
            )
            if removed_adapters:
                retained = (
                    "two rolling adapters for consecutive-policy KL"
                    if (spo_rs_tracker is not None
                        and len(_live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker)) > 1)
                    else Path(adapter_path).name
                )
                print(f"[checkpoint] removed {len(removed_adapters)} "
                      f"superseded adapter snapshot(s); retained {retained}",
                      flush=True)
            print(f"[checkpoint] completed step {step}; resume at step {step + 1}")
            step_plotter.submit(step)
            if (stats.get("exact_overflow_inprocess", False)
                    or getattr(cfg, "_exact_inprocess_recovery_step", None) == step):
                # Keep partial gradients until the *full* optimizer/sampler
                # checkpoint is durable. A crash before this point can resume
                # the same update; a successful step needs no process restart.
                _clear_exact_sharded_recovery(exp_dir)
                if hasattr(cfg, "_exact_inprocess_recovery_step"):
                    delattr(cfg, "_exact_inprocess_recovery_step")
            replica_capture_step = getattr(cfg, "_exact_replica_capture_step", None)
            if replica_capture_step is not None:
                if int(replica_capture_step) != int(step):
                    raise RuntimeError(
                        "exact replica-capture marker expected step "
                        f"{int(replica_capture_step)}, but completed step "
                        f"{step}")
                _clear_exact_sharded_recovery(exp_dir)
                print(
                    f"[recovery] legacy exact-OOM update for step {step} "
                    "completed on the fast replicas; recovery marker removed",
                    flush=True,
                )
                delattr(cfg, "_exact_replica_capture_step")
            replica_recovery_step = getattr(cfg, "_exact_replica_recovery_step", None)
            if replica_recovery_step is not None:
                if int(replica_recovery_step) != int(step):
                    raise RuntimeError(
                        "exact replica recovery marker expected step "
                        f"{int(replica_recovery_step)}, but completed step "
                        f"{step}")
                _clear_exact_sharded_recovery(exp_dir)
                print(
                    f"[recovery] RAM-bounded replicated overflow update for "
                    f"step {step} completed; recovery state removed",
                    flush=True,
                )
                delattr(cfg, "_exact_replica_recovery_step")
            recovery_step = getattr(cfg, "_exact_sharded_recovery_step", None)
            if recovery_step is not None:
                if int(recovery_step) != int(step):
                    raise RuntimeError(
                        "exact sharded recovery marker expected step "
                        f"{int(recovery_step)}, but completed step {step}")
                _clear_exact_sharded_recovery(exp_dir)
                restart_after_exact_recovery = True
                print(
                    f"[recovery] exact sharded update for step {step} is "
                    "checkpointed; restarting the normal fast replicated "
                    "trainer at the next step",
                    flush=True,
                )
                break
    except ExactLongContextTrainingOOM as error:
        if getattr(cfg, "_exact_sharded_recovery_step", None) is not None:
            raise RuntimeError(
                "an exact trajectory still failed after distributing the "
                "trainer across every selected GPU") from error
        if (active_step is None or configured_advantage_mode != "spo-rs"):
            raise
        recoverable = _recoverable_spo_rs_artifacts(exp_dir, active_step, cfg)
        if recoverable is None:
            raise RuntimeError(
                "the exact long-context trainer needs a multi-GPU retry, but "
                "this step does not yet have a complete durable SPO-RS "
                "rollout/evaluation set") from error
        marker = _write_exact_sharded_recovery(exp_dir, active_step, error)
        restart_for_exact_recovery = True
        recovery = _read_exact_sharded_recovery(exp_dir) or {}
        if recovery.get("overflow_gradient_file"):
            deferred_count = len(recovery.get("deferred_keys", ()))
            print(
                f"[recovery] step {active_step}: {deferred_count} exact "
                "trajectory/trajectories exceeded one replica; completed "
                "fast-path gradients and all rollout artifacts are durable. "
                "Restarting automatically so only the deferred work uses "
                "the trainer sharded across all GPUs "
                f"({marker.name}); generation, CPU evaluation, and completed "
                "backward passes will not repeat",
                flush=True,
            )
        else:
            print(
                f"[recovery] step {active_step}: one exact trajectory "
                "exceeded single-GPU training capacity; all generated "
                "tokens, rewards, advantages, and policy scores are already "
                "durable. Restarting automatically with the trainer sharded "
                f"across all GPUs ({marker.name}); generation and CPU "
                "evaluation will not repeat",
                flush=True,
            )
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
        if (parallel_trainer is not None and hasattr(parallel_trainer, "shutdown")):
            print("[shutdown] stopping process-distributed trainer ...")
            parallel_trainer.shutdown(for_restart=(restart_for_exact_recovery or restart_after_exact_recovery))

    if restart_for_exact_recovery or restart_after_exact_recovery:
        _restart_for_exact_resume(exp_dir)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    # Use the problem-native best-ever record, not merely the best state that
    # survived archive deduplication/pruning.
    best = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
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
                           best_raw_score=(float(best.raw_score) if best.raw_score is not None else None))
    else:
        print("No valid solution was ever produced.")
        save_final_summary(exp_dir, None, None, None)
    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()
