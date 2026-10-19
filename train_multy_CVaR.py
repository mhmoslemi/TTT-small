"""TTT-Discover multi-problem runner with entropic and SPO-RS training.

Configuration: self-contained problem YAML < resumed config < CLI flags.
Entropic training uses adaptive-beta, leave-one-out group advantages.
SPO-RS uses a historical run-wide baseline and clipped policy updates.
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
from output_retries import (coder_output_issue, coder_retry_gate,
                            coder_retry_prompt_job, final_answer_scope,
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
_EXACT_SHARDED_RECOVERY_FILE = ".exact_sharded_recovery.json"
_EXACT_OVERFLOW_GRADIENT_FILE = ".exact_spo_rs_overflow_step{step:04d}.pt"
_ANSI_ORANGE = "\033[38;5;208m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"


class ExactLongContextTrainingOOM(RuntimeError):
    """One exact trajectory needs the automatic multi-GPU training path."""


def _exact_sharded_recovery_path(run_dir):
    return Path(run_dir) / _EXACT_SHARDED_RECOVERY_FILE


def _read_exact_sharded_recovery(run_dir):
    if run_dir is None:
        return None
    path = _exact_sharded_recovery_path(run_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        step = int(payload["step"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid exact long-context recovery marker: {path}") from error
    if step < 0:
        raise RuntimeError(
            f"invalid exact long-context recovery step {step}: {path}")
    return {**payload, "step": step, "path": str(path)}


def _infer_exact_sharded_recovery(run_dir):
    """Recognize a pre-fix exact-OOM crash without repeating its failed pass."""
    if run_dir is None:
        return None
    log_path = Path(run_dir) / "temirnal.log"
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4 * 1024 * 1024), os.SEEK_SET)
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    failure_offset = tail.rfind("exact long-context training failed for a ")
    if failure_offset < 0:
        return None
    steps = re.findall(r"\[step (\d+)\]", tail[:failure_offset])
    if not steps:
        return None
    step = int(steps[-1])
    step_dir = Path(run_dir) / f"step{step:02d}"
    summary = step_dir / f"step{step:02d}.summary.json"
    rollout_pattern = f"step{step:02d}_group*_rollout*.meta.json"
    if summary.is_file() or not any(step_dir.glob(rollout_pattern)):
        return None
    token_match = re.search(
        r"exact long-context training failed for a (\d+)-token rollout",
        tail[failure_offset:])
    reason = "inferred from the previous exact single-GPU OOM"
    if token_match is not None:
        reason += f" ({int(token_match.group(1))}-token rollout)"
    return {
        "version": 1,
        "step": step,
        "reason": reason,
    }


def _write_exact_sharded_recovery(run_dir, step, error, **extra):
    path = _exact_sharded_recovery_path(run_dir)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if int(existing.get("step", -1)) == int(step):
                payload.update(existing)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    payload.update({
        "version": 2,
        "step": int(step),
        "reason": str(error),
        "created_at": time.time(),
    })
    payload.update(extra)
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _clear_exact_sharded_recovery(run_dir):
    path = _exact_sharded_recovery_path(run_dir)
    overflow_path = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        filename = payload.get("overflow_gradient_file")
        if filename and Path(str(filename)).name == str(filename):
            overflow_path = Path(run_dir) / str(filename)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    if overflow_path is not None:
        try:
            overflow_path.unlink()
        except FileNotFoundError:
            pass


def _save_exact_spo_rs_overflow(
        run_dir, step, model, rank_stats, *, normal_train_seconds=0.0):
    """Durably retain completed replica gradients for selective recovery."""
    import torch
    from fast_distributed import trainable_parameter_signature

    deferred_keys = sorted({
        (int(key[0]), int(key[1]))
        for stats in rank_stats
        for key in stats.get("deferred_keys", ())
    })
    deferred_count = sum(
        int(stats.get("deferred_examples", 0)) for stats in rank_stats)
    if deferred_count != len(deferred_keys) or not deferred_keys:
        raise RuntimeError(
            "exact SPO-RS overflow keys are missing or duplicated: "
            f"count={deferred_count}, unique={len(deferred_keys)}")

    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    normal_stats = {
        "totals": {
            key: sum(float(stats["totals"][key]) for stats in rank_stats)
            for key in metric_keys
        },
        "max_ratio": max(float(stats["max_ratio"]) for stats in rank_stats),
        "max_prefix_ratio": max(
            float(stats["max_prefix_ratio"]) for stats in rank_stats),
        "entropy_examples": sum(
            int(stats["entropy_examples"]) for stats in rank_stats),
        "trained_examples": sum(
            int(stats["trained_examples"]) for stats in rank_stats),
        "backward_batches": sum(
            int(stats["backward_batches"]) for stats in rank_stats),
        "training_seconds": max(0.0, float(normal_train_seconds)),
    }
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        gradients[name] = gradient.detach().cpu().clone()

    filename = _EXACT_OVERFLOW_GRADIENT_FILE.format(step=int(step))
    destination = Path(run_dir) / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "version": 2,
        "step": int(step),
        "parameter_signature": trainable_parameter_signature(model),
        "deferred_keys": deferred_keys,
        "remaining_keys": deferred_keys,
        "completed_keys": [],
        # These keys need cooperative GPU memory: the legacy CPU fallback
        # failed, or the warm path deliberately bypassed it. Do not replay
        # a huge CPU activation graph when resuming this residual tail.
        "replica_offload_exhausted": True,
        "replica_checkpoint_groups_attempted": any(
            hasattr(module, "_ttt_group_checkpoint_min_tokens")
            for module in model.modules()),
        "normal_stats": normal_stats,
        "overflow_stats": {
            "totals": {key: 0.0 for key in metric_keys},
            "max_ratio": 0.0,
            "max_prefix_ratio": 0.0,
            "entropy_examples": 0,
            "trained_examples": 0,
            "backward_batches": 0,
            "training_seconds": 0.0,
        },
        "gradients": gradients,
    }
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    del gradients, payload
    marker = _write_exact_sharded_recovery(
        run_dir, step,
        f"{deferred_count} exact trajectory/trajectories require the "
        "multi-GPU overflow lane",
        overflow_gradient_file=filename,
        deferred_keys=[list(key) for key in deferred_keys],
        normal_trained_examples=int(normal_stats["trained_examples"]),
        replica_offload_exhausted=True,
        replica_checkpoint_groups_attempted=any(
            hasattr(module, "_ttt_group_checkpoint_min_tokens")
            for module in model.modules()),
    )
    return destination, marker, deferred_keys, normal_stats


def _load_exact_spo_rs_overflow(run_dir, step):
    """Load and validate a selectively deferred exact SPO-RS update."""
    import torch

    marker = _read_exact_sharded_recovery(run_dir)
    if marker is None or int(marker["step"]) != int(step):
        return None
    filename = marker.get("overflow_gradient_file")
    if not filename:
        return None
    filename = str(filename)
    if Path(filename).name != filename:
        raise RuntimeError(
            "invalid exact SPO-RS overflow gradient filename in recovery "
            "marker")
    path = Path(run_dir) / filename
    if not path.is_file():
        print(
            f"[warn] exact SPO-RS overflow gradients are missing ({path}); "
            "falling back to the complete exact model-sharded update",
            flush=True,
        )
        return None
    try:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        print(
            f"[warn] exact SPO-RS overflow gradients could not be loaded "
            f"({error}); falling back to the complete exact model-sharded "
            "update",
            flush=True,
        )
        return None
    if (not isinstance(payload, dict)
            or int(payload.get("version", -1)) not in (1, 2)
            or int(payload.get("step", -1)) != int(step)):
        print(
            f"[warn] invalid exact SPO-RS overflow gradient checkpoint "
            f"({path}); falling back to the complete exact model-sharded "
            "update",
            flush=True,
        )
        return None
    deferred_keys = sorted({
        (int(key[0]), int(key[1]))
        for key in payload.get("deferred_keys", ())
    })
    if not deferred_keys:
        print(
            f"[warn] exact SPO-RS overflow checkpoint has no deferred keys "
            f"({path}); falling back to the complete exact model-sharded "
            "update",
            flush=True,
        )
        return None
    if int(payload.get("version", 1)) == 1:
        payload["version"] = 2
        payload["remaining_keys"] = list(deferred_keys)
        payload["completed_keys"] = []
    remaining_keys = sorted({
        (int(key[0]), int(key[1]))
        for key in payload.get("remaining_keys", deferred_keys)
    })
    completed_keys = sorted({
        (int(key[0]), int(key[1]))
        for key in payload.get("completed_keys", ())
    })
    if (set(remaining_keys) & set(completed_keys)
            or set(remaining_keys) | set(completed_keys)
            != set(deferred_keys)):
        print(
            f"[warn] exact SPO-RS overflow progress keys are inconsistent "
            f"({path}); falling back to the complete exact model-sharded "
            "update",
            flush=True,
        )
        return None
    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    overflow_stats = payload.get("overflow_stats")
    if not isinstance(overflow_stats, dict):
        overflow_stats = {}
    overflow_totals = overflow_stats.get("totals")
    if not isinstance(overflow_totals, dict):
        overflow_totals = {}
    payload["deferred_keys"] = deferred_keys
    payload["remaining_keys"] = remaining_keys
    payload["completed_keys"] = completed_keys
    payload["overflow_stats"] = {
        "totals": {
            key: float(overflow_totals.get(key, 0.0))
            for key in metric_keys
        },
        "max_ratio": float(overflow_stats.get("max_ratio", 0.0)),
        "max_prefix_ratio": float(
            overflow_stats.get("max_prefix_ratio", 0.0)),
        "entropy_examples": int(
            overflow_stats.get("entropy_examples", 0)),
        "trained_examples": int(
            overflow_stats.get("trained_examples", len(completed_keys))),
        "backward_batches": int(
            overflow_stats.get("backward_batches", len(completed_keys))),
        "training_seconds": max(
            0.0, float(overflow_stats.get("training_seconds", 0.0))),
    }
    if payload["overflow_stats"]["trained_examples"] != len(completed_keys):
        print(
            f"[warn] exact SPO-RS overflow progress count is inconsistent "
            f"({path}); falling back to the complete exact model-sharded "
            "update",
            flush=True,
        )
        return None
    payload["path"] = str(path)
    return payload


def _write_exact_spo_rs_overflow_state(overflow_state):
    """Atomically commit one exact-overflow progress checkpoint."""
    import torch

    path_value = overflow_state.get("path")
    if not path_value:
        raise RuntimeError(
            "exact SPO-RS overflow state has no durable checkpoint path")
    destination = Path(str(path_value))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = dict(overflow_state)
    payload.pop("path", None)
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    try:
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # The file itself is already flushed and atomically installed. Some
        # filesystems do not support fsync on a directory descriptor.
        pass


def _commit_exact_spo_rs_overflow_batch(
        model, overflow_state, step_idx, trajectory_keys, batch_stats,
        *, training_seconds):
    """Merge one exact overflow batch and make its progress restart-durable.

    The base replica gradients and every previously completed oversized
    trajectory remain in the CPU checkpoint. ``model.grad`` is the already
    reduced sum for exactly ``trajectory_keys``. The optimizer is deliberately
    untouched until *all* trajectories have been merged, preserving the one
    global clipped SPO-RS update exactly while allowing a replicated recovery
    wave to commit many trajectories at once.
    """
    import torch

    saved_gradients = _validate_precomputed_exact_gradients(
        model, overflow_state, step_idx)
    trajectory_keys = {
        tuple(int(value) for value in key) for key in trajectory_keys
    }
    if not trajectory_keys:
        return len(overflow_state.get("remaining_keys", ()))
    remaining = {
        tuple(int(value) for value in key)
        for key in overflow_state.get("remaining_keys", ())
    }
    completed = {
        tuple(int(value) for value in key)
        for key in overflow_state.get("completed_keys", ())
    }
    invalid = trajectory_keys - remaining
    if invalid or trajectory_keys & completed:
        raise RuntimeError(
            "exact SPO-RS overflow batch progress is duplicated or out of "
            f"order: {sorted(invalid or (trajectory_keys & completed))}")

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None:
                continue
            saved = saved_gradients[name]
            saved.add_(gradient.detach().to(
                device="cpu", dtype=saved.dtype))

    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    overflow_stats = overflow_state["overflow_stats"]
    totals = batch_stats["totals"]
    for key in metric_keys:
        overflow_stats["totals"][key] += float(totals[key])
    overflow_stats["max_ratio"] = max(
        float(overflow_stats["max_ratio"]),
        float(batch_stats["max_ratio"]))
    overflow_stats["max_prefix_ratio"] = max(
        float(overflow_stats["max_prefix_ratio"]),
        float(batch_stats["max_prefix_ratio"]))
    overflow_stats["entropy_examples"] += int(
        batch_stats["entropy_examples"])
    overflow_stats["trained_examples"] += len(trajectory_keys)
    overflow_stats["backward_batches"] += int(
        batch_stats.get("backward_batches", len(trajectory_keys)))
    overflow_stats["training_seconds"] += max(
        0.0, float(training_seconds))

    remaining.difference_update(trajectory_keys)
    completed.update(trajectory_keys)
    overflow_state["version"] = 2
    overflow_state["remaining_keys"] = sorted(remaining)
    overflow_state["completed_keys"] = sorted(completed)
    _write_exact_spo_rs_overflow_state(overflow_state)
    try:
        marker = _read_exact_sharded_recovery(
            Path(str(overflow_state["path"])).parent) or {}
        _write_exact_sharded_recovery(
            Path(str(overflow_state["path"])).parent,
            step_idx,
            marker.get(
                "reason",
                "selective exact SPO-RS overflow recovery in progress"),
            remaining_deferred_keys=[list(key) for key in sorted(remaining)],
            completed_deferred_keys=[list(key) for key in sorted(completed)],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # The authoritative gradient checkpoint was already atomically
        # committed. A stale human-readable marker does not invalidate it.
        print(
            f"[warn] exact overflow progress marker could not be refreshed: "
            f"{error}",
            flush=True,
        )
    return len(remaining)


def _commit_exact_spo_rs_overflow_progress(
        model, overflow_state, step_idx, trajectory_key, trajectory_stats,
        *, training_seconds):
    """Merge one finished oversized gradient and make it restart-durable."""
    return _commit_exact_spo_rs_overflow_batch(
        model, overflow_state, step_idx, [trajectory_key], trajectory_stats,
        training_seconds=training_seconds)


def _clip_sharded_grad_norm_(parameters, max_norm, *, error_if_nonfinite):
    """Clip one model-sharded gradient using one exact global L2 norm.

    ``torch.nn.utils.clip_grad_norm_`` is backend/version dependent when its
    input spans several CUDA devices.  Compute one FP32 norm per parameter on
    that parameter's own device, combine the scalar squares in host FP64, then
    apply the same global coefficient to every shard.
    """
    import torch

    parameters = [parameter for parameter in parameters
                  if parameter.grad is not None]
    squared_norms = []
    for parameter in parameters:
        gradient = parameter.grad.detach()
        local_norm = torch.linalg.vector_norm(
            gradient, ord=2, dtype=torch.float32)
        value = float(local_norm.item())
        squared_norms.append(value * value)
    total_norm = math.sqrt(math.fsum(squared_norms))
    if error_if_nonfinite and not math.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order 2.0 for gradients is non-finite, "
            f"so it cannot be clipped. Total norm: {total_norm}")
    coefficient = min(1.0, float(max_norm) / (total_norm + 1e-6))
    if coefficient < 1.0:
        with torch.no_grad():
            for parameter in parameters:
                parameter.grad.mul_(coefficient)
    return total_norm


def _resume_process_argv(run_dir):
    """Rebuild this Python invocation with exactly one authoritative resume."""
    original = list(sys.argv[1:])
    filtered = []
    index = 0
    while index < len(original):
        argument = original[index]
        if argument in {"--resume", "--resume-from"}:
            index += 2
            continue
        if (argument.startswith("--resume=")
                or argument.startswith("--resume-from=")):
            index += 1
            continue
        filtered.append(argument)
        index += 1
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *filtered,
        "--resume",
        str(Path(run_dir).resolve()),
    ]


def _restart_for_exact_resume(run_dir):
    """Replace the current process after every child/GPU has been stopped."""
    argv = _resume_process_argv(run_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(sys.executable, argv, dict(os.environ))


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


def _strategy_model_name(cfg):
    """Keep the configured local checkpoint available when the API is off."""
    return getattr(cfg, "api_strategy", "") or cfg.strategy_model_name


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
            "reasoning_effort": "high",
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
            "reasoning_effort": "medium",
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
    """Use the configured coder effort uniformly in every rollout phase."""
    return getattr(cfg, "coder_reasoning_effort", None)


def _coder_effort_entries(
        prompt_jobs, source_job_indices, counts, cfg, step_idx, phase):
    """Route the configured number of Qwen3.8 pilots through xhigh effort.

    The split changes only the chat-template reasoning control. Every source
    keeps exactly its requested sample count, allocation identity, and pilot
    status. All non-pilot work and all other coder models retain the run's
    configured effort unchanged.
    """
    del prompt_jobs, step_idx
    base_effort = _coder_effort_for_rollout_phase(cfg, phase)
    entries = [
        (int(source_idx), int(count), base_effort)
        for source_idx, count in zip(source_job_indices, counts)
        if int(count) > 0
    ]
    xhigh_per_strategy = int(getattr(
        cfg, "pilot_xhigh_programs_per_strategy", 0))
    if (str(phase) != "pilot" or xhigh_per_strategy <= 0
            or getattr(cfg, "coder_template_kind", "generic") != "qwen3.8"):
        return entries

    xhigh_entries = []
    base_entries = []
    for source_idx, count, _effort in entries:
        xhigh_count = min(xhigh_per_strategy, count)
        xhigh_entries.append((source_idx, xhigh_count, "xhigh"))
        if count > xhigh_count:
            base_entries.append(
                (source_idx, count - xhigh_count, base_effort))
    # Keep all xhigh variants in the same scheduler call and place those groups
    # first so round-robin assignment spreads the longer requests across the
    # available engines. This adds no rollout or allocation budget.
    return xhigh_entries + base_entries


def _phase_coder_prompt_job(
        prompt_jobs, source_idx, phase, cfg, render, cache, *, effort=None):
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
            "prompt_text": render(
                source["messages"], rollout_phase=phase,
                reasoning_effort=effort),
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

    # The model profile leaves memory outside vLLM's KV cache for CUDA/NCCL
    # and the bounded exact-token projection. The layout planner also accounts
    # for startup weight conversion before choosing the smallest fitting TP
    # group. Qwen3.8 needs a larger projection reserve than the other profiles.
    _profile_default(
        merged, "vllm_gpu_memory_utilization", 0.965, explicit_keys)
    _profile_default(merged, "vllm_quantization", "", explicit_keys)
    _profile_default(merged, "gen_micro_batch", "auto", explicit_keys)
    _profile_default(
        merged, "vllm_max_num_batched_tokens", "auto", explicit_keys)
    merged["vllm_runtime_reserve_override_gib"] = float(
        profile["vllm_runtime_reserve_gib"])
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
    reasoning_label = merged.get(
        "coder_reasoning_effort", "thinking on/off only")
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


def _detached_behavior_importance_ratio(cfg, current_logprobs,
                                        behavior_logprobs):
    """Return the existing token ratios or the A3B trajectory ratio.

    Entropic updates use a detached importance weight rather
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
    from strategy_prompts_openai import normalize_strategy_plan
    raw = final_answer_scope(normalize_strategy_plan(response_text)).strip()
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

    # The default deliberately preserves the original block-only contract:
    # accept the final complete, non-empty block, even when it is terse or
    # contains code-like/placeholder text.  Strict mode adds semantic format
    # checks, but is opt-in because those checks can create extra retries and
    # replace an otherwise usable strategy with the direct-solve fallback.
    if not strict:
        match = matches[-1]
        trailing = final_text[match.end():]
        if re.search(r"<strategy\b", trailing, flags=re.IGNORECASE):
            return _STRATEGY_FALLBACK, "unclosed final strategy block"
        body = match.group(1).strip()
        if not body:
            return _STRATEGY_FALLBACK, "empty final strategy"
        return body, None

    last_issue = "missing complete final strategy"
    for match in reversed(matches):
        body = match.group(1).strip()
        if not body:
            last_issue = "empty final strategy"
            continue
        if len(body) < 80:
            last_issue = "missing complete final strategy"
            continue
        # The strategy contract forbids executable source. A bare fence used
        # only to display a formula is harmless; an explicitly Python/code-like
        # fence means the planner attempted the coder's job and is retried.
        if (re.search(r"```\s*python\b", body, flags=re.IGNORECASE)
                or re.search(
                    r"```[^`]*(?:^|\n)\s*(?:def |class |from \S+ import |import \S+)",
                    body, flags=re.IGNORECASE | re.DOTALL)):
            last_issue = "strategy contained a code fence"
            continue
        # Reject explicit stand-ins for omitted work. They satisfy the XML
        # syntax but hand the coder an intentionally incomplete plan.
        if re.search(
                r"<\s*\.\.\.\s*>|\brest of (?:the )?(?:long )?strategy\b|"
                r"\bcontinue(?:d)? (?:in|with) (?:the )?next response\b|"
                r"\bplaceholder implementation\b|"
                r"\bcannot (?:provide|deliver|write) (?:the )?(?:full|complete)\b",
                body, flags=re.IGNORECASE):
            last_issue = "incomplete final strategy"
            continue
        return body, None
    return _STRATEGY_FALLBACK, last_issue


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
        raise ValueError("saved run uses an unsupported training objective; "
                         "only entropic and spo-rs can be resumed")
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


def compute_group_advantages(rewards_np):
    """Return entropic group advantages and their adaptive beta.

    SPO-RS obtains advantages from its frozen historical tracker instead.
    """
    from advantage import entropic_adaptive_advantages
    return entropic_adaptive_advantages(rewards_np)


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
    if (isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer))
            or epochs < 1):
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
    from spo_rs import BOUNDED_SHAPING_VERSION
    # The old adaptive-beta estimator did not have a historical baseline.
    merged["spo_rs_reward_shaping"] = (
        BOUNDED_SHAPING_VERSION if adaptive_beta else "exponential")
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
    p.add_argument("--num-steps", type=int, default=None,
                   help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=None,
                   help='Parent states per step.')
    p.add_argument("--group-size", type=int, default=None,
                   help='Rollouts per parent group.')
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
    p.add_argument("--strategy-thinking", dest="strategy_thinking",
                   action="store_const", const=True, default=None)
    p.add_argument("--no-strategy-thinking", dest="strategy_thinking",
                   action="store_const", const=False)
    p.add_argument("--strategy-reasoning-effort",
                   choices=["none", "low", "medium", "high", "xhigh", "max"],
                   default=None,
                   help="API reasoning effort; local GPT-OSS supports low, "
                        "medium, or high.")
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
        saved = _normalize_saved_advantage_config(
            json.loads(saved_config_path.read_text()))

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
        # A run saved before adaptive pilots existed must resume its original
        # one-pass rollout schedule even if today's YAML enables pilots.
        if "pilot_programs_per_strategy" not in saved:
            merged["pilot_programs_per_strategy"] = -1
        # Runs created before selectable phase-2 allocation used the original
        # median-quadrant rule.  Do not silently switch a resumed run to a new
        # allocator merely because its current problem YAML selects one.
        if "phase2_allocation_method" not in saved:
            merged["phase2_allocation_method"] = "rule_based"
        merged["spo_rs_adaptive_beta"] = bool(
            saved.get("spo_rs_adaptive_beta", False))
        # Introducing an API-enabled preset must not switch an older resumed
        # run to a paid service. An explicit CLI selector can still opt in.
        merged["api_strategy"] = saved.get("api_strategy", "")
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
    if (saved and args.spo_rs_adaptive_beta is not None
            and bool(saved.get("spo_rs_adaptive_beta", False))
            != bool(args.spo_rs_adaptive_beta)):
        raise ValueError(
            "cannot change fixed/adaptive SPO beta mode while resuming; "
            "start a fresh run for a different objective")
    if saved:
        from spo_rs import validate_shaping_resume
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
    if (saved.get("api_strategy") and args.api_strategy == ""
            and args.strategy_backend is None):
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
        completed_summary = (
            Path(resume_dir) / f"step{recovery_step:02d}"
            / f"step{recovery_step:02d}.summary.json")
        if completed_summary.is_file():
            _clear_exact_sharded_recovery(resume_dir)
            exact_sharded_recovery = None
            print(
                "[recovery] removed a stale exact sharded marker for an "
                "already checkpointed step")
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
        selective_overflow = bool(
            exact_sharded_recovery.get("overflow_gradient_file"))
        replicated_layout = bool(
            merged.get("fast", False)
            and str(merged.get("training_layout", "")).lower()
            == "replicated")
        from exact_overflow import supports_warm_overflow
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
            and not exact_sharded_recovery.get(
                "replica_offload_exhausted", False))
        can_capture_selectively = bool(
            not selective_overflow
            and replicated_layout)
        if warm_recovery:
            merged["_exact_inprocess_recovery_step"] = int(
                exact_sharded_recovery["step"])
            print("[recovery] saved overflow gradients will finish in-process; "
                  "Qwen checkpoint groups try all GPU replicas before "
                  "cooperative overflow when the saved work predates that mode")
        elif can_recover_on_replicas:
            merged["_exact_replica_recovery_step"] = int(
                exact_sharded_recovery["step"])
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
            merged["_exact_replica_capture_step"] = int(
                exact_sharded_recovery["step"])
            failed_length = re.search(
                r"(\d+)-token rollout",
                str(exact_sharded_recovery.get("reason", "")))
            if (failed_length is not None
                    and merged.get("coder_model_profile") != "qwen3.8-27b"):
                # Token sidecars are exact for current runs; the tiny margin
                # also covers legacy text recovery restoring an EOS/control
                # token without replaying the already-proven OOM workload.
                merged["_exact_replica_overflow_min_tokens"] = max(
                    1, int(failed_length.group(1)) - 16)
            print(
                "[recovery] legacy exact-OOM step will replay only adapter "
                "backward on the fast replicas; generation and evaluation "
                "remain skipped, and any oversized trajectories will be "
                "captured for selective model-sharded overflow")
        else:
            merged["training_layout"] = "sharded"
            merged["fast"] = False
            merged["_exact_sharded_recovery_step"] = int(
                exact_sharded_recovery["step"])
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
    pilot_xhigh_programs_per_strategy = merged.get(
        "pilot_xhigh_programs_per_strategy", 0)
    if (isinstance(pilot_xhigh_programs_per_strategy, bool)
            or not isinstance(pilot_xhigh_programs_per_strategy, int)
            or pilot_xhigh_programs_per_strategy < 0):
        raise ValueError(
            "pilot_xhigh_programs_per_strategy must be a non-negative integer")
    merged["pilot_xhigh_programs_per_strategy"] = (
        pilot_xhigh_programs_per_strategy)
    strategy_format_max_retries = merged.get("strategy_format_max_retries", 3)
    if (isinstance(strategy_format_max_retries, bool)
            or not isinstance(strategy_format_max_retries, int)
            or strategy_format_max_retries < 0):
        raise ValueError("strategy_format_max_retries must be a non-negative integer")
    merged["strategy_format_max_retries"] = strategy_format_max_retries
    strict_strategy_validation = merged.get(
        "strict_strategy_validation", False)
    if not isinstance(strict_strategy_validation, bool):
        raise ValueError("strict_strategy_validation must be a boolean")
    merged["strict_strategy_validation"] = strict_strategy_validation
    if merged["strategies"]:
        strategy_model_name = str(
            merged.get("strategy_model_name") or merged["model_name"]
        ).strip()
        if not merged["api_strategy"]:
            _apply_strategy_model_profile(
                merged, strategy_model_name, profile_explicit_keys)
        strategy_backend = "api" if merged["api_strategy"] else str(
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
        if merged["api_strategy"]:
            from strategy_api import OPENAI_MAX_OUTPUT_TOKENS
            if strategy_max_new_tokens > OPENAI_MAX_OUTPUT_TOKENS:
                print(f"[config] OpenAI strategy output budget capped at "
                      f"{OPENAI_MAX_OUTPUT_TOKENS} tokens (including reasoning)")
                strategy_max_new_tokens = OPENAI_MAX_OUTPUT_TOKENS
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
        if (pilot_xhigh_programs_per_strategy > 0
                and pilot_programs_per_strategy < 1):
            raise ValueError(
                "positive pilot_xhigh_programs_per_strategy requires "
                "adaptive pilots to be enabled")
        if (pilot_programs_per_strategy > 0
                and pilot_xhigh_programs_per_strategy
                > pilot_programs_per_strategy):
            raise ValueError(
                "pilot_xhigh_programs_per_strategy cannot exceed "
                "pilot_programs_per_strategy")
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
        if strategy_reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(
                "strategy_reasoning_effort must be none, low, medium, high, xhigh, or max")
        if (strategy_backend == "local" and "gpt-oss" in strategy_model_name.lower()
                and strategy_reasoning_effort not in {"low", "medium", "high"}):
            raise ValueError("local GPT-OSS strategy reasoning supports low, medium, or high")
        merged["strategy_reasoning_effort"] = strategy_reasoning_effort
        merged["strategy_vllm_quantization"] = str(merged.get('strategy_vllm_quantization') or '').strip()
        merged["strategy_api_base_url"] = str(
            merged.get("strategy_api_base_url")
            or "https://api.deepseek.com").strip().rstrip("/")
        merged["strategy_api_key_env"] = str(
            merged.get("strategy_api_key_env")
            or "DEEPSEEK_API_KEY").strip()
        if merged["api_strategy"]:
            merged["strategy_api_base_url"] = "https://api.openai.com/v1"
            merged["strategy_api_key_env"] = "OPENAI_API_KEY"
        merged["strategy_api_concurrency"] = int(
            merged.get("strategy_api_concurrency") or merged["groups_per_step"])
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
        if strategy_backend == "api":
            from strategy_api import strategy_api_key
            strategy_api_key(merged["strategy_api_key_env"])
        merged["group_size"] = effective_group_size
    # This ablation intentionally keeps two independent Qwen3-8B engines per
    # card resident together: a permanently base/no-LoRA strategist and a
    # coder that receives the current LoRA adapter. Keep the switch derived
    # rather than user-configurable so every other model pairing retains the
    # existing alternating-pool behavior.
    merged["dual_resident_qwen3_8b_strategy_coder_pools"] = (
        _dual_resident_qwen3_8b_strategy_coder_pools(merged))
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
        if not str(inventory_value or '').strip():
            inventory_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not str(inventory_value or '').strip():
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
        if not str(merged.get('gpu_type') or '').strip():
            merged["gpu_type"] = "H100"
            print("[config] gpu_mode gpu_type not set; defaulting explicitly to H100")
        merged["kernel_gpu_id"] = evaluation_gpu_id
        if int(merged.get('reward_workers') or 0) != 1:
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
            _uses_clipped_policy_loss(merged['advantage_mode'])
        )
        if (fast_clipped_policy
                and int(merged["spo_rs_update_epochs"]) != 1):
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
        policy_cfg = SimpleNamespace(**vars(cfg))
        policy_cfg.temperature = 1.0
        policy_cfg.top_p = 1.0
        policy_cfg.sampling_top_k = 0
        policy_cfg.sampling_min_p = 0.0
        if max_new_tokens is not None:
            policy_cfg.max_new_tokens = int(max_new_tokens)
        for group_idx, (prompt, count) in enumerate(zip(prompts_by_group, counts_by_group)):
            if int(count) > 0:
                responses, _ = generate_responses(
                    model, tokenizer, prompt, int(count), policy_cfg)
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
        "micro_batch": int(getattr(cfg, 'gen_micro_batch', 0) or 0),
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
    from training_checkpointing import grouped_qwen_hidden
    hidden = (grouped_qwen_hidden(decoder, input_ids, attention_mask)
              if with_grad else None)
    if hidden is None:
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


def _training_microbatches(examples, cfg):
    """Length-bucket examples into real model batches.

    train_examples_per_microbatch is an example-count ceiling.  The padded-token
    ceiling keeps one batch at or below max_seq_length total padded tokens, so a
    configured value such as 64 is useful for short responses without forcing
    64 long contexts into memory together.
    """
    example_cap = int(getattr(cfg, 'train_examples_per_microbatch', 1) or 0)
    if example_cap < 1:
        raise ValueError("train_examples_per_microbatch must be >= 1")
    token_cap = max(1, int(getattr(cfg, "max_seq_length", 1) or 1))

    batches = []
    ordered = sorted(
        examples,
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
        start_checkpointed=False, start_with_headroom=False,
        start_activation_offload=False,
        activation_offload_semaphore=None, defer_exact_oom=False,
        allow_activation_offload=True):
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
    outlier is quarantined only after every exact fallback has failed.  The
    strict SPO-RS replica path may instead return that singleton in the third
    result so its gradient can be computed by the exact multi-GPU overflow
    lane without discarding gradients already completed on other replicas.
    """
    import gc
    import torch
    from host_memory import (ProcessRSSPeakSampler,
                             host_cache_reclaim_supported,
                             process_rss_bytes,
                             reclaim_unused_host_memory)
    from model_backend import set_gradient_checkpointing

    # Saved-activation offload is safe in the replicated SPO-RS path only when
    # every process shares a bounded node-wide admission gate. Without that
    # gate, several 100K+ token graphs can consume host RAM concurrently and
    # OOM-kill the VM. The model-sharded fallback deliberately supplies no
    # gate and therefore continues to use GPU checkpointing/headroom only.
    if defer_exact_oom and activation_offload_semaphore is None:
        allow_activation_offload = False

    def offload_gate_can_admit(work):
        if activation_offload_semaphore is None:
            return True
        checker = getattr(
            activation_offload_semaphore, "can_admit", None)
        return bool(checker(work)) if callable(checker) else True

    def offload_admission(work):
        if activation_offload_semaphore is None:
            return nullcontext()
        reservation = getattr(
            activation_offload_semaphore, "reservation", None)
        if callable(reservation):
            return reservation(work)
        return activation_offload_semaphore

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
    expanded_fraction = None
    split_backoff = False
    offload_trigger_work = None
    headroom_trigger_work = None
    checkpoint_threshold = getattr(
        model, "_ttt_gradient_checkpointing_min_work", None)
    checkpointing_available = set_gradient_checkpointing(model, False)
    checkpointing = False

    def expand_to_next_memory_ceiling():
        """Request one strictly larger exact-training allocator ceiling."""
        nonlocal expanded_fraction
        if expand_memory is None:
            return None
        candidate = expand_memory()
        if candidate is None:
            return None
        candidate = float(candidate)
        if (expanded_fraction is not None
                and candidate <= float(expanded_fraction) + 1e-9):
            return None
        expanded_fraction = candidate
        return candidate

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
        memory_expanded = expand_to_next_memory_ceiling() is not None
    configured_offload_min_work = getattr(
        model, "_ttt_activation_offload_min_work", None)
    offload_min_work = profile.get(
        "activation_offload_min_work", configured_offload_min_work)
    if (allow_activation_offload
            and (bool(start_activation_offload)
                 or (offload_min_work is not None
                     and active_work >= int(offload_min_work)))
            and offload_gate_can_admit(active_work)
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
            # Pinned saved tensors make the GPU<->CPU replay substantially
            # faster, but PyTorch caches the released host blocks separately
            # from the CUDA device allocator.  Use pinned storage only when
            # this runtime exposes a way to return that cache to the OS.
            admission = (
                offload_admission(active_work)
                if activation_offload else nullcontext())
            # Keep the admission token until inactive pinned blocks and libc
            # arenas have been returned. Otherwise the next trainer can begin
            # a large offload while this process still owns the prior batch's
            # host cache, recreating the node-wide RAM spike.
            with admission:
                offload_rss_before = (
                    process_rss_bytes() if activation_offload else None)
                rss_sampler = (
                    ProcessRSSPeakSampler()
                    if activation_offload else nullcontext())
                offload_context = (
                    torch.autograd.graph.save_on_cpu(
                        pin_memory=host_cache_reclaim_supported(torch))
                    if activation_offload
                    and hasattr(torch.autograd.graph, "save_on_cpu")
                    else nullcontext()
                )
                try:
                    with rss_sampler as sampled_rss:
                        with offload_context:
                            result = attempt(active)
                except BaseException:
                    # Clean the failed graph before releasing the shared
                    # admission slot. Otherwise another rank can begin a CPU
                    # offload while this exception still retains saved
                    # tensors, recreating the node-wide host-RAM spike.
                    if activation_offload:
                        model.zero_grad(set_to_none=True)
                        gc.collect()
                        for cuda_device in cuda_devices:
                            with torch.cuda.device(cuda_device):
                                torch.cuda.empty_cache()
                        reclaim_unused_host_memory(
                            torch, cuda_devices=cuda_devices, force=True)
                    raise
                else:
                    if activation_offload:
                        cleanup = reclaim_unused_host_memory(
                            torch, cuda_devices=cuda_devices, force=True)
                        # Forced cleanup records the pinned allocator and RSS
                        # immediately before releasing the saved activations.
                        # Publish the larger measurement while this rank still
                        # owns its reservation; waiting ranks then recompute
                        # their token-weighted request before they can enter.
                        observer = getattr(
                            activation_offload_semaphore, "observe", None)
                        if callable(observer):
                            footprint = int(
                                cleanup.get("pinned_owned_before", 0) or 0)
                            rss_at_cleanup = cleanup.get("rss_before")
                            if (offload_rss_before is not None
                                    and rss_at_cleanup is not None):
                                footprint = max(
                                    footprint,
                                    max(0, int(rss_at_cleanup)
                                        - int(offload_rss_before)),
                                )
                            footprint = max(
                                footprint,
                                int(getattr(
                                    sampled_rss, "footprint_bytes", 0) or 0),
                            )
                            observer(active_work, footprint)
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
            if cuda_oom and expand_memory is not None:
                previous_fraction = expanded_fraction
                next_fraction = expand_to_next_memory_ceiling()
                if next_fraction is not None:
                    memory_expanded = True
                    headroom_trigger_work = active_work
                    retry_label = (
                        "singleton OOM"
                        if previous_fraction is None else
                        "singleton still OOM"
                    )
                    print(
                        f"[train-oom] {device_label}: {retry_label}; retrying "
                        "the exact update with "
                        f"reserved GPU headroom (allocator cap "
                        f"{100.0 * next_fraction:.1f}%)",
                        flush=True,
                    )
                    continue
            if (cuda_oom and allow_activation_offload
                    and not activation_offload
                    and offload_gate_can_admit(active_work)
                    and hasattr(
                    torch.autograd.graph, "save_on_cpu")):
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
                if defer_exact_oom:
                    quarantined.append(victim)
                    setting_log_only(
                        f"[train-overflow] {device_label}: deferring one "
                        f"{victim_tokens}-token rollout to the exact "
                        "multi-GPU overflow lane; completed replica "
                        "gradients remain valid",
                        flush=True,
                    )
                    # Continue with any other singleton batches in this local
                    # workload. ``attempt`` starts from zero after every OOM,
                    # while gradients from earlier work-queue items live in
                    # the caller's separate accumulators.
                    active_work = select_checkpointing()
                    continue
                set_gradient_checkpointing(model, False)
                model.zero_grad(set_to_none=True)
                if kernel_unavailable:
                    failure = (
                        "no compatible exact linear-memory attention kernel")
                elif allow_activation_offload:
                    failure = (
                        "insufficient memory after checkpointing, reserved "
                        "headroom, and CPU activation offload")
                else:
                    failure = (
                        "insufficient memory after checkpointing and all "
                        "reserved-headroom stages; CPU activation offload "
                        "was intentionally disabled for this host-RAM-safe "
                        "model-sharded lane")
                raise ExactLongContextTrainingOOM(
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


def _initialize_behavior_logprobs(examples):
    """Cache frozen rollout scores and return trajectories needing HF scoring."""
    missing = []
    for example in examples:
        if _valid_example_token_logprobs(example, "behavior_logprobs"):
            example["old_policy_logprobs"] = example["behavior_logprobs"].detach()
        else:
            example["old_policy_logprobs"] = None
            missing.append(example)
    return missing


def clipped_surrogate_loss(
        current_logprobs, old_logprobs, reference_logprob,
        advantage, *, clip_epsilon=SEQUENCE_CLIP_EPSILON_DEFAULT,
        clip_epsilon_low=None, clip_epsilon_high=None, kl_coef=0.0,
        sequence_level_ratio=False, return_tensor_metrics=False):
    """One trajectory's loss; callers average equally within each parent.

    Normally, per-token rho = exp(current_lp - old_lp) is differentiable. For
    the plain Qwen3-30B-A3B MoE coder, callers select the GSPO-style trajectory
    ratio exp(mean(current_lp - old_lp)); the same scalar ratio and asymmetric
    bounds are then used for the complete response. Old log-probabilities and
    the scalar group advantage are frozen. Policy/reference regularization uses
    the same centered log-probability correction as entropic mode:
        d = log pi_theta - log pi_ref
        A_KL = kl_coef * (mean(d) - d).
    Its policy-gradient surrogate is kept separate from the clipped PPO surrogate
    so SPO-RS advantages and asymmetric clipping retain their existing behavior.
    Clipping is asymmetric when requested: [1 - epsilon_low,
    1 + epsilon_high]. The legacy clip_epsilon remains the fallback for both.

    Accumulation and exponentiation use float64, and nonfinite ratios fail
    explicitly rather than silently clamping the objective.
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
    if not math.isfinite(float(kl_coef)) or kl_coef < 0.0:
        raise ValueError("KL coefficient must be finite and nonnegative")
    cur = current_logprobs.to(dtype=torch.float64)
    old = torch.as_tensor(old_logprobs, dtype=cur.dtype, device=cur.device).detach()
    if cur.ndim != 1 or old.shape != cur.shape or cur.numel() == 0:
        raise ValueError("current and old logprobs must be matching nonempty vectors")
    adv = torch.as_tensor(advantage, dtype=cur.dtype, device=cur.device).detach()
    if adv.numel() != 1 or not torch.isfinite(adv).all():
        raise ValueError("policy advantage must be a finite scalar")
    if not torch.isfinite(cur).all() or not torch.isfinite(old).all():
        raise ValueError("policy logprobs must be finite")

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
    loss = policy_loss + kl_policy_loss
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
def _policy_dropout_disabled(model):
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


def _clipped_policy_options(cfg):
    """Return PPO epsilon, lower epsilon, upper epsilon, and reference-KL coef."""
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
    epsilon = SEQUENCE_CLIP_EPSILON_DEFAULT
    epsilon_low = float(getattr(cfg, "sequence_clip_epsilon_low", epsilon))
    epsilon_high = float(getattr(cfg, "sequence_clip_epsilon_high", epsilon))
    return epsilon, epsilon_low, epsilon_high, float(cfg.kl_penalty_coef)


def clipped_policy_loss(cfg, current_logprobs, old_logprobs,
                       reference_logprob, advantage, *,
                       clip_epsilon, clip_epsilon_low,
                       clip_epsilon_high, kl_coef=0.0,
                       return_tensor_metrics=False):
    """Dispatch the shared fast trainer to the active clipped objective."""
    return clipped_surrogate_loss(
        current_logprobs, old_logprobs, reference_logprob, advantage,
        clip_epsilon=clip_epsilon,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        kl_coef=kl_coef,
        sequence_level_ratio=_uses_sequence_level_policy_ratio(cfg),
        return_tensor_metrics=return_tensor_metrics)


def _a3b_sequence_clipped_standard_loss(
        cfg, current_logprobs, behavior_logprobs, reference_logprobs,
        advantage):
    """Apply the A3B sequence surrogate inside entropic training.

    Entropic training retains its existing advantages, centered
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
    advantages, _beta = compute_group_advantages(rewards)
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
            "group_id": int(group_id),
            "rollout_index": int(rollout_index),
            "prompt_job_id": job_idx,
            "sample_weight": 1.0 / (int(num_groups) * len(responses)),
            "_overlap_key": (int(group_id), int(rollout_index)),
        }
        record["_entropic_overlap_example"] = example
        prepared.append(example)
    return prepared


def _prepare_spo_rs_overlap_inputs(
        tokenizer, group_responses, reward_futures, prompt_jobs):
    """Build every reward-independent SPO-RS training input on CPU.

    SPO-RS uses rollout-policy logprobs produced by vLLM and a frozen
    pre-step tracker baseline.  Prompt/response tensors, behavior scores, the
    exact step-wide trajectory weight, and scheduling keys therefore exist
    before sandbox rewards finish.  The reward future stays only on the local
    staging object and is removed before the example enters a multiprocessing
    queue.
    """
    import torch

    prompt_ids_by_job = {}
    prepared = []
    for group_id in sorted(group_responses):
        responses = group_responses[group_id]
        futures = reward_futures[group_id]
        if len(responses) != len(futures):
            raise RuntimeError(
                f"SPO-RS overlap group {group_id} has {len(responses)} "
                f"responses but {len(futures)} reward futures")
        for rollout_index, (record, reward_future) in enumerate(
                zip(responses, futures)):
            token_ids = list(record.get("token_ids") or ())
            behavior_values = record.get("behavior_logprobs")
            if not token_ids or behavior_values is None:
                continue
            try:
                behavior_array = np.asarray(
                    behavior_values, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if (behavior_array.ndim != 1
                    or behavior_array.size != len(token_ids)
                    or not np.isfinite(behavior_array).all()):
                continue

            job_idx = int(record["job_idx"])
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.cpu()
            example = {
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": torch.tensor(
                    [token_ids], dtype=torch.long, device="cpu"),
                "_entropy_measurement": record.get("_entropy_measurement"),
                "advantage": None,
                "behavior_logprobs": torch.as_tensor(
                    behavior_array, dtype=torch.float32).cpu(),
                "reference_logprobs": None,
                "reward_constant": False,
                "group_id": int(group_id),
                "rollout_index": int(rollout_index),
                "prompt_job_id": job_idx,
                "sample_weight": 0.0,
                "_overlap_key": (int(group_id), int(rollout_index)),
                "_overlap_reward_future": reward_future,
            }
            record["_spo_rs_overlap_example"] = example
            prepared.append(example)

    if prepared:
        trajectory_weight = 1.0 / len(prepared)
        for example in prepared:
            example["sample_weight"] = trajectory_weight
    return prepared


def _complete_spo_rs_overlap_examples(examples, spo_rs_tracker=None):
    """Attach historical SPO advantages as individual CPU rewards finish."""
    if not examples:
        return []
    futures = [example.get("_overlap_reward_future") for example in examples]
    if any(future is None or not future.done() for future in futures):
        raise RuntimeError(
            "SPO-RS overlap received an example before its reward completed")
    if spo_rs_tracker is None:
        raise RuntimeError("SPO-RS overlap requires its historical tracker")
    rewards = [float(future.result().reward) for future in futures]
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    advantages = spo_rs_tracker.normalized_advantages(transformed)
    if advantages is None:
        raise RuntimeError(
            "SPO-RS overlap lost its frozen pre-step tracker baseline")
    for example, advantage in zip(examples, advantages):
        example["advantage"] = float(advantage)
        # Futures cannot be sent through the process trainer's work queues.
        example.pop("_overlap_reward_future", None)
    return examples


def _validate_precomputed_exact_gradients(model, overflow_state, step_idx):
    """Validate and return a selective-overflow gradient mapping."""
    import torch
    from fast_distributed import trainable_parameter_signature

    if int(overflow_state.get("step", -1)) != int(step_idx):
        raise RuntimeError(
            "exact SPO-RS overflow gradients belong to a different step")

    def canonical_signature(value):
        return tuple(
            (str(name), tuple(int(size) for size in shape), str(dtype))
            for name, shape, dtype in value
        )

    saved_signature = canonical_signature(
        overflow_state.get("parameter_signature", ()))
    current_signature = canonical_signature(
        trainable_parameter_signature(model))
    if saved_signature != current_signature:
        raise RuntimeError(
            "exact SPO-RS overflow trainable-parameter layout changed; "
            "refusing to combine mismatched gradients")
    saved_gradients = overflow_state.get("gradients")
    if not isinstance(saved_gradients, dict):
        raise RuntimeError(
            "exact SPO-RS overflow checkpoint has no gradient mapping")
    expected_names = {name for name, _shape, _dtype in current_signature}
    if set(saved_gradients) != expected_names:
        raise RuntimeError(
            "exact SPO-RS overflow gradient names do not match the loaded "
            "adapter")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        saved = saved_gradients[name]
        if (not torch.is_tensor(saved)
                or tuple(saved.shape) != tuple(parameter.shape)
                or not torch.isfinite(saved).all()):
            raise RuntimeError(
                f"invalid saved exact SPO-RS gradient for {name}")
    return saved_gradients


def _apply_precomputed_exact_gradients(model, overflow_state, step_idx):
    """Add durable fast-path gradients to the model-sharded overflow pass."""
    import torch

    saved_gradients = _validate_precomputed_exact_gradients(
        model, overflow_state, step_idx)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            saved = saved_gradients[name]
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(saved.to(
                device=parameter.device, dtype=parameter.grad.dtype))


def _train_durable_exact_spo_rs_overflow(
        backend, model, tokenizer, optimizer, examples, cfg, step_idx,
        overflow_state, run_dir):
    """Finish selectively deferred SPO-RS trajectories one durable unit at a time."""
    import gc
    import torch
    from fast_distributed import set_long_rollout_memory_ceiling
    from host_memory import reclaim_unused_host_memory

    checkpoint_path = Path(str(overflow_state.get("path", ""))).resolve()
    expected_path = (
        Path(run_dir).resolve()
        / _EXACT_OVERFLOW_GRADIENT_FILE.format(step=int(step_idx)))
    if checkpoint_path != expected_path:
        raise RuntimeError(
            "exact SPO-RS overflow checkpoint is outside the active run or "
            "belongs to a different step")

    epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
    if epochs != 1:
        raise RuntimeError(
            "selective exact SPO-RS overflow requires one optimizer update")
    epsilon, epsilon_low, epsilon_high, kl_coef = (
        _clipped_policy_options(cfg))
    backend.set_training_mode()
    cuda_devices = sorted({
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
        and parameter.device.index is not None
    })
    if not cuda_devices:
        raise RuntimeError(
            "exact model-sharded overflow recovery found no CUDA shards")

    original_keys = {
        tuple(int(value) for value in key)
        for key in overflow_state.get("deferred_keys", ())
    }
    remaining_keys = {
        tuple(int(value) for value in key)
        for key in overflow_state.get("remaining_keys", ())
    }
    completed_before = {
        tuple(int(value) for value in key)
        for key in overflow_state.get("completed_keys", ())
    }
    example_by_key = {
        tuple(int(value) for value in example["_overlap_key"]): example
        for example in examples
    }
    if set(example_by_key) != remaining_keys:
        raise RuntimeError(
            "exact SPO-RS overflow examples do not match the durable "
            f"remaining keys: examples={sorted(example_by_key)}, "
            f"remaining={sorted(remaining_keys)}")

    missing_old = _initialize_behavior_logprobs(examples)
    print(
        f"[step {step_idx}] {cfg.advantage_mode} overflow logprobs: vLLM "
        f"supplied old={len(examples) - len(missing_old)}/{len(examples)}, "
        f"reference=off; "
        f"HF fallback old={len(missing_old)}, reference="
        f"off",
        flush=True,
    )

    with _policy_dropout_disabled(model):
        for batch in ([example] for example in missing_old):
            old_logprobs = compute_batched_token_logprobs(
                model, batch, with_grad=False, chunk=cfg.logprob_chunk,
                pad_token_id=tokenizer.pad_token_id)
            if any(not torch.isfinite(value).all()
                   for value in old_logprobs):
                raise FloatingPointError("nonfinite old-policy logprobs")
            for example, old_lp in zip(batch, old_logprobs):
                example["old_policy_logprobs"] = old_lp.detach()

        # The durable checkpoint already contains the fast-replica gradient and
        # any oversized trajectories completed by an earlier recovery process.
        # Commit after every trajectory, but keep the sharded model resident
        # and continue through the remaining work in this process. This path
        # never uses CPU activation offload, and the cleanup below releases the
        # completed graph and CUDA caches before the next trajectory. Progress
        # therefore remains independently resumable without repeatedly loading
        # the complete checkpoint.
        ordered_examples = sorted(
            examples,
            key=lambda example: int(
                example["prompt_ids"].shape[1]
                + example["response_ids"].shape[1]),
            reverse=True,
        )
        for ordinal, example in enumerate(ordered_examples, start=1):
            key = tuple(int(value) for value in example["_overlap_key"])
            token_count = int(
                example["prompt_ids"].shape[1]
                + example["response_ids"].shape[1])
            print(
                f"[recovery] step {step_idx}: exact oversized trajectory "
                f"{ordinal}/{len(ordered_examples)} key={key}, "
                f"tokens={token_count}; starting model-sharded backward",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            trajectory_started = time.time()
            headroom_stages = iter((0.96, 0.98, 0.99))

            def expand_sharded_memory():
                fraction = next(headroom_stages, None)
                if fraction is None:
                    return None
                for cuda_device in cuda_devices:
                    set_long_rollout_memory_ceiling(
                        cuda_device, fraction)
                return fraction

            def attempt(active_batches):
                totals = {key_name: 0.0 for key_name in (
                    "loss", "policy_loss", "kl_estimate",
                    "entropy_estimate", "ratio", "clipped_fraction")}
                max_ratio = 0.0
                max_prefix_ratio = 0.0
                entropy_examples = 0
                for batch in active_batches:
                    result = compute_batched_token_logprobs(
                        model, batch, with_grad=True,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id,
                        measure_entropy=True)
                    current_logprobs = result
                    weighted_losses = []
                    for item, current_lp in zip(
                            batch, current_logprobs):
                        loss, metrics = clipped_policy_loss(
                            cfg, current_lp, item["old_policy_logprobs"],
                            None, item["advantage"],
                            clip_epsilon=epsilon,
                            clip_epsilon_low=epsilon_low,
                            clip_epsilon_high=epsilon_high,
                            kl_coef=kl_coef)
                        weight = float(item["sample_weight"])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError("nonfinite SPO-RS loss")
                        weighted_losses.append(weight * loss)
                        totals["loss"] += weight * float(loss.detach().item())
                        for metric_name in (
                                "policy_loss", "kl_estimate",
                                "entropy_estimate", "ratio"):
                            totals[metric_name] += weight * metrics[metric_name]
                        totals["clipped_fraction"] += (
                            weight * float(metrics["clipped"]))
                        max_ratio = max(max_ratio, metrics["ratio"])
                        max_prefix_ratio = max(
                            max_prefix_ratio, metrics["prefix_ratio_max"])
                    if weighted_losses:
                        sum(weighted_losses[1:],
                            weighted_losses[0]).backward()
                return {
                    "totals": totals,
                    "max_ratio": max_ratio,
                    "max_prefix_ratio": max_prefix_ratio,
                    "entropy_examples": entropy_examples,
                }

            result, effective, quarantined = _run_oom_resilient_backward(
                model, [[example]], attempt,
                device_label="model-sharded GPUs",
                expand_memory=expand_sharded_memory,
                start_checkpointed=True,
                start_with_headroom=True,
                # A single 100K+ trajectory can make save_on_cpu retain an
                # unbounded host graph and OOM-kill the complete VM. The model
                # is already spread over every GPU here, so use staged card
                # headroom and exact checkpointed kernels instead.
                allow_activation_offload=False)
            if (result is None or quarantined
                    or sum(len(batch) for batch in effective) != 1):
                raise RuntimeError(
                    f"exact overflow trajectory {key} did not complete")
            remaining_count = _commit_exact_spo_rs_overflow_progress(
                model, overflow_state, step_idx, key, result,
                training_seconds=time.time() - trajectory_started)
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            for cuda_device in cuda_devices:
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()
            reclaim_unused_host_memory(
                torch, cuda_devices=cuda_devices, force=True)
            print(
                f"[recovery] step {step_idx}: committed exact gradient for "
                f"key={key}; {remaining_count} oversized trajectory/"
                "trajectories remain",
                flush=True,
            )
    for example in examples:
        for cache_key in ("old_policy_logprobs",):
            example.pop(cache_key, None)
    return _finalize_exact_spo_rs_overflow(
        model, optimizer, cfg, step_idx, overflow_state,
        completed_before=completed_before)


def _finalize_exact_spo_rs_overflow(
        model, optimizer, cfg, step_idx, overflow_state, *,
        completed_before=(), label="durable exact overflow recovery"):
    """Clip and apply the one full-batch update, never a partial overflow update."""
    if overflow_state.get("remaining_keys"):
        raise RuntimeError("cannot update the policy with unfinished trajectories")
    original_keys = overflow_state.get("deferred_keys", ())
    # All sample gradients now live in one durable CPU checkpoint. Materialize
    # them on their owning parameter shards, clip the single global update once,
    # and take exactly one optimizer step.
    finalize_started = time.time()
    optimizer.zero_grad(set_to_none=True)
    _apply_precomputed_exact_gradients(model, overflow_state, step_idx)
    grad_norm = _clip_sharded_grad_norm_(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()

    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    normal_stats = overflow_state.get("normal_stats", {})
    overflow_stats = overflow_state.get("overflow_stats", {})
    normal_totals = normal_stats.get("totals", {})
    overflow_totals = overflow_stats.get("totals", {})
    totals = {
        key: (float(normal_totals.get(key, 0.0))
              + float(overflow_totals.get(key, 0.0)))
        for key in metric_keys
    }
    max_ratio = max(
        float(normal_stats.get("max_ratio", 0.0)),
        float(overflow_stats.get("max_ratio", 0.0)))
    max_prefix_ratio = max(
        float(normal_stats.get("max_prefix_ratio", 0.0)),
        float(overflow_stats.get("max_prefix_ratio", 0.0)))
    entropy_examples = (
        int(normal_stats.get("entropy_examples", 0))
        + int(overflow_stats.get("entropy_examples", 0)))
    history = [{
        "epoch": 1,
        **totals,
        "ratio_max": max_ratio,
        "prefix_ratio_max": max_prefix_ratio,
        "entropy_examples": entropy_examples,
        "oom_quarantined_examples": 0,
        "grad_norm": float(grad_norm),
    }]
    print(
        f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
        f"loss={totals['loss']:.6f} "
        f"reference=off "
        f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
        f"clipped={totals['clipped_fraction']:.1%} "
        f"entropy examples={entropy_examples} OOM-quarantined=0 "
        f"({label})",
        flush=True,
    )
    return {
        "policy_updates": history,
        "training_seconds": (
            float(normal_stats.get("training_seconds", 0.0))
            + float(overflow_stats.get("training_seconds", 0.0))
            + (time.time() - finalize_started)),
        "oom_quarantined_examples": 0,
        "exact_overflow_reused_examples": (
            int(normal_stats.get("trained_examples", 0))
            + len(completed_before)),
        "exact_overflow_sharded_examples": len(original_keys),
        "exact_overflow_resumed_examples": len(completed_before),
    }


def _train_spo_rs_examples(backend, model, tokenizer, optimizer, examples,
                           cfg, step_idx, *, precomputed_gradient_state=None,
                           exact_overflow_run_dir=None):
    """Freeze behavior logprobs, then optimize the SPO-RS clipped surrogate."""
    import torch

    if precomputed_gradient_state is not None:
        if exact_overflow_run_dir is None:
            raise RuntimeError(
                "selective exact SPO-RS overflow requires its run directory")
        return _train_durable_exact_spo_rs_overflow(
            backend, model, tokenizer, optimizer, examples, cfg, step_idx,
            precomputed_gradient_state, exact_overflow_run_dir)

    epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
    epsilon, epsilon_low, epsilon_high, kl_coef = (
        _clipped_policy_options(cfg))
    backend.set_training_mode()
    started = time.time()
    update_batches = _training_microbatches(examples, cfg)
    largest_batch = max((len(batch) for batch in update_batches), default=0)
    print(f"[train] LoRA microbatches={len(update_batches)}; "
          f"configured max={int(cfg.train_examples_per_microbatch)}, "
          f"effective max={largest_batch}, "
          f"padded-token cap={int(cfg.max_seq_length)}", flush=True)
    missing_old = _initialize_behavior_logprobs(examples)
    print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: vLLM supplied old="
          f"{len(examples) - len(missing_old)}/{len(examples)}, reference=off; "
          f"HF fallback old={len(missing_old)} before {epochs} update(s)",
          flush=True)

    with _policy_dropout_disabled(model):
        # Usually this loop is empty because vLLM supplied the frozen values.
        # It retains exact compatibility with HF generation and with any vLLM
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
                ex["old_policy_logprobs"] = old_lp.detach()

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
                    result = compute_batched_token_logprobs(
                        model, batch, with_grad=True,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id,
                        measure_entropy=(epoch == 0))
                    current_logprobs = result
                    weighted_losses = []
                    for ex, cur_lp in zip(
                            batch, current_logprobs):
                        loss, metrics = clipped_policy_loss(
                            cfg, cur_lp, ex["old_policy_logprobs"],
                            None, ex["advantage"],
                            clip_epsilon=epsilon,
                            clip_epsilon_low=epsilon_low,
                            clip_epsilon_high=epsilon_high,
                            kl_coef=kl_coef)
                        weight = float(ex["sample_weight"])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError(
                                "nonfinite SPO-RS loss")
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

            if precomputed_gradient_state is not None:
                _apply_precomputed_exact_gradients(
                    model, precomputed_gradient_state, step_idx)
                normal_stats = precomputed_gradient_state.get(
                    "normal_stats", {})
                normal_totals = normal_stats.get("totals", {})
                for key in totals:
                    totals[key] += float(normal_totals.get(key, 0.0))
                max_ratio = max(
                    max_ratio, float(normal_stats.get("max_ratio", 0.0)))
                max_prefix_ratio = max(
                    max_prefix_ratio,
                    float(normal_stats.get("max_prefix_ratio", 0.0)))
                entropy_examples += int(
                    normal_stats.get("entropy_examples", 0))

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
                  f"reference=off "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={len(quarantined)}", flush=True)
    for ex in examples:
        for key in ("old_policy_logprobs",):
            ex.pop(key, None)
    recovery_seconds = float(
        (precomputed_gradient_state or {}).get(
            "normal_stats", {}).get("training_seconds", 0.0))
    return {
        "policy_updates": history,
        "training_seconds": recovery_seconds + (time.time() - started),
        "oom_quarantined_examples": sum(
            item["oom_quarantined_examples"] for item in history),
        "exact_overflow_reused_examples": int(
            (precomputed_gradient_state or {}).get(
                "normal_stats", {}).get("trained_examples", 0)),
        "exact_overflow_sharded_examples": (
            len(examples) if precomputed_gradient_state is not None else 0),
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

    def estimated_host_offload_bytes(self):
        """RAM required to move every live trainer replica off its GPU."""
        from host_memory import (GIB, estimated_module_bytes,
                                 estimated_optimizer_bytes)

        model_bytes = sum(
            estimated_module_bytes(model)
            for _, model, _, _ in self.replicas
        )
        optimizer_bytes = estimated_optimizer_bytes(self.optimizer)
        # The model estimate is storage-exact.  The small margin covers Python
        # tensor metadata, allocator alignment, and one GiB of workspace per
        # process without pretending that an additional model copy is needed.
        return int(
            1.05 * (model_bytes + optimizer_bytes)
            + self.world_size * GIB
        )

    def reclaim_host_memory(self):
        """Synchronize all replicas, then drop unused host allocations."""
        import torch
        from host_memory import reclaim_unused_host_memory

        devices = tuple(range(self.world_size))
        return [reclaim_unused_host_memory(
            torch, cuda_devices=devices, force=True)]

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

    def _train_spo_rs_fast(self, examples, cfg, step_idx):
        """Adaptive, work-stealing SPO-RS update selected only by --fast.

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

        epochs = int(getattr(cfg, "spo_rs_update_epochs",
                             SPO_RS_UPDATE_EPOCHS_DEFAULT))
        if epochs != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")
        epsilon, epsilon_low, epsilon_high, kl_coef = (
            _clipped_policy_options(cfg))

        cpu_examples = self._cpu_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in cpu_examples)
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(cpu_examples)}, reference="
              f"off; "
              "HF fallback runs only "
              "for missing values", flush=True)

        pending = sorted(
            cpu_examples,
            key=lambda example: int(
                example["prompt_ids"].shape[1]
                + example["response_ids"].shape[1]))

        queue_lock = threading.Lock()
        # A token budget is the real memory controller. This loose count guard
        # only prevents pathological thousands-of-tiny-sequences batches.
        max_examples_per_batch = min(64, max(1, len(cpu_examples)))

        def take_batch(token_budget):
            with queue_lock:
                if not pending:
                    return []
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

                        with _policy_dropout_disabled(model):
                            for batch in active_batches:
                                missing_old = (
                                    _initialize_behavior_logprobs(batch))
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
                                        example["old_policy_logprobs"] = (
                                            old_lp.detach())


                                result = compute_batched_token_logprobs(
                                    model, batch, with_grad=True,
                                    chunk=cfg.logprob_chunk,
                                    pad_token_id=tokenizer.pad_token_id)
                                current_logprobs = result

                                weighted_losses = []
                                for example, current_lp in zip(
                                        batch, current_logprobs):
                                    loss, metrics = clipped_policy_loss(
                                        cfg, current_lp,
                                        example["old_policy_logprobs"],
                                        None,
                                        example["advantage"],
                                        clip_epsilon=epsilon,
                                        clip_epsilon_low=epsilon_low,
                                        clip_epsilon_high=epsilon_high,
                                        kl_coef=kl_coef,
                                        return_tensor_metrics=True)
                                    weight = float(example["sample_weight"])
                                    if not torch.isfinite(loss).all():
                                        raise FloatingPointError(
                                            "nonfinite SPO-RS loss")
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
              f"reference=off "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        return {
            "policy_updates": history,
            "training_seconds": time.time() - started,
            "training_parallel_gpus": self.world_size,
            "training_fast": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(budgets),
            "fast_peak_memory_fraction": [
                result["peak_memory_fraction"] for result in results],
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_spo_rs(self, examples, cfg, step_idx):
        if bool(getattr(cfg, "fast", False)):
            return self._train_spo_rs_fast(
                examples, cfg, step_idx)

        import torch

        started = time.time()
        if self._offloaded:
            print(f"[train-parallel] restoring {self.world_size} trainer "
                  "replicas for the update", flush=True)
            self.restore_after_generation()
        epochs = int(getattr(cfg, "spo_rs_update_epochs",
                             SPO_RS_UPDATE_EPOCHS_DEFAULT))
        epsilon, epsilon_low, epsilon_high, kl_coef = (
            _clipped_policy_options(cfg))
        local_shards = self._prepare_shards(examples)
        update_batches = [
            _training_microbatches(shard, cfg)
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
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"off; "
              "HF fallback runs only "
              f"for missing values before {epochs} update(s)", flush=True)

        def cache(item):
            replica_index, shard = item
            backend, model, tokenizer, logical_id = self.replicas[replica_index]
            backend.set_training_mode()
            with torch.cuda.device(logical_id), _policy_dropout_disabled(model):
                missing_old = (
                    _initialize_behavior_logprobs(shard))
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
                        example["old_policy_logprobs"] = old_lp.detach()
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
                    with _policy_dropout_disabled(model):
                        for batch in active_batches:
                            result = compute_batched_token_logprobs(
                                model, batch, with_grad=True,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id,
                                measure_entropy=(epoch == 0))
                            current_logprobs = result
                            weighted_losses = []
                            for example, current_lp in zip(
                                    batch, current_logprobs):
                                loss, metrics = clipped_policy_loss(
                                    cfg, current_lp,
                                    example["old_policy_logprobs"],
                                    None,
                                    example["advantage"],
                                    clip_epsilon=epsilon,
                                    clip_epsilon_low=epsilon_low,
                                    clip_epsilon_high=epsilon_high,
                                    kl_coef=kl_coef)
                                weight = float(example["sample_weight"])
                                if not torch.isfinite(loss).all():
                                    raise FloatingPointError(
                                        "nonfinite SPO-RS loss")
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
                  f"reference=off "
                  f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
                  f"clipped={totals['clipped_fraction']:.1%} "
                  f"entropy examples={entropy_examples} "
                  f"OOM-quarantined={quarantined_examples}", flush=True)

        return {
            "policy_updates": history,
            "training_seconds": time.time() - started,
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

    def offload_for_generation(self, *, purge_host_cache=True):
        """Release every replica before an all-GPU vLLM phase."""
        if self._offloaded:
            return
        import gc
        import torch
        if purge_host_cache:
            self.reclaim_host_memory()
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
        from host_memory import reclaim_unused_host_memory
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
        reclaim_unused_host_memory(
            torch, cuda_devices=tuple(range(self.world_size)), force=True)
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
        self._exp_dir = Path(exp_dir).resolve()
        self._process_label = (
            "train-fast" if bool(cfg.fast) else "train-process")
        self._offloaded = False
        self._offloaded_ranks = set()
        self._closed = False
        self._workers_idle = True
        self._entropic_overlap = None
        self._spo_rs_overlap = None
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

        from host_memory import (
            GIB, TokenWeightedActivationOffloadGate,
            automatic_activation_offload_budget_bytes,
        )
        self._activation_offload_budget_bytes = (
            automatic_activation_offload_budget_bytes())
        self._activation_offload_semaphore = (
            TokenWeightedActivationOffloadGate(
                self._context,
                self._activation_offload_budget_bytes,
                max(1, int(cfg.max_seq_length)),
            ))
        self._activation_offload_slots = (
            self._activation_offload_semaphore.full_context_concurrency)
        self._activation_offload_budget_gib = (
            self._activation_offload_budget_bytes / GIB)

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
                        self._activation_offload_semaphore,
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

        if str(getattr(cfg, "advantage_mode", "")).lower() == "spo-rs":
            fallback_description = (
                "oversized exact trajectories use token-weighted replica "
                "activation offload from a "
                f"{self._activation_offload_budget_gib:.0f}-GiB RAM budget "
                f"({self._activation_offload_slots} full-context or up to "
                f"{self._world_size} shorter ranks); "
                "only the residual tail routes to the model-sharded lane")
        else:
            fallback_description = (
                "token-weighted CPU activation offload uses a "
                f"{self._activation_offload_budget_gib:.0f}-GiB RAM budget "
                f"({self._activation_offload_slots} full-context or up to "
                f"{self._world_size} shorter ranks)")
        setting_log_only(
            f"[{self._process_label}] process-distributed trainer active on "
            f"physical GPUs {self._physical_ids}; adaptive memory ceiling="
            f"{self._memory_percentage:g}%, "
            "exact long-rollout rescue staged at 96%, 98%, then 99%; "
            f"{fallback_description}; live RAM admission is shared with "
            "evaluation and model transfers", flush=True)

    @property
    def world_size(self):
        return self._world_size

    @property
    def is_offloaded(self):
        return self._offloaded

    def estimated_host_offload_bytes(self):
        """RAM required to move every live trainer replica off its GPU."""
        from host_memory import (GIB, estimated_module_bytes,
                                 estimated_optimizer_bytes)

        model_bytes = estimated_module_bytes(self.model)
        optimizer_bytes = estimated_optimizer_bytes(self.optimizer)
        # All worker models have the same storage layout as rank zero.  The
        # optimizer exists only on rank zero; workers retain gradients/model
        # state but do not construct duplicate Adam state.
        return int(
            1.05 * (model_bytes * self._world_size + optimizer_bytes)
            + self._world_size * GIB
        )

    @property
    def host_memory_admission(self):
        return self._activation_offload_semaphore.host_admission

    def reclaim_host_memory(self):
        """Synchronize every trainer process and free unused host caches."""
        import torch
        from host_memory import reclaim_unused_host_memory

        if not self._workers_idle:
            raise RuntimeError(
                "cannot reclaim trainer host caches while an update is active")
        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "purge_host_memory"})
        with torch.cuda.device(0):
            torch.cuda.synchronize(0)
        local_cleanup = reclaim_unused_host_memory(
            torch, cuda_devices=(0,), force=True)
        child_messages = self._collect_event(
            "host_memory_purged", range(1, self._world_size))
        return [local_cleanup] + [
            child_messages[rank]["cleanup"]
            for rank in range(1, self._world_size)
        ]

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
                traceback_text = message.get("traceback", "no traceback")
                if "ExactLongContextTrainingOOM" in traceback_text:
                    raise ExactLongContextTrainingOOM(
                        f"fast trainer rank {rank} exhausted exact "
                        "single-GPU long-context fallbacks:\n"
                        f"{traceback_text}")
                raise RuntimeError(
                    f"fast trainer rank {rank} failed:\n"
                    f"{traceback_text}")
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

    def _queue_examples(
            self, examples, *, seal=True, announce=True,
            weighted_offload=False):
        """Queue expensive examples first; idle ranks steal the next work."""
        import torch

        maximum_length = max(1, int(self.cfg.max_seq_length))

        def estimated_cost(example):
            prompt = int(example["prompt_ids"].shape[1])
            response = int(example["response_ids"].shape[1])
            total = prompt + response
            return total * total + response * maximum_length

        copied = self._cpu_examples(examples)
        overflow_floor = getattr(self, "_warm_overflow_min_tokens", None)
        if overflow_floor is not None:
            for example in copied:
                example["_cooperative_overflow"] = bool(
                    int(example["prompt_ids"].shape[1]
                        + example["response_ids"].shape[1]) >= overflow_floor)
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
            offload_waves = None
            if weighted_offload:
                # First-fit decreasing packs recovery trajectories into
                # RAM-safe concurrent waves. Unlike the former fixed rank
                # count, a wave can use every GPU when its shorter activation
                # graphs fit the same node-wide budget. Flattening the waves
                # retains asynchronous work stealing; the shared weighted
                # gate remains the authoritative safety mechanism.
                gate = self._activation_offload_semaphore
                waves = []
                for example in ordered:
                    work = int(example["prompt_ids"].shape[1]) + int(
                        example["response_ids"].shape[1])
                    units = gate.request_units(work)
                    if units is None:
                        waves.append({
                            "units": gate.capacity_units + 1,
                            "items": [example],
                        })
                        continue
                    placed = False
                    for wave in waves:
                        if (len(wave["items"]) < self._world_size
                                and int(wave["units"]) + int(units)
                                <= int(gate.capacity_units)):
                            wave["items"].append(example)
                            wave["units"] += int(units)
                            placed = True
                            break
                    if not placed:
                        waves.append({
                            "units": int(units),
                            "items": [example],
                        })
                ordered = [
                    example for wave in waves for example in wave["items"]
                ]
                offload_waves = [
                    len(wave["items"]) for wave in waves
                ]
            for example in ordered:
                self._work_queue.put(example)
            if announce:
                if offload_waves is None:
                    print(f"[{self._process_label}] queued {len(ordered)} "
                          "examples longest-first; each free GPU claims the "
                          "next adaptive batch", flush=True)
                else:
                    preview = ",".join(
                        str(value) for value in offload_waves[:8])
                    suffix = "..." if len(offload_waves) > 8 else ""
                    print(
                        f"[{self._process_label}] queued {len(ordered)} "
                        "oversized examples longest-first; conservative "
                        f"startup ranks/wave={preview}{suffix}, then measured "
                        "token-weighted admission expands concurrency; idle "
                        "ranks steal work as RAM capacity is freed",
                        flush=True,
                    )
        if seal:
            for _ in range(self._world_size):
                self._work_queue.put(None)
        return ordered


    def train_spo_rs(self, examples, cfg, step_idx):
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      local_spo_rs_update,
                                      reduce_trainable_gradients)

        if self._offloaded:
            print(f"[train-fast] restoring {self._world_size} process "
                  "trainers for the update", flush=True)
            self.restore_after_generation()
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")

        started = time.time()
        queued_examples = self._queue_examples(examples)
        supplied_old = sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued_examples)
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(examples)}, reference="
              f"off",
              flush=True)
        print(f"[train-fast] one process/GPU; memory ceiling="
              f"{self._memory_percentage:g}%; "
              "exact singleton rescue=96%->98%->99%; "
              f"initial padded-token budgets/GPU={self._token_budgets}",
              flush=True)

        defer_exact_oom = (
            str(getattr(cfg, "advantage_mode", "")).lower() == "spo-rs")
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "defer_exact_oom": defer_exact_oom,
            })

        local_stats = local_spo_rs_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            activation_offload_semaphore=(
                self._activation_offload_semaphore),
            defer_exact_oom=defer_exact_oom)
        child_messages = self._collect_event(
            "computed", range(1, self._world_size))
        rank_stats = [local_stats] + [
            child_messages[rank]["stats"]
            for rank in range(1, self._world_size)
        ]
        accounted_examples = sum(
            int(stats["trained_examples"])
            + int(stats["quarantined_examples"])
            + int(stats.get("deferred_examples", 0))
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
        overflow_error = None
        grad_norm = None
        try:
            deferred_examples = sum(
                int(stats.get("deferred_examples", 0))
                for stats in rank_stats)
            if deferred_examples:
                gradient_path, marker, deferred_keys, _normal_stats = (
                    _save_exact_spo_rs_overflow(
                        self._exp_dir, step_idx, self.model, rank_stats,
                        normal_train_seconds=time.time() - started))
                longest = max(
                    (int(length) for stats in rank_stats
                     for length in stats.get("deferred_token_lengths", ())),
                    default=0)
                print(
                    f"[train-overflow] step {step_idx}: retained exact "
                    f"gradients for "
                    f"{len(queued_examples) - deferred_examples}/"
                    f"{len(queued_examples)} trajectories; routing only "
                    f"{deferred_examples} oversized trajectory/trajectories "
                    "to cooperative GPU overflow "
                    f"(longest={longest} tokens, "
                    f"state={gradient_path.name}, marker={marker.name})",
                    flush=True,
                )
                overflow_error = ExactLongContextTrainingOOM(
                    f"step {step_idx} deferred {len(deferred_keys)} exact "
                    "trajectory/trajectories to the selective multi-GPU "
                    "overflow lane")
            else:
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
        if overflow_error is not None:
            from exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, queued_examples, cfg, step_idx,
                    _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

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
              f"reference=off "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        return {
            "policy_updates": history,
            "training_seconds": time.time() - started,
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

    def recover_spo_rs_overflow(
            self, examples, cfg, step_idx, overflow_state):
        """Finish saved SPO-RS overflow without replaying completed gradients.

        Older Qwen checkpoints first try grouped checkpointing on every GPU
        replica. Work that exhausted that mode uses warm cooperative groups.
        Legacy backends retain RAM-bounded replica recovery.
        Normal gradients already stored in ``overflow_state`` are never
        replayed. Remaining trajectories are work-stolen by all replicas. A
        trajectory that exhausts card memory may use the shared host-RAM
        admission gate; successful gradients are reduced once and atomically
        merged into the durable checkpoint. Only a trajectory that also fails
        that bounded fallback is left for the final model-sharded lane.
        """
        from exact_overflow import supports_warm_overflow, complete_warm_overflow
        grouped_recovery = bool(
            overflow_state is not None
            and overflow_state.get("remaining_keys")
            and not overflow_state.get("replica_checkpoint_groups_attempted", False)
            and any(hasattr(module, "_ttt_group_checkpoint_min_tokens")
                    for module in self.model.modules()))
        if supports_warm_overflow(cfg) and not grouped_recovery:
            return complete_warm_overflow(
                self, examples, cfg, step_idx, overflow_state)

        import torch
        from fast_distributed import (
            broadcast_trainable_parameters, local_spo_rs_update,
            reduce_trainable_gradients,
        )

        if str(getattr(cfg, "advantage_mode", "")).lower() != "spo-rs":
            raise ValueError("exact overflow recovery is SPO-RS-only")
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("exact overflow recovery requires one update epoch")
        if self._closed or not self._workers_idle:
            raise RuntimeError(
                "process trainer is not idle for exact overflow recovery")
        if self._offloaded:
            print(
                f"[{self._process_label}] restoring {self._world_size} "
                "process trainers for exact overflow recovery",
                flush=True,
            )
            self.restore_after_generation()

        _validate_precomputed_exact_gradients(
            self.model, overflow_state, step_idx)
        requested_keys = {
            tuple(int(value) for value in example["_overlap_key"])
            for example in examples
        }
        remaining_keys = {
            tuple(int(value) for value in key)
            for key in overflow_state.get("remaining_keys", ())
        }
        if not requested_keys or requested_keys != remaining_keys:
            raise RuntimeError(
                "replicated exact-overflow inputs do not match the durable "
                f"remaining keys: inputs={len(requested_keys)}, "
                f"remaining={len(remaining_keys)}")

        started = time.time()
        recovery_cfg = SimpleNamespace(**dict(vars(cfg)))
        recovery_cfg._exact_replica_force_activation_offload = not grouped_recovery
        if grouped_recovery:
            # A saved OOM belongs to the old layer-checkpoint execution mode.
            # Give every remaining trajectory the new GPU-only mode once.
            recovery_cfg._exact_replica_overflow_min_tokens = 0
            self.__dict__.pop("_warm_overflow_min_tokens", None)
            examples = [{key: value for key, value in item.items()
                         if key != "_cooperative_overflow"} for item in examples]
        queued_examples = self._queue_examples(
            examples, weighted_offload=not grouped_recovery)
        if grouped_recovery:
            print(f"[recovery] step {step_idx}: dispatching {len(queued_examples)} "
                  f"saved trajectories across all {self._world_size} GPU replicas "
                  "with grouped GPU checkpoints; completed gradients retained",
                  flush=True)
        else:
            print(
                f"[recovery] step {step_idx}: dispatching "
                f"{len(queued_examples)} saved oversized trajectories across "
                f"{self._world_size} GPU replicas; CPU activation admission "
                f"starts conservatively, measures real footprint, then shares "
                f"its {self._activation_offload_budget_gib:.0f}-GiB RAM budget "
                "across every rank whose current trajectory fits",
                flush=True,
            )
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(recovery_cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "defer_exact_oom": True,
            })

        try:
            local_stats = local_spo_rs_update(
                self.backend, self.model, self.tokenizer, (), recovery_cfg, 0,
                self._token_budgets[0],
                memory_fraction=self._memory_fraction,
                work_queue=self._work_queue,
                activation_offload_semaphore=(
                    self._activation_offload_semaphore),
                defer_exact_oom=True,
            )
            child_messages = self._collect_event(
                "computed", range(1, self._world_size))
            rank_stats = [local_stats] + [
                child_messages[rank]["stats"]
                for rank in range(1, self._world_size)
            ]
            accounted_examples = sum(
                int(stats["trained_examples"])
                + int(stats["quarantined_examples"])
                + int(stats.get("deferred_examples", 0))
                for stats in rank_stats
            )
            if accounted_examples != len(queued_examples):
                raise RuntimeError(
                    "replicated exact-overflow queue lost or duplicated "
                    f"work: accounted for {accounted_examples}/"
                    f"{len(queued_examples)} trajectories")

            deferred_keys = {
                (int(key[0]), int(key[1]))
                for stats in rank_stats
                for key in stats.get("deferred_keys", ())
            }
            deferred_count = sum(
                int(stats.get("deferred_examples", 0))
                for stats in rank_stats)
            quarantined_count = sum(
                int(stats["quarantined_examples"])
                for stats in rank_stats)
            trained_count = sum(
                int(stats["trained_examples"]) for stats in rank_stats)
            if (deferred_count != len(deferred_keys)
                    or not deferred_keys.issubset(requested_keys)
                    or quarantined_count):
                raise RuntimeError(
                    "replicated exact-overflow recovery produced invalid "
                    "trajectory accounting: "
                    f"trained={trained_count}, deferred={deferred_count}, "
                    f"unique deferred={len(deferred_keys)}, "
                    f"quarantined={quarantined_count}")
            successful_keys = requested_keys - deferred_keys
            if trained_count != len(successful_keys):
                raise RuntimeError(
                    "replicated exact-overflow successful-key accounting "
                    f"changed: trained={trained_count}, "
                    f"keys={len(successful_keys)}")
            metric_keys = (
                "loss", "policy_loss", "kl_estimate", "entropy_estimate",
                "ratio", "clipped_fraction",
            )
            wave_stats = {
                "totals": {
                    key: sum(float(stats["totals"][key])
                             for stats in rank_stats)
                    for key in metric_keys
                },
                "max_ratio": max(
                    float(stats["max_ratio"]) for stats in rank_stats),
                "max_prefix_ratio": max(
                    float(stats["max_prefix_ratio"])
                    for stats in rank_stats),
                "entropy_examples": sum(
                    int(stats["entropy_examples"]) for stats in rank_stats),
                "backward_batches": sum(
                    int(stats["backward_batches"]) for stats in rank_stats),
            }

            # Workers reduce their local sums into rank 0, then wait at the
            # parameter broadcast while rank 0 commits or applies the update.
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "finish_update"})
            reduce_trainable_gradients(self.model, destination_rank=0)

            update_error = None
            overflow_error = None
            grad_norm = None
            try:
                if successful_keys:
                    _commit_exact_spo_rs_overflow_batch(
                        self.model, overflow_state, step_idx,
                        successful_keys, wave_stats,
                        training_seconds=time.time() - started,
                    )

                # The durable checkpoint is now authoritative. Clear the
                # just-reduced wave before materializing the complete saved
                # gradient, otherwise successful recovery work is doubled.
                self.model.zero_grad(set_to_none=True)
                if deferred_keys:
                    if grouped_recovery:
                        overflow_state["replica_checkpoint_groups_attempted"] = True
                    overflow_state["replica_offload_exhausted"] = True
                    _write_exact_spo_rs_overflow_state(overflow_state)
                    marker = _read_exact_sharded_recovery(
                        self._exp_dir) or {}
                    _write_exact_sharded_recovery(
                        self._exp_dir, step_idx,
                        marker.get(
                            "reason",
                            "bounded replicated overflow recovery exhausted"),
                        replica_offload_exhausted=True,
                        remaining_deferred_keys=[
                            list(key) for key in sorted(deferred_keys)],
                    )
                    overflow_error = ExactLongContextTrainingOOM(
                        f"step {step_idx} retained "
                        f"{len(successful_keys)} additional exact gradient(s) "
                        f"but {len(deferred_keys)} trajectory/trajectories "
                        "still exceed bounded replica activation offload")
                else:
                    _apply_precomputed_exact_gradients(
                        self.model, overflow_state, step_idx)
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters()
                         if parameter.requires_grad],
                        max_norm=cfg.grad_clip,
                        error_if_nonfinite=True,
                    )
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
            if overflow_error is not None:
                if grouped_recovery and supports_warm_overflow(cfg):
                    return complete_warm_overflow(
                        self, examples, cfg, step_idx, overflow_state)
                raise overflow_error
        except BaseException:
            if not self._workers_idle:
                self._abort_workers()
            raise

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(
                1, int(stats["token_budget"]))
        normal_stats = overflow_state.get("normal_stats", {})
        overflow_stats = overflow_state.get("overflow_stats", {})
        normal_totals = normal_stats.get("totals", {})
        overflow_totals = overflow_stats.get("totals", {})
        totals = {
            key: (float(normal_totals.get(key, 0.0))
                  + float(overflow_totals.get(key, 0.0)))
            for key in metric_keys
        }
        max_ratio = max(
            float(normal_stats.get("max_ratio", 0.0)),
            float(overflow_stats.get("max_ratio", 0.0)),
        )
        max_prefix_ratio = max(
            float(normal_stats.get("max_prefix_ratio", 0.0)),
            float(overflow_stats.get("max_prefix_ratio", 0.0)),
        )
        entropy_examples = (
            int(normal_stats.get("entropy_examples", 0))
            + int(overflow_stats.get("entropy_examples", 0))
        )
        trained_per_gpu = [
            int(stats["trained_examples"]) for stats in rank_stats
        ]
        batches_per_gpu = [
            int(stats["backward_batches"]) for stats in rank_stats
        ]
        history = [{
            "epoch": 1,
            **totals,
            "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "entropy_examples": entropy_examples,
            "oom_quarantined_examples": 0,
            "grad_norm": grad_norm,
        }]
        print(
            f"[{self._process_label}] exact overflow trained examples/GPU="
            f"{trained_per_gpu}; backward batches/GPU={batches_per_gpu}; "
            f"final padded-token budgets/GPU={self._token_budgets}",
            flush=True,
        )
        print(
            f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
            f"loss={totals['loss']:.6f} "
            f"reference=off "
            f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
            f"clipped={totals['clipped_fraction']:.1%} "
            f"entropy examples={entropy_examples} OOM-quarantined=0 "
            + ("(grouped-checkpoint GPU replica recovery)" if grouped_recovery else
             "(RAM-bounded replicated exact overflow recovery)"),
            flush=True,
        )
        return {
            "policy_updates": history,
            "training_seconds": (
                float(normal_stats.get("training_seconds", 0.0))
                + float(overflow_stats.get("training_seconds", 0.0))),
            "training_parallel_gpus": self._world_size,
            "training_fast": True,
            "training_fast_processes": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "oom_quarantined_examples": 0,
            "exact_overflow_reused_examples": int(
                normal_stats.get("trained_examples", 0))
                + len(overflow_state.get("completed_keys", ()))
                - len(successful_keys),
            "exact_overflow_replica_examples": len(successful_keys),
            "exact_overflow_sharded_examples": 0,
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
            adaptive_batches=adaptive_batches,
            activation_offload_semaphore=(
                self._activation_offload_semaphore))
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

    def begin_spo_rs_overlap(self, cfg, step_idx):
        """Start exact clipped SPO-RS backward while CPU evaluation runs."""
        from concurrent.futures import ThreadPoolExecutor
        from fast_distributed import local_spo_rs_update

        if str(getattr(cfg, "advantage_mode", "")).lower() != "spo-rs":
            raise ValueError("streaming clipped overlap is SPO-RS-only")
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("streaming SPO-RS requires one update epoch")
        if self._closed or not self._workers_idle:
            raise RuntimeError(
                "process trainer is not idle for an SPO-RS overlap update")
        if self._spo_rs_overlap is not None:
            raise RuntimeError("an SPO-RS overlap update is already active")
        if self._entropic_overlap is not None:
            raise RuntimeError("an entropic overlap update is already active")
        if self._offloaded:
            print(f"[{self._process_label}] restoring {self._world_size} "
                  "process trainers while CPU evaluation continues",
                  flush=True)
            self.restore_after_generation()

        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "defer_exact_oom": True,
            })

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="spo-rs-rank0-overlap")
        state = {
            "overflow_examples": {},
            "step_idx": int(step_idx),
            "started_at": time.time(),
            "computed_at": None,
            "executor": executor,
            "future": None,
            "queued_examples": 0,
            "scheduled_keys": set(),
            "supplied_old": 0,
            "sealed": False,
        }
        future = executor.submit(
            local_spo_rs_update,
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            activation_offload_semaphore=(
                self._activation_offload_semaphore),
            defer_exact_oom=True)
        state["future"] = future

        def _record_completion(_future):
            state["computed_at"] = time.time()

        future.add_done_callback(_record_completion)
        self._spo_rs_overlap = state
        return state

    def queue_spo_rs_overlap(self, examples):
        """Feed reward-complete SPO-RS trajectories to all free GPUs."""
        state = self._spo_rs_overlap
        if state is None or state["sealed"]:
            raise RuntimeError("SPO-RS overlap queue is not open")
        if any(example.get("advantage") is None for example in examples):
            raise RuntimeError(
                "SPO-RS overlap cannot queue an unresolved advantage")
        queued = self._queue_examples(
            examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(
            _valid_example_token_logprobs(example, "behavior_logprobs")
            for example in queued)
        state["scheduled_keys"].update(
            example["_overlap_key"] for example in queued)
        state["overflow_examples"].update(
            (tuple(example["_overlap_key"]), example) for example in queued)
        return len(queued)

    def seal_spo_rs_overlap(self):
        """Tell every clipped-policy GPU consumer the stream is complete."""
        state = self._spo_rs_overlap
        if state is None:
            return
        if not state["sealed"]:
            for _ in range(self._world_size):
                self._work_queue.put(None)
            state["sealed"] = True

    def finish_spo_rs_overlap(self, cfg, step_idx, expected_examples, *,
                              apply_update=True):
        """Reduce and apply the exact streamed SPO-RS gradients once."""
        import torch
        from fast_distributed import (broadcast_trainable_parameters,
                                      reduce_trainable_gradients)

        state = self._spo_rs_overlap
        if state is None:
            raise RuntimeError("no SPO-RS overlap update is active")
        self.seal_spo_rs_overlap()
        expected_examples = int(expected_examples)
        queued_examples = int(state["queued_examples"])
        if apply_update and expected_examples != queued_examples:
            raise RuntimeError(
                "SPO-RS overlap final example count changed: "
                f"queued={queued_examples}, final={expected_examples}")

        try:
            local_stats = state["future"].result()
            child_messages = self._collect_event(
                "computed", range(1, self._world_size))
            # Rank 0 can finish before a longer worker. Record completion only
            # after every GPU has drained the shared queue.
            state["computed_at"] = time.time()
            rank_stats = [local_stats] + [
                child_messages[rank]["stats"]
                for rank in range(1, self._world_size)
            ]
            accounted_examples = sum(
                int(stats["trained_examples"])
                + int(stats["quarantined_examples"])
                + int(stats.get("deferred_examples", 0))
                for stats in rank_stats)
            if accounted_examples != queued_examples:
                raise RuntimeError(
                    "streaming SPO-RS queue lost or duplicated work: "
                    f"accounted for {accounted_examples}/{queued_examples} "
                    "examples")

            deferred_examples = sum(
                int(stats.get("deferred_examples", 0))
                for stats in rank_stats)
            for command_queue in self._command_queues.values():
                command_queue.put({
                    "kind": "finish_update",
                    "apply_update": bool(apply_update),
                })

            grad_norm = None
            update_error = None
            overflow_error = None
            if apply_update:
                reduce_trainable_gradients(
                    self.model, destination_rank=0)
                try:
                    if deferred_examples:
                        gradient_path, marker, deferred_keys, _normal_stats = (
                            _save_exact_spo_rs_overflow(
                                self._exp_dir, step_idx, self.model,
                                rank_stats,
                                normal_train_seconds=max(
                                    0.0, state["computed_at"]
                                    - state["started_at"])))
                        longest = max(
                            (int(length) for stats in rank_stats
                             for length in stats.get(
                                 "deferred_token_lengths", ())),
                            default=0)
                        print(
                            f"[train-overflow] step {step_idx}: retained "
                            f"exact gradients for "
                            f"{queued_examples - deferred_examples}/"
                            f"{queued_examples} trajectories; routing only "
                            f"{deferred_examples} oversized trajectory/"
                            "trajectories to cooperative GPU overflow "
                            f"(longest={longest} tokens, "
                            f"state={gradient_path.name}, "
                            f"marker={marker.name})",
                            flush=True,
                        )
                        overflow_error = ExactLongContextTrainingOOM(
                            f"step {step_idx} deferred "
                            f"{len(deferred_keys)} exact trajectory/"
                            "trajectories to the selective multi-GPU "
                            "overflow lane")
                    else:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in self.model.parameters()
                             if parameter.requires_grad],
                            max_norm=cfg.grad_clip,
                            error_if_nonfinite=True)
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
            self._spo_rs_overlap = None

        if overflow_error is not None:
            from exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, list(state["overflow_examples"].values()), cfg,
                    step_idx, _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(
                1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        metric_keys = (
            "loss", "policy_loss", "kl_estimate", "entropy_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {
            key: sum(float(stats["totals"][key]) for stats in rank_stats)
            for key in metric_keys
        }
        max_ratio = max(float(stats["max_ratio"]) for stats in rank_stats)
        max_prefix_ratio = max(
            float(stats["max_prefix_ratio"]) for stats in rank_stats)
        entropy_examples = sum(
            int(stats["entropy_examples"]) for stats in rank_stats)
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
        peak_percentages = [
            round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [
            round(100.0 * fraction, 1)
            for fraction in allocator_fractions]
        completed_at = state["computed_at"] or time.time()
        elapsed = max(0.0, completed_at - state["started_at"])
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "entropy_examples": entropy_examples,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        print(f"[{self._process_label}] trained examples/GPU="
              f"{trained_per_gpu}; backward batches/GPU={batches_per_gpu}; "
              f"final padded-token budgets/GPU={self._token_budgets}; "
              f"peak memory/GPU={peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        print(f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"reference=off "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"entropy examples={entropy_examples} "
              f"OOM-quarantined={quarantined_examples} "
              "(forward/backward overlapped CPU evaluation)", flush=True)
        return {
            "policy_updates": history,
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_fast": True,
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
              "exact singleton rescue=96%->98%->99%; "
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
            adaptive_batches=adaptive_batches,
            activation_offload_semaphore=(
                self._activation_offload_semaphore))
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

    def offload_for_generation(self, *, purge_host_cache=True):
        if self._offloaded:
            return
        import gc
        import torch

        # Long-context saved tensors use PyTorch's *host* pinned allocator.
        # CUDA empty_cache() cannot release that cache.  Clear it on every
        # replica before allocating eight complete CPU model copies; doing so
        # at this phase boundary does not change the update and avoids the
        # step-over-step host-RAM growth that can OOM-kill the whole VM.
        if purge_host_cache:
            self.reclaim_host_memory()
        self._offloaded = True
        self._offloaded_ranks = set(range(self._world_size))
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
        self.restore_on_gpus(self._physical_ids)

    def restore_on_gpus(self, physical_gpu_ids):
        """Move only released GPU groups back to training, in parallel.

        Called between coder sleep waves, so a complete set of CPU trainer
        copies never has to coexist with all new inference sleep backups.
        No collective or parameter update occurs until every rank is ready.
        """
        if not self._offloaded:
            return
        physical_gpu_ids = set(int(gpu) for gpu in physical_gpu_ids)
        ranks = {rank for rank in self._offloaded_ranks
                 if self._physical_ids[rank] in physical_gpu_ids}
        if not ranks:
            return
        if not self._dist_initialized:
            raise RuntimeError(
                "process-distributed trainer is unavailable after a worker "
                "failure")
        import gc
        import torch
        from fast_distributed import (set_total_memory_ceiling,
                                      validate_model_device)
        from host_memory import reclaim_unused_host_memory

        try:
            print(f"[{self._process_label}] restoring trainer GPUs "
                  f"{[self._physical_ids[rank] for rank in sorted(ranks)]} "
                  "as coder memory is released; CPU evaluation continues",
                  flush=True)
            for rank in sorted(ranks - {0}):
                self._command_queues[rank].put({"kind": "restore"})
            if 0 in ranks:
                gc.collect()
                with torch.cuda.device(0):
                    torch.cuda.empty_cache()
                    set_total_memory_ceiling(0, self._memory_fraction)
                    self.backend.restore_after_generation()
                    self.backend.set_training_mode()
                    torch.cuda.synchronize(0)
                validate_model_device(self.model, 0)
                _restore_optimizer_state_to_parameters(self.optimizer)
                reclaim_unused_host_memory(
                    torch, cuda_devices=(0,), force=True)
            self._collect_event("restored", ranks - {0})
            self._offloaded_ranks.difference_update(ranks)
            self._offloaded = bool(self._offloaded_ranks)
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise

    def _abort_workers(self):
        import torch.distributed as dist

        was_initialized = self._dist_initialized
        self._terminate_workers_for_restart()
        if was_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        self._dist_initialized = False

    def _close_queues(self):
        # After consumers exit, pending rollout tensors cannot drain from the
        # feeder pipe. Joining that feeder at interpreter exit hangs forever.
        # Discard pending messages only when this trainer has stopped/aborted.
        queues = [getattr(self, "_work_queue", None),
                  getattr(self, "_result_queue", None)]
        queues.extend(getattr(self, "_command_queues", {}).values())
        for queue in queues:
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()

    def _terminate_workers_for_restart(self):
        """Stop child ranks without entering another distributed collective.

        Exact-overflow recovery is durable before its restart request is
        raised.  At that point the current process will be replaced with
        ``exec``, so a graceful NCCL barrier is both unnecessary and unsafe:
        one child missing the matching barrier would prevent the recovery
        process from ever starting.  Use one bounded deadline for all ranks,
        then force-kill any child that ignored SIGTERM.  Deliberately leave
        rank zero's process group intact; ``exec`` replaces the process and
        releases that CUDA/NCCL state without another potentially blocking
        distributed operation.
        """
        import time

        for process in self._processes.values():
            if process.is_alive():
                process.terminate()

        deadline = time.monotonic() + 10.0
        for process in self._processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))

        for process in self._processes.values():
            if not process.is_alive():
                continue
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            else:
                process.terminate()

        deadline = time.monotonic() + 5.0
        for process in self._processes.values():
            if process.is_alive():
                process.join(timeout=max(0.0, deadline - time.monotonic()))
        self._dist_initialized = False
        self._close_queues()

    def shutdown(self, *, for_restart=False):
        if self._closed:
            return
        self._closed = True

        if for_restart:
            self._terminate_workers_for_restart()
            return

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

        if self._spo_rs_overlap is not None:
            state = self._spo_rs_overlap
            self.seal_spo_rs_overlap()
            try:
                state["future"].result()
            except Exception:
                pass
            finally:
                state["executor"].shutdown(
                    wait=True, cancel_futures=False)
                self._spo_rs_overlap = None

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
            self._close_queues()


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
    paths = sorted(step_dir.glob(
        f"step{step_idx:02d}_group*_rollout*.meta.json"))
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
        if (int(meta["step"]) != step_idx
                or str(meta["advantage_mode"]).lower() != "spo-rs"):
            return None
        try:
            reward = float(meta["reward"])
            advantage = float(meta["advantage"])
            response_tokens = int(meta["n_response_tokens"])
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(reward) or not math.isfinite(advantage)
                or response_tokens < 0):
            return None
        base = meta_path.name.removesuffix(".meta.json")
        response_path = meta_path.with_name(base + ".txt")
        prompt_path = meta_path.with_name(base + ".prompt.txt")
        token_path = meta_path.with_name(base + ".tokens.zlib")
        behavior_path = meta_path.with_name(
            base + ".behavior_logprobs.zlib")
        if not response_path.is_file() or not prompt_path.is_file():
            return None
        artifacts.append({
            "meta_path": meta_path,
            "response_path": response_path,
            "prompt_path": prompt_path,
            "token_path": token_path if token_path.is_file() else None,
            "behavior_logprobs_path": (
                behavior_path if behavior_path.is_file() else None),
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
    if (set(per_group) != expected_groups
            or any(count != int(cfg.group_size)
                   for count in per_group.values())):
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
    paths = sorted(step_dir.glob(
        f"step{step_idx:02d}_group*_rollout*.meta.json"))
    expected_primary = int(cfg.groups_per_step) * int(cfg.group_size)
    if len(paths) < expected_primary:
        return None

    artifacts = []
    for meta_path in paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if (int(meta.get("step", -1)) != step_idx
                or meta.get("status") != "evaluation_pending"):
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
        behavior_path = meta_path.with_name(
            base + ".behavior_logprobs.zlib")
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
            or any(count != int(cfg.group_size)
                   for count in per_group.values())):
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


def _reconstruct_saved_response_tokens(
        model, tokenizer, response_text, expected_count):
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
    token_ids = list(tokenizer(
        response_text, add_special_tokens=False).input_ids)
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
    chosen = (tokenizer_eos if tokenizer_eos in configured
              else configured[0] if len(configured) == 1 else None)
    if canonical_count < expected_count and chosen is not None:
        try:
            visible = tokenizer.decode(
                [chosen], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
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
        "unavailable_original_tokens": max(
            0, expected_count - recovered_count),
        "canonical_extra_tokens": max(
            0, recovered_count - expected_count),
    }
    return token_ids, recovery


def _prepare_recoverable_response_tokens(model, tokenizer, artifacts):
    """Validate/cache artifact token IDs before distributed replicas load."""
    from experiment_io import (load_rollout_logprobs,
                               load_rollout_token_ids)
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
            response_text = item["response_path"].read_text(
                encoding="utf-8", errors="replace")
            try:
                token_ids, recovery = (
                    _reconstruct_saved_response_tokens(
                        model, tokenizer, response_text, expected_count))
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
        summary["unavailable_original_tokens"] += int(
            recovery["unavailable_original_tokens"])
        summary["canonical_extra_tokens"] += int(
            recovery["canonical_extra_tokens"])
        summary["count_changed_responses"] += int(
            len(token_ids) != expected_count)
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
    from gen_workers import make_progress_bar
    from problems.base import ParentContext, RewardResult
    from sampler import State

    if cfg.generation_backend != "vllm":
        raise RuntimeError(
            "generated-step SPO-RS recovery requires exact vLLM rollout "
            "artifacts")
    if gen_pool is None:
        raise RuntimeError(
            "generated-step SPO-RS recovery requires the "
            "configured vLLM pool for exact consecutive-policy scoring")
    if (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError(
            "generated-step SPO-RS recovery requires the restored pre-step "
            "tracker")
    spo_rs_tracker.prepare_step()
    if sampling_adapter_path is None or not Path(
            sampling_adapter_path).is_dir():
        raise FileNotFoundError(
            "generated-step SPO-RS recovery cannot find its sampling adapter: "
            f"{sampling_adapter_path}")

    token_summary = _prepare_recoverable_response_tokens(
        None, tokenizer, artifacts)
    if token_summary["exact_sidecars"] != len(artifacts):
        raise RuntimeError(
            "generated-step recovery refuses non-exact response token IDs")
    if token_summary["behavior_logprob_sidecars"] != len(artifacts):
        raise RuntimeError(
            "generated-step recovery requires every rollout-policy logprob "
            "sidecar")

    parents = sampler.sample_states(int(cfg.groups_per_step))
    if len(parents) != int(cfg.groups_per_step):
        raise RuntimeError(
            "generated-step recovery could not reconstruct every parent")
    parents_path = (
        Path(exp_dir) / f"step{int(step_idx):02d}"
        / f"step{int(step_idx):02d}.parents.json")
    try:
        saved_parent_rows = json.loads(
            parents_path.read_text(encoding="utf-8"))["parents"]
    except (OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        raise RuntimeError(
            "generated-step recovery cannot read its saved parent selection") \
            from error
    saved_parents = {
        int(row["group"]): row for row in saved_parent_rows
    }
    for group, parent in enumerate(parents):
        row = saved_parents.get(group)
        if row is None:
            raise RuntimeError(
                f"generated-step recovery is missing parent group {group}")
        if (str(row.get("parent_id")) != str(parent.id)
                or int(row.get("parent_timestep", -1))
                != int(parent.timestep)
                or not math.isclose(
                    float(row.get("parent_reward")), float(parent.value),
                    rel_tol=0.0, abs_tol=1e-12)):
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
        workers = max(1, min(
            len(artifacts), len(cpu_ids) * processes_per_cpu))
        isolated_memory_limit = _automatic_sandbox_memory_limit_bytes(workers)
    else:
        workers = max(1, min(
            len(artifacts), _resolve_reward_workers(cfg, problem)))

    from host_memory import HostMemoryAdmission
    import multiprocessing
    host_admission = (getattr(gen_pool, "host_memory_admission", None)
                      or HostMemoryAdmission(multiprocessing.get_context("spawn")))

    def evaluate(item):
        meta = item["meta"]
        issue = meta.get("output_format_issue")
        if issue:
            return RewardResult(
                reward=float(problem.fail_score), msg=str(issue),
                failure_kind="code")
        response_text = item["response_path"].read_text(
            encoding="utf-8", errors="replace")
        parent_context = parent_contexts[int(meta["parent_group"])]
        if not isolated_eval:
            return problem.compute_reward(
                response_text, parent_context, cfg.sandbox_timeout_s)
        with host_admission.reservation(isolated_memory_limit):
            cpu_id = isolated_slots.get()
            try:
                return problem.compute_reward(
                    response_text, parent_context, cfg.sandbox_timeout_s,
                    cpu_id=cpu_id,
                    memory_limit_bytes=isolated_memory_limit)
            finally:
                isolated_slots.put(cpu_id)

    evaluation_pool = ThreadPoolExecutor(max_workers=workers)
    evaluation_futures = {
        evaluation_pool.submit(evaluate, item): index
        for index, item in enumerate(artifacts)
    }
    results = [None] * len(artifacts)
    try:
        prompt_cache = {}
        score_pairs = []
        current_scores = []
        context_ids = []
        for item in artifacts:
            prompt_text = item["prompt_path"].read_text(
                encoding="utf-8", errors="replace")
            if prompt_text not in prompt_cache:
                prompt_cache[prompt_text] = tokenizer(
                    prompt_text).input_ids
            score_pairs.append((
                prompt_cache[prompt_text], item["response_token_ids"]))
            current_scores.append(item["behavior_logprobs"])
            context_ids.append(int(item["meta"]["parent_group"]))

        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is None:
            raise RuntimeError(
                "generated-step recovery has no preceding SPO-RS adapter")
        previous_adapter_path = Path(exp_dir) / previous_name
        if not previous_adapter_path.is_dir():
            raise FileNotFoundError(
                "generated-step recovery cannot find the preceding policy "
                f"adapter: {previous_adapter_path}")
        if (previous_adapter_path.resolve()
                == Path(sampling_adapter_path).resolve()):
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
            raise RuntimeError(
                "generated-step recovery did not obtain every exact "
                "consecutive-policy score")

        eval_bar = make_progress_bar(
            len(evaluation_futures), desc="recovering evaluations")
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
        prompt_text = item["prompt_path"].read_text(
            encoding="utf-8", errors="replace")
        if prompt_text not in prompt_indices:
            prompt_indices[prompt_text] = len(prompt_indices)
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    advantages = spo_rs_tracker.normalized_advantages(transformed)
    if advantages is None:
        raise RuntimeError(
            "generated-step recovery lost the pre-step SPO-RS baseline")
    preview_tracker = deepcopy(spo_rs_tracker)
    update = preview_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx,
        raw_rewards=rewards)
    update.update({
        "groups": sorted({int(item["meta"]["group"])
                          for item in artifacts}),
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
    beta_by_index = np.full(
        len(artifacts), float(cfg.spo_rs_beta), dtype=np.float64)

    save_ctor = bool(getattr(problem, "saves_construction", False))
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False)
                    else "PUCTSampler")
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
            raw_strategy = strategy_path.read_text(
                encoding="utf-8", errors="replace")
            strategy, _issue = _extract_final_strategy(
                raw_strategy,
                strict=bool(getattr(cfg, "strict_strategy_validation", False)),
            )
        info = rollout_infos[index]
        meta = {
            key: value for key, value in pending.items()
            if key != "status"
        }
        meta.update({
            "node_id": child.id,
            "parent_id": parent.id,
            "parent_timestep": int(parent.timestep),
            "sampler_type": sampler_type,
            "reward": float(result.reward),
            "raw_score": (float(result.raw_score)
                          if result.raw_score is not None else None),
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
            "parent_value": (float(parent.value)
                             if parent.value is not None else None),
            "parent_raw_score": (float(parent.raw_score)
                                 if parent.raw_score is not None else None),
            "parent_is_seed": parent.id in sampler._seed_ids,
            "parent_visit_count": int(parent_row.get("visit_count", 0)),
            "parent_q_value": parent_row.get("q_value"),
            "parent_prior": parent_row.get("prior"),
            "parent_exploration_bonus": parent_row.get(
                "exploration_bonus"),
            "parent_selection_score": parent_row.get("selection_score"),
            "archive_eligible": archive_eligible,
            "two_stage_rollout": bool(cfg.strategies),
            "strategy": strategy,
            "strategy_index": strategy_index,
            "strategy_model_name": (
                cfg.strategy_model_name if bool(cfg.strategies) else None),
            "programs_per_strategy": (
                int(cfg.programs_per_strategy)
                if bool(cfg.strategies) else None),
            "construction": (
                _as_float_list(result.construction,
                               cfg.max_saved_construction)
                if save_ctor else None),
            "parent_construction": (
                _as_float_list(parent.construction,
                               cfg.max_saved_construction)
                if save_ctor else None),
            "seed": int(cfg.seed),
            "spo_rs": info,
        })
        response_text = item["response_path"].read_text(
            encoding="utf-8", errors="replace")
        save_rollout(
            exp_dir, step_idx, group, rollout, response_text, meta,
            artifacts_already_saved=True)

    print(
        f"[resume] step {step_idx}: reused {len(artifacts)} exact generated "
        "rollouts, recomputed their CPU rewards, and completed exact "
        "consecutive-policy scoring; no policy sampling was repeated",
        flush=True,
    )
    completed = _recoverable_spo_rs_artifacts(exp_dir, step_idx, cfg)
    if completed is None:
        raise RuntimeError(
            "generated-step recovery failed to seal complete rollout metadata")
    _prepare_recoverable_response_tokens(None, tokenizer, completed)
    return completed


def _resume_spo_rs_step_from_artifacts(
        backend, model, tokenizer, sampler, optimizer, step_idx, cfg, exp_dir,
        problem, parallel_trainer, spo_rs_tracker, sampling_adapter_path,
        artifacts):
    """Finish one interrupted SPO-RS step from its durable evaluations."""
    import torch
    from experiment_io import save_rollout
    from reward import extract_python_code
    from sampler import State

    adaptive_spo_beta = bool(getattr(cfg, "spo_rs_adaptive_beta", False))

    if (parallel_trainer is None
            and str(getattr(cfg, "training_layout", "")) != "sharded"):
        raise RuntimeError(
            "evaluated-step recovery requires either the configured fast "
            "distributed trainer or the automatic exact sharded trainer")
    if (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError(
            "evaluated-step recovery requires the pre-step SPO-RS tracker")
    spo_rs_tracker.prepare_step()
    if sampling_adapter_path is None or not Path(sampling_adapter_path).is_dir():
        raise FileNotFoundError(
            "evaluated-step recovery cannot find the sampling adapter: "
            f"{sampling_adapter_path}")

    parents = sampler.sample_states(int(cfg.groups_per_step))
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
            raise RuntimeError(
                f"evaluated-step recovery has no rollout group {group}")
        first = records[0]["meta"]
        try:
            saved_value = float(first["parent_value"])
            saved_timestep = int(first["parent_timestep"])
            saved_parent_id = str(first["parent_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"evaluated-step recovery lacks parent identity for group "
                f"{group}") from error
        current_value = float(parent.value)
        if (saved_timestep != int(parent.timestep)
                or not math.isclose(saved_value, current_value,
                                    rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(
                f"evaluated-step recovery parent {group} does not match the "
                "restored sampler state")
        if parent.id != saved_parent_id:
            if step_idx == 0 and parent.id in sampler._seed_ids:
                # Fresh seed State UUIDs are intentionally random. Reuse the
                # durable IDs already embedded in every rollout so the healed
                # checkpoint preserves the original lineage exactly.
                sampler._seed_ids.remove(parent.id)
                parent.id = saved_parent_id
                sampler._seed_ids.add(parent.id)
            else:
                raise RuntimeError(
                    f"evaluated-step recovery parent ID {group} differs from "
                    "the restored sampler checkpoint")
        sampler.record_expansion(parent, count=int(cfg.group_size))

    rewards = [float(item["meta"]["reward"]) for item in artifacts]
    saved_advantages = np.asarray(
        [float(item["meta"]["advantage"]) for item in artifacts],
        dtype=np.float64)
    spo_infos = [
        _literal_metadata_mapping(item["meta"].get("spo_rs"))
        for item in artifacts
    ]
    if any(not info or not bool(info.get("used_for_policy_update", False))
           for info in spo_infos):
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
        raise RuntimeError(
            "evaluated-step recovery does not match the saved pre-step "
            "SPO-RS value")
    if int(spo_info.get("group_size", -1)) != len(artifacts):
        raise RuntimeError(
            "evaluated-step recovery rollout count differs from the saved "
            "SPO-RS update")
    if Path(str(spo_info.get("policy_adapter", ""))).name != Path(
            sampling_adapter_path).name:
        raise RuntimeError(
            "evaluated-step recovery sampling adapter differs from the "
            "saved SPO-RS update")
    for index, (info, transformed_reward) in enumerate(
            zip(spo_infos, transformed)):
        saved_transformed = info.get("transformed_reward")
        if (saved_transformed is None
                or not math.isclose(float(saved_transformed),
                                    float(transformed_reward),
                                    rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError(
                "evaluated-step recovery transformed reward differs from "
                f"the saved SPO-RS record at rollout {index}")
    divergence = spo_info.get("divergence")
    if divergence is None:
        if bool(spo_info.get("divergence_missing", False)):
            divergence = math.inf
        else:
            raise RuntimeError(
                "evaluated-step recovery has no saved SPO-RS divergence")
    divergence = float(divergence)
    if math.isnan(divergence) or divergence < 0.0:
        raise RuntimeError(
            "evaluated-step recovery found an invalid SPO-RS divergence")
    tracker_update = spo_rs_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx,
        raw_rewards=rewards)
    recovery_updates = [tracker_update]
    saved_after = spo_info.get("value_after")
    if (saved_after is not None
            and not math.isclose(float(saved_after),
                                 float(tracker_update["value_after"]),
                                 rel_tol=0.0, abs_tol=1e-10)):
        raise RuntimeError(
            "evaluated-step recovery produced a different SPO-RS tracker "
            "value; refusing an inexact update")

    if (recovered_advantages is None
            or recovered_advantages.shape != saved_advantages.shape
            or not np.allclose(recovered_advantages, saved_advantages,
                               rtol=0.0, atol=1e-10)):
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
        from entropy_tools import (begin_step, observation_descriptor,
                                   rollout_observation)
        entropy_directory = begin_step(exp_dir, step_idx)
    group_sizes = {group: len(records) for group, records in by_group.items()}
    restored_terminal_tokens = 0
    canonical_recovery_responses = 0
    unavailable_original_tokens = 0
    canonical_extra_tokens = 0
    for item in artifacts:
        meta = item["meta"]
        response_text = item["response_path"].read_text(
            encoding="utf-8", errors="replace")
        prompt_text = item["prompt_path"].read_text(
            encoding="utf-8", errors="replace")
        expected_tokens = int(meta["n_response_tokens"])
        response_tokens = item.get("response_token_ids")
        if response_tokens is None:
            try:
                response_tokens, recovery = (
                    _reconstruct_saved_response_tokens(
                        model, tokenizer, response_text, expected_tokens))
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
                "restored_terminal": bool(
                    item.get("restored_terminal_token")),
                "unavailable_original_tokens": 0,
                "canonical_extra_tokens": 0,
            }
            if (bool(recovery.get("exact", False))
                    and len(response_tokens) != expected_tokens):
                raise RuntimeError(
                    "evaluated-step recovery exact token cache changed for "
                    f"{item['response_path'].name}")
        restored_terminal = bool(recovery.get("restored_terminal", False))
        restored_terminal_tokens += int(restored_terminal)
        canonical_recovery_responses += int(
            recovery.get("source") == "canonical-text")
        unavailable_original_tokens += int(recovery.get('unavailable_original_tokens', 0) or 0)
        canonical_extra_tokens += int(recovery.get('canonical_extra_tokens', 0) or 0)
        if prompt_text not in prompt_cache:
            prompt_cache[prompt_text] = tokenizer(
                prompt_text, return_tensors="pt").input_ids
            prompt_job_ids[prompt_text] = len(prompt_job_ids)
        group = int(meta["group"])
        entropy_descriptor = None
        if entropy_directory is not None:
            entropy_descriptor = observation_descriptor(
                entropy_directory, group, int(meta["rollout"]))
        if response_tokens:
            behavior_values = item.get("behavior_logprobs")
            behavior_logprobs = (
                torch.tensor(behavior_values, dtype=torch.float32)
                if behavior_values is not None
                and len(behavior_values) == len(response_tokens)
                else None)
            all_examples.append({
                "prompt_ids": prompt_cache[prompt_text],
                "response_ids": torch.tensor(
                    [response_tokens], dtype=torch.long),
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

        code = extract_python_code(response_text)
        if entropy_descriptor is not None:
            entropy_meta = dict(meta)
            entropy_meta["n_response_tokens"] = len(response_tokens)
            entropy_observations.append(rollout_observation(
                entropy_descriptor, entropy_meta, code))

        if bool(meta.get("archive_eligible", False)):
            if not code:
                raise RuntimeError(
                    "evaluated-step recovery lost code from an archive-eligible "
                    f"rollout: {item['response_path'].name}")
            child = State.make(
                timestep=step_idx, value=float(meta["reward"]), code=code,
                raw_score=(float(meta["raw_score"])
                           if meta.get("raw_score") is not None else None),
                construction=_literal_float_list(meta.get("construction")),
            )
            if meta.get("node_id"):
                child.id = str(meta["node_id"])
            parent_group = int(meta.get("parent_group", group))
            all_children.append((child, parents[parent_group]))
            strategy_keys.append(
                int(meta["strategy_index"])
                if meta.get("strategy_index") is not None else None)

    if not all_examples:
        raise RuntimeError(
            "evaluated-step recovery found no nonempty responses to train")
    trajectory_weight = 1.0 / len(all_examples)
    for example in all_examples:
        example["sample_weight"] = trajectory_weight

    overflow_state = _load_exact_spo_rs_overflow(exp_dir, step_idx)
    recovery_examples = all_examples
    if overflow_state is not None:
        try:
            _validate_precomputed_exact_gradients(
                model, overflow_state, step_idx)
            deferred_keys = {
                (int(key[0]), int(key[1]))
                for key in overflow_state.get("deferred_keys", ())
            }
            remaining_keys = {
                (int(key[0]), int(key[1]))
                for key in overflow_state.get(
                    "remaining_keys", overflow_state.get("deferred_keys", ()))
            }
            marker_keys = {
                (int(key[0]), int(key[1]))
                for key in (_read_exact_sharded_recovery(exp_dir) or {}).get(
                    "deferred_keys", ())
            }
            if not deferred_keys or deferred_keys != marker_keys:
                raise RuntimeError(
                    "trajectory keys disagree between the marker and "
                    "gradient checkpoint")
            if not remaining_keys.issubset(deferred_keys):
                raise RuntimeError(
                    "remaining overflow trajectories are not a subset of "
                    "the original deferred trajectories")
            example_by_key = {}
            for example in all_examples:
                key = tuple(example["_overlap_key"])
                if key in example_by_key:
                    raise RuntimeError(
                        f"durable trajectory key {key} is duplicated")
                example_by_key[key] = example
            missing_keys = deferred_keys - set(example_by_key)
            if missing_keys:
                raise RuntimeError(
                    "deferred trajectories are absent from durable "
                    f"rollouts: {sorted(missing_keys)}")
            normal_count = int(
                overflow_state.get("normal_stats", {}).get(
                    "trained_examples", -1))
            if normal_count + len(deferred_keys) != len(all_examples):
                raise RuntimeError(
                    "trajectory accounting changed: retained="
                    f"{normal_count}, deferred={len(deferred_keys)}, "
                    f"recovered={len(all_examples)}")
            recovery_examples = [
                example_by_key[key] for key in sorted(remaining_keys)]
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
        strategy_top_r=(int(cfg.strategy_archive_top_r)
                        if bool(cfg.strategies) else 0),
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
        from exact_overflow import supports_warm_overflow
        recovery_lane = (
            ("grouped-checkpoint GPU replicas, then cooperative groups if needed"
             if (not overflow_state.get("replica_checkpoint_groups_attempted", False)
                 and any(hasattr(module, "_ttt_group_checkpoint_min_tokens")
                         for module in model.modules())) else
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
    best_seen = (sampler.best_raw_state(maximize=bool(problem.maximize))
                 or sampler.best_state())
    stats = {
        "planned_rollouts": sum(
            bool(item["meta"].get("counts_toward_allocation", True))
            for item in artifacts),
        "coder_format_retries": sum(
            int(item['meta'].get('retry_attempt', 0) or 0) > 0
            for item in artifacts),
        "coder_retry_enabled": bool(getattr(cfg, "coder_retry", False)),
        "result_metric_name": str(problem.metric_name),
        "result_maximize": bool(problem.maximize),
        "step_best_raw_score": (
            float(step_best.raw_score) if step_best is not None else None),
        "step_best_reward": (
            float(step_best.value) if step_best is not None else None),
        "best_seen_raw_score": (
            float(best_seen.raw_score)
            if best_seen is not None and best_seen.raw_score is not None
            else None),
        "best_seen_reward": (
            float(best_seen.value)
            if best_seen is not None and best_seen.value is not None else None),
        "best_seen_step": (
            int(best_seen.timestep) if best_seen is not None else None),
        "valid_fraction": (
            sum(bool(item["meta"]["valid"]) for item in artifacts)
            / len(artifacts)),
        "code_valid_fraction": (
            1.0 - sum(_metadata_is_code_failure(item["meta"])
                      for item in artifacts) / len(artifacts)),
        "spo_rs_updates": recovery_updates,
        "spo_rs_adaptive_beta": adaptive_spo_beta,
        "spo_rs_tracker_size": (
            0 if spo_rs_tracker is None else len(spo_rs_tracker)),
        "spo_rs_divergence": (
            None if not math.isfinite(divergence) else divergence),
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
        stats.update(parallel_trainer.recover_spo_rs_overflow(
            recovery_examples, cfg, step_idx, overflow_state))
    elif parallel_trainer is not None:
        stats.update(parallel_trainer.train_spo_rs(
            all_examples, cfg, step_idx))
    else:
        try:
            input_device = next(
                model.get_input_embeddings().parameters()).device
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

    from sampler import State
    from experiment_io import (save_parent_selections, save_rollout,
                               save_rollout_logprobs,
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
    spo_rs_mode = active_advantage_mode == "spo-rs"
    adaptive_spo_beta = bool(
        spo_rs_mode and getattr(cfg, "spo_rs_adaptive_beta", False))
    if spo_rs_mode and spo_rs_tracker is None:
        raise ValueError("SPO-RS mode requires its persistent value tracker")
    if (spo_rs_mode and (not spo_rs_tracker.initialized)):
        raise ValueError(
            "SPO-RS tracker must be initialized from pre-step history before "
            "rollouts are sampled")
    if spo_rs_mode:
        spo_rs_tracker.prepare_step()
    clipped_policy_mode = _uses_clipped_policy_loss(active_advantage_mode)
    requested_parent_contexts = (
        int(cfg.groups_per_step))
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
        sampler.record_expansion(
            parent,
            count=int(cfg.group_size))
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
    elif gen_pool is not None:
        adapter_path = _save_adapter(
            model, exp_dir, step_idx, cfg.model_name)

    def _render(messages, *, rollout_phase=None, reasoning_effort=None):
        template_kind = str(
            getattr(cfg, "coder_template_kind", "generic"))
        template_options = {}
        if template_kind == "gpt-oss":
            template_options["reasoning_effort"] = str(
                cfg.coder_reasoning_effort)
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
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

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
        count = int(cfg.group_size)
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
    strict_strategy_validation = bool(
        getattr(cfg, "strict_strategy_validation", False))
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
        normalize_response = getattr(
            problem, "normalize_strategy_response", None)
        accepted_response = (
            str(normalize_response(raw_response))
            if callable(normalize_response) else raw_response)
        save_strategy_response(
            exp_dir,
            step_idx,
            int(chain["parent_group"]),
            int(strategy_index),
            accepted_response,
        )
        strategy, extraction_issue = _extract_final_strategy(
            accepted_response, strict=strict_strategy_validation)
        if extraction_issue is None and strict_strategy_validation:
            validate_strategy = getattr(
                problem, "strategy_output_issue", None)
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
                previous_strategies=[record["strategy"]
                                     for record in chain["strategies"]]))
        messages = deepcopy(strategy_prompt_cache[key])
        if retry:
            if getattr(cfg, "api_strategy", ""):
                from strategy_prompts_openai import openai_strategy_retry_messages
                messages = openai_strategy_retry_messages(messages)
            else:
                messages = output_retry_messages(messages, "strategy")
        prompt = _render_strategy(messages, reasoning_effort=None)
        strategy_attempt_prompts[(*key, int(retry))] = prompt
        return prompt

    def _handle_strategy_attempt(chain_index, strategy_index, response, attempt):
        chain = strategy_chains[int(chain_index)]
        raw_response = str(response or '')
        normalize_response = getattr(
            problem, "normalize_strategy_response", None)
        checked_response = (
            str(normalize_response(raw_response))
            if callable(normalize_response) else raw_response)
        strategy, issue = _extract_final_strategy(
            checked_response, strict=strict_strategy_validation)
        if issue is None and strict_strategy_validation:
            validate_strategy = getattr(
                problem, "strategy_output_issue", None)
            if callable(validate_strategy):
                issue = validate_strategy(strategy)
        save_strategy_response(
            exp_dir, step_idx, int(chain["parent_group"]),
            int(strategy_index), raw_response,
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
        elif issue is not None:
            # Preserve every malformed raw attempt above, but make the
            # canonical strategy for this slot an explicit direct-solve
            # instruction.  This keeps dependent stages and coder rollout
            # counts intact without admitting malformed planner text.
            fallback_response = (
                f"<strategy>{_STRATEGY_FALLBACK}</strategy>")
            _record_strategy_response(
                chain, strategy_index, fallback_response)
            chain["strategies"][-1].update({
                "used_fallback": True,
                "fallback_issue": issue,
                "failed_attempts": int(attempt) + 1,
            })
            strategy_prompt_cache.pop(
                (int(chain["chain_id"]), int(strategy_index)), None)
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
    from host_memory import HostMemoryAdmission
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
        isolated_memory_limit_bytes = (
            _automatic_sandbox_memory_limit_bytes(n_reward_workers))
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
        with host_admission.reservation(isolated_memory_limit_bytes):
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
        token_ids = list(token_ids)
        if behavior_logprobs is not None:
            behavior_values = list(behavior_logprobs)
            behavior_logprobs = (
                behavior_values
                if len(behavior_values) == len(token_ids)
                and all(math.isfinite(float(value))
                        for value in behavior_values)
                else None)
        job = prompt_jobs[int(job_idx)]
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
            "token_ids": token_ids,
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
        """Recover a low-validity batch; never return retries to the allocator."""
        if not bool(getattr(cfg, "coder_retry", False)):
            return
        triggered, total, valid, missing, retry_budget = coder_retry_gate(
            records)
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
            failed = [
                failed[(index * len(failed)) // retry_budget]
                for index in range(retry_budget)
            ]
        prompt_indices = [coder_retry_prompt_job(
            prompt_jobs, record, _render, coder_retry_prompt_cache,
            reasoning_effort=_coder_effort_for_rollout_phase(
                cfg, record.get("strategy_rollout_phase")),
            continue_from_failure=bool(getattr(
                problem, "coder_retry_uses_failed_response", False)))
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
        # Retain every planned record, including fully valid batches, because
        # the 50% trigger is defined over the complete pilot + phase-2 set.
        if records:
            deferred_coder_retry_batches.append(
                (list(records), int(seed_offset)))

    def _drain_deferred_coder_format_retries():
        if not bool(getattr(cfg, "coder_retry", False)):
            deferred_coder_retry_batches.clear()
            return
        batches = list(deferred_coder_retry_batches)
        deferred_coder_retry_batches.clear()
        all_records = [record for records, _seed_offset in batches
                       for record in records]
        _run_coder_format_retries(all_records, seed_offset=0)

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
        """Finish one or more ready parent pilot sets exactly."""
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
            if len(result.stdout or '') > 4096:
                result.stdout = (
                    (result.stdout or '')[:4096]
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
            key = int(job["parent_group"])
            chains.setdefault(key, []).append(int(source_idx))

        followup_counts = {int(source_idx): 0
                           for source_idx in source_job_indices}
        for parent_group, chain_indices in sorted(
                chains.items()):
            chain_indices.sort(
                key=lambda index: int(prompt_jobs[index]["strategy_index"]))
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
                    + 131
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
                    f": {allocation_label}; "
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
                    f": mean threshold="
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
            key = int(diagnostic["parent_group"])
            diagnostics_by_chain.setdefault(key, []).append(diagnostic)

        for parent_group, diagnostics in sorted(
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
                    f"{parent_group}: {allocation_label}; "
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
        same parent. There is no global all-parent reward barrier: a ready
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
            key = int(job["parent_group"])
            pending_chains.setdefault(key, []).append(source_idx)
        for key, chain_indices in pending_chains.items():
            chain_indices.sort(
                key=lambda index: int(
                    prompt_jobs[index]["strategy_index"]))
            if len(chain_indices) != strategies_per_parent:
                raise RuntimeError(
                    f"parent {key} has "
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
    coder_phase_complete = False
    try:
        try:
            api_planned = bool(two_stage_rollouts and
                               getattr(cfg, "strategy_backend", "local") == "api")
            if api_planned:
                from strategy_api import run_strategy_rounds
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
                                    "parent chain")
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
                            # Legacy local pools retain the compatible
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
                        active = _coder_effort_entries(
                            prompt_jobs, source_job_indices, counts, cfg,
                            step_idx, phase)
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
                cap_state = {"value": int(getattr(cfg, '_local_generation_cap', 0) or 0)}

                def _seed_local_generation(seed_step):
                    if cfg.deterministic:
                        torch.manual_seed((int(cfg.seed) * 1_000_003
                                           + int(seed_step) * 1009 + 13)
                                          % (2**31 - 1))

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
                                    raise RuntimeError("strategy generation omitted a parent chain")
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
                        active = _coder_effort_entries(
                            prompt_jobs, source_job_indices, counts, cfg,
                            step_idx, phase)
                        if not active:
                            return []
                        active_indices = [item[0] for item in active]
                        active_counts = [item[1] for item in active]
                        active_prompt_indices = [
                            _phase_coder_prompt_job(
                                prompt_jobs, index, phase, cfg, _render,
                                phase_prompt_job_cache, effort=effort)
                            for index, _count, effort in active]
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
                            for existing, retry in zip(
                                current_policy_scores, rescored_current)
                        ]
                    for record, values in zip(
                            vllm_logprob_records, current_policy_scores):
                        if values is None or len(values) != len(
                                record["token_ids"]):
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
                        spo_rs_context_divergences = {
                            int(record["context_id"]): 0.0
                            for record in vllm_logprob_records
                        }
                    elif (spo_rs_tracker.initialized):
                        if (spo_rs_previous_adapter_path is None
                                or not spo_rs_previous_adapter_path.is_dir()):
                            raise FileNotFoundError(
                                "SPO-RS cannot compute the consecutive-policy "
                                "KL because its previous sampling adapter is "
                                f"missing: {spo_rs_previous_adapter_path}")
                        else:
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
                    0 if spo_rs_mode else len(vllm_logprob_records) - reference_count)
                eval_done_after = sum(
                    future.done() for futures in reward_futures.values()
                    for future in futures)
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
                    and not getattr(cfg,
                        "dual_resident_qwen3_8b_strategy_coder_pools", False))
                if immediate_restore:
                    gen_pool.release_for_training(parallel_trainer)
                else:
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
                        _prepare_spo_rs_overlap_inputs(
                            tokenizer, group_responses, reward_futures,
                            prompt_jobs))
                    if staged_spo_rs_examples:
                        pending_before = sum(
                            not future.done() for future in all_futs)
                        pending_by_future = {
                            example["_overlap_reward_future"]: example
                            for example in staged_spo_rs_examples
                        }
                        if (len(pending_by_future)
                                != len(staged_spo_rs_examples)):
                            raise RuntimeError(
                                "SPO-RS overlap found a reward future shared "
                                "by multiple rollout records")
                        spo_rs_overlap_state = (
                            parallel_trainer.begin_spo_rs_overlap(
                                cfg, step_idx))
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
                            ready_futures = [
                                future for future in pending_by_future
                                if future.done()
                            ]
                            if not ready_futures:
                                wait(tuple(pending_by_future),
                                     return_when=FIRST_COMPLETED)
                                continue
                            ready_examples = [
                                pending_by_future.pop(future)
                                for future in ready_futures
                            ]
                            _complete_spo_rs_overlap_examples(
                                ready_examples, spo_rs_tracker)
                            parallel_trainer.queue_spo_rs_overlap(
                                ready_examples)

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

        with _policy_dropout_disabled(model):
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
                raw_rewards=[sample["reward"] for sample in samples],
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
            prepared = [spo_rs_prepared[(g, r_idx)]
                        for r_idx in range(len(responses))]
            advantages = np.asarray(
                [item["advantage"] for item in prepared],
                dtype=np.float64)
            spo_rs_trainable = [item["trainable"] for item in prepared]
            spo_rs_rollout_info = [item["info"] for item in prepared]
            adv_scale = float(cfg.spo_rs_beta)
            adv_scale_label = "beta"
        else:
            advantages, adv_scale = compute_group_advantages(rewards_np)
            adv_scale_label = "beta"
        constant = (
            float(rewards_np.max() - rewards_np.min()) < 1e-12)

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
                "sandbox_stdout": (res.stdout or '')[:2000],
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
            if spo_rs_mode:
                meta["spo_rs"] = spo_rs_rollout_info[r_idx]
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

        # Entropic training skips constant groups. SPO-RS compares every
        # rollout with its historical baseline, so tied groups can still learn.
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
                record.get("_entropic_overlap_example")
                or record.get("_spo_rs_overlap_example"))
            res = outs[r_idx]
            if prepared is not None:
                prepared_advantage = prepared.get("advantage")
                if (prepared_advantage is not None
                        and float(prepared_advantage) != float(adv)):
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
        # SPO-RS requires the frozen behavior probability. A malformed vLLM
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

    if spo_rs_mode and all_examples:
        trajectory_weight = 1.0 / len(all_examples)
        for example in all_examples:
            example["sample_weight"] = trajectory_weight


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

    if spo_rs_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(
            example.get("advantage") is not None
            and math.isfinite(float(example["advantage"]))
            for example in all_examples)
        advantages_match = not any(
            example.get("_overlap_label_mismatch", False)
            for example in all_examples)
        expected_keys = spo_rs_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys
                or not advantages_complete or not advantages_match):
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
    training_label = (str(len(all_examples))
                      if training_enabled else "disabled")
    rollout_phase_label = (
        "rollout+eval+overlapped-backward wall time"
        if (entropic_overlap_state is not None
            or spo_rs_overlap_state is not None)
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
                f"target={target:.12f}, beaten by {margin:.12f} 🏆")
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
    if spo_rs_mode:
        step_stats["spo_rs_updates"] = spo_rs_updates
        step_stats["spo_rs_adaptive_beta"] = adaptive_spo_beta
        step_stats["spo_rs_tracker_size"] = (
            0 if spo_rs_tracker is None else len(spo_rs_tracker))
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
    # real backward update to perform.
    if ensure_trainer_ready is not None:
        ensure_trainer_ready()


    if clipped_policy_mode:
        if spo_rs_overlap_state is not None:
            step_stats.update(parallel_trainer.finish_spo_rs_overlap(
                cfg, step_idx, len(all_examples), apply_update=True))
            spo_rs_overlap_state = None
        elif parallel_trainer is not None:
            step_stats.update(parallel_trainer.train_spo_rs(
                all_examples, cfg, step_idx))
        else:
            step_stats.update(_train_spo_rs_examples(
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
        best = (sampler.best_raw_state(maximize=bool(problem.maximize))
                or sampler.best_state())
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

    # Save the same problem-native best-ever state reported after each step.
    # The independent raw record includes seeds and candidates removed by
    # exact-code deduplication/archive pruning; ``best_state()`` alone can
    # therefore save an older or worse stochastic realization.
    best = (sampler.best_raw_state(maximize=bool(problem.maximize))
            or sampler.best_state())
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
        # This must precede every import path that can initialize CUDA. Worker
        # and evaluation children set their own physical groups before import.
        _pin_training_process(cfg.training_gpu_ids)
        if cfg.evaluation_gpu_id is not None:
            os.environ["TTT_EVALUATION_GPU_ID"] = str(cfg.evaluation_gpu_id)

        # Save the reusable base configuration before runtime-only adjustments.
        from experiment_io import (append_step_result, make_experiment_dir,
                                   save_best_state_snapshot,
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
    if cfg.generation_backend == "vllm":
        from host_memory import validate_memory_handoff_runtime
        runtime = validate_memory_handoff_runtime(__file__)
        with (Path(exp_dir) / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(runtime, handle, indent=2)
        runtime_message = (
            f"[runtime] checkout={runtime['checkout']}; "
            f"memory handoff: {runtime['memory_handoff']}")
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
    # Entropic updates use isolated GPU processes even
    # without --fast.  The former same-interpreter threaded replicas are not a
    # safe execution boundary for concurrent autograd.  --fast remains an
    # optional scheduling choice: it enables adaptive batches; without it the
    # configured microbatch ceiling is honored.  Clipped-policy modes retain
    # their existing default trainer unless --fast is explicitly requested.
    standard_policy_objective = not bool(
        _uses_clipped_policy_loss(cfg.advantage_mode))
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
        print(f"Strategy model:     {_strategy_model_name(cfg)} "
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
    print(f"Prompt target:      {cfg.target}")
    print(f"Print/SOTA target:  {getattr(cfg, 'target_print', cfg.target)}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Search selection:   {'UCT' if cfg.uct else 'PUCT'} "
          f"(c={cfg.puct_c})")
    configured_advantage_mode = getattr(cfg, "advantage_mode", "entropic")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    if configured_advantage_mode == "spo-rs":
        print("Reference KL:        off (SPO-RS PPO objective)")
    else:
        print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Advantage mode:     "
          f"{getattr(cfg, 'advantage_mode', 'entropic')}")
    if _uses_sequence_level_policy_ratio(cfg):
        print("MoE policy ratio:    sequence-level geometric mean "
              "(Qwen3-30B-A3B only; existing asymmetric low/high bounds)")
    if configured_advantage_mode == "spo-rs":
        if bool(getattr(cfg, "spo_rs_adaptive_beta", False)):
            print(f"SPO-RS shaping:     historical arctan, width=1.25; "
                  f"beta={cfg.spo_rs_beta}; all rollouts train")
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
        pilot_count = int(getattr(
            cfg, "pilot_programs_per_strategy", -1))
        if (getattr(cfg, "coder_template_kind", "generic") == "qwen3.8"
                and pilot_count > 0
                and int(getattr(
                    cfg, "pilot_xhigh_programs_per_strategy", 0)) > 0):
            xhigh_count = min(
                int(cfg.pilot_xhigh_programs_per_strategy), pilot_count)
            print(
                "Coder reasoning:    pilot/strategy="
                f"{xhigh_count} xhigh + {pilot_count - xhigh_count} "
                f"{base_effort}; phase 2={base_effort}")
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
            "only enough to target 60% when planned validity is below 50%"
            if bool(getattr(cfg, "coder_retry", False)) else
            "off (enable with --coder-retry)"
        )
        strategy_validation_label = (
            "strict" if bool(getattr(cfg, "strict_strategy_validation", False))
            else "previous block-only")
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
                print("[resume] legacy batch-growth state ignored; "
                      "using fixed configured batch sizes")
            if configured_advantage_mode == "spo-rs":
                interrupted_spo_rs_artifacts = (
                    _recoverable_spo_rs_artifacts(
                        exp_dir, start_step, cfg))
                if interrupted_spo_rs_artifacts is not None:
                    token_recovery = (
                        _prepare_recoverable_response_tokens(
                            model, tokenizer,
                            interrupted_spo_rs_artifacts))
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
                interrupted_spo_rs_artifacts = (
                    _recoverable_spo_rs_artifacts(exp_dir, 0, cfg))
            initial_adapter = Path(exp_dir) / "adapter_initial"
            if (interrupted_spo_rs_artifacts is not None
                    and initial_adapter.is_dir()):
                start_step = 0
                adapter_path = initial_adapter
                _load_adapter(model, adapter_path)
                token_recovery = _prepare_recoverable_response_tokens(
                    model, tokenizer, interrupted_spo_rs_artifacts)
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
    if (configured_advantage_mode == 'spo-rs'):
        from spo_rs import BoundedSPORSTracker, SPORSTracker
        bounded_shaping = bool(getattr(cfg, "spo_rs_adaptive_beta", False))
        tracker_class = BoundedSPORSTracker if bounded_shaping else SPORSTracker
        spo_rs_tracker = tracker_class(
            entropic_beta=cfg.spo_rs_beta,
            d_half=cfg.spo_rs_d_half,
            rho_min=cfg.spo_rs_rho_min,
            rho_max=cfg.spo_rs_rho_max,
            **({"fail_score": cfg.fail_score} if bounded_shaping else {}),
        )
        tracker_state = (resume_payload.get("spo_rs_tracker")
                         if resume_payload is not None else None)
        if tracker_state is not None and (
                bounded_shaping or tracker_state.get("version") == 2):
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
    # Generation callbacks consult it so all later steps follow the genuine
    # no-training residency path instead of waiting for a nonexistent update.
    runtime_training_active = {"value": not bool(cfg.no_train)}
    if (getattr(problem, "two_stage_rollouts", False)
            and cfg.strategy_backend == "api"):
        from strategy_api import StrategyAPIGenerationPool
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
            from host_memory import HostMemoryAdmission
            import multiprocessing
            pool_options["host_memory_admission"] = (
                getattr(parallel_trainer, "host_memory_admission", None)
                or HostMemoryAdmission(multiprocessing.get_context("spawn")))
            trainer_offloaded = False

            def _offload_trainer_for_generation(phase="coder"):
                nonlocal trainer_offloaded
                if (parallel_trainer is not None
                        and parallel_trainer.is_offloaded):
                    return
                if parallel_trainer is None and trainer_offloaded:
                    return
                from host_memory import (
                    GIB, estimated_module_bytes, estimated_optimizer_bytes,
                    host_memory_info, reclaim_unused_host_memory,
                )

                # Reclaim cached activation-offload blocks on *every* trainer
                # process before allocating complete CPU model replicas. This
                # is a phase-boundary operation: it leaves live tensors and
                # gradients untouched and runs only once per rollout step.
                if parallel_trainer is not None:
                    cleanups = parallel_trainer.reclaim_host_memory()
                    required_bytes = (
                        parallel_trainer.estimated_host_offload_bytes())
                else:
                    torch.cuda.synchronize()
                    cleanups = [reclaim_unused_host_memory(
                        torch,
                        cuda_devices=tuple(range(torch.cuda.device_count())),
                        force=True,
                    )]
                    required_bytes = int(
                        1.05 * (
                            estimated_module_bytes(model)
                            + estimated_optimizer_bytes(optimizer))
                        + GIB
                    )

                reclaimed_bytes = sum(max(
                    0,
                    int(cleanup.get("pinned_cached_before", 0))
                    - int(cleanup.get("pinned_cached_after", 0)),
                ) for cleanup in cleanups)
                memory = host_memory_info()
                if reclaimed_bytes >= GIB:
                    available = (
                        "unknown" if memory is None else
                        f"{memory['available'] / GIB:.1f} GiB")
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
                        discard = getattr(
                            pool, "discard_sleeping_host_residency", None)
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
                        parallel_trainer.offload_for_generation(
                            purge_host_cache=False)
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
                if (parallel_trainer is not None
                        and not parallel_trainer.is_offloaded):
                    trainer_offloaded = False
                    return
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
                from host_memory import reclaim_unused_host_memory
                reclaim_unused_host_memory(
                    torch,
                    cuda_devices=tuple(range(torch.cuda.device_count())),
                    force=True,
                )
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
                from host_memory import reclaim_unused_host_memory
                reclaim_unused_host_memory(
                    torch,
                    cuda_devices=tuple(range(torch.cuda.device_count())),
                    force=True,
                )
                trainer_offloaded = False

            if cfg.training_layout == "sharded":
                ensure_trainer_ready = _ensure_sharded_trainer_ready

            gen_pool = PhasedVLLMGenerationPool(
                before_start=lambda: _offload_trainer_for_generation(
                    "coder"),
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
                strategy_persistent_workers = getattr(
                    cfg, "strategy_vllm_persistent_workers", None)
                if str(getattr(
                        cfg, "coder_model_profile", "")).lower() == (
                            "gpt-oss-120b"):
                    strategy_persistent_workers = 0
                    setting_log_only(
                        "[memory] GPT-OSS-120B coder: local strategist "
                        "engines use transient residency so coder wake-up "
                        "retains the full 131k-context memory budget",
                        flush=True,
                    )
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
                    "vllm_persistent_workers": strategy_persistent_workers,
                    "vllm_staged_loading": (
                        cfg.strategy_vllm_staged_loading),
                    "vllm_log_path": strategy_vllm_log_path,
                })
                strategy_pool = PhasedVLLMGenerationPool(
                    before_start=lambda: _offload_trainer_for_generation(
                        "strategy"),
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
            completed_summary.get("training_seconds", 0.0) or 0.0)
    if cumulative_training_seconds:
        print(f"[resume] prior cumulative training time: "
              f"{cumulative_training_seconds:.1f}s")
    restart_for_exact_recovery = False
    restart_after_exact_recovery = False
    active_step = None
    try:
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

            if (resume_dir
                    and step == start_step
                    and configured_advantage_mode == "spo-rs"):
                pending_spo_rs_artifacts = None
                if interrupted_spo_rs_artifacts is None:
                    interrupted_spo_rs_artifacts = (
                        _recoverable_spo_rs_artifacts(exp_dir, step, step_cfg))
                if interrupted_spo_rs_artifacts is None:
                    pending_spo_rs_artifacts = (
                        _recoverable_pending_spo_rs_artifacts(
                            exp_dir, step, step_cfg))
            else:
                interrupted_spo_rs_artifacts = None
                pending_spo_rs_artifacts = None

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
            step_training_seconds = float(
                stats.get("training_seconds", 0.0) or 0.0)
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
            # Snapshot the sampler's independent native-metric optimum before
            # asynchronous plotting starts.  This includes unbeaten seeds and
            # evaluated children that no longer survive archive pruning.
            best_seen_state = (
                sampler.best_raw_state(maximize=bool(problem.maximize))
                or sampler.best_state()
            )
            best_snapshot_path = save_best_state_snapshot(
                exp_dir, step, best_seen_state,
                metric_name=str(problem.metric_name),
                maximize=bool(problem.maximize),
                construction=(
                    _as_float_list(
                        getattr(best_seen_state, "construction", None),
                        cfg.max_saved_construction)
                    if (best_seen_state is not None
                        and getattr(problem, "saves_construction", False))
                    else None),
            )
            step_summary["best_seen_snapshot"] = (
                str(Path(best_snapshot_path).relative_to(exp_dir))
                if best_snapshot_path is not None else None)
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
            if (stats.get("exact_overflow_inprocess", False)
                    or getattr(cfg, "_exact_inprocess_recovery_step", None) == step):
                # Keep partial gradients until the *full* optimizer/sampler
                # checkpoint is durable. A crash before this point can resume
                # the same update; a successful step needs no process restart.
                _clear_exact_sharded_recovery(exp_dir)
                if hasattr(cfg, "_exact_inprocess_recovery_step"):
                    delattr(cfg, "_exact_inprocess_recovery_step")
            replica_capture_step = getattr(
                cfg, "_exact_replica_capture_step", None)
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
            replica_recovery_step = getattr(
                cfg, "_exact_replica_recovery_step", None)
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
            recovery_step = getattr(
                cfg, "_exact_sharded_recovery_step", None)
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
        if (active_step is None
                or configured_advantage_mode != "spo-rs"):
            raise
        recoverable = _recoverable_spo_rs_artifacts(
            exp_dir, active_step, cfg)
        if recoverable is None:
            raise RuntimeError(
                "the exact long-context trainer needs a multi-GPU retry, but "
                "this step does not yet have a complete durable SPO-RS "
                "rollout/evaluation set") from error
        marker = _write_exact_sharded_recovery(
            exp_dir, active_step, error)
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
        if (parallel_trainer is not None
                and hasattr(parallel_trainer, "shutdown")):
            print("[shutdown] stopping process-distributed trainer ...")
            parallel_trainer.shutdown(for_restart=(
                restart_for_exact_recovery
                or restart_after_exact_recovery))

    if restart_for_exact_recovery or restart_after_exact_recovery:
        _restart_for_exact_resume(exp_dir)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    # Use the problem-native best-ever record, not merely the best state that
    # survived archive deduplication/pruning.
    best = (sampler.best_raw_state(maximize=bool(problem.maximize))
            or sampler.best_state())
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
