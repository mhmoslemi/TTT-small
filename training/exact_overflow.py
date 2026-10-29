"""Warm, in-process multi-GPU execution for oversized training trajectories.

Ordinary examples still use the existing replica/work-stealing path. Only its
overflow tail borrows GPUs from parked replicas. No model is loaded again, no
process is restarted, and an unsuccessful group is retried with more GPUs,
never fewer tokens. Gradient reduction and the single optimizer update remain
the coordinator's responsibility.
"""

from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time


METRICS = ("loss", "policy_loss", "kl_estimate", "entropy_estimate",
           "ratio", "clipped_fraction")


def supports_warm_overflow(cfg):
    # Quantized/custom backends may replace Parameters when moved. Do not
    # silently apply this placement protocol to those models.
    return (str(getattr(cfg, "backend", "")) == "hf"
            and not bool(getattr(cfg, "load_in_4bit", False))
            and not bool(getattr(cfg, "effective_load_in_4bit", False))
            and int(getattr(cfg, "num_training_gpus", 1)) > 1
            and str(getattr(cfg, "training_layout", "")) == "replicated")


def trajectory_key(example):
    return tuple(int(value) for value in example["_overlap_key"])


def plan_overflow_wave(examples, world_size, minimum_gpus=None):
    """Disjoint groups, expensive work first; every GPU joins a live group.

    One trajectory per group avoids padding and preserves its full context.
    Successful keys are removed by the caller; only failed keys get wider
    groups in a subsequent wave.
    """
    world_size = int(world_size)
    if world_size < 2:
        raise ValueError("cooperative overflow requires at least two GPUs")
    minimum_gpus = minimum_gpus or {}
    ordered = sorted(examples, key=lambda item: (
        int(minimum_gpus.get(trajectory_key(item), 2)),
        int(item["prompt_ids"].shape[1] + item["response_ids"].shape[1])),
        reverse=True)
    selected, widths, available = [], [], world_size
    for example in ordered:
        required = max(2, int(minimum_gpus.get(trajectory_key(example), 2)))
        if required > world_size:
            raise RuntimeError("trajectory exhausted all available GPU shards")
        if required <= available:
            selected.append(example)
            widths.append(required)
            available -= required
    if not selected:
        return []
    while available:
        index = min(range(len(widths)), key=lambda i: widths[i])
        widths[index] += 1
        available -= 1
    wave, offset = [], 0
    for example, width in zip(selected, widths):
        wave.append((example, tuple(range(offset, offset + width))))
        offset += width
    return wave


def require_parking_capacity(model, world_size, optimizer=None, *, generation_pools=()):
    """Reclaim sleeping inference backups before admitting CPU trainer copies."""
    from runtime.host_memory import (host_memory_info, GIB, estimated_module_bytes,
                             estimated_optimizer_bytes)

    weight_bytes = estimated_module_bytes(model)
    optimizer_bytes = estimated_optimizer_bytes(optimizer)
    required = int(1.10 * (weight_bytes * int(world_size) + optimizer_bytes))
    info = host_memory_info()
    if info is None:
        raise RuntimeError("cannot safely park replicas: host RAM is unknown")
    reserve = max(128 * GIB, int(0.10 * info["total"]))
    for pool in generation_pools:
        if required + reserve <= info["available"]:
            break
        discard = getattr(pool, "discard_sleeping_host_residency", None)
        if not callable(discard) or not discard():
            continue
        print("[train-overflow] released a sleeping generation pool's host backup "
              "to park trainer replicas; it will reload on its next use", flush=True)
        info = host_memory_info()
        if info is None:
            raise RuntimeError("cannot safely park replicas: host RAM is unknown")
        reserve = max(128 * GIB, int(0.10 * info["total"]))
    if required + reserve > info["available"]:
        raise RuntimeError(
            "insufficient host RAM to park the trainer replicas safely: "
            f"need {required / GIB:.1f} GiB plus {reserve / GIB:.1f} GiB "
            f"reserve, available {info['available'] / GIB:.1f} GiB; "
            "completed gradients are saved, no trajectory was discarded")


@contextmanager
def temporary_gpu_shards(backend, gpu_ids):
    """Redistribute a parked model without replacing optimizer Parameters.

    Accelerate's dispatch can replace Parameters during CPU->GPU placement.
    Move their *storage* first, then attach its normal autograd-enabled routing
    hooks. Finally return to the parked CPU layout; the coordinator restores
    all replicas only after every group has relinquished its borrowed GPUs.
    """
    import torch
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_submodules
    from accelerate.utils import get_balanced_memory
    from training.fast_distributed import set_long_rollout_memory_ceiling
    from runtime.host_memory import reclaim_unused_host_memory

    model = backend.model
    placement = getattr(backend, "_placement_model", model)
    gpu_ids = tuple(int(device) for device in gpu_ids)
    if not getattr(backend, "_trainer_is_offloaded", False):
        raise RuntimeError("overflow placement requires a parked trainer")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("overflow placement found a non-CPU parked parameter")
    original_parameters = dict(model.named_parameters())
    original_map = getattr(placement, "hf_device_map", None)
    saved_methods = {name: placement.__dict__.get(name)
                     for name in ("to", "cuda")}
    original_profile = getattr(model, "_ttt_oom_execution_profile", None)
    # A replica's failed memory profile is not the profile of this GPU group.
    model._ttt_oom_execution_profile = {}
    try:
        budgets = {}
        for device in gpu_ids:
            set_long_rollout_memory_ceiling(device, 0.96)
            with torch.cuda.device(device):
                free, total = torch.cuda.mem_get_info()
            # This is a weight-placement ceiling, not a token/activation cap.
            budgets[device] = max(0, int(free - 0.08 * total))
        no_split = list(getattr(placement, "_no_split_modules", None) or ())
        balanced = get_balanced_memory(
            placement, max_memory=budgets, no_split_module_classes=no_split)
        device_map = infer_auto_device_map(
            placement, max_memory={**balanced, "cpu": 0},
            no_split_module_classes=no_split)
        if any(target not in gpu_ids for target in device_map.values()):
            raise torch.cuda.OutOfMemoryError(
                "overflow group cannot hold its complete model on GPU")
        # Updating .data preserves the identities owned by the existing Adam
        # optimizer, including tied Parameters. No dtype conversion is made.
        def target_for(name):
            matches = [(prefix, target) for prefix, target in device_map.items()
                       if not prefix or name == prefix or name.startswith(prefix + ".")]
            return max(matches, key=lambda item: len(item[0]))[1]

        with torch.no_grad():
            for name, parameter in placement.named_parameters():
                parameter.data = parameter.data.to(target_for(name))
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.to(parameter.device)
            for name, buffer in list(placement.named_buffers()):
                parent, _, leaf = name.rpartition(".")
                module = placement.get_submodule(parent) if parent else placement
                module._buffers[leaf] = buffer.to(target_for(name))
        dispatch_model(placement, device_map=device_map,
                       main_device=gpu_ids[0], force_hooks=True)
        if any(dict(model.named_parameters()).get(name) is not parameter
               for name, parameter in original_parameters.items()):
            raise RuntimeError("overflow placement replaced a model Parameter")
        yield
    finally:
        remove_hook_from_submodules(placement)
        for name, original in saved_methods.items():
            if original is None:
                placement.__dict__.pop(name, None)
            else:
                setattr(placement, name, original)
        # Do not call Module.to here: preserve optimizer-owned objects even
        # under runtimes configured to replace Parameters on device moves.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.data = parameter.data.cpu()
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.cpu()
            for module in model.modules():
                for name, buffer in module._buffers.items():
                    if buffer is not None:
                        module._buffers[name] = buffer.cpu()
        if original_map is None:
            placement.__dict__.pop("hf_device_map", None)
        else:
            placement.hf_device_map = original_map
        model._ttt_oom_execution_profile = original_profile or {}
        gc.collect()
        for device in gpu_ids:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        reclaim_unused_host_memory(torch, cuda_devices=gpu_ids, force=True)


def backward_overflow_example(backend, tokenizer, example, cfg, gpu_ids):
    """One unchanged trajectory loss; leave its gradients on the parked model."""
    import torch
    import training.trainers as training
    from training.fast_distributed import set_long_rollout_memory_ceiling

    model = backend.model
    model.zero_grad(set_to_none=True)
    item = dict(example)
    try:
        with temporary_gpu_shards(backend, gpu_ids):
            backend.set_training_mode()
            input_device = next(model.get_input_embeddings().parameters()).device
            for key in ("prompt_ids", "response_ids"):
                item[key] = item[key].to(input_device)
            epsilon, low, high, kl = training._clipped_policy_options(cfg)
            missing_old = training._initialize_behavior_logprobs([item])
            with training._policy_dropout_disabled(model):
                if missing_old:
                    item["old_policy_logprobs"] = training.compute_batched_token_logprobs(
                        model, [item], False, cfg.logprob_chunk)[0].detach()

                def attempt(_batches):
                    result = training.compute_batched_token_logprobs(
                        model, [item], True, cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id,
                        measure_entropy=True)
                    current = result[0]
                    loss, metrics = training.clipped_policy_loss(
                        cfg, current, item["old_policy_logprobs"],
                        None, item["advantage"],
                        clip_epsilon=epsilon, clip_epsilon_low=low,
                        clip_epsilon_high=high, kl_coef=kl)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("nonfinite overflow policy loss")
                    weight = float(item["sample_weight"])
                    (weight * loss).backward()
                    totals = {name: weight * float(metrics[name]) for name in (
                        "policy_loss", "kl_estimate", "entropy_estimate", "ratio")}
                    totals.update(loss=weight * float(loss.detach()),
                                  clipped_fraction=weight * float(metrics["clipped"]))
                    return dict(totals=totals, max_ratio=float(metrics["ratio"]),
                                max_prefix_ratio=float(metrics["prefix_ratio_max"]),
                                entropy_examples=0, backward_batches=1)

                stages = iter((0.98, 0.99))

                def expand():
                    fraction = next(stages, None)
                    if fraction is not None:
                        for device in gpu_ids:
                            set_long_rollout_memory_ceiling(device, fraction)
                    return fraction

                result, effective, deferred = training._run_oom_resilient_backward(
                    model, [[item]], attempt,
                    device_label=f"cooperative GPUs {list(gpu_ids)}",
                    expand_memory=expand, start_checkpointed=True,
                    allow_activation_offload=False, defer_exact_oom=True)
                if deferred or not effective or result is None:
                    model.zero_grad(set_to_none=True)
                    return {"completed": False}
                return {"completed": True, "stats": result}
    except BaseException as error:
        model.zero_grad(set_to_none=True)
        if training._is_cuda_oom(error):
            return {"completed": False}
        raise
    finally:
        item.clear()


def compute_overflow_work(backend, tokenizer, example, cfg, gpu_ids):
    """Export only LoRA gradients, after the borrowed GPUs are released."""
    result = backward_overflow_example(backend, tokenizer, example, cfg, gpu_ids)
    if result["completed"]:
        gradients = {
            name: parameter.grad.detach()
            for name, parameter in backend.model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        result["gradient_pack"] = pack_cpu_gradients(gradients)
    backend.model.zero_grad(set_to_none=True)
    return result


def pack_cpu_gradients(gradients):
    """One IPC storage/FD per dtype, not thousands of per-Parameter FDs."""
    import torch

    groups, entries = {}, []
    for name, gradient in gradients.items():
        if gradient.device.type != "cpu":
            raise RuntimeError("overflow gradient transport must be CPU-only")
        groups.setdefault(gradient.dtype, []).append((name, gradient))
    buffers = []
    for buffer_index, group in enumerate(groups.values()):
        offset = 0
        for name, gradient in group:
            entries.append((name, buffer_index, offset, tuple(gradient.shape)))
            offset += gradient.numel()
        buffers.append(torch.cat([value.detach().reshape(-1) for _, value in group]))
    return {"buffers": buffers, "entries": entries}


def unpack_cpu_gradients(packed):
    import math

    buffers = packed["buffers"]
    if any(buffer.device.type != "cpu" or buffer.ndim != 1 for buffer in buffers):
        raise RuntimeError("invalid overflow gradient transport buffer")
    result = {}
    for name, index, offset, shape in packed["entries"]:
        if name in result or not 0 <= index < len(buffers):
            raise RuntimeError("invalid/duplicate overflow gradient entry")
        count = math.prod(shape)
        if offset < 0 or count < 0 or offset + count > buffers[index].numel():
            raise RuntimeError("overflow gradient entry exceeds its buffer")
        result[name] = buffers[index][offset:offset + count].view(shape)
    return result


def complete_warm_overflow(trainer, examples, cfg, step_idx, state):
    """Finish the missing gradients inside the existing job and update once."""
    import time
    import torch
    import training.trainers as training
    from training.fast_distributed import broadcast_trainable_parameters

    if state is None:
        raise RuntimeError("warm overflow requires the saved full-batch gradient state")
    if trainer._closed or not trainer._workers_idle:
        raise RuntimeError("replicas must finish their current work before borrowing GPUs")
    training._validate_precomputed_exact_gradients(trainer.model, state, step_idx)
    from training.context_parallel import complete_context_overflow

    context_result = complete_context_overflow(trainer, examples, cfg, step_idx, state)
    if context_result is not None:
        return context_result
    by_key = {}
    for item in examples:
        key = trajectory_key(item)
        if key in by_key:
            raise RuntimeError(f"duplicate overflow trajectory key: {key}")
        by_key[key] = item
    pending = {tuple(key) for key in state["remaining_keys"]}
    if pending - by_key.keys():
        raise RuntimeError("warm overflow is missing an original trajectory")
    examples = trainer._cpu_examples([by_key[key] for key in sorted(pending)])
    by_key = {trajectory_key(item): item for item in examples}
    completed_before = tuple(state["completed_keys"])
    previous_overflow_seconds = float(state["overflow_stats"]["training_seconds"])
    initial_pending = len(pending)
    started = time.time()
    minimum_gpus = {}
    if pending:
        smallest = min(int(item["prompt_ids"].shape[1] + item["response_ids"].shape[1])
                       for item in examples)
        # This is execution routing, not a context limit. Later equally long
        # trajectories skip the already-failed single-GPU ladder altogether.
        trainer._warm_overflow_min_tokens = min(
            int(getattr(trainer, "_warm_overflow_min_tokens", smallest)), smallest)
    waves = 0
    try:
        if pending:
            # Clear dead activation caches before checking the parking budget.
            trainer.reclaim_host_memory()
            if not trainer._offloaded:
                require_parking_capacity(trainer.model, trainer.world_size,
                                         trainer.optimizer,
                                         generation_pools=getattr(trainer, "_generation_pools", ()))
                trainer.offload_for_generation(purge_host_cache=False)
            # All replica weights remain resident in host RAM between waves.
            while pending:
                wave = plan_overflow_wave(
                    [by_key[key] for key in pending], trainer.world_size,
                    minimum_gpus)
                waves += 1
                layout = ", ".join(
                    f"{trajectory_key(item)}->{list(devices)}"
                    for item, devices in wave)
                print(f"[train-overflow] step {step_idx}: warm wave {waves}: "
                      f"{layout}; full trajectories, no worker restart", flush=True)
                wave_started = time.time()
                trainer._workers_idle = False
                child_leaders = []
                for item, devices in wave:
                    leader = devices[0]
                    if leader:
                        child_leaders.append(leader)
                        trainer._command_queues[leader].put({
                            "kind": "overflow_backward", "example": item,
                            "gpu_ids": devices, "step_cfg": dict(vars(cfg)),
                        })
                item, devices = wave[0]
                results = {0: compute_overflow_work(
                    trainer.backend, trainer.tokenizer, item, cfg, devices)}
                messages = trainer._collect_event("overflow_computed", child_leaders)
                results.update({rank: message["result"]
                                for rank, message in messages.items()})
                trainer._workers_idle = True
                # The long work has no outstanding NCCL collective: workers
                # can spend arbitrarily long here without a broadcast timeout.
                successful, failed, stats = set(), [], []
                parameters = {name: value for name, value in
                              trainer.model.named_parameters() if value.requires_grad}
                trainer.model.zero_grad(set_to_none=True)
                for item, devices in wave:
                    key = trajectory_key(item)
                    result = results[devices[0]]
                    if not result["completed"]:
                        failed.append((key, len(devices)))
                        continue
                    gradients = unpack_cpu_gradients(result["gradient_pack"])
                    if set(gradients) - parameters.keys():
                        raise RuntimeError("overflow worker returned unknown gradients")
                    with torch.no_grad():
                        for name, value in gradients.items():
                            parameter = parameters[name]
                            if (value.device.type != "cpu" or value.shape != parameter.shape
                                    or value.dtype != parameter.dtype
                                    or not torch.isfinite(value).all()):
                                raise RuntimeError(f"invalid overflow gradient: {name}")
                            if parameter.grad is None:
                                # Own the accumulator; never mutate a worker's
                                # multiprocessing-shared transport storage.
                                parameter.grad = value.clone()
                            else:
                                parameter.grad.add_(value)
                    successful.add(key)
                    stats.append(result["stats"])
                if successful:
                    combined = {
                        "totals": {name: sum(item["totals"][name] for item in stats)
                                   for name in METRICS},
                        "max_ratio": max(item["max_ratio"] for item in stats),
                        "max_prefix_ratio": max(item["max_prefix_ratio"] for item in stats),
                        "entropy_examples": sum(item["entropy_examples"] for item in stats),
                        "backward_batches": len(successful),
                    }
                    training._commit_exact_spo_rs_overflow_batch(
                        trainer.model, state, step_idx, successful, combined,
                        training_seconds=time.time() - wave_started)
                    pending.difference_update(successful)
                trainer.model.zero_grad(set_to_none=True)
                results.clear()
                messages.clear()
                for key, width in failed:
                    if width == trainer.world_size:
                        raise RuntimeError(
                            f"trajectory {key} still exceeds all {width} GPUs with "
                            "exact checkpointing; saved gradients remain resumable. "
                            "No samples were discarded and no partial optimizer "
                            "update was applied; refusing a restart loop.")
                    minimum_gpus[key] = min(trainer.world_size, 2 * width)

        # Restore existing objects, including the pre-update Adam state, once.
        # Only after every missing key has been committed is an update allowed.
        trainer.restore_after_generation()
        optimizer_parameters = {id(parameter)
                                for group in trainer.optimizer.param_groups
                                for parameter in group["params"]}
        if any(id(parameter) not in optimizer_parameters
               for parameter in trainer.model.parameters() if parameter.requires_grad):
            raise RuntimeError("restored model is not owned by the original optimizer")
        result = training._finalize_exact_spo_rs_overflow(
            trainer.model, trainer.optimizer, cfg, step_idx, state,
            completed_before=completed_before,
            label="warm cooperative overflow; no job restart")
        trainer._workers_idle = False
        for command_queue in trainer._command_queues.values():
            command_queue.put({"kind": "sync_update"})
        broadcast_trainable_parameters(trainer.model, source_rank=0)
        trainer._collect_event("updated", range(1, trainer.world_size))
        trainer.model.zero_grad(set_to_none=True)
        trainer._workers_idle = True
        result.update(
            training_parallel_gpus=trainer.world_size, training_fast=True,
            training_fast_processes=True, exact_overflow_inprocess=True,
            exact_overflow_waves=waves, exact_overflow_completed_examples=initial_pending,
            training_seconds=float(state["normal_stats"]["training_seconds"])
                + previous_overflow_seconds + time.time() - started)
        return result
    except BaseException:
        # Do not run a graceful NCCL barrier against a worker that may have
        # failed. The durable gradient checkpoint remains the recovery point.
        trainer._workers_idle = False
        trainer._abort_workers()
        raise


_EXACT_SHARDED_RECOVERY_FILE = ".exact_sharded_recovery.json"
_EXACT_OVERFLOW_GRADIENT_FILE = ".exact_spo_rs_overflow_step{step:04d}.pt"


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
        raise RuntimeError(f"invalid exact long-context recovery marker: {path}") from error
    if step < 0:
        raise RuntimeError(f"invalid exact long-context recovery step {step}: {path}")
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
    token_match = re.search(r"exact long-context training failed for a (\d+)-token rollout", tail[failure_offset:])
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
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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


def _save_exact_spo_rs_overflow(run_dir, step, model, rank_stats, *, normal_train_seconds=0.0):
    """Durably retain completed replica gradients for selective recovery."""
    import torch
    from training.fast_distributed import trainable_parameter_signature

    deferred_keys = sorted({
        (int(key[0]), int(key[1]))
        for stats in rank_stats
        for key in stats.get("deferred_keys", ())
    })
    deferred_count = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
    if deferred_count != len(deferred_keys) or not deferred_keys:
        raise RuntimeError(
            "exact SPO-RS overflow keys are missing or duplicated: "
            f"count={deferred_count}, unique={len(deferred_keys)}")

    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    normal_stats = {
        "totals": {key: sum(float(stats["totals"][key]) for stats in rank_stats) for key in metric_keys},
        "max_ratio": max(float(stats["max_ratio"]) for stats in rank_stats),
        "max_prefix_ratio": max(float(stats["max_prefix_ratio"]) for stats in rank_stats),
        "entropy_examples": sum(int(stats["entropy_examples"]) for stats in rank_stats),
        "trained_examples": sum(int(stats["trained_examples"]) for stats in rank_stats),
        "backward_batches": sum(int(stats["backward_batches"]) for stats in rank_stats),
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
        raise RuntimeError('invalid exact SPO-RS overflow gradient filename in recovery marker')
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
            payload = torch.load(path, map_location="cpu", weights_only=True)
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
    deferred_keys = sorted({(int(key[0]), int(key[1])) for key in payload.get("deferred_keys", ())})
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
    remaining_keys = sorted({(int(key[0]), int(key[1])) for key in payload.get("remaining_keys", deferred_keys)})
    completed_keys = sorted({(int(key[0]), int(key[1])) for key in payload.get("completed_keys", ())})
    if (set(remaining_keys) & set(completed_keys) or set(remaining_keys) | set(completed_keys) != set(deferred_keys)):
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
        "totals": {key: float(overflow_totals.get(key, 0.0)) for key in metric_keys},
        "max_ratio": float(overflow_stats.get("max_ratio", 0.0)),
        "max_prefix_ratio": float(overflow_stats.get("max_prefix_ratio", 0.0)),
        "entropy_examples": int(overflow_stats.get("entropy_examples", 0)),
        "trained_examples": int(overflow_stats.get("trained_examples", len(completed_keys))),
        "backward_batches": int(overflow_stats.get("backward_batches", len(completed_keys))),
        "training_seconds": max(0.0, float(overflow_stats.get("training_seconds", 0.0))),
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
        raise RuntimeError("exact SPO-RS overflow state has no durable checkpoint path")
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

    saved_gradients = _validate_precomputed_exact_gradients(model, overflow_state, step_idx)
    trajectory_keys = {tuple(int(value) for value in key) for key in trajectory_keys}
    if not trajectory_keys:
        return len(overflow_state.get("remaining_keys", ()))
    remaining = {tuple(int(value) for value in key) for key in overflow_state.get("remaining_keys", ())}
    completed = {tuple(int(value) for value in key) for key in overflow_state.get("completed_keys", ())}
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
            saved.add_(gradient.detach().to(device="cpu", dtype=saved.dtype))

    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    overflow_stats = overflow_state["overflow_stats"]
    totals = batch_stats["totals"]
    for key in metric_keys:
        overflow_stats["totals"][key] += float(totals[key])
    overflow_stats["max_ratio"] = max(float(overflow_stats["max_ratio"]), float(batch_stats["max_ratio"]))
    overflow_stats["max_prefix_ratio"] = max(
        float(overflow_stats["max_prefix_ratio"]),
        float(batch_stats["max_prefix_ratio"]))
    overflow_stats["entropy_examples"] += int(batch_stats["entropy_examples"])
    overflow_stats["trained_examples"] += len(trajectory_keys)
    overflow_stats["backward_batches"] += int(batch_stats.get("backward_batches", len(trajectory_keys)))
    overflow_stats["training_seconds"] += max(0.0, float(training_seconds))

    remaining.difference_update(trajectory_keys)
    completed.update(trajectory_keys)
    overflow_state["version"] = 2
    overflow_state["remaining_keys"] = sorted(remaining)
    overflow_state["completed_keys"] = sorted(completed)
    _write_exact_spo_rs_overflow_state(overflow_state)
    try:
        marker = _read_exact_sharded_recovery(Path(str(overflow_state["path"])).parent) or {}
        _write_exact_sharded_recovery(
            Path(str(overflow_state["path"])).parent,
            step_idx,
            marker.get("reason", "selective exact SPO-RS overflow recovery in progress"),
            remaining_deferred_keys=[list(key) for key in sorted(remaining)],
            completed_deferred_keys=[list(key) for key in sorted(completed)],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # The authoritative gradient checkpoint was already atomically
        # committed. A stale human-readable marker does not invalidate it.
        print(f'[warn] exact overflow progress marker could not be refreshed: {error}', flush=True)
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

    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    squared_norms = []
    for parameter in parameters:
        gradient = parameter.grad.detach()
        local_norm = torch.linalg.vector_norm(gradient, ord=2, dtype=torch.float32)
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
        if (argument.startswith("--resume=") or argument.startswith("--resume-from=")):
            index += 1
            continue
        filtered.append(argument)
        index += 1
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "train_multy_CVaR.py"),
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


def _validate_precomputed_exact_gradients(model, overflow_state, step_idx):
    """Validate and return a selective-overflow gradient mapping."""
    import torch
    from training.fast_distributed import trainable_parameter_signature

    if int(overflow_state.get("step", -1)) != int(step_idx):
        raise RuntimeError("exact SPO-RS overflow gradients belong to a different step")

    def canonical_signature(value):
        return tuple((str(name), tuple(int(size) for size in shape), str(dtype)) for name, shape, dtype in value)

    saved_signature = canonical_signature(overflow_state.get("parameter_signature", ()))
    current_signature = canonical_signature(trainable_parameter_signature(model))
    if saved_signature != current_signature:
        raise RuntimeError(
            "exact SPO-RS overflow trainable-parameter layout changed; "
            "refusing to combine mismatched gradients")
    saved_gradients = overflow_state.get("gradients")
    if not isinstance(saved_gradients, dict):
        raise RuntimeError("exact SPO-RS overflow checkpoint has no gradient mapping")
    expected_names = {name for name, _shape, _dtype in current_signature}
    if set(saved_gradients) != expected_names:
        raise RuntimeError('exact SPO-RS overflow gradient names do not match the loaded adapter')
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        saved = saved_gradients[name]
        if (not torch.is_tensor(saved)
                or tuple(saved.shape) != tuple(parameter.shape)
                or not torch.isfinite(saved).all()):
            raise RuntimeError(f"invalid saved exact SPO-RS gradient for {name}")
    return saved_gradients
