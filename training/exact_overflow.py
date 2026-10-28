"""Warm, in-process multi-GPU execution for oversized training trajectories.

Ordinary examples still use the existing replica/work-stealing path. Only its
overflow tail borrows GPUs from parked replicas. No model is loaded again, no
process is restarted, and an unsuccessful group is retried with more GPUs,
never fewer tokens. Gradient reduction and the single optimizer update remain
the coordinator's responsibility.
"""

from contextlib import contextmanager
import gc


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


def require_parking_capacity(model, world_size, optimizer=None):
    """Check RAM before moving replicas, not after committing the allocation."""
    from runtime.host_memory import (host_memory_info, GIB, estimated_module_bytes,
                             estimated_optimizer_bytes)

    weight_bytes = estimated_module_bytes(model)
    optimizer_bytes = estimated_optimizer_bytes(optimizer)
    required = int(1.10 * (weight_bytes * int(world_size) + optimizer_bytes))
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
    import train_multy_CVaR as training
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
    import train_multy_CVaR as training
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
                                         trainer.optimizer)
                trainer.offload_for_generation(purge_host_cache=False)
            # All replica weights remain resident in host RAM between waves.
            # Sleeping vLLM engines are not stopped or reloaded.
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
