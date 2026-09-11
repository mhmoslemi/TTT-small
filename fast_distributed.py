"""Process-per-GPU implementation for the opt-in fast rank trainer.

The main process is distributed rank 0. Persistent spawned workers own the
remaining GPUs, so transformer forward/checkpoint recomputation on different
cards does not contend for one Python interpreter. Only LoRA gradients and
parameters are communicated, each as one flattened NCCL buffer per update.
"""

from types import SimpleNamespace
import traceback


def trainable_parameters(model):
    return [parameter for parameter in model.parameters()
            if parameter.requires_grad]


def trainable_parameter_signature(model):
    """Plain-data layout used to reject mismatched replicas before NCCL."""
    return tuple(
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def validate_model_device(model, logical_id):
    """A fast replica must be wholly resident on its one assigned GPU."""
    import torch

    expected = torch.device(f"cuda:{int(logical_id)}")
    wrong = {
        str(parameter.device) for parameter in model.parameters()
        if parameter.device != expected
    }
    if wrong:
        raise RuntimeError(
            f"fast trainer rank {logical_id} expected every parameter on "
            f"{expected}, but also found {sorted(wrong)}")


def set_allocator_memory_ceiling(logical_id, fraction=0.80):
    """Make 80% an allocator limit, not merely a post-hoc target."""
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= 0.80:
        raise ValueError("fast trainer allocator ceiling must be in (0, 0.80]")
    with torch.cuda.device(int(logical_id)):
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=int(logical_id))


def set_total_memory_ceiling(logical_id, fraction=0.80):
    """Cap this allocator so it plus already-used device memory stays bounded."""
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= 0.80:
        raise ValueError("fast trainer total memory ceiling must be in (0, 0.80]")
    with torch.cuda.device(int(logical_id)):
        # Release only unused cache before measuring the model's persistent base.
        # This runs at worker load/update boundaries, never per microbatch.
        torch.cuda.empty_cache()
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        outside_allocator = max(
            0, int(total_bytes) - int(free_bytes) - reserved)
        process_limit = int(fraction * int(total_bytes)) - outside_allocator
        if process_limit <= allocated:
            occupied = (outside_allocator + allocated) / max(1, int(total_bytes))
            raise RuntimeError(
                f"fast trainer rank {logical_id} already needs "
                f"{100.0 * occupied:.1f}% GPU memory before a batch, so the "
                f"{100.0 * fraction:.0f}% total ceiling cannot be honored")
        allocator_fraction = min(
            fraction, process_limit / max(1, int(total_bytes)))
        set_allocator_memory_ceiling(logical_id, allocator_fraction)
        return allocator_fraction


def broadcast_trainable_parameters(model, source_rank=0):
    """Broadcast all LoRA parameters with one contiguous NCCL operation."""
    import torch
    import torch.distributed as dist

    parameters = trainable_parameters(model)
    if not parameters:
        raise RuntimeError("fast trainer found no trainable parameters")
    flat = torch.cat([
        parameter.detach().reshape(-1).to(dtype=torch.float32)
        for parameter in parameters
    ])
    dist.broadcast(flat, src=int(source_rank))
    if dist.get_rank() != int(source_rank):
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                count = parameter.numel()
                parameter.copy_(
                    flat[offset:offset + count].view_as(parameter).to(
                        dtype=parameter.dtype))
                offset += count
    del flat


def reduce_trainable_gradients(model, destination_rank=0):
    """Sum all LoRA gradients into rank 0 with one contiguous NCCL call."""
    import torch
    import torch.distributed as dist

    parameters = trainable_parameters(model)
    if not parameters:
        raise RuntimeError("fast trainer found no trainable parameters")
    flat = torch.cat([
        ((parameter.grad.detach() if parameter.grad is not None
          else torch.zeros_like(parameter)).reshape(-1).to(
              dtype=torch.float32))
        for parameter in parameters
    ])
    dist.reduce(flat, dst=int(destination_rank), op=dist.ReduceOp.SUM)
    if dist.get_rank() == int(destination_rank):
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                count = parameter.numel()
                reduced = flat[offset:offset + count].view_as(parameter).to(
                    dtype=parameter.dtype)
                if parameter.grad is None:
                    parameter.grad = torch.empty_like(parameter)
                parameter.grad.copy_(reduced)
                offset += count
    else:
        model.zero_grad(set_to_none=True)
    del flat


def _feedback_stats_dict(stats):
    return {
        "n": int(stats.n),
        "skipped": int(stats.skipped),
        "sum_abs": float(stats.sum_abs),
        "sum_pos": float(stats.sum_pos),
        "max_abs": float(stats.max_abs),
    }


def _merge_feedback_stats(parts):
    from feedback import FeedbackStats

    merged = FeedbackStats()
    for part in parts:
        if part is None:
            continue
        merged.n += int(part.n)
        merged.skipped += int(part.skipped)
        merged.sum_abs += float(part.sum_abs)
        merged.sum_pos += float(part.sum_pos)
        merged.max_abs = max(merged.max_abs, float(part.max_abs))
    return merged


def _padded_token_count(batch):
    if not batch:
        return 0
    return max(
        int(example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])
        for example in batch
    ) * len(batch)


def _move_examples(examples, logical_id):
    import torch

    device = torch.device(f"cuda:{logical_id}")
    tensor_cache = {}
    moved = []
    for example in examples:
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


def local_x_grpo_calibration(backend, model, tokenizer, examples, cfg,
                             logical_id, group_count, *, group_ids=None):
    """Evaluate this rank's diagnostic groups and cross-fit through NCCL sums."""
    import gc
    import math
    import torch
    import torch.distributed as dist
    import train_multy_CVaR as training

    group_count = int(group_count)
    if group_count < 3:
        raise ValueError("X-GRPO calibration requires at least three groups")
    backend.set_training_mode()
    set_total_memory_ceiling(logical_id, 0.80)
    local_examples = _move_examples(list(examples), logical_id)
    expected_group_ids = (
        sorted({int(group_id) for group_id in group_ids})
        if group_ids is not None else
        sorted({int(example["group_id"]) for example in local_examples})
    )
    local_groups = {group_id: [] for group_id in expected_group_ids}
    for example in local_examples:
        group_id = int(example["group_id"])
        if group_id not in local_groups:
            raise ValueError(
                f"unexpected local X-GRPO diagnostic group {group_id}")
        local_groups[group_id].append(example)

    parameters = trainable_parameters(model)
    flat_count = sum(parameter.numel() for parameter in parameters)
    if flat_count < 1:
        raise RuntimeError("X-GRPO found no trainable parameters")
    budgets = tuple(float(value) for value in cfg.x_grpo_budgets)
    selected = {group_id: 0.0 for group_id in local_groups}
    diagnostics = {group_id: [] for group_id in local_groups}
    quarantined_total = 0
    heldout_count = group_count - 1
    error_denominator = heldout_count * (heldout_count - 1)
    relative_error = float(cfg.x_grpo_relative_error)
    device = torch.device(f"cuda:{int(logical_id)}")

    try:
        for budget_index, budget in enumerate(budgets):
            gradients = {}
            for group_id in sorted(local_groups):
                gradient, quarantined = training._x_grpo_flat_group_gradient(
                    model, tokenizer, local_groups[group_id], cfg,
                    budget_index, device_label=str(device))
                gradients[group_id] = gradient
                quarantined_total += quarantined

            if gradients:
                local_sum = next(iter(gradients.values())).clone()
                for gradient in list(gradients.values())[1:]:
                    local_sum.add_(gradient)
                local_squared_norms = sum(
                    float(torch.dot(gradient, gradient).item())
                    for gradient in gradients.values())
            else:
                local_sum = torch.zeros(flat_count, dtype=torch.float32)
                local_squared_norms = 0.0

            with torch.cuda.device(int(logical_id)), torch.no_grad():
                total = local_sum.to(device, non_blocking=True)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                squared_norms = torch.tensor(
                    local_squared_norms, dtype=torch.float64, device=device)
                dist.all_reduce(squared_norms, op=dist.ReduceOp.SUM)
                total_norm_squared = float(torch.dot(total, total).item())
                all_squared_norms = float(squared_norms.item())
                total_cpu = total.cpu()

            for group_id, gradient in gradients.items():
                gradient_norm_squared = float(
                    torch.dot(gradient, gradient).item())
                total_dot_gradient = float(torch.dot(total_cpu, gradient).item())
                heldout_sum_norm_squared = max(
                    0.0,
                    total_norm_squared + gradient_norm_squared
                    - 2.0 * total_dot_gradient,
                )
                heldout_squared_norms = max(
                    0.0, all_squared_norms - gradient_norm_squared)
                centered_sum = max(
                    0.0,
                    heldout_squared_norms
                    - heldout_sum_norm_squared / heldout_count,
                )
                signal_squared = (
                    heldout_sum_norm_squared / (heldout_count ** 2))
                error_squared = centered_sum / error_denominator
                accepted = bool(
                    signal_squared > 0.0
                    and error_squared
                    <= relative_error ** 2 * signal_squared)
                signal_norm = math.sqrt(signal_squared)
                standard_error = math.sqrt(error_squared)
                diagnostics[group_id].append({
                    "budget": budget,
                    "gradient_norm": signal_norm,
                    "standard_error": standard_error,
                    "relative_error": (
                        standard_error / signal_norm
                        if signal_norm > 0.0 else None),
                    "accepted": accepted,
                })
                if accepted:
                    selected[group_id] = max(selected[group_id], budget)

            del gradients, local_sum, total, total_cpu, squared_norms
            gc.collect()
            with torch.cuda.device(int(logical_id)):
                torch.cuda.empty_cache()
    finally:
        model.zero_grad(set_to_none=True)

    return {
        "selected_budgets": selected,
        "groups": diagnostics,
        "diagnostic_quarantined_examples": quarantined_total,
    }


def local_rank_update(backend, model, tokenizer, examples, cfg, logical_id,
                      token_budget, *, memory_fraction=0.80, fb_cfg=None,
                      fb_on=False, fb_lambda=0.0, work_queue=None):
    """Accumulate one rank's gradients and leave them attached to the model."""
    import torch
    import train_multy_CVaR as training
    from feedback import (FeedbackStats, bound_feedback_advantage,
                          feedback_advantage)

    if not 0.0 < float(memory_fraction) <= 0.80:
        raise ValueError("fast trainer memory_fraction must be in (0, 0.80]")

    backend.set_training_mode()
    allocator_fraction = set_total_memory_ceiling(
        logical_id, float(memory_fraction))
    epsilon = float(getattr(
        cfg, "rank_clip_epsilon", training.RANK_CLIP_EPSILON_DEFAULT))
    epsilon_low = float(getattr(cfg, "rank_clip_epsilon_low", epsilon))
    epsilon_high = float(getattr(cfg, "rank_clip_epsilon_high", epsilon))
    entropy_coef = training._rank_entropy_coefficient(cfg)
    kl_coef = float(cfg.kl_penalty_coef)
    parameters = trainable_parameters(model)
    gradient_accumulators = [None for _ in parameters]
    metric_keys = (
        "loss", "policy_loss", "kl_estimate", "entropy_estimate",
        "ratio", "clipped_fraction",
    )
    totals = {key: 0.0 for key in metric_keys}
    max_ratio = 0.0
    max_prefix_ratio = 0.0
    entropy_examples = 0
    quarantined_examples = 0
    backward_batches = 0
    trained_examples = 0
    peak_memory_fraction = 0.0
    feedback_parts = []
    budget = max(1, int(token_budget or cfg.max_seq_length))

    examples = list(examples or ())
    if work_queue is not None and examples:
        raise ValueError(
            "fast trainer accepts either local examples or a shared queue")

    def example_length(example):
        return int(
            example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])

    def example_gate(example):
        return bool(example["rank_entropy_gate"] and entropy_coef > 0.0)

    max_examples_per_batch = 64
    if work_queue is None:
        partitions = {False: [], True: []}
        for example in examples:
            partitions[example_gate(example)].append(example)
        for partition in partitions.values():
            partition.sort(key=example_length)
        max_examples_per_batch = min(64, max(1, len(examples)))

        def take_batch():
            available = [key for key, value in partitions.items() if value]
            if not available:
                return []
            partition_key = max(
                available,
                key=lambda key: example_length(partitions[key][-1]))
            pending = partitions[partition_key]
            batch = [pending.pop()]
            maximum = example_length(batch[0])
            while pending and len(batch) < max_examples_per_batch:
                candidate = pending[-1]
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                if next_maximum * (len(batch) + 1) > budget:
                    break
                batch.append(pending.pop())
                maximum = next_maximum
            return batch
    else:
        # None is one end marker per rank. A single deferred example lets a rank
        # stop growing the current padded batch without putting work back or
        # disturbing the global queue's exactly-once behavior.
        queue_finished = False
        deferred = None

        def take_batch():
            nonlocal queue_finished, deferred
            if queue_finished and deferred is None:
                return []
            if deferred is not None:
                first, deferred = deferred, None
            else:
                first = work_queue.get()
                if first is None:
                    queue_finished = True
                    return []
            batch = [first]
            maximum = example_length(first)
            gate = example_gate(first)
            while (not queue_finished
                   and len(batch) < max_examples_per_batch):
                candidate = work_queue.get()
                if candidate is None:
                    queue_finished = True
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                if (example_gate(candidate) != gate
                        or next_maximum * (len(batch) + 1) > budget):
                    deferred = candidate
                    break
                batch.append(candidate)
                maximum = next_maximum
            return batch

    model.zero_grad(set_to_none=True)
    while True:
        cpu_batch = take_batch()
        if not cpu_batch:
            break
        requested_padded_tokens = _padded_token_count(cpu_batch)

        with torch.cuda.device(logical_id):
            base_allocated = int(torch.cuda.memory_allocated())
            base_reserved = int(torch.cuda.memory_reserved())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            external_bytes = max(
                0, int(total_bytes) - int(free_bytes) - base_reserved)
            torch.cuda.reset_peak_memory_stats()
            local_batch = _move_examples(cpu_batch, logical_id)

            def attempt(active_batches):
                attempt_values = {key: [] for key in metric_keys}
                attempt_ratio_values = []
                attempt_prefix_ratio_values = []
                attempt_entropy_examples = 0
                attempt_feedback = FeedbackStats()

                with training._rank_dropout_disabled(model):
                    for batch in active_batches:
                        missing_old, missing_reference = (
                            training._initialize_rank_logprob_caches(batch))
                        for fallback_batch in (
                                [example] for example in missing_old):
                            old_logprobs = training.compute_batched_token_logprobs(
                                model, fallback_batch, with_grad=False,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id)
                            if any(not torch.isfinite(value).all()
                                   for value in old_logprobs):
                                raise FloatingPointError(
                                    "nonfinite old-policy logprobs")
                            for example, old_lp in zip(
                                    fallback_batch, old_logprobs):
                                example["rank_old_logprobs"] = old_lp.detach()

                        if kl_coef and missing_reference:
                            with backend.disable_adapter(), torch.no_grad():
                                for fallback_batch in (
                                        [example] for example in
                                        missing_reference):
                                    reference_logprobs = (
                                        training.compute_batched_token_logprobs(
                                            model, fallback_batch,
                                            with_grad=False,
                                            chunk=cfg.logprob_chunk,
                                            pad_token_id=(
                                                tokenizer.pad_token_id)))
                                    if any(not torch.isfinite(value).all()
                                           for value in reference_logprobs):
                                        raise FloatingPointError(
                                            "nonfinite reference-policy "
                                            "logprobs")
                                    for example, reference_lp in zip(
                                            fallback_batch,
                                            reference_logprobs):
                                        example[
                                            "rank_reference_logprob"
                                        ] = reference_lp.detach()

                        if fb_on:
                            for example in batch:
                                if not example.get("reprompt_text"):
                                    continue
                                old_lp = example["rank_old_logprobs"].to(
                                    example["response_ids"].device)
                                fb_advantage = feedback_advantage(
                                    training.compute_token_logprobs, model,
                                    tokenizer, example["reprompt_text"],
                                    example["response_ids"], old_lp, fb_cfg,
                                    lam=fb_lambda, chunk=cfg.logprob_chunk)
                                if fb_advantage is None:
                                    attempt_feedback.skipped += 1
                                else:
                                    fb_advantage, _ = bound_feedback_advantage(
                                        fb_advantage,
                                        reward_advantage=example["advantage"],
                                        cfg=fb_cfg)
                                    attempt_feedback.add(fb_advantage)
                                    example[
                                        "rank_feedback_advantage"
                                    ] = fb_advantage.detach().cpu()

                        gate = bool(
                            batch[0]["rank_entropy_gate"]
                            and entropy_coef > 0.0)
                        result = training.compute_batched_token_logprobs(
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
                                batch, current_logprobs, token_entropies):
                            loss, metrics = training.rank_grpo_loss(
                                current_lp,
                                example["rank_old_logprobs"],
                                example["rank_reference_logprob"],
                                example["advantage"],
                                clip_epsilon=epsilon,
                                clip_epsilon_low=epsilon_low,
                                clip_epsilon_high=epsilon_high,
                                kl_coef=kl_coef,
                                entropy_coef=(entropy_coef if gate else 0.0),
                                token_entropies=token_entropy,
                                return_tensor_metrics=True)
                            fb_advantage = example[
                                "rank_feedback_advantage"]
                            if fb_advantage is not None:
                                loss = loss - (
                                    fb_advantage.to(current_lp.device)
                                    * current_lp).mean()
                            weight = float(example["sample_weight"])
                            if not torch.isfinite(loss).all():
                                raise FloatingPointError(
                                    "nonfinite rank/feedback loss")
                            weighted_losses.append(weight * loss)
                            attempt_values["loss"].append(
                                weight * loss.detach())
                            for key in (
                                    "policy_loss", "kl_estimate",
                                    "entropy_estimate", "ratio"):
                                attempt_values[key].append(
                                    weight * metrics[key])
                            attempt_values["clipped_fraction"].append(
                                weight * metrics["clipped"])
                            attempt_ratio_values.append(metrics["ratio"])
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
                    value.to(dtype=torch.float64) for value in reduced
                ]).detach().cpu().tolist()
                return (
                    dict(zip(metric_keys, packed[:len(metric_keys)])),
                    packed[len(metric_keys)], packed[len(metric_keys) + 1],
                    attempt_entropy_examples, attempt_feedback,
                )

            result, effective_batches, quarantined = (
                training._run_oom_resilient_backward(
                    model, [local_batch], attempt,
                    device_label=f"cuda:{logical_id}"))
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            used_at_peak = min(
                int(total_bytes), external_bytes + peak_reserved)
            observed_fraction = used_at_peak / max(1, int(total_bytes))
            peak_memory_fraction = max(
                peak_memory_fraction, observed_fraction)

            if result is not None:
                with torch.no_grad():
                    for parameter_index, parameter in enumerate(parameters):
                        gradient = parameter.grad
                        if gradient is None:
                            continue
                        accumulator = gradient_accumulators[parameter_index]
                        if accumulator is None:
                            gradient_accumulators[
                                parameter_index] = gradient.detach()
                        else:
                            accumulator.add_(gradient)
                        parameter.grad = None
                for key in metric_keys:
                    totals[key] += result[0][key]
                max_ratio = max(max_ratio, result[1])
                max_prefix_ratio = max(max_prefix_ratio, result[2])
                entropy_examples += result[3]
                feedback_parts.append(result[4])

            model.zero_grad(set_to_none=True)
            quarantined_examples += len(quarantined)
            backward_batches += len(effective_batches)
            trained_examples += sum(
                len(batch) for batch in effective_batches)
            successful_padded_tokens = max(
                (_padded_token_count(batch) for batch in effective_batches),
                default=0)
            longest_successful = max(
                (int(example["prompt_ids"].shape[1]
                     + example["response_ids"].shape[1])
                 for batch in effective_batches for example in batch),
                default=1)
            backed_off = bool(len(effective_batches) > 1 or quarantined)
            if not effective_batches:
                budget = max(1, budget // 2)
            elif backed_off:
                budget = max(
                    longest_successful,
                    min(budget, successful_padded_tokens))
            else:
                active_bytes = max(1, peak_allocated - base_allocated)
                target_process_bytes = max(
                    base_allocated + 1,
                    int(float(memory_fraction) * int(total_bytes))
                    - external_bytes)
                available_for_batch = max(
                    1, target_process_bytes - base_allocated)
                desired_budget = int(
                    requested_padded_tokens
                    * available_for_batch / active_bytes * 0.95)
                if observed_fraction >= float(memory_fraction):
                    desired_budget = min(
                        desired_budget,
                        int(budget * float(memory_fraction)
                            / max(observed_fraction, 1e-9) * 0.95))
                lower = max(longest_successful, int(budget * 0.70))
                upper = max(lower, int(budget * 1.20))
                desired_budget = min(upper, max(lower, desired_budget))
                budget = max(
                    longest_successful,
                    int(round(0.5 * budget + 0.5 * desired_budget)))

        del local_batch, effective_batches, quarantined

    with torch.cuda.device(logical_id), torch.no_grad():
        for parameter, accumulator in zip(
                parameters, gradient_accumulators):
            parameter.grad = accumulator
        torch.cuda.synchronize(logical_id)

    merged_feedback = _merge_feedback_stats(feedback_parts)
    return {
        "totals": totals,
        "max_ratio": max_ratio,
        "max_prefix_ratio": max_prefix_ratio,
        "entropy_examples": entropy_examples,
        "quarantined_examples": quarantined_examples,
        "backward_batches": backward_batches,
        "trained_examples": trained_examples,
        "peak_memory_fraction": peak_memory_fraction,
        "allocator_memory_fraction": allocator_fraction,
        "token_budget": budget,
        "feedback_stats": _feedback_stats_dict(merged_feedback),
    }


def worker_main(rank, world_size, cfg_dict, init_method, work_queue,
                command_queue, result_queue, dependency_log_path=None):
    """Persistent rank>0 worker command loop."""
    dist_initialized = False
    try:
        import gc
        import os
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        import train_multy_CVaR as training
        from model_backend import load_backend

        torch.set_num_threads(max(
            1, int(os.cpu_count() or world_size) // int(world_size)))
        torch.cuda.set_device(int(rank))
        set_total_memory_ceiling(int(rank), 0.80)
        cfg = SimpleNamespace(**dict(cfg_dict))
        cfg.training_replica_device = int(rank)
        cfg.num_training_gpus = 1
        rank_log = (f"{dependency_log_path}.trainer-rank{rank}"
                    if dependency_log_path else None)
        with training._route_dependency_notices(rank_log):
            backend = load_backend(cfg.backend, cfg)
            model, tokenizer = backend.load()
        validate_model_device(model, int(rank))
        result_queue.put({
            "event": "loaded", "rank": int(rank),
            "parameter_signature": trainable_parameter_signature(model),
        })

        command = command_queue.get()
        if command.get("kind") != "init_distributed":
            raise RuntimeError("fast worker expected init_distributed")
        dist.init_process_group(
            backend="nccl", init_method=init_method,
            rank=int(rank), world_size=int(world_size),
            timeout=timedelta(minutes=30))
        dist_initialized = True
        broadcast_trainable_parameters(model, source_rank=0)
        result_queue.put({"event": "ready", "rank": int(rank)})
        offloaded = False

        while True:
            command = command_queue.get()
            kind = command.get("kind")
            if kind == "offload":
                if not offloaded:
                    backend.offload_for_generation()
                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        torch.cuda.empty_cache()
                    offloaded = True
                result_queue.put({"event": "offloaded", "rank": int(rank)})
            elif kind == "restore":
                if offloaded:
                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        torch.cuda.empty_cache()
                        set_total_memory_ceiling(int(rank), 0.80)
                        backend.restore_after_generation()
                        backend.set_training_mode()
                        torch.cuda.synchronize(int(rank))
                    validate_model_device(model, int(rank))
                    offloaded = False
                result_queue.put({"event": "restored", "rank": int(rank)})
            elif kind == "train_rank":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received training while offloaded")
                stats = local_rank_update(
                    backend, model, tokenizer, (), cfg,
                    int(rank), command["token_budget"],
                    memory_fraction=command.get("memory_fraction", 0.80),
                    fb_cfg=command.get("fb_cfg"),
                    fb_on=bool(command.get("fb_on", False)),
                    fb_lambda=float(command.get("fb_lambda", 0.0)),
                    work_queue=work_queue)
                result_queue.put({
                    "event": "computed", "rank": int(rank),
                    "stats": stats,
                })
            elif kind == "calibrate_x_grpo":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received calibration while offloaded")
                calibration = local_x_grpo_calibration(
                    backend, model, tokenizer, command["examples"], cfg,
                    int(rank), int(command["group_count"]),
                    group_ids=command["group_ids"])
                result_queue.put({
                    "event": "calibrated", "rank": int(rank),
                    "calibration": calibration,
                })
            elif kind == "finish_update":
                reduce_trainable_gradients(model, destination_rank=0)
                broadcast_trainable_parameters(model, source_rank=0)
                model.zero_grad(set_to_none=True)
                result_queue.put({"event": "updated", "rank": int(rank)})
            elif kind == "stop":
                dist.barrier()
                result_queue.put({"event": "stopped", "rank": int(rank)})
                break
            else:
                raise RuntimeError(f"unknown fast-worker command: {kind!r}")
    except BaseException:
        try:
            result_queue.put({
                "event": "error", "rank": int(rank),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if dist_initialized:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
            except Exception:
                pass
