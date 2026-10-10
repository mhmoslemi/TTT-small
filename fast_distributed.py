"""Process-per-GPU implementation for opt-in fast LoRA training.

The main process is distributed rank 0. Persistent spawned workers own the
remaining GPUs, so transformer forward/checkpoint recomputation on different
cards does not contend for one Python interpreter. Only LoRA gradients and
parameters are communicated, each as one flattened NCCL buffer per update.
"""

from types import SimpleNamespace
import math
import traceback


STANDARD_TRAINING_MEMORY_FRACTION = 0.80
MAX_STANDARD_TRAINING_MEMORY_FRACTION = 0.90
LONG_ROLLOUT_MEMORY_FRACTION = 0.96
LONG_ROLLOUT_MEMORY_FRACTIONS = (0.96, 0.98)
MAX_LONG_ROLLOUT_MEMORY_FRACTION = max(LONG_ROLLOUT_MEMORY_FRACTIONS)
SHARED_PREFIX_WORK_KEY = "__ttt_shared_prefix_examples__"


def _shared_prefix_examples(batch):
    batch = list(batch or ())
    if len(batch) < 2:
        return False
    if not all(
            bool(example.get("_shared_prefix_packing_allowed", True))
            for example in batch):
        return False
    prompt_job_id = batch[0].get("prompt_job_id")
    if prompt_job_id is None:
        return False
    prompt_shape = tuple(batch[0]["prompt_ids"].shape)
    return all(
        example.get("prompt_job_id") == prompt_job_id
        and tuple(example["prompt_ids"].shape) == prompt_shape
        for example in batch[1:]
    )


def _shared_prefix_token_count(batch):
    """Physical packed tokens for one prompt and several response branches."""
    batch = list(batch)
    if not batch:
        return 0
    prompt = int(batch[0]["prompt_ids"].shape[1])
    return prompt + sum(max(
        0, int(example["response_ids"].shape[1]) - 1)
        for example in batch)


def _take_shared_prefix_chunk(examples, token_budget, example_cap=64,
                              singleton_predicate=None,
                              packed_token_limit=None):
    """Take the largest leading exact-prefix pack within the token budget."""
    examples = list(examples)
    if not examples:
        return [], []
    if singleton_predicate is not None and singleton_predicate(examples[0]):
        return examples[:1], examples[1:]
    prompt = int(examples[0]["prompt_ids"].shape[1])
    used = prompt
    count = 0
    for example in examples:
        if count and singleton_predicate is not None and singleton_predicate(
                example):
            break
        contribution = max(0, int(example["response_ids"].shape[1]) - 1)
        next_used = used + contribution
        if count and (count >= int(example_cap)
                      or next_used > int(token_budget)
                      or (packed_token_limit is not None
                          and next_used >= int(packed_token_limit))):
            break
        used = next_used
        count += 1
    count = max(1, count)
    return examples[:count], examples[count:]


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


def _set_allocator_memory_ceiling(logical_id, fraction, *, maximum, label):
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= float(maximum):
        raise ValueError(
            f"{label} allocator ceiling must be in (0, {maximum:.2f}]")
    with torch.cuda.device(int(logical_id)):
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=int(logical_id))


def set_allocator_memory_ceiling(
        logical_id, fraction=STANDARD_TRAINING_MEMORY_FRACTION):
    """Apply the configured normal-training allocator limit."""
    return _set_allocator_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_STANDARD_TRAINING_MEMORY_FRACTION,
        label="fast trainer")


def _set_total_memory_ceiling(logical_id, fraction, *, maximum, label):
    """Cap this allocator plus allocations owned by other GPU processes."""
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= float(maximum):
        raise ValueError(
            f"{label} total memory ceiling must be in (0, {maximum:.2f}]")
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
                f"{label} rank {logical_id} already needs "
                f"{100.0 * occupied:.1f}% GPU memory before a batch, so the "
                f"{100.0 * fraction:.0f}% total ceiling cannot be honored")
        allocator_fraction = min(
            fraction, process_limit / max(1, int(total_bytes)))
        _set_allocator_memory_ceiling(
            logical_id, allocator_fraction,
            maximum=maximum, label=label)
        return allocator_fraction


def set_total_memory_ceiling(
        logical_id, fraction=STANDARD_TRAINING_MEMORY_FRACTION):
    """Apply the normal conservative limit used by adaptive microbatches."""
    return _set_total_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_STANDARD_TRAINING_MEMORY_FRACTION,
        label="fast trainer")


def set_long_rollout_memory_ceiling(
        logical_id, fraction=LONG_ROLLOUT_MEMORY_FRACTION):
    """Use reserved card headroom for an otherwise untrainable singleton.

    This is used after the batch has been reduced to one rollout and before
    resorting to saved-activation CPU offload. External allocations (including
    sleeping vLLM processes and NCCL) remain part of the total. Callers first
    use the 96% ceiling and may explicitly escalate to 98% after a proven OOM;
    the last two percent of the card always remains outside the total ceiling.
    """
    return _set_total_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_LONG_ROLLOUT_MEMORY_FRACTION,
        label="long-rollout rescue")


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






def _padded_token_count(batch):
    if not batch:
        return 0
    if _shared_prefix_examples(batch):
        return _shared_prefix_token_count(batch)
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
                             logical_id, *, context_group_ids,
                             group_ids=None):
    """Evaluate assigned groups while cross-fitting each context separately."""
    import math
    from tempfile import TemporaryDirectory
    import torch
    import torch.distributed as dist
    import train_multy_CVaR as training

    backend.set_training_mode()
    set_total_memory_ceiling(logical_id, 0.80)
    local_examples = _move_examples(list(examples), logical_id)
    contexts = training._x_grpo_context_group_map(
        (), context_group_ids=context_group_ids)
    all_group_ids = {
        group_id for ids in contexts.values() for group_id in ids
    }
    expected_group_ids = (
        sorted({int(group_id) for group_id in group_ids})
        if group_ids is not None else
        sorted({int(example["group_id"]) for example in local_examples})
    )
    unexpected = set(expected_group_ids) - all_group_ids
    if unexpected:
        raise ValueError(
            f"unexpected local X-GRPO groups: {sorted(unexpected)}")
    local_groups = {group_id: [] for group_id in expected_group_ids}
    context_for_group = {
        group_id: context_id
        for context_id, ids in contexts.items()
        for group_id in ids
    }
    for example in local_examples:
        group_id = int(example["group_id"])
        if group_id not in local_groups:
            raise ValueError(
                f"unexpected local X-GRPO diagnostic group {group_id}")
        example_context = int(example.get(
            "x_grpo_context_id", context_for_group[group_id]))
        if example_context != context_for_group[group_id]:
            raise ValueError(
                f"X-GRPO group {group_id} was assigned to context "
                f"{context_for_group[group_id]} but contains context "
                f"{example_context}")
        local_groups[group_id].append(example)

    parameters = trainable_parameters(model)
    flat_count = sum(parameter.numel() for parameter in parameters)
    if flat_count < 1:
        raise RuntimeError("X-GRPO found no trainable parameters")
    zero_gradient = torch.zeros(
        flat_count, dtype=torch.float32, device="cpu")
    budgets = tuple(float(value) for value in cfg.x_grpo_budgets)
    selected = {group_id: 0.0 for group_id in local_groups}
    diagnostics = {group_id: [] for group_id in local_groups}
    quarantined_total = 0
    batch_plans = {
        group_id: training._training_microbatches(group_examples, cfg)
        for group_id, group_examples in local_groups.items()
    }
    budget_indices = tuple(range(len(budgets)))
    relative_error = float(cfg.x_grpo_relative_error)
    device = torch.device(f"cuda:{int(logical_id)}")

    try:
        with TemporaryDirectory(prefix="x-grpo-gradients-") as cache_dir:
            cached_by_group = {}
            with training._rank_dropout_disabled(model):
                for group_id in sorted(local_groups):
                    cached, quarantined, effective = (
                        training._x_grpo_cache_group_gradients(
                            model, tokenizer, local_groups[group_id], cfg,
                            budget_indices, group_id=group_id,
                            cache_dir=cache_dir,
                            device_label=str(device),
                            batches=batch_plans[group_id],
                            parameters=parameters))
                    batch_plans[group_id] = effective
                    quarantined_total += quarantined
                    cached_by_group[group_id] = cached

            for context_id, context_ids in contexts.items():
                group_count = len(context_ids)
                heldout_count = group_count - 1
                error_denominator = heldout_count * (heldout_count - 1)
                local_context_ids = tuple(
                    group_id for group_id in context_ids
                    if group_id in local_groups
                )
                context_caches = {
                    group_id: cached_by_group[group_id]
                    for group_id in local_context_ids
                }
                for budget_index, budget in enumerate(budgets):
                    gradients = {
                        group_id: training._x_grpo_load_cached_gradient(
                            cached_by_group[group_id][budget_index],
                            zero_gradient)
                        for group_id in local_context_ids
                    }
                    gradient_squared_norms = {
                        group_id: float(torch.dot(
                            gradient, gradient).item())
                        for group_id, gradient in gradients.items()
                    }
                    if gradients:
                        local_sum = next(iter(gradients.values())).clone()
                        for gradient in list(gradients.values())[1:]:
                            local_sum.add_(gradient)
                        local_squared_norms = sum(
                            gradient_squared_norms.values())
                    else:
                        local_sum = torch.zeros(
                            flat_count, dtype=torch.float32)
                        local_squared_norms = 0.0

                    with torch.cuda.device(int(logical_id)), torch.no_grad():
                        total = local_sum.to(device, non_blocking=True)
                        dist.all_reduce(total, op=dist.ReduceOp.SUM)
                        squared_norms = torch.tensor(
                            local_squared_norms, dtype=torch.float64,
                            device=device)
                        dist.all_reduce(squared_norms, op=dist.ReduceOp.SUM)
                        all_squared_norms = float(squared_norms.item())
                        total_cpu = total.cpu()
                        total_norm_squared = float(
                            torch.dot(total_cpu, total_cpu).item())

                    for group_id, gradient in gradients.items():
                        gradient_norm_squared = gradient_squared_norms[group_id]
                        heldout_sum = None
                        if gradient_norm_squared == 0.0:
                            heldout_sum_norm_squared = total_norm_squared
                        else:
                            heldout_sum = total_cpu - gradient
                            heldout_sum_norm_squared = float(
                                torch.dot(heldout_sum, heldout_sum).item())
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
                            selected[group_id] = max(
                                selected[group_id], budget)
                        if heldout_sum is not None:
                            del heldout_sum

                    del gradients, gradient_squared_norms, local_sum
                    del total, total_cpu, squared_norms
                    training._x_grpo_delete_consumed_cache(
                        context_caches, budget_index)
    finally:
        model.zero_grad(set_to_none=True)

    return {
        "selected_budgets": selected,
        "groups": diagnostics,
        "diagnostic_quarantined_examples": quarantined_total,
    }


def local_rank_update(backend, model, tokenizer, examples, cfg, logical_id,
                      token_budget, *, memory_fraction=0.80, work_queue=None):
    """Accumulate one rank's gradients and leave them attached to the model."""
    import torch
    import train_multy_CVaR as training
    from host_memory import reclaim_unused_host_memory

    if not 0.0 < float(memory_fraction) <= (
            MAX_STANDARD_TRAINING_MEMORY_FRACTION):
        raise ValueError(
            "fast trainer memory_fraction must be in (0, 0.90]")

    backend.set_training_mode()
    allocator_fraction = set_total_memory_ceiling(
        logical_id, float(memory_fraction))
    epsilon, epsilon_low, epsilon_high, kl_coef = (
        training._clipped_policy_options(cfg))
    entropy_coef = training._rank_entropy_coefficient(cfg)
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
    budget = max(1, int(token_budget or cfg.max_seq_length))
    emergency_allocator_fraction = None
    emergency_allocator_stage = 0

    def expand_for_long_rollout():
        nonlocal allocator_fraction, emergency_allocator_fraction
        nonlocal emergency_allocator_stage
        while emergency_allocator_stage < len(
                LONG_ROLLOUT_MEMORY_FRACTIONS):
            requested_fraction = LONG_ROLLOUT_MEMORY_FRACTIONS[
                emergency_allocator_stage]
            emergency_allocator_stage += 1
            try:
                expanded = set_long_rollout_memory_ceiling(
                    logical_id, requested_fraction)
            except RuntimeError as error:
                print(f"[train-oom] cuda:{logical_id}: "
                      f"{100.0 * requested_fraction:.0f}% reserved GPU "
                      f"headroom is unavailable ({error})", flush=True)
                continue
            if (emergency_allocator_fraction is not None
                    and expanded <= emergency_allocator_fraction):
                continue
            emergency_allocator_fraction = expanded
            allocator_fraction = max(allocator_fraction, expanded)
            return expanded
        return None

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
        # disturbing the global queue's exactly-once behavior. Strategy-mode
        # work items contain every program sharing one prompt; keep the item on
        # this rank and consume it in exact shared-prefix packs so no other rank
        # independently evaluates that prompt.
        queue_finished = False
        deferred = None
        prefix_remainder = []

        def take_batch():
            nonlocal queue_finished, deferred, prefix_remainder
            if prefix_remainder:
                batch, prefix_remainder = _take_shared_prefix_chunk(
                    prefix_remainder, budget, max_examples_per_batch)
                return batch
            if queue_finished and deferred is None:
                return []
            if deferred is not None:
                first, deferred = deferred, None
            else:
                first = work_queue.get()
                if first is None:
                    queue_finished = True
                    return []
            if (isinstance(first, dict)
                    and SHARED_PREFIX_WORK_KEY in first):
                batch, prefix_remainder = _take_shared_prefix_chunk(
                    first[SHARED_PREFIX_WORK_KEY], budget,
                    max_examples_per_batch)
                return batch
            batch = [first]
            maximum = example_length(first)
            gate = example_gate(first)
            while (not queue_finished
                   and len(batch) < max_examples_per_batch):
                candidate = work_queue.get()
                if candidate is None:
                    queue_finished = True
                    break
                if (isinstance(candidate, dict)
                        and SHARED_PREFIX_WORK_KEY in candidate):
                    deferred = candidate
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
                            loss, metrics = training.clipped_policy_loss(
                                cfg, current_lp,
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
                            weight = float(example["sample_weight"])
                            if not torch.isfinite(loss).all():
                                raise FloatingPointError(
                                    "nonfinite rank loss")
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
                    attempt_entropy_examples,
                )

            result, effective_batches, quarantined = (
                training._run_oom_resilient_backward(
                    model, [local_batch], attempt,
                    device_label=f"cuda:{logical_id}",
                    expand_memory=expand_for_long_rollout))
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
    reclaim_unused_host_memory(
        torch, cuda_devices=(logical_id,), force=True)

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
        "long_rollout_memory_rescue": (
            emergency_allocator_fraction is not None),
        "token_budget": budget,
    }


def local_policy_update(backend, model, tokenizer, examples, cfg, logical_id,
                        token_budget, total_examples, *,
                        memory_fraction=0.80, work_queue=None,
                        adaptive_batches=True):
    """Accumulate one rank's exact entropic/GRPO/CVaR gradients.

    This is the process-per-GPU counterpart of
    ``ReplicatedDataParallelTrainer.train_policy``.  The global example count
    is supplied by rank 0 so summing LoRA gradients across ranks reproduces the
    original global mean loss exactly, independent of dynamic work stealing.
    """
    from contextlib import nullcontext
    import torch
    import train_multy_CVaR as training
    from host_memory import reclaim_unused_host_memory
    from model_backend import fused_long_attention_is_active

    if not 0.0 < float(memory_fraction) <= (
            MAX_STANDARD_TRAINING_MEMORY_FRACTION):
        raise ValueError(
            "fast trainer memory_fraction must be in (0, 0.90]")
    total_examples = int(total_examples)
    if total_examples < 1:
        raise ValueError("fast policy update requires at least one example")
    sequence_policy = training._uses_sequence_level_policy_ratio(cfg)

    backend.set_training_mode()
    allocator_fraction = set_total_memory_ceiling(
        logical_id, float(memory_fraction))
    parameters = trainable_parameters(model)
    gradient_accumulators = [None for _ in parameters]
    total_loss = 0.0
    total_logp_delta = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    ratio_count = 0
    quarantined_examples = 0
    backward_batches = 0
    trained_examples = 0
    peak_memory_fraction = 0.0
    kl_errors = []
    budget = max(1, int(token_budget or cfg.max_seq_length))
    emergency_allocator_fraction = None
    emergency_allocator_stage = 0

    def expand_for_long_rollout():
        nonlocal allocator_fraction, emergency_allocator_fraction
        nonlocal emergency_allocator_stage
        while emergency_allocator_stage < len(
                LONG_ROLLOUT_MEMORY_FRACTIONS):
            requested_fraction = LONG_ROLLOUT_MEMORY_FRACTIONS[
                emergency_allocator_stage]
            emergency_allocator_stage += 1
            try:
                expanded = set_long_rollout_memory_ceiling(
                    logical_id, requested_fraction)
            except RuntimeError as error:
                print(f"[train-oom] cuda:{logical_id}: "
                      f"{100.0 * requested_fraction:.0f}% reserved GPU "
                      f"headroom is unavailable ({error})", flush=True)
                continue
            if (emergency_allocator_fraction is not None
                    and expanded <= emergency_allocator_fraction):
                continue
            emergency_allocator_fraction = expanded
            allocator_fraction = max(allocator_fraction, expanded)
            return expanded
        return None

    examples = list(examples or ())
    if work_queue is not None and examples:
        raise ValueError(
            "fast trainer accepts either local examples or a shared queue")

    def example_length(example):
        return int(
            example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])

    def needs_fused_long_singleton(example):
        return bool(
            fused_long_attention_is_active()
            and training._requires_fused_long_singleton(
                cfg, example_length(example))
        )

    def fused_pack_limit():
        if (fused_long_attention_is_active()
                and getattr(cfg, "fused_long_attention", False)):
            return training._FUSED_LONG_SINGLETON_MIN_TOKENS
        return None

    configured_example_cap = max(
        1, int(getattr(cfg, "train_examples_per_microbatch", 1) or 1))
    max_examples_per_batch = (
        64 if adaptive_batches else configured_example_cap)
    if work_queue is None:
        pending = sorted(examples, key=example_length)
        max_examples_per_batch = min(
            max_examples_per_batch, max(1, len(examples)))

        def take_batch():
            if not pending:
                return []
            batch = [pending.pop()]
            maximum = example_length(batch[0])
            if needs_fused_long_singleton(batch[0]):
                return batch
            while pending and len(batch) < max_examples_per_batch:
                candidate = pending[-1]
                if needs_fused_long_singleton(candidate):
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
                    break
                batch.append(pending.pop())
                maximum = next_maximum
            return batch
    else:
        # Strategy work items retain the complete shared prompt on one rank so
        # the exact blockwise scorer evaluates that prompt only once per pack.
        queue_finished = False
        deferred = None
        prefix_remainder = []

        def take_batch():
            nonlocal queue_finished, deferred, prefix_remainder
            if prefix_remainder:
                batch, prefix_remainder = _take_shared_prefix_chunk(
                    prefix_remainder, budget, max_examples_per_batch,
                    singleton_predicate=needs_fused_long_singleton,
                    packed_token_limit=fused_pack_limit())
                return batch
            if queue_finished and deferred is None:
                return []
            if deferred is not None:
                first, deferred = deferred, None
            else:
                first = work_queue.get()
                if first is None:
                    queue_finished = True
                    return []
            if (isinstance(first, dict)
                    and SHARED_PREFIX_WORK_KEY in first):
                batch, prefix_remainder = _take_shared_prefix_chunk(
                    first[SHARED_PREFIX_WORK_KEY], budget,
                    max_examples_per_batch,
                    singleton_predicate=needs_fused_long_singleton,
                    packed_token_limit=fused_pack_limit())
                return batch
            batch = [first]
            maximum = example_length(first)
            if needs_fused_long_singleton(first):
                return batch
            while (not queue_finished
                   and len(batch) < max_examples_per_batch):
                candidate = work_queue.get()
                if candidate is None:
                    queue_finished = True
                    break
                if (isinstance(candidate, dict)
                        and SHARED_PREFIX_WORK_KEY in candidate):
                    deferred = candidate
                    break
                if needs_fused_long_singleton(candidate):
                    deferred = candidate
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
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
        fused_long_singleton = bool(
            len(cpu_batch) == 1
            and needs_fused_long_singleton(cpu_batch[0]))

        with torch.cuda.device(logical_id):
            base_allocated = int(torch.cuda.memory_allocated())
            base_reserved = int(torch.cuda.memory_reserved())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            external_bytes = max(
                0, int(total_bytes) - int(free_bytes) - base_reserved)
            torch.cuda.reset_peak_memory_stats()
            local_batch = _move_examples(cpu_batch, logical_id)

            def attempt(active_batches):
                attempt_losses = []
                attempt_logp_deltas = []
                attempt_ratio_means = []
                attempt_ratio_maxima = []
                attempt_ratio_count = 0
                attempt_kl_error = None

                # Match the existing standard-policy trainer exactly. Unlike
                # PPO's frozen-policy likelihood pass, this objective retains
                # the model's ordinary training-mode dropout behavior.
                with nullcontext():
                    for batch in active_batches:
                        base_logprobs = [
                            example.get("reference_logprobs")
                            for example in batch
                        ]
                        supplied_reference = all(
                            training._valid_example_token_logprobs(
                                example, "reference_logprobs")
                            for example in batch)
                        if not supplied_reference:
                            try:
                                with (backend.disable_adapter(),
                                      torch.no_grad()):
                                    base_logprobs = (
                                        training.compute_batched_token_logprobs(
                                            model, batch, with_grad=False,
                                            chunk=cfg.logprob_chunk,
                                            pad_token_id=(
                                                tokenizer.pad_token_id)))
                            except Exception as error:
                                raise RuntimeError(
                                    "exact base-policy logprob fallback "
                                    "failed; refusing to replace the "
                                    "configured KL penalty with the current "
                                    "policy") from error
                        current_logprobs = (
                            training.compute_batched_token_logprobs(
                                model, batch, with_grad=True,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id))

                        batch_losses = []
                        for example, current_lp, base_lp in zip(
                                batch, current_logprobs, base_logprobs):
                            base_lp = base_lp.to(current_lp.device)
                            advantage = example["advantage"]
                            logp_difference = (
                                current_lp - base_lp).detach()
                            average_difference = logp_difference.mean()
                            kl_advantage = cfg.kl_penalty_coef * (
                                average_difference
                                - (current_lp - base_lp))
                            effective_advantage = advantage + kl_advantage

                            has_behavior = (
                                training._valid_example_token_logprobs(
                                    example, "behavior_logprobs"))
                            behavior_lp = (
                                example["behavior_logprobs"].to(
                                    current_lp.device)
                                if has_behavior else None)
                            if sequence_policy:
                                loss, policy_metrics = (
                                    training.
                                    _a3b_sequence_clipped_standard_loss(
                                        cfg, current_lp, behavior_lp,
                                        base_lp, advantage))
                                importance_ratio = policy_metrics["ratio"]
                            elif has_behavior:
                                importance_ratio = (
                                    training.
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
                                attempt_ratio_means.append(
                                    importance_ratio.mean().detach())
                                attempt_ratio_maxima.append(
                                    importance_ratio.max().detach())
                                attempt_ratio_count += 1
                            if not torch.isfinite(loss).all():
                                raise FloatingPointError(
                                    "nonfinite policy loss")
                            batch_losses.append(loss / total_examples)
                            attempt_losses.append(loss.detach())
                            attempt_logp_deltas.append(
                                logp_difference.mean().detach())
                        if batch_losses:
                            sum(batch_losses[1:],
                                batch_losses[0]).backward()

                zero = torch.zeros(
                    (), dtype=torch.float64,
                    device=torch.device(f"cuda:{logical_id}"))
                packed = torch.stack([
                    (torch.stack(attempt_losses).sum()
                     if attempt_losses else zero).to(torch.float64),
                    (torch.stack(attempt_logp_deltas).sum()
                     if attempt_logp_deltas else zero).to(torch.float64),
                    (torch.stack(attempt_ratio_means).sum()
                     if attempt_ratio_means else zero).to(torch.float64),
                    (torch.stack(attempt_ratio_maxima).max()
                     if attempt_ratio_maxima else zero).to(torch.float64),
                ]).detach().cpu().tolist()
                return (
                    packed[0], packed[1], packed[2], packed[3],
                    attempt_ratio_count,
                    attempt_kl_error,
                )

            result, effective_batches, quarantined = (
                training._run_oom_resilient_backward(
                    model, [local_batch], attempt,
                    device_label=f"cuda:{logical_id}",
                    expand_memory=expand_for_long_rollout,
                    start_checkpointed=fused_long_singleton,
                    start_with_headroom=fused_long_singleton))
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
                total_loss += result[0]
                total_logp_delta += result[1]
                ratio_sum += result[2]
                ratio_max = max(ratio_max, result[3])
                ratio_count += result[4]
                if result[5] is not None:
                    kl_errors.append(result[5])

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
            if adaptive_batches and not effective_batches:
                budget = max(1, budget // 2)
            elif adaptive_batches and backed_off:
                budget = max(
                    longest_successful,
                    min(budget, successful_padded_tokens))
            elif adaptive_batches:
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
    reclaim_unused_host_memory(
        torch, cuda_devices=(logical_id,), force=True)

    return {
        "total_loss": total_loss,
        "total_logp_delta": total_logp_delta,
        "ratio_sum": ratio_sum,
        "ratio_max": ratio_max,
        "ratio_count": ratio_count,
        "quarantined_examples": quarantined_examples,
        "backward_batches": backward_batches,
        "trained_examples": trained_examples,
        "peak_memory_fraction": peak_memory_fraction,
        "allocator_memory_fraction": allocator_fraction,
        "long_rollout_memory_rescue": (
            emergency_allocator_fraction is not None),
        "token_budget": budget,
        "kl_errors": kl_errors,
    }


def worker_main(rank, world_size, cfg_dict, init_method, work_queue,
                command_queue, result_queue, dependency_log_path=None):
    """Persistent rank>0 worker command loop."""
    dist_initialized = False
    try:
        import gc
        import os
        from datetime import timedelta
        import train_multy_CVaR as training

        training._install_console_timestamps()
        terminal_log = training._install_terminal_log()
        terminal_log_path = cfg_dict.get("_terminal_log_path")
        if terminal_log_path:
            terminal_log.bind(terminal_log_path)
        setting_log_path = cfg_dict.get("_setting_log_path")
        if setting_log_path:
            # Use the compatibility binding exported by the training module.
            # Logging-version skew must not abort a distributed update after
            # rollout generation and evaluation have completed.
            training.bind_setting_log(
                setting_log_path,
                time_offset=cfg_dict.get("_log_time_offset_seconds", 0),
            )

        from loading_logs import quiet_replica_load

        rank_log = (f"{dependency_log_path}.trainer-rank{rank}"
                    if dependency_log_path else None)
        with quiet_replica_load(rank_log):
            import torch
            import torch.distributed as dist
            from model_backend import load_backend

            torch.set_num_threads(max(
                1, int(os.cpu_count() or world_size) // int(world_size)))
            torch.cuda.set_device(int(rank))
            memory_fraction = float(
                cfg_dict.get("training_memory_fraction", 0.80))
            set_total_memory_ceiling(int(rank), memory_fraction)
            cfg = SimpleNamespace(**dict(cfg_dict))
            cfg.training_replica_device = int(rank)
            cfg.num_training_gpus = 1
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
            if kind == "purge_host_memory":
                from host_memory import reclaim_unused_host_memory

                with torch.cuda.device(int(rank)):
                    torch.cuda.synchronize(int(rank))
                cleanup = reclaim_unused_host_memory(
                    torch, cuda_devices=(int(rank),), force=True)
                result_queue.put({
                    "event": "host_memory_purged",
                    "rank": int(rank),
                    "cleanup": cleanup,
                })
            elif kind == "offload":
                if not offloaded:
                    backend.offload_for_generation()
                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        torch.cuda.empty_cache()
                    offloaded = True
                result_queue.put({"event": "offloaded", "rank": int(rank)})
            elif kind == "restore":
                if offloaded:
                    from host_memory import reclaim_unused_host_memory

                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        torch.cuda.empty_cache()
                        set_total_memory_ceiling(
                            int(rank), memory_fraction)
                        backend.restore_after_generation()
                        backend.set_training_mode()
                        torch.cuda.synchronize(int(rank))
                    validate_model_device(model, int(rank))
                    # ``Module.to(cuda)`` releases its former CPU storages,
                    # but glibc and the pinned allocator can retain those
                    # arenas. Return them before long-sequence training starts.
                    reclaim_unused_host_memory(
                        torch, cuda_devices=(int(rank),), force=True)
                    offloaded = False
                result_queue.put({"event": "restored", "rank": int(rank)})
            elif kind == "train_rank":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received training while offloaded")
                step_cfg = SimpleNamespace(**dict(
                    command.get("step_cfg") or vars(cfg)))
                stats = local_rank_update(
                    backend, model, tokenizer, (), step_cfg,
                    int(rank), command["token_budget"],
                    memory_fraction=command.get("memory_fraction", 0.80),
                    work_queue=work_queue)
                result_queue.put({
                    "event": "computed", "rank": int(rank),
                    "stats": stats,
                })
            elif kind == "train_policy":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received training while offloaded")
                step_cfg = SimpleNamespace(**dict(
                    command.get("step_cfg") or vars(cfg)))
                stats = local_policy_update(
                    backend, model, tokenizer, (), step_cfg,
                    int(rank), command["token_budget"],
                    command["total_examples"],
                    memory_fraction=command.get("memory_fraction", 0.80),
                    work_queue=work_queue,
                    adaptive_batches=bool(command.get(
                        "adaptive_batches", True)))
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
                    int(rank),
                    context_group_ids=command["context_group_ids"],
                    group_ids=command["group_ids"])
                result_queue.put({
                    "event": "calibrated", "rank": int(rank),
                    "calibration": calibration,
                })
            elif kind == "finish_update":
                gradient_scale = float(command.get("gradient_scale", 1.0))
                apply_update = bool(command.get("apply_update", True))
                if not math.isfinite(gradient_scale) or gradient_scale <= 0.0:
                    raise ValueError(
                        "finish_update gradient_scale must be finite and "
                        "positive")
                if apply_update:
                    if gradient_scale != 1.0:
                        with torch.no_grad():
                            for parameter in model.parameters():
                                if (parameter.requires_grad
                                        and parameter.grad is not None):
                                    parameter.grad.mul_(gradient_scale)
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
