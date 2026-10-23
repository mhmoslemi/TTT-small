"""Resident, exact context-parallel overflow for the hybrid Qwen HF trainer.

Every rank retains its own model and computes a contiguous token shard. Full
attention exchanges K/V with an autograd-aware gather; FLA exchanges DeltaNet
states in both directions. Only token logprobs, not vocabulary logits, are
gathered for the existing sequence-level policy loss.
"""

from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def context_parallel_groups(world_size):
    """All ranks create these groups once, in identical order."""
    return [tuple(range(start, start + width))
            for width in (2 ** power for power in range(1, world_size.bit_length()))
            for start in range(0, world_size, width)]


def initialize_groups(world_size):
    return {ranks: dist.new_group(list(ranks), timeout=timedelta(minutes=30))
            for ranks in context_parallel_groups(world_size)}


def availability(model, cfg, world_size):
    """Never substitute independent sequence chunks or a slow recurrent kernel."""
    if (getattr(cfg, "coder_model_profile", "") != "qwen3.8-27b"
            or getattr(cfg, "advantage_mode", "") != "spo-rs"
            or world_size < 2 or world_size & (world_size - 1)):
        return False, "not a supported hybrid Qwen replicated SPO-RS layout"
    try:
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from torch.nn.attention.bias import causal_lower_right
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if "cp_context" not in inspect.signature(chunk_gated_delta_rule).parameters:
            return False, "installed FLA does not expose DeltaNet context parallelism"
    except (ImportError, AttributeError, ValueError) as error:
        return False, f"context-parallel kernels unavailable: {error}"
    modules = [module for module in model.modules()
               if type(module).__name__ == "Qwen3_5GatedDeltaNet"]
    if not modules or any(not isinstance(getattr(module, "conv1d", None), torch.nn.Conv1d)
                          or not callable(getattr(module, "causal_conv1d_fn", None))
                          or not callable(getattr(module, "chunk_gated_delta_rule", None))
                          for module in modules):
        return False, "hybrid layer contract differs from the supported Qwen implementation"
    attention_modules = [module for module in model.modules()
                         if type(module).__name__ == "Qwen3_5Attention"]
    if not attention_modules or any(
            getattr(module.config, "_attn_implementation", None) != "ttt_blockwise_attention"
            for module in attention_modules):
        return False, "context parallelism requires the registered exact attention backend"
    config = attention_modules[0].config
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear", "yarn"):
        return False, "sequence-dependent rotary scaling needs a shared global frequency update"
    return True, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.rank = dist.get_rank(group)
        parts = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, value.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, gradient):
        # Summation, not averaging: earlier tokens also receive gradients from
        # every later rank that attended to them.
        source = gradient.movedim(ctx.dim, 0).contiguous()
        width = source.shape[0] // dist.get_world_size(ctx.group)
        if dist.get_backend(ctx.group) == "nccl":
            result = torch.empty_like(source[:width])
            dist.reduce_scatter_tensor(result, source, group=ctx.group)
        else:
            source = source.clone()
            dist.all_reduce(source, group=ctx.group)
            result = source.narrow(0, ctx.rank * width, width).contiguous()
        return result.movedim(0, ctx.dim), None, None


def gather(value, group, dim=0):
    return _Gather.apply(value, group, dim)


def attention(query, key, value, context, *, scale=None, dropout=0.0):
    """Local queries attend their complete global causal prefix, including GQA."""
    import torch.nn.functional as F

    if dropout:
        raise ValueError("context-parallel policy attention requires disabled dropout")
    keys = gather(key, context.group, 2)
    values = gather(value, context.group, 2)
    prefix = (context.rank + 1) * query.shape[2]
    keys, values = keys[:, :, :prefix], values[:, :, :prefix]
    options = dict(dropout_p=0.0, scale=scale)
    if query.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.attention.bias import causal_lower_right
        # A lower-right causal bias preserves global positions without ever
        # allocating a local_queries x global_tokens dense mask.
        options.update(attn_mask=causal_lower_right(query.shape[2], prefix),
                       enable_gqa=query.shape[1] != keys.shape[1])
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            result = F.scaled_dot_product_attention(query, keys, values, **options)
    else:
        # Small CPU equivalence tests only; CUDA is never allowed a math fallback.
        repeats = query.shape[1] // keys.shape[1]
        keys, values = (item.repeat_interleave(repeats, dim=1)
                        for item in (keys, values))
        positions = torch.arange(query.shape[2], device=query.device) + context.rank * query.shape[2]
        options["attn_mask"] = torch.arange(prefix, device=query.device)[None, :] <= positions[:, None]
        result = F.scaled_dot_product_attention(query, keys, values, **options)
    return result.transpose(1, 2).contiguous(), None


def causal_convolution(x, weight, bias=None, *, activation=None, seq_idx=None,
                       context, kernel):
    """Keep the native fused convolution, including its BF16 rounding and backward."""
    if seq_idx is not None:
        raise ValueError("context-parallel convolution requires one logical sequence")
    length, width = x.shape[-1], weight.shape[-1] - 1
    if width:
        tails = gather(x[:, :, -width:], context.group, 2)
        start = max(0, context.rank - 1) * width
        halo = tails[:, :, start:start + width] * int(context.rank > 0)
        x = torch.cat((halo.transpose(1, 2), x.transpose(1, 2)), dim=1).transpose(1, 2)
    # F.conv1d(...).silu() rounds the intermediate convolution to BF16;
    # the native kernel applies SiLU in FP32 before rounding just once.
    return kernel(x=x, weight=weight, bias=bias, activation=activation,
                  seq_idx=None)[:, :, -length:]


@contextmanager
def hybrid_context(model, group, total_tokens, kernel_size):
    from fla.ops.cp import build_cp_context
    from model_backend import set_gradient_checkpointing

    checkpointed = bool(getattr(model, "is_gradient_checkpointing", False))
    changed = []
    missing = object()
    try:
        set_gradient_checkpointing(model, True)
        device = next(model.get_input_embeddings().parameters()).device
        context = SimpleNamespace(group=group, rank=dist.get_rank(group),
                                  size=dist.get_world_size(group))
        context.fla_context = build_cp_context(
            torch.tensor([0, total_tokens], device=device, dtype=torch.long),
            group=group, conv1d_kernel_size=kernel_size)
        for module in model.modules():
            name = type(module).__name__
            if name not in ("Qwen3_5GatedDeltaNet", "Qwen3_5Attention"):
                continue
            names = ["_ttt_context_parallel"]
            if name == "Qwen3_5GatedDeltaNet":
                names.extend(("causal_conv1d_fn", "chunk_gated_delta_rule"))
            changed.append((module, {key: module.__dict__.get(key, missing) for key in names}))
            module._ttt_context_parallel = context
            if name == "Qwen3_5GatedDeltaNet":
                module.causal_conv1d_fn = partial(
                    causal_convolution, context=context, kernel=module.causal_conv1d_fn)
                module.chunk_gated_delta_rule = partial(
                    module.chunk_gated_delta_rule, cp_context=context.fla_context)
        yield context
    finally:
        for module, originals in reversed(changed):
            for name, original in originals.items():
                if original is missing:
                    module.__dict__.pop(name, None)
                else:
                    setattr(module, name, original)
        set_gradient_checkpointing(model, checkpointed)


def token_logprobs(model, example, tokenizer, cfg, context, *, with_grad=True, measure=False):
    """Preserve every response target, its predecessor and its global position."""
    import train_multy_CVaR as training
    from training_checkpointing import grouped_qwen_hidden

    decoder, head = training._causal_lm_decoder_and_head(model)
    device = next(model.get_input_embeddings().parameters()).device
    prompt, response = example["prompt_ids"][0], example["response_ids"][0]
    total = prompt.numel() + response.numel() - 1
    part = math.ceil(total / (64 * context.size)) * 64
    source = torch.cat((prompt, response[:-1])).to(device)
    source = torch.nn.functional.pad(source, (0, part * context.size - total), value=tokenizer.pad_token_id)
    start, end = context.rank * part, (context.rank + 1) * part
    ids = source[start:end].unsqueeze(0)
    positions = torch.arange(start, end, device=device).unsqueeze(0)
    with torch.set_grad_enabled(with_grad):
        hidden = grouped_qwen_hidden(
            decoder, ids, None, position_ids=positions,
            group_size=max(1, math.ceil(math.sqrt(len(decoder.layers)))), force=True)
        if hidden is None:
            hidden = decoder(input_ids=ids, position_ids=positions,
                             attention_mask=None, use_cache=False, return_dict=True).last_hidden_state
        first, last = max(start, prompt.numel() - 1), min(end, total)
        scores = hidden.sum() * 0.0 + hidden.new_zeros((part,), dtype=torch.float32)
        entropies = torch.zeros_like(scores, requires_grad=False) if measure else None
        if first < last:
            targets = response[first - prompt.numel() + 1:last - prompt.numel() + 1].to(device)
            result = training._score_hidden_token_chunks(
                hidden[0, first - start:last - start], targets, head,
                chunk=cfg.logprob_chunk, with_grad=with_grad,
                return_entropy="measure" if measure else False)
            chosen, entropy = result if measure else (result, None)
            scores = scores.index_copy(0, torch.arange(first - start, last - start, device=device), chosen)
            if measure:
                entropies[first - start:last - start] = entropy.detach()
        scores = gather(scores, context.group)[prompt.numel() - 1:total]
        if measure:
            entropies = gather(entropies, context.group)[prompt.numel() - 1:total]
        return scores, entropies


def compute_context_work(backend, tokenizer, example, cfg, group, ranks, *, validate=True):
    """Same weighted sequence loss; sum token-shard adapter gradients once."""
    import train_multy_CVaR as training
    from exact_overflow import pack_cpu_gradients
    from fast_distributed import set_long_rollout_memory_ceiling

    model = backend.model
    example = dict(example)
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    set_long_rollout_memory_ceiling(device.index, 0.96)
    backend.set_training_mode()
    if validate and ranks not in getattr(backend, "_validated_context_groups", set()):
        issue = validate_context_group(backend, tokenizer, example, cfg, group, ranks)
        if issue is not None:
            return dict(completed=False, validation_issue=issue)
        if not hasattr(backend, "_validated_context_groups"):
            backend._validated_context_groups = set()
        backend._validated_context_groups.add(ranks)
    total = example["prompt_ids"].shape[1] + example["response_ids"].shape[1] - 1
    part = math.ceil(total / (64 * len(ranks))) * 64
    delta = next(module for module in model.modules() if type(module).__name__ == "Qwen3_5GatedDeltaNet")
    descriptor = example.get("_entropy_measurement")
    measure = bool(descriptor and not Path(descriptor["path"]).exists())
    with hybrid_context(model, group, part * len(ranks), delta.conv1d.kernel_size[0]) as context:
        with training._policy_dropout_disabled(model):
            missing = training._initialize_behavior_logprobs([example])
            old = example.get("old_policy_logprobs")
            if missing:
                old, _ = token_logprobs(model, example, tokenizer, cfg, context, with_grad=False)
            current, entropy = token_logprobs(model, example, tokenizer, cfg, context, measure=measure)
            epsilon, low, high, kl = training._clipped_policy_options(cfg)
            loss, metrics = training.clipped_policy_loss(
                cfg, current, old.to(device), None, example["advantage"],
                clip_epsilon=epsilon, clip_epsilon_low=low,
                clip_epsilon_high=high, kl_coef=kl)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("nonfinite context-parallel policy loss")
            weight = float(example["sample_weight"])
            # Every rank sees the gathered loss. Its gather backward sums those
            # copies; divide here, then SUM parameter gradients across the group.
            (weight * loss / context.size).backward()
            if measure and context.rank == 0:
                from entropy_tools import _record_tensor, _warn_once
                try:
                    _record_tensor(descriptor, entropy)
                except (OSError, ValueError, RuntimeError) as error:
                    _warn_once("could not record policy entropy", error)
    from fast_distributed import reduce_trainable_gradients

    reduce_trainable_gradients(model, destination_rank=ranks[0], group=group)
    gradients = {name: parameter.grad.detach().cpu()
                 for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    totals = {name: weight * float(metrics[name]) for name in
              ("policy_loss", "kl_estimate", "entropy_estimate", "ratio")}
    totals.update(loss=weight * float(loss.detach()),
                  clipped_fraction=weight * float(metrics["clipped"]))
    result = dict(completed=True, stats=dict(
        totals=totals, max_ratio=float(metrics["ratio"]),
        max_prefix_ratio=float(metrics["prefix_ratio_max"]),
        entropy_examples=0, backward_batches=1))
    if dist.get_rank(group) == 0:
        result["gradient_pack"] = pack_cpu_gradients(gradients)
        if not validate:
            result["logprobs"] = current.detach().cpu()
    return result


def validate_context_group(backend, tokenizer, example, cfg, group, ranks):
    """Check real-model forward AND adapter gradients before using a new group.

    This tiny calibration is not a training sample and never updates weights.
    Incompatible kernel/model versions fall back before any new gradient is
    accepted into the durable accumulator.
    """
    import train_multy_CVaR as training
    from exact_overflow import unpack_cpu_gradients

    model = backend.model
    device = next(model.parameters()).device
    leader = dist.get_rank(group) == 0
    mini = dict(example, prompt_ids=example["prompt_ids"][:, :17],
                response_ids=example["response_ids"][:, :64 * len(ranks) - 33],
                advantage=1.0, sample_weight=1.0)
    mini.pop("_entropy_measurement", None)
    size = mini["response_ids"].shape[1]
    old = torch.empty(size, dtype=torch.float32, device=device)
    reference_gradients = None
    if leader:
        print(f"[train-context] checking forward/gradient equivalence on GPUs {list(ranks)} "
              "before accepting context-parallel training", flush=True)
        with training._policy_dropout_disabled(model):
            scores = training.compute_token_logprobs(
                model, mini["prompt_ids"].to(device), mini["response_ids"].to(device),
                with_grad=True, chunk=cfg.logprob_chunk)
            old.copy_(scores.detach())
            epsilon, low, high, kl = training._clipped_policy_options(cfg)
            loss, _ = training.clipped_policy_loss(
                cfg, scores, old, None, 1.0, clip_epsilon=epsilon,
                clip_epsilon_low=low, clip_epsilon_high=high, kl_coef=kl)
            loss.backward()
        reference_gradients = {name: parameter.grad.detach().cpu().clone()
                               for name, parameter in model.named_parameters()
                               if parameter.requires_grad and parameter.grad is not None}
        model.zero_grad(set_to_none=True)
    dist.broadcast(old, src=ranks[0], group=group)
    mini["old_policy_logprobs"] = old.detach().cpu()
    result = compute_context_work(backend, tokenizer, mini, cfg, group, ranks, validate=False)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    issue = None
    if leader:
        actual = unpack_cpu_gradients(result["gradient_pack"])
        missing = reference_gradients.keys() - actual.keys()
        difference = sum(float((actual[name].float() - value.float()).square().sum())
                         for name, value in reference_gradients.items() if name in actual)
        norm = sum(float(value.float().square().sum()) for value in reference_gradients.values())
        extra = sum(float(value.float().square().sum()) for name, value in actual.items()
                    if name not in reference_gradients)
        relative = math.sqrt((difference + extra) / max(norm, 1e-30))
        logprob_error = float((result["logprobs"] - old.cpu()).abs().max())
        if (missing or not math.isfinite(relative) or not math.isfinite(logprob_error)
                or relative > .03 or logprob_error > .05):
            accepted.zero_()
            issue = f"kernel equivalence check failed (gradient relative error={relative:.5g}, logprob error={logprob_error:.5g})"
        else:
            print(f"[train-context] GPUs {list(ranks)} equivalence passed: "
                  f"gradient relative error={relative:.5g}, logprob error={logprob_error:.5g}", flush=True)
    dist.broadcast(accepted, src=ranks[0], group=group)
    model.zero_grad(set_to_none=True)
    return None if accepted.item() else (issue or "context-parallel kernel equivalence check failed")


def plan_context_wave(examples, world_size, local_token_budget):
    """Fit activation shards, not full trajectories; keep group layouts reusable."""
    from exact_overflow import trajectory_key

    if world_size < 2 or world_size & (world_size - 1) or local_token_budget < 64:
        raise ValueError("context waves require power-of-two GPU groups and a positive token target")
    ordered = []
    for item in examples:
        tokens = item["prompt_ids"].shape[1] + item["response_ids"].shape[1] - 1
        width = 2
        while width < world_size and math.ceil(tokens / (64 * width)) * 64 > local_token_budget:
            width *= 2
        # This estimate came from a different, full-sequence replica path.
        # It chooses parallelism; it must not exclude full-context CP work.
        ordered.append((width, tokens, trajectory_key(item), item))
    ordered.sort(key=lambda row: row[:3], reverse=True)
    selected, available = [], world_size
    for width, tokens, key, item in ordered:
        if width <= available:
            selected.append([item, width])
            available -= width
    # Widen groups only in powers of two, so every later wave reuses the same
    # initialized NCCL groups rather than creating groups per trajectory.
    while available and selected:
        candidates = [row for row in selected if row[1] <= available]
        if not candidates:
            break
        row = min(candidates, key=lambda row: row[1])
        available -= row[1]
        row[1] *= 2
    selected.sort(key=lambda row: row[1], reverse=True)
    offset, result = 0, []
    for item, width in selected:
        result.append((item, tuple(range(offset, offset + width))))
        offset += width
    return result


def complete_context_overflow(trainer, examples, cfg, step_idx, state):
    """Reuse the durable gradient accumulator and take the same one update."""
    import time
    import train_multy_CVaR as training
    from exact_overflow import METRICS, trajectory_key, unpack_cpu_gradients
    from fast_distributed import broadcast_trainable_parameters

    if getattr(trainer, "_context_disabled_reason", None):
        return None
    supported, reason = availability(trainer.model, cfg, trainer.world_size)
    if not supported:
        if (getattr(cfg, "coder_model_profile", "") == "qwen3.8-27b"
                and not getattr(trainer, "_context_unavailable_reported", False)):
            print(f"[train-context] {reason}; retaining exact model-sharded overflow. "
                  "No dependencies are installed automatically.", flush=True)
            trainer._context_unavailable_reported = True
        return None
    pending = {tuple(key) for key in state["remaining_keys"]}
    by_key = {trajectory_key(item): item for item in trainer._cpu_examples(examples)}
    if len(by_key) != len(examples) or pending - by_key.keys():
        raise RuntimeError("context-parallel inputs do not match unfinished trajectories")
    if not pending:
        return None
    # The smallest trajectory that exhausted replica execution supplies an
    # observed activation budget. Reserve space for cross-rank K/V and comms.
    original = {tuple(key) for key in state["deferred_keys"]}
    lengths = [item["prompt_ids"].shape[1] + item["response_ids"].shape[1] - 1
               for key, item in by_key.items() if key in original]
    budget = max(64, int(min(lengths) * .8) // 64 * 64)
    budget = min(budget, state.get("context_local_token_budget", budget))
    state["context_local_token_budget"] = budget
    first_wave = plan_context_wave([by_key[key] for key in pending], trainer.world_size, budget)
    if not first_wave:
        return None
    completed_before = tuple(state["completed_keys"])
    trainer.restore_after_generation()
    if not getattr(trainer, "_context_parallel_groups", None):
        for queue in trainer._command_queues.values():
            queue.put({"kind": "init_context_parallel"})
        trainer._context_parallel_groups = initialize_groups(trainer.world_size)
        trainer._collect_event("context_ready", range(1, trainer.world_size))
    print(f"[train-context] step {step_idx}: resident context parallelism for "
          f"{len(pending)} pending trajectories; measured local activation target="
          f"{budget} tokens/GPU; full contexts and existing SPO-RS loss", flush=True)
    waves, processed = 0, 0
    try:
        while pending:
            wave = plan_context_wave([by_key[key] for key in pending], trainer.world_size, budget)
            waves += 1
            started = time.time()
            print(f"[train-context] step {step_idx}: wave {waves}: " + ", ".join(
                f"{trajectory_key(item)}->{list(ranks)}" for item, ranks in wave), flush=True)
            trainer._workers_idle = False
            active = set()
            for item, ranks in wave:
                for rank in ranks:
                    if rank:
                        active.add(rank)
                        trainer._command_queues[rank].put(dict(
                            kind="context_backward", example=item, gpu_ids=ranks,
                            step_cfg=dict(vars(cfg))))
            first, ranks = wave[0]
            results = {0: compute_context_work(
                trainer.backend, trainer.tokenizer, first, cfg,
                trainer._context_parallel_groups[ranks], ranks)}
            messages = trainer._collect_event("context_computed", sorted(active))
            results.update({rank: msg["result"] for rank, msg in messages.items()})
            trainer._workers_idle = True
            parameters = {name: value for name, value in trainer.model.named_parameters()
                          if value.requires_grad}
            trainer.model.zero_grad(set_to_none=True)
            stats, successful, validation_issue = [], set(), None
            for item, ranks in wave:
                result = results[ranks[0]]
                if not result["completed"]:
                    validation_issue = result.get("validation_issue")
                    if validation_issue:
                        continue
                    raise RuntimeError("context-parallel backward did not complete; no partial update applied")
                for name, value in unpack_cpu_gradients(result["gradient_pack"]).items():
                    parameter = parameters[name]
                    if value.shape != parameter.shape or not torch.isfinite(value).all():
                        raise RuntimeError(f"invalid context-parallel gradient: {name}")
                    value = value.to(parameter.device, dtype=parameter.dtype)
                    if parameter.grad is None:
                        parameter.grad = value.clone()
                    else:
                        parameter.grad.add_(value)
                stats.append(result["stats"])
                successful.add(trajectory_key(item))
            if successful:
                combined = dict(
                    totals={name: sum(row["totals"][name] for row in stats) for name in METRICS},
                    max_ratio=max(row["max_ratio"] for row in stats),
                    max_prefix_ratio=max(row["max_prefix_ratio"] for row in stats),
                    entropy_examples=sum(row["entropy_examples"] for row in stats),
                    backward_batches=len(successful))
                training._commit_exact_spo_rs_overflow_batch(
                    trainer.model, state, step_idx, successful, combined,
                    training_seconds=time.time() - started)
            trainer.model.zero_grad(set_to_none=True)
            pending.difference_update(successful)
            processed += len(successful)
            if validation_issue:
                trainer._context_disabled_reason = validation_issue
                print(f"[train-context] {validation_issue}; unfinished samples retain the "
                      "existing exact overflow path; no partial optimizer update", flush=True)
                return None
            print(f"[train-context] step {step_idx}: {len(pending)} trajectories remaining", flush=True)
            del results, messages, stats
        result = training._finalize_exact_spo_rs_overflow(
            trainer.model, trainer.optimizer, cfg, step_idx, state,
            completed_before=completed_before, label="resident context-parallel overflow")
        trainer._workers_idle = False
        for queue in trainer._command_queues.values():
            queue.put({"kind": "sync_update"})
        broadcast_trainable_parameters(trainer.model, source_rank=0)
        trainer._collect_event("updated", range(1, trainer.world_size))
        trainer.model.zero_grad(set_to_none=True)
        trainer._workers_idle = True
        result.update(training_parallel_gpus=trainer.world_size, training_fast=True,
                      training_fast_processes=True, exact_overflow_inprocess=True,
                      exact_overflow_context_parallel=True, exact_overflow_waves=waves,
                      exact_overflow_completed_examples=processed)
        return result
    except BaseException:
        trainer._workers_idle = False
        trainer._abort_workers()
        raise
