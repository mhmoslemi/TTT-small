"""Policy likelihood/loss helpers and replicated/process-distributed LoRA trainers."""

from contextlib import contextmanager, nullcontext
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from analysis.entropy_tools import measure_policy_entropy, token_entropy as _token_entropy
from diagnostics.console import _LOG_TIME_OFFSET_SECONDS, setting_log_only
from models.coder_profiles import _uses_sequence_level_policy_ratio
from runtime.configuration import (
    SEQUENCE_CLIP_EPSILON_DEFAULT, SPO_RS_CLIP_EPSILON_DEFAULT, SPO_RS_CLIP_EPSILON_HIGH_DEFAULT,
    SPO_RS_CLIP_EPSILON_LOW_DEFAULT, SPO_RS_UPDATE_EPOCHS_DEFAULT, _parse_gpu_ids,
)
from training.exact_overflow import (
    ExactLongContextTrainingOOM, _EXACT_OVERFLOW_GRADIENT_FILE, _clip_sharded_grad_norm_,
    _commit_exact_spo_rs_overflow_batch, _commit_exact_spo_rs_overflow_progress, _load_exact_spo_rs_overflow,
    _read_exact_sharded_recovery, _save_exact_spo_rs_overflow, _write_exact_sharded_recovery,
    _write_exact_spo_rs_overflow_state, _validate_precomputed_exact_gradients,
)
from training.fast_distributed import _move_optimizer_state, _restore_optimizer_state_to_parameters


_FUSED_LONG_SINGLETON_MIN_TOKENS = 8192


def _detached_behavior_importance_ratio(cfg, current_logprobs, behavior_logprobs):
    """Return the existing token ratios or the A3B trajectory ratio.

    Entropic updates use a detached importance weight rather
    than the differentiable clipped surrogate. Keep that objective unchanged
    except that plain Qwen3-30B-A3B receives one geometric-mean response weight
    instead of volatile individual-token MoE weights.
    """
    import torch

    current = current_logprobs.detach()
    behavior = torch.as_tensor(behavior_logprobs, dtype=current.dtype, device=current.device)
    log_ratios = current - behavior
    if _uses_sequence_level_policy_ratio(cfg):
        return log_ratios.mean().exp()
    return log_ratios.exp()


def compute_group_advantages(rewards_np):
    """Return entropic group advantages and their adaptive beta.

    SPO-RS obtains advantages from its frozen historical tracker instead.
    """
    from training.advantage import entropic_adaptive_advantages
    return entropic_adaptive_advantages(rewards_np)


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


def _score_hidden_token_chunks(hidden_states, targets, output_head, *, chunk, with_grad, return_entropy):
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
        chosen = log_probs.gather(1, target_chunk.unsqueeze(-1)).squeeze(-1)
        if not return_entropy:
            return chosen
        entropy = _token_entropy(log_probs, detached=(return_entropy == "measure"))
        return chosen, entropy

    chosen_parts = []
    entropy_parts = []
    for start in range(0, length, step):
        end = min(start + step, length)
        hidden_chunk = hidden_states[start:end]
        target_chunk = targets[start:end]
        if with_grad and hidden_chunk.requires_grad:
            from torch.utils.checkpoint import checkpoint
            result = checkpoint(project, hidden_chunk, target_chunk, use_reentrant=False)
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
    from training.training_checkpointing import grouped_qwen_hidden
    hidden = (grouped_qwen_hidden(decoder, input_ids, attention_mask) if with_grad else None)
    if hidden is None:
        outputs = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            try:
                hidden = outputs[0]
            except (IndexError, KeyError, TypeError):
                return None
    scored = []
    entropies = []
    for row, ((start, end), targets) in enumerate(zip(response_ranges, response_targets)):
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


def _shared_prefix_token_logprobs(model, examples, *, chunk, with_grad, return_entropy):
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
    if str(getattr(decoder_config, "_attn_implementation", "")) != ("ttt_blockwise_attention"):
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
    segment_parts = [torch.zeros(prompt_length, dtype=torch.long, device=device)]
    position_parts = [torch.arange(prompt_length, dtype=torch.long, device=device)]
    response_ranges = []
    attention_branches = []
    cursor = prompt_length
    for branch_index, example in enumerate(examples, start=1):
        response = example["response_ids"]
        if (response.ndim != 2 or int(response.shape[0]) != 1 or int(response.shape[1]) < 1):
            return None
        branch_input = response[0, :-1]
        branch_length = int(branch_input.shape[0])
        start = cursor
        end = start + branch_length
        response_ranges.append((start, end))
        if branch_length:
            input_parts.append(branch_input)
            segment_parts.append(torch.full((branch_length,), branch_index, dtype=torch.long, device=device))
            position_parts.append(torch.arange(
                prompt_length, prompt_length + branch_length,
                dtype=torch.long, device=device))
            attention_branches.append((start, end))
            cursor = end

    packed_ids = torch.cat(input_parts, dim=0).unsqueeze(0)
    segments = torch.cat(segment_parts, dim=0)
    positions = torch.cat(position_parts, dim=0)
    compact_mask = torch.stack((segments, positions), dim=0).unsqueeze(0)
    from models.model_backend import register_shared_prefix_attention_mask
    register_shared_prefix_attention_mask(compact_mask, prompt_length, attention_branches)

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
            prediction_hidden = torch.cat((prompt_last, hidden[0, start:end, :]), dim=0)
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
                    entropies.append(_token_entropy(lp, detached=(return_entropy == "measure")))
            gathered = torch.cat(parts, dim=1)  # (1, R)
            if return_entropy:
                entropy = torch.cat(entropies, dim=1)
        else:
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, R)
            if return_entropy:
                entropy = _token_entropy(log_probs, detached=(return_entropy == "measure"))
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
        if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or response_ids.ndim != 2 or response_ids.shape[0] != 1):
            raise ValueError("training examples must contain (1, length) token tensors")
        if prompt_ids.shape[1] < 1 or response_ids.shape[1] < 1:
            raise ValueError("training prompt/response tensors must be nonempty")

    device = prompts[0].device
    dtype = prompts[0].dtype
    totals = [int(prompt.shape[1] + response.shape[1]) for prompt, response in zip(prompts, responses)]
    max_total = max(totals)
    input_ids = torch.full((len(examples), max_total), int(pad_token_id), dtype=dtype, device=device)
    attention_mask = torch.zeros((len(examples), max_total), dtype=torch.long, device=device)
    for row, (prompt_ids, response_ids, total) in enumerate(zip(prompts, responses, totals)):
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
            response_ranges.append((prompt_len - 1, prompt_len - 1 + response_len))
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
        for row, (prompt_ids, response_ids) in enumerate(zip(prompts, responses)):
            prompt_len = int(prompt_ids.shape[1])
            response_len = int(response_ids.shape[1])
            pred_logits = logits[row, prompt_len - 1:prompt_len - 1 + response_len, :]
            targets = response_ids[0].to(device=pred_logits.device)
            step = (int(chunk) if chunk and 0 < int(chunk) < response_len else response_len)
            parts = []
            entropies = []
            for start in range(0, response_len, step):
                end = min(start + step, response_len)
                log_probs = F.log_softmax(pred_logits[start:end, :].float(), dim=-1)
                parts.append(log_probs.gather(1, targets[start:end].unsqueeze(-1)).squeeze(-1))
                if return_entropy:
                    entropies.append(_token_entropy(log_probs, detached=(return_entropy == "measure")))
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
    return bool(getattr(cfg, "fused_long_attention", False) and int(packed_tokens) >= _FUSED_LONG_SINGLETON_MIN_TOKENS)


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
        key=lambda example: int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1]),
        reverse=True,
    )
    current = []
    current_max = 0
    for example in ordered:
        length = int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
        if _requires_fused_long_singleton(cfg, length):
            if current:
                batches.append(current)
                current = []
                current_max = 0
            batches.append([example])
            continue
        next_max = max(current_max, length)
        exceeds_tokens = bool(current and next_max * (len(current) + 1) > token_cap)
        exceeds_fused_pack = bool(current and _exceeds_fused_long_pack(cfg, next_max * (len(current) + 1)))
        if current and (len(current) >= example_cap or exceeds_tokens or exceeds_fused_pack):
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
    return (isinstance(error, RuntimeError) and "out of memory" in str(error).lower() and "cuda" in str(error).lower())


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
        if (len(batch) <= 1 or _gradient_checkpointing_work_units([batch]) <= work_cap):
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
            and all(bool(example.get("_shared_prefix_packing_allowed", True)) for example in batch)
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
                int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
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
    from runtime.host_memory import (ProcessRSSPeakSampler,
                             host_cache_reclaim_supported,
                             process_rss_bytes,
                             reclaim_unused_host_memory)
    from models.model_backend import set_gradient_checkpointing

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
        checker = getattr(activation_offload_semaphore, "can_admit", None)
        return bool(checker(work)) if callable(checker) else True

    def offload_admission(work):
        if activation_offload_semaphore is None:
            return nullcontext()
        reservation = getattr(activation_offload_semaphore, "reservation", None)
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
        active, reused_split = _split_training_batches_to_work_cap(active, cached_work_cap)

    strict_exact_training = bool(getattr(model, "_ttt_strict_exact_long_training", False))
    activation_offload = False
    memory_expanded = False
    expanded_fraction = None
    split_backoff = False
    offload_trigger_work = None
    headroom_trigger_work = None
    checkpoint_threshold = getattr(model, "_ttt_gradient_checkpointing_min_work", None)
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
        if (expanded_fraction is not None and candidate <= float(expanded_fraction) + 1e-9):
            return None
        expanded_fraction = candidate
        return candidate

    def select_checkpointing():
        nonlocal checkpointing, checkpointing_available
        work = _gradient_checkpointing_work_units(active)
        wanted = bool(
            checkpointing_available
            and (bool(start_checkpointed) or (checkpoint_threshold is not None and work >= int(checkpoint_threshold)))
        )
        if wanted != checkpointing:
            if not set_gradient_checkpointing(model, wanted):
                checkpointing_available = False
                wanted = False
            checkpointing = wanted
        return work

    active_work = select_checkpointing()
    configured_headroom_min_work = getattr(model, "_ttt_reserved_headroom_min_work", None)
    headroom_min_work = profile.get("headroom_min_work", configured_headroom_min_work)
    if ((bool(start_with_headroom) or (headroom_min_work is not None and active_work >= int(headroom_min_work)))
            and expand_memory is not None):
        memory_expanded = expand_to_next_memory_ceiling() is not None
    configured_offload_min_work = getattr(model, "_ttt_activation_offload_min_work", None)
    offload_min_work = profile.get("activation_offload_min_work", configured_offload_min_work)
    if (allow_activation_offload
            and (bool(start_activation_offload)
                 or (offload_min_work is not None and active_work >= int(offload_min_work)))
            and offload_gate_can_admit(active_work)
            and hasattr(torch.autograd.graph, "save_on_cpu")):
        activation_offload = True

    reused_modes = []
    if reused_split:
        reused_modes.append(f"work cap {int(cached_work_cap)}")
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
            log_prefix = ("[train-memory]" if start_checkpointed or start_with_headroom else "[train-oom]")
            print(f"{log_prefix} {device_label}: {mode_source} ({', '.join(reused_modes)})", flush=True)
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
            admission = (offload_admission(active_work) if activation_offload else nullcontext())
            # Keep the admission token until inactive pinned blocks and libc
            # arenas have been returned. Otherwise the next trainer can begin
            # a large offload while this process still owns the prior batch's
            # host cache, recreating the node-wide RAM spike.
            with admission:
                offload_rss_before = (process_rss_bytes() if activation_offload else None)
                rss_sampler = (ProcessRSSPeakSampler() if activation_offload else nullcontext())
                offload_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=host_cache_reclaim_supported(torch))
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
                        reclaim_unused_host_memory(torch, cuda_devices=cuda_devices, force=True)
                    raise
                else:
                    if activation_offload:
                        cleanup = reclaim_unused_host_memory(torch, cuda_devices=cuda_devices, force=True)
                        # Forced cleanup records the pinned allocator and RSS
                        # immediately before releasing the saved activations.
                        # Publish the larger measurement while this rank still
                        # owns its reservation; waiting ranks then recompute
                        # their token-weighted request before they can enter.
                        observer = getattr(activation_offload_semaphore, "observe", None)
                        if callable(observer):
                            footprint = int(cleanup.get("pinned_owned_before", 0) or 0)
                            rss_at_cleanup = cleanup.get("rss_before")
                            if (offload_rss_before is not None and rss_at_cleanup is not None):
                                footprint = max(footprint, max(0, int(rss_at_cleanup) - int(offload_rss_before)))
                            footprint = max(footprint, int(getattr(sampled_rss, "footprint_bytes", 0) or 0))
                            observer(active_work, footprint)
            if split_backoff and active:
                successful_work = _gradient_checkpointing_work_units(active)
                _remember_oom_profile_minimum(profile, "safe_work_cap", successful_work)
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
                learned_checkpoint_floor = _oom_profile_work_floor(active_work)
                checkpoint_threshold = (
                    learned_checkpoint_floor
                    if checkpoint_threshold is None else
                    min(int(checkpoint_threshold), learned_checkpoint_floor)
                )
                setattr(model, "_ttt_gradient_checkpointing_min_work", int(checkpoint_threshold))
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
                reason = ("fused attention rejected the padded batch" if kernel_unavailable else "CUDA OOM")
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
                    retry_label = ("singleton OOM" if previous_fraction is None else "singleton still OOM")
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
                    and hasattr(torch.autograd.graph, "save_on_cpu")):
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
            victim_tokens = int(victim["prompt_ids"].shape[1] + victim["response_ids"].shape[1])
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
                    failure = ("no compatible exact linear-memory attention kernel")
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
    epsilon_low = (symmetric_epsilon if clip_epsilon_low is None else float(clip_epsilon_low))
    epsilon_high = (symmetric_epsilon if clip_epsilon_high is None else float(clip_epsilon_high))
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
    clipped_policy_ratios = policy_ratios.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    policy_loss = -torch.minimum(policy_ratios * adv, clipped_policy_ratios * adv).mean()
    ratio = policy_ratios.mean()
    clipped_fraction = (policy_ratios.detach() != clipped_policy_ratios.detach()).double().mean()
    policy_reference_delta = cur.new_zeros(())
    kl_policy_loss = cur.new_zeros(())
    if kl_coef:
        if reference_logprob is None:
            raise ValueError("reference logprobs are required for nonzero KL coefficient")
        ref = torch.as_tensor(reference_logprob, dtype=cur.dtype, device=cur.device).detach()
        if ref.shape != cur.shape or not torch.isfinite(ref).all():
            raise ValueError("reference logprobs must be a finite vector matching current logprobs")
        logp_difference = (cur - ref).detach()
        policy_reference_delta = logp_difference.mean()
        kl_advantage = kl_coef * (policy_reference_delta - (cur - ref))
        # This is the KL component of entropic mode's policy-gradient loss.
        # Keep the current/reference centering unchanged. In sequence-ratio
        # mode, one detached trajectory weight replaces volatile per-token MoE
        # weights without changing how pi_theta and pi_ref are compared.
        kl_policy_loss = -(policy_ratios.detach() * kl_advantage.detach() * cur).mean()

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
    return loss, {key: float(value.item()) for key, value in tensor_metrics.items()}


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
        epsilon = float(getattr(cfg, "spo_rs_clip_epsilon", SPO_RS_CLIP_EPSILON_DEFAULT))
        epsilon_low = float(getattr(cfg, "spo_rs_clip_epsilon_low", SPO_RS_CLIP_EPSILON_LOW_DEFAULT))
        epsilon_high = float(getattr(cfg, "spo_rs_clip_epsilon_high", SPO_RS_CLIP_EPSILON_HIGH_DEFAULT))
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


def _a3b_sequence_clipped_standard_loss(cfg, current_logprobs, behavior_logprobs, reference_logprobs, advantage):
    """Apply the A3B sequence surrogate inside entropic training.

    Entropic training retains its existing advantages, centered
    current/reference correction and sample normalization. Only the
    reward-policy importance ratio and clipping unit become one
    length-normalized response ratio. The helper is deliberately unavailable
    to every other coder model.
    """
    if not _uses_sequence_level_policy_ratio(cfg):
        raise ValueError("the sequence-clipped standard loss is Qwen3-30B-A3B only")
    epsilon, epsilon_low, epsilon_high, kl_coef = (_clipped_policy_options(cfg))
    old = (current_logprobs.detach() if behavior_logprobs is None else behavior_logprobs)
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
        raise RuntimeError(f'entropic overlap group {group_id} was prepared before all of its rewards completed')

    rewards = np.asarray([float(future.result().reward) for future in reward_futures], dtype=np.float64)
    if rewards.size == 0:
        return []
    advantages, _beta = compute_group_advantages(rewards)
    constant = float(rewards.max() - rewards.min()) < 1e-12
    if constant:
        return []

    prepared = []
    for rollout_index, (record, advantage) in enumerate(zip(responses, advantages)):
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
            "response_ids": torch.tensor([token_ids], dtype=torch.long, device="cpu"),
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


def _prepare_spo_rs_overlap_inputs(tokenizer, group_responses, reward_futures, prompt_jobs):
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
        for rollout_index, (record, reward_future) in enumerate(zip(responses, futures)):
            token_ids = list(record.get("token_ids") or ())
            behavior_values = record.get("behavior_logprobs")
            if not token_ids or behavior_values is None:
                continue
            try:
                behavior_array = np.asarray(behavior_values, dtype=np.float32)
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
                "response_ids": torch.tensor([token_ids], dtype=torch.long, device="cpu"),
                "_entropy_measurement": record.get("_entropy_measurement"),
                "advantage": None,
                "behavior_logprobs": torch.as_tensor(behavior_array, dtype=torch.float32).cpu(),
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
        raise RuntimeError("SPO-RS overlap received an example before its reward completed")
    if spo_rs_tracker is None:
        raise RuntimeError("SPO-RS overlap requires its historical tracker")
    rewards = [float(future.result().reward) for future in futures]
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    advantages = spo_rs_tracker.normalized_advantages(transformed)
    if advantages is None:
        raise RuntimeError("SPO-RS overlap lost its frozen pre-step tracker baseline")
    for example, advantage in zip(examples, advantages):
        example["advantage"] = float(advantage)
        # Futures cannot be sent through the process trainer's work queues.
        example.pop("_overlap_reward_future", None)
    return examples


def _apply_precomputed_exact_gradients(model, overflow_state, step_idx):
    """Add durable fast-path gradients to the model-sharded overflow pass."""
    import torch

    saved_gradients = _validate_precomputed_exact_gradients(model, overflow_state, step_idx)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            saved = saved_gradients[name]
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(saved.to(device=parameter.device, dtype=parameter.grad.dtype))


def _train_durable_exact_spo_rs_overflow(
        backend, model, tokenizer, optimizer, examples, cfg, step_idx,
        overflow_state, run_dir):
    """Finish selectively deferred SPO-RS trajectories one durable unit at a time."""
    import gc
    import torch
    from training.fast_distributed import set_long_rollout_memory_ceiling
    from runtime.host_memory import reclaim_unused_host_memory

    checkpoint_path = Path(str(overflow_state.get("path", ""))).resolve()
    expected_path = (Path(run_dir).resolve() / _EXACT_OVERFLOW_GRADIENT_FILE.format(step=int(step_idx)))
    if checkpoint_path != expected_path:
        raise RuntimeError(
            "exact SPO-RS overflow checkpoint is outside the active run or "
            "belongs to a different step")

    epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
    if epochs != 1:
        raise RuntimeError("selective exact SPO-RS overflow requires one optimizer update")
    epsilon, epsilon_low, epsilon_high, kl_coef = (_clipped_policy_options(cfg))
    backend.set_training_mode()
    cuda_devices = sorted({
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
        and parameter.device.index is not None
    })
    if not cuda_devices:
        raise RuntimeError("exact model-sharded overflow recovery found no CUDA shards")

    original_keys = {tuple(int(value) for value in key) for key in overflow_state.get("deferred_keys", ())}
    remaining_keys = {tuple(int(value) for value in key) for key in overflow_state.get("remaining_keys", ())}
    completed_before = {tuple(int(value) for value in key) for key in overflow_state.get("completed_keys", ())}
    example_by_key = {tuple(int(value) for value in example["_overlap_key"]): example for example in examples}
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
            if any(not torch.isfinite(value).all() for value in old_logprobs):
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
            key=lambda example: int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1]),
            reverse=True,
        )
        for ordinal, example in enumerate(ordered_examples, start=1):
            key = tuple(int(value) for value in example["_overlap_key"])
            token_count = int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
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
                    set_long_rollout_memory_ceiling(cuda_device, fraction)
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
                    for item, current_lp in zip(batch, current_logprobs):
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
                        totals["clipped_fraction"] += (weight * float(metrics["clipped"]))
                        max_ratio = max(max_ratio, metrics["ratio"])
                        max_prefix_ratio = max(max_prefix_ratio, metrics["prefix_ratio_max"])
                    if weighted_losses:
                        sum(weighted_losses[1:], weighted_losses[0]).backward()
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
            if (result is None or quarantined or sum(len(batch) for batch in effective) != 1):
                raise RuntimeError(f"exact overflow trajectory {key} did not complete")
            remaining_count = _commit_exact_spo_rs_overflow_progress(
                model, overflow_state, step_idx, key, result,
                training_seconds=time.time() - trajectory_started)
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            for cuda_device in cuda_devices:
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()
            reclaim_unused_host_memory(torch, cuda_devices=cuda_devices, force=True)
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
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    totals = {key: (float(normal_totals.get(key, 0.0)) + float(overflow_totals.get(key, 0.0))) for key in metric_keys}
    max_ratio = max(float(normal_stats.get("max_ratio", 0.0)), float(overflow_stats.get("max_ratio", 0.0)))
    max_prefix_ratio = max(
        float(normal_stats.get("max_prefix_ratio", 0.0)),
        float(overflow_stats.get("max_prefix_ratio", 0.0)))
    entropy_examples = (int(normal_stats.get("entropy_examples", 0)) + int(overflow_stats.get("entropy_examples", 0)))
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
        "exact_overflow_reused_examples": (int(normal_stats.get("trained_examples", 0)) + len(completed_before)),
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
            raise RuntimeError("selective exact SPO-RS overflow requires its run directory")
        return _train_durable_exact_spo_rs_overflow(
            backend, model, tokenizer, optimizer, examples, cfg, step_idx,
            precomputed_gradient_state, exact_overflow_run_dir)

    epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
    epsilon, epsilon_low, epsilon_high, kl_coef = (_clipped_policy_options(cfg))
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
            if any(not torch.isfinite(value).all() for value in old_logprobs):
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
                    for ex, cur_lp in zip(batch, current_logprobs):
                        loss, metrics = clipped_policy_loss(
                            cfg, cur_lp, ex["old_policy_logprobs"],
                            None, ex["advantage"],
                            clip_epsilon=epsilon,
                            clip_epsilon_low=epsilon_low,
                            clip_epsilon_high=epsilon_high,
                            kl_coef=kl_coef)
                        weight = float(ex["sample_weight"])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError("nonfinite SPO-RS loss")
                        weighted_losses.append(weight * loss)
                        totals["loss"] += (weight * float(loss.detach().item()))
                        for key in ("policy_loss", "kl_estimate",
                                    "entropy_estimate", "ratio"):
                            totals[key] += weight * metrics[key]
                        totals["clipped_fraction"] += (weight * float(metrics["clipped"]))
                        max_ratio = max(max_ratio, metrics["ratio"])
                        max_prefix_ratio = max(max_prefix_ratio, metrics["prefix_ratio_max"])
                    if weighted_losses:
                        sum(weighted_losses[1:], weighted_losses[0]).backward()
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
                _apply_precomputed_exact_gradients(model, precomputed_gradient_state, step_idx)
                normal_stats = precomputed_gradient_state.get("normal_stats", {})
                normal_totals = normal_stats.get("totals", {})
                for key in totals:
                    totals[key] += float(normal_totals.get(key, 0.0))
                max_ratio = max(max_ratio, float(normal_stats.get("max_ratio", 0.0)))
                max_prefix_ratio = max(max_prefix_ratio, float(normal_stats.get("max_prefix_ratio", 0.0)))
                entropy_examples += int(normal_stats.get("entropy_examples", 0))

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
    recovery_seconds = float((precomputed_gradient_state or {}).get("normal_stats", {}).get("training_seconds", 0.0))
    return {
        "policy_updates": history,
        "training_seconds": recovery_seconds + (time.time() - started),
        "oom_quarantined_examples": sum(item["oom_quarantined_examples"] for item in history),
        "exact_overflow_reused_examples": int(
            (precomputed_gradient_state or {}).get("normal_stats", {}).get("trained_examples", 0)),
        "exact_overflow_sharded_examples": (len(examples) if precomputed_gradient_state is not None else 0),
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

    def __init__(self, primary_backend, primary_model, primary_tokenizer, optimizer, cfg, dependency_log_path=None):
        import torch
        from models.model_backend import load_backend
        from diagnostics.loading_logs import quiet_replica_load

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
            replica_log = (f"{dependency_log_path}.trainer-rank{logical_id}" if dependency_log_path else None)
            with quiet_replica_load(replica_log):
                replica_backend = load_backend(cfg.backend, replica_cfg)
                replica_model, replica_tokenizer = replica_backend.load()
            self.replicas.append((replica_backend, replica_model, replica_tokenizer, logical_id))
            print(f"[train-parallel] replica {logical_id}/{world_size - 1} "
                  f"ready on physical GPU {physical_ids[logical_id]}", flush=True)

        self._validate_parameter_layouts()
        self._validate_replica_devices()
        self._broadcast_trainable_parameters()
        for logical_id in range(world_size):
            torch.cuda.synchronize(logical_id)
        print(f'[train-parallel] active on all {world_size} training GPUs {physical_ids}', flush=True)

    @property
    def world_size(self):
        return len(self.replicas)

    @property
    def is_offloaded(self):
        return self._offloaded

    def estimated_host_offload_bytes(self):
        """RAM required to move every live trainer replica off its GPU."""
        from runtime.host_memory import (GIB, estimated_module_bytes, estimated_optimizer_bytes)

        model_bytes = sum(estimated_module_bytes(model) for _, model, _, _ in self.replicas)
        optimizer_bytes = estimated_optimizer_bytes(self.optimizer)
        # The model estimate is storage-exact.  The small margin covers Python
        # tensor metadata, allocator alignment, and one GiB of workspace per
        # process without pretending that an additional model copy is needed.
        return int(1.05 * (model_bytes + optimizer_bytes) + self.world_size * GIB)

    def reclaim_host_memory(self):
        """Synchronize all replicas, then drop unused host allocations."""
        import torch
        from runtime.host_memory import reclaim_unused_host_memory

        devices = tuple(range(self.world_size))
        return [reclaim_unused_host_memory(torch, cuda_devices=devices, force=True)]

    @staticmethod
    def _trainable_parameters(model):
        return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    def _validate_parameter_layouts(self):
        expected = [
            (name, tuple(parameter.shape))
            for name, parameter in self._trainable_parameters(self.replicas[0][1])
        ]
        for _, model, _, logical_id in self.replicas[1:]:
            actual = [(name, tuple(parameter.shape)) for name, parameter in self._trainable_parameters(model)]
            if actual != expected:
                raise RuntimeError(f'trainer replica {logical_id} has a different trainable parameter layout')

    def _validate_replica_devices(self):
        import torch
        for _, model, _, logical_id in self.replicas:
            expected = torch.device(f"cuda:{logical_id}")
            wrong_devices = {str(parameter.device) for parameter in model.parameters() if parameter.device != expected}
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
                    target_parameter.copy_(source_parameter.detach().to(target_parameter.device, non_blocking=True))
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
            key=lambda ex: int(ex["prompt_ids"].numel() + ex["response_ids"].numel()),
            reverse=True,
        )
        for example in ordered:
            rank = min(range(self.world_size), key=lambda idx: loads[idx])
            shards[rank].append(example)
            loads[rank] += int(example["prompt_ids"].numel() + example["response_ids"].numel())
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
                        tensor_cache[cache_key] = value.to(device, non_blocking=True)
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

        local_shards = self._run_replicas(move, list(zip(self.replicas, shards)))
        counts = [len(shard) for shard in local_shards]
        print(f'[train-parallel] examples/GPU={counts}; tokens/GPU={token_loads}', flush=True)
        return local_shards

    def _zero_gradients(self):
        for _, model, _, _ in self.replicas:
            model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

    def _sum_gradients_to_primary(self):
        """Sum replica gradients without DDP's implicit world-size average."""
        import torch
        primary = self._trainable_parameters(self.replicas[0][1])
        others = [self._trainable_parameters(replica[1]) for replica in self.replicas[1:]]
        with torch.cuda.device(0), torch.no_grad():
            for parameter_index, (_, destination) in enumerate(primary):
                gradient = destination.grad
                if gradient is None:
                    gradient = torch.zeros_like(destination)
                    destination.grad = gradient
                for replica_parameters in others:
                    source_gradient = replica_parameters[parameter_index][1].grad
                    if source_gradient is not None:
                        gradient.add_(source_gradient.to(destination.device, non_blocking=True))
            torch.cuda.synchronize(0)


    def _clip_step_and_sync(self, cfg):
        import torch
        self._sum_gradients_to_primary()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in self._trainable_parameters(self.replicas[0][1])],
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
            int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
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
            print(f'[train-fast] restoring {self.world_size} trainer replicas for the update', flush=True)
            self.restore_after_generation()

        epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
        if epochs != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")
        epsilon, epsilon_low, epsilon_high, kl_coef = (_clipped_policy_options(cfg))

        cpu_examples = self._cpu_examples(examples)
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in cpu_examples)
        print(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
              "vLLM supplied old="
              f"{supplied_old}/{len(cpu_examples)}, reference="
              f"off; "
              "HF fallback runs only "
              "for missing values", flush=True)

        pending = sorted(
            cpu_examples,
            key=lambda example: int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1]))

        queue_lock = threading.Lock()
        # A token budget is the real memory controller. This loose count guard
        # only prevents pathological thousands-of-tiny-sequences batches.
        max_examples_per_batch = min(64, max(1, len(cpu_examples)))

        def take_batch(token_budget):
            with queue_lock:
                if not pending:
                    return []
                batch = [pending.pop()]
                maximum = int(batch[0]["prompt_ids"].shape[1] + batch[0]["response_ids"].shape[1])
                while pending and len(batch) < max_examples_per_batch:
                    candidate = pending[-1]
                    candidate_length = int(candidate["prompt_ids"].shape[1] + candidate["response_ids"].shape[1])
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
                budgets[replica_index] = max(1, int(budgets[replica_index] or initial_budget))

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
            parameters = [parameter for _, parameter in self._trainable_parameters(model)]
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
                    external_bytes = max(0, int(total_bytes) - int(free_bytes) - base_reserved)
                    torch.cuda.reset_peak_memory_stats()
                    local_batch = self._move_shard(cpu_batch, logical_id)

                    def attempt(active_batches):
                        attempt_values = {key: [] for key in metric_keys}
                        attempt_ratio_values = []
                        attempt_prefix_ratio_values = []
                        attempt_entropy_examples = 0

                        with _policy_dropout_disabled(model):
                            for batch in active_batches:
                                missing_old = (_initialize_behavior_logprobs(batch))
                                for fallback_batch in ([example] for example in missing_old):
                                    old_logprobs = (
                                        compute_batched_token_logprobs(
                                            model, fallback_batch,
                                            with_grad=False,
                                            chunk=cfg.logprob_chunk,
                                            pad_token_id=(tokenizer.pad_token_id)))
                                    if any(not torch.isfinite(value).all() for value in old_logprobs):
                                        raise FloatingPointError("nonfinite old-policy logprobs")
                                    for example, old_lp in zip(fallback_batch, old_logprobs):
                                        example["old_policy_logprobs"] = (old_lp.detach())


                                result = compute_batched_token_logprobs(
                                    model, batch, with_grad=True,
                                    chunk=cfg.logprob_chunk,
                                    pad_token_id=tokenizer.pad_token_id)
                                current_logprobs = result

                                weighted_losses = []
                                for example, current_lp in zip(batch, current_logprobs):
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
                                        raise FloatingPointError("nonfinite SPO-RS loss")
                                    weighted_losses.append(weight * loss)
                                    attempt_values["loss"].append(weight * loss.detach())
                                    for key in (
                                            "policy_loss", "kl_estimate",
                                            "entropy_estimate", "ratio"):
                                        attempt_values[key].append(weight * metrics[key])
                                    attempt_values["clipped_fraction"].append(weight * metrics["clipped"])
                                    attempt_ratio_values.append(metrics["ratio"])
                                    attempt_prefix_ratio_values.append(metrics["prefix_ratio_max"])
                                if weighted_losses:
                                    sum(weighted_losses[1:], weighted_losses[0]).backward()

                        zero = torch.zeros((), dtype=torch.float64, device=torch.device(f"cuda:{logical_id}"))
                        reduced = [
                            (torch.stack(attempt_values[key]).sum() if attempt_values[key] else zero)
                            for key in metric_keys
                        ]
                        reduced.extend((
                            (torch.stack(attempt_ratio_values).max() if attempt_ratio_values else zero),
                            (torch.stack(attempt_prefix_ratio_values).max() if attempt_prefix_ratio_values else zero),
                        ))
                        packed = torch.stack([
                            value.to(dtype=torch.float64)
                            for value in reduced
                        ]).detach().cpu().tolist()
                        attempt_totals = dict(zip(metric_keys, packed[:len(metric_keys)]))
                        attempt_max_ratio = packed[len(metric_keys)]
                        attempt_max_prefix_ratio = packed[len(metric_keys) + 1]
                        return (
                            attempt_totals, attempt_max_ratio,
                            attempt_max_prefix_ratio,
                            attempt_entropy_examples,
                        )

                    result, effective_batches, quarantined = (
                        _run_oom_resilient_backward(model, [local_batch], attempt, device_label=f"cuda:{logical_id}"))

                    peak_allocated = int(torch.cuda.max_memory_allocated())
                    peak_reserved = int(torch.cuda.max_memory_reserved())
                    used_at_peak = min(int(total_bytes), external_bytes + peak_reserved)
                    peak_memory_fraction = max(peak_memory_fraction, used_at_peak / max(1, int(total_bytes)))

                    if result is not None:
                        with torch.no_grad():
                            for parameter_index, parameter in enumerate(parameters):
                                gradient = parameter.grad
                                if gradient is None:
                                    continue
                                accumulator = gradient_accumulators[parameter_index]
                                if accumulator is None:
                                    gradient_accumulators[parameter_index] = gradient.detach()
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
                    trained_examples += sum(len(batch) for batch in effective_batches)

                    successful_padded_tokens = max(
                        (self._padded_token_count(batch) for batch in effective_batches),
                        default=0)
                    longest_successful = max(
                        (int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
                         for batch in effective_batches
                         for example in batch),
                        default=1)
                    backed_off = bool(len(effective_batches) > 1 or quarantined)
                    if not effective_batches:
                        budget = max(1, budget // 2)
                    elif backed_off:
                        budget = max(longest_successful, min(budget, successful_padded_tokens))
                    else:
                        active_bytes = max(1, peak_allocated - base_allocated)
                        target_process_bytes = max(base_allocated + 1, int(0.80 * int(total_bytes)) - external_bytes)
                        available_for_batch = max(1, target_process_bytes - base_allocated)
                        desired_budget = int(requested_padded_tokens * available_for_batch / active_bytes * 0.96)
                        lower = max(longest_successful, int(budget * 0.70))
                        upper = max(lower, int(budget * 1.35))
                        desired_budget = min(upper, max(lower, desired_budget))
                        budget = max(longest_successful, int(round(0.5 * budget + 0.5 * desired_budget)))
                    budgets[replica_index] = budget

                del local_batch, effective_batches, quarantined

            with torch.cuda.device(logical_id), torch.no_grad():
                for parameter, accumulator in zip(parameters, gradient_accumulators):
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
        totals = {key: sum(result["totals"][key] for result in results) for key in metric_keys}
        max_ratio = max(result["max_ratio"] for result in results)
        max_prefix_ratio = max(result["max_prefix_ratio"] for result in results)
        entropy_examples = sum(result["entropy_examples"] for result in results)
        quarantined_examples = sum(result["quarantined_examples"] for result in results)
        grad_norm = self._clip_step_and_sync(cfg)
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "entropy_examples": entropy_examples,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        trained_per_gpu = [result["trained_examples"] for result in results]
        batches_per_gpu = [result["backward_batches"] for result in results]
        peak_percentages = [round(100.0 * result["peak_memory_fraction"], 1) for result in results]
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
            "fast_peak_memory_fraction": [result["peak_memory_fraction"] for result in results],
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_spo_rs(self, examples, cfg, step_idx):
        if bool(getattr(cfg, "fast", False)):
            return self._train_spo_rs_fast(examples, cfg, step_idx)

        import torch

        started = time.time()
        if self._offloaded:
            print(f'[train-parallel] restoring {self.world_size} trainer replicas for the update', flush=True)
            self.restore_after_generation()
        epochs = int(getattr(cfg, "spo_rs_update_epochs", SPO_RS_UPDATE_EPOCHS_DEFAULT))
        epsilon, epsilon_low, epsilon_high, kl_coef = (_clipped_policy_options(cfg))
        local_shards = self._prepare_shards(examples)
        update_batches = [_training_microbatches(shard, cfg) for shard in local_shards]
        largest_batch = max((len(batch) for batches in update_batches for batch in batches), default=0)
        print(f"[train-parallel] LoRA microbatches/GPU="
              f"{[len(batches) for batches in update_batches]}; "
              f"configured max={int(cfg.train_examples_per_microbatch)}, "
              f"effective max={largest_batch}, "
              f"padded-token cap={int(cfg.max_seq_length)}", flush=True)
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in examples)
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
                missing_old = (_initialize_behavior_logprobs(shard))
                # vLLM normally fills these caches. Singleton fallback keeps a
                # rare incomplete payload from destabilizing all eight replicas.
                for batch in ([example] for example in missing_old):
                    old_logprobs = compute_batched_token_logprobs(
                        model, batch, with_grad=False,
                        chunk=cfg.logprob_chunk,
                        pad_token_id=tokenizer.pad_token_id)
                    if any(not torch.isfinite(value).all() for value in old_logprobs):
                        raise FloatingPointError("nonfinite old-policy logprobs")
                    for example, old_lp in zip(batch, old_logprobs):
                        example["old_policy_logprobs"] = old_lp.detach()
                torch.cuda.synchronize(logical_id)

        self._run_replicas(cache, list(enumerate(local_shards)))
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
                            for example, current_lp in zip(batch, current_logprobs):
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
                                    raise FloatingPointError("nonfinite SPO-RS loss")
                                weighted_losses.append(weight * loss)
                                totals["loss"] += (weight * float(loss.detach().item()))
                                for key in ("policy_loss", "kl_estimate",
                                            "entropy_estimate", "ratio"):
                                    totals[key] += weight * metrics[key]
                                totals["clipped_fraction"] += (weight * float(metrics["clipped"]))
                                max_ratio = max(max_ratio, metrics["ratio"])
                                max_prefix_ratio = max(max_prefix_ratio, metrics["prefix_ratio_max"])
                            if weighted_losses:
                                sum(weighted_losses[1:], weighted_losses[0]).backward()
                    torch.cuda.synchronize(logical_id)
                    return (totals, max_ratio, max_prefix_ratio,
                            entropy_examples)

                with torch.cuda.device(logical_id):
                    result, _effective, quarantined = (
                        _run_oom_resilient_backward(model, batches, attempt, device_label=f"cuda:{logical_id}"))
                if result is None:
                    result = (
                        {key: 0.0 for key in (
                            "loss", "policy_loss", "kl_estimate",
                            "entropy_estimate", "ratio",
                            "clipped_fraction")},
                        0.0, 0.0, 0,
                    )
                return (*result, len(quarantined))

            results = self._run_replicas(accumulate, list(enumerate(update_batches)))
            totals = {key: sum(result[0][key] for result in results) for key in results[0][0]}
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
            "oom_quarantined_examples": sum(item["oom_quarantined_examples"] for item in history),
        }

    def train_policy(self, examples, cfg, step_idx):
        import torch

        started = time.time()
        if self._offloaded:
            print(f'[train-parallel] restoring {self.world_size} trainer replicas for the update', flush=True)
            self.restore_after_generation()
        local_shards = self._prepare_shards(examples)
        n_examples = len(examples)
        microbatches = [_training_microbatches(shard, cfg) for shard in local_shards]
        largest_batch = max((len(batch) for batches in microbatches for batch in batches), default=0)
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
                    base_logprobs = [example.get("reference_logprobs") for example in batch]
                    supplied_reference = all(
                        _valid_example_token_logprobs(example, "reference_logprobs")
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
                    for example, current_lp, base_lp in zip(batch, current_logprobs, base_logprobs):
                        base_lp = base_lp.to(current_lp.device)
                        advantage = example["advantage"]
                        logp_difference = (current_lp - base_lp).detach()
                        average_difference = logp_difference.mean()
                        kl_advantage = cfg.kl_penalty_coef * (average_difference - (current_lp - base_lp))
                        effective_advantage = advantage + kl_advantage
                        behavior_lp = example.get("behavior_logprobs")
                        has_behavior = _valid_example_token_logprobs(example, "behavior_logprobs")
                        if sequence_policy:
                            loss, policy_metrics = (
                                _a3b_sequence_clipped_standard_loss(
                                    cfg, current_lp,
                                    behavior_lp if has_behavior else None,
                                    base_lp, advantage))
                            importance_ratio = policy_metrics["ratio"]
                        elif has_behavior:
                            importance_ratio = (_detached_behavior_importance_ratio(cfg, current_lp, behavior_lp))
                            loss = -(importance_ratio * effective_advantage.detach() * current_lp).mean()
                        else:
                            importance_ratio = 1.0
                            loss = -(effective_advantage.detach() * current_lp).mean()
                        if has_behavior:
                            ratio_sum += float(importance_ratio.mean().item())
                            ratio_max = max(ratio_max, float(importance_ratio.max().item()))
                            ratio_count += 1
                        batch_losses.append(loss / n_examples)
                        total_loss += float(loss.detach().item())
                        total_logp_delta += float(logp_difference.mean().item())
                    if batch_losses:
                        sum(batch_losses[1:], batch_losses[0]).backward()
                torch.cuda.synchronize(logical_id)
                return (total_loss, total_logp_delta, ratio_sum, ratio_max,
                        ratio_count, kl_error)

            with torch.cuda.device(logical_id):
                result, _effective, quarantined = (
                    _run_oom_resilient_backward(model, batches, attempt, device_label=f"cuda:{logical_id}"))
            if result is None:
                result = (0.0, 0.0, 0.0, 0.0, 0, None)
            return (*result, len(quarantined))

        results = self._run_replicas(accumulate, list(enumerate(microbatches)))
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
            ratio_message = (f'  IS ratio mean={ratio_sum / ratio_count:.9f} max={ratio_max:.3f}')
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
        from runtime.host_memory import reclaim_unused_host_memory
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
        reclaim_unused_host_memory(torch, cuda_devices=tuple(range(self.world_size)), force=True)
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
        from training.fast_distributed import (
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
        self._process_label = ("train-fast" if bool(cfg.fast) else "train-process")
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
        self._memory_fraction = float(getattr(cfg, "training_memory_fraction", 0.80))
        self._memory_percentage = round(100.0 * self._memory_fraction, 1)
        self._token_budgets = [max(1, int(cfg.max_seq_length)) for _ in range(self._world_size)]
        if self._world_size != len(self._physical_ids):
            raise ValueError("num_training_gpus does not match training_gpu_ids")
        if self._world_size < 2:
            raise ValueError("process-distributed training requires two GPUs")

        from runtime.host_memory import (
            GIB, TokenWeightedActivationOffloadGate,
            automatic_activation_offload_budget_bytes,
        )
        self._activation_offload_budget_bytes = (automatic_activation_offload_budget_bytes())
        self._activation_offload_semaphore = (
            TokenWeightedActivationOffloadGate(
                self._context,
                self._activation_offload_budget_bytes,
                max(1, int(cfg.max_seq_length)),
            ))
        self._activation_offload_slots = (self._activation_offload_semaphore.full_context_concurrency)
        self._activation_offload_budget_gib = (self._activation_offload_budget_bytes / GIB)

        # Eight independent Python processes must not each create a full-sized
        # host thread pool. GPU 0 was loaded before this class is constructed;
        # the allocator limit still constrains every subsequent allocation.
        torch.set_num_threads(max(1, int(os.cpu_count() or self._world_size) // self._world_size))
        torch.cuda.set_device(0)
        set_total_memory_ceiling(0, self._memory_fraction)
        validate_model_device(self.model, 0)
        expected_signature = trainable_parameter_signature(self.model)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        sync_dir = Path(exp_dir).resolve() / ".fast_trainer"
        sync_dir.mkdir(parents=True, exist_ok=True)
        rendezvous_path = sync_dir / (f"nccl-{os.getpid()}-{uuid.uuid4().hex}")
        self._init_method = rendezvous_path.as_uri()
        cfg_dict = dict(vars(cfg))
        cfg_dict["_terminal_log_path"] = str(Path(exp_dir).resolve() / "temirnal.log")
        cfg_dict["_setting_log_path"] = str(Path(exp_dir).resolve() / "setting.log")
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
                    raise RuntimeError(f'fast trainer rank {rank} has a different trainable parameter layout')
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
        from runtime.host_memory import (GIB, estimated_module_bytes, estimated_optimizer_bytes)

        model_bytes = estimated_module_bytes(self.model)
        optimizer_bytes = estimated_optimizer_bytes(self.optimizer)
        # All worker models have the same storage layout as rank zero.  The
        # optimizer exists only on rank zero; workers retain gradients/model
        # state but do not construct duplicate Adam state.
        return int(1.05 * (model_bytes * self._world_size + optimizer_bytes) + self._world_size * GIB)

    @property
    def host_memory_admission(self):
        return self._activation_offload_semaphore.host_admission

    def reclaim_host_memory(self):
        """Synchronize every trainer process and free unused host caches."""
        import torch
        from runtime.host_memory import reclaim_unused_host_memory

        if not self._workers_idle:
            raise RuntimeError("cannot reclaim trainer host caches while an update is active")
        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "purge_host_memory"})
        with torch.cuda.device(0):
            torch.cuda.synchronize(0)
        local_cleanup = reclaim_unused_host_memory(torch, cuda_devices=(0,), force=True)
        child_messages = self._collect_event("host_memory_purged", range(1, self._world_size))
        return [local_cleanup] + [child_messages[rank]["cleanup"] for rank in range(1, self._world_size)]

    def _dead_workers(self, pending):
        return [rank for rank in pending if rank in self._processes and not self._processes[rank].is_alive()]

    def _collect_event(self, event, ranks):
        pending = set(int(rank) for rank in ranks)
        results = {}
        while pending:
            try:
                message = self._result_queue.get(timeout=5.0)
            except self._queue_module.Empty:
                dead = self._dead_workers(pending)
                if dead:
                    codes = {rank: self._processes[rank].exitcode for rank in dead}
                    raise RuntimeError(f'fast trainer worker(s) exited while waiting for {event}: {codes}')
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
                raise RuntimeError(f'fast trainer rank {rank} failed:\n{traceback_text}')
            if message_event != event or rank not in pending:
                raise RuntimeError(f'unexpected fast trainer message while waiting for {event}: {message}')
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

    def _queue_examples(self, examples, *, seal=True, announce=True, weighted_offload=False):
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
                    int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1]) >= overflow_floor)
        # Packing response branches behind one shared prompt is exact for
        # ordinary transformer attention because the registered branch mask
        # isolates every response.  It is not valid for GPT-OSS (learned
        # attention sinks and alternating sliding attention), nor for
        # Qwen3.8's recurrent Gated-DeltaNet layers, whose state cannot be
        # reset by an attention mask.  Keep those profiles as independent
        # batch rows throughout scheduling and scoring.
        shared_prefix_packing = str(getattr(self.cfg, "coder_model_profile", "")) not in {
                "gpt-oss-120b", "qwen3.8-27b",
            }
        for example in copied:
            example["_shared_prefix_packing_allowed"] = shared_prefix_packing

        if (bool(getattr(self.cfg, "strategies", False)) and shared_prefix_packing):
            from training.fast_distributed import SHARED_PREFIX_WORK_KEY

            grouped = {}
            for example in copied:
                prompt_job_id = example.get("prompt_job_id")
                if prompt_job_id is None:
                    raise RuntimeError("strategy training example has no prompt_job_id")
                grouped.setdefault(int(prompt_job_id), []).append(example)

            def group_cost(group):
                prompt = int(group[0]["prompt_ids"].shape[1])
                responses = [int(example["response_ids"].shape[1]) for example in group]
                # The prompt is evaluated once. Each isolated response branch
                # attends to that prompt and its own causal prefix.
                return prompt * prompt + sum(response * (prompt + response) for response in responses)

            groups = []
            for prompt_job_id, group in grouped.items():
                prompt = group[0]["prompt_ids"]
                if any(
                        example["prompt_ids"].shape != prompt.shape
                        or not torch.equal(example["prompt_ids"], prompt)
                        for example in group[1:]):
                    raise RuntimeError(f'one strategy prompt_job_id contains different prompts: {prompt_job_id}')
                # Longest branches first makes any budget-driven split stable
                # and avoids leaving one pathological response until last.
                group.sort(key=lambda example: int(example["response_ids"].shape[1]), reverse=True)
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
                    work = int(example["prompt_ids"].shape[1]) + int(example["response_ids"].shape[1])
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
                ordered = [example for wave in waves for example in wave["items"]]
                offload_waves = [len(wave["items"]) for wave in waves]
            for example in ordered:
                self._work_queue.put(example)
            if announce:
                if offload_waves is None:
                    print(f"[{self._process_label}] queued {len(ordered)} "
                          "examples longest-first; each free GPU claims the "
                          "next adaptive batch", flush=True)
                else:
                    preview = ",".join(str(value) for value in offload_waves[:8])
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
        from training.fast_distributed import (broadcast_trainable_parameters,
                                      local_spo_rs_update,
                                      reduce_trainable_gradients)

        if self._offloaded:
            print(f'[train-fast] restoring {self._world_size} process trainers for the update', flush=True)
            self.restore_after_generation()
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("--fast requires one clipped-policy update epoch")

        started = time.time()
        queued_examples = self._queue_examples(examples)
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued_examples)
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

        defer_exact_oom = (str(getattr(cfg, "advantage_mode", "")).lower() == "spo-rs")
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
            activation_offload_semaphore=(self._activation_offload_semaphore),
            defer_exact_oom=defer_exact_oom)
        child_messages = self._collect_event("computed", range(1, self._world_size))
        rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
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
            deferred_examples = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            if deferred_examples:
                gradient_path, marker, deferred_keys, _normal_stats = (
                    _save_exact_spo_rs_overflow(
                        self._exp_dir, step_idx, self.model, rank_stats,
                        normal_train_seconds=time.time() - started))
                longest = max(
                    (int(length) for stats in rank_stats for length in stats.get("deferred_token_lengths", ())),
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
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
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
            from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, queued_examples, cfg, step_idx,
                    _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        metric_keys = (
            "loss", "policy_loss", "kl_estimate", "entropy_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {key: sum(stats["totals"][key] for stats in rank_stats) for key in metric_keys}
        max_ratio = max(stats["max_ratio"] for stats in rank_stats)
        max_prefix_ratio = max(stats["max_prefix_ratio"] for stats in rank_stats)
        entropy_examples = sum(stats["entropy_examples"] for stats in rank_stats)
        quarantined_examples = sum(stats["quarantined_examples"] for stats in rank_stats)
        trained_per_gpu = [stats["trained_examples"] for stats in rank_stats]
        batches_per_gpu = [stats["backward_batches"] for stats in rank_stats]
        peak_fractions = [stats["peak_memory_fraction"] for stats in rank_stats]
        allocator_fractions = [stats["allocator_memory_fraction"] for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
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

    def recover_spo_rs_overflow(self, examples, cfg, step_idx, overflow_state):
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
        from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
        grouped_recovery = bool(
            overflow_state is not None
            and overflow_state.get("remaining_keys")
            and not overflow_state.get("replica_checkpoint_groups_attempted", False)
            and any(hasattr(module, "_ttt_group_checkpoint_min_tokens") for module in self.model.modules()))
        from training.context_parallel import availability as context_parallel_availability
        context_supported, _ = context_parallel_availability(self.model, cfg, self._world_size)
        if supports_warm_overflow(cfg) and (context_supported or not grouped_recovery):
            return complete_warm_overflow(self, examples, cfg, step_idx, overflow_state)

        import torch
        from training.fast_distributed import (
            broadcast_trainable_parameters, local_spo_rs_update,
            reduce_trainable_gradients,
        )

        if str(getattr(cfg, "advantage_mode", "")).lower() != "spo-rs":
            raise ValueError("exact overflow recovery is SPO-RS-only")
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("exact overflow recovery requires one update epoch")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for exact overflow recovery")
        if self._offloaded:
            print(
                f"[{self._process_label}] restoring {self._world_size} "
                "process trainers for exact overflow recovery",
                flush=True,
            )
            self.restore_after_generation()

        _validate_precomputed_exact_gradients(self.model, overflow_state, step_idx)
        requested_keys = {tuple(int(value) for value in example["_overlap_key"]) for example in examples}
        remaining_keys = {tuple(int(value) for value in key) for key in overflow_state.get("remaining_keys", ())}
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
        queued_examples = self._queue_examples(examples, weighted_offload=not grouped_recovery)
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
                activation_offload_semaphore=(self._activation_offload_semaphore),
                defer_exact_oom=True,
            )
            child_messages = self._collect_event("computed", range(1, self._world_size))
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
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
            deferred_count = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            quarantined_count = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
            trained_count = sum(int(stats["trained_examples"]) for stats in rank_stats)
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
                "totals": {key: sum(float(stats["totals"][key]) for stats in rank_stats) for key in metric_keys},
                "max_ratio": max(float(stats["max_ratio"]) for stats in rank_stats),
                "max_prefix_ratio": max(float(stats["max_prefix_ratio"]) for stats in rank_stats),
                "entropy_examples": sum(int(stats["entropy_examples"]) for stats in rank_stats),
                "backward_batches": sum(int(stats["backward_batches"]) for stats in rank_stats),
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
                    marker = _read_exact_sharded_recovery(self._exp_dir) or {}
                    _write_exact_sharded_recovery(
                        self._exp_dir, step_idx,
                        marker.get("reason", "bounded replicated overflow recovery exhausted"),
                        replica_offload_exhausted=True,
                        remaining_deferred_keys=[list(key) for key in sorted(deferred_keys)],
                    )
                    overflow_error = ExactLongContextTrainingOOM(
                        f"step {step_idx} retained "
                        f"{len(successful_keys)} additional exact gradient(s) "
                        f"but {len(deferred_keys)} trajectory/trajectories "
                        "still exceed bounded replica activation offload")
                else:
                    _apply_precomputed_exact_gradients(self.model, overflow_state, step_idx)
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters() if parameter.requires_grad],
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
                    return complete_warm_overflow(self, examples, cfg, step_idx, overflow_state)
                raise overflow_error
        except BaseException:
            if not self._workers_idle:
                self._abort_workers()
            raise

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        normal_stats = overflow_state.get("normal_stats", {})
        overflow_stats = overflow_state.get("overflow_stats", {})
        normal_totals = normal_stats.get("totals", {})
        overflow_totals = overflow_stats.get("totals", {})
        totals = {
            key: (float(normal_totals.get(key, 0.0)) + float(overflow_totals.get(key, 0.0)))
            for key in metric_keys
        }
        max_ratio = max(float(normal_stats.get("max_ratio", 0.0)), float(overflow_stats.get("max_ratio", 0.0)))
        max_prefix_ratio = max(
            float(normal_stats.get("max_prefix_ratio", 0.0)),
            float(overflow_stats.get("max_prefix_ratio", 0.0)),
        )
        entropy_examples = (
            int(normal_stats.get("entropy_examples", 0))
            + int(overflow_stats.get("entropy_examples", 0))
        )
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
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
            "exact_overflow_reused_examples": int(normal_stats.get("trained_examples", 0))
                + len(overflow_state.get("completed_keys", ()))
                - len(successful_keys),
            "exact_overflow_replica_examples": len(successful_keys),
            "exact_overflow_sharded_examples": 0,
        }

    def begin_entropic_overlap(self, cfg, step_idx, normalization_examples):
        """Start a streaming entropic backward before CPU evaluation ends."""
        from concurrent.futures import ThreadPoolExecutor
        from training.fast_distributed import local_policy_update

        if str(getattr(cfg, "advantage_mode", "")).lower() != "entropic":
            raise ValueError("streaming policy overlap is entropic-only")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for an entropic overlap update")
        if self._entropic_overlap is not None:
            raise RuntimeError("an entropic overlap update is already active")
        normalization_examples = int(normalization_examples)
        if normalization_examples < 1:
            raise ValueError("entropic overlap needs a positive normalization count")
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

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entropic-rank0-overlap")
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
            activation_offload_semaphore=(self._activation_offload_semaphore))
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
        queued = self._queue_examples(examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued)
        state["supplied_reference"] += sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued)
        state["scheduled_keys"].update(example["_overlap_key"] for example in queued)
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

    def finish_entropic_overlap(self, cfg, step_idx, expected_examples, *, apply_update=True):
        """Finish, globally normalize, and optionally apply streamed gradients."""
        import torch
        from training.fast_distributed import (broadcast_trainable_parameters, reduce_trainable_gradients)

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
            child_messages = self._collect_event("computed", range(1, self._world_size))
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
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
                reduce_trainable_gradients(self.model, destination_rank=0)
                try:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                        max_norm=cfg.grad_clip, error_if_nonfinite=True)
                    self.optimizer.step()
                    grad_norm = float(grad_norm_tensor.item())
                except BaseException as error:
                    update_error = error
                finally:
                    self.model.zero_grad(set_to_none=True)
                    broadcast_trainable_parameters(self.model, source_rank=0)
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
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        total_loss = sum(float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        kl_errors = [error for stats in rank_stats for error in stats["kl_errors"]]
        if kl_errors:
            print(f"[warn] disable_adapter failed ({kl_errors[0]}); "
                  "training without KL penalty on affected examples",
                  flush=True)
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        print(f"[{self._process_label}] trained examples/GPU="
              f"{trained_per_gpu}; backward batches/GPU={batches_per_gpu}; "
              f"final padded-token budgets/GPU={self._token_budgets}; "
              f"peak memory/GPU={peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        completed_at = state["computed_at"] or time.time()
        elapsed = max(0.0, completed_at - state["started_at"])
        ratio_message = ""
        if ratio_count:
            ratio_message = (f'  IS ratio mean={ratio_sum / ratio_count:.9f} max={ratio_max:.3f}')
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
        from training.fast_distributed import local_spo_rs_update

        if str(getattr(cfg, "advantage_mode", "")).lower() != "spo-rs":
            raise ValueError("streaming clipped overlap is SPO-RS-only")
        if int(getattr(cfg, "spo_rs_update_epochs", 1)) != 1:
            raise ValueError("streaming SPO-RS requires one update epoch")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for an SPO-RS overlap update")
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

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spo-rs-rank0-overlap")
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
            activation_offload_semaphore=(self._activation_offload_semaphore),
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
            raise RuntimeError("SPO-RS overlap cannot queue an unresolved advantage")
        queued = self._queue_examples(examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued)
        state["scheduled_keys"].update(example["_overlap_key"] for example in queued)
        state["overflow_examples"].update((tuple(example["_overlap_key"]), example) for example in queued)
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

    def finish_spo_rs_overlap(self, cfg, step_idx, expected_examples, *, apply_update=True):
        """Reduce and apply the exact streamed SPO-RS gradients once."""
        import torch
        from training.fast_distributed import (broadcast_trainable_parameters, reduce_trainable_gradients)

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
            child_messages = self._collect_event("computed", range(1, self._world_size))
            # Rank 0 can finish before a longer worker. Record completion only
            # after every GPU has drained the shared queue.
            state["computed_at"] = time.time()
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
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

            deferred_examples = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            for command_queue in self._command_queues.values():
                command_queue.put({
                    "kind": "finish_update",
                    "apply_update": bool(apply_update),
                })

            grad_norm = None
            update_error = None
            overflow_error = None
            if apply_update:
                reduce_trainable_gradients(self.model, destination_rank=0)
                try:
                    if deferred_examples:
                        gradient_path, marker, deferred_keys, _normal_stats = (
                            _save_exact_spo_rs_overflow(
                                self._exp_dir, step_idx, self.model,
                                rank_stats,
                                normal_train_seconds=max(0.0, state["computed_at"] - state["started_at"])))
                        longest = max(
                            (int(length) for stats in rank_stats for length in stats.get("deferred_token_lengths", ())),
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
                            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                            max_norm=cfg.grad_clip,
                            error_if_nonfinite=True)
                        self.optimizer.step()
                        grad_norm = float(grad_norm_tensor.item())
                except BaseException as error:
                    update_error = error
                finally:
                    self.model.zero_grad(set_to_none=True)
                    broadcast_trainable_parameters(self.model, source_rank=0)
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
            from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, list(state["overflow_examples"].values()), cfg,
                    step_idx, _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        metric_keys = (
            "loss", "policy_loss", "kl_estimate", "entropy_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {key: sum(float(stats["totals"][key]) for stats in rank_stats) for key in metric_keys}
        max_ratio = max(float(stats["max_ratio"]) for stats in rank_stats)
        max_prefix_ratio = max(float(stats["max_prefix_ratio"]) for stats in rank_stats)
        entropy_examples = sum(int(stats["entropy_examples"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
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
        from training.fast_distributed import (broadcast_trainable_parameters,
                                      local_policy_update,
                                      reduce_trainable_gradients)

        process_label = self._process_label
        adaptive_batches = bool(cfg.fast)
        if self._offloaded:
            print(f'[{process_label}] restoring {self._world_size} process trainers for the update', flush=True)
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
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued_examples)
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
            activation_offload_semaphore=(self._activation_offload_semaphore))
        child_messages = self._collect_event("computed", range(1, self._world_size))
        rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
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
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
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
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        total_loss = sum(float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        kl_errors = [error for stats in rank_stats for error in stats["kl_errors"]]
        if kl_errors:
            print(f"[warn] disable_adapter failed ({kl_errors[0]}); "
                  "training without KL penalty on affected examples",
                  flush=True)
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        print(f"[{process_label}] trained examples/GPU={trained_per_gpu}; "
              f"backward batches/GPU={batches_per_gpu}; final padded-token "
              f"budgets/GPU={self._token_budgets}; peak memory/GPU="
              f"{peak_percentages}%; allocator caps/GPU="
              f"{allocator_percentages}%", flush=True)
        elapsed = time.time() - started
        ratio_message = ""
        if ratio_count:
            ratio_message = (f'  IS ratio mean={ratio_sum / ratio_count:.9f} max={ratio_max:.3f}')
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
        ranks = {rank for rank in self._offloaded_ranks if self._physical_ids[rank] in physical_gpu_ids}
        if not ranks:
            return
        if not self._dist_initialized:
            raise RuntimeError('process-distributed trainer is unavailable after a worker failure')
        import gc
        import torch
        from training.fast_distributed import (set_total_memory_ceiling, validate_model_device)
        from runtime.host_memory import reclaim_unused_host_memory

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
                reclaim_unused_host_memory(torch, cuda_devices=(0,), force=True)
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
                state["executor"].shutdown(wait=True, cancel_futures=False)
                self._entropic_overlap = None

        if self._spo_rs_overlap is not None:
            state = self._spo_rs_overlap
            self.seal_spo_rs_overlap()
            try:
                state["future"].result()
            except Exception:
                pass
            finally:
                state["executor"].shutdown(wait=True, cancel_futures=False)
                self._spo_rs_overlap = None

        if (not self._workers_idle or not self._dist_initialized or self._dead_workers(range(1, self._world_size))):
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
