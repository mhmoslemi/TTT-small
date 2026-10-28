"""GPU-resident, nested checkpoints for long Qwen text training.

Layer checkpoints alone retain one full hidden state per layer. Grouping
those checkpoints retains only group boundaries between forward and backward,
then reconstructs the layer boundaries of one group at a time. Parameters,
attention, recurrent state, tokens, and the loss are unchanged.
"""

import importlib
import math

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_group_size(layer_count, hidden_bytes, budget_bytes):
    """Smallest recomputation group fitting the boundary budget, or its minimum."""
    candidates = range(1, max(1, math.ceil(math.sqrt(layer_count))) + 1)
    for size in candidates:
        boundaries = layer_count if size == 1 else math.ceil(layer_count / size) + size
        if boundaries * hidden_bytes <= budget_bytes:
            return size
    return min(candidates, key=lambda size: math.ceil(layer_count / size) + size)


def checkpoint_layer_groups(layers, hidden, layer_kwargs, group_size):
    """Nest group checkpoints around the existing per-layer checkpoints."""
    for start in range(0, len(layers), group_size):
        # Capture each group by value: backward runs after this loop finishes.
        group = tuple(zip(layers[start:start + group_size],
                          layer_kwargs[start:start + group_size]))

        def run(value, group=group):
            for layer, kwargs in group:
                value = layer(value, **kwargs)
            return value

        hidden = (checkpoint(run, hidden, use_reentrant=False)
                  if group_size > 1 else run(hidden))
    return hidden


def grouped_qwen_hidden(decoder, input_ids, attention_mask, *, position_ids=None,
                        group_size=None, force=False):
    """Exact no-cache text forward, or None when the normal forward is suitable.

    The mask/rotary/layer implementations come from the installed Transformers
    version. This is restricted to the training scorer's text-only interface;
    generation and multimodal forwards keep their original implementation.
    Qwen3_5TextModel contract:
    https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """
    minimum = getattr(decoder, "_ttt_group_checkpoint_min_tokens", None)
    if ((not force and (minimum is None or input_ids.numel() < minimum))
            or not torch.is_grad_enabled()
            or not decoder.training or not decoder.gradient_checkpointing):
        return None
    # Accelerate's layer-sharded fallback owns placement/routing hooks. Keep
    # its original forward; this optimization is for independent GPU replicas.
    if len({parameter.device for parameter in decoder.parameters()}) != 1:
        return None

    layers = tuple(decoder.layers[:decoder.config.num_hidden_layers])
    if not all(getattr(layer, "gradient_checkpointing", False) for layer in layers):
        raise RuntimeError("group checkpoints require active per-layer checkpointing")
    hidden = decoder.embed_tokens(input_ids)
    if hidden.device.type == "cuda":
        device = hidden.device
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        available = free + torch.cuda.memory_reserved(device) - allocated
        get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        if callable(get_fraction):
            available = min(available, int(total * get_fraction(device)) - allocated)
        # Leave half the remaining card budget for attention/GDN backward,
        # chunked MLPs, output-head projection, and allocator workspace.
        budget = max(1, int(available * 0.5))
    else:
        # CPU permits small exactness tests without requiring a GPU runtime.
        budget = 0
    if group_size is None:
        group_size = checkpoint_group_size(
            len(layers), hidden.numel() * hidden.element_size(), budget)
    if group_size == 1:
        return None

    implementation = importlib.import_module(type(decoder).__module__)
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    text_positions = (positions.view(1, -1).expand(hidden.shape[0], -1)
                      if position_ids is None else position_ids)
    if text_positions.shape != input_ids.shape:
        raise ValueError("checkpointed global positions must match the local input")
    rotary_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
    mask_kwargs = dict(config=decoder.config, inputs_embeds=hidden,
                       attention_mask=attention_mask, past_key_values=None,
                       position_ids=text_positions)
    causal_mask = implementation.create_causal_mask(**mask_kwargs)
    old_mask_helper = getattr(decoder, "_update_linear_attn_mask", None)
    linear_mask = (old_mask_helper(attention_mask, None)
                   if callable(old_mask_helper)
                   else implementation.create_recurrent_attention_mask(**mask_kwargs))
    rotary = decoder.rotary_emb(hidden, rotary_positions)
    layer_kwargs = [dict(
        position_embeddings=rotary,
        attention_mask=(linear_mask if kind == "linear_attention" else causal_mask),
        position_ids=text_positions, past_key_values=None, use_cache=False,
    ) for kind in decoder.config.layer_types[:len(layers)]]
    if len(layer_kwargs) != len(layers):
        raise RuntimeError("Qwen decoder layer types do not match its layers")

    if getattr(decoder, "_ttt_reported_checkpoint_group", None) != group_size:
        print(f"[train-memory] {hidden.device}: GPU checkpoint groups of {group_size} "
              f"layers ({math.ceil(len(layers) / group_size)} saved group boundaries "
              f"instead of {len(layers)} layer boundaries); full trajectory",
              flush=True)
        decoder._ttt_reported_checkpoint_group = group_size
    hidden = checkpoint_layer_groups(layers, hidden, layer_kwargs, group_size)
    return decoder.norm(hidden)
