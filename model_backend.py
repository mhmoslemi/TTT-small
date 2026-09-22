

import importlib.util
import os
import torch
import weakref


_FUSED_LONG_ATTENTION = False
_FUSED_LONG_ATTENTION_DISABLED = False
_FUSED_LONG_ATTENTION_ACTIVE_REPORTED = False
_FUSED_LONG_ATTENTION_FALLBACK_REPORTED = False
_SHARED_PREFIX_ATTENTION_LAYOUTS = {}
_CURRENT_SHARED_PREFIX_ATTENTION_LAYOUT = None


def register_shared_prefix_attention_mask(mask, prefix_end, branches):
    """Associate a compact packed-branch mask with its contiguous layout.

    The registry keeps only a weak reference.  Gradient checkpointing retains
    ``mask`` until the backward recomputation has finished; once that graph is
    released, the callback deletes the temporary layout automatically.
    """
    if (not torch.is_tensor(mask) or mask.ndim != 3
            or tuple(mask.shape[:2]) != (1, 2)):
        raise ValueError("shared-prefix attention mask must have shape (1, 2, T)")
    prefix_end = int(prefix_end)
    normalized = tuple((int(start), int(end)) for start, end in branches)
    total = int(mask.shape[-1])
    if prefix_end < 1 or prefix_end > total:
        raise ValueError("shared-prefix boundary is outside the packed sequence")
    cursor = prefix_end
    for start, end in normalized:
        if start != cursor or end < start or end > total:
            raise ValueError("shared-prefix branches must be contiguous")
        cursor = end
    if cursor != total:
        raise ValueError("shared-prefix layout does not cover the packed sequence")
    key = (str(mask.device), int(mask.data_ptr()))

    def discard(reference, *, registry_key=key):
        current = _SHARED_PREFIX_ATTENTION_LAYOUTS.get(registry_key)
        if current is not None and current[0] is reference:
            _SHARED_PREFIX_ATTENTION_LAYOUTS.pop(registry_key, None)

    reference = weakref.ref(mask, discard)
    _SHARED_PREFIX_ATTENTION_LAYOUTS[key] = (
        reference, prefix_end, normalized)
    # Training performs one synchronous forward/backward at a time in each
    # process.  Accelerate may create one mask copy per sharded decoder layer;
    # this device-independent entry lets those copies recover the current
    # layout without a GPU-to-CPU synchronization on every layer.
    global _CURRENT_SHARED_PREFIX_ATTENTION_LAYOUT
    _CURRENT_SHARED_PREFIX_ATTENTION_LAYOUT = (
        total, prefix_end, normalized)


def _shared_prefix_attention_layout(mask):
    if (not torch.is_tensor(mask) or mask.ndim != 3
            or tuple(mask.shape[:2]) != (1, 2)):
        return None
    key = (str(mask.device), int(mask.data_ptr()))
    entry = _SHARED_PREFIX_ATTENTION_LAYOUTS.get(key)
    if entry is not None and entry[0]() is mask:
        return entry[1], entry[2]

    current = _CURRENT_SHARED_PREFIX_ATTENTION_LAYOUT
    if (current is not None
            and int(mask.shape[-1]) == current[0]
            and mask.dtype in (
                torch.int8, torch.int16, torch.int32,
                torch.int64, torch.uint8)):
        return current[1], current[2]
    raise RuntimeError("unregistered shared-prefix attention mask")


# ======================================================================
# Common helpers
# ======================================================================
def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def set_gradient_checkpointing(model, enabled):
    """Toggle transformer checkpointing without changing training mode.

    PEFT and Unsloth wrappers normally forward these methods to the underlying
    Transformers model.  Keeping this compatibility shim here lets the shared
    OOM controller select checkpointing per workload instead of forcing its
    recomputation cost on every update.
    """
    method_name = (
        "gradient_checkpointing_enable"
        if bool(enabled) else
        "gradient_checkpointing_disable"
    )
    method = getattr(model, method_name, None)
    if not callable(method):
        getter = getattr(model, "get_base_model", None)
        if callable(getter):
            try:
                method = getattr(getter(), method_name, None)
            except (AttributeError, TypeError):
                method = None
    if not callable(method):
        return False
    if enabled:
        try:
            method(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            method()
    else:
        method()
    return True


def _training_model_name(cfg):
    return str(getattr(cfg, "training_model_name",
                       getattr(cfg, "model_name", "")))


def _lora_rank(cfg):
    if bool(getattr(cfg, "binary_coder_training", False)):
        return int(getattr(cfg, "binary_coder_lora_rank"))
    return int(cfg.lora_rank)


def _requires_unsloth_gpt_oss_loader(model_name):
    name = str(model_name).strip().lower()
    return "gpt-oss" in name and "unsloth-bnb-4bit" in name


def _training_device_map(cfg):
    """Choose replicated data parallelism or single-process model sharding."""
    replica_device = getattr(cfg, "training_replica_device", None)
    if replica_device is not None:
        return {"": int(replica_device)}
    count = int(getattr(cfg, "num_training_gpus", 1) or 1)
    return "balanced" if count > 1 else {"": 0}


def _training_max_memory(cfg):
    budgets = list(getattr(cfg, "training_max_memory_gib", None) or [])
    if not budgets:
        return None
    replica_device = getattr(cfg, "training_replica_device", None)
    if replica_device is not None:
        logical_id = int(replica_device)
        if logical_id >= len(budgets):
            return None
        return {logical_id: f"{float(budgets[logical_id]):.1f}GiB"}
    return {logical_id: f"{float(gib):.1f}GiB"
            for logical_id, gib in enumerate(budgets)}


def _quantization_method(model_config):
    quantization = getattr(model_config, "quantization_config", None)
    if not quantization:
        return ""
    if isinstance(quantization, dict):
        method = quantization.get("quant_method", "")
        load_4bit = quantization.get("load_in_4bit",
                                     quantization.get("_load_in_4bit", False))
    else:
        method = getattr(quantization, "quant_method", "")
        load_4bit = getattr(quantization, "load_in_4bit",
                            getattr(quantization, "_load_in_4bit", False))
    method = str(method or "").lower()
    if method == "bitsandbytes" and load_4bit:
        return "bitsandbytes-4bit"
    return method


def _use_training_4bit(
        cfg, *, native_quantization=False, prequantized_bnb=False):
    """Resolve the requested QLoRA mode without affecting vLLM."""
    model_name = _training_model_name(cfg)
    if prequantized_bnb:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(
                f"{model_name} is a trainable BitsAndBytes checkpoint, but "
                "bitsandbytes is not installed")
        print(f"[precision] {model_name} is already BitsAndBytes 4-bit")
        return True
    if not bool(getattr(cfg, "load_in_4bit", False)):
        return False
    if native_quantization:
        print(f"[precision] {model_name} has checkpoint-native quantization; "
              "not applying BitsAndBytes 4-bit again")
        return False
    if importlib.util.find_spec("bitsandbytes") is None:
        print("[precision] bitsandbytes is unavailable; using checkpoint/default "
              "precision instead of requested 4-bit")
        return False
    return True


def _hf_training_attention_implementation():
    """Internal exact attention backend; no flash-attn package is required."""
    return "ttt_blockwise_attention"


def _configure_fused_long_attention(enabled):
    global _FUSED_LONG_ATTENTION
    global _FUSED_LONG_ATTENTION_DISABLED
    global _FUSED_LONG_ATTENTION_ACTIVE_REPORTED
    global _FUSED_LONG_ATTENTION_FALLBACK_REPORTED

    _FUSED_LONG_ATTENTION = bool(enabled)
    _FUSED_LONG_ATTENTION_DISABLED = False
    _FUSED_LONG_ATTENTION_ACTIVE_REPORTED = False
    _FUSED_LONG_ATTENTION_FALLBACK_REPORTED = False


def _disable_fused_long_attention(reason):
    global _FUSED_LONG_ATTENTION_DISABLED
    global _FUSED_LONG_ATTENTION_FALLBACK_REPORTED

    _FUSED_LONG_ATTENTION_DISABLED = True
    if not _FUSED_LONG_ATTENTION_FALLBACK_REPORTED:
        print(f"[memory] fused long attention unavailable ({reason}); "
              "using exact blockwise fallback", flush=True)
        _FUSED_LONG_ATTENTION_FALLBACK_REPORTED = True


def _try_fused_long_attention(query, key, value, *, dropout, scale, groups):
    """Return exact fused causal attention, or None when unsupported.

    This path is called only for full, unmasked self-attention. Forcing the
    FLASH_ATTENTION SDPA backend prevents PyTorch from silently selecting the
    quadratic-memory math kernel. GQA is kept native so long key/value tensors
    are not physically repeated across query heads.
    """
    global _FUSED_LONG_ATTENTION_ACTIVE_REPORTED

    if not _FUSED_LONG_ATTENTION or _FUSED_LONG_ATTENTION_DISABLED:
        return None
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as error:
        _disable_fused_long_attention(type(error).__name__)
        return None

    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=float(dropout),
                scale=scale,
                is_causal=True,
                enable_gqa=bool(groups != 1),
            )
    except torch.OutOfMemoryError:
        # The outer OOM-resilient trainer will retry this batch. Disable this
        # path first so that retry immediately uses bounded blockwise attention.
        _disable_fused_long_attention("CUDA OOM")
        raise
    except (TypeError, RuntimeError) as error:
        message = str(error).lower()
        unsupported = (
            isinstance(error, TypeError)
            or "no available kernel" in message
            or "no viable backend" in message
            or "flash attention" in message
            or "flash_attention" in message
            or "enable_gqa" in message
            or "grouped query" in message
            or "not supported" in message
        )
        if not unsupported:
            raise
        _disable_fused_long_attention(
            str(error).strip().splitlines()[0] or type(error).__name__)
        return None

    if not _FUSED_LONG_ATTENTION_ACTIVE_REPORTED:
        print(f"[memory] fused exact long attention active: "
              f"sequence={int(query.shape[2])} tokens, "
              f"query_heads={int(query.shape[1])}, "
              f"kv_heads={int(key.shape[1])}", flush=True)
        _FUSED_LONG_ATTENTION_ACTIVE_REPORTED = True
    return output


def _raw_padding_attention_mask(*args, attention_mask=None, **kwargs):
    """Keep the compact padding or registered shared-prefix mask intact."""
    return attention_mask


def _repeat_kv_heads(hidden_states, repetitions):
    if repetitions == 1:
        return hidden_states
    batch, heads, length, width = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, heads, repetitions, length, width)
    return expanded.reshape(batch, heads * repetitions, length, width)


def _attention_allowed_mask(attention_mask, *, batch, query_start,
                            query_end, query_offset, key_length, device,
                            sliding_window=None):
    """Build only one query block of the causal/padding mask."""
    query_positions = (
        torch.arange(query_start, query_end, device=device) + query_offset)
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions[None, :] <= query_positions[:, None]
    if sliding_window is not None:
        allowed = allowed & (
            key_positions[None, :]
            > query_positions[:, None] - int(sliding_window))
    allowed = allowed[None, None, :, :]
    additive = None
    if attention_mask is None:
        return allowed, additive
    if attention_mask.ndim == 2:
        padding = attention_mask[:, None, None, :key_length].bool()
        return allowed & padding, additive
    if attention_mask.ndim == 4:
        block = attention_mask[
            :, :, query_start:query_end, :key_length]
        if block.dtype == torch.bool:
            return allowed & block, additive
        additive = block
        return allowed, additive
    raise ValueError(
        "ttt blockwise attention accepts a 2D padding or 4D attention mask")


def _ttt_shared_prefix_attention_forward(
        module, query, key, value, layout, *, dropout, scaling,
        sliding_window, **kwargs):
    """Attend one shared prompt plus mutually isolated response branches."""
    prefix_end, branches = layout
    prefix_query = query[:, :, :prefix_end, :]
    prefix_key = key[:, :, :prefix_end, :]
    prefix_value = value[:, :, :prefix_end, :]
    prefix_output, _ = _ttt_blockwise_attention_forward(
        module, prefix_query, prefix_key, prefix_value, None,
        dropout=dropout, scaling=scaling,
        sliding_window=sliding_window, **kwargs)
    outputs = [prefix_output]
    cursor = int(prefix_end)
    for start, end in branches:
        if start != cursor:
            raise RuntimeError("noncontiguous shared-prefix attention branch")
        cursor = end
        if end == start:
            continue
        branch_query = query[:, :, start:end, :]
        branch_key = torch.cat(
            (prefix_key, key[:, :, start:end, :]), dim=2)
        branch_value = torch.cat(
            (prefix_value, value[:, :, start:end, :]), dim=2)
        branch_output, _ = _ttt_blockwise_attention_forward(
            module, branch_query, branch_key, branch_value, None,
            dropout=dropout, scaling=scaling,
            sliding_window=sliding_window, **kwargs)
        outputs.append(branch_output)
    if cursor != int(query.shape[2]):
        raise RuntimeError("shared-prefix attention layout length mismatch")
    # AttentionInterface outputs are (batch, sequence, heads, width), even
    # though its query/key/value inputs are (batch, heads, sequence, width).
    return torch.cat(outputs, dim=1), None


def _ttt_blockwise_attention_forward(
        module, query, key, value, attention_mask, dropout=0.0,
        scaling=None, sliding_window=None, **kwargs):
    """Exact causal attention with a bounded score tensor and autograd.

    Small calls retain native SDPA speed. Large calls split only the query
    dimension, attend each block to the complete key/value history, and
    checkpoint each block. This is mathematically full attention—not context
    truncation—while peak score memory is O(query_block * sequence_length).
    """
    import torch.nn.functional as F

    shared_layout = _shared_prefix_attention_layout(attention_mask)
    if shared_layout is not None:
        if int(query.shape[0]) != 1:
            raise ValueError("shared-prefix attention requires batch size one")
        return _ttt_shared_prefix_attention_forward(
            module, query, key, value, shared_layout,
            dropout=dropout, scaling=scaling,
            sliding_window=sliding_window, **kwargs)

    batch, query_heads, query_length, head_dim = query.shape
    key_heads = int(key.shape[1])
    key_length = int(key.shape[2])
    if query_heads % key_heads:
        raise ValueError(
            "query attention heads must be divisible by key/value heads")
    groups = query_heads // key_heads
    scale = float(scaling) if scaling is not None else head_dim ** -0.5
    query_offset = key_length - query_length
    score_elements = batch * query_heads * query_length * key_length
    native_limit = 134_217_728

    fused_eligible = bool(
        score_elements > native_limit
        and batch == 1
        and attention_mask is None
        and sliding_window is None
        and query_length == key_length
        and query_length > 1
    )
    if fused_eligible:
        fused = _try_fused_long_attention(
            query, key, value,
            dropout=dropout, scale=scale, groups=groups)
        if fused is not None:
            return fused.transpose(1, 2).contiguous(), None

    if score_elements <= native_limit:
        native_key = _repeat_kv_heads(key, groups)
        native_value = _repeat_kv_heads(value, groups)
        native_mask = None
        is_causal = bool(
            attention_mask is None and sliding_window is None
            and query_length == key_length and query_length > 1)
        if not is_causal:
            allowed, additive = _attention_allowed_mask(
                attention_mask, batch=batch, query_start=0,
                query_end=query_length, query_offset=query_offset,
                key_length=key_length, device=query.device,
                sliding_window=sliding_window)
            native_mask = allowed if additive is None else additive.masked_fill(
                ~allowed, torch.finfo(additive.dtype).min)
        output = F.scaled_dot_product_attention(
            query, native_key, native_value, attn_mask=native_mask,
            dropout_p=float(dropout), scale=scale, is_causal=is_causal)
        return output.transpose(1, 2).contiguous(), None

    # Bound the FP32 probability block to roughly 256 MiB (and the BF16/FP16
    # score block to roughly 128 MiB).
    score_budget = 67_108_864
    denominator = max(1, batch * query_heads * key_length)
    query_block = max(1, min(256, score_budget // denominator))

    def attend_block(query_part, key_states, value_states, start, end):
        local_query = query_part.reshape(
            batch, key_heads, groups, end - start, head_dim)
        scores = torch.matmul(
            local_query,
            key_states[:, :, None, :, :].transpose(-2, -1),
        ) * scale
        allowed, additive = _attention_allowed_mask(
            attention_mask, batch=batch, query_start=start, query_end=end,
            query_offset=query_offset, key_length=key_length,
            device=query.device, sliding_window=sliding_window)
        allowed = allowed[:, :, None, :, :]
        scores = scores.masked_fill(
            ~allowed, torch.finfo(scores.dtype).min)
        if additive is not None:
            scores = scores + additive[:, :, None, :, :]
        probabilities = F.softmax(
            scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout:
            probabilities = F.dropout(
                probabilities, p=float(dropout), training=module.training)
        result = torch.matmul(
            probabilities, value_states[:, :, None, :, :])
        return result.reshape(
            batch, query_heads, end - start, head_dim)

    output_blocks = []
    needs_grad = bool(
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad))
    for start in range(0, query_length, query_block):
        end = min(start + query_block, query_length)
        if needs_grad:
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(
                attend_block, query[:, :, start:end, :], key, value,
                start, end, use_reentrant=False)
        else:
            output = attend_block(
                query[:, :, start:end, :], key, value, start, end)
        output_blocks.append(output)
    output = torch.cat(output_blocks, dim=2)
    return output.transpose(1, 2).contiguous(), None


def _register_ttt_attention_backend(*, fused_long_attention=False):
    """Register runtime and mask functions before Transformers builds a model."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    _configure_fused_long_attention(fused_long_attention)
    name = _hf_training_attention_implementation()
    AttentionInterface.register(name, _ttt_blockwise_attention_forward)
    AttentionMaskInterface.register(name, _raw_padding_attention_mask)


class _ModelPlacementBackend:
    """Move a possibly sharded trainer out for an all-GPU vLLM phase."""

    def _remember_training_placement(self, loaded_model):
        self._placement_model = loaded_model
        device_map = getattr(loaded_model, "hf_device_map", None) or {}
        # A whole-model explicit device_map is not guaranteed to survive on
        # every Transformers/Accelerate combination. Replicated training owns
        # an unambiguous logical device, so retain it directly instead of
        # allowing restore_after_generation() to fall back to cuda:0. Without
        # this, all replicas can collapse onto GPU 0 after the first vLLM phase.
        replica_device = getattr(
            getattr(self, "cfg", None), "training_replica_device", None)
        if replica_device is not None:
            device_map = {"": int(replica_device)}
        self._training_device_map = dict(device_map)
        cpu_targets = [name for name, target in self._training_device_map.items()
                       if str(target).lower() in ("cpu", "disk")]
        if cpu_targets:
            raise RuntimeError(
                "the balanced training device map spilled model modules to "
                f"CPU/disk ({cpu_targets[:4]}). Reduce context/batch memory or "
                "provide more training GPUs; silent spill would be unstable")
        self._trainer_is_offloaded = False

    def offload_for_generation(self):
        if getattr(self, "_trainer_is_offloaded", False):
            return
        placement_model = getattr(self, "_placement_model", self.model)
        if getattr(self, "_training_device_map", None):
            # Remove Accelerate's routing hooks before collapsing the sharded
            # model to host memory. They are reconstructed during restore.
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(placement_model)
            except (ImportError, AttributeError):
                pass
        try:
            self.model.to("cpu")
            self._trainer_is_offloaded = True
        except Exception:
            # A quantized module can fail after earlier layers already moved.
            # Mark it offloaded so the normal placement restore can repair the
            # partially moved model before the original exception propagates.
            self._trainer_is_offloaded = True
            try:
                self.restore_after_generation()
            except Exception:
                pass
            raise

    def restore_after_generation(self):
        if not getattr(self, "_trainer_is_offloaded", False):
            return
        device_map = getattr(self, "_training_device_map", None) or {}
        placement_model = getattr(self, "_placement_model", self.model)
        devices = {str(target) for target in device_map.values()
                   if str(target).lower() not in ("cpu", "disk")}
        if len(devices) > 1:
            from accelerate import dispatch_model
            dispatch_model(
                placement_model, device_map=device_map, force_hooks=True)
        else:
            target = next(iter(devices), "0")
            target = target if str(target).startswith("cuda:") else f"cuda:{target}"
            self.model.to(target)
        self._trainer_is_offloaded = False


def _offline_mode() -> bool:
    truthy = {"1", "true", "yes", "on"}
    return any(str(os.environ.get(name, "")).strip().lower() in truthy
               for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _expert_count(model_config) -> int:
    for _ in range(2):
        if model_config is None:
            break
        for name in ("num_experts", "num_local_experts", "n_routed_experts"):
            value = getattr(model_config, name, None)
            if value:
                return int(value)
        model_config = getattr(model_config, "text_config", None)
    return 0


def _prepare_sparse_moe_for_kbit_training(model):
    """Freeze a partly quantized MoE without changing its forward dtypes.

    Some MoE implementations store all expert projections in packed 3-D
    Parameters instead of ``nn.Linear`` modules.  BitsAndBytes therefore leaves
    those frozen tensors in checkpoint precision.  PEFT's generic k-bit
    preparation casts every non-4-bit FP16/BF16 Parameter to FP32, which can
    nearly double a sparse model's resident size even though those expert
    weights will never be trained.  Mixing FP32 norm vectors with checkpoint-
    precision router/expert matrices is also invalid for implementations such
    as Qwen3-MoE: the norm promotes hidden states to FP32, then the BF16 router
    receives mismatched inputs.  Keep every frozen base tensor in its original
    dtype; LoRA parameters are attached separately below.
    """
    retained_parameters = 0
    retained_values = 0
    for parameter in model.parameters():
        parameter.requires_grad = False
        if parameter.dtype in (torch.float16, torch.bfloat16):
            retained_parameters += 1
            retained_values += int(parameter.numel())
    print(
        "[memory] sparse-MoE k-bit preparation: froze base weights; "
        f"kept {retained_parameters} FP16/BF16 tensors "
        f"({retained_values:,} values) in checkpoint precision so norms, "
        "routers, and experts use compatible dtypes",
        flush=True,
    )
    return model


def _resolve_lora_target_modules(cfg, model_config):
    """Avoid allocating a separate large LoRA across every MoE expert."""
    targets = list(cfg.target_modules)
    experts = _expert_count(model_config)
    expert_mlp_targets = {"gate_proj", "up_proj", "down_proj"}
    removed = [name for name in targets if name in expert_mlp_targets]
    kept = [name for name in targets if name not in expert_mlp_targets]
    if experts >= 64 and removed and kept:
        print(f"[memory] large MoE ({experts} experts): excluding expert-wide "
              f"LoRA targets {removed}; keeping {kept} at rank {_lora_rank(cfg)}")
        targets = kept
    cfg.effective_target_modules = tuple(targets)
    return targets


# ======================================================================
# Unsloth backend
# ======================================================================
class UnslothBackend(_ModelPlacementBackend):
    name = "unsloth"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._FastLanguageModel = None

    def load(self):
        # Unsloth must be imported BEFORE transformers/trl/peft
        from unsloth import FastLanguageModel
        self._FastLanguageModel = FastLanguageModel

        training_name = _training_model_name(self.cfg)
        prequantized_bnb = "bnb-4bit" in training_name.lower()
        use_4bit = _use_training_4bit(
            self.cfg, prequantized_bnb=prequantized_bnb)
        self.cfg.effective_load_in_4bit = use_4bit

        device_map = _training_device_map(self.cfg)
        print(f"[backend=unsloth] loading {training_name} across "
              f"{int(getattr(self.cfg, 'num_training_gpus', 1))} GPU(s) ...")
        offline = _offline_mode()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=training_name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=use_4bit,
            dtype=torch.bfloat16,
            # In offline mode Unsloth otherwise remaps a cached upstream model
            # to an uncached `unsloth/*-bnb-4bit` Hub repository. Pin the exact
            # requested repo and let BitsAndBytes quantize its cached weights.
            use_exact_model_name=offline,
            local_files_only=offline,
            device_map=device_map,
            **({"max_memory": _training_max_memory(self.cfg)}
               if _training_max_memory(self.cfg) else {}),
        )
        self._remember_training_placement(model)
        target_modules = _resolve_lora_target_modules(
            self.cfg, getattr(model, "config", None))
        print("[backend=unsloth] attaching LoRA ...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=_lora_rank(self.cfg),
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.cfg.seed,
        )
        set_gradient_checkpointing(model, False)
        tokenizer = _ensure_pad_token(tokenizer)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def set_inference_mode(self):
        self._FastLanguageModel.for_inference(self.model)

    def set_training_mode(self):
        self._FastLanguageModel.for_training(self.model)

    def disable_adapter(self):
        return self.model.disable_adapter()


# ======================================================================
# Plain HF + PEFT backend (fallback)
# ======================================================================
class HFBackend(_ModelPlacementBackend):
    name = "hf"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        training_name = _training_model_name(self.cfg)
        device_map = _training_device_map(self.cfg)
        replica_device = getattr(self.cfg, "training_replica_device", None)
        if replica_device is None:
            placement = (
                f"across {int(getattr(self.cfg, 'num_training_gpus', 1))} "
                "GPU(s)"
            )
        else:
            placement = (
                "as one replicated trainer on logical GPU "
                f"{int(replica_device)}"
            )
        print(f"[backend=hf] loading {training_name} {placement} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            training_name, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(
            training_name, trust_remote_code=True)
        quantization_method = _quantization_method(hf_config)
        if quantization_method == "mxfp4":
            raise RuntimeError(
                f"{training_name} uses inference-only MXFP4 weights. Native "
                "MXFP4 cannot be LoRA-trained by Transformers; enable "
                "load_in_4bit so GPT-OSS selects its trainable BitsAndBytes "
                "checkpoint, or set training_model_name explicitly.")
        prequantized_bnb = quantization_method == "bitsandbytes-4bit"
        native_quantization = bool(
            getattr(hf_config, "quantization_config", None)
            and not prequantized_bnb)
        use_4bit = _use_training_4bit(
            self.cfg, native_quantization=native_quantization,
            prequantized_bnb=prequantized_bnb)
        self.cfg.effective_load_in_4bit = use_4bit

        # GPU-mode's exclusive evaluation card was removed from visibility by
        # role allocation. "balanced" therefore uses every rollout/training GPU
        # without ever placing weights on the benchmark card.
        fused_long_attention = bool(
            getattr(self.cfg, "fused_long_attention", False))
        _register_ttt_attention_backend(
            fused_long_attention=fused_long_attention)
        attention_implementation = _hf_training_attention_implementation()
        model_kwargs = dict(
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attention_implementation,
        )
        attention_detail = (
            "native SDPA + exact fused long-context attention + exact "
            "bounded fallback"
            if fused_long_attention else
            "native SDPA + exact bounded long-context blocks"
        )
        print(f"[memory] HF training attention: "
              f"{attention_implementation} ({attention_detail}; "
              "no external package)")
        max_memory = _training_max_memory(self.cfg)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if use_4bit and not prequantized_bnb:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                use_4bit = False
                self.cfg.effective_load_in_4bit = False
                print("[backend=hf] bitsandbytes not available; using default precision")

        model = AutoModelForCausalLM.from_pretrained(
            training_name, **model_kwargs)
        self._remember_training_placement(model)

        if use_4bit:
            if _expert_count(hf_config) > 0:
                model = _prepare_sparse_moe_for_kbit_training(model)
            else:
                model = prepare_model_for_kbit_training(model)

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        target_modules = _resolve_lora_target_modules(self.cfg, hf_config)
        print("[backend=hf] attaching LoRA ...")
        peft_cfg = LoraConfig(
            r=_lora_rank(self.cfg),
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        set_gradient_checkpointing(model, False)
        model.print_trainable_parameters()

        tokenizer = _ensure_pad_token(tokenizer)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def set_inference_mode(self):
        self.model.eval()

    def set_training_mode(self):
        self.model.train()

    def disable_adapter(self):
        return self.model.disable_adapter()


# ======================================================================
# Factory with automatic fallback
# ======================================================================
def load_backend(name: str, cfg):
    """
    name in {"unsloth", "hf", "auto"}.

    "auto": use plain HF in offline mode; otherwise use Unsloth when installed.
    Importing Unsloth mutates Transformers classes globally, so a failed
    Unsloth load cannot safely fall back to HF in the same interpreter.
    """
    requires_unsloth = _requires_unsloth_gpt_oss_loader(
        _training_model_name(cfg))
    if name == "hf" and requires_unsloth:
        raise RuntimeError(
            f"{_training_model_name(cfg)} uses Unsloth's split quantized "
            "GPT-OSS expert layout and cannot be loaded safely by vanilla "
            "Transformers. Select backend: unsloth; normal configuration "
            "loading performs this routing automatically.")
    if name == "auto" and requires_unsloth:
        print("[backend=auto] GPT-OSS Unsloth BNB checkpoint requires the "
              "patched Unsloth loader")
        return UnslothBackend(cfg)
    if name == "hf":
        return HFBackend(cfg)
    if name == "unsloth":
        return UnslothBackend(cfg)
    if name == "auto":
        if _offline_mode():
            print("[backend=auto] offline mode: selecting HF before importing "
                  "Unsloth (avoids uncached Unsloth checkpoint remapping)")
            return HFBackend(cfg)
        if importlib.util.find_spec("unsloth") is not None:
            print("[backend=auto] Unsloth installed; selecting Unsloth")
            return _AutoFallbackBackend(cfg)
        print("[backend=auto] Unsloth is not installed; selecting HF")
        return HFBackend(cfg)
    raise ValueError(f"Unknown backend: {name}")


class _AutoFallbackBackend:
    """Online auto-selected Unsloth with a safe post-patch failure message."""
    name = "auto"

    def __init__(self, cfg):
        self.cfg = cfg
        self._inner = None

    def load(self):
        try:
            inner = UnslothBackend(self.cfg)
            m, t = inner.load()
            self._inner = inner
            return m, t
        except Exception as e:
            print(f"[backend=auto] Unsloth load failed: {e!r}")
            raise RuntimeError(
                "Unsloth patched Transformers before its model load failed, so "
                "an in-process HF fallback would be unsafe. Re-run with "
                "backend: hf (or fix/cache the Unsloth checkpoint and use "
                "backend: unsloth).") from e

    def set_inference_mode(self):
        self._inner.set_inference_mode()

    def set_training_mode(self):
        self._inner.set_training_mode()

    def disable_adapter(self):
        return self._inner.disable_adapter()

    def offload_for_generation(self):
        return self._inner.offload_for_generation()

    def restore_after_generation(self):
        return self._inner.restore_after_generation()
