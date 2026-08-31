

import importlib.util
import os
import torch


# ======================================================================
# Common helpers
# ======================================================================
def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _use_training_4bit(cfg, *, native_quantization=False):
    """Resolve the requested QLoRA mode without affecting vLLM."""
    if not bool(getattr(cfg, "load_in_4bit", False)):
        return False
    model_name = str(getattr(cfg, "model_name", ""))
    if native_quantization or "gpt-oss" in model_name.lower():
        print(f"[precision] {model_name} has checkpoint-native quantization; "
              "not applying BitsAndBytes 4-bit again")
        return False
    if importlib.util.find_spec("bitsandbytes") is None:
        print("[precision] bitsandbytes is unavailable; using checkpoint/default "
              "precision instead of requested 4-bit")
        return False
    return True


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


def _resolve_lora_target_modules(cfg, model_config):
    """Avoid allocating a separate large LoRA across every MoE expert."""
    targets = list(cfg.target_modules)
    experts = _expert_count(model_config)
    expert_mlp_targets = {"gate_proj", "up_proj", "down_proj"}
    removed = [name for name in targets if name in expert_mlp_targets]
    kept = [name for name in targets if name not in expert_mlp_targets]
    if experts >= 64 and removed and kept:
        print(f"[memory] large MoE ({experts} experts): excluding expert-wide "
              f"LoRA targets {removed}; keeping {kept} at rank {cfg.lora_rank}")
        targets = kept
    cfg.effective_target_modules = tuple(targets)
    return targets


# ======================================================================
# Unsloth backend
# ======================================================================
class UnslothBackend:
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

        use_4bit = _use_training_4bit(self.cfg)
        self.cfg.effective_load_in_4bit = use_4bit

        print(f"[backend=unsloth] loading {self.cfg.model_name} ...")
        offline = _offline_mode()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=use_4bit,
            dtype=torch.bfloat16,
            # In offline mode Unsloth otherwise remaps a cached upstream model
            # to an uncached `unsloth/*-bnb-4bit` Hub repository. Pin the exact
            # requested repo and let BitsAndBytes quantize its cached weights.
            use_exact_model_name=offline,
            local_files_only=offline,
            # The trainer process is restricted to its configured CUDA device,
            # and the explicit map prevents Accelerate/Unsloth from rediscovering
            # and replicating/sharding the policy over every visible card.
            device_map={"": 0},
        )
        target_modules = _resolve_lora_target_modules(
            self.cfg, getattr(model, "config", None))
        print("[backend=unsloth] attaching LoRA ...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.cfg.seed,
        )
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
class HFBackend:
    name = "hf"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"[backend=hf] loading {self.cfg.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(
            self.cfg.model_name, trust_remote_code=True)
        native_quantization = bool(getattr(hf_config, "quantization_config", None))
        use_4bit = _use_training_4bit(
            self.cfg, native_quantization=native_quantization)
        self.cfg.effective_load_in_4bit = use_4bit

        # Pinned to one device, NOT "auto". With two visible GPUs, "auto" shards
        # the model across both and puts training weights on the card that
        # kernel_gpu_id reserves for benchmarking, so the measured runtime
        # reflects contention with the backward pass instead of the kernel.
        # Index is into CUDA_VISIBLE_DEVICES, i.e. the first device this process
        # can see; the benchmark child re-sets CUDA_VISIBLE_DEVICES itself.
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        if use_4bit:
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

        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **model_kwargs)

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        target_modules = _resolve_lora_target_modules(self.cfg, hf_config)
        print("[backend=hf] attaching LoRA ...")
        peft_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
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
