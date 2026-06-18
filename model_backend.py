

import contextlib
import torch


# ======================================================================
# Common helpers
# ======================================================================
def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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

        print(f"[backend=unsloth] loading {self.cfg.model_name} ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=self.cfg.load_in_4bit,
            dtype=torch.bfloat16,
        )
        print("[backend=unsloth] attaching LoRA ...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.target_modules),
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"[backend=hf] loading {self.cfg.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("[backend=hf] bitsandbytes not available, ignoring load_in_4bit")

        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **model_kwargs)

        if self.cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        print("[backend=hf] attaching LoRA ...")
        peft_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.target_modules),
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

    "auto": try Unsloth first, fall back to HF if anything fails
    (import error, unsupported architecture, etc).
    """
    if name == "hf":
        return HFBackend(cfg)
    if name == "unsloth":
        return UnslothBackend(cfg)
    if name == "auto":
        try:
            b = UnslothBackend(cfg)
            import unsloth 
            print("[backend=auto] Unsloth available, will try unsloth first")
            return _AutoFallbackBackend(cfg)
        except Exception as e:
            print(f"[backend=auto] Unsloth not importable ({e}); falling back to HF")
            return HFBackend(cfg)
    raise ValueError(f"Unknown backend: {name}")


class _AutoFallbackBackend:
    """Wrapper that tries Unsloth on .load(); on failure, swaps to HF."""
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
            print("[backend=auto] Falling back to plain transformers + PEFT")
            inner = HFBackend(self.cfg)
            m, t = inner.load()
            self._inner = inner
            return m, t

    def set_inference_mode(self):
        self._inner.set_inference_mode()

    def set_training_mode(self):
        self._inner.set_training_mode()

    def disable_adapter(self):
        return self._inner.disable_adapter()
