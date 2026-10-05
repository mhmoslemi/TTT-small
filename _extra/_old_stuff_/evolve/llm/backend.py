"""
Model loading: Unsloth or plain transformers + PEFT, with automatic fallback.

A single backbone serves all three roles in the framework -- generator, Elo
judge and memory maker -- and only its LoRA adapter is ever trained.

Adapted from the reference implementation's model_backend.py; the difference is
that these read a ModelConfig rather than a flat namespace.
"""


def as_tokenizer(obj):
    """
    Unwrap a Processor to the text tokenizer inside it.

    Multimodal checkpoints (Qwen*-VL and friends) load with a Processor rather
    than a tokenizer, and a Processor's FIRST POSITIONAL argument is `images`,
    not `text`. Passing a prompt positionally therefore sends it into the image
    pipeline, which fails deep inside base64 decoding with a message that says
    nothing about tokenization. This framework is text-only, so reach through to
    the tokenizer and always call it with text= as a keyword.
    """
    inner = getattr(obj, "tokenizer", None)
    if inner is not None and hasattr(inner, "convert_tokens_to_ids"):
        return inner
    return obj


def _ensure_pad_token(obj):
    """Set the pad token on the tokenizer, and on the processor wrapping it."""
    tokenizer = as_tokenizer(obj)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if obj is not tokenizer and getattr(obj, "pad_token_id", None) is None:
        try:
            obj.pad_token = tokenizer.pad_token
        except AttributeError:
            pass          # not all processors expose it; the tokenizer is what we use
    return obj


class UnslothBackend:
    name = "unsloth"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._FastLanguageModel = None

    def load(self):
        # Unsloth patches transformers, so it must be imported first.
        from unsloth import FastLanguageModel
        import torch

        self._FastLanguageModel = FastLanguageModel
        print(f"[backend=unsloth] loading {self.cfg.name} ...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=self.cfg.load_in_4bit,
            dtype=torch.bfloat16,
        )
        print("[backend=unsloth] attaching LoRA ...", flush=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = _ensure_pad_token(tokenizer)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None

        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    def set_inference_mode(self):
        self._FastLanguageModel.for_inference(self.model)

    def set_training_mode(self):
        self._FastLanguageModel.for_training(self.model)

    def disable_adapter(self):
        return self.model.disable_adapter()


class HFBackend:
    name = "hf"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"[backend=hf] loading {self.cfg.name} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.name, trust_remote_code=True)

        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                      trust_remote_code=True)
        if self.cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("[backend=hf] bitsandbytes missing; ignoring load_in_4bit")

        model = AutoModelForCausalLM.from_pretrained(self.cfg.name, **kwargs)
        if self.cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        print("[backend=hf] attaching LoRA ...", flush=True)
        model = get_peft_model(model, LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

        tokenizer = _ensure_pad_token(tokenizer)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None

        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    def set_inference_mode(self):
        self.model.eval()

    def set_training_mode(self):
        self.model.train()

    def disable_adapter(self):
        return self.model.disable_adapter()


class _AutoFallbackBackend:
    """Try Unsloth on load(); swap to HF+PEFT if anything goes wrong."""
    name = "auto"

    def __init__(self, cfg):
        self.cfg = cfg
        self._inner = None

    def load(self):
        try:
            inner = UnslothBackend(self.cfg)
            out = inner.load()
        except Exception as e:
            print(f"[backend=auto] Unsloth load failed: {e!r}")
            print("[backend=auto] falling back to transformers + PEFT")
            inner = HFBackend(self.cfg)
            out = inner.load()
        self._inner = inner
        self.name = inner.name
        return out

    def set_inference_mode(self):
        self._inner.set_inference_mode()

    def set_training_mode(self):
        self._inner.set_training_mode()

    def disable_adapter(self):
        return self._inner.disable_adapter()

    @property
    def model(self):
        return self._inner.model

    @property
    def tokenizer(self):
        return self._inner.tokenizer


def load_backend(cfg):
    """cfg is a ModelConfig. cfg.backend selects unsloth | hf | auto."""
    name = (cfg.backend or "auto").lower()
    if name == "hf":
        return HFBackend(cfg)
    if name == "unsloth":
        return UnslothBackend(cfg)
    if name == "auto":
        try:
            import unsloth  # noqa: F401
            print("[backend=auto] Unsloth importable; trying it first")
            return _AutoFallbackBackend(cfg)
        except Exception as e:
            print(f"[backend=auto] Unsloth not importable ({e}); using HF")
            return HFBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.backend!r} (expected auto|unsloth|hf)")
