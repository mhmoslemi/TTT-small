

import torch

from model_io import ensure_pad_token, is_vlm


# ======================================================================
# Common helpers
# ======================================================================
def _ensure_pad_token(tokenizer):
    return ensure_pad_token(tokenizer)


def _language_lora_targets(model, target_modules):
    """Resolve language-side module names without touching the vision tower."""

    leaves = set(target_modules)
    vision_markers = (
        "vision", "visual", "image_tower", "image_encoder",
        "multi_modal_projector", "multimodal_projector", "mm_projector",
    )
    targets = []
    for name, _module in model.named_modules():
        lowered = name.lower()
        if name.rsplit(".", 1)[-1] not in leaves:
            continue
        if any(marker in lowered for marker in vision_markers):
            continue
        targets.append(name)
    if not targets:
        raise ValueError(
            "none of target_modules matched the VLM language stack; set "
            "vlm_finetune_vision_layers=true to use all linear layers, or "
            "update Config.target_modules for this architecture"
        )
    return targets


# ======================================================================
# Unsloth backend
# ======================================================================
class UnslothBackend:
    name = "unsloth"
    is_vision = False

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
    is_vision = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"[backend=hf] loading {self.cfg.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)

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
# Vision-language backends
# ======================================================================
class UnslothVisionBackend:
    name = "unsloth"
    is_vision = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._FastVisionModel = None

    def load(self):
        # As with FastLanguageModel, Unsloth must patch before transformers is
        # imported elsewhere in the process.
        from unsloth import FastVisionModel
        self._FastVisionModel = FastVisionModel

        print(f"[backend=unsloth, model=vlm] loading {self.cfg.model_name} ...")
        model, processor = FastVisionModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            load_in_4bit=self.cfg.load_in_4bit,
            dtype=torch.bfloat16,
        )
        if not hasattr(processor, "image_processor"):
            # Some Unsloth model families return the decoder tokenizer here.
            # The real AutoProcessor is still required to produce pixel values.
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                self.cfg.model_name, trust_remote_code=True)
        print("[backend=unsloth, model=vlm] attaching LoRA ...")
        tune_vision = bool(
            getattr(self.cfg, "vlm_finetune_vision_layers", False))
        model = FastVisionModel.get_peft_model(
            model,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.cfg.seed,
            finetune_vision_layers=tune_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        processor = _ensure_pad_token(processor)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        self.model = model
        self.tokenizer = processor
        return model, processor

    def set_inference_mode(self):
        self._FastVisionModel.for_inference(self.model)

    def set_training_mode(self):
        self._FastVisionModel.for_training(self.model)

    def disable_adapter(self):
        return self.model.disable_adapter()


class HFVisionBackend:
    name = "hf"
    is_vision = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self):
        import transformers
        from transformers import AutoProcessor
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        auto_cls = None
        for class_name in (
                "AutoModelForImageTextToText", "AutoModelForMultimodalLM",
                "AutoModelForVision2Seq"):
            auto_cls = getattr(transformers, class_name, None)
            if auto_cls is not None:
                break
        if auto_cls is None:
            raise ImportError(
                "this Transformers version has no multimodal generative auto "
                "model class; update Transformers"
            )

        print(f"[backend=hf, model=vlm] loading {self.cfg.model_name} ...")
        processor = AutoProcessor.from_pretrained(
            self.cfg.model_name, trust_remote_code=True)
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
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
                print("[backend=hf, model=vlm] bitsandbytes unavailable; "
                      "ignoring load_in_4bit")

        model = auto_cls.from_pretrained(self.cfg.model_name, **model_kwargs)
        if self.cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # The normal default adapts only the language-side attention/MLP
        # projections.  Opting into vision-layer tuning intentionally widens
        # LoRA to every linear layer because vision module names vary by model.
        targets = ("all-linear" if bool(
            getattr(self.cfg, "vlm_finetune_vision_layers", False))
            else _language_lora_targets(model, self.cfg.target_modules))
        print("[backend=hf, model=vlm] attaching LoRA ...")
        peft_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        processor = _ensure_pad_token(processor)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        self.model = model
        self.tokenizer = processor
        return model, processor

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
    vision = is_vlm(getattr(cfg, "model_kind", "llm"))
    unsloth_cls = UnslothVisionBackend if vision else UnslothBackend
    hf_cls = HFVisionBackend if vision else HFBackend
    if name == "hf":
        return hf_cls(cfg)
    if name == "unsloth":
        return unsloth_cls(cfg)
    if name == "auto":
        try:
            import unsloth 
            print("[backend=auto] Unsloth available, will try unsloth first"
                  + (" for the VLM" if vision else ""))
            return _AutoFallbackBackend(cfg, unsloth_cls, hf_cls)
        except Exception as e:
            print(f"[backend=auto] Unsloth not importable ({e}); falling back to HF")
            return hf_cls(cfg)
    raise ValueError(f"Unknown backend: {name}")


class _AutoFallbackBackend:
    """Wrapper that tries Unsloth on .load(); on failure, swaps to HF."""
    name = "auto"

    def __init__(self, cfg, unsloth_cls=UnslothBackend, hf_cls=HFBackend):
        self.cfg = cfg
        self.unsloth_cls = unsloth_cls
        self.hf_cls = hf_cls
        self.is_vision = bool(getattr(unsloth_cls, "is_vision", False))
        self._inner = None

    def load(self):
        try:
            inner = self.unsloth_cls(self.cfg)
            m, t = inner.load()
            self._inner = inner
            return m, t
        except Exception as e:
            print(f"[backend=auto] Unsloth load failed: {e!r}")
            print("[backend=auto] Falling back to plain transformers + PEFT")
            inner = self.hf_cls(self.cfg)
            m, t = inner.load()
            self._inner = inner
            return m, t

    def set_inference_mode(self):
        self._inner.set_inference_mode()

    def set_training_mode(self):
        self._inner.set_training_mode()

    def disable_adapter(self):
        return self._inner.disable_adapter()
