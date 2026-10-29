"""Native contracts for the two large, frozen local strategists (no CUDA imports)."""

from copy import deepcopy
from functools import lru_cache
from importlib import metadata, util
from pathlib import Path
import json
import re


STRATEGIST_ALIASES = {
    "qwen3.8-flash-next": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "deepseek-v4-flash-0731": "nvidia/DeepSeek-V4-Flash-0731-NVFP4",
}
DEEPSEEK_ENCODER_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_ENCODER_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def resolve_strategy_model(model_name):
    name = str(model_name or "").strip()
    # A full Hub id or local directory is an explicit checkpoint choice.
    if Path(name).is_dir():
        return name
    return STRATEGIST_ALIASES.get(name.lower(), name)


def strategist_profile(model_name):
    name = resolve_strategy_model(model_name).rstrip("/").rsplit("/", 1)[-1].lower()
    if name in {"qwen3.8-flash-next", "qwen3.8-flash-next-fp8",
                "qwen3.8-flash-next-nvfp4"}:
        return dict(
            family="qwen3.8-flash-next", heads=24, context=262_144,
            efforts=("low", "medium", "xhigh"), default_effort="xhigh",
            weights_gib=(126.0 if name.endswith("nvfp4") else
                         175.0 if name.endswith("fp8") else 338.0),
            # BF16 QSA K/V plus indexer state; two KV heads shard up to TP=2.
            ple_gib=48.0, kv_bytes_per_token=32_768, kv_heads=2,
            nvfp4=name.endswith("nvfp4"),
        )
    if name in {"deepseek-v4-flash-0731", "deepseek-v4-flash-0731-nvfp4"}:
        return dict(
            # User-selected runtime ceiling; the native model supports 1M.
            family="deepseek-v4-flash-0731", heads=64, context=262_144,
            efforts=("low", "high", "max"), default_effort="max",
            weights_gib=158.0, ple_gib=0.0, kv_bytes_per_token=8_192,
            nvfp4=name.endswith("nvfp4"),
        )
    return None


def resolve_strategy_effort(model_name, effort):
    profile = strategist_profile(model_name)
    effort = str(effort or "auto").strip().lower()
    if effort == "auto":
        return profile["default_effort"] if profile else "high"
    if profile and effort not in profile["efforts"]:
        raise ValueError(
            f"{model_name}: strategy_reasoning_effort must be auto or one of "
            f"{', '.join(profile['efforts'])}; got {effort!r}")
    return effort


def resolve_strategy_token_limits(model_name, max_seq_length="auto",
                                  max_new_tokens="auto", *, native_context=None):
    """Resolve native ceilings without loading weights or enabling context extension."""
    if max_seq_length == "auto":
        profile = strategist_profile(model_name)
        if native_context is not None:
            max_seq_length = native_context
        elif profile:
            max_seq_length = profile["context"]
        elif str(model_name).rstrip("/").rsplit("/", 1)[-1].lower() == (
                "deepseek-r1-0528-qwen3-8b"):
            max_seq_length = 131_072
        else:
            path = Path(model_name).expanduser() / "config.json"
            if not path.is_file():
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(
                    repo_id=model_name, filename="config.json"))
            config = json.loads(path.read_text())
            text_config = config.get("text_config") or config
            max_seq_length = text_config.get("max_position_embeddings")
            if max_seq_length is None:
                raise ValueError(f"{model_name}: no native context in config.json; "
                                 "set strategy_max_seq_length explicitly")
    if max_new_tokens == "auto":
        max_new_tokens = max_seq_length
    for key, value in (("strategy_max_seq_length", max_seq_length),
                       ("strategy_max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be auto or a positive integer")
    return max_seq_length, max_new_tokens


def strategist_memory_per_rank(model_name, parallel_size, max_length):
    """Account for native caches, replicated once TP exceeds the KV heads."""
    profile = strategist_profile(model_name)
    weight = (profile["weights_gib"] - profile["ple_gib"]) / parallel_size
    # Qwen's large lookup table is offloaded explicitly, not mistaken for
    # sharded GPU memory. vLLM asynchronously fetches the needed rows.
    kv_shards = min(parallel_size, profile.get("kv_heads", 1))
    kv = profile["kv_bytes_per_token"] * max_length / kv_shards / 2**30
    return weight, kv


def strategist_engine_options(model_name):
    profile = strategist_profile(model_name)
    # Checkpoint generation_config can impose its own server-wide output cap.
    options = dict(generation_config="vllm")
    if not profile:
        return options
    options.update(
        enable_lora=False, enable_expert_parallel=True,
        kv_cache_dtype="fp8", safetensors_load_strategy="lazy",
    )
    if profile["family"] == "qwen3.8-flash-next":
        # vLLM 0.30 QSA rejects FP8 KV; auto uses the BF16 model dtype.
        options.update(kv_cache_dtype="auto", engram_config={"cpu_offload": True},
                       limit_mm_per_prompt={"image": 0, "video": 0})
        if profile["nvfp4"]:
            options["quantization"] = "modelopt"
    else:
        # RTX PRO 6000 is SM120, not SM100: its sparse-MLA backend cannot
        # use the B200/GB200 FP4 indexer or deep_gemm_mega_moe.
        options.update(tokenizer_mode="deepseek_v4", block_size=256,
                       attention_config={"indexer_kv_dtype": "fp8"},
                       moe_backend="auto")
    # NVIDIA's cards do not validate DSpark on the NVFP4 release, and Qwen's
    # MTP needs an additional upstream fix. Do not silently enable either.
    return options


def validate_strategist_runtime(model_name, *, prepare_encoder=False):
    if not strategist_profile(model_name):
        return
    try:
        version = metadata.version("vllm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{model_name} requires a compatible recent vLLM build") from exc
    numbers = tuple(int(value) for value in re.findall(r"\d+", version)[:3])
    if numbers < (0, 30, 0):
        raise RuntimeError(
            f"{model_name} needs vLLM >= 0.30 with its current NVIDIA "
            f"NVFP4/SM120 kernels; installed {version}. Use a compatible "
            "recent build. No packages are installed automatically.")
    if (prepare_encoder and strategist_profile(model_name)["family"]
            == "deepseek-v4-flash-0731"):
        _deepseek_encoder(model_name)


@lru_cache(maxsize=4)
def _deepseek_encoder(model_name):
    path = Path(model_name) / "encoding" / "encoding_dsv4.py"
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(
            repo_id=DEEPSEEK_ENCODER_REPO,
            filename="encoding/encoding_dsv4.py",
            revision=DEEPSEEK_ENCODER_REVISION,
        ))
    # DeepSeek publishes this self-contained, stdlib-only encoder instead of
    # a Jinja template. Pin the official revision for repeatable prompts.
    spec = util.spec_from_file_location("ttt_deepseek_v4_encoding", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def render_strategy_prompt(tokenizer, messages, model_name, *, thinking, effort):
    profile = strategist_profile(model_name)
    effort = resolve_strategy_effort(model_name, effort)
    if profile["family"] == "deepseek-v4-flash-0731":
        return _deepseek_encoder(model_name)(
            deepcopy(messages), thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort if thinking else None,
        )
    # No generic fallback: silently dropping effort would change the policy.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(thinking), reasoning_effort=effort,
        preserve_thinking=True,
    )
