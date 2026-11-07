"""Launch-scoped hardware selection; never change the policy or task settings."""

from pathlib import Path


B200_MODEL_DIR = Path("/root/nas/Mohammad/models")
B200_LOCAL_MODELS = {
    "fastino/GLiDE-9B": "GLiDE-9B",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "Qwen/Qwen3.8-27B": "Qwen3.8-27B",
    "nvidia/Qwen3.8-Flash-Next-NVFP4": "Qwen3.8-Flash-Next-NVFP4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    "nvidia/DeepSeek-V4-Flash-0731-NVFP4": "DeepSeek-V4-Flash-0731-NVFP4",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
}


def local_checkpoint(model_name, profile):
    if profile == "B200":
        for model_id, folder in B200_LOCAL_MODELS.items():
            if str(model_name).lower() == model_id.lower():
                path = B200_MODEL_DIR / folder
                if path.is_dir():
                    print(f"[hardware] local checkpoint: {path}")
                    return str(path)
    return model_name


def checkpoint_identity(model_name):
    """Retain identity-based model behavior after resolving the NAS path."""
    for model_id, folder in B200_LOCAL_MODELS.items():
        if str(model_name).rstrip("/") == str(B200_MODEL_DIR / folder):
            return model_id
    return model_name


def hardware_profile(value):
    name = str(value).strip().upper()
    if name not in {"RTX6000", "B200"}:
        raise ValueError("hardware_profile must be RTX6000 or B200")
    return name


def restore_hardware_settings(merged, current_yaml, saved):
    """Do not reuse an old machine's resolved auto capacities on resume."""
    selected = hardware_profile(current_yaml.get("hardware_profile", "RTX6000"))
    previous = hardware_profile(saved.get("hardware_profile", "RTX6000"))
    merged["hardware_profile"] = selected
    if saved and previous != selected:
        for key, default in {
            "gen_micro_batch": "auto", "vllm_gpu_memory_utilization": "auto",
            "vllm_max_num_batched_tokens": "auto", "logprob_chunk": "auto",
        }.items():
            merged[key] = current_yaml.get(key, default)
