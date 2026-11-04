"""Launch-scoped hardware selection; never change the policy or task settings."""

import os
from pathlib import Path

from runtime.host_memory import host_memory_info, host_memory_reserve_bytes, GIB


B200_MODEL_DIR = Path("/root/nas/Mohammad/models")
B200_LOCAL_MODELS = {
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
    if selected == "B200" or previous != selected:
        for key, default in {
            "gen_micro_batch": "auto", "vllm_gpu_memory_utilization": "auto",
            "vllm_max_num_batched_tokens": "auto", "logprob_chunk": "auto",
            "vllm_staged_loading": False, "strategy_vllm_staged_loading": False,
            "strategy_vllm_persistent_workers": None,
        }.items():
            merged[key] = current_yaml.get(key, default)


def validate_hardware_profile(cfg, gpu_ids, memory):
    if cfg.get("hardware_profile") != "B200":
        return
    if any(gpu not in memory or "B200" not in memory[gpu].name.upper()
           for gpu in gpu_ids):
        raise ValueError("hardware_profile=B200 requires nvidia-smi to report B200 for every selected GPU")
    info = host_memory_info()
    if not info:
        raise RuntimeError("B200 profile requires readable host RAM limits for memory admission")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    print(f"[hardware] B200: {len(gpu_ids)} GPUs, "
          f"{min(memory[gpu].total_gib for gpu in gpu_ids):.1f} GiB/GPU; "
          f"{cpus} available CPUs; host RAM={info['total'] / GIB:.1f} GiB, "
          f"available={info['available'] / GIB:.1f} GiB, "
          f"reserve={host_memory_reserve_bytes(info) / GIB:.1f} GiB; live shared RAM admission")
