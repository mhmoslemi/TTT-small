"""Launch-scoped hardware capacity, checkpoint locations and residency."""

from itertools import zip_longest
import os
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
    if profile in {"B200", "B200_full_ram"}:
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
    if name == "B200_FULL_RAM":
        return "B200_full_ram"
    if name not in {"RTX6000", "B200"}:
        raise ValueError("hardware_profile must be RTX6000, B200 or B200_full_ram")
    return name


def generation_cpu_ids(profile, gpu_count):
    """Keep evaluation off inference cores, including their SMT siblings."""
    if profile not in {"B200", "B200_full_ram"} or not hasattr(os, "sched_getaffinity"):
        return []
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError):
            # Without physical topology, do not accidentally reserve half a core.
            return []
        cores.setdefault(key, []).append(cpu)
    reserve = min(2 + 2 * int(gpu_count), len(cores) // 2)
    if len(cores) < 4 or int(gpu_count) < 1:
        return []
    sockets = {}
    for (socket, _core), threads in cores.items():
        sockets.setdefault(socket, []).append(threads)
    # Spread inference across sockets rather than trapping all engines on one.
    ordered = [threads for row in zip_longest(*sockets.values())
               for threads in row if threads is not None]
    return sorted(cpu for threads in ordered[:reserve] for cpu in threads)


def restore_hardware_settings(merged, current_yaml, saved):
    """Do not reuse an old machine's resolved auto capacities on resume."""
    selected = hardware_profile(current_yaml.get("hardware_profile", "RTX6000"))
    previous = hardware_profile(saved.get("hardware_profile", "RTX6000"))
    merged["hardware_profile"] = selected
    if selected == "B200_full_ram" and previous != selected:
        # Saved low-RAM launch flags must not defeat the requested RTX policy.
        merged["reload_coder_between_phases"] = current_yaml.get("reload_coder_between_phases", False)
        merged["strategy_vllm_persistent_workers"] = current_yaml.get("strategy_vllm_persistent_workers")
    if saved and previous != selected:
        for key, default in {
            "gen_micro_batch": "auto", "vllm_gpu_memory_utilization": "auto",
            "vllm_max_num_batched_tokens": "auto", "logprob_chunk": "auto",
        }.items():
            merged[key] = current_yaml.get(key, default)
    if selected in {"B200", "B200_full_ram"}:
        # Resolved checkpoint values must not freeze vLLM's automatic B200
        # scheduler when the current launch still requests auto.
        merged["gen_micro_batch"] = current_yaml.get("gen_micro_batch", "auto")
        merged["vllm_max_num_batched_tokens"] = current_yaml.get(
            "vllm_max_num_batched_tokens", "auto")
