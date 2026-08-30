"""GPU inventory, role allocation, and conservative memory auto-tuning.

This module intentionally has no torch/Transformers imports.  Configuration is
resolved before the trainer narrows CUDA_VISIBLE_DEVICES, so physical ids still
mean what the operator wrote in AVAILABLE_GPUS.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Dict, Iterable, List, Optional


GPU_PROBLEM_NAMES = {
    "gpu_mode", "kernel", "kernel_engineering", "trimul",
    "mla_decode_nvidia", "mla",
}


def parse_gpu_ids(value, *, field: str = "AVAILABLE_GPUS") -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    try:
        ids = [int(part) for part in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be comma-separated physical GPU ids: "
                         f"{value!r}") from exc
    if not ids:
        raise ValueError(f"{field} must contain at least one physical GPU id")
    if any(gpu_id < 0 for gpu_id in ids):
        raise ValueError(f"{field} ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field} must not contain duplicates")
    return ids


@dataclass(frozen=True)
class GPURoles:
    available: List[int]
    training: int
    generation: List[int]
    evaluation: Optional[int]
    gpu_problem: bool
    sequential_generation: bool
    evaluation_shares_generation: bool


def allocate_gpu_roles(available: Iterable[int], problem: str) -> GPURoles:
    """Apply the role table defined by the launcher contract."""
    ids = parse_gpu_ids(list(available))
    gpu_problem = str(problem or "").strip().lower() in GPU_PROBLEM_NAMES
    training = ids[0]

    if not gpu_problem:
        generation = [training] if len(ids) == 1 else ids[1:]
        evaluation = None
    elif len(ids) == 1:
        generation = [training]
        evaluation = training
    elif len(ids) == 2:
        generation = [training]
        evaluation = ids[-1]
    else:
        generation = ids[1:-1]
        evaluation = ids[-1]

    return GPURoles(
        available=ids,
        training=training,
        generation=generation,
        evaluation=evaluation,
        gpu_problem=gpu_problem,
        sequential_generation=training in generation,
        evaluation_shares_generation=(evaluation is not None
                                      and evaluation in generation),
    )


@dataclass(frozen=True)
class GPUMemory:
    physical_id: int
    name: str
    total_gib: float
    free_gib: float


def query_gpu_memory() -> Dict[int, GPUMemory]:
    """Return physical-device memory from nvidia-smi, or {} when unavailable."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True,
                              timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}

    out: Dict[int, GPUMemory] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            physical_id = int(parts[0])
            total_gib = float(parts[2]) / 1024.0
            free_gib = float(parts[3]) / 1024.0
        except ValueError:
            continue
        out[physical_id] = GPUMemory(
            physical_id=physical_id,
            name=parts[1],
            total_gib=total_gib,
            free_gib=free_gib,
        )
    return out


def _auto(value) -> bool:
    return value is None or str(value).strip().lower() in ("", "auto")


def _model_size_billions(model_name: str) -> float:
    name = str(model_name or "").lower()
    # Prefer a size next to the conventional B suffix.  This handles 8B, 32B,
    # 30B-A3B and 120B without coupling to one repository naming scheme.
    matches = re.findall(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:$|[-_/])", name)
    return float(matches[0]) if matches else 8.0


def _native_weight_bytes(model_name: str) -> float:
    # GPT-OSS checkpoints are natively MXFP4.  Passing the training QLoRA flag
    # through as vLLM BitsAndBytes quantization is both redundant and fragile.
    return 0.55 if "gpt-oss" in str(model_name or "").lower() else 2.0


def _kv_bytes_per_token(model_name: str) -> int:
    name = str(model_name or "").lower()
    if "gpt-oss" in name:
        # 36 layers, 8 KV heads, head_dim 64, BF16 K+V. Sliding attention makes
        # this estimate conservative for most layers.
        return 73_728
    if "120b" in name or "32b" in name or "30b" in name:
        return 262_144
    return 147_456


def resolve_memory_settings(cfg: dict, roles: GPURoles,
                            memory: Dict[int, GPUMemory]) -> List[str]:
    """Resolve auto knobs in-place and return human-readable decisions.

    These are admission limits, not promises that arbitrary generated programs
    cannot allocate too much memory.  vLLM still owns its paged KV cache and HF
    retains its OOM-halving retry path.
    """
    notes: List[str] = []
    using_vllm = str(cfg.get("generation_backend", "hf")).lower() == "vllm"
    selected = [memory[g] for g in roles.generation if g in memory]
    min_total = min((item.total_gib for item in selected), default=80.0)
    min_free = min((item.free_gib for item in selected), default=min_total)

    raw_util = cfg.get("vllm_gpu_memory_utilization", "auto")
    if _auto(raw_util):
        # Leave at least 6 GiB outside vLLM and never claim more than 90% of a
        # card.  Account for memory already occupied before startup as well.
        util = min(0.90, (min_free - 6.0) / max(min_total, 1.0))
        util = max(0.50, util)
        cfg["vllm_gpu_memory_utilization"] = round(util, 3)
        if using_vllm:
            notes.append(f"vLLM memory utilization={util:.3f} "
                         f"(minimum generation GPU free={min_free:.1f}/"
                         f"{min_total:.1f} GiB)")
    else:
        util = float(raw_util)
        if not 0.0 < util <= 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be auto or in (0, 1]")
        cfg["vllm_gpu_memory_utilization"] = util

    max_len = max(1, int(cfg.get("max_seq_length", 4096)))
    if (bool(cfg.get("memory", False))
            and bool(cfg.get("memory_grant_context", True))):
        max_len += max(0, int(cfg.get("memory_token_budget", 0) or 0))
    size_b = _model_size_billions(cfg.get("model_name", ""))
    tp = max(1, len(roles.generation))
    quant = str(cfg.get("vllm_quantization", "") or "").lower()
    weight_bytes = (0.65 if quant not in ("", "auto", "none")
                    else _native_weight_bytes(cfg.get("model_name", "")))
    weight_gib_per_gpu = size_b * weight_bytes / tp * (1e9 / (1024.0 ** 3))
    kv_gib_per_seq_gpu = (
        _kv_bytes_per_token(cfg.get("model_name", "")) * max_len
        / tp / (1024.0 ** 3)
    )
    kv_budget = max(0.0, min_total * float(cfg["vllm_gpu_memory_utilization"])
                    - weight_gib_per_gpu - 4.0)
    estimated_sequences = int(kv_budget / max(kv_gib_per_seq_gpu, 0.01))
    if using_vllm and estimated_sequences < 1:
        required = weight_gib_per_gpu + 4.0 + kv_gib_per_seq_gpu
        raise ValueError(
            f"{cfg.get('model_name')} is estimated to require at least "
            f"{required:.1f} GiB per generation GPU for one "
            f"{max_len}-token request, but the vLLM budget is "
            f"{min_total * float(cfg['vllm_gpu_memory_utilization']):.1f} GiB. "
            "Use a checkpoint-native/explicit vllm_quantization, reduce "
            "max_seq_length, choose generation_backend=hf, or provide a "
            "compatible larger tensor-parallel group.")

    if _auto(cfg.get("gen_micro_batch", "auto")):
        # Keep CUDA graph/admission metadata bounded even when short responses
        # would permit hundreds of sequences.  One is retained as a safe floor;
        # vLLM will fail clearly at model load if the weights themselves do not fit.
        cap = max(1, min(32, estimated_sequences))
        if not using_vllm:
            cap = min(cap, 8 if size_b <= 10 else (2 if size_b <= 40 else 1))
        cfg["gen_micro_batch"] = cap
        notes.append(f"generation max sequences={cap} "
                     f"(estimated full-context KV capacity={max(0, estimated_sequences)})")
    else:
        cfg["gen_micro_batch"] = int(cfg.get("gen_micro_batch") or 0)
        if cfg["gen_micro_batch"] < 0:
            raise ValueError("gen_micro_batch must be auto or >= 0")

    if _auto(cfg.get("vllm_max_num_batched_tokens", "auto")):
        # Chunked prefill keeps a long prompt from monopolizing a giant
        # temporary batch.  This is scheduling work per iteration, not the KV
        # capacity or maximum request length.
        batched = min(max_len, 8192 if min_total >= 60 else 4096)
        cfg["vllm_max_num_batched_tokens"] = max(2048, batched)
        if using_vllm:
            notes.append("vLLM max batched tokens="
                         f"{cfg['vllm_max_num_batched_tokens']}")
    else:
        cfg["vllm_max_num_batched_tokens"] = int(
            cfg.get("vllm_max_num_batched_tokens") or 0)

    if _auto(cfg.get("logprob_chunk", "auto")):
        # Caps the float32 (tokens x vocabulary) log_softmax spike.  The model
        # forward remains exact and the feedback teacher uses the same cap.
        cfg["logprob_chunk"] = 256 if min_total >= 60 else 128
        notes.append(f"training/feedback logprob chunk={cfg['logprob_chunk']}")
    else:
        cfg["logprob_chunk"] = int(cfg.get("logprob_chunk") or 0)

    return notes


def validate_selected_gpus(roles: GPURoles,
                           memory: Dict[int, GPUMemory]) -> None:
    if not memory:
        return
    missing = [gpu_id for gpu_id in roles.available if gpu_id not in memory]
    if missing:
        raise ValueError(
            f"AVAILABLE_GPUS contains device(s) not reported by nvidia-smi: {missing}")


def validate_attention_heads(num_attention_heads: Optional[int], tp: int,
                             model_name: str) -> None:
    if not num_attention_heads or tp <= 1:
        return
    if int(num_attention_heads) % int(tp):
        raise ValueError(
            f"{model_name} has {num_attention_heads} attention heads, which "
            f"cannot be divided across vLLM tensor_parallel_size={tp}. "
            "AVAILABLE_GPUS is authoritative, so list a compatible subset "
            "(the first id remains training and, for gpu_mode, the last remains "
            "evaluation).")


def known_attention_heads(model_name: str) -> Optional[int]:
    """Fast preflight for the model families explicitly supported here."""
    name = str(model_name or "").lower()
    if "qwen3-coder-30b" in name:
        return 32
    if "qwen3-32b" in name:
        return 64
    if "qwen3-8b" in name:
        return 32
    if "gpt-oss-120b" in name:
        return 64
    return None
