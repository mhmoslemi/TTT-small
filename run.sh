#!/bin/sh
set -eu

# Runtime-only environment belongs here. Models, GPU roles, memory/feedback,
# sampling, and optimization hyperparameters belong in the selected YAML.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Authoritative ordered physical GPU inventory. Edit this one list (or export
# it before invoking the script); Python derives every role from it. The first
# card always trains. GPU-mode reserves the last card for evaluation when a
# separate card exists.
export AVAILABLE_GPUS="${AVAILABLE_GPUS:-0}"

# FlashInfer sampling can trigger runtime compilation and require nvcc. vLLM's
# native PyTorch/Triton sampler is the safe default; callers may explicitly opt
# back in with VLLM_USE_FLASHINFER_SAMPLER=1.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Override without editing this launcher:
#   TTT_CONFIG=configs/gpu_mode_trimul.yaml sh run.sh
# Resume keeps saved experiment settings; AVAILABLE_GPUS still defines this
# launch's physical inventory:
#   sh run.sh --resume /path/to/run
config_path="${TTT_CONFIG:-configs/erdos.yaml}"
exec python3 train_multy.py --config "$config_path" "$@"
