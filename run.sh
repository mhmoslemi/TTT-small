#!/bin/sh
set -eu

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export AVAILABLE_GPUS="${AVAILABLE_GPUS:-0,1,2}"
config_path="${TTT_CONFIG:-configs/erdos.yaml}"
memory_version="${MEMORY_VERSION:-}"



# Runtime-only environment belongs here. Models, GPU roles, memory/feedback,
# sampling, and optimization hyperparameters belong in the selected YAML.
# Authoritative ordered physical GPU inventory. Edit this one list (or export
# it before invoking the script); Python derives every role from it. The first
# card trains and also rejoins rollout generation. GPU-mode reserves only the
# last card for evaluation when a separate card exists. vLLM derives compatible
# TP groups/replicas that consume this complete list without dropping a card.
# FlashInfer sampling can trigger runtime compilation and require nvcc. vLLM's
# native PyTorch/Triton sampler is the safe default; callers may explicitly opt
# back in with VLLM_USE_FLASHINFER_SAMPLER=1.
# Override without editing this launcher:
#   TTT_CONFIG=configs/gpu_mode_trimul.yaml sh run.sh
# Select the memory implementation explicitly when desired. If omitted, the
# selected YAML (all checked-in presets currently say V1) remains authoritative:
#   sh run.sh --memory-version V2
# The short positional spelling is also accepted:
#   sh run.sh V2
# or:
#   MEMORY_VERSION=V2 sh run.sh
# Resume keeps saved experiment settings; AVAILABLE_GPUS still defines this
# launch's physical inventory:
#   sh run.sh --resume /path/to/run
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
case "${1:-}" in
    --memory-version)
        if [ "$#" -lt 2 ]; then
            echo "--memory-version requires V1 or V2" >&2
            exit 2
        fi
        memory_version="$2"
        shift 2
        ;;
    --memory-version=*)
        memory_version="${1#*=}"
        shift
        ;;
    V1|v1|V2|v2)
        memory_version="$1"
        shift
        ;;
esac
case "$memory_version" in
    "") ;;
    V1|v1) set -- --memory-version V1 "$@" ;;
    V2|v2)
        # V2 is a complete causal preset, including for YAMLs that still carry
        # legacy V1 memory settings. Explicit arguments in "$@" remain last and
        # may refine the preset; Python rejects combinations that break V2's
        # identification invariants.
        set -- \
            --memory-version V2 \
            --memory-lookup-mode select \
            --memory-lookup-max-select 1 \
            --memory-arm-control-fraction 0.2 \
            --memory-arm-explore-fraction 0.2 \
            --memory-arm-max-lessons 1 \
            --memory-outcome-credit \
            --memory-no-text-reinforce \
            "$@"
        ;;
    *)
        echo "MEMORY_VERSION must be V1 or V2 (got: $memory_version)" >&2
        exit 2
        ;;
esac

exec python3 train_multy.py --config "$config_path" "$@"
