#!/bin/sh
set -eu

# Runtime-only environment belongs here. Models, GPU roles, memory/feedback,
# sampling, and optimization hyperparameters belong in the selected YAML.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Override without editing this launcher:
#   TTT_CONFIG=configs/gpu_mode_trimul.yaml sh run.sh
# Resume keeps the saved run configuration authoritative:
#   sh run.sh --resume /path/to/run
config_path="${TTT_CONFIG:-configs/erdos.yaml}"
exec python3 train_multy.py --config "$config_path" "$@"
