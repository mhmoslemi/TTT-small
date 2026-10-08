#!/bin/sh
set -eu

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export AVAILABLE_GPUS="${AVAILABLE_GPUS:-0,1,2,3,4,5,6,7}"
# config_path="${TTT_CONFIG:-configs/circle_packing.yaml}"
config_path="${TTT_CONFIG:-configs/erdos.yaml}"
# config_path="${TTT_CONFIG:-configs/erdos-c4.yaml}"



# Runtime-only environment belongs here. Models, GPU roles,
# sampling, and optimization hyperparameters belong in the selected YAML.
# Authoritative ordered physical GPU inventory. Edit this one list (or export
# it before invoking the script); Python derives every role from it. Every
# rollout card hosts a parallel trainer replica and then rejoins generation.
# GPU-mode reserves only the last card for evaluation when a separate card
# exists. vLLM derives compatible TP groups/replicas that consume the remaining
# complete list without dropping a card.
# FlashInfer sampling can trigger runtime compilation and require nvcc. vLLM's
# native PyTorch/Triton sampler is the safe default; callers may explicitly opt
# back in with VLLM_USE_FLASHINFER_SAMPLER=1.
# Override without editing this launcher:
#   TTT_CONFIG=configs/gpu_mode_trimul.yaml sh run.sh
# Resume keeps saved experiment settings; AVAILABLE_GPUS still defines this
# launch's physical inventory:
#   sh run.sh --resume /path/to/run
# Opt into the process-per-GPU adaptive trainer. Omitting --fast keeps the
# current training implementation exactly as the default:
#   sh run.sh --fast
# Set asymmetric rank clipping as distances below/above 1, for example:
#   sh run.sh --fast --rank-clip-epsilon-low 0.1 --rank-clip-epsilon-high 0.3
# Isolate every candidate on one CPU. In this mode reward_workers from the
# selected YAML is the number of concurrent candidate processes per CPU:
#   sh run.sh --isolate-eval
# It can be combined with the fast trainer:
#   sh run.sh --fast --isolate-eval
# Skip adapter training completely while keeping rollout, evaluation, search,
# search and result/checkpoint persistence:
#   sh run.sh --no-train
# Measurement only: reuse policy forwards for full-vocabulary entropy and
# refresh entropy.jsonl / entropy.pdf each step. Strategy runs additionally
# refresh strategy_diversity.svg.
# No extra model forwards; skipped/unscored examples have missing coverage.
#   sh run.sh --measure-entropy
# Opt into hierarchical rollouts. The frozen strategist runs first, is fully
# offloaded, then the LoRA coder generates programs and trains. Without this
# flag the coder uses the direct one-stage rollout path:
#   sh run.sh --strategies
# Retry a coder response missing its required final code block once. Coder
# retries are off unless this flag is passed; strategy retries are unchanged:
#   sh run.sh --coder-retry
# An OpenAI-compatible remote strategist can be selected without loading its
# tokenizer or weights locally:
#   export DEEPSEEK_API_KEY='...'
#   sh run.sh --strategy-api --strategy-model-name deepseek-v4-pro
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# exec python3 train_multy_CVaR.py --config "$config_path" --backend hf "$@" --advantage-mode rank entropic spo-rs --fast
exec python3 train_multy_CVaR.py --config "$config_path" --backend hf --advantage-mode entropic --measure-entropy \
    --spo-rs-clip-epsilon-low 0.2 --spo-rs-clip-epsilon-high 0.38 --strategies \
    --isolate-eval --fused-long-attention "$@"
    # 
    # --coder-retry
    # 
