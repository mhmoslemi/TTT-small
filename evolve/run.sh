#!/usr/bin/env bash
#
# EVOLVE — launcher.
#
# Everything passed from here lands in the CLI layer, which outranks
# FRAMEWORK_OVERRIDES in config.py, which outranks examples/<name>/config.yaml,
# which outranks configs/base.yaml, which outranks the dataclass defaults.
#
#   bash run.sh                          # circle packing, all defaults
#   STEPS=5 N_SELECT=2 bash run.sh       # env-var override
#   DRY_RUN=1 bash run.sh                # resolve + print provenance, do not run
#   bash run.sh --alpha 0.8 --beta 4.0   # extra flags pass straight through
#
# Every variable below is `${VAR:-}`: unset means "don't pass the flag at all",
# so the lower layers keep control. Only what you actually set is overridden.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ----------------------------------------------------------------------
# Which example, and on which GPUs
# ----------------------------------------------------------------------
EXAMPLE="${EXAMPLE:-circle_packing}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ----------------------------------------------------------------------
# Overrides. Unset = inherit from config.py / YAML.
# ----------------------------------------------------------------------
# model / generation
MODEL_NAME="${MODEL_NAME:-}"
BACKEND="${BACKEND:-}"                     # auto | unsloth | hf
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
NUM_GPUS="${NUM_GPUS:-}"
GPU_IDS="${GPU_IDS:-}"

# §2.1 D-PUCT
N_SELECT="${N_SELECT:-}"                   # n
K_CHILDREN="${K_CHILDREN:-}"               # k
C_PUCT="${C_PUCT:-}"                       # c
ALPHA="${ALPHA:-}"                         # α  rank vs Elo
LAMBDA_VIRTUAL="${LAMBDA_VIRTUAL:-}"       # λ  virtual-child optimism
TAU="${TAU:-}"                             # τ  prior temperature

# Elo debate
ELO_ENABLED="${ELO_ENABLED:-}"
ELO_K="${ELO_K:-}"
ELO_SCALE="${ELO_SCALE:-}"                 # 1.0 = paper, 400.0 = classic Elo
ELO_TOP_K="${ELO_TOP_K:-}"

# §2.2 memory
MEMORY_ENABLED="${MEMORY_ENABLED:-}"
LESSONS_PER_GROUP="${LESSONS_PER_GROUP:-}" # L
TOP_M="${TOP_M:-}"                         # m

# §2.3 test-time RL
RL_ENABLED="${RL_ENABLED:-}"
BETA="${BETA:-}"                           # β  reward tilt
LAMBDA_FEEDBACK="${LAMBDA_FEEDBACK:-}"     # λ_f
CLIP_EPSILON="${CLIP_EPSILON:-}"           # ε
KL_COEF="${KL_COEF:-}"                     # η_KL
LR="${LR:-}"

# run
STEPS="${STEPS:-}"
SEED="${SEED:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-}"

# Example-specific knobs. These live in `example.params`, an open namespace the
# framework passes through untouched — reached with --set, never a named flag,
# so adding a problem never means editing the framework's schema.
NUM_CIRCLES="${NUM_CIRCLES:-}"
TARGET="${TARGET:-}"

# ----------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------
ARGS=(--example "$EXAMPLE")

add() {  # add <flag> <value> -- skipped entirely when value is empty
  [[ -n "${2:-}" ]] && ARGS+=("$1" "$2") || true
}
set_kv() {  # set_kv <dotted.path> <value> -- for open namespaces
  [[ -n "${2:-}" ]] && ARGS+=(--set "$1=$2") || true
}

add --model-name       "$MODEL_NAME"
add --backend          "$BACKEND"
add --max-seq-length   "$MAX_SEQ_LENGTH"
add --max-new-tokens   "$MAX_NEW_TOKENS"
add --temperature      "$TEMPERATURE"
add --num-gpus         "$NUM_GPUS"
add --gpu-ids          "$GPU_IDS"

add --n-select         "$N_SELECT"
add --k-children       "$K_CHILDREN"
add --c-puct           "$C_PUCT"
add --alpha            "$ALPHA"
add --lambda-virtual   "$LAMBDA_VIRTUAL"
add --tau              "$TAU"

add --elo-enabled      "$ELO_ENABLED"
add --elo-k            "$ELO_K"
add --elo-scale        "$ELO_SCALE"
add --elo-top-k        "$ELO_TOP_K"

add --memory-enabled   "$MEMORY_ENABLED"
add --lessons-per-group "$LESSONS_PER_GROUP"
add --top-m            "$TOP_M"

add --rl-enabled       "$RL_ENABLED"
add --beta             "$BETA"
add --lambda-feedback  "$LAMBDA_FEEDBACK"
add --clip-epsilon     "$CLIP_EPSILON"
add --kl-coef          "$KL_COEF"
add --lr               "$LR"

add --steps            "$STEPS"
add --seed             "$SEED"
add --run-name         "$RUN_NAME"
add --output-root      "$OUTPUT_ROOT"
add --verifier-timeout "$VERIFIER_TIMEOUT"

set_kv example.params.num_circles "$NUM_CIRCLES"
set_kv example.params.target      "$TARGET"

[[ "${DRY_RUN:-0}" == "1" ]] && ARGS+=(--print-config)

# "$@" goes last so ad-hoc flags on the command line beat the variables above.
echo "+ python main.py ${ARGS[*]} $*"
exec python main.py "${ARGS[@]}" "$@"


# ======================================================================
# Worked examples
# ======================================================================
#
# n=26, single GPU, short smoke test:
#   STEPS=3 N_SELECT=2 K_CHILDREN=2 MAX_NEW_TOKENS=1500 \
#   MODEL_NAME=LiquidAI/LFM2.5-350M bash run.sh
#
# n=32 instead of the example's n=26 (open namespace, no schema change):
#   NUM_CIRCLES=32 TARGET=2.940 bash run.sh
#
# Paper-scale on two GPUs:
#   CUDA_VISIBLE_DEVICES=6,7 NUM_GPUS=2 GPU_IDS=6,7 \
#   STEPS=50 N_SELECT=8 K_CHILDREN=8 bash run.sh
#
# Ablations:
#   RL_ENABLED=false bash run.sh                       # search + memory only
#   MEMORY_ENABLED=false bash run.sh                   # no lesson bank
#   ELO_ENABLED=false ALPHA=1.0 bash run.sh            # rank signal only (Eq. 3)
#   ALPHA=0.0 bash run.sh                              # Elo signal only
#   BETA=20 bash run.sh                                # near-max-seeking RL
#   LAMBDA_VIRTUAL=0 bash run.sh                       # no optimism on new siblings
#
# Confirm precedence without launching anything:
#   DRY_RUN=1 ALPHA=0.9 bash run.sh
#   -> search.alpha  0.9  cli
#      search.tau    1.0  yaml:base
