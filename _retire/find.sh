#!/bin/sh
# Extract the top-3 rollouts per step (by reward) into a results/ directory.
#
# Run from the TTT-local directory (or anywhere):
#   bash find.sh [output_dir]
#
# Automatically finds the latest experiment in runs/.
# Handles both flat (old) and per-step-subdir (new) run layouts.
#
# Output layout:
#   <run_dir>/results/
#     step00/
#       rank1_reward2.6225_group07_rollout011.txt
#       rank1_reward2.6225_group07_rollout011.meta.json
#       ...
#       summary.txt
#     step01/
#       ...
#     all_top3.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/runs"

if [ -n "$1" ]; then
    RUN_DIR="$(cd "$1" && pwd)"
else
    # Latest run dir by modification time
    RUN_DIR=$(ls -td "$RUNS_DIR"/*/ 2>/dev/null | head -1 | sed 's|/$||')
    if [ -z "$RUN_DIR" ]; then
        echo "No run directories found in $RUNS_DIR"
        exit 1
    fi
fi

echo "Run: $RUN_DIR"

OUT_DIR="$RUN_DIR/results"
mkdir -p "$OUT_DIR"

# Detect layout: new (per-step subdirs) or old (flat files)
HAS_STEP_DIRS=0
for d in "$RUN_DIR"/step*/; do
    [ -d "$d" ] && HAS_STEP_DIRS=1 && break
done

if [ "$HAS_STEP_DIRS" = "1" ]; then
    # New layout: step00/, step01/, ... each containing group*_rollout*.meta.json
    STEPS=$(for d in "$RUN_DIR"/step*/; do
                [ -d "$d" ] && basename "$d" | sed 's/^step//'
            done | sort -u)
else
    # Old layout: flat step00_group00_rollout000.meta.json files
    STEPS=$(ls "$RUN_DIR"/step*.meta.json 2>/dev/null \
            | sed -E "s|.*/step([0-9]+)_.*|\1|" \
            | sort -u)
fi

if [ -z "$STEPS" ]; then
    echo "No steps found in $RUN_DIR"
    exit 1
fi

TOTAL_STEPS=$(echo "$STEPS" | wc -l)
echo "Found $TOTAL_STEPS step(s). Extracting top-3 per step ..."

# Header for the global CSV
echo "step,rank,reward,group,rollout,valid,msg,file" > "$OUT_DIR/all_top3.csv"

for s in $STEPS; do
    step_dir="$OUT_DIR/step$s"
    mkdir -p "$step_dir"

    if [ "$HAS_STEP_DIRS" = "1" ]; then
        META_GLOB="$RUN_DIR/step${s}/*.meta.json"
    else
        META_GLOB="$RUN_DIR/step${s}_*.meta.json"
    fi

    # shellcheck disable=SC2086
    TOP3=$(jq -nr '
        [inputs
         | select(.reward != null)
         | {
             reward: .reward,
             group: .group,
             rollout: .rollout,
             valid: .valid,
             msg: .msg,
             file: input_filename
         }]
        | sort_by(-.reward)
        | .[0:6]
        | to_entries
        | .[]
        | "\(.key+1)|\(.value.reward)|\(.value.group)|\(.value.rollout)|\(.value.valid)|\(.value.msg)|\(.value.file)"
    ' $META_GLOB 2>/dev/null)

    if [ -z "$TOP3" ]; then
        echo "  step $s: no valid rollouts"
        echo "no rollouts with non-null reward" > "$step_dir/summary.txt"
        continue
    fi

    echo "  step $s:"
    {
        echo "Top 3 rollouts for step $s"
        echo "-----------------------------"
    } > "$step_dir/summary.txt"

    printf '%s\n' "$TOP3" | while IFS='|' read -r rank reward group rollout valid msg meta_file; do
        [ -z "$rank" ] && continue

        txt_file="${meta_file%.meta.json}.txt"
        reward_fmt=$(printf "%.10f" "$reward")
        group_fmt=$(printf "%02d" "$group")
        rollout_fmt=$(printf "%03d" "$rollout")

        dest_base="rank${rank}_reward${reward_fmt}_group${group_fmt}_rollout${rollout_fmt}"

        cp "$txt_file"  "$step_dir/${dest_base}.txt"
        cp "$meta_file" "$step_dir/${dest_base}.meta.json"

        echo "    rank $rank: reward=$reward_fmt  group=$group_fmt  rollout=$rollout_fmt  valid=$valid"
        echo "rank $rank  reward=$reward  group=$group  rollout=$rollout  valid=$valid  msg=$msg" \
            >> "$step_dir/summary.txt"

        echo "$s,$rank,$reward,$group,$rollout,$valid,\"$msg\",$txt_file" >> "$OUT_DIR/all_top3.csv"
    done
done

echo ""
echo "Done. Output written to: $OUT_DIR/"
echo "Global summary CSV:      $OUT_DIR/all_top3.csv"
