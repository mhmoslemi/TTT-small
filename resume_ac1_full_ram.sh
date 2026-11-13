#!/bin/sh
set -eu

ttt_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ttt_project_dir"

export AVAILABLE_GPUS="2,3,4,5"
export TTT_CONFIG="configs/ac1.yaml"
ttt_resume_dir="$ttt_project_dir/runs/ac1_Qwen3.8-27B_1107-0624"

if [ ! -f "$ttt_resume_dir/training_state.pt" ]; then
    printf '%s\n' "Missing $ttt_resume_dir/training_state.pt; run this in the server checkout containing the saved checkpoint." >&2
    exit 1
fi

# Do not silently regenerate step 3 if the server's exact sidecars are missing.
for ttt_meta in "$ttt_resume_dir"/step03/step03_group*_rollout*.meta.json; do
    ttt_base=${ttt_meta%.meta.json}
    for ttt_suffix in txt prompt.txt tokens.zlib; do
        if [ ! -f "$ttt_base.$ttt_suffix" ]; then
            printf '%s\n' "Missing recovery artifact: $ttt_base.$ttt_suffix" >&2
            exit 1
        fi
    done
done

exec sh "$ttt_project_dir/run_copy.sh" --resume "$ttt_resume_dir" --parent-selection puct "$@"
