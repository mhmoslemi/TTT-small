# # python train.py --num-circles 10 --target 1.6294 \
     # python train.py --num-circles 2 \
#                 --num-steps 5 \
#                 --groups-per-step 2 \
#                 --group-size 2 \
#                 --max-new-tokens 500 \
#                 --temperature 1.1 \
#                 --max-seq-length 40000 \
#                 --model-name LiquidAI/LFM2.5-350M
#             #    --model-name LiquidAI/LFM2.5-1.2B-Base \
# # #                --model-name LiquidAI/LFM2.5-350M


# # python train.py --num-circles 10 --target 1.6294 \ppo
# CUDA_VISIBLE_DEVICES=6 python train_multy.py --num-circles 26 \
#                 --num-steps 60 \
#                 --groups-per-step 8 \
#                 --group-size 64

#                #  --num-steps 60 \
#                #  --groups-per-step 8 \
#                #  --group-size 64 

#                 # \
#                 # --max-new-tokens 6700 \
#                 # --temperature 1 
#                 # \
#                 # --max-seq-length 40000 \
#                 # --model-name openai/gpt-oss-120b
#             #    --model-name LiquidAI/LFM2.5-1.2B-Base \
# #                --model-name LiquidAI/LFM2.5-350M



# CUDA_VISIBLE_DEVICES=6 python train_ppo.py --num-circles 26 \cd ../
#                 --num-steps 20 --groups-per-step 4 --group-size 24


# CUDA_VISIBLE_DEVICES=7 python train_a2c.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24



# CUDA_VISIBLE_DEVICES=7 python train_reinforce.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24


# CUDA_VISIBLE_DEVICES=6,7 python train_multy.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_multy.py --num-circles 26 \
# CUDA_VISIBLE_DEVICES=0,1 python train_multy.py --num-circles 26 \


# python train_multy.py --num-circles 26 \
#                 --num-steps 50 --groups-per-step 4 --group-size 12 
               #  --num-steps 50 --groups-per-step 8 --group-size 64 \
               #  --model-name Qwen/Qwen3-8B
               #  --model-name /mnt/storage/mohammad/models/Qwen3-8B



# python train_multy.py --problem gpu_mode --problem-type trimul --config configs/gpu_mode_trimul.yaml \
#     --num-steps 30 --groups-per-step 4 --group-size 10 --gpu-type H100

# python train_multy.py --problem gpu_mode --problem-type trimul \
#     --num-steps 30 --groups-per-step 4 --group-size 12 --gpu-type L40S


# python train_multy.py --problem erdos --num-steps 30 --groups-per-step 5 --group-size 15 --config configs/erdos.yaml


#!/bin/sh
export HF_HUB_OFFLINE=1  # gpt-oss MXFP4 kernel is cached locally; HF Hub is unreachable from this box
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# # trimul kernel search on a 4x H200 box.
# # GPU layout:
# #   GPU 0  -> training (Unsloth), heaviest card by design
# #   GPU 1  -> generation worker
# #   GPU 2  -> generation worker
# #   GPU 3  -> kernel benchmark, exclusive (must not appear in --gpu-ids)
# #
# # Note: GPU 0 sitting higher than 1/2 is expected. Training holds gradients,
# # optimizer state, and activations on top of the same weights the gen workers
# # hold; inference does not. It is not a leak and no flag rebalances it.




# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multy.py --deterministic --seed 42 \
#     --problem gpu_mode --problem-type trimul --config configs/gpu_mode_trimul.yaml \
#     --num-steps 100 --groups-per-step 8 --group-size 50 \
#     --gpu-type H200 --model-name /mnt/storage/mohammad/models/gpt-oss-120b \
#     --num-gpus 2 --gpu-ids 1,2 --kernel-gpu-id 3 \
#     --max-groups-per-step 8 --max-group-size 50 \
#     --gen-micro-batch 5  --logprob-chunk 128

# #   Qwen3-Coder-Next-FP8

# #   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_multy.py --deterministic --seed 42 \
# #     --problem gpu_mode --problem-type trimul --config configs/gpu_mode_trimul.yaml \
# #     --num-steps 100 --groups-per-step 8 --group-size 50 \
# #     --gpu-type H200 --model-name /mnt/storage/mohammad/models/gpt-oss-120b \
# #     --num-gpus 3 --gpu-ids 1,2,3 --kernel-gpu-id 4 \
# d#     --max-groups-per-step 8 --max-group-size 64 \
# #     --gen-micro-batch 5  --logprob-chunk 64



    



# # --deterministic --seed 42
# # --no-deterministic
# Fresh run:
#   sh run.sh
# Resume in-place from the next incomplete step:
#   sh run.sh --resume /path/to/runs/erdos_Qwen3-8B_0824-1000
# On resume, config.json supplies the original defaults. Extra CLI flags still
# win, e.g. `sh run.sh --resume RUN_DIR --num-steps 150`.
resume_run=0
for arg in "$@"; do
    case "$arg" in
        --resume|--resume-from|--resume=*|--resume-from=*) resume_run=1 ;;
    esac
done


# if [ "$resume_run" -eq 1 ]; then
#     CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py "$@"
# else
#     CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py --problem erdos --config configs/erdos.yaml \
#         --num-steps 12 --deterministic --seed 42 \
#         --groups-per-step 4 --group-size 16 \
#         --max-groups-per-step 4 --max-group-size 16 \
#         --growth-force-step 3 --num-gpus 3 --gpu-ids 4,5,6 --no-memory --no-feedback \
#         --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0 \
#         --model-name /mnt/storage/mohammad/models/Qwen3-8B "$@"
# fi
 


# if [ "$resume_run" -eq 1 ]; then
#     CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py "$@"
# else
#     CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py --problem erdos --config configs/erdos.yaml \
#         --num-steps 12 --deterministic --seed 42 \
#         --groups-per-step 4 --group-size 16 \
#         --max-groups-per-step 4 --max-group-size 16 \
#         --growth-force-step 3 --num-gpus 3 --gpu-ids 4,5,6 \
#         --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0 --memory --no-feedback \
#         --model-name /mnt/storage/mohammad/models/Qwen3-8B "$@"
# fi








# if [ "$resume_run" -eq 1 ]; then
#     CUDA_VISIBLE_DEVICES=6 python train_multy.py "$@"
# else
#     CUDA_VISIBLE_DEVICES=6 python train_multy.py --problem erdos --config configs/erdos.yaml \
#         --num-steps 12 --deterministic --seed 42 \
#         --groups-per-step 4 --group-size 16 \
#         --max-groups-per-step 4 --max-group-size 16 \
#         --growth-force-step 3 --num-gpus 1 --gpu-ids 6 \
#         --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0 --no-memory --feedback --feedback-max-per-step 16 --feedback-max-per-signature 4 \
#         --model-name /mnt/storage/mohammad/models/Qwen3-8B "$@"
# fi
 


# if [ "$resume_run" -eq 1 ]; then
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_multy.py "$@"
# else
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_multy.py --problem erdos --config configs/erdos.yaml \
#         --num-steps 22 --deterministic --seed 42 \
#         --groups-per-step 5 --group-size 16 \
#         --growth-force-step 3 --gpu-ids 0,1,2,3,4 \
#         --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0 --memory --feedback \
#         --model-name /mnt/storage/mohammad/models/Qwen3-8B "$@"
# fi


if [ "$resume_run" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py \
        --backend vllm --gpu-ids 1,2,3 "$@"
else
    CUDA_VISIBLE_DEVICES=4,5,6 python train_multy.py --problem erdos --config configs/erdos.yaml \
        --backend vllm \
        --num-steps 20 --deterministic --seed 42 \
        --groups-per-step 8 --group-size 64 \
        --growth-force-step 3 --gpu-ids 5,6 \
        --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0 --memory --feedback --thinking --temperature 0.6 --top-p 0.95 \
        --model-name /mnt/storage/mohammad/models/Qwen3-8B "$@"
fi
 


#  --thinking --temperature 0.6 --top-p 0.95