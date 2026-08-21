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


# #!/bin/sh
# export HF_HUB_OFFLINE=1  # gpt-oss MXFP4 kernel is cached locally; HF Hub is unreachable from this box
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

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multy.py \
#     --problem gpu_mode --problem-type trimul --config configs/gpu_mode_trimul.yaml \
#     --num-steps 100 --groups-per-step 6 --group-size 3 \
#     --gpu-type H200 --model-name /mnt/storage/mohammad/models/gpt-oss-120b \
#     --num-gpus 2 --gpu-ids 1,2 --kernel-gpu-id 3







# --deterministic --seed 42
# --no-deterministic
python train_multy.py --problem erdos --config configs/erdos.yaml \
    --num-steps 30 \
    --groups-per-step 8 --group-size 2 \
    --max-groups-per-step 8 --max-group-size 64 \
    --growth-force-step 10 \
    --growth-valid-yield 0.7 --growth-distinct-min 2 --growth-factor 2.0
 


# problem
# reinforce

# gpu 6 : lr 0 
# gpu 7 : lr 1


# srun --jobid=4918491 --overlap watch -n 0.1 nvidia-smi