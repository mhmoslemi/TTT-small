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


# CUDA_VISIBLE_DEVICES=6 python train_ppo.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24


# CUDA_VISIBLE_DEVICES=7 python train_a2c.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24



# CUDA_VISIBLE_DEVICES=7 python train_reinforce.py --num-circles 26 \
#                 --num-steps 20 --groups-per-step 4 --group-size 24


CUDA_VISIBLE_DEVICES=6,7 python train_multy.py --num-circles 26 \
                --num-steps 20 --groups-per-step 4 --group-size 24

# reinforce

# gpu 6 : lr 0 
# gpu 7 : lr 1
