


MODEL_NAME=Qwen/Qwen3.5-4B \
    STEPS=10 N_SELECT=8 K_CHILDREN=4 \
    ELO_ENABLED=false ALPHA=1.0 \
    NUM_CIRCLES=26 bash run.sh --max-new-tokens 8000 --think-budget 4500
