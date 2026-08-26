<!-- # TTT-Discover — Local 



## Model

**Works with ANY Hugging Face causal language model.** The default is
`Qwen/Qwen3-8B`; change it in `train.py` (`Config.model_name`) or pass it
on the command line:

```bash
python train.py --model-name "Qwen/Qwen3-8B"
python train.py --model-name "LiquidAI/LFM2.5-1.2B-Thinking"
python train.py --model-name "meta-llama/Llama-3.1-8B-Instruct"
```

The project was initially debugged and validated on **LiquidAI LFM2.5-350M**
(small enough for any GPU) and then scaled up to 8B models.

## Backends

Two interchangeable training/inference backends:

| Backend | How to select | Notes |
|---------|---------------|-------|
| **Unsloth** | `--backend unsloth` | Faster generation, lower VRAM. Recommended when available. |
| **HF + PEFT** | `--backend hf` | Plain `transformers` + `peft`. Works with every architecture. |
| **auto** (default) | `--backend auto` | Tries Unsloth first; silently falls back to HF+PEFT if Unsloth is unavailable or can't load the model. |

The auto-fallback means you never need to change code based on your environment — it just works.

See the [Unsloth install guide](https://github.com/unslothai/unsloth#-install)
for CUDA-version-specific wheels; if installation fails, the HF backend takes
over automatically.

For fine-tuning ideas and best practices (especially for thinking/reasoning
models), see:
- [LiquidAI LFM2.5-1.2B-Thinking fine-tuning guide](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#%F0%9F%94%A7-fine-tuning)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — HuggingFace's library for GRPO, REINFORCE, PPO, and other RL fine-tuning recipes that informed this implementation




## Output logging

Every single rollout is saved to disk under `runs/`. The directory name
encodes the key hyperparameters and timestamp:

```
runs/n10_target1.6294_steps50_g8x64_lr1e-05_T1.1_klc0.1_model-Qwen_Qwen3-8B_20260528-010417/
  config.json                          ← full hyperparameter dump
  step03_group02_rollout017.txt        ← raw model response (the generated Python program)
  step03_group02_rollout017.meta.json  ← reward, valid, parsed, ran, error msg, advantage, beta, ...
  ...
  final.summary.json                   ← best value, step, code
  best_code.py                         ← best packing found
```

The `.meta.json` files include:
- `reward` — sum of radii (0.0 if invalid)
- `valid` — whether the packing passed geometric validation
- `parsed` / `ran` — whether code extraction and execution succeeded
- `msg` — error message if anything failed
- `sandbox_stdout` — first 2000 chars of execution output
- `advantage`, `beta` — training signal for this rollout



## Installation

```bash
pip install -r requirements.txt
```

For Unsloth (optional but recommended), follow the
[official install guide](https://github.com/unslothai/unsloth#-install)
matching your CUDA version. If it fails, the script falls back to plain
transformers + PEFT automatically.

> [!CAUTION]
> **<font color="red">After setting up the Unsloth environment, you MUST install the remaining dependencies inside that same environment using `uv pip`:</font>**
>
> ```bash
> uv pip install -r requirements.txt
> ```
>
> **<font color="red">The LLM-generated Python programs are executed locally in a subprocess. If the packages (`numpy`, `scipy`, etc.) are not present in the active environment, every generated program will fail at runtime and all rewards will be zero.</font>**

## Usage

```bash
# Paper-scale defaults (50 steps × 8 groups × 64 rollouts)
python train.py

# Change model
python train.py --model-name "LiquidAI/LFM2.5-350M"

# n=10 problem (easier, good for validating the training loop)
python train.py --num-circles 10 --target 1.6294 --temperature 1.1

# n=26 paper problem
python train.py --num-circles 26 --target 2.636

# Force plain HF backend (useful if Unsloth is broken on your system)
python train.py --backend hf

```



## Files

| File | Purpose |
|------|---------|
| `train.py` | Main entry point — config, CLI, training loop |
| `model_backend.py` | Unsloth ↔ HF+PEFT backends with auto-fallback |
| `advantage.py` | Entropic adaptive-beta advantage via KL-budget bisection |
| `sampler.py` | PUCT archive + state struct |
| `reward.py` | Validator + reward (sum of radii) |
| `prompts.py` | Chat-format prompt builder |
| `sandbox.py` | Subprocess code execution with timeout (no Ray) |
| `experiment_io.py` | Per-run directory creation and rollout logging |
 -->
