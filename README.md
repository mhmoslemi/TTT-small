# TTT-small

## Directory guide

| Directory | Purpose |
|---|---|
| `models/` | Model loading, coder/strategist profiles and model-specific training kernels. |
| `training/` | RL objectives, trainers, parallel workers and exact long-rollout recovery. |
| `generation/` | Rollout workers, strategist prompts, API generation and output retries. |
| `search/` | Parent selection, candidate archives and rollout-budget allocation. |
| `evaluation/` | Python-code extraction and timed subprocess execution. |
| `problems/` | Problem definitions, task prompts, seed constructions and scoring rules. |
| `runtime/` | CLI/YAML/resume configuration, GPU placement and memory management. |
| `diagnostics/` | Startup displays, loading output and timestamped terminal/log routing. |
| `analysis/` | Entropy measurements, solution plots, progress curves and search trees. |
| `configs/` | Problem-specific YAML settings. |
| `tests/` | Regression and integration tests. |
| `vllm_compat/` | Worker-side vLLM compatibility hooks. |
| `examples/` | Reference task code, prompts and benchmark assets. |
| `requirements/` | Additional dependencies for particular tasks. |
| `runs/` | Experiment checkpoints, rollouts, scores, logs and plots. |
| `output/` | Generated artifacts outside individual experiment directories. |
| `_retire/` | Archived implementations and backups. |
