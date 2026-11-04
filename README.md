# TTT-small

## Hardware selection

Set `hardware_profile: B200` in the selected YAML for the B200 machine;
`RTX6000` preserves the existing placement policy. The current YAML hardware
selection also applies on resume. GPU capacity, CPU affinity and available host
RAM are detected, not assumed to equal the advertised machine size.
The seven installed checkpoints under `/root/nas/Mohammad/models/` are preferred
over their matching Hub IDs on B200 when present; explicit paths stay explicit.

B200 uses the same loading, sleep/wake and model-specific engine settings as
RTX6000, with replicas/TP groups and generation batches sized to its larger
GPU memory. Qwen Flash Next keeps its CPU-offloaded lookup table and the RTX
NVFP4 autotuning setting. No B200-only startup tuning or disk-spill path is used.
The existing adaptive parallel trainer, evaluation/transfer/offload RAM gate and
exact long-rollout handling remain in use. Sampling, precision, context limits
and the RL objective are unchanged. Explicit YAML/CLI capacity settings still
override automatic batch sizing. Inspect `setting.log` for resolved layouts.

B200 coder requests use a shared queue and refill engines after individual
rollouts finish. A parent's phase 2 starts as soon as its complete pilot set is
evaluated, while other parents can still be generating/evaluating pilots.
Automatic concurrency scales with available KV memory and the rollout workload,
with no fixed 64-request ceiling. vLLM controls paged-KV admission as outputs grow.

The original shared RAM admission and bounded transfers remain in place,
including OS headroom and Linux cgroup limits. Under genuine RAM pressure,
unretainable sleeping engines are released and reloaded next phase, as on RTX.
FlashInfer initialization and downloads use the original vLLM path; there is
no extra startup process, download hook or bulk-kernel preflight.
This is bounded resource planning, not a guarantee against arbitrary external
allocations or a claim of benchmarked maximum throughput.

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
