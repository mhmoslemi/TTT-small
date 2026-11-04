# TTT-small

## Hardware selection

Set `hardware_profile: B200` in the selected YAML for the B200 machine;
`RTX6000` preserves the existing placement policy. The current YAML hardware
selection also applies on resume. GPU capacity, CPU affinity and available host
RAM are detected, not assumed to equal the advertised machine size.
The seven installed checkpoints under `/root/nas/Mohammad/models/` are preferred
over their matching Hub IDs on B200 when present; explicit paths stay explicit.

B200 uses memory-fitting replicas/TP groups, GPU-resident Qwen Flash Next lookup
tables, larger automatic generation batches, and RAM-admitted model loading.
The existing adaptive parallel trainer, evaluation/transfer/offload RAM gate and
exact long-rollout handling remain in use. Sampling, precision, context limits
and the RL objective are unchanged. Explicit YAML/CLI capacity settings still
override automatic batch sizing. Inspect `setting.log` for resolved layouts.

Loading, sleeping model backups and CPU activation offloads share live RAM
admission with evaluation, leaving OS headroom and respecting Linux cgroup
limits. Engines whose sleep backups cannot fit are released and reloaded.
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
