# TTT-small

## Hardware selection

Set `hardware_profile: B200` in the selected YAML for the B200 machine;
`RTX6000` preserves the existing placement policy. The current YAML hardware
selection also applies on resume. GPU capacity, CPU affinity and available host
RAM are detected, not assumed to equal the advertised machine size.
Matching local checkpoints under `/root/nas/Mohammad/models/` are preferred
over their matching Hub IDs on B200 when present; explicit paths stay explicit.

B200 automatically uses a disk-backed trainer and transient inference workers,
including on resume, regardless of saved sleep/reload settings. Each trainer
writes its immutable base weights once per process to `.offload/`, reuses that
snapshot across steps, and transfers weights in 64-MiB chunks. Only adapters,
optimizer state and mutable buffers remain in host RAM while generation runs.
Two Qwen3.8-27B replicas need approximately 100 GiB of free disk for snapshots;
they are temporary files, not resume checkpoints. A new process rebuilds them
from its loaded checkpoint. `--trainer-offload-dir` can choose another disk.

GLiDE exits after parent selection. Strategist workers exit before coding;
coder workers stay loaded through pilot, phase 2, retries and scoring, then
exit before training resumes. No inference weights are backed up in host RAM.
Flash-Next's lookup tables stay on GPU and are included in GPU placement estimates.
The B200 RAM admission reserve is 16 GiB (capped at a quarter of node/cgroup
capacity); an explicit `TTT_HOST_MEMORY_RESERVE_GIB` still overrides it.

RTX6000 retains its existing placement and sleep/wake policy. Both profiles
retain shared RAM admission, bounded transfers and Linux cgroup accounting.
`run_copy.sh` uses the same execution flags as `run2.sh`; no extra B200 flags
are needed. Sampling, precision, context limits, prompts, rollout scheduling,
evaluation worker limits and the RL objective are unchanged.
FlashInfer initialization and downloads use the original vLLM path; there is
no extra startup process, download hook or bulk-kernel preflight.
Disk reload and worker initialization add phase-transition time; compute speed
is not benchmarked. This cannot protect against arbitrary external allocations.

## Semantic parent selection

Enable PUCT with GLiDE-9B comparisons using:

```sh
sh run_copy.sh --parent-selection puct-jev
```

Download the complete BF16 `fastino/GLiDE-9B` repository (including `glide/`
and `readout.safetensors`) to `/root/nas/Mohammad/models/GLiDE-9B` on B200,
or put `GLiDE-9B` beside the local coder checkpoint. Override its location
with `--jev-model-path /path/to/GLiDE-9B`. The worker uses the launch Python
environment and the checkpoint's native inference code; its `requirements.txt`
lists the vendor's validated dependencies. No packages are installed at runtime.

Step 0 uses ordinary PUCT and never loads GLiDE. On later steps PUCT proposes
parents in score order, considering at most the top
`jev_candidate_multiplier * groups_per_step` candidates. Every YAML defaults
`jev_candidate_multiplier` to `3`: eight requested parents means a window of
24 candidates, including the first accepted parent. Selection stops early when
the batch is full. One candidate can require comparisons with multiple accepted
parents, so this bounds candidates rather than individual model requests.
GLiDE receives the problem description and two complete
proposed Python programs, returning probabilities for `same_core_idea`,
`meaningful_variant`, and `different_approach`. A candidate is skipped if any
accepted parent matches with `same_core_idea >= 0.90`; uncertain cases remain
eligible. Original lineage blocking is relaxed within that window if needed,
then rejected candidates fill any remaining slots in PUCT order without
further comparisons or looking outside the window, explicitly logged
as `duplicate_fallback`. The archive and expansion statistics are preserved.

The trainer is parked before the first comparison, GLiDE loads on the first
rollout GPU, and its process exits before strategist/coder generation. It has
no persistent host-RAM weight backup. Full rendered prompts are checked against
40,000 tokens; oversized pairs and inference failures stop with an error rather
than silently truncating source or bypassing comparison.

Each step saves its problem, compared sources, probabilities, and final picks
in `stepNN/stepNN.jev.json`; worker diagnostics are in `stepNN.jev.worker.log`.
Completed comparisons and parent selections are reused on resume. An
unfinished selection retains its saved candidate multiplier; edits to the
current YAML apply to fresh selections. Older saved selections still replay.
A PUCT checkpoint can switch to `puct-jev` at the next fresh step while preserving
the archive, adapter, optimizer and visit counts.

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
