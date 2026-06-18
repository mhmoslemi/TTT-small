"""

Names accepted:
  circle_packing | circle          -> CirclePacking
  erdos | erdos_min_overlap        -> ErdosMinOverlap
  ac1 | ac2 | ac_inequalities      -> ACInequalities      (problem_type in cfg)
  denoising | single_cell          -> Denoising
  gpu_mode | kernel | trimul        -> GpuMode             (problem_type in cfg)
       | mla_decode_nvidia
  ahc | algorithm_engineering      -> Not Implemented 

"""

from __future__ import annotations


def available_problems():
    return [
        "circle_packing", "erdos", "ac1", "ac2",
        "denoising", "gpu_mode", "ahc",
    ]


def get_problem(name: str, cfg: dict):
    key = (name or "").strip().lower()

    if key in ("circle_packing", "circle", "circles"):
        from problems.circle_packing import CirclePacking
        return CirclePacking(cfg)

    if key in ("erdos", "erdos_min_overlap", "erdos_minimum_overlap"):
        from problems.erdos import ErdosMinOverlap
        return ErdosMinOverlap(cfg)

    if key in ("ac1", "ac2", "ac_inequalities", "autocorrelation", "autocorrelation_inequalities"):
        from problems.ac_inequalities import ACInequalities
        # problem_type lives in the config; the bare "ac1"/"ac2" name also sets it.
        if key in ("ac1", "ac2"):
            cfg = {**cfg, "problem_type": cfg.get("problem_type", key)}
        return ACInequalities(cfg)

    if key in ("denoising", "single_cell", "single_cell_analysis", "scrna"):
        from problems.denoising import Denoising
        return Denoising(cfg)

    if key in ("gpu_mode", "kernel", "kernel_engineering", "trimul", "mla_decode_nvidia", "mla"):
        from problems.gpu_mode import GpuMode
        if key in ("trimul", "mla_decode_nvidia"):
            cfg = {**cfg, "problem_type": cfg.get("problem_type", key)}
        elif key == "mla":
            cfg = {**cfg, "problem_type": cfg.get("problem_type", "mla_decode_nvidia")}
        return GpuMode(cfg)

    if key in ("ahc", "algorithm_engineering", "atcoder"):
        raise NotImplementedError(
            "AHC / Algorithm Engineering is not implemented: the original "
            "examples/ahc/env.py and its prompt were not part of the shared "
            "codebase, so the prompt cannot be reproduced 'exactly as original'. "
            "Share examples/ahc (env.py + prompt) and this slot can be filled in."
        )

    raise ValueError(
        f"Unknown problem '{name}'. Available: {', '.join(available_problems())}"
    )
