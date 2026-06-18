"""
Single-Cell Analysis (scRNA-seq denoising).

  - entrypoint:  run_denoising
  - reward:      1 / mse   (Poisson is a HARD CONSTRAINT: rejected if poisson_norm < 0.97)
  - preprocess injects the bio imports + the exact evaluate_mse / evaluate_poisson /
    run_denoising_eval sources + a wrapper, then the model's magic_denoise.
  - get_question reproduced verbatim (SYSTEM_PROMPT with placeholders filled).

 !!!!!  REQUIRMENTS ---- (scanpy, anndata, scprep, graphtools, magic-impute, molecular-cross-validation, openproblems, pancreas dataset ---- REQUIRMENTS!!!!!

 """

from __future__ import annotations
from typing import Any, List
import numpy as np
from problems.base import Problem, ParentContext, RewardResult, SeedState


BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
}


def verify_denoising(result) -> bool:
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return False
    mse, poisson = result[0], result[1]
    if not np.isfinite(mse) or not np.isfinite(poisson):
        return False
    baseline = BASELINES["pancreas"]
    if poisson < baseline["perfect_poisson"]:
        return False
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range if poisson_range > 0 else 0
    if poisson_norm < 0.97:
        return False
    return True


class Denoising(Problem):
    name = "denoising"
    entrypoint = "run_denoising"
    metric_name = "MSE"
    maximize = False   # minimize MSE; reward = 1/mse keeps higher-is-better

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.97  # poisson_norm constraint threshold (informational)
        self.eval_seed = int(cfg.get("eval_seed", 42))

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        from examples.denoising.prompt import SYSTEM_PROMPT
        from examples.denoising.utils import EVALUATE_MSE_FUNC, EVALUATE_POISSON_FUNC

        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<EVALUATE_MSE_FUNC>>>", EVALUATE_MSE_FUNC)
        prompt = prompt.replace("<<<EVALUATE_POISSON_FUNC>>>", EVALUATE_POISSON_FUNC)

        has_code = bool(parent.code and parent.code.strip())
        value_ctx = ""
        if parent.raw_score is not None:
            value_ctx = f"\nCurrent metrics (lower is better): MSE: {parent.raw_score:.6f}"

        if has_code:
            clean_code = parent.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the current implementation:
```python
{clean_code}
```

You are iteratively improving the denoising algorithm.{value_ctx}

Reason about how you could improve this approach.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to implement a denoising algorithm.
"""

        user = f"""{prompt}
{code_section}
Write your improved `magic_denoise` function."""
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        import inspect
        from examples.denoising.utils import (
            evaluate_mse, evaluate_poisson, run_denoising_eval,
        )

        imports = f"""import numpy as np
import scipy
import scipy.sparse
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import graphtools
import scprep
import anndata
import scanpy as sc
import sklearn.metrics
import math
import random
from molecular_cross_validation.mcv_sweep import poisson_nll_loss

_SEED = {self.eval_seed}
"""
        wrapper = """
def run_denoising():
    return run_denoising_eval(magic_denoise, seed=_SEED)
"""
        return (
            imports + "\n\n"
            + inspect.getsource(evaluate_mse) + "\n\n"
            + inspect.getsource(evaluate_poisson) + "\n\n"
            + inspect.getsource(run_denoising_eval) + "\n\n"
            + code + "\n\n"
            + wrapper
        )

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not verify_denoising(output):
            res.msg = "Invalid solution."
            return res
        mse, poisson = output[0], output[1]
        current_mse = mse if mse is not None else float("inf")
        res.valid = True
        res.raw_score = float(current_mse)
        res.reward = float(1.0 / current_mse) if current_mse > 0 else self.fail_score
        res.construction = []  # not reused
        res.msg = f"mse={current_mse}, poisson={poisson}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        # Initial state mirrors create_initial_state: MAGIC baseline.
        try:
            from examples.denoising.utils import MAGIC_FUNC
            code = MAGIC_FUNC
        except Exception:
            code = ""  # bio stack absent; model will write from scratch
        baseline_mse = 0.2316  # value used in create_initial_state
        value = float(1.0 / baseline_mse)
        return [SeedState(code=code, value=value, raw_score=baseline_mse, construction=[])
                for _ in range(self.num_seed_states)]
