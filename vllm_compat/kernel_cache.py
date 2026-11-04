"""Prepare shared FlashInfer attention artifacts before expensive GPU work."""

import os
from pathlib import Path
import subprocess
import sys


def prepare_kernel_cache(gpu_ids, log_path):
    from runtime.gpu_runtime import query_gpu_memory
    cards = query_gpu_memory()
    if not any("B200" in cards[gpu].name.upper() for gpu in gpu_ids if gpu in cards):
        return
    print("[init] checking shared FlashInfer attention kernel cache ...", flush=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    # Keep torch/FlashInfer imports out of the trainer: Unsloth must patch first.
    with open(log_path, "a", encoding="utf-8") as log:
        try:
            subprocess.run([sys.executable, "-m", "vllm_compat.kernel_cache"],
                           cwd=Path(__file__).resolve().parents[1], env=environment,
                           stdout=log, stderr=log, check=True, timeout=900)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"FlashInfer kernel cache preparation failed before model loading; see {log_path}") from exc
    print("[init] shared FlashInfer attention kernels ready", flush=True)


def cache_attention_artifacts():
    import functools
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import PurePosixPath
    from flashinfer import artifacts
    from flashinfer.jit import cubin_loader

    # The default 10s download/30s lock limits failed while eight engines
    # competed for one cold cache. Fetch once, retaining upstream SHA checks.
    cubin_loader.download_file = functools.partial(
        cubin_loader.download_file, retries=4, timeout=60, lock_timeout=600)
    get_artifact = getattr(cubin_loader, "get_artifact", None) or cubin_loader.get_cubin
    subdir = artifacts.ArtifactPath.TRTLLM_GEN_FMHA.rstrip("/")
    checksum = get_artifact(f"{subdir}/checksums.txt", artifacts.CheckSumHash.TRTLLM_GEN_FMHA)
    if not checksum:
        raise RuntimeError(f"cannot fetch verified FlashInfer manifest: {subdir}")
    files = []
    for line in checksum.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(digest) != 64:
            raise ValueError(f"invalid FlashInfer manifest entry: {name}")
        files.append((f"{subdir}/{path}", digest))
    if not files:
        raise RuntimeError("FlashInfer manifest contained no kernel files")

    def fetch(item):
        name, digest = item
        if not get_artifact(name, digest):
            raise RuntimeError(f"cannot fetch verified FlashInfer artifact: {name}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fetch, files))
    print(f"[kernel-cache] verified {len(files)} FlashInfer attention artifacts", flush=True)


if __name__ == "__main__":
    cache_attention_artifacts()
