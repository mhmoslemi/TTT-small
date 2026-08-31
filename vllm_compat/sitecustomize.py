"""Compatibility hooks inherited only by TTT-managed vLLM subprocesses."""

import os
import sys


if os.environ.get("TTT_VLLM_DISABLE_BROKEN_FLASHINFER_COMM") == "1":
    # vLLM treats an unavailable FlashInfer comm package as optional and falls
    # back to NCCL/custom all-reduce. FlashInfer attention remains available.
    sys.modules["flashinfer.comm"] = None
