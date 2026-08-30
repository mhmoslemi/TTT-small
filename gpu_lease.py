"""Cross-process exclusive leases for physical GPU evaluation devices."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import time


@contextmanager
def gpu_lease(gpu_id: int, timeout_s: float = 0.0,
              purpose: str = "evaluation"):
    """Hold an advisory lease for one physical GPU.

    A separate lock file is used for every physical id, so independent runs and
    reward-worker processes serialize benchmark evaluations on the same card.
    timeout_s <= 0 waits indefinitely.
    """
    gpu_id = int(gpu_id)
    lease_root = Path(os.environ.get(
        "TTT_GPU_LEASE_DIR",
        str(Path(tempfile.gettempdir()) / "ttt-gpu-leases"),
    ))
    lease_root.mkdir(parents=True, exist_ok=True)
    path = lease_root / f"gpu-{gpu_id}.lock"
    handle = path.open("a+")
    deadline = time.monotonic() + float(timeout_s) if timeout_s > 0 else None
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for physical GPU {gpu_id} "
                        f"{purpose} lease")
                time.sleep(0.1)

        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} purpose={purpose}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
