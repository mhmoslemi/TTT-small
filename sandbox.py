"""
Run model-generated code in a subprocess with a hard timeout.

Write the code to a temp file, then spawn a Python subprocess that:
  1. imports the file
  2. calls the named entrypoint function
  3. pickles the return value to a results file

The parent process reads the pickle when the child exits cleanly, or
kills the child (and its process group) on timeout.

"""

import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import threading


# Creating a Popen briefly allocates several parent-side descriptors. Serialize
# only that millisecond-scale setup; the child processes themselves still run
# fully concurrently on their assigned CPUs.
_SANDBOX_LAUNCH_LOCK = threading.Lock()


# Placeholders __PROGRAM_PATH__ / __FUNCTION_NAME__ / __RESULTS_PATH__
# are substituted before launch.
RUNNER_TEMPLATE = r'''
import os
import sys
import pickle
import traceback
import importlib.util

# Pin before importing the generated program. The affinity is inherited by
# every subprocess it creates, so one candidate and its whole process tree stay
# on the CPU assigned by the evaluation scheduler.
EVAL_CPU_ID = os.environ.pop("TTT_EVAL_CPU_ID", "")
if EVAL_CPU_ID:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("isolated evaluation requires Linux CPU affinity")
    os.sched_setaffinity(0, {int(EVAL_CPU_ID)})

# Force spawn for any multiprocessing the child code might do
try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
except Exception:
    pass

PROGRAM_PATH = "__PROGRAM_PATH__"
FUNCTION_NAME = "__FUNCTION_NAME__"
RESULTS_PATH = "__RESULTS_PATH__"

sys.path.insert(0, os.path.dirname(PROGRAM_PATH))

try:
    spec = importlib.util.spec_from_file_location("program", PROGRAM_PATH)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    fn = getattr(program, FUNCTION_NAME)
    result = fn()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"ok": True, "value": result}, f)
except Exception as e:
    tb = traceback.format_exc()
    try:
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump({"ok": False, "error": str(e), "traceback": tb}, f)
    except Exception:
        pass
    sys.stderr.write(tb)
'''


def _kill_tree(proc, pgid, hard=False):
    """Best-effort kill of the entire process tree."""
    sig = signal.SIGKILL if hard else signal.SIGTERM
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            # The session no longer exists, so it has no surviving children.
            return
        except Exception:
            pass
    # start_new_session normally makes killpg sufficient. Use pkill only on
    # platforms/races where no usable process group was available; spawning a
    # pkill process for every successful evaluation wastes hundreds of PIDs.
    if proc.poll() is None and shutil.which("pkill"):
        try:
            subprocess.run(
                ["pkill", "-KILL" if hard else "-TERM", "-P", str(proc.pid)],
                check=False,
            )
        except Exception:
            pass


def run_code(code: str, entrypoint: str, timeout_s: float, max_cpus: int = 1,
             *, cpu_id=None):
    """
    Execute `code` (Python source) in a subprocess. Calls `entrypoint()`
    and returns whatever it returns.

    Returns a dict:
      {"ok": True,  "value": <return value>, "stdout": "..."}
      {"ok": False, "error": "...", "stdout": "..."}
    """
    pinned_cpu = None
    if cpu_id is not None:
        if not (hasattr(os, "sched_getaffinity")
                and hasattr(os, "sched_setaffinity")):
            raise RuntimeError(
                "isolated evaluation requires Linux CPU affinity support")
        pinned_cpu = int(cpu_id)
        allowed_cpus = set(os.sched_getaffinity(0))
        if pinned_cpu not in allowed_cpus:
            raise ValueError(
                f"CPU {pinned_cpu} is outside this process's allowed CPU set")

    paths = []
    proc = None
    pgid = None
    try:
        # Write code to a temp file.
        with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w") as f:
            program_path = f.name
            paths.append(program_path)
            f.write(code)

        results_path = program_path + ".pkl"
        stdout_path = program_path + ".stdout"
        stderr_path = program_path + ".stderr"
        paths.extend((results_path, stdout_path, stderr_path))

        # Write the runner script.
        runner_src = (
            RUNNER_TEMPLATE
            .replace("__PROGRAM_PATH__", program_path)
            .replace("__FUNCTION_NAME__", entrypoint)
            .replace("__RESULTS_PATH__", results_path)
        )
        with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w") as f:
            runner_path = f.name
            paths.append(runner_path)
            f.write(runner_src)

        # Limit BLAS threads in the child so generated code cannot fork one
        # thread per host CPU.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env.pop("TTT_EVAL_CPU_ID", None)
        t = "1" if pinned_cpu is not None else str(max(1, int(max_cpus)))
        for key in [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS"]:
            env[key] = t
        if pinned_cpu is not None:
            env["TTT_EVAL_CPU_ID"] = str(pinned_cpu)

        # Do not retain PIPE descriptors for the lifetime of every sandbox.
        # The child writes to files, and the parent closes its handles as soon
        # as Popen returns. Serializing only this setup also bounds transient
        # errpipe descriptors without serializing any actual evaluation work.
        with _SANDBOX_LAUNCH_LOCK:
            with open(stdout_path, "wb") as stdout_file, open(
                    stderr_path, "wb") as stderr_file:
                proc = subprocess.Popen(
                    [sys.executable, runner_path],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    start_new_session=True,
                )

        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc, pgid, hard=False)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_tree(proc, pgid, hard=True)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

        # Belt-and-suspenders: make sure no descendant outlives evaluation.
        _kill_tree(proc, pgid, hard=True)

        def _read_capture(path):
            try:
                with open(path, "rb") as capture:
                    return capture.read()
            except OSError:
                return b""

        stdout_bytes = _read_capture(stdout_path)
        stderr_bytes = _read_capture(stderr_path)
        stdout_text = (
            stdout_bytes.decode(errors="ignore") if stdout_bytes else "")
        stderr_text = (
            stderr_bytes.decode(errors="ignore") if stderr_bytes else "")

        if timed_out:
            return {
                "ok": False,
                "error": f"Timeout after {timeout_s}s",
                "stdout": stdout_text,
            }
        if proc.returncode != 0:
            # The runner normally pickles its exception even though it exits
            # successfully; this branch handles import/interpreter crashes.
            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        payload = pickle.load(f)
                    payload["stdout"] = stdout_text
                    return payload
                except Exception as error:
                    return {
                        "ok": False,
                        "error": f"Failed to read results: {error}",
                        "stdout": stdout_text,
                    }
            return {
                "ok": False,
                "error": f"Process exited with code {proc.returncode}",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if not os.path.exists(results_path):
            return {
                "ok": False,
                "error": "No results file written",
                "stdout": stdout_text,
            }
        try:
            with open(results_path, "rb") as f:
                payload = pickle.load(f)
            payload["stdout"] = stdout_text
            return payload
        except Exception as error:
            return {
                "ok": False,
                "error": f"Failed to read results: {error}",
                "stdout": stdout_text,
            }
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc, pgid, hard=True)
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
        for path in paths:
            try:
                os.unlink(path)
            except (FileNotFoundError, OSError):
                pass


if __name__ == "__main__":
    # Quick self-test
    code = """
import numpy as np
def my_entry():
    return np.array([[0.5, 0.5]]), np.array([0.4]), 0.4
"""
    print(run_code(code, "my_entry", timeout_s=10))

    bad_code = """
def my_entry():
    while True:
        pass
"""
    print(run_code(bad_code, "my_entry", timeout_s=2))
