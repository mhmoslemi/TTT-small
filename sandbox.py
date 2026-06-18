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
import time


# Placeholders __PROGRAM_PATH__ / __FUNCTION_NAME__ / __RESULTS_PATH__
# are substituted before launch.
RUNNER_TEMPLATE = r'''
import os
import sys
import pickle
import traceback
import importlib.util

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
        except Exception:
            pass
    if shutil.which("pkill"):
        try:
            subprocess.run(
                ["pkill", "-KILL" if hard else "-TERM", "-P", str(proc.pid)],
                check=False,
            )
        except Exception:
            pass


def run_code(code: str, entrypoint: str, timeout_s: float, max_cpus: int = 1):
    """
    Execute `code` (Python source) in a subprocess. Calls `entrypoint()`
    and returns whatever it returns.

    Returns a dict:
      {"ok": True,  "value": <return value>, "stdout": "..."}
      {"ok": False, "error": "...", "stdout": "..."}
    """
    # Write code to a temp file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        program_path = f.name
        f.write(code)

    # Write the runner script
    runner_src = (
        RUNNER_TEMPLATE
        .replace("__PROGRAM_PATH__", program_path)
        .replace("__FUNCTION_NAME__", entrypoint)
        .replace("__RESULTS_PATH__", program_path + ".pkl")
    )
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        runner_path = f.name
        f.write(runner_src)

    results_path = program_path + ".pkl"

    # Limit BLAS threads in the child so generated code can't fork 200 threads
    env = os.environ.copy()
    t = str(max(1, int(max_cpus)))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        env.setdefault(key, t)

    proc = subprocess.Popen(
        [sys.executable, runner_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True, 
    )

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    stdout_bytes = b""
    stderr_bytes = b""
    timed_out = False

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc, pgid, hard=False)
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_tree(proc, pgid, hard=True)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    # Belt-and-suspenders: make sure nothing's left
    _kill_tree(proc, pgid, hard=True)

    stdout_text = stdout_bytes.decode(errors="ignore") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode(errors="ignore") if stderr_bytes else ""

    # Build result
    if timed_out:
        result = {"ok": False, "error": f"Timeout after {timeout_s}s", "stdout": stdout_text}
    elif proc.returncode != 0:
        # Child crashed without writing results
        if os.path.exists(results_path):
            try:
                with open(results_path, "rb") as f:
                    payload = pickle.load(f)
                payload["stdout"] = stdout_text
                result = payload
            except Exception as e:
                result = {"ok": False, "error": f"Failed to read results: {e}", "stdout": stdout_text}
        else:
            result = {
                "ok": False,
                "error": f"Process exited with code {proc.returncode}",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
    else:
        if not os.path.exists(results_path):
            result = {"ok": False, "error": "No results file written", "stdout": stdout_text}
        else:
            try:
                with open(results_path, "rb") as f:
                    payload = pickle.load(f)
                payload["stdout"] = stdout_text
                result = payload
            except Exception as e:
                result = {"ok": False, "error": f"Failed to read results: {e}", "stdout": stdout_text}

    # Cleanup
    for p in [program_path, runner_path, results_path]:
        try:
            os.unlink(p)
        except (FileNotFoundError, OSError):
            pass

    return result


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
