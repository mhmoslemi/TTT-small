"""
Run model-generated code in a subprocess with a hard timeout.

The code is written to a temp file; a runner subprocess imports it, calls the
named entrypoint and pickles the return value. The parent reads the pickle, or
kills the whole process group on timeout.

Adapted from the reference implementation's sandbox.py, with one change that
matters here: the traceback is returned alongside the error string. The
framework's feedback signal (Eq. 9) is only as informative as the text the
verifier hands back, and "ValueError" alone tells the model far less about
which tokens went wrong than the frame that raised it.

This is isolation for accident, not for malice: generated code runs with the
caller's privileges. Run untrusted models in a container.
"""

import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile

RUNNER_TEMPLATE = r'''
import os
import sys
import pickle
import traceback
import importlib.util

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
            pickle.dump({"ok": False, "error": f"{type(e).__name__}: {e}",
                         "traceback": tb}, f)
    except Exception:
        pass
    sys.stderr.write(tb)
    sys.exit(1)
'''


def _kill_tree(proc, pgid, hard=False):
    sig = signal.SIGKILL if hard else signal.SIGTERM
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
        except Exception:
            pass
    if shutil.which("pkill"):
        try:
            subprocess.run(["pkill", "-KILL" if hard else "-TERM", "-P",
                            str(proc.pid)], check=False)
        except Exception:
            pass


def run_code(code: str, entrypoint: str, timeout_s: float, max_cpus: int = 1) -> dict:
    """
    Execute `code` and call `entrypoint()`.

    Returns {"ok": True, "value": ..., "stdout": ...} or
            {"ok": False, "error": ..., "traceback": ..., "stdout": ...}
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        program_path = f.name
        f.write(code)

    runner_src = (RUNNER_TEMPLATE
                  .replace("__PROGRAM_PATH__", program_path)
                  .replace("__FUNCTION_NAME__", entrypoint)
                  .replace("__RESULTS_PATH__", program_path + ".pkl"))
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        runner_path = f.name
        f.write(runner_src)

    results_path = program_path + ".pkl"

    # Cap BLAS threads so one generated program cannot fork hundreds of threads.
    env = os.environ.copy()
    threads = str(max(1, int(max_cpus)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env.setdefault(key, threads)

    proc = subprocess.Popen([sys.executable, runner_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, start_new_session=True)
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    stdout_bytes = stderr_bytes = b""
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
    finally:
        _kill_tree(proc, pgid, hard=True)

    stdout_text = stdout_bytes.decode(errors="ignore") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode(errors="ignore") if stderr_bytes else ""

    def _read_payload():
        with open(results_path, "rb") as fh:
            return pickle.load(fh)

    if timed_out:
        result = {"ok": False, "error": f"Timeout after {timeout_s}s",
                  "traceback": "", "stdout": stdout_text}
    elif os.path.exists(results_path):
        try:
            result = _read_payload()
        except Exception as e:
            result = {"ok": False, "error": f"Failed to read results: {e}",
                      "traceback": stderr_text}
        result["stdout"] = stdout_text
    else:
        result = {"ok": False,
                  "error": f"Process exited with code {proc.returncode}",
                  "traceback": stderr_text, "stdout": stdout_text}

    for path in (program_path, runner_path, results_path):
        try:
            os.unlink(path)
        except OSError:
            pass
    return result
