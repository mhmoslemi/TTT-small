"""Restart a killed trainer from its durable run directory."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def request_restart(reason):
    path = os.environ.get("TTT_SUPERVISOR_RESTART_FILE")
    if path:
        Path(path).write_text(str(reason), encoding="utf-8")


def _resume_command(command, run_dir):
    cleaned = []
    skip = False
    for argument in command:
        if skip:
            skip = False
            continue
        if argument in ("--resume", "--resume-from"):
            skip = True
            continue
        if argument.startswith("--resume=") or argument.startswith("--resume-from="):
            continue
        cleaned.append(argument)
    return [*cleaned, "--resume", str(run_dir)]


def supervise(command, *, initial_delay=2.0, maximum_delay=60.0):
    if not command:
        raise ValueError("missing trainer command")

    with tempfile.TemporaryDirectory(prefix="ttt-supervisor-") as temporary:
        run_file = Path(temporary) / "run_dir"
        restart_file = Path(temporary) / "restart"
        environment = dict(
            os.environ,
            TTT_SUPERVISOR_RUN_FILE=str(run_file),
            TTT_SUPERVISOR_RESTART_FILE=str(restart_file),
        )
        child = None
        stopping = False

        def stop(signum, _frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

        previous = {
            signum: signal.signal(signum, stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        delay = float(initial_delay)
        try:
            active_command = list(command)
            while True:
                restart_file.unlink(missing_ok=True)
                child = subprocess.Popen(
                    active_command,
                    env=environment,
                    start_new_session=True,
                )
                returncode = child.wait()
                restart_requested = restart_file.is_file()
                killed = returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
                if stopping or not (killed or restart_requested):
                    return returncode
                try:
                    run_dir = Path(run_file.read_text(encoding="utf-8").strip()).resolve()
                except (OSError, ValueError):
                    print(
                        "[supervisor] trainer was SIGKILLed before its run "
                        "directory became durable; refusing an ambiguous restart",
                        file=sys.stderr,
                        flush=True,
                    )
                    return returncode
                if not (run_dir / "config.json").is_file():
                    print(
                        f"[supervisor] trainer was SIGKILLed but {run_dir} is "
                        "not resumable; stopping",
                        file=sys.stderr,
                        flush=True,
                    )
                    return returncode
                reason = (
                    restart_file.read_text(encoding="utf-8").strip()
                    if restart_requested else
                    "trainer was SIGKILLed; vLLM workers are being reaped"
                )
                print(f"[supervisor] {reason}. Resuming {run_dir} in "
                      f"{delay:g}s ...", flush=True)
                time.sleep(delay)
                if stopping:
                    return 128 + signal.SIGINT
                active_command = _resume_command(command, run_dir)
                delay = min(float(maximum_delay), max(delay, 1.0) * 2.0)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    raise SystemExit(supervise(arguments))


if __name__ == "__main__":
    main()
