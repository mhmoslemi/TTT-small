"""Small output helpers that do not depend on model or GPU libraries."""

import builtins
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo


_SETTING_LOG_LOCK = threading.RLock()
_SETTING_LOG_PATH = None
_LOG_TIMEZONE = ZoneInfo("America/Toronto")


def log_timestamp():
    return datetime.fromtimestamp(time.time(), _LOG_TIMEZONE).strftime("[%H:%M:%S] ")


def bind_setting_log(path):
    """Bind hidden configuration/runtime notices directly to setting.log.

    The direct append path works even while stdout is temporarily redirected
    by model-loading capture, and O_APPEND keeps independent trainer processes
    from overwriting one another's records.
    """
    global _SETTING_LOG_PATH
    with _SETTING_LOG_LOCK:
        _SETTING_LOG_PATH = (
            None if path is None else str(Path(path).expanduser().resolve()))


def setting_log_only(*values, sep=" ", end="\n", flush=False, file=None):
    """Append a timestamped notice to setting.log without touching stdout."""
    if file is not None:
        builtins.print(
            *values, sep=sep, end=end, file=file, flush=flush)
        return
    value = sep.join(str(item) for item in values) + end
    with _SETTING_LOG_LOCK:
        path = _SETTING_LOG_PATH
    if path is None:
        # Preserve diagnostics in standalone utilities and early failures.
        terminal_log_only(
            *values, sep=sep, end=end, flush=flush)
        return
    timestamp = log_timestamp()
    rendered = "".join(
        (timestamp + line if line else line)
        for line in value.splitlines(keepends=True)
    )
    if value and not rendered:
        rendered = timestamp + value
    payload = rendered.encode("utf-8", errors="replace")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("setting.log append made no progress")
            remaining = remaining[written:]
    finally:
        os.close(descriptor)


def terminal_log_only(*values, sep=" ", end="\n", flush=False, file=None):
    """Write a status line to the run's terminal log without showing it.

    The installed runtime stream supplies ``write_log_only``.  Falling back to
    a normal print keeps early startup failures and standalone utility calls
    visible when no run-local terminal log exists yet.
    """
    stream = sys.stdout if file is None else file
    writer = getattr(stream, "write_log_only", None)
    if writer is None:
        builtins.print(*values, sep=sep, end=end, flush=flush, file=stream)
        return
    writer(sep.join(str(value) for value in values) + end)
    if flush:
        stream.flush()
