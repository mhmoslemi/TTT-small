"""Quiet duplicate replica startup, without silencing subsequent training."""

from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import sys
import traceback


class _LoadingStream:
    """Cached library handlers return to the console when loading ends."""

    def __init__(self, diagnostic, visible):
        self.target = diagnostic
        self.visible = visible

    def write(self, value):
        return self.target.write(value)

    def flush(self):
        return self.target.flush()

    def write_log_only(self, value):
        """Keep explicit run diagnostics in temirnal.log during quiet loads."""
        writer = getattr(self.visible, "write_log_only", None)
        if writer is not None:
            return writer(value)
        return self.target.write(value)

    def detach(self):
        self.target = self.visible

    def __getattr__(self, name):
        return getattr(self.target, name)


def _restore_fd(fd, saved):
    try:
        os.dup2(saved, fd)
    finally:
        os.close(saved)


@contextmanager
def quiet_replica_load(log_path):
    """Send duplicate startup output to a diagnostic file, then restore it.

    Use only around sequential replica initialization, never around training
    or generation. Descriptor redirection also catches native HF download
    progress and logging handlers created before entering this context. Python
    stream wrappers remain usable if a library retains them after loading.
    Failures still propagate, with the full startup log's path on the console.
    Without a diagnostic path, keep output visible rather than discard it.
    """
    if not log_path:
        yield
        return

    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    visible_stdout, visible_stderr = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as diagnostic:
        diagnostic.write("\n--- trainer replica loading ---\n")
        stdout = _LoadingStream(diagnostic, visible_stdout)
        stderr = _LoadingStream(diagnostic, visible_stderr)
        failed = False
        try:
            # Flush before switching descriptors, so earlier terminal output
            # cannot accidentally end up in this worker's diagnostic file.
            visible_stdout.flush()
            visible_stderr.flush()
            with ExitStack() as stack:
                for fd in (1, 2):
                    saved = os.dup(fd)
                    stack.callback(_restore_fd, fd, saved)
                    os.dup2(diagnostic.fileno(), fd)
                stack.enter_context(redirect_stdout(stdout))
                stack.enter_context(redirect_stderr(stderr))
                try:
                    yield
                finally:
                    stdout.flush()
                    stderr.flush()
        except BaseException:
            failed = True
            traceback.print_exc(file=diagnostic)
            raise
        finally:
            stdout.detach()
            stderr.detach()
            if failed:
                print(f"[train-load] replica initialization failed; details: {path}",
                      file=visible_stderr, flush=True)
