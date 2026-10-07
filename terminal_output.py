"""Small output helpers that do not depend on model or GPU libraries."""

import builtins
import sys


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
