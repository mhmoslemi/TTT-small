"""
Progress reporting.

A step spends its time in three long, silent places -- one blocking
model.generate() call per target, a serial loop of sandbox subprocesses, and a
forward/backward pass per rollout. Without a bar the whole step looks hung,
which is indistinguishable from actually being hung.

tqdm is used when available and a dependency-free fallback otherwise, so
progress is never the reason a run needs an extra package.
"""

import shutil
import sys
import time
from typing import Optional

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _fmt_seconds(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class _FallbackBar:
    """Single-line stderr bar. Throttled, and quiet when not attached to a tty."""

    def __init__(self, total: int, desc: str, unit: str = "it",
                 min_interval: float = 0.25):
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.min_interval = min_interval
        self.n = 0
        self.started = time.time()
        self._last_draw = 0.0
        self._postfix = ""
        self._tty = sys.stderr.isatty()
        self._draw(force=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        self._draw()

    def set_postfix_str(self, text: str) -> None:
        self._postfix = text
        self._draw()

    def _draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now

        elapsed = now - self.started
        rate = self.n / elapsed if elapsed > 0 else 0.0
        if self.total and rate > 0:
            eta = _fmt_seconds((self.total - self.n) / rate)
            pct = min(100.0, 100.0 * self.n / self.total)
            counter = f"{pct:5.1f}% {self.n}/{self.total}"
        else:
            eta = "--:--"
            counter = f"{self.n}"

        line = (f"  {self.desc}: {counter} "
                f"[{_fmt_seconds(elapsed)}<{eta}, {rate:.1f} {self.unit}/s]")
        if self._postfix:
            line += f" {self._postfix}"

        if self._tty:
            width = shutil.get_terminal_size((100, 20)).columns
            sys.stderr.write("\r" + line[:width].ljust(width))
        else:
            # Piped to a file: one line per draw would flood the log.
            sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def close(self) -> None:
        self._draw(force=True)
        if self._tty:
            sys.stderr.write("\n")
        sys.stderr.flush()


class _NullBar:
    def update(self, n: int = 1) -> None: ...
    def set_postfix_str(self, text: str) -> None: ...
    def close(self) -> None: ...


def make_bar(total: int, desc: str, unit: str = "it", enabled: bool = True):
    """A progress bar with .update(n) / .set_postfix_str(s) / .close()."""
    if not enabled:
        return _NullBar()
    if _tqdm is not None:
        return _tqdm(total=max(int(total), 0), desc=f"  {desc}", unit=unit,
                     leave=False, dynamic_ncols=True, file=sys.stderr)
    return _FallbackBar(total, desc, unit)


class PhaseTimer:
    """Announce a phase, then report how long it took."""

    def __init__(self, label: str, enabled: bool = True):
        self.label = label
        self.enabled = enabled
        self.started = 0.0

    def __enter__(self):
        self.started = time.time()
        if self.enabled:
            sys.stderr.write(f"  {self.label} ...\n")
            sys.stderr.flush()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.stderr.write(
                f"  {self.label}: done in {time.time() - self.started:.1f}s\n")
            sys.stderr.flush()
        return False
