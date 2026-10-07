"""Startup diagnostics and a dependency-free, terminal-width-aware dashboard."""

import builtins
from contextlib import contextmanager, redirect_stdout
import io
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _StartupStream:
    def __init__(self, owner, visible, loading):
        self.owner, self.visible, self.loading = owner, visible, loading
        self.active = True
        self.progress = None

    def write(self, value):
        if not self.active:
            return self.visible.write(value)
        if self.loading and ("\r" in value or self.progress is not None):
            self.visible.write(value)
            if "\r" in value:
                self.progress = value.rsplit("\r", 1)[-1].strip()
            if "\n" in value:
                if self.progress:
                    self.owner._record(self.progress)
                self.progress = None
            return len(value)
        return self.owner.write(value, loading=self.loading)

    def flush(self):
        if self.active:
            self.owner.flush()
        self.visible.flush()

    def __getattr__(self, name):
        return getattr(self.visible, name)


class StartupLog:
    """Buffer pre-directory settings, then append them to setting.log.

    Only startup stdout is diverted. stderr stays visible, as do explicit
    warning/error lines. Streams cached by dependencies become console-only
    on scope exit, so later training and progress output cannot be captured.
    """

    def __init__(self, *, time_offset=0, console=None):
        self.console = sys.stdout if console is None else console
        self.time_offset = time_offset
        self.path = None
        self.pending = io.StringIO()
        self.partial = ""
        self.fields = {}
        self._summary = False

    def bind(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as log:
            log.write("\n=== Strategist Bandit | startup settings ===\n")
            log.write(self.pending.getvalue())
        self.pending = io.StringIO()

    def _record(self, line, *, loading=False):
        line = _ANSI.sub("", line)
        timestamp = time.strftime(
            "[%H:%M:%S] ", time.localtime(time.time() + self.time_offset))
        record = timestamp + line + "\n"
        if self.path is None:
            self.pending.write(record)
        else:
            with self.path.open("a", encoding="utf-8") as log:
                log.write(record)
        if self._summary and ":" in line:
            label, value = line.split(":", 1)
            self.fields[label.strip()] = value.strip()
        is_warning = re.search(r"\[warn(?:ing)?\]|\[error\]|\bwarning:",
                               line, re.IGNORECASE)
        # Keep one meaningful loading status and the first model's progress;
        # suppress its repeated kernel/precision/LoRA configuration messages.
        is_loading = loading and line.startswith("[backend=") and " loading " in line
        if is_warning or is_loading:
            builtins.print(line, file=self.console, flush=True)

    def write(self, value, *, loading=False):
        value = str(value)
        self.partial += value
        while "\n" in self.partial:
            line, self.partial = self.partial.split("\n", 1)
            self._record(line, loading=loading)
        return len(value)

    def flush(self):
        # Do not split a print() mid-line when a dependency flushes stdout.
        self.console.flush()

    def print(self, *values, **kwargs):
        kwargs.setdefault("file", self)
        builtins.print(*values, **kwargs)

    def begin_summary(self):
        self._summary = True

    def end_summary(self):
        self._summary = False

    @contextmanager
    def capture(self, *, loading=False):
        stream = _StartupStream(self, sys.stdout, loading)
        try:
            with redirect_stdout(stream):
                yield
        except BaseException as error:
            # Preserve argparse --help and validation output before a run dir
            # exists. A failed startup must never become a silent failure.
            if self.path is None:
                self.console.write(self.pending.getvalue())
            elif not (isinstance(error, SystemExit) and error.code == 0):
                builtins.print(f"[startup] details: {self.path}",
                               file=self.console, flush=True)
            raise
        finally:
            if self.partial:
                self._record(self.partial, loading=loading)
                self.partial = ""
            stream.active = False


def dashboard_sections(fields, run_dir, *, resuming=False):
    """Group the existing resolved banner fields; never reinterpret config."""
    used = set()

    def row(*keys):
        parts = []
        for key in keys:
            if key in fields:
                used.add(key)
                parts.append(f"{key}: {fields[key]}")
        return "   |   ".join(parts)

    sections = [
        ("RUN", 96, [
            row("Problem", "Entrypoint", "Steps"),
            row("Metric", "Target"),
            row("Search selection", "Seed", "Deterministic"),
        ]),
        ("MODELS & SAMPLING", 95, [
            row("Model"), row("Training model"), row("Coder sampling"),
            row("Coder reasoning"), row("Max new tokens", "Max seq length"),
            row("Strategy model"), row("Strategy sampling"),
        ]),
        ("ROLLOUTS & ALLOCATION", 93, [
            row("Groups per step", "Group size", "Total rollouts/step"),
            row("X-GRPO contexts P", "X-GRPO groups K", "X-GRPO group size G"),
            row("Rollout hierarchy"), row("Rollout pilot"),
            row("Strategy archive"), row("Output retries"),
        ]),
        ("TRAINING", 92, [
            row("Advantage mode", "LR", "KL coef", "Reference KL"),
            row("Training layout"), row("Training scheduler"), row("LoRA"),
            row("Train microbatch", "Logprob chunk"),
            row("Training memory cap"), row("Fused long attention"),
            row("MoE policy ratio"), row("Binary coder phase"), row("Binary coder clip"),
            row("CVaR alpha/lambda"), row("SPO-RS beta", "SPO-RS D_half"),
            row("SPO-RS rho min/max", "SPO-RS update epochs"),
            row("SPO-RS clip eps low/high"),
            row("Rank clip eps low/high", "Rank update epochs"),
            row("Rank gamma", "Rank entropy coef"),
            row("X-GRPO budgets", "X-GRPO rel. error", "X-GRPO entropy"),
        ]),
        ("HARDWARE & EVALUATION", 94, [
            row("Training backend", "Generation backend"),
            row("Training GPUs", "Generation GPUs"),
            row("vLLM parallelism", "Evaluation GPU"),
            row("CPU evaluation", "Sandbox timeout"),
        ]),
    ]
    remaining = [row(key) for key in fields if key not in used]
    if remaining:
        sections.append(("OTHER SETTINGS", 97, remaining))
    sections.append(("OUTPUT · RESUMING" if resuming else "OUTPUT", 96, [
        f"Run directory: {Path(run_dir).resolve()}",
        "Settings: setting.log   |   Terminal: temirnal.log   |   Results: result.txt",
    ]))
    return [(name, color, [r for r in rows if r]) for name, color, rows in sections]


def render_dashboard(fields, run_dir, *, width=104, color=False, resuming=False):
    """Render one bounded-width box, with colored section headings on a TTY."""
    width = max(36, int(width))
    inner = width - 4

    def paint(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text

    def content(text):
        return "│ " + text.ljust(inner) + " │"

    lines = [paint("╭" + "─" * (width - 2) + "╮", 90),
             content(paint("STRATEGIST BANDIT".ljust(inner), "1;96"))]
    for title, code, rows in dashboard_sections(fields, run_dir, resuming=resuming):
        header = f" {title} "
        lines.append(paint("├" + header + "─" * (width - 2 - len(header)) + "┤", code))
        for row in rows:
            for line in textwrap.wrap(row, width=inner, subsequent_indent="  ",
                                      break_long_words=True, break_on_hyphens=False):
                lines.append(content(line))
    lines.append(paint("╰" + "─" * (width - 2) + "╯", 90))
    return "\n".join(lines)


def print_dashboard(fields, run_dir, *, resuming=False):
    # Each console line already receives the project's 11-character timestamp.
    width = max(36, min(112, shutil.get_terminal_size((123, 24)).columns - 11))
    color = bool(sys.stdout.isatty() and "NO_COLOR" not in os.environ
                 and os.environ.get("TERM") != "dumb")
    builtins.print(render_dashboard(fields, run_dir, width=width,
                                    color=color, resuming=resuming), flush=True)
