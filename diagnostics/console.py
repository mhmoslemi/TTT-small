"""Timestamped terminal capture and dependency-notice routing."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import logging
import os
import re
import sys
import threading
import time

import diagnostics.terminal_output as _terminal_output

# Logging is diagnostic and must never make an otherwise valid trainer update
# fail.  Accept the pre-setting.log helper module during a rolling/mixed
# deployment; a complete deployment still uses the direct setting.log sink.
setting_log_only = getattr(_terminal_output, "setting_log_only", _terminal_output.terminal_log_only)
_native_bind_setting_log = getattr(_terminal_output, "bind_setting_log", None)


def bind_setting_log(path, *, time_offset=0):
    if callable(_native_bind_setting_log):
        return _native_bind_setting_log(path, time_offset=time_offset)
    return None


# Downstream modules in an older deployment may still import these names
# directly after importing this module. Publish the compatibility aliases so
# those imports remain non-fatal as well.
if not hasattr(_terminal_output, "setting_log_only"):
    _terminal_output.setting_log_only = setting_log_only
if not hasattr(_terminal_output, "bind_setting_log"):
    _terminal_output.bind_setting_log = bind_setting_log


_LOG_TIME_OFFSET_SECONDS = -4 * 60 * 60


class _TimestampedLineStream:
    def __init__(self, stream):
        self.stream = stream
        self._line_start = True
        self._dynamic_progress = False
        self._lock = threading.Lock()

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        progress_stream = getattr(self.stream, "progress_stream", None)
        route_progress = bool(
            progress_stream is not None
            and ("\r" in value or (self._dynamic_progress and "\n" in value and not value.strip())))
        if route_progress:
            with self._lock:
                progress_stream.write(value)
                if "\r" in value and "\n" not in value:
                    self._dynamic_progress = True
                if "\n" in value:
                    self._dynamic_progress = False
                    self._line_start = True
            return len(value)
        parts = value.split("\n")
        with self._lock:
            for index, part in enumerate(parts):
                if part and self._line_start:
                    timestamp = time.strftime("[%H:%M:%S] ", time.localtime(time.time() + _LOG_TIME_OFFSET_SECONDS))
                    self.stream.write(timestamp)
                if part:
                    self.stream.write(part)
                    self._line_start = False
                if index + 1 < len(parts):
                    self.stream.write("\n")
                    self._line_start = True
        return len(value)

    def flush(self):
        self.stream.flush()

    def write_log_only(self, value):
        """Timestamp complete diagnostic lines without touching the TTY."""
        value = str(value)
        if not value:
            return 0
        writer = getattr(self.stream, "write_log_only", None)
        if writer is None:
            return self.write(value)
        parts = value.split("\n")
        rendered = []
        for index, part in enumerate(parts):
            if part:
                timestamp = time.strftime("[%H:%M:%S] ", time.localtime(time.time() + _LOG_TIME_OFFSET_SECONDS))
                rendered.extend((timestamp, part))
            if index + 1 < len(parts):
                rendered.append("\n")
        with self._lock:
            writer("".join(rendered))
        return len(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _install_console_timestamps():
    if not isinstance(sys.stdout, _TimestampedLineStream):
        sys.stdout = _TimestampedLineStream(sys.stdout)
    if not isinstance(sys.stderr, _TimestampedLineStream):
        sys.stderr = _TimestampedLineStream(sys.stderr)


class _TerminalTeeStream:
    """Mirror one console stream to the run log without changing its TTY."""

    def __init__(self, console, sink):
        self.console = console
        self.sink = sink
        self._progress_stream = _TerminalProgressTeeStream(console, sink)

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.console.write(value)
            self.sink.write(value)
        return len(value)

    def flush(self):
        with self.sink.lock:
            self.console.flush()
            self.sink.flush()

    def write_log_only(self, value):
        """Append to temirnal.log while deliberately bypassing the console."""
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.sink.write(value)
        return len(value)

    @property
    def progress_stream(self):
        """TTY-preserving stream whose saved progress row updates in place."""
        return self._progress_stream

    def __getattr__(self, name):
        # Preserve fileno(), isatty(), encoding, etc. from the real terminal so
        # tqdm and libraries keep exactly their existing console behaviour.
        return getattr(self.console, name)


class _TerminalProgressTeeStream:
    """Show raw dynamic progress on the TTY and one live row in the log."""

    def __init__(self, console, sink):
        self.console = console
        self.sink = sink

    def write(self, value):
        value = str(value)
        if not value:
            return 0
        with self.sink.lock:
            self.console.write(value)
            self.sink.write_progress(value)
        return len(value)

    def flush(self):
        with self.sink.lock:
            self.console.flush()
            self.sink.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class _TerminalLogSink:
    """Append ordinary output while rewriting one dynamic progress row."""

    def __init__(self):
        self.lock = threading.RLock()
        self.log_file = None
        self.pending = []
        self._progress_offset = None
        self._progress_text = None

    @staticmethod
    def _clean_progress_text(value):
        # tqdm uses carriage returns for in-place updates and may include ANSI
        # cursor/colour controls. Keep the visible text, not terminal control
        # bytes, in the persistent plain-text log.
        value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
        candidates = []
        for line in value.replace("\n", "\r\n").split("\r"):
            cleaned = line.rstrip()
            if cleaned.strip():
                candidates.append(cleaned)
        return candidates[-1] if candidates else None

    def _remove_visible_progress(self):
        if self.log_file is None or self._progress_offset is None:
            return
        self.log_file.seek(self._progress_offset)
        self.log_file.truncate()
        self.log_file.seek(0, os.SEEK_END)
        self._progress_offset = None

    def _show_progress(self):
        if self.log_file is None or not self._progress_text:
            return
        self._remove_visible_progress()
        self.log_file.seek(0, os.SEEK_END)
        self._progress_offset = self.log_file.tell()
        timestamp = time.strftime("[%H:%M:%S] ", time.localtime(time.time() + _LOG_TIME_OFFSET_SECONDS))
        self.log_file.write(timestamp + self._progress_text)
        self.log_file.truncate()
        self.log_file.seek(0, os.SEEK_END)

    def _finish_progress(self):
        if self.log_file is None or self._progress_text is None:
            return
        if self._progress_offset is None:
            self._show_progress()
        self.log_file.seek(0, os.SEEK_END)
        self.log_file.write("\n")
        self._progress_offset = None
        self._progress_text = None

    def write(self, value):
        value = str(value)
        if self.log_file is None:
            self.pending.append(("ordinary", value))
            return
        # A regular status line takes precedence over a transient bar. Remove
        # the visible bar, append the status, and retain its latest text so a
        # later refresh/close can redraw or finalize it after the status line.
        self._remove_visible_progress()
        self.log_file.seek(0, os.SEEK_END)
        self.log_file.write(value)
        self.log_file.flush()

    def write_progress(self, value):
        if self.log_file is None:
            self.pending.append(("progress", str(value)))
            return
        cleaned = self._clean_progress_text(value)
        if cleaned is not None:
            self._progress_text = cleaned
            self._show_progress()
        if "\n" in str(value):
            self._finish_progress()
        self.log_file.flush()

    def flush(self):
        if self.log_file is not None:
            self.log_file.flush()

    def bind(self, log_path):
        with self.lock:
            if self.log_file is not None:
                return
            mode = "r+" if os.path.exists(log_path) else "w+"
            self.log_file = open(log_path, mode, buffering=1, encoding="utf-8")
            self.log_file.seek(0, os.SEEK_END)
            if self.pending:
                pending = list(self.pending)
                self.pending.clear()
                for kind, value in pending:
                    if kind == "progress":
                        self.write_progress(value)
                    else:
                        self.write(value)
            self.log_file.flush()


def _install_terminal_log():
    """Start capturing timestamped stdout/stderr before the run dir exists."""
    sink = _TerminalLogSink()

    def attach(stream):
        # Ordinary-line timestamps are added before the tee. Dynamic progress
        # bypasses that wrapper and receives its saved timestamp from the sink
        # each time the single persistent progress row is replaced.
        if isinstance(stream, _TimestampedLineStream):
            stream.stream = _TerminalTeeStream(stream.stream, sink)
            return stream
        return _TerminalTeeStream(stream, sink)

    sys.stdout = attach(sys.stdout)
    sys.stderr = attach(sys.stderr)
    return sink


_ROUTED_DEPENDENCY_NOTICES = (
    "Skipping import of cpp extensions due to incompatible torch version.",
    "No prebuilt binary for CUDA",
    "You are sending unauthenticated requests to the HF Hub.",
    "is an Enum subclass and is now natively supported by torch.compile",
    "`torch_dtype` is deprecated! Use `dtype` instead!",
)
_SETTING_LOG_ONLY_NOTICES = (
    "Unrecognized keys in `rope_parameters`",
)


class _NoticeRoutingStream:
    """Keep ordinary model-loading output visible and route known noise."""

    def __init__(self, visible, diagnostic, label):
        self.visible = visible
        self.diagnostic = diagnostic
        self.label = label
        self._routing_line = False
        self._setting_log_line = False

    def _diagnostic_is_open(self):
        return (self.diagnostic is not None and not bool(getattr(self.diagnostic, "closed", False)))

    def write(self, value):
        for piece in str(value).splitlines(keepends=True):
            setting_only = (self._setting_log_line or any(marker in piece for marker in _SETTING_LOG_ONLY_NOTICES))
            route = (self._diagnostic_is_open()
                     and not setting_only
                     and (self._routing_line or any(marker in piece for marker in _ROUTED_DEPENDENCY_NOTICES)))
            if setting_only:
                setting_log_only(piece, end="")
            elif route:
                if not self._routing_line:
                    self.diagnostic.write(f"[{self.label}] ")
                self.diagnostic.write(piece)
            else:
                self.visible.write(piece)
            self._setting_log_line = bool(setting_only and not piece.endswith(("\n", "\r")))
            self._routing_line = bool(route and not piece.endswith(("\n", "\r")))
        return len(value)

    def flush(self):
        self.visible.flush()
        if self._diagnostic_is_open():
            self.diagnostic.flush()

    def write_log_only(self, value):
        writer = getattr(self.visible, "write_log_only", None)
        if writer is not None:
            return writer(value)
        return self.diagnostic.write(value)

    def detach_diagnostic(self):
        """Make a handler-retained wrapper safe after its file is closed."""
        self._routing_line = False
        self._setting_log_line = False
        self.diagnostic = None

    def __getattr__(self, name):
        return getattr(self.visible, name)


def _restore_logging_stream(routed_stream, visible_stream):
    """Detach temporary routing streams retained by logging handlers."""
    root = logging.getLogger()
    loggers = [root]
    loggers.extend(
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    handlers = [logging.lastResort]
    for logger in loggers:
        handlers.extend(logger.handlers)

    seen = set()
    for handler in handlers:
        if handler is None or id(handler) in seen:
            continue
        seen.add(id(handler))
        if getattr(handler, "stream", None) is not routed_stream:
            continue
        # setStream flushes the old stream first. This runs while the
        # diagnostic file is still open, then points future warnings back to
        # the original console stream.
        handler.setStream(visible_stream)


@contextmanager
def _route_dependency_notices(log_path):
    if not log_path:
        yield
        return
    with open(log_path, "a", buffering=1, encoding="utf-8") as diagnostic:
        stdout = _NoticeRoutingStream(sys.stdout, diagnostic, "trainer dependency stdout")
        stderr = _NoticeRoutingStream(sys.stderr, diagnostic, "trainer dependency stderr")
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
        finally:
            _restore_logging_stream(stdout, stdout.visible)
            _restore_logging_stream(stderr, stderr.visible)
            # Some libraries cache a handler before attaching it to a logger,
            # so it cannot always be found by _restore_logging_stream. Such a
            # handler may retain this wrapper after the context closes. Leave
            # that wrapper usable, but make every future write console-only.
            stdout.detach_diagnostic()
            stderr.detach_diagnostic()
