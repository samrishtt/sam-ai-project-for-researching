"""
SAM-AI  ·  Structured Logging Facility
========================================
Provides hierarchical, color-coded console logging and optional
JSON-lines file logging for experiment reproducibility.
"""

import datetime
import json
import os
import sys
from enum import IntEnum
from typing import Any, Dict, Optional


# ── Severity levels ──────────────────────────────────────────────
class LogLevel(IntEnum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    REASONING = 25   # custom level for reasoning traces
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# ── ANSI colour codes (graceful no-op on non-TTY) ───────────────
_COLORS = {
    LogLevel.TRACE:     "\033[90m",      # grey
    LogLevel.DEBUG:     "\033[36m",       # cyan
    LogLevel.INFO:      "\033[32m",       # green
    LogLevel.REASONING: "\033[35m",       # magenta
    LogLevel.WARNING:   "\033[33m",       # yellow
    LogLevel.ERROR:     "\033[31m",       # red
    LogLevel.CRITICAL:  "\033[1;31m",     # bold red
}
_RESET = "\033[0m"


def _colorize(text: str, level: LogLevel) -> str:
    """Wrap *text* in ANSI colour if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{_COLORS.get(level, '')}{text}{_RESET}"


# ── Core Logger ──────────────────────────────────────────────────
class SAMLogger:
    """Lightweight, purpose-built logger for the SAM-AI pipeline.

    Parameters
    ----------
    name : str
        Logger namespace (e.g. ``"ReasoningEngine"``).
    level : LogLevel
        Minimum severity to emit.
    log_dir : str or None
        If given, JSON-lines logs are also written to this directory.
    """

    _instances: Dict[str, "SAMLogger"] = {}

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.DEBUG,
        log_dir: Optional[str] = None,
    ):
        self.name = name
        self.level = level
        self.log_dir = log_dir
        self._file_handle = None

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"{name.lower().replace(' ', '_')}.jsonl")
            self._file_handle = open(path, "a", encoding="utf-8")

    # ── factory ──────────────────────────────────────────────────
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: LogLevel = LogLevel.DEBUG,
        log_dir: Optional[str] = None,
    ) -> "SAMLogger":
        """Return a singleton logger for *name*."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, level, log_dir)
        return cls._instances[name]

    # ── internal emit ────────────────────────────────────────────
    def _emit(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        if level < self.level:
            return

        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_tag = level.name.ljust(9)

        # Console output
        header = _colorize(f"[{ts}] {level_tag} | {self.name}", level)
        print(f"{header} | {message}")
        if data:
            for key, val in data.items():
                print(f"    {_colorize('→', level)} {key}: {val}")

        # File output
        if self._file_handle:
            record = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "level": level.name,
                "logger": self.name,
                "message": message,
            }
            if data:
                record["data"] = data
            self._file_handle.write(json.dumps(record) + "\n")
            self._file_handle.flush()

    # ── public convenience methods ───────────────────────────────
    def trace(self, msg: str, **kw):
        self._emit(LogLevel.TRACE, msg, kw or None)

    def debug(self, msg: str, **kw):
        self._emit(LogLevel.DEBUG, msg, kw or None)

    def info(self, msg: str, **kw):
        self._emit(LogLevel.INFO, msg, kw or None)

    def reasoning(self, msg: str, **kw):
        self._emit(LogLevel.REASONING, msg, kw or None)

    def warning(self, msg: str, **kw):
        self._emit(LogLevel.WARNING, msg, kw or None)

    def error(self, msg: str, **kw):
        self._emit(LogLevel.ERROR, msg, kw or None)

    def critical(self, msg: str, **kw):
        self._emit(LogLevel.CRITICAL, msg, kw or None)

    # ── cleanup ──────────────────────────────────────────────────
    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
