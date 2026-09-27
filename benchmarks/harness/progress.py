"""Live benchmark progress. These lines are not scores.

Each line is flushed to stderr and, when a run directory exists, to
``progress.log``. The terminal shows the run moving; the final report stays
on stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import monotonic, strftime


def short_label(value: object, limit: int = 60) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def format_case_start(suite: str, index: int, total: int, case_id: object) -> str:
    return f"[{suite}] {index}/{total} {short_label(case_id)} ..."


def format_case_done(
    suite: str,
    index: int,
    total: int,
    case_id: object,
    *,
    passed: bool | None,
    latency_ms: float | None = None,
    detail: str | None = None,
) -> str:
    if passed is True:
        mark = "pass"
    elif passed is False:
        mark = "fail"
    else:
        mark = "done"
    latency = ""
    if isinstance(latency_ms, (int, float)) and not isinstance(latency_ms, bool):
        latency = f" {latency_ms:.0f}ms"
    extra = f" {short_label(detail, 40)}" if detail and passed is not True else ""
    return f"[{suite}] {index}/{total} {short_label(case_id)} {mark}{latency}{extra}"


class ProgressLog:
    """Append-only progress trail. Safe to call when no file is open."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._file = path.open("a", encoding="utf-8") if path is not None else None
        self.started = monotonic()

    def line(self, text: str) -> None:
        message = f"{strftime('%H:%M:%S')} {text}"
        print(message, file=sys.stderr, flush=True)
        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
