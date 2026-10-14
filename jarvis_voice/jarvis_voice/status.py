"""Terminal status display for the voice loop.

Single-line ANSI status (no threads, no timers): every state change calls
``StatusDisplay.set_state`` which repaints one line. Falls back to plain log
lines when stdout is not a TTY (tests, CI).
"""

from __future__ import annotations

import sys
from typing import Optional

from .interfaces import AssistantState

_STATE_COLORS = {
    AssistantState.LISTENING: "\033[36m",     # cyan
    AssistantState.WAKE_DETECTED: "\033[33m",  # yellow
    AssistantState.RECORDING: "\033[32m",      # green
    AssistantState.TRANSCRIBING: "\033[35m",   # magenta
    AssistantState.THINKING: "\033[34m",       # blue
    AssistantState.SPEAKING: "\033[37;1m",     # bright white
    AssistantState.ERROR: "\033[31m",          # red
    AssistantState.STOPPED: "\033[90m",        # grey
}
_RESET = "\033[0m"


class StatusDisplay:
    def __init__(self, *, stream=None, dedupe: bool = True) -> None:
        self._stream = stream or sys.stdout
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._current: Optional[AssistantState] = None
        self.dedupe = dedupe
        self.history: list[str] = []  # captured for tests / debugging

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        if (self.dedupe and state == self._current and not detail):
            return
        self._current = state
        line = f"[{state.value}] {detail}".rstrip()
        self.history.append(line)
        if self._tty:
            color = _STATE_COLORS.get(state, "")
            self._stream.write(f"\r\033[K{color}{line:<60}{_RESET}")
        else:
            self._stream.write(line + "\n")
        self._stream.flush()

    def info(self, message: str) -> None:
        """Transient info line that does not change state."""
        if self._tty:
            self._stream.write(f"\r\033[K{message}\n")
            self.set_state(self._current or AssistantState.LISTENING)
        else:
            self._stream.write(message + "\n")
        self._stream.flush()

    def finish(self) -> None:
        if self._tty:
            self._stream.write("\r\033[K\n")
        self._stream.flush()
