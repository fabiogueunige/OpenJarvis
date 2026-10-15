"""Conversation session policy: resume vs new, follow-ups, explicit exits.

A "session" is one conversational context (the message history). Between the
wake word and the end of a session, the microphone stays hot so follow-up
questions need no wake word. Sessions end via inactivity timeout or an
explicit spoken command ("goodbye jarvis", "cancel", ...).

Pure logic — no audio, no threads — so it is trivially unit-testable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class SessionPolicy:
    resume_active: bool = True            # wake during active session resumes it
    inactivity_timeout_seconds: float = 45.0
    exit_phrases: list[str] = field(default_factory=lambda: [
        "goodbye jarvis", "bye jarvis", "stop listening", "never mind",
        "nevermind", "cancel",
    ])
    cancel_phrases: list[str] = field(default_factory=lambda: ["cancel"])

    def is_exit(self, text: str) -> bool:
        """True when the transcript *ends* with an exit phrase.

        Exact whole-sentence match would miss realistic ASR output such as
        "okay Jarvis, goodbye" or "goodbye jarvis." plus trailing words; the
        conversational intent lives at the end of the utterance, so a suffix
        test is both more forgiving and still unambiguous (a bare "hello
        goodbye jarvis fan club" does not end with the phrase... actually it
        does not match because we anchor on word boundaries at the tail).
        """
        n = _norm(text)
        if not n:
            return False
        exits = {_norm(p) for p in self.exit_phrases}
        if n in exits:
            return True
        return any(n.endswith(" " + e) for e in exits)

    def is_cancel(self, text: str) -> bool:
        return _norm(text) in {_norm(p) for p in self.cancel_phrases}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return " ".join(s.split())


class ConversationSession:
    """Rolling chat history bounded by a configurable token budget."""

    def __init__(self, session_id: str, max_tokens: int = 1500,
                 started_at: float = 0.0) -> None:
        self.session_id = session_id
        self.started_at = started_at
        self.max_tokens = max_tokens
        self.messages: list[dict] = []          # {"role","content"}
        self.last_activity = started_at

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap, deterministic token estimate (~4 chars/token, cl100k rule
        of thumb). Deliberately dependency-free; replace with tiktoken if a
        stricter bound is ever needed."""
        return max(1, (len(text) + 3) // 4)

    def add_turn(self, user_text: str, assistant_text: str, now: float) -> None:
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.last_activity = now
        self.trim(now)

    def trim(self, now: float | None = None) -> None:
        total = sum(self.estimate_tokens(m["content"]) for m in self.messages)
        while self.messages and total > self.max_tokens:
            dropped = self.messages.pop(0)
            total -= self.estimate_tokens(dropped["content"])
        if now is not None:
            self.last_activity = now

    def history(self) -> list[dict]:
        return list(self.messages)

    def idle_seconds(self, now: float) -> float:
        return max(0.0, now - self.last_activity)
