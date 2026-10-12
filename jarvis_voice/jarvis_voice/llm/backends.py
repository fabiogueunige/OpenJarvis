"""LLM backends. Ollama is first-class; the interface keeps it replaceable."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Iterator, Optional

from ..interfaces import CancelToken, LLMBackend, StreamingLLMBackend

log = logging.getLogger("jarvis.llm")

SYSTEM_PROMPT = (
    "You are JARVIS, a helpful English-speaking voice assistant on Ubuntu. "
    "Always answer in English only. Keep replies short and conversational "
    "(1-3 sentences) because they will be read aloud. If the user asks for "
    "details, expand with a few more complete sentences. Never claim to have "
    "performed an action on the user's computer; this prototype has no "
    "system access. Never ask to run commands."
)

FAST_SYSTEM_PROMPT = (
    "You are JARVIS, an English voice assistant. Answer in ONE short spoken "
    "sentence. Plain text, English only."
)


class OllamaLLM(StreamingLLMBackend):
    """Ollama /api/chat backend over stdlib HTTP (no extra dependency).

    ``stream()`` consumes the NDJSON response incrementally so TTS can start
    before generation finishes; ``generate()`` stays available for callers
    that want the whole reply. Both poll the shared CancelToken between reads
    and close the connection promptly when cancelled.
    """

    def __init__(self, model: str = "qwen3:4b",
                 base_url: str = "http://localhost:11434",
                 timeout_seconds: float = 60.0,
                 keep_alive: str = "10m",
                 temperature: float = 0.7,
                 num_ctx: int = 8192) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.num_ctx = num_ctx

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------- internals
    def _open(self, user_text: str, history: list[dict], *, stream: bool,
              fast_path: bool = False):
        messages = [{"role": "system",
                     "content": FAST_SYSTEM_PROMPT if fast_path else SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url} ({exc}). "
                "Start it with 'ollama serve' and pull the model "
                f"'{self.model}' (ollama pull {self.model})."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama timed out after {self.timeout}s") from exc

    # ------------------------------------------------------------------- api
    def stream(self, user_text: str, history: list[dict],
               cancel: Optional[CancelToken] = None,
               fast_path: bool = False) -> Iterator[str]:
        resp = self._open(user_text, history, stream=True, fast_path=fast_path)
        deadline = time.monotonic() + self.timeout
        pending_think = ""   # carry partial <think> tags across deltas
        try:
            for raw in resp:
                if cancel is not None and cancel.cancelled:
                    log.info("LLM stream cancelled — closing connection")
                    return
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Ollama stream stalled past {self.timeout}s")
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                delta = (obj.get("message") or {}).get("content", "")
                if not delta:
                    continue
                chunk = pending_think + delta
                cleaned, pending_think = _strip_thinking_stream(chunk)
                if cleaned:
                    yield cleaned
        finally:
            try:
                resp.close()
            except Exception:  # pragma: no cover
                pass

    def generate(self, user_text: str, history: list[dict]) -> str:
        resp = self._open(user_text, history, stream=False)
        try:
            body = json.loads(resp.read().decode())
        finally:
            resp.close()
        content = (body.get("message") or {}).get("content", "")
        return _strip_thinking(content).strip()


def _strip_thinking_stream(text: str) -> tuple[str, str]:
    """Streaming-safe think-block removal.

    Returns (safe_text_to_emit, held_back_tail). A trailing partial tag
    prefix like '<thi' is held until more tokens arrive.
    """
    text = _strip_thinking(text)
    hold = ""
    low = text.lower()
    for probe_len in range(min(6, len(text)), 0, -1):
        tail = low[-probe_len:]
        if "<think".startswith(tail) or "</think".startswith(tail):
            hold = text[len(text) - probe_len:]
            text = text[:len(text) - probe_len]
            break
    return text, hold


class EchoLLM(LLMBackend):
    """Deterministic offline backend for --demo mode and tests."""

    def __init__(self) -> None:
        self._turn = 0

    def generate(self, user_text: str, history: list[dict]) -> str:
        self._turn += 1
        return (f"You said: {user_text}. This is demo reply number {self._turn}; "
                f"I am keeping context of {len(history)} earlier messages.")


# ---------------------------------------------------------------- fast path
_SHORT_ANSWER_WORDS = {
    "yes", "no", "ok", "okay", "thanks", "thank you", "hello", "hi", "hey",
    "goodbye", "bye", "stop", "cancel", "who are you", "what time is it",
    "what day is it", "are you there", "how are you", "ping", "test",
}
_COMPLEX_HINTS = re.compile(
    r"\b(explain|why|how do i|how to|write|code|program|compare|analyze|"
    r"analyse|research|plan|step[s]? by step|in detail|summarize|summary|"
    r"translate|debug|fix|design|calculate|compute)\b")
_GREETING_REPLIES = {
    "hello": "Hello. How can I help?",
    "hi": "Hi there.",
    "hey": "Yes, I'm here.",
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
    "goodbye": "Goodbye.",
    "bye": "Bye.",
    "ping": "Pong.",
    "are you there": "Yes, I'm here.",
    "how are you": "I'm doing well, thank you.",
    "who are you": "I'm Jarvis, your voice assistant.",
}


def classify_query(user_text: str) -> str:
    """Route a transcript to 'fast' or 'standard'.

    Fast path = deterministic canned replies or a one-sentence LLM answer;
    used for greetings and trivial questions so the user hears a response
    immediately without paying full generation cost. Anything that looks
    like a real task (explain/write/compare…) goes standard. Tools/reasoning
    tiers arrive in a later milestone; 'standard' is their home.
    """
    t = re.sub(r"[^\w\s]", "", (user_text or "").lower().strip())
    t = " ".join(t.split())
    if not t:
        return "fast"
    if t in _GREETING_REPLIES or t in _SHORT_ANSWER_WORDS:
        return "fast"
    if _COMPLEX_HINTS.search(t):
        return "standard"
    words = len(t.split())
    if words <= 6 and len(t) <= 48:
        return "fast"          # short simple question: one-sentence answer
    return "standard"


def canned_reply(user_text: str) -> Optional[str]:
    key = " ".join(re.sub(r"[^\w\s]", "", user_text.lower()).split())
    return _GREETING_REPLIES.get(key)


def _strip_thinking(text: str) -> str:
    """Remove chain-of-thought blocks some local models emit inline."""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # unterminated opening tag: drop everything up to closing tag or end
    if "<think" in text.lower():
        idx = text.lower().find("<think")
        end = text.lower().find("</think>", idx)
        text = text[:idx] + (text[end + 8:] if end != -1 else "")
    return text
