"""LLM backends. Ollama is first-class; the interface keeps it replaceable."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Optional

from ..interfaces import LLMBackend

log = logging.getLogger("jarvis.llm")

SYSTEM_PROMPT = (
    "You are JARVIS, a helpful English-speaking voice assistant on Ubuntu. "
    "Always answer in English only. Keep replies short and conversational "
    "(1-3 sentences) because they will be read aloud. Never ask to run "
    "commands; this prototype has no access to the user's system."
)


class OllamaLLM(LLMBackend):
    """Ollama /api/chat backend over stdlib HTTP (no extra dependency)."""

    def __init__(self, model: str = "qwen3:4b",
                 base_url: str = "http://localhost:11434",
                 timeout_seconds: float = 60.0,
                 keep_alive: str = "10m",
                 temperature: float = 0.7) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.keep_alive = keep_alive
        self.temperature = temperature

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def generate(self, user_text: str, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url} ({exc}). "
                "Start it with 'ollama serve' and pull the model "
                f"'{self.model}' (ollama pull {self.model})."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama timed out after {self.timeout}s") from exc
        content = (body.get("message") or {}).get("content", "")
        return _strip_thinking(content).strip()


class EchoLLM(LLMBackend):
    """Deterministic offline backend for --demo mode and tests."""

    def __init__(self) -> None:
        self._turn = 0

    def generate(self, user_text: str, history: list[dict]) -> str:
        self._turn += 1
        return (f"You said: {user_text}. This is demo reply number {self._turn}; "
                f"I am keeping context of {len(history)} earlier messages.")


def _strip_thinking(text: str) -> str:
    """Remove chain-of-thought blocks some local models emit inline."""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # unterminated opening tag: drop everything up to closing tag or end
    if "<think" in text.lower():
        idx = text.lower().find("<think")
        end = text.lower().find("</think>", idx)
        text = text[:idx] + (text[end + 8:] if end != -1 else "")
    return text
