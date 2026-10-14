"""Fixtures and helpers shared by unit tests (no real audio devices/models)."""

from __future__ import annotations

import struct
import math
from typing import Optional

from jarvis_voice.interfaces import (
    AudioCapture, AudioFrame, AudioPlayer, LLMBackend, PcmAudio,
    SpeechToText, TextToSpeech, WakeWordDetector,
)


def tone_pcm(seconds: float, sample_rate: int = 16000, amp: int = 8000,
             freq: float = 440.0) -> bytes:
    n = int(seconds * sample_rate)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(n)
    )


def silence_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


class FakeCapture(AudioCapture):
    """Replays a scripted list of frames on demand; no threads."""

    def __init__(self, frames: Optional[list[bytes]] = None,
                 sample_rate: int = 16000) -> None:
        self.frames = frames or []
        self.sample_rate = sample_rate
        self._sink = None
        self._running = False
        self.fail_on_start = False

    def start(self, sink) -> None:
        if self.fail_on_start:
            raise RuntimeError("microphone busy / device unavailable")
        self._sink = sink
        self._running = True

    def pump(self, count: int) -> None:
        """Deliver the next ``count`` scripted frames to the sink."""
        for _ in range(count):
            if not self.frames:
                return
            pcm = self.frames.pop(0)
            self._sink(AudioFrame(pcm=pcm, sample_rate=self.sample_rate))

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


class FakeWake(WakeWordDetector):
    def __init__(self, wake_word="jarvis", fire_after_calls: int = 1) -> None:
        self._wake_word = wake_word
        self.fire_after = fire_after_calls
        self.calls = 0
        self.resets = 0

    def process(self, frame: AudioFrame) -> bool:
        self.calls += 1
        if self.calls >= self.fire_after:
            self.calls = 0
            return True
        return False

    def reset(self) -> None:
        self.resets += 1
        self.calls = 0

    @property
    def wake_word(self) -> str:
        return f"{self._wake_word} [fake]"


class FakeSTT(SpeechToText):
    def __init__(self, scripts: Optional[list[str]] = None) -> None:
        self.scripts = list(scripts or ["hello world"])
        self.calls = 0

    def transcribe(self, pcm: bytes, sample_rate: int, language: str = "en") -> str:
        text = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return text


class FakeLLM(LLMBackend):
    def __init__(self, reply: str = "Sure, here is an English answer.",
                 fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.seen_history: list[list[dict]] = []

    def generate(self, user_text: str, history: list[dict]) -> str:
        if self.fail:
            raise RuntimeError("llm down")
        self.seen_history.append(list(history))
        return self.reply


class FakeTTS(TextToSpeech):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    @property
    def name(self) -> str:
        return "fake-tts"

    def synthesize(self, text: str, language: str = "en-US") -> PcmAudio:
        self.spoken.append(text)
        return PcmAudio(pcm=tone_pcm(0.05), sample_rate=22050)


class FakePlayer(AudioPlayer):
    def __init__(self) -> None:
        self.played: list[PcmAudio] = []
        self.aborts = 0

    def play(self, audio: PcmAudio) -> None:
        self.played.append(audio)

    def abort(self) -> None:
        self.aborts += 1
