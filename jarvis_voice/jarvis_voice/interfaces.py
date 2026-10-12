"""Abstract interfaces for every stage of the JARVIS voice pipeline.

Each stage (audio capture, wake-word detection, speech-to-text, LLM, TTS,
playback) is hidden behind a small ABC so implementations can be swapped or
mocked in tests without touching the interaction loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AssistantState(str, Enum):
    """States reported by the terminal status display."""

    LISTENING = "LISTENING"
    WAKE_DETECTED = "WAKE_DETECTED"
    RECORDING = "RECORDING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class AudioFrame:
    """One block of mono PCM audio delivered by the capture backend."""

    pcm: bytes  # 16-bit signed little-endian mono samples
    sample_rate: int


class AudioCapture(ABC):
    """Push-based microphone access. Implementations call ``sink`` per frame."""

    @abstractmethod
    def start(self, sink) -> None:
        """Begin streaming frames to ``sink(frame: AudioFrame)``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the stream and release the device."""

    @property
    @abstractmethod
    def running(self) -> bool: ...


class WakeWordDetector(ABC):
    """Consumes raw audio frames and reports wake-word hits.

    ``process`` must be cheap and non-blocking; it is called once per frame
    from the capture callback context.
    """

    @abstractmethod
    def process(self, frame: AudioFrame) -> bool:
        """Return True when the configured wake word was detected."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal buffers/hysteresis after a hit or state change."""

    @property
    @abstractmethod
    def wake_word(self) -> str: ...

    def close(self) -> None:  # pragma: no cover - optional hook
        """Release model resources."""


class SpeechToText(ABC):
    """Transcribes a complete utterance (mono 16-bit PCM at ``sample_rate``)."""

    @abstractmethod
    def transcribe(self, pcm: bytes, sample_rate: int, language: str = "en") -> str:
        """Return the transcript; empty string means nothing was heard."""


class LLMBackend(ABC):
    """Replaceable conversational text generator."""

    @abstractmethod
    def generate(self, user_text: str, history: list[dict]) -> str:
        """Generate an English reply given conversation history.

        ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``
        messages preceding ``user_text``.
        """


class TTSSynthesisResult:
    """Placeholder type kept explicit for clarity in signatures."""


class TextToSpeech(ABC):
    """Synthesizes English speech; returns playable PCM."""

    @abstractmethod
    def synthesize(self, text: str, language: str = "en-US") -> "PcmAudio":
        """Render ``text`` to PCM audio."""

    @property
    @abstractmethod
    def name(self) -> str: ...


@dataclass
class PcmAudio:
    """PCM audio produced by a TTS backend, ready for playback."""

    pcm: bytes  # 16-bit signed little-endian mono
    sample_rate: int

    def duration_seconds(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate if self.sample_rate else 0.0


class AudioPlayer(ABC):
    """Plays PCM audio to the system output."""

    @abstractmethod
    def play(self, audio: PcmAudio) -> None:
        """Blocking playback of one buffer."""


@dataclass
class VadConfig:
    """Energy-based voice-activity-detection tuning for utterance capture."""

    silence_threshold_rms: float = 500.0     # RMS below this counts as silence
    silence_seconds: float = 1.2             # trailing silence ends utterance
    startup_timeout_seconds: float = 6.0     # give up if speech never starts
    max_utterance_seconds: float = 30.0      # hard ceiling


@dataclass
class UtteranceBuffer:
    """Accumulates post-wake frames and applies VAD endpointing.

    Pure logic (no audio devices) so it is trivially unit-testable.
    """

    vad: VadConfig
    sample_rate: int = 16000
    _frames: list[bytes] = field(default_factory=list)
    _speech_seen: bool = False
    _silence_run: float = 0.0
    _elapsed: float = 0.0

    def add_frame(self, pcm: bytes) -> Optional[str]:
        """Feed one frame; returns 'silence' | 'max' | 'started' markers or None.

        Returns ``"end"`` when the utterance is finished, else ``None``.
        """
        from .audio_utils import rms

        chunk_seconds = (len(pcm) / 2) / self.sample_rate
        self._elapsed += chunk_seconds
        level = rms(pcm)
        if level >= self.vad.silence_threshold_rms:
            self._speech_seen = True
            self._silence_run = 0.0
        else:
            self._silence_run += chunk_seconds

        self._frames.append(pcm)

        if not self._speech_seen and self._elapsed >= self.vad.startup_timeout_seconds:
            return "timeout"
        if self._speech_seen and self._silence_run >= self.vad.silence_seconds:
            return "end"
        if self._elapsed >= self.vad.max_utterance_seconds:
            return "max"
        return None

    def get_pcm(self) -> bytes:
        return b"".join(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._speech_seen = False
        self._silence_run = 0.0
        self._elapsed = 0.0
