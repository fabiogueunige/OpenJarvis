"""TTS backends: Piper (fast, CPU) and Kokoro (quality), both English-only here.

Both return in-memory PcmAudio; nothing is written to disk. The module also
exposes ``evaluate_backends`` used by ``python -m jarvis_voice.bench_tts`` to
compare latency/quality on the target machine before committing a choice.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Optional

from ..interfaces import PcmAudio, TextToSpeech

log = logging.getLogger("jarvis.tts")


class PiperTTS(TextToSpeech):
    """piper-tts via CLI subprocess (pip install piper-tts).

    Very low CPU latency, intelligible but slightly robotic voice. Model must
    be downloaded once (see INSTALL.md); no audio files are persisted — we read
    stdout as raw PCM 44.1kHz mono int16 (piper's native output format).
    """

    def __init__(self, model_path: str, voice_name: str = "en_US-lessac-medium",
                 config_path: Optional[str] = None) -> None:
        if not shutil.which("piper"):
            raise RuntimeError(
                "piper CLI not found. Install with 'pip install piper-tts' inside "
                "the venv."
            )
        self.model_path = model_path
        self.config_path = config_path or (model_path.rsplit(".", 1)[0] + ".json")
        self._voice = voice_name

    @property
    def name(self) -> str:
        return f"piper:{self._voice}"

    def synthesize(self, text: str, language: str = "en-US") -> PcmAudio:
        if not text.strip():
            return PcmAudio(pcm=b"", sample_rate=44100)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["piper", "--model", self.model_path, "--config", self.config_path,
                 "--output-raw"],
                input=text.encode(), capture_output=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("piper synthesis timed out (60 s)") from exc
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"piper failed: {proc.stderr.decode(errors='replace')[:200]}")
        ms = (time.perf_counter() - t0) * 1000
        log.debug("piper synthesized %d chars in %.0f ms", len(text), ms)
        return PcmAudio(pcm=proc.stdout, sample_rate=44100)


class KokoroTTS(TextToSpeech):
    """kokoro-onnx via Python API (pip install kokoro-onnx soundfile).

    Near-neural quality at moderate CPU cost; outputs float32 WAV bytes which
    we convert to int16 PCM in memory. Requires espeak-ng data files available
    on the system (Ubuntu: apt install espeak-ng — ask before installing).
    Voice codes starting with 'a'/'b' map to American English, 'e'/'f' British.
    """

    def __init__(self, model_path: str = "~/.local/share/jarvis/kokoro/v1.0.onnx",
                 voices_path: str = "~/.local/share/jarvis/kokoro/voices.bin",
                 espeak_data_path: Optional[str] = None,
                 voice: str = "af_heart") -> None:
        import os

        self.model_path = os.path.expanduser(model_path)
        self.voices_path = os.path.expanduser(voices_path)
        self.espeak_data_path = espeak_data_path
        self.voice = voice
        self._engine = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx not installed (pip install kokoro-onnx soundfile)"
            ) from exc
        import os

        if not (os.path.exists(self.model_path) and os.path.exists(self.voices_path)):
            raise RuntimeError(
                f"Kokoro model files missing at {self.model_path} / "
                f"{self.voices_path}. Download them first (see INSTALL.md)."
            )
        kwargs = {}
        if self.espeak_data_path:
            kwargs["espeak_data_path"] = self.espeak_data_path
        t0 = time.perf_counter()
        self._engine = Kokoro(self.model_path, self.voices_path, **kwargs)
        log.info("kokoro engine loaded in %.1fs", time.perf_counter() - t0)
        return self._engine

    @property
    def name(self) -> str:
        return f"kokoro:{self.voice}"

    def synthesize(self, text: str, language: str = "en-US") -> PcmAudio:
        if not text.strip():
            return PcmAudio(pcm=b"", sample_rate=24000)
        engine = self._ensure_engine()
        t0 = time.perf_counter()
        samples, sr = engine.create(text, voice=self.voice, lang="en-us")
        ms = (time.perf_counter() - t0) * 1000
        log.debug("kokoro synthesized %d chars in %.0f ms", len(text), ms)
        import numpy as np

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return PcmAudio(pcm=pcm, sample_rate=int(sr))


class ScriptedTTS(TextToSpeech):
    """Emits a short beep — for --demo mode and unit tests."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self._sr = sample_rate

    @property
    def name(self) -> str:
        return "scripted-beep"

    def synthesize(self, text: str, language: str = "en-US") -> PcmAudio:
        import math
        import struct

        n = int(0.25 * self._sr)
        pcm = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / self._sr)))
            for i in range(n)
        )
        return PcmAudio(pcm=pcm, sample_rate=self._sr)
