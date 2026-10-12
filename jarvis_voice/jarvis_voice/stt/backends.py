"""Speech-to-text backends.

faster-whisper is the primary reliable English STT for this prototype.
The interface accepts raw PCM so no WAV file ever touches disk (requirement:
no raw audio stored by default).
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

from ..interfaces import SpeechToText

log = logging.getLogger("jarvis.stt")


class FasterWhisperSTT(SpeechToText):
    """Local faster-whisper transcription (lazy model load on first use)."""

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "auto", download_root: Optional[str] = None,
                 beam_size: int = 1) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.beam_size = beam_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper not installed (pip install faster-whisper==1.1.*"
                ") or its CTranslate2 native library is missing."
            ) from exc

        device, compute_type = self._resolve_device()
        log.info("loading faster-whisper '%s' on %s (%s)...",
                 self.model_size, device, compute_type)
        try:
            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute_type,
                download_root=self.download_root, local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"faster-whisper model '{self.model_size}' could not be loaded: "
                f"{exc}. Check network/model cache or pick another size."
            ) from exc
        return self._model

    def _resolve_device(self) -> tuple[str, str]:
        if self.device != "auto":
            return self.device, self.compute_type
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", (self.compute_type
                                if self.compute_type != "auto" else "float16")
        except Exception:
            pass
        return "cpu", (self.compute_type
                       if self.compute_type != "auto" else "int8")

    def transcribe(self, pcm: bytes, sample_rate: int, language: str = "en") -> str:
        if not pcm:
            return ""
        model = self._ensure_model()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        buf.seek(0)
        segments, info = model.transcribe(
            buf, language=language, beam_size=self.beam_size,
            vad_filter=True, without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        log.debug("stt lang=%s prob=%.2f text=%r", info.language,
                  info.language_probability, text)
        return text


class ScriptedSTT(SpeechToText):
    """Returns canned transcripts per call — used by --demo and unit tests."""

    def __init__(self, scripts: Optional[list[str]] = None) -> None:
        self._scripts = list(scripts or ["What is the weather today?"])
        self._i = 0

    def transcribe(self, pcm: bytes, sample_rate: int, language: str = "en") -> str:
        text = self._scripts[self._i % len(self._scripts)]
        self._i += 1
        return text
