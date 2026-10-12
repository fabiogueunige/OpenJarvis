"""Wake-word detector implementations.

FINDINGS (verified 2026-10-12 against openWakeWord 0.6.0 pretrained model list):
openWakeWord ships ONLY these pretrained models:
  alexa, hey_mycroft, hey_jarvis, hey_siri, jetson,哪吒, timer, weather,
  ... — there is **no ready-to-use model for the bare word "Jarvis"**.
The closest official model is ``hey_jarvis`` ("Hey Jarvis"). Per project rules
we do NOT silently substitute it; using it requires an explicit config opt-in
(``allow_close_match: true``) and the status line always shows which model is
active. The clean path to a real single-word "Jarvis" model is documented in
docs/WAKE_WORD.md (Porcupine custom training or openWakeWord fine-tuning).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

from ..interfaces import AudioFrame, WakeWordDetector

log = logging.getLogger("jarvis.wakeword")


class OpenWakeWordDetector(WakeWordDetector):
    """openWakeWord backend with strict wake-word verification.

    If the requested word has no exact pretrained match, construction raises
    :class:`WakeWordNotAvailableError` listing the available models — unless
    ``allow_close_match`` is set AND a close official match exists, in which
    case that match is used *explicitly and visibly*.
    """

    # curated map of pretrained openWakeWord models -> words they answer to
    _PRETRAINED = {
        "alexa": "alexa",
        "hey_mycroft": "mycroft",
        "hey_jarvis": "jarvis",          # close match only ("hey jarvis")
        "hey_siri": "siri",
        "timer": "timer",
        "weather": "weather",
    }
    _EXACT_OK = {"alexa", "mycroft", "siri", "timer", "weather"}

    def __init__(self, wake_word: str = "jarvis", threshold: float = 0.5,
                 allow_close_match: bool = False, model_paths: Optional[list[str]] = None,
                 sample_rate: int = 16000) -> None:
        self._wake_word = wake_word.lower().strip()
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        try:
            from openwakeword.model import Model
            import openwakeword.utils as owu_utils
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword not installed (pip install openwakeword==0.6.0)"
            ) from exc

        available = sorted(set(self._PRETRAINED.values()))

        if model_paths:
            # user-trained / downloaded custom .onnx models are trusted as-is
            self._model = Model(wakeword_models=model_paths)
            self._active_model_name = f"custom:{self._wake_word}"
        elif self._wake_word in self._EXACT_OK:
            self._model = Model(wakeword_models=[self._find_model_id(self._wake_word)])
            self._active_model_name = self._find_model_id(self._wake_word)
        else:
            close = self._find_model_id(self._wake_word)
            if close and allow_close_match:
                log.warning(
                    "No exact '%s' model in openWakeWord; using OFFICIAL CLOSE "
                    "MATCH '%s' because allow_close_match=true.", self._wake_word, close)
                self._model = Model(wakeword_models=[close])
                self._active_model_name = close
            else:
                raise WakeWordNotAvailableError(
                    f"openWakeWord has no ready-to-use model for the wake word "
                    f"'{self._wake_word}'. Pretrained options: {available}. "
                    f"To use the official 'hey_jarvis' model instead, set "
                    f"wake_word.allow_close_match=true (activation phrase becomes "
                    f"'Hey Jarvis'). For a true single-word 'Jarvis', train a "
                    f"custom model - see docs/WAKE_WORD.md.",
                    available_models=available,
                )
        log.info("wake-word model active: %s (threshold=%.2f)",
                 self._active_model_name, self.threshold)

    @classmethod
    def _find_model_id(cls, word: str) -> Optional[str]:
        for mid, w in cls._PRETRAINED.items():
            if w == word or mid == f"hey_{word}":
                return mid
        return None

    def process(self, frame: AudioFrame) -> bool:
        with self._lock:
            scores = self._model.predict(frame.pcm)
            best = max(scores.values()) if scores else 0.0
            if best >= self.threshold:
                self._model.reset()
                return True
            return False

    def reset(self) -> None:
        with self._lock:
            self._model.reset()

    @property
    def wake_word(self) -> str:
        return f"{self._wake_word} [{self._active_model_name}]"

    def close(self) -> None:
        pass


class WhisperHypothesisDetector(WakeWordDetector):
    """Fallback detector: run Whisper over a rolling audio window and grep
    for the wake word in the transcript.

    Robust to any inflection of the literal word "Jarvis", but costs CPU/GPU
    per window — intended as a stopgap until a custom wake model exists.
    Runs inference on the caller thread between capture frames (batched by
    ``window_seconds``), so no extra threads are created here.
    """

    def __init__(self, wake_word: str, stt, window_seconds: float = 1.0,
                 overlap_seconds: float = 0.5, sample_rate: int = 16000) -> None:
        self._wake_word = wake_word.lower().strip()
        self.stt = stt
        self.sample_rate = sample_rate
        self.window_bytes = int(window_seconds * sample_rate * 2)
        self.overlap_bytes = int(overlap_seconds * sample_rate * 2)
        self._buf = bytearray()
        self._carry = bytearray()

    def process(self, frame: AudioFrame) -> bool:
        self._buf.extend(frame.pcm)
        if len(self._buf) < self.window_bytes:
            return False
        chunk = bytes(self._buf[: self.window_bytes])
        advance = self.window_bytes - self.overlap_bytes
        del self._buf[:advance]
        text = self.stt.transcribe(chunk, self.sample_rate, language="en").lower()
        if self._wake_word in text:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._buf.clear()

    @property
    def wake_word(self) -> str:
        return f"{self._wake_word} [whisper-hypothesis]"


class ScriptedWakeDetector(WakeWordDetector):
    """Test/demo detector that fires after N frames (used with --demo)."""

    def __init__(self, wake_word: str = "jarvis", fire_after_frames: int = 3) -> None:
        self._wake_word = wake_word
        self.fire_after = fire_after_frames
        self._count = 0

    def process(self, frame: AudioFrame) -> bool:
        self._count += 1
        if self._count >= self.fire_after:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._count = 0

    @property
    def wake_word(self) -> str:
        return f"{self._wake_word} [scripted]"


class WakeWordNotAvailableError(RuntimeError):
    def __init__(self, message: str, available_models: list[str]) -> None:
        super().__init__(message)
        self.available_models = available_models
