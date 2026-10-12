"""sounddevice-based microphone capture and speaker playback.

The capture stream runs inside PortAudio's own callback thread (the only
background thread this prototype uses); frames are pushed to a sink callable.
``stop()`` blocks until the stream is closed, so the main loop stays
single-threaded apart from that callback.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .interfaces import AudioCapture, AudioFrame, AudioPlayer, PcmAudio

log = logging.getLogger("jarvis.audio")

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024  # 64 ms at 16 kHz — good wake-word/VAD granularity


def _require_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError: no PortAudio lib
        raise RuntimeError(
            "sounddevice/PortAudio unavailable. Install with "
            "'pip install sounddevice' and ensure libportaudio2 is present."
        ) from exc
    return sd


class SdAudioCapture(AudioCapture):
    """Continuous microphone stream via PortAudio default input device."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE,
                 device: Optional[str] = None) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._stream = None
        self._sink: Optional[Callable[[AudioFrame], None]] = None
        self._running = False
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()

    def start(self, sink: Callable[[AudioFrame], None]) -> None:
        sd = _require_sounddevice()
        self._sink = sink
        self._error = None

        def _callback(indata, frames, time_info, status):
            if status:
                log.warning("audio input status: %s", status)
            if self._error is not None:
                return
            try:
                pcm = bytes(indata[:frames, 0].tobytes())
                self._sink(AudioFrame(pcm=pcm, sample_rate=self.sample_rate))
            except BaseException as exc:  # surface callback errors to caller
                self._error = exc

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=_callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"cannot open microphone: {exc}") from exc
        self._running = True
        log.info("microphone capture started (device=%s)", self.device or "default")

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("error closing input stream: %s", exc)
            self._stream = None
        with self._lock:
            if self._error is not None:
                err, self._error = self._error, None
                raise RuntimeError(f"audio capture callback failed: {err}") from err

    @property
    def running(self) -> bool:
        return self._running


class SdAudioPlayer(AudioPlayer):
    """PCM playback on the default output device with barge-in abort.

    Uses a non-blocking sd.play + sd.wait so ``abort()`` (called from the
    main thread when the user interrupts) can stop the stream immediately
    instead of waiting for the rest of the chunk to drain.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._stream = None
        self._lock = threading.Lock()
        self._aborted = threading.Event()

    def play(self, audio: PcmAudio) -> None:
        if not audio.pcm:
            return
        sd = _require_sounddevice()
        import numpy as np

        samples = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            with self._lock:
                if self._stream is None:
                    self._stream = sd.OutputStream(
                        samplerate=audio.sample_rate, channels=1,
                        dtype="float32", device=self.device)
                    self._stream.start()
                self._stream.write(samples)
        except Exception as exc:
            raise RuntimeError(f"audio playback failed: {exc}") from exc
        # wait for this chunk to finish before returning; abort() stops the
        # stream so the wait ends promptly on barge-in.
        try:
            st = self._stream
            if st is not None:
                while st.active and not self._aborted.is_set():
                    time.sleep(0.01)
        except Exception:  # pragma: no cover
            pass

    def abort(self) -> None:
        """Stop playback promptly and discard any queued samples."""
        sd = _require_sounddevice()
        self._aborted.set()
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()   # drops pending buffer
                    self._stream.close()
                except Exception as exc:  # pragma: no cover
                    log.debug("player abort: %s", exc)
                self._stream = None
        self._aborted.clear()
        try:
            sd.stop()                     # global safety net (all streams)
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        self.abort()
