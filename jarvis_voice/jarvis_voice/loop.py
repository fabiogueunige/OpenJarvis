"""The voice interaction loop: wake -> record (VAD) -> STT -> LLM -> TTS -> play.

Design notes
------------
* One audio callback thread (PortAudio). Everything else runs on the main
  thread; heavy stages are triggered from the callback via a tiny event flag,
  executed in ``run_once`` — so no thread-pool sprawl and no GPU contention
  between stages (STT/LLM/TTS never overlap by construction).
* The loop is fully driven through injected interfaces, so tests can pump
  scripted frames with fakes and assert on state transitions.
* Raw audio lives only in memory (``UtteranceBuffer``); nothing is written to
  disk unless ``store_audio = true`` is explicitly set.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .interfaces import (
    AssistantState, AudioCapture, AudioFrame, AudioPlayer, LLMBackend,
    SpeechToText, TextToSpeech, UtteranceBuffer, VadConfig, WakeWordDetector,
)
from .status import StatusDisplay

log = logging.getLogger("jarvis.loop")


class VoiceLoop:
    def __init__(self, *, capture: AudioCapture, wake: WakeWordDetector,
                 stt: SpeechToText, llm: LLMBackend, tts: TextToSpeech,
                 player: AudioPlayer, vad: VadConfig, status: StatusDisplay,
                 sample_rate: int = 16000, language: str = "en",
                 max_history_turns: int = 8,
                 store_audio_path=None) -> None:
        self.capture = capture
        self.wake = wake
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.player = player
        self.status = status
        self.language = language
        self.history: list[dict] = []
        self._max_msgs = max_history_turns * 2
        self._buf = UtteranceBuffer(vad=vad, sample_rate=sample_rate)
        self._events: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._state = AssistantState.LISTENING
        self._busy = False   # True while a heavy stage owns the pipeline
        self.store_audio_path = store_audio_path  # pathlib.Path or None (default)
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> AssistantState:
        return self._state

    def _set_state(self, state: AssistantState, detail: str = "") -> None:
        self._state = state
        if state == AssistantState.ERROR:
            self.last_error = detail
        self.status.set_state(state, detail)

    # ------------------------------------------------------------- audio sink
    def _on_frame(self, frame: AudioFrame) -> None:
        """Called from the PortAudio callback thread. Must stay cheap.

        Heavy stages (STT/LLM/TTS/playback) run on the main thread; while a
        reply is being produced or spoken the state is outside
        LISTENING/RECORDING, so frames are simply ignored (no buffering).
        ``_busy`` additionally guards the tiny window between endpointing an
        utterance and the main thread starting to process it.
        """
        if self._stop.is_set() or self._busy:
            return
        if self._state == AssistantState.LISTENING:
            if self.wake.process(frame):
                # Flip synchronously inside the callback so no further frames
                # keep hitting the wake detector before the main thread drains
                # the event queue; the queued "wake" event drives the visible
                # WAKE_DETECTED -> RECORDING transition.
                self._state = AssistantState.WAKE_DETECTED
                self._events.put("wake")
        elif self._state in (AssistantState.WAKE_DETECTED, AssistantState.RECORDING):
            marker = self._buf.add_frame(frame.pcm)
            if marker is not None:
                self._busy = True   # stop feeding; processing starts shortly
                self._events.put(marker)

    # ------------------------------------------------------------ transitions
    def handle_wake(self) -> None:
        self._busy = False
        """Wake detected: acknowledge visibly, then start utterance capture.

        RECORDING becomes active immediately so the frames still queued in
        the event pipeline (delivered while we were in LISTENING) are kept.
        """
        self._set_state(AssistantState.WAKE_DETECTED, f"word={self.wake.wake_word}")
        self.wake.reset()
        self._buf.reset()
        self._set_state(AssistantState.RECORDING, "speak now")

    def finish_utterance(self, reason: str) -> None:
        # Heavy stages run on the main thread; during them the sink ignores
        # incoming frames, but anything already queued must be dropped so we
        # never re-enter RECORDING with stale post-reply audio.
        self._drain_pending()
        pcm = self._buf.get_pcm()
        self._buf.reset()
        if reason == "timeout":
            self._set_state(AssistantState.LISTENING, "no speech heard, back to wake")
            self._busy = False
            return
        self._set_state(AssistantState.TRANSCRIBING)
        try:
            text = self.stt.transcribe(pcm, self.capture_sr, self.language).strip()
            if self.store_audio_path is not None:
                # Opt-in only (config store_audio=true); off by default so raw
                # microphone audio never touches disk.
                import wave
                from datetime import datetime

                fn = datetime.now().strftime("utterance-%Y%m%d-%H%M%S.wav")
                with wave.open(str(self.store_audio_path / fn), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.capture_sr)
                    w.writeframes(pcm)
        except Exception as exc:
            self._fail(f"STT error: {exc}")
            return
        finally:
            del pcm  # drop audio reference promptly (privacy default)
        if not text:
            self._set_state(AssistantState.LISTENING, "empty transcription")
            self._busy = False
            return
        self.respond(text)

    def respond(self, user_text: str) -> None:
        self._set_state(AssistantState.THINKING, f'"{user_text}"')
        try:
            reply = self.llm.generate(user_text, list(self.history)).strip()
        except Exception as exc:
            self._fail(f"LLM error: {exc}")
            return
        if not reply:
            self._set_state(AssistantState.LISTENING, "empty LLM reply")
            self._busy = False
            return
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-self._max_msgs:]
        self._set_state(AssistantState.SPEAKING, reply[:60])
        try:
            audio = self.tts.synthesize(reply, "en-US")
            self._drain_pending()   # never speak over a newer wake request
            self.player.play(audio)
        except Exception as exc:
            self._fail(f"TTS/playback error: {exc}")
            return
        self._drain_pending()
        self._set_state(AssistantState.LISTENING, "awaiting wake word")
        self._busy = False

    def _fail(self, message: str) -> None:
        log.error(message)
        self._set_state(AssistantState.ERROR, message)
        # recover automatically into LISTENING unless stopped
        if not self._stop.is_set():
            self._drain_pending()
            self._set_state(AssistantState.LISTENING, "recovered")
            self._busy = False

    def _drain_pending(self) -> None:
        """Discard events queued while a heavy stage was running."""
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break

    # ---------------------------------------------------------------- driving
    capture_sr = 16000  # overwritten by run(); kept for tests

    def drain_events(self) -> bool:
        """Process all queued events. Returns False when stopping."""
        processed_any = False
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            processed_any = True
            if ev == "wake":
                self.handle_wake()
            elif ev in ("end", "max", "timeout"):
                if self._state in (AssistantState.WAKE_DETECTED,
                                   AssistantState.RECORDING):
                    self.finish_utterance(ev)
        return processed_any

    def run(self) -> None:
        """Blocking main loop until stop() or KeyboardInterrupt."""
        self.capture_sr = getattr(self.capture, "sample_rate", 16000)
        self._set_state(AssistantState.LISTENING,
                        f"say '{self.wake.wake_word}'")
        try:
            self.capture.start(self._on_frame)
        except RuntimeError as exc:
            self._set_state(AssistantState.ERROR, f"microphone: {exc}")
            raise
        try:
            while not self._stop.is_set():
                if not self.drain_events():
                    break
                self._stop.wait(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        try:
            self.capture.stop()
        except Exception as exc:  # pragma: no cover
            log.warning("capture stop error: %s", exc)
        self.wake.close()
        self._set_state(AssistantState.STOPPED, "bye")
        self.status.finish()
