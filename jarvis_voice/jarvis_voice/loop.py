"""The conversational voice loop (v2).

Pipeline per turn:
  wake -> (session resume/new) -> record (VAD endpointing) -> STT -> route
  (fast/standard) -> LLM stream -> sentence chunks -> TTS chunk -> playback,
  with barge-in monitoring while speaking and cooperative cancellation of
  downstream work when interrupted.

Design notes (preserved from v1 unless noted):
* One audio callback thread (PortAudio). Everything else runs on the main
  thread; heavy stages are triggered from the callback via a small event
  queue. No thread-pool sprawl, no GPU contention between stages.
* NEW: during SPEAKING the callback additionally evaluates frame energy for
  barge-in (pure integer RMS — microseconds), and can enqueue "barge". The
  main thread checks the same flag between playback chunks.
* Raw audio lives only in memory; nothing touches disk unless store_audio.
* Sessions: after a reply the mic stays hot (RECORDING follow-up window) so
  follow-ups need no wake word; an inactivity timeout or "goodbye jarvis"/
  "cancel" ends the session and returns to LISTENING.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .interfaces import (
    AssistantState, AudioCapture, AudioFrame, AudioPlayer, BargeInConfig,
    CancelToken, LLMBackend, PcmAudio, SpeechToText, TextToSpeech,
    UtteranceBuffer, VadConfig, WakeWordDetector,
)
from .llm.backends import canned_reply, classify_query
from .metrics import MetricsCollector
from .session import ConversationSession, SessionPolicy
from .status import StatusDisplay
from .streaming import SentenceChunker

log = logging.getLogger("jarvis.loop")


class VoiceLoop:
    def __init__(self, *, capture: AudioCapture, wake: WakeWordDetector,
                 stt: SpeechToText, llm: LLMBackend, tts: TextToSpeech,
                 player: AudioPlayer, vad: VadConfig, status: StatusDisplay,
                 sample_rate: int = 16000, language: str = "en",
                 max_history_turns: int = 8,
                 store_audio_path=None,
                 session_policy: Optional[SessionPolicy] = None,
                 barge_in: Optional[BargeInConfig] = None,
                 metrics: Optional[MetricsCollector] = None,
                 followup_window_seconds: float = 30.0,
                 max_context_tokens: int = 1500,
                 clock=time.monotonic) -> None:
        self.capture = capture
        self.wake = wake
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.player = player
        self.status = status
        self.language = language
        self._buf = UtteranceBuffer(vad=vad, sample_rate=sample_rate)
        # Frames buffered during the wake-confirmation window; replayed into
        # the utterance buffer when RECORDING opens so speech that starts in
        # the same audio block as the wake word is not lost.
        self._wake_pending: list[bytes] = []
        self._wake_pending_budget = 0.0      # seconds of parkable audio
        self._events: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._state = AssistantState.LISTENING
        self._busy = False            # True while a stage owns the pipeline
        self._wake_busy = False        # True between wake-hit and RECORDING opening
        self._in_followup = False      # True while the post-reply hot window is open
        self.store_audio_path = store_audio_path
        self.last_error: Optional[str] = None
        self.clock = clock
        self.sample_rate = sample_rate
        self.followup_window_seconds = followup_window_seconds

        self.policy = session_policy or SessionPolicy()
        self.barge_cfg = barge_in or BargeInConfig()
        self.metrics = metrics or MetricsCollector()
        self.max_context_tokens = max_context_tokens
        self.session: Optional[ConversationSession] = None
        self._max_msgs = max_history_turns * 2   # hard cap even if tokens allow more

        # ---- barge-in state (written by callback, read by main thread) ----
        self._speaking = False
        self._speak_started_at = 0.0
        self._loud_run = 0
        self._barge_frames: list[bytes] = []
        self._barge_hit = threading.Event()      # set by callback, cleared per chunk
        self._cancel: Optional[CancelToken] = None
        self.state_log: list[AssistantState] = []   # announced transitions (tests/observability)
        self.last_state_before_error = AssistantState.LISTENING

    capture_sr = 16000  # overwritten by run(); kept for tests

    @property
    def state(self) -> AssistantState:
        return self._state

    # ------------------------------------------------------------------ state
    def _set_state(self, state: AssistantState, detail: str = "") -> None:
        if state == AssistantState.ERROR:
            # Remember where we were so _fail() can decide the recovery
            # target (the ERROR announcement overwrites self._state).
            self.last_state_before_error = self._state
        self._state = state
        if state == AssistantState.ERROR:
            self.last_error = detail
        self.status.set_state(state, detail)
        # Observability hook (tests / instrumentation): record every
        # *announced* transition. The callback's synchronous LISTENING ->
        # WAKE_DETECTED flip is not an announcement and intentionally does
        # not appear here.
        self.state_log.append(state)

    # ------------------------------------------------------------ audio sink
    def _on_frame(self, frame: AudioFrame) -> None:
        """PortAudio callback context — must stay cheap and non-blocking."""
        if self._stop.is_set():
            return
        if self._speaking:
            if self.barge_cfg.enabled:
                self._barge_scan(frame)
            return
        if self._busy:
            return
        if self._state == AssistantState.LISTENING:
            if self.wake.process(frame):
                # Flip synchronously inside the callback so no further frames
                # keep hitting the wake detector before the main thread drains
                # the event queue. Announce via _set_state (not a raw field
                # write) so status display and state_log see the transition;
                # _wake_busy keeps frames out of the utterance buffer until
                # handle_wake() opens RECORDING — deliberately NOT the general
                # _busy flag, which finish_utterance checks to decide whether
                # the pipeline is wedged after an error recovery.
                self._wake_busy = True
                self._set_state(AssistantState.WAKE_DETECTED)
                self._events.put("wake")
        elif self._state in (AssistantState.WAKE_DETECTED, AssistantState.RECORDING):
            if self._state == AssistantState.WAKE_DETECTED or self._wake_busy:
                # Wake was confirmed in the callback; the main thread will
                # open RECORDING momentarily. Do NOT feed wake frames into
                # the utterance buffer — otherwise the "Hey Jarvis" itself
                # is transcribed as a phantom first request. Quiet frames are
                # dropped outright; loud frames are parked briefly and
                # replayed by handle_wake() so an immediate follow-on word is
                # not swallowed between the wake hit and RECORDING opening.
                # The park window is time-boxed to ~0.5 s of audio so a long
                # "Hey Jarvis!" can't pre-fill the buffer with wake-phrase
                # syllables.
                from .audio_utils import rms
                frame_seconds = (len(frame.pcm) / 2) / max(1, frame.sample_rate)
                budget = self._wake_pending_budget - frame_seconds
                if budget > 0 and rms(frame.pcm) >= vad.silence_threshold_rms:
                    self._wake_pending.append(frame.pcm)
                    self._wake_pending_budget = budget
                return
            marker = self._buf.add_frame(frame.pcm)
            if marker is not None:
                self._busy = True       # stop feeding; processing starts shortly
                self._events.put(marker)

    def _barge_scan(self, frame: AudioFrame) -> None:
        """Energy-based user-speech detection while we are speaking."""
        now = self.clock()
        if now - self._speak_started_at < self.barge_cfg.grace_seconds:
            return
        from .audio_utils import rms
        level = rms(frame.pcm)
        if level >= self.barge_cfg.threshold_rms:
            self._loud_run += 1
            self._barge_frames.append(frame.pcm)
            if self._loud_run >= self.barge_cfg.min_frames_in_row:
                self._barge_hit.set()
                self._events.put("barge")
        else:
            self._loud_run = 0
            self._barge_frames.clear()

    # ---------------------------------------------------------------- session
    def _new_session(self) -> ConversationSession:
        s = ConversationSession(session_id=f"s{int(self.clock() * 1000)}",
                                max_tokens=self.max_context_tokens,
                                started_at=self.clock())
        log.info("session %s started", s.session_id)
        return s

    def _end_session(self, reason: str) -> None:
        if self.session is not None:
            log.info("session %s ended (%s)", self.session.session_id, reason)
        self.session = None
        self._in_followup = False
        self.wake.reset()
        self._drain_pending()
        self._set_state(AssistantState.LISTENING,
                        f"say '{self.wake.wake_word}' ({reason})")
        self._busy = False

    # ------------------------------------------------------------- transitions
    def _begin_turn(self):
        """Open a TurnRecord stamped with the loop's (injectable) clock.

        MetricsCollector defaults to time.monotonic; tests inject a fake
        clock into the loop, so turns are created here rather than in the
        collector to keep all timestamps on one timeline.
        """
        return self.metrics.begin_turn(now=self.clock())

    def handle_wake(self) -> None:
        self._busy = False
        self._wake_busy = False
        self._in_followup = False      # fresh wake: no longer in hot window
        self._wake_pending_budget = 0.5      # seconds of audio parked per wake
        resumed = bool(self.session) and self.policy.resume_active
        if not self.session or not self.policy.resume_active:
            # resume_active=False => every wake starts a brand-new context.
            self.session = self._new_session()
        self._begin_turn()
        self._set_state(AssistantState.WAKE_DETECTED,
                        f"word={self.wake.wake_word}"
                        + (" [session resumed]" if resumed else " [new session]"))
        # Wake frames keep arriving while the main thread drains events;
        # they are dropped by _on_frame (see WAKE_DETECTED branch) so the
        # wake word itself is never transcribed. Buffer opens at RECORDING.
        self.wake.reset()
        self._buf.reset()
        # Replay speech parked during the wake-confirmation window (bounded
        # to 1 s so a long wake utterance can't pre-fill the buffer).
        pending = self._wake_pending[-16:]
        self._wake_pending.clear()
        marker = None
        for pcm in pending:
            marker = self._buf.add_frame(pcm)
            if marker is not None:
                break                      # endpoint/max reached already
        self._set_state(AssistantState.RECORDING, "speak now")
        if marker is not None:
            # Parked frames alone completed an endpoint (e.g. one-word
            # request spoken with the wake word): hand it straight to the
            # main thread instead of waiting on new audio.
            self._busy = True
            self._events.put(marker)

    def _active_turn(self):
        """Return the TurnRecord for the current turn.

        Normally created by handle_wake(). If respond() was called outside a
        wake cycle (tests, REPL), open one retroactively with zeroed stage
        timestamps so metrics never leak data across turns.

        A turn is considered *closed* once it has produced output — either a
        user transcript (``_user_text``) or an assistant reply
        (``reply_chars``). Closed turns are never reused; otherwise a second
        programmatic ``respond()`` after a completed turn would write its
        timestamps into the old record and corrupt latency stats. Turns that
        were opened but never used (e.g. ``handle_wake`` followed by a silent
        timeout) stay reusable so their wake timestamp is preserved.
        """
        last = self.metrics.turns[-1] if self.metrics.turns else None
        if last is not None and last.t_complete == 0.0 and not last.error \
                and not last._user_text and not last.reply_chars:
            return last
        return self._begin_turn()

    def finish_utterance(self, reason: str) -> None:
        self._drain_pending()
        pcm = self._buf.get_pcm()
        self._buf.reset()
        rec = self._active_turn()
        if reason == "timeout":
            # Nobody spoke after wake / follow-up opened then went quiet.
            self._set_state(AssistantState.LISTENING, "no speech heard")
            self._busy = False
            if self.session and self.session.idle_seconds(self.clock()) > \
                    self.policy.inactivity_timeout_seconds:
                self._end_session("idle timeout")
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
            self.metrics.note_error("stt")
            self._fail(f"STT error: {exc}")
            return
        finally:
            del pcm  # drop audio reference promptly (privacy default)
        if rec is not None:
            rec.t_speech_end = rec.t_speech_end or self.clock()
            rec.t_transcription = self.clock()
            rec.transcript_chars = len(text)
        if not text:
            # Nothing intelligible was heard. If we are mid-conversation,
            # stay in the follow-up window so the user can repeat without a
            # wake word; otherwise drop back to wake-word listening.
            if self.session and self._state == AssistantState.RECORDING:
                self._set_state(AssistantState.RECORDING,
                                "empty transcription, try again")
            else:
                self._set_state(AssistantState.LISTENING, "empty transcription")
            self._busy = False
            return
        self.respond(text)

    # ---------------------------------------------------------------- respond
    def respond(self, user_text: str) -> None:
        if self.session is None:
            # Lazily create a session (e.g. programmatic respond() without a
            # prior wake transition) instead of asserting — more robust.
            self.session = self._new_session()
        rec = self._active_turn()
        now = self.clock()
        self.session.last_activity = now
        if not rec.t_speech_end:
            rec.t_speech_end = now
        rec._user_text = user_text

        if self.policy.is_exit(user_text):
            self._set_state(AssistantState.THINKING, f'"{user_text}"')
            farewell = ("Goodbye." if "good" in user_text.lower()
                        else "Okay, cancelled.")
            cancel = CancelToken()
            self._cancel = cancel
            try:
                self._speak_generator(iter([farewell]), rec, cancel)
            except Exception as exc:   # farewell TTS failure must not wedge us
                log.warning("farewell speak failed: %s", exc)
            finally:
                self._cancel = None
            self._end_session("exit command")
            return

        tier = classify_query(user_text)
        rec.fast_path = (tier == "fast")
        self._set_state(AssistantState.THINKING, f'"{user_text[:40]}" [{tier}]')

        cancel = CancelToken()
        self._cancel = cancel
        history = self.session.history()[-self._max_msgs:]
        spoken_ok: Optional[bool] = False
        try:
            canned = canned_reply(user_text) if tier == "fast" else None
            if canned is not None:
                # Feed the canned reply through the *same* chunker + speak
                # path as streamed answers so behaviour stays uniform.
                gen = iter([canned])
            elif self.llm.supports_streaming():
                gen = self._safe_stream(user_text, history, cancel,
                                        fast_prompt=(tier == "fast"))
            else:
                gen = iter([self.llm.generate(user_text, history)])
            spoken_ok = self._speak_generator(gen, rec, cancel)
        except Exception as exc:
            self.metrics.note_error("llm/tts")
            self._fail(f"pipeline error: {exc}")
            return
        finally:
            self._cancel = None

        if spoken_ok is None:      # barge-in interrupted this turn
            return                 # handle_barge_in already moved to RECORDING

        rec.t_complete = self.clock()
        # Follow-up window: keep the mic hot; no wake word needed for the
        # next utterance until the session idles out or the user says bye.
        self._buf.reset()
        self._drain_pending()
        self._in_followup = True     # hot window open: errors keep the session
        self._set_state(AssistantState.RECORDING,
                        "follow-up? stay silent to end session")
        self._busy = False

    def _safe_stream(self, user_text, history, cancel, fast_prompt):
        try:
            yield from self.llm.stream(user_text, history, cancel=cancel,
                                       fast_path=fast_prompt)
        except TypeError:
            # backend stream() without fast_path support
            yield from self.llm.stream(user_text, history, cancel=cancel)

    # ------------------------------------------------- streaming speak engine
    def _speak_generator(self, gen, rec, cancel: CancelToken) -> Optional[bool]:
        """Consume LLM deltas, chunk into sentences, synthesize+play each one.

        Returns True when the turn completed, None when barge-in cancelled
        it. Partial replies that were already produced still enter history
        so the conversation stays coherent after an interruption.
        """
        chunker = SentenceChunker()
        reply_parts: list[str] = []
        first_chunk = True
        for delta in gen:
            if cancel.cancelled:
                break
            if delta:
                if not rec.t_first_token:
                    rec.t_first_token = self.clock()
                reply_parts.append(delta)
            for chunk in chunker.feed(delta):
                ok = self._emit_chunk(chunk, rec, cancel, first_chunk)
                if ok is None:
                    self._finalize_history(rec, "".join(reply_parts))
                    return None
                first_chunk = False
        for chunk in chunker.flush():
            if cancel.cancelled:
                break
            ok = self._emit_chunk(chunk, rec, cancel, first_chunk)
            if ok is None:
                self._finalize_history(rec, "".join(reply_parts))
                return None
            first_chunk = False
        self._finalize_history(rec, "".join(reply_parts))
        return not cancel.cancelled

    def _emit_chunk(self, chunk: str, rec, cancel: CancelToken,
                    first: bool) -> Optional[bool]:
        """Synthesize + play one sentence. Returns None => barge-in occurred."""
        self._set_state(AssistantState.SPEAKING, chunk[:60])
        self._start_speaking_window()
        try:
            audio = self.tts.synthesize(chunk, "en-US")
        except Exception as exc:
            self._stop_speaking_window()
            raise RuntimeError(f"TTS error: {exc}") from exc
        if first and not rec.t_first_audio:
            rec.t_first_audio = self.clock()
        self._drain_pending()   # never talk over a fresh wake request
        if cancel.cancelled or self._barge_hit.is_set():
            self._stop_speaking_window()
            return None
        try:
            self.player.play(audio)
        except Exception as exc:
            self._stop_speaking_window()
            raise RuntimeError(f"playback error: {exc}") from exc
        self._stop_speaking_window()
        if self._barge_hit.is_set():
            return None         # user spoke during this playback
        return True

    def _start_speaking_window(self) -> None:
        self._speaking = True
        self._speak_started_at = self.clock()
        self._loud_run = 0
        self._barge_frames.clear()
        self._barge_hit.clear()

    def _stop_speaking_window(self) -> None:
        self._speaking = False

    def _finalize_history(self, rec, reply_text: str) -> None:
        reply_text = reply_text.strip()
        user_text = getattr(rec, "_user_text", "") or ""
        if self.session and reply_text:
            self.session.add_turn(user_text, reply_text, self.clock())
        rec.reply_chars = len(reply_text)

    @property
    def history(self) -> list[dict]:
        """Back-compat accessor (v1 tests / REPL): current session messages."""
        return self.session.history() if self.session else []

    # -------------------------------------------------------------- barge-in
    def handle_barge_in(self) -> None:
        """Stop output promptly, discard queued speech, listen to the user."""
        rec = self.metrics.turns[-1] if self.metrics.turns else None
        self._stop_speaking_window()
        if self._cancel is not None:
            self._cancel.cancel()           # propagate: stop LLM/TTS work
        try:
            self.player.abort()             # discard in-flight/queued audio
        except Exception as exc:  # pragma: no cover
            log.debug("player abort: %s", exc)
        self.metrics.note_interruption()
        if rec is not None:
            rec.t_complete = self.clock()
        # Keep what the user has said so far as the start of the new turn.
        self._buf.reset()
        for pcm in self._barge_frames:
            self._buf.add_frame(pcm)
        self._barge_frames.clear()
        # The rest of the user's interrupting utterance keeps arriving on
        # the callback; feed it into the fresh buffer (endpointing will fire
        # normally). _busy stays True until finish_utterance runs.
        self._state = AssistantState.RECORDING
        self._set_state(AssistantState.RECORDING, "interrupted — listening to you")

    def _discard_barge_event(self) -> None:
        """A barge-in was confirmed but the turn already finished between the
        callback firing and the main thread draining events. Drop the stale
        interruption instead of re-opening a recording window that would
        capture room tone after the exchange is over."""
        self.metrics.note_interruption()
        self._barge_frames.clear()
        self._loud_run = 0

    # ---------------------------------------------------------------- failure
    def _fail(self, message: str) -> None:
        log.error(message)
        rec = self.metrics.turns[-1] if self.metrics.turns else None
        if rec is not None and not rec.t_complete:
            rec.error = message[:120]
        # Announce the error (ERROR state is transient by design — it exists
        # so the status display and any UI can show what went wrong), then
        # recover to wake-word listening. The recovery announcement always
        # carries a detail string so StatusDisplay.set_state cannot dedupe it
        # against the previous identical LISTENING line (history/observability
        # would otherwise miss the transition).
        self._set_state(AssistantState.ERROR, message)
        if not self._stop.is_set():
            self._stop_speaking_window()
            self._drain_pending()
            # Recovery policy depends on where the failure happened:
            #  * mid-conversation (the follow-up window was already open —
            #    i.e. a previous turn completed and this utterance came in
            #    without a wake word): keep the session hot and re-open
            #    RECORDING so the user need not repeat the wake word;
            #  * anything else (failure during the first exchange after a
            #    wake, or no session at all): back to wake-word listening.
            #    The session survives until its idle timeout either way.
            # NOTE: self._state has already been flipped to ERROR by the
            # announcement above, so recovery must use _in_followup (set
            # when a reply completes, cleared by handle_wake/_end_session)
            # rather than last_state_before_error — a failed first request
            # also runs with state==RECORDING but is NOT mid-conversation.
            in_conversation = (self.session is not None and
                               self._in_followup)
            if in_conversation:
                self._buf.reset()
                self._set_state(AssistantState.RECORDING,
                                "error recovered — try again")
            else:
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
    def drain_events(self) -> bool:
        processed_any = False
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            processed_any = True
            if ev == "wake":
                # The callback flips the state to WAKE_DETECTED synchronously
                # (so no further frames hit the detector) before queueing this
                # event; accept that state here. LISTENING is kept for callers
                # that enqueue "wake" manually (tests/REPL). RECORDING allows
                # re-waking mid follow-up window without losing the session.
                if self._state in (AssistantState.LISTENING,
                                   AssistantState.WAKE_DETECTED,
                                   AssistantState.RECORDING):
                    self.handle_wake()
            elif ev in ("end", "max", "timeout"):
                if self._state in (AssistantState.WAKE_DETECTED,
                                   AssistantState.RECORDING):
                    self.finish_utterance(ev)
            elif ev == "barge":
                if self._state == AssistantState.SPEAKING:
                    self.handle_barge_in()
                else:
                    # Turn finished before the main thread could react;
                    # don't open a phantom recording window.
                    self._discard_barge_event()
        return processed_any

    def check_timeouts(self) -> None:
        """Main-tick housekeeping: expire idle sessions between turns.

        Checked while LISTENING/RECORDING (mic hot, waiting for the user) and
        also while THINKING/SPEAKING — a long conversation must still expire
        on wall-clock idleness even if a turn was in flight when the deadline
        passed. In-flight states are never expired mid-processing; they end
        naturally into RECORDING/LISTENING where this check applies again.
        """
        now = self.clock()
        if not self.session:
            return
        if self._state in (AssistantState.LISTENING, AssistantState.RECORDING,
                           AssistantState.SPEAKING):
            limit = self.policy.inactivity_timeout_seconds
            if self._state == AssistantState.RECORDING:
                limit = max(limit, self.followup_window_seconds)
            if self.session.idle_seconds(now) > limit:
                self._end_session("idle timeout")

    def run(self) -> None:
        self.capture_sr = getattr(self.capture, "sample_rate", self.sample_rate)
        self._set_state(AssistantState.LISTENING,
                        f"say '{self.wake.wake_word}'")
        try:
            self.capture.start(self._on_frame)
        except RuntimeError as exc:
            self._set_state(AssistantState.ERROR, f"microphone: {exc}")
            raise
        try:
            while not self._stop.is_set():
                self.drain_events()
                self.check_timeouts()
                self._stop.wait(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._stop.set()
        if self._cancel is not None:
            self._cancel.cancel()

    def shutdown(self) -> None:
        try:
            self.capture.stop()
        except Exception as exc:  # pragma: no cover
            log.warning("capture stop error: %s", exc)
        self.wake.close()
        self._set_state(AssistantState.STOPPED, "bye")
        self.status.info("\n" + self.metrics.report())
        self.status.finish()
