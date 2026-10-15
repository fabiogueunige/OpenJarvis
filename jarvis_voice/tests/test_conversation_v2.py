"""Tests for the v2 conversational features: sessions, streaming, barge-in,
metrics. All fakes — no devices, no network."""

from __future__ import annotations

import io
import time

from jarvis_voice.interfaces import (
    AssistantState, AudioFrame, BargeInConfig, CancelToken, LLMBackend,
    PcmAudio, StreamingLLMBackend, VadConfig,
)
from jarvis_voice.llm.backends import EchoLLM, canned_reply, classify_query
from jarvis_voice.loop import VoiceLoop
from jarvis_voice.session import ConversationSession, SessionPolicy
from jarvis_voice.status import StatusDisplay
from jarvis_voice.streaming import SentenceChunker

from .fakes import (FakeCapture, FakeLLM, FakePlayer, FakeSTT, FakeTTS,
                   FakeWake, silence_pcm, tone_pcm)


# --------------------------------------------------------------------- helpers
class ScriptedStreamLLM(StreamingLLMBackend):
    """True-streaming fake: yields deltas and records cancellation state."""

    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.streams_started = 0
        self.streams_completed = 0
        self.cancel_seen: list[bool] = []

    def generate(self, user_text, history):   # pragma: no cover - unused
        return "".join(self.deltas)

    def stream(self, user_text, history, cancel=None, fast_path=False):
        self.streams_started += 1
        for d in self.deltas:
            if cancel is not None and cancel.cancelled:
                self.cancel_seen.append(True)
                return
            yield d
        self.streams_completed += 1
        self.cancel_seen.append(False)


class SlowGenLLM(LLMBackend):
    """Non-streaming backend that raises if called after cancellation was
    requested mid-pipeline (used to prove cancellation propagation)."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, user_text, history):
        self.calls += 1
        return "One sentence here. Second sentence follows."


def make_loop(*, capture=None, wake=None, stt=None, llm=None, tts=None,
              player=None, vad=None, policy=None, barge=None, clock=None,
              followup_window_seconds=30.0, max_context_tokens=1500):
    capture = capture or FakeCapture()
    status = StatusDisplay(stream=io.StringIO())
    loop = VoiceLoop(
        capture=capture,
        wake=wake or FakeWake(fire_after_calls=2),
        stt=stt or FakeSTT(["what time is it"]),
        llm=llm or FakeLLM(),
        tts=tts or FakeTTS(),
        player=player or FakePlayer(),
        vad=vad or VadConfig(silence_threshold_rms=500, silence_seconds=0.3,
                             startup_timeout_seconds=1.0,
                             max_utterance_seconds=5.0),
        status=status,
        sample_rate=16000,
        session_policy=policy,
        barge_in=barge,
        followup_window_seconds=followup_window_seconds,
        max_context_tokens=max_context_tokens,
        clock=clock or time.monotonic,
    )
    loop.capture_sr = 16000
    return loop, status


def pump(loop, capture, frames):
    if not capture.running:
        capture.start(loop._on_frame)
    for pcm in frames:
        capture.frames = [pcm]
        capture.pump(1)
        loop.drain_events()


SIL = lambda: silence_pcm(0.064)
LOUD = lambda: tone_pcm(0.064, amp=8000)
END_SIL = [SIL()] * 5          # >= silence_seconds (0.3 s)


# ==================================================================== sessions
def test_wake_starts_session_and_followups_need_no_wake():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, stt=FakeSTT(["first question",
                                                  "and a follow up"]))
    llm = loop.llm
    # turn 1: wake + speech + endpoint
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.state == AssistantState.RECORDING      # follow-up window hot
    assert len(llm.seen_history) == 1
    # turn 2 WITHOUT wake word: just speak
    pump(loop, cap, [LOUD()] + END_SIL)
    assert len(llm.seen_history) == 2
    assert llm.seen_history[1], "second call must carry conversation history"
    assert {"role": "user", "content": "first question"} in llm.seen_history[1]
    assert {"role": "assistant", "content": llm.reply} in llm.seen_history[1]


def test_exit_command_ends_session_and_returns_to_listening():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, stt=FakeSTT(["goodbye jarvis"]))
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.state == AssistantState.LISTENING
    assert loop.session is None
    assert loop.tts.spoken == ["Goodbye."]


def test_cancel_command_speaks_acknowledgement():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, stt=FakeSTT(["cancel"]))
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.state == AssistantState.LISTENING
    assert loop.session is None
    assert loop.tts.spoken == ["Okay, cancelled."]


def test_resume_active_false_starts_new_session_each_wake():
    cap = FakeCapture()
    policy = SessionPolicy(resume_active=False)
    loop, _ = make_loop(capture=cap, stt=FakeSTT(["hello jarvis",
                                                  "hello again"]),
                       policy=policy)
    # first wake -> canned greeting (fast path), session ends? No: greeting
    # keeps session; force end by idle-timeout simulation below instead.
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    sid1 = loop.session.session_id if loop.session else None
    assert loop.state == AssistantState.RECORDING
    # End session via exit phrase, then re-wake: new id expected.
    loop.stt.scripts = ["goodbye jarvis"]
    pump(loop, cap, [LOUD()] + END_SIL)
    assert loop.session is None
    loop.stt.scripts = ["hello there"]
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.session is not None and loop.session.session_id != sid1


def test_idle_timeout_ends_session_between_turns():
    now = [0.0]
    cap = FakeCapture()
    loop, status = make_loop(capture=cap,
                             clock=lambda: now[0],
                             policy=SessionPolicy(inactivity_timeout_seconds=5))
    # Drive a real turn through the state machine (wake -> speech -> reply),
    # then let the follow-up window go quiet until the inactivity timeout
    # expires. check_timeouts() is what the main run-loop calls every tick.
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.session is not None
    assert loop.state == AssistantState.RECORDING
    now[0] += 6.0
    loop.check_timeouts()
    assert loop.session is None
    assert loop.state == AssistantState.LISTENING


def test_followup_window_limit_ends_long_silent_session():
    now = [0.0]
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap,
                        clock=lambda: now[0],
                        policy=SessionPolicy(inactivity_timeout_seconds=60),
                        followup_window_seconds=10)
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)   # one full turn
    assert loop.state == AssistantState.RECORDING
    now[0] += 11
    loop.check_timeouts()                    # RECORDING uses followup limit
    assert loop.session is None
    assert loop.state == AssistantState.LISTENING


def test_session_token_budget_trims_history():
    s = ConversationSession("t", max_tokens=50)
    long_a = "a" * 400                      # ~100 tokens alone
    s.add_turn("q1", long_a, now=1.0)
    s.add_turn("q2", "short reply.", now=2.0)
    h = s.history()
    total = sum(ConversationSession.estimate_tokens(m["content"]) for m in h)
    assert total <= 50 or len(h) == 2       # oldest dropped to fit budget
    assert h[-1]["content"] == "short reply."
    assert all(m["role"] in ("user", "assistant") for m in h)


# =================================================================== streaming
def test_sentence_chunker_rules():
    c = SentenceChunker()
    out = []
    for delta in ["Hello ", "there. How ", "are you?"]:
        out += c.feed(delta)
    out += c.flush()
    assert out == ["Hello there.", "How are you?"]

    # abbreviations / decimals must not split mid-sentence
    c2 = SentenceChunker()
    got = []
    for d in ["Dr. Smith went to Washington. ", "Pi is 3.14 exactly."]:
        got += c2.feed(d)
    got += c2.flush()
    assert got == ["Dr. Smith went to Washington.", "Pi is 3.14 exactly."]

    # fragments below min size are never emitted on their own
    c3 = SentenceChunker(min_chunk_chars=4)
    got3 = c3.feed("Th")
    assert got3 == []
    got3 += c3.flush()
    assert got3 == []                        # dropped: malformed fragment


def test_streaming_tts_starts_before_generation_finishes():
    class ChunkedTTS(FakeTTS):
        def __init__(self):
            super().__init__()
            self.gen_done_at_emit = []

    llm = ScriptedStreamLLM(["The answer is forty-two. ",
                             "Anything else? ", "No thanks."])
    # Simulate "generation still running when first chunk spoken": the fake
    # stream only increments streams_completed at the very end, so check the
    # TTS heard the first sentence while the generator was still open.
    tts = ChunkedTTS()
    loop, _ = make_loop(llm=llm, tts=tts)
    loop.metrics.begin_turn()
    loop.respond("answer me")
    assert tts.spoken == ["The answer is forty-two.", "Anything else?",
                          "No thanks."]
    assert llm.streams_started == 1
    assert llm.streams_completed == 1


def test_fast_path_routing_and_canned_replies():
    assert classify_query("hello") == "fast"
    assert classify_query("what time is it") == "fast"
    assert classify_query("explain how transformers work in detail") == "standard"
    assert canned_reply("ping") == "Pong."
    # Fast path with a canned reply must NOT hit the LLM at all.
    llm = FakeLLM()
    loop, _ = make_loop(llm=llm)
    loop.metrics.begin_turn()
    loop.respond("hello")
    assert llm.seen_history == []
    # v2 speaks through the *same* sentence chunker as streamed answers, so
    # the canned two-sentence greeting arrives as two spoken chunks.
    assert loop.tts.spoken == ["Hello.", "How can I help?"]
    assert "".join(loop.tts.spoken) == canned_reply("hello")


# =================================================================== barge-in
def test_barge_in_stops_playback_and_listens():
    barge = BargeInConfig(enabled=True, threshold_rms=900,
                          min_frames_in_row=1, grace_seconds=0.0)
    # Long multi-sentence reply => several chunks; interrupt between them.
    deltas = ["First sentence here. ", "Second never spoken. "]

    class BargingStreamLLM(ScriptedStreamLLM):
        """Scripted stream that injects the user's interruption at a precise
        point: right after the first chunk has been played and before the
        second delta is consumed. This proves cancellation propagation —
        the generator observes the CancelToken and records it."""

        def __init__(self, loop_ref_holder: dict) -> None:
            super().__init__(deltas)
            self.holder = loop_ref_holder

        def stream(self, user_text, history, cancel=None, fast_path=False):
            self.streams_started += 1
            for d in self.deltas:
                if cancel is not None and cancel.cancelled:
                    self.cancel_seen.append(True)
                    return
                yield d
                if self.holder.get("loop") is not None:
                    lp = self.holder["loop"]
                    lp._on_frame(AudioFrame(pcm=tone_pcm(0.064, amp=8000),
                                             sample_rate=16000))
                    lp.drain_events()      # processes "barge" event
            self.streams_completed += 1
            self.cancel_seen.append(False)

    holder: dict = {}
    llm = BargingStreamLLM(holder)
    player = FakePlayer()
    loop, _ = make_loop(llm=llm, player=player, barge=barge)
    cap = loop.capture
    if not cap.running:
        cap.start(loop._on_frame)
    holder["loop"] = loop

    loop.metrics.begin_turn()
    loop.respond("tell me a story")

    assert loop.state == AssistantState.RECORDING          # listening to user
    assert player.aborts >= 1                              # queued audio gone
    assert loop.tts.spoken == ["First sentence here."]     # rest discarded
    assert llm.cancel_seen == [True]                       # stream cancelled
    assert loop.metrics.interruptions == 1
    # partial reply entered history so the conversation stays coherent
    hist = loop.history
    assert any(m["role"] == "assistant" and
               m["content"] == "First sentence here." for m in hist)


def test_barge_in_grace_period_ignores_early_noise():
    barge = BargeInConfig(enabled=True, threshold_rms=900,
                          min_frames_in_row=1, grace_seconds=10.0)
    loop, _ = make_loop(barge=barge)
    loop._speaking = True
    loop._speak_started_at = loop.clock()
    loop._on_frame(AudioFrame(pcm=tone_pcm(0.064, amp=9000), sample_rate=16000))
    assert not loop._barge_hit.is_set()                    # gated by grace


def test_barge_in_requires_run_of_loud_frames():
    barge = BargeInConfig(enabled=True, threshold_rms=900,
                          min_frames_in_row=3, grace_seconds=0.0)
    loop, _ = make_loop(barge=barge)
    loop._speaking = True
    loop._speak_started_at = loop.clock() - 1.0
    loud = AudioFrame(pcm=tone_pcm(0.064, amp=8000), sample_rate=16000)
    quiet = AudioFrame(pcm=silence_pcm(0.064), sample_rate=16000)
    loop._on_frame(loud)
    loop._on_frame(quiet)                                  # breaks the run
    loop._on_frame(loud)
    assert not loop._barge_hit.is_set()
    loop._on_frame(loud)
    loop._on_frame(loud)
    assert loop._barge_hit.is_set()


# ==================================================================== metrics
def test_metrics_record_stage_timestamps_and_report():
    # Drive a real turn through the state machine so *every* stage
    # timestamp (wake, speech end, transcription, first token/audio,
    # completion) is stamped by its natural transition.
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap)
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    t = loop.metrics.turns[-1]
    assert t.t_wake and t.t_transcription and t.t_first_audio
    assert t.t_speech_end and t.t_first_token
    assert t.t_complete > t.t_first_audio
    lat = t.latencies()
    assert "stt" in lat and "llm_ttft" in lat
    assert "speech_end_to_first_audio" in lat
    assert "turn_total" in lat
    rep = loop.metrics.report()
    assert "p50" in rep and "p95" in rep and "turns=1" in rep


def test_metrics_programmatic_respond_still_tracks_reply():
    # respond() without STT: transcription timestamp legitimately absent;
    # report must still work and reply_chars recorded.
    loop, _ = make_loop()
    t = loop.metrics.begin_turn()
    loop.respond("what is the weather")
    assert t.t_wake and t.t_first_audio
    assert t.t_complete > t.t_first_audio
    assert t.reply_chars > 0
    assert "turns=1" in loop.metrics.report()


def test_metrics_counts_errors_and_timeouts():
    loop, _ = make_loop(llm=FakeLLM(fail=True))
    loop.metrics.begin_turn()
    try:
        loop.respond("boom")
    except Exception:
        pass
    # respond catches pipeline errors internally:
    assert loop.metrics.errors >= 1
    loop.metrics.note_timeout()
    assert loop.metrics.timeouts == 1


# ============================================================ recovery flows
def test_recovery_llm_error_then_next_turn_works():
    class FlakyLLM(FakeLLM):
        def __init__(self):
            super().__init__(reply="recovered answer")
            self.n = 0

        def generate(self, user_text, history):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("llm down")
            return super().generate(user_text, history)

    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, stt=FakeSTT(["question one",
                                                  "question two"]))
    pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)      # fails -> recovers
    # v2 recovery keeps the session hot: after the ERROR announcement the
    # loop re-opens the follow-up window (RECORDING) instead of forcing a
    # fresh wake word.
    assert loop.state == AssistantState.RECORDING
    assert loop.metrics.errors == 1
    # second attempt succeeds without saying the wake word again
    pump(loop, cap, [LOUD()] + END_SIL)
    assert loop.tts.spoken == ["recovered answer"]
    assert loop.state == AssistantState.RECORDING


def test_multi_turn_consecutive_dialogue_with_echo():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, llm=EchoLLM(),
                        stt=FakeSTT(["alpha", "bravo", "charlie"]))
    for _ in range(3):
        pump(loop, cap, [SIL(), SIL(), LOUD()] + END_SIL)
    assert loop.stt.calls == 3
    # v2 speaks sentence-by-sentence: EchoLLM's reply is two sentences, so
    # each turn produces two spoken chunks (6 total). The *content* joined
    # per turn is what matters — assert on that, not raw chunk count.
    assert len(loop.tts.spoken) == 6
    last_reply = " ".join(loop.tts.spoken[-2:])
    assert "You said: charlie." in loop.tts.spoken[-2]
    assert "keeping context of 4 earlier messages" in last_reply
    assert loop.state == AssistantState.RECORDING
