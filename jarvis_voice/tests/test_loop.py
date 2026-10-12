"""Unit tests for the voice loop state machine (fakes only, no devices)."""

from __future__ import annotations

import io

from jarvis_voice.interfaces import AssistantState, AudioFrame, VadConfig
from jarvis_voice.loop import VoiceLoop
from jarvis_voice.status import StatusDisplay

from .fakes import (FakeCapture, FakeLLM, FakePlayer, FakeSTT, FakeTTS,
                   FakeWake, silence_pcm, tone_pcm)


def make_loop(capture=None, wake=None, stt=None, llm=None, tts=None, player=None):
    capture = capture or FakeCapture()
    status = StatusDisplay(stream=io.StringIO())
    loop = VoiceLoop(
        capture=capture,
        wake=wake or FakeWake(fire_after_calls=2),
        stt=stt or FakeSTT(["what time is it"]),
        llm=llm or FakeLLM(),
        tts=tts or FakeTTS(),
        player=player or FakePlayer(),
        vad=VadConfig(silence_threshold_rms=500, silence_seconds=0.3,
                      startup_timeout_seconds=1.0, max_utterance_seconds=5.0),
        status=status,
        sample_rate=16000,
    )
    loop.capture_sr = 16000
    return loop, status


def pump(loop, capture, frames):
    # FakeCapture only has a sink once start() was called; the loop normally
    # does that inside run(). Tests drive frames directly, so wire it here.
    if not capture.running:
        capture.start(loop._on_frame)
    # Deliver one frame at a time and drain events after each one — this
    # mirrors the real main loop (poll every ~50 ms while PortAudio emits a
    # frame every 64 ms), so state transitions interleave with delivery just
    # like they do on hardware.
    for pcm in frames:
        capture.frames = [pcm]
        capture.pump(1)
        loop.drain_events()


def test_happy_path_full_cycle():
    cap = FakeCapture()
    loop, status = make_loop(capture=cap)
    states = []
    orig = loop._set_state
    loop._set_state = lambda s, d="": (states.append(s), orig(s, d))[0]

    # two silent frames -> wake fires (fire_after_calls=2)
    frame_sil = silence_pcm(0.064)
    # then speech, then trailing silence >= silence_seconds (0.3 s needs
    # five 64 ms frames: 4 x 0.064 = 0.256 < 0.3 <= 5 x 0.064 = 0.32)
    speech = tone_pcm(0.064, amp=8000)
    pump(loop, cap, [frame_sil, frame_sil, speech, speech,
                     frame_sil, frame_sil, frame_sil, frame_sil,
                     frame_sil])

    assert AssistantState.WAKE_DETECTED in states
    assert AssistantState.RECORDING in states
    assert AssistantState.TRANSCRIBING in states
    assert AssistantState.THINKING in states
    assert AssistantState.SPEAKING in states
    assert states[-1] == AssistantState.LISTENING
    assert loop.tts.spoken == ["Sure, here is an English answer."]
    assert len(loop.history) == 2
    assert loop.history[0]["content"] == "what time is it"


def test_empty_transcription_returns_to_listening():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap, stt=FakeSTT([""]))
    sil = silence_pcm(0.064)
    speech = tone_pcm(0.064, amp=8000)
    pump(loop, cap, [sil, sil, speech] + [sil] * 5)
    assert loop.state == AssistantState.LISTENING
    # empty transcript -> the LLM must never be consulted, nothing spoken
    assert loop.llm.seen_history == []
    assert loop.tts.spoken == []


def test_startup_timeout_no_speech():
    cap = FakeCapture()
    loop, status = make_loop(capture=cap)
    sil = silence_pcm(0.064)
    # wake fires on 2nd frame; then long silence -> startup timeout.
    # startup_timeout_seconds=1.0 needs >15 frames of 0.064 s after wake.
    pump(loop, cap, [sil, sil] + [sil] * 17)
    assert loop.state == AssistantState.LISTENING
    assert "no speech" in status_last(status)
    # transcript never requested:
    assert loop.stt.calls == 0


def test_llm_failure_shows_error_then_recovers():
    cap = FakeCapture()
    loop, status = make_loop(capture=cap, llm=FakeLLM(fail=True))
    sil = silence_pcm(0.064)
    speech = tone_pcm(0.064, amp=8000)
    pump(loop, cap, [sil, sil, speech] + [sil] * 5)
    rendered = status.history
    assert any(line.startswith("[ERROR]") and "LLM error" in line
               for line in rendered)
    assert loop.state == AssistantState.LISTENING


def test_stt_failure_recovers():
    class BoomSTT(FakeSTT):
        def transcribe(self, pcm, sample_rate, language="en"):
            raise RuntimeError("model missing")

    cap = FakeCapture()
    loop, status = make_loop(capture=cap, stt=BoomSTT())
    sil = silence_pcm(0.064)
    speech = tone_pcm(0.064, amp=8000)
    pump(loop, cap, [sil, sil, speech] + [sil] * 5)
    assert any("STT error" in l for l in status.history)
    assert loop.state == AssistantState.LISTENING


def test_max_utterance_ceiling():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap)
    sil = silence_pcm(0.064)
    loud = tone_pcm(0.064, amp=8000)
    # continuous loud audio beyond max_utterance_seconds=5.0
    frames = [sil, sil] + [loud] * 90
    pump(loop, cap, frames)
    assert loop.stt.calls == 1  # forced endpoint at ceiling


def test_graceful_shutdown():
    cap = FakeCapture()
    loop, _ = make_loop(capture=cap)
    loop.stop()
    loop.shutdown()
    assert loop.state == AssistantState.STOPPED
    assert not cap.running


def test_microphone_start_failure_raises():
    cap = FakeCapture()
    cap.fail_on_start = True
    loop, status = make_loop(capture=cap)
    try:
        loop.run()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "microphone" in str(exc)


def status_last(status):
    return status.history[-1] if status.history else ""
