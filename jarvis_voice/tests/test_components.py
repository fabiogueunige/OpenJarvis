"""Unit tests for VAD endpointing, wake-word availability policy, and backends."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from jarvis_voice.interfaces import AudioFrame, UtteranceBuffer, VadConfig
from jarvis_voice.audio_utils import rms
from jarvis_voice.llm.backends import OllamaLLM, EchoLLM, _strip_thinking
from jarvis_voice.tts.backends import ScriptedTTS
from jarvis_voice.wakewords.detectors import (
    OpenWakeWordDetector, ScriptedWakeDetector, WakeWordNotAvailableError)

from .fakes import silence_pcm, tone_pcm


# --------------------------------------------------------------------- VAD
def test_rms_values():
    assert rms(b"") == 0.0
    assert rms(silence_pcm(0.1)) == 0.0
    loud = rms(tone_pcm(0.1, amp=8000))
    assert 5000 < loud < 9000


def test_utterance_buffer_end_on_silence():
    buf = UtteranceBuffer(VadConfig(silence_threshold_rms=500,
                                    silence_seconds=0.2,
                                    startup_timeout_seconds=5,
                                    max_utterance_seconds=30), sample_rate=16000)
    frame = int(0.064 * 16000) * 2
    speech = tone_pcm(0.064, amp=8000)[:frame]
    quiet = silence_pcm(0.064)[:frame]
    # each frame is 0.064 s of audio; endpoint fires once the trailing
    # silence run reaches silence_seconds (0.2 s -> on the 4th quiet frame,
    # since 3 x 0.064 = 0.192 < 0.2 <= 4 x 0.064 = 0.256)
    assert buf.add_frame(speech) is None
    assert buf.add_frame(quiet) is None
    assert buf.add_frame(quiet) is None
    assert buf.add_frame(quiet) is None
    assert buf.add_frame(quiet) == "end"
    assert len(buf.get_pcm()) > 0


def test_utterance_buffer_startup_timeout():
    buf = UtteranceBuffer(VadConfig(startup_timeout_seconds=0.2,
                                    silence_seconds=5,
                                    max_utterance_seconds=30), sample_rate=16000)
    marker = None
    for _ in range(10):
        marker = buf.add_frame(silence_pcm(0.064))
        if marker:
            break
    assert marker == "timeout"


def test_utterance_buffer_max_ceiling():
    buf = UtteranceBuffer(VadConfig(max_utterance_seconds=0.3,
                                    startup_timeout_seconds=10,
                                    silence_seconds=10), sample_rate=16000)
    markers = [buf.add_frame(tone_pcm(0.064, amp=8000)) for _ in range(10)]
    assert "max" in markers


# ------------------------------------------------------- wake word policy
def test_openwakeword_rejects_bare_jarvis_without_opt_in(monkeypatch):
    """VERIFIED requirement #3: no silent substitution of another wake word."""
    fake_model_cls = pytest.importorskip_stub = None  # noqa - we stub the import

    class FakeModel:
        def __init__(self, wakeword_models=None):
            raise AssertionError("model must not load when verification fails")

    import sys, types
    ow = types.ModuleType("openwakeword")
    model_mod = types.ModuleType("openwakeword.model")
    utils_mod = types.ModuleType("openwakeword.utils")
    model_mod.Model = FakeModel
    ow.model = model_mod
    monkeypatch.setitem(sys.modules, "openwakeword", ow)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_mod)

    with pytest.raises(WakeWordNotAvailableError) as ei:
        OpenWakeWordDetector(wake_word="jarvis", allow_close_match=False)
    msg = str(ei.value)
    assert "no ready-to-use model" in msg
    assert "hey_jarvis" in msg          # tells user about the close match
    assert "allow_close_match=true" in msg


def test_openwakeword_close_match_requires_explicit_opt_in(monkeypatch):
    loaded = {}

    class FakeModel:
        def __init__(self, wakeword_models=None):
            loaded["models"] = wakeword_models
        def predict(self, pcm): return {"m": 0.0}
        def reset(self): pass

    import sys, types
    ow = types.ModuleType("openwakeword")
    model_mod = types.ModuleType("openwakeword.model")
    utils_mod = types.ModuleType("openwakeword.utils")
    model_mod.Model = FakeModel
    ow.model = model_mod
    monkeypatch.setitem(sys.modules, "openwakeword", ow)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_mod)

    d = OpenWakeWordDetector(wake_word="jarvis", allow_close_match=True)
    assert loaded["models"] == ["hey_jarvis"]
    assert "hey_jarvis" in d.wake_word   # visible in status string


def test_scripted_wake_detector():
    d = ScriptedWakeDetector(fire_after_frames=2)
    f = AudioFrame(pcm=b"\x00\x00" * 512, sample_rate=16000)
    assert d.process(f) is False
    assert d.process(f) is True


# ---------------------------------------------------------------- LLM
def test_ollama_generate_posts_chat(monkeypatch):
    captured = {}

    class Resp:
        status = 200
        def read(self):
            return json.dumps({"message": {"content": "Hello sir."}}).encode()
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM(model="qwen3:4b")
    out = llm.generate("hi", [{"role": "user", "content": "prev"}])
    assert out == "Hello sir."
    assert captured["url"].endswith("/api/chat")
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["system", "user", "user"]  # system + history + current


def test_ollama_unreachable_gives_actionable_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="Ollama unreachable"):
        OllamaLLM().generate("hi", [])


def test_strip_thinking_blocks():
    assert _strip_thinking("<think>reasoning here</think>All good.") == "All good."
    assert _strip_thinking("plain text") == "plain text"


def test_echo_llm_counts_turns():
    llm = EchoLLM()
    r1 = llm.generate("a", [])
    r2 = llm.generate("b", [{"role": "user", "content": "a"}])
    assert "reply number 1" in r1 and "reply number 2" in r2
    assert "1 earlier messages" in r2


# ---------------------------------------------------------------- TTS/STT
def test_scripted_tts_returns_playable_pcm():
    audio = ScriptedTTS().synthesize("hello")
    assert audio.sample_rate == 22050
    assert len(audio.pcm) % 2 == 0
    assert audio.duration_seconds() > 0.2


def test_faster_whisper_missing_library_raises_helpful(monkeypatch):
    from jarvis_voice.stt.backends import FasterWhisperSTT
    import sys
    # force ImportError path
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    stt = FasterWhisperSTT()
    with pytest.raises(RuntimeError, match="faster-whisper not installed"):
        stt.transcribe(tone_pcm(0.5), 16000)


def test_piper_missing_cli_raises_helpful(monkeypatch):
    from jarvis_voice.tts.backends import PiperTTS
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="piper CLI not found"):
        PiperTTS(model_path="whatever.onnx")
