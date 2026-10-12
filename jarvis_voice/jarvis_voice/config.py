"""Configuration loading for the JARVIS voice prototype (TOML, stdlib tomllib)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WakeWordConfig:
    engine: str = "openwakeword"      # openwakeword | whisper_hypothesis | scripted
    word: str = "jarvis"
    threshold: float = 0.5
    allow_close_match: bool = False   # explicit opt-in to 'hey_jarvis' model
    model_paths: list[str] = field(default_factory=list)  # custom .onnx models


@dataclass
class SttConfig:
    engine: str = "faster_whisper"    # faster_whisper | scripted
    model_size: str = "base"          # tiny|base|small|distil-large-v3
    device: str = "auto"              # auto|cpu|cuda
    compute_type: str = "auto"
    language: str = "en"


@dataclass
class LlmConfig:
    engine: str = "ollama"            # ollama | echo
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 60.0
    keep_alive: str = "10m"
    temperature: float = 0.7
    max_history_turns: int = 8        # conversation context window (turns)


@dataclass
class TtsConfig:
    engine: str = "piper"             # piper | kokoro | scripted
    piper_model: str = "~/.local/share/jarvis/piper/en_US-lessac-medium.onnx"
    kokoro_model: str = "~/.local/share/jarvis/kokoro/v1.0.onnx"
    kokoro_voices: str = "~/.local/share/jarvis/kokoro/voices.bin"
    kokoro_voice: str = "af_heart"
    espeak_data_path: Optional[str] = None


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    silence_threshold_rms: float = 500.0
    silence_seconds: float = 1.2
    startup_timeout_seconds: float = 6.0
    max_utterance_seconds: float = 30.0


@dataclass
class SessionConfig:
    resume_active: bool = True          # wake word resumes an active session
    inactivity_timeout_seconds: float = 45.0
    followup_window_seconds: float = 30.0
    max_context_tokens: int = 1500      # rolling history budget (estimate-based)
    exit_phrases: list[str] = field(default_factory=lambda: [
        "goodbye jarvis", "bye jarvis", "stop listening", "never mind",
        "nevermind", "cancel",
    ])


@dataclass
class BargeInTuning:
    enabled: bool = True
    threshold_rms: float = 900.0
    min_frames_in_row: int = 4
    grace_seconds: float = 0.35


@dataclass
class AppConfig:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    barge_in: BargeInTuning = field(default_factory=BargeInTuning)
    store_audio: bool = False         # must be explicitly enabled; default off


def _fill(dc, data: dict):
    for k, v in (data or {}).items():
        if hasattr(dc, k):
            setattr(dc, k, v)
    return dc


def load_config(path: Optional[str] = None) -> AppConfig:
    cfg = AppConfig()
    path = path or os.environ.get("JARVIS_CONFIG", "config.toml")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        _fill(cfg.wake_word, raw.get("wake_word", {}))
        _fill(cfg.stt, raw.get("stt", {}))
        _fill(cfg.llm, raw.get("llm", {}))
        _fill(cfg.tts, raw.get("tts", {}))
        _fill(cfg.audio, raw.get("audio", {}))
        _fill(cfg.session, raw.get("session", {}))
        _fill(cfg.barge_in, raw.get("barge_in", {}))
        if "store_audio" in raw:
            cfg.store_audio = bool(raw["store_audio"])
    return cfg
