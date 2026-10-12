"""Factory that wires concrete backends from AppConfig."""

from __future__ import annotations

import os

from .config import AppConfig
from .interfaces import VadConfig


def build_wake(cfg: AppConfig):
    wc = cfg.wake_word
    if wc.engine == "scripted":
        from .wakewords.detectors import ScriptedWakeDetector
        return ScriptedWakeDetector(wc.word)
    if wc.engine == "whisper_hypothesis":
        from .wakewords.detectors import WhisperHypothesisDetector
        stt = build_stt(cfg)
        return WhisperHypothesisDetector(wc.word, stt)
    from .wakewords.detectors import OpenWakeWordDetector
    return OpenWakeWordDetector(
        wake_word=wc.word, threshold=wc.threshold,
        allow_close_match=wc.allow_close_match,
        model_paths=[os.path.expanduser(p) for p in wc.model_paths] or None,
    )


def build_stt(cfg: AppConfig):
    if cfg.stt.engine == "scripted":
        from .stt.backends import ScriptedSTT
        return ScriptedSTT()
    from .stt.backends import FasterWhisperSTT
    return FasterWhisperSTT(model_size=cfg.stt.model_size, device=cfg.stt.device,
                            compute_type=cfg.stt.compute_type)


def build_llm(cfg: AppConfig):
    if cfg.llm.engine == "echo":
        from .llm.backends import EchoLLM
        return EchoLLM()
    from .llm.backends import OllamaLLM
    return OllamaLLM(model=cfg.llm.model, base_url=cfg.llm.base_url,
                     timeout_seconds=cfg.llm.timeout_seconds,
                     keep_alive=cfg.llm.keep_alive,
                     temperature=cfg.llm.temperature)


def build_session_policy(cfg: AppConfig):
    from .session import SessionPolicy
    sc = cfg.session
    return SessionPolicy(resume_active=sc.resume_active,
                         inactivity_timeout_seconds=sc.inactivity_timeout_seconds,
                         exit_phrases=list(sc.exit_phrases))


def build_barge_in(cfg: AppConfig):
    from .interfaces import BargeInConfig
    bc = cfg.barge_in
    return BargeInConfig(enabled=bc.enabled, threshold_rms=bc.threshold_rms,
                         min_frames_in_row=bc.min_frames_in_row,
                         grace_seconds=bc.grace_seconds)


def build_tts(cfg: AppConfig):
    if cfg.tts.engine == "scripted":
        from .tts.backends import ScriptedTTS
        return ScriptedTTS()
    if cfg.tts.engine == "kokoro":
        from .tts.backends import KokoroTTS
        return KokoroTTS(model_path=cfg.tts.kokoro_model,
                         voices_path=cfg.tts.kokoro_voices,
                         voice=cfg.tts.kokoro_voice,
                         espeak_data_path=cfg.tts.espeak_data_path)
    from .tts.backends import PiperTTS
    return PiperTTS(model_path=os.path.expanduser(cfg.tts.piper_model))


def build_audio(cfg: AppConfig):
    from .audio_sd import SdAudioCapture, SdAudioPlayer
    capture = SdAudioCapture(sample_rate=cfg.audio.sample_rate,
                              block_size=cfg.audio.block_size,
                              device=cfg.audio.input_device)
    player = SdAudioPlayer(device=cfg.audio.output_device)
    return capture, player


def build_vad(cfg: AppConfig) -> VadConfig:
    a = cfg.audio
    return VadConfig(silence_threshold_rms=a.silence_threshold_rms,
                     silence_seconds=a.silence_seconds,
                     startup_timeout_seconds=a.startup_timeout_seconds,
                     max_utterance_seconds=a.max_utterance_seconds)
