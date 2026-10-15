"""Entry point: python -m jarvis_voice.main [--config config.toml] [--demo]"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .config import load_config
from .loop import VoiceLoop
from .status import StatusDisplay


def _demo_components(cfg):
    """All-fake pipeline so the loop can be exercised without models/devices."""
    from .wakewords.detectors import ScriptedWakeDetector
    from .stt.backends import ScriptedSTT
    from .llm.backends import EchoLLM
    from .tts.backends import ScriptedTTS
    from .interfaces import AudioCapture, AudioPlayer, PcmAudio

    class DemoCapture(AudioCapture):
        """Feeds a wake burst then one utterance then silence, on demand."""
        def __init__(self):
            self.sample_rate = cfg.audio.sample_rate
            self._sink = None
            self._running = False
            self._tick = 0

        def start(self, sink):
            self._sink = sink
            self._running = True

        def stop(self):
            self._running = False

        @property
        def running(self):
            return self._running

        def tick(self):
            import struct
            self._tick += 1
            if self._tick <= 3:
                pcm = b"\x00\x00" * cfg.audio.block_size          # silence
            elif self._tick <= 6:
                pcm = struct.pack(f"<{cfg.audio.block_size}h",
                                  *([4000] * cfg.audio.block_size))  # "speech"
            else:
                pcm = b"\x00\x00" * cfg.audio.block_size
            if self._sink:
                from .interfaces import AudioFrame
                self._sink(AudioFrame(pcm=pcm, sample_rate=self.sample_rate))

    class NullPlayer(AudioPlayer):
        def play(self, audio: PcmAudio) -> None:
            pass

    cap = DemoCapture()
    wake = ScriptedWakeDetector(cfg.wake_word.word, fire_after_frames=2)
    return cap, wake, ScriptedSTT(), EchoLLM(), ScriptedTTS(), NullPlayer(), cap.tick


def _audio_dir(cfg):
    """Opt-in raw-audio retention directory; None means never store audio."""
    if not cfg.store_audio:
        return None
    from pathlib import Path

    d = Path.home() / ".local" / "share" / "jarvis" / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS voice prototype")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--demo", action="store_true",
                        help="run the loop with scripted backends (no mic/models)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    cfg = load_config(args.config)
    status = StatusDisplay()

    if args.demo:
        cap, wake, stt, llm, tts, player, tick = _demo_components(cfg)
        from .interfaces import VadConfig
        loop = VoiceLoop(capture=cap, wake=wake, stt=stt, llm=llm, tts=tts,
                          player=player, vad=VadConfig(silence_seconds=0.3,
                          startup_timeout_seconds=1.0),
                          status=status, sample_rate=cfg.audio.sample_rate,
                          language=cfg.stt.language,
                          max_history_turns=cfg.llm.max_history_turns,
                          store_audio_path=_audio_dir(cfg),
                          session_policy=build_session_policy(cfg),
                          barge_in=build_barge_in(cfg),
                          followup_window_seconds=cfg.session.followup_window_seconds,
                          max_context_tokens=cfg.session.max_context_tokens)
        loop.capture_sr = cfg.audio.sample_rate
        loop._set_state(loop.state, "demo mode")
        try:
            cap.start(loop._on_frame)
            for _ in range(20):
                tick()
                loop.drain_events()
        finally:
            loop.shutdown()
        return 0

    # real hardware path
    from .factory import (build_audio, build_barge_in, build_llm, build_session_policy,
                          build_stt, build_tts, build_vad, build_wake)
    try:
        wake = build_wake(cfg)
    except Exception as exc:
        print(f"Wake word setup failed:\n{exc}", file=sys.stderr)
        print("See docs/WAKE_WORD.md for how to obtain a proper 'Jarvis' model.",
              file=sys.stderr)
        return 2

    capture, player = build_audio(cfg)
    loop = VoiceLoop(capture=capture, wake=wake, stt=build_stt(cfg),
                     llm=build_llm(cfg), tts=build_tts(cfg), player=player,
                     vad=build_vad(cfg), status=status,
                     sample_rate=cfg.audio.sample_rate,
                     language=cfg.stt.language,
                     max_history_turns=cfg.llm.max_history_turns,
                     store_audio_path=_audio_dir(cfg),
                     session_policy=build_session_policy(cfg),
                     barge_in=build_barge_in(cfg),
                     followup_window_seconds=cfg.session.followup_window_seconds,
                     max_context_tokens=cfg.session.max_context_tokens)

    def _sigint(*_):
        status.info("\nShutting down…")
        loop.stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        loop.run()
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
