"""Latency/health metrics for the voice pipeline.

Records per-turn timestamps (wake, end-of-speech, transcription done, first
LLM token, first audible chunk, completion) plus counters (errors, timeouts,
model load times, peak RSS). Derived stage latencies are computed from the
timestamps; ``report()`` prints p50/p95 across turns. No threads, no files
unless explicitly dumped to JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile on a small sample."""
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


@dataclass
class TurnRecord:
    turn_id: int
    t_wake: float = 0.0                 # monotonic when wake confirmed
    t_speech_end: float = 0.0           # VAD endpoint of user utterance
    t_transcription: float = 0.0        # STT returned
    t_first_token: float = 0.0          # first LLM delta (0 if unsupported)
    t_first_audio: float = 0.0          # first chunk handed to playback
    t_complete: float = 0.0             # playback finished / turn closed
    interrupted: bool = False
    error: Optional[str] = None
    timeout: bool = False
    fast_path: Optional[bool] = None
    transcript_chars: int = 0
    reply_chars: int = 0
    _user_text: str = ""            # set by the loop when the turn's request is known

    def latencies(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.t_speech_end and self.t_transcription:
            out["stt"] = (self.t_transcription - self.t_speech_end) * 1e3
        if self.t_transcription and self.t_first_token:
            out["llm_ttft"] = (self.t_first_token - self.t_transcription) * 1e3
        if self.t_transcription and self.t_complete and not self.t_first_token:
            out["llm_total"] = (self.t_complete - self.t_transcription) * 1e3
        if self.t_speech_end and self.t_first_audio:
            out["speech_end_to_first_audio"] = (self.t_first_audio - self.t_speech_end) * 1e3
        if self.t_wake and self.t_complete:
            out["turn_total"] = (self.t_complete - self.t_wake) * 1e3
        if self.t_first_audio and self.t_complete:
            out["speaking_time"] = (self.t_complete - self.t_first_audio) * 1e3
        return out


@dataclass
class MetricsCollector:
    turns: list[TurnRecord] = field(default_factory=list)
    model_load_seconds: dict[str, float] = field(default_factory=dict)
    errors: int = 0
    timeouts: int = 0
    interruptions: int = 0
    barge_false_triggers: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _next_turn: int = 0

    # ------------------------------------------------------------------ api
    def begin_turn(self, now: Optional[float] = None) -> TurnRecord:
        """Open a turn record.

        ``now`` lets callers (the VoiceLoop) stamp the wake timestamp with
        their own injectable clock so tests can run on a fake timeline;
        defaults to time.monotonic for production use.
        """
        rec = TurnRecord(turn_id=self._next_turn,
                         t_wake=time.monotonic() if now is None else now)
        self._next_turn += 1
        self.turns.append(rec)
        return rec

    def note_error(self, kind: str = "") -> None:
        self.errors += 1
        if self.turns:
            self.turns[-1].error = kind or "error"

    def note_timeout(self) -> None:
        self.timeouts += 1
        if self.turns:
            self.turns[-1].timeout = True

    def note_interruption(self) -> None:
        self.interruptions += 1
        if self.turns:
            self.turns[-1].interrupted = True

    def note_model_load(self, name: str, seconds: float) -> None:
        self.model_load_seconds[name] = seconds

    @staticmethod
    def peak_rss_mb() -> Optional[float]:
        try:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return kb / 1024.0
        except Exception:
            return None

    # ---------------------------------------------------------------- report
    def report(self) -> str:
        lines = [f"turns={len(self.turns)} errors={self.errors} "
                 f"timeouts={self.timeouts} interruptions={self.interruptions} "
                 f"barge_false_triggers={self.barge_false_triggers}"]
        stages: dict[str, list[float]] = {}
        for t in self.turns:
            for k, v in t.latencies().items():
                stages.setdefault(k, []).append(v)
        for stage, vals in stages.items():
            vals.sort()
            lines.append(f"{stage:>28}: p50={_pct(vals, 0.50):8.1f} ms  "
                         f"p95={_pct(vals, 0.95):8.1f} ms  n={len(vals)}")
        for name, secs in self.model_load_seconds.items():
            lines.append(f"model_load[{name}] = {secs:.2f} s")
        rss = self.peak_rss_mb()
        if rss is not None:
            lines.append(f"peak_rss = {rss:.0f} MB")
        return "\n".join(lines)
