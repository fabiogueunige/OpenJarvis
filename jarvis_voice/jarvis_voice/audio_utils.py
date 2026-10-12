"""Small pure-Python helpers shared by pipeline stages (no device access)."""

from __future__ import annotations

import struct


def rms(pcm: bytes) -> float:
    """RMS amplitude of 16-bit signed little-endian mono PCM."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"<{n}h", pcm[: n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Naive linear resampling of 16-bit mono PCM (used only for playback prep)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    n = len(pcm) // 2
    src = struct.unpack(f"<{n}h", pcm[: n * 2])
    ratio = dst_rate / src_rate
    out_n = max(1, int(n * ratio))
    out = []
    for i in range(out_n):
        pos = i / ratio
        i0 = int(pos)
        i1 = min(i0 + 1, n - 1)
        frac = pos - i0
        val = src[i0] * (1 - frac) + src[i1] * frac
        out.append(max(-32768, min(32767, int(val))))
    return struct.pack(f"<{out_n}h", *out)
