"""Sentence/phrase chunking of streamed LLM output for speak-as-you-generate.

The streamer is deliberately conservative: it only emits *complete* sentences
(terminated by . ! ? :, followed by whitespace or end-of-stream) so TTS never
speaks malformed fragments. Mid-sentence text is buffered until a sentence
boundary, an abbreviation-safe soft break, or the size cap forces emission.

Guards:
* ``min_chunk_chars`` — don't emit absurdly short blips ("Hi." alone is fine;
  "Th" is not).
* ``max_chunk_chars`` — safety valve if a model streams one giant run-on.
* unterminated quotes/parens at a candidate boundary delay the cut (heuristic
  against mid-quote splits).
* final flush at end-of-stream emits any trailing complete clause; an
  incomplete trailing fragment below ``min_chunk_chars`` is dropped rather
  than spoken.
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

_TERMINATORS = ".!?"
_SOFT_BREAKS = ",:"
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "no", "yes?", "approx", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec", "am", "pm", "fig", "ref",
}
# "3.5", "$1.00", "e.g.", "U.S." — decimal points and dotted initials are
# NOT sentence ends.
_DECIMAL_RE = re.compile(r"\d\.\d$")
_DOTTED_INIT_RE = re.compile(r"(?:\b[A-Za-z]\.){1,}$")


class SentenceChunker:
    """Incrementally turns token deltas into speakable sentence chunks."""

    def __init__(self, min_chunk_chars: int = 4, max_chunk_chars: int = 260,
                 first_chunk_max_chars: int = 80) -> None:
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        # Speak sooner on the *first* chunk: allow a soft break (comma/colon)
        # there so the user hears something quickly even with long sentences.
        self.first_chunk_max_chars = first_chunk_max_chars
        self._buf = ""
        self._emitted_any = False

    # ------------------------------------------------------------------ api
    def feed(self, delta: str) -> list[str]:
        """Add streamed text; return zero or more completed chunks."""
        out: list[str] = []
        self._buf += delta
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if len(chunk) >= self.min_chunk_chars:
                out.append(chunk)
            else:  # too small to be worth speaking alone — keep buffering
                self._buf = chunk + " " + self._buf
                break
            self._emitted_any = True
        return out

    def flush(self) -> list[str]:
        """End-of-stream: emit remaining complete-looking text, drop crumbs."""
        out: list[str] = []
        tail = self._buf.strip()
        self._buf = ""
        if tail and len(tail) >= self.min_chunk_chars:
            # If the tail ends mid-sentence (no terminator), still speak it —
            # the answer is over; better a full tail than silence.
            out.append(tail)
            self._emitted_any = True
        return out

    @property
    def emitted_any(self) -> bool:
        return self._emitted_any

    # ----------------------------------------------------------------- cut
    def _find_cut(self) -> Optional[int]:
        buf = self._buf
        limit = (self.first_chunk_max_chars
                 if not self._emitted_any else self.max_chunk_chars)
        if len(buf) < self.min_chunk_chars:
            return None
        scan_from = max(0, self.min_chunk_chars - 1)
        for i in range(scan_from, len(buf)):
            ch = buf[i]
            room_ok = i + 1 <= limit or i + 1 >= self.max_chunk_chars
            if ch in _TERMINATORS:
                nxt = buf[i + 1] if i + 1 < len(buf) else ""
                if nxt and not nxt.isspace():
                    continue                      # "3.5", "U.S." handled below
                word = _last_word(buf[:i])
                if word.lower().rstrip(".") in _ABBREVIATIONS:
                    continue
                if _DECIMAL_RE.search(buf[max(0, i - 2):i]):
                    continue
                if _DOTTED_INIT_RE.search(buf[max(0, i - 4):i + 1]):
                    continue
                if _unbalanced(buf[:i + 1]):
                    continue                      # inside quotes/parens
                if room_ok:
                    return i + 1
            elif ch in _SOFT_BREAKS and not self._emitted_any:
                # early soft break only for the first chunk, once we already
                # have something meaningful to say
                if i + 1 >= 25 and buf[i + 1:i + 2].isspace() and not _unbalanced(buf[:i + 1]):
                    return i + 1
        if len(buf) >= self.max_chunk_chars:
            # forced cut at last whitespace within the window
            j = buf.rfind(" ", 0, self.max_chunk_chars)
            return j if j > self.min_chunk_chars else self.max_chunk_chars
        return None


def _last_word(prefix: str) -> str:
    words = prefix.rstrip().split(" ")
    return words[-1] if words else ""


def _unbalanced(text: str) -> bool:
    q = text.count('"') - text.count('\u201c') * 0  # simple double-quote parity
    return (q % 2 == 1) or (text.count("(") != text.count(")"))


def iter_chunks(deltas: Iterator[str], chunker: Optional[SentenceChunker] = None
                ) -> Iterator[str]:
    """Convenience: map an iterator of token deltas to an iterator of chunks."""
    c = chunker or SentenceChunker()
    for d in deltas:
        for chunk in c.feed(d):
            yield chunk
    for chunk in c.flush():
        yield chunk
