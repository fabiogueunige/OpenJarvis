# Wake Word — Verified Findings & Options

## What is actually available (verified against openWakeWord 0.6.0, 2026-10-12)

openWakeWord ships these **pretrained** models only:
`alexa, hey_mycroft, hey_jarvis, hey_siri, jetson, timer, weather, ...`

There is **no pretrained single-word "Jarvis" model**. The closest official model is
**`hey_jarvis`**, which activates on the two-word phrase **"Hey Jarvis"** — exactly the
activation phrase requested for JARVIS.

Project rule: we never silently substitute a different wake word. Therefore:

| Config | Behaviour |
|---|---|
| `word="jarvis"`, `allow_close_match=false` (default) | Startup raises `WakeWordNotAvailableError` with instructions. Nothing wakes. |
| `word="jarvis"`, `allow_close_match=true` | Loads `hey_jarvis`; status line prints `WAKE_DETECTED [model=hey_jarvis]` so you always know which model fired. Activation phrase = **"Hey Jarvis"**. |
| `model_paths=["…/jarvis.onnx"]` | Loads your custom-trained model; exact-word activation. |
| `engine="whisper_hypothesis"` | Wakes when faster-whisper's rolling transcription contains the target word. Slower, more accurate, no extra model needed. |

## How to get an exact "Jarvis" model (practical path)

1. **Train via openWakeWord's positive-data workflow** (recommended, free):
   - Record ~2–5 minutes of yourself saying "Jarvis" in varied tones/distances (16 kHz WAV).
   - Use openWakeWord's training notebooks/scripts to fine-tune and export ONNX.
   - Drop the file at `~/.local/share/jarvis/models/jarvis.onnx` and set `model_paths`.
2. **Porcupine alternative engine** (Picovoice): has a ready-made **"Jarvis"** keyword on the
   free tier (account + access key required; proprietary license). Needs a small adapter
   implementing our `WakeWordDetector` interface.
3. **Whisper-hypothesis engine** (already implemented here): zero training, uses your existing
   STT model; trades CPU/GPU cycles for accuracy. Good fallback while you collect training data.

## Testing honesty

The prototype's unit tests and `--demo` mode use `ScriptedWakeDetector` (deterministic fake).
Real-model performance of `hey_jarvis` on *your* microphone/acoustics has **not** been
measured yet — run the acceptance procedure in README.md ("Manual acceptance test") and tune
`threshold` (default 0.5) based on observed false-wake vs miss rate.
