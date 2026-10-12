# JARVIS Voice Prototype — Install & Run

Target: **Ubuntu 24.04 GNOME**, Python **3.10–3.12**. Everything runs locally; no cloud services, no telemetry.

## 0. System packages (needs your approval — I do not touch system config)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev espeak-ng-data alsa-utils
# PipeWire-based sound on Ubuntu 24.04 works out of the box; pipewire-pulse is usually preinstalled.
```

`espeak-ng-data` is needed by Piper/Kokoro for phonemization. `portaudio19-dev` only if you want to rebuild the audio bindings (sounddevice ships wheels).

## 1. Python environment

```bash
cd jarvis_voice
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[tts-piper,dev]"        # core + Piper TTS + pytest
# or:  pip install -e ".[tts-kokoro,dev]"   for Kokoro instead
```

Dependencies are pinned in `pyproject.toml` (sounddevice, numpy, openwakeword, faster-whisper, piper-tts/kokoro-onnx, pytest).

## 2. Models (downloaded once, stored under ~/.local/share/jarvis)

```bash
mkdir -p ~/.local/share/jarvis/{piper,kokoro,models}

# STT — faster-whisper downloads 'base' automatically on first run (~145 MB, CPU int8).
# Optional GPU: device = "cuda", model_size = "small".

# LLM — via Ollama (already installed? check with `ollama --version`)
ollama pull qwen3:4b          # ~2.5 GB, fits your RTX 4060 8 GB alongside whisper-base

# TTS voice — Piper (recommended first: fastest, lowest VRAM):
python -m piper.download_model en_US-lessac-medium ~/.local/share/jarvis/piper \
  || wget -P ~/.local/share/jarvis/piper \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
  && wget -P ~/.local/share/jarvis/piper https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# TTS alternative — Kokoro (better quality, needs ONNX weights ~330 MB):
wget -P ~/.local/share/jarvis/kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/v1.0.onnx
wget -P ~/.local/share/jarvis/kokoro https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices.bin
```

## 3. Configure

```bash
cp config.example.toml config.toml   # then edit [wake_word] — see docs/WAKE_WORD.md
```

**Wake-word honesty rule:** openWakeWord has **no pretrained single-word "Jarvis" model**. Options:
1. `allow_close_match = true` → uses the community **"hey_jarvis"** model (reacts to *"Hey Jarvis"* — matches your requested phrase, but it is a close match, not a custom-trained word).
2. Train/export your own `jarvis.onnx` and set `model_paths = ["~/.local/share/jarvis/models/jarvis.onnx"]`.
3. `engine = "whisper_hypothesis"` → wakes when Whisper's transcription of the live buffer contains "jarvis" (more accurate, more CPU/GPU).

If none of these is configured, startup **fails loudly** rather than silently waking on a different word.

## 4. Run

```bash
source .venv/bin/activate
python -m jarvis_voice.main --demo            # no mic/models needed; exercises the full state machine
python -m jarvis_voice.main --config config.toml
```

Status line shows: `LISTENING → WAKE_DETECTED → RECORDING → TRANSCRIBING → THINKING → SPEAKING → LISTENING`. Ctrl-C shuts down gracefully.

## 5. Tests

```bash
pytest            # 22 mocked unit tests (fake audio + fake models, no devices required)
```

## Troubleshooting (quick list — full guide below)

| Symptom | Fix |
|---|---|
| `PortAudioError -9996 / no default input device` | Check `pactl info`; pick device: `[audio] input_device = "USB Mic"` substring; `pavucontrol` to set profile (not "Off"). |
| No wake at all | Verify config path (§3); test mic: `python -c "import sounddevice as sd; print(sd.query_devices())"`; lower `silence_threshold_rms` won't help wake — check `threshold`, try `whisper_hypothesis`. |
| `Ollama connection refused` | `systemctl --user status ollama` or run `ollama serve`; confirm `curl localhost:11434/api/tags` lists your model. |
| Slow first reply | Model load happens at startup; keep `keep_alive="10m"`; use `tiny`/`base` whisper + `qwen3:4b`. |
| Piper says nothing / crashes | Ensure `en_US-lessac-medium.onnx` AND `.onnx.json` exist; ensure espeak-ng data installed. |
| `ModuleNotFoundError: piper` | You didn't install the extra: `pip install -e ".[tts-piper]"`. |
| Device disappears after suspend | The loop reopens the stream on error and returns to LISTENING; if stuck, restart the app. |
| GPU contention / CUDA OOM | Set `[stt] device = "cpu"` (base int8 is fast enough) and let Ollama own the GPU. |

## Manual acceptance test (5 min)

1. `python -m jarvis_voice.main --demo` → states cycle automatically, exit with Ctrl-C. ✅ pipeline logic
2. Start normally → status `LISTENING`. Say **"Hey Jarvis"** → `WAKE_DETECTED`, chime optional. ✅ wake
3. Immediately say *"What time is it?"* → `RECORDING` ends after ~1.2 s silence → `TRANSCRIBING` shows your text → `THINKING` → `SPEAKING` (you hear English answer) → back to `LISTENING`. ✅ full loop
4. While it speaks, say "stop" — prototype doesn't barge-in yet (phase 2); note it. 
5. Unplug mic mid-run → status `ERROR`, then auto-retry → `LISTENING`. ✅ fault handling
6. Kill Ollama (`ollama stop` / stop server) → `ERROR: llm unavailable`, returns to LISTENING. ✅ graceful degrade
7. Say nothing after wake → empty transcript → returns to LISTENING without crash. ✅ edge case

## Privacy

Raw audio is never written to disk unless `store_audio = true`. Logs contain transcripts only.
