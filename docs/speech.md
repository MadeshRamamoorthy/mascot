# Speech

Four independent subsystems, all reachable from the kiosk:

| Subsystem | Path |
|---|---|
| Wake word | `wake_word.py` (Vosk) |
| Chat voice (STT) | `chat_voice.py` + `asr.py` |
| Chat LLM | `chat.py` |
| TTS | `tts.py` + `async_tts.py` |

---

## Wake word (Vosk)

A small Vosk model with a tight grammar listens continuously while
the kiosk is IDLE. The grammar accepts the primary phrase plus
`WAKE_WORD_ALIASES`:

```
"hello echo scope"  /  "hello echo"  /  "echo scope"  /  "hey echo"
```

The recogniser is recreated on every stream session, which avoids a
known Vosk `FinalizeDecoding` bug after pause/resume cycles.

While ACTIVE, the wake-word listener is **paused** entirely
(`listener.pause()`) — saves CPU, releases the mic for chat-voice,
and prevents a stray "hello" from confusing the session.

**Tuning**: change `WAKE_WORD` and `WAKE_WORD_ALIASES` in `config.py`.
Keep aliases short (1-3 words) — the small Vosk model is shaky on
longer phrases.

---

## Chat-voice (Whisper)

Tap the listening pill → the wake-word listener releases the mic →
`ChatVoiceCapture` opens its own input stream → VAD finalises after
`CHAT_VOICE_SILENCE_SEC` (0.9 s) of silence or `CHAT_VOICE_MAX_SEC`
(12 s) max → ASR backend transcribes.

### Backend auto-precedence

`CHAT_ASR_BACKEND = "auto"` goes through:

| # | Backend | Latency (5 s clip) | Cost | Local? |
|---|---|---|---|---|
| 1 | **Hailo Whisper-Base** | ~250–500 ms | $0 | ✅ |
| 2 | OpenAI Whisper API | ~1–2 s | $0.0002/min | ❌ |
| 3 | faster-whisper tiny.en (CPU) | ~3–4 s | $0 | ✅ |

For Hailo Whisper setup, see [Hailo Whisper](hailo-whisper.md).

To pin a specific backend in `config.py`:

```python
CHAT_ASR_BACKEND = "hailo"      # or "openai" / "faster-whisper" / "vosk"
```

### Visible processing state

The kiosk shows a full-frame mic overlay for **every busy phase** so
the UI never looks frozen during processing:

| State | Label |
|---|---|
| `listening` | "Listening…" |
| `transcribing` | "Got it — transcribing…" |
| `chat_pending` | "Looking that up…" |

Tap-to-cancel works during `listening` only; the processing variant
shows a wait cursor and ignores taps.

---

## Chat (LLM)

`config.CHAT_BACKEND_ORDER = ("openai", "ollama")` tries OpenAI first,
falls back to Hailo-Ollama (`qwen3:1.7b`) if the network call fails.

Per-session budget enforced — once a user hits
`CHAT_MAX_QUESTIONS_PER_SESSION`, the next attempt receives a polite
"that's enough for now" reply.

### OpenAI

Set `OPENAI_API_KEY` in `.env`. The kiosk uses the chat completions
endpoint with a small system prompt that hints at the kiosk persona.

### Hailo-Ollama (offline fallback)

The `hailo-ollama` daemon binds to `:8000` and serves `qwen3:1.7b`
on the Hailo NPU. By default `start.sh` skips it when
`OPENAI_API_KEY` is set (saves 2-3 GB of resident memory). Set
`PRELOAD_OLLAMA=1` to always warm it.

## TTS

`async_tts.AsyncTTS` wraps the synchronous backend with a queue so
the camera loop never blocks waiting for `aplay`. The `on_start`
callback fires from the worker thread immediately *before* audio
plays, so toasts and spoken lines land together rather than the UI
racing ahead of the queue.

| Backend | Quality | Latency |
|---|---|---|
| Piper (`en_US-hfc_female-medium`) | Natural | ~200–400 ms |
| `pyttsx3` + espeak-ng | Robotic | ~50 ms |

Piper auto-loads when its ONNX voice is present in `models/`; the
kiosk falls back to pyttsx3 if not.

### Piper voice install

Piper isn't pip-installable on Python 3.13 yet. On Trixie:

```bash
# Use the espeak fallback OR install Piper on Python 3.11/3.12
python3.12 -m pip install piper-tts
python3.12 -m piper.download_voices en_US-hfc_female-medium --data-dir models/
```

Alternatively use OpenAI TTS for higher quality (`audio.speech.create`
in OpenAI's SDK) — not currently wired in but the integration is
straightforward.
