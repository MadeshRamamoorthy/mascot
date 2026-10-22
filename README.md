# ECHO SCOPE

**A face-recognising kiosk for the Raspberry Pi 5 + Hailo-10H.**
Wake it with your voice, chat with an LLM, the kiosk remembers
returning visitors.

A browser SPA presents a glassmorphic 1280×800 idle dashboard
(weather, projects, upcoming sessions, hi-5 metrics) that switches
to a live camera + transcript view the moment it hears the wake
phrase "hello echo scope". Face recognition, anti-spoof liveness,
voice transcription, and LLM chat all run locally on the Hailo NPU
when possible, with OpenAI as a network-side option.

| | |
|---|---|
| **Hardware** | Raspberry Pi 5 (8 GB) · Pi AI Camera (IMX500) · Hailo-10H M.2 AI HAT 2+ |
| **OS** | Raspberry Pi OS Trixie · Python 3.13 · HailoRT 5.x |
| **NPU workloads** | SCRFD (detect) · ArcFace (embed) · Whisper-Base (STT) · optional `qwen3:1.7b` LLM |
| **Backends** | Chat: OpenAI / Hailo-Ollama · STT: Hailo / OpenAI / faster-whisper / Vosk · TTS: Piper / pyttsx3 |
| **UI** | Chromium kiosk pointed at a local Flask app on `:8090` |

## Quick start

```bash
git clone <your-fork-url> pi5
cd pi5
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env                        # set OPENAI_API_KEY=<API_KEY>

./start.sh
```

Chromium opens in kiosk mode at `http://127.0.0.1:8090`. Say
**"hello echo scope"** to wake.

If you don't yet have HailoRT, models, or the venv, work through
[docs/installation.md](docs/installation.md) first.

## Documentation

Full docs live in [`docs/`](docs/).

| | |
|---|---|
| 📚 [Documentation index](docs/README.md) | Browse all pages |
| 🚀 [Quickstart](docs/quickstart.md) | Three commands to a running kiosk |
| 🧱 [Architecture](docs/architecture.md) | What the SPA looks like, where each piece runs |
| 🔧 [Installation](docs/installation.md) | OS prep, HailoRT, venv, models, audio, browser |
| ⚙️ [Configuration](docs/configuration.md) | `.env` knobs and `config.py` defaults |
| ▶️ [Running the kiosk](docs/running.md) | `start.sh`, `stop.sh`, live logs, auto-start |
| 🎙️ [Speech](docs/speech.md) | Wake word, chat-voice, chat LLM, TTS |
| ⚡ [Hailo Whisper](docs/hailo-whisper.md) | NPU speech-to-text setup |
| 😀 [Face recognition](docs/face-recognition.md) | Pipeline, registration, learning, liveness |
| 🔒 [Admin UI](docs/admin.md) | Standalone admin app on port 8081 (runs independent of kiosk) |
| 🔁 [State machine](docs/state-machine.md) | IDLE / ACTIVE transitions |
| 🎛️ [Tuning](docs/tuning.md) | Every `config.py` knob in one place |
| 🩺 [Troubleshooting](docs/troubleshooting.md) | Symptom → fix |
| 📂 [Project layout](docs/project-layout.md) | Annotated file tree |
| 🔐 [Privacy and limitations](docs/privacy.md) | What's stored, what's sent off-device |
| 🤝 [Contributing](docs/contributing.md) | Branch / commit conventions |

## License

This is an internal project. No license headers are claimed on the
source files. ArcFace, SCRFD, Whisper, Vosk, and Piper models are
each subject to their respective licences — see their upstream
repos.
