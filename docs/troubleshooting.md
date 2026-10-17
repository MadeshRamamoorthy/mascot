# Troubleshooting

## Symptom → fix

| Symptom | Likely cause / fix |
|---|---|
| `Port 8090 is in use by another program` | Something else (Open WebUI?) on 8090. Override `KIOSK_PORT` in `.env` |
| `[camera-worker] init failed: ...` | Hailo HEF wrong arch or models missing. `hailortcli parse-hef models/scrfd_10g.hef \| head -1` should say HAILO10H |
| `Address already in use` | Previous backend didn't shut down. `./stop.sh` or `sudo fuser -k 8090/tcp` |
| `Package 'chromium-browser' has no installation candidate` | Pi OS package is `chromium` — `sudo apt install -y chromium`. `start.sh` tries both |
| `[wake-word] disabled: ...` | Vosk or sounddevice failed to import, or `models/vosk-model-small-en-us-0.15/` is missing |
| `sqlite3.ProgrammingError: SQLite objects created in a thread...` | Pull the latest — fixed in commit df21737 (`check_same_thread=False`) |
| Recognising wrong people | Silent learning was over-eager. Cleanup: `sqlite3 faces.db "DELETE FROM face_embeddings WHERE emp_id='X';"` then re-register. Confirm `SILENT_LEARN_MIN_MARGIN >= 0.15` |
| Chat says "Chat unavailable — no backend reachable" | `OPENAI_API_KEY` unset AND Hailo-Ollama not running. Either paste a key in `.env` or `PRELOAD_OLLAMA=1 ./start.sh` |
| Gibberish Whisper output | If using Hailo on Hailo-8/8L, set `HAILO_WHISPER_ADD_EMBED=true` |
| Kiosk drops out mid-chat | Bump `CHAT_KEEPALIVE_SEC`. Watch `[state] ACTIVE -> IDLE` log line for the actual reason |
| First chat takes 3-4 s, subsequent ones are 1 s | Whisper model load on first request. `start.sh` warms it in the background after launch — make sure `[asr] loading...` appears within ~5 s of boot |
| Pi 5 freezes / system unresponsive | Pull latest — likely on the old branch with `vosk-model-en-us-0.22` (2-3 GB). The current code is pinned to the small model |
| Camera shows fine in `libcamera-hello` but blank in the kiosk | The camera worker died. `tail -F /tmp/echo-backend.log` for the traceback |
| `hailortcli scan` shows nothing | PCIe link issue. Run `sudo raspi-config nonint do_pcie_gen 3` and reboot |
| ECHO wakes up by itself | Check the log line right before `[state] IDLE -> ACTIVE`. `[wake] touch on idle screen` means a tap (set `WAKE_ON_TOUCH=false` for a ghost-tapping screen). `[wake] wake word` means Vosk matched: look at the `conf` on the `WAKE on` line and raise `WAKE_WORD_MIN_CONF` above it. Re-wakes right after a goodbye are blocked by `WAKE_COOLDOWN_AFTER_IDLE_SEC` |
| External display stays on / never switches | `grep display-api` in the log. Each call logs its HTTP status. Empty `DISPLAY_API_URL` disables the feature. Test against `python mock_display_api.py` first |
| Duplicate or broken employee records | `python admin.py cleanup` reports them (dry run). `--apply` fixes whitespace, zero-sample rows, orphan samples and over-cap samples. Duplicate names are only reported |
| Wake word never fires | `tail -F /tmp/echo-backend.log \| grep wake-word` to see what Vosk hears. If it's `"the"` / `"a"` / nothing, the mic isn't getting your voice — check `SD_DEVICE` |
| Browser shows "ERR_CONNECTION_REFUSED" | Backend hasn't started. Look for `Error: backend didn't come up` in the terminal where `start.sh` ran |
| `error: externally-managed-environment` on `pip install` | PEP 668 on Trixie. Activate the venv first: `source .venv/bin/activate && pip install ...`. The venv writes inside `.venv/lib/python3.13/`, leaving the apt-managed system Python alone |
| `HailoRTException: ... HAILO_OUT_OF_PHYSICAL_DEVICES (74)` when chat-voice starts | Two VDevices want exclusive access to the NPU. Fixed in `hailo_infer.py` — our `HailoFacePipeline` now creates the VDevice with `scheduling_algorithm=ROUND_ROBIN` and `group_id="SHARED"`, matching hailo-apps's whisper pipeline so they share the chip. Pull the latest |

## Verifying the chip + models

```bash
hailortcli --version            # 5.x required for Hailo-10H
hailortcli fw-control identify  # Architecture: HAILO10H
hailortcli scan                 # pci/0001:01:00.0 listed
hailortcli parse-hef models/scrfd_10g.hef             | head -1
hailortcli parse-hef models/arcface_mobilefacenet.hef | head -1
hailortcli run2 -t 5 set-net models/scrfd_10g.hef     # expect fps: 240+
```

## Verifying mic + speaker

```bash
arecord -l                                          # capture devices
arecord -d 3 -f cd /tmp/test.wav && aplay /tmp/test.wav    # round-trip
python -c "import sounddevice as sd; print(sd.query_devices())"
```

If the mic that shows up isn't picked automatically, set
`SD_DEVICE=<index>` in `.env`.

## Verifying the backend

```bash
curl -s http://localhost:8090/api/state | python -m json.tool | head
```

Expect a JSON document with `state: "IDLE"`, `brand: "ECHO SCOPE"`,
and so on. If the curl returns nothing, the backend isn't up — check
`/tmp/echo-backend.log`.

## Verifying Whisper backend selection

```bash
grep -E 'asr|chat-voice' /tmp/echo-backend.log | head -5
```

Expect one of:

```
[asr] auto-selected backend: hailo-whisper
[asr] auto-selected backend: openai
[asr] auto-selected backend: faster-whisper
```

If you wanted Hailo but got openai, the auto-selector found a missing
file. The wrapper prints the expected path:

```
Hailo Whisper HEF not found: models/whisper-base-encoder.hef
Install hailo-apps (not on PyPI -- clone the repo):
  git clone https://github.com/hailo-ai/hailo-apps.git
  cd hailo-apps
  pip install -e '.[speech-rec]'
```

## Resetting everything

If the database is hopelessly tangled:

```bash
./stop.sh
mv faces.db faces.db.backup
./start.sh
# Re-register everyone from photos via /admin
```

`messages.py`, `config.py`, and `models/` are unaffected.

## Asking for help

When reporting issues, include:

1. The last 50 lines of `/tmp/echo-backend.log`.
2. `hailortcli fw-control identify` output.
3. The output of `pip list | grep -E 'hailo|faster|vosk|piper'`.
4. Which commit you're on: `git log -1 --oneline`.
