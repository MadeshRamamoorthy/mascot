# Privacy and limitations

## What's stored

| Data | Where | Notes |
|---|---|---|
| Face embeddings | `faces.db` table `face_embeddings` | 512-d float32 vectors |
| Face photo | `faces.db` table `employee_photos` | **One JPEG face crop per person**, kept so it can be sent to the hologram app. Saved at registration, or on first recognition for people registered earlier (`HOLOGRAM_BACKFILL_ON_RECOGNITION`). Deleted with the employee. Other camera frames still exist only in RAM |
| Names + emp_ids | `faces.db` table `employees` | Whatever the user / admin typed in |
| Interactions | `faces.db` table `interactions` | `emp_id` + timestamp per recognition |
| Projects | `faces.db` table `projects` | Admin-curated content for the idle dashboard |
| Sessions | `faces.db` table `sessions` | Admin-curated content for the upcoming-session card |
| Backend log | `/tmp/echo-backend.log` | Wiped on every `start.sh`. Contains Vosk transcripts + chat questions |

## What's sent off-device

- **Chat (when `OPENAI_API_KEY` is set)**: each user question goes to
  OpenAI's chat completions endpoint as plain text. OpenAI's
  retention / use policy applies — read theirs before deploying.
- **Whisper (when `CHAT_ASR_BACKEND` resolves to `openai`)**: the raw
  audio of each question (a few seconds of 16 kHz mono WAV) goes to
  OpenAI's `audio.transcriptions` endpoint.
- **Weather**: the kiosk uses IP-geolocation
  (`ipapi.co` / `ipwho.is` / `ip-api.com`) for the city name on the
  dashboard. Each request reveals the kiosk's public IP and gets
  back coordinates + city.
- **Hologram app (when `HOLOGRAM_SYNC_URL` is set)**: each employee's
  ID, name and face photo are sent to that URL. See `hologram_sync.py`.
- **Display app (when `DISPLAY_API_URL` is set)**: the ID and name of
  the person in the current session, plus an "off" when it ends.
- **No analytics, telemetry, or third-party scripts.**

For a fully-local deployment:

1. Leave `OPENAI_API_KEY` unset.
2. Install Hailo Whisper (see [hailo-whisper](hailo-whisper.md)).
3. Set `PRELOAD_OLLAMA=1` to use the local LLM.
4. Disable the weather fetch by editing `weather.py` (or just accept
   the geo lookup — it's not a privacy concern for most setups).

Chat then has no internet dependency.

## Liveness limits

When `LIVENESS_ENABLED = True`, the kiosk runs:

- Texture variance check (printed photos have lower variance)
- Specular highlight ratio (screens reflect more)
- Pixel jitter (frame-to-frame natural micro-motion)
- Relative landmark motion over the window
- Optional active blink challenge

**Effective against** printed photos, basic phone-screen replays,
and most static reproductions.

**Not effective against** a high-quality video on a large monitor
with natural ambient motion. The blink challenge raises the bar but
isn't bulletproof either.

For high-stakes deployments (access control, payments), combine
with:

- Depth sensing (3D structured-light or stereo)
- A dedicated liveness model
- Multi-factor (e.g. PIN + face)

The kiosk in this repo is designed for **friendly recognition** —
greeting visitors, tracking interaction counts, opening a chat. It
is not designed as a security gate.

## Recognition drift

ArcFace is a 512-d embedding cosine-matched against a per-person
gallery. Silent learning expands the gallery opportunistically but
is gated by:

- Match score ≥ 0.70
- Runner-up score ≥ 0.15 below the best
- New sample isn't a near-duplicate (cos ≤ 0.92)
- Max one new sample per minute per person

False positives still happen — especially for genuinely similar
people. Clean drifted galleries via the Employees admin tab or
`sqlite3 faces.db`:

```bash
sqlite3 faces.db "DELETE FROM face_embeddings WHERE emp_id='E001';"
# Then re-register via the kiosk or /admin -> Register from photo
```

## Chat budget

Each ACTIVE session is capped at
`CHAT_MAX_QUESTIONS_PER_SESSION` (default 25). The cap is a backstop
against a visitor camping in front of the kiosk running up an API
bill; it's high enough that a normal conversation never hits it, and
the countdown badge is hidden in the UI.

Budget resets on every IDLE → ACTIVE transition.

## Data retention & event reset

Registration stores face embeddings + a name in `faces.db`. Two ways
to clear that biometric data:

- **Manual event reset (default):** in the admin panel's Employees
  tab, click **Wipe all faces**. This deletes every employee + their
  embeddings. Hi-5 interaction *counts* are kept (they're anonymous
  aggregates). Irreversible.
- **Automatic retention:** set `DATA_RETENTION_HOURS` (env var, default
  `0` = off). When > 0, a background sweep deletes face data older than
  that window — e.g. `DATA_RETENTION_HOURS=24` clears a one-day booth's
  registrations automatically the next day.

The registration overlay shows a consent line ("Your face is stored
only on this device to recognise you, and is deleted after the event.")
so visitors know before they register.

## Audio capture

While the kiosk is IDLE, the wake-word listener is **always on**.
It runs a grammar-locked Vosk recogniser over a continuous mic
stream and only recognises the four wake phrases — anything else
maps to `[unk]` and is discarded. The audio buffer is small (~500 ms
rolling) and never written to disk.

When `chat-voice` records a question, the raw PCM is held in RAM
just long enough to transcribe (either locally or via the API).
It's not persisted.

If you want to turn off mic access entirely, run with
`--no-wake-word`. The kiosk then only wakes from a screen tap.
