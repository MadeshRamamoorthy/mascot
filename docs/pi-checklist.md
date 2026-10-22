# Pi checklist: branch `claude/hologram-echo`

Things to do on the Pi to roll out and verify this branch. Work top to
bottom. Each section says what changed, what to run, and what "good"
looks like.

"Restart" below means `sudo systemctl restart echo-scope` if the
service is installed, otherwise `./stop.sh && ./start.sh`.

Log file used throughout:

```bash
tail -F /tmp/echo-backend.log
```

---

## 0. Get the code and back up the database

```bash
cd ~/Documents/code/pi5
git fetch
git checkout claude/hologram-echo
git pull
cp faces.db faces.db.bak-$(date +%F)
```

- [ ] Branch checked out
- [ ] `faces.db` backed up

---

## 1. Check `.env` before starting

If `OPENAI_API_KEY` in `.env` holds placeholder text instead of a real
key, ECHO thinks OpenAI is configured. It then skips starting the local
model, and every question fails.

```bash
grep -E "OPENAI_API_KEY|PRELOAD_OLLAMA|DISPLAY_API|WAKE_" .env
```

- [ ] `OPENAI_API_KEY` is a real key (starts with `sk-`), or empty
- [ ] Decide on `PRELOAD_OLLAMA` (see section 6)

---

## 2. Restart and confirm it comes up

```bash
sudo systemctl restart echo-scope
grep -E "asr\] auto-selected|chat-voice\] using|display-api|wake-word\] listening" /tmp/echo-backend.log | tail
```

- [ ] Kiosk shows the idle dashboard
- [ ] Note which speech-to-text backend was picked: `[asr] auto-selected backend: ...`
      (matters for section 6)

---

## 3. Speed

What changed: face detection no longer runs while idle. The camera
isn't JPEG-encoded when no screen shows it. The clock isn't pushed to
the browser every frame. Face matching re-uses the last result for
0.5 s while the same face stays in frame.

```bash
top -p $(pgrep -f main.py | head -1)
```

- [ ] Idle CPU is clearly lower than before
- [ ] A session (walk up, get greeted, ask a question) feels snappier
- [ ] Recognition still names people correctly

If recognition feels sluggish to switch between two people, set
`RECOGNITION_REUSE_SEC=0` in `.env` to embed every frame again.

---

## 4. Wake word waking on its own

What changed:

- Matches below a confidence score are rejected (`WAKE_WORD_MIN_CONF`, default 0.80).
- A wake latched during the end of a session is cleared.
- Wake triggers are ignored for 4 s after a session ends, so ECHO's own goodbye can't re-wake it.
- Every wake now logs its source.

```bash
tail -F /tmp/echo-backend.log | grep -E "wake|\[state\]"
```

Let people talk nearby for a few minutes **without** saying the phrase.

| Log line | Meaning | Action |
|---|---|---|
| `[wake-word] rejected 'hello echo': conf 0.62 < 0.80` | False wake blocked | Nothing, it works |
| `[wake-word] WAKE on 'hello echo' (conf 0.85)` but nobody said it | Still too loose | Raise `WAKE_WORD_MIN_CONF` above that number |
| Real "hello echo scope" gets `rejected` | Too strict | Lower `WAKE_WORD_MIN_CONF` by 0.05 |
| `[wake] touch on idle screen` with nobody touching | Ghost taps | Set `WAKE_ON_TOUCH=false` |

- [ ] No self-wakes over ~10 minutes of background talk
- [ ] Saying the phrase still wakes ECHO reliably
- [ ] Saying "bye" ends the session and ECHO stays asleep afterwards

---

## 5. Employee data cleanup

What changed: `admin.py list` and `export` were crashing and now work.
New `cleanup` command, dry run by default.

```bash
source .venv/bin/activate
python admin.py list
python admin.py cleanup            # report only
python admin.py cleanup --apply    # fix the safe items
python admin.py export employees.csv
```

`--apply` trims stray spaces in names, deletes employees with zero
face samples, deletes orphaned samples, and trims anyone over the
sample cap. Duplicate names are only reported.

- [ ] Reviewed the dry-run report
- [ ] Applied the fixes
- [ ] For each duplicate name, kept one ID and removed the rest: `python admin.py delete <id>`
- [ ] Employee ID format matches the display system (e.g. `E0001`)
- [ ] Exported `employees.csv` for the display-side DB alignment

---

## 6. Offline chat ("I'm having a little trouble reaching the chat service")

**What the local model is.** `qwen3:1.7b` served by Hailo-Ollama on
the Pi at `http://localhost:8000`, running on the Hailo NPU.

**When it is used.** Only when OpenAI is unavailable: no key, or the
Pi can't open a connection to `api.openai.com`. OpenAI is always tried
first.

**Why it failed when the internet was down.**

1. `start.sh` does **not** start Hailo-Ollama when an OpenAI key is
   set, unless `PRELOAD_OLLAMA=1`. It saves ~2.5 GB of memory. So when
   the internet dropped, there was no local model running, and ECHO
   answered "trouble reaching the chat service (No chat backend
   reachable)".
2. If the network is half-up (DNS slow, Wi-Fi up but no route), the
   OpenAI connection check can pass and the request then fails. That
   error is not retried on the local model.
3. Speech-to-text has the same problem. If the Pi has no Hailo Whisper
   files, ECHO uses OpenAI Whisper, which needs internet. It doesn't
   fall back to the local CPU Whisper.

**Check on the Pi:**

```bash
curl -s http://localhost:8000/api/tags | head -c 300; echo   # is the local model up?
which hailo-ollama
grep "auto-selected backend" /tmp/echo-backend.log | tail -1  # hailo-* = offline-safe, openai = needs internet
free -h                                                        # room for +2.5 GB?
```

**Quick fix to test:** add `PRELOAD_OLLAMA=1` to `.env`, restart, then
unplug the network and ask a question. It should answer through the
local model. The first local answer can take several seconds.

- [ ] `hailo-ollama` is installed
- [ ] Offline question answered with `PRELOAD_OLLAMA=1`
- [ ] Speech-to-text works offline (needs a `hailo-*` ASR backend)
- [ ] Memory is OK with the local model loaded (`free -h`)

Points 2 and 3 need code changes. Not done yet:

- retry on the local model when OpenAI fails
- shorter OpenAI timeouts
- fall back to CPU Whisper when OpenAI Whisper fails

---

## 7. Display API (on / off)

What changed: when a recognised person starts a session, ECHO POSTs
`{"state":"on","employee_id":...,"full_name":...}`. When the session
ends (bye, idle timeout, or X), it POSTs `{"state":"off"}`. Calls run
in the background and never block ECHO.

**Test with the mock first.** On the laptop, same network as the Pi:

```bash
python mock_display_api.py --host 0.0.0.0
```

On the Pi, in `.env`:

```bash
DISPLAY_API_URL=http://<laptop-ip>:8765/api/display
DISPLAY_API_KEY=test-key
```

Restart, walk up, get greeted, say "bye". The mock prints:

```
[mock-display] ON E0001 Madesh Ramamoorthy
[mock-display] OFF
```

On the Pi, `grep display-api /tmp/echo-backend.log` shows the status
code of each call.

- [ ] ON arrives when recognised, with the right ID and name
- [ ] OFF arrives after "bye"
- [ ] OFF arrives after the idle timeout and after tapping X
- [ ] No ON for an unrecognised visitor
- [ ] Switched `.env` to the real endpoint and key

---

## 8. Hologram app: employee details + photos

What changed: ECHO now keeps **one face photo per person** (a JPEG crop
in `faces.db`). Each person's ID, name and photo are POSTed to the
hologram app as multipart form data. Fields are `employee_id`,
`full_name` and `image`.

- **New registration** (kiosk or `/admin` photo upload): sent right
  away in the background.
- **Hologram app offline:** the person stays "not sent". The kiosk
  retries every 10 min, and `start.sh` retries before every launch.
- **People registered before this change have no photo.** ECHO saves
  one the next time it recognises them, then sends it.

**Bulk upload (preferred, the hologram app now takes a ZIP).** Build
the ZIP on the Pi, or download it from `/admin` on the Employees tab
with the "Download hologram ZIP" button. Then upload it in the hologram
app. The ZIP holds `employees.csv` (columns `employee_id`, `full_name`,
`photo`) plus one `<employee_id>.jpg` per person. The `photo` cell is
that file name.

```bash
python hologram_sync.py --export-zip hologram_upload.zip
python hologram_sync.py --export-zip hologram_upload.zip --include-no-photo
```

People without a photo are left out unless `--include-no-photo` is
given; their `photo` cell is then empty.

- [ ] ZIP built / downloaded
- [ ] Hologram app accepted it and shows every row with the right photo

**Live sending (optional).** It stays off unless `HOLOGRAM_SYNC_URL` is
set. **Confirm the hologram API format first.** The fields above are an
assumption. If the hologram app expects something else, change
`_build_request` in `hologram_sync.py`.

**Test with the mock** (same laptop mock as section 7):

```bash
# Pi .env
HOLOGRAM_SYNC_URL=http://<laptop-ip>:8765/api/employees
HOLOGRAM_API_KEY=test-key
```

**One-time sync, before the kiosk starts:**

```bash
./stop.sh                                # free memory first
python hologram_sync.py --status         # who is pending / has no photo
python hologram_sync.py --dry-run
python hologram_sync.py                  # send everyone pending, one by one
python hologram_sync.py --all            # force a full re-send
python hologram_sync.py --include-no-photo   # also send people without a photo
```

The exit code tells you how it went:

| Code | Meaning |
|---|---|
| 0 | All sent |
| 1 | Some were rejected (see "last error" in `--status`) |
| 2 | The hologram app was unreachable. Nothing is lost, everyone stays pending |

- [ ] Agreed the request format with the hologram app owner
- [ ] `--status` shows the expected people
- [ ] One-time sync sends everyone with a photo
- [ ] Photos in the laptop's `mock_hologram_photos/` folder look right:
      face centred, **colours correct** (not blue-tinted)
- [ ] Register a new person on the kiosk: it shows up on the mock at once
- [ ] Stop the mock, register someone, start the mock, run
      `python hologram_sync.py`: the missed person is sent
- [ ] An existing person without a photo gets one after being recognised
      once (`[hologram] back-filled photo for ...` in the log)
- [ ] `start.sh` prints `hologram : syncing pending employees` at launch
- [ ] Switched `.env` to the real hologram URL and key

Settings: `HOLOGRAM_SYNC_ON_START=0` skips the start-up sync.
`HOLOGRAM_RETRY_SEC=0` turns off the in-kiosk retry.
`HOLOGRAM_BACKFILL_ON_RECOGNITION=false` stops saving photos for
existing people.

---

## If something goes wrong

Roll back to the previous branch and restore the DB:

```bash
git checkout claude/echo-scope-ui
cp faces.db.bak-<date> faces.db
sudo systemctl restart echo-scope
```
