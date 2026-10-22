# Quickstart

For a Pi 5 + AI HAT 2+ that already has HailoRT 5.x installed, models
in `models/`, and a working `python3.13 -m venv .venv`:

```bash
git clone <your-fork-url> pi5
cd pi5
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env                                  # set OPENAI_API_KEY=<API_KEY>

./start.sh
```

Chromium opens in kiosk mode at `http://127.0.0.1:8090`. Say
**"hello echo scope"** to wake.

Stop with **Ctrl+C** in the launching terminal, or from another
terminal:

```bash
./stop.sh
```

If you don't yet have HailoRT / models / venv, work through
[Installation](installation.md) first.

## Sanity check

```bash
hailortcli --version            # must show 5.x
hailortcli fw-control identify  # Architecture: HAILO10H
arecord -l                       # mic listed?
aplay /usr/share/sounds/alsa/Front_Center.wav  # speaker works?
```

## First wake

1. Stand 0.5–1.5 m from the camera.
2. Say "hello echo scope" (or "hello echo" / "hey echo" — the
   wake-word grammar accepts aliases).
3. Look at the screen. You should hear "Hello! Lovely to see you."
   and see your face inside a yellow "Unknown" bounding box.
4. The registration overlay pops after ~1 s. Fill in `emp_id` +
   name → tap **Register** → follow the pose prompts.
5. On subsequent wakes, the box turns green and the kiosk greets you
   by name.

## Watching what it's doing

```bash
tail -F /tmp/echo-backend.log | grep -E 'wake-word|state|GREET|asr'
```

You'll see something like:

```
[wake-word] partial: 'hello'
[wake-word] partial: 'hello echo'
[wake-word] FINAL  : 'hello echo'
[state] IDLE -> ACTIVE (session 2c4c54cddc77)
[GREET] Welcome, Parakh!
```
