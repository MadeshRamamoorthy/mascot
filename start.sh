#!/usr/bin/env bash
# Start the face-recognition app with the right mic + speaker pinned.
#
# Override any of these by exporting them before running, or by editing
# the defaults below. The script also forwards any extra args you pass
# straight through to main.py (e.g. ./start.sh --no-display).
#
#   SD_DEVICE             sounddevice index for the mic.
#                         List with: python debug_mic.py --list
#   AUDIO_OUTPUT_DEVICE   ALSA name for the speaker.
#                         List with: aplay -L
#   PI5_DIR               project directory.
#   EXTRA_ARGS            default args passed to main.py.
#
# Examples:
#   ./start.sh
#   ./start.sh --no-display
#   SD_DEVICE=3 ./start.sh
#   EXTRA_ARGS="--no-wake-word --auto-register" ./start.sh

set -euo pipefail

# ---- Defaults (edit these once for your machine) -------------------------
: "${PI5_DIR:=/home/echo/Documents/code/pi5}"

# ---- Local config (.env) -------------------------------------------------
# If a .env file exists in PI5_DIR, source it so the user can keep their
# OpenAI key + admin password + any other knobs in one git-ignored file
# rather than editing this script. Run `cp .env.example .env` once,
# then `nano .env` to fill in the values.
if [[ -f "$PI5_DIR/.env" ]]; then
    set -a       # auto-export every variable sourced below
    # shellcheck disable=SC1091
    source "$PI5_DIR/.env"
    set +a
    ECHO_ENV_LOADED=1
else
    ECHO_ENV_LOADED=0
fi
: "${SD_DEVICE:=1}"
: "${AUDIO_OUTPUT_DEVICE:=plughw:CARD=PowerConf,DEV=0}"
: "${EXTRA_ARGS:=--auto-register}"
# OpenAI is the preferred chat backend; Ollama on localhost is the fallback
# when this is unset or unreachable.
: "${OPENAI_API_KEY:=}"

# Silence the harmless cv2/Qt font warning (main.py also sets this; doing
# it here too means the warning is gone even before Python starts).
export QT_LOGGING_RULES="qt.qpa.fonts.warning=false"

# ---- Sanity checks -------------------------------------------------------
cd "$PI5_DIR"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Error: .venv not found at $PI5_DIR/.venv" >&2
    echo "Set up the venv first (see README §2.5)." >&2
    exit 1
fi

if [[ ! -f models/scrfd_10g.hef || ! -f models/arcface_mobilefacenet.hef ]]; then
    echo "Error: HEFs missing in $PI5_DIR/models/" >&2
    echo "Download them (see README §2.7)." >&2
    exit 1
fi

# Optional: warn if Vosk model dir is missing -- we still proceed because
# main.py falls back to ACTIVE state without the wake word.
if [[ ! -d models/vosk-model-small-en-us-0.15 ]]; then
    echo "Note: Vosk model not found; wake-word will be disabled." >&2
    echo "      Download (see README §2.8) to enable 'hello echo'." >&2
fi

# ---- Display / X handling -----------------------------------------------
# Common pitfall: running this script as root drops the user's X session
# variables, so cv2's Qt preview can't connect to the display. Fix it up
# here automatically. If we still can't reach an X display, fall back to
# --no-display so the app at least runs headless.
if [[ $EUID -eq 0 ]]; then
    echo "Warning: running as root. Prefer running as the 'echo' user." >&2
    if [[ -z "${DISPLAY:-}" ]]; then
        export DISPLAY=":0"
    fi
    if [[ -z "${XAUTHORITY:-}" && -f /home/echo/.Xauthority ]]; then
        export XAUTHORITY=/home/echo/.Xauthority
    fi
fi

# (X probe runs after venv activation -- needs python from venv.)

# ---- Activate venv & launch ---------------------------------------------
# shellcheck source=/dev/null
source .venv/bin/activate

export SD_DEVICE
export AUDIO_OUTPUT_DEVICE
export OPENAI_API_KEY

# ---- Admin credentials ---------------------------------------------------
# Required to reach /admin. If unset we generate a short random password
# at every launch so the page is always behind auth. Set KIOSK_ADMIN_PASS
# in ~/.bashrc or systemd unit to make it stable across restarts.
: "${KIOSK_ADMIN_USER:=admin}"
if [[ -z "${KIOSK_ADMIN_PASS:-}" ]]; then
    KIOSK_ADMIN_PASS="$(python -c 'import secrets; print(secrets.token_urlsafe(6))')"
    ADMIN_PASS_GENERATED=1
else
    ADMIN_PASS_GENERATED=0
fi
export KIOSK_ADMIN_USER KIOSK_ADMIN_PASS

# ---- Hailo-Ollama (offline-only fallback chat backend) -------------------
# Online OpenAI is the preferred backend. Hailo-Ollama is the offline
# fallback -- runs LLM inference on the Hailo-10H. It also holds 2-3 GB
# resident *just sitting there*, so on a Pi 5 8 GB we skip starting it
# when OpenAI is configured. Override with PRELOAD_OLLAMA=1 if you want
# instant chat replies if/when OpenAI is unreachable.
# NOTE: hailo-ollama on this Pi binds to :8000. Upstream Ollama uses
# 11434, and Open WebUI uses 8080 -- override OLLAMA_URL if your build
# differs.
: "${OLLAMA_URL:=http://localhost:8000}"
: "${PRELOAD_OLLAMA:=0}"

if [[ -n "$OPENAI_API_KEY" ]] && [[ "$PRELOAD_OLLAMA" != 1 ]]; then
    echo "   hailo-ollama : skipped (OpenAI primary; PRELOAD_OLLAMA=1 to start)"
else
# Sanity-probe the API by asking for the JSON tag list. /api/tags on
# real Ollama returns a {"models":[...]} document; uvicorn / Open WebUI
# returns HTML, which is how we caught the previous misconfiguration.
ollama_up() {
    local out
    out=$(curl -sf -m 1 -H "Accept: application/json" \
                "${OLLAMA_URL}/api/tags" 2>/dev/null) || return 1
    [[ "$out" == \{* ]] || return 1
    return 0
}

    if ollama_up; then
        echo "   hailo-ollama : up at ${OLLAMA_URL}"
    elif command -v hailo-ollama >/dev/null 2>&1; then
        echo "   hailo-ollama : not running, launching in background..."
        : "${HAILO_OLLAMA_LOG:=/tmp/hailo-ollama.log}"
        # The Pi build runs the daemon by invoking the binary with no args.
        # Override via HAILO_OLLAMA_CMD if your build uses a subcommand.
        : "${HAILO_OLLAMA_CMD:=hailo-ollama}"
        nohup $HAILO_OLLAMA_CMD >"$HAILO_OLLAMA_LOG" 2>&1 &
        # Wait up to ~10 s for the API to answer.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            ollama_up && break
            sleep 1
        done
        if ollama_up; then
            echo "   hailo-ollama : up at ${OLLAMA_URL} (log: $HAILO_OLLAMA_LOG)"
        else
            echo "   hailo-ollama : failed to start -- see $HAILO_OLLAMA_LOG" >&2
            echo "                  chat will use OpenAI only (or be unavailable" >&2
            echo "                  if OPENAI_API_KEY is unset)" >&2
        fi
    else
        echo "   hailo-ollama : binary not found in PATH -- offline chat disabled" >&2
    fi
fi

# Probe X display reliably *after* venv activation -- uses libX11 via
# ctypes so we don't depend on x11-utils being installed.
x_probe="$(python - <<'PY' 2>/dev/null
import os, ctypes, sys
disp = os.environ.get("DISPLAY", "")
if not disp:
    print("none"); sys.exit(0)
try:
    libx = ctypes.CDLL("libX11.so.6")
    libx.XOpenDisplay.restype = ctypes.c_void_p
    libx.XCloseDisplay.argtypes = [ctypes.c_void_p]
    d = libx.XOpenDisplay(disp.encode())
    if d:
        libx.XCloseDisplay(d)
        print("ok")
    else:
        print("unreachable")
except Exception:
    # libX11 not present -- almost certainly no X.
    print("none")
PY
)"

case "$x_probe" in
    ok)
        ;;
    none)
        echo "Note: no X display ('$DISPLAY'); running headless." >&2
        case " $EXTRA_ARGS $* " in
            *" --no-display "*) ;;
            *) EXTRA_ARGS="$EXTRA_ARGS --no-display" ;;
        esac
        ;;
    unreachable)
        echo "Note: DISPLAY='$DISPLAY' set but unreachable (auth or no server)." >&2
        echo "      Falling back to headless. To get the GUI:" >&2
        echo "        - run as the user that owns the desktop session," >&2
        echo "        - or switch the session to X11 (sudo raspi-config" >&2
        echo "          -> Advanced -> Wayland -> X11) and reboot." >&2
        case " $EXTRA_ARGS $* " in
            *" --no-display "*) ;;
            *) EXTRA_ARGS="$EXTRA_ARGS --no-display" ;;
        esac
        ;;
esac

# ASR warm-up now happens IN-PROCESS inside main.py -- a separate
# python -c invocation here can't share the loaded pipeline with the
# kiosk backend, so it never actually helped first-chat latency.

# ---- Hologram catch-up sync -----------------------------------------------
# Send any employees not yet delivered to the hologram app (new since the
# last run, or missed while it was offline). Runs here, BEFORE the kiosk,
# so it doesn't compete with the camera + NPU for memory. Never blocks
# start-up: bounded by a timeout and failures are ignored.
if [[ -n "${HOLOGRAM_SYNC_URL:-}" ]] && [[ "${HOLOGRAM_SYNC_ON_START:-1}" != 0 ]]; then
    echo "   hologram   : syncing pending employees -> $HOLOGRAM_SYNC_URL"
    timeout "${HOLOGRAM_SYNC_START_TIMEOUT:-180}" python hologram_sync.py         2>&1 | sed 's/^/                /' || true
fi

# ---- Launch the kiosk ----------------------------------------------------
# Backend (Flask + camera worker) runs in the background; once it's
# serving /api/state we open Chromium in kiosk mode pointed at the SPA.
: "${KIOSK_PORT:=8090}"
: "${KIOSK_URL:=http://127.0.0.1:${KIOSK_PORT}}"
: "${KIOSK_BROWSER:=auto}"
: "${KIOSK_BACKEND_LOG:=/tmp/echo-backend.log}"

echo "---------------------------------------------------------------"
echo " ECHO SCOPE kiosk"
echo "   project    : $PI5_DIR"
if [[ "$ECHO_ENV_LOADED" == 1 ]]; then
    echo "   config     : .env loaded"
else
    echo "   config     : (no .env -- copy .env.example to .env to set keys)"
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "   openai     : key set (***${OPENAI_API_KEY: -4})"
else
    echo "   openai     : no key -- chat will fall back to hailo-ollama"
fi
echo "   mic        : sounddevice index $SD_DEVICE"
echo "   speaker    : $AUDIO_OUTPUT_DEVICE"
echo "   kiosk URL  : $KIOSK_URL"
echo "   admin URL  : http://<host>:8081/   (start with ./start_admin.sh)"
echo "   backend log: $KIOSK_BACKEND_LOG"
if [[ "$ADMIN_PASS_GENERATED" == 1 ]]; then
    echo "   admin user : $KIOSK_ADMIN_USER"
    echo "   admin pass : $KIOSK_ADMIN_PASS    (generated -- set KIOSK_ADMIN_PASS to pin)"
else
    echo "   admin auth : $KIOSK_ADMIN_USER (KIOSK_ADMIN_PASS from env)"
fi
echo "   args       : $EXTRA_ARGS $*"
echo "---------------------------------------------------------------"

# Start backend in the background; trap shutdown so killing this script
# stops the python process too.
python main.py $EXTRA_ARGS "$@" >"$KIOSK_BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null; wait $BACKEND_PID 2>/dev/null; exit' \
     INT TERM

# Wait up to 30 s for the backend to answer.
for _ in $(seq 1 60); do
    if curl -sf -m 1 "${KIOSK_URL}/api/state" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
if ! curl -sf -m 1 "${KIOSK_URL}/api/state" >/dev/null 2>&1; then
    echo "Error: backend didn't come up -- see $KIOSK_BACKEND_LOG" >&2
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
echo "   backend    : up (pid $BACKEND_PID)"

# Pick a browser binary.
pick_browser() {
    if [[ "$KIOSK_BROWSER" != "auto" ]]; then
        echo "$KIOSK_BROWSER"; return
    fi
    for b in chromium-browser chromium google-chrome chrome firefox; do
        if command -v "$b" >/dev/null 2>&1; then
            echo "$b"; return
        fi
    done
    echo ""
}

BROWSER=$(pick_browser)
if [[ -z "$BROWSER" ]]; then
    echo "Note: no kiosk browser found. Open ${KIOSK_URL} manually." >&2
    wait $BACKEND_PID
else
    echo "   browser    : $BROWSER (kiosk mode)"
    case "$BROWSER" in
        chromium*|google-chrome|chrome)
            BROWSER_CMD=("$BROWSER"
                --kiosk --noerrdialogs --disable-infobars
                --disable-features=TranslateUI
                --autoplay-policy=no-user-gesture-required
                --app="$KIOSK_URL")
            ;;
        *)
            BROWSER_CMD=("$BROWSER" "$KIOSK_URL")
            ;;
    esac
    if [[ -n "${INVOCATION_ID:-}" ]]; then
        # Under systemd: run the browser in the background and exit as soon
        # as EITHER the backend or browser dies, tearing the other down, so
        # Restart=always gives a clean full relaunch. The `if` consumes
        # wait's status so `set -e` doesn't abort before cleanup.
        "${BROWSER_CMD[@]}" &
        BROWSER_PID=$!
        if wait -n "$BACKEND_PID" "$BROWSER_PID"; then
            status=0
        else
            status=$?
        fi
        kill "$BACKEND_PID" "$BROWSER_PID" 2>/dev/null || true
        wait 2>/dev/null || true
        exit $status
    else
        # Run by hand: exec the browser (old behaviour). The backend keeps
        # running independently of this shell, so a browser hiccup never
        # takes the web server down.
        exec "${BROWSER_CMD[@]}"
    fi
fi
