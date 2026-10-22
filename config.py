from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "faces.db"

# Vosk model directory. We use the SMALL model only -- it's enough for
# the four-word wake phrase and uses ~200 MB instead of 2-3 GB. Chat
# free-form dictation goes through Whisper (see CHAT_ASR_BACKEND below)
# so the big Vosk model isn't needed.
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-small-en-us-0.15"

# Hailo-10H compiled models (HEF). Download via the install guide.
DETECTOR_HEF = MODELS_DIR / "scrfd_10g.hef"
EMBEDDER_HEF = MODELS_DIR / "arcface_mobilefacenet.hef"

# Detector
DETECTOR_INPUT = (640, 640)
DETECTOR_SCORE_THRESHOLD = 0.5
DETECTOR_NMS_IOU = 0.4

# Embedder
EMBEDDER_INPUT = (112, 112)
EMBEDDING_DIM = 512

# Recognition
COSINE_MATCH_THRESHOLD = 0.42  # ArcFace: 0.35-0.45 typical; lower => looser match. Bumped after a live deployment where Parakh's gallery had silent-learned 25 extra polluted samples and was matching other faces -- tighter threshold + tighter silent-learn margins (below) prevent recurrence.
# Each emp_id is greeted at most once per this many seconds. Lets the
# system greet every recognised face in a multi-person scene without
# spamming when someone keeps stepping in and out of frame.
GREET_COOLDOWN_SEC = 60
# Appended to a recognised person's greeting so they know how to start
# talking. Set to "" to disable.
# Spoken after the personal welcome and BEFORE GREET_TAP_PROMPT --
# introduces Leon (the sibling AI persona) to every recognised visitor
# so they know he exists. Set to "" to disable.
LEON_INTRODUCTION = "I am my distant cousin of Leon, Infosys AI Mascot."
GREET_TAP_PROMPT = "Tap to speak to chat with me!"

# Camera
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FRAMERATE = 30

# ---- HUD: side transcript panel ------------------------------------------
# When True, the OpenCV window is split: camera on the left, a live
# transcript of mic input + TTS output + recognition / state events on
# the right. Set False to keep the simple camera-only window.
SHOW_TRANSCRIPT_PANEL = True
TRANSCRIPT_PANEL_WIDTH = 420       # pixels added to the right of the camera
TRANSCRIPT_MAX_EVENTS = 16         # ring-buffer length

# Final display scale applied after compositing camera + transcript.
# Keep at 1.0 for full-resolution display. Lower values fit the window
# onto smaller screens (recommended values below). Detection / liveness
# / quality logic still runs on the full-resolution camera frame, so
# accuracy is unaffected -- only the displayed window shrinks.
#
#   1.0  -> native (camera + panel = ~1700 x 720)
#   0.75 -> ~1275 x 540   (fits 1280 x 800 panels)
#   0.6  -> ~1020 x 432   (fits 1024 x 600 panels)
#   0.5  -> ~850  x 360
DISPLAY_SCALE = 1.0

# ---- Face quality gate ---------------------------------------------------
# A "real" face must clear ALL of these before recognition / registration.
QUALITY_SCORE_THRESHOLD = 0.70    # min detector score (overrides DETECTOR_SCORE_THRESHOLD for gating)
QUALITY_MIN_FACE_PIXELS = 110     # min bbox width AND height in pixels
QUALITY_FRAME_EDGE_MARGIN = 12    # face must be at least this far from any frame edge
QUALITY_MIN_EYE_DISTANCE = 28     # min inter-pupil distance in pixels
QUALITY_MAX_EYE_TILT = 0.45       # max |dy/dx| between the two eyes (rejects extreme roll)

# Anti-flicker: the largest face must be of acceptable quality AND below the
# match threshold for this many consecutive frames before we offer to register.
UNKNOWN_FRAMES_BEFORE_REGISTER = 15

# ---- Registration: voice-guided poses ------------------------------------
# Each entry: (spoken prompt, expected landmark-shift direction relative to the
# previous capture). The shift is checked in normalised units of the inter-eye
# distance, so it works at any range. Use None for the first/calibration pose.
#   ("right",  +x shift in image)   ("left", -x)   ("up", -y)   ("down", +y)
# Single frontal pose: registration is now "one good picture" -- the user
# just looks at the camera and we grab POSE_FRAMES_PER_POSE stable frames in
# a couple of seconds. Add the head-turn lines back below for denser angle
# coverage if recognition struggles at extreme angles:
#   ("Slowly turn a little to your right.", "right"),
#   ("Now turn a little to your left.",     "left"),
POSE_PROMPTS = [
    ("Look straight at the camera and hold still.", None),
]

# How many stable frames to capture at each pose. With a single frontal pose
# this is the total number of enrolment embeddings stored for a new person.
# 3 gives the matcher a little natural variation (micro head movements,
# blinks) without dragging the capture out.
POSE_FRAMES_PER_POSE = 3

POSE_HOLD_SEC = 1.0            # min time after prompt before we even *try* to capture
POSE_STABLE_SEC = 0.4          # face landmarks must stay still for this long before capture
POSE_STABLE_PIXEL_TOL = 4.0    # max landmark drift (px) within the stability window
POSE_MIN_SHIFT = 0.25          # min directional shift, in eye-distance units
POSE_CAPTURE_TIMEOUT_SEC = 7.0 # give up on this pose if user doesn't comply

# Re-register: the new face must self-match this strongly to count as the
# same person before we append samples to an existing emp_id.
REREGISTER_MATCH_THRESHOLD = 0.35

# ---- Active liveness (blink challenge) ------------------------------------
# When True, every emp_id must blink once per ACTIVE session before they
# get greeted. Clears on IDLE so the next session re-prompts. Defeats the
# remaining attack vector: a high-quality video replay on a screen.
LIVENESS_REQUIRE_BLINK = False
LIVENESS_BLINK_PROMPT = "Just a quick blink so I know it's you!"
LIVENESS_BLINK_TIMEOUT_SEC = 5
LIVENESS_BLINK_PATCH_PX = 14         # half-extent (px) around each eye landmark
LIVENESS_BLINK_DELTA_MIN = 6.0       # min std-dev range across the window
LIVENESS_BLINK_REVERIFY_SEC = 600    # within an ACTIVE session, this is huge
                                     # (1h); blinks are session-scoped anyway

# ---- TTS backend ----------------------------------------------------------
# "piper"   -> neural Piper via the piper1-gpl Python package
#              (pip install piper-tts). Falls back to pyttsx3 automatically
#              if the package or the voice model isn't installed yet.
# "pyttsx3" -> espeak-ng. Robotic but always available.
TTS_BACKEND = "piper"

# Default voice. Drop more .onnx + .onnx.json pairs into models/piper/
# and point this at any of them to switch voices. See README §2.9.
PIPER_MODEL_PATH = MODELS_DIR / "piper" / "en_US-hfc_female-medium.onnx"

# Pad the start of each TTS utterance with this much silence. USB speakers
# (Anker A3301 etc.) often clip the first ~300 ms while their amp wakes up.
# 0 disables. In the streaming TTS path the silence is written to aplay's
# stdin before the first real audio chunk; in the legacy file path it's
# inserted into the WAV. Kept small (150 ms) so the voice starts promptly
# after a reply is posted -- bump back toward 300-500 if the speaker clips
# the first syllable.
TTS_PREBUFFER_MS = 150

# Stream TTS audio to aplay as soon as Piper emits each chunk, rather than
# synthesising the whole utterance to a temp WAV first. Saves ~1-2 s on
# long replies before audio starts. Falls back to the file path
# automatically if the installed piper version doesn't expose the
# generator API.
# Set TTS_STREAMING=false in .env to force the file path (useful if a
# specific voice / hardware combination has audio glitches on streaming).
TTS_STREAMING = os.environ.get("TTS_STREAMING", "true").lower() in (
    "1", "true", "yes",
)

# ---- Audio output for TTS -------------------------------------------------
# None  -> use system default audio sink (HDMI / 3.5 mm / whatever PipeWire
#          decides). To pin TTS to the Anker (or any specific output), set
#          this to an ALSA name from `aplay -L`, e.g.:
#            "plughw:CARD=PowerConf,DEV=0"
#            "plughw:2,0"
# Also overridable at runtime via the AUDIO_OUTPUT_DEVICE env var (the
# start.sh script uses this).
# When set, pyttsx3 synthesises to a temporary WAV and aplay plays it on
# the chosen device, so this works even when the system default is HDMI.
AUDIO_OUTPUT_DEVICE: str | None = os.environ.get("AUDIO_OUTPUT_DEVICE") or None

# ---- Wake-word / activation -----------------------------------------------
BRAND_NAME = "ECHO SCOPE"
# Spoken when a visitor asks "what is ECHO?" / "what does ECHO stand for?".
ECHO_FULL_FORM = "Enterprise Center for Human-AI Outcomes, Amplifying Intent"
# Spoken when a visitor asks about the AI lab itself ("what is the AI lab?",
# "what's special about Calgary's AI lab?", "what's the motive of the lab?").
AI_LAB_ABOUT = (
    "ECHO (Enterprise Center for Human-AI Outcomes) is Calgary's dedicated "
    "AI lab — a collaborative space where professionals, builders, and "
    "learners come together to work on real AI projects, share knowledge, "
    "and push the boundaries of what's possible with AI. Whether you're "
    "building something new or just getting started, ECHO is where intent "
    "meets action."
)
# Spoken when a visitor asks "do you have a message for the staff meeting?"
# / "any message for the team / all-hands / town hall?". Routed locally
# (no LLM, no budget slot), so the wording here is exactly what ECHO says.
STAFF_MEETING_MESSAGE = (
    "A warm welcome to our executive leaders from ECHO SCOPE. We hope your "
    "meeting is going wonderfully. This is an extraordinary time for AI — "
    "and we'd be delighted to host you at Infosys Calgary. Come visit us "
    "and experience our AI journey first-hand."
)
CONTACT_EMAIL = "Calgary_AIClub@infosys.com"
WAKE_WORD = "hello echo scope"        # primary phrase shown in the UI
# Aliases also accepted by the wake-word grammar. We accept the short
# "hello echo" / "hi echo" alongside the full "hello echo scope" -- the
# small Vosk model is more reliable on the shorter phrases, and all are
# natural to say.
# Dropped "hi echo": in production logs Vosk was finalising ambient
# chatter as "hi echo ..." often enough to wake the kiosk to an empty
# room. Keeping "hello echo" (still short) and the full "hello echo
# scope" (via WAKE_WORD).
WAKE_WORD_ALIASES = ["hello echo"]
WAKE_WORD_SAMPLERATE = 16000
WAKE_WORD_BLOCKSIZE = 8000
# Vosk's grammar-restricted recognizer forces EVERY utterance onto one of
# the grammar phrases or [unk], so ambient chatter is regularly decoded as
# "hello echo" with low confidence. That was the main cause of ECHO waking
# on its own. Require the mean per-word confidence of a final result to be
# at least this high. Raise it if ECHO still wakes on its own; lower it if
# the real phrase stops working. The [wake-word] log prints the confidence.
WAKE_WORD_MIN_CONF = float(os.environ.get("WAKE_WORD_MIN_CONF", "0.80"))
# After a session ends, ignore wake triggers for this long. Stops ECHO's own
# farewell (played through the speaker into the mic) or a tap that landed
# during the ACTIVE -> IDLE transition from instantly re-waking the kiosk.
WAKE_COOLDOWN_AFTER_IDLE_SEC = float(
    os.environ.get("WAKE_COOLDOWN_AFTER_IDLE_SEC", "4"))
# Tap anywhere on the idle dashboard wakes ECHO. Set to false if a flaky
# touchscreen produces ghost taps.
WAKE_ON_TOUCH = os.environ.get("WAKE_ON_TOUCH", "true").lower() in (
    "1", "true", "yes", "on")
# Drop back to IDLE this long after the last activity. ECHO speaking and
# a face in front of the camera both count as activity, so this countdown
# only starts once the greeting has finished AND the person has stepped
# away (or there's no chat / registration / mic in progress).
IDLE_AFTER_LAST_INTERACTION_SEC = 60
# A face seen within this many seconds counts as "someone is still here"
# and keeps the screen ACTIVE so they can start interacting.
FACE_PRESENT_KEEPALIVE_SEC = 3
# If the kiosk wakes (voice or touch) but the camera detects NO face at
# all for this many seconds, it closes silently back to the dashboard.
# Handles false wakes -- Vosk mis-hearing ambient chatter as a wake
# phrase -- so the kiosk doesn't sit ACTIVE talking to an empty room.
NO_FACE_AUTO_IDLE_SEC = 5
# While there's chat activity, keep the kiosk in ACTIVE mode for at
# least this long after the last chat message. Gives the user time to
# read the answer and ask a follow-up without the kiosk dropping out.
CHAT_KEEPALIVE_SEC = 180
# After a chat conversation goes quiet for this long, the kiosk speaks a
# friendly sign-off (by name) and returns to the dashboard, ready for the
# next person to start fresh.
CHAT_IDLE_GOODBYE_SEC = 60

# Phrases that end the chat session without calling the LLM. Matched
# against the bottom of the transcribed user utterance (case-insensitive,
# trailing punctuation stripped). On match: kiosk speaks a random line
# from messages.FAREWELL_RESPONSES and drops back to the dashboard.
# Tune for local idioms ("tata", "ciao", "see ya") if needed.
CHAT_GOODBYE_TOKENS = (
    "bye",
    "goodbye",
    "good bye",
    "bye bye",
    "thank you",            # gratitude = done (exact / end-of-sentence)
    "thanks",
    "no thank you",
    "no thanks",
    "thanks bye",
    "thank you bye",
    "bye thank you",       # bye-first phrasings (startswith match)
    "bye thanks",
    "thanks goodbye",
    "goodbye thank you",
    "i'm good",
    "im good",
    "i'm all good",
    "that's it",
    "thats it",
    "nothing else",
    "no more questions",
    "catch you later",
    "have a good day",
    "have a good one",
    "okay bye",
    "ok bye",
    "alright bye",
    "see you",
    "see you later",
    "see ya",
    "talk to you later",
    "that's all",
    "thats all",
    "i'm done",
    "im done",
    "we're done",
    "were done",
    "all done",
    "end conversation",
    "end chat",
    "stop chat",
    "exit",
)
# After the user taps "No thanks" on the auto-register prompt, don't
# re-open it for this many seconds even if their face is still unknown.
REGISTER_DECLINE_COOLDOWN_SEC = 120
# Backwards-compat alias used by older code paths.
SLEEP_AFTER_NO_LIVE_FACE_SEC = IDLE_AFTER_LAST_INTERACTION_SEC
# Hard back-stop on any single ACTIVE session.
ACTIVE_SESSION_MAX_SEC = 600

# ---- Resilience / watchdog ------------------------------------------------
# The camera worker stamps a heartbeat each loop iteration. A watchdog
# thread exits the process (so the systemd unit's Restart=always relaunches
# it with clean camera + NPU state) if the worker thread dies or its
# heartbeat goes stale. Clean process restart beats trying to re-acquire
# leaked hardware handles in-process.
WATCHDOG_POLL_SEC = 5
# Generous so a long registration (which blocks the main loop while it
# captures poses) never trips the watchdog. A real hang is far longer.
WORKER_HEARTBEAT_STALL_SEC = 90
# How many times to retry opening the camera + Hailo pipeline at startup
# before giving up and exiting for a systemd restart. Rides out transient
# boot races (camera not enumerated yet, NPU busy from a prior run).
WORKER_INIT_RETRIES = 5

# ---- Privacy / biometric data retention -----------------------------------
# Auto-delete stored face data this many hours after registration. 0 (the
# default) disables auto-purge -- wipe manually via the admin "Wipe all
# faces" button after the event. Set e.g. 24 to clear a one-day booth's
# data automatically the next day. Interaction *counts* are always kept;
# only the biometric embeddings + names are removed.
DATA_RETENTION_HOURS = int(os.environ.get("DATA_RETENTION_HOURS", "0"))
# How often the retention sweeper runs (minutes). Ignored when retention
# is disabled.
DATA_PURGE_SWEEP_MIN = 30

# ---- Idle / Active screen layout -----------------------------------------
# Welcome banner colours, BGR triplets.
THEME = {
    "bg":     (180, 110, 30),   # blue background
    "fg":     (0, 0, 0),         # black text
    "accent": (0, 0, 0),
}
# Window size for the cv2 output. The camera frame is centred / scaled
# into the left half during ACTIVE; the right half is the panel. Pick a
# size that fits your display.
WINDOW_SIZE = (1280, 720)        # (width, height)

# Right-panel width in ACTIVE state. Camera left half = WINDOW_SIZE[0] - PANEL_WIDTH.
PANEL_WIDTH = 540

# ---- Weather widget -------------------------------------------------------
KIOSK_HOST = "127.0.0.1"
KIOSK_PORT = 8090   # 8080 is commonly taken by Open WebUI / similar.
FUN_FACT_ROTATE_SEC = 15
MJPEG_QUALITY = 80                # JPEG quality for the camera stream
MJPEG_MAX_FPS = 20                # display only; detection runs at full rate
# Re-use the last face embedding / match for this long while the same face
# (by box overlap) stays in frame, instead of running ArcFace on every
# frame. Detection + liveness still run every frame. 0 = embed every frame.
RECOGNITION_REUSE_SEC = float(os.environ.get("RECOGNITION_REUSE_SEC", "0.5"))
# How often the idle dashboard (projects / sessions / metrics) is re-read
# from SQLite while IDLE, and while ACTIVE (where it isn't visible).
IDLE_REFRESH_SEC = 5
ACTIVE_REFRESH_SEC = 60

# ---- External display API ---------------------------------------------------
# When a recognised person starts talking to ECHO we POST
#   {"state": "on", "employee_id": ..., "full_name": ...}
# and when the session ends (goodbye, idle timeout, or X) we POST
#   {"state": "off"}
# to DISPLAY_API_URL with header X-API-Key. Unset URL = feature off.
DISPLAY_API_URL = os.environ.get("DISPLAY_API_URL", "").strip()
DISPLAY_API_KEY = os.environ.get("DISPLAY_API_KEY", "").strip()
DISPLAY_API_TIMEOUT_SEC = float(os.environ.get("DISPLAY_API_TIMEOUT_SEC", "3"))

# ---- Hologram app: employee details + face photo -------------------------
# Each employee is POSTed (multipart: employee_id, full_name, image) to
# HOLOGRAM_SYNC_URL. See hologram_sync.py. Unset URL = feature off.
HOLOGRAM_SYNC_URL = os.environ.get("HOLOGRAM_SYNC_URL", "").strip()
# Defaults to the display API key when not set separately.
HOLOGRAM_API_KEY = (os.environ.get("HOLOGRAM_API_KEY", "").strip()
                    or DISPLAY_API_KEY)
HOLOGRAM_TIMEOUT_SEC = float(os.environ.get("HOLOGRAM_TIMEOUT_SEC", "10"))
# While the kiosk runs, retry not-yet-sent people this often. 0 = only the
# start-up utility retries.
HOLOGRAM_RETRY_SEC = float(os.environ.get("HOLOGRAM_RETRY_SEC", "600"))
# Send people who have no photo yet (details only). Default off: they are
# sent once a photo exists.
HOLOGRAM_SEND_WITHOUT_PHOTO = os.environ.get(
    "HOLOGRAM_SEND_WITHOUT_PHOTO", "false").lower() in ("1", "true", "yes", "on")
# People registered before photos were stored have none. When true, the
# kiosk saves a photo the next time it recognises them, then sends it.
HOLOGRAM_BACKFILL_ON_RECOGNITION = os.environ.get(
    "HOLOGRAM_BACKFILL_ON_RECOGNITION", "true").lower() in (
    "1", "true", "yes", "on")
# Face crop: extra margin around the face box (fraction of box size) and
# the longest side of the saved JPEG.
HOLOGRAM_PHOTO_MARGIN = 0.6
HOLOGRAM_PHOTO_MAX_SIDE = 480

WEATHER_REFRESH_SEC = 1800        # 30 minutes
# Pinned to Calgary. IP geolocation was resolving the Pi's public IP to
# Edmonton (ISP routing), so the kiosk reported the wrong city. With
# lat/lon set, weather.py skips geolocation entirely and always reports
# Calgary -- both on the idle widget and in chat ("It's N°C ... in Calgary").
WEATHER_LATITUDE = 51.0447        # Calgary, AB
WEATHER_LONGITUDE = -114.0719
WEATHER_FALLBACK_CITY = "Calgary"  # label shown when lat/lon are pinned

# ---- OpenAI / Ollama chat -------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_URL = "http://localhost:8000"  # Hailo-Ollama on this Pi binds to :8000
                                      # (`ss -tlnp` shows hailo-ollama owning :8000).
                                      # :8080 is Open WebUI; upstream Ollama uses 11434.
OLLAMA_MODEL = "qwen3:1.7b"       # qwen3 instruct on Hailo-Ollama; bigger
                                  # alternatives that fit on Pi 5: "qwen2.5:1.5b"
                                  # avoid "deepseek_r1:*" (emits <think> blocks)
                                  # and "qwen2.5-coder:*" (code-tuned, weak chat)
# Backend probe order. First available wins. Set to ("ollama", "openai")
# if you want to prefer the local model.
CHAT_BACKEND_ORDER = ("openai", "ollama")

# Web search: route OpenAI chat through the Responses API with the
# web_search tool so answers can include current / real-time info
# instead of only the model's training-cutoff knowledge. Adds ~3-6 s
# latency per question. Falls back to a plain chat.completions call if
# the Responses API or the tool isn't available on the installed SDK.
# Only affects the OpenAI backend; Ollama (offline) ignores it.
CHAT_WEB_SEARCH = os.environ.get("CHAT_WEB_SEARCH", "true").lower() in (
    "1", "true", "yes",
)

# How many prior conversation turns (user+assistant messages) to send
# with each question so follow-ups have context. 0 = stateless one-shot.
# Each turn adds tokens; 6 (= 3 exchanges) is a good balance for a kiosk.
CHAT_HISTORY_TURNS = 6

# Max tokens in a chat reply. Kiosk answers should be short + spoken.
# ~150 tokens ~= 3-4 short sentences -- a hard ceiling so web-search
# answers can't run long even if the model wants to ramble.
CHAT_MAX_REPLY_TOKENS = 150
# Per-session question cap. Set high (effectively "no limit" for a normal
# kiosk visit) -- the budget still exists as a backstop against runaway
# sessions, but the UI no longer shows a countdown badge (see echo.js).
CHAT_MAX_QUESTIONS_PER_SESSION = 25
CHAT_VOICE_MODE_DEFAULT = "voice"   # "voice" or "keyboard"

# ---- AI Lab sessions (answered in chat) -----------------------------------
# When a visitor asks about AI Lab sessions, ECHO answers from these lists
# locally (no LLM). Edit to keep current. Completed sessions are names only;
# planned sessions carry a human-readable date string. This is separate from
# the `sessions` DB table (which drives the dashboard's dated "Upcoming
# Session" card) -- this list also covers already-completed sessions.
AI_LAB_SESSIONS_COMPLETED = [
    "Basics of AI Application",
    "Prompt Engineering",
    "Build a RAG-based Chatbot",
    "MCP Servers",
]
AI_LAB_SESSIONS_PLANNED = [
    ("Claude Certified Associate - Foundation", "September 12th at 2:54 PM"),
]

# Headline count quoted for "how many tools/solutions?" questions and the
# listing intro. The solutions DB catalog (set_solutions.py) holds the
# detailed entries used to answer specific "do you have a tool for X?"
# questions; this is just the public number we advertise.
SOLUTIONS_COUNT_CLAIM = "50+"

# ---- Chat-voice ASR -------------------------------------------------------
# Wake-word listener stays on the small Vosk model. Long-form chat
# dictation goes through this separate backend, lazy-loaded so the
# kiosk doesn't pay the memory bill until the user actually opens
# chat voice mode.
#
#   "auto"           -> precedence: hailo-whisper -> openai -> faster-whisper.
#                       Hailo wins if the HEFs are on disk; OpenAI wins
#                       if OPENAI_API_KEY is set; otherwise local CPU.
#                       Default.
#   "hailo"          -> Hailo NPU Whisper-Base via hailo-apps. ~200-700 ms
#                       per utterance on Hailo-10H, $0, fully local.
#                       Needs:
#                         pip install hailo-apps
#                         + encoder/decoder HEFs in models/ (see docs/
#                           hailo-whisper.md for download instructions)
#   "faster-whisper" -> CTranslate2 + tiny.en, ~250 MB resident, ~3-4 s
#                       per 10 s of speech on Pi 5 CPU.
#   "openai"         -> openai.audio.transcriptions.create (cloud).
#                       ~1-2 s round-trip, paid per minute.
#   "vosk"           -> legacy: small Vosk model in a no-grammar recognizer.
#                       Lower quality, kept as a fully-offline fallback.
CHAT_ASR_BACKEND = "auto"

# Hailo Whisper config. The "auto" path looks here first.
#
# Variant: "tiny" / "base" / "tiny.en" -- all three downloaded by the
# hailo-apps CLI's --variant flag.
#   python -m hailo_apps.python.standalone_apps.speech_recognition...
#                                  --arch hailo10h --variant <name>
# Latency on Hailo-10H:
#   tiny      ~150-300 ms   basic accuracy
#   base      ~250-500 ms   recommended balance (default)
#   tiny.en   ~150-300 ms   best English accuracy among CLI-supported
#                           variants. Hailo-10H only.
# Whisper-Small has a Model Explorer page on hailo.ai but isn't in
# the CLI's argparse choices yet. If you obtain the HEFs separately,
# set HAILO_WHISPER_MODEL="small" and point HAILO_WHISPER_*_HEF /
# HAILO_WHISPER_NPY_DIR at the files. See docs/hailo-whisper.md.
HAILO_WHISPER_MODEL = os.environ.get("HAILO_WHISPER_MODEL", "base")

# Combined HEF path for HailoRT 5.2+'s native Speech2Text API. When
# set + the file exists, the auto-selector picks the new native path
# instead of hailo-apps's whisper_pipeline.py. One file = encoder +
# decoder packed together. See docs/hailo-whisper.md.
HAILO_WHISPER_HEF = os.environ.get("HAILO_WHISPER_HEF") or None

# File resolution. Each setting picks the first option that exists,
# unless the corresponding env var is set (in which case the env var
# wins outright -- the user pinned a specific path on purpose).
#   1. HAILO_WHISPER_{ENCODER,DECODER}_HEF / _NPY_DIR env var
#   2. pi5/models/whisper-{variant}-{role}.hef        (symlink convention)
#   3. /usr/local/hailo/resources/models/hailo10h/... (hailo-apps install)
def _first_existing(*paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    # Nothing exists yet -- return the local path so the error message
    # points at the project's preferred install spot.
    return Path(paths[0])

def _env_path_or_search(env_var: str, *fallback_paths) -> Path:
    """If env_var is set, use it verbatim (user pinned a path on
    purpose). Otherwise fall back to the first-existing search."""
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)
    return _first_existing(*fallback_paths)

# Note the embedded "-10s" / "-10s-out-seq-64" suffixes -- Hailo's
# downloader bakes them into the filenames. Tiny.en is Hailo-10H
# only and uses the same naming with "tiny.en" prefix.
HAILO_WHISPER_ENCODER_HEF = _env_path_or_search(
    "HAILO_WHISPER_ENCODER_HEF",
    MODELS_DIR / f"whisper-{HAILO_WHISPER_MODEL}-encoder.hef",
    f"/usr/local/hailo/resources/models/hailo10h/"
    f"{HAILO_WHISPER_MODEL}-whisper-encoder-10s.hef",
)
HAILO_WHISPER_DECODER_HEF = _env_path_or_search(
    "HAILO_WHISPER_DECODER_HEF",
    MODELS_DIR / f"whisper-{HAILO_WHISPER_MODEL}-decoder.hef",
    f"/usr/local/hailo/resources/models/hailo10h/"
    f"{HAILO_WHISPER_MODEL}-whisper-decoder-10s-out-seq-64.hef",
)
# The .npy assets live in a shared directory; Hailo's pipeline picks
# the right ones by filename suffix (e.g. token_embedding_weight_base.npy
# vs ..._tiny.npy).
HAILO_WHISPER_NPY_DIR = _env_path_or_search(
    "HAILO_WHISPER_NPY_DIR",
    MODELS_DIR / f"whisper-{HAILO_WHISPER_MODEL}-assets",
    "/usr/local/hailo/resources/npy",
)
# add_embed: hailo-apps sets True for Hailo-8 / Hailo-8L (embedding
# matmul runs on the host) and False for Hailo-10H (embedding runs on
# the chip). False is the default for the Pi 5 + AI HAT 2+.
HAILO_WHISPER_ADD_EMBED = (
    os.environ.get("HAILO_WHISPER_ADD_EMBED", "false").lower()
    in ("1", "true", "yes")
)
WHISPER_MODEL = "tiny.en"           # "tiny.en" | "base.en" | "small.en"
WHISPER_DEVICE = "cpu"              # Pi 5 has no GPU; keep "cpu".
WHISPER_COMPUTE_TYPE = "int8"       # 8-bit quantisation for memory.
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
# Voice capture timing knobs.
CHAT_VOICE_MAX_SEC = 12.0           # hard cap on a single utterance.
# How long to wait for the user to START talking after they tap the mic.
# If they don't speak within this window, the kiosk closes the mic and
# nudges them ("please speak when you're ready"), then the normal
# chat-idle goodbye takes over.
CHAT_VOICE_NO_SPEECH_SEC = 30.0
CHAT_VOICE_SILENCE_SEC = 1.5        # auto-finalise after this much trailing
                                    # silence. Generous so a natural pause
                                    # mid-question doesn't truncate it. (Users
                                    # can also tap to stop for full control.)
CHAT_VOICE_SILENCE_RMS = 600        # int16 RMS threshold below which audio
                                    # counts as silence. Raised from 350 --
                                    # the PowerConf picks up enough ambient
                                    # room noise to sit above 350, so the VAD
                                    # never saw "silence" and recordings ran
                                    # 8+ s. 600 treats normal room tone as
                                    # silence so the cutoff fires ~0.6 s after
                                    # the user stops talking. Lower it if the
                                    # VAD starts cutting people off mid-word.
CHAT_VOICE_DEBUG_RMS = os.environ.get("CHAT_VOICE_DEBUG_RMS", "0") == "1"

# ---- Silent learning ------------------------------------------------------
# When a confidently recognised face passes the quality + liveness gates,
# append the new embedding to that person's gallery so recognition gets
# more robust over time. Rate-limited and capped so the DB doesn't grow
# unbounded.
SILENT_LEARN_ENABLED = True
SILENT_LEARN_MIN_SCORE = 0.75          # only learn from very confident matches (was 0.70)
SILENT_LEARN_MIN_MARGIN = 0.20         # best score must beat runner-up by this much (was 0.15)
SILENT_LEARN_MAX_SIMILARITY = 0.92     # skip if new sample is ~ a duplicate of an existing one
SILENT_LEARN_MIN_INTERVAL_SEC = 60     # at most one new sample per person per minute
SILENT_LEARN_MAX_SAMPLES_PER_PERSON = 10  # cap; oldest non-enrolment samples drop first (was 30 -- smaller cap = less room for drift to compound)

# ---- Liveness (passive anti-spoofing) -------------------------------------
# Master switch. When False, every face is treated as live -- no motion /
# jitter / texture / specular checks run. Useful when the kiosk lives in
# a controlled space (e.g. a private office) and the anti-spoofing is
# more friction than protection.
LIVENESS_ENABLED = False
# Sliding window length over which we compute liveness signals.
LIVENESS_WINDOW_FRAMES = 24
# Min per-frame face crop size used by the pixel-jitter check.
LIVENESS_CROP_PX = 96

# Single-frame screen-attack defences (run BEFORE temporal checks; if any
# fails, the temporal window is cleared so glare/flat content can never
# accumulate enough frames to be considered live).
#
# Specular highlight ratio: fraction of pixels in the face crop whose HSV
# V channel is > 240. Real indoor faces produce 0-5% (glasses, forehead
# sheen, glints in eyes). Phone/monitor screens with glare typically
# light up 10-40% of the visible "face". Lower this if a screen still
# passes; raise if real faces are flagged.
LIVENESS_MAX_SPECULAR_RATIO = 0.10

# Texture variance: variance of the Laplacian of the face crop. Real
# skin has fine pores/wrinkles -> high variance. Phones/monitors smooth
# the face out -> low variance. Bump this up if a screen still passes.
LIVENESS_MIN_TEXTURE_VAR = 60.0

# Temporal signals (computed over the sliding window). Both must clear:
# 1. relative landmark motion (subtracting whole-face translation).
LIVENESS_REL_MOTION_MIN = 0.45    # pixels (std), in original-image scale
# 2. face-region pixel jitter beyond camera read noise.
LIVENESS_PIXEL_JITTER_MIN = 4.0   # mean abs frame-to-frame diff in [0..255]
