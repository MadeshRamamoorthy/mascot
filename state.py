"""Thread-safe state bus shared between the camera worker and the
Flask app.

The camera worker pushes state changes via setter methods (or
`update(...)`); Flask's SSE endpoint registers as a subscriber via
`subscribe()` and yields JSON-serialised diffs as they arrive.

There's no per-field locking complexity here -- a single RLock guards
everything. The state is small and updates are infrequent (a few per
second at most), so contention is a non-issue.
"""

from __future__ import annotations

import copy
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Generator


@dataclass
class KioskState:
    # IDLE | ACTIVE
    state: str = "IDLE"
    # epoch seconds when the current state started
    state_since: float = field(default_factory=time.time)

    # Branding (read by the SPA template).
    brand: str = "ECHO SCOPE"
    wake_phrase: str = "hello echo scope"
    contact_email: str = "Calgary_AIClub@infosys.com"

    # Live clock (server-side; SPA can compute its own too).
    now_epoch: float = field(default_factory=time.time)

    # Weather.
    weather: dict = field(default_factory=lambda: {
        "ok": False, "temp_c": None, "label": None, "city": None,
        "humidity": None, "icon": "❓",
    })

    # Hi-5 stats (interaction counter aggregates).
    metrics: dict = field(default_factory=lambda: {
        "today": 0, "week": 0, "best_day": None, "best_count": 0,
        "total": 0,
    })

    # Projects shown on idle (Calgary AI files on display today).
    projects: list = field(default_factory=list)

    # Upcoming session card.
    next_session: dict | None = None

    # Currently recognised person, if any.
    person: dict | None = None      # {"emp_id": ..., "name": ...}

    # Active-screen content.
    fun_fact: dict = field(default_factory=lambda: {
        "type": "AI FUN FACT", "icon": "💡",
        "content": "Welcome to ECHO SCOPE."
    })

    # Chat-voice state. listening = mic is recording. transcribing =
    # audio captured, waiting for Whisper / OpenAI to return text.
    listening: bool = False
    transcribing: bool = False
    # Whisper / OpenAI label, e.g. "OpenAI · gpt-4o-mini"
    chat_backend: str = ""
    chat_history: list = field(default_factory=list)
    chat_remaining: int = 5
    chat_pending: bool = False

    # Registration overlay.
    register_open: bool = False
    register_step: str = "form"      # "form" | "capturing" | "done"
    register_message: str = ""
    register_pose: dict | None = None  # {"idx": 1, "total": 5, "prompt": "...", "status": "..."}

    # Last toast message (e.g. greeting just spoken).
    toast: dict | None = None        # {"text": "...", "since": epoch}


class StateBus:
    """Single source of truth. Mutate via setters; read via snapshot()."""

    def __init__(self):
        self._lock = threading.RLock()
        self._state = KioskState()
        self._subscribers: list[queue.Queue] = []
        self._subscribers_lock = threading.Lock()

    # ---- read ----------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return _to_json(self._state)

    def get(self, *keys: str) -> dict:
        """Cheap read of a few fields, for hot paths like the camera loop.
        snapshot() runs dataclasses.asdict over the whole state (a full
        recursive deep copy, chat history and project list included),
        which is far too expensive to call several times per frame.
        Values are returned as-is; callers must not mutate them."""
        with self._lock:
            return {k: getattr(self._state, k, None) for k in keys}

    # ---- write ---------------------------------------------------------

    def update(self, **kwargs) -> None:
        """Patch one or more fields. Any keys that change broadcast a
        diff. Unknown keys are silently ignored to keep callers
        forward-compatible."""
        diff: dict[str, Any] = {}
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self._state, k):
                    continue
                cur = getattr(self._state, k)
                if cur != v:
                    setattr(self._state, k, v)
                    diff[k] = _to_json_value(v)
        if diff:
            self._broadcast(diff)

    def patch(self, field_name: str, **mutations) -> None:
        """Apply a dict patch to a dict-valued field and broadcast the
        merged result."""
        with self._lock:
            cur = getattr(self._state, field_name, None)
            if not isinstance(cur, dict):
                return
            merged = {**cur, **mutations}
            setattr(self._state, field_name, merged)
        self._broadcast({field_name: copy.deepcopy(merged)})

    def set_now(self) -> None:
        self.update(now_epoch=time.time())

    def go_active(self) -> None:
        self.update(state="ACTIVE", state_since=time.time())

    def go_idle(self) -> None:
        self.update(
            state="IDLE",
            state_since=time.time(),
            listening=False,
            transcribing=False,
            register_open=False,
            chat_pending=False,
            chat_history=[],
            person=None,
            toast=None,
        )

    # ---- pub/sub -------------------------------------------------------

    def subscribe(self) -> "Subscription":
        q: queue.Queue = queue.Queue(maxsize=128)
        with self._subscribers_lock:
            self._subscribers.append(q)
        # Seed the new subscriber with the current full snapshot.
        q.put(self.snapshot())
        return Subscription(self, q)

    def _unsubscribe(self, q: queue.Queue) -> None:
        with self._subscribers_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, diff: dict) -> None:
        with self._subscribers_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(diff)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass


class Subscription:
    def __init__(self, bus: StateBus, q: queue.Queue):
        self._bus = bus
        self._q = q
        self._closed = False

    def stream(self) -> Generator[dict, None, None]:
        try:
            while not self._closed:
                try:
                    yield self._q.get(timeout=15.0)
                except queue.Empty:
                    # Heartbeat so clients on flaky networks don't drop us.
                    yield {"_heartbeat": time.time()}
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self._q)


# --------------- JSON helpers ----------------------------------------------


def _to_json_value(v: Any) -> Any:
    if hasattr(v, "__dataclass_fields__"):
        return asdict(v)
    if isinstance(v, (list, tuple)):
        return [_to_json_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json_value(x) for k, x in v.items()}
    return v


def _to_json(state: KioskState) -> dict:
    return {k: _to_json_value(v) for k, v in asdict(state).items()}


def encode_sse(data: dict) -> bytes:
    """Format a dict as a Server-Sent Events frame."""
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")
