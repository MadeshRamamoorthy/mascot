"""Notify the external display service when a session starts / ends.

    POST {DISPLAY_API_URL}
    X-API-Key: {DISPLAY_API_KEY}
    Content-Type: application/json

    {"state": "on", "employee_id": "E0001", "full_name": "Madesh Ramamoorthy"}
    {"state": "off"}

Calls run on a single background thread fed by a queue, so the camera
loop never blocks on the network. Order is preserved (an "off" can't
overtake the "on" before it). When DISPLAY_API_URL is unset the
notifier is a silent no-op, so the kiosk runs unchanged without it.

The notifier also tracks whether the display is currently "on" for
this kiosk: `off()` is skipped when no "on" was sent, so an idle
timeout with nobody recognised doesn't spam the endpoint.
"""

from __future__ import annotations

import queue
import threading

import config

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None


class DisplayNotifier:
    def __init__(self, url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.url = (url if url is not None else config.DISPLAY_API_URL) or ""
        self.api_key = (api_key if api_key is not None
                        else config.DISPLAY_API_KEY) or ""
        self.timeout = timeout or config.DISPLAY_API_TIMEOUT_SEC
        self.enabled = bool(self.url) and requests is not None
        self._is_on = False
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._session = requests.Session() if self.enabled else None
        if self.enabled:
            threading.Thread(target=self._run, daemon=True,
                             name="display-api").start()
            print(f"[display-api] enabled -> {self.url}", flush=True)
        elif self.url and requests is None:
            print("[display-api] disabled: 'requests' not installed",
                  flush=True)

    # ---- public API (non-blocking) ------------------------------------

    def on(self, employee_id: str, full_name: str) -> None:
        with self._lock:
            self._is_on = True
        self._enqueue({"state": "on", "employee_id": employee_id,
                       "full_name": full_name})

    def off(self) -> None:
        with self._lock:
            if not self._is_on:
                return
            self._is_on = False
        self._enqueue({"state": "off"})

    # ---- internals -----------------------------------------------------

    def _enqueue(self, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            print(f"[display-api] queue full, dropped {payload}", flush=True)

    def _run(self) -> None:
        while True:
            payload = self._q.get()
            self._post(payload)

    def _post(self, payload: dict) -> bool:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        try:
            r = self._session.post(self.url, json=payload, headers=headers,
                                   timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[display-api] POST {payload.get('state')} failed: "
                  f"{exc!r}", flush=True)
            return False
        ok = 200 <= r.status_code < 300
        print(f"[display-api] POST {payload.get('state')} -> "
              f"{r.status_code}{'' if ok else ' ' + r.text[:120]}",
              flush=True)
        return ok
