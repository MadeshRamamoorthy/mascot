"""MJPEG ring buffer for the camera stream.

The camera worker pushes BGR frames via push(); HTTP clients pull them
out via generator(). Only the *latest* frame is kept -- if a client is
slow we drop frames rather than buffer.

Each subscriber gets its own threading.Event to wake on; that means
multiple browser tabs / windows can stream concurrently from the same
ring without clobbering each other.
"""

from __future__ import annotations

import threading
import time
from typing import Generator

import cv2

import config


_BOUNDARY = b"--echoframe"


class FrameStreamer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._subscribers: list[threading.Event] = []
        self._last_push = 0.0
        self._min_interval = 1.0 / max(1, config.MJPEG_MAX_FPS)

    def push(self, frame_bgr) -> None:
        """Encode frame to JPEG and wake subscribers. Throttled to
        MJPEG_MAX_FPS to keep CPU light when nobody's watching."""
        # Nobody is watching (the dashboard drops the stream while IDLE):
        # skip the JPEG encode entirely, it's the costliest step here.
        if not self._subscribers:
            return
        now = time.time()
        if now - self._last_push < self._min_interval:
            return
        self._last_push = now
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(config.MJPEG_QUALITY)],
        )
        if not ok:
            return
        with self._lock:
            self._jpeg = bytes(buf)
            self._frame_id += 1
            subs = list(self._subscribers)
        for ev in subs:
            ev.set()

    def generator(self) -> Generator[bytes, None, None]:
        """Yield multipart/x-mixed-replace chunks for Flask's
        Response(..., mimetype='multipart/x-mixed-replace; boundary=echoframe')."""
        ev = threading.Event()
        with self._lock:
            self._subscribers.append(ev)
            last_id = -1
        try:
            while True:
                ev.wait(timeout=10.0)
                ev.clear()
                with self._lock:
                    jpeg = self._jpeg
                    frame_id = self._frame_id
                if jpeg is None or frame_id == last_id:
                    continue
                last_id = frame_id
                yield (b"\r\n" + _BOUNDARY + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
        finally:
            with self._lock:
                try:
                    self._subscribers.remove(ev)
                except ValueError:
                    pass
