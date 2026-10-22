"""Stand-in for the external display + hologram services, for local testing.

Accepts the same calls ECHO makes and prints each one, so you can watch
the on / off flow without the real display.

    python mock_display_api.py                      # :8765, key "test-key"
    python mock_display_api.py --port 9000 --key abc
    python mock_display_api.py --host 0.0.0.0       # reachable from the Pi

Then point ECHO at it in .env:

    DISPLAY_API_URL=http://<this-machine-ip>:8765/api/display
    DISPLAY_API_KEY=test-key

GET /api/display returns the current state and the call log, handy for
scripted checks.

It also accepts the hologram employee upload (see hologram_sync.py):

    HOLOGRAM_SYNC_URL=http://<this-machine-ip>:8765/api/employees

Received photos are saved to ./mock_hologram_photos/<employee_id>.jpg so
you can check them by eye. GET /api/employees lists what arrived.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from flask import Flask, jsonify, request


def create_app(api_key: str) -> Flask:
    app = Flask(__name__)
    current = {"state": "off"}
    log: list[dict] = []

    @app.post("/api/display")
    def display():
        if api_key and request.headers.get("X-API-Key") != api_key:
            print("[mock-display] 401 bad or missing X-API-Key", flush=True)
            return jsonify(error="invalid api key"), 401
        body = request.get_json(silent=True) or {}
        state = body.get("state")
        if state not in ("on", "off"):
            return jsonify(error="state must be 'on' or 'off'"), 400
        if state == "on" and not (body.get("employee_id")
                                  and body.get("full_name")):
            return jsonify(error="employee_id and full_name required"), 400
        current.clear()
        current.update(body)
        log.append({"at": time.strftime("%H:%M:%S"), **body})
        who = (f" {body['employee_id']} {body['full_name']}"
               if state == "on" else "")
        print(f"[mock-display] {state.upper()}{who}", flush=True)
        return jsonify(ok=True, **current)

    @app.get("/api/display")
    def status():
        return jsonify(current=current, log=log)

    employees: dict[str, dict] = {}
    photo_dir = Path("mock_hologram_photos")

    @app.post("/api/employees")
    def employee_upload():
        if api_key and request.headers.get("X-API-Key") != api_key:
            print("[mock-hologram] 401 bad or missing X-API-Key", flush=True)
            return jsonify(error="invalid api key"), 401
        emp_id = (request.form.get("employee_id") or "").strip()
        name = (request.form.get("full_name") or "").strip()
        if not emp_id or not name:
            return jsonify(error="employee_id and full_name required"), 400
        img = request.files.get("image")
        size = 0
        if img is not None:
            data = img.read()
            size = len(data)
            photo_dir.mkdir(exist_ok=True)
            (photo_dir / f"{emp_id}.jpg").write_bytes(data)
        employees[emp_id] = {"full_name": name, "image_bytes": size,
                             "at": time.strftime("%H:%M:%S")}
        print(f"[mock-hologram] got {emp_id} {name} "
              f"({size} byte photo)" if size else
              f"[mock-hologram] got {emp_id} {name} (no photo)", flush=True)
        return jsonify(ok=True)

    @app.get("/api/employees")
    def employee_list():
        return jsonify(employees)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--key", default="test-key",
                   help="expected X-API-Key ('' disables the check)")
    a = p.parse_args()
    print(f"[mock] listening on http://{a.host}:{a.port}  "
          f"display: /api/display  hologram: /api/employees  "
          f"(key={a.key!r})", flush=True)
    create_app(a.key).run(host=a.host, port=a.port)


if __name__ == "__main__":
    main()
