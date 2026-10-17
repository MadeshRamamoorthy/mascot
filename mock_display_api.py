"""Stand-in for the external display service, for local testing.

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
"""

from __future__ import annotations

import argparse
import time

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

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--key", default="test-key",
                   help="expected X-API-Key ('' disables the check)")
    a = p.parse_args()
    print(f"[mock-display] listening on http://{a.host}:{a.port}/api/display "
          f"(key={a.key!r})", flush=True)
    create_app(a.key).run(host=a.host, port=a.port)


if __name__ == "__main__":
    main()
