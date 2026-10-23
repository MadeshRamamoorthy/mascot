"""One-shot registration for Jamie (client visit).

Usage on the Pi (from the project root, venv active):

    python register_jamie.py path/to/jamie.jpg
    # or with multiple angles:
    python register_jamie.py jamie_front.jpg jamie_left.jpg

Registers a new employee with name=NAME (the E0001-style ID is generated) below, using the
photo(s) you point at, then pins CUSTOM_WELCOME as the exact greeting the
kiosk speaks whenever Jamie is recognised (overrides any LLM-generated
welcome). Safe to re-run: appends embeddings if that name is already registered.

Edit NAME / CUSTOM_WELCOME below to change any of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

from database import FaceDB
from hailo_infer import HailoFacePipeline
from photo_register import register_from_photos
import config


NAME = "Jamie"
CUSTOM_WELCOME = (
    "Hi Jamie, welcome to Echo AI Lab! Great to have you with us today. "
    "We're proud of our Nutrien-Infosys partnership -- together, feeding "
    "the future. This is where ideas turn into real AI solutions. Let's "
    "build the future of AI together."
)


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: python register_jamie.py <photo1> [<photo2> ...]",
              file=sys.stderr)
        return 2
    blobs = []
    for p in paths:
        f = Path(p)
        if not f.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 1
        blobs.append(f.read_bytes())

    db = FaceDB()
    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    try:
        # Re-runs add to the existing person; otherwise a new E-number.
        result = register_from_photos(
            db, pipe, db.find_emp_id_by_name(NAME), NAME, blobs)
        print(f"register_from_photos: {result}")
        if not result.get("ok"):
            return 1
        emp_id = result["emp_id"]
        db.set_employee_profile(emp_id, custom_welcome=CUSTOM_WELCOME)
        print(f"custom_welcome set for {emp_id}.")
        # Quick read-back so you can eyeball the row.
        profile = db.get_employee_profile(emp_id)
        print(f"profile: {profile}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
