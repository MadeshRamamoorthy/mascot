"""One-shot registration for Abhinav Jain (TPD head Americas).

Usage on the Pi (from the project root, venv active, kiosk stopped):

    ./stop.sh
    python register_abhinav.py path/to/abhinav.jpg [path/to/abhinav2.jpg ...]
    ./start.sh

Two photos give a much more robust match than one -- if you have both
the headshot and a photo taken with the kiosk's own camera in the venue
lighting, pass them both. Registers a new employee with the emp_id /
name below, then pins CUSTOM_WELCOME as the exact spoken greeting.
Safe to re-run: appends embeddings if the emp_id already exists.

Edit EMP_ID / NAME / CUSTOM_WELCOME to change any of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

from database import FaceDB
from hailo_infer import HailoFacePipeline
from photo_register import register_from_photos
import config


EMP_ID = "abhinav"
NAME = "Abhinav Jain"
CUSTOM_WELCOME = (
    "Namaste Abhinav! It's a true honor to welcome you to ECHO SCOPE -- "
    "our AI Lab here at Infosys Calgary. As TPD head for the Americas, "
    "your leadership inspires the work we do every day. Please explore "
    "the demos, meet the team, and see how our ideas turn into real AI "
    "solutions. Welcome to the Lab!"
)


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: python register_abhinav.py <photo1> [<photo2> ...]",
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
        result = register_from_photos(db, pipe, EMP_ID, NAME, blobs)
        print(f"register_from_photos: {result}")
        if not result.get("ok"):
            return 1
        db.set_employee_profile(EMP_ID, custom_welcome=CUSTOM_WELCOME)
        print(f"custom_welcome set for {EMP_ID}.")
        profile = db.get_employee_profile(EMP_ID)
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
