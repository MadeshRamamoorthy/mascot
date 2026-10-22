"""Send employees (details + face photo) to the hologram application.

Two ways in, one send function:

0. Bulk-upload ZIP for the hologram app (no network needed):

       python hologram_sync.py --export-zip hologram_upload.zip
       python hologram_sync.py --export-zip out.zip --include-no-photo

   The ZIP holds employees.csv (employee_id, full_name, photo) plus one
   <employee_id>.jpg per person, side by side at the top level. The
   `photo` column is that file name. Also downloadable from /admin
   (Employees tab).

1. One-time / catch-up utility, run BEFORE the kiosk starts so it
   doesn't compete with the camera + NPU for memory:

       python hologram_sync.py              # send everyone not sent yet
       python hologram_sync.py --all        # re-send everyone
       python hologram_sync.py --status     # just show what's pending
       python hologram_sync.py --dry-run    # list what would be sent

   start.sh runs the default form automatically when HOLOGRAM_SYNC_URL
   is set (HOLOGRAM_SYNC_ON_START=0 to skip).

2. Live: when someone is registered (kiosk or /admin photo upload) the
   kiosk calls HologramSync.push(emp_id). It sends in the background.
   If the hologram app is offline the person simply stays "not sent",
   and the utility above picks them up next time.

Request sent per employee (one at a time):

    POST {HOLOGRAM_SYNC_URL}
    X-API-Key: {HOLOGRAM_API_KEY}
    Content-Type: multipart/form-data

    employee_id = E0001
    full_name   = Madesh Ramamoorthy
    image       = <face.jpg, image/jpeg>      (omitted if no photo)

Change `_build_request` if the hologram app expects a different shape.

Status lives in faces.db: employees.hologram_synced_at is NULL until a
send succeeds, and is reset to NULL whenever the photo or name changes.
"""

from __future__ import annotations

import argparse
import csv
import io
import queue
import re
import sys
import threading
import time

import config

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None


class HologramOffline(RuntimeError):
    """The hologram app couldn't be reached at all (vs. rejecting one
    employee). The batch stops: everything left stays pending."""


# ---------- face crop -------------------------------------------------------


def face_crop_jpeg(frame_bgr, bbox, margin: float | None = None,
                   max_side: int | None = None) -> bytes | None:
    """Crop the face (plus some head / shoulders) out of a BGR frame and
    JPEG-encode it. Returns None if the crop is empty."""
    import cv2
    margin = config.HOLOGRAM_PHOTO_MARGIN if margin is None else margin
    max_side = config.HOLOGRAM_PHOTO_MAX_SIDE if max_side is None else max_side
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * margin))
    x2 = int(min(w, x2 + bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    y2 = int(min(h, y2 + bh * margin))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    side = max(crop.shape[:2])
    if side > max_side:
        scale = max_side / side
        crop = cv2.resize(crop, (int(crop.shape[1] * scale),
                                 int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf) if ok else None


# ---------- bulk-upload ZIP -------------------------------------------------


BULK_CSV_NAME = "employees.csv"
BULK_COLUMNS = ("employee_id", "full_name", "photo")


def _photo_name(emp_id: str) -> str:
    # Keep file names safe on any OS / unzip tool.
    return re.sub(r"[^A-Za-z0-9._-]", "_", emp_id) + ".jpg"


def build_bulk_zip(db, include_no_photo: bool = False) -> tuple[bytes, dict]:
    """Return (zip_bytes, counts) for the hologram bulk upload.

    employees.csv has columns employee_id, full_name, photo. `photo` is the
    file name of that person's JPEG inside the same ZIP. People without a
    photo are left out unless include_no_photo, then `photo` is empty."""
    import zipfile
    rows = db.hologram_rows(pending_only=False)
    out_rows, used, skipped = [], set(), 0
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for emp_id, name, has_photo, _at, _err in rows:
            photo = ""
            if has_photo:
                jpeg = db.get_photo(emp_id)
                photo = _photo_name(emp_id)
                n = 2
                while photo in used:            # two IDs mapping to one name
                    photo = f"{_photo_name(emp_id)[:-4]}_{n}.jpg"
                    n += 1
                used.add(photo)
                # JPEGs don't shrink further; store them as-is.
                zf.writestr(photo, jpeg, compress_type=zipfile.ZIP_STORED)
            elif not include_no_photo:
                skipped += 1
                continue
            out_rows.append((emp_id, name, photo))
        text = io.StringIO()
        w = csv.writer(text, lineterminator="\n")
        w.writerow(BULK_COLUMNS)
        w.writerows(out_rows)
        zf.writestr(BULK_CSV_NAME, text.getvalue().encode("utf-8"))
    counts = {"rows": len(out_rows),
              "photos": sum(1 for r in out_rows if r[2]),
              "skipped_no_photo": skipped}
    return buf.getvalue(), counts


# ---------- sending ---------------------------------------------------------


def enabled() -> bool:
    return bool(config.HOLOGRAM_SYNC_URL) and requests is not None


def _build_request(emp_id: str, name: str, jpeg: bytes | None):
    data = {"employee_id": emp_id, "full_name": name}
    files = ({"image": (f"{emp_id}.jpg", jpeg, "image/jpeg")}
             if jpeg else None)
    return data, files


def send_employee(db, emp_id: str, session=None) -> bool:
    """Send one employee. Marks the row sent / failed in the DB.
    Returns True on success, False if the app rejected it.
    Raises HologramOffline if the app can't be reached."""
    name = db.get_name(emp_id)
    if name is None:
        return False
    jpeg = db.get_photo(emp_id)
    data, files = _build_request(emp_id, name, jpeg)
    headers = {}
    if config.HOLOGRAM_API_KEY:
        headers["X-API-Key"] = config.HOLOGRAM_API_KEY
    http = session or requests
    try:
        r = http.post(config.HOLOGRAM_SYNC_URL, data=data, files=files,
                      headers=headers, timeout=config.HOLOGRAM_TIMEOUT_SEC)
    except (requests.ConnectionError, requests.Timeout) as exc:
        db.mark_hologram_failed(emp_id, f"offline: {exc.__class__.__name__}")
        raise HologramOffline(str(exc)) from exc
    if 200 <= r.status_code < 300:
        db.mark_hologram_sent(emp_id)
        return True
    db.mark_hologram_failed(emp_id, f"HTTP {r.status_code}: {r.text[:200]}")
    return False


def sync_pending(db, *, include_no_photo: bool | None = None,
                 delay: float = 0.0, log=print) -> dict:
    """Send every not-yet-sent employee, one by one. Stops at the first
    sign the app is offline. Returns counts."""
    if include_no_photo is None:
        include_no_photo = config.HOLOGRAM_SEND_WITHOUT_PHOTO
    sent = failed = skipped = 0
    rows = db.hologram_rows(pending_only=True)
    session = requests.Session()
    for emp_id, name, has_photo, _at, _err in rows:
        if not has_photo and not include_no_photo:
            skipped += 1
            continue
        try:
            ok = send_employee(db, emp_id, session=session)
        except HologramOffline:
            left = len(rows) - sent - failed - skipped
            log(f"[hologram] app unreachable at {config.HOLOGRAM_SYNC_URL}; "
                f"{left} left pending for next run")
            return {"sent": sent, "failed": failed, "skipped": skipped,
                    "offline": True, "left": left}
        if ok:
            sent += 1
            log(f"[hologram] sent   {emp_id} {name}"
                f"{'' if has_photo else ' (no photo)'}")
        else:
            failed += 1
            log(f"[hologram] FAILED {emp_id} {name}: "
                f"{_last_error(db, emp_id)}")
        if delay:
            time.sleep(delay)
    return {"sent": sent, "failed": failed, "skipped": skipped,
            "offline": False, "left": 0}


def _last_error(db, emp_id: str) -> str:
    row = db.conn.execute("SELECT hologram_error FROM employees "
                          "WHERE emp_id = ?", (emp_id,)).fetchone()
    return (row[0] if row else "") or ""


# ---------- live background sender (used by the kiosk) --------------------


class HologramSync:
    """Non-blocking sender for the running kiosk. push(emp_id) queues a
    send. A failure leaves the person pending. With
    HOLOGRAM_RETRY_SEC > 0 the thread also retries pending people on
    that interval while the kiosk runs."""

    def __init__(self):
        self.enabled = enabled()
        self._q: queue.Queue = queue.Queue(maxsize=256)
        if self.enabled:
            threading.Thread(target=self._run, daemon=True,
                             name="hologram-sync").start()
            print(f"[hologram] live sync on -> {config.HOLOGRAM_SYNC_URL}",
                  flush=True)

    def push(self, emp_id: str) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(emp_id)
        except queue.Full:
            pass  # stays pending in the DB; next retry / utility run sends it

    def _run(self) -> None:
        from database import FaceDB
        db = FaceDB()  # own connection for this thread
        session = requests.Session()
        retry = config.HOLOGRAM_RETRY_SEC
        last_retry = time.time()
        while True:
            try:
                emp_id = self._q.get(timeout=5.0)
            except queue.Empty:
                emp_id = None
            if emp_id is not None:
                try:
                    ok = send_employee(db, emp_id, session=session)
                    print(f"[hologram] live send {emp_id}: "
                          f"{'ok' if ok else 'rejected, ' + _last_error(db, emp_id)}",
                          flush=True)
                except HologramOffline:
                    print(f"[hologram] app offline; {emp_id} left pending",
                          flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[hologram] live send {emp_id} crashed: {exc!r}",
                          flush=True)
            if retry > 0 and time.time() - last_retry >= retry:
                last_retry = time.time()
                try:
                    res = sync_pending(db, log=lambda m: print(m, flush=True))
                    if res["sent"]:
                        print(f"[hologram] retry sent {res['sent']}",
                              flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[hologram] retry crashed: {exc!r}", flush=True)


# ---------- CLI -------------------------------------------------------------


def _print_status(db) -> None:
    rows = db.hologram_rows(pending_only=False)
    pending = [r for r in rows if r[3] is None]
    no_photo = [r for r in rows if not r[2]]
    print(f"Employees: {len(rows)}   sent: {len(rows) - len(pending)}   "
          f"pending: {len(pending)}   without photo: {len(no_photo)}")
    for emp_id, name, has_photo, _at, err in pending:
        flag = "" if has_photo else "  [no photo]"
        print(f"  pending {emp_id:<10} {name:<28}{flag}"
              f"{'  last error: ' + err if err else ''}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--all", action="store_true",
                   help="mark everyone unsent first, then send all")
    p.add_argument("--include-no-photo", action="store_true",
                   help="also send people who have no photo yet")
    p.add_argument("--status", action="store_true",
                   help="show sync status and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be sent and exit")
    p.add_argument("--export-zip", metavar="PATH",
                   help="write the bulk-upload ZIP (employees.csv + photos) "
                        "to PATH and exit; no network needed")
    p.add_argument("--delay", type=float, default=0.2,
                   help="seconds between sends (default 0.2)")
    a = p.parse_args()

    from database import FaceDB
    db = FaceDB()
    if a.export_zip:
        data, counts = build_bulk_zip(db, include_no_photo=a.include_no_photo)
        with open(a.export_zip, "wb") as fp:
            fp.write(data)
        print(f"[hologram] wrote {a.export_zip}: {counts['rows']} employees, "
              f"{counts['photos']} photos"
              + (f", {counts['skipped_no_photo']} left out (no photo; "
                 "--include-no-photo to add them)"
                 if counts['skipped_no_photo'] else ""))
        return 0
    if a.status or a.dry_run:
        _print_status(db)
        return 0
    if not enabled():
        print("[hologram] HOLOGRAM_SYNC_URL not set (or 'requests' missing); "
              "nothing to do.")
        return 0
    if a.all:
        n = db.reset_hologram_sync()
        print(f"[hologram] marked {n} employees for re-send")
    res = sync_pending(db, include_no_photo=a.include_no_photo or None,
                       delay=a.delay)
    print(f"[hologram] done: sent {res['sent']}, failed {res['failed']}, "
          f"skipped (no photo) {res['skipped']}"
          f"{', app OFFLINE, ' + str(res['left']) + ' left pending' if res['offline'] else ''}")
    db.close()
    return 2 if res["offline"] else (1 if res["failed"] else 0)


if __name__ == "__main__":
    sys.exit(main())
