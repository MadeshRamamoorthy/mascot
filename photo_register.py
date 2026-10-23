"""Photo-only face registration -- detect, embed, persist.

Shared between the standalone admin app (web/admin_app.py) and the
deprecated kiosk-side admin path. Pure function: callers own the
HailoFacePipeline lifecycle and the database connection.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

import config


def _largest_detection(dets):
    if not dets:
        return None
    return max(
        dets,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )


def register_from_photos(
    db,
    pipeline,
    emp_id: str | None,
    name: str,
    photos: Iterable[bytes],
    add_embeddings: bool = True,
    check_match: bool = True,
) -> dict:
    """Process one or more photo bytes blobs into face embeddings.

    Returns:
        {"ok": True,  "samples": N, "emp_id": "E0007", "created": bool}
                                                   if N >= 1 face embeddings saved.
        {"ok": False, "error": "<reason>"}         if zero usable photos.

    Args:
        db:        FaceDB instance. Writer's responsibility.
        pipeline:  HailoFacePipeline instance with .detect / .embed.
                   Caller owns the lifecycle (close after this returns).
        emp_id:    None (normal): a new person is created and the next
                   E0001-style ID is generated. An existing ID appends the
                   photos to that person instead; unknown IDs are refused
                   (IDs are never typed in, see database.add_employee_auto).
        name:      Human-readable name. Ignored when emp_id already exists.
        photos:    Iterable of raw image bytes (JPEG / PNG / etc.).
        add_embeddings: existing person only. False = just replace their
                   stored photo, don't add the faces to recognition.
        check_match: existing person only. Refuse (error "mismatch") when
                   the uploaded face doesn't look like the faces already
                   stored for them, so a wrong file can't pollute their
                   recognition or reach the hologram app. Pass False to
                   override after a human has checked.
    """
    # Local import so the admin process doesn't fail to load when
    # hailo_infer's heavy deps aren't installed.
    from hailo_infer import align_face

    photos = list(photos or [])
    if not photos:
        return {"ok": False, "error": "no photos provided"}

    embeddings = []
    skipped = 0
    best_photo = None          # (detector score, jpeg) of the clearest face
    for blob in photos:
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        # cv2 decodes as BGR; Hailo's SCRFD was trained on RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = pipeline.detect(
            img,
            config.DETECTOR_SCORE_THRESHOLD,
            config.DETECTOR_NMS_IOU,
        )
        det = _largest_detection(dets)
        if det is None:
            skipped += 1
            continue
        aligned = align_face(img, det.landmarks)
        embeddings.append(pipeline.embed(aligned))
        if best_photo is None or det.score > best_photo[0]:
            from hologram_sync import face_crop_jpeg
            # Crop from the BGR decode so colours are right in the JPEG.
            jpeg = face_crop_jpeg(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                  det.bbox)
            if jpeg:
                best_photo = (det.score, jpeg)

    if not embeddings:
        return {
            "ok": False,
            "error": (
                f"no clear face found in any of {len(photos)} photo(s); "
                "try frontal, well-lit images"
            ),
        }

    created = emp_id is None
    match_score = None
    if not created and not db.employee_exists(emp_id):
        return {"ok": False, "error": f"unknown employee {emp_id!r}"}
    if not created:
        existing = db.get_embeddings(emp_id)
        if existing.shape[0] > 0:
            match_score = max(float((existing @ e).max()) for e in embeddings)
            if check_match and match_score < config.REREGISTER_MATCH_THRESHOLD:
                return {
                    "ok": False, "error": "mismatch",
                    "match_score": round(match_score, 2),
                    "message": (
                        f"This face doesn't look like {db.get_name(emp_id)} "
                        f"(similarity {match_score:.2f}, needs "
                        f"{config.REREGISTER_MATCH_THRESHOLD:.2f}). Check it's "
                        "the right person, then use 'Upload anyway'."),
                }
    if created:
        emp_id = db.add_employee_auto(name, embeddings)
    elif add_embeddings:
        for e in embeddings:
            db.add_embedding(emp_id, e)
    if best_photo is not None:
        db.set_photo(emp_id, best_photo[1])

    return {"ok": True, "emp_id": emp_id, "created": created,
            "samples": len(embeddings) if (created or add_embeddings) else 0,
            "skipped": skipped, "photo": best_photo is not None,
            "match_score": (round(match_score, 2)
                            if match_score is not None else None)}
