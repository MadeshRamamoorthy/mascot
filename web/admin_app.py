"""Standalone Flask app for /admin.

Runs independently of the kiosk runtime. No StateBus, no FrameStreamer,
no camera worker -- just CRUD against `faces.db` plus a synchronous
photo-register endpoint that lazily acquires a HailoFacePipeline.

Entrypoint: admin_web.py (top-level) calls `create_admin_app(FaceDB())`
and runs it on port 8081.
"""

from __future__ import annotations

from typing import Callable

from flask import Flask, jsonify, render_template, request

from web._auth import check_admin_auth, request_admin_auth


def _default_pipeline_factory():
    """Open a fresh HailoFacePipeline. Caller closes it. Per the user's
    chosen lifecycle (create+release per upload), this runs on every
    /api/register/photo request and the pipeline is closed before the
    response returns."""
    import config
    from hailo_infer import HailoFacePipeline
    return HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)


def create_admin_app(
    db,
    *,
    pipeline_factory: Callable[[], object] | None = None,
) -> Flask:
    """Build the Flask app that serves /admin and its API.

    Args:
        db: FaceDB instance. The app holds the reference; readers and
            writers go through it.
        pipeline_factory: callable returning a HailoFacePipeline (or a
            mock for tests). Defaults to the real Hailo path. Called once
            per photo-register request and closed afterwards.
    """
    if pipeline_factory is None:
        pipeline_factory = _default_pipeline_factory
    hologram = None            # created on first registration

    def _send_to_hologram(emp_id: str) -> None:
        nonlocal hologram
        try:
            if hologram is None:
                from hologram_sync import HologramSync
                hologram = HologramSync()
            hologram.push(emp_id)
        except Exception as exc:  # noqa: BLE001
            # Stays pending in the DB; the next sync run sends it.
            print(f"[hologram] admin push failed for {emp_id}: {exc!r}")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # -------- admin HTML --------------------------------------------------

    @app.route("/")
    def admin_index():
        if not check_admin_auth():
            return request_admin_auth()
        return render_template("admin.html")

    # -------- projects ---------------------------------------------------

    @app.route("/api/projects", methods=["GET"])
    def projects_list():
        rows = db.list_projects()
        return jsonify([
            {"id": r[0], "title": r[1], "description": r[2],
             "ordering": r[3], "created_at": r[4]} for r in rows
        ])

    @app.route("/api/projects", methods=["POST"])
    def projects_create():
        if not check_admin_auth():
            return request_admin_auth()
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        if not title:
            return ("title required", 400)
        desc = (body.get("description") or "").strip()
        try:
            ordering = int(body.get("ordering", 0))
        except (TypeError, ValueError):
            ordering = 0
        pid = db.add_project(title, desc, ordering)
        return jsonify({"id": pid})

    @app.route("/api/projects/<int:pid>", methods=["POST", "DELETE"])
    def projects_edit(pid):
        if not check_admin_auth():
            return request_admin_auth()
        if request.method == "DELETE":
            if not db.delete_project(pid):
                return ("not found", 404)
            return ("", 204)
        body = request.get_json(silent=True) or {}
        kw = {}
        for key in ("title", "description"):
            if key in body:
                kw[key] = (body[key] or "").strip()
        if "ordering" in body:
            try:
                kw["ordering"] = int(body["ordering"])
            except (TypeError, ValueError):
                pass
        if not kw or not db.update_project(pid, **kw):
            return ("no change", 400)
        return ("", 204)

    # -------- sessions ---------------------------------------------------

    @app.route("/api/sessions", methods=["GET"])
    def sessions_list():
        rows = db.list_sessions(upcoming_only=False)
        return jsonify([
            {"id": r[0], "title": r[1], "starts_at": r[2], "ends_at": r[3],
             "notes": r[4]} for r in rows
        ])

    @app.route("/api/sessions", methods=["POST"])
    def sessions_create():
        if not check_admin_auth():
            return request_admin_auth()
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        starts_at = (body.get("starts_at") or "").strip()
        if not title or not starts_at:
            return ("title and starts_at required", 400)
        sid = db.add_session(
            title, starts_at,
            body.get("ends_at") or None,
            (body.get("notes") or "").strip(),
        )
        return jsonify({"id": sid})

    @app.route("/api/sessions/<int:sid>", methods=["POST", "DELETE"])
    def sessions_edit(sid):
        if not check_admin_auth():
            return request_admin_auth()
        if request.method == "DELETE":
            if not db.delete_session(sid):
                return ("not found", 404)
            return ("", 204)
        body = request.get_json(silent=True) or {}
        ok = db.update_session(sid, **{
            k: (body[k] or "").strip() if isinstance(body.get(k), str) else body.get(k)
            for k in ("title", "starts_at", "ends_at", "notes") if k in body
        })
        return ("", 204) if ok else ("no change", 400)

    # -------- employees --------------------------------------------------

    @app.route("/api/employees", methods=["GET"])
    def emp_list():
        if not check_admin_auth():
            return request_admin_auth()
        rows = db.list_employees()
        return jsonify([
            {"emp_id": r[0], "name": r[1], "samples": r[2],
             "created_at": r[3], "profile_url": r[4] or "",
             "custom_welcome": r[5] or "",
             "welcome_cache": r[6] or ""}
            for r in rows
        ])

    @app.route("/api/employees/<emp_id>", methods=["POST", "DELETE"])
    def emp_edit(emp_id):
        if not check_admin_auth():
            return request_admin_auth()
        if request.method == "DELETE":
            if not db.delete_employee(emp_id):
                return ("not found", 404)
            return ("", 204)
        body = request.get_json(silent=True) or {}
        if not db.employee_exists(emp_id):
            return ("not found", 404)
        update = {}
        if "name" in body:
            name = (body.get("name") or "").strip()
            if not name:
                return ("name cannot be empty", 400)
            update["name"] = name
        if "profile_url" in body:
            update["profile_url"] = (body.get("profile_url") or "").strip() or None
        if "custom_welcome" in body:
            update["custom_welcome"] = (body.get("custom_welcome") or "").strip() or None
        if not update:
            return ("nothing to update", 400)
        db.set_employee_profile(emp_id, **update)
        return ("", 204)

    @app.route("/api/faces/purge", methods=["POST"])
    def faces_purge():
        """Event reset: wipe ALL registered faces (employees + their
        embeddings). Interaction counts are kept. Irreversible."""
        if not check_admin_auth():
            return request_admin_auth()
        deleted = db.purge_all_faces()
        return jsonify({"ok": True, "deleted": deleted})

    @app.route("/api/employees/<emp_id>/welcome", methods=["POST", "DELETE"])
    def emp_regen_welcome(emp_id):
        """POST   : re-fetch profile_url and regenerate welcome cache.
        DELETE : clear the cached welcome (back to default random
                 greeting at next recognition)."""
        if not check_admin_auth():
            return request_admin_auth()
        if not db.employee_exists(emp_id):
            return ("not found", 404)
        if request.method == "DELETE":
            db.set_employee_profile(emp_id, welcome_cache=None)
            return jsonify({"ok": True, "welcome": ""})
        # POST -- regenerate from URL.
        profile = db.get_employee_profile(emp_id)
        url = (profile.get("profile_url") or "").strip()
        if not url:
            return jsonify({"ok": False,
                             "error": "no profile_url set on this employee"}), 400
        from profile_fetch import regenerate
        welcome, err = regenerate(profile["name"], url)
        if err:
            return jsonify({"ok": False, "error": err}), 502
        db.set_employee_profile(emp_id, welcome_cache=welcome)
        return jsonify({"ok": True, "welcome": welcome})

    # -------- photo register (synchronous, lazy pipeline) ---------------

    @app.route("/api/register/photo", methods=["POST"])
    def register_photo():
        """Multipart: emp_id, name, photos[].

        Per-request HailoFacePipeline lifecycle: acquire on entry,
        release before responding. Costs ~3 s of NPU acquire + HEF load
        per upload but keeps the admin process's idle memory footprint
        small. (Flip to a process-lifetime pipeline by holding the
        instance on the closure if photo onboarding becomes batchy.)
        """
        if not check_admin_auth():
            return request_admin_auth()
        emp_id = (request.form.get("emp_id") or "").strip()
        name = (request.form.get("name") or "").strip()
        if not emp_id or not name:
            return ("emp_id and name required", 400)
        photos = request.files.getlist("photos")
        if not photos:
            return ("at least one photo file required", 400)
        blobs = [f.read() for f in photos]
        blobs = [b for b in blobs if b]
        if not blobs:
            return ("uploaded photos are empty", 400)

        from photo_register import register_from_photos
        pipeline = pipeline_factory()
        try:
            result = register_from_photos(db, pipeline, emp_id, name, blobs)
        finally:
            try:
                pipeline.close()
            except Exception:
                pass

        if result.get("ok"):
            _send_to_hologram(emp_id)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    # -------- hologram bulk-upload ZIP ----------------------------------

    @app.route("/api/hologram/export.zip")
    def hologram_export():
        """employees.csv (employee_id, full_name, photo) + photos, for the
        hologram app's bulk upload. ?include_no_photo=1 adds people who
        have no photo yet (empty photo cell)."""
        if not check_admin_auth():
            return request_admin_auth()
        import io
        import time
        from flask import send_file
        from hologram_sync import build_bulk_zip
        include = request.args.get("include_no_photo") in ("1", "true", "yes")
        data, counts = build_bulk_zip(db, include_no_photo=include)
        resp = send_file(
            io.BytesIO(data), mimetype="application/zip", as_attachment=True,
            download_name=f"hologram_upload_{time.strftime('%Y%m%d_%H%M')}.zip",
        )
        resp.headers["X-Employee-Rows"] = str(counts["rows"])
        resp.headers["X-Skipped-No-Photo"] = str(counts["skipped_no_photo"])
        return resp

    # -------- metrics ----------------------------------------------------

    @app.route("/api/metrics")
    def metrics():
        best_day, best_count = db.interaction_best_day()
        return jsonify({
            "total": db.interaction_count_total(),
            "today": db.interaction_count_today(),
            "week": db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        })

    return app
