from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from config import DB_PATH, EMBEDDING_DIM

# Employee IDs follow the hologram app's format: capital E + 4 digits
# (E0001, E0002 ...). New IDs are generated here, never typed by a user.
EMP_ID_PREFIX = "E"
EMP_ID_DIGITS = 4
_EMP_ID_OK = re.compile(rf"{EMP_ID_PREFIX}\d{{{EMP_ID_DIGITS},}}")
# Anything that is an E-number in disguise: "e1", "E7", "e0042" ...
_EMP_ID_LOOSE = re.compile(rf"[{EMP_ID_PREFIX}{EMP_ID_PREFIX.lower()}]0*(\d+)")


def format_emp_id(n: int) -> str:
    return f"{EMP_ID_PREFIX}{n:0{EMP_ID_DIGITS}d}"


def is_emp_id(value: str) -> bool:
    """True for an ID already in the E0001 format."""
    return bool(_EMP_ID_OK.fullmatch(value or ""))


def emp_id_number(value: str) -> int | None:
    """Numeric part of an E-style ID, tolerating case / short padding
    ("e7" -> 7). None for legacy IDs like "madesh"."""
    m = _EMP_ID_LOOSE.fullmatch((value or "").strip())
    return int(m.group(1)) if m else None

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    emp_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id     TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_emp ON face_embeddings(emp_id);

-- One face photo (JPEG crop) per person, sent to the hologram app.
CREATE TABLE IF NOT EXISTS employee_photos (
    emp_id      TEXT PRIMARY KEY,
    jpeg        BLOB NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    description TEXT,
    ordering    INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(emp_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    starts_at   TIMESTAMP NOT NULL,
    ends_at     TIMESTAMP,
    notes       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_starts ON sessions(starts_at);

CREATE TABLE IF NOT EXISTS solutions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT,
    domain      TEXT,
    link        TEXT,
    ordering    INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class FaceDB:
    def __init__(self, path: Path = DB_PATH):
        # check_same_thread=False: FaceDB is shared between the Flask
        # request threads (admin endpoints) and the camera worker
        # thread. SQLite itself serialises writes internally, and our
        # workload is light enough that lock contention is a non-issue.
        self.path = Path(path)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        # Serialises ID allocation / migration between threads of this
        # process; BEGIN IMMEDIATE covers the other process (kiosk vs admin).
        self._id_lock = threading.Lock()
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL + busy_timeout: two processes (kiosk + standalone admin) share
        # this file. WAL allows concurrent readers + one writer without
        # blocking, busy_timeout retries silently if both write at once.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 2000")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Idempotent column-adds for existing databases. Each ADD COLUMN
        is wrapped in a try so re-running on a current schema is a no-op."""
        emp_cols = {r[1] for r in self.conn.execute(
            "PRAGMA table_info(employees)").fetchall()}
        for col_def in (
            "profile_url TEXT",            # admin-set link to a public bio page
            "custom_welcome TEXT",         # admin-written welcome line (overrides cache)
            "welcome_cache TEXT",          # LLM-generated welcome from profile_url
            "profile_updated_at TIMESTAMP",
            # Hologram sync: NULL synced_at = not sent yet (or changed
            # since). hologram_error holds the last failure reason.
            "hologram_synced_at TIMESTAMP",
            "hologram_error TEXT",
        ):
            col_name = col_def.split()[0]
            if col_name in emp_cols:
                continue
            try:
                self.conn.execute(
                    f"ALTER TABLE employees ADD COLUMN {col_def}"
                )
            except sqlite3.OperationalError:
                pass

    def close(self):
        self.conn.close()

    def add_employee(self, emp_id: str, name: str, embeddings: Iterable[np.ndarray]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO employees (emp_id, name) VALUES (?, ?)",
            (emp_id, name),
        )
        for emb in embeddings:
            cur.execute(
                "INSERT INTO face_embeddings (emp_id, embedding) VALUES (?, ?)",
                (emp_id, _to_blob(emb)),
            )
        self.conn.commit()

    # ---- employee IDs: E0001 format, generated here -----------------------

    def _begin_write(self) -> None:
        """Take the write lock up front (sqlite would otherwise only grab
        it at the first INSERT, after the ID was already chosen)."""
        if self.conn.in_transaction:
            self.conn.commit()
        self.conn.execute("BEGIN IMMEDIATE")

    def _next_emp_number(self) -> int:
        """Highest E-number in use + 1 (legacy IDs like "madesh" are
        ignored). Caller holds the write lock."""
        top = 0
        for (eid,) in self.conn.execute("SELECT emp_id FROM employees"):
            n = emp_id_number(eid)
            if n is not None and n > top:
                top = n
        return top + 1

    def next_emp_id(self) -> str:
        """Preview of the next ID. add_employee_auto() allocates for real."""
        return format_emp_id(self._next_emp_number())

    def add_employee_auto(self, name: str,
                          embeddings: Iterable[np.ndarray]) -> str:
        """Create a new employee with the next free E-number and return
        it. Pick + insert happen in one write transaction, so the kiosk
        and the admin app can't hand out the same ID."""
        embeddings = list(embeddings)
        with self._id_lock:
            self._begin_write()
            try:
                emp_id = format_emp_id(self._next_emp_number())
                self.conn.execute(
                    "INSERT INTO employees (emp_id, name) VALUES (?, ?)",
                    (emp_id, name),
                )
                for emb in embeddings:
                    self.conn.execute(
                        "INSERT INTO face_embeddings (emp_id, embedding) "
                        "VALUES (?, ?)", (emp_id, _to_blob(emb)),
                    )
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise
        return emp_id

    def find_emp_id_by_name(self, name: str) -> str | None:
        """Employee with exactly this name (case-insensitive), if any.
        When several share it, the oldest wins."""
        row = self.conn.execute(
            "SELECT emp_id FROM employees WHERE lower(name) = lower(?) "
            "ORDER BY created_at, emp_id LIMIT 1", (name.strip(),),
        ).fetchone()
        return row[0] if row else None

    def data_version(self) -> int:
        """Changes whenever ANOTHER connection (e.g. the admin app) commits.
        The kiosk polls it to know when to reload its in-memory gallery."""
        return self.conn.execute("PRAGMA data_version").fetchone()[0]

    # ---- one-off migration of legacy IDs to the E0001 format -------------

    def plan_emp_id_migration(self, overrides: dict[str, str] | None = None
                              ) -> list[dict]:
        """Work out old -> new IDs without touching the DB.

        Rules, in priority order:
          1. an ID the admin typed in `overrides` wins;
          2. IDs that are already E-numbers keep their number
             (E0001 stays, "e7" becomes E0007);
          3. everyone else gets the next free number, oldest first.
        If two rows claim the same number the older one keeps it.

        Returns [{old, name, new, action, reason}] oldest first. Raises
        ValueError on a bad / duplicate override.
        """
        overrides = {k: (v or "").strip().upper()
                     for k, v in (overrides or {}).items() if (v or "").strip()}
        rows = self.conn.execute(
            "SELECT emp_id, name FROM employees ORDER BY created_at, emp_id"
        ).fetchall()
        known = {r[0] for r in rows}
        for old, new in overrides.items():
            if old not in known:
                raise ValueError(f"unknown employee {old!r}")
            if not is_emp_id(new):
                raise ValueError(
                    f"{new!r} is not in the format "
                    f"{format_emp_id(1)} ({EMP_ID_PREFIX} + at least "
                    f"{EMP_ID_DIGITS} digits)")
        dupes = {v for v in overrides.values()
                 if list(overrides.values()).count(v) > 1}
        if dupes:
            raise ValueError(f"same new ID used twice: {sorted(dupes)}")

        target: dict[str, str] = {}
        reason: dict[str, str] = {}
        taken: set[str] = set()
        for old, new in overrides.items():           # 1. manual
            target[old], reason[old] = new, "set by admin"
            taken.add(new)
        for old, _name in rows:                      # 2. already E-style
            if old in target:
                continue
            n = emp_id_number(old)
            if n is None:
                continue
            cand = format_emp_id(n)
            if cand in taken:
                continue                             # lost the tie -> step 3
            target[old] = cand
            taken.add(cand)
            reason[old] = ("already in format" if cand == old
                           else "normalised")
        top = max([emp_id_number(t) or 0 for t in taken] or [0])
        for old, _name in rows:                      # 3. new numbers
            if old in target:
                continue
            top += 1
            while format_emp_id(top) in taken:
                top += 1
            target[old] = format_emp_id(top)
            taken.add(target[old])
            reason[old] = "new number"
        return [{
            "old": old, "name": name, "new": target[old],
            "action": "keep" if old == target[old] else "rename",
            "reason": reason[old],
        } for old, name in rows]

    def backup_to(self, dest: Path | None = None) -> Path:
        """Consistent copy of the whole DB (safe while the kiosk is running)."""
        dest = Path(dest) if dest else self.path.with_name(
            f"{self.path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        out = sqlite3.connect(str(dest))
        try:
            self.conn.backup(out)
        finally:
            out.close()
        return dest

    def _rename_emp(self, old: str, new: str, resend: bool) -> None:
        self.conn.execute(
            "UPDATE employees SET emp_id = ?"
            + (", hologram_synced_at = NULL, hologram_error = NULL"
               if resend else "") + " WHERE emp_id = ?", (new, old))
        for table in ("face_embeddings", "employee_photos", "interactions"):
            self.conn.execute(
                f"UPDATE {table} SET emp_id = ? WHERE emp_id = ?", (new, old))

    def migrate_emp_ids(self, overrides: dict[str, str] | None = None) -> dict:
        """Rename every employee to its planned E0001-style ID.

        All-or-nothing: one transaction, preceded by a full backup. Two
        phases (old -> temp -> final) so IDs can swap or chain safely.
        Renamed people are marked "not sent" so the hologram app gets them
        again under the new ID. Returns {changed, kept, backup, mapping}.
        """
        with self._id_lock:
            plan = self.plan_emp_id_migration(overrides)
            changes = [p for p in plan if p["action"] == "rename"]
            result = {"changed": len(changes),
                      "kept": len(plan) - len(changes),
                      "backup": None,
                      "mapping": [{"old": p["old"], "new": p["new"],
                                   "name": p["name"]} for p in changes]}
            if not changes:
                return result
            result["backup"] = str(self.backup_to())
            self._begin_write()
            try:
                # Parent + child rows move together; check FKs at COMMIT.
                self.conn.execute("PRAGMA defer_foreign_keys = ON")
                for i, p in enumerate(changes):
                    self._rename_emp(p["old"], f"__migrating_{i}__", False)
                for i, p in enumerate(changes):
                    self._rename_emp(f"__migrating_{i}__", p["new"], True)
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise
        return result

    def add_embedding(self, emp_id: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT INTO face_embeddings (emp_id, embedding) VALUES (?, ?)",
            (emp_id, _to_blob(embedding)),
        )
        self.conn.commit()

    def load_all(self):
        """Return (emp_ids, names, matrix) where matrix is (N, D) float32."""
        rows = self.conn.execute(
            """
            SELECT f.emp_id, e.name, f.embedding
            FROM face_embeddings f
            JOIN employees e ON e.emp_id = f.emp_id
            """
        ).fetchall()
        if not rows:
            return [], [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        emp_ids = [r[0] for r in rows]
        names = [r[1] for r in rows]
        matrix = np.stack([_from_blob(r[2]) for r in rows]).astype(np.float32)
        return emp_ids, names, matrix

    def employee_exists(self, emp_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM employees WHERE emp_id = ?", (emp_id,)
        ).fetchone()
        return row is not None

    def get_name(self, emp_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT name FROM employees WHERE emp_id = ?", (emp_id,)
        ).fetchone()
        return row[0] if row else None

    def list_employees(self):
        """Return [(emp_id, name, sample_count, created_at, profile_url,
        custom_welcome, welcome_cache)] sorted by emp_id."""
        return self.conn.execute(
            """
            SELECT e.emp_id, e.name, COUNT(f.id), e.created_at,
                   e.profile_url, e.custom_welcome, e.welcome_cache
            FROM employees e
            LEFT JOIN face_embeddings f ON f.emp_id = e.emp_id
            GROUP BY e.emp_id
            ORDER BY e.emp_id
            """
        ).fetchall()

    def get_employee_profile(self, emp_id: str):
        """Return {emp_id, name, profile_url, custom_welcome, welcome_cache}
        or None when the employee doesn't exist."""
        row = self.conn.execute(
            "SELECT emp_id, name, profile_url, custom_welcome, welcome_cache "
            "FROM employees WHERE emp_id = ?", (emp_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "emp_id": row[0], "name": row[1],
            "profile_url": row[2], "custom_welcome": row[3],
            "welcome_cache": row[4],
        }

    def set_employee_profile(self, emp_id: str, **fields) -> bool:
        """Update any of: name, profile_url, custom_welcome, welcome_cache.
        None values clear the field; absent keys leave it alone."""
        cols = ("name", "profile_url", "custom_welcome", "welcome_cache")
        sets, vals = [], []
        for k in cols:
            if k in fields:
                sets.append(f"{k} = ?")
                vals.append(fields[k])
        if not sets:
            return False
        sets.append("profile_updated_at = CURRENT_TIMESTAMP")
        if "name" in fields:
            # The hologram app holds the name too: re-send it.
            sets.append("hologram_synced_at = NULL")
        vals.append(emp_id)
        cur = self.conn.execute(
            f"UPDATE employees SET {', '.join(sets)} WHERE emp_id = ?", vals,
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- face photo + hologram sync ---------------------------------------

    def set_photo(self, emp_id: str, jpeg: bytes) -> None:
        """Store / replace this person's face photo and mark them as not
        yet sent to the hologram app."""
        self.conn.execute(
            "INSERT INTO employee_photos (emp_id, jpeg) VALUES (?, ?) "
            "ON CONFLICT(emp_id) DO UPDATE SET jpeg = excluded.jpeg, "
            "captured_at = CURRENT_TIMESTAMP",
            (emp_id, jpeg),
        )
        self.conn.execute(
            "UPDATE employees SET hologram_synced_at = NULL WHERE emp_id = ?",
            (emp_id,),
        )
        self.conn.commit()

    def get_photo(self, emp_id: str) -> bytes | None:
        row = self.conn.execute(
            "SELECT jpeg FROM employee_photos WHERE emp_id = ?", (emp_id,)
        ).fetchone()
        return bytes(row[0]) if row else None

    def photo_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute(
            "SELECT emp_id FROM employee_photos").fetchall()}

    def has_photo(self, emp_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM employee_photos WHERE emp_id = ?", (emp_id,)
        ).fetchone() is not None

    def hologram_rows(self, pending_only: bool = True):
        """Return [(emp_id, name, has_photo, synced_at, last_error)],
        sorted by emp_id. pending_only=True keeps only not-yet-sent rows."""
        where = "WHERE e.hologram_synced_at IS NULL" if pending_only else ""
        return self.conn.execute(
            f"""
            SELECT e.emp_id, e.name, p.emp_id IS NOT NULL,
                   e.hologram_synced_at, e.hologram_error
            FROM employees e
            LEFT JOIN employee_photos p ON p.emp_id = e.emp_id
            {where}
            ORDER BY e.emp_id
            """
        ).fetchall()

    def mark_hologram_sent(self, emp_id: str) -> None:
        self.conn.execute(
            "UPDATE employees SET hologram_synced_at = CURRENT_TIMESTAMP, "
            "hologram_error = NULL WHERE emp_id = ?", (emp_id,),
        )
        self.conn.commit()

    def mark_hologram_failed(self, emp_id: str, error: str) -> None:
        self.conn.execute(
            "UPDATE employees SET hologram_synced_at = NULL, "
            "hologram_error = ? WHERE emp_id = ?", (error[:300], emp_id),
        )
        self.conn.commit()

    def reset_hologram_sync(self) -> int:
        """Mark everyone as not sent (for a full re-send)."""
        cur = self.conn.execute(
            "UPDATE employees SET hologram_synced_at = NULL")
        self.conn.commit()
        return cur.rowcount

    def delete_employee(self, emp_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM employees WHERE emp_id = ?", (emp_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def purge_all_faces(self) -> int:
        """Wipe every registered face: deletes all employees, which
        CASCADE-deletes their embeddings. Interaction counts (the hi-5
        aggregates) are intentionally left intact. Returns the number of
        employee rows removed."""
        cur = self.conn.execute("DELETE FROM employees")
        self.conn.commit()
        return cur.rowcount

    def purge_faces_older_than(self, hours: int) -> int:
        """Delete employees registered more than `hours` ago (CASCADE
        removes their embeddings). No-op when hours <= 0. Returns the
        number of employee rows removed."""
        if hours <= 0:
            return 0
        cur = self.conn.execute(
            "DELETE FROM employees WHERE created_at < datetime('now', ?)",
            (f"-{int(hours)} hours",),
        )
        self.conn.commit()
        return cur.rowcount

    def get_embeddings(self, emp_id: str) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT embedding FROM face_embeddings WHERE emp_id = ?",
            (emp_id,),
        ).fetchall()
        if not rows:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([_from_blob(r[0]) for r in rows]).astype(np.float32)

    def count_embeddings(self, emp_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE emp_id = ?",
            (emp_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def trim_embeddings(self, emp_id: str, keep_n: int) -> int:
        """Keep the newest `keep_n` embeddings for this emp_id, drop older
        ones. Returns the number deleted."""
        if keep_n <= 0:
            return 0
        rows = self.conn.execute(
            "SELECT id FROM face_embeddings WHERE emp_id = ? "
            "ORDER BY id DESC LIMIT -1 OFFSET ?",
            (emp_id, keep_n),
        ).fetchall()
        if not rows:
            return 0
        ids = [r[0] for r in rows]
        self.conn.executemany(
            "DELETE FROM face_embeddings WHERE id = ?",
            [(i,) for i in ids],
        )
        self.conn.commit()
        return len(ids)

    # ---- projects --------------------------------------------------------

    def list_projects(self):
        """Return [(id, title, description, ordering, created_at)] sorted
        by (ordering, id)."""
        return self.conn.execute(
            "SELECT id, title, description, ordering, created_at "
            "FROM projects ORDER BY ordering, id"
        ).fetchall()

    def get_project(self, project_id: int):
        return self.conn.execute(
            "SELECT id, title, description, ordering, created_at "
            "FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    def add_project(self, title: str, description: str = "",
                    ordering: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO projects (title, description, ordering) VALUES (?, ?, ?)",
            (title, description, ordering),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_project(self, project_id: int, *, title: str | None = None,
                       description: str | None = None,
                       ordering: int | None = None) -> bool:
        sets, vals = [], []
        if title is not None:
            sets.append("title = ?")
            vals.append(title)
        if description is not None:
            sets.append("description = ?")
            vals.append(description)
        if ordering is not None:
            sets.append("ordering = ?")
            vals.append(ordering)
        if not sets:
            return False
        vals.append(project_id)
        cur = self.conn.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id = ?",
            vals,
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_project(self, project_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM projects WHERE id = ?", (project_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- solutions catalog (searched from chat) -------------------------

    def list_solutions(self):
        """Return [(id, name, description, domain, link, ordering)] sorted
        by (ordering, id)."""
        return self.conn.execute(
            "SELECT id, name, description, domain, link, ordering "
            "FROM solutions ORDER BY ordering, id"
        ).fetchall()

    def add_solution(self, name: str, description: str = "", domain: str = "",
                     link: str = "", ordering: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO solutions (name, description, domain, link, ordering) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, description, domain, link, ordering),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def delete_solution(self, solution_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM solutions WHERE id = ?", (solution_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- interaction counter --------------------------------------------

    def record_interaction(self, emp_id: str, session_id: str) -> bool:
        """Returns True iff this is a new (emp_id, session_id) pair."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO interactions (emp_id, session_id) VALUES (?, ?)",
            (emp_id, session_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def interaction_count_total(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()
        return int(row[0]) if row else 0

    def interaction_count_session(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def interaction_count_today(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions "
            "WHERE date(ts, 'localtime') = date('now', 'localtime')"
        ).fetchone()
        return int(row[0]) if row else 0

    def interaction_count_this_week(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions "
            "WHERE ts >= datetime('now', '-7 days')"
        ).fetchone()
        return int(row[0]) if row else 0

    def interaction_best_day(self) -> tuple[str | None, int]:
        """Returns (weekday_name, count) of the historical best day,
        or (None, 0) if no interactions yet."""
        row = self.conn.execute(
            "SELECT strftime('%w', ts, 'localtime') AS dow, "
            "       date(ts, 'localtime') AS d, "
            "       COUNT(*) AS c "
            "FROM interactions "
            "GROUP BY d ORDER BY c DESC LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return (None, 0)
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        try:
            name = days[int(row[0])]
        except (ValueError, IndexError):
            name = None
        return (name, int(row[2]))

    # ---- sessions (upcoming events on the idle dashboard) --------------

    def list_sessions(self, upcoming_only: bool = True):
        if upcoming_only:
            return self.conn.execute(
                "SELECT id, title, starts_at, ends_at, notes "
                "FROM sessions WHERE starts_at >= datetime('now', '-1 hour') "
                "ORDER BY starts_at"
            ).fetchall()
        return self.conn.execute(
            "SELECT id, title, starts_at, ends_at, notes "
            "FROM sessions ORDER BY starts_at DESC"
        ).fetchall()

    def next_session(self):
        return self.conn.execute(
            "SELECT id, title, starts_at, ends_at, notes "
            "FROM sessions WHERE starts_at >= datetime('now') "
            "ORDER BY starts_at LIMIT 1"
        ).fetchone()

    def add_session(self, title: str, starts_at: str,
                    ends_at: str | None = None, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (title, starts_at, ends_at, notes) "
            "VALUES (?, ?, ?, ?)",
            (title, starts_at, ends_at, notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_session(self, session_id: int, *, title: str | None = None,
                       starts_at: str | None = None,
                       ends_at: str | None = None,
                       notes: str | None = None) -> bool:
        sets, vals = [], []
        for col, val in (("title", title), ("starts_at", starts_at),
                          ("ends_at", ends_at), ("notes", notes)):
            if val is not None:
                sets.append(f"{col} = ?")
                vals.append(val)
        if not sets:
            return False
        vals.append(session_id)
        cur = self.conn.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", vals,
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_session(self, session_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)
