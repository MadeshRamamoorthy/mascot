"""Admin CLI for the face DB.

Usage:
    python admin.py list                 # all employees + sample counts
    python admin.py show E001            # one employee, with embedding metadata
    python admin.py delete E001          # remove employee + all their embeddings
    python admin.py export employees.csv # dump employees to CSV (no embeddings)
    python admin.py export employees.json  # same, as JSON (by extension)
    python admin.py cleanup              # report data problems (dry run)
    python admin.py cleanup --apply      # fix the safe ones (see cmd_cleanup)
    python admin.py path                 # print DB file location
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict

from config import DB_PATH
from database import FaceDB


def cmd_list(_args):
    db = FaceDB()
    rows = db.list_employees()
    if not rows:
        print("(no employees registered)")
        return
    print(f"{'EMP_ID':<12} {'NAME':<24} {'SAMPLES':>8}  CREATED_AT")
    print("-" * 70)
    for emp_id, name, count, created, *_profile in rows:
        print(f"{emp_id:<12} {name:<24} {count:>8}  {created}")
    print(f"\nTotal: {len(rows)} employees, DB at {DB_PATH}")
    db.close()


def cmd_show(args):
    db = FaceDB()
    if not db.employee_exists(args.emp_id):
        print(f"No such employee: {args.emp_id}")
        sys.exit(1)
    name = db.get_name(args.emp_id)
    rows = db.conn.execute(
        "SELECT id, length(embedding), created_at "
        "FROM face_embeddings WHERE emp_id = ? ORDER BY id",
        (args.emp_id,),
    ).fetchall()
    print(f"Employee: {args.emp_id}  ({name})")
    print(f"Embeddings: {len(rows)}")
    for row_id, blob_len, created in rows:
        dim = blob_len // 4  # float32
        print(f"  #{row_id:<4} dim={dim:<4} created={created}")
    db.close()


def cmd_delete(args):
    db = FaceDB()
    if not db.employee_exists(args.emp_id):
        print(f"No such employee: {args.emp_id}")
        sys.exit(1)
    name = db.get_name(args.emp_id)
    if not args.yes:
        ans = input(f"Delete {args.emp_id} ({name}) and all embeddings? [y/N] ")
        if ans.strip().lower() != "y":
            print("Cancelled.")
            return
    db.delete_employee(args.emp_id)
    print(f"Deleted {args.emp_id}.")
    db.close()


EXPORT_COLUMNS = ["emp_id", "name", "sample_count", "created_at",
                  "profile_url", "custom_welcome", "welcome_cache"]


def cmd_export(args):
    db = FaceDB()
    rows = [dict(zip(EXPORT_COLUMNS, r)) for r in db.list_employees()]
    db.close()
    if args.csv_path.lower().endswith(".json"):
        with open(args.csv_path, "w", encoding="utf-8") as fp:
            json.dump(rows, fp, indent=2, ensure_ascii=False)
    else:
        with open(args.csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.csv_path}")


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def cmd_cleanup(args):
    """Find (and with --apply, fix) employee-data problems.

    Fixed automatically with --apply:
      - names with stray / doubled whitespace      -> trimmed
      - employees with zero face samples           -> deleted (can never
                                                      be recognised)
      - face samples whose employee row is gone    -> deleted
      - people over SILENT_LEARN_MAX_SAMPLES_PER_PERSON samples
                                                   -> trimmed to the cap
    Only reported, because a human has to decide:
      - the same name registered under several emp_ids
    """
    import config
    db = FaceDB()
    c = db.conn
    rows = db.list_employees()
    cap = getattr(config, "SILENT_LEARN_MAX_SAMPLES_PER_PERSON", 0)

    messy = [(e, n) for e, n, *_ in rows
             if n != re.sub(r"\s+", " ", (n or "").strip())]
    empty = [(e, n) for e, n, cnt, *_ in rows if cnt == 0]
    over = [(e, n, cnt) for e, n, cnt, *_ in rows if cap and cnt > cap]
    orphans = c.execute(
        "SELECT COUNT(*) FROM face_embeddings f "
        "LEFT JOIN employees e ON e.emp_id = f.emp_id "
        "WHERE e.emp_id IS NULL").fetchone()[0]
    by_name = defaultdict(list)
    for e, n, cnt, *_ in rows:
        by_name[_norm_name(n)].append((e, cnt))
    dupes = {k: v for k, v in by_name.items() if len(v) > 1}

    print(f"Employees: {len(rows)}   DB: {DB_PATH}\n")
    print(f"Names with stray whitespace : {len(messy)}")
    for e, n in messy:
        print(f"    {e:<12} {n!r}")
    print(f"No face samples (unusable)  : {len(empty)}")
    for e, n in empty:
        print(f"    {e:<12} {n}")
    print(f"Over the {cap}-sample cap      : {len(over)}")
    for e, n, cnt in over:
        print(f"    {e:<12} {n:<24} {cnt} samples")
    print(f"Orphaned face samples       : {orphans}")
    print(f"Same name, several IDs      : {len(dupes)}  (review by hand)")
    for _k, ids in dupes.items():
        print("    " + ", ".join(f"{e} ({cnt} samples)" for e, cnt in ids))

    if not args.apply:
        print("\nDry run. Re-run with --apply to fix everything above "
              "except the duplicate names. Back up faces.db first.")
        db.close()
        return

    for e, n in messy:
        db.set_employee_profile(e, name=re.sub(r"\s+", " ", n.strip()))
    for e, _n in empty:
        db.delete_employee(e)
    for e, _n, _cnt in over:
        db.trim_embeddings(e, cap)
    c.execute("DELETE FROM face_embeddings WHERE emp_id NOT IN "
              "(SELECT emp_id FROM employees)")
    c.commit()
    c.execute("VACUUM")
    print(f"\nApplied: {len(messy)} renamed, {len(empty)} deleted, "
          f"{len(over)} trimmed, {orphans} orphan samples removed.")
    db.close()


def cmd_path(_args):
    print(DB_PATH)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("emp_id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("delete")
    s.add_argument("emp_id")
    s.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("export")
    s.add_argument("csv_path")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("cleanup", help="report / fix employee data problems")
    s.add_argument("--apply", action="store_true",
                   help="apply the safe fixes (default is a dry run)")
    s.set_defaults(func=cmd_cleanup)

    sub.add_parser("path").set_defaults(func=cmd_path)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
