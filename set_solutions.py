"""Set the kiosk's solutions catalog (the things ECHO can match a
visitor's question against).

REPLACES the solutions table with exactly the list below. Run on the Pi:

    python set_solutions.py

This drives the answers to "what solutions have you built?" / "what
tools do you have?" / "do you have something for X?". The idle-screen
projects card is a separate list managed by set_projects.py.

Each entry is (name, description, domain, link); ordering is taken from
the list index. Descriptions/domains/links are left blank for now --
fill them in per solution when you're ready and re-run this script.
"""

from __future__ import annotations

from database import FaceDB


SOLUTIONS = [
    ("ECHO SCOPE Interaction", "", "", ""),
    ("Railcar Quality Passport", "", "", ""),
    ("Jarvis", "", "", ""),
    ("Performax (AI in IoT)", "", "", ""),
    ("Recruitment Buddy", "", "", ""),
    ("Landing Zone for Hyperscale (Infrastructure)", "", "", ""),
    ("Subscription360 (AI Agentic Chatbot)", "", "", ""),
    ("RFP Governance Model", "", "", ""),
    ("Data Marketplace", "", "", ""),
    ("Data Migration", "", "", ""),
    ("ReconFlow - Topaz Vibathon", "", "", ""),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_solutions()
        for row in existing:
            db.delete_solution(row[0])
        for i, (name, desc, domain, link) in enumerate(SOLUTIONS, start=1):
            db.add_solution(name, desc, domain, link, ordering=i)
        print(f"solutions: removed {len(existing)}, inserted {len(SOLUTIONS)}")
        for i, (name, *_rest) in enumerate(SOLUTIONS, start=1):
            print(f"  {i:>2}. {name}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
