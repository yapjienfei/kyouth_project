"""PostgreSQL layer — jobs storage, user history, skill snapshots, path feedback.

Falls back gracefully: if DATABASE_URL is unreachable, callers get None/[] and
the app keeps serving from the JSON snapshot.
"""
import json
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://career:career@localhost:5432/career")
DATA_DIR = Path(__file__).parent.parent / "data"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    company         TEXT,
    salary_min      INTEGER DEFAULT 0,
    salary_max      INTEGER DEFAULT 0,
    salary_display  TEXT,
    description     TEXT,
    skills_ai       JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS search_history (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    search_type TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_history_user ON search_history (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS skill_snapshots (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    skills      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_snapshots_user ON skill_snapshots (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS path_feedback (
    id           SERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL,
    target_title TEXT NOT NULL,
    path_key     TEXT NOT NULL,
    path_titles  JSONB NOT NULL,
    rating       SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, path_key)
);
CREATE INDEX IF NOT EXISTS idx_feedback_target ON path_feedback (target_title);
"""


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db() -> bool:
    """Create tables and seed jobs from jobs_snapshot.json if the table is empty.
    Returns True if the database is available."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            cur.execute("SELECT count(*) FROM jobs")
            if cur.fetchone()[0] == 0:
                snapshot = DATA_DIR / "jobs_snapshot.json"
                if snapshot.exists():
                    with open(snapshot) as f:
                        jobs = json.load(f)
                    rows = [
                        (
                            j.get("title", ""),
                            j.get("company", ""),
                            j.get("salary_min") or 0,
                            j.get("salary_max") or 0,
                            j.get("salary_display", ""),
                            (j.get("description") or "")[:8000],
                            json.dumps(j.get("skills_ai", [])),
                        )
                        for j in jobs
                    ]
                    psycopg2.extras.execute_values(
                        cur,
                        """INSERT INTO jobs
                           (title, company, salary_min, salary_max, salary_display, description, skills_ai)
                           VALUES %s""",
                        rows,
                    )
                    print(f"[db] Seeded {len(rows)} jobs from jobs_snapshot.json")
        return True
    except Exception as e:
        print(f"[db] init failed ({e}) — falling back to JSON files")
        return False


def load_jobs() -> list[dict] | None:
    """Load all jobs from postgres. Returns None if unavailable."""
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT title, company, salary_min, salary_max, salary_display, skills_ai
                   FROM jobs"""
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[db] load_jobs failed ({e})")
        return None


# ── User history ─────────────────────────────────────────────────────────────

def log_search(user_id: str, search_type: str, payload: dict) -> bool:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO search_history (user_id, search_type, payload) VALUES (%s, %s, %s)",
                (user_id, search_type, json.dumps(payload)),
            )
        return True
    except Exception as e:
        print(f"[db] log_search failed ({e})")
        return False


def snapshot_skills(user_id: str, skills: list[str]) -> bool:
    """Store a skill snapshot, but only when it differs from the latest one
    (so the timeline records acquisition events, not every page load)."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT skills FROM skill_snapshots WHERE user_id=%s ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            normalized = sorted(set(s.lower() for s in skills))
            if row and sorted(row[0]) == normalized:
                return False
            cur.execute(
                "INSERT INTO skill_snapshots (user_id, skills) VALUES (%s, %s)",
                (user_id, json.dumps(normalized)),
            )
        return True
    except Exception as e:
        print(f"[db] snapshot_skills failed ({e})")
        return False


def get_history(user_id: str, limit: int = 25) -> dict:
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT search_type, payload, created_at FROM search_history
                   WHERE user_id=%s ORDER BY created_at DESC LIMIT %s""",
                (user_id, limit),
            )
            searches = [
                {
                    "type": r["search_type"],
                    "payload": r["payload"],
                    "at": r["created_at"].isoformat(),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """SELECT skills, created_at FROM skill_snapshots
                   WHERE user_id=%s ORDER BY created_at ASC LIMIT 100""",
                (user_id,),
            )
            timeline = []
            prev: set = set()
            for r in cur.fetchall():
                now_set = set(r["skills"])
                timeline.append(
                    {
                        "at": r["created_at"].isoformat(),
                        "count": len(now_set),
                        "added": sorted(now_set - prev),
                        "removed": sorted(prev - now_set),
                    }
                )
                prev = now_set
            return {"searches": searches, "skill_timeline": timeline}
    except Exception as e:
        print(f"[db] get_history failed ({e})")
        return {"searches": [], "skill_timeline": [], "error": "history unavailable"}


# ── Path feedback ────────────────────────────────────────────────────────────

def save_feedback(user_id: str, target_title: str, path_titles: list[str], rating: int) -> bool:
    path_key = "→".join(path_titles)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO path_feedback (user_id, target_title, path_key, path_titles, rating)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (user_id, path_key)
                   DO UPDATE SET rating = EXCLUDED.rating, created_at = now()""",
                (user_id, target_title, path_key, json.dumps(path_titles), rating),
            )
        return True
    except Exception as e:
        print(f"[db] save_feedback failed ({e})")
        return False


def delete_feedback(user_id: str, path_titles: list[str]) -> bool:
    """Remove a user's vote on a path (clicking the same thumb again)."""
    path_key = "→".join(path_titles)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM path_feedback WHERE user_id=%s AND path_key=%s",
                (user_id, path_key),
            )
        return True
    except Exception as e:
        print(f"[db] delete_feedback failed ({e})")
        return False


def get_feedback(target_title: str) -> dict[str, dict]:
    """Aggregate net votes per path for a target role: {path_key: {up, down, net}}"""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT path_key,
                          count(*) FILTER (WHERE rating = 1)  AS up,
                          count(*) FILTER (WHERE rating = -1) AS down
                   FROM path_feedback WHERE target_title=%s GROUP BY path_key""",
                (target_title,),
            )
            return {
                r[0]: {"up": r[1], "down": r[2], "net": r[1] - r[2]}
                for r in cur.fetchall()
            }
    except Exception as e:
        print(f"[db] get_feedback failed ({e})")
        return {}
