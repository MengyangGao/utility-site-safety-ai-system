"""Durable event index, append-only review history, and leased delivery outbox."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .event import SafetyEvent

REVIEW_STATUSES = ("unreviewed", "confirmed", "false_positive", "needs_follow_up")
SAFE_METADATA = frozenset(
    {
        "run_id",
        "stream_segment",
        "incident_id",
        "lifecycle_state",
        "confirmation_status",
        "confirmation_count",
        "confirmation_required",
        "ppe_type",
        "compliance",
        "confidence",
        "evidence_class",
        "zone_risk",
        "dwell_seconds",
        "time_in_zone_seconds",
        "required_ppe",
        "privacy_blur_enabled",
        "privacy_mode",
    }
)


def public_event(event: SafetyEvent) -> dict[str, Any]:
    """Allowlist the integration payload; never transmit source URIs, media or paths."""
    record = asdict(event)
    record["source_path"] = None
    record["snapshot_path"] = None
    record["metadata"] = {
        key: value for key, value in event.metadata.items() if key in SAFE_METADATA
    }
    return record


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL,
                    payload TEXT NOT NULL, snapshot_path TEXT, created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL REFERENCES events(event_id),
                    status TEXT NOT NULL, note TEXT NOT NULL,
                    reviewer TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reviews_event ON reviews(event_id, id);
                CREATE TABLE IF NOT EXISTS outbox (
                    event_id TEXT NOT NULL REFERENCES events(event_id), target TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_at REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
                    lease_token TEXT, last_error TEXT,
                    PRIMARY KEY (event_id, target)
                );
                CREATE INDEX IF NOT EXISTS outbox_due ON outbox(state, next_at, lease_until);
                CREATE TABLE IF NOT EXISTS delivery_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                    target TEXT NOT NULL, outcome TEXT NOT NULL, created_at REAL NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, event: SafetyEvent, targets: tuple[str, ...] = ()) -> None:
        payload = json.dumps(public_event(event), sort_keys=True, allow_nan=False)
        run_id = str(event.metadata.get("run_id", ""))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT payload FROM events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if existing and existing[0] != payload:
                raise ValueError("An event identifier cannot be reused for changed evidence.")
            db.execute(
                "INSERT OR IGNORE INTO events(event_id,run_id,payload,snapshot_path,created_at) VALUES(?,?,?,?,?)",
                (event.event_id, run_id, payload, event.snapshot_path, time.time()),
            )
            db.executemany(
                "INSERT OR IGNORE INTO outbox(event_id,target) VALUES(?,?)",
                [(event.event_id, target) for target in targets],
            )

    def list_events(
        self, *, after: int = 0, limit: int = 100, run_id: str | None = None
    ) -> list[dict]:
        if not 1 <= limit <= 500 or after < 0:
            raise ValueError("limit must be 1–500 and after must be non-negative")
        with self.connect() as db:
            rows = db.execute(
                """SELECT e.sequence,e.payload,
                (SELECT status FROM reviews r WHERE r.event_id=e.event_id ORDER BY id DESC LIMIT 1) AS review_status,
                (SELECT note FROM reviews r WHERE r.event_id=e.event_id ORDER BY id DESC LIMIT 1) AS operator_note
                FROM events e WHERE sequence>? AND (? IS NULL OR run_id=?) ORDER BY sequence LIMIT ?""",
                (after, run_id, run_id, limit),
            ).fetchall()
            return [
                {
                    **json.loads(r["payload"]),
                    "sequence": r["sequence"],
                    "review_status": r["review_status"] or "unreviewed",
                    "operator_note": r["operator_note"] or "",
                }
                for r in rows
            ]

    def review(self, event_id: str, status: str, note: str, reviewer: str) -> int:
        if (
            status not in REVIEW_STATUSES
            or len(note) > 2000
            or not 1 <= len(reviewer.strip()) <= 100
        ):
            raise ValueError("Invalid review decision, reviewer or note length.")
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone():
                raise KeyError("Event not found")
            result = db.execute(
                "INSERT INTO reviews(event_id,status,note,reviewer,created_at) VALUES(?,?,?,?,?)",
                (event_id, status, note, reviewer.strip(), time.time()),
            )
            return int(result.lastrowid)

    def review_history(self, event_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reviews WHERE event_id=? ORDER BY id", (event_id,)
                )
            ]

    def run_reviews(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT r.* FROM reviews r JOIN events e USING(event_id) WHERE e.run_id=? ORDER BY r.id",
                    (run_id,),
                )
            ]

    def claim(
        self, targets: tuple[str, ...], *, now: float | None = None, lease_seconds: float = 30
    ) -> dict | None:
        if not targets:
            return None
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in targets)
            row = db.execute(
                f"""SELECT o.*,e.payload FROM outbox o JOIN events e USING(event_id)
                WHERE target IN ({placeholders}) AND next_at<=? AND
                (state='pending' OR (state='sending' AND lease_until<=?))
                ORDER BY next_at,e.sequence LIMIT 1""",
                (*targets, now, now),
            ).fetchone()
            if not row:
                return None
            token = uuid.uuid4().hex
            db.execute(
                "UPDATE outbox SET state='sending', attempts=attempts+1, lease_until=?, lease_token=? WHERE event_id=? AND target=?",
                (now + lease_seconds, token, row["event_id"], row["target"]),
            )
            return {**dict(row), "attempts": row["attempts"] + 1, "lease_token": token}

    def finish(
        self,
        item: dict,
        *,
        error: str | None = None,
        permanent: bool = False,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        state = (
            "delivered"
            if error is None
            else "dead"
            if permanent or item["attempts"] >= 6
            else "pending"
        )
        # Error values are local classifications, never upstream response text.
        safe_error = error[:120] if error else None
        with self.connect() as db:
            updated = db.execute(
                """UPDATE outbox SET state=?,next_at=?,lease_until=0,lease_token=NULL,last_error=?
                WHERE event_id=? AND target=? AND lease_token=?""",
                (
                    state,
                    now + min(60, 2 ** min(item["attempts"], 6)),
                    safe_error,
                    item["event_id"],
                    item["target"],
                    item["lease_token"],
                ),
            ).rowcount
            if updated:
                db.execute(
                    "INSERT INTO delivery_attempts(event_id,target,outcome,created_at) VALUES(?,?,?,?)",
                    (item["event_id"], item["target"], safe_error or "delivered", now),
                )
            return bool(updated)

    def delivery_summary(self) -> dict[str, int]:
        with self.connect() as db:
            counts = dict.fromkeys(("pending", "sending", "delivered", "dead"), 0)
            counts.update(
                {r[0]: r[1] for r in db.execute("SELECT state,count(*) FROM outbox GROUP BY state")}
            )
            return counts

    def retry_failed(self) -> int:
        with self.connect() as db:
            return db.execute(
                "UPDATE outbox SET state='pending',attempts=0,next_at=0,last_error=NULL WHERE state='dead'"
            ).rowcount

    def evidence_path(self, event_id: str, output_root: Path) -> Path | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT run_id,snapshot_path FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        if not row or not row["snapshot_path"]:
            return None
        root = (output_root / "runs" / row["run_id"]).resolve()
        if root.parent != (output_root / "runs").resolve():
            raise ValueError("Invalid run path")
        path = (root / row["snapshot_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence path escapes the run")
        return path if path.is_file() else None
