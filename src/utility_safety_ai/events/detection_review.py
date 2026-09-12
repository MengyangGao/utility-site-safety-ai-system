"""Review raw predictions independently from policy events; never rewrite run evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .store import REVIEW_STATUSES, EventStore

LOG_PATH = "events/detections.jsonl"
PAGE_SIZE = 25


def detection_page(run_dir: Path, page: int = 0) -> dict:
    """Read one bounded page and verify the complete log against its completed manifest.

    Memory use is bounded by one page and a 64 KiB record. The log is streamed once
    for hashing/counting, including records outside the requested page.
    """
    if page < 0:
        raise ValueError("Page must be non-negative.")
    manifest_path, path = run_dir / "manifest.json", run_dir / LOG_PATH
    if not path.resolve().is_relative_to(run_dir.resolve()) or path.is_symlink():
        raise ValueError("Detection log escapes the run.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("run_id") != run_dir.name:
        raise ValueError("Only completed, matching runs can be reviewed.")
    expected = next(
        (
            a.get("sha256")
            for a in manifest.get("artifacts", [])
            if a.get("path", "").replace("\\", "/") == LOG_PATH
        ),
        None,
    )
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("This run has no verifiable detection log.")
    hasher, rows, total = hashlib.sha256(), [], 0
    with path.open("rb") as stream:
        while line := stream.readline(65537):
            if len(line) > 65536:
                raise ValueError("Detection record exceeds the 64 KiB review limit.")
            hasher.update(line)
            if not line.strip():
                raise ValueError("Detection log contains an empty record.")
            total += 1
            if page * PAGE_SIZE < total <= (page + 1) * PAGE_SIZE:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Invalid detection record.")
                rows.append(
                    {
                        "detection_id": hashlib.sha256(
                            f"{run_dir.name}:{expected}:{total}".encode()
                        ).hexdigest(),
                        "line_number": total,
                        **{
                            key: record.get(key)
                            for key in (
                                "frame_index",
                                "time_seconds",
                                "class_name",
                                "confidence",
                                "bbox",
                                "track_id",
                            )
                        },
                    }
                )
    if hasher.hexdigest() != expected:
        raise ValueError("Detection evidence changed. Restore the original log before reviewing.")
    return {
        "run_id": run_dir.name,
        "log_sha256": expected,
        "total": total,
        "page": page,
        "rows": rows,
    }


def review_history(store: EventStore, run_id: str, log_sha256: str | None = None) -> list[dict]:
    with store.connect() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT * FROM detection_reviews WHERE run_id=? AND (? IS NULL OR log_sha256=?) ORDER BY id",
                (run_id, log_sha256, log_sha256),
            )
        ]


def latest_reviews(store: EventStore, run_id: str, log_sha256: str) -> dict[str, dict]:
    with store.connect() as db:
        return {
            row["detection_id"]: dict(row)
            for row in db.execute(
                "SELECT * FROM detection_reviews WHERE id IN (SELECT MAX(id) FROM detection_reviews WHERE run_id=? AND log_sha256=? GROUP BY detection_id)",
                (run_id, log_sha256),
            )
        }


def save_reviews(
    store: EventStore,
    run_dir: Path,
    page: int,
    expected_sha256: str,
    decisions: list[dict],
    reviewer: str,
) -> int:
    """Validate the current evidence again and append a page's decisions atomically."""
    if not 1 <= len(reviewer.strip()) <= 100 or len(decisions) > PAGE_SIZE:
        raise ValueError("Provide a reviewer label of 1–100 characters and at most 25 decisions.")
    evidence = detection_page(run_dir, page)
    if evidence["log_sha256"] != expected_sha256:
        raise ValueError("This run was replaced. Reopen it before reviewing.")
    records = {row["detection_id"]: row for row in evidence["rows"]}
    seen, inserts = set(), []
    for decision in decisions:
        key, status, note = decision["detection_id"], decision["status"], decision["note"]
        if (
            key not in records
            or key in seen
            or status not in REVIEW_STATUSES
            or not isinstance(note, str)
            or len(note) > 2000
        ):
            raise ValueError("Invalid detection, decision or note (maximum 2000 characters).")
        seen.add(key)
        inserts.append(
            (
                run_dir.name,
                expected_sha256,
                key,
                records[key]["line_number"],
                status,
                note,
                reviewer.strip(),
                time.time(),
            )
        )
    with store.connect() as db:
        db.executemany(
            "INSERT INTO detection_reviews(run_id,log_sha256,detection_id,line_number,status,note,reviewer,created_at) VALUES(?,?,?,?,?,?,?,?)",
            inserts,
        )
    return len(inserts)
