import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utility_safety_ai.events.detection_logger import DetectionLogger
from utility_safety_ai.events.detection_review import (
    detection_page,
    latest_reviews,
    review_history,
    save_reviews,
)
from utility_safety_ai.events.event import Detection
from utility_safety_ai.events.store import EventStore
from utility_safety_ai.utils.paths import OutputPaths
from utility_safety_ai.web.results import _bundle_run


@pytest.fixture
def run(tmp_path):
    paths = OutputPaths(tmp_path, run_id="predictions-only")
    paths.start_manifest(source_type="image", source="fixture.jpg", model={}, config={})
    logger = DetectionLogger(paths.events)
    for number in range(29):
        logger.log(
            Detection(2, "gloves", 0.7, (1, 2, 3, 4)), "image", "fixture.jpg", frame_index=number
        )
    paths.complete_manifest(metrics={"detections": 29, "events": 0})
    return paths.root


def decision(row, status="false_positive"):
    return {
        "detection_id": row["detection_id"],
        "status": status,
        "note": "A CCTV fixture was mistaken for gloves.",
    }


def test_prediction_reviews_are_paginated_append_only_and_independent_of_events(run):
    original = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    first, second = detection_page(run), detection_page(run, 1)
    assert first["total"] == 29 and len(first["rows"]) == 25 and len(second["rows"]) == 4
    assert second["rows"][0]["line_number"] == 26
    assert not {r["detection_id"] for r in first["rows"]} & {
        r["detection_id"] for r in second["rows"]
    }
    store = EventStore(run.parent.parent / "events.sqlite3")
    assert not store.list_events()
    row = first["rows"][0]
    save_reviews(store, run, 0, first["log_sha256"], [decision(row)], "Operator")
    save_reviews(
        store, run, 0, first["log_sha256"], [decision(row, "needs_follow_up")], "Second operator"
    )
    assert len(review_history(store, run.name)) == 2
    assert (
        latest_reviews(store, run.name, first["log_sha256"])[row["detection_id"]]["status"]
        == "needs_follow_up"
    )
    with zipfile.ZipFile(io.BytesIO(_bundle_run(run))) as archive:
        assert len(json.loads(archive.read("detection_review_history.json"))) == 2
        for name, data in original.items():
            assert archive.read(name) == data == (run / name).read_bytes()


@pytest.mark.parametrize(
    "invalid", ["unknown", "other_page", "long_note", "bad_status", "duplicate"]
)
def test_invalid_batch_cannot_partially_save(run, invalid):
    first = detection_page(run)
    good = decision(first["rows"][0])
    bad = decision(first["rows"][1])
    if invalid == "unknown":
        bad["detection_id"] = "missing"
    if invalid == "other_page":
        bad = decision(detection_page(run, 1)["rows"][0])
    if invalid == "long_note":
        bad["note"] = "x" * 2001
    if invalid == "bad_status":
        bad["status"] = "certainly-safe"
    if invalid == "duplicate":
        bad = good.copy()
    store = EventStore(run.parent.parent / "events.sqlite3")
    with pytest.raises(ValueError):
        save_reviews(store, run, 0, first["log_sha256"], [good, bad], "Operator")
    assert not review_history(store, run.name)


def test_changed_evidence_and_replaced_runs_fail_closed(run):
    first = detection_page(run)
    path = run / "events/detections.jsonl"
    path.write_bytes(path.read_bytes().replace(b"gloves", b"helmet"))
    with pytest.raises(ValueError, match="evidence changed"):
        detection_page(run)
    manifest = json.loads((run / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        if artifact["path"].replace("\\", "/") == "events/detections.jsonl":
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps(manifest))
    replacement = detection_page(run)
    assert replacement["rows"][0]["detection_id"] != first["rows"][0]["detection_id"]
    store = EventStore(run.parent.parent / "events.sqlite3")
    with pytest.raises(ValueError, match="replaced"):
        save_reviews(store, run, 0, first["log_sha256"], [decision(first["rows"][0])], "Operator")


def test_review_ui_saves_predictions_without_any_event(run):
    evidence = detection_page(run)
    script = Path(__file__).resolve().parents[1] / "src/utility_safety_ai/web/app.py"
    app = AppTest.from_file(str(script), default_timeout=20).run()
    app.session_state.selected_run = str(run)
    app.session_state.workspace_view = "results"
    app.run()
    assert not app.exception
    key = f"detection_review_{run.name}_{evidence['log_sha256']}_0"
    app.session_state[key] = {
        "edited_rows": {0: {"review_status": "false_positive", "operator_note": "CCTV fixture"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    next(b for b in app.button if b.label == "Save detection reviews").click().run()
    assert not app.exception
    assert any(m.label == "Marked false positive" and m.value == "1" for m in app.metric)
    store = EventStore(run.parent.parent / "events.sqlite3")
    assert review_history(store, run.name)[0]["note"] == "CCTV fixture"
    assert not store.list_events()
