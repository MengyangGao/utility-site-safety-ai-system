"""Execute the real Streamlit script without starting a browser server."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utility_safety_ai.web.app import _normalize_preset
from utility_safety_ai.web_helpers import parse_zone_yaml


def test_streamlit_app_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "src" / "utility_safety_ai" / "web" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    privacy_controls = [
        control for control in app.toggle if "privacy blur" in control.label.lower()
    ]
    assert len(privacy_controls) == 1
    assert privacy_controls[0].value is True
    assert any(button.label == "Run included sample" for button in app.button)
    assert any(select.label == "Model profile" for select in app.selectbox)
    assert any(select.label == "Privacy redaction style" for select in app.selectbox)
    assert any(radio.label == "Video processing" for radio in app.radio)


def test_pixel_zone_presets_use_their_own_reference_dimensions():
    zones = parse_zone_yaml(_normalize_preset("PPE work area"))

    assert zones[0].polygon[0] == pytest.approx((100 / 640, 280 / 640))
    assert zones[0].polygon[2] == pytest.approx((540 / 640, 620 / 640))


def test_review_save_appends_history_without_rewriting_manifest(tmp_path):
    import json

    from test_pipeline_e2e import FakeDetector, _write_image

    from utility_safety_ai.events.store import EventStore
    from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline

    source = tmp_path / "sample.jpg"
    _write_image(source)
    output = tmp_path / "outputs"
    _, events = run_image_pipeline(source, output, FakeDetector(), zones=[], run_id="review-test")
    run_dir = output / "runs/review-test"
    original_manifest = (run_dir / "manifest.json").read_bytes()
    app_path = Path(__file__).resolve().parents[1] / "src/utility_safety_ai/web/app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    app.session_state.selected_run = str(run_dir)
    app.session_state.workspace_view = "results"
    app.run()
    assert not app.exception
    app.session_state["review_review-test"] = {
        "edited_rows": {
            0: {"review_status": "confirmed", "operator_note": "Reviewed in the test fixture."}
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    button = next(button for button in app.button if button.label == "Save review decisions")
    button.click().run()
    assert not app.exception
    history = EventStore(output / "events.sqlite3").review_history(events[0].event_id)
    assert len(history) == 1
    assert history[0]["status"] == "confirmed"
    assert (run_dir / "manifest.json").read_bytes() == original_manifest
    assert json.loads(original_manifest)["status"] == "completed"
