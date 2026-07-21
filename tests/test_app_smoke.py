"""Execute the real Streamlit script without starting a browser server."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utility_safety_ai.web.app import _normalize_preset
from utility_safety_ai.web_helpers import parse_zone_yaml


def test_streamlit_app_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    privacy_controls = [
        control
        for control in app.toggle
        if "privacy blur" in control.label.lower()
    ]
    assert len(privacy_controls) == 1
    assert privacy_controls[0].value is True
    assert any(button.label == "Run portfolio sample" for button in app.button)
    assert any(select.label == "Verified model profile" for select in app.selectbox)
    assert any(select.label == "Privacy redaction style" for select in app.selectbox)
    assert any(radio.label == "Video processing" for radio in app.radio)


def test_pixel_zone_presets_use_their_own_reference_dimensions():
    zones = parse_zone_yaml(_normalize_preset("PPE work area"))

    assert zones[0].polygon[0] == pytest.approx((100 / 640, 280 / 640))
    assert zones[0].polygon[2] == pytest.approx((540 / 640, 620 / 640))
