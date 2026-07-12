"""Execute the real Streamlit script without starting a browser server."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


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
