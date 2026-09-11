"""Run the real dashboard, model inference, report download and browser video playback."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/browser-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as temporary:
        env = {k: v for k, v in os.environ.items() if not k.startswith("UTILITY_SAFETY_")}
        env.update(UTILITY_SAFETY_OUTPUT_DIR=temporary, UTILITY_SAFETY_REPO_ROOT=str(root))
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(root / "src/utility_safety_ai/web/app.py"),
                    f"--server.port={port}",
                    "--server.address=127.0.0.1",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                    "--theme.base=dark",
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                for _ in range(100):
                    try:
                        if (
                            httpx.get(
                                f"http://127.0.0.1:{port}/_stcore/health", timeout=1
                            ).status_code
                            == 200
                        ):
                            break
                    except httpx.HTTPError:
                        pass
                    if server.poll() is not None:
                        raise RuntimeError("Dashboard server exited; inspect server.log.")
                    time.sleep(0.1)
                with sync_playwright() as playwright:
                    expect.set_options(timeout=60000)
                    browser = playwright.chromium.launch()
                    page = browser.new_page(
                        viewport={"width": 1440, "height": 1100}, device_scale_factor=1
                    )
                    page.set_default_timeout(60000)
                    page.goto(f"http://127.0.0.1:{port}")
                    expect(
                        page.get_by_role("button", name="Run included sample", exact=True)
                    ).to_be_visible()
                    page.screenshot(path=str(args.output / "dashboard-v2.2.png"), full_page=True)
                    page.get_by_role("button", name="Run included sample", exact=True).click()
                    expect(page.get_by_text("Frames analysed", exact=True)).to_be_visible()
                    expect(
                        page.get_by_role("button", name="Save review decisions", exact=True)
                    ).to_be_visible()
                    page.screenshot(path=str(args.output / "review-v2.2.png"), full_page=True)
                    with page.expect_download() as download:
                        page.get_by_role(
                            "button", name="Download complete report", exact=True
                        ).click()
                    archive_path = Path(temporary) / "report.zip"
                    download.value.save_as(archive_path)
                    with zipfile.ZipFile(archive_path) as archive:
                        manifest = json.loads(archive.read("manifest.json"))
                        assert manifest["status"] == "completed"
                        assert manifest["config"]["privacy_blur_enabled"] is True
                        assert "review_history.json" in archive.namelist()
                        for artifact in manifest["artifacts"]:
                            assert (
                                hashlib.sha256(archive.read(artifact["path"])).hexdigest()
                                == artifact["sha256"]
                            )
                    page.get_by_text("Event timeline", exact=True).scroll_into_view_if_needed()
                    page.screenshot(
                        path=str(args.output / "review-details-v2.2.png"), full_page=True
                    )
                    page.get_by_text("Monitor", exact=True).click()
                    page.get_by_text("Video", exact=True).click()
                    expect(page.get_by_role("region", name="Upload worksite video")).to_be_visible()
                    page.locator("input[type=file]").set_input_files(
                        str(root / "examples/sample_videos/construction_ppe_pan.mp4")
                    )
                    page.get_by_role("button", name="Run safety analysis", exact=True).click()
                    expect(page.locator("video")).to_be_visible(timeout=90000)
                    page.wait_for_function(
                        'document.querySelector("video")?.readyState >= 2', timeout=30000
                    )
                    assert page.locator("video").evaluate("(video) => video.duration") > 0
                    page.locator("video").evaluate(
                        "(video) => {video.muted=true; return video.play()}"
                    )
                    page.wait_for_function('document.querySelector("video")?.currentTime > 0')
                    page.screenshot(path=str(args.output / "video-v2.2.png"), full_page=True)
                    assert page.locator('[data-testid="stException"]').count() == 0
                    browser.close()
                print(
                    "PASS: dashboard, real PPE image/video inference, automatic results navigation, private report download, artifact hashes and Chromium H.264 playback."
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
