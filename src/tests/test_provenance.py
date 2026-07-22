"""Provenance policy tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from utility_safety_ai.governance import audit_provenance


def test_audit_rejects_unapproved_license_hash_drift_and_unregistered_file(tmp_path: Path):
    images = tmp_path / "examples" / "sample_images"
    images.mkdir(parents=True)
    (images / "registered.jpg").write_bytes(b"registered")
    (images / "unregistered.jpg").write_bytes(b"unregistered")
    manifest = tmp_path / "provenance.yaml"
    manifest.write_text(
        """
artifacts:
  - path: examples/sample_images/registered.jpg
    sha256: deadbeef
    license: Proprietary
""",
        encoding="utf-8",
    )

    report = audit_provenance(manifest, tmp_path)

    assert report["passed"] is False
    assert any("Unapproved license" in error for error in report["errors"])
    assert any("SHA-256 mismatch" in error for error in report["errors"])
    assert any("Unregistered audited artifact" in error for error in report["errors"])


def test_audit_rejects_duplicate_and_missing_records(tmp_path: Path):
    manifest = tmp_path / "provenance.yaml"
    manifest.write_text(
        """
artifacts:
  - path: docs/assets/missing.jpg
    sha256: deadbeef
    license: AGPL-3.0-only
  - path: docs/assets/missing.jpg
    sha256: deadbeef
    license: AGPL-3.0-only
""",
        encoding="utf-8",
    )

    report = audit_provenance(manifest, tmp_path)

    assert report["passed"] is False
    assert any("Duplicate artifact record" in error for error in report["errors"])
    assert any("Missing artifact" in error for error in report["errors"])


def test_text_lf_hash_is_stable_across_windows_checkout_line_endings(tmp_path: Path):
    evidence = tmp_path / "docs" / "model-evaluation" / "metrics.json"
    evidence.parent.mkdir(parents=True)
    canonical = b'{\n  "metric": 0.9\n}\n'
    evidence.write_bytes(canonical.replace(b"\n", b"\r\n"))
    manifest = tmp_path / "provenance.yaml"
    manifest.write_text(
        f"""
artifacts:
  - path: docs/model-evaluation/metrics.json
    sha256: {hashlib.sha256(canonical).hexdigest()}
    hash_mode: text-lf
    license: AGPL-3.0-only
""",
        encoding="utf-8",
    )

    report = audit_provenance(manifest, tmp_path)

    assert report["passed"], report["errors"]
