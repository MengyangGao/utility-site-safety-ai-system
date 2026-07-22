"""Validate the repository's third-party and generated-artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

APPROVED_LICENSES = frozenset({"AGPL-3.0-only", "CC0-1.0"})
AUDITED_SUFFIXES = frozenset({".csv", ".gif", ".jpg", ".json", ".mp4", ".png", ".yaml"})
AUDITED_DIRECTORIES = (
    Path("examples/sample_images"),
    Path("examples/sample_videos"),
    Path("docs/assets"),
    Path("docs/model-evaluation"),
)
AUDITED_MODEL_DIRECTORY = Path("models")


def _sha256(path: Path, hash_mode: str = "raw") -> str:
    digest = hashlib.sha256()
    if hash_mode == "text-lf":
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(content)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _audited_files(repo_root: Path) -> set[str]:
    files: set[str] = set()
    for relative_dir in AUDITED_DIRECTORIES:
        directory = repo_root / relative_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDITED_SUFFIXES:
                files.add(path.relative_to(repo_root).as_posix())
    model_directory = repo_root / AUDITED_MODEL_DIRECTORY
    if model_directory.exists():
        for path in model_directory.glob("*.pt"):
            if path.is_file():
                files.add(path.relative_to(repo_root).as_posix())
    return files


def audit_provenance(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Return a deterministic report for the repository provenance manifest."""
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    artifacts = payload.get("artifacts") or []
    errors: list[str] = []
    recorded: set[str] = set()

    for artifact in artifacts:
        relative_path = str(artifact.get("path", ""))
        license_id = str(artifact.get("license", ""))
        expected_hash = str(artifact.get("sha256", ""))
        hash_mode = str(artifact.get("hash_mode", "raw"))
        if not relative_path:
            errors.append("Artifact record is missing path")
            continue
        if relative_path in recorded:
            errors.append(f"Duplicate artifact record: {relative_path}")
        recorded.add(relative_path)
        if license_id not in APPROVED_LICENSES:
            errors.append(f"Unapproved license {license_id!r}: {relative_path}")
        if hash_mode not in {"raw", "text-lf"}:
            errors.append(f"Unsupported hash mode {hash_mode!r}: {relative_path}")
            continue
        path = repo_root / relative_path
        if not path.is_file():
            errors.append(f"Missing artifact: {relative_path}")
        elif _sha256(path, hash_mode) != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative_path}")

    audited = _audited_files(repo_root)
    for relative_path in sorted(audited - recorded):
        errors.append(f"Unregistered audited artifact: {relative_path}")
    for relative_path in sorted(recorded - audited):
        errors.append(f"Manifest path is outside the audited artifact set: {relative_path}")

    return {
        "passed": not errors,
        "artifact_count": len(recorded),
        "approved_licenses": sorted(APPROVED_LICENSES),
        "errors": errors,
    }
