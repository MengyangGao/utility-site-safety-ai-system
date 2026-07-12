"""Run-scoped output paths and transactional audit manifest helpers."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA_VERSION = "1.0"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def generate_run_id() -> str:
    """Return a collision-resistant, chronologically sortable run identifier."""
    return _new_run_id()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace ``path`` with bytes written on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace ``path`` with a UTF-8 JSON document."""
    content = (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()
    _atomic_write_bytes(path, content)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must be a JSON object: {path}")
    return manifest


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unavailable"


def _runtime_manifest() -> dict[str, Any]:
    """Return reproducibility metadata without hostnames or filesystem paths."""
    return {
        "utility_safety_ai_version": _distribution_version("utility-safety-ai"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "ultralytics": _distribution_version("ultralytics"),
            "torch": _distribution_version("torch"),
            "opencv": _distribution_version("opencv-python", "opencv-python-headless"),
        },
    }


def _find_git_root() -> Path | None:
    # In the source/editable layout this is the project root. Do not fall back
    # to the current directory: a wheel may be invoked from an unrelated Git
    # checkout, whose commit must never be misreported as this package's code.
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / ".git").exists() else None


def _git_provenance() -> dict[str, Any]:
    """Return commit identity and dirty state without recording changed paths."""
    root = _find_git_root()
    unavailable = {
        "commit_sha": None,
        "dirty": None,
        "unavailable_reason": "git repository metadata unavailable",
    }
    if root is None:
        return unavailable
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return unavailable
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        return unavailable
    return {
        "commit_sha": commit.lower(),
        "dirty": bool(status.strip()),
        "unavailable_reason": None,
    }


def resolve_latest_run(
    output_root: str | Path,
    *,
    include_failed: bool = False,
) -> Path | None:
    """Return the most recent published run or diagnostic run.

    By default, the atomic ``latest.json`` pointer is accepted only when it
    resolves to a direct child of ``runs/`` whose manifest is completed and
    whose run ID matches the pointer. This fail-closed validation prevents a
    stale pointer from presenting a failed run as successful.

    ``include_failed=True`` scans published and ``failed-runs/`` manifests and
    may return a running or failed diagnostic run.
    """
    output_root = Path(output_root)
    runs_root = output_root / "runs"

    if not include_failed:
        pointer_path = output_root / "latest.json"
        if not pointer_path.is_file():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            relative = pointer["run_dir"]
            pointer_run_id = pointer["run_id"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid latest-run pointer: {pointer_path}") from exc

        candidate = (output_root / relative).resolve()
        expected_parent = runs_root.resolve()
        if candidate.parent != expected_parent:
            raise ValueError(f"Unsafe latest-run pointer in {pointer_path}")
        if not candidate.is_dir():
            return None
        manifest = _read_manifest(candidate / "manifest.json")
        if manifest.get("status") != "completed":
            raise ValueError(
                f"Latest-run pointer does not reference a completed run: {candidate}"
            )
        if manifest.get("run_id") != pointer_run_id or candidate.name != pointer_run_id:
            raise ValueError(f"Latest-run pointer and manifest disagree: {pointer_path}")
        return candidate

    candidates: list[Path] = []
    for parent in (runs_root, output_root / "failed-runs"):
        if parent.is_dir():
            candidates.extend(
                path
                for path in parent.iterdir()
                if path.is_dir() and (path / "manifest.json").is_file()
            )

    def created_at(path: Path) -> tuple[str, int]:
        try:
            value = _read_manifest(path / "manifest.json").get("created_at", "")
        except ValueError:
            value = ""
        return str(value), path.stat().st_mtime_ns

    return max(candidates, key=created_at, default=None)


class OutputPaths:
    """Manage immutable, run-scoped artifacts beneath an output root.

    A new run writes directly to ``<base>/runs/<run_id>``. When explicitly
    overwriting an existing ID, the replacement is built under a hidden staging
    directory while the old published run remains intact. Only a fully completed
    replacement is atomically swapped into place. A failed replacement is moved
    to ``failed-runs/`` and cannot invalidate the old run or latest pointer.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str | None = None,
        overwrite: bool = False,
    ) -> None:
        self.base_root = Path(root)
        self.run_id = run_id or _new_run_id()
        if not _RUN_ID_RE.fullmatch(self.run_id) or ".." in self.run_id:
            raise ValueError(
                "run_id must contain only letters, numbers, '.', '_' or '-' and must not contain '..'"
            )

        self.final_root = self.base_root / "runs" / self.run_id
        self.root = self.final_root
        self._set_working_root(self.root)
        self._overwrite = overwrite
        self._transactional_overwrite = False
        self._prepared = False
        self._manifest: dict[str, Any] = {}

    @property
    def published_root(self) -> Path:
        """Stable run path exposed to callers after successful publication."""
        return self.final_root

    def ensure_directories(self) -> None:
        """Create an isolated run or overwrite staging directory."""
        if self._prepared:
            return
        if self.final_root.exists():
            if not self._overwrite:
                raise FileExistsError(
                    f"Run output already exists: {self.final_root}. Use overwrite=True only intentionally."
                )
            transaction_root = self.base_root / ".transactions"
            transaction_root.mkdir(parents=True, exist_ok=True)
            staging_root = transaction_root / f"{self.run_id}-{uuid.uuid4().hex}.staging"
            self._set_working_root(staging_root)
            self._transactional_overwrite = True
        for directory in (self.images, self.videos, self.events, self.snapshots):
            directory.mkdir(parents=True, exist_ok=False)
        self._prepared = True

    def start_manifest(
        self,
        *,
        source_type: str,
        source: str,
        model: dict[str, Any],
        config: dict[str, Any],
        source_integrity: dict[str, Any] | None = None,
    ) -> Path:
        """Create a running manifest before inference begins."""
        self.ensure_directories()
        self._manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "created_at": _utc_now(),
            "completed_at": None,
            "published_run_dir": str(self.final_root.relative_to(self.base_root)),
            "source": {
                "type": source_type,
                "value": source,
                **(source_integrity or {}),
            },
            "model": model,
            "config": config,
            "runtime": _runtime_manifest(),
            "code": _git_provenance(),
            "artifacts": [],
            "error": None,
        }
        _atomic_write_json(self.manifest_path, self._manifest)
        return self.manifest_path

    def complete_manifest(self, *, metrics: dict[str, Any] | None = None) -> Path:
        """Finalize the manifest and transactionally publish this run."""
        if not self._manifest:
            raise RuntimeError("start_manifest() must be called before complete_manifest()")
        self._manifest.update(
            status="completed",
            completed_at=_utc_now(),
            artifacts=self._artifact_inventory(),
            metrics=metrics or {},
        )
        _atomic_write_json(self.manifest_path, self._manifest)
        if self._transactional_overwrite:
            self._publish_overwrite()
        else:
            _atomic_write_json(self.base_root / "latest.json", self._latest_pointer())
        return self.manifest_path

    def fail_manifest(self, error: BaseException) -> Path | None:
        """Record a failed run without changing successful evidence or latest."""
        if not self._manifest:
            return None
        diagnostic_root: Path | None = None
        if self._transactional_overwrite:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            diagnostic_root = (
                self.base_root
                / "failed-runs"
                / f"{self.run_id}-{timestamp}-{uuid.uuid4().hex[:8]}"
            )
        self._manifest.update(
            status="failed",
            completed_at=_utc_now(),
            artifacts=self._artifact_inventory(),
            error={"type": type(error).__name__, "message": str(error)},
            diagnostic_run_dir=(
                str(diagnostic_root.relative_to(self.base_root))
                if diagnostic_root is not None
                else str(self.root.relative_to(self.base_root))
            ),
        )
        _atomic_write_json(self.manifest_path, self._manifest)
        if diagnostic_root is not None:
            diagnostic_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.root, diagnostic_root)
            self._set_working_root(diagnostic_root)
            self._transactional_overwrite = False
        return self.manifest_path

    def reset_logs(self) -> None:
        """Clear files only inside this unpublished run (legacy helper)."""
        if not self._prepared:
            raise RuntimeError("ensure_directories() must be called before reset_logs()")
        for directory in (self.images, self.videos, self.events, self.snapshots):
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()

    def _publish_overwrite(self) -> None:
        """Swap a completed staged run into place, rolling back on any failure."""
        staging_root = self.root
        backup_root = (
            self.base_root
            / ".transactions"
            / f"{self.run_id}-{uuid.uuid4().hex}.backup"
        )
        latest_path = self.base_root / "latest.json"
        previous_latest = latest_path.read_bytes() if latest_path.is_file() else None
        old_moved = False
        new_moved = False
        try:
            if not self.final_root.is_dir():
                raise FileNotFoundError(
                    f"Published run disappeared during overwrite: {self.final_root}"
                )
            os.replace(self.final_root, backup_root)
            old_moved = True
            os.replace(staging_root, self.final_root)
            new_moved = True
            _atomic_write_json(latest_path, self._latest_pointer())
        except BaseException:
            if new_moved and self.final_root.exists():
                os.replace(self.final_root, staging_root)
                new_moved = False
            if old_moved and backup_root.exists():
                os.replace(backup_root, self.final_root)
                old_moved = False
            self._restore_latest(latest_path, previous_latest)
            self._set_working_root(staging_root)
            raise

        self._set_working_root(self.final_root)
        self._transactional_overwrite = False
        if backup_root.exists():
            # Publication is already durable. A hidden backup is safer than
            # reporting failure and attempting to undo a successful commit.
            with suppress(OSError):
                shutil.rmtree(backup_root)

    def _restore_latest(self, path: Path, previous: bytes | None) -> None:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(path, previous)

    def _latest_pointer(self) -> dict[str, Any]:
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_dir": str(self.final_root.relative_to(self.base_root)),
            "manifest": str(
                (self.final_root / "manifest.json").relative_to(self.base_root)
            ),
            "updated_at": _utc_now(),
        }

    def _set_working_root(self, root: Path) -> None:
        self.root = root
        self.images = root / "images"
        self.videos = root / "videos"
        self.events = root / "events"
        self.snapshots = root / "snapshots"
        self.manifest_path = root / "manifest.json"

    def _artifact_inventory(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        artifacts: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path == self.manifest_path:
                continue
            artifacts.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
            )
        return artifacts

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
