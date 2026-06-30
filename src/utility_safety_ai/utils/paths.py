"""Output path helpers."""

from __future__ import annotations

from pathlib import Path


class OutputPaths:
    """Manage output subdirectories for a single inference run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images = self.root / "images"
        self.videos = self.root / "videos"
        self.events = self.root / "events"
        self.snapshots = self.root / "snapshots"

    def ensure_directories(self) -> None:
        for directory in (self.images, self.videos, self.events, self.snapshots):
            directory.mkdir(parents=True, exist_ok=True)

    def reset_logs(self) -> None:
        """Remove previous logs and snapshots so each run starts clean.

        Annotated images and videos are overwritten by name, but event snapshots
        use unique IDs and would otherwise accumulate across runs.
        """
        for name in (
            "events.jsonl",
            "events.csv",
            "detections.jsonl",
            "detections.csv",
            "compliance.jsonl",
            "compliance.csv",
        ):
            path = self.events / name
            if path.exists():
                path.unlink()

        for path in self.snapshots.iterdir():
            if path.is_file():
                path.unlink()
