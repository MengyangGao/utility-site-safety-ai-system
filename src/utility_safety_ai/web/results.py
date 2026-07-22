"""Evidence workspace, quality diagnostics, downloads, and human review."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from .theme import section


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _bundle_run(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return buffer.getvalue()


def _display_primary_artifact(run_dir: Path) -> None:
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    images = sorted((run_dir / "images").glob("*"))
    if videos:
        st.video(str(videos[0]))
    elif images:
        st.image(str(images[0]), width="stretch")


def _percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.0f}%"
    return "—"


def _quality_panel(run_dir: Path, quality_title: str) -> None:
    quality = _read_json(run_dir / "quality.json")
    if not quality:
        return
    section(quality_title, "Operational indicators · not ground-truth accuracy")
    st.markdown(
        f"""
        <div class="usi-quality">
          <div><strong>{_percent(quality.get('tracked_person_rate'))}</strong><small>Tracked person coverage</small></div>
          <div><strong>{_percent(quality.get('ppe_assignment_rate'))}</strong><small>Reliable PPE assignment</small></div>
          <div><strong>{quality.get('effective_fps', '—')}</strong><small>Effective frames / second</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quality diagnostics and interpretation"):
        st.json(quality)


def _review_workspace(run_dir: Path, events: pd.DataFrame, evidence_title: str) -> None:
    section(evidence_title, "Every alert remains a reviewable hypothesis")
    if events.empty:
        st.success("No confirmed safety event was emitted in this run.")
        return
    visible = [
        column
        for column in (
            "event_id",
            "time_seconds",
            "risk_level",
            "event_type",
            "description",
            "person_track_id",
            "zone_name",
        )
        if column in events.columns
    ]
    review = events[visible].copy()
    review["review_status"] = "unreviewed"
    review["operator_note"] = ""
    saved_path = run_dir / "operator_reviews.csv"
    saved = _read_csv(saved_path)
    if not saved.empty and "event_id" in saved and "event_id" in review:
        saved_by_id = saved.set_index("event_id")
        for column in ("review_status", "operator_note"):
            if column in saved_by_id:
                review[column] = review["event_id"].map(saved_by_id[column]).fillna(
                    review[column]
                )
    edited = st.data_editor(
        review,
        hide_index=True,
        width="stretch",
        disabled=list(visible),
        column_config={
            "review_status": st.column_config.SelectboxColumn(
                "Review decision",
                options=["unreviewed", "confirmed", "false_positive", "needs_follow_up"],
                required=True,
            ),
            "operator_note": st.column_config.TextColumn("Reviewer note"),
        },
        key=f"review_{run_dir.name}",
    )
    if st.button("Save review decisions", key=f"save_review_{run_dir.name}"):
        edited.to_csv(saved_path, index=False)
        st.success("Review decisions saved with this run.")

    snapshots = sorted((run_dir / "snapshots").glob("*.jpg"))
    if snapshots:
        columns = st.columns(min(3, len(snapshots)))
        for index, snapshot in enumerate(snapshots):
            columns[index % len(columns)].image(str(snapshot), caption=snapshot.stem[:12])


def render_run(run_dir: Path, *, quality_title: str, evidence_title: str) -> None:
    """Render one completed run as a review workspace."""
    if not run_dir.is_dir():
        st.warning("The selected run is no longer available.")
        return
    manifest = _read_json(run_dir / "manifest.json")
    metrics = manifest.get("metrics", {}) if isinstance(manifest.get("metrics"), dict) else {}
    events = _read_csv(run_dir / "events" / "events.csv")

    source = manifest.get("source", {})
    model = manifest.get("model", {})
    status = str(manifest.get("status", "unknown")).upper()
    st.caption(
        f"RUN {run_dir.name}  ·  {status}  ·  {source.get('type', 'source')}  ·  {model.get('model_kind', 'model')}"
    )
    cols = st.columns(4)
    cols[0].metric("Frames analysed", metrics.get("frames_processed", 0))
    cols[1].metric("Detections", metrics.get("detections", 0))
    cols[2].metric("Confirmed events", metrics.get("events", len(events)))
    critical = 0
    if not events.empty and "risk_level" in events:
        critical = int(events["risk_level"].isin(["critical", "high"]).sum())
    cols[3].metric("High / critical", critical)

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        _display_primary_artifact(run_dir)
    with right:
        st.markdown("**Run details**")
        st.caption(f"Model · {model.get('model_path') or 'runtime'}")
        model_hash = model.get("model_sha256")
        st.code(str(model_hash)[:20] + "…" if model_hash else "hash unavailable")
        capabilities = model.get("capabilities") or []
        st.caption(
            " · ".join(str(item).replace("_", " ") for item in capabilities)
            or "Model capabilities loading"
        )
        st.download_button(
            "Download complete report",
            _bundle_run(run_dir),
            file_name=f"utility-safety-{run_dir.name}.zip",
            mime="application/zip",
            key=f"bundle_{run_dir.name}",
            width="stretch",
        )

    _quality_panel(run_dir, quality_title)
    _review_workspace(run_dir, events, evidence_title)
    with st.expander("Run settings and file hashes"):
        st.json(manifest)


def render_history(history: list[dict[str, str]]) -> None:
    """Render compact session-local run history."""
    if not history:
        st.info("No completed runs in this private browser session yet.")
        return
    rows = []
    for item in history:
        run_dir = Path(item["path"])
        manifest = _read_json(run_dir / "manifest.json")
        metrics = manifest.get("metrics", {})
        rows.append(
            {
                "run": run_dir.name,
                "source": item.get("source", "source"),
                "status": manifest.get("status", "missing"),
                "frames": metrics.get("frames_processed", 0),
                "events": metrics.get("events", 0),
                "completed": manifest.get("completed_at", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
