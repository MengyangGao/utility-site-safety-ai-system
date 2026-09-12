"""Evidence workspace, quality diagnostics, downloads, and human review."""

from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ..events.detection_review import review_history as detection_review_history
from ..events.event import SafetyEvent
from ..events.store import EventStore
from .detection_review import render_detection_review
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
            if (
                path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(run_dir.resolve())
            ):
                archive.write(path, path.relative_to(run_dir))
        store = EventStore(run_dir.parent.parent / "events.sqlite3")
        archive.writestr(
            "review_history.json", json.dumps(store.run_reviews(run_dir.name), indent=2)
        )
        archive.writestr(
            "detection_review_history.json",
            json.dumps(detection_review_history(store, run_dir.name), indent=2),
        )
    return buffer.getvalue()


def _display_primary_artifact(run_dir: Path) -> None:
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    images = sorted((run_dir / "images").glob("*"))
    if videos:
        encoding = (
            _read_json(run_dir / "manifest.json").get("metrics", {}).get("video_encoding", {})
        )
        if encoding.get("browser_compatible") is False:
            st.warning(
                "Install FFmpeg and rerun to create browser-playable video. This MPEG-4 recording remains available in the report ZIP for desktop players."
            )
        else:
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
          <div><strong>{_percent(quality.get("tracked_person_rate"))}</strong><small>Tracked person coverage</small></div>
          <div><strong>{_percent(quality.get("ppe_assignment_rate"))}</strong><small>PPE association coverage</small></div>
          <div><strong>{quality.get("effective_fps", "—")}</strong><small>Effective frames / second</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quality diagnostics and interpretation"):
        st.json(quality)


def _review_workspace(run_dir: Path, events: pd.DataFrame, evidence_title: str) -> None:
    section(evidence_title, "Confirm a finding or record a false positive")
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
    store = EventStore(run_dir.parent.parent / "events.sqlite3")
    # Import older runs into the integration index without rewriting their evidence.
    log_path = run_dir / "events" / "events.jsonl"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("schema_version", None)
            record.setdefault("metadata", {}).setdefault("run_id", run_dir.name)
            store.append(SafetyEvent(**record))
    history = store.run_reviews(run_dir.name)
    saved = pd.DataFrame(history).rename(
        columns={"status": "review_status", "note": "operator_note"}
    )
    if not saved.empty:
        saved = saved.drop_duplicates("event_id", keep="last")
    else:
        saved = _read_csv(run_dir / "operator_reviews.csv")
    if not saved.empty and "event_id" in saved and "event_id" in review:
        saved_by_id = saved.set_index("event_id")
        for column in ("review_status", "operator_note"):
            if column in saved_by_id:
                review[column] = review["event_id"].map(saved_by_id[column]).fillna(review[column])
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
    reviewer = st.text_input(
        "Reviewer label", value="Local reviewer", key=f"reviewer_{run_dir.name}", max_chars=100
    )
    if st.button(
        "Save review decisions", key=f"save_review_{run_dir.name}", disabled=not reviewer.strip()
    ):
        for index, row in edited.fillna("").iterrows():
            if (
                row["review_status"] != review.loc[index, "review_status"]
                or row["operator_note"] != review.loc[index, "operator_note"]
            ):
                store.review(
                    str(row["event_id"]),
                    str(row["review_status"]),
                    str(row["operator_note"]),
                    reviewer,
                )
        st.success("Review history appended. Original evidence remains unchanged.")

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
    config = manifest.get("config", {})
    source_name = Path(str(source.get("value", "Inspection"))).name
    status = str(manifest.get("status", "unknown")).replace("_", " ").title()
    st.markdown(
        f'<div class="usi-run-heading"><div><h2>Inspection result</h2><p>{html.escape(source_name)}</p></div><span>{html.escape(status)}</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Frames analysed", metrics.get("frames_processed", 0))
    cols[1].metric("Detections", metrics.get("detections", 0))
    cols[2].metric("Confirmed events", metrics.get("events", len(events)))
    critical = 0
    if not events.empty and "risk_level" in events:
        critical = int(events["risk_level"].isin(["critical", "high"]).sum())
    cols[3].metric("High / critical", critical)

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        _display_primary_artifact(run_dir)
        st.caption("Recorded evidence · detections and policy observations require human review.")
    with right:
        with st.container(border=True):
            st.markdown("#### Observations")
            if events.empty:
                st.write("No confirmed event in this run.")
            elif "event_type" in events:
                for kind, count in events["event_type"].value_counts().items():
                    st.markdown(
                        f'<div class="usi-summary-row"><span>{html.escape(str(kind).replace("_", " ").title())}</span><b>{int(count)}</b></div>',
                        unsafe_allow_html=True,
                    )
            st.caption("Event counts describe this run, not an overall site-safety rating.")
            reason = str(config.get("privacy_reason", ""))
            if reason.startswith("reviewed_rear_view_sample:") and not config.get(
                "privacy_blur_enabled"
            ):
                privacy_label = "Reviewed rear view · no visible faces"
            else:
                privacy_label = (
                    "Privacy redaction on"
                    if config.get("privacy_blur_enabled")
                    else "Privacy redaction off"
                )
            st.markdown(f"**{privacy_label}**")
            st.download_button(
                "Download complete report",
                _bundle_run(run_dir),
                file_name=f"utility-safety-{run_dir.name}.zip",
                mime="application/zip",
                key=f"bundle_{run_dir.name}",
                width="stretch",
                type="primary",
            )
        with st.expander("Model and provenance"):
            st.caption(f"Model · {Path(str(model.get('model_path') or 'runtime')).name}")
            model_hash = model.get("model_sha256")
            st.code(str(model_hash) if model_hash else "hash unavailable")
            capabilities = model.get("capabilities") or []
            st.caption(" · ".join(str(item).replace("_", " ") for item in capabilities))
            st.caption(f"Run · {run_dir.name}")

    capture = metrics.get("capture")
    if capture:
        with st.expander("Stream continuity and capture health", expanded=True):
            health_cols = st.columns(3)
            health_cols[0].metric("Reconnections", capture.get("reconnections", 0))
            health_cols[1].metric("Dropped backlog frames", capture.get("frames_dropped", 0))
            health_cols[2].metric(
                "Stale frames discarded", capture.get("stale_frames_discarded", 0)
            )
            st.caption(
                f"Stop reason: {capture.get('stop_reason', 'unknown')}. MP4 timing may be compressed; use the timestamp log for capture timing."
            )
    _quality_panel(run_dir, quality_title)
    render_detection_review(run_dir)
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
    selected = st.selectbox(
        "Open a previous run",
        range(len(history)),
        format_func=lambda index: (
            f"{history[index]['source']} · {Path(history[index]['path']).name}"
        ),
    )
    if st.button("Open selected run"):
        st.session_state.selected_run = history[selected]["path"]
        st.session_state.next_workspace_view = "results"
        st.rerun()
