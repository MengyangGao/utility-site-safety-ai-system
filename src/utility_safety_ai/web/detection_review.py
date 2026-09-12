"""A paginated prediction review surface, including runs with no policy events."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from ..events.detection_review import PAGE_SIZE, detection_page, latest_reviews, save_reviews
from ..events.store import REVIEW_STATUSES, EventStore
from .theme import section


def render_detection_review(run_dir: Path) -> None:
    section(
        "Detection review", "Inspect individual predictions, including those that emitted no event"
    )
    try:
        first = detection_page(run_dir)
        if not first["total"]:
            st.info("No detections were recorded. This does not establish that the site is safe.")
            return
        pages = (first["total"] + PAGE_SIZE - 1) // PAGE_SIZE
        page = (
            int(
                st.number_input(
                    "Detection page",
                    min_value=1,
                    max_value=pages,
                    value=1,
                    step=1,
                    key=f"detection_page_{run_dir.name}_{first['log_sha256']}",
                )
            )
            - 1
            if pages > 1
            else 0
        )
        evidence = first if page == 0 else detection_page(run_dir, page)
        store = EventStore(run_dir.parent.parent / "events.sqlite3")
        latest = latest_reviews(store, run_dir.name, evidence["log_sha256"])
        counts = Counter(row["status"] for row in latest.values())
        cols = st.columns(4)
        cols[0].metric("Raw predictions", first["total"])
        cols[1].metric("Reviewer confirmed", counts["confirmed"])
        cols[2].metric("Marked false positive", counts["false_positive"])
        cols[3].metric("Needs follow-up", counts["needs_follow_up"])
        st.caption(
            "Model confidence is not verified accuracy. Review decisions do not change original boxes, policy events or delivered alerts."
        )
        frame = pd.DataFrame(
            [
                {
                    **row,
                    "bbox": ", ".join(str(round(v, 1)) for v in row["bbox"])
                    if isinstance(row["bbox"], list)
                    else str(row["bbox"]),
                    "review_status": latest.get(row["detection_id"], {}).get(
                        "status", "unreviewed"
                    ),
                    "operator_note": latest.get(row["detection_id"], {}).get("note", ""),
                }
                for row in evidence["rows"]
            ]
        )
        if frame.empty:
            return
        edited = st.data_editor(
            frame,
            hide_index=True,
            width="stretch",
            disabled=list(evidence["rows"][0]),
            column_config={
                "detection_id": None,
                "line_number": st.column_config.NumberColumn("Record"),
                "frame_index": st.column_config.NumberColumn("Frame")
                if frame["frame_index"].notna().any()
                else None,
                "time_seconds": st.column_config.NumberColumn("Time (s)", format="%.2f")
                if frame["time_seconds"].notna().any()
                else None,
                "bbox": st.column_config.TextColumn(
                    "Box coordinates", help="Left, top, right, bottom in source-image pixels."
                ),
                "track_id": st.column_config.NumberColumn("Track"),
                "class_name": st.column_config.TextColumn("Model label"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                "review_status": st.column_config.SelectboxColumn(
                    "Review decision", options=list(REVIEW_STATUSES), required=True
                ),
                "operator_note": st.column_config.TextColumn("Reviewer note", max_chars=2000),
            },
            key=f"detection_review_{run_dir.name}_{evidence['log_sha256']}_{page}",
        )
        with st.expander("Save prediction reviews"):
            reviewer = st.text_input(
                "Detection reviewer label",
                value="Local reviewer",
                max_chars=100,
                key=f"detection_reviewer_{run_dir.name}",
            )
            if st.button(
                "Save detection reviews",
                disabled=not reviewer.strip(),
                key=f"save_detection_{run_dir.name}",
            ):
                decisions = [
                    {
                        "detection_id": row["detection_id"],
                        "status": row["review_status"],
                        "note": row["operator_note"],
                    }
                    for index, row in edited.fillna("").iterrows()
                    if row["review_status"] != frame.loc[index, "review_status"]
                    or row["operator_note"] != frame.loc[index, "operator_note"]
                ]
                saved = save_reviews(
                    store, run_dir, page, evidence["log_sha256"], decisions, reviewer
                )
                st.session_state[f"detection_saved_{run_dir.name}"] = (
                    f"{saved} prediction review(s) appended. Original evidence preserved."
                )
                st.rerun()
        message = st.session_state.pop(f"detection_saved_{run_dir.name}", None)
        if message:
            st.success(message)
        st.caption(
            f"Records {page * PAGE_SIZE + 1}–{min((page + 1) * PAGE_SIZE, first['total'])} of {first['total']} · Save decisions before changing pages. The report includes prediction review history."
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"Detection review unavailable: {error}")
