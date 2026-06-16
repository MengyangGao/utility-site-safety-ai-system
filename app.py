"""Streamlit web demo for Utility Site Safety AI."""

from __future__ import annotations

import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import yaml

from utility_safety_ai.compliance.person_ppe_association import PPE_TYPES
from utility_safety_ai.detection.model_loader import resolve_model_path
from utility_safety_ai.detection.yolo_detector import YoloDetector
from utility_safety_ai.events.summary import aggregate_events
from utility_safety_ai.i18n import _, set_language
from utility_safety_ai.pipelines.camera_pipeline import run_camera_pipeline
from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline
from utility_safety_ai.pipelines.video_pipeline import run_video_pipeline
from utility_safety_ai.privacy.face_blur import blur_faces as _blur_faces
from utility_safety_ai.rules.rule_engine import RuleEngine
from utility_safety_ai.visualization.annotator import annotate_image
from utility_safety_ai.zones.zone import Zone

DEFAULT_MODEL = str(resolve_model_path(None))

REPO_ROOT = Path(__file__).resolve().parent
ZONE_PRESETS = {
    "Construction zone (construction_zone_01.jpg)": REPO_ROOT / "examples" / "zones_construction_zone_01.yaml",
    "Solar farm (solar_farm_01.jpg)": REPO_ROOT / "examples" / "zones_solar_farm_01.yaml",
    "Construction site PPE (construction_site_ppe_01.jpg)": REPO_ROOT / "examples" / "zones_construction_site_ppe_01.yaml",
    "Full-PPE reference (no zones)": REPO_ROOT / "examples" / "zones_electrical_engineer_01.yaml",
    "Custom": None,
}

DEFAULT_ZONE_TEXT = (ZONE_PRESETS["Construction zone (construction_zone_01.jpg)"]).read_text()

LANG_OPTIONS = {
    "English": "en",
    "简体中文": "zh-hans",
    "繁體中文": "zh-hant",
}


def _load_preset_zones(preset_name: str) -> str:
    path = ZONE_PRESETS[preset_name]
    if path is None or not path.exists():
        return "zones:\n  []\n"
    return path.read_text()


def _parse_zones(text: str) -> list[Zone]:
    data = yaml.safe_load(text) or {}
    zones: list[Zone] = []
    for item in data.get("zones", []):
        polygon = [tuple(float(v) for v in point) for point in item["polygon"]]
        zones.append(
            Zone(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                risk_level=str(item.get("risk_level", "high")),
                polygon=polygon,
            )
        )
    return zones


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def _risk_badge(level: str) -> str:
    emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(level, "⚪")
    return f"{emoji} {_(level).upper()}"


def _status_badge(status: str) -> str:
    return {
        "yes": f"✅ {_('compliance_yes')}",
        "no": f"❌ {_('compliance_no')}",
        "unknown": f"❓ {_('compliance_unknown')}",
    }.get(status, status)


def _zip_reports(events_dir: Path) -> bytes:
    """Return a ZIP archive containing all report files in ``events_dir``."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in events_dir.iterdir():
            if path.is_file():
                zf.write(path, path.name)
    buffer.seek(0)
    return buffer.getvalue()


def _build_compliance_table(events_dir: Path) -> pd.DataFrame | None:
    compliance_csv = events_dir / "compliance.csv"
    if not compliance_csv.exists():
        return None
    df = pd.read_csv(compliance_csv)
    if df.empty:
        return None
    # Keep the latest record per person_track_id.
    df = df.sort_values(by=["person_track_id", "frame_index"]).drop_duplicates(
        subset=["person_track_id"], keep="last"
    )
    return df


def _inject_custom_css() -> None:
    """Apply a professional dark/industrial theme to the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stSlider > div > div {
            color: #93c5fd !important;
        }
        .title-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #0ea5e9 100%);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            color: white;
            box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.25);
        }
        .title-card h1 { margin: 0; font-weight: 700; font-size: 1.6rem; }
        .title-card p { margin: 0.25rem 0 0; opacity: 0.9; font-size: 0.9rem; }
        div[data-testid="stTabs"] button[role="tab"] {
            background-color: #1e293b;
            color: #cbd5e1;
            border-radius: 10px 10px 0 0;
            margin-right: 6px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%);
            color: white;
        }
        div[data-baseweb="tab-highlight"] { background: transparent !important; }
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.3);
        }
        [data-testid="stMetric"] {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }
        [data-testid="stMetric"] label { color: #94a3b8; font-weight: 600; }
        [data-testid="stMetric"] div { color: #f8fafc; font-weight: 700; font-size: 1.6rem; }
        .info-card {
            background: #0f172a;
            border-left: 4px solid #0ea5e9;
            border-radius: 0 12px 12px 0;
            padding: 1rem;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_detector(model_path: str, device: str | None, conf: float, iou: float) -> YoloDetector:
    """Cache the heavy YOLO detector across Streamlit reruns."""
    return YoloDetector(model_path=model_path, device=device, conf=conf, iou=iou)


def _stop_preview() -> None:
    """Callback that requests the live preview loop to stop."""
    st.session_state.preview_running = False


def _release_preview() -> None:
    """Release the camera and clear preview session state."""
    cap = st.session_state.get("preview_cap")
    if cap is not None:
        cap.release()
    st.session_state.preview_running = False
    st.session_state.preview_frame = 0
    for key in ("preview_cap", "preview_source", "preview_max_frames"):
        st.session_state.pop(key, None)


def _run_live_preview(
    model_path: str,
    device: str | None,
    conf: float,
    iou: float,
    blur_faces: bool,
    cooldown: float,
    zones: list[Zone],
) -> None:
    """Process one camera frame and trigger a rerun for the next frame."""
    st.subheader("🎥 " + _("ui.preview_active"))
    st.button("⏹️ " + _("ui.stop_preview"), on_click=_stop_preview, key="btn_stop_preview")
    placeholder = st.empty()

    source = st.session_state.get("preview_source", "0")
    max_frames = st.session_state.get("preview_max_frames", 60)

    cap = st.session_state.get("preview_cap")
    if cap is None or not cap.isOpened():
        cap_source = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        st.session_state.preview_cap = cap

    if not cap.isOpened():
        st.error(_("ui.camera_open_error"))
        _release_preview()
        return

    detector = _get_detector(model_path, device, conf, iou)
    engine = RuleEngine(zones=zones, cooldown_seconds=cooldown)

    ret, frame = cap.read()
    if not ret:
        st.warning(_("ui.camera_ended"))
        _release_preview()
        return

    frame_index = st.session_state.get("preview_frame", 0)
    time_seconds = frame_index / 30.0
    detections = detector(frame)
    if blur_faces:
        frame = _blur_faces(frame, detections, enabled=True)
    events = engine.evaluate(
        detections,
        source_type="camera",
        source_path=str(source),
        frame_index=frame_index,
        time_seconds=time_seconds,
    )
    annotated = annotate_image(frame, zones, detections, events)
    placeholder.image(annotated, channels="BGR", use_container_width=True)

    st.session_state.preview_frame = frame_index + 1
    if st.session_state.preview_frame >= max_frames:
        st.success(_("ui.preview_complete"))
        _release_preview()
        return

    time.sleep(0.03)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=_("ui.title"),
        page_icon="🦺",
        layout="wide",
    )
    _inject_custom_css()

    st.session_state.setdefault("preview_running", False)
    st.session_state.setdefault("preview_frame", 0)

    with st.sidebar:
        lang_label = st.selectbox(
            "🌐 Language / 语言 / 語言",
            list(LANG_OPTIONS.keys()),
            index=0,
            key="language_selector",
        )
        set_language(LANG_OPTIONS[lang_label])

        st.header("⚙️ " + _("ui.sidebar_settings"))
        model_path = st.text_input(_("ui.model_path"), value=DEFAULT_MODEL)
        col1, col2 = st.columns(2)
        with col1:
            conf = st.slider(_("ui.confidence"), 0.0, 1.0, 0.25, 0.05)
        with col2:
            iou = st.slider(_("ui.nms_iou"), 0.0, 1.0, 0.45, 0.05)
        device = st.selectbox(_("ui.device"), ["auto", "cpu", "mps", "cuda"], index=0)
        blur_faces = st.toggle(
            _("ui.blur_faces"),
            value=False,
            help=_("ui.blur_faces_help"),
        )
        cooldown = st.slider(_("ui.cooldown"), 0, 60, 10, 1)
        st.caption(_("ui.confidence_floors_caption"))

        st.markdown("---")
        st.header("🚧 " + _("ui.zones_config"))
        preset = st.selectbox(_("ui.zone_preset"), list(ZONE_PRESETS.keys()))
        zones_yaml = st.text_area(
            _("ui.zone_editor"),
            value=_load_preset_zones(preset),
            height=280,
        )
        st.markdown("_" + _("ui.zone_tip") + "_")

    st.markdown(
        f"""
        <div class="title-card">
            <h1>🦺 {_('ui.title')}</h1>
            <p>{_('ui.page_caption')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    zones = _parse_zones(zones_yaml)

    if st.session_state.preview_running:
        _run_live_preview(
            model_path=model_path,
            device=None if device == "auto" else device,
            conf=conf,
            iou=iou,
            blur_faces=blur_faces,
            cooldown=cooldown,
            zones=zones,
        )
        return

    source_tab, camera_tab, rtsp_tab, live_tab = st.tabs(
        [
            "📁 " + _("ui.upload_image") + " / " + _("ui.upload_video"),
            "📷 " + _("ui.camera_tab"),
            "🎥 " + _("ui.camera_source"),
            "📡 " + _("ui.live_preview"),
        ]
    )

    output_root = REPO_ROOT / "outputs" / "web_demo"
    events: list = []
    source_type = "upload"

    with source_tab:
        uploaded_file = st.file_uploader(
            _("ui.upload_image") + " / " + _("ui.upload_video"),
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )
        run_upload = st.button(
            "🚀 " + _("ui.run_inference"), type="primary", use_container_width=True
        )

    with camera_tab:
        camera_image = st.camera_input(_("ui.camera_tab"))
        run_camera = st.button(
            "🚀 " + _("ui.run_inference") + " (" + _("ui.camera_tab") + ")",
            type="primary",
            use_container_width=True,
        )

    with rtsp_tab:
        rtsp_source = st.text_input(
            _("ui.camera_source"),
            value="0",
            help=_("ui.camera_source_help"),
            key="rtsp_source",
        )
        rtsp_frames = st.slider(_("ui.live_frames"), 1, 300, 60, 10, key="rtsp_frames")
        run_rtsp = st.button(
            "🚀 " + _("ui.run_inference") + " (" + _("ui.camera_source") + ")",
            type="primary",
            use_container_width=True,
        )

    with live_tab:
        live_source = st.text_input(
            _("ui.camera_source"),
            value="0",
            help=_("ui.camera_source_help"),
            key="live_source_input",
        )
        live_max_frames = st.slider(
            _("ui.live_frames"), 1, 300, 60, 10, key="live_max_frames"
        )
        start_preview = st.button(
            "▶️ " + _("ui.start_preview"), type="primary", use_container_width=True
        )
        if start_preview:
            st.session_state.preview_running = True
            st.session_state.preview_source = live_source
            st.session_state.preview_max_frames = live_max_frames
            st.session_state.preview_frame = 0
            _release_preview()
            st.rerun()

    run_button = run_upload or run_camera or run_rtsp
    source_path: Path | str | None = None
    is_video = False
    annotated = None

    if run_upload and uploaded_file is not None:
        source_path = _save_upload(uploaded_file)
        is_video = source_path.suffix.lower() in (".mp4", ".avi", ".mov")
        source_type = "upload"
    elif run_camera and camera_image is not None:
        source_path = _save_upload(camera_image)
        source_type = "camera"
    elif run_rtsp:
        source_path = rtsp_source
        is_video = True
        source_type = "rtsp"
    else:
        st.markdown(
            f'<div class="info-card">{_("ui.start_prompt")}</div>',
            unsafe_allow_html=True,
        )
        return

    if run_button:
        with st.spinner(_("ui.loading")):
            detector = _get_detector(
                model_path, None if device == "auto" else device, conf, iou
            )
            engine = RuleEngine(zones=zones, cooldown_seconds=cooldown)

            if source_type == "rtsp":
                events = run_camera_pipeline(
                    source=str(source_path),
                    output_root=output_root,
                    detector=detector,
                    zones=zones,
                    rule_engine=engine,
                    blur_faces_enabled=blur_faces,
                    max_frames=rtsp_frames,
                )
            elif is_video:
                events = run_video_pipeline(
                    source_path=source_path,
                    output_root=output_root,
                    detector=detector,
                    zones=zones,
                    rule_engine=engine,
                    blur_faces_enabled=blur_faces,
                    max_frames=300,
                )
            else:
                annotated, events = run_image_pipeline(
                    source_path=source_path,
                    output_root=output_root,
                    detector=detector,
                    zones=zones,
                    rule_engine=engine,
                    blur_faces_enabled=blur_faces,
                )

        summary = aggregate_events(events)
        st.success(
            f"{_('ui.inference_complete')}: {summary['total_events']} {_('ui.total_events').lower()}, "
            f"{summary['zone_intrusions']} {_('ui.zone_intrusions').lower()}, "
            f"{summary['ppe_violations']} {_('ui.ppe_violations').lower()}."
        )

        tab_result, tab_events, tab_detections, tab_compliance, tab_summary, tab_reports = st.tabs(
            [
                "🖼️ " + _("ui.result_tab"),
                "📋 " + _("ui.events_title"),
                "🔍 " + _("ui.detections_tab"),
                "👷 " + _("ui.compliance_tab"),
                "📊 " + _("ui.summary_tab"),
                "📥 " + _("ui.reports_tab"),
            ]
        )

        with tab_result:
            if source_type == "rtsp":
                out_video = output_root / "videos" / "camera_output.mp4"
                if out_video.exists():
                    st.video(str(out_video))
                else:
                    st.warning(_("ui.annotated_clip_not_found"))
            elif is_video:
                out_video = output_root / "videos" / Path(source_path).name
                if out_video.exists():
                    st.video(str(out_video))
                else:
                    st.warning(_("ui.annotated_video_not_found"))
            else:
                st.image(annotated, channels="BGR", use_container_width=True)

        with tab_events:
            if events:
                rows = []
                for e in events:
                    rows.append(
                        {
                            "risk": _risk_badge(e.risk_level),
                            "type": _(e.event_type).replace("_", " "),
                            "description": e.description,
                            "time (s)": e.time_seconds,
                            "zone": e.zone_name or "—",
                            "track": e.person_track_id,
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.info(_("ui.no_events"))

        with tab_detections:
            detections_csv = output_root / "events" / "detections.csv"
            if detections_csv.exists():
                df_det = pd.read_csv(detections_csv)
                st.subheader(_("ui.detected_objects"))
                st.bar_chart(df_det["class_name"].value_counts())
                st.dataframe(
                    df_det[["class_name", "confidence", "bbox", "track_id"]],
                    use_container_width=True,
                )
            else:
                st.info(_("ui.no_detection_log"))

        with tab_compliance:
            compliance_df = _build_compliance_table(output_root / "events")
            if compliance_df is not None:
                display = compliance_df.copy()
                for col in PPE_TYPES:
                    if col in display.columns:
                        display[col] = display[col].apply(_status_badge)
                st.subheader(_("ui.per_person_compliance"))
                st.dataframe(
                    display[["person_track_id", "frame_index", "time_seconds", *PPE_TYPES, "violations"]],
                    use_container_width=True,
                )
            else:
                st.info(_("ui.no_compliance_records"))

        with tab_summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(_("ui.total_events"), summary["total_events"])
            c2.metric(_("ui.zone_intrusions"), summary["zone_intrusions"])
            c3.metric(_("ui.ppe_violations"), summary["ppe_violations"])
            c4.metric(_("ui.unique_persons"), summary["unique_persons"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(_("ui.by_risk_level"))
                st.json(summary["by_risk_level"])
            with col_b:
                st.subheader(_("ui.by_event_type"))
                st.json(summary["by_event_type"])

        with tab_reports:
            events_dir = output_root / "events"
            files = {
                "events CSV": events_dir / "events.csv",
                "events JSONL": events_dir / "events.jsonl",
                "summary JSON": events_dir / "summary.json",
                "detections CSV": events_dir / "detections.csv",
                "detections JSONL": events_dir / "detections.jsonl",
                "compliance CSV": events_dir / "compliance.csv",
                "compliance JSONL": events_dir / "compliance.jsonl",
            }
            for label, path in files.items():
                if path.exists():
                    st.download_button(
                        f"{_('ui.download')} {label}",
                        path.read_bytes(),
                        file_name=path.name,
                        mime="text/csv" if path.suffix == ".csv" else "application/jsonlines+json" if path.suffix == ".jsonl" else "application/json",
                    )

            st.markdown("---")
            st.download_button(
                "📦 " + _("ui.download_all_reports"),
                _zip_reports(events_dir),
                file_name="safety_reports.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
