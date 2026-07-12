"""Session-isolated Streamlit product demo for Utility Site Safety AI."""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utility_safety_ai.compliance.person_ppe_association import PPE_TYPES
from utility_safety_ai.detection.model_loader import resolve_model_path
from utility_safety_ai.detection.yolo_detector import YoloDetector
from utility_safety_ai.events.summary import aggregate_events
from utility_safety_ai.i18n import _
from utility_safety_ai.pipelines.camera_pipeline import run_camera_pipeline
from utility_safety_ai.pipelines.image_pipeline import run_image_pipeline
from utility_safety_ai.pipelines.video_pipeline import run_video_pipeline
from utility_safety_ai.privacy.face_blur import blur_faces as _blur_faces
from utility_safety_ai.rules.rule_engine import RuleEngine
from utility_safety_ai.tracking.simple_tracker import SimpleTracker
from utility_safety_ai.utils.paths import resolve_latest_run
from utility_safety_ai.visualization.annotator import annotate_image
from utility_safety_ai.web_helpers import (
    RISK_LEVELS,
    NormalizedZone,
    RunContext,
    ZoneValidationError,
    draw_zone_preview,
    inspect_detector,
    new_session_id,
    parse_zone_yaml,
    redact_uri_credentials,
    resolve_run_artifact,
    sanitize_run_artifacts,
    temporary_upload,
    utc_run_label,
    zones_from_rows,
    zones_to_domain,
    zones_to_rows,
    zones_to_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent
_resolved_default_model = Path(resolve_model_path(None))
try:
    DEFAULT_MODEL = str(_resolved_default_model.resolve().relative_to(REPO_ROOT))
except ValueError:
    DEFAULT_MODEL = str(_resolved_default_model)

ZONE_PRESETS: dict[str, tuple[Path | None, Path | None]] = {
    "Construction zone": (
        REPO_ROOT / "examples" / "zones_construction_zone_01.yaml",
        REPO_ROOT / "examples" / "sample_images" / "construction_zone_01.jpg",
    ),
    "Solar inspection (verified Pexels asset)": (
        REPO_ROOT / "examples" / "zones_solar_inspection_pexels_4254172.yaml",
        REPO_ROOT / "examples" / "sample_images" / "solar_inspection_pexels_4254172.jpg",
    ),
    "Construction PPE": (
        REPO_ROOT / "examples" / "zones_construction_site_ppe_01.yaml",
        REPO_ROOT / "examples" / "sample_images" / "construction_site_ppe_01.jpg",
    ),
    "No zones (verified Pexels solar asset)": (
        None,
        REPO_ROOT / "examples" / "sample_images" / "solar_inspection_pexels_4254172.jpg",
    ),
    "Custom": (None, None),
}

LANG_OPTIONS = {"English": "en", "简体中文": "zh-hans", "繁體中文": "zh-hant"}

WEB_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "zone_mode": "Zone editor mode",
        "table": "Visual table",
        "yaml": "Normalized YAML",
        "zone_help": "Coordinates are normalized from 0 to 1 and scale safely to every source resolution.",
        "polygon_help": "Polygon format: x,y; x,y; x,y (all values 0–1)",
        "zone_preview": "Live zone preview",
        "invalid_zones": "Invalid zone configuration",
        "download_zones": "Download normalized zone YAML",
        "model_audit": "Active model and capability",
        "model_pending": "The model is loaded only when a run starts.",
        "classes": "Classes",
        "capabilities": "Capabilities",
        "model_hash": "SHA-256",
        "generic_model": "This model has no PPE classes. The run supports person and zone detection only.",
        "ppe_model": "PPE-capable model detected.",
        "privacy_default": "Privacy blur defaults to ON. Disable it only for controlled test material.",
        "camera_source": "Camera / RTSP source",
        "credentials": "Credentials are masked in the UI, run history, manifests, and downloadable text reports.",
        "run_failed": "Inference failed",
        "source_probe_failed": "Could not determine the source resolution.",
        "run_history": "Run history",
        "no_history": "No completed runs in this browser session yet.",
        "event_filter": "Event type filter",
        "risk_filter": "Risk filter",
        "evidence": "Evidence snapshots",
        "no_evidence": "No evidence snapshots match the current filters.",
        "unknown_notice": "Unknown means the PPE item was not observed; it is not counted as compliant.",
        "model_tab": "Model & zones",
        "run_source": "Source",
        "run_time": "Completed",
        "session_output": "Private session output",
        "preview_events": "New events in this preview",
        "preview_findings": "Active findings",
        "manifest": "Run manifest",
        "history_limit": "The latest 20 runs are kept in this browser session.",
        "start_blocked": "Fix the zone configuration before starting inference.",
    },
    "zh-hans": {
        "zone_mode": "区域编辑方式",
        "table": "可视化表格",
        "yaml": "归一化 YAML",
        "zone_help": "坐标统一使用 0 到 1 的归一化值，可安全适配不同分辨率。",
        "polygon_help": "多边形格式：x,y; x,y; x,y（全部数值为 0–1）",
        "zone_preview": "区域实时预览",
        "invalid_zones": "区域配置无效",
        "download_zones": "下载归一化区域 YAML",
        "model_audit": "当前模型与能力",
        "model_pending": "模型仅在开始运行时加载。",
        "classes": "类别",
        "capabilities": "能力",
        "model_hash": "SHA-256",
        "generic_model": "该模型不含 PPE 类别，本次运行仅支持人员与区域入侵检测。",
        "ppe_model": "已识别支持 PPE 的模型。",
        "privacy_default": "隐私模糊默认开启。仅应在受控测试素材上关闭。",
        "camera_source": "摄像头 / RTSP 源",
        "credentials": "凭据会在界面、运行历史、清单和可下载文本报告中脱敏。",
        "run_failed": "推理失败",
        "source_probe_failed": "无法确定输入源分辨率。",
        "run_history": "运行历史",
        "no_history": "本浏览器会话尚无已完成运行。",
        "event_filter": "事件类型筛选",
        "risk_filter": "风险等级筛选",
        "evidence": "证据快照",
        "no_evidence": "当前筛选条件下没有证据快照。",
        "unknown_notice": "“未知”表示未观察到该 PPE，不能视为合规。",
        "model_tab": "模型与区域",
        "run_source": "来源",
        "run_time": "完成时间",
        "session_output": "独立会话输出",
        "preview_events": "本次预览新增事件",
        "preview_findings": "当前有效风险",
        "manifest": "运行清单",
        "history_limit": "本浏览器会话保留最近 20 次运行。",
        "start_blocked": "请先修复区域配置再开始推理。",
    },
    "zh-hant": {
        "zone_mode": "區域編輯方式",
        "table": "視覺化表格",
        "yaml": "正規化 YAML",
        "zone_help": "座標統一使用 0 到 1 的正規化值，可安全適配不同解析度。",
        "polygon_help": "多邊形格式：x,y; x,y; x,y（全部數值為 0–1）",
        "zone_preview": "區域即時預覽",
        "invalid_zones": "區域設定無效",
        "download_zones": "下載正規化區域 YAML",
        "model_audit": "目前模型與能力",
        "model_pending": "模型只會在開始執行時載入。",
        "classes": "類別",
        "capabilities": "能力",
        "model_hash": "SHA-256",
        "generic_model": "此模型不含 PPE 類別，本次執行只支援人員與區域入侵偵測。",
        "ppe_model": "已識別支援 PPE 的模型。",
        "privacy_default": "隱私模糊預設開啟。只應在受控測試素材上關閉。",
        "camera_source": "攝影機 / RTSP 來源",
        "credentials": "憑據會在介面、執行歷史、清單及可下載文字報告中遮罩。",
        "run_failed": "推理失敗",
        "source_probe_failed": "無法確定輸入來源解析度。",
        "run_history": "執行歷史",
        "no_history": "本瀏覽器工作階段尚無已完成執行。",
        "event_filter": "事件類型篩選",
        "risk_filter": "風險等級篩選",
        "evidence": "證據快照",
        "no_evidence": "目前篩選條件下沒有證據快照。",
        "unknown_notice": "「未知」表示未觀察到該 PPE，不能視為合規。",
        "model_tab": "模型與區域",
        "run_source": "來源",
        "run_time": "完成時間",
        "session_output": "獨立工作階段輸出",
        "preview_events": "本次預覽新增事件",
        "preview_findings": "目前有效風險",
        "manifest": "執行清單",
        "history_limit": "本瀏覽器工作階段保留最近 20 次執行。",
        "start_blocked": "請先修正區域設定再開始推理。",
    },
}


def _t(key: str) -> str:
    lang = st.session_state.get("ui_language", "en")
    return WEB_TEXT.get(lang, WEB_TEXT["en"]).get(key, WEB_TEXT["en"].get(key, key))


def _tr(key: str) -> str:
    """Use the session-selected language instead of relying on the process default."""

    return _(key, lang=st.session_state.get("ui_language", "en"))


def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1220,#111827); }
        [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown { color:#e2e8f0!important; }
        .title-card { background:linear-gradient(90deg,#1e3a8a,#0ea5e9);border-radius:16px;
          padding:1.25rem 1.5rem;margin-bottom:1rem;color:white;
          box-shadow:0 10px 25px -5px rgba(14,165,233,.25); }
        .title-card h1 { margin:0;font-weight:700;font-size:1.65rem; }
        .title-card p { margin:.3rem 0 0;opacity:.92;font-size:.9rem; }
        .info-card { background:#0f172a;border-left:4px solid #0ea5e9;border-radius:0 12px 12px 0;
          padding:1rem;color:#e2e8f0; }
        [data-testid="stMetric"] { background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1rem; }
        .stButton>button { border-radius:10px;font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session() -> None:
    st.session_state.setdefault("web_session_id", new_session_id())
    output_root = REPO_ROOT / "outputs" / "web_demo" / "sessions" / st.session_state.web_session_id
    output_root.mkdir(parents=True, exist_ok=True)
    st.session_state.setdefault("web_output_root", output_root)
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("preview_running", False)
    st.session_state.setdefault("preview_frame", 0)
    st.session_state.setdefault("zone_preset_last", None)
    st.session_state.setdefault("zone_mode_last", None)
    st.session_state.setdefault("zone_editor_revision", 0)
    st.session_state.setdefault("last_model_info", None)
    st.session_state.setdefault("ui_language", "en")


def _load_preset(name: str) -> tuple[list[NormalizedZone], np.ndarray | None]:
    yaml_path, image_path = ZONE_PRESETS[name]
    image = cv2.imread(str(image_path)) if image_path and image_path.exists() else None
    if yaml_path is None:
        return [], image
    if not yaml_path.exists():
        raise ZoneValidationError(f"Preset file not found: {yaml_path.name}")
    if image is None:
        raise ZoneValidationError("Preset reference image could not be read.")
    height, width = image.shape[:2]
    return parse_zone_yaml(
        yaml_path.read_text(encoding="utf-8"),
        source_width=width,
        source_height=height,
    ), image


def _zone_editor() -> tuple[list[NormalizedZone], bool, np.ndarray | None]:
    preset = st.selectbox(_tr("ui.zone_preset"), list(ZONE_PRESETS), key="zone_preset")
    if st.session_state.zone_preset_last != preset:
        zones, _ = _load_preset(preset)
        st.session_state.zone_rows = zones_to_rows(zones)
        st.session_state.zone_yaml_text = zones_to_yaml(zones)
        st.session_state.zone_current_yaml = zones_to_yaml(zones)
        st.session_state.zone_preset_last = preset
        st.session_state.zone_editor_revision += 1

    mode = st.radio(
        _t("zone_mode"),
        ["table", "yaml"],
        format_func=lambda value: _t(value),
        horizontal=True,
        key="zone_editor_mode",
    )
    table_mode = mode == "table"
    if st.session_state.zone_mode_last != mode:
        if table_mode and st.session_state.get("zone_current_yaml"):
            try:
                parsed = parse_zone_yaml(st.session_state.zone_current_yaml)
                st.session_state.zone_rows = zones_to_rows(parsed)
                st.session_state.zone_editor_revision += 1
            except ZoneValidationError:
                pass
        elif not table_mode and st.session_state.get("zone_current_yaml"):
            st.session_state.zone_yaml_text = st.session_state.zone_current_yaml
        st.session_state.zone_mode_last = mode

    st.caption(_t("zone_help"))
    try:
        if table_mode:
            rows = st.session_state.get("zone_rows", [])
            frame = pd.DataFrame(
                rows,
                columns=[
                    "id",
                    "name",
                    "risk_level",
                    "required_ppe",
                    "dwell_seconds",
                    "polygon",
                ],
            )
            edited = st.data_editor(
                frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=f"zone_table_{st.session_state.zone_editor_revision}",
                column_config={
                    "risk_level": st.column_config.SelectboxColumn(
                        "risk_level", options=list(RISK_LEVELS), required=True
                    ),
                    "required_ppe": st.column_config.TextColumn(
                        "required_ppe",
                        help="Comma-separated PPE types: helmet, vest, gloves, boots, goggles.",
                    ),
                    "dwell_seconds": st.column_config.NumberColumn(
                        "dwell_seconds",
                        help="Minimum continuous zone occupancy before an intrusion event.",
                        min_value=0.0,
                        step=0.5,
                        required=True,
                    ),
                    "polygon": st.column_config.TextColumn(
                        "polygon", help=_t("polygon_help"), required=True
                    ),
                },
            )
            zones = zones_from_rows(edited.fillna("").to_dict(orient="records"))
            st.session_state.zone_rows = zones_to_rows(zones)
        else:
            text = st.text_area(
                _tr("ui.zone_editor"),
                key="zone_yaml_text",
                height=280,
            )
            zones = parse_zone_yaml(text)
        st.session_state.zone_current_yaml = zones_to_yaml(zones)
        valid = True
    except ZoneValidationError as exc:
        zones = []
        valid = False
        st.error(f"{_t('invalid_zones')}: {exc}")

    _, image_path = ZONE_PRESETS[preset]
    preview_image = cv2.imread(str(image_path)) if image_path and image_path.exists() else None
    st.caption(_t("zone_preview"))
    st.image(
        draw_zone_preview(zones, preview_image),
        channels="BGR",
        width="stretch",
    )
    st.download_button(
        _t("download_zones"),
        zones_to_yaml(zones).encode("utf-8"),
        file_name="zones.normalized.yaml",
        mime="application/yaml",
        disabled=not valid,
    )
    return zones, valid, preview_image


def _create_context(
    model_path: str,
    device: str | None,
    conf: float,
    iou: float,
    cooldown: float,
    zones: list[NormalizedZone],
) -> RunContext:
    detector = YoloDetector(model_path=model_path, device=device, conf=conf, iou=iou)
    detector.reset_tracking()
    tracker = SimpleTracker()
    tracker.reset()
    engine = RuleEngine(zones=zones_to_domain(zones), cooldown_seconds=cooldown)
    engine.reset()
    model_info = inspect_detector(detector, model_path)
    context = RunContext(
        context_id=uuid.uuid4().hex,
        output_root=Path(st.session_state.web_output_root),
        detector=detector,
        engine=engine,
        tracker=tracker,
        model_info=model_info,
    )
    st.session_state.active_run_context = context
    st.session_state.last_model_info = model_info
    return context


def _release_preview() -> None:
    cap = st.session_state.get("preview_cap")
    if cap is not None:
        cap.release()
    context = st.session_state.get("preview_context")
    if context is not None:
        try:
            context.detector.reset_tracking()
            context.tracker.reset()
            context.engine.reset()
        except Exception:
            pass
    st.session_state.preview_running = False
    st.session_state.preview_frame = 0
    for key in ("preview_cap", "preview_context", "preview_settings"):
        st.session_state.pop(key, None)


def _stop_preview() -> None:
    _release_preview()


def _run_live_preview() -> None:
    settings = st.session_state.preview_settings
    source = settings["source"]
    redacted_source = redact_uri_credentials(source)
    st.subheader("🎥 " + _tr("ui.preview_active"))
    st.caption(f"{_t('run_source')}: {redacted_source}")
    st.button(
        "⏹️ " + _tr("ui.stop_preview"),
        on_click=_stop_preview,
        key="btn_stop_preview",
    )
    placeholder = st.empty()

    cap = st.session_state.get("preview_cap")
    if cap is None or not cap.isOpened():
        capture_source: str | int = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(capture_source)
        st.session_state.preview_cap = cap
    if not cap.isOpened():
        st.error(_tr("ui.camera_open_error"))
        _release_preview()
        return

    ret, frame = cap.read()
    if not ret:
        st.warning(_tr("ui.camera_ended"))
        _release_preview()
        return

    context = st.session_state.get("preview_context")
    if context is None:
        with st.spinner(_tr("ui.loading")):
            context = _create_context(
                settings["model_path"],
                settings["device"],
                settings["conf"],
                settings["iou"],
                settings["cooldown"],
                settings["zones"],
            )
        st.session_state.preview_context = context

    frame_index = st.session_state.preview_frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    time_seconds = frame_index / fps
    detections = context.detector.track(frame)
    if any(d.track_id is None for d in detections):
        detections = context.tracker.update(detections)
    height, width = frame.shape[:2]
    evaluation = context.engine.evaluate_frame(
        detections,
        source_type="camera",
        source_path=redacted_source,
        frame_index=frame_index,
        time_seconds=time_seconds,
        metadata={"frame_size": (width, height), "privacy_blur_enabled": settings["blur_faces"]},
    )
    context.events.extend(evaluation.new_events)
    display_frame = _blur_faces(frame.copy(), detections, enabled=settings["blur_faces"])
    annotated = annotate_image(
        display_frame,
        zones_to_domain(settings["zones"]),
        detections,
        evaluation.active_findings,
    )
    placeholder.image(annotated, channels="BGR", width="stretch")
    c1, c2 = st.columns(2)
    c1.metric(_t("preview_findings"), len(evaluation.active_findings))
    c2.metric(_t("preview_events"), len(context.events))

    st.session_state.preview_frame = frame_index + 1
    if st.session_state.preview_frame >= settings["max_frames"]:
        st.success(_tr("ui.preview_complete"))
        st.session_state.last_preview_events = list(context.events)
        _release_preview()
        return
    time.sleep(0.03)
    st.rerun()


def _probe_source(source: str | Path, *, video: bool) -> tuple[int, int]:
    if not video:
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(_t("source_probe_failed"))
        height, width = image.shape[:2]
        return width, height
    source_text = str(source)
    capture_source: str | int = int(source_text) if source_text.isdigit() else source_text
    cap = cv2.VideoCapture(capture_source)
    try:
        if not cap.isOpened():
            raise ValueError(_t("source_probe_failed"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(_t("source_probe_failed"))
            height, width = frame.shape[:2]
        return width, height
    finally:
        cap.release()


def _write_web_manifest(
    run_dir: Path,
    *,
    source: str,
    source_type: str,
    privacy_blur: bool,
    zones: list[NormalizedZone],
    model_info: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    manifest = {
        "completed_at": utc_run_label(),
        "source": source,
        "source_type": source_type,
        "privacy_blur_enabled": privacy_blur,
        "zones": json.loads(json.dumps([zone.__dict__ for zone in zones], default=list)),
        "model": model_info,
        "summary": summary,
    }
    path = run_dir / "web_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _finish_run(
    *,
    context: RunContext,
    events: list[Any],
    source_type: str,
    source_label: str,
    result_path: Path | None,
    zones: list[NormalizedZone],
    privacy_blur: bool,
) -> dict[str, Any]:
    run_dir = resolve_latest_run(context.output_root)
    if run_dir is None:
        raise RuntimeError("Pipeline completed without publishing a run directory.")
    summary = aggregate_events(events)
    web_manifest_path = _write_web_manifest(
        run_dir,
        source=source_label,
        source_type=source_type,
        privacy_blur=privacy_blur,
        zones=zones,
        model_info=context.model_info,
        summary=summary,
    )
    record = {
        "id": context.context_id,
        "completed_at": utc_run_label(),
        "source_type": source_type,
        "source_label": source_label,
        "run_dir": run_dir,
        "result_path": result_path,
        "events": list(events),
        "summary": summary,
        "model_info": context.model_info,
        "zones_yaml": zones_to_yaml(zones),
        "privacy_blur": privacy_blur,
        "core_manifest_path": run_dir / "manifest.json",
        "web_manifest_path": web_manifest_path,
    }
    history = st.session_state.run_history
    history.insert(0, record)
    del history[20:]
    st.session_state.selected_run_id = context.context_id
    st.session_state.active_run_context = None
    return record


def _find_annotated_artifact(run_dir: Path, media_dir: str) -> Path | None:
    """Locate the pipeline-published annotated artifact without source-name assumptions."""

    candidates = sorted((run_dir / media_dir).glob("*_annotated.*"))
    return candidates[0] if candidates else None


def _run_uploaded(
    uploaded_file: Any,
    *,
    source_type: str,
    model_path: str,
    device: str | None,
    conf: float,
    iou: float,
    cooldown: float,
    blur_faces: bool,
    zones: list[NormalizedZone],
) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov"}
    with temporary_upload(uploaded_file) as source_path:
        _probe_source(source_path, video=is_video)
        context = _create_context(model_path, device, conf, iou, cooldown, zones)
        if is_video:
            events = run_video_pipeline(
                source_path=source_path,
                output_root=context.output_root,
                detector=context.detector,
                zones=zones_to_domain(zones),
                rule_engine=context.engine,
                blur_faces_enabled=blur_faces,
                max_frames=300,
                audit_source=Path(uploaded_file.name).name,
            )
        else:
            _, events = run_image_pipeline(
                source_path=source_path,
                output_root=context.output_root,
                detector=context.detector,
                zones=zones_to_domain(zones),
                rule_engine=context.engine,
                blur_faces_enabled=blur_faces,
                audit_source=Path(uploaded_file.name).name,
            )
        run_dir = resolve_latest_run(context.output_root)
        if run_dir is None:
            raise RuntimeError("Pipeline did not publish its output run.")
        result_path = _find_annotated_artifact(run_dir, "videos" if is_video else "images")
        return _finish_run(
            context=context,
            events=events,
            source_type="video" if is_video else source_type,
            source_label=Path(uploaded_file.name).name,
            result_path=result_path,
            zones=zones,
            privacy_blur=blur_faces,
        )


def _run_stream(
    source: str,
    *,
    max_frames: int,
    model_path: str,
    device: str | None,
    conf: float,
    iou: float,
    cooldown: float,
    blur_faces: bool,
    zones: list[NormalizedZone],
) -> dict[str, Any]:
    redacted_source = redact_uri_credentials(source)
    context = _create_context(model_path, device, conf, iou, cooldown, zones)
    try:
        events = run_camera_pipeline(
            source=source,
            output_root=context.output_root,
            detector=context.detector,
            zones=zones_to_domain(zones),
            rule_engine=context.engine,
            blur_faces_enabled=blur_faces,
            max_frames=max_frames,
            audit_source=redacted_source,
        )
    finally:
        partial_run = resolve_latest_run(context.output_root, include_failed=True)
        if partial_run is not None:
            sanitize_run_artifacts(partial_run, source, redacted_source)
    run_dir = resolve_latest_run(context.output_root)
    result_path = _find_annotated_artifact(run_dir, "videos") if run_dir else None
    return _finish_run(
        context=context,
        events=events,
        source_type="camera",
        source_label=redacted_source,
        result_path=result_path,
        zones=zones,
        privacy_blur=blur_faces,
    )


def _status_badge(status: str) -> str:
    return {
        "yes": f"✅ {_tr('compliance_yes')}",
        "no": f"❌ {_tr('compliance_no')}",
        "unknown": f"❓ {_tr('compliance_unknown')}",
    }.get(status, status)


def _risk_badge(level: str) -> str:
    emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(level, "⚪")
    return f"{emoji} {_tr(level).upper()}"


def _safe_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    return None if frame.empty else frame


def _build_compliance_table(events_dir: Path) -> pd.DataFrame | None:
    frame = _safe_csv(events_dir / "compliance.csv")
    if frame is None:
        return None
    sort_columns = [column for column in ("person_track_id", "frame_index") if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    if "person_track_id" in frame:
        frame = frame.drop_duplicates(subset=["person_track_id"], keep="last")
    return frame


def _zip_reports(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        candidates = [run_dir / "events", run_dir / "snapshots"]
        for root in candidates:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(run_dir))
        for name in ("manifest.json", "web_run_manifest.json"):
            path = run_dir / name
            if path.exists():
                archive.write(path, path.name)
    return buffer.getvalue()


def _render_model_info(info: dict[str, Any], zones_yaml: str | None = None) -> None:
    st.subheader("🧠 " + _t("model_audit"))
    c1, c2, c3 = st.columns(3)
    c1.metric(_tr("ui.model_path"), Path(info["requested_model"]).name)
    c2.metric(_tr("ui.device"), info.get("device", "unknown"))
    digest = info.get("sha256")
    c3.metric(_t("model_hash"), digest[:12] if digest else "hub / unavailable")
    classes = info.get("classes", [])
    st.caption(f"{_t('classes')}: {', '.join(classes) if classes else '—'}")
    capabilities = info.get("capabilities", [])
    st.caption(f"{_t('capabilities')}: {', '.join(capabilities) if capabilities else '—'}")
    if info.get("ppe_classes"):
        st.success(_t("ppe_model"))
    else:
        st.warning(_t("generic_model"))
    if zones_yaml is not None:
        st.code(zones_yaml, language="yaml")


def _event_rows(events: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "risk": event.risk_level,
            "risk_label": _risk_badge(event.risk_level),
            "type": event.event_type,
            "description": event.description,
            "time_seconds": event.time_seconds,
            "zone": event.zone_name or "—",
            "track": event.person_track_id,
            "snapshot": event.snapshot_path,
        }
        for event in events
    ]


def _render_run(record: dict[str, Any]) -> None:
    run_dir = Path(record["run_dir"])
    events_dir = run_dir / "events"
    summary = record["summary"]
    st.caption(
        f"{_t('run_source')}: {record['source_label']} · {_t('run_time')}: {record['completed_at']} · "
        f"{_t('session_output')}: {run_dir.name}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_tr("ui.total_events"), summary["total_events"])
    c2.metric(_tr("ui.zone_intrusions"), summary["zone_intrusions"])
    c3.metric(_tr("ui.ppe_violations"), summary["ppe_violations"])
    c4.metric(_tr("ui.unique_track_ids"), summary["unique_track_ids"])

    result_tab, events_tab, detections_tab, compliance_tab, evidence_tab, model_tab, reports_tab = st.tabs(
        [
            "🖼️ " + _tr("ui.result_tab"),
            "📋 " + _tr("ui.events_title"),
            "🔍 " + _tr("ui.detections_tab"),
            "👷 " + _tr("ui.compliance_tab"),
            "📸 " + _t("evidence"),
            "🧠 " + _t("model_tab"),
            "📥 " + _tr("ui.reports_tab"),
        ]
    )

    with result_tab:
        result_path = record.get("result_path")
        if result_path and Path(result_path).exists():
            result_path = Path(result_path)
            if result_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
                st.video(str(result_path))
            else:
                image = cv2.imread(str(result_path))
                if image is not None:
                    st.image(image, channels="BGR", width="stretch")
        else:
            st.warning(_tr("ui.annotated_video_not_found"))

    rows = _event_rows(record["events"])
    risks = sorted({row["risk"] for row in rows})
    event_types = sorted({row["type"] for row in rows})
    with events_tab:
        if not rows:
            st.info(_tr("ui.no_events"))
        else:
            left, right = st.columns(2)
            selected_risks = left.multiselect(
                _t("risk_filter"), risks, default=risks, key=f"risk_{record['id']}"
            )
            selected_types = right.multiselect(
                _t("event_filter"), event_types, default=event_types, key=f"type_{record['id']}"
            )
            filtered = [
                row for row in rows if row["risk"] in selected_risks and row["type"] in selected_types
            ]
            display = pd.DataFrame(filtered)
            if not display.empty:
                st.dataframe(
                    display[["risk_label", "type", "description", "time_seconds", "zone", "track"]],
                    width="stretch",
                    hide_index=True,
                )

    with detections_tab:
        detections = _safe_csv(events_dir / "detections.csv")
        if detections is None:
            st.info(_tr("ui.no_detection_log"))
        else:
            if "class_name" in detections:
                st.bar_chart(detections["class_name"].value_counts())
            columns = [c for c in ("class_name", "confidence", "bbox", "track_id") if c in detections]
            st.dataframe(detections[columns], width="stretch", hide_index=True)

    with compliance_tab:
        st.info(_t("unknown_notice"))
        compliance = _build_compliance_table(events_dir)
        if compliance is None:
            st.info(_tr("ui.no_compliance_records"))
        else:
            display = compliance.copy()
            for column in PPE_TYPES:
                if column in display:
                    display[column] = display[column].fillna("unknown").apply(_status_badge)
            columns = [
                column
                for column in ("person_track_id", "frame_index", "time_seconds", *PPE_TYPES, "violations")
                if column in display
            ]
            st.dataframe(display[columns], width="stretch", hide_index=True)

    with evidence_tab:
        snapshots = [
            (event, resolve_run_artifact(run_dir, event.snapshot_path))
            for event in record["events"]
            if event.snapshot_path
        ]
        snapshots = [(event, path) for event, path in snapshots if path.exists()]
        if not snapshots:
            st.info(_t("no_evidence"))
        else:
            evidence_risks = sorted({event.risk_level for event, _ in snapshots})
            evidence_types = sorted({event.event_type for event, _ in snapshots})
            left, right = st.columns(2)
            selected_evidence_risks = left.multiselect(
                _t("risk_filter"),
                evidence_risks,
                default=evidence_risks,
                key=f"evidence_risk_{record['id']}",
            )
            selected_evidence_types = right.multiselect(
                _t("event_filter"),
                evidence_types,
                default=evidence_types,
                key=f"evidence_type_{record['id']}",
            )
            snapshots = [
                (event, path)
                for event, path in snapshots
                if event.risk_level in selected_evidence_risks
                and event.event_type in selected_evidence_types
            ]
            if not snapshots:
                st.info(_t("no_evidence"))
            snapshot_columns = st.columns(3)
            for index, (event, path) in enumerate(snapshots):
                with snapshot_columns[index % 3]:
                    st.image(str(path), width="stretch")
                    st.caption(f"{_risk_badge(event.risk_level)} · {_tr(event.event_type)}")

    with model_tab:
        _render_model_info(record["model_info"], record["zones_yaml"])

    with reports_tab:
        for path in sorted(events_dir.glob("*")) if events_dir.exists() else []:
            if path.is_file():
                mime = "text/csv" if path.suffix == ".csv" else "application/json"
                st.download_button(
                    f"{_tr('ui.download')} {path.name}",
                    path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"download_{record['id']}_{path.name}",
                )
        core_manifest = Path(record["core_manifest_path"])
        if core_manifest.exists():
            st.download_button(
                f"{_t('manifest')} (core audit)",
                core_manifest.read_bytes(),
                file_name=core_manifest.name,
                mime="application/json",
                key=f"core_manifest_{record['id']}",
            )
        web_manifest = Path(record["web_manifest_path"])
        if web_manifest.exists():
            st.download_button(
                f"{_t('manifest')} (Web session)",
                web_manifest.read_bytes(),
                file_name=web_manifest.name,
                mime="application/json",
                key=f"web_manifest_{record['id']}",
            )
        st.download_button(
            "📦 " + _tr("ui.download_all_reports"),
            _zip_reports(run_dir),
            file_name=f"safety_reports_{run_dir.name}.zip",
            mime="application/zip",
            key=f"zip_{record['id']}",
        )


def _render_history() -> None:
    history = st.session_state.run_history
    st.header("🕘 " + _t("run_history"))
    st.caption(_t("history_limit"))
    if not history:
        st.info(_t("no_history"))
        return
    by_id = {record["id"]: record for record in history}
    selected = st.selectbox(
        _t("run_history"),
        list(by_id),
        key="selected_run_id",
        format_func=lambda run_id: (
            f"{by_id[run_id]['completed_at']} · {by_id[run_id]['source_type']} · "
            f"{by_id[run_id]['source_label']}"
        ),
        label_visibility="collapsed",
    )
    _render_run(by_id[selected])


def main() -> None:
    st.set_page_config(page_title="Utility Site Safety AI", page_icon="🦺", layout="wide")
    _inject_custom_css()
    _initialize_session()

    with st.sidebar:
        lang_label = st.selectbox(
            "🌐 Language / 语言 / 語言",
            list(LANG_OPTIONS),
            key="language_selector",
        )
        st.session_state.ui_language = LANG_OPTIONS[lang_label]
        # Never call the process-global i18n setter from a multi-session Web app.
        # All Web translations below receive this session's language explicitly;
        # saved annotations retain the server's fixed/default language.
        st.header("⚙️ " + _tr("ui.sidebar_settings"))
        model_path = st.text_input(_tr("ui.model_path"), value=DEFAULT_MODEL)
        col1, col2 = st.columns(2)
        conf = col1.slider(_tr("ui.confidence"), 0.0, 1.0, 0.25, 0.05)
        iou = col2.slider(_tr("ui.nms_iou"), 0.0, 1.0, 0.45, 0.05)
        device_label = st.selectbox(_tr("ui.device"), ["auto", "cpu", "mps", "cuda"], index=0)
        device = None if device_label == "auto" else device_label
        blur_faces = st.toggle(
            _tr("ui.blur_faces"),
            value=True,
            help=_tr("ui.blur_faces_help"),
        )
        st.caption(_t("privacy_default"))
        cooldown = float(st.slider(_tr("ui.cooldown"), 0, 60, 10, 1))
        st.caption(_tr("ui.confidence_floors_caption"))

        st.markdown("---")
        st.header("🚧 " + _tr("ui.zones_config"))
        normalized_zones, zones_valid, _ = _zone_editor()
        st.markdown("---")
        model_info = st.session_state.last_model_info
        st.subheader("🧠 " + _t("model_audit"))
        if model_info:
            st.caption(Path(model_info["requested_model"]).name)
            st.caption(", ".join(model_info.get("classes", [])) or "—")
        else:
            st.caption(_t("model_pending"))

    st.markdown(
        f'<div class="title-card"><h1>🦺 {_tr("ui.title")}</h1><p>{_tr("ui.page_caption")}</p></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.preview_running:
        _run_live_preview()
        return

    source_tab, camera_tab, rtsp_tab, live_tab = st.tabs(
        [
            "📁 " + _tr("ui.upload_image") + " / " + _tr("ui.upload_video"),
            "📷 " + _tr("ui.camera_tab"),
            "🎥 " + _t("camera_source"),
            "📡 " + _tr("ui.live_preview"),
        ]
    )

    with source_tab:
        uploaded_file = st.file_uploader(
            _tr("ui.upload_image") + " / " + _tr("ui.upload_video"),
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )
        if uploaded_file and Path(uploaded_file.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            preview = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if preview is not None:
                st.image(
                    draw_zone_preview(normalized_zones, preview),
                    channels="BGR",
                    caption=_t("zone_preview"),
                    width="stretch",
                )
        run_upload = st.button(
            "🚀 " + _tr("ui.run_inference"),
            type="primary",
            width="stretch",
            disabled=not zones_valid,
            key="run_upload",
        )

    with camera_tab:
        camera_image = st.camera_input(_tr("ui.camera_tab"))
        run_camera = st.button(
            "🚀 " + _tr("ui.run_inference") + " (" + _tr("ui.camera_tab") + ")",
            type="primary",
            width="stretch",
            disabled=not zones_valid,
            key="run_camera_image",
        )

    with rtsp_tab:
        rtsp_source = st.text_input(
            _t("camera_source"),
            value="0",
            type="password",
            help=_tr("ui.camera_source_help"),
            key="rtsp_source",
        )
        st.caption(_t("credentials"))
        rtsp_frames = st.slider(_tr("ui.live_frames"), 1, 300, 60, 10, key="rtsp_frames")
        run_rtsp = st.button(
            "🚀 " + _tr("ui.run_inference") + " (" + _t("camera_source") + ")",
            type="primary",
            width="stretch",
            disabled=not zones_valid,
            key="run_rtsp",
        )

    with live_tab:
        live_source = st.text_input(
            _t("camera_source"),
            value="0",
            type="password",
            help=_tr("ui.camera_source_help"),
            key="live_source_input",
        )
        st.caption(_t("credentials"))
        live_max_frames = st.slider(
            _tr("ui.live_frames"), 1, 300, 60, 10, key="live_max_frames"
        )
        start_preview = st.button(
            "▶️ " + _tr("ui.start_preview"),
            type="primary",
            width="stretch",
            disabled=not zones_valid,
            key="start_preview",
        )
        if start_preview:
            # Release the previous source first, then mark the new preview active.
            # This ordering fixes the former start-then-immediately-release bug.
            _release_preview()
            st.session_state.preview_settings = {
                "source": live_source,
                "max_frames": live_max_frames,
                "model_path": model_path,
                "device": device,
                "conf": conf,
                "iou": iou,
                "cooldown": cooldown,
                "blur_faces": blur_faces,
                "zones": list(normalized_zones),
            }
            st.session_state.preview_running = True
            st.session_state.preview_frame = 0
            st.rerun()

    if not zones_valid:
        st.error(_t("start_blocked"))

    try:
        record: dict[str, Any] | None = None
        if run_upload:
            if uploaded_file is None:
                st.warning(_tr("ui.start_prompt"))
            else:
                with st.spinner(_tr("ui.loading")):
                    record = _run_uploaded(
                        uploaded_file,
                        source_type="upload",
                        model_path=model_path,
                        device=device,
                        conf=conf,
                        iou=iou,
                        cooldown=cooldown,
                        blur_faces=blur_faces,
                        zones=normalized_zones,
                    )
        elif run_camera:
            if camera_image is None:
                st.warning(_tr("ui.start_prompt"))
            else:
                with st.spinner(_tr("ui.loading")):
                    record = _run_uploaded(
                        camera_image,
                        source_type="camera",
                        model_path=model_path,
                        device=device,
                        conf=conf,
                        iou=iou,
                        cooldown=cooldown,
                        blur_faces=blur_faces,
                        zones=normalized_zones,
                    )
        elif run_rtsp:
            with st.spinner(_tr("ui.loading")):
                record = _run_stream(
                    rtsp_source,
                    max_frames=rtsp_frames,
                    model_path=model_path,
                    device=device,
                    conf=conf,
                    iou=iou,
                    cooldown=cooldown,
                    blur_faces=blur_faces,
                    zones=normalized_zones,
                )
        if record is not None:
            summary = record["summary"]
            st.success(
                f"{_tr('ui.inference_complete')}: {summary['total_events']} "
                f"{_tr('ui.total_events').lower()}"
            )
    except Exception as exc:
        raw_source = rtsp_source if run_rtsp else ""
        message = str(exc).replace(raw_source, redact_uri_credentials(raw_source)) if raw_source else str(exc)
        st.error(f"{_t('run_failed')}: {message}")
        st.session_state.active_run_context = None

    st.markdown("---")
    _render_history()


if __name__ == "__main__":
    main()
