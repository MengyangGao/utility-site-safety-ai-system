"""Session-isolated Streamlit product demo for Utility Site Safety AI."""

from __future__ import annotations

import io
import json
import os
import shutil
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
from utility_safety_ai.i18n import _, set_language
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
    cleanup_session_outputs,
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
DEMO_IMAGE = REPO_ROOT / "examples" / "sample_images" / "construction_zone_01.jpg"

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
        "video_processing": "Video processing",
        "complete": "Complete video",
        "preview": "Quick preview",
        "preview_frame_limit": "Preview frame limit",
        "complete_video_notice": "The complete video will be processed. Runtime depends on its duration and device.",
        "preview_video_notice": "Only the selected leading frames will be processed; the run manifest records the limit.",
        "privacy_style": "Privacy redaction style",
        "trusted_model": "Verified model profile",
        "custom_model": "Use an advanced custom model path",
        "custom_model_warning": "Only load checkpoints you trust. PyTorch model files can contain executable code.",
        "run_sample": "Run portfolio sample",
        "review_queue": "Incident review queue",
        "save_review": "Save review decisions",
        "review_saved": "Review decisions saved",
        "compliance_view": "Compliance view",
        "latest_state": "Latest state per track",
        "timeline": "Full timeline",
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
        "video_processing": "视频处理范围",
        "complete": "完整视频",
        "preview": "快速预览",
        "preview_frame_limit": "预览帧数上限",
        "complete_video_notice": "将处理完整视频，耗时取决于视频长度与运行设备。",
        "preview_video_notice": "仅处理开头指定帧数，运行清单会记录该限制。",
        "privacy_style": "隐私遮挡样式",
        "trusted_model": "可信模型配置",
        "custom_model": "使用高级自定义模型路径",
        "custom_model_warning": "仅加载可信权重。PyTorch 模型文件可能包含可执行代码。",
        "run_sample": "运行作品集示例",
        "review_queue": "事件复核队列",
        "save_review": "保存复核结果",
        "review_saved": "复核结果已保存",
        "compliance_view": "合规数据视图",
        "latest_state": "每条轨迹最新状态",
        "timeline": "完整时间线",
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
        "video_processing": "影片處理範圍",
        "complete": "完整影片",
        "preview": "快速預覽",
        "preview_frame_limit": "預覽影格上限",
        "complete_video_notice": "將處理完整影片，耗時取決於影片長度與執行裝置。",
        "preview_video_notice": "只處理開頭指定影格數，執行清單會記錄該限制。",
        "privacy_style": "隱私遮擋樣式",
        "trusted_model": "可信模型設定",
        "custom_model": "使用進階自訂模型路徑",
        "custom_model_warning": "只載入可信權重。PyTorch 模型檔案可能包含可執行程式碼。",
        "run_sample": "執行作品集範例",
        "review_queue": "事件覆核佇列",
        "save_review": "儲存覆核結果",
        "review_saved": "覆核結果已儲存",
        "compliance_view": "合規資料檢視",
        "latest_state": "每條軌跡最新狀態",
        "timeline": "完整時間軸",
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
        :root { --canvas:#070b14; --panel:#0f172a; --line:#243044; --muted:#94a3b8;
          --brand:#f97316; --brand-soft:rgba(249,115,22,.14); --cyan:#22d3ee; }
        .stApp { background:radial-gradient(circle at 75% 0%,rgba(34,211,238,.06),transparent 30%),var(--canvas); }
        .main .block-container { max-width:1440px;padding-top:1.1rem;padding-bottom:3rem; }
        [data-testid="stSidebar"] { background:#0a101c;border-right:1px solid var(--line); }
        [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown { color:#e2e8f0!important; }
        .title-card { position:relative;overflow:hidden;background:linear-gradient(135deg,#111827,#0b1324);
          border:1px solid var(--line);border-radius:20px;padding:1.45rem 1.6rem;margin-bottom:1rem;color:white;
          box-shadow:0 24px 55px -32px rgba(34,211,238,.45); }
        .title-card:after { content:"";position:absolute;inset:0 0 0 auto;width:38%;
          background:linear-gradient(135deg,transparent,rgba(34,211,238,.08));pointer-events:none; }
        .eyebrow { color:var(--brand);font-size:.72rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase; }
        .title-card h1 { margin:.35rem 0 0;font-weight:760;font-size:1.85rem;letter-spacing:-.025em; }
        .title-card p { margin:.45rem 0 0;color:#aebbd0;font-size:.92rem;max-width:780px; }
        .system-badge { display:inline-flex;align-items:center;gap:.45rem;border:1px solid rgba(34,211,238,.25);
          background:rgba(34,211,238,.07);color:#a5f3fc;border-radius:999px;padding:.32rem .65rem;font-size:.72rem; }
        .system-dot { width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.12); }
        .workflow { display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:.2rem 0 1.15rem; }
        .workflow-step { border:1px solid var(--line);background:#0c1321;border-radius:12px;padding:.72rem .8rem;
          color:var(--muted);font-size:.78rem; }
        .workflow-step b { display:block;color:#f8fafc;font-size:.82rem;margin-top:.18rem; }
        .workflow-step span { color:var(--brand);font-weight:800; }
        .info-card { background:var(--panel);border-left:3px solid var(--brand);border-radius:0 12px 12px 0;
          padding:1rem;color:#e2e8f0; }
        [data-testid="stMetric"] { background:linear-gradient(160deg,#111827,#0c1321);border:1px solid var(--line);
          border-radius:14px;padding:1rem;box-shadow:0 14px 35px -28px #000; }
        [data-testid="stMetricValue"] { font-weight:750;letter-spacing:-.03em; }
        .stButton>button { border-radius:10px;font-weight:700;border-color:#334155;min-height:2.65rem; }
        .stButton>button[kind="primary"] { background:var(--brand);border-color:var(--brand);color:#fff; }
        [data-baseweb="tab-list"] { gap:.35rem;background:#0c1321;border:1px solid var(--line);border-radius:12px;padding:.3rem; }
        [data-baseweb="tab"] { border-radius:8px;padding:.45rem .8rem; }
        [data-baseweb="tab-highlight"] { background:var(--brand); }
        [data-testid="stFileUploader"] { border:1px dashed #334155;border-radius:14px;background:#0a101c;padding:.4rem; }
        hr { border-color:var(--line)!important; }
        @media(max-width:900px){ .workflow{grid-template-columns:repeat(2,1fr)} .main .block-container{padding-left:1rem;padding-right:1rem} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session() -> None:
    st.session_state.setdefault("web_session_id", new_session_id())
    sessions_root = REPO_ROOT / "outputs" / "web_demo" / "sessions"
    output_root = sessions_root / st.session_state.web_session_id
    output_root.mkdir(parents=True, exist_ok=True)
    if not st.session_state.get("retention_checked"):
        cleanup_session_outputs(
            sessions_root,
            keep_session_id=st.session_state.web_session_id,
            max_age_hours=24.0,
            max_sessions=20,
        )
        st.session_state.retention_checked = True
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
        metadata={
            "frame_size": (width, height),
            "privacy_blur_enabled": settings["blur_faces"],
            "privacy_mode": settings["privacy_mode"],
        },
    )
    context.events.extend(evaluation.new_events)
    display_frame = _blur_faces(
        frame.copy(),
        detections,
        enabled=settings["blur_faces"],
        mode=settings["privacy_mode"],
    )
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
    privacy_mode: str,
    zones: list[NormalizedZone],
    model_info: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    manifest = {
        "completed_at": utc_run_label(),
        "source": source,
        "source_type": source_type,
        "privacy_blur_enabled": privacy_blur,
        "privacy_mode": privacy_mode,
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
    privacy_mode: str,
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
        privacy_mode=privacy_mode,
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
        "privacy_mode": privacy_mode,
        "core_manifest_path": run_dir / "manifest.json",
        "web_manifest_path": web_manifest_path,
    }
    history = st.session_state.run_history
    history.insert(0, record)
    pruned = history[20:]
    del history[20:]
    session_root = Path(st.session_state.web_output_root).resolve()
    for old_record in pruned:
        old_run = Path(old_record["run_dir"]).resolve()
        if old_run.is_relative_to(session_root):
            shutil.rmtree(old_run, ignore_errors=True)
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
    privacy_mode: str,
    zones: list[NormalizedZone],
    max_frames: int | None = None,
) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov"}
    with temporary_upload(uploaded_file) as source_path:
        _probe_source(source_path, video=is_video)
        context = _create_context(model_path, device, conf, iou, cooldown, zones)
        if is_video:
            progress = st.progress(0, text=_tr("ui.loading"))

            def update_progress(processed: int, total: int | None) -> None:
                if total:
                    progress.progress(
                        min(1.0, processed / total),
                        text=f"{processed:,} / {total:,} frames",
                    )
                else:
                    progress.progress(0, text=f"{processed:,} frames")

            try:
                events = run_video_pipeline(
                    source_path=source_path,
                    output_root=context.output_root,
                    detector=context.detector,
                    zones=zones_to_domain(zones),
                    rule_engine=context.engine,
                    blur_faces_enabled=blur_faces,
                    privacy_mode=privacy_mode,
                    max_frames=max_frames,
                    audit_source=Path(uploaded_file.name).name,
                    progress_callback=update_progress,
                )
            finally:
                progress.empty()
        else:
            _, events = run_image_pipeline(
                source_path=source_path,
                output_root=context.output_root,
                detector=context.detector,
                zones=zones_to_domain(zones),
                rule_engine=context.engine,
                blur_faces_enabled=blur_faces,
                privacy_mode=privacy_mode,
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
            privacy_mode=privacy_mode,
        )


def _run_portfolio_sample(
    *,
    model_path: str,
    device: str | None,
    conf: float,
    iou: float,
    cooldown: float,
    blur_faces: bool,
    privacy_mode: str,
    zones: list[NormalizedZone],
) -> dict[str, Any]:
    """Run the bundled, provenance-documented image without an upload step."""

    if not DEMO_IMAGE.is_file():
        raise FileNotFoundError(f"Bundled demo image not found: {DEMO_IMAGE.name}")
    context = _create_context(model_path, device, conf, iou, cooldown, zones)
    _, events = run_image_pipeline(
        source_path=DEMO_IMAGE,
        output_root=context.output_root,
        detector=context.detector,
        zones=zones_to_domain(zones),
        rule_engine=context.engine,
        blur_faces_enabled=blur_faces,
        privacy_mode=privacy_mode,
        audit_source=DEMO_IMAGE.name,
    )
    run_dir = resolve_latest_run(context.output_root)
    if run_dir is None:
        raise RuntimeError("Sample pipeline did not publish its output run.")
    return _finish_run(
        context=context,
        events=events,
        source_type="sample",
        source_label=DEMO_IMAGE.name,
        result_path=_find_annotated_artifact(run_dir, "images"),
        zones=zones,
        privacy_blur=blur_faces,
        privacy_mode=privacy_mode,
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
    privacy_mode: str,
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
            privacy_mode=privacy_mode,
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
        privacy_mode=privacy_mode,
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


def _build_compliance_table(
    events_dir: Path, *, latest_only: bool = True
) -> pd.DataFrame | None:
    frame = _safe_csv(events_dir / "compliance.csv")
    if frame is None:
        return None
    sort_columns = [column for column in ("person_track_id", "frame_index") if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    if latest_only and "person_track_id" in frame:
        frame = frame.drop_duplicates(subset=["person_track_id"], keep="last")
    return frame


def _zip_reports(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        candidates = [
            run_dir / "events",
            run_dir / "snapshots",
            run_dir / "images",
            run_dir / "videos",
        ]
        for root in candidates:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(run_dir))
        for name in ("manifest.json", "web_run_manifest.json", "review_state.json"):
            path = run_dir / name
            if path.exists():
                archive.write(path, path.name)
    return buffer.getvalue()


def _render_review_queue(record: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Render and persist an operator acknowledgement workflow per run."""

    if not rows:
        return
    run_dir = Path(record["run_dir"])
    review_path = run_dir / "review_state.json"
    existing: dict[str, dict[str, Any]] = {}
    if review_path.exists():
        try:
            stored = json.loads(review_path.read_text(encoding="utf-8"))
            existing = {item["event_id"]: item for item in stored.get("reviews", [])}
        except (OSError, ValueError, KeyError, TypeError):
            existing = {}
    review_rows = []
    for event, row in zip(record["events"], rows, strict=True):
        saved = existing.get(event.event_id, {})
        review_rows.append(
            {
                "event_id": event.event_id,
                "risk": row["risk"],
                "type": row["type"],
                "time_seconds": row["time_seconds"],
                "status": saved.get("status", "new"),
                "assignee": saved.get("assignee", ""),
                "note": saved.get("note", ""),
            }
        )
    st.markdown(f"#### {_t('review_queue')}")
    edited = st.data_editor(
        pd.DataFrame(review_rows),
        width="stretch",
        hide_index=True,
        disabled=["event_id", "risk", "type", "time_seconds"],
        column_config={
            "status": st.column_config.SelectboxColumn(
                "status", options=["new", "acknowledged", "investigating", "resolved", "false_positive"]
            )
        },
        key=f"review_{record['id']}",
    )
    if st.button(_t("save_review"), key=f"save_review_{record['id']}"):
        payload = {
            "updated_at": utc_run_label(),
            "reviews": edited.to_dict(orient="records"),
        }
        review_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        st.success(_t("review_saved"))


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
            _render_review_queue(record, rows)

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
        compliance_view = st.radio(
            _t("compliance_view"),
            ["latest_state", "timeline"],
            format_func=lambda value: _t(value),
            horizontal=True,
            key=f"compliance_view_{record['id']}",
        )
        compliance = _build_compliance_table(
            events_dir, latest_only=compliance_view == "latest_state"
        )
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
    st.set_page_config(page_title="Utility Site Safety AI", page_icon="🛡️", layout="wide")
    _inject_custom_css()
    _initialize_session()

    with st.sidebar:
        lang_label = st.selectbox(
            "🌐 Language / 语言 / 語言",
            list(LANG_OPTIONS),
            key="language_selector",
        )
        st.session_state.ui_language = LANG_OPTIONS[lang_label]
        # Context-local language selection keeps concurrent browser sessions
        # isolated while making saved annotations match the selected UI language.
        set_language(st.session_state.ui_language)
        st.caption("CONTROL CENTER")
        st.header(_tr("ui.sidebar_settings"))
        trusted_models = list(
            dict.fromkeys(
                [
                    DEFAULT_MODEL,
                    *[
                        str(path.relative_to(REPO_ROOT))
                        for path in (
                            REPO_ROOT / "models" / "ppe_yolo11n.pt",
                            REPO_ROOT / "models" / "ppe_yolo11s.pt",
                            REPO_ROOT / "models" / "yolo11n.pt",
                        )
                        if path.is_file()
                    ],
                    "yolo11n.pt",
                ]
            )
        )
        model_path = st.selectbox(_t("trusted_model"), trusted_models)
        trusted_only = os.environ.get("UTILITY_SAFETY_TRUSTED_MODELS_ONLY", "0") == "1"
        use_custom_model = st.toggle(
            _t("custom_model"), value=False, disabled=trusted_only
        )
        if use_custom_model:
            st.warning(_t("custom_model_warning"))
            model_path = st.text_input(_tr("ui.model_path"), value=model_path)
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
        privacy_mode = st.selectbox(
            _t("privacy_style"),
            ["gaussian", "pixelate", "solid"],
            disabled=not blur_faces,
        )
        st.caption(_t("privacy_default"))
        cooldown = float(st.slider(_tr("ui.cooldown"), 0, 60, 10, 1))
        st.caption(_tr("ui.confidence_floors_caption"))

        st.markdown("---")
        st.header(_tr("ui.zones_config"))
        normalized_zones, zones_valid, _ = _zone_editor()
        st.markdown("---")
        model_info = st.session_state.last_model_info
        st.subheader(_t("model_audit"))
        if model_info:
            st.caption(Path(model_info["requested_model"]).name)
            st.caption(", ".join(model_info.get("classes", [])) or "—")
        else:
            st.caption(_t("model_pending"))

    st.markdown(
        f'<div class="title-card"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start">'
        f'<div><div class="eyebrow">Safety intelligence platform</div><h1>{_tr("ui.title")}</h1>'
        f'<p>{_tr("ui.page_caption")}</p></div><div class="system-badge"><i class="system-dot"></i> SYSTEM READY</div>'
        f'</div></div><div class="workflow">'
        f'<div class="workflow-step"><span>01</span><b>Connect source</b></div>'
        f'<div class="workflow-step"><span>02</span><b>Configure controls</b></div>'
        f'<div class="workflow-step"><span>03</span><b>Run analysis</b></div>'
        f'<div class="workflow-step"><span>04</span><b>Review evidence</b></div></div>',
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
        upload_is_video = bool(
            uploaded_file
            and Path(uploaded_file.name).suffix.lower() in {".mp4", ".avi", ".mov"}
        )
        video_mode = st.radio(
            _t("video_processing"),
            ["complete", "preview"],
            format_func=lambda value: _t(value),
            horizontal=True,
            disabled=not upload_is_video,
            key="video_processing_mode",
        )
        upload_max_frames = st.number_input(
            _t("preview_frame_limit"),
            min_value=30,
            max_value=10000,
            value=300,
            step=30,
            disabled=not upload_is_video or video_mode != "preview",
        )
        if upload_is_video:
            st.caption(
                _t("complete_video_notice")
                if video_mode == "complete"
                else _t("preview_video_notice")
            )
        run_upload = st.button(
            "🚀 " + _tr("ui.run_inference"),
            type="primary",
            width="stretch",
            disabled=not zones_valid,
            key="run_upload",
        )
        run_sample = st.button(
            _t("run_sample"),
            width="stretch",
            disabled=not zones_valid,
            key="run_sample",
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
                "privacy_mode": privacy_mode,
                "zones": list(normalized_zones),
            }
            st.session_state.preview_running = True
            st.session_state.preview_frame = 0
            st.rerun()

    if not zones_valid:
        st.error(_t("start_blocked"))

    try:
        record: dict[str, Any] | None = None
        if run_sample:
            with st.spinner(_tr("ui.loading")):
                record = _run_portfolio_sample(
                    model_path=model_path,
                    device=device,
                    conf=conf,
                    iou=iou,
                    cooldown=cooldown,
                    blur_faces=blur_faces,
                    privacy_mode=privacy_mode,
                    zones=normalized_zones,
                )
        elif run_upload:
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
                        privacy_mode=privacy_mode,
                        zones=normalized_zones,
                        max_frames=(
                            int(upload_max_frames)
                            if upload_is_video and video_mode == "preview"
                            else None
                        ),
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
                        privacy_mode=privacy_mode,
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
                    privacy_mode=privacy_mode,
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
