"""Modern Streamlit monitoring console."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from utility_safety_ai.i18n import set_language
from utility_safety_ai.monitoring.profiles import (
    get_monitoring_profile,
    monitoring_profile_names,
)
from utility_safety_ai.web.config import (
    DEMO_IMAGE,
    LANGUAGES,
    MODEL_PROFILES,
    REPO_ROOT,
    ZONE_PRESETS,
    text,
)
from utility_safety_ai.web.results import render_history, render_run
from utility_safety_ai.web.services import (
    AnalysisResult,
    AnalysisSettings,
    analyse_camera,
    analyse_file,
    analyse_path,
)
from utility_safety_ai.web.state import initialize_session, record_run
from utility_safety_ai.web.theme import brand, hero, inject_theme, section
from utility_safety_ai.web_helpers import (
    ZoneValidationError,
    draw_zone_preview,
    parse_zone_yaml,
    zones_to_yaml,
)


def _t(key: str) -> str:
    return text(key, st.session_state.get("ui_language", "en"))


def _resolve_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _sidebar_settings() -> AnalysisSettings:
    with st.sidebar:
        brand()
        language_label = st.selectbox("Language / 语言", list(LANGUAGES))
        language = LANGUAGES[language_label]
        st.session_state.ui_language = language
        set_language(language)

        st.markdown("### Monitoring setup")
        profile_name = st.selectbox(
            "Monitoring profile",
            monitoring_profile_names(),
            index=monitoring_profile_names().index(
                st.session_state.get("monitoring_profile", "balanced")
            ),
            format_func=lambda name: get_monitoring_profile(name).label,
        )
        st.session_state.monitoring_profile = profile_name
        profile = get_monitoring_profile(profile_name)
        st.caption(profile.description)

        model_label = st.selectbox("Model profile", list(MODEL_PROFILES))
        use_custom = st.toggle("Advanced custom model", value=False)
        if use_custom:
            st.warning("Only load checkpoints you trust. PyTorch weights can contain code.")
            model_value = st.text_input("Custom model path", value=MODEL_PROFILES[model_label])
        else:
            model_value = MODEL_PROFILES[model_label]
        model_path = _resolve_model_path(model_value)

        device_label = st.selectbox("Processing device", ["CPU", "Auto", "Apple MPS", "CUDA"])
        device = {"Auto": None, "CPU": "cpu", "Apple MPS": "mps", "CUDA": "cuda:0"}[
            device_label
        ]
        with st.expander("Detection tuning"):
            confidence = st.slider(
                "Detection confidence",
                0.05,
                0.90,
                float(profile.confidence),
                0.05,
                key=f"confidence_{profile_name}",
            )
            nms_iou = st.slider(
                "Overlap suppression",
                0.10,
                0.90,
                float(profile.nms_iou),
                0.05,
                key=f"iou_{profile_name}",
            )
            cooldown = st.slider("Repeat-alert cooldown (seconds)", 0, 60, 10)

        st.markdown("### Privacy & reports")
        blur_faces = st.toggle("Privacy blur", value=True)
        privacy_mode = st.selectbox(
            "Privacy redaction style",
            ["gaussian", "pixelate", "solid"],
            format_func=lambda value: value.title(),
        )
        st.radio(
            "Video processing",
            ["Complete video", "Quick preview"],
            horizontal=True,
            key="video_processing_mode",
        )
        st.caption(_t("privacy_note"))
        st.markdown(
            f'<div class="usi-boundary">{_t("not_certified")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Open source under AGPL-3.0 · "
            "[Source](https://github.com/MengyangGao/utility-site-safety-ai-system) · "
            "[License](https://github.com/MengyangGao/utility-site-safety-ai-system/blob/main/LICENSE)"
        )

    return AnalysisSettings(
        model_path=model_path,
        device=device,
        confidence=confidence,
        nms_iou=nms_iou,
        cooldown_seconds=float(cooldown),
        blur_faces=blur_faces,
        privacy_mode=privacy_mode,
        profile=profile,
    )


def _normalize_preset(name: str) -> str:
    path, image_path = ZONE_PRESETS[name]
    if path is None:
        return "zones: []\n"
    raw = path.read_text(encoding="utf-8")
    if image_path is None:
        raise ZoneValidationError(f"Preset {name!r} has no reference image")
    reference = cv2.imread(str(image_path))
    if reference is None:
        raise ZoneValidationError(f"Preset reference image could not be read: {image_path.name}")
    height, width = reference.shape[:2]
    zones = parse_zone_yaml(raw, source_width=width, source_height=height)
    return zones_to_yaml(zones)


def _current_zones() -> tuple[list, str | None]:
    try:
        return parse_zone_yaml(st.session_state.zone_yaml), None
    except ZoneValidationError as exc:
        return [], str(exc)


def _policy_workspace() -> tuple[list, bool]:
    preset = st.selectbox("Zone policy preset", list(ZONE_PRESETS))
    if st.session_state.zone_preset_loaded != preset:
        st.session_state.zone_yaml = _normalize_preset(preset)
        st.session_state.zone_preset_loaded = preset
    st.caption("Normalized coordinates stay aligned when image resolution changes.")
    raw = st.text_area("Normalized zone policy (YAML)", key="zone_yaml", height=270)
    try:
        zones = parse_zone_yaml(raw)
        valid = True
        st.success(f"{_t('zone_valid')} · {len(zones)} zone(s)")
    except ZoneValidationError as exc:
        zones = []
        valid = False
        st.error(f"{_t('zone_invalid')}: {exc}")

    _yaml_path, preview_path = ZONE_PRESETS[preset]
    preview = cv2.imread(str(preview_path)) if preview_path and preview_path.is_file() else None
    st.image(draw_zone_preview(zones, preview), channels="BGR", width="stretch")
    st.download_button(
        "Download normalized policy",
        zones_to_yaml(zones).encode("utf-8"),
        file_name="zones.normalized.yaml",
        mime="application/yaml",
        disabled=not valid,
    )
    return zones, valid


def _finish(result: AnalysisResult) -> None:
    record_run(result.run_dir, result.source_label)
    if result.event_count:
        st.session_state.run_flash = (
            "warning",
            f"Analysis complete · {result.event_count} confirmed event(s) require review.",
        )
    else:
        st.session_state.run_flash = (
            "success",
            "Analysis complete · no confirmed event emitted.",
        )


def _run_action(operation) -> None:
    try:
        with st.spinner("Running detection and preparing results…"):
            _finish(operation())
        flash = st.session_state.pop("run_flash", None)
        if flash:
            getattr(st, flash[0])(flash[1])
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def _monitor_workspace(settings: AnalysisSettings, zones: list, zones_valid: bool) -> None:
    section(_t("monitor"), _t("workflow"))
    flash = st.session_state.pop("run_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    source_type = st.radio(
        "Source type",
        ["Image", "Video", "Camera / RTSP"],
        horizontal=True,
    )
    output_root = Path(st.session_state.web_output_root)

    if source_type == "Image":
        upload = st.file_uploader("Upload worksite image", type=["jpg", "jpeg", "png"])
        sample_available = DEMO_IMAGE.is_file()
        left, right = st.columns(2)
        run_upload = left.button(
            _t("run"),
            type="primary",
            disabled=upload is None or not zones_valid,
            width="stretch",
        )
        run_sample = right.button(
            _t("run_sample"),
            disabled=not zones_valid or not sample_available,
            width="stretch",
        )
        if not sample_available:
            st.caption("The included sample is available when the app runs from the repository.")
        if run_upload and upload is not None:
            _run_action(
                lambda: analyse_file(
                    upload,
                    is_video=False,
                    output_root=output_root,
                    zones=zones,
                    settings=settings,
                )
            )
        if run_sample:
            _run_action(
                lambda: analyse_path(
                    DEMO_IMAGE,
                    output_root=output_root,
                    zones=zones,
                    settings=settings,
                )
            )
    elif source_type == "Video":
        upload = st.file_uploader("Upload worksite video", type=["mp4", "mov", "avi", "mkv"])
        processing = st.session_state.video_processing_mode
        max_frames = None
        if processing == "Quick preview":
            max_frames = st.slider("Preview frame limit", 30, 900, 180, 30)
        if st.button(
            _t("run"),
            type="primary",
            disabled=upload is None or not zones_valid,
            width="stretch",
        ) and upload is not None:
            _run_action(
                lambda: analyse_file(
                    upload,
                    is_video=True,
                    output_root=output_root,
                    zones=zones,
                    settings=settings,
                    max_frames=max_frames,
                )
            )
    else:
        source_mode = st.radio("Camera input", ["Browser snapshot", "Camera / RTSP stream"], horizontal=True)
        if source_mode == "Browser snapshot":
            capture = st.camera_input("Capture a privacy-protected inspection frame")
            if st.button(
                _t("run"),
                type="primary",
                disabled=capture is None or not zones_valid,
                width="stretch",
            ) and capture is not None:
                _run_action(
                    lambda: analyse_file(
                        capture,
                        is_video=False,
                        output_root=output_root,
                        zones=zones,
                        settings=settings,
                    )
                )
        else:
            source = st.text_input("Camera index or RTSP URL", value="0", type="password")
            max_frames = st.slider("Capture frame limit", 30, 1800, 300, 30)
            st.caption("Credentials are redacted from manifests and downloads.")
            if st.button(
                _t("run"),
                type="primary",
                disabled=not source or not zones_valid,
                width="stretch",
            ):
                _run_action(
                    lambda: analyse_camera(
                        source,
                        output_root=output_root,
                        zones=zones,
                        settings=settings,
                        max_frames=max_frames,
                    )
                )

    profile = settings.profile
    cols = st.columns(3)
    cols[0].metric("Alert confirmation", f"{profile.ppe_confirmation_frames} frames")
    cols[1].metric("Track memory", f"{profile.tracker_max_age} frames")
    cols[2].metric("Association threshold", f"{profile.association_min_score:.2f}")


def main() -> None:
    st.set_page_config(
        page_title="Utility Safety Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session()
    inject_theme()
    settings = _sidebar_settings()
    hero(_t("product"), _t("subtitle"), _t("ready"), _t("not_certified"))

    zones, zone_error = _current_zones()
    views = ["monitor", "policy", "results", "history"]
    selected_view = st.segmented_control(
        "Workspace",
        views,
        default="monitor",
        format_func=_t,
        label_visibility="collapsed",
        key="workspace_view",
    )
    if selected_view == "monitor":
        _monitor_workspace(settings, zones, zone_error is None)
        if zone_error:
            st.error(f"{_t('zone_invalid')}: {zone_error}")
    elif selected_view == "policy":
        section(_t("policy"), "Resolution-independent safety boundaries")
        _policy_workspace()
    elif selected_view == "results":
        selected = st.session_state.get("selected_run")
        if selected:
            render_run(Path(selected), quality_title=_t("quality"), evidence_title=_t("evidence"))
        else:
            st.markdown(
                f'<div class="usi-empty"><b>Evidence workspace</b>{_t("no_result")}</div>',
                unsafe_allow_html=True,
            )
    else:
        section(_t("history"), "Private to this browser session · retained for 24 hours")
        render_history(st.session_state.run_history)


if __name__ == "__main__":
    main()
