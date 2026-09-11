"""Internationalization helpers for UI text, class labels, and risk levels."""

from __future__ import annotations

import os
from contextvars import ContextVar

SUPPORTED_LANGUAGES = {"en", "zh-hans", "zh-hant"}
DEFAULT_LANGUAGE = "en"

_ENV_KEY = "UTILITY_SAFETY_AI_LANG"
_ACTIVE_LANGUAGE: ContextVar[str | None] = ContextVar("utility_safety_ai_language", default=None)


_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Class labels
        "person": "person",
        "helmet": "helmet",
        "vest": "vest",
        "gloves": "gloves",
        "boots": "boots",
        "goggles": "goggles",
        "no_helmet": "no helmet",
        "no_vest": "no vest",
        "no_gloves": "no gloves",
        "no_boots": "no boots",
        "no_goggles": "no goggles",
        "no_goggle": "no goggles",
        "none": "none",
        # Risk levels
        "low": "LOW",
        "medium": "MED",
        "high": "HIGH",
        "critical": "CRIT",
        # Annotation / UI
        "safety_events": "SAFETY EVENTS",
        "ui.events_title": "SAFETY EVENTS",
        "ui.title": "Utility Site Safety AI",
        "ui.sidebar_settings": "Settings",
        "ui.choose_language": "Language",
        "ui.model_path": "Model path",
        "ui.confidence": "Confidence threshold",
        "ui.blur_faces": "Enable face / privacy blur",
        "ui.blur_faces_help": "Blurs detected faces or the upper body before saving outputs.",
        "ui.upload_image": "Upload an image",
        "ui.upload_video": "Upload a video",
        "ui.run_inference": "Run inference",
        "ui.download_csv": "Download CSV report",
        "ui.no_events": "No safety events detected.",
        "ui.start_prompt": "Upload a file, take a snapshot, or configure a live source, then click Run inference.",
        "ui.event_summary": "Event summary",
        "ui.zones_config": "Zones configuration",
        "ui.source_type": "Source type",
        "ui.image_tab": "Image",
        "ui.video_tab": "Video",
        "ui.camera_tab": "Camera",
        "ui.result_tab": "Result",
        "ui.detections_tab": "Detections",
        "ui.compliance_tab": "Compliance",
        "ui.summary_tab": "Summary",
        "ui.reports_tab": "Reports",
        "ui.device": "Device",
        "ui.nms_iou": "NMS IoU",
        "ui.loading": "Loading model and running inference...",
        "ui.annotated_clip_not_found": "Annotated camera clip not found.",
        "ui.annotated_video_not_found": "Annotated video not found.",
        "ui.no_detection_log": "No detection log found.",
        "ui.no_compliance_records": "No compliance records found.",
        "ui.per_person_compliance": "Per-person PPE compliance (latest state)",
        "ui.detected_objects": "Detected objects",
        "ui.cooldown": "Event cooldown (s)",
        "ui.live_frames": "Live frames to process",
        "ui.total_events": "Total events",
        "ui.zone_intrusions": "Zone intrusions",
        "ui.ppe_violations": "PPE violations",
        "ui.unique_track_ids": "Temporary track IDs",
        "ui.by_risk_level": "By risk level",
        "ui.by_event_type": "By event type",
        "ui.inference_complete": "Inference complete",
        "ui.page_caption": "PPE, people, and restricted-zone monitoring for utility and construction worksites.",
        "ui.confidence_floors_caption": "The global confidence is a minimum for every class; stricter per-class floors are then applied (for example helmet 0.35).",
        "ui.zone_tip": "Tip: polygon coordinates must match the image/video resolution you upload.",
        "ui.camera_source_help": "Webcam index (0), RTSP URL, or local camera device path.",
        "ui.zone_preset": "Load preset",
        "ui.zone_editor": "Edit zones (YAML)",
        "ui.download": "Download",
        "ui.download_all_reports": "Download all reports (ZIP)",
        "ui.live_preview": "Live preview",
        "ui.start_preview": "Start live preview",
        "ui.stop_preview": "Stop live preview",
        "ui.preview_active": "Live preview active",
        "ui.preview_complete": "Preview complete",
        "ui.camera_open_error": "Could not open camera source.",
        "ui.camera_ended": "Camera stream ended.",
        # Event descriptions
        "missing_helmet": "Missing helmet",
        "missing_vest": "Missing high-visibility vest",
        "missing_gloves": "Missing gloves",
        "missing_boots": "Missing safety boots",
        "missing_goggles": "Missing safety goggles",
        "zone_intrusion": "Person entered restricted zone",
        "compliance_yes": "yes",
        "compliance_no": "no",
        "compliance_unknown": "unknown",
    },
    "zh-hans": {
        "person": "人员",
        "helmet": "安全帽",
        "vest": "反光背心",
        "gloves": "手套",
        "boots": "安全靴",
        "goggles": "护目镜",
        "no_helmet": "未戴安全帽",
        "no_vest": "未穿反光背心",
        "no_gloves": "未戴手套",
        "no_boots": "未穿安全靴",
        "no_goggles": "未戴护目镜",
        "no_goggle": "未戴护目镜",
        "none": "无",
        "low": "低",
        "medium": "中",
        "high": "高",
        "critical": "危急",
        "safety_events": "安全事件",
        "ui.events_title": "安全事件",
        "ui.title": "工地安全 AI 监控系统",
        "ui.sidebar_settings": "设置",
        "ui.choose_language": "语言",
        "ui.model_path": "模型路径",
        "ui.confidence": "置信度阈值",
        "ui.blur_faces": "启用人脸/隐私模糊",
        "ui.blur_faces_help": "在保存结果前对检测到的人脸或上半身进行模糊处理。",
        "ui.upload_image": "上传图片",
        "ui.upload_video": "上传视频",
        "ui.camera_source": "摄像头 / RTSP 源",
        "ui.start_camera": "启动摄像头",
        "ui.stop_camera": "停止摄像头",
        "ui.run_inference": "运行推理",
        "ui.download_csv": "下载 CSV 报告",
        "ui.no_events": "未检测到安全事件。",
        "ui.start_prompt": "上传文件、拍摄快照或配置实时源，然后点击运行推理。",
        "ui.event_summary": "事件摘要",
        "ui.zones_config": "区域配置",
        "ui.source_type": "来源类型",
        "ui.image_tab": "图片",
        "ui.video_tab": "视频",
        "ui.camera_tab": "摄像头",
        "ui.result_tab": "结果",
        "ui.detections_tab": "检测",
        "ui.compliance_tab": "合规",
        "ui.summary_tab": "摘要",
        "ui.reports_tab": "报告",
        "ui.device": "设备",
        "ui.nms_iou": "NMS IoU",
        "ui.loading": "正在加载模型并运行推理...",
        "ui.annotated_clip_not_found": "未找到标注后的摄像头片段。",
        "ui.annotated_video_not_found": "未找到标注后的视频。",
        "ui.no_detection_log": "未找到检测日志。",
        "ui.no_compliance_records": "未找到合规记录。",
        "ui.per_person_compliance": "每人 PPE 合规情况（最新状态）",
        "ui.detected_objects": "检测到的对象",
        "ui.page_caption": "面向公用事业和施工现场的人员、防护装备与限制区域监测。",
        "ui.confidence_floors_caption": "全局置信度是所有类别的最低门槛，随后再应用更严格的每类下限（例如安全帽 0.35）。",
        "ui.zone_tip": "提示：多边形坐标必须与您上传的图片/视频分辨率匹配。",
        "ui.camera_source_help": "摄像头索引（0）、RTSP 地址或本地摄像头设备路径。",
        "ui.zone_preset": "加载预设",
        "ui.zone_editor": "编辑区域（YAML）",
        "ui.download": "下载",
        "ui.download_all_reports": "下载全部报告（ZIP）",
        "ui.live_preview": "实时预览",
        "ui.start_preview": "开始实时预览",
        "ui.stop_preview": "停止实时预览",
        "ui.preview_active": "实时预览中",
        "ui.preview_complete": "预览完成",
        "ui.camera_open_error": "无法打开摄像头源。",
        "ui.camera_ended": "摄像头流已结束。",
        "ui.cooldown": "事件冷却时间（秒）",
        "ui.live_frames": "要处理的实时帧数",
        "ui.total_events": "事件总数",
        "ui.zone_intrusions": "区域入侵",
        "ui.ppe_violations": "PPE 违规",
        "ui.unique_track_ids": "临时跟踪 ID",
        "ui.by_risk_level": "按风险等级",
        "ui.by_event_type": "按事件类型",
        "ui.inference_complete": "推理完成",
        "missing_helmet": "未佩戴安全帽",
        "missing_vest": "未穿着反光背心",
        "missing_gloves": "未佩戴手套",
        "missing_boots": "未穿着安全靴",
        "missing_goggles": "未佩戴护目镜",
        "zone_intrusion": "人员进入受限区域",
        "compliance_yes": "是",
        "compliance_no": "否",
        "compliance_unknown": "未知",
    },
    "zh-hant": {
        "person": "人員",
        "helmet": "安全帽",
        "vest": "反光背心",
        "gloves": "手套",
        "boots": "安全靴",
        "goggles": "護目鏡",
        "no_helmet": "未戴安全帽",
        "no_vest": "未穿反光背心",
        "no_gloves": "未戴手套",
        "no_boots": "未穿安全靴",
        "no_goggles": "未戴護目鏡",
        "no_goggle": "未戴護目鏡",
        "none": "無",
        "low": "低",
        "medium": "中",
        "high": "高",
        "critical": "危急",
        "safety_events": "安全事件",
        "ui.events_title": "安全事件",
        "ui.title": "工地安全 AI 監控系統",
        "ui.sidebar_settings": "設定",
        "ui.choose_language": "語言",
        "ui.model_path": "模型路徑",
        "ui.confidence": "置信度閾值",
        "ui.blur_faces": "啟用人臉/隱私模糊",
        "ui.blur_faces_help": "在儲存結果前對偵測到的人臉或上半身進行模糊處理。",
        "ui.upload_image": "上傳圖片",
        "ui.upload_video": "上傳影片",
        "ui.camera_source": "攝影機 / RTSP 來源",
        "ui.start_camera": "啟動攝影機",
        "ui.stop_camera": "停止攝影機",
        "ui.run_inference": "執行推理",
        "ui.download_csv": "下載 CSV 報告",
        "ui.no_events": "未偵測到安全事件。",
        "ui.start_prompt": "上傳檔案、拍攝快照或設定即時來源，然後點擊執行推理。",
        "ui.event_summary": "事件摘要",
        "ui.zones_config": "區域設定",
        "ui.source_type": "來源類型",
        "ui.image_tab": "圖片",
        "ui.video_tab": "影片",
        "ui.camera_tab": "攝影機",
        "ui.result_tab": "結果",
        "ui.detections_tab": "偵測",
        "ui.compliance_tab": "合規",
        "ui.summary_tab": "摘要",
        "ui.reports_tab": "報告",
        "ui.device": "裝置",
        "ui.nms_iou": "NMS IoU",
        "ui.loading": "正在載入模型並執行推理...",
        "ui.annotated_clip_not_found": "未找到標註後的攝影機片段。",
        "ui.annotated_video_not_found": "未找到標註後的影片。",
        "ui.no_detection_log": "未找到偵測紀錄。",
        "ui.no_compliance_records": "未找到合規紀錄。",
        "ui.per_person_compliance": "每人 PPE 合規情況（最新狀態）",
        "ui.detected_objects": "偵測到的物件",
        "ui.page_caption": "面向公用事業和施工現場的人員、防護裝備與限制區域監測。",
        "ui.confidence_floors_caption": "全域置信度是所有類別的最低門檻，隨後再套用更嚴格的每類下限（例如安全帽 0.35）。",
        "ui.zone_tip": "提示：多邊形座標必須與您上傳的圖片/影片解析度相符。",
        "ui.camera_source_help": "攝影機索引（0）、RTSP 網址或本地攝影機裝置路徑。",
        "ui.zone_preset": "載入預設",
        "ui.zone_editor": "編輯區域（YAML）",
        "ui.download": "下載",
        "ui.download_all_reports": "下載全部報告（ZIP）",
        "ui.live_preview": "即時預覽",
        "ui.start_preview": "開始即時預覽",
        "ui.stop_preview": "停止即時預覽",
        "ui.preview_active": "即時預覽中",
        "ui.preview_complete": "預覽完成",
        "ui.camera_open_error": "無法開啟攝影機來源。",
        "ui.camera_ended": "攝影機串流已結束。",
        "ui.cooldown": "事件冷卻時間（秒）",
        "ui.live_frames": "要處理的即時幀數",
        "ui.total_events": "事件總數",
        "ui.zone_intrusions": "區域入侵",
        "ui.ppe_violations": "PPE 違規",
        "ui.unique_track_ids": "臨時追蹤 ID",
        "ui.by_risk_level": "按風險等級",
        "ui.by_event_type": "按事件類型",
        "ui.inference_complete": "推理完成",
        "missing_helmet": "未佩戴安全帽",
        "missing_vest": "未穿著反光背心",
        "missing_gloves": "未佩戴手套",
        "missing_boots": "未穿著安全靴",
        "missing_goggles": "未佩戴護目鏡",
        "zone_intrusion": "人員進入受限區域",
        "compliance_yes": "是",
        "compliance_no": "否",
        "compliance_unknown": "未知",
    },
}


def current_language() -> str:
    """Return the task-local language, then the environment default."""
    active = _ACTIVE_LANGUAGE.get()
    lang = (active or os.environ.get(_ENV_KEY, DEFAULT_LANGUAGE)).lower().replace("_", "-")
    if lang in SUPPORTED_LANGUAGES:
        return lang
    # Accept bare "zh" as simplified.
    if lang.startswith("zh"):
        return "zh-hans"
    return DEFAULT_LANGUAGE


def set_language(lang: str) -> str:
    """Set the active language for the current thread/task context.

    Returns the normalized language code that was set.
    """
    lang = lang.lower().replace("_", "-")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "zh-hans" if lang.startswith("zh") else DEFAULT_LANGUAGE
    _ACTIVE_LANGUAGE.set(lang)
    return lang


def _(key: str, lang: str | None = None) -> str:
    """Look up a translation for ``key``.

    Falls back to the English string, then to the key itself.
    """
    lang = lang or current_language()
    return _TRANSLATIONS.get(lang, _TRANSLATIONS[DEFAULT_LANGUAGE]).get(
        key, _TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    )


def class_label(class_name: str, lang: str | None = None) -> str:
    """Return the localized display label for a detection class."""
    return _(class_name.lower(), lang=lang)


def risk_label(risk_level: str, lang: str | None = None) -> str:
    """Return the localized short label for a risk level."""
    return _(risk_level.lower(), lang=lang)
