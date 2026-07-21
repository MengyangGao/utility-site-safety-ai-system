"""Stable web-product configuration, labels, and repository assets."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_OUTPUT_ROOT = REPO_ROOT / "outputs" / "web_demo" / "sessions"
DEMO_IMAGE = REPO_ROOT / "examples" / "sample_images" / "construction_zone_01.jpg"

MODEL_PROFILES: dict[str, str] = {}
if (REPO_ROOT / "models" / "ppe_yolo11n.pt").is_file():
    MODEL_PROFILES["PPE monitor · local portfolio checkpoint"] = "models/ppe_yolo11n.pt"
MODEL_PROFILES["Person + restricted zone · YOLO11n"] = (
    "models/yolo11n.pt"
    if (REPO_ROOT / "models" / "yolo11n.pt").is_file()
    else "yolo11n.pt"
)

ZONE_PRESETS: dict[str, tuple[Path | None, Path | None]] = {
    "Construction perimeter": (
        REPO_ROOT / "examples" / "zones_construction_zone_01.yaml",
        REPO_ROOT / "examples" / "sample_images" / "construction_zone_01.jpg",
    ),
    "Solar inspection": (
        REPO_ROOT / "examples" / "zones_solar_inspection_pexels_4254172.yaml",
        REPO_ROOT / "examples" / "sample_images" / "solar_inspection_pexels_4254172.jpg",
    ),
    "PPE work area": (
        REPO_ROOT / "examples" / "zones_construction_site_ppe_01.yaml",
        REPO_ROOT / "examples" / "sample_images" / "construction_site_ppe_01.jpg",
    ),
    "No restricted zones": (None, DEMO_IMAGE),
}

LANGUAGES = {"English": "en", "简体中文": "zh-hans", "繁體中文": "zh-hant"}

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "product": "Utility Safety Intelligence",
        "subtitle": "Evidence-first visual monitoring for high-risk field operations.",
        "monitor": "Monitor",
        "policy": "Zones & policy",
        "results": "Results & review",
        "history": "Run history",
        "ready": "SYSTEM READY",
        "workflow": "Select source · Apply policy · Analyse · Review evidence",
        "run": "Run safety analysis",
        "run_sample": "Run portfolio sample",
        "no_result": "Run an analysis to open the evidence workspace.",
        "privacy_note": "Privacy protection is enabled by default for every saved artifact.",
        "quality": "Monitoring quality",
        "evidence": "Evidence timeline",
        "capability": "Capability boundary",
        "not_certified": "Decision-support prototype · Human review required",
        "zone_valid": "Policy ready",
        "zone_invalid": "Policy needs attention",
    },
    "zh-hans": {
        "product": "电力现场安全智能平台",
        "subtitle": "面向高风险作业的证据化视觉监测与复核。",
        "monitor": "监测中心",
        "policy": "区域与策略",
        "results": "结果与复核",
        "history": "运行历史",
        "ready": "系统就绪",
        "workflow": "选择来源 · 应用策略 · 智能分析 · 人工复核",
        "run": "开始安全分析",
        "run_sample": "运行作品集示例",
        "no_result": "运行一次分析后，将在这里打开证据工作区。",
        "privacy_note": "所有保存的结果默认启用隐私保护。",
        "quality": "监测质量",
        "evidence": "证据时间线",
        "capability": "能力边界",
        "not_certified": "辅助决策原型 · 必须人工复核",
        "zone_valid": "策略已就绪",
        "zone_invalid": "策略需要修正",
    },
    "zh-hant": {
        "product": "電力現場安全智能平台",
        "subtitle": "面向高風險作業的證據化視覺監測與覆核。",
        "monitor": "監測中心",
        "policy": "區域與策略",
        "results": "結果與覆核",
        "history": "執行歷史",
        "ready": "系統就緒",
        "workflow": "選擇來源 · 套用策略 · 智慧分析 · 人工覆核",
        "run": "開始安全分析",
        "run_sample": "執行作品集範例",
        "no_result": "執行一次分析後，將在這裡開啟證據工作區。",
        "privacy_note": "所有儲存的結果預設啟用隱私保護。",
        "quality": "監測品質",
        "evidence": "證據時間線",
        "capability": "能力邊界",
        "not_certified": "輔助決策原型 · 必須人工覆核",
        "zone_valid": "策略已就緒",
        "zone_invalid": "策略需要修正",
    },
}


def text(key: str, language: str) -> str:
    """Return a UI string with an English fallback."""
    return TEXT.get(language, TEXT["en"]).get(key, TEXT["en"].get(key, key))
