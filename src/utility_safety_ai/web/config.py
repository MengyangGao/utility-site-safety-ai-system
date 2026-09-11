"""Stable web-product configuration, labels, and repository assets."""

from __future__ import annotations

import os
from pathlib import Path

from ..resources import resource_root

REPO_ROOT = resource_root()
WEB_OUTPUT_ROOT = (
    Path(os.getenv("UTILITY_SAFETY_OUTPUT_DIR", str(Path.cwd() / "outputs")))
    / "web_demo"
    / "sessions"
)
DEMO_IMAGE = REPO_ROOT / "examples" / "sample_images" / "construction_site_ppe_01.jpg"

MODEL_PROFILES: dict[str, str] = {}
if (REPO_ROOT / "models" / "ppe_yolo11n.pt").is_file():
    MODEL_PROFILES["PPE + restricted-zone monitor"] = "models/ppe_yolo11n.pt"
MODEL_PROFILES["Person + restricted-zone monitor"] = (
    "models/yolo11n.pt" if (REPO_ROOT / "models" / "yolo11n.pt").is_file() else "yolo11n.pt"
)

ZONE_PRESETS: dict[str, tuple[Path | None, Path | None]] = {}
piling_zone = REPO_ROOT / "examples/zones_piling_rear_view.yaml"
piling_image = REPO_ROOT / "examples/sample_images/piling_rear_view_01.jpg"
if piling_zone.is_file() and piling_image.is_file():
    ZONE_PRESETS["Piling works · example zone"] = (piling_zone, piling_image)
ppe_zone = REPO_ROOT / "examples" / "zones_construction_site_ppe_01.yaml"
if ppe_zone.is_file() and DEMO_IMAGE.is_file():
    ZONE_PRESETS["PPE work area"] = (ppe_zone, DEMO_IMAGE)
ZONE_PRESETS["No restricted zones"] = (None, DEMO_IMAGE if DEMO_IMAGE.is_file() else None)

LANGUAGES = {"English": "en", "简体中文": "zh-hans", "繁體中文": "zh-hant"}

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "product": "Safety inspection",
        "subtitle": "Review people, protective equipment and work-zone observations.",
        "monitor": "Monitor",
        "policy": "Zones & policy",
        "results": "Results & review",
        "history": "Run history",
        "ready": "Local workspace",
        "workflow": "Select source · Apply policy · Analyse · Review evidence",
        "run": "Run safety analysis",
        "run_sample": "Run included sample",
        "no_result": "Run an analysis to view detections, events, and reports.",
        "privacy_note": "Uploads use privacy protection by default. Reviewed rear-view samples can remain clear.",
        "quality": "Monitoring quality",
        "evidence": "Event timeline",
        "capability": "Model capabilities",
        "not_certified": "Human review required",
        "zone_valid": "Policy ready",
        "zone_invalid": "Policy needs attention",
    },
    "zh-hans": {
        "product": "现场安全检查",
        "subtitle": "实时监测人员、个人防护装备与作业区域。",
        "monitor": "监测中心",
        "policy": "区域与策略",
        "results": "结果与复核",
        "history": "运行历史",
        "ready": "系统就绪",
        "workflow": "选择来源 · 应用策略 · 智能分析 · 人工复核",
        "run": "开始安全分析",
        "run_sample": "运行内置示例",
        "no_result": "运行一次分析后，在这里查看检测、事件与报告。",
        "privacy_note": "上传素材默认启用隐私保护；已核验的背面示例可保持清晰。",
        "quality": "监测质量",
        "evidence": "事件时间线",
        "capability": "模型能力",
        "not_certified": "检测结果需人工复核",
        "zone_valid": "策略已就绪",
        "zone_invalid": "策略需要修正",
    },
    "zh-hant": {
        "product": "現場安全檢查",
        "subtitle": "即時監測人員、個人防護裝備與作業區域。",
        "monitor": "監測中心",
        "policy": "區域與策略",
        "results": "結果與覆核",
        "history": "執行歷史",
        "ready": "系統就緒",
        "workflow": "選擇來源 · 套用策略 · 智慧分析 · 人工覆核",
        "run": "開始安全分析",
        "run_sample": "執行內建範例",
        "no_result": "執行一次分析後，在這裡查看偵測、事件與報告。",
        "privacy_note": "上傳素材預設啟用隱私保護；已核驗的背面範例可保持清晰。",
        "quality": "監測品質",
        "evidence": "事件時間線",
        "capability": "模型能力",
        "not_certified": "偵測結果需人工覆核",
        "zone_valid": "策略已就緒",
        "zone_invalid": "策略需要修正",
    },
}


def text(key: str, language: str) -> str:
    """Return a UI string with an English fallback."""
    return TEXT.get(language, TEXT["en"]).get(key, TEXT["en"].get(key, key))
