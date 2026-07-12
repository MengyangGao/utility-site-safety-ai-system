# Demo Script

This is a repeatable 6–8 minute Bilibili/GitHub portfolio demo. It deliberately starts with the honest clean-clone path and labels custom PPE as an optional second tier.

## Before recording

```bash
conda activate utility-safety-ai
pip install -e ".[dev]"
make verify
make fetch-model
```

Confirm:

- `models/yolo11n.pt` and its `.pt.json` checksum record exist;
- the selected demo asset appears in `examples/assets.yaml`;
- no RTSP credential, personal path, private image, or training token is visible;
- privacy blur is enabled;
- previous demo output is either preserved or intentionally replaced with a deterministic run ID;
- terminal font and browser zoom are readable at video resolution.

Do not quote PPE mAP/FPS unless the matching validation/benchmark artifacts have been reviewed and linked from the model card.

## Storyboard

### 0:00–0:35 — The problem

On screen: project title and architecture excerpt.

Suggested narration:

> This project is not a production safety system and it does not replace a safety officer. It demonstrates how detection, worksite geometry, temporal rules, privacy, and audit evidence can be engineered into one reviewable workflow.

Show these five words: **detect → associate → localize → aggregate → audit**.

### 0:35–1:20 — Honest model capability

Show the two-mode table in the README.

Explain:

- a clean clone uses COCO YOLO11n for people and zones;
- PPE appears only when a compatible custom model is loaded;
- absence of a helmet box becomes `unknown`, not an accusation;
- explicit `no_*` detections are associated to a person and conflicts are resolved.

### 1:20–2:20 — CLI clean-clone run

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_zone_01.jpg \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_zone_01.yaml \
  --output outputs/video-demo \
  --run-id bilibili-clean \
  --overwrite \
  --blur-faces
```

Show:

1. CLI summary and exact run path.
2. Annotated image.
3. `manifest.json` status/model/config.
4. `events.csv` and `detections.csv`.
5. `latest.json`.

Call out that event cooldown affects logging, while active findings stay visible.

### 2:20–3:00 — Run isolation and failure evidence

Show the tree:

```text
outputs/video-demo/runs/bilibili-clean/
```

Explain:

- runs do not delete each other;
- explicit overwrite is required for the same ID;
- artifact hashes make the run inspectable;
- a failed run does not replace the latest successful pointer.

### 3:00–4:30 — Streamlit experience

```bash
streamlit run app.py
```

Record:

1. Language selector.
2. Privacy toggle enabled by default.
3. Model classes/capability area.
4. Normalized zone editor in table mode.
5. YAML mode and zone download.
6. Resolution-aware preview.
7. Image upload and inference.
8. Result, events, detections, compliance, evidence, model, and reports tabs.
9. ZIP report download.
10. Run history selector.

If demonstrating RTSP, use a local disposable source with no visible real credentials. Explain that application artifacts redact user-info/tokens but shell history and external systems still need protection.

### 4:30–5:20 — Video and temporal logic

```bash
utility-safety-ai infer-video \
  --source examples/sample_videos/construction_rebar_pexels_10294768.mp4 \
  --model models/yolo11n.pt \
  --zones examples/zones_construction_rebar_pexels_10294768.yaml \
  --output outputs/video-demo \
  --run-id bilibili-video \
  --overwrite \
  --max-frames 120 \
  --blur-faces
```

Explain temporary track IDs, blank-frame aging, active findings, event cooldown, and why IDs are not identities.

### 5:20–6:20 — Optional PPE model

Only include this section if `models/ppe_yolo11n.pt` exists and its model card/validation artifacts are ready.

```bash
utility-safety-ai infer-image \
  --source examples/sample_images/construction_site_ppe_01.jpg \
  --model models/ppe_yolo11n.pt \
  --zones examples/zones_construction_site_ppe_01.yaml \
  --output outputs/video-demo \
  --run-id bilibili-ppe \
  --overwrite \
  --blur-faces
```

Show the model class list before the result. Describe `yes`, explicit `no`, and `unknown`. If a class is weak, show and say so; do not hide it.

If the exact validated checkpoint is unavailable, skip this section. Do not attach the documented
metrics to a newly retrained weight unless it has its own hash and retained validation artifacts.

### 6:20–7:10 — Engineering quality

Show:

```bash
make verify
```

Briefly highlight:

- offline unit/E2E tests;
- strict zone validation;
- checked media writes;
- privacy snapshot test;
- CI across supported Python versions;
- architecture/model/dataset/security/license documentation.

### 7:10–end — Limitations and next step

Say explicitly:

- default model is not PPE-capable;
- site-specific validation is mandatory;
- privacy blur can miss identities;
- image-plane zones are not calibrated 3D safety distances;
- Ultralytics/model/dataset terms require license review;
- production deployment needs security, retention, access control, monitoring, and safety governance.

Close with the project's strongest portfolio message: the value is not a single neural network—it is the auditable engineering around uncertain perception.

## Capture checklist

- [x] `docs/assets/web-dashboard.jpg` — full Web workbench view
- [x] `docs/assets/annotated-zone.jpg` — clean-clone person + zone output
- [x] `docs/assets/annotated-ppe-zone.jpg` — validated-checkpoint image result
- [x] `docs/assets/annotated-video-frame-2.jpg` — privacy-enabled PPE video result
- [ ] Capture an optional run-artifact tree during recording
- [ ] Capture an optional compliance/model-audit close-up during recording
- [ ] Terminal output does not reveal home directory, credentials, or tokens
- [ ] Every shown asset is present in `examples/assets.yaml`
- [ ] Captions distinguish clean-clone and PPE-enabled modes
- [ ] No unsupported safety or production claim is spoken or shown

## Suggested title and chapter text

**Title:** 从 YOLO 到可审计安全事件：Utility Site Safety AI 工程化实战

**Chapters:**

1. 为什么不是“安全帽检测 Demo”
2. 模型能力边界与规则真值
3. 归一化危险区域和事件生命周期
4. 隐私优先的证据与运行审计
5. Streamlit 多语言演示
6. 测试、限制与下一步
