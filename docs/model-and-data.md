# Models, data, and evaluation

## Source policy

All committed third-party media and generated evaluation artifacts must trace to an approved
open-source or public-domain source. The authoritative record is
[`legal/provenance.yaml`](legal/provenance.yaml); its paths, SPDX-style license identifiers, and
SHA-256 values are checked by CI.

Currently committed external material comes only from **Ultralytics Construction-PPE 1.0.0**, which
the official dataset documentation identifies as AGPL-3.0. Pexels and unverifiable stock sources
are intentionally excluded.

## Model variants

### General detector

`yolo11n.pt` is the clean-clone default. It supports person detection, temporary tracking, zone
intrusion, privacy fallback, and audit output. It does not support PPE unless its class map exposes
the required PPE classes.

### Optional PPE detector

Local custom checkpoints may expose:

```text
person
helmet, vest, gloves, boots, goggles
no_helmet, no_vest, no_gloves, no_boots, no_goggles/no_goggle
```

The application never turns a missing positive detection into a violation. Only an explicit,
person-associated negative class can produce a missing-PPE finding. Contradictory positive evidence
wins, and uncertain observations remain `unknown`.

Checkpoint binaries are ignored by Git. [`models/registry.json`](../models/registry.json) records
the public capability and provenance boundary.

## Construction-PPE

| Field | Value |
|---|---|
| Source | [Official Ultralytics documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| License | AGPL-3.0-only |
| Images | 1,416 |
| Train / validation / test | 1,132 / 143 / 141 |
| Classes | 11 |
| Local configuration | [`datasets/data.yaml`](../datasets/data.yaml) |

The class map has `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots`, but no `no_vest`. It is
also not a utility-site benchmark: public construction imagery may not represent substations,
electrical cabinets, night work, weather, local PPE standards, or each target camera.

## Training and evaluation

```bash
utility-safety-ai train \
  --data construction-ppe.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640

python scripts/validate_ppe_model.py \
  --model models/ppe_yolo11n.pt \
  --data datasets/data.yaml \
  --output outputs/validation/ppe_yolo11n

utility-safety-ai model-gate \
  --metrics docs/model-evaluation/ppe_yolo11n-v1/metrics.json \
  --required-class no_helmet \
  --required-class no_vest \
  --required-class no_boots
```

A publishable evaluation must retain the base model, checkpoint SHA-256, dataset archive and split,
class map/counts, training configuration, software environment, per-class metrics, plots, runtime
scope, and representative failure analysis. A result never transfers to a newly trained weight.

## Retained engineering result

The local-only checkpoint with SHA-256
`b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff` produced:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.6903 | 0.5515 | 0.5786 | 0.2860 |

The model gate rejects it: aggregate recall and mAP50 are below the sample thresholds,
`no_helmet` is weak, `no_boots` recall is zero on four validation instances, and `no_vest` is absent.
The checkpoint is suitable only as retained engineering evidence, not a field-safety claim. Raw
artifacts are under [`model-evaluation/ppe_yolo11n-v1/`](model-evaluation/ppe_yolo11n-v1/).

## Field validation gate

Before a supervised pilot, create consented data separated by site, camera, and recording session;
never split adjacent frames across train and test. Report per camera and class:

1. detection precision, recall, and mAP;
2. event precision/recall after association, tracking, and rules;
3. false alerts per camera-hour and missed hazards per reviewed hour;
4. track fragmentation and ID-switch rate;
5. privacy-redaction recall;
6. end-to-end FPS, p95 latency, memory, storage growth, and reconnect recovery.

No aggregate number may hide a class with zero recall. Roll out only through offline replay, shadow
mode, human-reviewed notifications, and a limited pilot with rollback, privacy, security, licensing,
and safety approvals.

## Known risks

Small PPE, occlusion, glare, rain, motion blur, camera shift, clothing variation, and domain shift can
all degrade results. Performance may vary with body size, skin tone, mobility aids, PPE design, task,
and camera placement. Automated employment or disciplinary use is out of scope.
