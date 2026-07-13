# Model Card

## Scope

Utility Site Safety AI is model-agnostic at its internal `Detection` boundary but currently ships an Ultralytics YOLO adapter. This card distinguishes the default general detector from optional custom PPE checkpoints.

One local checkpoint has retained engineering-evaluation evidence below. It is not distributed in
Git and is not approved for field safety decisions.

## Model variants

### General clean-clone detector

| Field | Value |
|---|---|
| Default name | `yolo11n.pt` |
| Local pinned path | `models/yolo11n.pt` |
| Source | Ultralytics YOLO11 COCO-pretrained checkpoint |
| Application capability | Person detection, temporary tracking, zone intrusion, privacy fallback, audit outputs |
| PPE capability | None unless the checkpoint's class map actually contains PPE classes |
| Reference device | CPU-compatible; MPS/CUDA optional |

The application may receive other COCO classes from the model, but safety zone logic uses `person`. It must not advertise PPE compliance from this checkpoint.

### Custom PPE detector

Suggested local names:

- `models/ppe_yolo11n.pt` — small, laptop-oriented custom checkpoint;
- `models/ppe_yolo11s.pt` — optional larger alternative.

The application recognizes these normalized PPE names when the model exposes them:

```text
person
helmet, vest, gloves, boots, goggles
no_helmet, no_vest, no_gloves, no_boots, no_goggles/no_goggle
```

The Construction-PPE dataset itself has no `no_vest` class, so a model trained only on that source cannot learn a dedicated missing-vest label.

## Intended use

- Educational computer-vision engineering
- Open-source portfolio and technical demonstrations
- Offline review of licensed/synthetic worksite media
- Controlled proof-of-concept experiments with human review
- Research on model/geometry/rule integration

## Out-of-scope use

- Certified safety interlocks or autonomous shutdown decisions
- Replacing safety officers or formal inspections
- Automated discipline, worker scoring, payroll, or employment decisions
- Face recognition, identity tracking, or cross-camera re-identification
- Covert surveillance
- Unreviewed real-time alerting in a hazardous operation
- Production deployment without site validation, security/privacy controls, and license review

## Input and output

Input is a BGR image/frame or supported path accepted by Ultralytics. The wrapper emits normalized detections:

```text
class_id: int
class_name: lowercase string
confidence: float
bbox: x1, y1, x2, y2 in source pixels
track_id: temporary int or null
```

Application decisions are not raw model outputs. They also depend on class thresholds, tracking, person-PPE association, zone geometry, dwell, and cooldown.

## Confidence policy

Default post-model floors are:

| Class | Floor |
|---|---:|
| person | 0.30 |
| helmet | 0.35 |
| vest | 0.30 |
| gloves | 0.15 |
| boots | 0.20 |
| goggles | 0.20 |
| no_helmet | 0.30 |
| no_vest | 0.30 |
| no_gloves | 0.15 |
| no_boots | 0.20 |
| no_goggles / no_goggle | 0.20 |

These are engineering defaults, not calibrated probabilities or universal operating points. The
user's global confidence is a minimum for every class; a listed class may require a stricter floor.
Ultralytics receives the global threshold, and the adapter then applies the per-class maximum.

## Decision semantics

- Only positive and explicit negative boxes associated to a detected person participate in PPE rules.
- Unassociated PPE boxes remain in detection audit logs but cannot emit a person-level event.
- Positive evidence overrides contradictory same-type negative evidence.
- Repeated same-type negatives produce one resolved PPE violation.
- Absence of a positive box produces `unknown`, never an automatic `no`.
- Two or more distinct explicit PPE violation types can escalate to high risk.
- An active missing-helmet finding with a mature zone intrusion escalates the zone event to critical.

These choices reduce false accusations but may miss true non-compliance when the negative detector fails.

## Evaluation contract

A publishable PPE model evaluation must retain:

- checkpoint path and SHA-256;
- base checkpoint and license;
- Ultralytics/PyTorch/package versions;
- dataset identifier/version/license and exact split;
- class map and instance counts;
- seed, image size, epochs, batch, optimizer/augmentation configuration;
- hardware and precision mode;
- aggregate and per-class precision, recall, mAP50, and mAP50-95;
- confusion matrices and PR/F1/precision/recall curves;
- representative false-positive/false-negative galleries;
- application-level person association, zone, event, and privacy checks;
- benchmark warm-up, iterations, resolution, device, and raw JSON.

### Current local engineering result

Checkpoint SHA-256 `b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff`
was independently evaluated on the 143-image Construction-PPE validation split: precision 0.6903,
recall 0.5515, mAP50 0.5786, and mAP50-95 0.2860. Explicit negative classes are substantially
weaker; `no_boots` recall was zero on only four validation instances. Detector-only Apple M4 Pro
timings and the complete per-class table are retained in the
[`ppe_yolo11n-v1 evidence directory`](model-evaluation/ppe_yolo11n-v1/README.md).

The exact weight is local-only and training was non-deterministic, so retraining will not reproduce
this hash. These metrics are evidence for the recorded checkpoint, not a promise about a clean clone,
a newly trained model, another site, or end-to-end pipeline throughput.

The machine-readable `utility-safety-ai model-gate` command rejects this checkpoint under the sample
field thresholds (recall 0.65, mAP50 0.60 and required-class recall 0.50): aggregate recall and mAP50
are below threshold, `no_helmet` recall is below threshold, `no_boots` recall is zero, and `no_vest`
is missing. This deliberate failure prevents a portfolio model from being silently promoted as
field-ready. See [`field-pilot-plan.md`](field-pilot-plan.md) for the data and acceptance work needed
to pass a future gate.

## Expected failure modes

- Small/distant PPE occupies few pixels.
- Gloves/boots are occluded or confused with hands/footwear/background.
- Negative PPE labels are visually ambiguous and less abundant.
- Crowding causes PPE-person association errors.
- Backlighting, glare, rain, night scenes, motion blur, compression, and camera shifts change the domain.
- PPE style, color, cultural/site practice, and camera viewpoint differ from training data.
- A face detector or privacy fallback may miss identity-revealing regions.
- Track IDs can change after occlusion or scene cuts.

## Bias and fairness considerations

Performance can vary with body size, clothing, skin tone, PPE design, occupation/task, mobility aids, camera placement, and environmental conditions. Aggregate mAP is insufficient to establish equitable behavior. Any field evaluation should stratify failure analysis across relevant site conditions and ensure that alerts are reviewed rather than used as automatic judgments about individuals.

## Operational monitoring

A future pilot should monitor:

- detection and event rate drift by camera/time/condition;
- `unknown` PPE rate;
- track fragmentation;
- per-zone occupancy duration;
- human-confirmed false-positive/false-negative categories;
- privacy misses;
- model hash/config changes;
- camera movement and zone alignment.

## Licensing

Ultralytics documents YOLO11 software/models under AGPL-3.0 with separate Enterprise options: <https://docs.ultralytics.com/models/yolo11/> and <https://www.ultralytics.com/license>.

A custom checkpoint may also be affected by its dataset and other input licenses. The root project-code MIT license does not relicense a YOLO checkpoint. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
