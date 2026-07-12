# Dataset and Example-Asset Card

## Overview

The repository does not require a PPE dataset for clean-clone person + zone inference. PPE fine-tuning optionally uses the Ultralytics Construction-PPE dataset. A small set of example media is committed for demos and tests; its machine-readable provenance is [`examples/assets.yaml`](../examples/assets.yaml).

## Construction-PPE

According to the official [Ultralytics Construction-PPE documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe/):

| Field | Value |
|---|---|
| Dataset | Construction-PPE |
| Version/citation year | 1.0.0 / 2025 in the upstream citation |
| Images | 1,416 |
| Train / validation / test | 1,132 / 143 / 141 images |
| Classes | 11 |
| Download | Ultralytics dataset alias `construction-ppe.yaml` |
| Upstream license | AGPL-3.0 |

Upstream class map:

```text
helmet, gloves, vest, boots, goggles, none, Person,
no_helmet, no_goggle, no_gloves, no_boots
```

Important consequences:

- There is no dedicated `no_vest` class.
- `none` is ignored by the application.
- `Person` is normalized to lowercase `person` by the detector adapter.
- Missing-PPE classes are explicit labels; an absent positive annotation must not be treated as a negative label.

The repository carries `datasets/data.yaml` and `datasets/LICENSE` as configuration/license material. Dataset images may be downloaded by Ultralytics outside the repository. Users are responsible for storage, redistribution, attribution, and license compliance.

## Intended dataset use

- Fine-tuning a small object detector for engineering research/demos
- Studying positive and explicit negative PPE classes
- Creating validation artifacts for the exact trained checkpoint
- Testing the integration among detections, person association, zones, and rules

## Dataset limitations

- Public construction imagery may not match substations, cabinets, utility poles, renewable-energy sites, or local PPE standards.
- Negative PPE classes can be imbalanced and visually ambiguous.
- Small gloves, goggles, and boots are sensitive to resolution and occlusion.
- Class definitions do not encode whether PPE is correctly worn, fastened, certified, or appropriate to a specific task.
- Dataset labels do not provide camera calibration, job role, authorization, or full site policy.
- Image-level train/validation/test separation does not by itself prove independence by site, worker, camera, or capture sequence.
- Upstream documentation is not evidence of performance for a locally trained checkpoint.

## Example media inventory

The manifest currently records:

| Asset group | Provenance | Terms/status |
|---|---|---|
| `construction_zone_01.jpg` | Verified PickPik source page | CC0-1.0 claim in manifest; primary clean-clone demo |
| `solar_inspection_pexels_4254172.jpg` | Verified Pexels source page; Gustavo Fring | Pexels License; suitable alternate verified demo |
| `construction_site_ppe_01.jpg` | Construction-PPE validation sample | AGPL-3.0; optional PPE demo |
| `construction_worker_gloves_01.jpg` | Construction-PPE validation sample | AGPL-3.0; optional PPE demo |
| `construction_rebar_pexels_10294768.mp4` | Verified Pexels source page; This Viktọ | Pexels License; primary real-video demo |
| `electrical_engineer_01.jpg` | Verified Pexels source page; Fatih Yurtman | Pexels License; electrical-panel maintenance demo |
| `solar_farm_01.jpg` | Verified Pexels source page; ThisIsEngineering | Pexels License; rooftop solar demo |
| Generated `.mp4` demos | Derived by `scripts/make_demo_assets.py` | Inherit relevant source-image terms |

Refer to the manifest for exact SHA-256 values and paths. A generated derivative does not erase source license, privacy, publicity, or endorsement constraints.

## Provenance policy

Every new asset must record:

1. repository-relative path;
2. SHA-256;
3. exact source page;
4. author where known;
5. explicit license/terms;
6. derivation details;
7. provenance status.

An asset with a missing exact source page should be replaced or reverified before it is featured in release media.

## Data preparation

For a custom site dataset:

- define classes and labeling instructions before annotation;
- separate sites/cameras/sequences across splits to reduce leakage;
- retain source consent and data-processing authority;
- minimize personal data and define retention/deletion;
- balance normal, compliant, non-compliant, and difficult negative examples;
- audit tiny-object label quality;
- document excluded/corrupt/duplicate samples;
- version raw data, labels, split manifests, and transformations;
- never commit real private worksite images to a public repository by default.

Example YOLO YAML:

```yaml
path: /absolute/or/resolved/path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: helmet
  2: vest
  3: gloves
  4: boots
  5: goggles
  6: no_helmet
  7: no_vest
  8: no_gloves
  9: no_boots
  10: no_goggles
```

The class order must match the checkpoint. Do not copy this illustrative order over an upstream dataset with a different map.

## Evaluation and release checks

- Verify asset hashes against `examples/assets.yaml`.
- Record exact split and class counts.
- Inspect labels and predictions visually.
- Report per-class, not only aggregate, metrics.
- Evaluate application-level event behavior and privacy.
- Document domain shift and known unsupported site conditions.
- Review dataset/model licenses together before distributing weights.

## Ethical and privacy considerations

Worksite media may reveal faces, health/mobility information, location, employer, job role, shift patterns, or unsafe behavior. A license to use an image is not necessarily consent for employee monitoring or automated discipline. Use this dataset workflow for engineering evaluation with appropriate governance and human review.

CC0 and stock-media terms do not eliminate privacy/publicity rights or permit implied endorsement. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
