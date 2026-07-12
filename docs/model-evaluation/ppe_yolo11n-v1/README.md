# PPE YOLO11n v1 evaluation

This directory retains auditable local evaluation evidence for checkpoint SHA-256
`b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff`.
The checkpoint itself is intentionally excluded from Git. Running the documented non-deterministic
training recipe creates a new, non-identical checkpoint; published metrics apply only to the exact
hash above.

## Independent validation result

The promoted checkpoint was evaluated after training on the 143-image, 1,172-instance
Construction-PPE validation split using CPU inference.

| Metric | Result |
|---|---:|
| Precision | 0.6903 |
| Recall | 0.5515 |
| mAP50 | 0.5786 |
| mAP50-95 | 0.2860 |

Strong mAP50 classes were `Person` (0.9026), vest (0.8635), helmet (0.8448), gloves
(0.8184), goggles (0.7748), and boots (0.7672). Explicit negative classes were much weaker:
`no_helmet` 0.3998, `no_goggle` 0.1986, `no_gloves` 0.1835, and `no_boots` 0.0285.
The validation split contains only four `no_boots` instances, and recall for that class was zero.
There is no `no_vest` class in this dataset.

These numbers support an engineering demonstration, not a safety deployment claim. The weak and
imbalanced explicit-negative classes are the primary model-quality bottleneck; the application
therefore reports missing PPE only from an explicit negative detection and treats absence as
`unknown`.

## Detector-only benchmark

The benchmark used one deterministic 640×640 synthetic noise frame, five warm-up calls, and 30
timed `detector.predict` calls per device on an Apple M4 Pro.

| Device | FPS | ms/frame |
|---|---:|---:|
| CPU | 28.40 | 35.21 |
| Apple MPS | 117.95 | 8.48 |

The JSON file is authoritative because reruns vary. This is not end-to-end throughput: it excludes
media decode, tracking, rules, privacy blur, annotation, logging, and output encoding.

## Artifact index

- [`metadata.json`](metadata.json) — checkpoint, dataset, command, environment, and split identity
- [`metrics.json`](metrics.json) — aggregate and per-class validation metrics
- [`benchmark.json`](benchmark.json) — raw CPU/MPS timing evidence and scope
- [`training-args.yaml`](training-args.yaml) — sanitized Ultralytics training configuration
- [`training-results.csv`](training-results.csv) — per-epoch training/validation curve
- [`confusion-matrix-normalized.png`](confusion-matrix-normalized.png)
- [`confusion-matrix.png`](confusion-matrix.png)
- [`pr-curve.png`](pr-curve.png)
- [`f1-curve.png`](f1-curve.png)
- [`precision-curve.png`](precision-curve.png)
- [`recall-curve.png`](recall-curve.png)
- [`validation-labels.jpg`](validation-labels.jpg) and
  [`validation-predictions.jpg`](validation-predictions.jpg) — one label/prediction batch pair

The single retained prediction batch and public validation split do not replace a site-specific
false-positive/false-negative gallery. That work remains required before any field pilot.

## License and provenance boundary

The base YOLO11 checkpoint and Ultralytics runtime are offered under AGPL-3.0 with separate
Enterprise terms available. Construction-PPE is documented by Ultralytics as AGPL-3.0. The two
validation mosaics in this directory are derived from that dataset and are not relicensed by the
repository's MIT license. The exact individual source filenames used inside Ultralytics' saved
batch mosaic were not retained by the current validation helper; this is an evidence limitation.
See [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) before redistributing weights or
validation media.
