# User guide

## Monitoring workflow

1. Choose the bundled PPE model or the general person model.
2. Upload an image/video or select a camera/RTSP stream.
3. Choose a monitoring profile and configure one or more zones.
4. Run analysis and review annotated media, findings, and PPE status.
5. Add operator notes or download the complete report ZIP.

Start the Web console with:

```bash
utility-safety-ai web
```

## Monitoring profiles

| Profile | Use when |
|---|---|
| `balanced` | You want the default balance of stability and sensitivity |
| `high_precision` | You want fewer weak or transient alerts |
| `high_sensitivity` | You want to surface weaker evidence earlier |

Profiles tune confidence, overlap suppression, person-PPE association, tracking memory, and the
number of frames required to confirm a video alert.

## CLI workflows

```bash
# Image
utility-safety-ai infer-image --source image.jpg --zones zones.yaml --blur-faces

# Video
utility-safety-ai infer-video --source video.mp4 --zones zones.yaml --blur-faces

# Webcam
utility-safety-ai infer-camera --source 0 --max-frames 300 --blur-faces

# RTSP stream
utility-safety-ai infer-camera --source "rtsp://camera/stream" --blur-faces

# Convert an event log to CSV
utility-safety-ai export-report --events events.jsonl --format csv --output report.csv
```

Run `utility-safety-ai --help` or `utility-safety-ai <command> --help` for every option.

## Zone policies

The system checks the bottom-center point of each tracked person against polygon zones. Normalized
coordinates stay aligned when the input resolution changes.

```yaml
zones:
  - id: restricted_area
    name: Restricted Area
    coordinate_space: normalized
    risk_level: high
    min_dwell_seconds: 0.5
    required_ppe: [helmet, vest]
    polygon:
      - [0.15, 0.40]
      - [0.85, 0.40]
      - [0.90, 0.95]
      - [0.10, 0.95]
```

Pixel-coordinate zones are also supported. Recheck the zone whenever a camera is moved.

## Findings and risk levels

The rule engine combines model detections, person-PPE association, and zone entry into four risk
levels: `low`, `medium`, `high`, and `critical`. Video findings are confirmed across consecutive
frames, and a cooldown prevents the same tracked person from generating an event every frame.

## Privacy controls

Enable `--blur-faces` in the CLI or keep **Privacy blur** enabled in the Web console. Choose Gaussian
blur, pixelation, or solid redaction. The system does not perform face recognition and track IDs last
only for the current stream.

## Run history and downloads

Successful runs appear in the Web console's **Run history** tab. Each run contains annotated media,
event snapshots, CSV/JSONL records, a summary, processing indicators, and a manifest with hashes.
`latest.json` always points to the latest completed run.
