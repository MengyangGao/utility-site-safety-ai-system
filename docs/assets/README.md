# Release media evidence

These images were generated from completed, privacy-enabled v1 acceptance runs on 2026-07-13.
They are documentation evidence, not claims of production accuracy.

| File | Source run | Detector | Source-media terms |
|---|---|---|---|
| `annotated-zone.jpg` | `clean-image` | General YOLO11n, SHA `0ebbc80d…` | PickPik CC0 source recorded in `examples/assets.yaml` |
| `annotated-ppe-zone.jpg` | `ppe-image` | Local PPE YOLO11n, SHA `b05d39db…` | Construction-PPE / AGPL-3.0-derived media |
| `annotated-video-frame.jpg` | `ppe-video`, frame near 2 s | Local PPE YOLO11n, SHA `b05d39db…` | Pexels source, This Viktọ |
| `annotated-video-frame-2.jpg` | `ppe-video`, frame near 3.5 s | Local PPE YOLO11n, SHA `b05d39db…` | Pexels source, This Viktọ |
| `web-dashboard.jpg` | Final Streamlit visual smoke | Local PPE model selected; not loaded | Original project UI |

The source run produced 120 annotated frames, 468 detections, five cooldown-filtered zone events,
five privacy-processed evidence snapshots, and a completed artifact manifest. Runtime output remains
under ignored `outputs/`; the two selected frames are retained here for the public project page.

Before replacing release media:

1. use only sources recorded in `examples/assets.yaml`;
2. keep privacy blur enabled;
3. verify the depicted model hash and completed run manifest;
4. remove credentials, personal host paths, tokens, and unrelated browser content;
5. update this registry with the source run and applicable media terms.

The repository MIT license does not relicense Pexels, Construction-PPE, or model-derived media.
