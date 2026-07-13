# Local model directory

Checkpoint binaries are intentionally ignored by Git. They retain their upstream model/dataset
license terms and can be large or deployment-specific.

[`registry.json`](registry.json) is the machine-readable source of truth for public capabilities,
installation commands, hashes, and redistribution status. The Web app exposes verified registry/local
profiles by default; set `UTILITY_SAFETY_TRUSTED_MODELS_ONLY=1` for hosted deployments to disable
arbitrary local checkpoint paths.

Default discovery order:

1. `ppe_yolo11n.pt`
2. `ppe_yolo11s.pt`
3. `yolo11n.pt`
4. Ultralytics runtime download of `yolo11n.pt`

Install the clean-clone general detector with:

```bash
utility-safety-ai fetch-model --model yolo11n.pt --output models
```

The command also writes an ignored `yolo11n.pt.json` containing the local hash and Ultralytics
version. A custom PPE checkpoint can be trained with the README recipe and promoted manually.

The 2026-07-13 engineering evaluation applies only to local PPE checkpoint SHA-256
`b05d39dba9d9a5a19855b9cc7dc4c613e979a64a280929e593522685c19cefff`. That binary is not part of
the source repository, and non-deterministic retraining will not reproduce it. See
[`docs/model-evaluation/ppe_yolo11n-v1`](../docs/model-evaluation/ppe_yolo11n-v1/README.md).

Review [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before redistributing a model or using
Ultralytics in a hosted, proprietary, or commercial context.
