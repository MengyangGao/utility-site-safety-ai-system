# Field-pilot validation plan

This project is an engineering prototype. A field pilot is a validation exercise under human
supervision, not authorization to automate safety decisions.

## Dataset and split design

- Collect consented, policy-approved footage from every intended camera position, shift, weather and
  lighting regime.
- Split by site, camera and recording session. Adjacent frames from one clip must never cross the
  train/validation/test boundary.
- Balance every positive and explicit-negative PPE class, especially `no_vest`, `no_boots`, gloves
  and goggles. Retain hard negatives such as caps, ordinary clothing, reflections, tools and partial
  occlusions.
- Maintain a frozen, hash-addressed test split and a separate adverse-condition challenge set.

## Release gates

The owner must choose thresholds with qualified safety stakeholders before a pilot. At minimum,
report per class and per camera:

1. detection precision, recall and mAP;
2. event-level precision and recall after tracking/rules;
3. false alerts per camera-hour and missed hazards per reviewed hour;
4. track fragmentation and ID-switch rate;
5. face-redaction recall under representative pose, scale and lighting;
6. end-to-end FPS, p95 latency, memory, output growth and reconnect recovery time.

No aggregate score may hide a class with zero recall. Failed examples and adjudication notes must be
retained alongside the numeric report.

## Staged rollout

1. Offline replay with no notifications.
2. Shadow-mode live run visible only to the evaluation team.
3. Human-reviewed notifications with acknowledgement and false-positive labeling.
4. Limited operational pilot after privacy, cybersecurity, licensing and safety reviews.

Every stage needs a rollback owner, incident procedure, retention period and signed acceptance
record. The system must never be described as replacing a safety officer.
