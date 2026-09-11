# Architecture

The existing YOLO adapter, person-PPE association, rule engine and three media pipelines are
preserved. The modernization adds bounded I/O and delivery around those components.

## Capture and continuity

`FrameCapture` owns a live OpenCV capture in one producer thread. Network captures select FFmpeg
and set its open-only timeout properties during construction. RTSP uses TCP unless the operator
has already configured `OPENCV_FFMPEG_CAPTURE_OPTIONS`. FFmpeg decoding uses one thread to avoid
large decode queues on a small edge host. Open and read timeouts default to five seconds.

The mailbox retains one frame. Slow inference drops backlog rather than allowing latency to grow
without bound. Local file replay stays sequential. Frame age measures time since local decode,
not camera-to-host latency; upstream camera buffers are outside this measurement.

A recovered connection increments `stream_segment` and resets detection tracking, fallback
tracking and rule state. Track counts and compliance records include that segment. Resolution
changes end the run with an explicit failure so an old zone policy is not silently misapplied.
The default camera duration is five minutes; supervise bounded runs for longer monitoring.

## Events and evidence

`RunArtifacts` handles snapshots, CSV/JSONL, compliance changes, lifecycle records and the event
index. Stable `incident_id` metadata links an observation across frames and repeated emissions.
A disappearance creates a resolved-observation record; ending a run or losing a stream creates
an interrupted-observation record. Neither means that the physical worksite is safe.

Redaction precedes snapshots, annotations and persisted media. With FFmpeg installed, annotated
video is converted to H.264 with fast-start metadata before the manifest hashes are finalized.
Without FFmpeg, the run explicitly records MPEG-4 browser incompatibility and the UI offers the
report for a desktop player. Live output contains processed frames at nominal input FPS and can
be time-compressed; `frame_timestamps.jsonl` records actual host timing.

Only completed runs update the latest pointer. Premature file decoding, failed codecs, failed
required writes and exhausted stream retries fail the run. Partial evidence remains available
for diagnosis. Intentional `--overwrite` retains the original transactional replacement behavior.
Prefer new run IDs for retained evidence.

## Delivery and review

SQLite stores a sanitized event index, an outbox, delivery attempts and append-only review rows.
Recording an event and queueing its targets is one transaction. Network workers never perform
inference or touch raw frames. Leases let a new process reclaim an interrupted delivery; bounded
backoff ends in a visible dead-letter state. Delivery is at least once. Receivers must deduplicate
stable event identifiers/idempotency keys; exactly-once delivery is not claimed.

Reviews do not rewrite original CSVs, snapshots or manifest hashes. The report ZIP adds a review
history snapshot from the database. This is an application-level append-only audit trail, not
a tamper-proof ledger or an independently authenticated reviewer identity system.

## Scope

The dashboard remains Streamlit. REST and MQTT dependencies are optional; SQLite, threads and
small callables handle delivery without a task broker. There is no face recognition, automated
actuation, multi-camera orchestration service or worker-performance scoring. Operational metrics
report processing and association coverage, not model accuracy.

References: [OpenCV capture properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html),
[Ultralytics tracking](https://docs.ultralytics.com/modes/track/),
[Paho MQTT acknowledgement](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).
