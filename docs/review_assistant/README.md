# Railway Person Review Assistant

Local operator-in-the-loop MVP for reviewing potential person appearances in
railway videos.

## Safety contract

```text
PRODUCT_STATUS = OPERATOR_ASSISTANT_MVP
AUTONOMOUS_ALARMING = DISABLED
SAFETY_ACTUATION = DISABLED
HUMAN_CONFIRMATION_REQUIRED = true
```

The application does not issue alarms or control railway equipment. It creates
a compact review queue from frozen detector and temporal outputs. The operator
must label each event as person, false positive, uncertain, or skipped.

## Installation

```bash
./scripts/review_assistant/install.sh
```

Models are never downloaded. Start the local UI with explicit frozen artifacts:

```bash
./scripts/review_assistant/run.sh \
  --checkpoint /local/models/person_detector.pt \
  --verifier /local/models/combined.pkl \
  --encoder /local/models/verifier_encoder.pt
```

Expected SHA-256 values are printed when a model is missing or incompatible.
For conservative review only, the temporal verifier paths can be left empty.

## CLI processing

```bash
python scripts/review_assistant/process_videos.py \
  --input /local/video-or-folder \
  --mode combined_queue \
  --camera-id camera-01 \
  --checkpoint /local/models/person_detector.pt \
  --verifier /local/models/combined.pkl \
  --encoder /local/models/verifier_encoder.pt
```

Modes:

- `conservative_review`: frozen frame baseline only;
- `high_recall_review`: frozen OC-SORT and combined verifier;
- `combined_queue`: baseline and temporal-only evidence in one queue.

## Event semantics

Candidates join one event when they belong to the same camera, occur within
three seconds, and overlap or have nearby centers. Track-ID changes do not
create duplicate events. A closed event reopens for up to 30 seconds when the
same spatial region becomes active again.

Every event has:

```text
data/runs/<run_id>/events/<event_id>/
  clip.mp4
  thumbnail.jpg
  event.json
  detections.csv
```

Clips include three seconds before and after the event. Interpolated detections
remain explicitly marked.

## Review workflow

The Streamlit application provides:

- processing progress, FPS and ETA;
- normal and `Muted` queues;
- source labels `BASELINE`, `TEMPORAL_ONLY`, and `BOTH`;
- reversible operator decisions and comments;
- keyboard controls `1`, `2`, `3`, `Space`, `N`, and `P`;
- manual, camera-specific suppression rules;
- HTML/PDF and CSV exports.

Suppression rules never hide events. They lower queue priority and move stable
repetitions to `Muted`. A changed box, confidence or motion raises the event
again.

## Reports

```bash
python scripts/review_assistant/generate_report.py --run-id R-...
```

The primary operational metric is `review minutes per video hour`. Reports
also contain unique events per hour, raw detections per event, confirmed
temporal-only people, and false events per hour.

## Long benchmark

```bash
python scripts/review_assistant/benchmark_long_video.py \
  --input development_sample.mp4 \
  --output outputs/review_assistant/benchmark \
  --checkpoint /local/models/person_detector.pt \
  --verifier /local/models/combined.pkl \
  --encoder /local/models/verifier_encoder.pt \
  --frames 1000 --encode
```

The benchmark requires at least 1,000 post-warm-up frames or a ten-minute
video. It is a throughput measurement, not accuracy evidence. Offline
processing is supported and no real-time claim is made.

## Privacy and local storage

Video, clips, thumbnails and SQLite decisions stay local. Public release
bundles exclude all models, videos, crops, datasets, local databases, sealed
test material and absolute paths.
