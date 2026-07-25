# Person Canonical V5

`person-canonical-v5-range-aware-v1` is a prospective, development-only
protocol for improving person detection across independent OSDaR23 scenes.
Railway test images and labels remain sealed, and adversarial/T-norm experiments
remain blocked until a clean detector passes the complete OOF gate.

The protocol separates two primary interventions:

1. a YOLO11m P2 detection level, with Coordinate Attention as a separate
   ablation;
2. masked, perspective-aware person instance pasting using only the current
   fold's training scenes.

LiDAR geometry may provide a training-only distance group only when an RGB
person annotation is linked to the synchronized LiDAR object. Otherwise the
implementation records a `bbox_area_scale_fallback`; it never labels apparent
box size as geometric range.

Before candidate training, a common evaluator compares B0, CrowdHuman zero-shot,
D1-best and D1-last on folds 0 and 1. Gradual transfer is permitted only when
the frozen diagnostic rule is satisfied. Exact settings and hashes are in
[`configs/person_v5/protocol.yaml`](configs/person_v5/protocol.yaml).

