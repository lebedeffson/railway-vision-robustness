# Canonical v6 person temporal T-norm protocol

`canonical-v6-person-temporal-tnorm-v1` is a prospective development-only
experiment. It does not alter B0 weights and cannot open railway test or
adversarial attacks.

T0 uses the already saved low-floor (`confidence >= 0.001`) B0 detections.
Temporal windows are causal and contain at most the current plus four previous
frames. Windows never cross a `subsequence_id`; `grouped_scene_id` remains the
independent unit for folds and claims.

The four frozen ablations are:

- T0-A: raw B0;
- T0-B: ByteTrack-style high/low-confidence association and short gap filling;
- T0-C: Bayesian track-existence fusion with arithmetic reliability;
- T0-D: the same Bayesian fusion with Product T-norm reliability.

Global camera motion is estimated only between consecutive development frames
with ORB features and a RANSAC homography. Detection regions are masked when
estimating background motion. A transform below the frozen match/inlier gate
contributes exactly zero temporal evidence. Missing detections may be propagated
for no more than two frames and only after two matched observations.

T0 uses a deterministic RoI appearance descriptor composed of normalized RGB
and gradient histograms. Learned P3/P4 aggregation and pre-NMS score heatmaps
belong to T1 and are explicitly blocked until T0-D passes.

All temporal parameters are frozen in
`configs/canonical_v6_person_temporal_tnorm.yaml` before fold 0. Fold 0 is the
development decision. Fold 1 is not read by the T0 runner unless T0-D passes
every frozen fold-0 improvement gate. Test remains physically sealed.

