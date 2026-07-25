# Canonical v6 temporal T-norm T0 final report

Status: `T0_FAIL`

Protocol: `canonical-v6-person-temporal-tnorm-v1`

Implementation commit: `258a039831d20f0d1278d0d2037d429e2dde7575`

Protocol lock commit: `1f22948`

## Claim boundary

This is development fold-0 screening evidence, not article evidence. Fold 1 was
not read because the frozen fold-0 gate failed. Railway test remained sealed,
attacks remained blocked and the trainable T1 temporal head was not started.

## Results

| Variant | mAP50 | Recall | Small Recall | FP/frame |
|---|---:|---:|---:|---:|
| T0-A B0 | 0.251139 | 0.201288 | 0.146559 | 0.885496 |
| T0-B ByteTrack-style | 0.250379 | 0.306763 | 0.259160 | 4.458015 |
| T0-C Bayesian | 0.256573 | 0.205314 | 0.151028 | 0.908397 |
| T0-D Bayesian + Product | 0.256394 | 0.204509 | 0.150134 | 0.908397 |

T0-D versus B0:

- delta mAP50: `+0.005254`;
- delta Recall: `+0.003221`;
- delta small Recall: `+0.003575`;
- relative FP/frame: `1.025862`.

The frozen requirements were delta Recall at least `+0.05`, delta small Recall
at least `+0.08`, delta mAP50 at least `-0.02`, FP/frame growth at most 25% and
fold Recall at least `0.25`. T0-D failed the Recall, small-Recall and absolute
fold-Recall checks.

T0-B demonstrates that temporal propagation can recover detections, but its
FP/frame increased by approximately five times. This is not an accepted
positive result.

## Integrity

- evaluator consistency: PASS;
- lost GT: 0;
- sequence leakage: 0;
- homography pairs: 125/125 valid;
- minimum homography inlier ratio: 0.6162;
- median homography inlier ratio: 0.8426;
- test access: 0;
- fold 1: `SKIPPED_BY_FOLD_0_GATE`;
- T1: BLOCKED.

