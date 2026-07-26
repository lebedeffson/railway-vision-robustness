# Temporal verifier v1

Protocol: `railway-person-temporal-verifier-v1`

- Parent temporal-safety result stayed immutable at commit `2239ddd`.
- Implementation commit: `c619681892853d4a793aabc9bd086a5083782248`.
- Primary tracker was prospectively fixed to OC-SORT; ByteTrack was sensitivity.
- The verifier preserves all original frame-detector outputs and filters only
  temporal-confirmed/interpolated additions.
- Support, screening fold 0 and confirmation fold 1 are scene-disjoint.
- Twelve frozen rule candidates were evaluated first. The selected rule
  (`K=3`, `W=3`, high confidence `0.25`) failed the Recall, FN, false-alarm and
  improved-scene checks.
- Nested L2 logistic regression used support-scene GroupKFold OOF scores,
  train-only scaling and Platt calibration. One global threshold (`0.17`) was
  selected on fold 0 before confirmation fold 1.

Final two-fold OC-SORT logistic effects:

| metric | effect |
|---|---:|
| delta Recall | +0.129282 |
| relative FN/frame reduction | 17.57% |
| relative false-alarm increase | 325.11% |
| delta F1 | -0.059607 |
| improved scenes | 4/6 |
| worst-scene Recall delta | 0.0 |

Paired Recall bootstrap CI was `[0.02585, 0.14731]`, but the frozen false-alarm,
F1 and improved-scene checks failed. Status is `DEVELOPMENT_FAIL`; test access
count remains zero. Do not retune rule/logistic thresholds under this protocol.
The next allowed step is a crop verifier under a new amendment.

The false-track audit found 288 unambiguous false support tracks: 89.58% had no
confidence at or above 0.25 and 33.33% lasted only one or two frames. The private
crop gallery is excluded from public bundles.
