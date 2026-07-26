# Railway person crop verifier v1

This amendment is the final local temporal experiment after
`railway-person-temporal-verifier-v1` ended at `DEVELOPMENT_FAIL`.

The detector, OC-SORT parameters, low-confidence predictions, tracks, frame
ordering, GT and prior results are immutable. The only new component is a
visual verifier over three real detector crops per track.

## Prospective comparison

```text
track_only  = fixed track-quality features
visual_only = mean frozen-backbone embedding of three crops
combined    = visual embedding + fixed track-quality features
```

All three heads are L2-regularized logistic regressions. Scaling and Platt
calibration are fitted on grouped support-scene OOF predictions only. A small
MLP is not part of the primary matrix and is released only if the logistic
triage passes.

## Crop selection

Only real detector hits are eligible. The slots are:

1. the second chronological detector hit (or the first if only one exists);
2. the maximum-confidence detector hit;
3. the latest detector hit.

When fewer than three distinct real frames exist, an existing real crop is
repeated deterministically and the number of unique frames is audited.
Interpolated boxes are never encoded.

## Leakage controls

- `grouped_scene_id` is the independent unit.
- Support, screening fold 0 and confirmation fold 1 stay disjoint.
- No held-out crop, normalization statistic, calibration target or GT label is
  used to fit the corresponding verifier.
- The selected model and one global threshold are frozen after fold 0 before
  confirmation fold 1 is read.
- Test is physically sealed.

## Gates

Two-fold triage requires Recall gain at least `0.10`, relative FN/frame
reduction at least `15%`, false-alarm growth at most `75%`, and F1 degradation
at most `0.03`.

Only a triage PASS releases 15-scene cross-fitted development evaluation. Its
final false-alarm limit is `20%`, with the additional worst-scene, improved
scene fraction and paired bootstrap requirements from the locked config.

Any FAIL closes the temporal direction as `CLOSED_NO_PRACTICAL_GATE`. It does
not authorize test access or further tuning under this protocol ID.

