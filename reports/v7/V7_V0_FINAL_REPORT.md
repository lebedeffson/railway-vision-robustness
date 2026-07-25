# Canonical v7 V0 final report

Protocol: `canonical-v7-person-tracklet-verifier-v1`  
Implementation commit: `e12d186b0e21f867e5c482d7beb536a925bfd3c0`  
Protocol-lock commit: `42e9125`  
Config SHA-256: `6ac5a59fe1f1d75c347689d60b4cf04c35f3a5792c391543b99d1ec4c6bdc59d`

## Status

```text
V0_GATE: FAIL
fold 1: BLOCKED / unread
crop verifier: eligible only through a new prospective lock
railway test: SEALED
attacks: BLOCKED
```

V0 was trained and calibrated from the 12 fold-0 train scenes only. The
high-recall proposals produced 990 train tracklets: 190 positive, 394
negative and 406 ambiguous. Ambiguous tracklets were excluded from fitting.
Model selection and standard/safety thresholds used five-fold grouped OOF
scores by `grouped_scene_id`.

The train-only OOF rule selected
`V0-B_monotone_hist_gradient_boosting` before held-out fold 0 was read.

## Held-out fold 0

| Method / point | mAP50 | Precision | Recall | F1 | Small Recall | FP/frame |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.25114 | 0.68306 | 0.20129 | 0.31095 | 0.14656 | 0.88550 |
| V0-A standard | 0.24415 | 1.00000 | 0.07407 | 0.13793 | 0.01877 | 0.00000 |
| V0-A safety | 0.24415 | 0.83408 | 0.14976 | 0.25392 | 0.09026 | 0.28244 |
| V0-B standard | 0.22048 | 0.99091 | 0.08776 | 0.16124 | 0.02681 | 0.00763 |
| V0-B safety | 0.22048 | 0.55858 | 0.16506 | 0.25482 | 0.10456 | 1.23664 |

The frozen standard point failed mAP50, Recall, small Recall and F1 relative
to B0. The safety point also remained below B0 Recall and small Recall.
Consequently V0 did not solve the scene-transfer problem and fold 1 was not
opened.

## Integrity

- verifier-train and held-out scene intersection: empty;
- lost GT: 0;
- NaN/Inf: 0;
- test access: 0;
- fold-1 access: 0.

Reparsing the selected prediction CSV reproduced every operating-point metric
exactly. AP differed by `9.9251e-6` because equal calibrated scores changed
ordering after decimal CSV serialization; the frozen `1e-12` parity tolerance
therefore marked evaluator parity `FAIL`. This integrity failure reinforces
the stop decision but is not the reason the scientific gate failed: the direct
in-memory metrics were already below all primary quality thresholds.

## Interpretation

The learned monotone tabular verifier overfit the train-scene tracklet
distribution. Train-scene OOF was high because B0 had been fitted on those
same detector-training scenes, but the verifier suppressed many real
held-out tracklets. A crop verifier may be evaluated only as a separately
locked next stage; V0 parameters must not be retuned from this held-out result.

