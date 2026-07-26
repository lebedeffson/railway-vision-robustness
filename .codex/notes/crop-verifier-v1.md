# Crop verifier v1

Protocol: `railway-person-crop-verifier-v1`

- Implementation commit: `c98347a2d8e9765e2ef70aaf8337978a6926b56a`.
- Parent detector, OC-SORT tracks and temporal-verifier results were frozen.
- Encoder checkpoint SHA-256:
  `c16f796a297b0418cf315212a7d12a00b25825d857782da84f24653909723de6`.
- Frozen encoder used three real detector crops and mean aggregation.
  Interpolated crops: 0. Cross-role scene/crop-source overlap: 0.
- Models were L2 logistic `track_only`, `visual_only` and `combined`.
  Scaling and Platt calibration used support grouped-scene OOF only.

Support OOF:

| model | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| track_only | 0.883673 | 0.908288 | 0.108039 |
| visual_only | 0.601197 | 0.542732 | 0.240938 |
| combined | 0.848636 | 0.847435 | 0.145010 |

`combined` and global threshold `0.275` were frozen before confirmation.
Two-fold effects:

| metric | effect |
|---|---:|
| delta Recall | +0.126740 |
| relative FN/frame reduction | 17.23% |
| relative false-alarm increase | 275.88% |
| delta F1 | -0.044440 |

All three models failed the false-alarm and F1 triage checks. Full development,
MLP sensitivity and test were not run. Final status is
`CLOSED_NO_PRACTICAL_GATE`; test access count is zero.

The visual-only OOF result and confusion audit show that the frozen person
backbone does not reliably separate persistent railway hard negatives from
people. Do not tune deeper heads, trackers or thresholds on the same six held-out
scenes. New independent railway scenes are required.
