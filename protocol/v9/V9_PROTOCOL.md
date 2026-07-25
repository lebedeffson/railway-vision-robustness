# Canonical v9 residual-temporal T-norm protocol

Protocol ID: `canonical-v9-person-residual-temporal-v1`

Status: `BLOCKED_PREREQUISITES`

## Activation boundary

V9 may start only after at least one condition is independently audited:

1. at least five genuinely new railway development scenes are added without
   sequence overlap; or
2. all 15 existing scenes receive detector predictions and features from a
   detector that was not trained on the held-out scene.

The current repository satisfies neither condition. V9 therefore contains a
frozen design and a synthetic implementation proxy only. It has no article
evidence and does not read the sealed test.

## Confirmatory question

Do residualized, causal temporal T-norm features contain information about
`FN/frame` that is not already explained by detector statistics and standard
representation metrics?

The primary endpoint remains scene-macro MAE for `FN/frame`. This choice
continues V8b and was not selected from V8b secondary results.

## Models

- `V2`: detector statistics plus standard representation metrics.
- `V3`: `V2` prediction plus a residual model fitted to train-scene-OOF
  residuals using T-norm features residualized against V2 features.

The primary model is hierarchical mixed-effects regression with a grouped
scene random intercept. Strictly nested scene-CV gradient boosting is a
sensitivity model. Neural risk models are forbidden.

## Causal temporal features

For Product, Łukasiewicz, and proposal-background gaps, compute only current
and past-frame differences, rolling median, rolling MAD, slope, and
change-point scores over fixed 3/5-frame windows. Windows never cross
`subsequence_id`; future frames are unavailable.

## Validation and gate

All normalization, residualization, hyperparameters, calibration, and model
fitting occur inside the outer training scenes. The statistical unit is
`grouped_scene_id`.

V3 passes only when all conditions hold:

```text
relative scene-macro MAE reduction >= 5%
paired scene-bootstrap upper 95% CI < 0
scene wins >= 70%
worst leave-one-scene-out relative reduction >= 0
```

Secondary outcomes cannot rescue primary failure. Test access remains zero
until a new-data/detector-OOF audit, design lock, full development execution,
and confirmatory PASS are complete.
