# Canonical v8b person failure-risk protocol

Protocol ID: `canonical-v8b-person-failure-risk-v1`

The detector is frozen. This protocol does not train or improve YOLO and does
not evaluate adversarial robustness. It asks whether T-norm representation
consistency predicts failures of the fixed B0 person detector.

## Fixed detector and evidence boundary

The fixed detector is the person-v3 fold-0 B0 checkpoint with SHA-256
`5a8483b1d40938f49b05225cfe4b3c92414e874fe72008746396ea32bcfb0a79`.
It was trained without three development scenes and with the other twelve.
This overlap is disclosed. LOSO therefore evaluates generalization of the risk
estimator, not a uniformly detector-OOF estimate across all 15 scenes.

No new detector training is allowed. Test remains sealed until a development
PASS and a pre-test risk-model freeze.

## Primary endpoint

The confirmatory comparison is U3 versus U2 for scene-macro MAE of frame-level
`FN/frame`. U3 must reduce this MAE by at least 5%, have a paired scene
bootstrap CI wholly below zero for `MAE_U3 - MAE_U2`, win on at least 10 of 15
scenes, and retain non-negative improvement after deleting any one scene.

Secondary metrics cannot rescue a failed primary endpoint.

## Outcome definitions

The primary target is the number of unmatched person GT boxes in each frame at
the frozen B0 threshold and IoU. Recall regression excludes frames with no
person GT. A frame is unsafe when `FN >= 1`, or when it contains person GT and
Recall is below 0.50.

## Deployable features

U0-U3 use only information available at inference time. Object-region features
are pooled over frozen B0 person proposals. GT-person regions are an oracle-only
supplement and are forbidden in fitted risk models, threshold selection, gates,
and test inference.

For every outer and inner scene split, quantile normalization and reference
prototypes are refit only on that split's training scenes.

## Nested hierarchy

```text
U0 confidence only
U1 U0 + detector output statistics
U2 U1 + standard P3/P4/P5 representation distances
U3 U2 + Product/Lukasiewicz consistency
```

Ridge is the primary regression model. ElasticNet is sensitivity-only.
Logistic regression with train-scene OOF isotonic calibration estimates unsafe
frame probability. Neural predictors and primary boosting are forbidden.

## Stages

```text
F0 source/checkpoint/hook/test-seal audit
F1 resumable development feature extraction
F2 nested 15-scene LOSO, bootstrap, Holm and risk-coverage
development gate
pre-test freeze on PASS
one risk-only test evaluation
```

If the primary gate fails, test remains sealed. FGSM, PGD and adaptive PGD are
outside this article.

