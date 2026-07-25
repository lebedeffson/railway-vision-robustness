# Canonical v8b final report

## Decision

`canonical-v8b-person-failure-risk-v1` ended as `DEVELOPMENT_FAIL`.
The railway test remained sealed (`test_access_count = 0`), the detector was
not retrained, and attacks were outside this protocol.

The prospectively frozen primary endpoint was equal-weight scene-macro MAE for
predicting `FN/frame`, comparing U3 against U2 across nested 15-scene LOSO:

| Model | Scene-macro MAE FN/frame |
|---|---:|
| U2: detector outputs + standard representation distances | 1.59549 |
| U3: U2 + Product/Łukasiewicz consistency | 1.76798 |

The U3 change was `+0.17249` MAE, equivalent to a `-10.81%` relative MAE
reduction: the T-norm extension made the primary endpoint worse.

## Frozen gate

| Check | Required | Observed | Status |
|---|---:|---:|---|
| Relative MAE reduction | at least 5% | -10.81% | FAIL |
| Paired scene-bootstrap CI for U3-U2 MAE | upper bound below 0 | [-0.00850, 0.40791] | FAIL |
| Scene wins | at least 10/15 | 5/15 | FAIL |
| Leave-one-scene-out stability | no negative reduction | minimum -15.77% | FAIL |
| Brier noninferiority | required | delta -0.00440 | PASS |
| ECE noninferiority | required | delta -0.00515 | PASS |
| Technical integrity | all checks pass | PASS | PASS |

The small calibration improvements are descriptive secondary observations.
They do not rescue the failed primary endpoint.

## Hierarchical comparison

Equal-weight scene means:

| Model | MAE FN/frame | MAE Recall | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| U0 | 2.60385 | 0.22633 | 0.42045 | 0.36975 | 0.27183 | 0.43177 |
| U1 | 1.81067 | 0.22199 | 0.68055 | 0.60975 | 0.23958 | 0.40080 |
| U2 | 1.59549 | 0.27421 | 0.64241 | 0.49792 | 0.26170 | 0.39198 |
| U3 | 1.76798 | 0.26474 | 0.61388 | 0.49754 | 0.25730 | 0.38683 |

No U3-vs-U2 endpoint survived the frozen Holm correction. Risk-coverage
curves were generated, but U3 did not consistently dominate U2.

## Integrity and scope

- F0 audit: PASS.
- F1 extraction: 1,085 frames from 15 grouped development scenes.
- Evaluator consistency: PASS.
- Train-only normalization and prototypes: PASS.
- Nested train-only hyperparameter selection: PASS.
- Scene leakage: 0.
- NaN/Inf: 0.
- Frozen B0 checkpoint SHA-256:
  `5a8483b1d40938f49b05225cfe4b3c92414e874fe72008746396ea32bcfb0a79`.
- Protocol-lock SHA-256:
  `93c67625c243130b640c35da9447252a2913b4fb59d35e65987c638fdda68e32`.

B0 had been trained on 12 of the 15 risk-development scenes. Therefore, this
experiment tests LOSO generalization of the risk estimator over frozen B0
outputs; it is not a uniformly detector-OOF study. The limitation was frozen
and disclosed before feature extraction.

## Claim-safe conclusion

On the current development pool, Product/Łukasiewicz representation
consistency did not add reliable failure-risk information beyond standard
detector and representation-distance features. The risk-only test was not
opened. A positive T-norm failure-prediction claim is not supported.

The defensible article outcome is a negative methodological result or a
development-only report. A new positive claim requires a new prospectively
frozen question or genuinely new independent data, not post-hoc endpoint
selection.
