# Canonical v8b negative result

The T-norm extension U3 failed the frozen primary development endpoint.

```text
U2 scene-macro MAE FN/frame: 1.59549
U3 scene-macro MAE FN/frame: 1.76798
relative MAE reduction:      -10.81%
paired bootstrap 95% CI:     [-0.00850, 0.40791]
scene wins:                  5/15
development status:          FAIL
test access count:           0
```

U3 slightly improved Brier score and ECE, but these secondary calibration
changes were not statistically confirmed after correction and cannot replace
the failed confirmatory endpoint.

No detector retraining, adversarial attack, risk-only test evaluation, or
post-hoc protocol amendment was performed.
