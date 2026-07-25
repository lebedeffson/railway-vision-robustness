# V8b article scope

Working title:

> T-Norm Representation Consistency for Failure-Risk Estimation in Railway
> Person Detection

The article evaluates prediction of failures made by a frozen person detector.
It does not claim that the detector was improved, production-ready, or
adversarially robust.

Primary hypothesis:

> Product and Lukasiewicz representation-consistency features reduce
> scene-macro MAE for predicting frame-level false negatives relative to the
> same risk model with standard representation distances only.

The primary result is U3 versus U2 for `FN/frame`. Object regions in the
deployable model are B0 proposal regions. GT-person regions are oracle-only.

Development wording before F2:

> The failure-risk protocol and implementation were prospectively frozen.
> Fifteen-scene nested LOSO evaluation is pending. Railway test remains sealed
> and no scientific result has been established.

Failure wording:

> T-norm consistency did not pass the prospectively defined incremental-value
> gate over standard representation distances. Test remained sealed.

Pass wording:

> T-norm consistency passed the prospectively defined development gate for
> false-negative risk estimation. Model coefficients, normalization,
> prototypes and the operating rule were frozen before one test evaluation.

