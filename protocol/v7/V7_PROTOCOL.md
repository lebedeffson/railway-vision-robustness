# Canonical v7 person tracklet verifier

`canonical-v7-person-tracklet-verifier-v1` is a development-only successor to
the completed v6 T0 failure. It does not alter B0 or train a detector.

The experiment converts the frozen high-recall ByteTrack-style proposal stream
into tracklets and trains two monotone, interpretable verifiers:

1. a sign-constrained logistic scorer trained with weighted BCE and pairwise
   ranking loss;
2. monotonic histogram gradient boosting.

Tracklet labels and calibration use only the train scenes of the corresponding
fold. Ambiguous tracklets are excluded. Model selection, score blending and
standard/safety thresholds use grouped out-of-fold predictions from those
train scenes. Held-out fold 0 is read once after those choices are frozen.
Fold 1 remains unread unless fold 0 passes the frozen V0 gate.

The railway test remains sealed and adversarial attacks remain blocked.
Failure of V0 permits a separately locked crop-verifier amendment; it does not
permit retuning V0 on held-out results.

