# Railway person temporal verifier v1

This protocol starts after the terminal `DEVELOPMENT_FAIL` of
`railway-person-temporal-safety-v1`. The parent outputs, V8b, detector
predictions, tracker choice and railway test seal are immutable.

OC-SORT is the primary tracker because it had the smaller false-alarm increase
and F1 loss at nearly the same Recall gain. ByteTrack is sensitivity only.

Tracker/verifier fitting uses nine support scenes. Fold 0 is the single global
rule/threshold screening fold and fold 1 is confirmation without retuning.
Tracks never cross `subsequence_id`; all learned preprocessing and calibration
are fitted on grouped support-scene folds only.

Every system retains the frozen frame-detector predictions. The verifier may
only accept or reject temporal additions. Interpolated frames never count as
detector hits. No ground-truth field is part of the inference feature matrix.

The first stage contains exactly 12 prospectively fixed rule candidates. If the
selected rule fails the two-fold development gate, one L2 logistic verifier is
fitted. Gradient boosting is sensitivity only after logistic PASS. A crop model
is outside this protocol and requires a new amendment.

Railway test inference is forbidden until a full 15-scene development PASS.
