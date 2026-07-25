# Railway Person Temporal Safety v1

This protocol is an engineering experiment separate from the final negative V8b
T-norm study. V8b remains immutable.

The frozen canonical-v3 person detector recipe supplies low-confidence proposals.
ByteTrack-style and OC-SORT-style causal adapters consume exactly the same proposal
table for each fold. Tracker parameters are selected on nine support scenes:
outer folds 0 and 1 are both excluded from tracker fitting. Fold 0 is screening
and fold 1 is confirmation; both are evaluated without retuning.

The primary false-alarm quantity is the number of unmatched frame-level detections
per observed minute. Timestamp cadence is parsed from official source filenames;
the expected and observed nominal rate is 10 FPS. False-track events are reported
separately when reliable identities are available.

The two-fold triage requires all of:

- absolute Recall improvement of at least 0.07;
- relative FN/frame reduction of at least 10%;
- false alarms/min increase of at most 25%;
- F1 degradation of at most 0.03;
- no scene Recall degradation greater than 0.10.

Full development requires five scene-disjoint detector OOF folds. The temporal
protocol does not authorize detector retraining. If folds 2-4 are unavailable, the
full gate remains blocked rather than evaluating detector-training scenes.

Railway test images, labels, predictions, and videos remain inaccessible until a
full development PASS and a pre-test freeze. T-norms and adversarial attacks are
out of scope.
