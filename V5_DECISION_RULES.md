# V5 Decision Rules

The primary development metrics are scene-macro Recall, scene-macro small
Recall and worst-fold Recall. mAP50 is a secondary guard against uncontrolled
false positives.

A candidate passes only if all frozen absolute gates pass:

- macro mAP50 at least 0.45;
- macro Recall at least 0.45;
- macro small Recall at least 0.30;
- worst-fold Recall at least 0.30;
- evaluator parity PASS, with no lost GT or non-finite values.

Relative to B0 it must improve macro Recall by 0.03 or macro small Recall by
0.05, while changing macro mAP50 by no less than -0.01.

Selection is lexicographic: all gates, worst-fold Recall, macro small Recall,
macro Recall, macro mAP50, latency, then complexity. No test observation can
change these rules.

