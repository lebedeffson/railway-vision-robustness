# Canonical V5 expedited screening amendment

Protocol ID: `canonical-v5-expedited-screening-v1`

This prospective amendment is a compute-allocation screen only. Its outputs are
not article evidence and cannot open railway test or adversarial attacks.

The formal `person-canonical-v5-candidate-runtime-v1c` results remain immutable:
V5-A and V5-B completed both folds and failed their frozen gates. The original
V5-C run was stopped before a completed epoch after an integrity audit found
that unchanged pasted tiles linked multiclass labels into an `nc: 1` dataset.
Those partial artifacts were quarantined and are not scientific results.

The amendment introduces separate IDs that do not rename formal V5 candidates:

- C0: original person-only B0 control;
- C1: C0 with audited 25% person instance pasting;
- C2: C1 with train-only hard-negative background selection.

All candidates start from the same `yolo11m.pt`, seed, fold, image resolution,
tiling inference and independent evaluator. The frozen reference is B0 fold 0
at confidence 0.07.

Successive halving is fixed before new results:

1. level 0: data, graph, finite gradient and shared initialization checks;
2. level 1: every candidate reaches five cumulative epochs, at most two advance;
3. level 2: survivors reach ten cumulative epochs, at most one advances;
4. level 3: the winner reaches twenty cumulative epochs on fold 0;
5. level 4: only that winner receives a twenty-epoch confirmation on fold 1.

Intermediate rungs retain optimizer state and resume the same 20-epoch run.
Every promotion/rejection is appended to `decision_trace.json`; no metric may
be edited after a decision is recorded. Screening terminates after the
two-fold decision. A PASS still requires a separate locked canonical OOF
continuation before test can be opened.

The exact thresholds and candidate definitions are in
`configs/person_v5/expedited_screening.yaml`. The runtime refuses to start
unless `V5_EXPEDITED_SCREENING_LOCK.json` verifies that configuration and all
screening implementation hashes.

