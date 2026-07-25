# Person Canonical V5 Release Status

Status: `CANDIDATE_MATRIX_RUNNING_TEST_SEALED`

- D1 two-fold gate: FAIL
- B0/B1/D1-best/D1-last common evaluator: PASS
- Gradual transfer V5-E: EXCLUDED BY PROSPECTIVE RULE
- Allowed main candidates: V5-A, V5-B, V5-C, V5-D
- Railway test: SEALED
- Attacks and T-norm H1-H4: BLOCKED
- V5 candidate runtime: `person-canonical-v5-candidate-runtime-v1c`
- V5 candidate training: RUNNING under `tnorm-person-v5-candidates.service`
- Instance bank fold 0: PASS (2,392 accepted / 4,905 candidates)
- Pasting 25%: PASS (238/954 changed frames, 376 inserted GT)
- Pasting 50%: PASS (477/954 changed frames, 747 inserted GT)
- Scientific claim: NONE

The diagnostic is development-only. It shows that direct railway fine-tuning raises Recall over CrowdHuman zero-shot but produces a large FP increase; the last checkpoints further collapse AP and confidence quality.
