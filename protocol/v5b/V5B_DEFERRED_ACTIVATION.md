# Deferred activation: person-canonical-v5b-starcoord

Status: `INACTIVE`

This document freezes the transition rule before the remaining
person-canonical-v5 results are known. It is not the v5b protocol lock and does
not authorize v5b training.

## Activation condition

The v5b branch may be created only when all four primary v5 candidates have
finished folds 0 and 1 and `results/v5/TWO_FOLD_GATE.json` records:

```text
V5-A: absolute/comparative gate FAIL
V5-B: absolute/comparative gate FAIL
V5-C: absolute/comparative gate FAIL
V5-D: absolute/comparative gate FAIL
PERSON_CANONICAL_V5: SCIENTIFIC_FAIL
```

If any primary v5 candidate passes both gates, v5b remains inactive. The v5
winner proceeds unchanged to folds 2–4 and the full OOF gate.

## Isolation

After the activation condition is met:

1. preserve the completed v5 branch and release bundle;
2. create branch `feat/person-canonical-v5b-starcoord`;
3. create `protocol/v5b/V5B_PROTOCOL_LOCK.json` before the first v5b metric;
4. use runtime `person-canonical-v5b-starcoord-runtime-v1`;
5. use service `tnorm-person-v5b-starcoord.service`;
6. keep railway test sealed and attacks blocked.

The current `V5-A`, `V5-B`, `V5-C`, and `V5-D` definitions must never be
renamed or modified. In particular, StarBlock must not be added to V5-B or
V5-D.

## Frozen prospective v5b matrix

| Candidate | Only experimental change |
|---|---|
| V5B-B0 | exact baseline reproduction |
| V5B-S | baseline plus StarBlock |
| V5B-C | baseline plus Coordinate Attention |
| V5B-SC | baseline plus StarBlock and Coordinate Attention |

The primary v5b matrix does not contain P2, person instance pasting, TTA, new
losses, new assignment rules, or a new image size. StarBlock is limited to
predeclared neck fusion stages. Coordinate Attention uses identical placement
in V5B-C and V5B-SC.

The v5b absolute gate remains `mAP50 >= 0.45`, `Recall >= 0.45`, small Recall
`>= 0.30`, and worst-fold Recall `>= 0.30`, with evaluator/data-integrity
checks. Comparative evidence additionally requires Recall gain `>= 0.03` or
small-Recall gain `>= 0.05`, mAP50 delta `>= -0.01`, and relative FP/frame
growth `<= 15%`.

No performance value from an external paper is a target or acceptance
criterion. Primary architecture sources must be checked before the v5b
implementation is locked.
