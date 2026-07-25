# Canonical v7 crop verifier final report

Protocol: `canonical-v7-person-tracklet-crop-verifier-v1`  
Implementation commit: `c567fc1115b560dc146ffad4fdc04c7a6da025ae`  
Protocol-lock commit: `4638b5d`  
Config SHA-256: `242939de7eb76c627ac046301d0c942917b788c974b89fac51c090194f5fc1d4`  
Frozen CrowdHuman checkpoint SHA-256:
`c16f796a297b0418cf315212a7d12a00b25825d857782da84f24653909723de6`

## Status

```text
V1_V2_GATE: FAIL
fold 1: BLOCKED / unread
railway test: SEALED
attacks: BLOCKED
canonical v7 scientific status: FAIL
```

The amendment was activated prospectively after the immutable V0 failure. It
used the CrowdHuman checkpoint only as a frozen crop encoder. No CrowdHuman
test data, images or derived crops were read or published.

For every train tracklet the highest-, median- and lowest-confidence contextual
crops were encoded by the frozen YOLO backbone. V1 was a five-epoch binary MLP
with BCE and pairwise ranking loss. V2 fused the OOF V0-B and V1 probabilities
with non-negative monotone weights. Model selection and both thresholds used
grouped train-scene OOF predictions only.

The train-only rule selected `V2_monotone_tabular_crop_fusion` and froze
standard threshold `0.758` and safety threshold `0.682` before held-out crop
encoding.

## Held-out fold 0

| Method / point | mAP50 | Precision | Recall | F1 | Small Recall | FP/frame |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.25114 | 0.68306 | 0.20129 | 0.31095 | 0.14656 | 0.88550 |
| V1 crop standard | 0.24576 | 0.85625 | 0.11031 | 0.19544 | 0.04736 | 0.17557 |
| V1 crop safety | 0.24576 | 0.60714 | 0.21900 | 0.32189 | 0.16443 | 1.34351 |
| V2 fusion standard | 0.22571 | 0.98261 | 0.09098 | 0.16654 | 0.02949 | 0.01527 |
| V2 fusion safety | 0.22571 | 0.61044 | 0.12238 | 0.20389 | 0.05719 | 0.74046 |

V2 failed mAP50, Recall, small Recall and F1 relative to B0. V1 safety is
reported as a descriptive negative result: it reached Recall `0.21900` and
small Recall `0.16443`, but missed the frozen Recall gate `0.25`, was not the
train-OOF-selected candidate, and therefore cannot release fold 1.

## Integrity

- parent V0 gate hash verified;
- CrowdHuman checkpoint hash verified;
- encoder parameters frozen;
- verifier-train and held-out scenes remained disjoint;
- lost GT: 0;
- NaN/Inf: 0;
- fold-1 access: 0;
- railway-test access: 0;
- attack runs: 0.

All operating-point metrics reproduced exactly from the saved CSV. AP changed
by `5.1515e-7` after coordinate serialization near an IoU tie, exceeding the
prospectively frozen `1e-12` parity tolerance. Parity therefore remained
`FAIL`; the direct in-memory result already failed every primary quality gate.

## Conclusion

The high-recall temporal proposal stream contains additional true people, but
tracklet statistics and frozen CrowdHuman crop appearance did not transfer
reliably to the weak held-out scene. Canonical v7 ends at
`SCIENTIFIC_FAIL`. Retuning V0/V1/V2 on fold 0 or opening fold 1 is prohibited.
The next defensible step requires additional independent railway development
scenes or a separately defined temporal task/metric, not another verifier
trained on the same scene pool.

