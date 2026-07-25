# Temporal safety v1

Protocol: `railway-person-temporal-safety-v1`

- Implementation commit: `9fb352625d675fcdf2e62fb84e8fecfad9be2656`.
- V8b stayed immutable; V9 closed `ABANDONED_BEFORE_EXECUTION`.
- One frozen canonical-v3 detector recipe supplied proposals. Fold-specific
  checkpoints were required for scene-disjoint folds 0 and 1.
- Tracker parameters used nine detector-train support scenes outside both
  screening fold 0 and confirmation fold 1.
- Candidate floor was 0.001; frame operating threshold was 0.07; inference was
  causal; interpolation was limited to one frame.
- ByteTrack and OC-SORT adapters used identical proposal streams per fold.
- Sequence audit: 1085 development frames, 15 scenes, 32 subsequences, 0
  duplicates, observed median 10 FPS, no test rows/images/labels read.

Two-fold fold-macro effects:

| tracker | delta Recall | relative FN reduction | relative false-alarm increase | delta F1 |
|---|---:|---:|---:|---:|
| ByteTrack | +0.137716 | 18.78% | 390.87% | -0.074058 |
| OC-SORT | +0.132768 | 18.20% | 341.32% | -0.062290 |

Both candidates failed the frozen false-alarm and F1 checks. Full development
was blocked, test access count stayed zero, and no positive article was built.
The valid practical lesson is narrow: low-confidence temporal association
recovers people, but the present confirmation logic cannot suppress false
alarms enough for the specified operating envelope.

Cached full-pipeline mean latency was 115.9-156.7 ms/frame. p95, VRAM and CPU
were not remeasured after FAIL and must not be imputed.
