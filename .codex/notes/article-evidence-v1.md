# Article evidence computation v1

Protocol: `article-evidence-computation-v1`

- Branch: `feat/article-evidence-computation-v1`.
- Inputs are saved development predictions, tracker outputs, frozen verifier
  models, the 288-row false-track audit and saved V8b P3/P4/P5 tensors.
- Detector grid is exactly
  `0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01`.
- The threshold matrix contains five methods, 45 aggregate rows and 270
  per-scene rows over the six frozen screening/confirmation scenes. No point
  passed the complete operational gate.
- The verifier sweep contains all 201 unique thresholds and no complete gate
  pass.
- All 288 false tracks remain represented. The stored audit lacks complete
  human-verified signal/catenary/pole/train-part labels, so those semantic
  subcategories are explicitly `BLOCKED_MISSING_ARTIFACT`; deterministic
  categories and geometry flags are retained without visual inference.
- Runtime evidence uses 100 warm-up and 1,000 measured development frames.
  Tiling is not used by the product pipeline; coordinate restoration remains
  inside the Ultralytics postprocess timing and is labelled accordingly.
- The public archive excludes crops, images, videos, checkpoints, local paths
  and test material. Full acceptance observed `428 passed`; artifact checks
  observed `20 passed`; manifest verification passed.
- Railway test remains `SEALED`, access count zero. No training or article edit
  is part of this protocol.
- Event-sensitivity evidence requires the complete pre-aggregation candidate
  stream. The product SQLite retains only 62 observations from accepted events;
  replaying them yields four default events rather than the published three, so
  the 108-configuration diagnostic is explicitly
  `BLOCKED_MISSING_PRE_AGGREGATION_STREAM`.
- The 100-frame encoded smoke video has no immutable exact frame-to-GT-episode
  mapping. Fragmentation, false merges and episode coverage remain
  `BLOCKED_MISSING_ARTIFACT` rather than being inferred from the default events.
- One-stream scaling reuses the verified 1,000-frame run. Two- and four-stream
  loads are `BLOCKED_RESOURCE_LIMIT` after interactive-host instability; do not
  impute parallel throughput.
