# Operator assistant evidence v2

Protocol: `operator-assistant-evidence-v2`

- The development-only benchmark uses frames 110–209 of
  `8_station_altona_8.2`, high-resolution center RGB, with persistent OpenLABEL
  person UUIDs. It contains 100 frames, 15 person episodes and 1,322 person
  boxes.
- The complete pre-aggregation trace has stable candidate IDs. The locked
  baseline is reproducible as 2,410 raw detector boxes, 1,635 aggregator
  observations and five unique events; direct execution and replay signatures
  match exactly. The old demonstration `296 -> 3` is not v2 evidence.
- All 108 prospectively defined event configurations were calculated without
  choosing a replacement baseline. The locked 3 s / IoU 0.20 / center 0.10 /
  reopen 30 s configuration has event recall 0.866667, fragmentation 0.266667,
  false-merge rate 0.800000 and episode-frame coverage 0.866667.
- Fail-safe behavior lives in amendment modules
  `event_aggregator_v2.py` and `processor_v2.py`; the source files hashed by the
  prior published event-engineering lock remain byte-identical.
- All ten fail-safe checks pass. Verifier exceptions enter general review as
  `VERIFIER_UNAVAILABLE`; recoverable processing errors create a technical
  review event; storage-integrity failures remain hard stops.
- The one-stream runtime protocol uses 100 warm-up and 1,000 measured frames.
  Multi-stream scaling remains deferred to avoid overloading the workstation.
- Railway test remains `SEALED`, access count zero. No model was trained.
