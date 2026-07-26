# Operator assistant evidence v3

- Five independent development scenes are frozen: fire site, pedestrian
  bridge, Klein Flottbek, Altona and vegetation steady. They provide 479
  frames and 48 source-ID person episodes.
- Every scene has a complete pre-aggregation stream, PersonEpisode table,
  HazardEvent JSONL and exact hierarchy replay audit. Across scenes, 4,943
  aggregator inputs produce 25 B4/B5 cards; this is card aggregation evidence,
  not operator workload evidence.
- All 15 v2 GT episodes are traced. Thirteen are covered, one has
  `NO_DETECTOR_MATCH`, and one has 77 detector matches but
  `NO_TRACKER_MATCH`.
- B0-B5 use identical per-scene inputs. B4 and B5 each have the frozen
  108-configuration sensitivity sweep; B5 preserves every PersonEpisode child.
- Semantic hazard recall/precision/false-merge metrics remain blocked until two
  authors independently group episodes and submit an adjudicated
  `GT_HAZARD_EVENTS.csv`. Code must not infer this grouping.
- FULL_SYNC reuses the frozen 100/1000-frame v2 runtime. CACHED_SYNC and
  DEFERRED_BATCH are explicitly blocked until parity-preserving implementations
  are measured; no speed claim is allowed.
- Railway test remains sealed with access count zero; no training or
  multi-stream benchmark is permitted.
