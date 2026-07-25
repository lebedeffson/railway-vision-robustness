# Person V5 data-leakage audit

- Protocol: `person-canonical-v5-range-aware-v1`
- Diagnostic inputs: original train plus validation development folds
- Independent unit: `grouped_scene_id`
- Railway test: sealed and absent from evaluator sources
- Instance-bank rule: current-fold train scenes only
- LiDAR linkage rule: RGB bbox and LiDAR cuboid must share the same OpenLABEL object UUID
- Area fallback is labelled `bbox_area_scale_fallback`, never range
- Diagnostic evaluator parity: `PASS`
