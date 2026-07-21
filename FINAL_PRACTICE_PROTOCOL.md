# Final practice protocol

This protocol supersedes the image-grouped statistical claims in the supplied PDF.
The immutable received-code baseline is the Git tag `diagnostics_v1_current`.

## Hard gates

1. The frozen raw-data policy contains 1424 OpenLABEL-referenced frames: 1405
   usable frames and 19 pre-split exclusions from `3_fire_site_3.1` (18 missing,
   one truncated). The exact exclusions are versioned in
   `config/raw_frame_exclusions.json` and audited
   in `outputs/final_practice/00_audit/raw_frame_exclusions.*`.
2. `python final_practice_preflight.py` must report `READY`.
3. `python audit_final_practice.py` must report `PASS` before any new attack run.
4. Every raw row must have `sequence_id`; the independent unit is the manifest
   `group`, which joins adjacent numbered OSDaR23 sequence parts.
5. Test is opened only after attack parameters, filters, normalization, seeds and
   formulas are frozen on validation.
6. Bootstrap and GroupKFold use `sequence_id`; bootstrap defaults to 2000 draws.

## Attack objective

YOLO11 uses the Ultralytics detection objective with box, class and DFL terms.
There is no legacy YOLOv5-style standalone objectness loss. The attack extractor
records the actual `box`, `cls` and `dfl` weights from the loaded model configuration.

## Required run order

The persistent low-impact controller runs this order automatically through
`wait_and_train.py` and `run_training_pipeline.py`. Completed stages and final
matrix images are checkpointed, so a reboot resumes rather than discarding the
whole run.

```bash
python final_practice_preflight.py
python audit_final_practice.py
python evaluate_baseline.py

python extract_feature_consistency.py --eval-split val --quick --rebuild-stats

python run_final_matrix.py --split val \
  --output outputs/final_practice/unified_diagnostics_val_raw.csv
# Freeze formulas, normalization, confidence, defenses, budgets and seeds here.
# Run test only after the validation freeze.
python run_final_matrix.py --split test \
  --output outputs/final_practice/unified_diagnostics_raw.csv

python analyze_feature_diagnostics.py \
  --features outputs/diagnostics/feature_consistency/feature_consistency_val.csv \
  --detections outputs/diagnostics/image_detection/image_detection_val.csv \
  --output outputs/diagnostics/analysis_val
python analyze_feature_diagnostics.py
python final_diagnostics.py --bootstrap 2000

python analyze_final_practice.py --bootstrap 2000
python benchmark_latency.py --warmup 30 --repetitions 100 --batch 1
```

`run_final_matrix.py` writes detection, feature, attack-side and defense-side rows
from the exact same adversarial examples, avoiding cross-script seed mismatch.
The PGD pilot budgets are `0.1,0.25,0.5,1.0 / 255`, PGD-20, random start,
three restarts/seeds (`42,123,999`), with the maximum-loss restart selected per
image. Add `2,4 / 255` and PGD-40 only if the `1/255` pilot is above the floor.
The attacked final matrix uses the mandatory `none`, Product T-norm,
bilateral and Gaussian branches. JPEG and median remain in clean-utility
evidence only; no white-box robustness claim is made for either one.

## Claim gates

- If `mAP50 <= 0.01`, use Recall, image F1, false negatives/frame, confidence
  drop and IoU shift as primary responses; retain mAP only as context.
- If T-norm `delta R2 < 0.05` and MAE reduction is below 5%, the permitted claim
  is only “small complementary diagnostic signal”.
- If the sequence bootstrap interval includes zero, label the result exploratory.
- If adaptive PGD removes the apparent defense gain, do not claim white-box
  robustness.
- Keep the failed Lukasiewicz threshold policy as a negative result; do not tune
  further thresholds on test.

## Expected outputs

`outputs/final_practice/00_audit/` contains split/annotation/baseline evidence.
Attack-side rows are written to `attack_consistency_raw.csv`; feature-side
`P/A/R/G/C_def` and entropy are written by `extract_feature_consistency.py`.
`unified_diagnostics_raw.csv` is the only input to M0-M4 final statistics.
Latency and environment evidence is under `08_latency/`; sequence statistics are
under `09_statistics/`.
