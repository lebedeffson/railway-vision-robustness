# Final-practice implementation note

The received `project.zip` contained source files, pretrained `yolo11m.pt` and
`yolo26n.pt`, but not the OSDaR23 data, trained stage-2 `best.pt`, raw diagnostic
CSVs, or the `outputs/` tree referenced by `HANDOFF.md`.

OSDaR23 is downloaded from the official FID move endpoint used by the supplied
script: `https://download.data.fid-move.de/dzsf/osdar23/<sequence>.zip`. The 43
non-calibration archives total about 109.94 GiB compressed. Do not retain them:
extract only `rgb_highres_center`, OpenLABEL JSON, README and license, then delete
each archive. `download_osdar23_direct.py` supports parallel workers but not
resume: the official server ignores Range, so interrupted `.part` files restart.
The local resolver did not resolve the official host on 2026-07-21; public DNS
resolved it to `194.95.114.28`, while HTTPS with the original host name and curl
`--resolve` worked.

The dataset builder already groups numbered subsequences by scene. That `group`
is now emitted as `sequence_id` and is mandatory for GroupKFold/bootstrap. The
audit fails on any split intersection or annotation inconsistency.

YOLO11's configured loss has box, class and DFL components, not a separate legacy
objectness term. Attack metadata must record the actual box/cls/dfl weights.

After data extraction: build and validate the YOLO dataset, inspect the generated
annotation examples, train stage 1 and stage 2, evaluate clean test performance,
then execute the frozen validation-first final-practice protocol. Only
`build_final_delivery.py` may produce the handoff ZIP; it refuses to run without
the final stage-2 `best.pt`.

On the local RTX 4060 Laptop GPU (8 GiB), training is fixed at batch size 1 and
1280 px. Batch size 2 is unsafe for the fully unfrozen YOLO11m stage.

Fast mode uses two non-overlapping download shards, each with four 24 MiB/s
workers, a 1.5-core CPU quota and a 2.5 GiB soft memory limit. Training keeps
batch size 1 for VRAM safety but uses two dataloader workers, a two-core CPU
quota and no inter-epoch cooldown.
Each download shard has a 3 GiB hard memory ceiling; the training controller has
an 11 GiB hard ceiling and an elevated OOM score so the research job is stopped
before the desktop session under system-wide memory pressure.
Stage training resumes from `last.pt`. Final-matrix work checkpoints after every
fully completed image, and `run_training_pipeline.py` records a marker after
each stage so reboots do not repeat completed practice blocks.
The training user unit runs Python directly: wrapping it in `systemd-inhibit`
failed with an interactive-authorization error after reboot. Sleep may pause the
job, but checkpoints make that safer than preventing unattended startup.

The `3_fire_site_3.1` archive repeatedly failed through the required VPN and the
official host does not support Range. The frozen fallback policy excludes 18
missing and one truncated OpenLABEL-referenced RGB frame before splitting. The
resulting corpus has 1405 referenced images (1057 train, 198 val, 150 test) across all 43
downloaded subsequences. One extra physical PNG in `15_construction_vehicle_15.1`
is not referenced by OpenLABEL and is therefore not a dataset sample.

The canonical final-stage ledger is `outputs/pipeline_status.json`. It has one
record for each of the 13 acceptance stages and is updated atomically. The raw
scene audit writes `outputs/final_practice/audit/{scene_manifest.csv,
split_manifest.csv,split_audit.json}`; the accepted audit has 43/43 scenes
checked, 42 PASS, `3_fire_site_3.1` PASS_WITH_EXCLUSIONS, 1405 readable frames,
and zero sequence split intersections.

The frozen final matrix uses FGSM at 0.5/1/2/4/8 pixel levels and PGD pilot at
0.1/0.25/0.5/1, with seeds 42/123/999 and maximum-loss restart selection.
Non-adaptive clean/attack analysis includes none, Product, bilateral, Gaussian,
median and JPEG. Adaptive Product evaluation uses PGD-20 and PGD-40; JPEG and
median remain outside the adaptive ranking because no BPDA is implemented.

Post-training checkpoint selection is frozen in
`config/checkpoint_selection.json` and exported to the final audit. A common
198-image validation evaluator measured mAP50-95 of 0.095797 for Stage 1 best,
0.097000 for Stage 2 best (epoch 1), and 0.073284 for Stage 2 last. Stage 2 best
is therefore the primary checkpoint by mAP50-95. Stage 1 best is retained as a
sensitivity result because it has higher precision, recall, F1 and mAP50. Test
metrics were not used for selection.

The Q1 revision is frozen in `config/revision_q1_protocol.yaml` before test
analysis. Its main inference uses 5000 paired cluster-bootstrap resamples with
seed 20260720, GroupKFold by `sequence_id`, Holm-Bonferroni for the primary
hypothesis family and BH-FDR as a secondary correction. Per-channel P3/P4/P5
normalization is fit once on clean validation only. N1 quantile is preferred
only if validation saturation and cross-validated diagnostics pass; test data
must never select normalization, features, thresholds, checkpoints or models.
Stage 2 best remains primary and Stage 1 best is a sensitivity checkpoint, not
an opportunity for retrospective checkpoint replacement.
The normalization scheme is selected once on Stage 2 clean validation. Stage 1
sensitivity uses the same frozen scheme but refits only its per-channel clean
validation distribution statistics, so test data cannot influence scaling.

Deadline mode (2026-07-22) preserves the active 198-image validation run and
masks `tnorm-revision-q1.service`. After validation, the pipeline snapshots the
legacy matrix, audits row integrity and NMS warnings, reruns only warning images,
fits clean-validation N1/N2/N3 statistics, and runs a 12-frame smoke pilot over
all three validation sequences. Because the split contains only three validation
sequences, those 12 frames are not independent and must not appear as article
results.

The validation-gated canonical test run uses only attack budgets whose validation
pilot floor fraction is below 50%. If the clean detector itself has a floor, the
conditional clean-detectable analysis is frozen with an explicit limitation.
The final deadline stage computes 5000 sequence-cluster bootstraps and a paired
Stage 1 sensitivity run on 45 deterministic unique frames, balanced as the
three scene sizes permit (10/18/17 rather than duplicating the 10-frame scene).
Full spatial stress, transfer, PGD-40, the second architecture and
the full normalization/filter sweep remain deferred.

The immutable 129-frame interim audit established that the live 198-frame stage
is correctly `val`; the dataset manifest has train/val/test counts 1057/198/150,
while an older narrative reversed val and test. The legacy matrix has 450 rows
per complete frame and remains useful only as a compatibility baseline. Its
feature `product`, `godel` and `lukasiewicz` columns are not the formal pointwise
T-norm conjunctions, and its `g_recovery` is clipped. Formal Q1 metrics now use
`a*b`, `min(a,b)` and `max(0,a+b-1)` on memberships, preserve raw G, and clip
only inside C_def. Do not open canonical test until the clean-validation pilot,
targeted NMS rerun and independent recheck pass.

Deadline mode was frozen on 2026-07-22 because the broad Q1 matrix could not
finish before the article deadline. The active validation matrix remains the
baseline and must not be interrupted. `tnorm-revision-q1.service` is runtime
masked, and the main service has no `OnSuccess` unit, preventing a second GPU
pipeline. `freeze_validation_protocol.py` runs the matrix/NMS audit and the
validation-only pilot gate before canonical test is opened.

After a passing pilot, `deadline_select_budgets.py` freezes the non-floor
validation budgets and `run_final_matrix.py` applies only that grid to
`outputs/final_practice/unified_diagnostics_raw.csv`, with three PGD seeds and
none/Product/bilateral/median defenses. N1 statistics are fit on clean validation
and all Q1 metrics are emitted in the same Stage 2 inference pass.

`build_final_delivery.py` first preserves the standard practice bundle, then
calls the resumable `deadline_finalize.py`. The latter runs a 45-frame,
three-sequence paired Stage 1/Stage 2 sensitivity check, 5000 sequence-cluster
bootstraps, per-scene/macro/LOSO summaries, the deadline tables and figures, and creates
`outputs/bundles/TNormFilter_deadline_final.zip`. Only three independent val and
test scenes exist, so the pilot is smoke-only, Stage 1 sensitivity is
exploratory, and confirmatory scene-difficulty strata are deferred.

Model-gain signs are frozen as `delta_mae = MAE_new - MAE_baseline` and
`relative_mae_reduction = (MAE_baseline - MAE_new) / MAE_baseline * 100`.
Product and Lukasiewicz are primary canonical metrics; Gödel remains a
supplementary redundancy ablation because legacy Product/Gödel rho was 0.99199.
Legacy G is reconstructed from stored A/R without inference, preserved as raw
and clipped variants, and clipped only inside C_def and visualizations.

Canonical v2 supersedes the three-scene deadline test. The 43 downloaded raw
subsequence directories represent only 20 independent grouped railway scenes;
the original split is therefore 14/3/3, not 37/3/3. A model-result-independent
100,000-candidate search froze `data/yolo_osdar23_v2` at 10 train, 5 validation
and 5 test scenes (774/311/320 frames), with all six classes present and zero
sequence overlap. Its manifest hash and audit live under
`outputs/canonical_v2/split`.

The active legacy validation child remains untouched. Its already-loaded parent
controller is paused so it cannot launch the obsolete legacy test. The enabled
`tnorm-canonical-v2.service` waits for the atomic 198-frame validation output,
stops the obsolete service only after the child exits, performs targeted NMS/G
repair, creates `TNormFilter_legacy_baseline.zip`, audits train/val/test and
calibrates standard F1 and safety F2 thresholds, retrains one leakage-free model
from `yolo11m.pt`, and starts canonical attacks only when validation Recall is at
least 0.35 and mAP50 at least 0.25. Canonical attack budgets must additionally
pass a strict absolute (<50%) validation floor gate; no clean-detectable fallback
is allowed for the article experiment.

Canonical v2 threshold handling is model-specific. The frozen order is split,
one v2 training, train/validation clean evaluation, validation-only F1/F2 sweep,
quality gate, one post-gate clean test evaluation with frozen thresholds, then
canonical attacks. The legacy checkpoint sweep is isolated under
`legacy_threshold_calibration` and is never reused. Threshold selection, clean
test and both matrix configs carry the exact v2 checkpoint SHA-256 and fail on
any mismatch. The service waits for `legacy_validation.complete.json`, which is
written only after the 198x450-row CSV has no partial frames, missing conditions
or duplicate condition keys, and holds `outputs/locks/canonical_v2.lock`.

The v2 manifest explicitly records raw `subsequence_id` and independent
`grouped_scene_id`; `sequence_id` is an alias of the latter. Its summary includes
per-class object and grouped-scene counts, small/medium/large counts and each
scene's frame/object share. Some rare classes occur in only one training or
validation grouped scene, so class-specific claims must report that limitation
even though all six classes occur in every split.

Legacy validation completed at 198/198 frames and 89,100 rows with no partial,
missing or duplicate conditions. The post-audit CSV hash changed only because
raw/clipped G and NMS audit columns were appended; the original completion hash
is retained in the baseline snapshot. The targeted audit covers 55 warnings in
six frames, reruns only those frames with instrumented NMS, and exports the exact
recheck contract before the legacy ZIP is sealed.

Canonical normalization fitting remains exact on all sampled clean-validation
positions, separately per layer/channel. Only distribution diagnostics use a
deterministic bounded sample (maximum one million values), because a full
`torch.quantile` over the diagnostic tensor exceeded its input-size limit. This
does not change q01/q99, median, MAD, mean or standard deviation used by N1-N3.

The final canonical chain now includes a preselected five-scene pilot, v2 model
training, validation-only F1/F2 calibration, quality and pilot gates, one clean
test evaluation, validation-frozen budgets, test attacks, 5000 grouped-scene
bootstraps, latency, 15 tables, 10 figures, article fill/validation and the final
ZIP. The canonical analysis config defines D3 as Product/Lukasiewicz added to
D2 and R3 as Product/Lukasiewicz recovery plus raw G and C_def added to R2.
Article output is generated from a frozen internal DOCX reference; the template
is hash-checked before and after filling and is never overwritten.

The completed three-scene legacy smoke pilot retains a FAIL because N1 P4 has
20.63% values below 0.01, narrowly exceeding the frozen 20% saturation warning.
All critical attack, gradient, formula, leakage, finite-value and model checks
passed. This result is archived as a legacy limitation and does not gate the
separately trained/fitted five-scene canonical v2 pilot; the latter keeps the
strict saturation gate and is the only gate allowed to open canonical test.
Legacy threshold calibration is explicitly validation-only (`--splits val`);
the default all-split evaluator must never be used for that isolated baseline.

The final canonical contract supersedes the earlier post-gate ordering: after
v2 calibration and quality gate, fit/select N1 or the predeclared N2 fallback on
clean validation, run the five-scene deterministic frame-order pilot, compute
the full validation budget sweep, freeze budgets, and only then open clean test
once. Matrix checkpoints are retained by complete attack condition rather than
only by complete frame. The final article uses the read-only expanded source
`/home/lebedeffson/Downloads/TNorm_RZD_article_expanded_internal_review.docx`
and writes `TNorm_RZD_article_final.docx`, its PDF, and
`TNorm_RZD_supplementary.pdf` under `outputs/article`.
