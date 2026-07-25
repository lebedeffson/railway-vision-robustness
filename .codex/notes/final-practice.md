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

Person-only canonical v3 was frozen before test and evaluated on two
scene-disjoint development folds. It stopped at the prospective hard-fail rule:
macro mAP50 0.358828, macro Recall 0.354316, macro small Recall 0.211301 and
worst-fold Recall 0.201288. Both independent evaluator checks passed, no GT or
scene was lost, and `outputs/person_v3/test/TEST_OPENED.json` was never created.
The verified failure bundle is
`outputs/person_v3/bundles/TNormFilter_person_v3_triage_failed.zip` with SHA-256
`645b4c60a4311ddd5d24c6f6f017db98767a28d9b1a871ec3bcf950ed4ec84a4`.

Canonical v4 (`canonical-v4-person-dg-nwd-v1`) keeps those folds, initialization,
tiling, seed and evaluator fixed. A1 adds hybrid NWD-aware task assignment,
CIoU/NWD localization and QFL; A2 adds scene-round-robin GroupDRO; A3 adds a
different-scene MixStyle bank, constrained scale-aware zoom and validation-only
SWAD. A candidate must improve macro mAP50, Recall and small Recall by at least
0.05, not reduce worst-fold Recall, and reduce FP. Two-fold results are only
selection evidence; test remains sealed until the unchanged winner passes the
full five-fold OOF gate.

The v4 expedited selection and the train-only proxy both rejected the custom
DG/NWD stack. Canonical v5 therefore freezes a data-first experiment: standard
YOLO box/class/DFL losses, CrowdHuman visible-person pretraining, and bounded
hard-positive/hard-negative mining using only each railway fold's train scenes.
The only v5-v1 GPU candidate is D1 on folds 0/1; RT-DETR is deferred and requires
a new prospective amendment. CrowdHuman test is unused, and image data must
never enter Git or a release bundle.

Person canonical v5 range-aware runs the immutable V5-A=P2,
V5-B=P2+Coordinate Attention, V5-C=25% person pasting and V5-D=P2+25% person
pasting matrix. StarBlock is forbidden in those candidates. The prospective
v5b transition is documented in `protocol/v5b/V5B_DEFERRED_ACTIVATION.md` but
remains inactive until every primary v5 candidate finishes two folds and fails
the frozen gates. Only then may a separate `feat/person-canonical-v5b-starcoord`
branch lock B0, StarBlock, Coordinate Attention and StarBlock+Coordinate
Attention candidates without P2; railway test and attacks remain sealed.

The final canonical contract supersedes the earlier post-gate ordering: after
v2 calibration and quality gate, fit/select N1 or the predeclared N2 fallback on
clean validation, run the five-scene deterministic frame-order pilot, compute
the full validation budget sweep, freeze budgets, and only then open clean test
once. Matrix checkpoints are retained by complete attack condition rather than
only by complete frame. The final article uses the read-only expanded source
`/home/lebedeffson/Downloads/TNorm_RZD_article_expanded_internal_review.docx`
and writes `TNorm_RZD_article_final.docx`, its PDF, and
`TNorm_RZD_supplementary.pdf` under `outputs/article`.

Canonical v2 training completed 63 epochs and stopped early. The validation-only
threshold sweep selected standard confidence 0.081 (Recall 0.17310, F1 0.20989)
and safety confidence 0.017 (Recall 0.27233, F2 0.22387); validation mAP50 was
0.11896. The predeclared gate required Recall 0.35 and mAP50 0.25, so the run
stopped before normalization, pilot attacks, test evaluation, canonical attacks,
statistics, latency and article finalization. The diagnostic handoff is
`outputs/bundles/TNormFilter_baseline_gate_failed.zip`; downstream stages are
explicitly `skipped`, and the inactive linked service is not enabled at login.

The prospectively committed rescue protocol is `canonical-v2-rescue-v1` on
`feat/canonical-v2-rescue-v1`. Full split/image/label hashing found 20 independent
groups, 1405 readable frames, zero cross-split exact or perceptual duplicates,
zero fatal bbox errors and a consistent six-class mapping. Visual review of 104
frames confirmed one exact duplicated person annotation in a train frame; the
separate `data/yolo_osdar23_rescue_v1` removes that one line while retaining the
frozen split and leaving validation/test labels unchanged. The golden evaluator
audit passed. The current managed execution environment exposes neither
`/dev/nvidia*` nor the user D-Bus, so the committed resumable pipeline is
`blocked_infrastructure` at micro-overfit; this is not a failed scientific gate.
`systemd/tnorm-rescue-v1.service` is the host-GPU continuation unit.

The frozen rescue-v1 micro-overfit completed 150 epochs but failed its
prospective 0.90/0.90 gate (mAP50 0.787435, Recall 0.859155). Of 80 false
negatives, 75 were signals; small/medium/large Recall was
0.801508/0.993333/1.0. The loss decrease and changed weight norm confirm
parameter updates. Ultralytics 8.4.102 defines but does not dispatch
`on_before_zero_grad`, so the all-zero gradient log from this run is invalid
instrumentation, not evidence of disconnected gradients. Future runs attach
the logger to `on_train_batch_end`. Under `canonical-v2-rescue-v1`, R0-R4,
test, attacks and article finalization remain skipped; a new protocol is
required before any further scientific run.

The prospectively frozen expedited M4 triage stopped the full two-day protocol
after scene-CV fold 0. Independent global tiling/fusion evaluation on 306
held-out train frames from two grouped scenes produced mAP50 0.216448,
mAP50-95 0.075490 and safety Recall 0.179917, below the frozen 0.25/0.35
go/no-go thresholds. Small/medium/large Recall was
0.099374/0.232023/0.945455. Evaluator CSV reparse consistency passed, no source
GT was lost and test remained sealed. The 10-frame held-out scene contained no
GT objects, so its zero Recall is not treated as a catastrophic scene failure;
the hard fail is caused by mAP50 and Recall alone. Remaining CV folds, full
training, official validation, test, attacks and H1-H4 analysis are
`skipped_by_expedited_triage`.

Canonical v3 merged the old 10-scene train and 5-scene validation into a
1085-frame, 15-scene development pool while preserving the same five sealed
test scenes. A deterministic 100,000-partition constrained split search proved
the requested fivefold class-support contract infeasible: animal occurs in only
two development scenes, so at least one held-out animal fold has fewer than two
training scenes (the best partition groups both animal scenes and leaves zero
animal train support in that fold). In the old M4 fold 0, road vehicle was not
strictly absent but had only seven training objects from one scene versus 1192
held-out objects.

The fold-0 fusion audit also corrected the interpretation of the tile/global
metric gap. At the model prefilter 1169 GT were matchable; the safety threshold
reduced this to 842 and global fusion to 826. Thus fusion lost only 16 matched
GT, while thresholding lost 327. Border recall was 0.07385 versus 0.19741 for
center objects after fusion, confirming a context/local-detector problem.
Tile-level mAP is not numerically comparable with global mAP because overlapping
tiles duplicate GT in the tile evaluator. Canonical v3 is
`BLOCKED_DEVELOPMENT_DATA_SUPPORT`; C1/C2/P2, test and attacks remain blocked
until additional development-only scenes are introduced under a new protocol.

Before the final A1 fold-0 evaluation was read, the computational-budget
amendment `canonical-v4-person-dg-nwd-expedited-v1` was frozen. It preserves the
same initialization, seed, 5+15 epochs, folds, tiling, evaluator and loss
coefficients. A1 continues to fold 1 only after fold-0 gains of at least 0.05
in mAP50, Recall and small Recall with all technical checks passing. If A1 is
eliminated, A2 is skipped and the same gate is applied to the full A3 stack.
The first candidate passing the two-fold macro gate (mAP50 and Recall at least
0.45, small-Recall gain at least 0.05, worst-fold Recall at least 0.25) alone
runs folds 2-4. This selection remains development-only and cannot open test
before the existing full OOF gate passes.

The prospective `person-v4-train-only-proxy-v1` protocol is independent of the
active A3 fold and is frozen only as a cheap compute filter. Its three proxy
splits use four source scenes and one held-out scene drawn exclusively from the
12-scene training portion of outer fold 0. The three outer-fold-0 held-out
scenes, official validation and test images/labels are forbidden. A0 and A3 use
the same initialization, seed, tiling, fusion and evaluator for 12 epochs.
Proxy PASS requires paired median gains of 0.03 mAP50, 0.03 Recall and 0.05
small Recall, non-negative worst-split Recall delta, at least two improved
splits and all technical checks. It permits only an external scene fold and is
never article evidence. Synthetic four-domain results are implementation
evidence only.
