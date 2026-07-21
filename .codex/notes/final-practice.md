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
