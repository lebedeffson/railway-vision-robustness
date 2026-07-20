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

Long work is intentionally desktop-safe: the persistent download service uses
at most three 12 MiB/s workers under a shared 50% CPU quota, low CPU/IO weights
and selective extraction; training uses one dataloader worker, batch size 1 and
a three-second inter-epoch cooldown.
Stage training resumes from `last.pt`. Final-matrix work checkpoints after every
fully completed image, and `run_training_pipeline.py` records a marker after
each stage so reboots do not repeat completed practice blocks.
