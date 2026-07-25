# Project instructions

- Treat manifest `group` as the independent `sequence_id`; adjacent numbered
  OSDaR23 parts must never cross train/val/test or bootstrap folds.
- Run `final_practice_preflight.py` and `audit_final_practice.py` before attacks.
- Do not claim white-box robustness without adaptive PGD through the Product filter.
- Keep the failed single-threshold defense policy as a negative result.
- Do not create a final delivery ZIP before stage-2 `weights/best.pt` exists.
- The official download host may fail local DNS; use `download_osdar23_direct.py`
  with its pinned official-host IP and delete each full archive after selective extraction.
  The host does not support Range, so incomplete `.part` files must restart from zero.
- Treat a raw sequence as complete only when every high-resolution camera URI in
  its OpenLABEL JSON resolves to an existing file.
- If the non-resumable host repeatedly drops a large archive, use
  `stream_extract_osdar23.py`; it retains extracted RGB frames between attempts.
- The frozen fallback corpus is 1405 OpenLABEL-referenced images; only the 19
  paths in `config/raw_frame_exclusions.json` may be absent or skipped.
- Fast mode uses two non-overlapping dataset shards with four workers each;
  training keeps batch size 1 but uses two dataloader workers and no cooldown.
- The persistent training unit must not wrap `ExecStart` in `systemd-inhibit`;
  after reboot the user service may lack interactive authorization for inhibitors.
- Treat `outputs/pipeline_status.json` as the canonical resumable stage contract;
  reconcile it from checkpoint/output evidence with `pipeline_status.py sync`.
- The final matrix keeps six clean/non-adaptive defenses, while white-box Product
  evaluation uses adaptive PGD-20 and PGD-40 only; JPEG/median require BPDA.
- Resolve every post-training evaluator through `config/checkpoint_selection.json`;
  never silently substitute `last.pt` or hard-code a stage checkpoint.
- Keep Q1 revision outputs isolated under `outputs/final_practice/revision_q1`;
  fit normalization and scene thresholds on clean validation only, never test.
- Q1 inference uses Stage 2 best as primary and Stage 1 best only for the frozen
  cross-checkpoint sensitivity protocol; all resampling clusters by `sequence_id`.
- Validation has only three independent sequences. The 12-frame deadline pilot
  is a code smoke test, never article evidence or a substitute for cluster CIs.
- On-disk split identity is train=1057, val=198, test=150; an older narrative
  reversed val/test counts. Trust `split_manifest.csv`, not the prose count.
- Legacy feature `product/godel/lukasiewicz` columns are compatibility scores.
  Scientific Q1 claims must use the formal pointwise T-norm implementation.
- Deadline mode keeps full Q1 masked/deferred. Canonical test budgets must be
  frozen from the validation pilot's floor-effect audit; test cannot select them.
- Report `delta_mae = MAE_new - MAE_baseline` and relative MAE reduction in
  percent, so an improvement has negative delta and positive reduction.
- With three independent scenes, always report per-scene, equal-weight macro and
  leave-one-scene-out results; do not turn bootstrap repeats into generality claims.
- The 43 OSDaR23 directories collapse to 20 independent grouped scenes. Canonical
  v2 is frozen at 10/5/5 scenes; never treat numbered subsequences as independent.
- The current-split legacy test is prohibited. Preserve validation as a legacy
  baseline, then use `tnorm-canonical-v2.service` and its quality/floor gates.
- Calibrate canonical confidence only after v2 training on v2 validation; the
  calibration, clean test and attack configs must share one checkpoint SHA-256.
- Start canonical v2 only from the audited atomic legacy completion marker and
  hold `outputs/locks/canonical_v2.lock` for the entire pipeline.
- Canonical v2 primary recovery R3 includes Product/Lukasiewicz recovery plus
  raw G and C_def; use `config/canonical_v2_analysis.yaml`, not the legacy Q1 R3.
- Do not treat quantile/histogram diagnostic sampling as normalization fitting:
  fit statistics stay exact on clean validation, while diagnostics may use the
  deterministic bounded sample to avoid `torch.quantile` allocation failure.
- Final article generation must preserve the read-only expanded source DOCX;
  only generated files under `outputs/article` may be filled or converted.
- Canonical v2 clean test is opened only after validation pilot, normalization
  selection and non-floor attack budgets are frozen; validation attacks are not test.
- Final article content comes from the read-only expanded DOCX in `Downloads`;
  canonical outputs are `TNorm_RZD_article_final.{docx,pdf}` plus supplementary PDF.
- Canonical v2 stopped at the frozen validation quality gate: mAP50=0.11896
  and standard-point Recall=0.17310. Do not open test or attacks without a new,
  prospectively frozen rescue protocol.
- Rescue v1 is frozen on `feat/canonical-v2-rescue-v1`; its CPU audits passed
  and a separate rescue dataset removes one exact train bbox duplicate without
  changing split or test labels. Resume at micro-overfit; CUDA invisibility is
  an infrastructure block, not a scientific gate failure.
- Canonical v3 development-pool CPU audit is blocked by class-scene support:
  animal occurs in only two development scenes. Do not run C1/C2/P2 training
  until new development-only scenes are frozen under a new protocol.
- Person-only v3 also stopped before test: two-fold macro mAP50=0.35883,
  Recall=0.35432 and worst-fold Recall=0.20129. Preserve this hard-fail baseline.
- Canonical v4 is the frozen A0-A3 DG/NWD matrix. Test and attacks remain blocked
  unless one candidate clears all two-fold deltas and then the full five-fold OOF gate.
- The prospective `canonical-v4-person-dg-nwd-expedited-v1` amendment stops the
  full A0-A3 matrix after A1 fold 0 and selects only A1 or A3 through locked
  fold-0 and two-fold gates. A2 is skipped; test remains sealed until full OOF PASS.
- `person-v4-train-only-proxy-v1` is a compute filter using only outer-fold-0
  training scenes. Its PASS permits one external fold, never test or article claims.
- Canonical v5 is data-first: simple YOLO ERM with CrowdHuman visible-person
  pretraining and railway train-only hard mining. Do not reintroduce NWD/QFL,
  GroupDRO, MixStyle or SWAD, and do not run RT-DETR without a new amendment.
- CrowdHuman images are non-commercial research/education only and may not be
  redistributed. Require explicit terms acknowledgement; never place its images,
  archives or derived dataset in Git or release bundles.
- Person canonical v5 range-aware candidate runtime `v1c` is locked on
  `feat/person-canonical-v5-range-aware`: V5-E is excluded by the common
  B0/B1/D1 diagnostic, V5-A/B/C/D run under the user service, and railway test
  and attacks stay sealed until a full development gate passes.
- Preserve `protocol/v5/V5_EXECUTION_LOCK.json` for official person-v5 runs:
  25% pasting is primary, 50% is fold-0 sensitivity only, evaluation uses
  confidence 0.07, and test/attacks stay blocked until full OOF PASS.
- Never add StarBlock to current V5-B or V5-D. Activate the separate
  `feat/person-canonical-v5b-starcoord` protocol only after all four primary
  v5 candidates receive terminal two-fold FAIL; v5b excludes P2.
- `canonical-v5-expedited-screening-v1` is compute screening only: C0/C1/C2
  use 5/10/20-epoch successive halving and cannot open test or support article
  claims. Every derived `nc: 1` label file must be re-encoded as class 0.
- Canonical v6 T0 is causal and development-only: temporal windows never cross
  `subsequence_id`, invalid homographies contribute zero evidence, and fold 1
  must not be read unless T0-D passes the frozen fold-0 gate.
- Canonical v6 T0 ended at fold-0 FAIL: Product fusion improved Recall only
  +0.00322 and small Recall +0.00357. Keep fold 1, T1, test and attacks blocked.

## Notes

- `.codex/notes/final-practice.md` — protocol decisions, data footprint and run order.
