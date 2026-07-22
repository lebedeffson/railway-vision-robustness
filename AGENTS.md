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

## Notes

- `.codex/notes/final-practice.md` — protocol decisions, data footprint and run order.
