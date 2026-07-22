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
- Deadline mode keeps full Q1 masked/deferred; after the validation pilot passes,
  canonical test inference uses the frozen 114-row minimal grid and is finalized
  only through `deadline_finalize.py` after the standard delivery bundle.

## Notes

- `.codex/notes/final-practice.md` — protocol decisions, data footprint and run order.
