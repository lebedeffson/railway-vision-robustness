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
- Keep long local runs desktop-safe by default: at most three rate-limited download
  workers under a shared CPU/memory cgroup, one Ultralytics dataloader worker,
  low process/IO priority, and an inter-epoch cooldown.

## Notes

- `.codex/notes/final-practice.md` — protocol decisions, data footprint and run order.
