# Railway vision final closure v1

`protocol/project_closure_v1/PROJECT_CLOSURE_LOCK.json` is the terminal project
contract. The project is `COMPLETED_RESEARCH`, the practical artifact is
`RESEARCH_DEMONSTRATOR_ONLY`, deployment is `FAIL`, railway test is `SEALED`
with access count zero, and further tuning on the existing development pool is
prohibited.

The final demonstrator has two frozen modes:

- `frame_baseline` for conservative frame detections;
- `temporal_research` for recall-oriented OC-SORT plus the frozen combined
  verifier. It must display the high-false-alarm and non-deployment warning.

Generated closure artifacts live under `outputs/project_closure_v1/`. Rebuild
them with `/home/lebedeffson/Code/venv/bin/python -m
scripts.project_closure.finalize`. The public archive must exclude data,
videos, checkpoints, restricted annotations, feature tensors, local paths and
secrets. The internal audit archive is never a release asset.

Acceptance requires the complete repository test suite, a clean worktree,
verified SHA-256 sidecars, the `v1.0-final-project-closure` tag, and a GitHub
release containing only the approved public artifacts.
