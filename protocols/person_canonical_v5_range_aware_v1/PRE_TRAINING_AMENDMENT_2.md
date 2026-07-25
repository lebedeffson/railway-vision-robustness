# Candidate-runtime amendment 2

Status: `NEW_PROTOCOL_AFTER_PARTIAL_RUN_WITHOUT_RESULT`

Runtime v1a selected exactly 25% or 50% of source frames for an augmentation
attempt. Geometric rejection meant the first completed materialization changed
191 of 954 frames (20.02%) instead of the declared 25%. The output was therefore
rejected before any V5-C/V5-D training.

V5-A fold 0 had started but had no completed stage marker, checkpoint selection,
independent evaluation or viewed metric when the mismatch was detected. The
service was stopped normally and no candidate result was produced.

Runtime `person-canonical-v5-candidate-runtime-v1b` keeps the models, folds,
training recipe, gates and 25%/50% alternatives unchanged. It changes only the
materialization contract:

- deterministically order all train-only geometrically eligible frames;
- continue attempts until the exact requested number of source frames has at
  least one accepted insertion;
- fail if the eligible pool is insufficient;
- require `accepted_frames == target_changed_frames`;
- retain the maximum of two insertions per source frame.

Railway test and attacks remained sealed throughout.
