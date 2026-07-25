# Candidate-runtime amendment 3

Status: `PROVENANCE_FIX_BEFORE_FIRST_COMPLETED_EPOCH`

After runtime v1b was started, the resumable status file retained the prior
runtime string `v1a`. The locked configuration and code hashes were v1b, but
keeping the stale display identifier would make the execution trace ambiguous.

The service was stopped before a completed epoch, checkpoint, evaluator output
or metric existed. The partial run directories were preserved as rejected
infrastructure artifacts.

Runtime `person-canonical-v5-candidate-runtime-v1c` changes no model, fold,
training, augmentation, gate or selection parameter. Its status writer now
always reads the protocol ID from the current locked runtime and records any
prior ID under `superseded_protocol_ids`.

Railway test and attacks remained sealed.
