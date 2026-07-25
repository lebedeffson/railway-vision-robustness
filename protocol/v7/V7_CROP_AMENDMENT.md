# Canonical v7 crop-verifier amendment

This prospective amendment is activated only because the locked V0 fold-0
gate ended in `FAIL`. It does not edit or rerun V0.

The frozen CrowdHuman visible-person checkpoint is used only as an RGB crop
encoder. Three contextual proposal crops are selected deterministically for
each tracklet. The encoder remains in evaluation mode and its parameters never
receive gradients.

`V1` trains a small binary MLP with BCE and pairwise ranking loss. `V2` fuses
the OOF V0-B probability and OOF V1 crop probability with non-negative
monotone weights. All fitting, calibration, model selection and threshold
selection use grouped train-scene OOF predictions. Held-out fold 0 is evaluated
only after a new pre-heldout freeze marker is written.

Fold 1, railway test and attacks remain blocked unless the frozen V1/V2 fold-0
gate passes. CrowdHuman images, derived crops and checkpoints are excluded
from public bundles.

