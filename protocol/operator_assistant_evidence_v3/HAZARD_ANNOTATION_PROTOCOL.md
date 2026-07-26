# Hazard event annotation protocol

Two authors independently assign every existing GT person episode in one scene
to a hazard situation. A hazard situation is a shared spatial-temporal context,
not a detector track. Episodes from different scenes can never share a hazard
identifier. Temporary occlusion does not create a new hazard event. A new
spatial context or a non-overlapping later occurrence requires a new identifier.

Each author submits a complete CSV using the locked template. The adjudicated
CSV is accepted only when every person episode is referenced exactly once,
intervals are valid, authors are distinct, and every disagreement has an
explicit adjudication status. Missing or unknown assignments block hazard-level
evaluation.
