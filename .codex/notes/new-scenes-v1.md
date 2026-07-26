# Railway person new scenes v1

Protocol `railway-person-new-scenes-v1` is an acquisition-only boundary after
the terminal crop-verifier FAIL. Its implementation commit is recorded in
`protocol/new_scenes_v1/ACQUISITION_LOCK.json`; do not edit locked files under
the same protocol ID.

Current state:

- `WAITING_FOR_NEW_SCENES`;
- zero accepted scenes and no training authorization;
- `railway-person-independent-data-v1` is
  `BLOCKED_BY_NEW_SCENES_GATE`;
- railway test is `SEALED`, access count zero;
- release v0.11 and its negative results are immutable.

The audit needs four local CSV inputs under `data/new_scenes_v1/`, using the
tracked templates. It requires 8–12 scenes, 1,500–3,000 frames, at least three
camera/capture points, two illumination conditions, and a causal 20-frame
fragment in every scene. `scene_id`, `sequence_id`, source video, and split role
must remain isolated. Exact duplicates, cross-role perceptual duplicates,
non-monotonic timestamps, and reused old/sealed identifiers fail the gate.

After a real audit PASS, `scripts.new_scenes.authorize_independent_data` hashes
all acquisition inputs and authorizes B1 only. B2 requires the detector gate;
B3 requires the temporal gate. Test remains sealed until the full-system gate.

The public waiting bundle contains configs, schemas, templates, audit status,
and reproduction code only. It excludes images, annotations, checkpoints,
feature tensors, and test material.
