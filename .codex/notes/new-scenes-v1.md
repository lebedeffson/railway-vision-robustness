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

Acquisition execution on 2026-07-26:

- the supplied starter ZIP passed outer SHA-256
  `14a56ecbd9b6b449fe2441c37e5c53fee97e5e7e3ef8aea5bdb6fa5822b3909c`
  and its internal manifest;
- RailEye3D public annotations were frozen at commit
  `ce1b2cf8434712a28c1fc2757724687c52844cd6` (10 MOT files), while images
  still require owner approval;
- the RailGoerl24 host did not support HTTP Range and reset two transfers, the
  best at 4,051,124,215 of 4,055,218,180 bytes. No SHA-256, extraction, or
  accepted scene may be recorded until a complete archive passes `7z t`;
- RAIL-BENCH terms were not accepted and RAWPED was not requested.
