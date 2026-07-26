# Railway person independent data v1

Status: `BLOCKED_BY_NEW_SCENES_GATE`

This protocol is intentionally not execution-locked. It becomes eligible only
after `railway-person-new-scenes-v1` passes its acquisition CPU gate.

The staged comparison is fixed:

```text
B0 frozen current frame detector
B1 B0 recipe fine-tuned on accepted new train scenes
B2 B1 plus frozen temporal pipeline
B3 B2 plus frozen crop verifier
```

B2 runs only after the detector gate passes. B3 runs only after the temporal
stage passes. Architecture, loss and tracker parameters cannot be changed
together. Test remains sealed until the full system gate passes.

