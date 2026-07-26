# Railway person new scenes v1

Status: `WAITING_FOR_NEW_SCENES`

This is a data-only acquisition protocol. It does not authorize detector
training, tracker tuning, crop-verifier tuning, test access or a scientific
claim.

The complete local temporal direction is immutable. Release `v0.11` is not
recalculated. The next experiment may start only after genuinely new,
independent railway scenes pass the CPU gate.

## Required data

- 8–12 independent railway scenes;
- 1,500–3,000 manually verified frames;
- 100–300 frames per scene;
- at least three camera/capture points;
- at least two illumination conditions;
- at least one 20-frame causal temporal fragment per scene.

Scenes must not be adjacent fragments of an existing development or sealed-test
video. Frames are stratified across person presence, hard-negative-only frames,
small persons, occlusions and entry/exit events.

## Annotation contract

Person annotations are stored separately from the image manifest and include
absolute `xyxy`, occlusion, size bin and border flag. Railway hard negatives are
audit metadata, not detector classes.

The permitted hard-negative categories are signal, pole, catenary structure,
train part, shadow, sign, tile border and other vertical object.

## Leakage boundary

`scene_id` and `sequence_id` are indivisible. A scene and all of its sequences
have exactly one role: train support, held-out or untouched confirmation.
Exact duplicates are forbidden. Perceptual near-duplicates are forbidden
across roles; adjacent frames within one sequence remain allowed for temporal
evaluation.

The sealed-test manifest may be read only as metadata to reject prohibited
scene/sequence identifiers. Test images and labels are never opened.

## Activation

`railway-person-independent-data-v1` remains
`BLOCKED_BY_NEW_SCENES_GATE` until:

1. all four acquisition CSVs exist;
2. `NEW_SCENES_AUDIT.json` is `PASS`;
3. the data inputs and audit are hashed into a new execution lock;
4. test access count is still zero.

Only B1 detector fine-tuning is released first. B2 temporal and B3 crop stages
remain conditional on the preceding prospectively frozen gate.

