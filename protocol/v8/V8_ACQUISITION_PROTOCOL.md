# Canonical v8 person active-data acquisition protocol

Protocol ID: `canonical-v8-person-active-data-v1`

Status: `WAITING_FOR_NEW_DATA`

This protocol is a prospective data-acquisition amendment. It does not authorize
GPU training, validation, test access, adversarial attacks, or scientific claims.

## Fixed scientific decision

The v2-v7 development history is immutable. B0 remains the best independent
development baseline. Fold 0 is a tuning fold and is not reused as untouched
confirmation evidence. The only permitted v8 intervention is the acquisition
and manual annotation of genuinely new, independent railway scenes followed by
the frozen simple B0 recipe.

No NWD, QFL, GroupDRO, MixStyle, SWAD, P2, attention, temporal fusion, tracklet
verification, or instance pasting is permitted in v8.

## Required acquisition

Before training, the development extension must contain at least:

- 3 independent new railway scenes;
- 300 manually verified frames;
- 500 person boxes;
- 40% small or distant persons.

Five scenes, 600 frames, and 1,500 boxes are preferred, not post-hoc gates. New
scenes must differ in station, camera, perspective, illumination, distance,
occlusion, or background. Adjacent frames or numbered parts of an existing
scene are not independent scenes.

B0 may rank frames for annotation but its predictions are never accepted as
ground truth without manual review. Every selected frame has a correction-log
entry, including frames whose annotation was verified unchanged.

## Two locks

`V8_ACQUISITION_LOCK.json` freezes this document, schemas, templates, audit code,
and acquisition rules before data collection.

`outputs/person_v8/protocol/V8_EXECUTION_LOCK.json` may be created only after:

1. the acquisition manifest and correction log are complete;
2. the CPU data gate passes;
3. deterministic screening and OOF folds are generated;
4. all relevant inputs and outputs are hashed;
5. railway test access remains zero.

The GPU runner requires the second lock and verifies every hash. Missing data
therefore produces `WAITING_FOR_NEW_DATA`, not a fabricated PASS or FAIL.

## Test boundary

The sealed-test manifest may be read only as metadata to detect prohibited scene
or source identifiers. Test images and labels must not be opened. Test access,
threshold calibration, attack-budget selection, and attacks remain physically
blocked until full OOF PASS and a separate pre-test freeze.

## Run order

```text
new independent scenes
-> manual person annotation
-> CPU audit
-> deterministic scene-disjoint splits
-> execution lock
-> B0 5/10/20 screening
-> untouched confirmation
-> five-fold OOF
-> OOF gate
-> final model and pre-test freeze
-> one clean test
-> attacks and T-norm diagnostics
```

