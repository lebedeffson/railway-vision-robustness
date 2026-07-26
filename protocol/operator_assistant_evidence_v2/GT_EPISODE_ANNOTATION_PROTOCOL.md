# GT person episode protocol

Protocol: `operator-assistant-evidence-v2`

The benchmark uses one development-only OSDaR23 OpenLABEL sequence. Railway
test material is prohibited.

## Episode identity

- A source OpenLABEL object with `type=person` defines one person episode.
- The immutable OpenLABEL object UUID is the episode and person identifier.
- Only boxes for the frozen `rgb_highres_center` stream are used.
- An episode starts at its first frame containing a valid stream-specific box
  and ends at its last such frame.
- Missing boxes inside the object interval are recorded as temporary
  occlusion/missing annotation; they do not create a new episode.
- Leaving and later re-entering under a different source object UUID is a new
  episode. No UUIDs are merged manually.

## Geometry

OpenLABEL boxes are center-x, center-y, width, height. They are converted to
corner coordinates and scaled exactly to the generated 1280 by 720 benchmark.
Every box retains its source frame, source object UUID and source annotation
hash.

## Verification

This benchmark does not use manual identity reconstruction because persistent
source object UUIDs are available. The generator validates object intervals,
box presence, frame order, image dimensions and bounds. Any ambiguous or
missing identity blocks generation rather than being repaired by inference.

The protocol does not involve railway operators and does not measure operator
workload.
