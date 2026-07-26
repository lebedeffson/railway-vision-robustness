# Railway person new scenes v1

Protocol `railway-person-new-scenes-v1` is an acquisition-only boundary after
the terminal crop-verifier FAIL. Its implementation commit is recorded in
`protocol/new_scenes_v1/ACQUISITION_LOCK.json`; do not edit locked files under
the same protocol ID.

Current state:

- `CLOSED_DATA_UNAVAILABLE`;
- zero accepted scenes and no training authorization;
- automatic download and training are disabled;
- `railway-person-independent-data-v1` will not be activated;
- railway test is `SEALED`, access count zero;
- release v0.11 and its negative results are immutable.

The acquisition phase was closed by
`railway-vision-final-closure-v1`. Do not resume downloads, retry mirrors,
request restricted sources, or authorize a new training run under this
protocol.

The former acquisition requirements and staged B0-B3 activation remain
historical protocol evidence only. They no longer authorize downloads,
ingestion or training.

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
