# Railway person review assistant v1

The product branch adds an operator queue without changing scientific tag
`v1.0-final-project-closure`. Product status is `OPERATOR_ASSISTANT_MVP`;
autonomous alarming and safety actuation are disabled, and human confirmation
is required.

Frozen inputs are the closure frame detector, OC-SORT configuration and
combined crop verifier. No weights or thresholds are retrained. Candidates are
deduplicated across track-ID changes using a three-second join window and
spatial overlap/center distance. Events reopen for up to 30 seconds in the same
camera region. Clips include three-second pre/post roll.

SQLite stores videos, runs, events, detections, reversible reviews, manual
suppression rules and audit logs. Suppression is never automatic and never
hides an event; stable repetitions move to the lower-priority `Muted` queue,
while changed confidence, motion or geometry raises them again.

Acceptance evidence:

- 1,000 post-warm-up development-train throughput frames;
- 15.69 end-to-end FPS, no real-time claim;
- real frozen-pipeline 100-frame smoke: 296 detections to three events;
- report/database parity;
- public bundle contains no model, video, SQLite, restricted data or test;
- Streamlit screenshots use synthetic media only.

Rebuild the public package with
`python scripts/review_assistant/build_release.py`. The synthetic screenshot
database and benchmark source video stay under ignored local outputs and are
never release assets.
