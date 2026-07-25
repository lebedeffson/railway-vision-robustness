# Pre-training amendment 1

Status: `PROSPECTIVE_BEFORE_FIRST_CANDIDATE_TRAINING`

The first candidate runtime lock exposed a contract gap in the dataset
materializer: the parent protocol permits at most two inserted instances per
source frame, while the tile loop could independently accept one insertion in
more than two tiles.

Before any V5-A/V5-B/V5-C/V5-D training, the runtime was amended to:

- enforce at most two accepted insertions per source frame;
- match an instance's source frame by the exact frame token rather than a
  substring;
- record the observed maximum in the materialization summary;
- fail if the limit is exceeded.

No model result, validation metric, threshold, test label or attack result was
available when this amendment was made. The earlier runtime lock is superseded
by `person-canonical-v5-candidate-runtime-v1a`.
