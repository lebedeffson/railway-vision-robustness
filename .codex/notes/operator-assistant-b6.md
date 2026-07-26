# Operator assistant B6

- B6 is the final computation over the five frozen v3 development scenes:
  479 frames, 48 evaluation-only GT PersonEpisode and 4,943 identical
  pre-aggregation observations. The railway test stayed sealed with access
  count zero.
- Frozen DINOv2 Small appearance and Video Depth Anything Small depth were
  used without fine-tuning. The official depth input size was reduced from
  518 to 280 after a pre-metric CUDA OOM; the amendment is recorded under
  `protocol/operator_assistant_b6/`.
- B6 logistic association models were trained only on automatic pseudo-pairs
  in five LOSO folds. GT PersonEpisode identifiers were assigned only after
  feature extraction and were used only for evaluation.
- B6-FULL versus B5: macro association F1 0.26561 versus 0.19959;
  cross-person merge rate 0.10466 versus 0.19667; split recovery 0.23741
  versus 0.12218; coverage was unchanged at 0.90462. B6-FULL improved
  association F1 on only 3/5 scenes and median pairwise perturbation ARI was
  0.53446, below the frozen 0.90 gate.
- Final decision is `FAIL`. Direct/replay matched, all 646 TrackFragment were
  preserved, and the B6 hierarchy explicitly uses ReviewBundle rather than
  semantic HazardEvent.
- Compute status is `FROZEN_AFTER_B6`; further model experiments and threshold
  tuning are prohibited. The public bundle excludes frames, models, feature
  tensors and absolute local paths.
