# Operator assistant B7-SCF

- B7-SCF was the single owner-authorized terminal amendment after B6. It used
  the same five development scenes, 479 frames, 4,943 observations, 646 frozen
  TrackFragment and 48 GT PersonEpisode for evaluation only. The railway test
  stayed sealed with access count zero.
- Before B7, the B6 audit restored TP=44, FP=16 and FN=750, giving micro
  association F1 0.10304, micro cross-person merge rate 0.26667 and split
  recovery 0.05542. LOSO coefficients, scaler parameters and 36,864 selected
  B6 pseudopair rows were exported.
- The B7 protocol froze robust per-scene median/MAD normalization, a 50/25/15/10
  hard-negative mix, optional one-predecessor/one-successor flow, path penalty
  and 32-run consensus before computation. No GT identity entered training.
- B7-FLOW improved micro F1 to 0.19600 and recovery to 0.11713, but worsened
  cross-person merge rate to 0.40000 and perturbation stability. This is a
  mechanism ablation, not a passed primary result.
- Primary B7-CONSENSUS produced micro F1 0.03206, merge rate 0.23529, recovery
  0.01637 and median perturbation ARI 0.61780. It improved 0/5 scenes and failed
  the frozen scientific gate. Operational gate also failed.
- Direct/replay and two independent process runs matched exactly. Final status
  is `FROZEN_AFTER_B7`; B8 and further threshold/model tuning are prohibited.
