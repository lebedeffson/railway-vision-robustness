# TNormFilter revision Q1 hypotheses

The protocol is frozen before revision test analysis. Test data cannot select
checkpoints, normalization, features, thresholds, folds, or model families.

## H1 — feature damage

After accounting for attack type, epsilon, perturbation L1/L2/Linf and standard
feature distances, canonical Product and Lukasiewicz metrics improve prediction of
`f1_clean - f1_attacked`. Recall damage, confidence drop and false-negative
increase are secondary endpoints. Goedel is retained only as a supplementary
redundancy ablation because the legacy Product/Goedel rank correlation is 0.99199.

## H2 — feature recovery

T-norm recovery metrics add predictive information for
`f1_defended - f1_attacked` beyond cosine, L1, normalized L2, MSE, Pearson and
entropy recovery. Normalized quality recovery is not clipped for inference.

## H3 — checkpoint sensitivity

The sign and ranking of the main T-norm increment are consistent between the
validation-selected Stage 2 best checkpoint and Stage 1 best. This is a
cross-checkpoint sensitivity test, not evidence across architectures.

## H4 — scene difficulty

The T-norm increment remains directionally consistent after stratification by
validation-frozen object-count and small-object-fraction tertiles. Strata with
fewer than five independent sequences are explicitly exploratory.

## Claim rule

Scientific advantage requires a Holm-corrected p-value below 0.05 and a paired
sequence-bootstrap 95% interval excluding zero. Practical importance is reported
separately using the frozen MAE, R2, and Spearman thresholds. Negative outcomes
are retained. With only three independent validation/test scenes, corrected
p-values remain descriptive for the observed scenes and cannot support a strong
generalization claim; scene-wise, macro-average and leave-one-scene-out results
are mandatory.
