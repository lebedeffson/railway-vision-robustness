# Person V5 development diagnostic

All four states were evaluated with the same frozen 2x2 tiling, fusion, confidence threshold 0.07, IoU 0.50, folds and class mapping.

| state   | macro_mAP50 | macro_mAP50_95 | macro_precision | macro_recall | macro_F1 | worst_fold_recall | mean_FN_per_frame | mean_FP_per_frame |
| ------- | ----------- | -------------- | --------------- | ------------ | -------- | ----------------- | ----------------- | ----------------- |
| B0      | 0.358828    | 0.195942       | 0.548954        | 0.354316     | 0.383701 | 0.201288          | 4.632590          | 1.672106          |
| B1      | 0.270704    | 0.154533       | 0.522935        | 0.242420     | 0.296396 | 0.090982          | 5.350445          | 1.049933          |
| D1_best | 0.298774    | 0.173443       | 0.248900        | 0.332638     | 0.181783 | 0.181965          | 4.765477          | 14.157329         |
| D1_last | 0.073107    | 0.031254       | 0.147078        | 0.345872     | 0.140207 | 0.188406          | 4.700539          | 28.098589         |

Evaluator parity: **PASS**. No test image or label was accessed. No GT was lost and no NaN/Inf was observed.

CrowdHuman zero-shot does not exceed D1-best in macro Recall or worst-fold Recall. Therefore gradual transfer (V5-E) is excluded by the rule frozen before this diagnostic.
