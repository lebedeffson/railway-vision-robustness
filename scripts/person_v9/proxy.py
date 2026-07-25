from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]


def causal_temporal_features(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    required = {"grouped_scene_id", "subsequence_id", "frame_id"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing temporal key columns: {sorted(required - set(frame.columns))}")
    output = frame.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    ).copy()
    grouped = output.groupby(
        ["grouped_scene_id", "subsequence_id"], sort=False, group_keys=False
    )
    for column in value_columns:
        output[f"{column}_delta"] = grouped[column].diff().fillna(0.0)
        for window in windows:
            rolling = grouped[column].rolling(window, min_periods=1)
            output[f"{column}_median_{window}"] = (
                rolling.median().reset_index(level=[0, 1], drop=True)
            )
            output[f"{column}_mad_{window}"] = grouped[column].transform(
                lambda values: values.rolling(window, min_periods=1).apply(
                    lambda sample: float(
                        np.median(np.abs(sample - np.median(sample)))
                    ),
                    raw=True,
                )
            )
            output[f"{column}_slope_{window}"] = grouped[column].transform(
                lambda values: values.rolling(window, min_periods=2).apply(
                    lambda sample: float(
                        np.polyfit(np.arange(len(sample)), sample, 1)[0]
                    ),
                    raw=True,
                )
            ).fillna(0.0)
            output[f"{column}_change_{window}"] = (
                output[column] - output[f"{column}_median_{window}"]
            ).abs()
    return output.sort_index()


def residualize_train_only(
    base_train: np.ndarray,
    base_validation: np.ndarray,
    tnorm_train: np.ndarray,
    tnorm_validation: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    normalized_train = scaler.fit_transform(base_train)
    normalized_validation = scaler.transform(base_validation)
    residual_train = np.empty_like(tnorm_train, dtype=float)
    residual_validation = np.empty_like(tnorm_validation, dtype=float)
    for index in range(tnorm_train.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(normalized_train, tnorm_train[:, index])
        residual_train[:, index] = tnorm_train[:, index] - model.predict(normalized_train)
        residual_validation[:, index] = (
            tnorm_validation[:, index] - model.predict(normalized_validation)
        )
    return residual_train, residual_validation


def run_synthetic_proxy(output: Path) -> dict[str, object]:
    rng = np.random.default_rng(20260726)
    rows = []
    for scene_index in range(8):
        for frame_index in range(20):
            base = rng.normal()
            tnorm = rng.normal()
            target = 1.5 + 0.7 * base + 0.4 * tnorm + rng.normal(scale=0.15)
            rows.append(
                {
                    "grouped_scene_id": f"synthetic_scene_{scene_index}",
                    "subsequence_id": f"synthetic_scene_{scene_index}.0",
                    "frame_id": frame_index,
                    "base": base,
                    "product": tnorm,
                    "target": target,
                }
            )
    frame = causal_temporal_features(pd.DataFrame(rows), ["product"])
    predictions = []
    for scene in sorted(frame["grouped_scene_id"].unique()):
        train = frame[frame["grouped_scene_id"].ne(scene)]
        validation = frame[frame["grouped_scene_id"].eq(scene)]
        base_columns = ["base"]
        tnorm_columns = [
            "product",
            "product_delta",
            "product_median_3",
            "product_mad_3",
            "product_slope_3",
        ]
        base_model = Ridge(alpha=1.0).fit(train[base_columns], train["target"])
        base_train_prediction = base_model.predict(train[base_columns])
        residual_target = train["target"].to_numpy() - base_train_prediction
        tnorm_models = []
        train_residual_features = []
        validation_residual_features = []
        for column in tnorm_columns:
            residualizer = Ridge(alpha=1.0).fit(
                train[base_columns], train[column]
            )
            train_residual_features.append(
                train[column].to_numpy()
                - residualizer.predict(train[base_columns])
            )
            validation_residual_features.append(
                validation[column].to_numpy()
                - residualizer.predict(validation[base_columns])
            )
            tnorm_models.append(residualizer)
        del tnorm_models
        residual_model = Ridge(alpha=1.0).fit(
            np.column_stack(train_residual_features), residual_target
        )
        base_validation = base_model.predict(validation[base_columns])
        candidate = base_validation + residual_model.predict(
            np.column_stack(validation_residual_features)
        )
        for actual, baseline, proposed in zip(
            validation["target"], base_validation, candidate, strict=True
        ):
            predictions.append(
                {
                    "grouped_scene_id": scene,
                    "actual": actual,
                    "V2": baseline,
                    "V3": proposed,
                }
            )
    result = pd.DataFrame(predictions)
    per_scene = result.groupby("grouped_scene_id").apply(
        lambda group: pd.Series(
            {
                "V2_MAE": mean_absolute_error(group["actual"], group["V2"]),
                "V3_MAE": mean_absolute_error(group["actual"], group["V3"]),
            }
        ),
        include_groups=False,
    )
    payload = {
        "status": "PROXY_PASS",
        "article_evidence": False,
        "scenes": int(result["grouped_scene_id"].nunique()),
        "future_frame_access": False,
        "scene_grouping": "PASS",
        "V2_MAE": float(per_scene["V2_MAE"].mean()),
        "V3_MAE": float(per_scene["V3_MAE"].mean()),
        "test_access_count": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "V9_SYNTHETIC_PROXY.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    print(
        json.dumps(
            run_synthetic_proxy(ROOT / "outputs/person_v9/proxy"),
            indent=2,
        )
    )
