from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
MATRIX = ROOT / "raw/canonical_test.csv"
CONFIG = MATRIX.with_suffix(".json")
KEY = [
    "grouped_scene_id", "subsequence_id", "image_path", "checkpoint_sha256",
    "threshold", "attack", "adaptive", "epsilon", "steps", "restart", "seed",
    "defense", "layer", "normalization",
]


def boolean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def signature(row: pd.Series) -> tuple[str, ...]:
    return (
        str(row["attack"]), str(bool(row["adaptive_bool"])).lower(),
        f"{float(row['epsilon_px']):.12g}", str(int(row["steps"])),
        str(int(row["restart"])), str(int(row["seed"])),
        str(row["defense"]), str(row["layer"]), str(row["normalization"]),
    )


def expected_signatures(config: dict) -> set[tuple[str, ...]]:
    rows: set[tuple[str, ...]] = set()
    normalization = config["normalizations"][0]
    for attack, epsilon_px, steps, adaptive in config["conditions"]:
        seeds = [config["seeds"][0]] if attack == "fgsm" else config["seeds"]
        defenses = (
            [name for name in config["defenses"] if name in {"none", "tnorm"}]
            if adaptive else config["defenses"]
        )
        for restart, seed in enumerate(seeds):
            for defense in defenses:
                for layer in ("P3", "P4", "P5"):
                    rows.add((
                        str(attack), str(bool(adaptive)).lower(),
                        f"{float(epsilon_px):.12g}", str(int(steps)),
                        str(restart), str(int(seed)), str(defense), layer,
                        normalization,
                    ))
    return rows


def main() -> None:
    frame = pd.read_csv(MATRIX, low_memory=False)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    missing_columns = sorted(set(KEY + [
        "actual_L1", "actual_L2", "actual_Linf", "attack_loss_clean",
        "attack_loss_final", "nms_timeout", "nms_output_complete",
    ]) - set(frame))
    duplicates = int(frame.duplicated(KEY).sum()) if not missing_columns else -1
    frame["adaptive_bool"] = boolean(frame["adaptive"])
    expected_set = expected_signatures(config)
    missing_condition_count = 0
    unexpected_condition_count = 0
    partial = 0
    for _, scope in frame.groupby("image_path"):
        observed = {signature(row) for _, row in scope.iterrows()}
        missing_condition_count += len(expected_set - observed)
        unexpected_condition_count += len(observed - expected_set)
        partial += int(observed != expected_set)
    expected = len(expected_set)
    linf = pd.to_numeric(frame.get("actual_Linf"), errors="coerce")
    epsilon = pd.to_numeric(frame.get("epsilon"), errors="coerce")
    linf_violations = int((linf > epsilon + 1e-6).sum())
    selected = frame[boolean(frame["selected_best"])]
    restart_selection_violations = 0
    group_key = ["image_path", "attack", "adaptive", "epsilon", "steps"]
    for _, scope in frame.groupby(group_key):
        losses = scope.groupby("restart")["attack_loss_final"].mean()
        selected_restarts = set(scope.loc[boolean(scope["selected_best"]), "restart"])
        if selected_restarts != {int(losses.idxmax())}:
            restart_selection_violations += 1
    timeout = frame[boolean(frame["nms_timeout"])].copy()
    timeout.to_csv(ROOT / "audit/canonical_nms_audit.csv", index=False)
    finite_columns = ["actual_L1", "actual_L2", "actual_Linf", "attack_loss_clean", "attack_loss_final"]
    nonfinite = {
        name: int((~np.isfinite(pd.to_numeric(frame[name], errors="coerce"))).sum())
        for name in finite_columns if name in frame
    }
    checks = {
        "required_columns": not missing_columns,
        "duplicates_zero": duplicates == 0,
        "missing_conditions_zero": missing_condition_count == 0,
        "unexpected_conditions_zero": unexpected_condition_count == 0,
        "partial_frames_zero": partial == 0,
        "expected_320_frames": frame["image_path"].nunique() == 320,
        "five_grouped_scenes": frame["grouped_scene_id"].nunique() == 5,
        "actual_linf_within_budget": linf_violations == 0,
        "best_restart_by_attack_loss": restart_selection_violations == 0,
        "finite_attack_fields": not any(nonfinite.values()),
        "nms_timeout_not_silent_empty": bool(
            timeout.empty or boolean(timeout["nms_output_complete"]).all()
        ),
        "checkpoint_hash_matches_config": frame["checkpoint_sha256"].nunique() == 1
            and frame["checkpoint_sha256"].iloc[0] == config["checkpoint_sha256"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "rows": len(frame), "frames": int(frame["image_path"].nunique()),
        "grouped_scenes": int(frame["grouped_scene_id"].nunique()),
        "rows_per_complete_frame": expected, "partial_frames": partial,
        "duplicates": duplicates, "missing_columns": missing_columns,
        "missing_conditions": missing_condition_count,
        "unexpected_conditions": unexpected_condition_count,
        "linf_violations": linf_violations,
        "restart_selection_violations": restart_selection_violations,
        "nms_timeout_rows": len(timeout), "nonfinite": nonfinite,
    }
    (ROOT / "audit").mkdir(parents=True, exist_ok=True)
    (ROOT / "audit/canonical_matrix_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("Canonical matrix audit failed")


if __name__ == "__main__":
    main()
