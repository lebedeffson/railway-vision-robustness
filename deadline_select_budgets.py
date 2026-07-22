from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "outputs/final_practice/deadline/pilot/raw/pilot_matrix.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "outputs/final_practice/deadline/config/canonical_budget_selection.json"
)


def boolean_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def budget_table(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sequence_id", "image_path", "attack", "adaptive", "epsilon_px", "steps",
        "defense", "layer", "selected_best", "f1_clean", "recall_clean",
        "f1_attack", "recall_attack",
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(f"Budget selection input lacks {sorted(missing)}")
    scope = data[
        boolean_series(data["selected_best"])
        & data["defense"].eq("none")
        & data["layer"].eq("P3")
    ].copy()
    scope["adaptive_bool"] = boolean_series(scope["adaptive"])
    scope = scope.drop_duplicates([
        "sequence_id", "image_path", "attack", "adaptive_bool", "epsilon_px", "steps"
    ])
    rows: list[dict[str, object]] = []
    keys = ["attack", "adaptive_bool", "epsilon_px", "steps"]
    for key, group in scope.groupby(keys, sort=True):
        attack, adaptive, epsilon_px, steps = key
        absolute_floor = (group["f1_attack"] <= 1e-12) | (
            group["recall_attack"] <= 1e-12
        )
        clean_detectable = (group["f1_clean"] > 1e-12) & (
            group["recall_clean"] > 1e-12
        )
        eligible = group[clean_detectable]
        conditional_floor = (eligible["f1_attack"] <= 1e-12) | (
            eligible["recall_attack"] <= 1e-12
        )
        rows.append({
            "attack": attack, "adaptive": bool(adaptive),
            "epsilon_px": float(epsilon_px), "steps": int(steps),
            "frames": int(group["image_path"].nunique()),
            "sequences": int(group["sequence_id"].nunique()),
            "absolute_floor_fraction": float(absolute_floor.mean()),
            "clean_detectable_frames": int(clean_detectable.sum()),
            "conditional_floor_fraction": (
                float(conditional_floor.mean()) if len(eligible) else np.nan
            ),
            "absolute_pass": bool(absolute_floor.mean() < 0.5),
            "conditional_pass": bool(
                len(eligible) and conditional_floor.mean() < 0.5
            ),
        })
    return pd.DataFrame(rows)


def select_budgets(table: pd.DataFrame, allow_conditional: bool = True) -> dict[str, object]:
    selected: dict[str, list[float]] = {}
    evidence: dict[str, object] = {}
    families = {
        "fgsm_epsilon_px": ("fgsm", False, 2),
        "pgd_epsilon_px": ("pgd", False, 2),
        "adaptive_pgd_epsilon_px": ("pgd", True, 1),
    }
    failed: list[str] = []
    for output_name, (attack, adaptive, limit) in families.items():
        scope = table[
            table["attack"].eq(attack) & table["adaptive"].eq(adaptive)
            & table["steps"].eq(1 if attack == "fgsm" else 20)
        ].sort_values("epsilon_px")
        absolute = scope[scope["absolute_pass"]]
        if len(absolute):
            accepted, basis = absolute, "absolute_all_validation_frames"
        elif allow_conditional:
            accepted, basis = scope[scope["conditional_pass"]], (
                "conditional_on_clean_detectable_due_to_baseline_floor"
            )
        else:
            accepted, basis = scope.iloc[0:0], "strict_absolute_floor_required"
        values = accepted["epsilon_px"].head(limit).astype(float).tolist()
        selected[output_name] = values
        evidence[output_name] = {
            "selection_basis": basis, "selected": values,
            "available_absolute_pass": absolute["epsilon_px"].astype(float).tolist(),
            "available_conditional_pass": scope.loc[
                scope["conditional_pass"], "epsilon_px"
            ].astype(float).tolist(),
        }
        if not values:
            failed.append(output_name)
    return {
        "status": "PASS" if not failed else "FAIL",
        "selected": selected,
        "evidence": evidence,
        "failed_families": failed,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "floor_threshold": 0.5,
        "warning": (
            "Conditional selection is descriptive because the clean detector itself "
            "has a substantial image-level floor."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze non-floor canonical budgets on validation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-absolute", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False)
    table = budget_table(data)
    payload = select_budgets(table, allow_conditional=not args.strict_absolute)
    payload["validation_matrix_path"] = str(args.input.resolve())
    payload["validation_matrix_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    payload["tested_budgets"] = table.to_dict(orient="records")
    payload["selection_rule"] = "absolute floor fraction below 0.5 on validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "frozen_attack_budgets.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
