from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit_final_practice import canonical_path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
AUDIT = PROJECT_DIR / "outputs/final_practice/audit"
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
EVENT = re.compile(r"(?P<progress>(?P<done>\d+)/(?P<total>\d+))|(?P<warning>NMS time limit)")
KEYS = [
    "sequence_id", "image_path", "attack", "adaptive", "epsilon", "steps", "restart", "seed",
    "defense", "layer",
]
DETECTION_COLUMNS = [
    "f1_clean", "f1_attack", "f1_defended", "recall_clean", "recall_attack",
    "recall_defended", "fn_clean", "fn_attack", "fn_defended", "false_negatives",
    "confidence_drop", "iou_shift",
]
NMS_COLUMNS = [
    "nms_timeout", "nms_timeout_clean", "nms_timeout_attack",
    "nms_timeout_defended", "nms_runtime_ms", "nms_runtime_clean_ms",
    "nms_runtime_attack_ms", "nms_runtime_defended_ms",
    "predictions_before_or_after_timeout", "predictions_after_nms",
    "nms_candidates_before", "nms_output_complete",
]


def message_text(value: object) -> str:
    if isinstance(value, list):
        return bytes(int(item) for item in value).decode("utf-8", errors="replace")
    return str(value)


def warning_image_indices(total: int) -> list[dict[str, object]]:
    command = [
        "journalctl", "--user", "-u", "tnorm-wait-train.service",
        "--since", "2026-07-22 00:20:00", "-o", "json", "--all", "--no-pager",
    ]
    output = subprocess.check_output(command, text=True)
    completed = 0
    current_total = None
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = message_text(record.get("MESSAGE", ""))
        for event in EVENT.finditer(text):
            if event.group("progress"):
                completed = int(event.group("done"))
                current_total = int(event.group("total"))
            elif current_total == total:
                timestamp = int(record.get("__REALTIME_TIMESTAMP", 0)) / 1_000_000
                rows.append({
                    "journal_timestamp_unix": timestamp,
                    "completed_images_before_warning": completed,
                    "image_index_zero_based": completed,
                    "matrix_total_images": current_total,
                })
    return rows


def image_order(data: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(data["image_path"].astype(str)))


def subset_data_yaml(images: list[str], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    image_list = destination / "nms_timeout_images.txt"
    image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    config = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    config["val"] = str(image_list)
    config["test"] = str(image_list)
    data_yaml = destination / "nms_timeout_data.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_yaml


def run_matrix(data_yaml: Path, output: Path, max_time: float) -> None:
    if output.is_file() and output.with_suffix(".json").is_file():
        return
    subprocess.run([
        sys.executable, "run_final_matrix.py", "--data", str(data_yaml),
        "--split", "val", "--workers", "0", "--checkpoint-name", "stage2_best",
        "--nms-max-time-img", str(max_time), "--output", str(output),
    ], cwd=PROJECT_DIR, check=True)


def bool_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted legacy NMS warning audit")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, low_memory=False)
    ordered_images = image_order(data)
    warning_rows = warning_image_indices(len(ordered_images))
    for row in warning_rows:
        index = int(row["image_index_zero_based"])
        row["image_path"] = ordered_images[index] if index < len(ordered_images) else None
    legacy = pd.DataFrame(warning_rows)
    affected = [
        image for image in dict.fromkeys(legacy.get("image_path", pd.Series(dtype=str)).dropna())
    ]
    low_path = args.output / f"nms_timeout_low_{args.split}.csv"
    high_path = args.output / f"nms_timeout_high_{args.split}.csv"
    corrected_rows = 0
    exact_timeout_rows = 0
    rerun_error = None
    cases = pd.DataFrame()
    if affected:
        try:
            data_yaml = subset_data_yaml(affected, args.output / "nms_timeout_subset")
            run_matrix(data_yaml, low_path, 0.05)
            run_matrix(data_yaml, high_path, 10.0)
            low = pd.read_csv(low_path, low_memory=False)
            high = pd.read_csv(high_path, low_memory=False)
            low["image_path"] = low["image_path"].map(canonical_path)
            high["image_path"] = high["image_path"].map(canonical_path)
            data["image_path"] = data["image_path"].map(canonical_path)
            exact = low[bool_values(low["nms_timeout"])].copy()
            exact_timeout_rows = len(exact)
            cases = exact.merge(
                high[[*KEYS, *NMS_COLUMNS, *DETECTION_COLUMNS]],
                on=KEYS, suffixes=("_low", "_high"), validate="one_to_one",
            )
            changed = np.zeros(len(cases), dtype=bool)
            for column in DETECTION_COLUMNS:
                changed |= ~np.isclose(
                    cases[f"{column}_low"].to_numpy(float),
                    cases[f"{column}_high"].to_numpy(float),
                    equal_nan=True, atol=1e-9, rtol=1e-9,
                )
            cases["metrics_changed"] = changed
            cases["legacy_warning_count_for_image"] = cases["image_path"].map(
                legacy.assign(image_path=legacy["image_path"].map(canonical_path))[
                    "image_path"
                ].value_counts()
            )
            cases["batch_size"] = 1
            cases["legacy_output_complete_by_control_flow"] = True
            # Add audit fields to every legacy row, then patch only confirmed exact conditions.
            for column in NMS_COLUMNS:
                if column not in data:
                    data[column] = False if "timeout" in column else np.nan
            data["exclude_from_primary_statistics"] = False
            low_audit = low[[*KEYS, *NMS_COLUMNS]].copy()
            data = data.merge(low_audit, on=KEYS, how="left", suffixes=("", "_audit"))
            for column in NMS_COLUMNS:
                audit_column = f"{column}_audit"
                data[column] = data[audit_column].combine_first(data[column])
                data.drop(columns=[audit_column], inplace=True)
            if changed.any():
                replacement = high.merge(
                    cases.loc[cases["metrics_changed"], KEYS], on=KEYS, how="inner"
                )
                replacement = replacement.set_index(KEYS)
                indexed = data.set_index(KEYS)
                for key, row in replacement.iterrows():
                    for column in DETECTION_COLUMNS:
                        indexed.loc[key, column] = row[column]
                corrected_rows = len(replacement)
                data = indexed.reset_index()
            temporary = args.input.with_suffix(args.input.suffix + ".nms-audit.tmp")
            data.to_csv(temporary, index=False)
            temporary.replace(args.input)
        except Exception as error:
            rerun_error = f"{type(error).__name__}: {error}"
            affected_canonical = {canonical_path(value) for value in affected}
            data["exclude_from_primary_statistics"] = data["image_path"].map(
                canonical_path
            ).isin(affected_canonical)
            temporary = args.input.with_suffix(args.input.suffix + ".nms-audit.tmp")
            data.to_csv(temporary, index=False)
            temporary.replace(args.input)
    case_columns = [
        "sequence_id", "image_path", "attack", "adaptive", "epsilon", "steps",
        "restart", "seed", "defense",
        "nms_timeout_clean_low", "nms_timeout_attack_low", "nms_timeout_defended_low",
        "nms_runtime_ms_low", "predictions_after_nms_low", "nms_runtime_ms_high",
        "predictions_after_nms_high", "metrics_changed", "batch_size",
        "legacy_output_complete_by_control_flow",
    ]
    for column in case_columns:
        if column not in cases:
            cases[column] = pd.Series(dtype=object)
    cases[case_columns].drop_duplicates().to_csv(
        args.output / "nms_timeout_cases.csv", index=False
    )
    sensitivity = {
        "rows_excluded": (
            int(bool_values(data["exclude_from_primary_statistics"]).sum())
            if "exclude_from_primary_statistics" in data else 0
        ),
        "reason": rerun_error,
    }
    (args.output / "nms_timeout_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "status": "PASS" if rerun_error is None else "RERUN_FAILED_EXCLUDED",
        "legacy_warning_count": len(legacy),
        "legacy_affected_images": len(affected),
        "exact_timeout_rows_in_instrumented_rerun": exact_timeout_rows,
        "corrected_rows": corrected_rows,
        "rerun_error": rerun_error,
        "batch_size": 1,
        "ultralytics_control_flow": (
            "time limit is checked after output assignment; batch=1 legacy outputs are complete"
        ),
    }
    (args.output / "nms_timeout_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
