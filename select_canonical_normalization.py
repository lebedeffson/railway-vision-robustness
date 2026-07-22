from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
AUDIT = ROOT / "tables/02_normalization_ablation.csv"
MANIFEST = ROOT / "normalization/normalization_manifest.json"
OUTPUT = ROOT / "normalization/normalization_selection.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit = pd.read_csv(AUDIT)
    candidates = ["N1_quantile", "N2_robust_sigmoid"]
    diagnostics = {}
    selected = None
    for mode in candidates:
        scope = audit[audit["normalization"].eq(mode)]
        maximum = float(scope[["fraction_below_0.01", "fraction_above_0.99"]].max().max())
        diagnostics[mode] = {
            "maximum_saturation_fraction": maximum,
            "layers": sorted(scope["layer"].astype(str).unique()),
            "passed": maximum < .20,
        }
        if selected is None and maximum < .20:
            selected = mode
    if selected is None:
        raise RuntimeError(f"Neither N1 nor predeclared N2 passes saturation: {diagnostics}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("fit_split") != "val" or manifest.get("fit_inputs") != "clean_only":
        raise RuntimeError("Normalization was not fitted on clean validation only")
    payload = {
        "status": "PASS", "selected_normalization": selected,
        "selection_split": "val", "test_used": False,
        "rule": "N1_if_saturation_below_20_percent_else_predeclared_N2",
        "diagnostics": diagnostics, "N3_role": "supplementary_only",
        "statistics_sha256": sha256(ROOT / "normalization/layer_channel_statistics.pt"),
        "fit_manifest_hash": manifest["image_path_hash"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest["selected_normalization"] = selected
    manifest["selection_split"] = "val"
    manifest["test_used_for_selection"] = False
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "pilot/normalization_audit.csv").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "pilot/normalization_audit.csv").write_bytes(AUDIT.read_bytes())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
