from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR / "outputs/canonical_v2"
ARTICLE = PROJECT_DIR / "outputs/article"
FORBIDDEN = ["[TBD", "вычисления выполняются", "будут внесены", "статус вычислений"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    docx = ARTICLE / "TNorm_RZD_article_final.docx"
    pdf = ARTICLE / "TNorm_RZD_article_final.pdf"
    supplementary = ARTICLE / "TNorm_RZD_supplementary.pdf"
    resolved = json.loads((ARTICLE / "result_mapping_resolved.json").read_text(encoding="utf-8"))
    plain = subprocess.check_output(["pandoc", str(docx), "-t", "plain"], text=True)
    pdf_plain = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    supplementary_plain = subprocess.check_output(
        ["pdftotext", str(supplementary), "-"], text=True
    )
    quality = json.loads((ROOT / "baseline_rescue_v2/quality_gate.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "config/provenance_final.json").read_text(encoding="utf-8"))
    checkpoint_hashes = set()
    for path in [ROOT / "calibration/threshold_selection.json", ROOT / "raw/canonical_validation.json", ROOT / "raw/canonical_test.json"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_hashes.add(payload.get("checkpoint_sha256"))
    damage = pd.read_csv(ROOT / "tables/08_damage_D2_D3.csv")
    recovery = pd.read_csv(ROOT / "tables/09_recovery_R2_R3.csv")
    signs_valid = True
    for frame in (damage, recovery):
        rows = frame[frame["metric"].eq("delta_mae")]
        for row in rows.itertuples(index=False):
            if math.isfinite(row.estimate) and math.isfinite(row.relative_mae_reduction):
                signs_valid &= (row.estimate < 0) == (row.relative_mae_reduction > 0) or abs(row.estimate) < 1e-15
    required_tables = all(any(path.name.startswith(f"{index:02d}_") for path in (ROOT / "tables").glob("*.csv")) for index in range(1, 16))
    required_figures = all(any(path.name.startswith(f"{index:02d}_") for path in (ROOT / "figures").glob("*.png")) for index in range(1, 11))
    anchors = [
        f"{resolved['clean']['mAP50']:.4f}", f"{resolved['clean']['recall']:.4f}",
        f"{resolved['damage']['delta_r2']:.4f}", f"{resolved['recovery']['delta_r2']:.4f}",
    ]
    checks = {
        "docx_exists": docx.is_file(), "pdf_exists": pdf.is_file(),
        "supplementary_exists": supplementary.is_file(),
        "no_forbidden_status_text": not any(
            value.lower() in plain.lower() or value.lower() in pdf_plain.lower()
            or value.lower() in supplementary_plain.lower()
            for value in FORBIDDEN
        ),
        "quality_gate_passed": bool(quality["passed"]),
        "manifest_hash_matches": (ROOT / "split/split_v2_hash.txt").read_text().split()[0] == "bfd82413e4f92a935e8efb048a2c0edb03a97d932dd2a9303fa6220996b5e5db",
        "checkpoint_hash_consistent": len(checkpoint_hashes) == 1 and None not in checkpoint_hashes,
        "provenance_passed": provenance.get("status") == "PASS",
        "required_tables_present": required_tables,
        "required_figures_present": required_figures,
        "delta_mae_signs_valid": bool(signs_valid),
        "abstract_and_conclusion_numbers_match": all(anchor in plain for anchor in anchors),
        "source_docx_unchanged": (
            resolved["source_docx_sha256_before"] == resolved["source_docx_sha256_after"]
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "docx": str(docx), "docx_sha256": sha256(docx),
        "pdf": str(pdf), "pdf_sha256": sha256(pdf),
        "supplementary": str(supplementary),
        "supplementary_sha256": sha256(supplementary),
    }
    (ARTICLE / "article_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("Final article validation failed")


if __name__ == "__main__":
    main()
