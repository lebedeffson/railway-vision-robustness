from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path


EXPECTED = {
    "B6-FULL": (0.10304449648711946, 0.26666666666666666, 0.055415617128463476, 0.5344611922965019),
    "B7-FLOW": (0.19599578503688095, 0.4, 0.11712846347607053, 0.3572405706468634),
    "B7-HARD": (0.10514018691588785, 0.27419354838709675, 0.05667506297229219, 0.6100347696725423),
    "B7-CONSENSUS": (0.032059186189889025, 0.23529411764705882, 0.0163727959697733, 0.6178043479873313),
}

REQUIRED = {
    "B7_CANDIDATE_EDGES.csv",
    "B7_CROSS_PROCESS_DETERMINISM.json",
    "B7_DECISION.json",
    "B7_FLOW_EDGES.csv",
    "B7_HARD_PSEUDO_PAIR_INDEX.csv",
    "B7_HIERARCHY.jsonl",
    "B7_INPUT_PROVENANCE.json",
    "B7_MODEL_COEFFICIENTS.csv",
    "B7_PROTOCOL_LOCK.json",
    "B7_REPLAY_AUDIT.json",
    "B7_RESULTS_OVERALL.csv",
    "B7_RESULTS_PER_SCENE.csv",
    "B7_SCENE_NORMALIZATION.csv",
    "B7_STABILITY_PER_SCENE.csv",
    "COMPUTE_FREEZE.json",
    "FINAL_TECHNICAL_SUMMARY.md",
    "MANIFEST.sha256",
    "TEST_RESULTS.txt",
    "b6_audit/B6_AUDIT.json",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def verify(archive_path: Path) -> dict[str, object]:
    archive_sha256 = sha256(archive_path.read_bytes())
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if sidecar.is_file():
        expected_sha = sidecar.read_text().split()[0]
        if expected_sha != archive_sha256:
            raise ValueError("ZIP SHA-256 does not match its sidecar")

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED - names)
        if missing:
            raise ValueError(f"Public evidence is incomplete: {missing}")

        for line in archive.read("MANIFEST.sha256").decode().splitlines():
            digest, name = line.split("  ", 1)
            if sha256(archive.read(name)) != digest:
                raise ValueError(f"Manifest mismatch: {name}")

        rows = {
            row["method"]: row
            for row in csv.DictReader(
                io.StringIO(archive.read("B7_RESULTS_OVERALL.csv").decode())
            )
        }
        if set(rows) != set(EXPECTED):
            raise ValueError("Unexpected B6/B7 method set")
        fields = (
            "micro_association_f1",
            "micro_cross_person_merge_rate",
            "micro_same_person_split_recovery",
            "median_perturbation_ari",
        )
        for method, expected_values in EXPECTED.items():
            actual = tuple(float(rows[method][field]) for field in fields)
            if any(abs(a - e) > 1e-12 for a, e in zip(actual, expected_values)):
                raise ValueError(f"Frozen metric mismatch: {method}")

        decision = _json(archive, "B7_DECISION.json")
        freeze = _json(archive, "COMPUTE_FREEZE.json")
        lock = _json(archive, "B7_PROTOCOL_LOCK.json")
        replay = _json(archive, "B7_REPLAY_AUDIT.json")
        determinism = _json(archive, "B7_CROSS_PROCESS_DETERMINISM.json")
        if decision["scientific_decision"] != "FAIL":
            raise ValueError("Scientific decision is not the frozen FAIL")
        if decision["operational_decision"] != "FAIL":
            raise ValueError("Operational decision is not the frozen FAIL")
        if freeze["status"] != "FROZEN_AFTER_B7" or freeze["b8_allowed"]:
            raise ValueError("Compute freeze is inconsistent")
        if lock["test_status"] != "SEALED" or lock["test_access_count"] != 0:
            raise ValueError("Sealed-test contract is inconsistent")
        if not replay["exact_match"] or not replay["all_fragments_preserved"]:
            raise ValueError("Direct/replay audit did not pass")
        if not determinism["exact_match"]:
            raise ValueError("Cross-process determinism did not pass")
        for name, digest in determinism["files"].items():
            if sha256(archive.read(name)) != digest:
                raise ValueError(f"Determinism hash mismatch: {name}")

        forbidden_suffixes = {".pt", ".pth", ".mp4", ".jpg", ".jpeg", ".parquet"}
        forbidden_payloads = (b"/home/", b"/mnt/", b"railway_test")
        for name in names:
            if Path(name).suffix.lower() in forbidden_suffixes:
                raise ValueError(f"Restricted file type in public evidence: {name}")
            payload = archive.read(name)
            if any(marker in payload.lower() for marker in forbidden_payloads):
                raise ValueError(f"Restricted payload in public evidence: {name}")

    return {
        "archive": archive_path.name,
        "sha256": archive_sha256,
        "manifest": "PASS",
        "b6_audit": "PASS",
        "b7_metrics": "PASS",
        "replay": "PASS",
        "cross_process_determinism": "PASS",
        "scientific_gate": "FAIL",
        "operational_gate": "FAIL",
        "test_status": "SEALED",
        "test_access_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen public B7 evidence")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    result = verify(arguments.archive)
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"verification: FAIL ({error})", file=sys.stderr)
        raise SystemExit(1) from error
