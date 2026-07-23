from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    PROTOCOL_PATH,
    TEST_MARKER,
    atomic_json,
    git,
    load_protocol,
    now,
    sha256,
)


PYTHON = PROJECT_DIR / ".venv/bin/python"
STATUS = OUTPUT_ROOT / "final/post_gate_pipeline_status.json"


def status(stage: str, value: str, **extra: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {"protocol_id": load_protocol()["protocol_id"], "stages": {}}
    )
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {
        **payload["stages"].get(stage, {}),
        "status": value,
        **extra,
    }
    atomic_json(STATUS, payload)


def run_script(name: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", f"scripts/{name}", *arguments],
        cwd=PROJECT_DIR,
        check=True,
    )


def build_prefinal() -> Path:
    protocol = load_protocol()
    source = Path(protocol["article"]["source_template"])
    output = PROJECT_DIR / protocol["article"]["pre_final"]
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    seed = json.loads(
        (OUTPUT_ROOT / "selection/checkpoint_selection.json")
        .read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="canonical-m4-prefinal-") as directory:
        markdown = Path(directory) / "source.md"
        subprocess.run(
            ["pandoc", str(source), "-t", "gfm", "-o", str(markdown)],
            cwd=PROJECT_DIR,
            check=True,
        )
        text = markdown.read_text(encoding="utf-8")
        text += (
            "\n\n# Canonical M4 pre-final validation status\n\n"
            "Micro-overfit применялся только как sanity-проверка технической "
            "обучаемости. M4 был единственным кандидатом, прошедшим micro gate; "
            "обобщение проверено отдельно scene-level CV и validation.\n\n"
            f"Selected seed: {seed['selected_seed']}; checkpoint SHA-256: "
            f"`{gate['checkpoint_sha256']}`; validation mAP50="
            f"{gate['validation_map50']:.4f}; safety Recall="
            f"{gate['validation_safety_recall']:.4f}; quality gate=PASS.\n\n"
            "Test, D2/D3 и R2/R3 на момент этой pre-final версии остаются "
            "зафиксированными маркерами [TBD_*].\n"
        )
        markdown.write_text(text, encoding="utf-8")
        subprocess.run(
            [
                "pandoc", str(markdown), "--reference-doc", str(source),
                "-o", str(output),
            ],
            cwd=PROJECT_DIR,
            check=True,
        )
    return output


def open_test_once() -> dict[str, Any]:
    protocol = load_protocol()
    gate_path = OUTPUT_ROOT / "validation/quality_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("quality_gate_passed"):
        raise RuntimeError("Canonical M4 test remains blocked by failed gate")
    threshold = OUTPUT_ROOT / "selection/checkpoint_selection.json"
    attack = OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json"
    normalization = OUTPUT_ROOT / "normalization/normalization_manifest.json"
    payload = {
        "status": "OPENED_ONCE",
        "opened_at": now(),
        "commit": git("rev-parse", "HEAD"),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "manifest_sha256": protocol["dataset"]["split_manifest_sha256"],
        "threshold_lock_sha256": sha256(threshold),
        "attack_protocol_sha256": sha256(attack),
        "normalization_manifest_sha256": sha256(normalization),
        "test_open_count": 1,
        "test_used_for_tuning": False,
    }
    if TEST_MARKER.is_file():
        current = json.loads(TEST_MARKER.read_text(encoding="utf-8"))
        comparable = {key: current.get(key) for key in payload if key != "opened_at"}
        expected = {key: value for key, value in payload.items() if key != "opened_at"}
        if comparable != expected:
            raise RuntimeError("Existing TEST_OPENED marker differs from frozen locks")
        return current
    atomic_json(TEST_MARKER, payload)
    gate["test_opened"] = True
    gate["test_open_marker_sha256"] = sha256(TEST_MARKER)
    atomic_json(gate_path, gate)
    return payload


def main() -> None:
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    if not gate.get("quality_gate_passed"):
        raise RuntimeError("Post-gate pipeline called after validation FAIL")
    try:
        if TEST_MARKER.is_file():
            required = [
                OUTPUT_ROOT / "normalization/normalization_manifest.json",
                OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json",
            ]
            if not all(path.is_file() for path in required):
                raise RuntimeError(
                    "Test marker exists without completed validation-only locks"
                )
            status("test_open", "success", resumed_after_restart=True)
        else:
            status("pre_final_article", "running", started_at=now())
            prefinal = build_prefinal()
            status(
                "pre_final_article", "success", finished_at=now(),
                output=str(prefinal.resolve()), output_sha256=sha256(prefinal),
            )
            checkpoint = str(gate["checkpoint"])
            normalization_lock = (
                OUTPUT_ROOT / "normalization/normalization_manifest.json"
            )
            if not normalization_lock.is_file():
                status("normalization", "running", started_at=now())
                run_script(
                    "fit_canonical_m4_normalization.py",
                    "--checkpoint",
                    checkpoint,
                )
                status("normalization", "success", finished_at=now())
            attack_lock = (
                OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json"
            )
            if not attack_lock.is_file():
                status("attack_calibration", "running", started_at=now())
                run_script("calibrate_canonical_m4_attacks.py")
                status("attack_calibration", "success", finished_at=now())
            open_test_once()
            status(
                "test_open", "success", finished_at=now(),
                marker_sha256=sha256(TEST_MARKER), test_open_count=1,
            )
        status("clean_test", "running", started_at=now())
        run_script("evaluate_canonical_m4_clean_test.py")
        status("clean_test", "success", finished_at=now())
        status("canonical_attack_matrix", "running", started_at=now())
        run_script("run_canonical_m4_test_matrix.py")
        status("canonical_attack_matrix", "success", finished_at=now())
        status("statistics", "running", started_at=now())
        run_script("analyze_canonical_m4.py")
        status("statistics", "success", finished_at=now())
        status("latency", "running", started_at=now())
        run_script("benchmark_canonical_m4_latency.py")
        status("latency", "success", finished_at=now())
        status("article_and_bundle", "running", started_at=now())
        run_script("finalize_canonical_m4.py")
        final = json.loads(
            (OUTPUT_ROOT / "final/run_summary.json").read_text(encoding="utf-8")
        )
        status(
            "article_and_bundle", "success", finished_at=now(),
            bundle=final["bundle"]["path"],
            bundle_sha256=final["bundle"]["sha256"],
        )
        status("pipeline", "success", finished_at=now(), test_opened=True)
    except Exception as error:
        status(
            "pipeline",
            "failed",
            finished_at=now(),
            error=f"{type(error).__name__}: {error}",
            test_opened=TEST_MARKER.is_file(),
        )
        raise


if __name__ == "__main__":
    main()
