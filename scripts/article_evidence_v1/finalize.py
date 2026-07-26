from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from scripts.article_evidence_v1.common import (
    OUTPUT,
    PROJECT,
    atomic_text,
    check_no_test_markers,
    sha256,
)


EXCLUDED = {
    "MANIFEST.sha256",
    "article_evidence_v1_public.zip",
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".svg", ".yaml", ".yml"}


def _load(relative: str) -> dict:
    return json.loads((OUTPUT / relative).read_text(encoding="utf-8"))


def technical_summary() -> str:
    provenance = _load("INPUT_PROVENANCE.json")
    threshold = _load("threshold_baseline/THRESHOLD_AUDIT.json")
    verifier = _load("track_verifier/TRACK_VERIFIER_AUDIT.json")
    false = _load("false_tracks/FALSE_TRACK_AUDIT.json")
    event = _load("operator_assistant/EVENT_AGGREGATION_SUMMARY.json")
    event_parameters = _load("operator_assistant/EVENT_PARAMETERS_ACTUAL.json")
    runtime = _load("runtime/RUNTIME_FINAL.json")
    environment = _load("runtime/RUNTIME_ENVIRONMENT.json")
    tnorm = _load("tnorm/TNORM_AUDIT.json")
    test_results = (
        (OUTPUT / "TEST_RESULTS.txt").read_text(encoding="utf-8").strip()
        if (OUTPUT / "TEST_RESULTS.txt").exists()
        else "PENDING"
    )
    files = sorted(
        path.relative_to(OUTPUT).as_posix()
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and not path.name.startswith(".")
    )
    return f"""# Article evidence computation v1 — technical summary

## 1. Commit and frozen input hashes

- git commit: `{provenance['git_commit']}`
- detector predictions: `{provenance['detector_predictions_sha256']}`
- tracker outputs: `{provenance['tracker_outputs_sha256']}`
- track-verifier sweep: `{provenance['track_verifier_outputs_sha256']}`
- false-track audit: `{provenance['false_track_audit_sha256']}`
- event config: `{provenance['event_config_sha256']}`
- development manifest: `{provenance['development_manifest_sha256']}`

## 2. Closed-test boundary

- test status: `SEALED`
- test access count: `0`
- new training: `FORBIDDEN`
- new data: `FORBIDDEN`

## 3. Processed development material

- threshold comparison: `{threshold['development_scenes']}` grouped scenes
- T-norm redundancy analysis: `{tnorm['outer_folds']}` grouped-scene LOSO folds, `{tnorm['frames']}` saved feature rows

## 4. Threshold computations

- detector thresholds: `{len(threshold['threshold_grid'])}`
- methods: `{len(threshold['methods'])}`
- threshold-method rows: `{threshold['metric_rows']}`
- verifier thresholds: `{verifier['number_of_unique_thresholds']}`
- complete operational-gate passes in detector grid: `{threshold['full_gate_passes']}`
- complete operational-gate passes in verifier sweep: `{verifier['full_gate_passes']}`

## 5. False tracks

- classified rows: `{false['false_tracks']}`
- category sum: `{false['category_sum']}`
- duplicate track rows: `{false['duplicate_track_rows']}`
- full semantic subtyping: `{false['semantic_subcategory_status']}`

## 6. Event aggregation

- actual parameters: `{json.dumps(event_parameters, sort_keys=True)}`
- real short run: `{event['real_short']}`
- real long run: `{event['real_long']}`
- synthetic run: `{event['synthetic']}`

## 7. Runtime environment

- CPU: `{environment['cpu']}`
- GPU: `{environment['gpu']}`
- OS: `{environment['operating_system']}`
- Python: `{environment['python']}`
- PyTorch/CUDA: `{environment['pytorch']} / {environment['cuda']}`
- precision: `{environment['precision']}`
- warm-up / measured frames: `{runtime['warmup_frames']} / {runtime['measured_frames']}`

## 8. Runtime

- end-to-end FPS: `{runtime['end_to_end_fps']:.6f}`
- peak GPU memory MiB: `{runtime['peak_gpu_memory_mib']:.3f}`
- peak RAM MiB: `{runtime['peak_ram_mib']:.3f}`
- mean / p95 tracks per frame: `{runtime['mean_tracks_per_frame']:.3f} / {runtime['p95_tracks_per_frame']:.3f}`

## 9. Created files

{chr(10).join(f'- `{name}`' for name in files)}

## 10. Tests

```text
{test_results}
```

## 11. Blocked computations

- `BLOCKED_MISSING_ARTIFACT`: complete human-verified semantic labels separating signal, catenary support, pole/sign and train-part for all 288 false tracks were not stored. Existing deterministic categories and binary geometry flags were preserved; no visual class was invented.
- `BLOCKED_MISSING_ARTIFACT`: the prior 1,000-frame event benchmark stored total event count and FPS but not the per-event raw-track/detection decomposition. The available long-run values were retained without approximation.
- `BLOCKED_PUBLICATION_RIGHTS`: false-track source images were excluded; `FIG_07_FALSE_TRACK_EXAMPLES` was not generated.
"""


def _files_for_bundle() -> list[Path]:
    return sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and not any(part.startswith(".") for part in path.relative_to(OUTPUT).parts)
    )


def _validate_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8", errors="strict")
    forbidden = (
        "/home/",
        "lebedeffson",
        "TNormFilter_handoff/data/",
        "sealed_test_manifest",
    )
    hits = [token for token in forbidden if token.lower() in text.lower()]
    if hits:
        raise RuntimeError(f"Public output contains local/restricted tokens in {path}: {hits}")
    secret_patterns = [
        r"(?i)api[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_-]{12,}",
        r"(?i)bearer\s+[A-Za-z0-9._-]{12,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]
    for pattern in secret_patterns:
        if re.search(pattern, text):
            raise RuntimeError(f"Potential secret in {path}")


def main() -> None:
    check_no_test_markers()
    required = [
        "COMPUTATION_LOCK.json",
        "INPUT_PROVENANCE.json",
        "threshold_baseline/THRESHOLD_AUDIT.json",
        "track_verifier/TRACK_VERIFIER_AUDIT.json",
        "false_tracks/FALSE_TRACK_AUDIT.json",
        "operator_assistant/EVENT_AUDIT.json",
        "runtime/RUNTIME_AUDIT.json",
        "tnorm/TNORM_AUDIT.json",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Evidence outputs incomplete: {missing}")
    atomic_text(OUTPUT / "FINAL_TECHNICAL_SUMMARY.md", technical_summary())
    atomic_text(
        OUTPUT / "REPRODUCE.md",
        """# Reproduce article evidence v1

Use the repository environment and the frozen saved development artifacts:

```bash
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.lock_inputs
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.compute_saved
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.threshold_baseline
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.runtime_analysis
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.plot_figures
/home/lebedeffson/Code/venv/bin/python -m scripts.article_evidence_v1.finalize
```

The commands do not train models and abort if a test-open marker exists.
Local paths shown above are execution examples and are intentionally omitted
from the public archive copy of this file.
""".replace(
            "/home/lebedeffson/Code/venv/bin/python", "python"
        ),
    )
    files = _files_for_bundle()
    for path in files:
        _validate_text(path)
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in files
    ]
    atomic_text(OUTPUT / "MANIFEST.sha256", "\n".join(manifest_lines) + "\n")
    archive = OUTPUT / "article_evidence_v1_public.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(
                path,
                arcname=f"article_evidence_v1/{path.relative_to(OUTPUT).as_posix()}",
            )
        handle.write(
            OUTPUT / "MANIFEST.sha256",
            arcname="article_evidence_v1/MANIFEST.sha256",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "files": len(files),
                "archive": archive.name,
                "archive_sha256": sha256(archive),
                "test_status": "SEALED",
                "test_access_count": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
