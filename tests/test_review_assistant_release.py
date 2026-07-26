from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/review_assistant_release_v1"
BUNDLE = OUTPUT / "railway_person_review_assistant_v1_public.zip"


def test_scientific_closure_unchanged() -> None:
    current = (
        ROOT / "protocol/project_closure_v1/PROJECT_CLOSURE_LOCK.json"
    ).read_bytes()
    tagged = subprocess.check_output(
        [
            "git",
            "show",
            "v1.0-final-project-closure:protocol/project_closure_v1/"
            "PROJECT_CLOSURE_LOCK.json",
        ],
        cwd=ROOT,
    )
    assert current == tagged


def test_product_lock_disables_autonomy() -> None:
    lock = json.loads(
        (ROOT / "protocol/review_assistant_v1/PRODUCT_LOCK.json").read_text()
    )
    assert lock["product_status"] == "OPERATOR_ASSISTANT_MVP"
    assert lock["autonomous_alarming"] == "DISABLED"
    assert lock["safety_actuation"] == "DISABLED"
    assert lock["human_confirmation_required"] is True
    assert lock["railway_test_status"] == "SEALED"
    assert lock["railway_test_access_count"] == 0


def test_installer_never_downloads_models() -> None:
    installer = (ROOT / "scripts/review_assistant/install.sh").read_text()
    assert "curl " not in installer
    assert "wget " not in installer
    assert "http://" not in installer
    assert "https://" not in installer


def test_docker_package_has_no_model_copy() -> None:
    dockerfile = (ROOT / "docker/Dockerfile.review-assistant").read_text()
    assert "COPY models" not in dockerfile
    assert "COPY outputs" not in dockerfile
    compose = yaml.safe_load(
        (ROOT / "docker/compose.review-assistant.yaml").read_text()
    )
    volumes = compose["services"]["review-assistant"]["volumes"]
    assert any(str(value).endswith(":/models:ro") for value in volumes)


def test_long_benchmark_completed() -> None:
    result = json.loads(
        (OUTPUT / "benchmark/LONG_VIDEO_BENCHMARK.json").read_text()
    )
    assert result["status"] == "PASS"
    assert result["measured_frames"] >= 1000
    assert result["warmup_frames"] >= 1
    assert result["real_time_claim"] is False
    assert result["test_status"] == "SEALED"
    assert result["test_access_count"] == 0


def test_ui_screenshots_are_present() -> None:
    screenshots = sorted((OUTPUT / "screenshots").glob("*.png"))
    assert len(screenshots) >= 3
    assert all(path.stat().st_size > 10_000 for path in screenshots)


def test_example_report_contains_no_real_media() -> None:
    run_id = (OUTPUT / "SYNTHETIC_RUN_ID.txt").read_text().strip()
    report = OUTPUT / "synthetic_example" / run_id
    assert (report / "REVIEW_REPORT.html").is_file()
    assert (report / "REVIEW_REPORT.pdf").is_file()
    assert not list(report.rglob("*.jpg"))
    assert not list(report.rglob("*.mp4"))
    metrics = json.loads((report / "report_metrics.json").read_text())
    assert metrics["human_confirmation_required"] is True
    assert metrics["autonomous_alarm_output"] is False


def test_public_bundle_safety_audit() -> None:
    audit = json.loads((OUTPUT / "PRODUCT_RELEASE_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["models_included"] is False
    assert audit["videos_included"] is False
    assert audit["sqlite_included"] is False
    assert audit["railway_test_included"] is False


def test_public_bundle_manifest_and_contents() -> None:
    prohibited = {
        ".pt",
        ".pth",
        ".ckpt",
        ".mp4",
        ".avi",
        ".mov",
        ".sqlite",
        ".db",
        ".jpg",
        ".jpeg",
        ".pkl",
    }
    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        assert len([name for name in names if name.endswith("MANIFEST.sha256")]) == 1
        assert not any(Path(name).suffix.lower() in prohibited for name in names)
        assert any(name.endswith("configs/review_assistant_v1.yaml") for name in names)
        assert any(name.endswith("screenshots/02_event_queue.png") for name in names)
        assert any(name.endswith("example_report/REVIEW_REPORT.pdf") for name in names)
