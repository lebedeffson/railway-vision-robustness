from __future__ import annotations

import csv
import json
import re
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.final_demo.app import FinalDemoApp
from src.final_demo.output_writer import annotate
from src.final_demo.safety_banner import WARNING_LINES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/project_closure_v1"
LOCK = ROOT / "protocol/project_closure_v1/PROJECT_CLOSURE_LOCK.json"
PUBLIC = OUTPUT / "bundles/railway_vision_final_closure_public.zip"


def payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_project_closure_lock() -> None:
    lock = payload(LOCK)
    assert lock["protocol_id"] == "railway-vision-final-closure-v1"
    assert lock["project_status"] == "COMPLETED_RESEARCH"
    assert lock["practical_status"] == "RESEARCH_DEMONSTRATOR_ONLY"
    assert lock["deployment_gate"] == "FAIL"
    assert lock["further_tuning_allowed"] is False


def test_all_experiments_registered() -> None:
    rows = list(csv.DictReader((OUTPUT / "tables/EXPERIMENT_REGISTRY.csv").open()))
    identifiers = {row["experiment_id"] for row in rows}
    assert {
        "canonical-v8b-person-failure-risk-v1",
        "railway-person-temporal-safety-v1",
        "railway-person-temporal-verifier-v1",
        "railway-person-crop-verifier-v1",
        "railway-person-new-scenes-v1",
        "railway-vision-final-closure-v1",
    } <= identifiers


def test_all_release_tags_registered() -> None:
    rows = list(csv.DictReader((OUTPUT / "tables/RELEASE_REGISTRY.csv").open()))
    tags = {row["release_tag"] for row in rows}
    assert "v0.8b-negative-result" in tags
    assert "v1.0-final-project-closure" in tags


def test_final_metrics_match_source_artifacts() -> None:
    audit = payload(OUTPUT / "FINAL_CLOSURE_AUDIT.json")
    assert audit["status"] == "PASS"
    assert all(audit["source_checks"].values())


def test_v8b_metrics_unchanged() -> None:
    source = payload(ROOT / "outputs/person_v8b/final/FINAL_METRICS.json")
    assert source["U2_scene_macro_MAE"] == 1.595488930156677
    assert source["U3_scene_macro_MAE"] == 1.7679771024802038
    assert source["scene_wins"] == 5


def test_temporal_metrics_unchanged() -> None:
    source = payload(
        ROOT / "outputs/temporal_safety_v1/FINAL_DEVELOPMENT_SUMMARY.json"
    )
    assert source["recall_signal"]["bytetrack"] == 0.13771609596477535
    assert source["recall_signal"]["ocsort"] == 0.13276800629502783


def test_track_verifier_metrics_unchanged() -> None:
    source = payload(ROOT / "outputs/temporal_verifier_v1/FINAL_SUMMARY.json")
    assert source["final_gate"]["deltas"]["recall"] == 0.12928241412597363
    assert source["status"] == "DEVELOPMENT_FAIL"


def test_crop_verifier_metrics_unchanged() -> None:
    source = payload(ROOT / "outputs/crop_verifier_v1/FINAL_SUMMARY.json")
    assert source["two_fold_triage"]["deltas"]["recall"] == 0.12673957117272844
    assert source["status"] == "CLOSED_NO_PRACTICAL_GATE"


def test_acquisition_closed() -> None:
    source = payload(
        ROOT / "acquisition/new_scenes_v1/ACQUISITION_RUNTIME_STATUS.json"
    )
    assert source["status"] == "CLOSED_DATA_UNAVAILABLE"
    assert source["accepted_scenes"] == 0


def test_training_disabled() -> None:
    source = payload(
        ROOT / "acquisition/new_scenes_v1/ACQUISITION_RUNTIME_STATUS.json"
    )
    assert source["training_allowed"] is False
    assert source["project_state"]["automatic_training_disabled"] is True


def test_test_sealed() -> None:
    assert payload(LOCK)["test_status"] == "SEALED"
    assert not (ROOT / "outputs/person_v3/test/TEST_OPENED.json").exists()


def test_test_access_count_zero() -> None:
    assert payload(LOCK)["test_access_count"] == 0


def _running_project_units() -> str:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--state=running",
            "--type=service",
            "--type=timer",
            "--no-pager",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return "\n".join(
        line
        for line in result.stdout.splitlines()
        if re.search(r"tnorm|railway|person|temporal", line, re.I)
    )


def test_no_active_download_service() -> None:
    assert "download" not in _running_project_units().lower()


def test_no_active_training_service() -> None:
    assert not _running_project_units()


class FakeDetector:
    checkpoint_sha256 = "fake-detector-sha256"

    def candidates(self, frame: np.ndarray) -> list[dict]:
        return [
            {
                "box": [5.0, 5.0, 25.0, 35.0],
                "confidence": 0.8,
                "raw_confidence": 0.8,
                "track_id": None,
                "source": "detector",
                "interpolated": False,
                "confirmed": True,
            }
        ]

    def standard(self, candidates: list[dict]) -> list[dict]:
        return candidates


class FakeTemporal:
    def update(self, frame: np.ndarray, candidates: list[dict]):
        return [
            {
                **candidates[0],
                "track_id": 1,
                "source": "verifier",
                "interpolated": True,
            }
        ], []


def test_demo_frame_mode(tmp_path: Path) -> None:
    app = FinalDemoApp(detector=FakeDetector(), temporal=FakeTemporal())
    result = app.process_frames(
        [np.zeros((48, 64, 3), dtype=np.uint8)],
        tmp_path / "frame",
        "frame_baseline",
    )
    assert result["frames"] == 1
    assert (tmp_path / "frame/annotated_video.mp4").is_file()


def test_demo_temporal_research_warning() -> None:
    image = np.zeros((120, 300, 3), dtype=np.uint8)
    rendered = annotate(image, [], research=True)
    assert tuple(rendered[5, 5]) != (0, 0, 0)
    config = yaml.safe_load((ROOT / "configs/final_demo.yaml").read_text())
    assert tuple(config["output"]["warning"]) == WARNING_LINES


def test_interpolated_boxes_visually_marked() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    rendered = annotate(
        image,
        [
            {
                "box": [10, 10, 60, 60],
                "confidence": 0.5,
                "source": "temporal_interpolation",
                "interpolated": True,
            }
        ],
        research=False,
    )
    assert np.count_nonzero(rendered[10, 10:60]) > 0
    assert np.count_nonzero(rendered[60, 19:21]) == 0


def test_demo_provenance_complete(tmp_path: Path) -> None:
    app = FinalDemoApp(detector=FakeDetector(), temporal=FakeTemporal())
    root = tmp_path / "temporal"
    app.process_frames(
        [np.zeros((48, 64, 3), dtype=np.uint8)],
        root,
        "temporal_research",
    )
    provenance = payload(root / "provenance.json")
    assert {
        "git_commit",
        "detector_checkpoint_sha256",
        "tracker_config_sha256",
        "verifier_config_sha256",
        "input_file_sha256",
        "mode",
        "start_time",
        "completion_status",
    } <= provenance.keys()


def test_runtime_metrics_complete() -> None:
    frame = pd_read(OUTPUT / "tables/RUNTIME_BENCHMARK.csv")
    assert set(frame["mode"]) >= {"frame_baseline", "temporal_research"}
    assert (frame["frames"].astype(int) > 0).all()
    assert (frame["p95_latency_ms"].astype(float) >= 0).all()


def pd_read(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _public_names() -> list[str]:
    with zipfile.ZipFile(PUBLIC) as handle:
        return handle.namelist()


def test_public_zip_no_images() -> None:
    names = _public_names()
    assert not any(
        Path(name).suffix.lower() in {".jpg", ".jpeg", ".bmp"}
        for name in names
    )
    assert all(
        "/figures/" in name
        for name in names
        if Path(name).suffix.lower() == ".png"
    )


def test_public_zip_no_videos() -> None:
    assert not any(Path(name).suffix.lower() in {".mp4", ".avi", ".mov"} for name in _public_names())


def test_public_zip_no_checkpoints() -> None:
    assert not any(Path(name).suffix.lower() in {".pt", ".pth", ".ckpt"} for name in _public_names())


def test_public_zip_no_restricted_annotations() -> None:
    assert not any("raileye3d/anno" in name.lower() or "crowdhuman" in name.lower() for name in _public_names())


def test_public_zip_no_local_paths() -> None:
    with zipfile.ZipFile(PUBLIC) as handle:
        for name in handle.namelist():
            if Path(name).suffix.lower() in {".md", ".txt", ".json", ".csv", ".yaml", ".py", ".cff"}:
                assert "/" + "home/" not in handle.read(name).decode(
                    "utf-8", errors="replace"
                )


def test_public_zip_no_secrets() -> None:
    validation = payload(OUTPUT / "bundles/PUBLIC_VALIDATION.json")
    assert validation["status"] == "PASS"
    assert validation["failures"] == []


def test_public_zip_manifest() -> None:
    with zipfile.ZipFile(PUBLIC) as handle:
        manifests = [name for name in handle.namelist() if name.endswith("MANIFEST.sha256")]
        assert len(manifests) == 1
        root = manifests[0].rsplit("/", 1)[0]
        for line in handle.read(manifests[0]).decode().splitlines():
            expected, relative = line.split("  ", 1)
            assert hashlib_sha(handle.read(f"{root}/{relative}")) == expected


def hashlib_sha(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_release_sha256() -> None:
    expected, name = (
        OUTPUT / "bundles/railway_vision_final_closure_public.zip.sha256"
    ).read_text().split()
    assert name == PUBLIC.name
    assert hashlib_sha(PUBLIC.read_bytes()) == expected
