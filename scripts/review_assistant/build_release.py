from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.review_assistant.database import ReviewDatabase
from src.review_assistant.models import EventDetection, ReviewEvent
from src.review_assistant.reporting import generate_report


OUTPUT = PROJECT / "outputs/review_assistant_release_v1"
BUNDLE = OUTPUT / "railway_person_review_assistant_v1_public.zip"
FIXED_TIME = (2026, 7, 26, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_report() -> None:
    demo = OUTPUT / "synthetic_example"
    if demo.exists():
        shutil.rmtree(demo)
    demo.mkdir(parents=True)
    database_path = demo / "synthetic.sqlite"
    with ReviewDatabase(database_path) as database:
        video_id, _ = database.register_video(
            path=Path("synthetic_example.mp4"),
            sha256="1" * 64,
            camera_id="demo-camera",
            fps=25,
            frame_count=9000,
            width=1280,
            height=720,
        )
        run_id, _ = database.create_run(
            video_id=video_id,
            mode="combined_queue",
            total_frames=9000,
            config_sha256="2" * 64,
            resume_policy="restart_incomplete_run",
            run_id_override="R-SYNTHETIC-DEMO",
        )
        events = []
        specifications = [
            ("BASELINE", [150, 130, 250, 500], 0.82, "HUMAN"),
            ("TEMPORAL_ONLY", [610, 190, 655, 390], 0.18, "UNCERTAIN"),
            ("BOTH", [910, 170, 1010, 540], 0.73, "FALSE_POSITIVE"),
        ]
        for index, (source, box, confidence, status) in enumerate(specifications, 1):
            event = ReviewEvent(
                event_id=f"E{index:06d}",
                video_id=video_id,
                camera_id="demo-camera",
                start_time=20.0 * index,
                end_time=20.0 * index + 4.0,
                state="CLOSED",
            )
            event.add(
                EventDetection(
                    frame_number=500 * index,
                    timestamp=20.0 * index,
                    box=box,
                    confidence=confidence,
                    source=source,
                    track_id=index,
                )
            )
            events.append(event)
        database.save_events(run_id, events)
        media_root = demo / "ui_media"
        for index, event in enumerate(events, 1):
            event_root = media_root / event.event_id
            event_root.mkdir(parents=True, exist_ok=True)
            clip = event_root / "clip.mp4"
            thumbnail = event_root / "thumbnail.png"
            writer = cv2.VideoWriter(
                str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 360)
            )
            for frame_number in range(48):
                frame = np.full((360, 640, 3), (27, 43, 55), dtype=np.uint8)
                x = 90 + index * 120 + (frame_number % 12)
                color = (
                    (60, 200, 100)
                    if event.source_label == "BASELINE"
                    else (0, 155, 255)
                )
                cv2.rectangle(frame, (x, 95), (x + 55, 285), color, 3)
                cv2.putText(
                    frame,
                    f"SYNTHETIC EVENT {event.event_id}",
                    (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    event.source_label,
                    (24, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                if frame_number == 20:
                    cv2.imwrite(str(thumbnail), frame)
            writer.release()
            database.set_event_media(run_id, event.event_id, clip, thumbnail)
        database.update_progress(run_id, 9000, 247)
        database.complete_run(
            run_id, processing_seconds=722, track_count=11, event_count=3
        )
        database.add_review_seconds(run_id, 94)
        for event, (_, _, _, status) in zip(events, specifications):
            database.review_event(
                run_id,
                event.event_id,
                operator="demo-operator",
                new_status=status,
                comment="Synthetic release example",
            )
    generate_report(database_path, run_id, demo)
    (OUTPUT / "SYNTHETIC_RUN_ID.txt").write_text(run_id + "\n", encoding="utf-8")


def public_entries() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []

    def add(path: Path, target: str | None = None) -> None:
        if not path.is_file():
            raise RuntimeError(f"Missing release input: {path}")
        entries.append((path, target or path.relative_to(PROJECT).as_posix()))

    for path in (
        PROJECT / "README.md",
        PROJECT / "requirements.txt",
        PROJECT / "configs/review_assistant_v1.yaml",
        PROJECT / "protocol/review_assistant_v1/PRODUCT_LOCK.json",
        PROJECT / "docker/Dockerfile.review-assistant",
        PROJECT / "docker/compose.review-assistant.yaml",
        PROJECT / "tests/test_review_assistant_v1.py",
    ):
        add(path)
    for directory in (
        PROJECT / "src/review_assistant",
        PROJECT / "src/final_demo",
        PROJECT / "src/crop_verifier_v1",
        PROJECT / "src/temporal_safety",
        PROJECT / "src/temporal",
        PROJECT / "scripts/review_assistant",
        PROJECT / "docs/review_assistant",
    ):
        for path in sorted(directory.glob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".py",
                ".sh",
                ".md",
            }:
                add(path)
    run_id = (OUTPUT / "SYNTHETIC_RUN_ID.txt").read_text(encoding="utf-8").strip()
    report_root = OUTPUT / "synthetic_example" / run_id
    for path in sorted(report_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".html",
            ".pdf",
            ".csv",
            ".json",
            ".yaml",
        }:
            add(path, f"example_report/{path.relative_to(report_root).as_posix()}")
    screenshots = OUTPUT / "screenshots"
    for path in sorted(screenshots.glob("*.png")):
        add(path, f"screenshots/{path.name}")
    benchmark = OUTPUT / "benchmark"
    for path in sorted(benchmark.glob("LONG_VIDEO_BENCHMARK.*")):
        add(path, f"benchmark/{path.name}")
    return entries


def write_zip(entries: Iterable[tuple[Path, str]]) -> None:
    payloads = [(target, source.read_bytes()) for source, target in entries]
    manifest = "\n".join(
        f"{hashlib.sha256(payload).hexdigest()}  {target}"
        for target, payload in payloads
    ) + "\n"
    payloads.append(("MANIFEST.sha256", manifest.encode()))
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
        for target, payload in payloads:
            info = zipfile.ZipInfo(
                f"railway_person_review_assistant_v1/{target}", FIXED_TIME
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)


def validate() -> dict:
    prohibited_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".sqlite",
        ".db",
        ".jpg",
        ".jpeg",
        ".npy",
        ".npz",
        ".pkl",
    }
    secret = re.compile(
        r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    failures: list[str] = []
    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix in prohibited_suffixes:
                failures.append(f"prohibited file: {name}")
            if any(
                token in name.lower()
                for token in (".codex", "sealed_test", "crowdhuman", "raileye3d")
            ):
                failures.append(f"restricted path: {name}")
            if suffix in {
                ".py",
                ".sh",
                ".md",
                ".json",
                ".csv",
                ".yaml",
                ".yml",
                ".html",
            }:
                text = archive.read(name).decode("utf-8", errors="replace")
                if ("/" + "home/") in text:
                    failures.append(f"absolute local path: {name}")
                if secret.search(text):
                    failures.append(f"secret-like content: {name}")
        manifests = [name for name in names if name.endswith("MANIFEST.sha256")]
    return {
        "status": "PASS" if not failures and len(manifests) == 1 else "FAIL",
        "files": len(names),
        "failures": failures,
        "archive_sha256": sha256(BUNDLE),
        "archive_bytes": BUNDLE.stat().st_size,
        "models_included": False,
        "videos_included": False,
        "sqlite_included": False,
        "railway_test_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-example", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_example:
        synthetic_report()
    screenshots = list((OUTPUT / "screenshots").glob("*.png"))
    if not screenshots:
        raise SystemExit("At least one UI screenshot is required before release.")
    write_zip(public_entries())
    audit = validate()
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (OUTPUT / "PRODUCT_RELEASE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / f"{BUNDLE.name}.sha256").write_text(
        f"{audit['archive_sha256']}  {BUNDLE.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
