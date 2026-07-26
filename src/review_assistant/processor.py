from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import yaml

from src.final_demo.frame_pipeline import FrozenPersonDetector
from src.final_demo.temporal_pipeline import TemporalResearchPipeline

from .clips import EventClipWriter
from .config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    assert_safe_input,
    load_config,
    resolve_path,
    sha256_file,
    verify_model_file,
)
from .database import ReviewDatabase
from .event_aggregator import EventAggregator, box_iou


ProgressCallback = Callable[[dict[str, Any]], None]


def _config_sha(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(
        yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def merge_review_candidates(
    baseline: list[dict[str, Any]],
    temporal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    matched_temporal: set[int] = set()
    for base in baseline:
        row = dict(base)
        row["review_source"] = "BASELINE"
        for index, candidate in enumerate(temporal):
            if box_iou(row["box"], candidate["box"]) < 0.5:
                continue
            row["review_source"] = "BOTH"
            row["track_id"] = candidate.get("track_id")
            row["confirmed"] = bool(
                row.get("confirmed", False) or candidate.get("confirmed", False)
            )
            row["interpolated"] = False
            matched_temporal.add(index)
            break
        output.append(row)
    for index, candidate in enumerate(temporal):
        if index in matched_temporal:
            continue
        if float(candidate.get("raw_confidence", 0.0)) >= 0.07 and any(
            box_iou(candidate["box"], base["box"]) >= 0.5 for base in baseline
        ):
            continue
        row = dict(candidate)
        row["review_source"] = "TEMPORAL_ONLY"
        output.append(row)
    return output


class ReviewProcessor:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        *,
        checkpoint: str | Path | None = None,
        verifier: str | Path | None = None,
        encoder: str | Path | None = None,
        database: str | Path | None = None,
        run_root: str | Path | None = None,
        output_root: str | Path | None = None,
        device: str | None = None,
        detector: Any | None = None,
        temporal: Any | None = None,
    ) -> None:
        overrides = {
            "detector.checkpoint": checkpoint,
            "verifier.model": verifier,
            "verifier.encoder_checkpoint": encoder,
            "storage.database": database,
            "storage.run_root": run_root,
            "storage.output_root": output_root,
        }
        self.config = load_config(config_path, overrides=overrides)
        self.device = device or str(self.config["processing"]["device"])
        self.detector = detector
        self.temporal = temporal
        self.config_sha256 = _config_sha(self.config)

    def _models(self, mode: str) -> tuple[Any, Any | None]:
        if self.detector is None:
            checkpoint = verify_model_file(
                self.config["detector"]["checkpoint"],
                self.config["detector"]["expected_sha256"],
            )
            self.detector = FrozenPersonDetector(
                self.config["detector"], checkpoint, self.device
            )
        if mode in {"high_recall_review", "combined_queue"} and self.temporal is None:
            verify_model_file(
                self.config["verifier"]["model"],
                self.config["verifier"]["model_expected_sha256"],
            )
            verify_model_file(
                self.config["verifier"]["encoder_checkpoint"],
                self.config["verifier"]["encoder_expected_sha256"],
            )
            self.temporal = TemporalResearchPipeline(
                self.config["tracker"],
                self.config["verifier"],
                float(self.config["detector"]["standard_threshold"]),
                self.device,
            )
        return self.detector, self.temporal

    def _metadata(self, video_path: Path) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        metadata = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 1.0),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        capture.release()
        if metadata["frame_count"] <= 0 or metadata["width"] <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")
        return metadata

    def process(
        self,
        video_path: str | Path,
        *,
        mode: str = "combined_queue",
        camera_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if mode not in {
            "conservative_review",
            "high_recall_review",
            "combined_queue",
        }:
            raise ValueError(f"Unsupported review mode: {mode}")
        path = Path(video_path).resolve()
        assert_safe_input(path, self.config)
        metadata = self._metadata(path)
        video_sha = sha256_file(path)
        selected_camera = camera_id or str(self.config["input"]["camera_id"])
        db_path = resolve_path(self.config["storage"]["database"])
        run_root = resolve_path(self.config["storage"]["run_root"])
        output_root = resolve_path(self.config["storage"]["output_root"])
        started = time.perf_counter()
        detector, temporal = self._models(mode)
        with ReviewDatabase(db_path) as database:
            video_id, duplicate = database.register_video(
                path=path,
                sha256=video_sha,
                camera_id=selected_camera,
                **metadata,
            )
            run_id, resumed = database.create_run(
                video_id=video_id,
                mode=mode,
                total_frames=metadata["frame_count"],
                config_sha256=self.config_sha256,
                resume_policy=str(self.config["processing"]["resume_policy"]),
            )
            aggregator = EventAggregator(
                self.config["events"],
                video_id=video_id,
                camera_id=selected_camera,
                frame_width=metadata["width"],
                frame_height=metadata["height"],
            )
            capture = cv2.VideoCapture(str(path))
            raw_count = 0
            track_ids: set[int] = set()
            frame_number = -1
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_number += 1
                    timestamp = frame_number / metadata["fps"]
                    candidates = detector.candidates(frame)
                    raw_count += len(candidates)
                    baseline = detector.standard(candidates)
                    temporal_rows: list[dict[str, Any]] = []
                    if temporal is not None:
                        temporal_rows, _ = temporal.update(frame, candidates)
                        track_ids.update(
                            int(row["track_id"])
                            for row in temporal_rows
                            if row.get("track_id") is not None
                        )
                    if mode == "conservative_review":
                        review_rows = [
                            {**row, "review_source": "BASELINE"} for row in baseline
                        ]
                    elif mode == "high_recall_review":
                        review_rows = [
                            {**row, "review_source": "TEMPORAL_ONLY"}
                            for row in temporal_rows
                        ]
                    else:
                        review_rows = merge_review_candidates(baseline, temporal_rows)
                    aggregator.observe(frame_number, timestamp, review_rows)
                    interval = int(
                        self.config["processing"]["progress_commit_interval_frames"]
                    )
                    if frame_number % max(interval, 1) == 0:
                        database.update_progress(run_id, frame_number + 1, raw_count)
                    if progress is not None:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        processed = frame_number + 1
                        progress(
                            {
                                "run_id": run_id,
                                "processed_frames": processed,
                                "total_frames": metadata["frame_count"],
                                "progress": processed / metadata["frame_count"],
                                "fps": processed / elapsed,
                                "eta_seconds": (
                                    metadata["frame_count"] - processed
                                )
                                / max(processed / elapsed, 1e-9),
                                "events": len(aggregator.accepted_events()),
                                "error": None,
                            }
                        )
                capture.release()
                events = aggregator.finalize()
                database.update_progress(run_id, frame_number + 1, raw_count)
                database.save_events(run_id, events)
                event_root = run_root / run_id / "events"
                clip_writer = EventClipWriter(
                    self.config["events"]["pre_roll_seconds"],
                    self.config["events"]["post_roll_seconds"],
                )
                if self.config["storage"]["retain_event_clips"]:
                    for event in events:
                        media_root = event_root / event.event_id
                        clip, thumbnail = clip_writer.write(path, event, media_root)
                        database.set_event_media(
                            run_id, event.event_id, clip, thumbnail
                        )
                        database.apply_suppression(
                            run_id,
                            event.event_id,
                            self.config["suppression"],
                            frame_width=metadata["width"],
                            frame_height=metadata["height"],
                        )
                elapsed = time.perf_counter() - started
                database.complete_run(
                    run_id,
                    processing_seconds=elapsed,
                    track_count=len(track_ids),
                    event_count=len(events),
                )
                summary = {
                    "product_id": self.config["product_id"],
                    "product_status": "OPERATOR_ASSISTANT_MVP",
                    "run_id": run_id,
                    "video_id": video_id,
                    "mode": mode,
                    "duplicate_video": duplicate,
                    "resumed_by_restart": resumed,
                    "frames": metadata["frame_count"],
                    "video_duration_seconds": metadata["frame_count"]
                    / metadata["fps"],
                    "processing_seconds": elapsed,
                    "end_to_end_fps": metadata["frame_count"] / max(elapsed, 1e-9),
                    "raw_detections": raw_count,
                    "tracks": len(track_ids),
                    "unique_events": len(events),
                    "autonomous_alarm_output": False,
                    "safety_actuation": False,
                    "human_confirmation_required": True,
                    "test_status": "SEALED",
                    "test_access_count": 0,
                }
                self._write_run_outputs(
                    output_root / run_id,
                    summary,
                    path,
                    video_sha,
                )
                return summary
            except Exception as error:
                capture.release()
                database.fail_run(run_id, str(error))
                if progress is not None:
                    progress({"run_id": run_id, "error": str(error)})
                raise

    def _write_run_outputs(
        self,
        root: Path,
        runtime: dict[str, Any],
        input_path: Path,
        input_sha256: str,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved = {
            key: value for key, value in self.config.items() if not key.startswith("_")
        }
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        provenance = {
            "git_commit": (
                commit.stdout.strip()
                if commit.returncode == 0
                else "UNAVAILABLE_NOT_GIT"
            ),
            "input_file_sha256": input_sha256,
            "input_name": input_path.name,
            "detector_checkpoint_sha256": getattr(
                self.detector, "checkpoint_sha256", "INJECTED_TEST_DETECTOR"
            ),
            "verifier_model_sha256": (
                sha256_file(resolve_path(self.config["verifier"]["model"]))
                if self.temporal is not None
                and resolve_path(self.config["verifier"]["model"]).is_file()
                else "NOT_USED_OR_INJECTED"
            ),
            "config_sha256": self.config_sha256,
            "product_status": "OPERATOR_ASSISTANT_MVP",
            "autonomous_alarming": "DISABLED",
            "safety_actuation": "DISABLED",
            "human_confirmation_required": True,
            "test_status": "SEALED",
            "test_access_count": 0,
        }
        (root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
