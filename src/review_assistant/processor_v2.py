from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from .clips import EventClipWriter
from .config import assert_safe_input, resolve_path, sha256_file
from .database import ReviewDatabase
from .event_aggregator_v2 import EvidenceEventAggregator
from .processor import ProgressCallback, ReviewProcessor, merge_review_candidates


class FailSafeReviewProcessor(ReviewProcessor):
    """Review processor amendment with fail-open review-queue semantics."""

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
            aggregator = EvidenceEventAggregator(
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
                    try:
                        candidates = detector.candidates(frame)
                        for ordinal, candidate in enumerate(candidates):
                            candidate.setdefault(
                                "candidate_id",
                                f"D-{frame_number:06d}-{ordinal:04d}",
                            )
                            candidate.setdefault("processing_status", "NORMAL")
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
                            for fallback in getattr(
                                temporal, "last_fallbacks", []
                            ):
                                database.audit(
                                    "system",
                                    "VERIFIER_FALLBACK",
                                    "candidate",
                                    str(fallback.get("candidate_id", "")),
                                    {
                                        "frame_number": frame_number,
                                        "track_id": fallback.get("track_id"),
                                        "error_type": fallback.get("error_type"),
                                        "queue": "GENERAL_REVIEW",
                                    },
                                )
                        if mode == "conservative_review":
                            review_rows = [
                                {**row, "review_source": "BASELINE"}
                                for row in baseline
                            ]
                        elif mode == "high_recall_review":
                            review_rows = [
                                {**row, "review_source": "TEMPORAL_ONLY"}
                                for row in temporal_rows
                            ]
                        else:
                            review_rows = merge_review_candidates(
                                baseline, temporal_rows
                            )
                    except Exception as error:
                        technical_id = f"F-{frame_number:06d}-0000"
                        review_rows = [
                            {
                                "candidate_id": technical_id,
                                "box": [
                                    0.0,
                                    0.0,
                                    float(metadata["width"]),
                                    float(metadata["height"]),
                                ],
                                "confidence": 0.0,
                                "raw_confidence": 0.0,
                                "track_id": None,
                                "source": "processing_fallback",
                                "review_source": "TEMPORAL_ONLY",
                                "confirmed": True,
                                "interpolated": False,
                                "processing_status": "PROCESSING_FALLBACK",
                                "error_type": type(error).__name__,
                            }
                        ]
                        database.audit(
                            "system",
                            "PROCESSING_FALLBACK",
                            "candidate",
                            technical_id,
                            {
                                "frame_number": frame_number,
                                "error_type": type(error).__name__,
                                "queue": "TECHNICAL_REVIEW",
                                "raw_observation_preserved": True,
                            },
                        )
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
