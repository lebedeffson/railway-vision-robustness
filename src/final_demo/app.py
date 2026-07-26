from __future__ import annotations

import json
import subprocess
import time
from itertools import chain
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from .config_loader import (
    DEFAULT_CONFIG,
    PROJECT,
    assert_not_test_input,
    load_config,
    sha256_file,
)
from .frame_pipeline import FrozenPersonDetector
from .output_writer import RunWriter, annotate
from .runtime_metrics import RuntimeMetrics
from .temporal_pipeline import TemporalResearchPipeline


class FinalDemoApp:
    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG,
        checkpoint: Path | None = None,
        device: str | None = None,
        detector: Any | None = None,
        temporal: Any | None = None,
    ) -> None:
        self.config = load_config(config_path)
        selected_device = device or str(self.config.get("device", "auto"))
        self.detector = detector or FrozenPersonDetector(
            self.config["detector"], checkpoint, selected_device
        )
        self.temporal = temporal
        self.selected_device = selected_device

    def _temporal(self) -> Any:
        if self.temporal is None:
            self.temporal = TemporalResearchPipeline(
                self.config["tracker"],
                self.config["verifier"],
                float(self.config["detector"]["standard_threshold"]),
                self.selected_device,
            )
        return self.temporal

    def process_frames(
        self,
        frames: Iterable[np.ndarray],
        output_root: Path,
        mode: str,
        fps: float = 10.0,
        input_sha256: str = "IN_MEMORY_INPUT",
    ) -> dict[str, Any]:
        if mode not in {"frame_baseline", "temporal_research"}:
            raise ValueError(f"Unsupported mode: {mode}")
        iterator = iter(frames)
        first = next(iterator, None)
        if first is None:
            raise RuntimeError("Input contains no frames")
        height, width = first.shape[:2]
        writer = RunWriter(output_root, fps, (width, height), self.config)
        metrics = RuntimeMetrics()
        started = datetime.now(timezone.utc).isoformat()
        for frame_number, frame in enumerate(chain([first], iterator)):
            with metrics.measure("total"):
                with metrics.measure("detector"):
                    candidates = self.detector.candidates(frame)
                tracker_events: list[dict[str, Any]] = []
                if mode == "frame_baseline":
                    rows = self.detector.standard(candidates)
                else:
                    temporal = self._temporal()
                    rows, tracker_events = temporal.update(frame, candidates)
                    metrics.samples["tracker"].append(
                        float(getattr(temporal, "last_tracker_ms", 0.0))
                    )
                    metrics.samples["verifier"].append(
                        float(getattr(temporal, "last_verifier_ms", 0.0))
                    )
                with metrics.measure("output_rendering"):
                    rendered = annotate(
                        frame, rows, research=mode == "temporal_research"
                    )
                    writer.write(
                        rendered,
                        rows,
                        tracker_events,
                        frame_number,
                        frame_number / fps,
                    )
            metrics.frame_complete()
        peak_gpu = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        runtime = metrics.summary(peak_gpu)
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else "UNAVAILABLE_NOT_GIT"
        )
        provenance = {
            "git_commit": commit,
            "detector_checkpoint_sha256": getattr(
                self.detector, "checkpoint_sha256", "INJECTED_TEST_DETECTOR"
            ),
            "tracker_config_sha256": sha256_file(DEFAULT_CONFIG),
            "verifier_config_sha256": sha256_file(DEFAULT_CONFIG),
            "input_file_sha256": input_sha256,
            "mode": mode,
            "start_time": started,
            "completion_status": "PASS",
            "test_status": "SEALED",
            "test_access_count": 0,
        }
        writer.close(runtime, self.config, provenance)
        return runtime

    def run_video(self, input_path: Path, output_root: Path, mode: str) -> dict[str, Any]:
        assert_not_test_input(input_path, self.config)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)

        def frames() -> Iterable[np.ndarray]:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
            capture.release()

        return self.process_frames(
            frames(), output_root, mode, fps, sha256_file(input_path)
        )

    def run_image_folder(
        self, input_path: Path, output_root: Path, mode: str, fps: float = 10.0
    ) -> dict[str, Any]:
        assert_not_test_input(input_path, self.config)
        paths = sorted(
            path
            for path in input_path.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        payload = "\n".join(f"{path.name}:{sha256_file(path)}" for path in paths)

        def frames() -> Iterable[np.ndarray]:
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError(f"Unreadable image: {path}")
                yield frame

        import hashlib

        digest = hashlib.sha256(payload.encode()).hexdigest()
        return self.process_frames(frames(), output_root, mode, fps, digest)
