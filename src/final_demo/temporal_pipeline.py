from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.crop_verifier_v1.encoder import FrozenYoloTrackEncoder, context_crop
from src.crop_verifier_v1.model import FittedVerifier, predict_verifier
from src.temporal_safety.ocsort_adapter import OCSortAdapter

from .config_loader import resolve_local_path, sha256_file


class FrozenCombinedVerifier:
    def __init__(self, settings: dict[str, Any], device: str = "auto") -> None:
        model_path = resolve_local_path(settings["model"])
        encoder_path = resolve_local_path(settings["encoder_checkpoint"])
        if not model_path.is_file() or not encoder_path.is_file():
            raise FileNotFoundError("Frozen combined verifier artifacts are missing")
        if sha256_file(encoder_path) != settings["encoder_expected_sha256"]:
            raise RuntimeError("Frozen verifier encoder SHA-256 mismatch")
        with model_path.open("rb") as handle:
            self.model: FittedVerifier = pickle.load(handle)
        self.model_sha256 = sha256_file(model_path)
        self.encoder_sha256 = sha256_file(encoder_path)
        self.threshold = float(settings["threshold"])
        encoder_device = None if device == "auto" else device
        if encoder_device is not None and encoder_device.isdigit():
            encoder_device = f"cuda:{encoder_device}"
        self.encoder = FrozenYoloTrackEncoder(
            encoder_path,
            int(settings["backbone_last_layer_index"]),
            int(settings["input_size"]),
            float(settings["crop_context_multiplier"]),
            int(settings["padding_value"]),
            3,
            encoder_device,
        )
        self.crops: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)

    def observe(self, track_id: int, frame: np.ndarray, row: dict[str, Any]) -> None:
        if row.get("interpolated"):
            return
        crop = context_crop(
            frame,
            list(row["box"]),
            self.encoder.context_multiplier,
            self.encoder.input_size,
            self.encoder.padding_value,
        )
        self.crops[int(track_id)].append((float(row["raw_confidence"]), crop))
        self.crops[int(track_id)] = self.crops[int(track_id)][-12:]

    def _visual(self, track_id: int) -> np.ndarray:
        history = self.crops.get(int(track_id), [])
        if not history:
            return np.zeros(512, dtype=np.float64)
        first = history[min(1, len(history) - 1)][1]
        maximum = max(history, key=lambda item: item[0])[1]
        latest = history[-1][1]
        batch = np.stack([first, maximum, latest])
        tensor = torch.from_numpy(
            batch[:, :, :, ::-1].copy().transpose(0, 3, 1, 2)
        ).float().div_(255.0).to(self.encoder.device)
        return self.encoder._forward(tensor).mean(dim=0).numpy().astype(np.float64)

    def probability(self, track: Any) -> float:
        age = max(int(track.age), 1)
        track_features = np.asarray(
            [
                float(np.mean(track.confidence_history)),
                float(np.max(track.confidence_history)),
                float(age),
                float(track.hits / age),
                float(max(age - track.hits, 0) / age),
            ],
            dtype=np.float64,
        )
        matrix = np.concatenate([self._visual(track.track_id), track_features])[None, :]
        return float(predict_verifier(self.model, matrix)[0])


class TemporalResearchPipeline:
    def __init__(
        self,
        tracker_settings: dict[str, Any],
        verifier_settings: dict[str, Any],
        standard_threshold: float,
        device: str = "auto",
    ) -> None:
        self.tracker = OCSortAdapter(
            dict(tracker_settings["parameters"]),
            dict(tracker_settings["temporal_logic"]),
        )
        self.tracker.reset("demo")
        self.verifier = FrozenCombinedVerifier(verifier_settings, device)
        self.standard_threshold = float(standard_threshold)
        self.last_tracker_ms = 0.0
        self.last_verifier_ms = 0.0
        self.frame_index = -1
        self.last_trace_rows: list[dict[str, Any]] = []
        self.last_fallbacks: list[dict[str, Any]] = []

    def update(
        self, frame: np.ndarray, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.frame_index += 1
        self.last_trace_rows = []
        self.last_fallbacks = []
        height, width = frame.shape[:2]
        tracker_started = time.perf_counter()
        emitted, events = self.tracker.update(candidates, width, height)
        self.last_tracker_ms = (time.perf_counter() - tracker_started) * 1000.0
        verifier_started = time.perf_counter()
        states = {track.track_id: track for track in self.tracker.tracks}
        output: list[dict[str, Any]] = []
        for ordinal, row in enumerate(emitted):
            track_id = int(row["track_id"])
            if not row.get("candidate_id"):
                row["candidate_id"] = (
                    f"T-{self.frame_index:06d}-{track_id:06d}-{ordinal:04d}"
                )
            state = states.get(track_id)
            baseline = float(row.get("raw_confidence", 0.0)) >= self.standard_threshold
            try:
                self.verifier.observe(track_id, frame, row)
                probability = (
                    self.verifier.probability(state) if state is not None else 0.0
                )
                accepted = baseline or probability >= self.verifier.threshold
                decision = "ACCEPT" if accepted else "REJECT"
                processing_status = "NORMAL"
                error_type = ""
            except Exception as error:
                probability = None
                accepted = True
                decision = "BYPASS"
                processing_status = "VERIFIER_UNAVAILABLE"
                error_type = type(error).__name__
                self.last_fallbacks.append(
                    {
                        "frame_number": self.frame_index,
                        "track_id": track_id,
                        "candidate_id": row["candidate_id"],
                        "error_type": error_type,
                    }
                )
            self.last_trace_rows.append(
                {
                    **row,
                    "verifier_score": probability,
                    "verifier_decision": decision,
                    "verifier_error_type": error_type,
                    "processing_status": processing_status,
                    "accepted": accepted,
                }
            )
            if accepted:
                item = dict(row)
                item["confirmed"] = bool(state.confirmed if state is not None else baseline)
                item["verifier_probability"] = probability
                item["verifier_decision"] = decision
                item["processing_status"] = processing_status
                item["verifier_error_type"] = error_type
                if processing_status == "VERIFIER_UNAVAILABLE":
                    item["original_source"] = item["source"]
                    item["source"] = "verifier_fallback"
                elif not baseline:
                    item["original_source"] = item["source"]
                    item["source"] = "verifier"
                output.append(item)
        self.last_verifier_ms = (time.perf_counter() - verifier_started) * 1000.0
        return output, events
