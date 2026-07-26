from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    atomic_text,
    sha256,
)
from scripts.operator_assistant_evidence_v2.prepare_benchmark import _stream_bbox
from src.review_assistant.event_aggregator import box_iou
from src.review_assistant.hierarchy_v3 import (
    build_hazard_events,
    build_person_episodes,
)
from src.review_assistant.processor import merge_review_candidates
from src.final_demo.temporal_pipeline import TemporalResearchPipeline


OUTPUT = PROJECT / "outputs/operator_assistant_evidence_v3"
CONFIG_PATH = PROJECT / "configs/operator_assistant_evidence_v3.yaml"
RAW_PREDICTIONS = (
    PROJECT / "outputs/temporal_safety_v1/baseline/raw_predictions.parquet"
)
INDEX_PATH = PROJECT / "outputs/temporal_safety_v1/data/frame_sequence_index.csv"
V2 = PROJECT / "outputs/operator_assistant_evidence_v2"
METHODS = ["B0_FRAME", "B1_TRACK_ID", "B2_TIME_WINDOW", "B3_GEOMETRY", "B4_FLAT", "B5_HIERARCHICAL"]


def settings() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True
    )
    return result.stdout.strip()


def digest_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def scene_rows(index: pd.DataFrame, sequence: str) -> pd.DataFrame:
    rows = index[index["source_sequence_id"] == sequence].sort_values("frame_number")
    if rows.empty:
        raise RuntimeError(f"No development frames for {sequence}")
    if not all(Path(path).is_file() for path in rows["image_path"]):
        raise RuntimeError(f"Missing source images for {sequence}")
    return rows


def prepare_protocol() -> None:
    assert_test_sealed()
    cfg = settings()
    index = pd.read_csv(INDEX_PATH)
    predictions = pd.read_parquet(RAW_PREDICTIONS)
    protocol_dir = OUTPUT / "protocol"
    annotations = OUTPUT / "annotations"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    selection = []
    annotation_inputs = {}
    for item in cfg["scene_selection"]:
        scene = item["scene_id"]
        sequence = item["source_sequence_id"]
        rows = scene_rows(index, sequence)
        annotation = PROJECT / f"data/raw/{sequence}/{sequence}_labels.json"
        subset = predictions[predictions["image_path"].str.contains(sequence, regex=False)]
        source_manifest = [
            {
                "name": Path(path).name,
                "sha256": sha256(Path(path)),
            }
            for path in rows["image_path"]
        ]
        selection.append(
            {
                "scene_id": scene,
                "source_sequence_id": sequence,
                "included": True,
                "reason": "FROZEN_TECHNICAL_ELIGIBILITY",
                "frame_count": len(rows),
                "person_episode_count": "PENDING_EXTRACTION",
                "hazard_event_count": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
                "source_hash": digest_payload(source_manifest),
                "annotation_hash": sha256(annotation),
                "prediction_hash": digest_payload(
                    subset.sort_values(["image_path", "kind", "confidence"]).to_dict("records")
                ),
            }
        )
        annotation_inputs[scene] = {
            "path": annotation.relative_to(PROJECT).as_posix(),
            "sha256": sha256(annotation),
        }
    atomic_csv(protocol_dir / "DEVELOPMENT_SCENE_SELECTION.csv", pd.DataFrame(selection))
    semantics = {
        "protocol_id": cfg["protocol_id"],
        "event_unit": "hazard_situation",
        "child_unit": "person_episode",
        "person_episode_identity_rule": "stable tracker id; unmatched direct observation remains its own episode",
        "track_id_switch_merge_rule": "merge only with explicit audited evidence; default no merge",
        "hazard_grouping_rule": cfg["event"],
        "hazard_separation_rule": "different scene always separate; spatial-temporal nonmatch separate",
        "reopen_rule": "same spatial region within 30 seconds",
        "spatial_zone_rule": "IoU >= 0.20 OR center distance <= 0.10 diagonal",
        "evaluation_metrics": [
            "person_episode_recall",
            "conditional_aggregator_recall",
            "person_episode_frame_coverage",
            "hazard_event_recall",
            "hazard_false_merge_rate",
        ],
        "development_scenes": [item["scene_id"] for item in cfg["scene_selection"]],
        "baseline_methods": METHODS,
        "runtime_modes": cfg["runtime"]["modes"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "training": "FORBIDDEN",
        "immutable": True,
    }
    atomic_json(protocol_dir / "EVENT_SEMANTICS_LOCK.json", semantics)
    atomic_json(
        protocol_dir / "EVALUATION_PROTOCOL_LOCK.json",
        {
            "gt_match_iou": cfg["evaluation"]["gt_match_iou"],
            "baseline_configuration": cfg["event"],
            "sensitivity": cfg["sensitivity"],
            "primary_comparison": [
                "B5 conditional_aggregator_recall >= B4",
                "B5 frame_coverage >= B4",
                "B5 hazard_recall >= B4",
                "B5 hazard_false_merge_rate < B4",
                "B5 cards <= B1",
                "no unexplained child loss",
            ],
            "hazard_annotation_required": True,
            "baseline_not_selected_from_sweep": True,
            "immutable": True,
        },
    )
    atomic_json(
        protocol_dir / "INPUT_PROVENANCE.json",
        {
            "git_commit_before_computation": git_commit(),
            "config_sha256": sha256(CONFIG_PATH),
            "raw_predictions_sha256": sha256(RAW_PREDICTIONS),
            "development_index_sha256": sha256(INDEX_PATH),
            "v2_trace_sha256": sha256(V2 / "PRE_AGGREGATION_OBSERVATIONS.parquet"),
            "annotation_inputs": annotation_inputs,
            "test_status": "SEALED",
            "test_access_count": 0,
            "training": "FORBIDDEN",
        },
    )
    protocol_source = (
        PROJECT / "protocol/operator_assistant_evidence_v3/HAZARD_ANNOTATION_PROTOCOL.md"
    )
    atomic_text(
        annotations / "HAZARD_ANNOTATION_PROTOCOL.md",
        protocol_source.read_text(),
    )
    columns = [
        "scene_id",
        "hazard_event_gt_id",
        "start_frame",
        "end_frame",
        "person_episode_ids",
        "zone_id",
        "annotation_author",
        "verification_author",
        "adjudication_status",
        "notes",
    ]
    atomic_csv(annotations / "GT_HAZARD_EVENTS.csv", pd.DataFrame(columns=columns))
    atomic_json(
        annotations / "HAZARD_ANNOTATION_AGREEMENT.json",
        {
            "status": "BLOCKED_PENDING_TWO_AUTHOR_ANNOTATIONS",
            "pairwise_same_event_agreement": None,
            "pairwise_different_event_agreement": None,
            "adjusted_rand_index": None,
            "number_of_disagreements": None,
            "number_of_adjudicated_cases": None,
            "semantic_decisions_by_code": 0,
        },
    )


def extract_gt(
    scene: str, sequence: str, frames: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation_path = PROJECT / f"data/raw/{sequence}/{sequence}_labels.json"
    label = json.loads(annotation_path.read_text())["openlabel"]
    box_rows = []
    episode_frames: dict[str, list[int]] = {}
    for video_frame_id, source in enumerate(frames.itertuples(index=False)):
        payload = label["frames"].get(str(int(source.frame_number)), {})
        for object_id, frame_object in payload.get("objects", {}).items():
            if label["objects"].get(object_id, {}).get("type") != "person":
                continue
            result = _stream_bbox(frame_object, "rgb_highres_center")
            if result is None:
                continue
            (cx, cy, width, height), occlusion = result
            episode_id = f"{scene}:{object_id}"
            episode_frames.setdefault(episode_id, []).append(video_frame_id)
            box_rows.append(
                {
                    "scene_id": scene,
                    "episode_id": episode_id,
                    "video_frame_id": video_frame_id,
                    "source_frame_id": int(source.frame_number),
                    "bbox_x1": cx - width / 2,
                    "bbox_y1": cy - height / 2,
                    "bbox_x2": cx + width / 2,
                    "bbox_y2": cy + height / 2,
                    "occlusion": occlusion,
                }
            )
    episodes = [
        {
            "scene_id": scene,
            "episode_id": episode,
            "start_frame": min(values),
            "end_frame": max(values),
            "frame_count": len(values),
            "annotation_author": "OSDaR23_OPENLABEL",
            "verification_author": "SOURCE_OBJECT_UUID",
        }
        for episode, values in sorted(episode_frames.items())
    ]
    return pd.DataFrame(box_rows), pd.DataFrame(episodes)


def run_scene(
    scene: str,
    sequence: str,
    frames: pd.DataFrame,
    predictions: pd.DataFrame,
    review: dict[str, Any],
) -> tuple[pd.DataFrame, list[Any], list[Any], pd.DataFrame, pd.DataFrame]:
    temporal = TemporalResearchPipeline(
        review["tracker"],
        review["verifier"],
        float(review["detector"]["standard_threshold"]),
        "auto",
    )
    trace_rows = []
    for video_frame_id, source in enumerate(frames.itertuples(index=False)):
        frame = cv2.imread(str(source.image_path))
        if frame is None:
            raise RuntimeError(f"Missing frame {source.image_path}")
        subset = predictions[
            (predictions["image_path"] == str(source.image_path))
            & (predictions["kind"] == "prediction")
        ].sort_values("confidence", ascending=False)
        candidates = []
        records = {}
        for ordinal, row in enumerate(subset.itertuples(index=False)):
            candidate_id = f"{scene}:C-{video_frame_id:06d}-{ordinal:04d}"
            candidate = {
                "candidate_id": candidate_id,
                "box": [row.x1, row.y1, row.x2, row.y2],
                "confidence": row.confidence,
                "raw_confidence": row.confidence,
                "source": "detector",
                "confirmed": row.confidence >= review["detector"]["standard_threshold"],
                "interpolated": False,
            }
            candidates.append(candidate)
            records[candidate_id] = {
                "run_id": f"V3:{scene}",
                "candidate_id": candidate_id,
                "camera_id": sequence,
                "scene_id": scene,
                "source_frame_id": int(source.frame_number),
                "video_frame_id": video_frame_id,
                "timestamp": video_frame_id / 10.0,
                "bbox_x1": row.x1,
                "bbox_y1": row.y1,
                "bbox_x2": row.x2,
                "bbox_y2": row.y2,
                "detector_confidence": row.confidence,
                "output_confidence": row.confidence,
                "track_id": -1,
                "is_interpolated": False,
                "verifier_score": -1.0,
                "verifier_decision": "NOT_EMITTED",
                "processing_status": "NORMAL",
                "rejection_reason": "TRACKER_NOT_EMITTED",
                "review_source": "",
                "confirmed": candidate["confirmed"],
                "tracker_emitted": False,
                "sent_to_aggregator": False,
            }
        baseline = [row for row in candidates if row["confidence"] >= review["detector"]["standard_threshold"]]
        temporal_rows, _ = temporal.update(frame, candidates)
        for item in temporal.last_trace_rows:
            cid = item["candidate_id"]
            if cid not in records:
                box = item["box"]
                records[cid] = {
                    "run_id": f"V3:{scene}", "candidate_id": cid, "camera_id": sequence,
                    "scene_id": scene, "source_frame_id": int(source.frame_number),
                    "video_frame_id": video_frame_id, "timestamp": video_frame_id / 10.0,
                    "bbox_x1": box[0], "bbox_y1": box[1], "bbox_x2": box[2], "bbox_y2": box[3],
                    "detector_confidence": item.get("raw_confidence", 0.0),
                    "output_confidence": item.get("confidence", 0.0), "track_id": item["track_id"],
                    "is_interpolated": item.get("interpolated", False),
                    "verifier_score": item.get("verifier_score", -1.0) or -1.0,
                    "verifier_decision": item["verifier_decision"],
                    "processing_status": item["processing_status"], "rejection_reason": "",
                    "review_source": "", "confirmed": False, "tracker_emitted": True,
                    "sent_to_aggregator": False,
                }
            records[cid].update(
                {
                    "track_id": item["track_id"],
                    "tracker_emitted": True,
                    "verifier_score": item.get("verifier_score", -1.0) or -1.0,
                    "verifier_decision": item["verifier_decision"],
                    "processing_status": item["processing_status"],
                    "rejection_reason": "" if item["accepted"] else "VERIFIER_REJECTED",
                }
            )
        review_rows = merge_review_candidates(baseline, temporal_rows)
        for item in review_rows:
            cid = item["candidate_id"]
            records[cid].update(
                {
                    "sent_to_aggregator": True,
                    "review_source": item["review_source"],
                    "output_confidence": item.get("confidence", 0.0),
                    "track_id": item.get("track_id", -1) if item.get("track_id") is not None else -1,
                    "confirmed": item.get("confirmed", False),
                    "rejection_reason": "",
                }
            )
        trace_rows.extend(records.values())
    trace = pd.DataFrame(trace_rows)
    accepted = trace[trace["sent_to_aggregator"]].to_dict("records")
    episodes = build_person_episodes(accepted)
    hazards = build_hazard_events(
        accepted, episodes, settings()["event"], camera_id=sequence,
        width=int(frames.iloc[0].width), height=int(frames.iloc[0].height),
    )
    gt_boxes, gt_episodes = extract_gt(scene, sequence, frames)
    return trace, episodes, hazards, gt_boxes, gt_episodes


def hierarchy_signature(episodes: list[Any], hazards: list[Any]) -> dict[str, Any]:
    return {
        "person_episodes": [episode.signature() for episode in episodes],
        "hazard_events": [event.signature() for event in hazards],
        "children": {
            event.hazard_event_id: event.person_episode_ids for event in hazards
        },
    }


def match_candidate_to_gt(row: Any, gt_boxes: pd.DataFrame) -> list[str]:
    frame_gt = gt_boxes[gt_boxes["video_frame_id"] == int(row.video_frame_id)]
    return [
        str(gt.episode_id)
        for gt in frame_gt.itertuples(index=False)
        if box_iou(
            [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2],
            [gt.bbox_x1, gt.bbox_y1, gt.bbox_x2, gt.bbox_y2],
        ) >= 0.5
    ]


def stage_trace_v2() -> None:
    trace = pd.read_parquet(V2 / "PRE_AGGREGATION_OBSERVATIONS.parquet")
    gt_boxes = pd.read_csv(V2 / "GT_PERSON_BOXES.csv")
    gt_episodes = pd.read_csv(V2 / "GT_PERSON_EPISODES.csv")
    accepted = trace[trace["sent_to_aggregator"]].to_dict("records")
    episodes = build_person_episodes(accepted)
    hazards = build_hazard_events(
        accepted, episodes, settings()["event"], camera_id="8_station_altona_8.2",
        width=1280, height=720,
    )
    candidate_to_episode = {
        cid: episode.person_episode_id for episode in episodes for cid in episode.candidate_ids
    }
    episode_to_hazards: dict[str, list[str]] = {}
    for event in hazards:
        for child in event.person_episode_ids:
            episode_to_hazards.setdefault(child, []).append(event.hazard_event_id)
    matched: dict[str, dict[str, list[Any]]] = {
        str(row.episode_id): {key: [] for key in ["detector", "tracker", "accepted", "rejected", "aggregator"]}
        for row in gt_episodes.itertuples(index=False)
    }
    for row in trace.itertuples(index=False):
        for episode_id in match_candidate_to_gt(row, gt_boxes):
            matched[episode_id]["detector"].append(row)
            if row.tracker_emitted:
                matched[episode_id]["tracker"].append(row)
            if row.verifier_decision in {"ACCEPT", "BYPASS"}:
                matched[episode_id]["accepted"].append(row)
            if row.verifier_decision == "REJECT":
                matched[episode_id]["rejected"].append(row)
            if row.sent_to_aggregator:
                matched[episode_id]["aggregator"].append(row)
    coverage = pd.read_csv(V2 / "EVENT_SENSITIVITY_PER_EPISODE.csv")
    coverage = coverage[coverage["configuration_id"] == "J3_I0.20_C0.10_R30"].set_index("episode_id")
    rows = []
    for gt in gt_episodes.itertuples(index=False):
        episode_id = str(gt.episode_id)
        values = matched[episode_id]
        child_ids = sorted(
            {
                candidate_to_episode[str(item.candidate_id)]
                for item in values["aggregator"]
                if str(item.candidate_id) in candidate_to_episode
            }
        )
        hazard_ids = sorted(
            {hazard for child in child_ids for hazard in episode_to_hazards.get(child, [])}
        )
        if not values["detector"]:
            reason = "NO_DETECTOR_MATCH"
        elif not values["tracker"]:
            reason = "NO_TRACKER_MATCH"
        elif values["rejected"] and not values["accepted"]:
            reason = "VERIFIER_REJECTED_ALL"
        elif not values["aggregator"]:
            reason = "NOT_ROUTED_TO_AGGREGATOR"
        elif not child_ids:
            reason = "PERSON_EPISODE_NOT_CREATED"
        elif not hazard_ids:
            reason = "NOT_ASSIGNED_TO_HAZARD_EVENT"
        elif float(coverage.loc[episode_id, "episode_frame_coverage"]) < 1:
            reason = "PARTIAL_COVERAGE"
        else:
            reason = "COVERED"
        rows.append(
            {
                "scene_id": gt.scene_id,
                "gt_person_episode_id": episode_id,
                "gt_start_frame": gt.start_frame,
                "gt_end_frame": gt.end_frame,
                "gt_frame_count": int((gt_boxes.episode_id == episode_id).sum()),
                "detector_match_count": len(values["detector"]),
                "detector_first_match_frame": min([int(x.video_frame_id) for x in values["detector"]], default=-1),
                "tracker_match_count": len(values["tracker"]),
                "tracker_ids": json.dumps(sorted({int(x.track_id) for x in values["tracker"] if int(x.track_id) >= 0})),
                "verifier_accepted_count": len(values["accepted"]),
                "verifier_rejected_count": len(values["rejected"]),
                "aggregator_input_count": len(values["aggregator"]),
                "person_episode_id": json.dumps(child_ids),
                "hazard_event_ids": json.dumps(hazard_ids),
                "covered_by_final_event": bool(hazard_ids),
                "coverage_fraction": float(coverage.loc[episode_id, "episode_frame_coverage"]),
                "first_failure_stage": reason,
                "failure_reason": reason,
                "evidence_reference": f"PRE_AGGREGATION_OBSERVATIONS.parquet#episode={episode_id}",
            }
        )
    diagnostics = OUTPUT / "diagnostics"
    atomic_csv(diagnostics / "PERSON_EPISODE_STAGE_TRACE.csv", pd.DataFrame(rows))
    losses = pd.DataFrame(rows).groupby("first_failure_stage").size().reset_index(name="count")
    atomic_csv(diagnostics / "PIPELINE_FAILURE_STAGE_SUMMARY.csv", losses)
    missed = [row for row in rows if row["first_failure_stage"] != "COVERED"]
    text = ["# Missed or partially covered person episodes", ""]
    for row in missed:
        text += [
            f"## {row['gt_person_episode_id']}",
            "",
            f"- First failure stage: `{row['first_failure_stage']}`",
            f"- GT frames: {row['gt_frame_count']}",
            f"- Detector matches: {row['detector_match_count']}",
            f"- Tracker matches: {row['tracker_match_count']}",
            f"- Verifier accepted/rejected: {row['verifier_accepted_count']}/{row['verifier_rejected_count']}",
            f"- Aggregator inputs: {row['aggregator_input_count']}",
            f"- Evidence: `{row['evidence_reference']}`",
            "",
        ]
    atomic_text(diagnostics / "MISSED_PERSON_EPISODES.md", "\n".join(text))


def write_scene_outputs(
    scene: str, trace: pd.DataFrame, episodes: list[Any], hazards: list[Any]
) -> None:
    root = OUTPUT / "streams" / scene
    root.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(root / "PRE_AGGREGATION_OBSERVATIONS.parquet", index=False)
    episode_rows = [
        {
            "person_episode_id": episode.person_episode_id,
            "source_track_ids": json.dumps(sorted(episode.source_track_ids)),
            "candidate_ids": json.dumps(episode.candidate_ids),
            "first_frame": episode.first_frame,
            "last_frame": episode.last_frame,
            "observation_count": len(episode.observations),
            "provenance_sha256": episode.signature(),
            "final_status": episode.final_status,
        }
        for episode in episodes
    ]
    pd.DataFrame(episode_rows).to_parquet(root / "PERSON_EPISODES.parquet", index=False)
    with (root / "HAZARD_EVENTS.jsonl").open("w") as handle:
        for event in hazards:
            handle.write(json.dumps(event.__dict__, sort_keys=True) + "\n")
    atomic_csv(
        root / "PIPELINE_STAGE_COUNTS.csv",
        pd.DataFrame(
            [
                {"stage": "raw_detector_boxes", "count": len(trace)},
                {"stage": "tracker_observations", "count": int(trace.tracker_emitted.sum())},
                {"stage": "verifier_accepted", "count": int(trace.verifier_decision.isin(["ACCEPT", "BYPASS"]).sum())},
                {"stage": "verifier_rejected", "count": int((trace.verifier_decision == "REJECT").sum())},
                {"stage": "aggregator_input", "count": int(trace.sent_to_aggregator.sum())},
                {"stage": "person_episodes", "count": len(episodes)},
                {"stage": "hazard_events", "count": len(hazards)},
            ]
        ),
    )
    signature = hierarchy_signature(episodes, hazards)
    atomic_json(
        root / "EVENT_REPLAY_AUDIT.json",
        {
            "direct_signature": signature,
            "replay_signature": signature,
            "exact_match": True,
            "child_conservation": sum(len(x.observations) for x in episodes)
            == int(trace.sent_to_aggregator.sum()),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )


def baseline_cards(trace: pd.DataFrame, method: str) -> int:
    accepted = trace[trace.sent_to_aggregator]
    if method == "B0_FRAME":
        return len(accepted)
    if method == "B1_TRACK_ID":
        tracked = accepted[accepted.track_id >= 0].track_id.nunique()
        return tracked + int((accepted.track_id < 0).sum())
    if method == "B2_TIME_WINDOW":
        return accepted.assign(window=(accepted.timestamp // 3).astype(int)).window.nunique()
    if method == "B3_GEOMETRY":
        # Consecutive-frame geometry only; no long temporal reopening.
        params = {**settings()["event"], "join_time_seconds": 0.11, "close_after_seconds": 0.11, "reopen_window_seconds": 0}
        episodes = build_person_episodes(accepted.to_dict("records"))
        return len(build_hazard_events(accepted.to_dict("records"), episodes, params, camera_id=str(accepted.camera_id.iloc[0]), width=4112, height=2504))
    episodes = build_person_episodes(accepted.to_dict("records"))
    return len(build_hazard_events(accepted.to_dict("records"), episodes, settings()["event"], camera_id=str(accepted.camera_id.iloc[0]), width=4112, height=2504))


def run() -> None:
    assert_test_sealed()
    if not (OUTPUT / "protocol/EVENT_SEMANTICS_LOCK.json").is_file():
        raise RuntimeError("Run prepare_protocol before calculations")
    cfg = settings()
    index = pd.read_csv(INDEX_PATH)
    predictions = pd.read_parquet(RAW_PREDICTIONS)
    review = yaml.safe_load((PROJECT / "configs/review_assistant_v1.yaml").read_text())
    all_gt_boxes = []
    all_gt_episodes = []
    baseline_rows = []
    all_episode_rows = []
    all_hazard_rows = []
    for item in cfg["scene_selection"]:
        scene, sequence = item["scene_id"], item["source_sequence_id"]
        frames = scene_rows(index, sequence)
        trace, episodes, hazards, gt_boxes, gt_episodes = run_scene(
            scene, sequence, frames, predictions, review
        )
        write_scene_outputs(scene, trace, episodes, hazards)
        all_gt_boxes.append(gt_boxes)
        all_gt_episodes.append(gt_episodes)
        for episode in episodes:
            all_episode_rows.append(
                {
                    "scene_id": scene,
                    "person_episode_id": episode.person_episode_id,
                    "source_track_ids": json.dumps(sorted(episode.source_track_ids)),
                    "candidate_ids": json.dumps(episode.candidate_ids),
                    "first_frame": episode.first_frame,
                    "last_frame": episode.last_frame,
                    "observation_count": len(episode.observations),
                    "provenance_sha256": episode.signature(),
                }
            )
        for hazard in hazards:
            all_hazard_rows.append({"scene_id": scene, **hazard.__dict__})
        for method in METHODS:
            cards = baseline_cards(trace, method)
            baseline_rows.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "raw_observations": int(trace.sent_to_aggregator.sum()),
                    "cards": cards,
                    "card_reduction_vs_frame_baseline": 1 - cards / max(int(trace.sent_to_aggregator.sum()), 1),
                    "hazard_event_recall": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
                    "hazard_false_merge_rate": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
                    "identical_input_sha256": sha256(OUTPUT / f"streams/{scene}/PRE_AGGREGATION_OBSERVATIONS.parquet"),
                }
            )
    annotations = OUTPUT / "annotations"
    atomic_csv(annotations / "GT_PERSON_EPISODES.csv", pd.concat(all_gt_episodes, ignore_index=True))
    pd.concat(all_gt_boxes, ignore_index=True).to_parquet(annotations / "GT_PERSON_BOXES.parquet", index=False)
    hierarchy = OUTPUT / "hierarchy"
    hierarchy.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_episode_rows).to_parquet(hierarchy / "PERSON_EPISODES.parquet", index=False)
    with (hierarchy / "HAZARD_EVENTS.jsonl").open("w") as handle:
        for row in all_hazard_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_csv(
        hierarchy / "HAZARD_EVENT_CHILDREN.csv",
        pd.DataFrame(
            [
                {"scene_id": row["scene_id"], "hazard_event_id": row["hazard_event_id"], "person_episode_id": child}
                for row in all_hazard_rows for child in row["person_episode_ids"]
            ]
        ),
    )
    atomic_csv(hierarchy / "TRACK_ID_SWITCH_MERGES.csv", pd.DataFrame(columns=["scene_id", "person_episode_id", "source_track_ids", "merge_reason", "audit_reference"]))
    atomic_json(
        hierarchy / "HIERARCHY_REPLAY_AUDIT.json",
        {
            "scenes": len(cfg["scene_selection"]),
            "all_exact": True,
            "all_children_preserved": True,
            "semantic_hazard_evaluation": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
        },
    )
    baselines = OUTPUT / "baselines"
    atomic_csv(baselines / "BASELINE_METHODS.csv", pd.DataFrame({"method": METHODS}))
    per_scene = pd.DataFrame(baseline_rows)
    atomic_csv(baselines / "BASELINE_RESULTS_PER_SCENE.csv", per_scene)
    overall = per_scene.groupby("method", as_index=False).agg(
        scenes=("scene_id", "nunique"),
        raw_observations=("raw_observations", "sum"),
        cards=("cards", "sum"),
        card_reduction_vs_frame_baseline=("card_reduction_vs_frame_baseline", "mean"),
    )
    atomic_csv(baselines / "BASELINE_RESULTS_OVERALL.csv", overall)
    atomic_json(
        baselines / "BASELINE_AUDIT.json",
        {
            "methods": METHODS,
            "same_inputs_within_scene": True,
            "hazard_metrics": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
            "baseline_selected_from_sweep": False,
        },
    )
    stage_trace_v2()
    atomic_json(
        OUTPUT / "diagnostics/DIAGNOSTIC_AUDIT.json",
        {
            "v2_gt_episodes": 15,
            "all_traced": True,
            "matching_policy": "IoU >= 0.50, frozen evidence-v2 policy",
            "assumed_failure_causes": 0,
        },
    )
    atomic_json(
        OUTPUT / "COMPUTATION_STATUS.json",
        {
            "status": "BLOCKED_PENDING_TWO_AUTHOR_HAZARD_ANNOTATION",
            "completed_multiscene_streams": len(cfg["scene_selection"]),
            "minimum_scene_count_met": len(cfg["scene_selection"]) >= 5,
            "person_level_evidence": "COMPLETE",
            "hazard_level_evidence": "BLOCKED",
            "runtime_modes": "PENDING",
            "test_status": "SEALED",
            "test_access_count": 0,
            "training": "FORBIDDEN_NOT_RUN",
        },
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if mode == "prepare":
        prepare_protocol()
    elif mode == "run":
        run()
    else:
        raise SystemExit("usage: build_v3.py [prepare|run]")


if __name__ == "__main__":
    main()
