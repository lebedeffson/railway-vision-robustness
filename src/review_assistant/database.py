from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .event_aggregator import box_iou, center_distance_ratio
from .models import ReviewEvent


SCHEMA_VERSION = 1
REVIEW_STATUSES = {
    "PENDING",
    "HUMAN",
    "FALSE_POSITIVE",
    "UNCERTAIN",
    "SKIPPED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReviewDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def migrate(self) -> None:
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    fps REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES videos(video_id),
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    processed_frames INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER NOT NULL,
                    raw_detection_count INTEGER NOT NULL DEFAULT 0,
                    track_count INTEGER NOT NULL DEFAULT 0,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    processing_seconds REAL NOT NULL DEFAULT 0,
                    review_seconds REAL NOT NULL DEFAULT 0,
                    config_sha256 TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL,
                    video_id TEXT NOT NULL REFERENCES videos(video_id),
                    camera_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    source_label TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    track_ids_json TEXT NOT NULL,
                    maximum_confidence REAL NOT NULL,
                    real_detection_count INTEGER NOT NULL,
                    interpolated_count INTEGER NOT NULL,
                    spatial_region_json TEXT NOT NULL,
                    representative_motion REAL NOT NULL DEFAULT 0,
                    review_status TEXT NOT NULL DEFAULT 'PENDING',
                    operator_comment TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 100,
                    muted INTEGER NOT NULL DEFAULT 0,
                    clip_path TEXT,
                    thumbnail_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS event_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    track_id INTEGER,
                    interpolated INTEGER NOT NULL,
                    confirmed INTEGER NOT NULL,
                    motion REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY (run_id, event_id)
                      REFERENCES events(run_id, event_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    old_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    old_comment TEXT NOT NULL,
                    new_comment TEXT NOT NULL,
                    action TEXT NOT NULL,
                    FOREIGN KEY (run_id, event_id)
                      REFERENCES events(run_id, event_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS camera_profiles (
                    camera_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    nominal_fps REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS suppression_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    region_json TEXT NOT NULL,
                    reference_confidence REAL NOT NULL,
                    reference_motion REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    manual_confirmed INTEGER NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_queue
                  ON events(run_id, muted, review_status, priority DESC, start_time);
                CREATE INDEX IF NOT EXISTS idx_event_detections_event
                  ON event_detections(run_id, event_id, frame_number);
                """
            )
            db.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    @property
    def schema_version(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row["value"])

    def audit(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        details: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO audit_log(
              timestamp, actor, action, entity_type, entity_id, details_json
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                actor,
                action,
                entity_type,
                entity_id,
                json.dumps(details, sort_keys=True),
            ),
        )

    def register_video(
        self,
        *,
        path: Path,
        sha256: str,
        camera_id: str,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
    ) -> tuple[str, bool]:
        existing = self.connection.execute(
            "SELECT video_id FROM videos WHERE sha256=?", (sha256,)
        ).fetchone()
        if existing:
            return str(existing["video_id"]), True
        video_id = f"V-{sha256[:16]}"
        duration = frame_count / max(fps, 1e-9)
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO videos VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    sha256,
                    str(path.resolve()),
                    camera_id,
                    float(fps),
                    int(frame_count),
                    int(width),
                    int(height),
                    duration,
                    utc_now(),
                ),
            )
            self.audit(
                "system", "VIDEO_REGISTERED", "video", video_id, {"sha256": sha256}
            )
        return video_id, False

    def create_run(
        self,
        *,
        video_id: str,
        mode: str,
        total_frames: int,
        config_sha256: str,
        resume_policy: str,
        run_id_override: str | None = None,
    ) -> tuple[str, bool]:
        incomplete = self.connection.execute(
            """
            SELECT run_id FROM runs
            WHERE video_id=? AND mode=? AND status IN ('QUEUED','PROCESSING','FAILED')
            ORDER BY started_at DESC LIMIT 1
            """,
            (video_id, mode),
        ).fetchone()
        if incomplete and resume_policy == "restart_incomplete_run":
            run_id = str(incomplete["run_id"])
            with self.transaction() as db:
                db.execute("DELETE FROM events WHERE run_id=?", (run_id,))
                db.execute(
                    """
                    UPDATE runs SET status='QUEUED', processed_frames=0,
                      raw_detection_count=0, track_count=0, event_count=0,
                      processing_seconds=0, completed_at=NULL, error=NULL,
                      started_at=?
                    WHERE run_id=?
                    """,
                    (utc_now(), run_id),
                )
                self.audit(
                    "system",
                    "RUN_RESTARTED_FROM_BEGINNING",
                    "run",
                    run_id,
                    {"reason": "causal tracker state is intentionally rebuilt"},
                )
            return run_id, True
        seed = f"{video_id}|{mode}|{utc_now()}".encode()
        run_id = run_id_override or f"R-{hashlib.sha256(seed).hexdigest()[:16]}"
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO runs(
                  run_id, video_id, mode, status, started_at, total_frames,
                  config_sha256
                ) VALUES(?, ?, ?, 'QUEUED', ?, ?, ?)
                """,
                (run_id, video_id, mode, utc_now(), total_frames, config_sha256),
            )
            self.audit("system", "RUN_CREATED", "run", run_id, {"mode": mode})
        return run_id, False

    def update_progress(
        self, run_id: str, processed_frames: int, raw_detection_count: int
    ) -> None:
        with self.transaction() as db:
            db.execute(
                """
                UPDATE runs SET status='PROCESSING', processed_frames=?,
                  raw_detection_count=? WHERE run_id=?
                """,
                (int(processed_frames), int(raw_detection_count), run_id),
            )

    def fail_run(self, run_id: str, error: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE runs SET status='FAILED', error=? WHERE run_id=?",
                (error, run_id),
            )
            self.audit("system", "RUN_FAILED", "run", run_id, {"error": error})

    def save_events(self, run_id: str, events: list[ReviewEvent]) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute("DELETE FROM events WHERE run_id=?", (run_id,))
            for event in events:
                record = event.to_record()
                db.execute(
                    """
                    INSERT INTO events(
                      run_id, event_id, video_id, camera_id, start_time,
                      end_time, duration, source_label, sources_json,
                      track_ids_json, maximum_confidence, real_detection_count,
                      interpolated_count, spatial_region_json,
                      representative_motion, review_status, operator_comment,
                      priority, muted, clip_path, thumbnail_path, created_at,
                      updated_at
                    ) VALUES(
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      NULL, NULL, ?, ?
                    )
                    """,
                    (
                        run_id,
                        record["event_id"],
                        record["video_id"],
                        record["camera_id"],
                        record["start_time"],
                        record["end_time"],
                        record["duration"],
                        record["source_label"],
                        json.dumps(record["sources"]),
                        json.dumps(record["track_ids"]),
                        record["maximum_confidence"],
                        record["real_detection_count"],
                        record["interpolated_count"],
                        json.dumps(record["spatial_region"]),
                        record["representative_motion"],
                        record["review_status"],
                        record["operator_comment"],
                        record["priority"],
                        int(record["muted"]),
                        now,
                        now,
                    ),
                )
                for detection in event.detections:
                    db.execute(
                        """
                        INSERT INTO event_detections(
                          run_id, event_id, frame_number, timestamp, bbox_json,
                          confidence, source, track_id, interpolated, confirmed,
                          motion
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            event.event_id,
                            detection.frame_number,
                            detection.timestamp,
                            json.dumps(detection.box),
                            detection.confidence,
                            detection.source,
                            detection.track_id,
                            int(detection.interpolated),
                            int(detection.confirmed),
                            detection.motion,
                        ),
                    )

    def set_event_media(
        self, run_id: str, event_id: str, clip_path: Path, thumbnail_path: Path
    ) -> None:
        with self.transaction() as db:
            db.execute(
                """
                UPDATE events SET clip_path=?, thumbnail_path=?, updated_at=?
                WHERE run_id=? AND event_id=?
                """,
                (
                    str(clip_path.resolve()),
                    str(thumbnail_path.resolve()),
                    utc_now(),
                    run_id,
                    event_id,
                ),
            )

    def complete_run(
        self,
        run_id: str,
        *,
        processing_seconds: float,
        track_count: int,
        event_count: int,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                """
                UPDATE runs SET status='COMPLETED', completed_at=?,
                  processed_frames=total_frames, processing_seconds=?,
                  track_count=?, event_count=? WHERE run_id=?
                """,
                (
                    utc_now(),
                    float(processing_seconds),
                    int(track_count),
                    int(event_count),
                    run_id,
                ),
            )
            self.audit(
                "system",
                "RUN_COMPLETED",
                "run",
                run_id,
                {"event_count": event_count},
            )

    def add_review_seconds(self, run_id: str, seconds: float) -> None:
        with self.transaction() as db:
            db.execute(
                """
                UPDATE runs SET review_seconds=review_seconds+?
                WHERE run_id=?
                """,
                (max(float(seconds), 0.0), run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT runs.*, videos.path, videos.camera_id, videos.duration_seconds
                FROM runs JOIN videos USING(video_id)
                ORDER BY runs.started_at DESC
                """
            )
        ]

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT runs.*, videos.path, videos.sha256 AS video_sha256,
              videos.camera_id, videos.fps, videos.frame_count,
              videos.width, videos.height, videos.duration_seconds
            FROM runs JOIN videos USING(video_id) WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def event_detections(self, run_id: str, event_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM event_detections
            WHERE run_id=? AND event_id=? ORDER BY frame_number, id
            """,
            (run_id, event_id),
        )
        return [dict(row) for row in rows]

    def list_events(
        self,
        run_id: str,
        *,
        muted: bool | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id=?"]
        values: list[Any] = [run_id]
        if muted is not None:
            clauses.append("muted=?")
            values.append(int(muted))
        if status is not None:
            clauses.append("review_status=?")
            values.append(status)
        rows = self.connection.execute(
            f"""
            SELECT * FROM events WHERE {' AND '.join(clauses)}
            ORDER BY priority DESC, start_time, event_id
            """,
            values,
        )
        return [self._decode_event(dict(row)) for row in rows]

    def get_event(self, run_id: str, event_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM events WHERE run_id=? AND event_id=?",
            (run_id, event_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, event_id))
        return self._decode_event(dict(row))

    @staticmethod
    def _decode_event(row: dict[str, Any]) -> dict[str, Any]:
        for field in ("sources_json", "track_ids_json", "spatial_region_json"):
            row[field.removesuffix("_json")] = json.loads(row[field])
        row["muted"] = bool(row["muted"])
        return row

    def review_event(
        self,
        run_id: str,
        event_id: str,
        *,
        operator: str,
        new_status: str,
        comment: str = "",
        action: str = "REVIEW",
    ) -> None:
        if new_status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid review status: {new_status}")
        event = self.get_event(run_id, event_id)
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO reviews(
                  run_id, event_id, operator, timestamp, old_status, new_status,
                  old_comment, new_comment, action
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event_id,
                    operator,
                    utc_now(),
                    event["review_status"],
                    new_status,
                    event["operator_comment"],
                    comment,
                    action,
                ),
            )
            db.execute(
                """
                UPDATE events SET review_status=?, operator_comment=?,
                  updated_at=? WHERE run_id=? AND event_id=?
                """,
                (new_status, comment, utc_now(), run_id, event_id),
            )
            self.audit(
                operator,
                action,
                "event",
                f"{run_id}/{event_id}",
                {
                    "old_status": event["review_status"],
                    "new_status": new_status,
                },
            )

    def undo_last_review(self, run_id: str, event_id: str, operator: str) -> None:
        last = self.connection.execute(
            """
            SELECT * FROM reviews WHERE run_id=? AND event_id=?
            ORDER BY id DESC LIMIT 1
            """,
            (run_id, event_id),
        ).fetchone()
        if last is None:
            raise RuntimeError("No review decision to undo")
        self.review_event(
            run_id,
            event_id,
            operator=operator,
            new_status=str(last["old_status"]),
            comment=str(last["old_comment"]),
            action="UNDO_REVIEW",
        )

    def create_suppression_rule(
        self,
        *,
        camera_id: str,
        region: list[float],
        reference_confidence: float,
        reference_motion: float,
        operator: str,
        reason: str,
        manual_confirmed: bool,
    ) -> int:
        if not manual_confirmed:
            raise RuntimeError("Suppression rules require explicit operator confirmation")
        with self.transaction() as db:
            cursor = db.execute(
                """
                INSERT INTO suppression_rules(
                  camera_id, region_json, reference_confidence,
                  reference_motion, manual_confirmed, created_by,
                  created_at, reason
                ) VALUES(?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    camera_id,
                    json.dumps(region),
                    float(reference_confidence),
                    float(reference_motion),
                    operator,
                    utc_now(),
                    reason,
                ),
            )
            rule_id = int(cursor.lastrowid)
            self.audit(
                operator,
                "SUPPRESSION_RULE_CREATED",
                "suppression_rule",
                str(rule_id),
                {"camera_id": camera_id, "region": region},
            )
        return rule_id

    def apply_suppression(
        self,
        run_id: str,
        event_id: str,
        settings: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        event = self.get_event(run_id, event_id)
        rules = self.connection.execute(
            """
            SELECT * FROM suppression_rules
            WHERE camera_id=? AND active=1 AND manual_confirmed=1
            """,
            (event["camera_id"],),
        )
        muted = False
        for rule in rules:
            region = json.loads(rule["region_json"])
            spatial_match = box_iou(region, event["spatial_region"]) >= float(
                settings["region_iou"]
            )
            if not spatial_match:
                continue
            confidence_delta = abs(
                event["maximum_confidence"] - rule["reference_confidence"]
            ) / max(abs(rule["reference_confidence"]), 1e-6)
            motion_delta = abs(
                event["representative_motion"] - rule["reference_motion"]
            )
            center_delta = center_distance_ratio(
                region,
                event["spatial_region"],
                frame_width,
                frame_height,
            )
            changed = (
                confidence_delta > float(settings["confidence_change_ratio"])
                or motion_delta > float(settings["motion_change_ratio"])
                or center_delta > float(settings["center_change_ratio"])
            )
            muted = not changed
            if muted:
                break
        with self.transaction() as db:
            db.execute(
                "UPDATE events SET muted=?, priority=?, updated_at=? "
                "WHERE run_id=? AND event_id=?",
                (
                    int(muted),
                    int(
                        settings["muted_priority"]
                        if muted
                        else settings["normal_priority"]
                    ),
                    utc_now(),
                    run_id,
                    event_id,
                ),
            )
        return muted

    def suppression_suggestion_count(
        self, camera_id: str, region: list[float], minimum_iou: float
    ) -> int:
        rows = self.connection.execute(
            """
            SELECT spatial_region_json FROM events
            WHERE camera_id=? AND review_status='FALSE_POSITIVE'
            """,
            (camera_id,),
        )
        return sum(
            box_iou(region, json.loads(row["spatial_region_json"])) >= minimum_iou
            for row in rows
        )

    def audit_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM audit_log ORDER BY id"
            )
        ]
