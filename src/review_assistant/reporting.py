from __future__ import annotations

import csv
import html
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .database import ReviewDatabase


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["event_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_metrics(
    run: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    duration_hours = float(run["duration_seconds"]) / 3600.0
    safe_hours = max(duration_hours, 1e-9)
    unique_events = len(events)
    humans = [row for row in events if row["review_status"] == "HUMAN"]
    false_events = [
        row for row in events if row["review_status"] == "FALSE_POSITIVE"
    ]
    uncertain = [row for row in events if row["review_status"] == "UNCERTAIN"]
    temporal_humans = [
        row
        for row in humans
        if row["source_label"] == "TEMPORAL_ONLY"
    ]
    return {
        "run_id": run["run_id"],
        "video_duration_seconds": run["duration_seconds"],
        "processing_seconds": run["processing_seconds"],
        "manual_review_seconds": run["review_seconds"],
        "raw_detections": run["raw_detection_count"],
        "tracks": run["track_count"],
        "unique_events": unique_events,
        "confirmed_people": len(humans),
        "false_events": len(false_events),
        "uncertain_events": len(uncertain),
        "pending_or_skipped_events": unique_events
        - len(humans)
        - len(false_events)
        - len(uncertain),
        "temporal_only_confirmed_people": len(temporal_humans),
        "review_minutes_per_video_hour": (run["review_seconds"] / 60.0)
        / safe_hours,
        "raw_detections_per_unique_event": run["raw_detection_count"]
        / max(unique_events, 1),
        "events_per_video_hour": unique_events / safe_hours,
        "confirmed_person_events_per_video_hour": len(humans) / safe_hours,
        "false_events_per_video_hour": len(false_events) / safe_hours,
        "human_confirmation_required": True,
        "autonomous_alarm_output": False,
        "safety_actuation": False,
    }


def _html_report(metrics: dict[str, Any], events: list[dict[str, Any]]) -> str:
    cards = "".join(
        f"""
        <div class="metric"><span>{html.escape(label)}</span><strong>{value}</strong></div>
        """
        for label, value in (
            ("Unique events", metrics["unique_events"]),
            ("Confirmed people", metrics["confirmed_people"]),
            ("False events", metrics["false_events"]),
            ("Uncertain", metrics["uncertain_events"]),
            (
                "Review min / video hour",
                f'{metrics["review_minutes_per_video_hour"]:.2f}',
            ),
            (
                "Raw detections / event",
                f'{metrics["raw_detections_per_unique_event"]:.1f}',
            ),
        )
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['event_id'])}</td>"
        f"<td>{row['start_time']:.1f}–{row['end_time']:.1f}</td>"
        f"<td><span class='pill {row['source_label'].lower()}'>{row['source_label']}</span></td>"
        f"<td>{row['maximum_confidence']:.3f}</td>"
        f"<td>{row['real_detection_count']}</td>"
        f"<td>{row['interpolated_count']}</td>"
        f"<td>{html.escape(row['review_status'])}</td>"
        "</tr>"
        for row in events
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Railway Person Review Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:0;background:#f3f5f7;color:#18222c}}
header{{background:#122637;color:white;padding:28px 40px;border-bottom:5px solid #e8a317}}
header p{{max-width:900px;color:#dce6ec}} main{{padding:28px 40px}}
.warning{{background:#fff3cd;border:1px solid #e8a317;padding:14px;margin:18px 0}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.metric{{background:white;border-radius:9px;padding:18px;border-left:5px solid #1c6e8c}}
.metric span{{display:block;color:#667784;font-size:13px}}.metric strong{{font-size:26px}}
table{{width:100%;border-collapse:collapse;background:white;margin-top:24px}}
th,td{{padding:10px;border-bottom:1px solid #dae1e5;text-align:left}}
th{{background:#e7eef2}}.pill{{padding:3px 7px;border-radius:9px;font-size:11px}}
.baseline{{background:#d5efde}}.temporal_only{{background:#ffe3b3}}.both{{background:#dbe8ff}}
footer{{margin-top:28px;color:#6d7a83;font-size:12px}}
</style></head><body>
<header><h1>Railway Person Review Assistant</h1>
<p>Operator-in-the-loop review report. No autonomous alarming or safety
actuation is performed.</p></header><main>
<div class="warning"><strong>HUMAN CONFIRMATION REQUIRED.</strong>
Temporal-only events have an elevated false-positive risk.</div>
<div class="metrics">{cards}</div>
<h2>Event queue</h2><table><thead><tr><th>Event</th><th>Time</th>
<th>Source</th><th>Max confidence</th><th>Detector frames</th>
<th>Interpolated</th><th>Review</th></tr></thead><tbody>{rows}</tbody></table>
<footer>Product status: OPERATOR_ASSISTANT_MVP · Autonomous alarming:
DISABLED · Safety actuation: DISABLED</footer></main></body></html>"""


def _render_pdf(html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="review-report-") as temporary:
        temp = Path(temporary)
        markdown = temp / "report.md"
        # Pandoc handles the plain-text extract predictably; HTML remains the rich report.
        markdown.write_text(
            "# Railway Person Review Assistant\n\n"
            "**HUMAN CONFIRMATION REQUIRED**  \n"
            "Autonomous alarming: DISABLED  \n"
            "Safety actuation: DISABLED\n\n"
            + html_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        docx = temp / "report.docx"
        subprocess.run(
            ["pandoc", str(html_path), "-o", str(docx)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        profile = temp / "lo-profile"
        profile.mkdir()
        subprocess.run(
            [
                "libreoffice",
                f"-env:UserInstallation=file://{profile}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temp),
                str(docx),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        generated = temp / "report.pdf"
        pdf_path.write_bytes(generated.read_bytes())


def generate_report(
    database_path: str | Path,
    run_id: str,
    output_root: str | Path,
    *,
    resolved_config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    with ReviewDatabase(database_path) as database:
        run = database.get_run(run_id)
        events = database.list_events(run_id)
        audit = database.audit_rows()
    export_rows = []
    for row in events:
        clean = {
            key: value
            for key, value in row.items()
            if key not in {"clip_path", "thumbnail_path"}
        }
        for key, value in list(clean.items()):
            if isinstance(value, (list, dict)):
                clean[key] = json.dumps(value, sort_keys=True)
        export_rows.append(clean)
    metrics = report_metrics(run, events)
    _write_csv(root / "events.csv", export_rows)
    _write_csv(
        root / "confirmed_people.csv",
        [row for row in export_rows if row["review_status"] == "HUMAN"],
    )
    _write_csv(
        root / "false_events.csv",
        [row for row in export_rows if row["review_status"] == "FALSE_POSITIVE"],
    )
    _write_csv(
        root / "uncertain_events.csv",
        [row for row in export_rows if row["review_status"] == "UNCERTAIN"],
    )
    _write_csv(root / "audit_log.csv", audit)
    html_path = root / "REVIEW_REPORT.html"
    html_path.write_text(_html_report(metrics, events), encoding="utf-8")
    _render_pdf(html_path, root / "REVIEW_REPORT.pdf")
    (root / "report_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if resolved_config is not None:
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )
    if provenance is not None:
        (root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    runtime = {
        "processing_seconds": run["processing_seconds"],
        "review_seconds": run["review_seconds"],
        "video_duration_seconds": run["duration_seconds"],
        "frames": run["frame_count"],
        "status": run["status"],
    }
    (root / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics
