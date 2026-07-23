from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVES_DIR = PROJECT_DIR / "data/archives"
RAW_DIR = PROJECT_DIR / "data/raw"
RAW_EXCLUSIONS_PATH = PROJECT_DIR / "config/raw_frame_exclusions.json"
DOWNLOAD_BASE_URL = "https://download.data.fid-move.de/dzsf/osdar23"
CAMERA_FOLDER = "rgb_highres_center"
SEQUENCES = [
    "2_station_berliner_tor_2.1",
    "3_fire_site_3.1", "3_fire_site_3.2", "3_fire_site_3.3", "3_fire_site_3.4",
    "4_station_pedestrian_bridge_4.1", "4_station_pedestrian_bridge_4.2",
    "4_station_pedestrian_bridge_4.3", "4_station_pedestrian_bridge_4.4",
    "4_station_pedestrian_bridge_4.5",
    "5_station_bergedorf_5.1", "5_station_bergedorf_5.2",
    "6_station_klein_flottbek_6.1", "6_station_klein_flottbek_6.2",
    "7_approach_underground_station_7.1", "7_approach_underground_station_7.2",
    "7_approach_underground_station_7.3",
    "8_station_altona_8.1", "8_station_altona_8.2", "8_station_altona_8.3",
    "9_station_ruebenkamp_9.1", "9_station_ruebenkamp_9.2",
    "9_station_ruebenkamp_9.3", "9_station_ruebenkamp_9.4",
    "9_station_ruebenkamp_9.5", "9_station_ruebenkamp_9.6",
    "9_station_ruebenkamp_9.7",
    "10_station_suelldorf_10.1", "11_main_station_11.1",
    "12_vegetation_steady_12.1", "13_station_ohlsdorf_13.1",
    "14_signals_station_14.1", "14_signals_station_14.2",
    "14_signals_station_14.3", "15_construction_vehicle_15.1",
    "16_under_bridge_16.1", "17_signal_bridge_17.1",
    "18_vegetation_switch_18.1", "19_vegetation_curve_19.1",
    "20_vegetation_squirrel_20.1", "21_station_wedel_21.1",
    "21_station_wedel_21.2", "21_station_wedel_21.3",
]


DEFAULT_DOWNLOAD_IP = "194.95.114.28"
DEFAULT_WORKERS = 3
DEFAULT_RATE_LIMIT = "12M"
MIN_FREE_GIB = 20
PRINT_LOCK = threading.Lock()


def prepare_directories() -> None:
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip"


def partial_archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip.part"


def expected_camera_paths(
    sequence: str,
    raw_dir: Path = RAW_DIR,
) -> list[Path]:
    """Return all high-resolution camera frames referenced by OpenLABEL."""
    root = raw_dir / sequence
    labels = root / f"{sequence}_labels.json"
    payload = json.loads(labels.read_text(encoding="utf-8-sig"))
    frames = payload["openlabel"]["frames"]
    if not isinstance(frames, dict):
        raise ValueError("openlabel.frames must be an object")

    root_resolved = root.resolve()
    paths: list[Path] = []
    for frame_id, frame in frames.items():
        try:
            uri = frame["frame_properties"]["streams"][CAMERA_FOLDER]["uri"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Frame {frame_id} has no {CAMERA_FOLDER} URI"
            ) from error
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"Frame {frame_id} has an invalid camera URI")
        relative = PurePosixPath(uri.lstrip("/"))
        if ".." in relative.parts:
            raise ValueError(f"Unsafe camera URI in frame {frame_id}: {uri}")
        target = (root / Path(*relative.parts)).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(f"Unsafe camera URI in frame {frame_id}: {uri}") from error
        paths.append(target)
    if not paths:
        raise ValueError("No camera frames are referenced by OpenLABEL")
    return paths


def missing_sequence_files(
    sequence: str,
    raw_dir: Path = RAW_DIR,
) -> list[Path]:
    try:
        expected = expected_camera_paths(sequence, raw_dir=raw_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return [raw_dir / sequence / f"{sequence}_labels.json"]
    return [path for path in expected if not path.is_file()]


def sequence_is_complete(sequence: str, raw_dir: Path = RAW_DIR) -> bool:
    return not missing_sequence_files(sequence, raw_dir=raw_dir)


def load_frame_exclusions(path: Path = RAW_EXCLUSIONS_PATH) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("excluded_relative_paths", [])
    if not isinstance(values, list):
        raise ValueError("excluded_relative_paths must be a list")
    exclusions: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every raw frame exclusion must be a non-empty string")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe raw frame exclusion: {value}")
        exclusions.add(relative.as_posix())
    return exclusions


def missing_outside_exclusions(
    sequence: str,
    raw_dir: Path = RAW_DIR,
    exclusions_path: Path = RAW_EXCLUSIONS_PATH,
) -> list[Path]:
    allowed = load_frame_exclusions(exclusions_path)
    raw_root = raw_dir.resolve()
    result: list[Path] = []
    for path in missing_sequence_files(sequence, raw_dir=raw_dir):
        try:
            relative = path.resolve().relative_to(raw_root).as_posix()
        except ValueError:
            result.append(path)
            continue
        if relative not in allowed:
            result.append(path)
    return result


def sequence_is_usable(
    sequence: str,
    raw_dir: Path = RAW_DIR,
    exclusions_path: Path = RAW_EXCLUSIONS_PATH,
) -> bool:
    return not missing_outside_exclusions(
        sequence,
        raw_dir=raw_dir,
        exclusions_path=exclusions_path,
    )


def relative_archive_path(name: str, sequence: str) -> PurePosixPath | None:
    member = PurePosixPath(name)
    if not member.parts:
        return None
    if sequence in member.parts:
        parts = member.parts[member.parts.index(sequence) + 1:]
        return PurePosixPath(*parts) if parts else None
    return member


def member_is_required(path: PurePosixPath, sequence: str) -> bool:
    if path.parts and path.parts[0] == CAMERA_FOLDER:
        return True
    return path.name.lower() in {
        f"{sequence.lower()}_labels.json", "readme.md", "license.md"
    }


def extract_required_files(zip_path: Path, sequence: str) -> None:
    output = RAW_DIR / sequence
    output.mkdir(parents=True, exist_ok=True)
    output_root = output.resolve()
    selected = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            relative = relative_archive_path(member.filename, sequence)
            if relative is None or not member_is_required(relative, sequence):
                continue
            target = output / Path(*relative.parts)
            resolved = target.resolve()
            try:
                resolved.relative_to(output_root)
            except ValueError as error:
                raise RuntimeError(f"Unsafe archive path: {member.filename}") from error
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            selected += 1
    if selected == 0:
        raise RuntimeError(f"No required members in {zip_path}")


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def download_sequence(
    sequence: str,
    ip_address: str,
    keep_archive: bool,
    rate_limit: str,
) -> str:
    if sequence_is_complete(sequence):
        if not keep_archive:
            archive_path(sequence).unlink(missing_ok=True)
            partial_archive_path(sequence).unlink(missing_ok=True)
        log(f"SKIP {sequence}: already extracted")
        return "skipped"

    free_bytes = shutil.disk_usage(ARCHIVES_DIR).free
    if free_bytes < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(
            f"Free space safety gate: {free_bytes / 1024**3:.1f} GiB < {MIN_FREE_GIB} GiB"
        )

    final_path = archive_path(sequence)
    partial_path = partial_archive_path(sequence)
    if final_path.is_file() and not zipfile.is_zipfile(final_path):
        raise RuntimeError(f"Invalid existing archive: {final_path}")

    if not final_path.is_file():
        url = f"{DOWNLOAD_BASE_URL}/{sequence}.zip"
        # The official server ignores Range and curl cannot resume safely.
        # Remove only this sequence's incomplete file before a fresh transfer.
        partial_path.unlink(missing_ok=True)
        log(f"GET  {sequence}")
        subprocess.run(
            [
                "curl", "--fail", "--location", "--retry", "5",
                "--retry-all-errors",
                "--limit-rate", rate_limit,
                "--resolve", f"download.data.fid-move.de:443:{ip_address}",
                url, "--output", str(partial_path),
            ],
            check=True,
        )
        partial_path.replace(final_path)

    if not zipfile.is_zipfile(final_path):
        raise RuntimeError(f"Downloaded file is not ZIP: {final_path}")

    log(f"EXTRACT {sequence}")
    extract_required_files(zip_path=final_path, sequence=sequence)
    if not sequence_is_complete(sequence):
        raise RuntimeError(f"Extraction completeness check failed: {sequence}")

    if not keep_archive:
        final_path.unlink()
    log(f"DONE {sequence}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct OSDaR23 downloader without browser/Anubis"
    )
    parser.add_argument("--download-ip", default=DEFAULT_DOWNLOAD_IP)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--rate-limit",
        default=DEFAULT_RATE_LIMIT,
        help="Per-worker curl transfer limit (default: 12M)",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--sequences", nargs="*", default=SEQUENCES)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate referenced camera frames without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    sequences = args.sequences[args.shard_index::args.shard_count]
    prepare_directories()
    if args.audit_only:
        failures = 0
        for sequence in sequences:
            missing = missing_sequence_files(sequence)
            if missing:
                failures += 1
                log(f"INCOMPLETE {sequence}: {len(missing)} missing or invalid")
                for path in missing[:5]:
                    log(f"  {path}")
            else:
                log(f"OK {sequence}")
        log(f"Audit: {len(sequences) - failures} complete, {failures} incomplete")
        if failures:
            raise SystemExit(1)
        return
    log(f"Raw output: {RAW_DIR}")
    log(
        f"Sequences: {len(sequences)}, shard: {args.shard_index}/{args.shard_count}, "
        f"workers: {args.workers}, "
        f"rate limit: {args.rate_limit}"
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_sequence,
                sequence,
                args.download_ip,
                args.keep_archives,
                args.rate_limit,
            ): sequence
            for sequence in sequences
        }
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as error:
                counts["failed"] += 1
                failures.append((sequence, f"{type(error).__name__}: {error}"))
                log(f"FAIL {sequence}: {error}")
    log(f"Summary: {counts}")
    if failures:
        for sequence, error in failures:
            log(f"  {sequence}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
