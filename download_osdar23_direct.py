from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVES_DIR = PROJECT_DIR / "data/archives"
RAW_DIR = PROJECT_DIR / "data/raw"
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
DEFAULT_WORKERS = 4
MIN_FREE_GIB = 20
PRINT_LOCK = threading.Lock()


def prepare_directories() -> None:
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip"


def partial_archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip.part"


def sequence_is_complete(sequence: str) -> bool:
    root = RAW_DIR / sequence
    labels = root / f"{sequence}_labels.json"
    camera = root / CAMERA_FOLDER
    return labels.is_file() and camera.is_dir() and any(camera.iterdir())


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


def download_sequence(sequence: str, ip_address: str, keep_archive: bool) -> str:
    if sequence_is_complete(sequence):
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
        log(f"GET  {sequence}")
        subprocess.run(
            [
                "curl", "--fail", "--location", "--retry", "5",
                "--retry-all-errors", "--continue-at", "-",
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
        description="Direct resumable OSDaR23 downloader without browser/Anubis"
    )
    parser.add_argument("--download-ip", default=DEFAULT_DOWNLOAD_IP)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--sequences", nargs="*", default=SEQUENCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    prepare_directories()
    log(f"Raw output: {RAW_DIR}")
    log(f"Sequences: {len(args.sequences)}, workers: {args.workers}")
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_sequence, sequence, args.download_ip, args.keep_archives
            ): sequence
            for sequence in args.sequences
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
