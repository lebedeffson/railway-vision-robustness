from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from download_osdar23_direct import (
    CAMERA_FOLDER,
    DEFAULT_DOWNLOAD_IP,
    DEFAULT_RATE_LIMIT,
    DOWNLOAD_BASE_URL,
    RAW_DIR,
    missing_sequence_files,
    partial_archive_path,
)


def curl_command(url: str, download_ip: str | None, rate_limit: str | None) -> list[str]:
    command = ["curl", "--fail", "--location", "--show-error"]
    if rate_limit:
        command.extend(["--limit-rate", rate_limit])
    if download_ip and url.startswith("https://download.data.fid-move.de/"):
        command.extend([
            "--resolve", f"download.data.fid-move.de:443:{download_ip}"
        ])
    command.append(url)
    return command


def stream_extract(
    sequence: str,
    url: str,
    raw_dir: Path = RAW_DIR,
    download_ip: str | None = DEFAULT_DOWNLOAD_IP,
    rate_limit: str | None = DEFAULT_RATE_LIMIT,
) -> tuple[int, int, list[Path]]:
    """Extract useful ZIP members while downloading, retaining partial progress."""
    output = raw_dir / sequence
    output.mkdir(parents=True, exist_ok=True)
    curl = subprocess.Popen(
        curl_command(url, download_ip, rate_limit),
        stdout=subprocess.PIPE,
    )
    assert curl.stdout is not None
    archive = subprocess.Popen(
        [
            "bsdtar", "-xvf", "-", "-C", str(output), "--no-same-owner",
            f"{CAMERA_FOLDER}/*", f"{sequence}_labels.json",
        ],
        stdin=curl.stdout,
    )
    curl.stdout.close()
    archive_status = archive.wait()
    curl_status = curl.wait()
    missing = missing_sequence_files(sequence, raw_dir=raw_dir)
    return curl_status, archive_status, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retain selected OSDaR23 files from an unreliable ZIP stream"
    )
    parser.add_argument("sequence")
    parser.add_argument("--download-ip", default=DEFAULT_DOWNLOAD_IP)
    parser.add_argument("--rate-limit", default=DEFAULT_RATE_LIMIT)
    parser.add_argument("--url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_before = missing_sequence_files(args.sequence)
    if not missing_before:
        print(f"SKIP {args.sequence}: already complete", flush=True)
        return
    # A non-resumable archive cannot help the streaming recovery and only costs disk.
    partial_archive_path(args.sequence).unlink(missing_ok=True)
    url = args.url or f"{DOWNLOAD_BASE_URL}/{args.sequence}.zip"
    print(
        f"STREAM {args.sequence}: {len(missing_before)} referenced frames missing",
        flush=True,
    )
    curl_status, archive_status, missing = stream_extract(
        args.sequence,
        url,
        download_ip=args.download_ip,
        rate_limit=args.rate_limit,
    )
    if not missing:
        print(
            f"DONE {args.sequence}: all referenced frames extracted "
            f"(curl={curl_status}, bsdtar={archive_status})",
            flush=True,
        )
        return
    print(
        f"RETRY {args.sequence}: {len(missing)} frames still missing "
        f"(curl={curl_status}, bsdtar={archive_status})",
        flush=True,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
