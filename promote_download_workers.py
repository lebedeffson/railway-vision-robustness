from __future__ import annotations

import argparse
import subprocess
import time

from download_osdar23_direct import sequence_is_complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart downloader after a safe sequence boundary")
    parser.add_argument("sequence")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    while not sequence_is_complete(args.sequence):
        time.sleep(args.poll_seconds)
    subprocess.run(
        ["systemctl", "--user", "restart", "tnorm-osdar-download.service"],
        check=True,
    )


if __name__ == "__main__":
    main()
