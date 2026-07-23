from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "outputs/canonical_v2/tests"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [str(PROJECT_DIR / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=PROJECT_DIR, text=True, capture_output=True)
    transcript = result.stdout + result.stderr
    (OUTPUT / "unittest.txt").write_text(transcript, encoding="utf-8")
    payload = {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "returncode": result.returncode,
    }
    (OUTPUT / "test_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(transcript)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
