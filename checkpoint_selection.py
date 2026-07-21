from __future__ import annotations

import json
import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config/checkpoint_selection.json"


def load_selection(path: Path = CONFIG_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_checkpoint")
    if not isinstance(selected, str) or not selected.strip():
        raise RuntimeError(f"Invalid selected_checkpoint in {path}")
    checkpoint = (PROJECT_DIR / selected).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload["resolved_checkpoint"] = str(checkpoint)
    return payload


def selected_checkpoint(path: Path = CONFIG_PATH) -> Path:
    return Path(str(load_selection(path)["resolved_checkpoint"]))


def export_selection(output: Path) -> None:
    payload = load_selection()
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("checkpoint selection requires non-empty candidates")
    output.mkdir(parents=True, exist_ok=True)
    fields = list(candidates[0])
    with (output / "checkpoint_selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    (output / "checkpoint_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    export_selection(PROJECT_DIR / "outputs/final_practice/audit")
