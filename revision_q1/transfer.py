from __future__ import annotations

from pathlib import Path


def validate_transfer_pair(source: str | Path, target: str | Path) -> tuple[Path, Path]:
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if source_path == target_path:
        raise ValueError("Transfer source and target checkpoints must differ")
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Transfer checkpoint is missing")
    return source_path, target_path
