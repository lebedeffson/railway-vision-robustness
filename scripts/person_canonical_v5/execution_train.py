from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.person_canonical_v5.common import atomic_json, now, sha256
from scripts.person_canonical_v5.execution_common import assert_execution_locked
from scripts.person_canonical_v5.train_candidates import (
    CANDIDATE_ROOT,
    data_config,
    environment,
    model_config,
    runtime,
    train_stage,
)


def train(candidate: str, fold: int) -> dict[str, Any]:
    assert_execution_locked()
    if fold not in range(5):
        raise RuntimeError("Person-v5 execution only permits folds 0..4")
    if not torch.cuda.is_available():
        raise RuntimeError("Person-v5 execution requires CUDA")
    data = data_config(candidate, fold)
    if not data.is_file():
        raise RuntimeError(f"Candidate dataset is not ready: {data}")
    root = CANDIDATE_ROOT / candidate / f"fold_{fold}"
    completion = root / "TRAINING_COMPLETE.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        checkpoint = Path(payload["checkpoint"])
        if checkpoint.is_file() and sha256(checkpoint) == payload["checkpoint_sha256"]:
            return payload
        raise RuntimeError(f"Completed training artifact changed: {root}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environment.json", environment())
    atomic_json(root / "resolved_config.json", {
        "candidate": candidate,
        "fold": fold,
        "execution_runtime": "person-canonical-v5-candidate-runtime-v1c",
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "model": str(model_config(candidate).resolve()),
        "model_sha256": sha256(model_config(candidate)),
        "seed": runtime()["training"]["seed"],
        "test_used": False,
    })
    checkpoint: Path | None = None
    for stage_name in ("stage1", "stage2"):
        checkpoint = train_stage(
            candidate=candidate,
            fold=fold,
            stage_name=stage_name,
            initialization=checkpoint,
            data=data,
        )
    assert checkpoint is not None
    payload = {
        "status": "RUNTIME_PASS",
        "candidate": candidate,
        "fold": fold,
        "finished_at": now(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "best_checkpoint": str(checkpoint.resolve()),
        "last_checkpoint": str((root / "stage2/weights/last.pt").resolve()),
        "test_used": False,
    }
    atomic_json(completion, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.candidate, args.fold), indent=2))


if __name__ == "__main__":
    main()
