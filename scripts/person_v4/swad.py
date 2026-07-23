from __future__ import annotations

import hashlib

import torch


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating SWAD state and explicitly merge integer BN counters."""
    if not states:
        raise ValueError("SWAD requires at least one state dictionary")
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("SWAD state dictionaries have different keys")
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        first = values[0]
        if any(
            value.shape != first.shape or value.dtype != first.dtype
            for value in values[1:]
        ):
            raise ValueError(f"SWAD tensor contract differs for {key}")
        if torch.is_floating_point(first):
            total = torch.zeros_like(first, dtype=torch.float64)
            for value in values:
                total += value.to(dtype=torch.float64)
            averaged[key] = (total / len(values)).to(dtype=first.dtype)
        else:
            # BatchNorm num_batches_tracked is a non-floating state buffer.
            # Keep the latest/highest counter instead of averaging it as if it
            # were a learnable parameter.
            averaged[key] = torch.stack(values).amax(dim=0)
    return averaged


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor state independently of torch.save container metadata."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()
