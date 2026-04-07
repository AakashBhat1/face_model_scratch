from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def _to_checkpoint_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_checkpoint_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_checkpoint_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_checkpoint_safe(v) for v in value)
    return value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_metric: float,
    config_dict: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler | None = None,
    head: nn.Module | None = None,
    class_names: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": _to_checkpoint_safe(config_dict),
        "class_names": class_names or [],
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if head is not None:
        payload["head_state"] = head.state_dict()
    # Atomic save: write to temp file then rename so a crash mid-save
    # never leaves a corrupted checkpoint.
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        if weights_only:
            raise RuntimeError("weights_only loading is not supported by this PyTorch version")
        return torch.load(path, map_location=map_location)
    except RuntimeError as exc:
        if "zip archive" in str(exc) or "central directory" in str(exc):
            raise RuntimeError(
                f"Checkpoint file is corrupted (likely truncated by a disconnection): {path}\n"
                f"Delete it and restart training from scratch, or use a different checkpoint.\n"
                f"Original error: {exc}"
            ) from exc
        raise
