from pathlib import Path

import pytest

from face_model_core.config import TrainConfig


def test_config_defaults_load() -> None:
    cfg = TrainConfig(data_root=Path("data"))
    assert cfg.embedding_dim in {128, 512}
    assert cfg.learning_rate > 0
    assert cfg.backbone_lr > 0
    assert cfg.freeze_backbone_epochs >= 0
    assert cfg.grad_clip_norm > 0
    assert cfg.resume_from is None


def test_config_accepts_resume_path() -> None:
    cfg = TrainConfig(data_root=Path("data"), resume_from=Path("checkpoints/last.pt"))
    assert cfg.resume_from == Path("checkpoints/last.pt")


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), embedding_dim=42)

    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), arcface_margin=-0.1)

    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), backbone_lr=0)

    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), freeze_backbone_epochs=-1)

    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), grad_clip_norm=0)
