from pathlib import Path

import pytest

from face_model_core.config import TrainConfig


def test_config_defaults_load() -> None:
    cfg = TrainConfig(data_root=Path("data"))
    assert cfg.embedding_dim in {128, 512}
    assert cfg.learning_rate > 0
    assert cfg.resume_from is None


def test_config_accepts_resume_path() -> None:
    cfg = TrainConfig(data_root=Path("data"), resume_from=Path("checkpoints/last.pt"))
    assert cfg.resume_from == Path("checkpoints/last.pt")


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), embedding_dim=42)

    with pytest.raises(ValueError):
        TrainConfig(data_root=Path("data"), arcface_margin=-0.1)
