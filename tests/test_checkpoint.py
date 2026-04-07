from pathlib import Path

import torch

from face_model_core.checkpoint import load_checkpoint, save_checkpoint
from face_model_core.model import FaceEmbeddingModel


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = FaceEmbeddingModel(backbone="mobilenet_v2", embedding_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_metric=0.75,
        config_dict={"backbone": "mobilenet_v2", "embedding_dim": 128, "image_size": 112},
        scaler=None,
        head=None,
        class_names=["a", "b"],
    )
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 3
    assert abs(loaded["best_metric"] - 0.75) < 1e-8
    assert loaded["config"]["backbone"] == "mobilenet_v2"


def test_checkpoint_weights_only_load_supports_path_values(tmp_path: Path) -> None:
    model = FaceEmbeddingModel(backbone="mobilenet_v2", embedding_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt_weights_only.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_metric=0.1,
        config_dict={"data_root": Path("data"), "checkpoint_dir": Path("checkpoints")},
        scaler=None,
        head=None,
        class_names=None,
    )

    loaded = load_checkpoint(path, weights_only=True)
    assert loaded["config"]["data_root"] == "data"
    assert loaded["config"]["checkpoint_dir"] == "checkpoints"
