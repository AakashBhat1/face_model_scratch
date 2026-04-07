from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from face_model_core.config import TrainConfig
from face_model_core.train import train_model


class _DummyModel(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3 * 8 * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.view(x.shape[0], -1))
        return torch.nn.functional.normalize(out, dim=1)


class _DummyHead(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings @ torch.nn.functional.normalize(self.weight, dim=1).t()


def _build_resume_checkpoint(epoch: int, best_metric: float, embedding_dim: int = 128) -> dict:
    model = _DummyModel(embedding_dim=embedding_dim)
    head = _DummyHead(embedding_dim=embedding_dim, num_classes=2)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    return {
        "model_state": model.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }


def test_train_model_skips_when_resume_already_complete(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"placeholder")

    monkeypatch.setattr("face_model_core.train.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "face_model_core.train.FaceEmbeddingModel",
        lambda backbone, embedding_dim: _DummyModel(embedding_dim=embedding_dim),
    )
    monkeypatch.setattr("face_model_core.train.ArcFaceHead", _DummyHead)
    def _fail_create_dataloaders(*args, **kwargs):
        raise AssertionError("create_dataloaders should not be called when resume already completed")

    monkeypatch.setattr("face_model_core.train.create_dataloaders", _fail_create_dataloaders)

    load_calls = {"weights_only": None}

    def _load_checkpoint(path, map_location, weights_only=False):
        load_calls["weights_only"] = weights_only
        return _build_resume_checkpoint(3, 0.7, embedding_dim=128)

    monkeypatch.setattr("face_model_core.train.load_checkpoint", _load_checkpoint)

    save_calls = {"count": 0}

    def _save_checkpoint(*args, **kwargs):
        save_calls["count"] += 1

    monkeypatch.setattr("face_model_core.train.save_checkpoint", _save_checkpoint)

    config = TrainConfig(
        data_root=tmp_path,
        resume_from=checkpoint_path,
        backbone="resnet50",
        embedding_dim=128,
        loss_type="arcface",
        image_size=8,
        batch_size=2,
        epochs=3,
        num_workers=0,
        mixed_precision=False,
        checkpoint_dir=tmp_path / "ckpts",
    )

    returned = train_model(config)
    assert returned == checkpoint_path
    assert save_calls["count"] == 0
    assert load_calls["weights_only"] is False


def test_train_model_resumes_and_runs_next_epoch(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"placeholder")

    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1], dtype=torch.long)

    monkeypatch.setattr("face_model_core.train.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "face_model_core.train.FaceEmbeddingModel",
        lambda backbone, embedding_dim: _DummyModel(embedding_dim=embedding_dim),
    )
    monkeypatch.setattr("face_model_core.train.ArcFaceHead", _DummyHead)
    monkeypatch.setattr(
        "face_model_core.train.create_dataloaders",
        lambda data_root, image_size, batch_size, num_workers: ([(images, labels)], [(images, labels)], 2, ["a", "b"]),
    )
    load_calls = {"weights_only": None}

    def _load_checkpoint(path, map_location, weights_only=False):
        load_calls["weights_only"] = weights_only
        return _build_resume_checkpoint(0, 0.1, embedding_dim=128)

    monkeypatch.setattr("face_model_core.train.load_checkpoint", _load_checkpoint)
    monkeypatch.setattr(
        "face_model_core.train.collect_embeddings",
        lambda model, loader, device, max_images: (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        "face_model_core.train.quick_similarity_eval",
        lambda embeddings, labels, threshold, seed: {
            "same_mean": 0.9,
            "diff_mean": 0.1,
            "pair_acc": 0.5,
            "num_pairs": 2.0,
        },
    )

    config = TrainConfig(
        data_root=tmp_path,
        resume_from=checkpoint_path,
        backbone="resnet50",
        embedding_dim=128,
        loss_type="arcface",
        image_size=8,
        batch_size=2,
        epochs=1,
        num_workers=0,
        mixed_precision=False,
        checkpoint_dir=tmp_path / "ckpts",
    )

    returned = train_model(config)
    assert returned == config.checkpoint_dir / "best.pt"
    assert (config.checkpoint_dir / "best.pt").exists()
    assert (config.checkpoint_dir / "last.pt").exists()
    assert load_calls["weights_only"] is False


def test_train_model_raises_when_resume_file_missing(tmp_path: Path) -> None:
    config = TrainConfig(
        data_root=tmp_path,
        resume_from=tmp_path / "does_not_exist.pt",
        checkpoint_dir=tmp_path / "ckpts",
    )

    try:
        train_model(config)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing resume checkpoint")
