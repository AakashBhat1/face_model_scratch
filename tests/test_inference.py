from pathlib import Path

import numpy as np
import torch
from PIL import Image

from face_model_core.inference import build_gallery, image_to_embedding, infer_with_gallery


class _DummyModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self._output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._output.repeat(x.shape[0], 1)


def test_image_to_embedding_returns_float32(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_path)

    output = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    def fake_build_model_from_checkpoint(_checkpoint_path: Path):
        return _DummyModel(output), {"image_size": 8}

    def fake_build_transforms(image_size: int):
        del image_size

        def val_t(_image):
            return torch.ones(3, 8, 8, dtype=torch.float32)

        return None, val_t

    monkeypatch.setattr("face_model_core.inference._build_model_from_checkpoint", fake_build_model_from_checkpoint)
    monkeypatch.setattr("face_model_core.inference.build_transforms", fake_build_transforms)
    monkeypatch.setattr("face_model_core.inference.get_device", lambda: torch.device("cpu"))

    emb = image_to_embedding(image_path=image_path, checkpoint_path=tmp_path / "model.pt")
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32
    assert emb.shape == (4,)
    assert np.allclose(emb, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), atol=1e-6)


def test_infer_with_gallery_match_and_no_match(monkeypatch, tmp_path: Path) -> None:
    gallery_path = tmp_path / "gallery.npz"
    identities = np.array(["alice", "bob"])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.savez_compressed(gallery_path, identities=identities, vectors=vectors)

    monkeypatch.setattr(
        "face_model_core.inference.image_to_embedding",
        lambda image_path, checkpoint_path: np.array([0.9, 0.1], dtype=np.float32),
    )

    result_match = infer_with_gallery(
        image_path=tmp_path / "query.jpg",
        checkpoint_path=tmp_path / "model.pt",
        gallery_path=gallery_path,
        threshold=0.8,
    )
    assert result_match["identity"] == "alice"
    assert isinstance(result_match["score"], float)
    assert result_match["match"] == "match"

    result_no_match = infer_with_gallery(
        image_path=tmp_path / "query.jpg",
        checkpoint_path=tmp_path / "model.pt",
        gallery_path=gallery_path,
        threshold=0.999,
    )
    assert result_no_match["identity"] == "alice"
    assert result_no_match["match"] == "no_match"


def test_infer_with_gallery_rejects_invalid_threshold(monkeypatch, tmp_path: Path) -> None:
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        identities=np.array(["alice"]),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "face_model_core.inference.image_to_embedding",
        lambda image_path, checkpoint_path: np.array([1.0, 0.0], dtype=np.float32),
    )

    try:
        infer_with_gallery(
            image_path=tmp_path / "query.jpg",
            checkpoint_path=tmp_path / "model.pt",
            gallery_path=gallery_path,
            threshold=1.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for out-of-range threshold")


def test_infer_with_gallery_rejects_non_finite_threshold(monkeypatch, tmp_path: Path) -> None:
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        identities=np.array(["alice"]),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "face_model_core.inference.image_to_embedding",
        lambda image_path, checkpoint_path: np.array([1.0, 0.0], dtype=np.float32),
    )

    try:
        infer_with_gallery(
            image_path=tmp_path / "query.jpg",
            checkpoint_path=tmp_path / "model.pt",
            gallery_path=gallery_path,
            threshold=float("nan"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-finite threshold")


def test_build_gallery_loads_model_once(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "gallery"
    for identity in ["alice", "bob"]:
        identity_dir = root / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            Image.new("RGB", (8, 8), color=(50 + idx, 80, 120)).save(identity_dir / f"{idx}.jpg")

    calls = {"count": 0}

    def fake_build_model_from_checkpoint(_checkpoint_path: Path):
        calls["count"] += 1
        return _DummyModel(torch.tensor([[0.3, 0.7]], dtype=torch.float32)), {"image_size": 8}

    def fake_build_transforms(image_size: int):
        del image_size

        def val_t(_image):
            return torch.ones(3, 8, 8, dtype=torch.float32)

        return None, val_t

    monkeypatch.setattr("face_model_core.inference._build_model_from_checkpoint", fake_build_model_from_checkpoint)
    monkeypatch.setattr("face_model_core.inference.build_transforms", fake_build_transforms)
    monkeypatch.setattr("face_model_core.inference.get_device", lambda: torch.device("cpu"))

    output_path = tmp_path / "gallery.npz"
    build_gallery(gallery_root=root, checkpoint_path=tmp_path / "model.pt", output_path=output_path)

    assert calls["count"] == 1
    assert output_path.exists()


def test_infer_with_gallery_rejects_empty_gallery_vectors(monkeypatch, tmp_path: Path) -> None:
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        identities=np.array([], dtype="<U1"),
        vectors=np.empty((0, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "face_model_core.inference.image_to_embedding",
        lambda image_path, checkpoint_path: np.array([1.0, 0.0], dtype=np.float32),
    )

    try:
        infer_with_gallery(
            image_path=tmp_path / "query.jpg",
            checkpoint_path=tmp_path / "model.pt",
            gallery_path=gallery_path,
            threshold=0.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty gallery vectors")


def test_infer_with_gallery_rejects_mismatched_gallery_shapes(monkeypatch, tmp_path: Path) -> None:
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        identities=np.array(["alice", "bob"]),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "face_model_core.inference.image_to_embedding",
        lambda image_path, checkpoint_path: np.array([1.0, 0.0], dtype=np.float32),
    )

    try:
        infer_with_gallery(
            image_path=tmp_path / "query.jpg",
            checkpoint_path=tmp_path / "model.pt",
            gallery_path=gallery_path,
            threshold=0.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched gallery arrays")


def test_build_gallery_raises_for_empty_gallery(tmp_path: Path) -> None:
    empty_root = tmp_path / "gallery"
    empty_root.mkdir()
    output = tmp_path / "gallery.npz"

    try:
        build_gallery(gallery_root=empty_root, checkpoint_path=tmp_path / "model.pt", output_path=output)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty gallery")

    assert not output.exists()
