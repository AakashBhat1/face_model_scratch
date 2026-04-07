from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from face_model_core.checkpoint import load_checkpoint
from face_model_core.data import build_transforms
from face_model_core.model import FaceEmbeddingModel
from face_model_core.similarity import pairwise_cosine_similarity
from face_model_core.utils import get_device


def _build_model_from_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = FaceEmbeddingModel(backbone=cfg["backbone"], embedding_dim=cfg["embedding_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def _image_to_embedding_with_runtime(
    image_path: Path,
    model: nn.Module,
    val_transform,
    device: torch.device,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(device)
    emb = model(tensor).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


@torch.no_grad()
def image_to_embedding(image_path: Path, checkpoint_path: Path) -> np.ndarray:
    model, cfg = _build_model_from_checkpoint(checkpoint_path)
    _, val_t = build_transforms(image_size=int(cfg["image_size"]))
    return _image_to_embedding_with_runtime(
        image_path=image_path,
        model=model,
        val_transform=val_t,
        device=get_device(),
    )


def build_gallery(gallery_root: Path, checkpoint_path: Path, output_path: Path) -> None:
    grouped_image_paths: list[tuple[str, list[Path]]] = []

    for identity_dir in sorted(gallery_root.iterdir()):
        if not identity_dir.is_dir():
            continue
        image_paths: list[Path] = []
        for image_path in identity_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            image_paths.append(image_path)
        if image_paths:
            grouped_image_paths.append((identity_dir.name, image_paths))

    if not grouped_image_paths:
        raise ValueError(f"No valid gallery images found in: {gallery_root}")

    model, cfg = _build_model_from_checkpoint(checkpoint_path)
    _, val_t = build_transforms(image_size=int(cfg["image_size"]))
    device = get_device()

    identities: list[str] = []
    vectors: list[np.ndarray] = []
    for identity, image_paths in grouped_image_paths:
        identity_embeddings = [
            _image_to_embedding_with_runtime(
                image_path=image_path,
                model=model,
                val_transform=val_t,
                device=device,
            )
            for image_path in image_paths
        ]
        identities.append(identity)
        vectors.append(np.mean(identity_embeddings, axis=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, identities=np.array(identities), vectors=np.stack(vectors, axis=0))


def infer_with_gallery(
    image_path: Path,
    checkpoint_path: Path,
    gallery_path: Path,
    threshold: float,
) -> dict[str, str | float]:
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [-1.0, 1.0], got {threshold}")

    query = image_to_embedding(image_path=image_path, checkpoint_path=checkpoint_path)
    with np.load(gallery_path, allow_pickle=False) as gallery:
        if "identities" not in gallery or "vectors" not in gallery:
            raise ValueError("gallery file must contain 'identities' and 'vectors' arrays")
        identities = gallery["identities"]
        vectors = gallery["vectors"]

    if identities.ndim != 1:
        raise ValueError("gallery identities must be a 1D array")
    if vectors.ndim != 2:
        raise ValueError("gallery vectors must be a 2D array")
    if vectors.shape[0] == 0:
        raise ValueError("gallery vectors must contain at least one identity")
    if identities.shape[0] != vectors.shape[0]:
        raise ValueError("gallery identities and vectors must have the same first dimension")
    if query.shape[0] != vectors.shape[1]:
        raise ValueError("query embedding dimension must match gallery vector dimension")

    scores = pairwise_cosine_similarity(query[None, :], vectors)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return {
        "identity": str(identities[best_idx]),
        "score": best_score,
        "match": "match" if best_score >= threshold else "no_match",
    }
