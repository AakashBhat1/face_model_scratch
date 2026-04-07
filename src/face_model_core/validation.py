from __future__ import annotations

from collections import defaultdict
from random import Random

import numpy as np
import torch
from torch.utils.data import DataLoader

from face_model_core.similarity import cosine_similarity


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    seen = 0
    for images, labels in loader:
        if seen >= max_images:
            break
        images = images.to(device, non_blocking=True)
        embeddings = model(images).cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(labels.numpy())
        seen += images.shape[0]

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def quick_similarity_eval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, float]:
    if embeddings.shape[0] < 4:
        return {
            "same_mean": 0.0,
            "diff_mean": 0.0,
            "pair_acc": 0.0,
            "num_pairs": 0.0,
        }

    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        indices_by_label[int(label)].append(idx)

    same_pairs: list[tuple[int, int]] = []
    diff_pairs: list[tuple[int, int]] = []
    rng = Random(seed)

    labels_with_pairs = [k for k, v in indices_by_label.items() if len(v) >= 2]
    all_labels = list(indices_by_label.keys())
    for label in labels_with_pairs:
        picks = indices_by_label[label]
        i, j = rng.sample(picks, 2)
        same_pairs.append((i, j))

        other_labels = [x for x in all_labels if x != label]
        if other_labels:
            other_label = rng.choice(other_labels)
            k = rng.choice(indices_by_label[other_label])
            diff_pairs.append((i, k))

    if not same_pairs or not diff_pairs:
        return {
            "same_mean": 0.0,
            "diff_mean": 0.0,
            "pair_acc": 0.0,
            "num_pairs": 0.0,
        }

    same_scores = [cosine_similarity(embeddings[i], embeddings[j]) for i, j in same_pairs]
    diff_scores = [cosine_similarity(embeddings[i], embeddings[j]) for i, j in diff_pairs]

    correct = sum(score >= threshold for score in same_scores)
    correct += sum(score < threshold for score in diff_scores)
    total = len(same_scores) + len(diff_scores)

    return {
        "same_mean": float(np.mean(same_scores)),
        "diff_mean": float(np.mean(diff_scores)),
        "pair_acc": float(correct / max(total, 1)),
        "num_pairs": float(total),
    }
