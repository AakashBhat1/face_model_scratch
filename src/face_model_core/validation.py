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
    _zero_metrics: dict[str, float] = {
        "same_mean": 0.0,
        "diff_mean": 0.0,
        "pair_acc": 0.0,
        "num_pairs": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "far": 0.0,
        "frr": 0.0,
        "eer": 0.0,
        "auc_roc": 0.0,
        "optimal_threshold": 0.0,
    }

    if embeddings.shape[0] < 4:
        return dict(_zero_metrics)

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
        return dict(_zero_metrics)

    same_scores = [cosine_similarity(embeddings[i], embeddings[j]) for i, j in same_pairs]
    diff_scores = [cosine_similarity(embeddings[i], embeddings[j]) for i, j in diff_pairs]

    same_arr = np.array(same_scores, dtype=np.float64)
    diff_arr = np.array(diff_scores, dtype=np.float64)

    # Basic accuracy at the given threshold.
    tp = int(np.sum(same_arr >= threshold))
    fn = int(np.sum(same_arr < threshold))
    fp = int(np.sum(diff_arr >= threshold))
    tn = int(np.sum(diff_arr < threshold))
    total = tp + fn + fp + tn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    far = fp / max(fp + tn, 1)   # False Accept Rate
    frr = fn / max(fn + tp, 1)   # False Reject Rate

    # AUC-ROC and EER via threshold sweep.
    all_scores = np.concatenate([same_arr, diff_arr])
    thresholds = np.sort(np.unique(all_scores))
    # Add boundary values for complete ROC curve.
    thresholds = np.concatenate([[all_scores.min() - 0.01], thresholds, [all_scores.max() + 0.01]])

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    far_list: list[float] = []
    frr_list: list[float] = []

    for t in thresholds:
        t_tp = float(np.sum(same_arr >= t))
        t_fn = float(np.sum(same_arr < t))
        t_fp = float(np.sum(diff_arr >= t))
        t_tn = float(np.sum(diff_arr < t))

        tpr_list.append(t_tp / max(t_tp + t_fn, 1))
        fpr_list.append(t_fp / max(t_fp + t_tn, 1))
        far_list.append(t_fp / max(t_fp + t_tn, 1))
        frr_list.append(t_fn / max(t_fn + t_tp, 1))

    # AUC via trapezoidal rule (sort by FPR ascending).
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_idx = np.argsort(fpr_arr)
    auc_roc = float(np.trapezoid(tpr_arr[sorted_idx], fpr_arr[sorted_idx]))

    # EER: where FAR and FRR cross.
    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)

    # Optimal threshold: maximizes (TPR - FPR).
    optimal_idx = int(np.argmax(tpr_arr - fpr_arr))
    optimal_threshold = float(thresholds[optimal_idx])

    return {
        "same_mean": float(np.mean(same_scores)),
        "diff_mean": float(np.mean(diff_scores)),
        "pair_acc": float((tp + tn) / max(total, 1)),
        "num_pairs": float(total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "eer": float(eer),
        "auc_roc": float(auc_roc),
        "optimal_threshold": float(optimal_threshold),
    }
