from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def pairwise_cosine_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    a = matrix_a / (np.linalg.norm(matrix_a, axis=1, keepdims=True) + 1e-12)
    b = matrix_b / (np.linalg.norm(matrix_b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def is_match(score: float, threshold: float) -> bool:
    return score >= threshold
