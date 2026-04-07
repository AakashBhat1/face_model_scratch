import numpy as np

from face_model_core.validation import quick_similarity_eval


def test_quick_similarity_eval_separates_same_and_diff() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    metrics = quick_similarity_eval(embeddings=embeddings, labels=labels, threshold=0.7, seed=123)
    assert metrics["same_mean"] > metrics["diff_mean"]
    assert metrics["pair_acc"] == 1.0
    assert metrics["num_pairs"] > 0.0


def test_quick_similarity_eval_small_batch_returns_zeros() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1], dtype=np.int64)

    metrics = quick_similarity_eval(embeddings=embeddings, labels=labels, threshold=0.7, seed=123)
    assert metrics["same_mean"] == 0.0
    assert metrics["diff_mean"] == 0.0
    assert metrics["pair_acc"] == 0.0
    assert metrics["num_pairs"] == 0.0
