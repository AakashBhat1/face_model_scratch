import numpy as np

from face_model_core.similarity import cosine_similarity, is_match, pairwise_cosine_similarity


def test_similarity_cosine_known_vectors() -> None:
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)

    assert cosine_similarity(v1, v2) > 0.999
    assert abs(cosine_similarity(v1, v3)) < 1e-6


def test_pairwise_similarity_shape_and_symmetry() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = pairwise_cosine_similarity(x, x)
    assert m.shape == (2, 2)
    assert np.allclose(m, m.T, atol=1e-6)
    assert np.allclose(np.diag(m), np.ones(2), atol=1e-6)


def test_is_match_threshold_logic() -> None:
    assert is_match(0.9, 0.4)
    assert not is_match(0.2, 0.4)
