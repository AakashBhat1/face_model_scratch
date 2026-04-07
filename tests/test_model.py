import torch

from face_model_core.model import FaceEmbeddingModel


def test_model_forward_output_shape_and_unit_norm() -> None:
    model = FaceEmbeddingModel(backbone="mobilenet_v2", embedding_dim=128)
    x = torch.randn(4, 3, 112, 112)
    y = model(x)
    assert y.shape == (4, 128)
    norms = torch.linalg.norm(y, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
