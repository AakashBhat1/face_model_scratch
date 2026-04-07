import torch

from face_model_core.losses import ArcFaceLoss, BatchTripletLoss


def test_arcface_loss_returns_finite_scalar() -> None:
    logits = torch.tensor(
        [[0.8, 0.1, -0.2], [0.2, 0.7, -0.1], [0.0, -0.2, 0.9]], dtype=torch.float32
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = ArcFaceLoss(margin=0.3, scale=32.0)(logits, labels)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_triplet_loss_margin_behavior() -> None:
    embeddings_good = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ],
        dtype=torch.float32,
    )
    embeddings_good = torch.nn.functional.normalize(embeddings_good, dim=1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss_good = BatchTripletLoss(margin=0.2)(embeddings_good, labels)
    assert loss_good >= 0

    embeddings_bad = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    embeddings_bad = torch.nn.functional.normalize(embeddings_bad, dim=1)
    bad_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss_bad = BatchTripletLoss(margin=0.2)(embeddings_bad, bad_labels)
    assert loss_bad > 0


def test_triplet_loss_supports_backward_when_no_valid_triplets() -> None:
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    embeddings = torch.nn.functional.normalize(x, dim=1)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    loss = BatchTripletLoss(margin=0.2)(embeddings, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None
