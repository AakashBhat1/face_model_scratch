from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine_logits = cosine_logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine_logits)
        target_cosine = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine_logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = one_hot * target_cosine + (1.0 - one_hot) * cosine_logits
        output = output * self.scale
        return F.cross_entropy(output, labels)


class BatchTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pairwise = 1.0 - embeddings @ embeddings.t()

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_neq = ~label_eq

        eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        positive_mask = label_eq & ~eye
        negative_mask = label_neq

        hardest_positive = torch.where(positive_mask, pairwise, torch.full_like(pairwise, -1.0)).max(dim=1).values
        hardest_negative = torch.where(negative_mask, pairwise, torch.full_like(pairwise, 1e6)).min(dim=1).values

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not torch.any(valid):
            return embeddings.sum() * 0.0

        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        return losses[valid].mean()
