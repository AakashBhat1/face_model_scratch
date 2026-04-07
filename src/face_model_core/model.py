from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2, resnet50


class FaceEmbeddingModel(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int) -> None:
        super().__init__()
        if backbone == "resnet50":
            base = resnet50(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "mobilenet_v2":
            base = mobilenet_v2(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return embeddings @ weight_norm.t()
