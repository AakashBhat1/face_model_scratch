from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_BACKBONES = {"resnet50", "mobilenet_v2"}
VALID_EMBED_DIMS = {128, 512}
VALID_LOSS_TYPES = {"arcface", "triplet"}


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    backbone: str = "resnet50"
    embedding_dim: int = 512
    loss_type: str = "arcface"
    image_size: int = 112
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    mixed_precision: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    val_max_images: int = 1200
    val_threshold: float = 0.4
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    triplet_margin: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backbone not in VALID_BACKBONES:
            raise ValueError(f"Invalid backbone: {self.backbone}")
        if self.embedding_dim not in VALID_EMBED_DIMS:
            raise ValueError(f"Invalid embedding_dim: {self.embedding_dim}")
        if self.loss_type not in VALID_LOSS_TYPES:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.triplet_margin < 0:
            raise ValueError("triplet_margin must be non-negative")
        if self.arcface_margin <= 0:
            raise ValueError("arcface_margin must be positive")
        if self.arcface_scale <= 0:
            raise ValueError("arcface_scale must be positive")
