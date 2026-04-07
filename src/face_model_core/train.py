from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from face_model_core.checkpoint import save_checkpoint
from face_model_core.config import TrainConfig
from face_model_core.data import create_dataloaders
from face_model_core.losses import ArcFaceLoss, BatchTripletLoss
from face_model_core.model import ArcFaceHead, FaceEmbeddingModel
from face_model_core.utils import get_device, set_seed
from face_model_core.validation import collect_embeddings, quick_similarity_eval


def train_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = get_device()
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        data_root=config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = FaceEmbeddingModel(backbone=config.backbone, embedding_dim=config.embedding_dim).to(device)
    head: ArcFaceHead | None = None

    if config.loss_type == "arcface":
        head = ArcFaceHead(embedding_dim=config.embedding_dim, num_classes=num_classes).to(device)
        criterion: nn.Module = ArcFaceLoss(margin=config.arcface_margin, scale=config.arcface_scale)
        params = list(model.parameters()) + list(head.parameters())
    else:
        criterion = BatchTripletLoss(margin=config.triplet_margin)
        params = list(model.parameters())

    optimizer = AdamW(params=params, lr=config.learning_rate, weight_decay=config.weight_decay)
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    best_metric = -1.0
    best_path = config.checkpoint_dir / "best.pt"
    last_path = config.checkpoint_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                embeddings = model(images)
                if config.loss_type == "arcface":
                    assert head is not None
                    logits = head(embeddings)
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().item())

        val_embeddings, val_labels = collect_embeddings(
            model=model,
            loader=val_loader,
            device=device,
            max_images=config.val_max_images,
        )
        metrics = quick_similarity_eval(
            embeddings=val_embeddings,
            labels=val_labels,
            threshold=config.val_threshold,
            seed=config.seed + epoch,
        )

        train_loss = epoch_loss / max(len(train_loader), 1)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"same_mean={metrics['same_mean']:.4f} diff_mean={metrics['diff_mean']:.4f} "
            f"pair_acc={metrics['pair_acc']:.4f}"
        )

        if metrics["pair_acc"] > best_metric:
            best_metric = metrics["pair_acc"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                config_dict=asdict(config),
                scaler=scaler,
                head=head,
                class_names=class_names,
            )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            config_dict=asdict(config),
            scaler=scaler,
            head=head,
            class_names=class_names,
        )

    return best_path if best_path.exists() else last_path
