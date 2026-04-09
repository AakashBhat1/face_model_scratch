from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from face_model_core.checkpoint import load_checkpoint, save_checkpoint
from face_model_core.config import TrainConfig
from face_model_core.data import create_dataloaders
from face_model_core.losses import ArcFaceLoss, BatchTripletLoss
from face_model_core.model import ArcFaceHead, FaceEmbeddingModel
from face_model_core.utils import get_device, set_seed
from face_model_core.validation import collect_embeddings, quick_similarity_eval


def train_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = get_device()
    print(f"Initializing training on device={device.type}", flush=True)

    resume_checkpoint: dict | None = None
    start_epoch = 1
    best_metric = -1.0
    best_path = config.checkpoint_dir / "best.pt"
    last_path = config.checkpoint_dir / "last.pt"

    if config.resume_from is not None:
        if not config.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {config.resume_from}")

        resume_checkpoint = load_checkpoint(config.resume_from, map_location=device, weights_only=False)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_metric = float(resume_checkpoint.get("best_metric", -1.0)) 
        if start_epoch > config.epochs:
            print("Resume checkpoint already reached requested epochs; skipping training.")
            if best_path.exists():
                return best_path
            return config.resume_from

    print(
        f"Scanning dataset and building dataloaders from data_root={config.data_root} "
        f"(num_workers={config.num_workers}, batch_size={config.batch_size})",
        flush=True,
    )
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        data_root=config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(
        f"Dataloaders ready: train_steps={len(train_loader)} val_steps={len(val_loader)} "
        f"num_classes={num_classes}",
        flush=True,
    )

    model = FaceEmbeddingModel(backbone=config.backbone, embedding_dim=config.embedding_dim).to(device)
    head: ArcFaceHead | None = None

    if config.loss_type == "arcface":
        head = ArcFaceHead(embedding_dim=config.embedding_dim, num_classes=num_classes).to(device)
        criterion: nn.Module = ArcFaceLoss(margin=config.arcface_margin, scale=config.arcface_scale).to(device)
    else:
        criterion = BatchTripletLoss(margin=config.triplet_margin).to(device)

    param_groups = [
        {"params": list(model.backbone.parameters()), "lr": config.backbone_lr},
        {"params": list(model.embedding.parameters()), "lr": config.learning_rate},
    ]
    if head is not None:
        param_groups.append({"params": list(head.parameters()), "lr": config.learning_rate})

    if device.type == "cuda":
        gpu_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"GPU memory after model load: {gpu_mb:.0f} MB", flush=True)

    optimizer = AdamW(params=param_groups, weight_decay=config.weight_decay)
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    params_to_clip: list[nn.Parameter] = list(model.parameters())
    if head is not None:
        params_to_clip.extend(list(head.parameters()))

    # LR scheduler: 1-epoch linear warmup then cosine decay to near-zero.
    warmup_epochs = 1
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs - warmup_epochs, 1))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer_state_loaded = True
        if "optimizer_state" in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            except ValueError:
                optimizer_state_loaded = False
                print(
                    "Resume checkpoint optimizer state is incompatible with current "
                    "optimizer layout; continuing with fresh optimizer/scheduler state.",
                    flush=True,
                )

        if head is not None:
            if "head_state" not in resume_checkpoint:
                raise ValueError("Missing head_state in resume checkpoint for arcface training")
            head.load_state_dict(resume_checkpoint["head_state"])

        if optimizer_state_loaded and "scaler_state" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])

        if optimizer_state_loaded and "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])

        print(f"Resumed from {config.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"Starting epoch {epoch}/{config.epochs}", flush=True)
        model.train()
        if epoch <= config.freeze_backbone_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = False
            # Keep BatchNorm stats stable while the backbone is frozen.
            model.backbone.eval()
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
            model.backbone.train()
            if epoch == config.freeze_backbone_epochs + 1:
                print(f"Unfreezing backbone at epoch {epoch}", flush=True)

        epoch_loss = 0.0
        total_steps = len(train_loader)
        for step, (images, labels) in enumerate(train_loader, start=1):
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            step_loss = float(loss.detach().item())
            epoch_loss += step_loss

            # Heartbeat logs help detect stalled runs on long epochs in Colab.
            if step == 1 or step == total_steps or step % 100 == 0:
                print(
                    f"epoch={epoch} step={step}/{total_steps} batch_loss={step_loss:.4f}",
                    flush=True,
                )

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
        gpu_info = ""
        if device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
            gpu_info = f" gpu_alloc={alloc_mb:.0f}MB gpu_reserved={reserved_mb:.0f}MB"
        current_backbone_lr = optimizer.param_groups[0]["lr"]
        current_main_lr = optimizer.param_groups[1]["lr"]
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"same_mean={metrics['same_mean']:.4f} diff_mean={metrics['diff_mean']:.4f} "
            f"pair_acc={metrics['pair_acc']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} far={metrics['far']:.4f} frr={metrics['frr']:.4f} "
            f"eer={metrics['eer']:.4f} auc={metrics['auc_roc']:.4f} "
            f"optimal_thr={metrics['optimal_threshold']:.4f} "
            f"lr_backbone={current_backbone_lr:.6f} lr_main={current_main_lr:.6f}{gpu_info}",
            flush=True,
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
                scheduler=scheduler,
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
            scheduler=scheduler,
        )

        # Sync checkpoints to backup dir (e.g. Google Drive) after each epoch.
        if config.backup_dir is not None:
            config.backup_dir.mkdir(parents=True, exist_ok=True)
            start_sync = time.time()
            for ckpt in (best_path, last_path):
                if ckpt.exists():
                    shutil.copy2(ckpt, config.backup_dir / ckpt.name)
            elapsed = time.time() - start_sync
            print(f"Synced checkpoints to {config.backup_dir} ({elapsed:.1f}s)", flush=True)

    return best_path if best_path.exists() else last_path
