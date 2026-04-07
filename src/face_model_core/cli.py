from __future__ import annotations

import argparse
import math
from pathlib import Path

from face_model_core.config import TrainConfig
from face_model_core.inference import build_gallery, image_to_embedding, infer_with_gallery
from face_model_core.train import train_model


def _cosine_threshold(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"threshold must be finite, got {parsed}")
    if parsed < -1.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"threshold must be in [-1.0, 1.0], got {parsed}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face embedding core pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train embedding model")
    train_p.add_argument("--data-root", type=Path, required=True)
    train_p.add_argument("--backbone", choices=["resnet50", "mobilenet_v2"], default="resnet50")
    train_p.add_argument("--embedding-dim", type=int, choices=[128, 512], default=512)
    train_p.add_argument("--loss-type", choices=["arcface", "triplet"], default="arcface")
    train_p.add_argument("--image-size", type=int, default=112)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--epochs", type=int, default=12)
    train_p.add_argument("--learning-rate", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--num-workers", type=int, default=2)
    mixed_precision_group = train_p.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    train_p.set_defaults(mixed_precision=True)
    train_p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    train_p.add_argument("--val-max-images", type=int, default=1200)
    train_p.add_argument("--val-threshold", type=_cosine_threshold, default=0.4)

    gallery_p = sub.add_parser("build-gallery", help="Build identity gallery embeddings")
    gallery_p.add_argument("--gallery-root", type=Path, required=True)
    gallery_p.add_argument("--checkpoint", type=Path, required=True)
    gallery_p.add_argument("--output", type=Path, required=True)

    infer_p = sub.add_parser("infer", help="Infer one image embedding or match with gallery")
    infer_p.add_argument("--image", type=Path, required=True)
    infer_p.add_argument("--checkpoint", type=Path, required=True)
    infer_p.add_argument("--gallery", type=Path, default=None)
    infer_p.add_argument("--threshold", type=_cosine_threshold, default=0.4)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
            loss_type=args.loss_type,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            mixed_precision=args.mixed_precision,
            checkpoint_dir=args.checkpoint_dir,
            val_max_images=args.val_max_images,
            val_threshold=args.val_threshold,
        )
        best = train_model(config)
        print(f"best_checkpoint={best}")
        return

    if args.command == "build-gallery":
        build_gallery(gallery_root=args.gallery_root, checkpoint_path=args.checkpoint, output_path=args.output)
        print(f"gallery_saved={args.output}")
        return

    if args.command == "infer":
        if args.gallery is None:
            emb = image_to_embedding(image_path=args.image, checkpoint_path=args.checkpoint)
            print(emb.tolist())
            return
        result = infer_with_gallery(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            gallery_path=args.gallery,
            threshold=args.threshold,
        )
        print(result)
        return

    raise RuntimeError("Unsupported command")


if __name__ == "__main__":
    main()
