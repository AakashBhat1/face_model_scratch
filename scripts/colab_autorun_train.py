"""One-cell Colab bootstrap script for face model training.

Usage in Colab:
1. Open a new Python notebook.
2. Paste this entire file content into one code cell.
3. Update CONFIG values if needed.
4. Run the cell.

The script mounts Google Drive, clones/pulls the repo, installs dependencies,
checks dataset structure, and starts training.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# =========================
# CONFIGURATION
# =========================
REPO_URL = "https://github.com/AakashBhat1/face_model_scratch.git"
REPO_DIR = Path("/content/face_model_scratch")

DATA_ROOT = Path("/content/drive/MyDrive/face_data")
CHECKPOINT_DIR = Path("/content/drive/MyDrive/face_checkpoints")

AUTO_DOWNLOAD_DATASET = True
KAGGLE_DATASET = "hearfool/vggface2"

BACKBONE = "resnet50"  # choices: resnet50, mobilenet_v2
EMBEDDING_DIM = 512     # choices: 128, 512
LOSS_TYPE = "arcface"  # choices: arcface, triplet

EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMAGE_SIZE = 112
NUM_WORKERS = 2
VAL_MAX_IMAGES = 1200
VAL_THRESHOLD = 0.4
MIXED_PRECISION = True
ALLOW_CPU_TRAINING = False
RESUME_FROM: Path | None = None  # example: Path("/content/drive/MyDrive/face_checkpoints/last.pt")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    display = " ".join(command)
    print(f"\n$ {display}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def ensure_repo() -> None:
    if REPO_DIR.exists():
        print(f"Repo exists at {REPO_DIR}, pulling latest changes...")
        run_command(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    else:
        print(f"Cloning repo into {REPO_DIR}...")
        run_command(["git", "clone", REPO_URL, str(REPO_DIR)])


def install_dependencies() -> None:
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_command([sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "requirements.txt")])
    run_command([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR)])
    if AUTO_DOWNLOAD_DATASET:
        run_command([sys.executable, "-m", "pip", "install", "kagglehub"])


def validate_data_layout(data_root: Path) -> None:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset path invalid. Expected directories:\n"
            f"- {train_dir}\n"
            f"- {val_dir}\n"
            "Upload/move your VGGFace2 subset so both exist before running."
        )


def _find_dataset_root(base_path: Path) -> Path | None:
    if (base_path / "train").is_dir() and (base_path / "val").is_dir():
        return base_path

    for path in sorted(base_path.rglob("*")):
        if path.is_dir() and (path / "train").is_dir() and (path / "val").is_dir():
            return path
    return None


def resolve_data_root() -> Path:
    if (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "val").is_dir():
        print(f"Using DATA_ROOT: {DATA_ROOT}")
        return DATA_ROOT

    if not AUTO_DOWNLOAD_DATASET:
        raise FileNotFoundError(
            "DATA_ROOT does not contain train/ and val/.\n"
            f"Configured DATA_ROOT: {DATA_ROOT}\n"
            "Either fix DATA_ROOT or set AUTO_DOWNLOAD_DATASET=True."
        )

    print(f"Downloading dataset from KaggleHub: {KAGGLE_DATASET}")
    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise RuntimeError("kagglehub is not installed. Set AUTO_DOWNLOAD_DATASET=False or install kagglehub.") from exc

    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"KaggleHub download path: {downloaded_path}")

    resolved = _find_dataset_root(downloaded_path)
    if resolved is not None:
        print(f"Resolved dataset root: {resolved}")
        return resolved

    raise FileNotFoundError(
        "Downloaded dataset does not include expected train/ and val/ folders.\n"
        f"Downloaded path: {downloaded_path}\n"
        "Inspect the folder structure and set DATA_ROOT manually."
    )


def start_training(data_root: Path) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        "-m",
        "face_model_core.cli",
        "train",
        "--data-root",
        str(data_root),
        "--backbone",
        BACKBONE,
        "--embedding-dim",
        str(EMBEDDING_DIM),
        "--loss-type",
        LOSS_TYPE,
        "--image-size",
        str(IMAGE_SIZE),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        str(LEARNING_RATE),
        "--num-workers",
        str(NUM_WORKERS),
        "--val-max-images",
        str(VAL_MAX_IMAGES),
        "--val-threshold",
        str(VAL_THRESHOLD),
        "--checkpoint-dir",
        str(CHECKPOINT_DIR),
    ]

    if not MIXED_PRECISION:
        command.append("--no-mixed-precision")

    if RESUME_FROM is not None:
        command.extend(["--resume-from", str(RESUME_FROM)])

    run_command(command, cwd=REPO_DIR)


def preflight_runtime_check() -> None:
    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        print("GPU detected. Training will use CUDA.")
        return

    if ALLOW_CPU_TRAINING:
        print("No GPU detected. Continuing with CPU training because ALLOW_CPU_TRAINING=True.")
        return

    raise RuntimeError(
        "No GPU detected in Colab runtime. Enable GPU via Runtime > Change runtime type > T4/A100, "
        "or set ALLOW_CPU_TRAINING=True to continue on CPU."
    )


def main() -> None:
    print("Mounting Google Drive...")
    from google.colab import drive  # pylint: disable=import-outside-toplevel

    drive.mount("/content/drive")
    ensure_repo()
    install_dependencies()
    resolved_data_root = resolve_data_root()
    validate_data_layout(resolved_data_root)
    preflight_runtime_check()
    print("Starting training...")
    start_training(resolved_data_root)
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
