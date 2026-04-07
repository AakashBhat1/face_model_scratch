"""Colab shell 2: train/resume only.

Run this after `scripts/colab_shell_1_setup.py`.
It reads resolved data root, checks runtime, and starts training.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path
import shutil


# =========================
# CONFIGURATION
# =========================
REPO_DIR = Path("/content/face_model_scratch")
CHECKPOINT_DIR = Path("/content/drive/MyDrive/face_checkpoints")

# Set explicitly if needed, otherwise shell 2 uses auto-resume from last.pt.
RESUME_FROM: Path | None = None
AUTO_RESUME_IF_LAST_EXISTS = True

# Cross-account helpers:
# 1) upload last.pt into /content in Colab and leave this path as-is,
# 2) or set RESUME_FROM_URL to download a checkpoint.
IMPORT_RESUME_FROM_CONTENT = Path("/content/last.pt")
RESUME_FROM_URL: str | None = None
DOWNLOADED_RESUME_PATH = Path("/content/downloaded_last.pt")

BACKBONE = "resnet50"  # choices: resnet50, mobilenet_v2
EMBEDDING_DIM = 512     # choices: 128, 512
LOSS_TYPE = "arcface"  # choices: arcface, triplet

EPOCHS = 12
BATCH_SIZE = 256        # T4 has 15GB; 256 at 112x112 with AMP uses ~8-10GB
LEARNING_RATE = 5e-3    # linear scaling: 256/32 = 8x, so 8 * 1e-3 ≈ 5e-3 (capped)
IMAGE_SIZE = 112
NUM_WORKERS = 4         # Colab has 2 CPU cores; 4 workers keeps GPU pipeline full
VAL_MAX_IMAGES = 1200
VAL_THRESHOLD = 0.4
MIXED_PRECISION = True  # AMP halves memory, doubles throughput on T4
ALLOW_CPU_TRAINING = False

RESOLVED_DATA_ROOT_FILE = REPO_DIR / ".colab_resolved_data_root.txt"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    display = " ".join(command)
    print(f"\n$ {display}", flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}.\n"
            f"Output:\n{result.stdout or '(no output captured)'}"
        )


def preflight_runtime_check() -> None:
    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        print("GPU detected. Training will use CUDA.")
        return

    if ALLOW_CPU_TRAINING:
        print("No GPU detected. Continuing with CPU training because ALLOW_CPU_TRAINING=True.")
        return

    raise RuntimeError(
        "No GPU detected in Colab runtime. Enable GPU via Runtime > Change runtime type > GPU, "
        "or set ALLOW_CPU_TRAINING=True to continue on CPU."
    )


def resolve_data_root() -> Path:
    if RESOLVED_DATA_ROOT_FILE.exists():
        value = RESOLVED_DATA_ROOT_FILE.read_text(encoding="utf-8").strip()
        resolved = Path(value)
        if (resolved / "train").is_dir() and (resolved / "val").is_dir():
            print(f"Using resolved data root from shell 1: {resolved}")
            return resolved

    raise FileNotFoundError(
        "Could not find a valid resolved data root. Run scripts/colab_shell_1_setup.py first."
    )


def resolve_resume_checkpoint() -> Path | None:
    if RESUME_FROM is not None:
        if not RESUME_FROM.exists():
            raise FileNotFoundError(f"Configured RESUME_FROM not found: {RESUME_FROM}")
        return RESUME_FROM

    if AUTO_RESUME_IF_LAST_EXISTS:
        last_path = CHECKPOINT_DIR / "last.pt"
        if last_path.exists():
            print(f"Auto-resume enabled. Using checkpoint: {last_path}")
            return last_path

    if IMPORT_RESUME_FROM_CONTENT.exists():
        target_path = CHECKPOINT_DIR / "last.pt"
        shutil.copy2(IMPORT_RESUME_FROM_CONTENT, target_path)
        print(f"Imported uploaded checkpoint from {IMPORT_RESUME_FROM_CONTENT} to {target_path}")
        return target_path

    if RESUME_FROM_URL is not None:
        print(f"Downloading resume checkpoint from URL: {RESUME_FROM_URL}")
        urllib.request.urlretrieve(RESUME_FROM_URL, DOWNLOADED_RESUME_PATH)
        target_path = CHECKPOINT_DIR / "last.pt"
        shutil.copy2(DOWNLOADED_RESUME_PATH, target_path)
        print(f"Downloaded checkpoint copied to {target_path}")
        return target_path

    return None


def start_training(data_root: Path) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_checkpoint()

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

    if resume_path is not None:
        command.extend(["--resume-from", str(resume_path)])

    run_command(command, cwd=REPO_DIR)


def main() -> None:
    print("Mounting Google Drive...")
    from google.colab import drive  # pylint: disable=import-outside-toplevel

    drive.mount("/content/drive")

    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"Repo not found at {REPO_DIR}. Run scripts/colab_shell_1_setup.py first."
        )

    preflight_runtime_check()
    data_root = resolve_data_root()
    print("Starting training...")
    start_training(data_root)
    print("\nTraining completed successfully.")


def colab_cell_snippet() -> str:
    """Return a small snippet users can paste into a Colab cell.

    This snippet executes the latest repo version of this script,
    so git-pull updates take effect without re-pasting the cell.
    """
    return (
        "# Colab Shell 2 — paste this once, it always runs the latest repo version\n"
        "import subprocess, sys\n"
        "subprocess.run(\n"
        '    [sys.executable, "/content/face_model_scratch/scripts/colab_shell_2_train.py"],\n'
        "    check=True,\n"
        ")\n"
    )


if __name__ == "__main__":
    main()
