"""Colab shell 2: train/resume only.

Run this after `scripts/colab_shell_1_setup.py`.

Strategy: train with checkpoints on LOCAL disk (/content/checkpoints/) for
fast I/O, then sync to Google Drive after each epoch for persistence.
On startup, any existing Drive checkpoints are copied to local first.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import shutil


# =========================
# CONFIGURATION
# =========================
REPO_DIR = Path("/content/face_model_scratch")

# Local (fast) checkpoint dir — on the Colab server SSD.
LOCAL_CHECKPOINT_DIR = Path("/content/checkpoints")

# Drive (persistent) checkpoint dir — survives runtime disconnects.
DRIVE_CHECKPOINT_DIR = Path("/content/drive/MyDrive/face_checkpoints")

# Resume behaviour
AUTO_RESUME_IF_LAST_EXISTS = True

# Cross-account helpers:
IMPORT_RESUME_FROM_CONTENT = Path("/content/last.pt")
RESUME_FROM_URL: str | None = None

BACKBONE = "resnet50"  # choices: resnet50, mobilenet_v2
EMBEDDING_DIM = 512     # choices: 128, 512
LOSS_TYPE = "arcface"  # choices: arcface, triplet

EPOCHS = 10             # lower LR needs more epochs; pretrained backbone still converges fast
BATCH_SIZE = 768        # T4 has 15GB; 1024 OOMs after backbone unfreeze, 768 is safe
LEARNING_RATE = 1e-4    # 1e-3 caused head collapse with arcface_scale=64; 1e-4 is stable
BACKBONE_LR = 1e-5      # backbone should adapt slowly vs randomly initialized head
FREEZE_BACKBONE_EPOCHS = 2
GRAD_CLIP_NORM = 5.0
ARCFACE_SCALE = 30.0    # default 64 causes head collapse; 30 gives gentler gradients
IMAGE_SIZE = 112
NUM_WORKERS = 2         # 2 is stable on Colab; 4 can deadlock
VAL_MAX_IMAGES = 1200
VAL_THRESHOLD = 0.4
MIXED_PRECISION = True  # AMP halves memory, doubles throughput on T4
ALLOW_CPU_TRAINING = False

RESOLVED_DATA_ROOT_FILE = REPO_DIR / ".colab_resolved_data_root.txt"


# =========================
# HELPERS
# =========================

def run_command(command: list[str], cwd: Path | None = None) -> None:
    display = " ".join(command)
    print(f"\n$ {display}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    process.wait()
    if process.returncode != 0:
        captured = "".join(output_lines) or "(no output captured)"
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}.\n"
            f"Output:\n{captured}"
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


# =========================
# CHECKPOINT SYNC
# =========================

def _copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy src to dst if src exists. Returns True if copied."""
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def pull_checkpoints_from_drive() -> None:
    """Copy Drive checkpoints to local disk for fast training I/O."""
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pulled = False

    for name in ("last.pt", "best.pt"):
        src = DRIVE_CHECKPOINT_DIR / name
        dst = LOCAL_CHECKPOINT_DIR / name
        if _copy_if_exists(src, dst):
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"Pulled {name} from Drive to local ({size_mb:.1f} MB)", flush=True)
            pulled = True

    # Also handle uploaded checkpoint in /content
    if IMPORT_RESUME_FROM_CONTENT.is_file():
        dst = LOCAL_CHECKPOINT_DIR / "last.pt"
        shutil.copy2(IMPORT_RESUME_FROM_CONTENT, dst)
        print(f"Imported uploaded checkpoint from {IMPORT_RESUME_FROM_CONTENT}", flush=True)
        pulled = True

    # Handle URL download
    if RESUME_FROM_URL is not None:
        print(f"Downloading resume checkpoint from URL: {RESUME_FROM_URL}", flush=True)
        tmp = Path("/content/downloaded_last.pt")
        urllib.request.urlretrieve(RESUME_FROM_URL, tmp)
        dst = LOCAL_CHECKPOINT_DIR / "last.pt"
        shutil.copy2(tmp, dst)
        print(f"Downloaded checkpoint saved to local", flush=True)
        pulled = True

    if not pulled:
        print("No existing checkpoints found. Starting fresh.", flush=True)


def push_checkpoints_to_drive() -> None:
    """Copy local checkpoints to Drive for persistence."""
    DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "best.pt"):
        src = LOCAL_CHECKPOINT_DIR / name
        if src.is_file():
            dst = DRIVE_CHECKPOINT_DIR / name
            start = time.time()
            shutil.copy2(src, dst)
            elapsed = time.time() - start
            print(f"Synced {name} to Drive ({elapsed:.1f}s)", flush=True)


# =========================
# TRAINING
# =========================

def resolve_resume_checkpoint() -> Path | None:
    """Find a local checkpoint to resume from."""
    if AUTO_RESUME_IF_LAST_EXISTS:
        last_path = LOCAL_CHECKPOINT_DIR / "last.pt"
        if last_path.is_file():
            print(f"Auto-resume: found local {last_path}", flush=True)
            return last_path

    return None


def start_training(data_root: Path) -> None:
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
        "--backbone-lr",
        str(BACKBONE_LR),
        "--freeze-backbone-epochs",
        str(FREEZE_BACKBONE_EPOCHS),
        "--grad-clip-norm",
        str(GRAD_CLIP_NORM),
        "--arcface-scale",
        str(ARCFACE_SCALE),
        "--num-workers",
        str(NUM_WORKERS),
        "--val-max-images",
        str(VAL_MAX_IMAGES),
        "--val-threshold",
        str(VAL_THRESHOLD),
        "--checkpoint-dir",
        str(LOCAL_CHECKPOINT_DIR),
        "--backup-dir",
        str(DRIVE_CHECKPOINT_DIR),
    ]

    if not MIXED_PRECISION:
        command.append("--no-mixed-precision")

    if resume_path is not None:
        command.extend(["--resume-from", str(resume_path)])

    run_command(command, cwd=REPO_DIR)


def main() -> None:
    print("Mounting Google Drive...", flush=True)
    from google.colab import drive  # pylint: disable=import-outside-toplevel

    drive.mount("/content/drive")

    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"Repo not found at {REPO_DIR}. Run scripts/colab_shell_1_setup.py first."
        )

    preflight_runtime_check()
    data_root = resolve_data_root()

    # Pull checkpoints from Drive to fast local disk
    print("\n=== Syncing checkpoints from Drive ===", flush=True)
    pull_checkpoints_from_drive()

    # Train (reads/writes to LOCAL_CHECKPOINT_DIR)
    print("\n=== Starting training ===", flush=True)
    start_training(data_root)

    # Push final checkpoints back to Drive
    print("\n=== Syncing checkpoints to Drive ===", flush=True)
    push_checkpoints_to_drive()

    print("\nTraining completed successfully.", flush=True)


def colab_cell_snippet() -> str:
    """Return a small snippet users can paste into a Colab cell."""
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
