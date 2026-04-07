"""Colab shell 1: setup only.

Run this first in Colab to:
- mount Google Drive
- clone/pull repo
- install dependencies
- optionally download dataset via KaggleHub
- validate dataset layout

Then run `scripts/colab_shell_2_train.py` to start/resume training.
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

RESOLVED_DATA_ROOT_FILE = REPO_DIR / ".colab_resolved_data_root.txt"


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
            "Upload/move your dataset so both exist before running."
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
        raise RuntimeError("kagglehub is not installed.") from exc

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


def main() -> None:
    print("Mounting Google Drive...")
    from google.colab import drive  # pylint: disable=import-outside-toplevel

    drive.mount("/content/drive")
    ensure_repo()
    install_dependencies()

    resolved_data_root = resolve_data_root()
    validate_data_layout(resolved_data_root)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESOLVED_DATA_ROOT_FILE.write_text(str(resolved_data_root), encoding="utf-8")

    print("\nShell 1 setup complete.")
    print(f"Resolved data root: {resolved_data_root}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print("You can now upload or verify best.pt/last.pt in checkpoint dir, then run shell 2.")


if __name__ == "__main__":
    main()
