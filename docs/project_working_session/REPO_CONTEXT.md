# Repo Context

## Project Name
Face Recognition Model (Core Only) for learning face embeddings from identity-labeled image folders.

## Architecture Overview
- `src/face_model_core/config.py`: immutable training configuration and validation, including optional `resume_from` checkpoint path.
- `src/face_model_core/data.py`: dataset loading from `train/` and `val/` with augmentation.
- `src/face_model_core/model.py`: face embedding backbones (`resnet50`, `mobilenet_v2`) and ArcFace head.
- `src/face_model_core/losses.py`: ArcFace loss and batch-hard triplet loss with stable zero-valid-triplet backward behavior.
- `src/face_model_core/train.py`: end-to-end training loop with AdamW, AMP support, checkpointing, secure checkpoint resume support, explicit dataloader-startup logs, and periodic batch heartbeat logs for long-epoch visibility.
- `src/face_model_core/validation.py`: minimal pairwise similarity evaluation on validation embeddings.
- `src/face_model_core/inference.py`: image to embedding, gallery build, and gallery matching with threshold and gallery-shape validation.
- `src/face_model_core/cli.py`: `train`, `build-gallery`, and `infer` commands with bounded threshold parsing, mixed precision opt-out, and `--resume-from` checkpoint continuation.
- `src/face_model_core/checkpoint.py`: checkpoint save/load utilities with weights-only safe loading support.
- `scripts/colab_autorun_train.py`: one-cell Colab bootstrap script that mounts Drive, clones/pulls repo, installs deps, optionally downloads dataset from KaggleHub, auto-resolves dataset root (`train/` + `val/`), checks GPU runtime, and starts/resumes training with unbuffered Python output; default `NUM_WORKERS=0` for stability (prefer shell 2 for tuned GPU settings).
- `scripts/colab_shell_1_setup.py`: Colab shell 1 for setup (mount Drive, sync repo, install deps, resolve dataset root, and prepare checkpoint directory).
- `scripts/colab_shell_2_train.py`: Colab shell 2 for training/resume (auto-reads resolved dataset root and can auto-resume from `last.pt`); default Colab settings tuned for T4 GPU (`BATCH_SIZE=256`, `NUM_WORKERS=4`, `LR=5e-3`, AMP enabled).
- `local_model_testing/quick_eval_best.py`: local smoke-test script (untracked) that evaluates a checkpoint (default `models/best.pt`) on validation pairs and prints `same_mean`, `diff_mean`, and `pair_acc`.
- `local_model_testing/quick_infer_best.py`: local smoke-test script (untracked) that builds a gallery from `val/` and runs one query image match against it.
- `tests/`: unit tests for config, model, losses, similarity, checkpointing, CLI, inference, and validation including malformed gallery and threshold edge cases.

## Key Dependencies
| Package | Purpose |
|---------|---------|
| torch | model training and inference |
| torchvision | backbones, transforms, ImageFolder |
| numpy | embedding and similarity operations |
| Pillow | image loading |
| pytest | test execution |

## Environment
- Language/runtime: Python 3.13 (venv)
- Package manager: pip
- Test command: `".venv/Scripts/python.exe" -m pytest -q`

## Assumptions and Constraints
- Dataset is folder-structured by identity with separate `train/` and `val/` directories.
- Scope is core embedding model only (not production search or deployment).
- Minimal validation is pair-based similarity, not benchmark-grade evaluation.
- Mixed precision is enabled by default in CLI and activates on CUDA.
- Inference expects trusted checkpoints; inference load path enforces safer `weights_only` deserialization.
- Local throwaway testing scripts are placed in `local_model_testing/`, and this folder is gitignored.
