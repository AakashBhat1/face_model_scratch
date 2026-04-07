# Changes Log

## Pass 2026-04-07-01
- Updated: README.md
- Updated: src/face_model_core/cli.py
- Updated: src/face_model_core/inference.py
- Updated: tests/test_cli.py
- Added: tests/test_inference.py
- Added: tests/test_validation.py
- Added: docs/project_working_session/REPO_CONTEXT.md
- Added: docs/project_working_session/CURRENT_STEP.md
- Added: docs/project_working_session/changes.md
- Notes: Completed PRD alignment pass for core model pipeline; strengthened CLI defaults, gallery robustness, and test coverage for inference/validation.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 17 passed.
- Follow-up: run full training on VGGFace2 subset and monitor pair metrics/checkpoint quality.

## Pass 2026-04-07-02
- Updated: src/face_model_core/cli.py
- Updated: src/face_model_core/inference.py
- Updated: src/face_model_core/losses.py
- Updated: src/face_model_core/checkpoint.py
- Updated: tests/test_cli.py
- Updated: tests/test_inference.py
- Updated: tests/test_losses.py
- Updated: tests/test_checkpoint.py
- Updated: README.md
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Completed final PRD pass with runtime fixes (triplet no-grad edge case), inference performance improvements, threshold validation, malformed gallery guards, and secure inference checkpoint loading compatibility.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 26 passed.
- Additional smoke check: one-epoch synthetic training run succeeded and wrote `best.pt`.
- Follow-up: train on real VGGFace2 subset and track `pair_acc`, `same_mean`, and `diff_mean` across epochs.

## Pass 2026-04-07-03
- Added: scripts/colab_autorun_train.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added one-cell Colab automation script to mount Drive, sync repo, install dependencies, validate dataset layout, enforce GPU preflight (or explicit CPU override), and start training automatically.
- Verification: `".venv/Scripts/python.exe" -m py_compile "scripts/colab_autorun_train.py"` -> success.
- Follow-up: run the script in Colab and adjust CONFIG values (`DATA_ROOT`, `CHECKPOINT_DIR`, epochs) as needed.

## Pass 2026-04-07-04
- Updated: src/face_model_core/config.py
- Updated: src/face_model_core/cli.py
- Updated: src/face_model_core/train.py
- Updated: scripts/colab_autorun_train.py
- Updated: tests/test_config.py
- Updated: tests/test_cli.py
- Added: tests/test_train_resume.py
- Updated: README.md
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Implemented pause/resume support with `resume_from` in config and `--resume-from` in CLI; added secure resume checkpoint loading (`weights_only=True`), early skip when target epochs are already completed, and Colab script support for optional `RESUME_FROM` path.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 31 passed.
- Follow-up: in Colab, set `RESUME_FROM` to Drive checkpoint path when resuming interrupted runs.

## Pass 2026-04-07-05
- Updated: scripts/colab_autorun_train.py
- Updated: README.md
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added optional KaggleHub automation in Colab script (`AUTO_DOWNLOAD_DATASET`, `KAGGLE_DATASET`) with automatic detection of dataset root containing `train/` and `val/` before training starts.
- Verification: `".venv/Scripts/python.exe" -m py_compile "scripts/colab_autorun_train.py"` -> success.
- Follow-up: enable KaggleHub download in Colab and ensure Kaggle credentials are available if required.

## Pass 2026-04-07-06
- Updated: src/face_model_core/train.py
- Updated: scripts/colab_autorun_train.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Improved runtime observability by adding heartbeat logging every 100 batches (and first/last batch each epoch) plus unbuffered Python execution in Colab script so logs appear immediately.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 31 passed.
- Verification: `".venv/Scripts/python.exe" -m py_compile "scripts/colab_autorun_train.py"` -> success.
- Follow-up: run in Colab and monitor heartbeat logs to distinguish active training from stalled setup.

## Pass 2026-04-07-07
- Updated: src/face_model_core/train.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added explicit startup logs before dataloader creation and at epoch start so users can distinguish dataset scanning from actual stalls when Colab output is quiet.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 31 passed.
- Follow-up: if startup still appears silent in Colab, retry with `NUM_WORKERS=0` to rule out dataloader worker stalls.

## Pass 2026-04-07-08
- Updated: scripts/colab_autorun_train.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Set Colab autorun default `NUM_WORKERS` to 0 so fresh runs avoid common multiprocessing worker stalls in Colab and show logs reliably.
- Verification: `".venv/Scripts/python.exe" -m py_compile "scripts/colab_autorun_train.py"` -> success.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 31 passed.
- Follow-up: after stable runs are confirmed, optionally test `NUM_WORKERS=1` for speed.
