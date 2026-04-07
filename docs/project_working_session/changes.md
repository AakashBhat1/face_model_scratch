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

## Pass 2026-04-07-09
- Added: scripts/colab_shell_1_setup.py
- Added: scripts/colab_shell_2_train.py
- Updated: README.md
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added split Colab workflow with two scripts: shell 1 performs setup/mount/sync/dependency+dataset resolution, and shell 2 runs train/resume so users can manually place `best.pt`/`last.pt` between steps.
- Verification: `".venv/Scripts/python.exe" -m py_compile "scripts/colab_shell_1_setup.py" "scripts/colab_shell_2_train.py" "scripts/colab_autorun_train.py"` -> success.
- Verification: `".venv/Scripts/python.exe" -m pytest -q` -> 31 passed.
- Follow-up: use shell 1 then shell 2 in Colab; optionally set `RESUME_FROM` or rely on auto-resume from `last.pt`.

## Pass 2026-04-07-10
- Updated: scripts/colab_shell_2_train.py
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Increased BATCH_SIZE from 32 to 128, NUM_WORKERS from 0 to 2, and scaled LR from 1e-3 to 3e-3 to utilize T4 GPU (was using ~1GB of 15GB due to tiny batches and no data workers).
- Verification: `py_compile scripts/colab_shell_2_train.py` -> success.
- Verification: `pytest -q` -> 31 passed in 3.02s.
- Follow-up: if OOM occurs, reduce to BATCH_SIZE=96. If training is unstable, reduce LR to 2e-3. For even higher utilization try BATCH_SIZE=256.

## Pass 2026-04-07-11
- Added: .gitignore
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added root `.gitignore` with `models/` excluded from version control.
- Verification: `pytest -q` -> 31 passed (no code changes, config-only).

## Pass 2026-04-07-12
- Updated: src/face_model_core/train.py
- Updated: scripts/colab_shell_2_train.py
- Updated: tests/test_train_resume.py
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Fixed Colab resume crash — `weights_only=True` rejects non-tensor data (config dict, class_names, epoch int) in checkpoint. Changed to `weights_only=False` for resume path. Also improved `run_command` to surface subprocess errors in Colab notebooks instead of swallowing them.
- Verification: `pytest -q` -> 31 passed in 2.81s.
- Follow-up: re-run shell 2 in Colab; resume should now succeed.
