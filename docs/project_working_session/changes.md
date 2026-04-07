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

## Pass 2026-04-07-13
- Updated: .gitignore
- Added: local_model_testing/quick_eval_best.py
- Added: local_model_testing/quick_infer_best.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Added a separate local testing folder so checkpoint testing can run without any training-code edits. `quick_eval_best.py` reports validation pair metrics, validates CLI input ranges, and uses safe `weights_only` loading by default with explicit `--allow-unsafe-deserialization` opt-in for trusted legacy checkpoints only. `quick_infer_best.py` builds a gallery and runs a query match, requires explicit `--image`, blocks query-in-gallery by default unless `--allow-query-in-gallery` is set, and safely handles Windows cross-drive path comparisons. Added `local_model_testing/` to `.gitignore`.
- Verification: `python -m py_compile local_model_testing/quick_eval_best.py local_model_testing/quick_infer_best.py` -> success.
- Verification: `python local_model_testing/quick_eval_best.py --help` -> success.
- Verification: `python local_model_testing/quick_infer_best.py --help` -> success.
- Follow-up: run `quick_eval_best.py` with your dataset root to validate first-epoch `models/best.pt` quality.

## Pass 2026-04-07-14
- Updated: src/face_model_core/utils.py
- Updated: src/face_model_core/data.py
- Updated: src/face_model_core/train.py
- Updated: scripts/colab_shell_2_train.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Full GPU utilization overhaul. Enabled `cudnn.benchmark` for fixed-size input speedup. Added `persistent_workers` and `prefetch_factor=4` to dataloaders to eliminate worker restart overhead and keep GPU fed. Moved loss criterion to GPU. Added GPU memory logging (allocated + reserved) at model load and per epoch. Bumped Colab defaults to `BATCH_SIZE=256`, `NUM_WORKERS=4`, `LR=5e-3` to properly fill T4 15GB VRAM.
- Verification: `pytest -q` -> 31 passed in 2.83s.
- Follow-up: re-run shell 2 in Colab; expect 4-8GB allocated, 8-12GB reserved. If OOM, reduce to BATCH_SIZE=128.

## Pass 2026-04-07-15
- Added: local_model_testing/README.md
- Added: local_model_testing/input/query_images/PUT_YOUR_FACE_HERE.txt
- Added: local_model_testing/scripts/quick_eval_best.py
- Added: local_model_testing/scripts/quick_infer_best.py
- Added: local_model_testing/output/galleries/
- Deleted: local_model_testing/quick_eval_best.py
- Deleted: local_model_testing/quick_infer_best.py
- Updated: docs/project_working_session/REPO_CONTEXT.md
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Reorganized local smoke testing into `scripts/`, `input/query_images/`, and `output/galleries/` folders. Added a default face photo drop location at `local_model_testing/input/query_images/my_face.jpg`, updated inference defaults to use it, added strict threshold validation, and aligned both smoke scripts with explicit trusted-only unsafe-deserialization fallback behavior.
- Verification: `python -m py_compile local_model_testing/scripts/quick_eval_best.py local_model_testing/scripts/quick_infer_best.py` -> success.
- Verification: `python local_model_testing/scripts/quick_infer_best.py --help` -> success.
- Follow-up: place a photo at `local_model_testing/input/query_images/my_face.jpg` and run quick inference.

## Pass 2026-04-07-16
- Updated: scripts/colab_shell_2_train.py
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Fixed stale Colab cell problem. Shell 2 now includes a `colab_cell_snippet()` that returns a thin 4-line launcher. Users paste the launcher once; it executes the repo file directly so `git pull` updates always take effect. Also improved `run_command` to merge stdout+stderr for visible error output in notebooks.
- Verification: `pytest -q` -> 31 passed in 2.58s.
- Follow-up: replace Colab shell 2 cell with the thin launcher, re-run shell 1 then shell 2.

## Pass 2026-04-07-17
- Updated: src/face_model_core/checkpoint.py
- Updated: docs/project_working_session/CURRENT_STEP.md
- Updated: docs/project_working_session/changes.md
- Notes: Fixed corrupt checkpoint crash. `last.pt` on Drive was truncated (Colab disconnect mid-save). Added atomic checkpoint saving (write to `.pt.tmp` then rename) to prevent future corruption. Added clear error message for corrupted zip archives on load.
- Verification: `pytest -q` -> 31 passed in 2.83s.
- Follow-up: delete corrupt `last.pt` from Drive, re-run shell 1 + shell 2 to start fresh.
