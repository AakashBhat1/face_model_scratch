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
