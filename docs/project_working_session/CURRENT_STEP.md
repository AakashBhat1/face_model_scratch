# Current Step

Last updated: 2026-04-07 (local)
Owner: Claude

## Where We Are
- Step ID: gpu-max-utilization
- Status: READY
- Summary: Overhauled training pipeline for maximum T4 GPU utilization — batch 256, 4 workers, cuDNN benchmark, persistent workers, prefetch, GPU memory logging.

## Completed In This Pass
- Enabled `cudnn.benchmark = True` in `utils.py` for consistent-size input speedup.
- Added `persistent_workers=True` and `prefetch_factor=4` to dataloaders in `data.py`.
- Moved loss criterion to GPU device in `train.py`.
- Added GPU memory logging (alloc + reserved) at model load and after each epoch.
- Bumped Colab defaults: `BATCH_SIZE=256`, `NUM_WORKERS=4`, `LR=5e-3`.

## Next Exact Action
- Command: Re-run `scripts/colab_shell_2_train.py` in Colab. Watch for GPU memory logs in output.
- Expected result: `gpu_alloc` should show 4-8GB, `gpu_reserved` 8-12GB. Epochs should complete significantly faster.

## If Blocked
- Blocker: CUDA OOM with batch size 256.
- Fix: Reduce `BATCH_SIZE` to 128 in `scripts/colab_shell_2_train.py`.
- Blocker: Training diverges (NaN loss).
- Fix: Reduce `LEARNING_RATE` to 3e-3.
