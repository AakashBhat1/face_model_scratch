# Current Step

Last updated: 2026-04-07 (local)
Owner: Claude

## Where We Are
- Step ID: colab-gpu-utilization-tuning
- Status: READY
- Summary: Tuned Colab training config to utilize T4 GPU properly — increased batch size to 128, workers to 2, and scaled LR.

## Completed In This Pass
- Increased `BATCH_SIZE` from 32 to 128 in `scripts/colab_shell_2_train.py` to fill T4 VRAM.
- Increased `NUM_WORKERS` from 0 to 2 to keep GPU fed with data.
- Scaled `LEARNING_RATE` from 1e-3 to 3e-3 (linear scaling rule for 4x batch size).

## Next Exact Action
- Command: Re-run `scripts/colab_shell_2_train.py` in Colab and monitor GPU utilization via `!nvidia-smi`.
- Expected result: GPU memory usage should rise to ~5-7GB. If OOM, reduce `BATCH_SIZE` to 96.

## If Blocked
- Blocker: OOM error with batch size 128.
- Fix: Reduce `BATCH_SIZE` to 96 or 64 in `scripts/colab_shell_2_train.py`.
- Blocker: Training diverges (loss spikes or NaN).
- Fix: Reduce `LEARNING_RATE` to 2e-3 or 1e-3.
