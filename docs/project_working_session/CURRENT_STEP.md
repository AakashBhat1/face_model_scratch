# Current Step

Last updated: 2026-04-07 (local)
Owner: Claude

## Where We Are
- Step ID: local-checkpoint-with-drive-sync
- Status: READY
- Summary: Overhauled checkpoint strategy — train on local SSD, sync to Drive after each epoch. Eliminates Drive I/O bottleneck that caused hangs.

## Completed In This Pass
- Rewrote `scripts/colab_shell_2_train.py` with local/Drive split: `LOCAL_CHECKPOINT_DIR=/content/checkpoints/` for fast training I/O, `DRIVE_CHECKPOINT_DIR` for persistence.
- On startup: pulls existing checkpoints from Drive to local.
- Added `backup_dir` field to `TrainConfig` and `--backup-dir` CLI flag.
- Training loop syncs `best.pt`/`last.pt` to `backup_dir` (Drive) after every epoch with timing log.
- After training completes: final sync to Drive.
- Reduced `NUM_WORKERS` to 2 (4 deadlocks on Colab).

## Next Exact Action
- Command: Push, re-run shell 1 then shell 2 in Colab. Training reads/writes locally; Drive gets synced per epoch.
- Expected result: No more Drive I/O hangs. Logs stream in real-time. Per-epoch `Synced checkpoints to Drive (Xs)` confirms backup.

## If Blocked
- Blocker: CUDA OOM with batch size 512.
- Fix: Reduce `BATCH_SIZE` to 256 in `scripts/colab_shell_2_train.py`.
