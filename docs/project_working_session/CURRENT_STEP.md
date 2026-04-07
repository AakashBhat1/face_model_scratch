# Current Step

Last updated: 2026-04-07 (local)
Owner: Claude

## Where We Are
- Step ID: pretrained-backbone-and-lr-scheduler
- Status: READY
- Summary: Switched to pretrained ImageNet backbone and added cosine LR scheduler with warmup for faster convergence.

## Completed In This Pass
- Changed `FaceEmbeddingModel` to use pretrained ImageNet weights by default (`pretrained=True`).
- Added cosine annealing LR scheduler with 1-epoch linear warmup to training loop.
- Scheduler state is saved/restored in checkpoints for correct resume behavior.
- Current LR is logged per epoch for visibility.

## Next Exact Action
- Command: Delete old checkpoints (trained from scratch) and retrain: `!rm -f /content/drive/MyDrive/face_checkpoints/*.pt` then re-run shell 1 + shell 2.
- Expected result: Much faster convergence — expect usable `pair_acc` within 3-5 epochs instead of 12+. LR starts low (warmup), peaks, then decays via cosine.

## If Blocked
- Blocker: Resume from old checkpoint fails (architecture mismatch with pretrained weights).
- Fix: Start fresh — delete old checkpoints.
