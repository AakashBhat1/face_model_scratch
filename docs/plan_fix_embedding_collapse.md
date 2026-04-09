# Plan: Fix Embedding Collapse During Training

## Context

The model collapses around epoch 3-4 every time: all embeddings become identical (`same_mean=1.0, diff_mean=1.0, pair_acc=0.5`). Root cause is the pretrained backbone getting destabilized before the randomly-initialized head learns anything useful — compounded by no gradient clipping and ArcFace's 64x scale multiplier amplifying gradient spikes.

### Evidence from latest run

| Epoch | same_mean | diff_mean | pair_acc | Status |
|-------|-----------|-----------|----------|--------|
| 1 | 0.45 | 0.15 | 0.80 | Learning |
| 2 | 0.31 | -0.03 | 0.80 | Learning |
| 3 | 0.32 | 0.57 | 0.40 | Collapsing |
| 4+ | 1.00 | 1.00 | 0.50 | Dead — constant output |

Loss flatlines at ~1.169 from epoch 4 onward. All embeddings are identical vectors.

## Root Causes

1. **No gradient clipping** — ArcFace multiplies logits by `scale=64`. Gradient spikes go unchecked.
2. **Single learning rate** — Pretrained backbone, random embedding layer, and random ArcFace head all share `lr=1e-3`. Backbone needs 10-100x lower LR.
3. **No backbone freezing** — Backbone is fine-tuned from step 1. By epoch 3 the pretrained features are corrupted before the head stabilizes.

## Changes

### 1. Gradient clipping (`train.py` lines 125-127)

Insert `scaler.unscale_(optimizer)` + `clip_grad_norm_(params, max_norm)` between `backward()` and `step()`.

New config field: `grad_clip_norm: float = 5.0`

### 2. Differential learning rates (`train.py` lines 70-79)

Replace flat `params` list with param groups:

```python
param_groups = [
    {"params": list(model.backbone.parameters()), "lr": config.backbone_lr},
    {"params": list(model.embedding.parameters()), "lr": config.learning_rate},
]
if head is not None:
    param_groups.append({"params": list(head.parameters()), "lr": config.learning_rate})
```

New config field: `backbone_lr: float = 1e-5`

### 3. Freeze backbone for first N epochs (`train.py` epoch loop ~line 106)

```python
if epoch <= config.freeze_backbone_epochs:
    for p in model.backbone.parameters():
        p.requires_grad = False
elif epoch == config.freeze_backbone_epochs + 1:
    for p in model.backbone.parameters():
        p.requires_grad = True
    print(f"Unfreezing backbone at epoch {epoch}")
```

New config field: `freeze_backbone_epochs: int = 2`

### 4. Config (`config.py`)

Add to `TrainConfig`:
- `backbone_lr: float = 1e-5`
- `freeze_backbone_epochs: int = 2`
- `grad_clip_norm: float = 5.0`

Validation: all must be positive (freeze can be 0 to disable).

### 5. CLI (`cli.py`)

Add flags: `--backbone-lr`, `--freeze-backbone-epochs`, `--grad-clip-norm`

### 6. Colab shell 2 (`scripts/colab_shell_2_train.py`)

Add config vars + pass as CLI args:
- `BACKBONE_LR = 1e-5`
- `FREEZE_BACKBONE_EPOCHS = 2`
- `GRAD_CLIP_NORM = 5.0`

### 7. Tests

- `tests/test_train_resume.py` — update `_build_resume_checkpoint` optimizer to use param groups
- `tests/test_config.py` — add validation tests for the 3 new fields

## File Change Summary

| File | Change |
|------|--------|
| `src/face_model_core/config.py` | Add 3 fields + validation |
| `src/face_model_core/train.py` | Param groups, grad clip, freeze logic |
| `src/face_model_core/cli.py` | Add 3 CLI flags |
| `scripts/colab_shell_2_train.py` | Add 3 config vars + CLI args |
| `tests/test_config.py` | Validation tests for new fields |
| `tests/test_train_resume.py` | Update optimizer structure in helper |

## Expected Training Behavior After Fix

- **Epochs 1-2**: Backbone frozen. Only embedding + ArcFace head train. Loss drops, `pair_acc` climbs.
- **Epoch 3+**: Backbone unfreezes at `backbone_lr=1e-5` (100x lower than head). Gradients clipped at 5.0. Continued improvement, no collapse.
- **Key signal**: `same_mean` and `diff_mean` stay separated, never both converge to 1.0.

## Verification

1. `python -m pytest -q` — all tests pass
2. Retrain fresh in Colab, confirm no collapse through all epochs
