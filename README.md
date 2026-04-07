# Face Recognition Model (Core Only)

This repository implements the face embedding core from your PRD.

## 1. Objective

Train a model that maps a face image to an embedding vector such that:

- Same person: embeddings are close.
- Different people: embeddings are far.

This scope intentionally excludes large-scale benchmarking and production deployment.

## 2. Dataset

Primary dataset target: VGGFace2 subset style layout.

Expected structure:

data/
  train/
    person_0001/
      img1.jpg
      img2.jpg
    person_0002/
      ...
  val/
    person_0001/
      ...
    person_0002/
      ...

Usage:

- Train on `train/`.
- Validate on `val/`.
- No separate test split is required for this stage.

## 3. System Pipeline

Training pipeline:

1. Load image from dataset.
2. Apply augmentation.
3. Forward through CNN backbone.
4. Project to embedding vector.
5. Compute metric-learning loss.
6. Backpropagate and update weights.

Inference pipeline:

1. Input face image.
2. Convert to embedding.
3. Compare with stored embeddings using cosine similarity.
4. Return closest identity and match/no-match decision.

## 4. Model Design

Backbones:

- `resnet50` (recommended default)
- `mobilenet_v2` (faster/lighter option)

Embedding dimensions:

- `128`
- `512`

Loss functions:

- `arcface` (default)
- `triplet` (batch-hard variant)

## 5. Training Configuration (Colab-friendly defaults)

Defaults in this repo:

- Batch size: `32`
- Epochs: `12` (typical target range 10 to 20)
- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Mixed precision: enabled by default (used automatically on CUDA)

Augmentations:

- Random crop
- Horizontal flip
- Brightness/contrast jitter

## 6. Estimated Training Time

For around 185K images (hardware dependent):

- 1 epoch: about 45 to 90 minutes
- 10 to 15 epochs: about 8 to 18 hours
- With Colab interruptions: often 1 to 2 days wall-clock

## 7. Outputs

Model outputs:

- Embedding generator from image to vector
- Example output: `[0.12, -0.44, ..., 0.89]`

Saved artifacts:

- Model checkpoints: `.pt` (`best.pt`, `last.pt`)
- Optional gallery embeddings: `.npz`

## 8. Minimal Evaluation

Current validation is intentionally lightweight:

- Sample identity pairs from validation embeddings
- Report same-identity similarity mean
- Report different-identity similarity mean
- Report threshold-based pair accuracy

## 9. Storage

Recommended persistence:

- Save checkpoints frequently during training
- Save gallery embeddings when needed
- Use Google Drive mount in Colab for long runs

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting | Use augmentation and monitor validation pair metrics |
| Slow training | Use `mobilenet_v2`, lower image size, lower embedding dim |
| Colab timeout | Save checkpoints every epoch and resume |

## 11. Deliverables

This repository delivers:

- Trained face embedding model checkpoints
- Inference function (image to embedding)
- Similarity comparison flow (gallery match/no-match)

## 12. Final Scope

In scope:

- Face embedding model core
- Minimal validation and inference utilities

Out of scope for now:

- Large-scale nearest-neighbor search service
- Full benchmark suite
- Production deployment system

## 13. Next Step Extensions

After this core is stable, extend to:

- Face verification API/service
- Face search index (for example FAISS)
- Real-time webcam recognition

## Quick Start

1. Create and activate venv.
2. Install dependencies.
3. Train.
4. Build gallery.
5. Run inference.

Commands:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

python -m face_model_core.cli train \
  --data-root ./data \
  --backbone resnet50 \
  --embedding-dim 512 \
  --loss-type arcface \
  --epochs 12 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --checkpoint-dir ./checkpoints

# Resume from the last checkpoint
python -m face_model_core.cli train \
  --data-root ./data \
  --resume-from ./checkpoints/last.pt \
  --epochs 12 \
  --checkpoint-dir ./checkpoints

python -m face_model_core.cli build-gallery \
  --gallery-root ./data/val \
  --checkpoint ./checkpoints/best.pt \
  --output ./artifacts/gallery.npz

python -m face_model_core.cli infer \
  --image ./sample.jpg \
  --checkpoint ./checkpoints/best.pt \
  --gallery ./artifacts/gallery.npz \
  --threshold 0.4
```

## Colab Auto-Download Option

If your dataset is not already organized in Google Drive, `scripts/colab_autorun_train.py` can optionally download from KaggleHub and auto-detect the folder that contains both `train/` and `val/`.

In `scripts/colab_autorun_train.py` set:

- `AUTO_DOWNLOAD_DATASET = True`
- `KAGGLE_DATASET = "hearfool/vggface2"`

Then run the script in Colab. If your Kaggle credentials are required, configure Kaggle access in Colab first.

## Colab Two-Shell Workflow

If you want to split setup and training into two separate Colab cells/scripts:

1. Run `scripts/colab_shell_1_setup.py`
2. Upload or verify `best.pt` / `last.pt` in your checkpoint folder if needed
3. Run `scripts/colab_shell_2_train.py`

Notes:

- Shell 1 writes the resolved dataset root into `/content/face_model_scratch/.colab_resolved_data_root.txt`.
- Shell 2 reads that file and starts training from the same dataset root.
- Shell 2 can auto-resume from `last.pt` if it exists in the configured checkpoint directory.
- If you switch Google accounts (new Drive), upload your previous `last.pt` to `/content/last.pt` in Colab before running shell 2; it will be imported automatically.
- You can also set `RESUME_FROM_URL` in shell 2 to download a checkpoint and continue from it.
