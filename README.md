# RallyTemPose — Badminton Stroke Prediction

A Transformer-based deep learning framework for **next-stroke prediction** in badminton rallies. Given the pose sequences and court positions of both players up to the current shot, the model predicts the type of the next stroke. The architecture couples a spatio-temporal pose encoder (**TemPoseIII**) with a sequence decoder (**Decoder**) that is conditioned on rich text descriptions of stroke types via a frozen BERT embedding layer.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [TemPoseIII Encoder](#temposeiii-encoder)
  - [Sequence Decoder](#sequence-decoder)
  - [RallyTemPose (Full Model)](#rallytemposeiii-full-model)
- [Dataset](#dataset)
  - [Stroke Classes (ShuttleSet)](#stroke-classes-shuttleset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Training](#training)
  - [Command-Line Arguments](#command-line-arguments)
  - [Resuming / Fine-tuning](#resuming--fine-tuning)
- [Output & Checkpoints](#output--checkpoints)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Implementation Details](#key-implementation-details)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Badminton is a turn-based sport where each rally consists of a sequence of strokes alternating between two players. This project frames rally understanding as a **sequence prediction problem**: given all observed strokes (poses + court positions) in a rally so far, predict which stroke type the next player will hit.

Key highlights:
- **Factorised spatio-temporal attention**: separate spatial and temporal Transformer stacks operate on skeleton keypoints and rally timestamps respectively.
- **Cross-attention interaction module**: models the interaction between the two players (the active hitter and the reactive opponent) at each shot.
- **Player identity embedding**: a learnable lookup table encodes each player's identity so the model can exploit player-specific tendencies.
- **BERT-conditioned decoder**: stroke-type token embeddings are initialised from BERT text representations of stroke descriptions, grounding the prediction space in natural language semantics.
- **Auxiliary probe loss**: an auxiliary linear probe on encoder representations provides an extra training signal, improving generalisation.
- **Cosine-annealing LR with warmup** and optional gradient clipping / gradient accumulation (pseudo-large-batch training).

---

## Architecture

### TemPoseIII Encoder

The encoder operates in two stages:

1. **Spatial stage** – For every shot in the rally, the 2-D skeleton keypoints of both players are projected into a joint embedding space and processed by `N_spat` Spatial Transformer blocks with learnable positional encoding. A cross-attention block further models the opponent's reactive pose against the active hitter.

2. **Temporal Convolutional Network (TCN)** – Court position trajectories for each player are processed by two separate dilated 1-D TCNs to extract local and global motion context before being fused with the spatial features.

3. **Temporal stage** – The fused per-shot tokens are enriched with learnable temporal positional encoding and passed through `N_temp` Temporal Transformer blocks. A second cross-attention stream captures the inter-player dynamics along the time axis.

4. **Grouped Pooling Block (GPB)** – An adaptive max-pooling operation over grouped player streams produces compact rally-level representations.

### Sequence Decoder

A standard causal Transformer decoder that:
- Embeds each stroke-type token using BERT token embeddings (from `bert-base-uncased`) projected to the model dimension.
- Adds learnable positional encoding for the rally position.
- Injects the **player identity** of the current hitter as an additive embedding.
- Attends to the encoder output via cross-attention.
- Outputs logits over the 10 stroke classes.

### RallyTemPose (Full Model)

`RallyTemPose` wraps the encoder and decoder, handles causal masking (triangular mask for the target sequence and temporal padding mask for the source), and exposes:
- `forward(src, trg, ID, temp, ...)` — returns class logits (+ aux logits if `prob_bool=True`)
- `predict(...)` — returns argmax class indices
- `predict_sample(...)` — samples from the predicted distribution
- `get_probs(...)` — returns full softmax probabilities
- `extract_attention(...)` — returns all four attention maps (key-spatial, temporal, self-decoder, cross-decoder) for interpretability

---

## Dataset

The model is trained and evaluated on the **ShuttleSet** dataset — a publicly available professional badminton match dataset containing:
- **Pose sequences**: 2-player skeleton keypoints (17 COCO joints → converted to a standard 16-joint representation) at each shot.
- **Court positions**: 2-D normalised court coordinates of both players.
- **Stroke labels**: Shot-type annotations per stroke.
- **Match metadata** (`match.csv`): winner, loser, court-side orientation, set information — used to resolve player identities.

Data files are stored under `Data/`:

| File | Description |
|------|-------------|
| `merged23_poses.pkl` | Pose sequences (merged dataset) |
| `merged23_labels.pkl` | Stroke label arrays |
| `merged23_positions.pkl` | Court position arrays |
| `shuttleset/match.csv` | Match metadata |
| `shuttleset/{poses,labels,positions}.pkl` | Raw ShuttleSet files |

### Stroke Classes (ShuttleSet)

| ID | Stroke Type | Chinese Label(s) |
|----|-------------|-----------------|
| 0 | Net Shot / Cross-court Net Shot | 放小球, 勾球 |
| 1 | Defensive Block / Lob / Drive | 擋小球, 防守回挑, 防守回抽 |
| 2 | Smash / Wrist Smash | 殺球, 點扣 |
| 3 | Lob / Lift | 挑球 |
| 4 | Clear | 長球 |
| 5 | Drive | 平球, 小平球, 後場抽平球 |
| 6 | Drop | 切球, 過度切球 |
| 7 | Push / Rush | 推球, 撲球 |
| 8 | Serve (short & long) | 發短球, 發長球 |
| 9 | Unknown | 未知球種 |

---

## Repository Structure

```
stroke-prediction/
│
├── run.py                    # Main training & evaluation script
├── main_req.txt              # Full pip requirements
│
├── Models/
│   └── nn_models.py          # All model classes:
│                             #   SelfAttention, TransformerBlock,
│                             #   TemPoseIII, TemPoseII,
│                             #   Decoder, RallyTemPose, …
│
├── Utils/
│   ├── tools.py              # Dataset class (PoseData_Forecast),
│   │                         # normalisation, LR scheduling,
│   │                         # checkpoint save/load, accuracy helpers
│   └── playerID_utils.py     # Player-ID encoding, data grouping,
│                             #   train/val/test splitting,
│                             #   keypoint format conversion & smoothing
│
├── Data/
│   ├── merged23_poses.pkl
│   ├── merged23_labels.pkl
│   ├── merged23_positions.pkl
│   └── shuttleset/
│       ├── match.csv
│       ├── poses.pkl
│       ├── labels.pkl
│       └── positions.pkl
│
├── best_model.pt             # Best model checkpoint (by val accuracy)
├── training_log.csv          # Per-epoch training / val / test metrics
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd stroke-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r main_req.txt
```

> **GPU**: PyTorch with CUDA is strongly recommended. The script automatically detects and uses the available GPU (`cuda`) or falls back to CPU.

---

## Training

```bash
python run.py \
  --T_layers 2 \
  --N_layers 2 \
  --file_save best_model.pt \
  --split 12 \
  --dropout_a 0.3 \
  --dropout_e 0.3 \
  --drop_path 0.0 \
  --aux_r 0.3 \
  --LR 5e-5 \
  --clip_grad 1 \
  --pseudo_bs 4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--T_layers` | int | 2 | Number of temporal Transformer encoder layers |
| `--N_layers` | int | 2 | Number of decoder (interaction) Transformer layers |
| `--file_save` | str | `best_model.pt` | Output checkpoint filename |
| `--split` | int | 12 | Random seed for train/val/test splitting |
| `--dropout_a` | float | 0.3 | Attention dropout rate |
| `--dropout_e` | float | 0.3 | Embedding / projection dropout rate |
| `--drop_path` | float | 0.0 | Stochastic depth (drop-path) rate |
| `--aux_r` | float | 0.3 | Weight of the auxiliary probe loss |
| `--LR` | float | 5e-5 | Peak learning rate |
| `--clip_grad` | int | 1 | Enable gradient clipping (max norm = 2.0) |
| `--pseudo_bs` | int | 4 | Gradient accumulation steps (effective batch size multiplier) |
| `--pretrained_weights` | str | None | Path to a `.pt` checkpoint to load |
| `--resume_optimizer` | int | 1 | Load optimizer state from checkpoint (1 = yes) |
| `--resume_training` | int | 0 | Resume epoch counter from checkpoint (1 = yes) |
| `--reset_optimizer` | int | 0 | Force-reset optimizer even when loading weights (1 = yes) |

### Resuming / Fine-tuning

To resume training from a saved checkpoint:

```bash
python run.py \
  --pretrained_weights best_model.pt \
  --resume_training 1 \
  --resume_optimizer 1
```

To fine-tune from a pre-trained model with a fresh optimiser:

```bash
python run.py \
  --pretrained_weights best_model.pt \
  --resume_training 0 \
  --reset_optimizer 1 \
  --LR 1e-5
```

---

## Output & Checkpoints

- **`best_model.pt`** (or the name given via `--file_save`): saved whenever the validation top-1 accuracy improves. The checkpoint contains:
  - `state_dict` — model weights
  - `optimizer` — optimiser state
  - `best_accuracy` — best validation accuracy achieved
  - `epoch` — epoch at which the checkpoint was saved

- **`training_log.csv`**: a CSV file logging per-epoch metrics every `print_int` (25) epochs:

  | Column | Description |
  |--------|-------------|
  | `epoch` | Epoch number |
  | `train_acc` | Training top-1 accuracy |
  | `train_loss` | Training loss |
  | `val_acc` | Validation top-1 accuracy |
  | `val_top2` | Validation top-2 accuracy |
  | `val_top3` | Validation top-3 accuracy |
  | `val_loss` | Validation loss |
  | `test_acc` | Test top-1 accuracy |
  | `test_top2` | Test top-2 accuracy |
  | `test_top3` | Test top-3 accuracy |

---

## Evaluation Metrics

The model is evaluated using three accuracy metrics (ignoring shots whose target label is the "unknown" class, index 9, during validation):

- **Top-1 Accuracy** — fraction of shots where the predicted stroke equals the ground truth.
- **Top-2 Accuracy** — fraction of shots where the ground truth is within the top-2 predicted classes.
- **Top-3 Accuracy** — fraction of shots where the ground truth is within the top-3 predicted classes.

---

## Key Implementation Details

| Feature | Details |
|---------|---------|
| **Skeleton format** | COCO-17 keypoints converted to a standard 16-joint skeleton (`convert_AlphaOpenposeCoco_to_standard16Joint`) |
| **Keypoint smoothing** | Optional Savitzky–Golay filtering (window=5, poly=2) along the time axis |
| **Normalisation** | Per-skeleton Euclidean normalisation of keypoint coordinates |
| **Court position scaling** | X: `/300`, Y: `/700` (court dimensions in pixels) |
| **Sequence length** | Rallies truncated / sub-sampled to `max_len = 30` shots |
| **LR schedule** | Linear warmup (30% of epochs) followed by cosine annealing down to `1e-6` |
| **Positional encoding** | Learnable (initialised with σ=0.02 normal) for both spatial and temporal axes |
| **Player splitting** | Data is split proportionally at match level (`prop_match_test_train_split`) to avoid data leakage across train / val / test |
| **Early stopping** | Patience of 200 epochs on validation loss (after warmup) |

---

## Dependencies

All dependencies are listed in `main_req.txt`. Key packages:

| Package | Role |
|---------|------|
| `torch` | Deep learning framework |
| `einops` | Tensor rearrangement |
| `transformers` | BERT tokeniser & model (`bert-base-uncased`) |
| `scikit-learn` | Metrics & train/test splitting |
| `scipy` | Savitzky–Golay keypoint smoothing |
| `pandas` | Data loading and logging |
| `numpy` | Numerical operations |
| `optuna` | Hyperparameter search (utility, not used in `run.py`) |
| `tensorboard` | (Optional) training visualisation |

---

## License

See [LICENSE](LICENSE) for details.


