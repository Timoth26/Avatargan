# AvatarGAN

A conditional Generative Adversarial Network (cGAN) that generates cartoon avatars conditioned on discrete facial attributes. Built on the [CartoonSet](https://google.github.io/cartoonset/) dataset.

---

## Overview

AvatarGAN learns to generate 128×128 RGB cartoon faces that match a given combination of facial attributes. Both the Generator and the Discriminator receive attribute embeddings alongside the image/noise input, which guides the model towards attribute-consistent outputs.

## Architecture

| Component | Description |
|---|---|
| **AttributeBlock** | Per-attribute two-layer MLP that refines each embedding before it is concatenated |
| **Generator** | Latent vector `z` + attribute embeddings → fully-connected layers → 128×128 RGB image (Tanh output) |
| **Discriminator** | Flattened image + L2-normalised attribute embeddings → fully-connected layers → real/fake score (Sigmoid) |

## Attributes

Four attributes are used, each encoded as a learned embedding:

| Attribute | Variants |
|---|---|
| `facial_hair` | 15 |
| `hair` | 111 |
| `face_color` | 11 |
| `hair_color` | 10 |

## Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 128 × 128 |
| Latent dimension | 128 |
| Embedding dimension | 64 |
| Batch size | 64 |
| Epochs | 360 |
| Generator LR | 0.0002 |
| Discriminator LR | 0.0001 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Loss | Binary Cross-Entropy |

## Dataset

[CartoonSet10k](https://google.github.io/cartoonset/) — 10,000 cartoon avatar images, each paired with a CSV file describing its visual attributes. Place the dataset at `../cartoonset10k` relative to the notebook.

```
cartoonset10k/
├── 00000.png
├── 00000.csv
├── 00001.png
├── 00001.csv
└── ...
```

## Requirements

```
torch
torchvision
Pillow
matplotlib
```

Install with:

```bash
pip install torch torchvision Pillow matplotlib
```

## Usage

Open and run `avatargan.ipynb` in Jupyter. The notebook is organised into the following sections:

1. **Imports and Configuration** — libraries and global hyperparameters
2. **Dataset** — `CustomImageDataset` with CSV attribute loading
3. **Model Architecture** — `AttributeBlock`, `Generator`, `Discriminator`
4. **Dataset and Model Setup** — dataloader, model initialisation, optimizers
5. **Helper Functions** — checkpoint saving, fixed sample preparation
6. **Training** — main training loop with per-epoch visualisation

Checkpoints are saved every 20 epochs to the `models4/` directory.

## Training Progress

Every epoch a side-by-side comparison of 6 fixed original images and their generated counterparts is displayed, allowing visual tracking of generation quality over time.