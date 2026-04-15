# MNIST Digit Classification with PyTorch

A feedforward neural network trained on the MNIST handwritten digit dataset, implemented in PyTorch and run on Google Colab with GPU acceleration.

## Overview

This notebook builds and trains a multi-layer neural network to classify handwritten digits (0–9) from the MNIST dataset. The model achieves **~98.2% accuracy** on the test set and **~99.7% accuracy** on the training set.

## Dataset

- **Source:** MNIST via `keras.datasets`
- **Training samples:** 60,000 images
- **Image format:** 28×28 grayscale pixels, flattened to 784 features
- **Classes:** 10 (digits 0–9)
- **Train/Test split:** 80% / 20%

## Model Architecture

A custom feedforward neural network (`MyNN`) with the following layers:

| Layer | Details |
|-------|---------|
| Linear | 784 → 128 |
| BatchNorm1d | 128 |
| ReLU | — |
| Dropout | p = 0.3 |
| Linear | 128 → 64 |
| BatchNorm1d | 64 |
| ReLU | — |
| Dropout | p = 0.3 |
| Linear | 64 → 10 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 100 |
| Loss Function | CrossEntropyLoss |

## Results

| Split | Accuracy |
|-------|----------|
| Training | 99.72% |
| Test | 98.17% |

## Requirements

```
torch
torchvision
keras
scikit-learn
pandas
matplotlib
```

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) with a T4 GPU runtime.
2. Install dependencies if needed.
3. Run all cells in order — data loading, preprocessing, model definition, training, and evaluation.

## Notebook Structure

1. **Imports** — Load libraries (PyTorch, Keras, pandas, matplotlib)
2. **GPU Check** — Detect and use CUDA if available
3. **Data Loading** — Load MNIST, reshape to DataFrame
4. **Visualization** — Plot a 4×4 grid of sample images
5. **Preprocessing** — Train/test split, pixel normalization (÷ 255)
6. **Dataset & DataLoader** — Custom PyTorch `Dataset` and `DataLoader` setup
7. **Model Definition** — Define `MyNN` class
8. **Training** — Run training loop for 100 epochs
9. **Evaluation** — Compute accuracy on test and training sets
