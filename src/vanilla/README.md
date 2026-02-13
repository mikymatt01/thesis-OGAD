# Vanilla Diffusion Baseline for Graph Generation

A simple DDPM-style diffusion model for generating graphs represented as adjacency matrices.

## Overview

This module implements a minimal baseline diffusion model that operates directly on adjacency matrices. It serves as a clean reference implementation for thesis experiments on graph generation.

## Architecture

### Components

1. **Dataset (`dataset.py`)**: Loads and preprocesses adjacency matrices
2. **Diffusion (`diffusion.py`)**: Implements forward noise addition and reverse denoising
3. **Model (`model.py`)**: Neural networks for denoising (MLP and UNet variants)
4. **Training (`train.py`)**: Training loop and optimization
5. **Sampling (`sample.py`)**: Generation of new adjacency matrices

### Diffusion Process

- **Forward Process**: Progressively adds Gaussian noise to adjacency matrices following a noise schedule
- **Reverse Process**: Iteratively denoises from pure noise to generate clean adjacency matrices
- **Training Objective**: MSE loss between predicted and actual clean matrices

## Usage

### Training

Train on synthetic data (for testing):

```bash
python -m src.vanilla.train \
    --use_synthetic \
    --num_synthetic 1000 \
    --matrix_size 10 \
    --model_type mlp \
    --num_epochs 100 \
    --batch_size 32 \
    --save_dir ./checkpoints
```

Train on real adjacency matrix data:

```bash
python -m src.vanilla.train \
    --data_dir /path/to/adjacency/matrices \
    --matrix_size 20 \
    --model_type unet \
    --num_epochs 200 \
    --batch_size 16 \
    --save_dir ./checkpoints
```

### Sampling

Generate new adjacency matrices from a trained model:

```bash
python -m src.vanilla.sample \
    --checkpoint ./checkpoints/final_model.pt \
    --model_type mlp \
    --matrix_size 10 \
    --num_samples 100 \
    --save_dir ./generated_samples
```

### Arguments

#### Training Arguments

- `--use_synthetic`: Use synthetic random graphs for testing
- `--data_dir`: Directory containing .npy adjacency matrix files
- `--matrix_size`: Size of adjacency matrices (N x N)
- `--model_type`: Model architecture (mlp, unet)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate (default: 1e-4)
- `--save_dir`: Directory to save checkpoints
- `--num_timesteps`: Number of diffusion timesteps (default: 1000)
- `--schedule`: Noise schedule (linear, cosine)

#### Sampling Arguments

- `--checkpoint`: Path to trained model checkpoint
- `--model_type`: Model architecture (must match training)
- `--matrix_size`: Size of matrices (must match training)
- `--num_samples`: Number of graphs to generate
- `--threshold`: Binarization threshold (default: 0.5)
- `--no_symmetric`: Generate directed graphs
- `--save_dir`: Directory to save generated samples

## Data Format

### Input Data

Adjacency matrices should be:
- Stored as `.npy` files (NumPy arrays)
- Square matrices (N x N)
- Binary (0/1) or weighted values
- One matrix per file OR a single file with multiple matrices

Example:
```python
import numpy as np

# Create adjacency matrix
adj_matrix = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

# Save to file
np.save("graph_001.npy", adj_matrix)
```

### Output Data

Generated samples are saved as:
- Individual `.npy` files: `sample_00000.npy`, `sample_00001.npy`, ...
- Combined file: `sample_all.npy` (array of shape `(num_samples, matrix_size, matrix_size)`)

## Model Architectures

### MLP Denoiser

Simple multi-layer perceptron that:
- Flattens adjacency matrix to vector
- Concatenates with sinusoidal timestep embedding
- Processes through fully-connected layers
- Reshapes output back to matrix

Parameters:
- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of hidden layers (default: 3)

### UNet Denoiser

Convolutional U-Net with skip connections:
- Encoder-decoder architecture
- Timestep embeddings integrated in middle layer
- Skip connections preserve spatial information
- Better for larger matrices

Parameters:
- `base_channels`: Base channel multiplier (default: 32)

## Implementation Details

### Noise Schedule

Two schedules available:
1. **Linear**: Linearly interpolates between `beta_start` and `beta_end`
2. **Cosine**: Smoother schedule following cosine function (often better performance)

### Training Objective

Model predicts clean matrix x₀ from noised matrix x_t:

```
Loss = MSE(model(x_t, t), x₀)
```

### Sampling Process

1. Start from pure Gaussian noise
2. Iteratively denoise for T timesteps
3. Post-process (binarize, symmetrize, remove self-loops)
4. Save generated matrices

## Example Workflow

```python
from src.vanilla.dataset import create_synthetic_dataset
from src.vanilla.diffusion import GaussianDiffusion
from src.vanilla.model import create_model
import torch

# Create dataset
dataset = create_synthetic_dataset(num_samples=1000, matrix_size=10)

# Create diffusion process
diffusion = GaussianDiffusion(num_timesteps=1000)

# Create and train model
model = create_model('mlp', matrix_size=10)
# ... training loop ...

# Generate samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
samples = diffusion.p_sample_loop(
    model=model,
    shape=(100, 1, 10, 10),
    device=device
)
```

## Limitations

This is a **baseline implementation** with intentional simplifications:

- Operates on fixed-size matrices (requires padding for variable sizes)
- Does not leverage graph structure (treats matrices as images)
- No advanced graph-specific inductive biases
- Simple architectures (not state-of-the-art)

For more sophisticated graph diffusion models, see literature on:
- Score-based graph generation
- Equivariant diffusion models
- Discrete diffusion on graphs

## Requirements

- Python >= 3.10
- PyTorch >= 1.13.1
- NumPy >= 1.23.5
- tqdm (for progress bars)

All dependencies are included in the parent project's `pyproject.toml`.

## Citation

This implementation is part of a thesis on graph generation. If you use this code, please cite accordingly.
