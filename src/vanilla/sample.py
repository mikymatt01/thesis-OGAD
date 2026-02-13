"""
Sampling script for generating adjacency matrices.

This module implements sampling from trained diffusion models
to generate new adjacency matrices.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

from .diffusion import GaussianDiffusion
from .model import create_model


def sample_adjacency_matrices(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    num_samples: int,
    matrix_size: int,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate adjacency matrices using the trained diffusion model.
    
    Args:
        model: Trained denoising model
        diffusion: Diffusion process
        num_samples: Number of adjacency matrices to generate
        matrix_size: Size of matrices (N x N)
        device: Device to generate on
        batch_size: Batch size for generation
        verbose: Whether to show progress
        
    Returns:
        Array of generated adjacency matrices (num_samples, matrix_size, matrix_size)
    """
    model.to(device)
    model.eval()
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Generating samples")
    
    with torch.no_grad():
        for i in iterator:
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate samples
            shape = (current_batch_size, 1, matrix_size, matrix_size)
            samples = diffusion.p_sample_loop(
                model=model,
                shape=shape,
                device=device,
                progress=False  # Disable inner progress bar
            )
            
            # Convert to numpy and remove channel dimension
            samples = samples.cpu().numpy()[:, 0, :, :]
            all_samples.append(samples)
    
    # Concatenate all batches
    all_samples = np.concatenate(all_samples, axis=0)
    
    return all_samples


def post_process_adjacency_matrices(
    matrices: np.ndarray,
    threshold: float = 0.5,
    make_symmetric: bool = True,
    remove_self_loops: bool = True
) -> np.ndarray:
    """
    Post-process generated adjacency matrices.
    
    Args:
        matrices: Generated matrices (N, H, W)
        threshold: Threshold for binarization
        make_symmetric: Whether to make matrices symmetric (undirected graphs)
        remove_self_loops: Whether to zero out diagonal
        
    Returns:
        Post-processed matrices
    """
    processed = matrices.copy()
    
    # Binarize if threshold is provided
    if threshold is not None:
        processed = (processed > threshold).astype(np.float32)
    
    # Make symmetric for undirected graphs
    if make_symmetric:
        processed = np.maximum(processed, np.transpose(processed, (0, 2, 1)))
    
    # Remove self-loops
    if remove_self_loops:
        for i in range(len(processed)):
            np.fill_diagonal(processed[i], 0)
    
    return processed


def save_samples(
    samples: np.ndarray,
    save_dir: str,
    prefix: str = "sample"
) -> None:
    """
    Save generated adjacency matrices to disk.
    
    Args:
        samples: Generated matrices (N, H, W)
        save_dir: Directory to save to
        prefix: Prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as individual .npy files
    for i, sample in enumerate(samples):
        filename = os.path.join(save_dir, f"{prefix}_{i:05d}.npy")
        np.save(filename, sample)
    
    # Also save all samples in a single file
    all_samples_path = os.path.join(save_dir, f"{prefix}_all.npy")
    np.save(all_samples_path, samples)
    
    print(f"Saved {len(samples)} samples to {save_dir}")
    print(f"Individual files: {prefix}_XXXXX.npy")
    print(f"All samples: {prefix}_all.npy")


def main():
    """Main sampling function with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate graphs using trained diffusion model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "unet"],
                        help="Type of denoising model")
    parser.add_argument("--matrix_size", type=int, required=True,
                        help="Size of adjacency matrices")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for MLP model")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of layers for MLP model")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channels for UNet model")
    
    # Diffusion arguments
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Start value for beta schedule")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="End value for beta schedule")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"],
                        help="Noise schedule type")
    
    # Sampling arguments
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation")
    parser.add_argument("--save_dir", type=str, default="./generated_samples",
                        help="Directory to save generated samples")
    parser.add_argument("--prefix", type=str, default="sample",
                        help="Prefix for saved files")
    
    # Post-processing arguments
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binarization (set to -1 to disable)")
    parser.add_argument("--no_symmetric", action="store_true",
                        help="Do not make matrices symmetric")
    parser.add_argument("--keep_self_loops", action="store_true",
                        help="Keep self-loops (diagonal entries)")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule
    )
    print(f"Created diffusion process with {args.num_timesteps} timesteps")
    
    # Create model
    model_kwargs = {}
    if args.model_type == "mlp":
        model_kwargs = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers
        }
    elif args.model_type == "unet":
        model_kwargs = {
            "base_channels": args.base_channels
        }
    
    model = create_model(
        args.model_type,
        matrix_size=args.matrix_size,
        **model_kwargs
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    samples = sample_adjacency_matrices(
        model=model,
        diffusion=diffusion,
        num_samples=args.num_samples,
        matrix_size=args.matrix_size,
        device=device,
        batch_size=args.batch_size,
        verbose=True
    )
    
    print(f"Generated samples shape: {samples.shape}")
    print(f"Value range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    # Post-process samples
    print("\nPost-processing samples...")
    threshold = args.threshold if args.threshold >= 0 else None
    samples = post_process_adjacency_matrices(
        samples,
        threshold=threshold,
        make_symmetric=not args.no_symmetric,
        remove_self_loops=not args.keep_self_loops
    )
    
    # Save samples
    print("\nSaving samples...")
    save_samples(samples, args.save_dir, args.prefix)
    
    # Print statistics
    print("\nGenerated graph statistics:")
    num_edges = [np.sum(mat) for mat in samples]
    density = [np.mean(mat) for mat in samples]
    print(f"Average number of edges: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
    print(f"Average density: {np.mean(density):.4f} ± {np.std(density):.4f}")
    
    print("\nSampling complete!")


if __name__ == "__main__":
    main()
