"""
Training script for vanilla diffusion model.

This module implements the training loop for the diffusion model
on adjacency matrix datasets.

Example: python -m src.vanilla.train --use_synthetic --matrix_size 10
Real input: --data_dir accepts a directory or one .npy batch. Edge weights are
preserved without per-graph scaling. Smaller graphs are zero-padded; generated
graphs retain the fixed padded size. Only NumPy, PyTorch and tqdm are required.

"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .dataset import AdjacencyMatrixDataset, create_synthetic_dataset, get_dataloader
from .diffusion import GaussianDiffusion
from .model import create_model


def train(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    save_interval: int = 10,
    verbose: bool = True
) -> None:
    """
    Train the diffusion model.
    
    Args:
        model: Denoising model
        diffusion: Diffusion process
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N epochs
        verbose: Whether to print progress
    """
    if num_epochs < 1 or save_interval < 1 or len(dataloader) == 0:
        raise ValueError("Require positive epochs, save_interval and a nonempty dataloader")
    model.to(device)
    model.train()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else dataloader
        
        for batch in pbar:
            batch = batch.to(device)
            
            # Compute loss
            loss = diffusion.training_losses(model, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / num_batches
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'model_config': model.config,
                'diffusion_config': diffusion.config,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            if verbose:
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'diffusion_config': diffusion.config,
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    if verbose:
        print(f"Saved final model to {final_path}")


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train vanilla diffusion model for graph generation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory of .npy files, or a single (B,N,N) .npy file")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic dataset for testing")
    parser.add_argument("--num_synthetic", type=int, default=1000,
                        help="Number of synthetic graphs to generate")
    parser.add_argument("--matrix_size", type=int, default=10,
                        help="Size of adjacency matrices")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "unet"],
                        help="Type of denoising model")
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
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create or load dataset
    if args.use_synthetic or args.data_dir is None:
        print(f"Creating synthetic dataset with {args.num_synthetic} graphs...")
        dataset = create_synthetic_dataset(
            num_samples=args.num_synthetic,
            matrix_size=args.matrix_size,
            edge_prob=0.3
        )
    else:
        print(f"Loading dataset from {args.data_dir}...")
        from .dataset import load_adjacency_matrices_from_dir
        dataset = load_adjacency_matrices_from_dir(
            args.data_dir,
            max_size=args.matrix_size
        )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Matrix size: {dataset.max_size} x {dataset.max_size}")
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
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
        matrix_size=dataset.max_size,
        **model_kwargs
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created {args.model_type.upper()} model with {num_params:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    print(f"\nStarting training for {args.num_epochs} epochs...")
    train(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        verbose=True
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

