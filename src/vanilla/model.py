"""
Neural network model for denoising adjacency matrices.

This module provides a simple MLP-based architecture for predicting
clean adjacency matrices from noised inputs.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timestep encoding.
    
    Encodes timesteps using sine and cosine functions of different frequencies,
    similar to the position encoding in Transformers.
    
    Args:
        embedding_dim: Dimension of the embedding
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class MLPDenoiser(nn.Module):
    """
    Simple MLP-based denoising model for adjacency matrices.
    
    Takes a noised adjacency matrix and a timestep, and predicts the clean matrix.
    Uses sinusoidal timestep embeddings and several fully-connected layers.
    
    Args:
        matrix_size: Size of the adjacency matrix (N x N)
        hidden_dim: Dimension of hidden layers
        num_layers: Number of hidden layers
        time_embedding_dim: Dimension of timestep embeddings
    """
    
    def __init__(
        self,
        matrix_size: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embedding_dim: int = 128
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(time_embedding_dim)
        
        # Input: flattened matrix (1 * matrix_size * matrix_size) + time embedding
        input_dim = matrix_size * matrix_size + time_embedding_dim
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output: flattened matrix
        layers.append(nn.Linear(hidden_dim, matrix_size * matrix_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noised adjacency matrix (batch_size, 1, matrix_size, matrix_size)
            t: Timestep (batch_size,)
            
        Returns:
            Predicted clean adjacency matrix (batch_size, 1, matrix_size, matrix_size)
        """
        batch_size = x.shape[0]
        
        # Flatten matrix
        x_flat = x.view(batch_size, -1)
        
        # Get timestep embedding
        t_emb = self.time_embed(t)
        
        # Concatenate matrix and timestep
        h = torch.cat([x_flat, t_emb], dim=-1)
        
        # Pass through MLP
        out = self.mlp(h)
        
        # Reshape to matrix
        out = out.view(batch_size, 1, self.matrix_size, self.matrix_size)
        
        return out


class UNetDenoiser(nn.Module):
    """
    Simple U-Net style denoiser for adjacency matrices.
    
    Uses convolutional layers with skip connections to denoise matrices.
    More sophisticated than MLP but still relatively simple.
    
    Args:
        matrix_size: Size of the adjacency matrix (N x N)
        base_channels: Number of channels in the first layer
        time_embedding_dim: Dimension of timestep embeddings
    """
    
    def __init__(
        self,
        matrix_size: int,
        base_channels: int = 32,
        time_embedding_dim: int = 128
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.base_channels = base_channels
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # Encoder (downsampling)
        self.enc1 = self._conv_block(1, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Middle
        self.middle = self._conv_block(base_channels * 4, base_channels * 4)
        
        # Decoder (upsampling with skip connections)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with GroupNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noised adjacency matrix (batch_size, 1, matrix_size, matrix_size)
            t: Timestep (batch_size,)
            
        Returns:
            Predicted clean adjacency matrix (batch_size, 1, matrix_size, matrix_size)
        """
        # Get timestep embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.view(-1, self.base_channels * 4, 1, 1)
        
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Middle with time embedding
        m = self.middle(e3)
        m = m + t_emb  # Add time embedding
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Output
        out = self.out(d1)
        
        return out


def create_model(
    model_type: str,
    matrix_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create denoising models.
    
    Args:
        model_type: Type of model ('mlp' or 'unet')
        matrix_size: Size of adjacency matrix
        **kwargs: Additional arguments for the model
        
    Returns:
        Denoising model
    """
    if model_type == 'mlp':
        return MLPDenoiser(matrix_size, **kwargs)
    elif model_type == 'unet':
        return UNetDenoiser(matrix_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
