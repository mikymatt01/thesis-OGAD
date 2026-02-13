"""
Diffusion process implementation for adjacency matrices.

This module implements the forward and reverse diffusion processes
following the DDPM (Denoising Diffusion Probabilistic Models) framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class GaussianDiffusion:
    """
    Gaussian diffusion process for adjacency matrices.
    
    Implements:
    - Forward process: progressively adds Gaussian noise to data
    - Reverse process: denoises from noise to data using a learned model
    - Simple linear noise schedule
    
    Args:
        num_timesteps: Number of diffusion timesteps (T)
        beta_start: Starting value of beta (noise level at t=0)
        beta_end: Ending value of beta (noise level at t=T)
        schedule: Type of noise schedule ('linear', 'cosine')
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'linear'
    ):
        self.num_timesteps = num_timesteps
        
        # Create noise schedule
        if schedule == 'linear':
            self.betas = self._linear_schedule(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Convert to tensors for GPU usage
        self.register_tensors()
    
    def _linear_schedule(self, beta_start: float, beta_end: float, num_timesteps: int) -> np.ndarray:
        """Linear noise schedule."""
        return np.linspace(beta_start, beta_end, num_timesteps)
    
    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine noise schedule as proposed in improved DDPM."""
        steps = num_timesteps + 1
        t = np.linspace(0, num_timesteps, steps)
        alphas_cumprod = np.cos(((t / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    
    def register_tensors(self):
        """Convert numpy arrays to tensors (will be moved to device when needed)."""
        self.betas_t = torch.from_numpy(self.betas).float()
        self.alphas_t = torch.from_numpy(self.alphas).float()
        self.alphas_cumprod_t = torch.from_numpy(self.alphas_cumprod).float()
        self.alphas_cumprod_prev_t = torch.from_numpy(self.alphas_cumprod_prev).float()
        self.posterior_variance_t = torch.from_numpy(self.posterior_variance).float()
        
        # sqrt terms for forward diffusion
        self.sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod_t)
        self.sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod_t)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: sample from q(x_t | x_0).
        
        Args:
            x_start: Clean data (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional pre-generated noise
            
        Returns:
            Noised data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Move tensors to same device as x_start
        device = x_start.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod_t.to(device)[t]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod_t.to(device)[t]
        
        # Reshape for broadcasting: (B, 1, 1, 1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)
        
        # Apply noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Reverse diffusion: sample from p(x_{t-1} | x_t).
        
        Args:
            model: Denoising model that predicts x_0 from x_t
            x_t: Noised data at timestep t
            t: Current timestep
            clip_denoised: Whether to clip predictions to [0, 1]
            
        Returns:
            Denoised sample x_{t-1}
        """
        device = x_t.device
        
        # Predict x_0 from x_t
        x_0_pred = model(x_t, t)
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
        
        # Get parameters for this timestep
        alpha_t = self.alphas_t.to(device)[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod_t.to(device)[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev_t.to(device)[t].view(-1, 1, 1, 1)
        beta_t = self.betas_t.to(device)[t].view(-1, 1, 1, 1)
        
        # Compute mean of p(x_{t-1} | x_t, x_0)
        # Using formula from DDPM paper
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t)
        
        # Predict noise from x_t and x_0_pred
        pred_noise = (x_t - torch.sqrt(alpha_cumprod_t) * x_0_pred) / sqrt_one_minus_alpha_cumprod
        
        # Compute x_{t-1}
        mean = (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) * x_t +
                sqrt_alpha_cumprod_prev * beta_t * x_0_pred) / (1.0 - alpha_cumprod_t)
        
        if t[0] == 0:
            # No noise at t=0
            return mean
        else:
            variance = self.posterior_variance_t.to(device)[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling loop.
        
        Generates samples from pure noise by iteratively denoising.
        
        Args:
            model: Denoising model
            shape: Shape of samples to generate (B, C, H, W)
            device: Device to generate on
            progress: Whether to show progress bar
            
        Returns:
            Generated samples (B, C, H, W)
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps = reversed(range(self.num_timesteps))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t_idx in timesteps:
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        
        return x
    
    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        The model is trained to predict x_0 from x_t.
        Loss is MSE between predicted and actual x_0.
        
        Args:
            model: Denoising model
            x_start: Clean data
            t: Timesteps (if None, sampled uniformly)
            noise: Noise to add (if None, sampled from Gaussian)
            
        Returns:
            MSE loss
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to get x_t
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict x_0 from x_t
        x_0_pred = model(x_t, t)
        
        # MSE loss between predicted and actual x_0
        loss = nn.functional.mse_loss(x_0_pred, x_start)
        
        return loss
