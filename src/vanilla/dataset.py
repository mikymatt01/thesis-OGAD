"""
Dataset loader for adjacency matrices.

This module provides utilities to load and preprocess adjacency matrices
for graph generation with diffusion models.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List


class AdjacencyMatrixDataset(Dataset):
    """
    Dataset for adjacency matrices.
    
    Loads adjacency matrices from numpy arrays and provides them as PyTorch tensors.
    Matrices are assumed to be square and of fixed size (or padded to fixed size).
    
    Args:
        adjacency_matrices: List or array of adjacency matrices
        max_size: Maximum matrix size (for padding). If None, uses the max dimension found.
        normalize: Optional per-matrix normalization; disabled to preserve edge weights
    """
    
    def __init__(
        self, 
        adjacency_matrices: np.ndarray,
        max_size: Optional[int] = None,
        normalize: bool = False
    ):
        if isinstance(adjacency_matrices, np.ndarray) and adjacency_matrices.ndim == 2:
            adjacency_matrices = [adjacency_matrices]
        adjacency_matrices = [np.asarray(mat, dtype=np.float32) for mat in adjacency_matrices]
        if not adjacency_matrices:
            raise ValueError("Dataset must contain at least one matrix")
        for mat in adjacency_matrices:
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or not mat.shape[0]:
                raise ValueError("Adjacency matrices must be nonempty and square")
            if not np.isfinite(mat).all():
                raise ValueError("Adjacency matrices must contain finite values")
        self.matrices = adjacency_matrices
        self.normalize = normalize
        
        # Determine max size if not provided
        if max_size is None:
            self.max_size = max(mat.shape[0] for mat in adjacency_matrices)
        else:
            self.max_size = max_size
        
        if self.max_size <= 0 or any(mat.shape[0] > self.max_size for mat in adjacency_matrices):
            raise ValueError("max_size must fit every matrix; truncation is not supported")
        # Pad all matrices to max_size if needed
        self.matrices = self._pad_matrices(adjacency_matrices)
        
    def _pad_matrices(self, matrices: np.ndarray) -> np.ndarray:
        """Pad matrices to max_size x max_size."""
        padded = []
        for mat in matrices:
            h, w = mat.shape
            if h < self.max_size or w < self.max_size:
                padded_mat = np.zeros((self.max_size, self.max_size), dtype=mat.dtype)
                padded_mat[:h, :w] = mat
                padded.append(padded_mat)
            else:
                padded.append(mat)
        return np.array(padded)
    
    def __len__(self) -> int:
        return len(self.matrices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get adjacency matrix as a tensor.
        
        Returns:
            Tensor of shape (1, max_size, max_size) with values in [0, 1] if normalized
        """
        matrix = self.matrices[idx].astype(np.float32)
        
        if self.normalize:
            # Normalize to [0, 1] range
            min_val = matrix.min()
            max_val = matrix.max()
            if max_val > min_val:
                matrix = (matrix - min_val) / (max_val - min_val)
        
        # Convert to tensor and add channel dimension (1, H, W)
        return torch.from_numpy(matrix).unsqueeze(0)


def load_adjacency_matrices_from_dir(
    data_dir: str,
    file_pattern: str = "*.npy",
    max_size: Optional[int] = None,
    normalize: bool = False
) -> AdjacencyMatrixDataset:
    """
    Load adjacency matrices from a directory of .npy files.
    
    Args:
        data_dir: Directory containing .npy files
        file_pattern: Pattern to match files (default: "*.npy")
        max_size: Maximum matrix size for padding
        normalize: Whether to normalize values
        
    Returns:
        AdjacencyMatrixDataset instance
    """
    import glob
    
    files = [data_dir] if os.path.isfile(data_dir) else glob.glob(os.path.join(data_dir, file_pattern))
    matrices = []
    
    for file_path in sorted(files):
        mat = np.load(file_path)
        if mat.ndim == 2:
            matrices.append(mat)
        elif mat.ndim == 3:
            matrices.extend(mat)
        else:
            raise ValueError(f"{file_path}: expected (N,N) or (B,N,N)")
    
    if not matrices:
        raise ValueError(f"No adjacency matrices found in {data_dir}")
    
    return AdjacencyMatrixDataset(matrices, max_size=max_size, normalize=normalize)


def create_synthetic_dataset(
    num_samples: int = 100,
    matrix_size: int = 10,
    edge_prob: float = 0.3,
    directed: bool = False
) -> AdjacencyMatrixDataset:
    """
    Create a synthetic dataset of random adjacency matrices for testing.
    
    Args:
        num_samples: Number of graphs to generate
        matrix_size: Size of adjacency matrices (N x N)
        edge_prob: Probability of an edge existing
        directed: Whether graphs are directed
        
    Returns:
        AdjacencyMatrixDataset with synthetic graphs
    """
    matrices = []
    
    for _ in range(num_samples):
        # Generate random adjacency matrix
        mat = (np.random.rand(matrix_size, matrix_size) < edge_prob).astype(np.float32)
        
        # Make symmetric if undirected
        if not directed:
            mat = np.triu(mat, 1)
            mat = mat + mat.T
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(mat, 0)
        
        matrices.append(mat)
    
    return AdjacencyMatrixDataset(np.array(matrices), normalize=False)


def get_dataloader(
    dataset: AdjacencyMatrixDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for the adjacency matrix dataset.
    
    Args:
        dataset: AdjacencyMatrixDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

