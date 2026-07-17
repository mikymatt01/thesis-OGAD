"""Utilities for discretized graphons used by the DiGress adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image


_GRAPHON_FILE_RE = re.compile(r"^sample_(\d+)\.png$")


def resolve_data_root(data_root: str | Path, digress_root: str | Path | None = None) -> Path:
    """Resolve an OGAD data root relative to the DiGress repository."""
    path = Path(data_root).expanduser()
    if path.is_absolute():
        return path.resolve()

    root = Path(digress_root) if digress_root is not None else Path(__file__).resolve().parents[1]
    return (root / path).resolve()


def discover_graphon_images(graphon_dir: str | Path) -> list[Path]:
    """Return ``sample_<id>.png`` files sorted by numeric id."""
    directory = Path(graphon_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Graphon directory does not exist: {directory}")

    indexed_paths = []
    for path in directory.iterdir():
        match = _GRAPHON_FILE_RE.match(path.name)
        if match is not None and path.is_file():
            indexed_paths.append((int(match.group(1)), path))

    indexed_paths.sort(key=lambda item: item[0])
    paths = [path for _, path in indexed_paths]
    if len(paths) < 3:
        raise ValueError(
            f"At least three graphon images are required in {directory}; found {len(paths)}"
        )
    return paths


def split_graphon_paths(
    paths: Sequence[Path], seed: int = 0
) -> dict[str, list[Path]]:
    """Create deterministic 80/10/10 train, validation and test splits."""
    if len(paths) < 3:
        raise ValueError("At least three graphons are required for train/val/test splits")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(paths), generator=generator).tolist()
    n_val = max(1, int(round(0.1 * len(paths))))
    n_test = max(1, int(round(0.1 * len(paths))))
    n_train = len(paths) - n_val - n_test
    if n_train < 1:
        raise ValueError("The graphon split must contain at least one training sample")

    train_indices = permutation[:n_train]
    val_indices = permutation[n_train:n_train + n_val]
    test_indices = permutation[n_train + n_val:]
    return {
        "train": [paths[index] for index in train_indices],
        "val": [paths[index] for index in val_indices],
        "test": [paths[index] for index in test_indices],
    }


def sanitize_graphon(graphon: torch.Tensor) -> torch.Tensor:
    """Return a finite, symmetric probability matrix with a zero diagonal."""
    graphon = torch.as_tensor(graphon, dtype=torch.float32)
    if graphon.ndim != 2 or graphon.shape[0] != graphon.shape[1]:
        raise ValueError(f"Expected a square graphon matrix, got shape {tuple(graphon.shape)}")
    if not torch.isfinite(graphon).all():
        raise ValueError("Graphon contains non-finite values")

    graphon = ((graphon + graphon.transpose(0, 1)) / 2).clamp(0, 1).clone()
    graphon.fill_diagonal_(0)
    return graphon


def quantize_graphon(graphon: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Map graphon probabilities to categorical values in ``[0, num_bins - 1]``."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    return (sanitize_graphon(graphon) * (num_bins - 1)).round().long()


def dequantize_graphon(categories: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Map categorical edge values back to a sanitized graphon matrix."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    categories = torch.as_tensor(categories)
    if categories.numel() > 0:
        minimum = int(categories.min().item())
        maximum = int(categories.max().item())
        if minimum < 0 or maximum >= num_bins:
            raise ValueError(
                f"Graphon categories must be in [0, {num_bins - 1}], got [{minimum}, {maximum}]"
            )
    return sanitize_graphon(categories.float() / (num_bins - 1))


def off_diagonal_density(graphons: torch.Tensor) -> torch.Tensor:
    """Return mean off-diagonal probability for each graphon."""
    graphons = torch.as_tensor(graphons, dtype=torch.float32)
    squeeze_result = False
    if graphons.ndim == 2:
        graphons = graphons.unsqueeze(0).unsqueeze(0)
        squeeze_result = True
    elif graphons.ndim == 3:
        graphons = graphons.unsqueeze(1)
    elif graphons.ndim != 4:
        raise ValueError(f"Expected graphon shape [R,R], [N,R,R] or [N,1,R,R], got {tuple(graphons.shape)}")

    if graphons.shape[1] != 1 or graphons.shape[-1] != graphons.shape[-2]:
        raise ValueError(f"Expected graphon shape [N,1,R,R], got {tuple(graphons.shape)}")

    resolution = graphons.shape[-1]
    mask = ~torch.eye(resolution, dtype=torch.bool, device=graphons.device)
    density = graphons[:, 0][:, mask].mean(dim=1)
    return density[0] if squeeze_result else density


def mean_graphon_density(paths: Sequence[Path]) -> float:
    """Compute mean off-diagonal density over graphon PNG files."""
    densities = []
    for path in paths:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            pixels = (
                grayscale.get_flattened_data()
                if hasattr(grayscale, "get_flattened_data")
                else grayscale.getdata()
            )
            matrix = torch.tensor(list(pixels), dtype=torch.float32)
            matrix = matrix.reshape(grayscale.height, grayscale.width) / 255
        densities.append(off_diagonal_density(sanitize_graphon(matrix)))
    if not densities:
        raise ValueError("Cannot compute graphon density from an empty path list")
    return float(torch.stack(densities).mean().item())


def match_graphon_density(
    graphons: torch.Tensor, target_density: float, eps: float = 1e-8
) -> torch.Tensor:
    """Scale each graphon so its off-diagonal mean approaches ``target_density``."""
    if target_density < 0:
        raise ValueError(f"target_density must be non-negative, got {target_density}")

    graphons = torch.as_tensor(graphons, dtype=torch.float32)
    if graphons.ndim != 4 or graphons.shape[1] != 1:
        raise ValueError(f"Expected graphon shape [N,1,R,R], got {tuple(graphons.shape)}")

    densities = off_diagonal_density(graphons).clamp_min(eps)
    scale = torch.as_tensor(target_density, dtype=graphons.dtype, device=graphons.device) / densities
    corrected = graphons * scale.view(-1, 1, 1, 1)
    corrected = corrected.clamp(0, 1)
    corrected = (corrected + corrected.transpose(-1, -2)) / 2
    diagonal = corrected.diagonal(dim1=-2, dim2=-1)
    diagonal.zero_()
    return corrected.cpu().float()


def density_summary(densities: torch.Tensor) -> dict[str, float]:
    densities = torch.as_tensor(densities, dtype=torch.float32)
    if densities.numel() == 0:
        raise ValueError("Cannot summarize empty densities")
    return {
        "mean": float(densities.mean().item()),
        "std": float(densities.std(unbiased=False).item()),
        "min": float(densities.min().item()),
        "max": float(densities.max().item()),
    }


def generated_samples_to_graphons(
    samples: Iterable[Sequence[torch.Tensor]], num_bins: int
) -> torch.Tensor:
    """Convert DiGress ``[node_types, edge_types]`` samples to ``[N, 1, R, R]``."""
    graphons = [dequantize_graphon(sample[1], num_bins) for sample in samples]
    if not graphons:
        raise ValueError("No generated DiGress samples were provided")

    resolution = graphons[0].shape
    if any(graphon.shape != resolution for graphon in graphons):
        raise ValueError("Generated graphons do not all have the same resolution")
    return torch.stack(graphons, dim=0).unsqueeze(1).cpu().float()
