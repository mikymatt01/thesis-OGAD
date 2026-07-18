"""Utilities for discretized graphons used by the DiGress adapter."""

from __future__ import annotations

import json
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


def uniform_codebook(num_bins: int) -> torch.Tensor:
    """Evenly spaced reconstruction levels covering ``[0, 1]``."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    return torch.linspace(0.0, 1.0, num_bins, dtype=torch.float32)


def _histogram(values: torch.Tensor, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin ``values`` on ``[0, 1]`` and drop empty bins."""
    weights = torch.histc(values, bins=resolution, min=0.0, max=1.0)
    centers = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    occupied = weights > 0
    return weights[occupied], centers[occupied]


def fit_codebook(
    values: torch.Tensor,
    num_bins: int,
    histogram_resolution: int = 4096,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Fit ``num_bins`` reconstruction levels to the empirical value distribution.

    Graphon probabilities are extremely skewed: on community graphs roughly half
    the mass sits at (near) zero and the inter-community probability is ~0.01.
    Evenly spaced levels round that 0.01 down to exactly 0, which makes generated
    graphs disconnect into one component per community. A Lloyd-Max quantizer
    spends its levels where the mass actually is, so both the near-zero and the
    intra-community modes survive the round trip at the same class count.

    Runs Lloyd's algorithm on a histogram of the values, so cost is independent
    of how many pixels were passed in.
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")

    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Cannot fit a codebook to an empty value tensor")
    if not torch.isfinite(values).all():
        raise ValueError("Graphon values contain non-finite entries")

    weights, centers = _histogram(values.clamp(0, 1), histogram_resolution)
    if centers.numel() <= num_bins:
        # Fewer occupied levels than classes. The surplus classes cannot carry
        # training mass whatever we do, so just keep the codebook strictly
        # increasing; edge_counts smooths the resulting zero marginals.
        return _pad_levels(centers, num_bins)

    codebook = _initial_codebook(weights, centers, num_bins)
    for _ in range(max_iterations):
        assignment = torch.bucketize(centers, (codebook[1:] + codebook[:-1]) / 2)
        mass = torch.zeros(num_bins).index_add_(0, assignment, weights)
        total = torch.zeros(num_bins).index_add_(0, assignment, weights * centers)
        updated = torch.where(mass > 0, total / mass.clamp_min(1e-12), codebook)
        if bool((updated - codebook).abs().max() < tolerance):
            codebook = updated
            break
        codebook = updated

    codebook = torch.sort(codebook)[0]
    _validate_codebook(codebook, weights, centers)
    return codebook


def _pad_levels(levels: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Grow a level set to ``num_bins`` by halving its widest gaps."""
    values = sorted(set(torch.unique(levels).tolist()))
    while len(values) < num_bins:
        anchors = sorted({0.0, *values, 1.0})
        width, index = max(
            (anchors[i + 1] - anchors[i], i) for i in range(len(anchors) - 1)
        )
        midpoint = (anchors[index] + anchors[index + 1]) / 2
        if width <= 1e-6 or midpoint in values:
            raise ValueError(
                f"Cannot build {num_bins} distinct levels from {len(values)} graphon values; "
                f"lower dataset.num_bins"
            )
        values = sorted(set(values + [midpoint]))
    return torch.tensor(values[:num_bins], dtype=torch.float32)


def _initial_codebook(
    weights: torch.Tensor, centers: torch.Tensor, num_bins: int
) -> torch.Tensor:
    """Seed Lloyd's algorithm with distinct, mass-weighted levels.

    Quantile seeding alone collapses onto the zero spike, leaving empty classes;
    LBG-style splitting of the highest-distortion cluster keeps them distinct.
    """
    cdf = torch.cumsum(weights, dim=0) / weights.sum()
    quantiles = (torch.arange(num_bins, dtype=torch.float32) + 0.5) / num_bins
    seeded = centers[torch.searchsorted(cdf, quantiles).clamp(max=centers.numel() - 1)]
    codebook = torch.unique(seeded)

    while codebook.numel() < num_bins:
        assignment = torch.bucketize(centers, (codebook[1:] + codebook[:-1]) / 2)
        distortion = torch.zeros(codebook.numel()).index_add_(
            0, assignment, weights * (centers - codebook[assignment]) ** 2
        )
        worst = int(distortion.argmax())
        members = centers[assignment == worst]
        if members.numel() < 2:
            # Degenerate cluster: nudge a new level halfway to the next one.
            upper = codebook[worst + 1] if worst + 1 < codebook.numel() else torch.tensor(1.0)
            candidate = (codebook[worst] + upper) / 2
        else:
            candidate = members[members != codebook[worst]].median()
        codebook = torch.unique(torch.cat([codebook, candidate.reshape(1)]))

    return torch.sort(codebook)[0]


def _validate_codebook(
    codebook: torch.Tensor, weights: torch.Tensor, centers: torch.Tensor
) -> None:
    """Reject codebooks with an unreachable class.

    An empty class would make its edge-type marginal zero, which produces a
    log(0) in the marginal transition model.
    """
    assignment = torch.bucketize(centers, (codebook[1:] + codebook[:-1]) / 2)
    mass = torch.zeros(codebook.numel()).index_add_(0, assignment, weights)
    empty = (mass <= 0).nonzero().flatten().tolist()
    if empty:
        raise ValueError(
            f"Codebook classes {empty} carry no training mass; "
            f"reduce dataset.num_bins below {codebook.numel()}"
        )


def as_codebook(codebook) -> torch.Tensor:
    """Coerce a config/JSON codebook into a validated 1-D float tensor."""
    tensor = torch.as_tensor(list(codebook) if not torch.is_tensor(codebook) else codebook,
                             dtype=torch.float32).flatten()
    if tensor.numel() < 2:
        raise ValueError(f"A codebook needs at least two levels, got {tensor.numel()}")
    if bool((tensor[1:] <= tensor[:-1]).any()):
        raise ValueError("Codebook levels must be strictly increasing")
    if float(tensor.min()) < 0 or float(tensor.max()) > 1:
        raise ValueError("Codebook levels must lie in [0, 1]")
    return tensor


def quantize_graphon(graphon: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map graphon probabilities to the nearest codebook class."""
    codebook = as_codebook(codebook)
    thresholds = (codebook[1:] + codebook[:-1]) / 2
    return torch.bucketize(sanitize_graphon(graphon), thresholds).long()


def dequantize_graphon(categories: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map categorical edge values back to a sanitized graphon matrix."""
    codebook = as_codebook(codebook)
    categories = torch.as_tensor(categories)
    if categories.numel() > 0:
        minimum = int(categories.min().item())
        maximum = int(categories.max().item())
        if minimum < 0 or maximum >= codebook.numel():
            raise ValueError(
                f"Graphon categories must be in [0, {codebook.numel() - 1}], "
                f"got [{minimum}, {maximum}]"
            )
    return sanitize_graphon(codebook.to(categories.device)[categories.long()])


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
    densities = [off_diagonal_density(load_graphon_image(path)) for path in paths]
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


def load_graphon_image(path: str | Path) -> torch.Tensor:
    """Read a grayscale graphon PNG as a sanitized probability matrix."""
    with Image.open(path) as image:
        grayscale = image.convert("L")
        matrix = torch.tensor(list(grayscale.getdata()), dtype=torch.float32)
        matrix = matrix.reshape(grayscale.height, grayscale.width) / 255
    return sanitize_graphon(matrix)


def collect_graphon_values(paths: Sequence[Path]) -> torch.Tensor:
    """Flatten the off-diagonal probabilities of every graphon in ``paths``."""
    values = []
    for path in paths:
        matrix = load_graphon_image(path)
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
        values.append(matrix[mask])
    if not values:
        raise ValueError("Cannot collect graphon values from an empty path list")
    return torch.cat(values)


def save_codebook(codebook: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_codebook(codebook).tolist()), encoding="utf-8")


def load_codebook(path: str | Path) -> torch.Tensor:
    return as_codebook(json.loads(Path(path).read_text(encoding="utf-8")))


def generated_samples_to_graphons(
    samples: Iterable[Sequence[torch.Tensor]], codebook: torch.Tensor
) -> torch.Tensor:
    """Convert DiGress ``[node_types, edge_types]`` samples to ``[N, 1, R, R]``."""
    graphons = [dequantize_graphon(sample[1], codebook) for sample in samples]
    if not graphons:
        raise ValueError("No generated DiGress samples were provided")

    resolution = graphons[0].shape
    if any(graphon.shape != resolution for graphon in graphons):
        raise ValueError("Generated graphons do not all have the same resolution")
    return torch.stack(graphons, dim=0).unsqueeze(1).cpu().float()
