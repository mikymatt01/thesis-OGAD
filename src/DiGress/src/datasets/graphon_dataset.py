"""PyG dataset for categorical diffusion over discretized graphons."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision.transforms.functional import pil_to_tensor

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.graphon_utils import (
    discover_graphon_images,
    quantize_graphon,
    resolve_data_root,
    split_graphon_paths,
)


def complete_directed_edge_index(num_nodes: int) -> torch.Tensor:
    """Return every directed pair except self-loops."""
    nodes = torch.arange(num_nodes)
    source = nodes.repeat_interleave(num_nodes)
    target = nodes.repeat(num_nodes)
    mask = source != target
    return torch.stack((source[mask], target[mask]), dim=0)


class GraphonDataset(Dataset):
    def __init__(self, paths: list[Path], resolution: int, num_bins: int):
        if resolution < 2:
            raise ValueError(f"resolution must be at least 2, got {resolution}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        self.paths = list(paths)
        self.resolution = resolution
        self.num_bins = num_bins
        self.edge_index = complete_directed_edge_index(resolution)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Data:
        path = self.paths[index]
        with Image.open(path) as image:
            graphon = pil_to_tensor(image.convert("L")).squeeze(0).float() / 255

        expected_shape = (self.resolution, self.resolution)
        if tuple(graphon.shape) != expected_shape:
            raise ValueError(
                f"Unexpected graphon shape in {path}: {tuple(graphon.shape)}; "
                f"expected {expected_shape}"
            )

        categories = quantize_graphon(graphon, self.num_bins)
        source, target = self.edge_index
        edge_attr = F.one_hot(
            categories[source, target], num_classes=self.num_bins
        ).float()

        return Data(
            x=torch.ones(self.resolution, 1, dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=torch.zeros(1, 0, dtype=torch.float32),
            n_nodes=torch.tensor([self.resolution], dtype=torch.long),
        )


class GraphonDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        digress_root = Path(__file__).resolve().parents[2]
        self.data_root = resolve_data_root(cfg.dataset.data_root, digress_root)
        self.graphon_dir = self.data_root / cfg.dataset.graphon_dir
        self.resolution = int(cfg.dataset.resolution)
        self.num_bins = int(cfg.dataset.num_bins)

        paths = discover_graphon_images(self.graphon_dir)
        split_paths = split_graphon_paths(paths, seed=int(cfg.dataset.split_seed))
        datasets = {
            split: GraphonDataset(
                split_paths[split], resolution=self.resolution, num_bins=self.num_bins
            )
            for split in ("train", "val", "test")
        }
        print(
            "Graphons: train {} val {} test {} (resolution={}, bins={})".format(
                len(datasets["train"]),
                len(datasets["val"]),
                len(datasets["test"]),
                self.resolution,
                self.num_bins,
            )
        )
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

    def edge_counts(self):
        """Count all explicit graphon bins, including bin zero."""
        counts = torch.zeros(self.num_bins, dtype=torch.float32)
        for data in self.train_dataset:
            counts += data.edge_attr.sum(dim=0)
        if counts.sum() == 0:
            raise ValueError("The graphon training split contains no edge values")
        return counts / counts.sum()


class GraphonDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: GraphonDataModule, dataset_config):
        self.datamodule = datamodule
        self.name = "graphons"
        self.num_bins = int(dataset_config.num_bins)
        self.n_nodes = torch.zeros(datamodule.resolution + 1)
        self.n_nodes[datamodule.resolution] = 1
        self.node_types = torch.tensor([1.0])
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
