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
    collect_graphon_values,
    discover_graphon_images,
    fit_codebook,
    quantize_graphon,
    resolve_data_root,
    save_codebook,
    split_graphon_paths,
    uniform_codebook,
)


def complete_directed_edge_index(num_nodes: int) -> torch.Tensor:
    """Return every directed pair except self-loops."""
    nodes = torch.arange(num_nodes)
    source = nodes.repeat_interleave(num_nodes)
    target = nodes.repeat(num_nodes)
    mask = source != target
    return torch.stack((source[mask], target[mask]), dim=0)


class GraphonDataset(Dataset):
    def __init__(self, paths: list[Path], resolution: int, codebook: torch.Tensor):
        if resolution < 2:
            raise ValueError(f"resolution must be at least 2, got {resolution}")
        self.paths = list(paths)
        self.resolution = resolution
        self.codebook = codebook
        self.num_bins = int(codebook.numel())
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

        categories = quantize_graphon(graphon, self.codebook)
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
        self.quantization = str(getattr(cfg.dataset, "quantization", "codebook"))

        paths = discover_graphon_images(self.graphon_dir)
        split_paths = split_graphon_paths(paths, seed=int(cfg.dataset.split_seed))
        self.codebook = self._build_codebook(split_paths["train"])
        datasets = {
            split: GraphonDataset(
                split_paths[split], resolution=self.resolution, codebook=self.codebook
            )
            for split in ("train", "val", "test")
        }
        print(
            "Graphons: train {} val {} test {} (resolution={}, bins={}, quantization={})".format(
                len(datasets["train"]),
                len(datasets["val"]),
                len(datasets["test"]),
                self.resolution,
                self.num_bins,
                self.quantization,
            )
        )
        print("Graphon codebook: {}".format(
            [round(level, 5) for level in self.codebook.tolist()]
        ))
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
        self._edge_counts = None

    def _build_codebook(self, train_paths: list[Path]) -> torch.Tensor:
        """Choose the reconstruction levels the diffusion classes stand for.

        Fitted on the training split only, so val/test never leak into the
        quantizer, and persisted next to the generated tensors so a run can be
        decoded after the fact.
        """
        if self.quantization == "uniform":
            codebook = uniform_codebook(self.num_bins)
        elif self.quantization == "codebook":
            codebook = fit_codebook(collect_graphon_values(train_paths), self.num_bins)
        else:
            raise ValueError(
                f"dataset.quantization must be 'codebook' or 'uniform', got {self.quantization!r}"
            )

        codebook_path = getattr(self.cfg.dataset, "codebook_path", None)
        if codebook_path:
            path = Path(codebook_path)
            save_codebook(codebook, path if path.is_absolute() else self.data_root / path)
        return codebook

    def __getitem__(self, item):
        return self.inner[item]

    def edge_counts(self):
        """Marginal over all explicit graphon bins, including bin zero.

        Overriding the abstract version is required, not cosmetic: it derives
        the bin-0 count from *absent* edges, and this dense encoding has none,
        so every bin-0 entry would be discarded.
        """
        if self._edge_counts is not None:
            return self._edge_counts

        counts = torch.zeros(self.num_bins, dtype=torch.float32)
        for data in self.train_dataset:
            counts += data.edge_attr.sum(dim=0)
        if counts.sum() == 0:
            raise ValueError("The graphon training split contains no edge values")

        # A zero-probability class would make the marginal transition model's
        # limit distribution contain a zero, and kl_prior takes its log.
        counts = (counts / counts.sum()).clamp_min(1e-6)
        self._edge_counts = counts / counts.sum()
        return self._edge_counts


class GraphonDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: GraphonDataModule, dataset_config):
        self.datamodule = datamodule
        self.name = "graphons"
        self.codebook = datamodule.codebook
        self.num_bins = int(self.codebook.numel())
        self.n_nodes = torch.zeros(datamodule.resolution + 1)
        self.n_nodes[datamodule.resolution] = 1
        self.node_types = torch.tensor([1.0])
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
