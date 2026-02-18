"""Dataset that loads graphs from CommunityGraphs.pickle (OGAD data format)."""
import os
import pathlib
import pickle

import numpy as np
import torch
import torch_geometric.utils
from scipy.sparse import csr_matrix
from torch_geometric.data import InMemoryDataset, Data

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

try:
    import networkx as nx
except ImportError:
    nx = None


def _graphs_to_adjs(graphs):
    """Convert list of graphs (nx.Graph, np.ndarray, or csr_matrix) to list of numpy adjacency matrices."""
    if not graphs:
        return []
    first = graphs[0]
    if isinstance(first, np.ndarray):
        return [np.asarray(g, dtype=np.float32) for g in graphs]
    if isinstance(first, csr_matrix):
        return [np.asarray(g.toarray(), dtype=np.float32) for g in graphs]
    if nx is not None and isinstance(first, nx.Graph):
        return [np.asarray(nx.to_numpy_array(g), dtype=np.float32) for g in graphs]
    raise ValueError(f"Unsupported graph type: {type(first)}")


class CommunityGraphDataset(InMemoryDataset):
    """Loads graphs from a pickle file and converts to PyG format (same as Spectre)."""

    def __init__(self, split, root, pickle_path, transform=None, pre_transform=None, pre_filter=None,
                 train_ratio=0.8, val_ratio=0.1, seed=0):
        self.split = split
        self.pickle_path = pathlib.Path(pickle_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        """Load graphs from pickle and save train/val/test splits as raw .pt files."""
        with open(self.pickle_path, 'rb') as f:
            graphs = pickle.load(f)
        adjs = _graphs_to_adjs(graphs)
        # Convert to torch tensors for consistency with Spectre raw format
        adjs = [torch.from_numpy(adj).float() for adj in adjs]
        n_total = len(adjs)
        g_cpu = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_total, generator=g_cpu)
        n_test = max(1, int(round(n_total * self.test_ratio)))
        n_val = max(1, int(round(n_total * self.val_ratio)))
        n_train = n_total - n_test - n_val
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        train_data = [adjs[i] for i in train_indices]
        val_data = [adjs[i] for i in val_indices]
        test_data = [adjs[i] for i in test_indices]
        print(f'CommunityGraphs: train {n_train}, val {n_val}, test {n_test}')
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        """Convert raw adjacency lists to PyG Data (same format as Spectre)."""
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        data_list = []
        for adj in raw_dataset:
            if isinstance(adj, torch.Tensor):
                adj = adj.numpy()
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            adj_t = torch.from_numpy(adj).float() if not isinstance(adj, torch.Tensor) else adj
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj_t)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class CommunityGraphDataModule(AbstractDataModule):
    """DataModule for CommunityGraphs.pickle."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        # Pickle path: config can be relative to DiGress root (e.g. ../../../data/...)
        pickle_path = cfg.dataset.get('pickle_path', '../../../data/train/graphs/CommunityGraphs.pickle')
        pickle_path = os.path.normpath(os.path.join(base_path, pickle_path))
        self.pickle_path = pickle_path
        datasets = {
            'train': CommunityGraphDataset(split='train', root=root_path, pickle_path=pickle_path),
            'val': CommunityGraphDataset(split='val', root=root_path, pickle_path=pickle_path),
            'test': CommunityGraphDataset(split='test', root=root_path, pickle_path=pickle_path),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class CommunityGraphDatasetInfos(AbstractDatasetInfos):
    """Dataset infos for CommunityGraphs (same interface as Spectre)."""

    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
