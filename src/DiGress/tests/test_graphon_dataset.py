"""Integration tests for the graphon PyG representation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REQUIRED_MODULES = ("torch", "torch_geometric", "torchvision", "pytorch_lightning")
HAS_GRAPH_STACK = all(importlib.util.find_spec(module) is not None for module in REQUIRED_MODULES)


@unittest.skipUnless(HAS_GRAPH_STACK, "DiGress graph dependencies are not installed")
class TestGraphonDataset(unittest.TestCase):
    def _write_graphons(self, root: Path, count: int = 3, resolution: int = 4):
        import numpy as np
        from PIL import Image

        graphon_dir = root / "train" / "graphons"
        graphon_dir.mkdir(parents=True)
        for index in range(count):
            matrix = np.zeros((resolution, resolution), dtype=np.uint8)
            matrix[0, 1] = matrix[1, 0] = 255
            matrix[2, 3] = matrix[3, 2] = 64 + index
            Image.fromarray(matrix, mode="L").save(graphon_dir / f"sample_{index}.png")
        return graphon_dir

    def _cfg(self, data_root: Path):
        return SimpleNamespace(
            dataset=SimpleNamespace(
                data_root=str(data_root),
                graphon_dir="train/graphons",
                output_path="predictions/graphons/digress/tensors.pt",
                stats_output_path="predictions/graphons/digress/stats.json",
                resolution=4,
                num_bins=10,
                quantization="uniform",
                codebook_path=None,
                match_training_density=False,
                split_seed=0,
                pin_memory=False,
            ),
            train=SimpleNamespace(batch_size=1, num_workers=0),
            general=SimpleNamespace(name="graphon-test"),
        )

    def _model_cfg(self, data_root: Path):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "dataset": {
                    "data_root": str(data_root),
                    "graphon_dir": "train/graphons",
                    "output_path": "predictions/graphons/digress/tensors.pt",
                    "stats_output_path": "predictions/graphons/digress/stats.json",
                    "resolution": 4,
                    "num_bins": 10,
                    "quantization": "uniform",
                    "codebook_path": None,
                    "match_training_density": False,
                    "split_seed": 0,
                    "pin_memory": False,
                },
                "general": {
                    "name": "graphon-smoke",
                    "log_every_steps": 1000,
                    "number_chain_steps": 1,
                    "val_loss_repeats": 1,
                },
                "train": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "lr": 0.0002,
                    "weight_decay": 1e-12,
                },
                "model": {
                    "type": "discrete",
                    "diffusion_steps": 3,
                    "diffusion_noise_schedule": "cosine",
                    "transition": "uniform",
                    "n_layers": 1,
                    "lambda_train": [5, 0],
                    "hidden_mlp_dims": {"X": 8, "E": 8, "y": 8},
                    "hidden_dims": {
                        "dx": 8,
                        "de": 8,
                        "dy": 8,
                        "n_head": 1,
                        "dim_ffX": 8,
                        "dim_ffE": 8,
                        "dim_ffy": 8,
                    },
                },
            }
        )

    def test_zero_bin_is_an_explicit_dense_edge_class(self):
        import torch

        from src import utils
        from src.datasets.graphon_dataset import GraphonDataModule, GraphonDatasetInfos

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self._write_graphons(data_root)
            datamodule = GraphonDataModule(self._cfg(data_root))
            item = datamodule.train_dataset[0]

            self.assertEqual(item.edge_index.shape[1], 4 * 3)
            self.assertGreater(float(item.edge_attr[:, 0].sum()), 0)

            batch = torch.zeros(item.x.shape[0], dtype=torch.long)
            dense, node_mask = utils.to_dense(item.x, item.edge_index, item.edge_attr, batch)
            off_diagonal = ~torch.eye(4, dtype=torch.bool)
            dense_edges = dense.E[0][off_diagonal]
            self.assertTrue((dense_edges.sum(dim=-1) == 1).all())
            self.assertTrue((dense_edges[:, 0] == 1).any())
            self.assertTrue((dense.E[0].diagonal(dim1=0, dim2=1) == 0).all())
            self.assertTrue(node_mask.all())

            infos = GraphonDatasetInfos(datamodule, datamodule.cfg.dataset)
            self.assertEqual(infos.edge_types.shape[0], 10)
            self.assertGreater(float(infos.edge_types[0]), 0)
            self.assertAlmostEqual(float(infos.edge_types.sum()), 1.0, places=6)

    def test_sampling_metric_exports_tensor(self):
        import torch

        from src.metrics.graphon_metrics import GraphonSamplingMetrics

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            cfg = self._cfg(data_root)
            metric = GraphonSamplingMetrics(cfg)
            edge_types = torch.tensor(
                [[0, 2, 3, 4], [2, 0, 5, 6], [3, 5, 0, 7], [4, 6, 7, 0]]
            )
            samples = [[torch.zeros(4, dtype=torch.long), edge_types]]

            metric(samples, "graphons", 0, 0, local_rank=1, test=True)
            self.assertFalse(metric.output_path.exists())
            metric(samples, "graphons", 0, 0, local_rank=0, test=True)

            saved = torch.load(metric.output_path)
            self.assertEqual(tuple(saved.shape), (1, 1, 4, 4))
            self.assertEqual(saved.dtype, torch.float32)
            stats = json.loads(metric.stats_output_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["num_graphons"], 1)
            self.assertFalse(stats["match_training_density"])

    def test_sampling_metric_matches_training_density_and_writes_stats(self):
        import torch

        from src.graphon_utils import mean_graphon_density, off_diagonal_density
        from src.metrics.graphon_metrics import GraphonSamplingMetrics

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            graphon_dir = self._write_graphons(data_root)
            cfg = self._cfg(data_root)
            cfg.dataset.match_training_density = True
            metric = GraphonSamplingMetrics(cfg)
            edge_types = torch.tensor(
                [[0, 9, 9, 9], [9, 0, 9, 9], [9, 9, 0, 9], [9, 9, 9, 0]]
            )
            samples = [[torch.zeros(4, dtype=torch.long), edge_types]]

            metric(samples, "graphons", 0, 0, local_rank=0, test=True)

            saved = torch.load(metric.output_path)
            target_density = mean_graphon_density(sorted(graphon_dir.glob("sample_*.png")))
            self.assertAlmostEqual(float(off_diagonal_density(saved)[0]), target_density, places=5)
            self.assertTrue(torch.allclose(saved, saved.transpose(-1, -2)))
            self.assertTrue(torch.equal(saved.diagonal(dim1=-2, dim2=-1), torch.zeros(1, 1, 4)))

            stats = json.loads(metric.stats_output_path.read_text(encoding="utf-8"))
            self.assertTrue(stats["match_training_density"])
            self.assertEqual(stats["num_graphons"], 1)
            self.assertAlmostEqual(stats["target_density"], target_density, places=6)
            self.assertGreater(stats["generated_density_before_mean"], stats["generated_density_after_mean"])

    def test_discrete_model_smoke_training_step_and_sampling(self):
        import torch

        from diffusion.extra_features import DummyExtraFeatures
        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from src.datasets.graphon_dataset import GraphonDataModule, GraphonDatasetInfos
        from src.metrics.graphon_metrics import GraphonSamplingMetrics

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self._write_graphons(data_root)
            cfg = self._model_cfg(data_root)
            datamodule = GraphonDataModule(cfg)
            dataset_infos = GraphonDatasetInfos(datamodule, cfg.dataset)
            extra_features = DummyExtraFeatures()
            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=extra_features,
            )
            model = DiscreteDenoisingDiffusion(
                cfg=cfg,
                dataset_infos=dataset_infos,
                train_metrics=TrainAbstractMetricsDiscrete(),
                sampling_metrics=GraphonSamplingMetrics(cfg),
                visualization_tools=None,
                extra_features=extra_features,
                domain_features=extra_features,
            )

            batch = next(iter(datamodule.train_dataloader()))
            output = model.training_step(batch, 1)
            self.assertTrue(torch.isfinite(output["loss"]))

            samples = model.sample_batch(
                batch_id=0,
                batch_size=1,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                num_nodes=4,
            )
            self.assertEqual(len(samples), 1)
            self.assertEqual(tuple(samples[0][1].shape), (4, 4))
            self.assertTrue(torch.equal(samples[0][1], samples[0][1].T))


@unittest.skipUnless(importlib.util.find_spec("hydra") is not None, "hydra-core is not installed")
class TestGraphonHydraConfig(unittest.TestCase):
    def test_experiment_selects_graphons(self):
        from hydra import compose, initialize_config_dir

        digress_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(digress_root / "configs"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=["+experiment=graphons"])

        self.assertEqual(cfg.dataset.name, "graphons")
        self.assertEqual(cfg.dataset.num_bins, 16)
        self.assertEqual(cfg.dataset.quantization, "codebook")
        self.assertFalse(cfg.dataset.match_training_density)
        self.assertEqual(cfg.dataset.stats_output_path, "predictions/graphons/digress/stats.json")
        self.assertEqual(cfg.model.transition, "marginal")
        self.assertEqual(cfg.model.extra_features, "all")
        self.assertEqual(cfg.general.monitor_metric, "val/E_CE")
        self.assertEqual(cfg.train.accumulate_grad_batches, 1)


if __name__ == "__main__":
    unittest.main()
