"""Unit tests for graphon quantization and DiGress sample export."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestGraphonUtils(unittest.TestCase):
    def test_quantization_and_dequantization(self):
        import torch

        from src.graphon_utils import dequantize_graphon, quantize_graphon

        graphon = torch.tensor(
            [
                [0.0, 0.24, 1.0],
                [0.24, 0.0, 0.51],
                [1.0, 0.51, 0.0],
            ]
        )
        categories = quantize_graphon(graphon, num_bins=10)
        self.assertEqual(categories.tolist(), [[0, 2, 9], [2, 0, 5], [9, 5, 0]])

        restored = dequantize_graphon(categories, num_bins=10)
        self.assertTrue(torch.allclose(restored, restored.T))
        self.assertTrue(torch.equal(torch.diag(restored), torch.zeros(3)))
        self.assertGreaterEqual(float(restored.min()), 0)
        self.assertLessEqual(float(restored.max()), 1)

    def test_image_discovery_is_numeric_and_split_is_reproducible(self):
        from src.graphon_utils import discover_graphon_images, split_graphon_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("sample_10.png", "sample_2.png", "sample_1.png", "ignored.png"):
                (root / name).touch()

            paths = discover_graphon_images(root)
            self.assertEqual([path.name for path in paths], ["sample_1.png", "sample_2.png", "sample_10.png"])
            first = split_graphon_paths(paths, seed=7)
            second = split_graphon_paths(paths, seed=7)
            self.assertEqual(first, second)
            self.assertEqual({key: len(value) for key, value in first.items()}, {"train": 1, "val": 1, "test": 1})

    def test_generated_samples_have_ggme_shape(self):
        import torch

        from src.graphon_utils import generated_samples_to_graphons

        edge_types = torch.tensor(
            [
                [0, 3, 9],
                [3, 0, 6],
                [9, 6, 0],
            ]
        )
        samples = [[torch.zeros(3, dtype=torch.long), edge_types]] * 2
        graphons = generated_samples_to_graphons(samples, num_bins=10)

        self.assertEqual(tuple(graphons.shape), (2, 1, 3, 3))
        self.assertEqual(graphons.dtype, torch.float32)
        self.assertTrue(torch.allclose(graphons, graphons.transpose(-1, -2)))
        self.assertTrue(torch.equal(graphons.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 1, 3)))

    def test_density_matching_preserves_graphon_contract(self):
        import torch

        from src.graphon_utils import match_graphon_density, off_diagonal_density

        graphons = torch.tensor(
            [
                [
                    [
                        [0.0, 0.9, 0.9],
                        [0.9, 0.0, 0.9],
                        [0.9, 0.9, 0.0],
                    ]
                ]
            ]
        )
        corrected = match_graphon_density(graphons, target_density=0.3)

        self.assertEqual(tuple(corrected.shape), (1, 1, 3, 3))
        self.assertAlmostEqual(float(off_diagonal_density(corrected)[0]), 0.3, places=6)
        self.assertTrue(torch.allclose(corrected, corrected.transpose(-1, -2)))
        self.assertTrue(torch.equal(corrected.diagonal(dim1=-2, dim2=-1), torch.zeros(1, 1, 3)))
        self.assertGreaterEqual(float(corrected.min()), 0)
        self.assertLessEqual(float(corrected.max()), 1)


if __name__ == "__main__":
    unittest.main()
