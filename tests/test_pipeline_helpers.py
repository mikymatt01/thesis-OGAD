import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "d3pm"))
sys.path.insert(0, str(ROOT / "src" / "ggme"))
sys.path.insert(0, str(ROOT / "src" / "sigl"))

from bootstrap_utils import (  # noqa: E402
    build_cluster_bootstrap_plan,
    expected_graphon_count,
)
from graphon_utils import (  # noqa: E402
    cyclic_graphon_index,
    probability_matrix,
    sanitize_graphon_matrix,
    validate_generation_request,
)
from training_utils import (  # noqa: E402
    combine_d3pm_losses,
    deterministic_split,
    load_experiment_config,
)


class D3PMHelperTests(unittest.TestCase):
    def test_loss_matches_paper_objective(self):
        self.assertAlmostEqual(combine_d3pm_losses(2.0, 3.0, 0.1), 2.3)
        self.assertEqual(combine_d3pm_losses(2.0, 3.0, 0.0), 2.0)

    def test_runner_uses_the_shared_loss_composer(self):
        runner_path = ROOT / "src" / "d3pm" / "d3pm_runner.py"
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "combine_d3pm_losses"
        ]
        self.assertEqual(len(calls), 1)

    def test_split_is_deterministic_disjoint_and_complete(self):
        first = deterministic_split(10, 0.2, 7)
        second = deterministic_split(10, 0.2, 7)
        self.assertEqual(first, second)
        train_indices, validation_indices = first
        self.assertEqual(len(validation_indices), 2)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(
            sorted(train_indices + validation_indices), list(range(10))
        )

    def test_new_config_sections_are_parsed_separately(self):
        config = {
            "d3pm": {
                "dim": [128],
                "num_classes": [8, 16],
                "image_size": [64],
                "lr": [1e-5],
                "batch_size": [16],
                "hybrid_loss_coeff": [0.001],
                "num_workers": [0],
            },
            "d3pm_training": {
                "validation_fraction": 0.25,
                "split_seed": 11,
                "patience": 4,
                "min_delta": 0.01,
                "max_epochs": 12,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            grid, training = load_experiment_config(str(path))

        self.assertNotIn("validation_fraction", grid)
        self.assertEqual(grid["num_classes"], [8, 16])
        self.assertEqual(training["validation_fraction"], 0.25)
        self.assertEqual(training["split_seed"], 11)
        self.assertEqual(training["max_epochs"], 12)

    def test_repository_experiment_is_reduced_for_initial_run(self):
        config_path = ROOT / "experiments" / "test_merged.json"
        grid, training = load_experiment_config(str(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["sigl"]["n_epoch"], [100])
        self.assertEqual(config["sigl"]["Res"], [64])
        self.assertEqual(config["sigl"]["n_clusters"], [50])
        self.assertEqual(config["sigl"]["bootstrap_replicates"], [5])
        self.assertEqual(config["sigl"]["bootstrap_seed"], [0])
        self.assertEqual(config["sigl"]["include_original_clusters"], [False])
        self.assertEqual(grid["dim"], [128])
        self.assertEqual(grid["num_classes"], [16])
        self.assertEqual(grid["image_size"], [64])
        self.assertEqual(training["validation_fraction"], 0.2)
        self.assertEqual(config["ggme"]["num_generated_graphs"], 100)
        validate_generation_request(
            config["ggme"]["num_generated_graphs"], [45] * 100
        )


class SIGLBootstrapTests(unittest.TestCase):
    def test_bootstrap_plan_is_deterministic_and_cluster_local(self):
        clusters = [[0, 1, 2], [3, 4]]
        first = build_cluster_bootstrap_plan(clusters, 3, seed=0)
        second = build_cluster_bootstrap_plan(clusters, 3, seed=0)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

        cluster_sets = [set(indices) for indices in clusters]
        for job in first:
            cluster_id = job["cluster_id"]
            original_graph_indices = job["original_graph_indices"]
            self.assertEqual(len(original_graph_indices), len(clusters[cluster_id]))
            self.assertTrue(set(original_graph_indices).issubset(cluster_sets[cluster_id]))
            self.assertFalse(job["is_original"])

    def test_bootstrap_count_includes_optional_originals(self):
        self.assertEqual(expected_graphon_count(50, 5, False), 250)
        self.assertEqual(expected_graphon_count(50, 5, True), 300)
        self.assertEqual(expected_graphon_count(50, 1, False), 50)

        plan = build_cluster_bootstrap_plan([[0, 1], [2, 3]], 2, seed=0, include_original_clusters=True)
        self.assertEqual(len(plan), 6)
        self.assertEqual(sum(1 for job in plan if job["is_original"]), 2)


class GraphonHelperTests(unittest.TestCase):
    def test_probability_matrix_reuses_one_latent_index_vector(self):
        graphon = np.arange(25).reshape(5, 5)
        indices = np.array([4, 1, 3])
        actual = probability_matrix(graphon, indices)
        expected = graphon[np.ix_(indices, indices)]
        np.testing.assert_array_equal(actual, expected)

    def test_graphon_sanitization_enforces_structural_constraints(self):
        graphon = np.array([[2.0, -0.4, 0.2], [0.8, 0.5, 1.4], [0.6, 0.4, -1.0]])
        sanitized = sanitize_graphon_matrix(graphon)

        np.testing.assert_allclose(sanitized, sanitized.T)
        np.testing.assert_array_equal(np.diag(sanitized), np.zeros(3))
        self.assertGreaterEqual(float(sanitized.min()), 0.0)
        self.assertLessEqual(float(sanitized.max()), 1.0)

    def test_graphons_are_selected_cyclically_for_exact_count(self):
        indices = [cyclic_graphon_index(index, 3) for index in range(8)]
        self.assertEqual(indices, [0, 1, 2, 0, 1, 2, 0, 1])
        validate_generation_request(100, [45] * 100)


if __name__ == "__main__":
    unittest.main()
