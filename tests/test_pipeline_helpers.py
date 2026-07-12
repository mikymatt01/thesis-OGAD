import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "d3pm"))
sys.path.insert(0, str(ROOT / "src" / "ggme"))
sys.path.insert(0, str(ROOT / "src" / "sigl"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_utils import (  # noqa: E402
    build_cluster_bootstrap_plan,
    expected_graphon_count,
)
from graph_split_utils import deterministic_graph_split  # noqa: E402
from graphon_utils import (  # noqa: E402
    cyclic_graphon_index,
    probability_matrix,
    sanitize_graphon_matrix,
    validate_generation_request,
)
from training_utils import (  # noqa: E402
    combine_d3pm_losses,
    deterministic_split,
    load_graph_selection_config,
    load_experiment_config,
)
from selection_utils import digress_target_ratio_scores, select_best_candidate  # noqa: E402
from experiment_grid import assign_gpu, expand_merged_experiment  # noqa: E402
import run_from_merged_parallel as parallel_runner  # noqa: E402


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
        self.assertEqual(config["sigl"]["Res"], [128])
        self.assertEqual(config["sigl"]["n_clusters"], [50])
        self.assertEqual(config["sigl"]["bootstrap_replicates"], [10, 12, 14, 16])
        self.assertEqual(config["sigl"]["bootstrap_seed"], [0])
        self.assertEqual(config["sigl"]["include_original_clusters"], [False])
        self.assertEqual(grid["dim"], [256])
        self.assertEqual(grid["lr"], [3e-5, 5e-5, 7e-5])
        self.assertEqual(grid["num_classes"], [10, 12, 14])
        self.assertEqual(grid["image_size"], [128])
        self.assertEqual(grid["hybrid_loss_coeff"], [0.01])
        self.assertEqual(training["validation_fraction"], 0.2)
        self.assertEqual(training["patience"], 25)
        self.assertEqual(training["max_epochs"], 300)
        graph_selection = load_graph_selection_config(str(config_path))
        self.assertEqual(graph_selection["validation_fraction"], 0.2)
        self.assertEqual(graph_selection["split_seed"], 0)
        self.assertEqual(graph_selection["selection_metric"], "mean_rank")
        self.assertEqual(config["ggme"]["num_generated_graphs"], 100)
        validate_generation_request(
            config["ggme"]["num_generated_graphs"], [45] * 100
        )

    def test_digress_target_experiment_uses_target_selection(self):
        config_path = ROOT / "experiments" / "5_d3pm_digress_target.json"
        grid, training = load_experiment_config(str(config_path))
        graph_selection = load_graph_selection_config(str(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["sigl"]["bootstrap_replicates"], [12, 14])
        self.assertEqual(grid["dim"], [256, 512])
        self.assertEqual(grid["lr"], [1e-5, 2e-5, 3e-5, 5e-5])
        self.assertEqual(grid["num_classes"], [8, 10, 12])
        self.assertEqual(grid["hybrid_loss_coeff"], [0.001, 0.003, 0.01])
        self.assertEqual(training["max_epochs"], 300)
        self.assertEqual(graph_selection["selection_metric"], "digress_target_ratio")


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

    def test_graph_selection_split_is_deterministic_disjoint_and_complete(self):
        first = deterministic_graph_split(10, 0.2, seed=0)
        second = deterministic_graph_split(10, 0.2, seed=0)
        self.assertEqual(first, second)
        model_train_indices, ggme_validation_indices = first
        self.assertEqual(len(ggme_validation_indices), 2)
        self.assertFalse(set(model_train_indices) & set(ggme_validation_indices))
        self.assertEqual(
            sorted(model_train_indices + ggme_validation_indices),
            list(range(10)),
        )


class GGMECandidateSelectionTests(unittest.TestCase):
    def test_mean_rank_selects_best_candidate_and_breaks_ties_with_loss(self):
        candidates = [
            {
                "candidate_id": "a",
                "validation_loss": 0.2,
                "mmd": {
                    "degree_distribution": 1.0,
                    "clustering_coefficient": 1.0,
                    "normalised_laplacian_spectrum": 3.0,
                },
            },
            {
                "candidate_id": "b",
                "validation_loss": 0.1,
                "mmd": {
                    "degree_distribution": 1.0,
                    "clustering_coefficient": 1.0,
                    "normalised_laplacian_spectrum": 3.0,
                },
            },
            {
                "candidate_id": "c",
                "validation_loss": 0.01,
                "mmd": {
                    "degree_distribution": 3.0,
                    "clustering_coefficient": 3.0,
                    "normalised_laplacian_spectrum": 1.0,
                },
            },
        ]
        best, ranked = select_best_candidate(candidates)
        self.assertEqual(best["candidate_id"], "b")
        self.assertLess(ranked[0]["ggme_score"], ranked[-1]["ggme_score"])

    def test_digress_target_ratio_selects_nearest_target_candidate(self):
        candidates = [
            {
                "candidate_id": "degree_only",
                "validation_loss": 0.01,
                "mmd": {
                    "degree_distribution": 0.0125,
                    "clustering_coefficient": 500.0,
                    "normalised_laplacian_spectrum": 10.0,
                },
            },
            {
                "candidate_id": "balanced",
                "validation_loss": 0.2,
                "mmd": {
                    "degree_distribution": 0.025,
                    "clustering_coefficient": 120.0,
                    "normalised_laplacian_spectrum": 2.0,
                },
            },
        ]

        best, ranked = select_best_candidate(
            candidates,
            selection_metric="digress_target_ratio",
        )

        self.assertEqual(best["candidate_id"], "balanced")
        self.assertIn("descriptor_target_ratios", ranked[0])
        self.assertLess(ranked[0]["ggme_score"], ranked[1]["ggme_score"])

    def test_digress_target_ratio_breaks_ties_with_loss(self):
        candidates = [
            {
                "candidate_id": "slow",
                "validation_loss": 0.2,
                "mmd": {
                    "degree_distribution": 0.0125,
                    "clustering_coefficient": 99.7936,
                    "normalised_laplacian_spectrum": 1.8530,
                },
            },
            {
                "candidate_id": "fast",
                "validation_loss": 0.1,
                "mmd": {
                    "degree_distribution": 0.0125,
                    "clustering_coefficient": 99.7936,
                    "normalised_laplacian_spectrum": 1.8530,
                },
            },
        ]

        ranked = digress_target_ratio_scores(candidates)

        self.assertEqual(ranked[0]["candidate_id"], "fast")


class ParallelSchedulerTests(unittest.TestCase):
    def test_merged_expansion_and_gpu_assignment_are_deterministic(self):
        raw = {
            "name": "base",
            "shared": {},
            "sigl": {"Res": [64, 128], "n_clusters": [50]},
            "d3pm": {"image_size": [64], "num_classes": [8]},
        }
        experiments = expand_merged_experiment(raw)
        self.assertEqual(len(experiments), 2)
        self.assertEqual(experiments[0]["name"], "test_nc0")
        self.assertEqual(experiments[0]["d3pm"]["image_size"], [64])
        self.assertEqual(experiments[1]["d3pm"]["image_size"], [128])
        self.assertEqual(assign_gpu(0, ["0", "1"]), "0")
        self.assertEqual(assign_gpu(1, ["0", "1"]), "1")
        self.assertEqual(assign_gpu(2, ["0", "1"]), "0")

    def test_parallel_retry_state_is_written_and_bounded(self):
        original_run_sh = parallel_runner.RUN_SH
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            temp = Path(temp_dir)
            fake_runner = temp / "fake_runner.sh"
            fake_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "marker=\"$OGAD_RUNS_ROOT/retry_marker\"\n"
                "if [[ ! -f \"$marker\" ]]; then\n"
                "  mkdir -p \"$(dirname \"$marker\")\"\n"
                "  touch \"$marker\"\n"
                "  exit 7\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            os.chmod(fake_runner, 0o755)
            parallel_runner.RUN_SH = fake_runner
            try:
                exp_path = temp / "test_nc0.json"
                exp_path.write_text("{}", encoding="utf-8")
                job = {
                    "index": 0,
                    "slug": "nc0",
                    "experiment_path": exp_path,
                    "experiment_rel": exp_path.relative_to(ROOT),
                }
                result = parallel_runner.run_job(
                    job,
                    gpu_id="0",
                    runs_root=temp / "runs",
                    max_retries=1,
                )
            finally:
                parallel_runner.RUN_SH = original_run_sh

            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["attempts"]), 2)
            status = json.loads(
                (temp / "runs" / "queue" / "nc0" / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "complete")

    def test_run_script_uses_run_local_experiment_config(self):
        script_path = ROOT / "scripts" / "run_ogad_experiment.sh"
        script = script_path.read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn('cp "$EXP_ABS" "$RUN_DIR/experiment.json"', script)
        self.assertIn('CONTAINER_CONFIG="/app/runs/current/experiment.json"', script)
        self.assertIn('--config "$CONTAINER_CONFIG"', script)
        self.assertIn('-e "CUDA_VISIBLE_DEVICES=0"', script)
        self.assertIn('device_ids: ["${OGAD_GPU_ID:-0}"]', compose)
        self.assertNotIn('-v "$ROOT/experiments:/app/experiments:ro"', script)
        self.assertNotIn('-v "$EXP_DIR:/app/experiments:ro"', script)


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
