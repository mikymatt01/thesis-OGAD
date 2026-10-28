from __future__ import annotations

import pickle
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from src.baselines.classical_graphs.models import (
    MODEL_CHOICES,
    fit_model,
    generate_baseline_graphs,
    load_graphs,
)


ROOT = Path(__file__).resolve().parents[1]


def _hashes(graphs):
    return [
        tuple(sorted((min(u, v), max(u, v)) for u, v in graph.edges()))
        for graph in graphs
    ]


def _assert_simple_outputs(test_case, graphs, count):
    test_case.assertEqual(len(graphs), count)
    for graph in graphs:
        test_case.assertIs(type(graph), nx.Graph)
        test_case.assertFalse(any(u == v for u, v in graph.edges()))


def _training_graphs():
    return [
        nx.planted_partition_graph(2, 8, 0.75, 0.02, seed=1),
        nx.planted_partition_graph(2, 10, 0.70, 0.03, seed=2),
        nx.planted_partition_graph(2, 9, 0.80, 0.01, seed=3),
    ]


class ClassicalGraphBaselineTests(unittest.TestCase):
    def test_classical_baselines_generate_simple_deterministic_graphs(self):
        train_graphs = _training_graphs()
        test_graphs = [nx.path_graph(16), nx.path_graph(20)]
        for model_name in MODEL_CHOICES:
            first = generate_baseline_graphs(
                model_name=model_name,
                train_graphs=train_graphs,
                test_graphs=test_graphs,
                num_graphs=8,
                seed=42,
            )
            second = generate_baseline_graphs(
                model_name=model_name,
                train_graphs=train_graphs,
                test_graphs=test_graphs,
                num_graphs=8,
                seed=42,
            )
            changed = generate_baseline_graphs(
                model_name=model_name,
                train_graphs=train_graphs,
                test_graphs=test_graphs,
                num_graphs=8,
                seed=43,
            )
            _assert_simple_outputs(self, first, 8)
            self.assertEqual(_hashes(first), _hashes(second))
            self.assertNotEqual(_hashes(first), _hashes(changed))

    def test_erdos_renyi_fit_estimates_valid_probability(self):
        model = fit_model("erdos_renyi", [nx.complete_graph(4), nx.empty_graph(4)])
        self.assertGreaterEqual(model.edge_probability, 0.0)
        self.assertLessEqual(model.edge_probability, 1.0)
        self.assertEqual(model.edge_probability, 0.5)

    def test_sbm_fit_finds_multiple_blocks_for_clear_communities(self):
        graph = nx.planted_partition_graph(2, 12, 0.9, 0.0, seed=7)
        model = fit_model("sbm", [graph])
        self.assertGreaterEqual(len(model.sbm_templates[0].block_sizes), 2)

    def test_degree_corrected_sbm_preserves_more_degree_heterogeneity_than_plain_sbm(self):
        graph = nx.disjoint_union(nx.star_graph(9), nx.star_graph(9))
        graph.add_edge(0, 10)
        train_graphs = [graph]
        test_graphs = [nx.path_graph(20)]
        sbm_graphs = generate_baseline_graphs(
            model_name="sbm",
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            num_graphs=25,
            seed=5,
        )
        dc_graphs = generate_baseline_graphs(
            model_name="degree_corrected_sbm",
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            num_graphs=25,
            seed=5,
        )

        def mean_degree_std(graphs):
            return float(np.mean([np.std([degree for _, degree in graph.degree()]) for graph in graphs]))

        self.assertGreater(mean_degree_std(dc_graphs), mean_degree_std(sbm_graphs))

    def test_classical_baseline_cli_writes_all_pickles(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            temp = Path(temp_dir)
            train_path = temp / "train.pkl"
            test_path = temp / "test.pkl"
            output_dir = temp / "baselines"
            with train_path.open("wb") as file:
                pickle.dump(_training_graphs(), file)
            with test_path.open("wb") as file:
                pickle.dump([nx.path_graph(16), nx.path_graph(20)], file)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.baselines.classical_graphs",
                    "--model",
                    "all",
                    "--train-graphs",
                    str(train_path),
                    "--test-graphs",
                    str(test_path),
                    "--output-dir",
                    str(output_dir),
                    "--num-graphs",
                    "6",
                    "--seed",
                    "11",
                ],
                cwd=ROOT,
                check=True,
            )

            for model_name in MODEL_CHOICES:
                graphs = load_graphs(output_dir / f"{model_name}.pickle")
                _assert_simple_outputs(self, graphs, 6)

    def test_classical_baseline_cli_reads_experiment_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            temp = Path(temp_dir)
            train_path = temp / "train.pkl"
            test_path = temp / "test.pkl"
            output_dir = temp / "classical"
            config_path = temp / "experiment.json"
            with train_path.open("wb") as file:
                pickle.dump(_training_graphs(), file)
            with test_path.open("wb") as file:
                pickle.dump([nx.path_graph(16), nx.path_graph(20)], file)
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "train_graphs_path": str(train_path),
                            "test_graphs_path": str(test_path),
                            "generated_graph_count": 4,
                        },
                        "baselines": {
                            "enabled": True,
                            "models": ["sbm", "degree_corrected_sbm"],
                            "output_dir": str(output_dir),
                            "seed": 9,
                        },
                        "ggme": {"num_generated_graphs": 4},
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.baselines.classical_graphs",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                check=True,
            )

            self.assertFalse((output_dir / "erdos_renyi.pickle").exists())
            for model_name in ("sbm", "degree_corrected_sbm"):
                graphs = load_graphs(output_dir / f"{model_name}.pickle")
                _assert_simple_outputs(self, graphs, 4)


if __name__ == "__main__":
    unittest.main()
