"""Unit tests for the candidate export that feeds GGME's selection stage."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestCheckpointPlan(unittest.TestCase):
    def test_orders_by_score_and_expands_seeds(self):
        from src.graphon_candidates import checkpoint_plan

        callback = SimpleNamespace(
            best_k_models={"b.ckpt": 0.9, "a.ckpt": 0.4, "c.ckpt": 1.5},
            best_model_path="a.ckpt",
        )
        plan = checkpoint_plan(callback, max_checkpoints=2, seeds=3)

        self.assertEqual(len(plan), 6)
        self.assertEqual([p for p, _, _ in plan[:3]], ["a.ckpt"] * 3)
        self.assertEqual([p for p, _, _ in plan[3:]], ["b.ckpt"] * 3)
        self.assertEqual([s for _, _, s in plan[:3]], [0, 1, 2])

    def test_falls_back_to_best_model_path_when_untracked(self):
        from src.graphon_candidates import checkpoint_plan

        callback = SimpleNamespace(best_k_models={}, best_model_path="only.ckpt")
        plan = checkpoint_plan(callback, max_checkpoints=5, seeds=1)
        self.assertEqual([p for p, _, _ in plan], ["only.ckpt"])

    def test_disabled_or_unavailable_yields_no_plan(self):
        from src.graphon_candidates import checkpoint_plan

        callback = SimpleNamespace(best_k_models={"a.ckpt": 1.0}, best_model_path="a.ckpt")
        self.assertEqual(checkpoint_plan(callback, 0, 3), [])
        self.assertEqual(checkpoint_plan(callback, 5, 0), [])
        self.assertEqual(checkpoint_plan(None, 5, 3), [])
        self.assertEqual(
            checkpoint_plan(SimpleNamespace(best_k_models={}, best_model_path=""), 5, 1), []
        )

    def test_epoch_is_parsed_from_the_checkpoint_name(self):
        from src.graphon_candidates import epoch_of

        self.assertEqual(epoch_of("/ckpt/epoch=189.ckpt"), 189)
        self.assertEqual(epoch_of("/ckpt/last.ckpt"), -1)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestGraphSelectionDiscovery(unittest.TestCase):
    def _with_run_dir(self, payload):
        tmp = tempfile.TemporaryDirectory()
        if payload is not None:
            (Path(tmp.name) / "experiment.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        os.environ["OGAD_RUN_DIR"] = tmp.name
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.environ.pop, "OGAD_RUN_DIR", None)

    def test_reads_graph_selection_from_the_experiment_file(self):
        from src.graphon_candidates import experiment_graph_selection

        self._with_run_dir({"graph_selection": {"validation_fraction": 0.2}})
        self.assertEqual(
            experiment_graph_selection(), {"validation_fraction": 0.2}
        )

    def test_absent_section_disables_selection_rather_than_failing(self):
        from src.graphon_candidates import experiment_graph_selection

        self._with_run_dir({"sigl": {}})
        self.assertEqual(experiment_graph_selection(), {})

        self._with_run_dir(None)
        self.assertEqual(experiment_graph_selection(), {})

    def test_unset_run_dir_disables_selection(self):
        from src.graphon_candidates import experiment_graph_selection

        os.environ.pop("OGAD_RUN_DIR", None)
        self.assertEqual(experiment_graph_selection(), {})


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestManifestContract(unittest.TestCase):
    """The manifest must carry every key src/ggme/main.py reads."""

    REQUIRED = {
        "candidate_id",
        "params",
        "best_epoch",
        "best_validation_loss",
        "checkpoint_path",
        "params_path",
        "graphons_path",
    }

    def test_export_writes_a_manifest_ggme_can_consume(self):
        import torch

        from src.graphon_candidates import export_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment.json").write_text(
                json.dumps({"graph_selection": {"validation_fraction": 0.2,
                                                "selection_metric": "mean_rank"}}),
                encoding="utf-8",
            )
            os.environ["OGAD_RUN_DIR"] = str(root)
            self.addCleanup(os.environ.pop, "OGAD_RUN_DIR", None)

            cfg = SimpleNamespace(
                general=SimpleNamespace(candidate_checkpoints=2, candidate_seeds=2,
                                        candidate_samples=4,
                                        final_model_samples_to_generate=100),
                dataset=SimpleNamespace(num_bins=16, quantization="codebook",
                                        candidate_dir="candidates",
                                        candidate_manifest_path="candidates.json"),
                model=SimpleNamespace(transition="marginal", extra_features="all"),
            )
            sampling_metrics = SimpleNamespace(
                output_path=root / "tensors.pt",
                stats_output_path=root / "stats.json",
            )
            callback = SimpleNamespace(
                best_k_models={str(root / "epoch=10.ckpt"): 0.8,
                               str(root / "epoch=20.ckpt"): 0.5},
                best_model_path=str(root / "epoch=20.ckpt"),
            )

            class FakeTrainer:
                """Stands in for Lightning: writes where the metric points."""

                def test(self, model, datamodule=None, ckpt_path=None):
                    torch.save(torch.zeros(1, 1, 4, 4), sampling_metrics.output_path)
                    Path(sampling_metrics.stats_output_path).write_text("{}", encoding="utf-8")

            manifest_path = export_candidates(
                cfg=cfg, trainer=FakeTrainer(), model=None, datamodule=None,
                sampling_metrics=sampling_metrics, checkpoint_callback=callback,
                data_root=root,
            )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertTrue(manifest["selection_enabled"])
            self.assertEqual(manifest["graph_selection"]["selection_metric"], "mean_rank")
            self.assertEqual(len(manifest["candidates"]), 4)
            for candidate in manifest["candidates"]:
                self.assertTrue(self.REQUIRED <= set(candidate))
                self.assertTrue(Path(candidate["graphons_path"]).is_file())
                self.assertTrue(Path(candidate["params_path"]).is_file())
            # best checkpoint first, and the per-candidate sample count applied
            self.assertEqual(manifest["candidates"][0]["best_epoch"], 20)
            self.assertEqual(cfg.general.final_model_samples_to_generate, 4)
            # the canonical output still exists for a selection-disabled pipeline
            self.assertTrue((root / "tensors.pt").is_file())
            self.assertEqual(sampling_metrics.output_path, root / "tensors.pt")

    def test_no_checkpoint_means_no_manifest_and_a_normal_test_run(self):
        from src.graphon_candidates import export_candidates

        cfg = SimpleNamespace(
            general=SimpleNamespace(candidate_checkpoints=0, candidate_seeds=1,
                                    candidate_samples=4,
                                    final_model_samples_to_generate=100),
            dataset=SimpleNamespace(num_bins=16, quantization="codebook",
                                    candidate_dir="candidates",
                                    candidate_manifest_path="candidates.json"),
            model=SimpleNamespace(transition="marginal", extra_features="all"),
        )
        self.assertIsNone(
            export_candidates(cfg=cfg, trainer=None, model=None, datamodule=None,
                              sampling_metrics=None, checkpoint_callback=None,
                              data_root=Path("/nonexistent"))
        )
        self.assertEqual(cfg.general.final_model_samples_to_generate, 100)


if __name__ == "__main__":
    unittest.main()
