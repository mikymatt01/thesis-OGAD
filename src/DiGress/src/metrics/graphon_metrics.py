"""Sampling output adapter for graphon-valued DiGress experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.graphon_utils import (
    density_summary,
    discover_graphon_images,
    generated_samples_to_graphons,
    match_graphon_density,
    mean_graphon_density,
    off_diagonal_density,
    resolve_data_root,
)


class GraphonSamplingMetrics(nn.Module):
    """Export final categorical samples using the tensor contract expected by GGME."""

    def __init__(self, cfg):
        super().__init__()
        digress_root = Path(__file__).resolve().parents[2]
        data_root = resolve_data_root(cfg.dataset.data_root, digress_root)
        configured_output = Path(cfg.dataset.output_path)
        self.output_path = (
            configured_output
            if configured_output.is_absolute()
            else data_root / configured_output
        )
        configured_stats_output = Path(
            getattr(cfg.dataset, "stats_output_path", "predictions/graphons/digress/stats.json")
        )
        self.stats_output_path = (
            configured_stats_output
            if configured_stats_output.is_absolute()
            else data_root / configured_stats_output
        )
        self.graphon_dir = data_root / cfg.dataset.graphon_dir
        self.num_bins = int(cfg.dataset.num_bins)
        self.match_training_density = bool(getattr(cfg.dataset, "match_training_density", False))

    def reset(self):
        return None

    def _target_density(self) -> float:
        graphon_paths = discover_graphon_images(self.graphon_dir)
        return mean_graphon_density(graphon_paths)

    def _write_stats(self, stats: dict):
        self.stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_output_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    def forward(
        self,
        generated_graphs,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
    ):
        if not test or local_rank != 0:
            return {}

        graphons = generated_samples_to_graphons(generated_graphs, self.num_bins)
        densities_before = off_diagonal_density(graphons)
        target_density = None
        if self.match_training_density:
            target_density = self._target_density()
            graphons = match_graphon_density(graphons, target_density)
        densities_after = off_diagonal_density(graphons)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphons, self.output_path)
        stats = {
            "num_graphons": int(graphons.shape[0]),
            "target_density": target_density,
            "generated_density_before_mean": density_summary(densities_before)["mean"],
            "generated_density_before_std": density_summary(densities_before)["std"],
            "generated_density_after_mean": density_summary(densities_after)["mean"],
            "generated_density_after_std": density_summary(densities_after)["std"],
            "generated_density_before": density_summary(densities_before),
            "generated_density_after": density_summary(densities_after),
            "match_training_density": self.match_training_density,
        }
        self._write_stats(stats)
        print(
            f"Saved {graphons.shape[0]} generated graphons with shape "
            f"{tuple(graphons.shape)} to {self.output_path}"
        )
        return {"graphons/saved": graphons.shape[0]}
