"""Emit several candidate graphon sets so GGME can select one on validation data.

The D3PM branch trains a hyperparameter grid and lets GGME rank every candidate
against a held-out validation split, then copies the winner into tensors.pt.
The DiGress branch produced a single unselected sample, so the two were being
compared as "best of 72" against "one draw" - on one recorded run the median
D3PM candidate scored worse than the single DiGress sample.

Candidates here come from the top-k checkpoints ModelCheckpoint already keeps,
times a few sampling seeds, so no extra training is needed: only extra sampling.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch

EPOCH_RE = re.compile(r"epoch=(\d+)")


def resolve_under(data_root: Path, configured, default: str) -> Path:
    path = Path(configured or default)
    return path if path.is_absolute() else data_root / path


def experiment_graph_selection() -> dict:
    """Read graph_selection from the OGAD experiment JSON, as the D3PM stage does.

    Keeping the experiment file as the single source of truth avoids a second
    place where selection can be enabled in one stage but not the other.
    """
    run_dir = os.environ.get("OGAD_RUN_DIR")
    if not run_dir:
        return {}
    path = Path(run_dir) / "experiment.json"
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return config.get("graph_selection") or {}


def checkpoint_plan(checkpoint_callback, max_checkpoints: int, seeds: int):
    """Return ``[(checkpoint_path, monitored_score, seed), ...]``, best first."""
    if checkpoint_callback is None or max_checkpoints < 1 or seeds < 1:
        return []

    tracked = getattr(checkpoint_callback, "best_k_models", None) or {}
    ranked = sorted((float(score), path) for path, score in tracked.items())
    if not ranked:
        fallback = getattr(checkpoint_callback, "best_model_path", "")
        if not fallback:
            return []
        ranked = [(float("nan"), fallback)]

    return [
        (path, score, seed)
        for score, path in ranked[:max_checkpoints]
        for seed in range(seeds)
    ]


def epoch_of(checkpoint_path: str) -> int:
    match = EPOCH_RE.search(Path(checkpoint_path).name)
    return int(match.group(1)) if match else -1


def export_candidates(
    cfg,
    trainer,
    model,
    datamodule,
    sampling_metrics,
    checkpoint_callback,
    data_root: Path,
) -> Path | None:
    """Sample one graphon set per (checkpoint, seed) and write GGME's manifest.

    Returns the manifest path, or None when candidate export is disabled or no
    checkpoint is available (in which case the caller should test normally).
    """
    plan = checkpoint_plan(
        checkpoint_callback,
        int(getattr(cfg.general, "candidate_checkpoints", 0)),
        int(getattr(cfg.general, "candidate_seeds", 1)),
    )
    if not plan:
        return None

    candidate_dir = resolve_under(
        data_root,
        getattr(cfg.dataset, "candidate_dir", None),
        "predictions/graphons/digress/candidates",
    )
    manifest_path = resolve_under(
        data_root,
        getattr(cfg.dataset, "candidate_manifest_path", None),
        "predictions/graphons/digress/candidates.json",
    )
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    samples = int(getattr(cfg.general, "candidate_samples", 0))
    if samples > 0:
        cfg.general.final_model_samples_to_generate = samples

    canonical_output = sampling_metrics.output_path
    canonical_stats = sampling_metrics.stats_output_path
    candidates = []

    for index, (checkpoint_path, score, seed) in enumerate(plan):
        candidate_id = f"candidate_{index:03d}"
        graphons_path = candidate_dir / f"{candidate_id}.pt"
        params = {
            "checkpoint": Path(checkpoint_path).name,
            "sampling_seed": seed,
            "num_bins": int(cfg.dataset.num_bins),
            "quantization": str(getattr(cfg.dataset, "quantization", "codebook")),
            "transition": str(cfg.model.transition),
            "extra_features": str(cfg.model.extra_features),
        }
        params_path = candidate_dir / f"{candidate_id}_params.json"
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

        sampling_metrics.output_path = graphons_path
        sampling_metrics.stats_output_path = candidate_dir / f"{candidate_id}_stats.json"
        # Seeds are what makes two candidates from the same checkpoint differ.
        torch.manual_seed(seed)
        print(f"[candidates] {candidate_id}: {Path(checkpoint_path).name} seed={seed}")
        trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        if not graphons_path.is_file():
            raise RuntimeError(
                f"{candidate_id} produced no graphons at {graphons_path}; "
                "GraphonSamplingMetrics only writes on local_rank 0, so candidate "
                "export needs general.gpus=1"
            )

        candidates.append({
            "candidate_id": candidate_id,
            "params": params,
            "best_epoch": epoch_of(checkpoint_path),
            "best_validation_loss": None if score != score else score,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "params_path": str(params_path.resolve()),
            "graphons_path": str(graphons_path.resolve()),
        })

    sampling_metrics.output_path = canonical_output
    sampling_metrics.stats_output_path = canonical_stats

    # The pipeline must still work when selection is off, so the best-scoring
    # candidate is what tensors.pt holds unless GGME overwrites it.
    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidates[0]["graphons_path"], canonical_output)
    first_stats = candidate_dir / "candidate_000_stats.json"
    if first_stats.is_file():
        shutil.copyfile(first_stats, canonical_stats)

    graph_selection = experiment_graph_selection()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "selection_enabled": bool(graph_selection),
                "graph_selection": graph_selection,
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[candidates] wrote {len(candidates)} candidates to {manifest_path} "
        f"(selection_enabled={bool(graph_selection)})"
    )
    return manifest_path
