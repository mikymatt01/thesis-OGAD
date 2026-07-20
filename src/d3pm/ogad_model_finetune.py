from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import ParameterGrid
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from d3pm_runner import D3PM
from dit import DiT_Llama
from training_utils import (
    deterministic_split,
    load_experiment_config,
    load_graph_selection_config,
    update_run_manifest,
)


transform = transforms.ToTensor()

graphons_dataset = load_dataset(
    "imagefolder",
    data_dir="../../data/train/graphons",
    split="train",
)


def apply_transform(example):
    image = example["image"].convert("L")
    tensor = transform(image)
    return {"data": tensor}


gr_dataset = graphons_dataset.map(
    apply_transform, remove_columns=["image"], num_proc=None
)
gr_dataset.set_format(type="torch")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def off_diagonal_density(graphons: torch.Tensor) -> torch.Tensor:
    graphons = torch.as_tensor(graphons, dtype=torch.float32)
    squeeze_result = False
    if graphons.ndim == 2:
        graphons = graphons.unsqueeze(0).unsqueeze(0)
        squeeze_result = True
    elif graphons.ndim == 3:
        graphons = graphons.unsqueeze(1)
    if graphons.ndim != 4 or graphons.shape[1] != 1:
        raise ValueError(f"Expected graphon shape [N,1,R,R], got {tuple(graphons.shape)}")

    mask = ~torch.eye(graphons.shape[-1], dtype=torch.bool, device=graphons.device)
    densities = graphons[:, 0][:, mask].mean(dim=1)
    return densities[0] if squeeze_result else densities


def match_graphon_density(graphons: torch.Tensor, target_density: float, eps: float = 1e-8) -> torch.Tensor:
    if target_density < 0:
        raise ValueError(f"target_density must be non-negative, got {target_density}")

    graphons = torch.as_tensor(graphons, dtype=torch.float32)
    if graphons.ndim != 4 or graphons.shape[1] != 1:
        raise ValueError(f"Expected graphon shape [N,1,R,R], got {tuple(graphons.shape)}")

    densities = off_diagonal_density(graphons).clamp_min(eps)
    scale = torch.as_tensor(target_density, dtype=graphons.dtype, device=graphons.device) / densities
    corrected = (graphons * scale.view(-1, 1, 1, 1)).clamp(0, 1)
    corrected = (corrected + corrected.transpose(-1, -2)) / 2
    corrected.diagonal(dim1=-2, dim2=-1).zero_()
    return corrected.cpu().float()


def mean_dataset_density(dataset) -> float:
    densities = [off_diagonal_density(item["data"]) for item in dataset]
    if not densities:
        raise ValueError("Cannot compute density from an empty graphon dataset")
    return float(torch.stack(densities).mean().item())


def append_metrics_jsonl(run_dir: str | None, payload: dict) -> None:
    if not run_dir:
        return
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(payload, default=str) + "\n")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_batch(item, num_classes: int, image_size: int):
    x = item["data"].to(device)
    if x.shape[-2:] != (image_size, image_size):
        raise ValueError(
            f"Unexpected image size {tuple(x.shape[-2:])}; "
            f"expected {(image_size, image_size)}"
        )
    return (x * (num_classes - 1)).round().long().clamp(0, num_classes - 1)


def train_epoch(d3pm, dataloader, optim, scaler, num_classes: int, image_size: int):
    d3pm.train()
    totals = {"loss": 0.0, "vb_loss": 0.0, "ce_loss": 0.0}
    samples = 0
    progress = tqdm(dataloader)

    for item in progress:
        x_cat = _prepare_batch(item, num_classes, image_size)
        optim.zero_grad()

        with autocast():
            loss, info = d3pm(x_cat)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)
        scaler.step(optim)
        scaler.update()

        batch_size = x_cat.shape[0]
        samples += batch_size
        for key in totals:
            totals[key] += float(info[key]) * batch_size
        progress.set_description(
            "loss: {:.4f}, norm: {:.4f}, vb_loss: {:.4f}, ce_loss: {:.4f}".format(
                totals["loss"] / samples,
                norm,
                totals["vb_loss"] / samples,
                totals["ce_loss"] / samples,
            )
        )

    if samples == 0:
        raise ValueError("Empty D3PM training dataloader")
    return {key: value / samples for key, value in totals.items()}


def validate_epoch(d3pm, dataloader, num_classes: int, image_size: int):
    d3pm.eval()
    totals = {"loss": 0.0, "vb_loss": 0.0, "ce_loss": 0.0}
    samples = 0

    with torch.no_grad():
        for item in dataloader:
            x_cat = _prepare_batch(item, num_classes, image_size)
            _, info = d3pm(x_cat)
            batch_size = x_cat.shape[0]
            samples += batch_size
            for key in totals:
                totals[key] += float(info[key]) * batch_size

    if samples == 0:
        raise ValueError("Empty D3PM validation dataloader")
    return {key: value / samples for key, value in totals.items()}


def _cpu_state_dict(model) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def finetune_model(params, training_config, train_indices, validation_indices):
    num_classes = params["num_classes"]
    batch_size = params["batch_size"]
    dim = params["dim"]
    learning_rate = params["lr"]
    hybrid_loss_coeff = params["hybrid_loss_coeff"]
    num_workers = params["num_workers"]
    image_size = params["image_size"]
    seed = training_config["split_seed"]

    set_reproducible_seed(seed)
    print(f"Params {params}")

    d3pm = D3PM(
        DiT_Llama(1, num_classes, dim=dim),
        1000,
        num_classes=num_classes,
        hybrid_loss_coeff=hybrid_loss_coeff,
    ).to(device)
    print(
        f"Total Param Count: "
        f"{sum(param.numel() for param in d3pm.x0_model.parameters())}"
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        Subset(gr_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        Subset(gr_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_validation_loss = None
    best_epoch = None
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, training_config["max_epochs"] + 1):
        train_metrics = train_epoch(
            d3pm, train_loader, optim, scaler, num_classes, image_size
        )
        validation_metrics = validate_epoch(
            d3pm, validation_loader, num_classes, image_size
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}, "
            f"validation_loss={validation_metrics['loss']:.6f}"
        )

        improved = (
            best_validation_loss is None
            or validation_metrics["loss"]
            < best_validation_loss - training_config["min_delta"]
        )
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state_dict = _cpu_state_dict(d3pm)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state_dict is None:
        raise RuntimeError("D3PM training completed without a valid checkpoint")

    return {
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "state_dict": best_state_dict,
        "history": history,
    }


def save_gzip_torch(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as compressed_file:
        torch.save(payload, compressed_file)


def generate_graphons_from_checkpoint(
    state_dict,
    params,
    out_path: Path,
    seed: int,
    target_density: float | None = None,
) -> dict:
    """Generate sanitized graphon tensors for one trained D3PM candidate."""
    set_reproducible_seed(seed)
    num_classes = params["num_classes"]
    dim = params["dim"]
    image_size = params["image_size"]
    hybrid_loss_coeff = params["hybrid_loss_coeff"]
    d3pm = D3PM(
        DiT_Llama(1, num_classes, dim=dim),
        1000,
        num_classes=num_classes,
        hybrid_loss_coeff=hybrid_loss_coeff,
    ).to(device)
    d3pm.load_state_dict(state_dict)
    d3pm.eval()
    with torch.no_grad():
        init_noise = torch.randint(
            0,
            num_classes,
            (16, 1, image_size, image_size),
            device=device,
        )
        images = d3pm.sample_with_image_sequence(init_noise, None, stride=40)
        graphons = images[-1].float().cpu() / (num_classes - 1)
        graphons = ((graphons + graphons.transpose(-1, -2)) / 2).clamp(0, 1)
        graphons.diagonal(dim1=-2, dim2=-1).zero_()
    density_before = off_diagonal_density(graphons)
    if target_density is not None:
        graphons = match_graphon_density(graphons, target_density)
    density_after = off_diagonal_density(graphons)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphons, out_path)
    return {
        "density_before_mean": float(density_before.mean().item()),
        "density_after_mean": float(density_after.mean().item()),
        "target_density": target_density,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D3PM finetuning with validation and early stopping."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment JSON (d3pm grid + d3pm_training controls).",
    )
    args = parser.parse_args()
    param_grid, training_config = load_experiment_config(args.config)
    graph_selection_config = load_graph_selection_config(args.config)
    target_density = (
        mean_dataset_density(gr_dataset)
        if training_config.get("match_training_density")
        else None
    )
    if target_density is not None:
        print(f"Matching generated graphon density to training mean {target_density:.6f}")
    train_indices, validation_indices = deterministic_split(
        len(gr_dataset),
        training_config["validation_fraction"],
        training_config["split_seed"],
    )

    run_dir = os.environ.get("OGAD_RUN_DIR")
    run_id = os.environ.get("OGAD_RUN_ID")
    split_metadata = {
        "split_seed": training_config["split_seed"],
        "validation_fraction": training_config["validation_fraction"],
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    update_run_manifest(run_dir, {"split": split_metadata, "status": "training"})
    append_metrics_jsonl(
        run_dir,
        {"stage": "d3pm_split", "run_id": run_id, **split_metadata},
    )

    best_result = None
    best_params = None
    candidate_summaries = []
    candidate_manifest = []
    candidate_dir = Path("../../data/models/d3pm_candidates")
    candidate_graphon_dir = Path("../../data/predictions/graphons/candidates")
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    if candidate_graphon_dir.exists():
        shutil.rmtree(candidate_graphon_dir)

    for candidate_index, params in enumerate(ParameterGrid(param_grid)):
        candidate_id = f"candidate_{candidate_index:03d}"
        result = finetune_model(
            params, training_config, train_indices, validation_indices
        )
        checkpoint_path = candidate_dir / f"{candidate_id}_model.pth.gz"
        params_path = candidate_dir / f"{candidate_id}_params.pth.gz"
        graphons_path = candidate_graphon_dir / f"{candidate_id}.pt"
        candidate_metadata = {
            **dict(params),
            "candidate_id": candidate_id,
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"],
            "training": training_config,
            "match_training_density": bool(training_config.get("match_training_density")),
            "target_density": target_density,
            "split_seed": training_config["split_seed"],
        }
        save_gzip_torch(result["state_dict"], checkpoint_path)
        save_gzip_torch(candidate_metadata, params_path)
        generation_stats = generate_graphons_from_checkpoint(
            result["state_dict"],
            dict(params),
            graphons_path,
            seed=training_config["split_seed"],
            target_density=target_density,
        )
        summary = {
            "candidate_id": candidate_id,
            "params": dict(params),
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"],
            "generation_stats": generation_stats,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "params_path": str(params_path.resolve()),
            "graphons_path": str(graphons_path.resolve()),
        }
        candidate_summaries.append(summary)
        candidate_manifest.append(summary)
        append_metrics_jsonl(
            run_dir,
            {
                "stage": "d3pm_candidate",
                "run_id": run_id,
                **summary,
                "history": result["history"],
            },
        )

        if (
            best_result is None
            or result["best_validation_loss"]
            < best_result["best_validation_loss"]
        ):
            best_result = result
            best_params = dict(params)

    if best_result is None or best_params is None:
        raise RuntimeError("The D3PM parameter grid produced no candidate")

    candidate_manifest_path = candidate_dir / "candidates.json"
    candidate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_manifest_path.write_text(
        json.dumps(
            {
                "selection_enabled": bool(graph_selection_config),
                "graph_selection": graph_selection_config,
                "candidates": candidate_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint_metadata = {
        **best_params,
        "best_epoch": best_result["best_epoch"],
        "best_validation_loss": best_result["best_validation_loss"],
        "training": training_config,
        "match_training_density": bool(training_config.get("match_training_density")),
        "target_density": target_density,
        "split_seed": training_config["split_seed"],
    }
    save_gzip_torch(
        best_result["state_dict"],
        Path("../../data/models/best_model_d3pm.pth.gz"),
    )
    save_gzip_torch(
        checkpoint_metadata,
        Path("../../data/models/best_params_d3pm.pth.gz"),
    )

    final_metadata = {
        "status": "complete",
        "split": split_metadata,
        "training_config": training_config,
        "best_params": best_params,
        "best_epoch": best_result["best_epoch"],
        "best_validation_loss": best_result["best_validation_loss"],
        "target_density": target_density,
        "candidates": candidate_summaries,
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
    }
    update_run_manifest(run_dir, final_metadata)
    append_metrics_jsonl(
        run_dir,
        {"stage": "d3pm_selection", "run_id": run_id, **final_metadata},
    )
    print(
        f"Best model {best_params} at epoch {best_result['best_epoch']} "
        f"with validation loss {best_result['best_validation_loss']}"
    )
