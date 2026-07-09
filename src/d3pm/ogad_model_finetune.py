from __future__ import annotations

import argparse
import gzip
import json
import os
import random
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

    for params in ParameterGrid(param_grid):
        result = finetune_model(
            params, training_config, train_indices, validation_indices
        )
        summary = {
            "params": dict(params),
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"],
        }
        candidate_summaries.append(summary)
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

    checkpoint_metadata = {
        **best_params,
        "best_epoch": best_result["best_epoch"],
        "best_validation_loss": best_result["best_validation_loss"],
        "training": training_config,
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
        "candidates": candidate_summaries,
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
