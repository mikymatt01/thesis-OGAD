from sklearn.model_selection import ParameterGrid
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
import argparse
import gzip
import json
import os
from pathlib import Path
import torch

from d3pm_runner import D3PM
from dit import DiT_Llama

transform = transforms.ToTensor()

graphons_dataset = load_dataset(
    "imagefolder",
    data_dir="../../data/train/graphons",
    split="train"
)

def apply_transform(example):
    image = example['image'].convert("L")
    tensor = transform(image)
    return { 'data': tensor }
gr_dataset = graphons_dataset.map(apply_transform, remove_columns=["image"], num_proc=None)
gr_dataset.set_format(type="torch")

DEFAULT_PARAM_GRID = {
    "dim": [512],
    "lr": [1e-5, 2e-5, 5e-5],
    "num_classes": [8],
    "image_size": [128],
    "batch_size": [16, 32],
    "n_epoch": [50],
    "hybrid_loss_coeff": [1e-2, 0.0, 1e-3],
    "num_workers": [0],
}


def _as_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def build_param_grid(config_path: str | None) -> dict:
    """Merge experiment JSON: d3pm section + shared keys lr/batch_size/n_epoch when absent in d3pm."""
    if not config_path:
        config_path = os.environ.get("OGAD_EXPERIMENT_JSON")
    if not config_path:
        return {k: list(v) for k, v in DEFAULT_PARAM_GRID.items()}
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    shared = cfg.get("shared") or {}
    d3pm = cfg.get("d3pm") or {}
    grid = {k: _as_list(v) for k, v in d3pm.items()}
    for key in ("lr", "batch_size", "n_epoch"):
        if key in shared and key not in grid:
            grid[key] = _as_list(shared[key])
    for key, default_vals in DEFAULT_PARAM_GRID.items():
        if key not in grid:
            grid[key] = list(default_vals)
    return grid


def append_metrics_jsonl(run_dir: str | None, payload: dict) -> None:
    if not run_dir:
        return
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

scaler = GradScaler()

def train(d3pm, dataloader, optim, N, image_size: int):
    pbar = tqdm(dataloader)
    loss_ema = None
    
    for item in pbar:
        x = item['data']
        optim.zero_grad()
        x = x.to(device)
        if x.shape[-2:] != (image_size, image_size):
            raise ValueError(
                f"Unexpected image size {tuple(x.shape[-2:])}; expected {(image_size, image_size)}"
            )

        x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)

        with autocast():
            loss, info = d3pm(x_cat)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)
        scaler.step(optim)
        scaler.update()

        with torch.no_grad():
            param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

        pbar.set_description(
            f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
        )
    
    return loss_ema
    
def finetune_model(params):
    N = params['num_classes']  # number of classes for discretized state per pixel
    batch_size = params['batch_size']
    n_epoch = params['n_epoch']
    dim = params['dim']
    lr = params['lr']
    hybrid_loss_coeff = params['hybrid_loss_coeff']
    num_workers = params['num_workers']
    image_size = params['image_size']

    print(f"Params {params}")
    
    d3pm = D3PM(
        DiT_Llama(1, N, input_size=image_size, dim=dim), 1000, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff
    ).to(device)
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")

    dataloader = DataLoader(gr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=lr)
    d3pm.train()

    for _ in range(n_epoch):
        loss_ema = train(
            d3pm,
            dataloader,
            optim,
            N,
            image_size,
        )
        
    return loss_ema, d3pm

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="D3PM finetune with optional experiment JSON grid.")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment JSON (shared + d3pm). Overrides OGAD_EXPERIMENT_JSON if set.",
    )
    args = ap.parse_args()
    param_grid = build_param_grid(args.config)

    best_loss = None
    best_params = None

    for params in ParameterGrid(param_grid):
        loss_ema, _ = finetune_model(params)

        if best_loss is None or best_loss > loss_ema:
            best_loss = loss_ema
            best_params = params

    print(f"Best model {best_params} with loss {best_loss}")

    append_metrics_jsonl(
        os.environ.get("OGAD_RUN_DIR"),
        {
            "stage": "d3pm_finetune",
            "run_id": os.environ.get("OGAD_RUN_ID"),
            "best_loss": best_loss,
            "best_params": dict(best_params) if best_params else None,
        },
    )

    best_params = dict(best_params)
    best_params["n_epoch"] = 200
    loss_ema, best_model = finetune_model(best_params)

    state_dict = best_model.state_dict()
    with gzip.open("../../data/models/best_model_d3pm.pth.gz", "wb", compresslevel=9) as f:
        torch.save(state_dict, f)
    with gzip.open("../../data/models/best_params_d3pm.pth.gz", "wb", compresslevel=9) as f:
        torch.save(best_params, f)

    append_metrics_jsonl(
        os.environ.get("OGAD_RUN_DIR"),
        {
            "stage": "d3pm_finetune_final",
            "run_id": os.environ.get("OGAD_RUN_ID"),
            "n_epoch": best_params.get("n_epoch"),
            "loss_ema": loss_ema,
            "params": best_params,
        },
    )
