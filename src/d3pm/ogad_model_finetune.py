from sklearn.model_selection import ParameterGrid
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
import torch
import gzip

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

param_grid = {
    "dim": [512],
    "lr": [1e-5, 2e-5, 5e-5],
    "num_classes": [8],
    "batch_size": [16, 32],
    "n_epoch": [50],
    "hybrid_loss_coeff": [1e-2, 0.0, 1e-3],
    "num_workers": [0],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

scaler = GradScaler()

def train(d3pm, dataloader, optim, N):
    pbar = tqdm(dataloader)
    loss_ema = None
    
    for item in pbar:
        x = item['data']
        optim.zero_grad()
        x = x.to(device)

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

    print(f"Params {params}")
    
    d3pm = D3PM(
        DiT_Llama(1, N, dim=dim), 1000, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff
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
            N
        )
        
    return loss_ema, d3pm

if __name__ == '__main__':
    best_loss = None
    best_params = None
    
    for params in ParameterGrid(param_grid): 
        loss_ema, _ = finetune_model(params)

        if best_loss == None or best_loss > loss_ema:
            best_loss = loss_ema
            best_params = params
 
    print(f"Best model {best_params} with loss {best_loss}")
    
    best_params['n_epoch'] = 200
    loss_ema, best_model = finetune_model(best_params)
    
    state_dict = best_model.state_dict()
    with gzip.open("../../data/models/best_model_d3pm.pth.gz", "wb", compresslevel=9) as f:
        torch.save(state_dict, f)
    with gzip.open("../../data/models/best_params_d3pm.pth.gz", "wb", compresslevel=9) as f:
        torch.save(best_params, f)
    # torch.save(best_model.state_dict(), "../../data/models/d3pm_state_dict.pth")
