import torch
import gzip
import io
from pathlib import Path

from d3pm_runner import D3PM
from dit import DiT_Llama

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def off_diagonal_density(graphons: torch.Tensor) -> torch.Tensor:
    graphons = torch.as_tensor(graphons, dtype=torch.float32)
    if graphons.ndim == 3:
        graphons = graphons.unsqueeze(1)
    if graphons.ndim != 4 or graphons.shape[1] != 1:
        raise ValueError(f"Expected graphon shape [N,1,R,R], got {tuple(graphons.shape)}")
    mask = ~torch.eye(graphons.shape[-1], dtype=torch.bool, device=graphons.device)
    return graphons[:, 0][:, mask].mean(dim=1)


def match_graphon_density(graphons: torch.Tensor, target_density: float, eps: float = 1e-8) -> torch.Tensor:
    densities = off_diagonal_density(graphons).clamp_min(eps)
    scale = torch.as_tensor(target_density, dtype=graphons.dtype, device=graphons.device) / densities
    corrected = (graphons * scale.view(-1, 1, 1, 1)).clamp(0, 1)
    corrected = (corrected + corrected.transpose(-1, -2)) / 2
    corrected.diagonal(dim1=-2, dim2=-1).zero_()
    return corrected.cpu().float()


with gzip.open("../../data/models/best_params_d3pm.pth.gz", "rb") as f:
    buffer = io.BytesIO(f.read())
best_model = torch.load(buffer, map_location=device)
  
with gzip.open("../../data/models/best_model_d3pm.pth.gz", "rb") as f:
    buffer = io.BytesIO(f.read())
state_dict = torch.load(buffer, map_location=device)


if __name__ == "__main__":
    hybrid_loss_coeff = best_model['hybrid_loss_coeff']
    N = best_model['num_classes']
    dim = best_model['dim']
    image_size = best_model.get("image_size", 32)
    
    d3pm = D3PM(
        DiT_Llama(1, N, dim=dim), 1000, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff
    ).to(device)

    d3pm.load_state_dict(state_dict)

    global_step = 0
    d3pm.eval()
    with torch.no_grad():
        init_noise = torch.randint(0, N, (16, 1, image_size, image_size)).to(device)
        images = d3pm.sample_with_image_sequence(
            init_noise, None, stride=40
        )
        image = images[-1]
        this_image = image.float().cpu() / (N - 1)
        this_image = ((this_image + this_image.transpose(-1, -2)) / 2).clamp(0, 1)
        this_image.diagonal(dim1=-2, dim2=-1).zero_()
        target_density = best_model.get("target_density")
        if best_model.get("match_training_density") and target_density is not None:
            before = off_diagonal_density(this_image).mean().item()
            this_image = match_graphon_density(this_image, float(target_density))
            after = off_diagonal_density(this_image).mean().item()
            print(
                "Matched D3PM graphon density from {:.6f} to {:.6f} "
                "(target {:.6f})".format(before, after, float(target_density))
            )
        out_path = Path("../../data/predictions/graphons/tensors.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(this_image, out_path)
