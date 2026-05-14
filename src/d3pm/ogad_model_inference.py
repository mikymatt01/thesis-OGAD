import torchvision.transforms as transforms
from torchvision import transforms
from datasets import load_dataset
import torch
import gzip
import io
from pathlib import Path

from d3pm_runner import D3PM
from dit import DiT_Llama

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.ToTensor()

graphons_dataset = load_dataset(
    "imagefolder",
    data_dir="../../data/train/graphons",
    split="train"
)

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
        DiT_Llama(1, N, input_size=image_size, dim=dim), 1000, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff
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
        out_path = Path("../../data/predictions/graphons/tensors.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(this_image, out_path)