import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
# import Image
import torch
import gzip
import io

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

dataloader = DataLoader(
    gr_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

with gzip.open("../../data/models/best_params_d3pm.pth.gz", "rb") as f:
    buffer = io.BytesIO(f.read())

best_model = torch.load(buffer)
    
hybrid_loss_coeff = best_model['hybrid_loss_coeff']
N = best_model['num_classes']
dim = best_model['dim']

d3pm = D3PM(
    DiT_Llama(1, N, dim=dim), 1000, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff
) #.cuda()

with gzip.open("../../data/models/best_model_d3pm.pth.gz", "rb") as f:
    buffer = io.BytesIO(f.read())

state_dict = torch.load(buffer)
d3pm.load_state_dict(state_dict)

global_step = 0
d3pm.eval()
with torch.no_grad():
    # cond = torch.arange(0, 16).cuda() % 10
    init_noise = torch.randint(0, N, (16, 1, 32, 32)) #.cuda()
    images = d3pm.sample_with_image_sequence(
        init_noise, None, stride=40
    )
    # image sequences to gif
    gif = []
    for image in images:
        this_image = image.float().cpu() / (N - 1)
        all_images = torch.cat([this_image], dim=0)
        x_as_image = make_grid(all_images, nrow=4)
        img = x_as_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))

    gif[0].save(
        f"contents/sample_{global_step}.gif",
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )

    last_img = gif[-1]
    last_img.save(f"contents/sample_{global_step}_last.png")
