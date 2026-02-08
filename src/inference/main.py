from ..ggme.metrics.descriptor_functions import degree_distribution
from ..ggme.metrics.kernels import gaussian_kernel
from ..ggme.evaluate import evaluate_mmd
from torchvision.utils import make_grid
from ..ggme.utils import load_graphs
from PIL import Image
import numpy as np

import pickle
import torch

def generate_graphon():
    with open("../../models/d3pm.pkl", "rb") as f:
        d3pm = pickle.load(f)
    global_step = 0
    N = 8
    d3pm.eval()
    with torch.no_grad():
        # cond = torch.arange(0, 16).cuda() % 10
        init_noise = torch.randint(0, N, (16, 1, 32, 32)).cuda()
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
        return last_img



def sample_graph_from_graphon(W, n):
    # W: [32,32] tensor con valori in [0,1]
    res = W.shape[0]
    # STEP 1: sample latent positions
    u = torch.rand(n)
    # STEP 2: convert to grid indices
    idx = torch.clamp((u * res).long(), max=res-1)
    # STEP 3: compute probability matrix
    p = W[idx[:,None], idx[None,:]]
    # symmetrize (se graph undirected)
    p = (p + p.T) / 2
    p.fill_diagonal_(0)
    # STEP 4: sample adjacency
    A = torch.bernoulli(p)
    return A

def evaluate_graph():
    predicted_graphs = load_graphs("../data/predictions/graphs/CommunityGraphs.pickle")
    test_graphs = load_graphs("../data/test/graphs/CommunityGraphs.pickle")

    mmd = evaluate_mmd(
            graphs_dist_1=test_graphs,
            graphs_dist_2=predicted_graphs,
            function=degree_distribution,
            kernel=gaussian_kernel,
            use_linear_approximation=False,
            density=True,
            sigma=0.1
            )

    print("Option 3:")
    print("The MMD between the test and predicted graphs is {:.4f}.".format(mmd))
    
if __name__ == '__main__':
    test_graphs = load_graphs("../data/test/CommunityGraphs.pickle")
    
    # generate a graphon
    tensor = generate_graphon()
    # sample graph from graphon
    adj_matrix = sample_graph_from_graphon(tensor, 10)
    # use ggme to compute the metric on the generation
    evaluate_graph()