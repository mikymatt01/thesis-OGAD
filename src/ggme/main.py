from evaluate import evaluate_mmd
from correlation import compute_correlation
from correlation import select_kernel_and_hyperparameters

from metrics.kernels import linear_kernel
from metrics.kernels import laplacian_total_variation_kernel
from metrics.kernels import gaussian_emd
from metrics.kernels import gaussian_kernel

from metrics.descriptor_functions import degree_distribution
from metrics.descriptor_functions import clustering_coefficient
from metrics.descriptor_functions import normalised_laplacian_spectrum

from perturb_graphs import perturb_graphs
from utils import load_graphs, load_digress_generated_samples
import networkx as nx
import torch
import os

def sample_graph_from_graphon(W, M):
    # W_matrix: torch tensor NxN con valori in [0,1]
    N = W.shape[0]
    # campiona valori latenti uniformi
    u = torch.rand(M)
    v = torch.rand(M)
    # mappa su indici della griglia W
    idx_u = (u * N).long().clamp(0, N-1)
    idx_v = (v * N).long().clamp(0, N-1)
    # costruisci la matrice di probabilità
    probs = W[idx_u][:, idx_v]   # risultato MxM
    # campiona gli edges
    A = torch.bernoulli(probs)
    # se vuoi grafo non orientato senza self-loop:
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    return A

def generate_graphs(graphons):
    generated_graphs = []
    for graphon in graphons:
        graphon = graphon[0]
        for i in range(0, 10):
            adj_matrix = sample_graph_from_graphon(graphon, 32)
            generated_graphs.append(nx.from_numpy_array(adj_matrix.numpy()))
    return generated_graphs
    
if __name__ == "__main__":
    print("path for: ", os.getcwd())
    
    # take a set of graphs and perturb them for a given perturbation
    test_graphs = load_graphs("../../data/test/graphs/CommunityGraphs.pickle")
    print("esempio test graphs: ", len(test_graphs))
    perturbation_values, perturbed_graphs = perturb_graphs(
            graphs=test_graphs,
            perturbation_type="AddEdges"
            )
    # OPTION 1: to find the best parameters for a grid of kernel/hyperparameter
    # combination, run the following function, which will return the
    # kernel, sigma, and correlation of the best configuration
    best_params = select_kernel_and_hyperparameters(
        test_graphs=test_graphs,
        perturbed_graphs=perturbed_graphs,
        perturbation_values=perturbation_values,
        kernels=[gaussian_kernel, linear_kernel],
        sigmas=[0.1, 1, 10],
        descriptor_fn=degree_distribution,
        correlation_type="pearson"
        )
    print("Option 1:")
    print("The best kernel-hyperparameter combination is the {} with sigma={}. This had a correlation of {:.4f}."
        .format(
        best_params["kernel"].__name__,
        best_params["sigma"],
        best_params["corr"]
        ))
    best_kernel = best_params["kernel"].__name__
    best_sigma = best_params["sigma"]
    best_corr = best_params["corr"]
    print("======================")

    # OPTION 2: to compute the correlation for a single kernel/param combo,
    # take perturbed graphs and true graphs and calculate MMD
    # for each level of perturbation (specify the kernel/sigma value)
    list_of_mmd = [evaluate_mmd(
        graphs_dist_1=test_graphs,
        graphs_dist_2=graphs,
        function=degree_distribution,
        kernel=gaussian_kernel,
        use_linear_approximation=False,
        density=True,
        sigma=best_sigma
        ) for graphs in perturbed_graphs]

    correlation = compute_correlation(
            perturbation_values,
            list_of_mmd,
            correlation_type="pearson"
            )

    print("Option 2:")
    print("The correlation is {:.6f}.".format(correlation))
    print("======================")

    # OPTION 3: just compute MMD between predicted graphs and test graphs
    paper_graphs = load_graphs("../../data/predictions/graphs/ggme_paper/CommunityGraphs.pickle")
    # DiGress community graphs from generated_samples1.txt (adjacency matrices)
    digress_graphs = load_digress_generated_samples("../../data/predictions/graphs/digress/generated_samples1.txt")
    print("Loaded {} DiGress graphs from generated_samples1.txt".format(len(digress_graphs)))
    test_graphs = load_graphs("../../data/test/graphs/CommunityGraphs.pickle")
    
    graphons = torch.load("../../data/predictions/graphons/tensors.pt")
    generated_graphs = generate_graphs(graphons)
    print("Loaded {} OGAD graphs from tensors.pt".format(len(generated_graphs)))
    
    print("Option 3:")
    for key, dataset in [
        ('DiGress', digress_graphs),
        ('OGAD', generated_graphs),
        ('Paper', paper_graphs),
    ]:
        generated_graphs_mmd = evaluate_mmd(
            graphs_dist_1=test_graphs,
            graphs_dist_2=dataset,
            function=degree_distribution,
            kernel=gaussian_kernel,
            use_linear_approximation=False,
            density=True,
            sigma=best_sigma
        )
        print("The MMD between the test and {} graphs is {:.4f}.".format(key, generated_graphs_mmd))
    
    # paper_graphs_mmd = evaluate_mmd(
    #     graphs_dist_1=test_graphs,
    #     graphs_dist_2=paper_graphs,
    #     function=degree_distribution,
    #     kernel=gaussian_kernel,
    #     use_linear_approximation=False,
    #     density=True,
    #     sigma=best_sigma
    # )

