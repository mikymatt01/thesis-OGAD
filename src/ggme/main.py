from evaluate import evaluate_mmd
from correlation import compute_correlation
from correlation import select_kernel_and_hyperparameters

from metrics.kernels import linear_kernel
from metrics.kernels import gaussian_kernel

from metrics.descriptor_functions import degree_distribution, clustering_coefficient, normalised_laplacian_spectrum

from perturb_graphs import perturb_graphs
from utils import load_graphs, load_digress_generated_samples
from graphon_utils import (
    cyclic_graphon_index,
    probability_matrix,
    sanitize_graphon_matrix,
    validate_generation_request,
)
import argparse
import json
import numpy as np
import networkx as nx
import torch
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def sanitize_graphon(W):
    """Enforce graphon symmetry, probability range, and a zero diagonal."""
    return sanitize_graphon_matrix(W)


def sample_graph_from_graphon(W, M, generator=None):
    # W_matrix: torch tensor NxN con valori in [0,1]
    N = W.shape[0]
    # campiona valori latenti uniformi
    eta = torch.rand(M, device=W.device, generator=generator)
    # mappa su indici della griglia W
    idx = (eta * N).long().clamp(0, N-1)
    # costruisci la matrice di probabilità
    probs = probability_matrix(W, idx)
    # campiona gli edges
    A = torch.bernoulli(probs, generator=generator)
    # se vuoi grafo non orientato senza self-loop:
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    return A


def _test_node_sizes(test_graphs):
    return [G.number_of_nodes() for G in test_graphs]


def _sample_M(rng, sizes, n_max, max_tries=64):
    """Sample M from empirical test sizes; reject M > n_max and resample."""
    for _ in range(max_tries):
        m = int(rng.choice(sizes))
        if 1 <= m <= n_max:
            return m
    # fallback: clamp last draw
    return max(1, min(int(rng.choice(sizes)), n_max))


def generate_graphs(graphons, test_sizes, num_graphs=100, seed=0):
    """Generate a fixed count of OGAD graphs with test-set node sizes."""
    validate_generation_request(num_graphs, test_sizes)
    if len(graphons) < 1:
        raise ValueError("No graphons available for graph generation")
    rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device=graphons.device)
    torch_generator.manual_seed(seed)
    sanitized_graphons = [
        sanitize_graphon(graphon[0]) for graphon in graphons
    ]
    generated_graphs = []
    for sample_index in range(num_graphs):
        graphon_index = cyclic_graphon_index(sample_index, len(sanitized_graphons))
        W = sanitized_graphons[graphon_index]
        n_max = W.shape[0]
        M = _sample_M(rng, test_sizes, n_max)
        adj_matrix = sample_graph_from_graphon(W, M, generator=torch_generator)
        generated_graphs.append(nx.from_numpy_array(adj_matrix.detach().cpu().numpy()))
    return generated_graphs

def compute_ggme(
    test_graphs,
    paper_graphs,
    generated_graphs,
    digress_graphs,
    perturbed_graphs,
    perturbation_values,
    descriptor_fn
): 
    descriptor_name = getattr(descriptor_fn, "__name__", str(descriptor_fn))
    prefix = f"[descriptor={descriptor_name}] "
    def log(msg: str) -> None:
        print(prefix + msg)

    # OPTION 1: to find the best parameters for a grid of kernel/hyperparameter
    # combination, run the following function, which will return the
    # kernel, sigma, and correlation of the best configuration
    best_params = select_kernel_and_hyperparameters(
        test_graphs=test_graphs,
        perturbed_graphs=perturbed_graphs,
        perturbation_values=perturbation_values,
        kernels=[gaussian_kernel, linear_kernel],
        sigmas=[0.1, 1, 10],
        descriptor_fn=descriptor_fn,
        correlation_type="pearson"
    )
    log("Option 1:")
    log("The best kernel-hyperparameter combination is the {} with sigma={}. This had a correlation of {:.4f}."
        .format(
        best_params["kernel"],
        best_params["sigma"],
        best_params["corr"]
        ))
    # Keep the kernel as a callable; only use __name__ for display.
    best_kernel = best_params["kernel"]
    best_sigma = best_params["sigma"]
    best_corr = best_params["corr"]
    log("======================")

    # OPTION 2: to compute the correlation for a single kernel/param combo,
    # take perturbed graphs and true graphs and calculate MMD
    # for each level of perturbation (specify the kernel/sigma value)
    list_of_mmd = [evaluate_mmd(
        graphs_dist_1=test_graphs,
        graphs_dist_2=graphs,
        function=descriptor_fn,
        kernel=best_kernel,
        use_linear_approximation=False,
        density=True,
        sigma=best_sigma
        ) for graphs in perturbed_graphs]

    correlation = compute_correlation(
            perturbation_values,
            list_of_mmd,
            correlation_type="pearson"
            )

    log("Option 2:")
    log("The correlation is {:.6f}.".format(correlation))
    log("======================")
    
    log("Option 3:")
    mmd_by_key = {}
    for key, dataset in [
        ('DiGress', digress_graphs),
        ('OGAD', generated_graphs),
        ('Paper', paper_graphs),
    ]:
        if key == "DiGress" and not dataset:
            log("DiGress baseline skipped (no graphs).")
            mmd_by_key[key] = None
            continue
        generated_graphs_mmd = evaluate_mmd(
            graphs_dist_1=test_graphs,
            graphs_dist_2=dataset,
            function=descriptor_fn,
            kernel=best_kernel,
            use_linear_approximation=False,
            density=True,
            sigma=best_sigma
        )
        mmd_by_key[key] = float(generated_graphs_mmd)
        log("The MMD between the test and {} graphs is {:.4f}.".format(key, generated_graphs_mmd))

    return {
        "descriptor": descriptor_name,
        "best_kernel": getattr(best_kernel, "__name__", str(best_kernel)),
        "best_sigma": float(best_sigma),
        "corr_option1": float(best_corr),
        "correlation_option2": float(correlation),
        "mmd": mmd_by_key,
    }


def append_metrics_jsonl(run_dir: str | None, payload: dict) -> None:
    if not run_dir:
        return
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


if __name__ == "__main__":
    print("path for: ", os.getcwd())

    parser = argparse.ArgumentParser(description="GGME evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional experiment JSON: uses 'ggme' object for paths (Docker: /app/data/...).",
    )
    parser.add_argument(
        "--skip-digress",
        action="store_true",
        help="Do not load DiGress baseline samples.",
    )
    parser.add_argument(
        "--digress_samples",
        type=str,
        default="../../data/predictions/graphs/digress/generated_samples1.txt",
        help="Path to DiGress generated_samples1.txt",
    )
    parser.add_argument(
        "--train_graphs",
        type=str,
        default="../../data/train/graphs/CommunityGraphs.pickle",
    )
    parser.add_argument(
        "--test_graphs",
        type=str,
        default="../../data/test/graphs/CommunityGraphs.pickle",
    )
    parser.add_argument(
        "--paper_graphs",
        type=str,
        default="../../data/predictions/graphs/ggme_paper/CommunityGraphs.pickle",
    )
    parser.add_argument(
        "--graphons_pt",
        type=str,
        default="../../data/predictions/graphons/tensors.pt",
    )
    parser.add_argument("--num_generated_graphs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
        for key, val in (cfg.get("ggme") or {}).items():
            if hasattr(args, key):
                setattr(args, key, val)

    test_graphs = load_graphs(args.test_graphs)
    test_sizes = _test_node_sizes(test_graphs)
    paper_graphs = load_graphs(args.paper_graphs)

    digress_graphs = []
    if not args.skip_digress and args.digress_samples and os.path.isfile(args.digress_samples):
        digress_graphs = load_digress_generated_samples(args.digress_samples)
    elif not args.skip_digress and args.digress_samples:
        print(
            "DiGress samples not found at {}; continuing without baseline.".format(
                args.digress_samples
            )
        )
    if digress_graphs and len(digress_graphs) != args.num_generated_graphs:
        raise ValueError(
            "DiGress and OGAD must use the same number of generated graphs: "
            f"expected {args.num_generated_graphs}, found {len(digress_graphs)}"
        )

    graphons = torch.load(args.graphons_pt, map_location=device)
    generated_graphs = generate_graphs(
        graphons,
        test_sizes,
        num_graphs=args.num_generated_graphs,
        seed=args.seed,
    )

    gen_sizes = [G.number_of_nodes() for G in generated_graphs]
    print(
        "OGAD generated sizes: min={} max={} mean={:.1f}".format(
            min(gen_sizes), max(gen_sizes), sum(gen_sizes) / len(gen_sizes)
        )
    )
    print(
        "Test sizes:          min={} max={} mean={:.1f}".format(
            min(test_sizes), max(test_sizes), sum(test_sizes) / len(test_sizes)
        )
    )

    print("Loaded {} DiGress graphs".format(len(digress_graphs)))
    print("Loaded {} OGAD graphs from {}".format(len(generated_graphs), args.graphons_pt))
    print("Test graphs: ", len(test_graphs))

    perturbation_values, perturbed_graphs = perturb_graphs(
        graphs=test_graphs,
        perturbation_type="AddEdges"
    )

    run_dir = os.environ.get("OGAD_RUN_DIR")
    run_id = os.environ.get("OGAD_RUN_ID")

    for descriptor_fn in [
        degree_distribution,
        clustering_coefficient,
        normalised_laplacian_spectrum
    ]:
        row = compute_ggme(
            test_graphs,
            paper_graphs,
            generated_graphs,
            digress_graphs,
            perturbed_graphs,
            perturbation_values,
            descriptor_fn
        )
        append_metrics_jsonl(
            run_dir,
            {
                "stage": "ggme",
                "run_id": run_id,
                "seed": args.seed,
                "num_generated_graphs": args.num_generated_graphs,
                **row,
            },
        )
