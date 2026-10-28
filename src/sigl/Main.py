from bootstrap_utils import build_cluster_bootstrap_plan
from clusterize import cluster_graphs_from_arrays
from graph_split_utils import deterministic_graph_split, select_by_indices
from tools import *
import concurrent.futures
import numpy as np
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import torch
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def learn_and_save_graphon_job(payload):
    job = payload["job"]
    params = payload["params"]
    graph_inr = payload["graph_inr"]
    output_path = payload["output_path"]

    print("Graphon job {}: Step1 learning latent node variables".format(job["sample_id"]))
    model_SIGL, _ = coords_prediction(
        params["inr_dim_hidden"],
        params["gnn_dim_hidden"],
        params["n_epochs"],
        params["epoch_show"],
        params["w0"],
        graph_inr,
        params["lr"],
    )
    print("Graphon job {}: Step2 sorting graphs and pooling".format(job["sample_id"]))
    X_all, y_all, w_all = graph2XY(graph_inr, model_SIGL)
    print("Graphon job {}: Step3 training INR histograms".format(job["sample_id"]))
    trained_inr = train_graphon(
        params["inr_dim_hidden"],
        params["w0"],
        X_all,
        y_all,
        w_all,
        params["n_epochs"],
        params["epoch_show"],
        params["lr"],
        params["batch_size"],
    )
    estimated_graphon = get_graphon(params["Res"], trained_inr)
    print("graphon {} generated {}".format(output_path, len(estimated_graphon)))
    save_graphon_for_diff(estimated_graphon, output_path)


def run_graphon_payload(payload_path):
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    learn_and_save_graphon_job(payload)


def build_graphon_payload(job, graphs_inr, params, output_dir):
    graph_inr = [
        graphs_inr[source_index]
        for source_index in job["model_train_graph_indices"]
    ]
    return {
        "job": job,
        "params": params,
        "graph_inr": graph_inr,
        "output_path": os.path.join(output_dir, "sample_{}.png".format(job["sample_id"])),
    }


def graphon_worker_log_dir(output_dir):
    run_dir = os.environ.get("OGAD_RUN_DIR")
    if run_dir:
        return os.path.join(run_dir, "sigl_workers")
    return os.path.join(os.path.dirname(output_dir), "sigl_workers")


def run_graphon_subprocess(payload_path, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--sigl_job_payload", payload_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_graphon_jobs(graphon_jobs, graphs_inr, params, output_dir, sigl_workers):
    sigl_workers = max(1, int(sigl_workers))
    if sigl_workers == 1:
        for job in graphon_jobs:
            payload = build_graphon_payload(job, graphs_inr, params, output_dir)
            learn_and_save_graphon_job(payload)
        return

    print("Running {} SIGL graphon jobs with {} workers".format(len(graphon_jobs), sigl_workers))
    log_dir = graphon_worker_log_dir(output_dir)
    os.makedirs(log_dir, exist_ok=True)
    print("SIGL worker logs: {}".format(log_dir))
    with tempfile.TemporaryDirectory(prefix="sigl_graphon_jobs_") as temp_dir:
        worker_jobs = []
        for job in graphon_jobs:
            payload = build_graphon_payload(job, graphs_inr, params, output_dir)
            payload_path = os.path.join(temp_dir, "job_{}.pickle".format(job["sample_id"]))
            with open(payload_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            log_path = os.path.join(log_dir, "job_{}.log".format(job["sample_id"]))
            worker_jobs.append((job["sample_id"], payload["output_path"], payload_path, log_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=sigl_workers) as executor:
            futures = {}
            for sample_id, output_path, payload_path, log_path in worker_jobs:
                print("job {} started -> {}".format(sample_id, log_path))
                future = executor.submit(run_graphon_subprocess, payload_path, log_path)
                futures[future] = (sample_id, output_path, log_path)
            for future in concurrent.futures.as_completed(futures):
                sample_id, output_path, log_path = futures[future]
                try:
                    future.result()
                except Exception:
                    print("job {} failed -> {}".format(sample_id, log_path))
                    raise
                print("job {} completed -> {}".format(sample_id, output_path))

def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


parser = argparse.ArgumentParser(description="Arguments for training and evaluating graphon learning models.")
parser.add_argument('--test_name', type=str, default=None, help="Test name")
parser.add_argument('--n_graphon', type=int, default=1, help="Index of the graphon to learn.")
parser.add_argument('--alpha', type=float, default=0.5, help="Parameter for the SBM graphons.")
parser.add_argument('--n_trials', type=int, default=10, help="Number of trials.")
parser.add_argument('--n_G', type=int, default=10, help="Number of graphs generated per trial.")
parser.add_argument('--offset', type=int, default=0, help="Offset for the number of nodes in each graph.")
parser.add_argument('--Res', type=int, default=1000, help="Resolution of the samples graphon from the INR.")
parser.add_argument('--n_epoch', type=int, default=100, help="Number of training epochs.")
parser.add_argument('--epoch_show', type=int, default=20, help="Training checkpoint interval.")
parser.add_argument('--gnn_dim_hidden', type=str, default="8,8,8", help="Hidden layer sizes for GNN in step 1.")
parser.add_argument('--inr_dim_hidden', type=str, default="20,20", help="Hidden layer sizes for SIREN.")
parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training INR in step 3.")
parser.add_argument('--w0', type=float, default=10, help="Default frequency for sine activation in INR.")
parser.add_argument('--lr', type=float, default=0.06, help="Learning rate.")
parser.add_argument('--h_size', type=int, default=4, 
                    help="Pooling window size (1: 1, 2: 2, 3: 3, 4: log(n), 5: sqrt(n)). Default: 4.")
parser.add_argument('--Adjs_dir', type=str, default='../../datasets/community_graphs/CommunityGraphs_train.pkl',
                    help="Path to existing adjacency matrices (e.g., myGraphs/Adjs.pkl).")
parser.add_argument('--output_dir', type=str, default='../../data/train/graphons', help="Output directory.")
parser.add_argument('--n_clusters', type=int, default=None, help="Number of clasters")
parser.add_argument('--bootstrap_replicates', type=int, default=1, help="Bootstrap graphon replicas per cluster.")
parser.add_argument('--bootstrap_seed', type=int, default=0, help="Seed for deterministic cluster bootstrap.")
parser.add_argument('--sigl_workers', type=int, default=1, help="Parallel graphon workers inside one SIGL run.")
parser.add_argument(
    '--include_original_clusters',
    type=str2bool,
    default=False,
    help="Whether to save one original whole-cluster graphon in addition to bootstrap replicas.",
)
parser.add_argument(
    '--config',
    type=str,
    default=None,
    help="Optional JSON experiment file: merges top-level 'shared' then 'sigl' into CLI args.",
)
parser.add_argument('--sigl_job_payload', type=str, default=None, help=argparse.SUPPRESS)

graph_selection_config = {}
args = parser.parse_args()
if args.sigl_job_payload:
    run_graphon_payload(args.sigl_job_payload)
    exit(0)
if args.config:
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    merged = {**(cfg.get("shared") or {}), **(cfg.get("sigl") or {})}
    graph_selection_config = cfg.get("graph_selection") or {}
    for key, val in merged.items():
        if hasattr(args, key):
            setattr(args, key, val)
print("Parameters: \n", args)

test_name = args.test_name
n_graphon = args.n_graphon - 1
alpha = args.alpha
n_trials = args.n_trials
n_G = args.n_G
offset = args.offset
Res = args.Res

n_epochs = args.n_epoch
epoch_show = args.epoch_show
gnn_dim_hidden = [int(x) for x in args.gnn_dim_hidden.split(',')]
inr_dim_hidden = [int(x) for x in args.inr_dim_hidden.split(',')]
batch_size = args.batch_size
w0 = args.w0
lr = args.lr
hsize = args.h_size
Adjs_dir = args.Adjs_dir
n_clusters = args.n_clusters
bootstrap_replicates = int(args.bootstrap_replicates)
bootstrap_seed = int(args.bootstrap_seed)
sigl_workers = int(args.sigl_workers)
include_original_clusters = str2bool(args.include_original_clusters)

output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# Initialization
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if Adjs_dir is None: # Generate the adjacency matrices from a true graphon (synthetic data) to reproduce the paper results
    errors = np.zeros(n_trials)
    All_time = np.zeros(n_trials)
    min_error = 1e10

    for trial in range(n_trials):
        print("Trial: ", trial)
        # Generate the true graphon
        true_graphon = synthesize_graphon(r=Res, type_idx=n_graphon, alpha=alpha)
        # Observe graphs from the true graphon
        graphs_inr = simulate_graphs(true_graphon, num_graphs=n_G, graph_size='vary', offset=offset)
        start_time = time.time()
        ### Learn the coordinates and graphon as step 1
        print("Step1: Learning the laten variable of the nodes")
        model_SIGL, _ = coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, graphs_inr, lr)
        ### Sort the graphs based on the learned latent variables
        print("Step2: Sorting the graphs and calculating the pooling")
        X_all, y_all, w_all = graph2XY(graphs_inr, model_SIGL)
        ### train INR using histogram approximation
        print("Step3: Training the INR using histograms")
        trained_inr = train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size)

        # Sample the estimated graphon from inr:
        predictions_graphon = get_graphon(Res, trained_inr)
        end_time = time.time()

        # Calculate the GW distance between the true and estimated graphon as error
        error_trial = gw_distance(true_graphon, predictions_graphon)
        print("GW : ", error_trial)
        All_time[trial] = end_time - start_time
        errors[trial] = error_trial

        # plot the samples for the best trial
        if error_trial <= min_error:
            min_error = error_trial
            plot_smaples_syn(true_graphon, predictions_graphon, output_dir, model_SIGL, offset)
            with open(output_dir + '/sigl.pkl', "wb") as f:
                pickle.dump(trained_inr, f)
            list_of_tensors = [predictions_graphon for _ in range(10)]
            arrays = {f"tensor_{i:05d}": t for i, t in enumerate(list_of_tensors)}
            np.savez("output/SIGL/graphon_dataset.npz", **arrays)

    # print the average error
    print("Average time: ", np.mean(All_time))
    print("Standard deviation of time: ", np.std(All_time))
    # remove nan values from errors
    errors = errors[~np.isnan(errors)]
    print("Average GW distance between true and predicted graphon: ", np.round(np.mean(errors), 4))
    print("Standard deviation of GW distance between true and predicted graphon: ", np.round(np.std(errors), 4))


else: # load the existing adjacency matrices
    existing = [f for f in os.listdir(output_dir) if f.startswith("sample_") and f.endswith(".png")]
    if existing:
        print(f"Dataset already exists in {output_dir} ({len(existing)} graphons found). Skipping generation.")
        exit(0)

    graphs_inr = load_graphs(Adjs_dir)
    print("graphs: ", len(graphs_inr))
    split_metadata = None
    graph_index_map = list(range(len(graphs_inr)))
    if graph_selection_config:
        selection_fraction = float(graph_selection_config.get("validation_fraction", 0.2))
        selection_seed = int(graph_selection_config.get("split_seed", 0))
        model_train_indices, ggme_validation_indices = deterministic_graph_split(
            len(graphs_inr),
            selection_fraction,
            selection_seed,
        )
        ggme_validation_graphs = select_by_indices(graphs_inr, ggme_validation_indices)
        graphs_inr = select_by_indices(graphs_inr, model_train_indices)
        graph_index_map = model_train_indices
        run_dir = os.environ.get("OGAD_RUN_DIR")
        if run_dir:
            validation_path = os.path.join(run_dir, "ggme_validation_graphs.pickle")
            model_train_path = os.path.join(run_dir, "model_train_graphs.pickle")
            with open(validation_path, "wb") as f:
                pickle.dump(ggme_validation_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(model_train_path, "wb") as f:
                pickle.dump(graphs_inr, f, protocol=pickle.HIGHEST_PROTOCOL)
        split_metadata = {
            "validation_fraction": selection_fraction,
            "split_seed": selection_seed,
            "model_train_indices": model_train_indices,
            "ggme_validation_indices": ggme_validation_indices,
            "model_train_count": len(graphs_inr),
            "ggme_validation_count": len(ggme_validation_graphs),
        }
        print(
            "GGME validation split: model_train={} ggme_validation={}".format(
                len(graphs_inr), len(ggme_validation_graphs)
            )
        )
    graphon_jobs = [
        {
            "sample_id": index,
            "cluster_id": index,
            "bootstrap_id": None,
            "is_original": True,
            "original_graph_indices": [index],
        }
        for index in range(len(graphs_inr))
    ]
    cluster_sizes = [1 for _ in graphs_inr]

    if n_clusters != None:
        clustered_graphs = cluster_graphs_from_arrays(
            graphs_inr,
            n_clusters,
            return_indices=True
        )
        cluster_sizes = [len(cluster["indices"]) for cluster in clustered_graphs]
        graphon_jobs = build_cluster_bootstrap_plan(
            [cluster["indices"] for cluster in clustered_graphs],
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            include_original_clusters=include_original_clusters,
        )
        print(
            "bootstrap graphons: ",
            len(graphon_jobs),
            "from clusters: ",
            len(clustered_graphs),
        )

    for job in graphon_jobs:
        model_train_graph_indices = list(job["original_graph_indices"])
        job["model_train_graph_indices"] = model_train_graph_indices
        job["original_graph_indices"] = [
            graph_index_map[index] for index in model_train_graph_indices
        ]

    metadata = {
        "n_clusters": n_clusters,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "include_original_clusters": include_original_clusters,
        "cluster_sizes": cluster_sizes,
        "graph_selection_split": split_metadata,
        "samples": graphon_jobs,
    }

    graphon_params = {
        "inr_dim_hidden": inr_dim_hidden,
        "gnn_dim_hidden": gnn_dim_hidden,
        "n_epochs": n_epochs,
        "epoch_show": epoch_show,
        "w0": w0,
        "lr": lr,
        "batch_size": batch_size,
        "Res": Res,
    }
    run_graphon_jobs(graphon_jobs, graphs_inr, graphon_params, output_dir, sigl_workers)

    with open(os.path.join(output_dir, "bootstrap_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
