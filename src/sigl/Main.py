from bootstrap_utils import build_cluster_bootstrap_plan, parse_job_ids, select_graphon_jobs
from clusterize import cluster_graphs_from_arrays
from graph_split_utils import deterministic_graph_split, select_by_indices
from tools import *
import numpy as np
import argparse
import json
import pickle
import subprocess
import sys
import torch
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

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
parser.add_argument(
    '--node_features',
    type=str,
    default="structural",
    choices=["structural", "degree", "ones", "random"],
    help="Node features for SIGL coordinate learning. 'random' preserves the original behavior.",
)
parser.add_argument(
    '--degree_correction',
    type=str,
    default="none",
    choices=["none", "sorted_profile"],
    help="Optional graphon row/column scaling to match the sorted empirical degree profile.",
)
parser.add_argument(
    '--degree_correction_iters',
    type=int,
    default=32,
    help="Number of symmetric scaling iterations for degree_correction=sorted_profile.",
)
parser.add_argument('--Adjs_dir', type=str, default='../../data/train/graphs/CommunityGraphs.pickle', 
                    help="Path to existing adjacency matrices (e.g., myGraphs/Adjs.pkl).")
parser.add_argument('--output_dir', type=str, default='../../data/train/graphons', help="Output directory.")
parser.add_argument('--n_clusters', type=int, default=None, help="Number of clasters")
parser.add_argument(
    '--min_cluster_size',
    type=int,
    default=1,
    help="Merge clusters smaller than this size before graphon estimation.",
)
parser.add_argument('--bootstrap_replicates', type=int, default=1, help="Bootstrap graphon replicas per cluster.")
parser.add_argument('--bootstrap_seed', type=int, default=0, help="Seed for deterministic cluster bootstrap.")
parser.add_argument(
    '--min_bootstrap_cluster_size',
    type=int,
    default=1,
    help="Clusters smaller than this produce one original graphon instead of bootstrap replicas.",
)
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
parser.add_argument(
    '--job_ids',
    type=str,
    default=None,
    help="Comma-separated graphon sample ids to run. Empty means all jobs.",
)
parser.add_argument(
    '--job_shard_index',
    type=int,
    default=0,
    help="Worker shard index for job-level parallel SIGL generation.",
)
parser.add_argument(
    '--job_shard_count',
    type=int,
    default=1,
    help="Total number of SIGL job shards.",
)
parser.add_argument(
    '--skip_existing',
    type=str2bool,
    default=False,
    help="Skip individual sample_i.png files that already exist instead of exiting globally.",
)
parser.add_argument(
    '--write_metadata',
    type=str2bool,
    default=True,
    help="Whether this process may write shared metadata/split files.",
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help="Launch this many internal SIGL worker processes using deterministic job shards.",
)

graph_selection_config = {}
args = parser.parse_args()
explicit_cli_args = {
    item.split("=", 1)[0].lstrip("-").replace("-", "_")
    for item in sys.argv[1:]
    if item.startswith("--")
}
if args.config:
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    merged = {**(cfg.get("shared") or {}), **(cfg.get("sigl") or {})}
    graph_selection_config = cfg.get("graph_selection") or {}
    for key, val in merged.items():
        if hasattr(args, key):
            if key in explicit_cli_args:
                continue
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
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
node_features = args.node_features
degree_correction = args.degree_correction
degree_correction_iters = int(args.degree_correction_iters)
Adjs_dir = args.Adjs_dir
n_clusters = args.n_clusters
min_cluster_size = int(args.min_cluster_size)
bootstrap_replicates = int(args.bootstrap_replicates)
bootstrap_seed = int(args.bootstrap_seed)
min_bootstrap_cluster_size = int(args.min_bootstrap_cluster_size)
include_original_clusters = str2bool(args.include_original_clusters)
job_ids = parse_job_ids(args.job_ids)
job_shard_index = int(args.job_shard_index)
job_shard_count = int(args.job_shard_count)
skip_existing = str2bool(args.skip_existing)
write_metadata = str2bool(args.write_metadata)
num_workers = int(args.num_workers)

output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# Initialization
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def coalesce_small_clusters(clustered_graphs, min_size):
    if min_size <= 1:
        return clustered_graphs

    large_clusters = []
    small_graphs = []
    small_indices = []
    for cluster in clustered_graphs:
        if len(cluster["indices"]) >= min_size:
            large_clusters.append(cluster)
        else:
            small_graphs.extend(cluster["graphs"])
            small_indices.extend(cluster["indices"])

    if not small_indices:
        return large_clusters

    merged_small = {"graphs": small_graphs, "indices": small_indices}
    if len(small_indices) >= min_size or not large_clusters:
        return large_clusters + [merged_small]

    target = min(large_clusters, key=lambda cluster: len(cluster["indices"]))
    target["graphs"].extend(small_graphs)
    target["indices"].extend(small_indices)
    return large_clusters


def set_job_seed(base_seed, sample_id):
    seed = int(base_seed) + int(sample_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def write_metadata_file(metadata, output_dir):
    tmp_path = os.path.join(output_dir, "bootstrap_metadata.json.tmp")
    final_path = os.path.join(output_dir, "bootstrap_metadata.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_path, final_path)


def launch_parallel_workers():
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_workers == 1:
        return False
    if job_shard_count != 1 or job_shard_index != 0:
        raise ValueError("Use either --num_workers or manual --job_shard_* arguments, not both")

    log_dir = os.path.join(output_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    child_base_cmd = [sys.executable, os.path.abspath(__file__)]
    if args.config:
        child_base_cmd.extend(["--config", args.config])
    else:
        child_base_cmd.extend(sys.argv[1:])

    print("Launching {} SIGL workers".format(num_workers))
    processes = []
    log_handles = []
    for worker_index in range(num_workers):
        child_cmd = child_base_cmd + [
            "--num_workers",
            "1",
            "--job_shard_index",
            str(worker_index),
            "--job_shard_count",
            str(num_workers),
            "--skip_existing",
            "true",
            "--write_metadata",
            "true" if worker_index == 0 else "false",
        ]
        log_path = os.path.join(log_dir, "sigl_worker_{}.log".format(worker_index))
        log_handle = open(log_path, "w", encoding="utf-8")
        log_handles.append(log_handle)
        print("worker {} log: {}".format(worker_index, log_path))
        processes.append(subprocess.Popen(child_cmd, stdout=log_handle, stderr=subprocess.STDOUT))

    return_codes = []
    try:
        for process in processes:
            return_codes.append(process.wait())
    finally:
        for log_handle in log_handles:
            log_handle.close()

    failed = [index for index, code in enumerate(return_codes) if code != 0]
    if failed:
        raise RuntimeError("SIGL workers failed: {}".format(failed))
    print("All SIGL workers completed")
    return True


if launch_parallel_workers():
    exit(0)


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
        model_SIGL, _ = coords_prediction(
            inr_dim_hidden,
            gnn_dim_hidden,
            n_epochs,
            epoch_show,
            w0,
            graphs_inr,
            lr,
            node_feature_mode=node_features,
        )
        ### Sort the graphs based on the learned latent variables
        print("Step2: Sorting the graphs and calculating the pooling")
        X_all, y_all, w_all = graph2XY(
            graphs_inr,
            model_SIGL,
            hsize=hsize,
            node_feature_mode=node_features,
        )
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
    if existing and not skip_existing:
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
        should_write_shared_files = write_metadata and job_shard_index == 0
        if run_dir and should_write_shared_files:
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
        before_cluster_sizes = [len(cluster["indices"]) for cluster in clustered_graphs]
        clustered_graphs = coalesce_small_clusters(clustered_graphs, min_cluster_size)
        if min_cluster_size > 1:
            print(
                "coalesced clusters: {} -> {} with min_cluster_size={}".format(
                    len(before_cluster_sizes),
                    len(clustered_graphs),
                    min_cluster_size,
                )
            )
        cluster_sizes = [len(cluster["indices"]) for cluster in clustered_graphs]
        graphon_jobs = build_cluster_bootstrap_plan(
            [cluster["indices"] for cluster in clustered_graphs],
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            include_original_clusters=include_original_clusters,
            min_bootstrap_cluster_size=min_bootstrap_cluster_size,
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
        "min_cluster_size": min_cluster_size,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "min_bootstrap_cluster_size": min_bootstrap_cluster_size,
        "include_original_clusters": include_original_clusters,
        "node_features": node_features,
        "h_size": hsize,
        "degree_correction": degree_correction,
        "degree_correction_iters": degree_correction_iters,
        "cluster_sizes": cluster_sizes,
        "graph_selection_split": split_metadata,
        "samples": graphon_jobs,
        "job_shard_index": job_shard_index,
        "job_shard_count": job_shard_count,
        "job_ids": sorted(job_ids) if job_ids is not None else None,
    }

    should_write_metadata = write_metadata and job_shard_index == 0
    if should_write_metadata:
        write_metadata_file(metadata, output_dir)

    selected_jobs = select_graphon_jobs(
        graphon_jobs,
        job_ids=job_ids,
        shard_index=job_shard_index,
        shard_count=job_shard_count,
    )
    print(
        "selected graphon jobs: {} of {} (shard {}/{})".format(
            len(selected_jobs),
            len(graphon_jobs),
            job_shard_index,
            job_shard_count,
        )
    )

    for job in selected_jobs:
        index = job["sample_id"]
        sample_path = output_dir + f'/sample_{index}.png'
        if skip_existing and os.path.exists(sample_path):
            print(f"graphon {sample_path} already exists; skipping")
            continue
        job_seed = set_job_seed(bootstrap_seed, index)
        job["seed"] = job_seed
        graph_inr = [
            graphs_inr[source_index]
            for source_index in job["model_train_graph_indices"]
        ]
        ### Learn the coordinates and graphon as step 1
        print("Step1: Learning the laten variable of the nodes")
        model_SIGL, _ = coords_prediction(
            inr_dim_hidden,
            gnn_dim_hidden,
            n_epochs,
            epoch_show,
            w0,
            graph_inr,
            lr,
            node_feature_mode=node_features,
        )
        ### Sort the graphs based on the learned latent variables
        print("Step2: Sorting the graphs and calculating the pooling")
        X_all, y_all, w_all = graph2XY(
            graph_inr,
            model_SIGL,
            hsize=hsize,
            node_feature_mode=node_features,
        )
        ### train INR using histogram approximation
        print("Step3: Training the INR using histograms")
        trained_inr = train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size)
        # Sample the estimated graphon from inr:
        estimated_graphon = get_graphon(Res, trained_inr)
        if degree_correction == "sorted_profile":
            target_degree_profile = sorted_degree_profile(
                graph_inr,
                model_SIGL,
                Res,
                node_feature_mode=node_features,
            )
            estimated_graphon = degree_correct_graphon(
                estimated_graphon,
                target_degree_profile,
                num_iters=degree_correction_iters,
            )
            job["degree_profile_mean"] = float(np.mean(target_degree_profile))
            job["degree_profile_min"] = float(np.min(target_degree_profile))
            job["degree_profile_max"] = float(np.max(target_degree_profile))

        # plot a sample of the graphs with the estimated graphon and the learned coordinates
        # sample_graph = graph_inr[-1]
        # plot_smaples_real(sample_graph, estimated_graphon, output_dir, model_SIGL)
        print(f'graphon {sample_path} generated', len(estimated_graphon))
        save_graphon_for_diff(estimated_graphon, sample_path)

    if should_write_metadata and job_shard_count == 1 and job_ids is None:
        write_metadata_file(metadata, output_dir)
