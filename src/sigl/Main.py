from clusterize import cluster_graphs_from_arrays
from tools import *
import numpy as np
import argparse
import json
import pickle
import torch
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

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
parser.add_argument('--Adjs_dir', type=str, default='../../data/train/graphs/CommunityGraphs.pickle', 
                    help="Path to existing adjacency matrices (e.g., myGraphs/Adjs.pkl).")
parser.add_argument('--output_dir', type=str, default='../../data/train/graphons', help="Output directory.")
parser.add_argument('--n_clusters', type=int, default=None, help="Number of clasters")
parser.add_argument(
    '--config',
    type=str,
    default=None,
    help="Optional JSON experiment file: merges top-level 'shared' then 'sigl' into CLI args.",
)

args = parser.parse_args()
if args.config:
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    merged = {**(cfg.get("shared") or {}), **(cfg.get("sigl") or {})}
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
    graph_collections = graphs_inr

    if n_clusters != None:
        graph_collections = cluster_graphs_from_arrays(
            graphs_inr,
            n_clusters
        )

    for index in range(0, len(graph_collections)):
        if n_clusters != None:
            graph_inr = graph_collections[index]
        else:
            graph_inr = [graph_collections[index]]
        ### Learn the coordinates and graphon as step 1
        print("Step1: Learning the laten variable of the nodes")
        model_SIGL, _ = coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, graph_inr, lr)
        ### Sort the graphs based on the learned latent variables
        print("Step2: Sorting the graphs and calculating the pooling")
        X_all, y_all, w_all = graph2XY(graph_inr, model_SIGL)
        ### train INR using histogram approximation
        print("Step3: Training the INR using histograms")
        trained_inr = train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size)
        # Sample the estimated graphon from inr:
        estimated_graphon = get_graphon(Res, trained_inr)

        # plot a sample of the graphs with the estimated graphon and the learned coordinates
        # sample_graph = graph_inr[-1]
        # plot_smaples_real(sample_graph, estimated_graphon, output_dir, model_SIGL)
        print(f'graphon {output_dir}/sample_{index}.png generated', len(estimated_graphon))
        save_graphon_for_diff(estimated_graphon, output_dir + f'/sample_{index}.png')