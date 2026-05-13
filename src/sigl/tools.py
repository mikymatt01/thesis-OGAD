from torchvision.utils import save_image
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib as mpl
from typing import List
import torch.nn as nn
import numpy as np
import pickle
import random
import torch
import math
import ot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 0):
    """Make torch / numpy / python RNG deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


'''Classes'''
# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# Siren layer:
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = 'sine'):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.act = activation

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

       
        if activation=='sine':
            self.activation=Sine(w0)
        elif activation=='relu':
            self.activation=nn.ReLU(inplace=False)
        elif activation=='id':
            self.activation=nn.Identity()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        else:
            raise ValueError('No mlp activation specified')


    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        act = self.act

        if act =='relu':
            w_std = math.sqrt(1/dim)
        else:
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# Siren network:    
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, activation = 'sine', final_activation = 'sigmoid', symmetric = False):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden)
        self.symmetric = symmetric
        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden[ind-1]
            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden[ind],
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                activation = activation
            ))

        final_activation = 'id' if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden[num_layers-1], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def _symmetrize_input(self, x):
        # Enforce f(x, y) = f(y, x) by replacing the first two coords with (min, max).
        xy = x[..., :2]
        rest = x[..., 2:]
        lo = xy.min(dim=-1, keepdim=True).values
        hi = xy.max(dim=-1, keepdim=True).values
        return torch.cat([lo, hi, rest], dim=-1)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)
        if self.symmetric:
            x = self._symmetrize_input(x)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x = x*rearrange(mod, 'd -> () d')   
                # x = x * mod

        return self.last_layer(x)

# GNN network using GCNConv:
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_dimensions):
        super(GCN, self).__init__()        
        self.layers = torch.nn.ModuleList()        
        self.layers.append(GCNConv(in_channels, hidden_channels[0]))        
        for i in range(1, len(hidden_channels)):
            self.layers.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))  
        
        # Final fully connected layer
        self.fc1 = torch.nn.Linear(hidden_channels[-1], num_dimensions)        
    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x_pre_fc = x
        x_post_fc = self.fc1(x_pre_fc)
        # Straight-through rank-normalization: forward yields exact uniform ranks in [0,1],
        # backward passes the gradient through identity to avoid outlier-driven min-max collapse.
        n = x_post_fc.size(0)
        denom = max(n - 1, 1)
        with torch.no_grad():
            ranks = torch.argsort(torch.argsort(x_post_fc, dim=0), dim=0).float() / denom
        x_post_fc = x_post_fc + (ranks - x_post_fc).detach()

        return x_pre_fc, x_post_fc
    
# SIGL network without pooling:
class SIGL(nn.Module):
    def __init__(self, model1, model2):
        super(SIGL, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, g):
        output_gnn_pre, output_gnn_post = self.model1(g.x, g.edge_index)
        n = output_gnn_post.size(0)
        z = output_gnn_post.squeeze(-1)
        
        x_coords = z.repeat_interleave(n)
        y_coords = z.repeat(n)
        edge_coords = torch.stack([x_coords, y_coords], dim=1)
        edge_coords = edge_coords.to(device)

        output_inr = self.model2(edge_coords)
        return output_inr, output_gnn_post


'''Functions'''
NODE_FEAT_DIM = 5


def _ldp_node_features(A: torch.Tensor) -> torch.Tensor:
    """Deterministic permutation-equivariant node features (LDP-like).

    Returns a (n, NODE_FEAT_DIM) tensor with:
      0: normalized degree d_i / (n-1)
      1: local clustering coefficient (triangles / (d_i choose 2))
      2: mean of neighbor degrees / (n-1)
      3: min of neighbor degrees / (n-1)  (0 if isolated)
      4: max of neighbor degrees / (n-1)  (0 if isolated)
    """
    n = A.size(0)
    A = A.float()
    A = A - torch.diag(torch.diagonal(A))
    deg = A.sum(dim=1)
    denom = max(n - 1, 1)
    deg_n = deg / denom

    triangles = torch.einsum('ij,jk,ki->i', A, A, A)
    pairs = deg * (deg - 1.0)
    clustering = torch.where(pairs > 0, triangles / pairs, torch.zeros_like(deg))

    deg_row = deg.unsqueeze(0).expand(n, -1)
    mask = A > 0
    safe_deg = torch.where(mask, deg_row, torch.zeros_like(deg_row))

    has_neigh = deg > 0
    sum_n = safe_deg.sum(dim=1)
    mean_n = torch.where(has_neigh, sum_n / torch.clamp(deg, min=1.0), torch.zeros_like(deg))
    big = torch.full_like(deg_row, float('inf'))
    min_neigh = torch.where(mask, deg_row, big).min(dim=1).values
    min_neigh = torch.where(has_neigh, min_neigh, torch.zeros_like(deg))
    max_neigh = torch.where(mask, deg_row, torch.zeros_like(deg_row)).max(dim=1).values

    feats = torch.stack([
        deg_n,
        clustering.clamp(0.0, 1.0),
        mean_n / denom,
        min_neigh / denom,
        max_neigh / denom,
    ], dim=1)
    return feats


def nx2torch(adjacency_matrix):
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
    edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t().contiguous()
    edge_value = adjacency_matrix.reshape(-1)
    x = _ldp_node_features(adjacency_matrix)
    data = Data(x=x, edge_index=edge_index, y=edge_value).to(device)
    return data

def graph2XY(graphs, model_SIGL, latent_val=None, hsize=4):
    num_nodes = sum([graph_i.shape[0] for graph_i in graphs])
    X_all = []
    y_all = []
    w_all = []

    # Loop through each graph
    for graph_idx in range(len(graphs)):
        # Sorting:
        graph_test = graphs[graph_idx]
        data_i = nx2torch(graph_test)
        _, output_gnn_post = model_SIGL.model1(data_i.x, data_i.edge_index)
        perm_i = torch.argsort(output_gnn_post.squeeze(-1))
        perm = perm_i.cpu().numpy()  # Convert torch permutation to numpy array
        weight = graph_test.shape[0] / num_nodes
        graph_test_sort = graph_test[perm, :][:, perm]
        
        # Average pooling to have the histogram
        N = graph_test_sort.shape[0]
        if hsize == 1:
            h = 1
        if hsize == 2:
            h = 2
        if hsize == 3:
            h = 3
        if hsize == 4:
            h = int(np.log(N))
        if hsize == 5:
            h = int(np.sqrt(N))
        A = torch.tensor(graph_test_sort, dtype=torch.float, device=device)
        A = A.view(1, 1, N, N)
        H = F.avg_pool2d(A, kernel_size=(h, h))
        H = H.cpu().numpy().squeeze()  # Convert pooled result back to numpy
        k = H.shape[0]

        # Get the coordinates:
        x_coord = (np.arange(k) + 0.5) / k
        y_coord = (np.arange(k) + 0.5) / k
        xx, yy = np.meshgrid(x_coord, y_coord)
        X = np.column_stack((xx.ravel(), yy.ravel()))
        y = H.ravel().reshape(-1, 1)

        w = np.tile(weight, (X.shape[0], 1))
        if latent_val is not None:
            z_this = latent_val[graph_idx]
            zz = np.tile(z_this, (X.shape[0], 1))
            X = np.column_stack((X, zz))

        X_all.append(X)
        y_all.append(y)
        w_all.append(w)

    # Convert lists to tensors
    X_all = torch.tensor(np.concatenate(X_all, axis=0), dtype=torch.float, device=device)
    y_all = torch.tensor(np.concatenate(y_all, axis=0), dtype=torch.float, device=device)
    w_all = torch.tensor(np.concatenate(w_all, axis=0), dtype=torch.float, device=device)

    return X_all, y_all, w_all

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def synthesize_graphon(r: int = 1000, type_idx: int = 0, alpha: int = 1) -> np.ndarray:
    """
    Synthesize graphons of Table 1
    :param r: the resolution of discretized graphon
    :param type_idx: the type of graphon
    :return:
        w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 101:
        theta = np.pi * 3 / 4
        w = 0.9*np.exp((-(v)**2-(u-1)**2)/alpha**2)+0.9*np.exp((-(u)**2-(v-1)**2)/alpha**2)+0.9*np.exp(-((np.sin(theta)*u+np.cos(theta)*v)/alpha)**2)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    
    # parametrized case of graphon 2 and SBM
    elif type_idx == 19:
        w = np.exp(-(u ** 0.7 + v ** 0.7)/alpha)
    elif type_idx == 14:
        w = np.zeros((r, r))
        R = int(r * alpha)
        w[:R, :R] = 0.8
        w[R:, R:] = 0.8
        w[:R, R:] = 0.1
        w[R:, :R] = 0.1
    else:
        raise ValueError('Unknown graphon type')

    np.fill_diagonal(w, 0.)
    return w

def simulate_graphs(w: np.ndarray, seed_gsize: int=123, seed_edge:int=123, num_graphs: int = 10,
                    num_nodes: int = 200, graph_size: str = 'fixed', offset:int=0) -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: a (r, r) discretized graphon
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []
    r = w.shape[0]
	
    if graph_size == 'vary':
        numbers = np.linspace(100+offset,500+offset,num_graphs).astype(int).tolist()

    else:
        numbers = [num_nodes for _ in range(num_graphs)]
    
    np.random.seed(seed_edge) #add random seed for reproducibility
    for n in range(num_graphs):
        node_locs = (r * np.random.rand(numbers[n])).astype('int')
        graph = w[node_locs, :]
        graph = graph[:, node_locs]
        noise = np.random.rand(graph.shape[0], graph.shape[1])
        noise = np.triu(noise)
        noise = noise + noise.T - np.diag(np.diag(noise))   
        graph -= noise
        np.fill_diagonal(graph, 0)
        graphs.append((graph > 0).astype('float'))

    return graphs

def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)

def coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, graphs, lr, patience: int = 30):
    # Auxiliary graphon represented by an INR (symmetric in (x,y) by construction).
    model_inr = SirenNet(dim_in = 2, # input [x,y] coordinate
                dim_hidden = inr_dim_hidden,
                dim_out = 1, # output graphon (edge) probability 
                num_layers = len(inr_dim_hidden), # f_theta number of layers
                final_activation = 'sigmoid',
                w0_initial = w0,
                symmetric = True).to(device)
    # GNN model to output the latent variables of the nodes.
    model_gnn = GCN(NODE_FEAT_DIM, gnn_dim_hidden, 1).to(device)
    # End-to-end model of Step 1
    model_SIGL = SIGL(model_gnn, model_inr).to(device)
    # Train the model
    graph_data = [nx2torch(graph_i) for graph_i in graphs]
    optimizer = torch.optim.Adam(model_SIGL.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    model_SIGL.train()
    best_loss = float('inf')
    bad_epochs = 0
    for epoch in range(1, n_epochs + 1):
        random.shuffle(graph_data)
        loss_e = 0
        for data_i in graph_data:
            edges_pred, _ = model_SIGL(data_i)
            preds = edges_pred.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
            loss = criterion(preds, data_i.y)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_e = loss_e + loss.item()
        if epoch == 1 or epoch % epoch_show == 0:
            print("epoch: ", epoch, " loss: ", loss_e)
        if loss_e < best_loss - 1e-5:
            best_loss = loss_e
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping Step1 at epoch {epoch} (best loss {best_loss:.5f})")
                break

    return model_SIGL, loss_e

def train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size, isparametric=0, patience: int = 30):
    inr_model = SirenNet(dim_in = 2 + isparametric , # input [x,y] coordinate
                dim_hidden = inr_dim_hidden,
                dim_out = 1, # output graphon (edge) probability 
                num_layers = len(inr_dim_hidden), # f_theta number of layers
                final_activation = 'sigmoid',
                w0_initial = w0,
                symmetric = True).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_all, y_all, w_all), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(inr_model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss(reduction='none')
    best_loss = float('inf')
    bad_epochs = 0
    for epoch in range(1, n_epochs+1):
        loss_e = 0
        for loader_idx, (X, y, w) in enumerate(train_loader):
            X, y, w = X.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            y_pred = inr_model(X).clamp(1e-6, 1.0 - 1e-6)
            y_target = y.clamp(0.0, 1.0)
            loss = criterion(y_pred, y_target)
            weighted_loss = torch.mean(loss * w)
            weighted_loss.backward()
            optimizer.step()
            loss_e = loss_e + weighted_loss.item()
        if epoch==1 or epoch % epoch_show == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss_e))
        if loss_e < best_loss - 1e-5:
            best_loss = loss_e
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping Step3 at epoch {epoch} (best loss {best_loss:.5f})")
                break

    return inr_model

def get_graphon(Res, model):
    # Square grid coordinates:
    x_coord = (np.arange(Res) + 0.5) / Res
    y_coord = (np.arange(Res) + 0.5) / Res
    xx, yy = np.meshgrid(x_coord, y_coord)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    X_torch = torch.tensor(X, dtype=torch.float).to(device)
    # get the graphon on grid
    model.eval()
    with torch.no_grad():
        graphon_upper = model(X_torch)
    graphon = graphon_upper.cpu().numpy().reshape(Res, Res)
    graphon = (graphon + graphon.T) / 2
    np.fill_diagonal(graphon, 0)
    return graphon

def save_graphon_for_diff(true_graphon, output_dir):
    true_graphon = torch.from_numpy(true_graphon).unsqueeze(0)
    save_image(true_graphon, output_dir)

def plot_smaples_syn(true_graphon, predictions_graphon, output_dir, model_SIGL, offset):
        # plot the true graphon and the predicted graphon
        plt.figure(figsize=(4, 4))
        mpl.rcParams['font.family'] = 'serif'
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.imshow(true_graphon)
        plt.axis('off')
        plt.title(r'True graphon $\omega$', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_dir + "/True_graphon.jpg")

        plt.figure(figsize=(4, 4))
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.imshow(predictions_graphon)
        plt.axis('off')
        plt.title(r'Predicted graphon $f_{\theta^*}(x,y)$', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_dir + "/Predicted_graphon.jpg")

        # Generate a new graph with 150 nodes and plot the learned latent variables of nodes and the sorted graph
        N = 150
        model_SIGL.eval()
        test_graph = simulate_graphs(true_graphon, num_graphs=1, graph_size='fixed', num_nodes=N)[0]
        data_i = nx2torch(test_graph)
        _, coords = model_SIGL(data_i)
        coords = coords.cpu().detach().numpy()
        
        plt.figure(figsize=(4, 4))
        plt.scatter([np.sum(test_graph[:,j]) for j in range(test_graph.shape[0])], coords)
        plt.xlabel(r'Node degree', fontsize=16)
        plt.ylabel(r'$\hat{\boldsymbol{\eta}}$', fontsize=16)
        plt.title(r'Estimated latent variables', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_dir + "/Latent_vars.jpg")

        sorted_idx = np.argsort(coords, axis=0).squeeze()
        test_graph_sorted = test_graph[sorted_idx,:][:, sorted_idx]
        h = int(np.log(N))
        A = torch.tensor(test_graph_sorted, dtype=torch.float, device=device)
        A = A.view(1, 1, N, N)
        H = F.avg_pool2d(A, kernel_size=(h, h))
        H = H.cpu().numpy().squeeze()

        plt.figure(figsize=(4, 4))
        plt.imshow(H)
        plt.axis('off')
        plt.title(r'(Sorted) histogram approximation', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_dir + "/Histogram_app.jpg")



def plot_smaples_real(sample_graph, predictions_graphon, output_dir, model_ISGL):
        model_ISGL.eval()
        data_i = nx2torch(sample_graph)
        _, coords = model_ISGL(data_i)
        perm = torch.argsort(coords.squeeze(-1))
        perm = perm.cpu().numpy()  # Convert torch permutation to numpy array
        true_graph_sorted = sample_graph[perm,:][:, perm]
        coords = coords.cpu().detach().numpy()

        # plot the true graph and sorted graph based on the coordinates
        plt.figure(figsize=(15, 5))
        mpl.rcParams['font.family'] = 'serif'
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

        plt.subplot(1, 4, 1)
        plt.spy(sample_graph, markersize=1)
        plt.title(r'True graph', fontsize=18)

        plt.subplot(1, 4, 2)
        plt.scatter([np.sum(sample_graph[:,j]) for j in range(sample_graph.shape[0])], coords)
        plt.xlabel(r'Node degree', fontsize=16)
        plt.ylabel(r'$\hat{\boldsymbol{\eta}}$', fontsize=16)
        plt.title(r'Estimated latent variables', fontsize=18)

        plt.subplot(1, 4, 3)
        plt.spy(true_graph_sorted, markersize=1)
        plt.title(r'True graph sorted', fontsize=18)

        plt.subplot(1, 4, 4)
        plt.imshow(predictions_graphon, cmap='viridis')
        plt.axis('off')
        plt.title(r'Predicted graphon $f_{\theta^*}(x,y)$', fontsize=18)

        plt.tight_layout()
        plt.savefig(output_dir + "/output.jpg", dpi=300)
        

def load_graphs(filepath):
    """Load graphs from eventually compressed pickle."""

    with open(filepath, 'rb') as f:
        graphs = pickle.load(f)
    if isinstance(graphs[0], np.ndarray):
        return graphs
    elif isinstance(graphs[0], csr_matrix):
        return [graph.toarray() for graph in graphs]
    else:
        raise ValueError('Unsupported input type.')


def graphon_total_variation(W: np.ndarray) -> float:
    """Anisotropic TV of a Res x Res graphon, normalized per pixel."""
    dx = np.abs(np.diff(W, axis=0)).sum()
    dy = np.abs(np.diff(W, axis=1)).sum()
    return float((dx + dy) / (W.size + 1e-12))


def graphon_to_degree_dist(W: np.ndarray, n_bins: int = 200) -> np.ndarray:
    """Marginal degree density implied by the estimated graphon."""
    deg = W.mean(axis=1)
    hist, _ = np.histogram(deg, bins=n_bins, range=(0.0, 1.0), density=True)
    return hist


def _normalized_degrees(graphs):
    out = []
    for A in graphs:
        n = A.shape[0]
        if n <= 1:
            continue
        out.append(A.sum(axis=1) / (n - 1))
    return np.concatenate(out) if out else np.zeros(0)


def _spectral_samples(graphs):
    """Normalized Laplacian eigenvalues, concatenated across input graphs."""
    eigs = []
    for A in graphs:
        n = A.shape[0]
        if n <= 1:
            continue
        d = A.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        L = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
        eigs.append(np.clip(np.linalg.eigvalsh(L), 0.0, 2.0))
    return np.concatenate(eigs) if eigs else np.zeros(0)


def proxy_metrics(W: np.ndarray, graphs) -> dict:
    """Return degree-W1, spectral-W1 and TV between graphon W and its input graphs.

    These are diagnostic metrics for the real-data branch where no ground-truth
    graphon is available. Smaller W1 means the estimated graphon better matches
    the empirical degree / spectral distributions; TV measures smoothness.
    """
    emp_deg = _normalized_degrees(graphs)
    pred_deg = W.mean(axis=1)
    deg_w1 = float(wasserstein_distance(emp_deg, pred_deg)) if emp_deg.size else float("nan")

    emp_spec = _spectral_samples(graphs)
    n = W.shape[0]
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L_w = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    pred_spec = np.clip(np.linalg.eigvalsh(L_w), 0.0, 2.0)
    spec_w1 = float(wasserstein_distance(emp_spec, pred_spec)) if emp_spec.size else float("nan")

    return {
        "degree_w1": deg_w1,
        "spectral_w1": spec_w1,
        "tv": graphon_total_variation(W),
    }
