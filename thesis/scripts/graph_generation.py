import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# Global configuration
# ============================================================

SEED = 42
N = 60

OUTDIR = "graph_models"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Plotting utilities
# ============================================================

def save_graph(
    G,
    filename,
    pos,
    node_size=28,
    node_color="#111111",
    edge_color="#808080",
    edge_width=0.65,
    edge_alpha=0.55,
):
    """
    Save one graph as both vector PDF and high-resolution PNG.

    No title is placed inside the figure:
    model name and parameters should be reported in the LaTeX caption.
    """

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_width,
        alpha=edge_alpha,
        edge_color=edge_color,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        linewidths=0,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.tight_layout(pad=0.02)

    fig.savefig(
        os.path.join(OUTDIR, filename + ".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        os.path.join(OUTDIR, filename + ".png"),
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


def four_block_layout(
    groups,
    center_radius=1.45,
    block_radius=0.52,
):
    """
    Place four communities at the corners of a square.

    Nodes belonging to each block are placed on a small circle.
    Using the same layout for SBM and DC-SBM makes their
    structural difference easier to interpret.
    """

    centers = [
        (-center_radius, +center_radius),
        (+center_radius, +center_radius),
        (-center_radius, -center_radius),
        (+center_radius, -center_radius),
    ]

    pos = {}

    for block_id, nodes in enumerate(groups):

        angles = np.linspace(
            0,
            2 * np.pi,
            len(nodes),
            endpoint=False,
        )

        # Slightly rotate each block
        angles += block_id * np.pi / 8

        cx, cy = centers[block_id]

        for node, angle in zip(nodes, angles):

            pos[node] = np.array([
                cx + block_radius * np.cos(angle),
                cy + block_radius * np.sin(angle),
            ])

    return pos


# Consistent block colors for SBM and DC-SBM.
BLOCK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
]


# ============================================================
# 3.1.1 Erdős-Rényi
# ============================================================
#
# Homogeneous random connectivity:
# every possible edge has the same probability p.
#
# p=0.10 is intentionally a little larger than the thesis
# experimental dataset value so that the illustrative graph
# is connected and readable with this fixed seed.
#

G_er = nx.erdos_renyi_graph(
    n=N,
    p=0.10,
    seed=SEED,
)

pos_er = nx.spring_layout(
    G_er,
    seed=SEED,
    k=0.30,
    iterations=500,
)

save_graph(
    G_er,
    "erdos_renyi",
    pos_er,
)


# ============================================================
# 3.1.2 Stochastic Block Model
# ============================================================
#
# Four very clear communities.
#
# High intra-community probability:
#     p_in = 0.55
#
# Very low inter-community probability:
#     p_out = 0.006
#

block_sizes = [15, 15, 15, 15]

p_in = 0.55
p_out = 0.006

P = np.full(
    (4, 4),
    p_out,
)

np.fill_diagonal(
    P,
    p_in,
)

G_sbm = nx.stochastic_block_model(
    block_sizes,
    P,
    seed=SEED,
)

groups = [
    list(range(0, 15)),
    list(range(15, 30)),
    list(range(30, 45)),
    list(range(45, 60)),
]

pos_blocks = four_block_layout(groups)

sbm_colors = [
    BLOCK_COLORS[b]
    for b in range(4)
    for _ in range(15)
]

save_graph(
    G_sbm,
    "stochastic_block_model",
    pos_blocks,
    node_color=sbm_colors,
    node_size=30,
    edge_alpha=0.50,
)


# ============================================================
# 3.1.3 Degree-Corrected Stochastic Block Model
# ============================================================
#
# Same four-community organization as the SBM,
# but nodes inside each block have very different
# degree propensities theta_i.
#
# We use:
#
# lambda_ij = theta_i * theta_j * omega[g_i, g_j]
#
# and then, for a simple graph:
#
# p_ij = 1 - exp(-lambda_ij)
#
# This makes degree heterogeneity within communities
# visually evident.
#

rng = np.random.default_rng(SEED)

z = np.repeat(
    np.arange(4),
    15,
)

# Strongly heterogeneous propensity profile.
#
# Each block receives the same profile, but randomly
# permuted among its nodes.
degree_profile = np.array([
    4.00,
    3.00,
    2.00,
    1.50,
    1.20,
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
    0.35,
])

# Normalize so that the average propensity is 1.
degree_profile /= degree_profile.mean()

theta = np.zeros(N)

for block in range(4):

    start = block * 15
    end = start + 15

    theta[start:end] = rng.permutation(
        degree_profile
    )


# Block interaction intensities.
#
# Communities are still clearly separated, but
# intra-community degrees can differ dramatically.
omega = np.full(
    (4, 4),
    0.004,
)

np.fill_diagonal(
    omega,
    0.28,
)


G_dcsbm = nx.Graph()
G_dcsbm.add_nodes_from(range(N))

for i in range(N):

    for j in range(i + 1, N):

        lam = (
            theta[i]
            * theta[j]
            * omega[z[i], z[j]]
        )

        p = 1.0 - np.exp(-lam)

        if rng.random() < p:
            G_dcsbm.add_edge(i, j)


dcsbm_colors = [
    BLOCK_COLORS[int(z[v])]
    for v in G_dcsbm.nodes()
]

# Degree is also encoded through node size.
#
# This is intentionally NOT done for the ordinary SBM.
dcsbm_sizes = [
    25 + 11 * G_dcsbm.degree(v)
    for v in G_dcsbm.nodes()
]

save_graph(
    G_dcsbm,
    "degree_corrected_sbm",
    pos_blocks,
    node_color=dcsbm_colors,
    node_size=dcsbm_sizes,
    edge_alpha=0.48,
)


# ============================================================
# 3.1.4 Configuration Model
# ============================================================
#
# Deliberately bimodal degree sequence:
#
# 10 nodes have degree 10
# 50 nodes have degree 2
#
# This makes the prescribed-degree property visually clear.
#
# random_degree_sequence_graph returns a SIMPLE graph while
# preserving the supplied degree sequence exactly.
#

degree_sequence = (
    [10] * 10
    + [2] * 50
)

assert len(degree_sequence) == N
assert sum(degree_sequence) % 2 == 0
assert nx.is_graphical(degree_sequence)

G_conf = nx.random_degree_sequence_graph(
    degree_sequence,
    seed=SEED,
    tries=1000,
)

pos_conf = nx.kamada_kawai_layout(
    G_conf
)

save_graph(
    G_conf,
    "configuration_model",
    pos_conf,
)


# ============================================================
# 3.1.5 Expected-degree model
# ============================================================
#
# The node parameters vary smoothly from low to high
# expected degree.
#
# Unlike the Configuration Model, these degrees are not
# fixed exactly: they determine edge probabilities.
#

expected_degrees = np.linspace(
    1.0,
    12.0,
    N,
)

G_exp = nx.expected_degree_graph(
    expected_degrees,
    seed=SEED,
    selfloops=False,
)

pos_exp = nx.spring_layout(
    G_exp,
    seed=SEED,
    k=0.25,
    iterations=500,
)

save_graph(
    G_exp,
    "expected_degree",
    pos_exp,
)


# ============================================================
# 3.1.6 Barabási-Albert
# ============================================================
#
# m=1 is intentionally chosen for the illustrative figure.
#
# Every new node introduces one edge.
# Preferential attachment therefore produces a sparse
# tree-like graph with extremely visible hubs.
#
# This is NOT the m=3 value used in the OGAD experiments.
#

G_ba = nx.barabasi_albert_graph(
    n=N,
    m=1,
    seed=SEED,
)

pos_ba = nx.spring_layout(
    G_ba,
    seed=SEED,
    k=0.28,
    iterations=500,
)

ba_sizes = [
    24 + 8 * G_ba.degree(v)
    for v in G_ba.nodes()
]

save_graph(
    G_ba,
    "barabasi_albert",
    pos_ba,
    node_size=ba_sizes,
)


# ============================================================
# 3.1.7 Watts-Strogatz
# ============================================================
#
# Dense local ring:
#     k = 8
#
# Very low rewiring probability:
#     p = 0.03
#
# This preserves the ring-lattice appearance while
# introducing a few clearly visible long-range shortcuts.
#

G_ws = nx.watts_strogatz_graph(
    n=N,
    k=8,
    p=0.03,
    seed=SEED,
)

pos_ws = nx.circular_layout(
    G_ws
)

save_graph(
    G_ws,
    "watts_strogatz",
    pos_ws,
    node_size=26,
    edge_alpha=0.42,
    edge_width=0.55,
)


# ============================================================
# Optional sanity check
# ============================================================

def print_stats(name, G):

    degrees = np.array([
        d
        for _, d in G.degree()
    ])

    print(
        f"{name:20s} "
        f"n={G.number_of_nodes():2d}  "
        f"m={G.number_of_edges():3d}  "
        f"avg_deg={degrees.mean():5.2f}  "
        f"min={degrees.min():2d}  "
        f"max={degrees.max():2d}  "
        f"clustering={nx.average_clustering(G):.3f}"
    )


print("\nGenerated graphs:\n")

print_stats(
    "Erdos-Renyi",
    G_er,
)

print_stats(
    "SBM",
    G_sbm,
)

print_stats(
    "DC-SBM",
    G_dcsbm,
)

print_stats(
    "Configuration",
    G_conf,
)

print_stats(
    "Expected-degree",
    G_exp,
)

print_stats(
    "Barabasi-Albert",
    G_ba,
)

print_stats(
    "Watts-Strogatz",
    G_ws,
)