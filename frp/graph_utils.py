import random
import copy
from typing import Literal, Optional, Union
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import markers
import scipy.optimize as sopt
import networkx as nx
from matplotlib import rc


def get_adj_matrix(graph: ig.Graph):
    return np.array(graph.get_adjacency().data)


# networkx to plot graph
def plot_graph(
    ax: plt.Axes,
    graph: ig.Graph,
    latex: bool = False,
    node_size=400,
    node_color="white",
    edgecolors="k",
    **kwargs,
):
    # enable latex rendering
    if latex:
        rc("text", usetex=True)
        plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(len(graph.vs)))
    not_two_cycle_edgelist = []
    two_cycle_edgelist = []
    full_edgelist = graph.get_edgelist()
    for edge in full_edgelist:
        if (edge[1], edge[0]) in full_edgelist:
            two_cycle_edgelist.append(edge)
        else:
            not_two_cycle_edgelist.append(edge)

    pos = nx.circular_layout(nx_graph)
    nx.draw_networkx_edges(
        nx_graph, pos, two_cycle_edgelist, ax=ax, connectionstyle="arc3,rad=0.15"
    )
    nx.draw_networkx_edges(nx_graph, pos, not_two_cycle_edgelist, ax=ax)

    if "labels" not in kwargs:
        kwargs["labels"] = {i: f"${i+1}$" for i in range(len(graph.vs))}

    nx.draw(
        nx_graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors=edgecolors,
        **kwargs,
    )

    ax.set_title(kwargs.get("title", "Graph"))
    ax.set_axis_off()

    x_min = min([vertex[0] for vertex in pos.values()])
    x_max = max([vertex[0] for vertex in pos.values()])
    y_min = min([vertex[1] for vertex in pos.values()])
    y_max = max([vertex[1] for vertex in pos.values()])
    margin = kwargs.get("margin", 0.2)
    ax.set_xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
    ax.set_ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))

    if latex:
        rc("text", usetex=False)


def theo_theta_sparsity(p, edge_num):
    c = edge_num / (p * (p - 1))
    return (1 - c) ** 2 * (1 - c**2) ** (p - 2)


def get_edge_num_from_sparsity(p, target):
    res = sopt.minimize(
        lambda e: (theo_theta_sparsity(p, e) - target) ** 2, 0, method="Nelder-Mead"
    )
    return int(np.round(res.x[0]))


def generate_uniform_graph(p: int, deg: int, mode: Literal["in", "out"]):
    edges = []

    for idx in range(p):
        adjs = (np.random.permutation(p - 1) + 1)[:deg]
        if mode == "out":
            edges += [[idx, (idx + adj) % p] for adj in adjs]
        elif mode == "in":
            edges += [[(idx - adj) % p, idx] for adj in adjs]
        else:
            raise ValueError(f"mode should be in ['in', 'out'].")

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def generate_sparse_graph(p: int, edge_prob: float):
    edges = []

    for idx in range(p):
        for jdx in range(p):
            if idx != jdx and np.random.rand(1) < edge_prob:
                edges.append([idx, jdx])

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def generate_erdos_renyi_dg(p: int, edge_num: int, seed: Optional[int] = None):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    graph = ig.Graph.Erdos_Renyi(n=p, m=edge_num, directed=True)
    B = get_adj_matrix(graph)
    graph = ig.Graph.Adjacency(B)

    return graph


def generate_dag(p: int, edge_num: int, seed: Optional[int] = None):
    assert edge_num <= p * (p - 1) / 2, "edge_num should be less than p * (p - 1) / 2"
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    # select edge_num edges
    edges = np.random.choice(np.arange(p * (p - 1) / 2), edge_num, replace=False)

    # build adjacency matrix
    B = np.zeros((p, p))
    for edge in edges:
        i = int((-1 + (1 + 8 * edge) ** 0.5) / 2)
        j = int(edge - i * (i + 1) / 2)
        B[i + 1, j] = 1

    # create a topological order
    order = np.random.permutation(p)
    B = B[order, :][:, order]
    graph = ig.Graph.Adjacency(B)

    assert graph.is_dag(), "generated graph is not a DAG"

    return graph


def generate_tree(p: int, children: Optional[int] = None, seed: Optional[int] = None):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    if children is None:
        return ig.Graph.Tree_Game(p, directed=True)
    else:
        return ig.Graph.Tree(p, children, type="out")


def generate_in_tree(p: int, children: int, seed: Optional[int] = None):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    return ig.Graph.Tree(p, children, type="in")


def generate_converging_tree(p: int):
    """
    p: number of nodes
    edges = {2 -> 1, 3 -> 1, ..., p -> 1}
    """
    edges = []

    for idx in range(p):
        if idx == 0:
            continue
        edges.append([idx, 0])

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def generate_diverging_tree(p: int):
    """
    p: number of nodes
    edges = {1 -> 2, 1 -> 3, ..., 1 -> p}
    """
    edges = []

    for idx in range(p):
        if idx == 0:
            continue
        edges.append([0, idx])

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def generate_two_cycles_chain(p: int):
    """
    p: number of nodes
    edges = {1 -> 2, 2 -> 3, ..., p-1 -> p} + {2 -> 1, 3 -> 2, ..., p -> p-1}
    """
    edges = []

    for idx in range(p):
        edges.append([(idx - 1) % p, idx % p])
        edges.append([idx % p, (idx - 1) % p])

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def generate_full_cycle(p: int):
    """
    p: number of nodes
    edges = {1 -> 2, 2 -> 3, ..., p-1 -> p}
    """
    edges = []

    for idx in range(p):
        edges.append([(idx - 1) % p, idx % p])

    graph = ig.Graph(n=p, edges=edges, directed=True)
    return graph


def edge_recall(
    g_true: Union[ig.Graph, np.ndarray], g_est: Union[ig.Graph, np.ndarray]
):
    """Compute recall of estimated graph."""
    B_true = get_adj_matrix(g_true) if isinstance(g_true, ig.Graph) else g_true
    B_est = get_adj_matrix(g_est) if isinstance(g_est, ig.Graph) else g_est

    assert B_true.shape == B_est.shape

    if B_true.sum() > 0:
        return (B_true & B_est).sum() / B_true.sum()
    else:
        return np.nan


def edge_precision(g_true: ig.Graph, g_est: ig.Graph):
    """Compute precision of estimated graph."""
    B_true = get_adj_matrix(g_true) if isinstance(g_true, ig.Graph) else g_true
    B_est = get_adj_matrix(g_est) if isinstance(g_est, ig.Graph) else g_est

    assert B_true.shape == B_est.shape

    if B_est.sum() > 0:
        return (B_true & B_est).sum() / B_est.sum()
    else:
        return np.nan


def two_cycle_recall(g_true: ig.Graph, g_est: ig.Graph):
    """Compute recall of 2-cycles of estimated graph."""
    B_true = get_adj_matrix(g_true)
    B_est = get_adj_matrix(g_est)

    assert B_true.shape[0] == B_est.shape[0]

    TP = 0
    FN = 0

    for i in range(B_true.shape[0]):
        for j in range(i):
            if B_true[i, j] and B_true[j, i]:
                if B_est[i, j] and B_est[j, i]:
                    TP += 1
                else:
                    FN += 1

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return np.nan


def two_cycle_precision(g_true: ig.Graph, g_est: ig.Graph):
    """Compute precision of 2-cycles of estimated graph."""
    B_true = get_adj_matrix(g_true)
    B_est = get_adj_matrix(g_est)

    assert B_true.shape[0] == B_est.shape[0]

    TP = 0
    FP = 0

    for i in range(B_true.shape[0]):
        for j in range(i):
            if B_est[i, j] and B_est[j, i]:
                if B_true[i, j] and B_true[j, i]:
                    TP += 1
                else:
                    FP += 1

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return np.nan


def check_equal(graph_1, graph_2):
    """Check if two graphs are equal."""
    B_1 = get_adj_matrix(graph_1)
    B_2 = get_adj_matrix(graph_2)

    return np.array_equal(B_1, B_2)
