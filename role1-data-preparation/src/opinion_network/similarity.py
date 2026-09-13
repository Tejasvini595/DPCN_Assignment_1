"""Respondent-respondent similarity and sparsification."""
import networkx as nx
import numpy as np


def _cosine(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = M / norms
    return U @ U.T


def profile_correlation(centred):
    """Pearson correlation between respondents' item-centred profiles.

    Item-centring removes the class consensus on each statement; the
    row-centring implicit in Pearson removes each respondent's overall
    agreement intensity (acquiescence). What remains is *relative*
    opinion content: which statements a person endorses more or less
    than the class does, compared with their own baseline.
    """
    M = np.asarray(centred, dtype=float)
    M = M - M.mean(axis=1, keepdims=True)
    return _cosine(M)


def agreement_similarity(num):
    """Naive similarity: 1 - mean absolute Likert distance / 4 (mean-imputed)."""
    R = num.fillna(num.mean()).to_numpy()
    D = np.abs(R[:, None, :] - R[None, :, :]).mean(axis=2)
    return 1 - D / 4.0


def knn_graph(S, k, labels=None):
    """Symmetric (union) k-nearest-neighbour graph keeping positive similarities.

    Node attribute `hubness` counts how many respondents list the node among
    their k most similar peers (k-occurrence).
    """
    n = S.shape[0]
    labels = list(range(n)) if labels is None else list(labels)
    S = np.array(S, dtype=float, copy=True)
    np.fill_diagonal(S, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(labels)
    hub = np.zeros(n, dtype=int)
    for i in range(n):
        order = np.argsort(-S[i], kind="stable")[:k]
        for j in order:
            if S[i, j] > 0:
                hub[j] += 1
                G.add_edge(labels[i], labels[j], weight=float(S[i, j]))
    nx.set_node_attributes(G, {labels[i]: int(hub[i]) for i in range(n)}, "hubness")
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 - d["weight"]   # for path-based centralities
    return G


def default_k(n):
    return int(round(np.sqrt(n)))
