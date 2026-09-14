"""Role 3 - detailed analysis of the Role 1/Role 2 opinion network.

Consumes Role 2's constructed graphs (never rebuilds the node/edge definition
itself) and answers the questions Role 2 explicitly left open: which
respondents are central, whether the network's community structure is real
or a k-NN artefact, whether opinion camps persist across topics, and whether
enthusiasm (intensity) versus actual opinion content is what organises the
network.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

DOMAINS = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics & Society",
    "V": "Environment",
}
DOMAIN_COLORS = {"T": "#3B6FB6", "E": "#E08A2E", "S": "#8E5EA2", "V": "#3E9651"}

SEED = 42


@dataclass(frozen=True)
class Role3Paths:
    root: Path

    @property
    def role1_tables(self) -> Path:
        return self.root.parent / "role1-data-preparation" / "outputs" / "tables"

    @property
    def role2_tables(self) -> Path:
        return self.root.parent / "role2-network-construction" / "outputs" / "tables"

    @property
    def role2_graphs(self) -> Path:
        return self.root.parent / "role2-network-construction" / "outputs" / "graphs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"


def ensure_output_dirs(paths: Role3Paths) -> None:
    for d in (paths.tables, paths.figures):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- inputs

def read_inputs(paths: Role3Paths) -> dict:
    """Load Role 1's prepared tables (needed to rebuild the pipeline under
    conditions Role 1/2 never ran: shuffled data, alternative k, full
    per-domain similarity matrices) plus the one Role 2 table with no graph
    equivalent (the exploratory statement edges, for the FDR comparison).
    The respondent and topic-layer graphs themselves are NOT re-parsed from
    CSV here -- see `load_role2_graph` / `load_role2_topic_layer`, which read
    Role 2's own GraphML/GEXF exports directly.
    """

    r1, r2 = paths.role1_tables, paths.role2_tables
    data = {
        "network_sample": pd.read_csv(r1 / "network_sample.csv", index_col=0),
        "similarity": pd.read_csv(r1 / "similarity_matrix.csv", index_col=0),
        "prepared_responses": pd.read_csv(r1 / "prepared_responses.csv", index_col=0),
        "codebook": pd.read_csv(r1 / "codebook.csv", index_col=0),
        "item_statistics": pd.read_csv(r1 / "item_statistics.csv", index_col=0),
        "item_edges": pd.read_csv(r2 / "item_edges_visual.csv"),
    }
    data["similarity"].index = data["similarity"].index.astype(int)
    data["similarity"].columns = data["similarity"].columns.astype(int)
    return data


def load_role2_graph(paths: Role3Paths) -> nx.Graph:
    """Load Role 2's main respondent network exactly as it exported it."""

    return nx.read_graphml(paths.role2_graphs / "respondent_opinion_network.graphml", node_type=int)


def load_role2_topic_layer(paths: Role3Paths, prefix: str) -> nx.Graph:
    """Load one of Role 2's topic-layer networks exactly as it exported it."""

    return nx.read_gexf(paths.role2_graphs / f"topic_layer_{prefix}.gexf", node_type=int)


# ------------------------------------------------- shared construction
# Reproduced identically from Role 1 (preprocess.py / similarity.py) so the
# null model and the k-sensitivity scan rebuild the *same* network, not an
# approximation of it.

def item_centre(num: pd.DataFrame) -> pd.DataFrame:
    return (num - num.mean()).fillna(0.0)


def profile_correlation(centred: pd.DataFrame) -> np.ndarray:
    M = np.asarray(centred, dtype=float)
    M = M - M.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = M / norms
    return U @ U.T


def knn_graph(S: np.ndarray, k: int, labels) -> nx.Graph:
    labels = list(labels)
    S = np.array(S, dtype=float, copy=True)
    np.fill_diagonal(S, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(labels)
    for i in range(len(labels)):
        for j in np.argsort(-S[i], kind="stable")[:k]:
            if S[i, j] > 0:
                G.add_edge(labels[i], labels[j], weight=float(S[i, j]), distance=float(1 - S[i, j]))
    return G


def default_k(n: int) -> int:
    return int(round(np.sqrt(n)))


# --------------------------------------------------- centrality & roles

def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True, seed=SEED)
    closeness = nx.closeness_centrality(G, distance="distance")
    eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    clustering = nx.clustering(G, weight="weight")

    df = pd.DataFrame({
        "degree": degree,
        "strength": strength,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
        "clustering": clustering,
    })
    df.index.name = "respondent"
    extra = ["intensity", "distance_to_class_mean", "n_answered",
             "T_mean", "E_mean", "S_mean", "V_mean"]
    for col in extra:
        df[col] = pd.Series({n: G.nodes[n].get(col) for n in G.nodes()})
    return df.sort_index()


def detect_communities(G: nx.Graph, seed: int = SEED, weight: str = "weight"):
    communities = nx.algorithms.community.louvain_communities(G, weight=weight, seed=seed)
    modularity = nx.algorithms.community.modularity(G, communities, weight=weight)
    membership = {node: i for i, com in enumerate(communities) for node in com}
    return communities, membership, modularity


def guimera_amaral_roles(G: nx.Graph, communities, weight: str = "weight") -> pd.DataFrame:
    """Within-module strength z-score and participation coefficient (Guimera & Amaral, 2005).

    Uses edge weight (not raw degree) as the connection strength, consistent
    with this being a weighted similarity network.
    """

    membership = {node: i for i, com in enumerate(communities) for node in com}
    rows = []
    within_by_module: dict[int, list[float]] = {i: [] for i in range(len(communities))}
    node_within = {}
    node_total = {}
    for node in G.nodes():
        m = membership[node]
        total = 0.0
        within = 0.0
        to_module: dict[int, float] = {}
        for nbr in G.neighbors(node):
            w = G[node][nbr].get(weight, 1.0)
            total += w
            to_module[membership[nbr]] = to_module.get(membership[nbr], 0.0) + w
            if membership[nbr] == m:
                within += w
        node_within[node] = within
        node_total[node] = total
        rows.append((node, m, total, within, to_module))
        within_by_module[m].append(within)

    module_stats = {m: (np.mean(v), np.std(v) if np.std(v) > 0 else 1.0) for m, v in within_by_module.items()}

    out = []
    for node, m, total, within, to_module in rows:
        mean_k, std_k = module_stats[m]
        z = (within - mean_k) / std_k
        participation = 1.0 - sum((w / total) ** 2 for w in to_module.values()) if total > 0 else 0.0
        out.append({"respondent": node, "community": m, "within_module_strength": within,
                    "within_module_z": z, "participation_coefficient": participation})
    return pd.DataFrame(out).set_index("respondent").sort_index()


# -------------------------------------------------------- null models

def permute_items(num: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = num.copy()
    for c in out.columns:
        out[c] = rng.permutation(out[c].to_numpy())
    return out


def null_model_modularity(sample: pd.DataFrame, k: int, n_rep: int, seed: int = SEED) -> pd.DataFrame:
    """Rebuild the full pipeline (shuffle items -> centre -> correlate -> k-NN)
    n_rep times and record community/structure statistics of the null graphs.
    This is the validation Role 1 recommended: k-NN graphs are clustered and
    modular *by construction*, so raw modularity needs a null comparison.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rep):
        null_sample = permute_items(sample, rng)
        S = profile_correlation(item_centre(null_sample))
        G = knn_graph(S, k, labels=sample.index)
        communities, _, modularity = detect_communities(G, seed=seed)
        clustering = nx.average_clustering(G, weight="weight")
        components = nx.number_connected_components(G)
        rows.append({"rep": i, "modularity": modularity, "n_communities": len(communities),
                     "avg_clustering": clustering, "components": components})
    return pd.DataFrame(rows)


def k_sensitivity(similarity: pd.DataFrame, k_values, seed: int = SEED) -> pd.DataFrame:
    labels = similarity.index
    S = similarity.to_numpy()
    rows = []
    for k in k_values:
        G = knn_graph(S, k, labels=labels)
        degrees = [d for _, d in G.degree()]
        communities, _, modularity = detect_communities(G, seed=seed)
        rows.append({
            "k": k, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "density": nx.density(G), "components": nx.number_connected_components(G),
            "min_degree": min(degrees), "max_degree": max(degrees),
            "mean_degree": float(np.mean(degrees)),
            "modularity": modularity, "n_communities": len(communities),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------- topic-layer comparison

def topic_layer_similarity_matrices(sample: pd.DataFrame, common_respondents) -> dict[str, pd.DataFrame]:
    """Full (unsparsified) domain-level profile-correlation matrices, restricted
    to the respondents common to every topic layer -- for Mantel comparison.
    """

    matrices = {}
    for prefix in DOMAINS:
        cols = [c for c in sample.columns if c.startswith(prefix)]
        sub = sample.loc[common_respondents, cols]
        S = profile_correlation(item_centre(sub))
        matrices[prefix] = pd.DataFrame(S, index=common_respondents, columns=common_respondents)
    return matrices


def mantel_test(A: pd.DataFrame, B: pd.DataFrame, n_perm: int = 999, seed: int = SEED):
    """Mantel test: correlation between two distance/similarity matrices over
    the same node set, with permutation-based significance (labels of one
    matrix are shuffled to build the null distribution of the correlation).
    """

    common = A.index.intersection(B.index)
    a = A.loc[common, common].to_numpy()
    b = B.loc[common, common].to_numpy()
    iu = np.triu_indices(len(common), 1)
    av, bv = a[iu], b[iu]
    observed = float(np.corrcoef(av, bv)[0, 1])

    rng = np.random.default_rng(seed)
    n = len(common)
    perm_r = np.empty(n_perm)
    for p in range(n_perm):
        order = rng.permutation(n)
        bp = b[np.ix_(order, order)]
        perm_r[p] = np.corrcoef(av, bp[iu])[0, 1]
    p_value = float((np.abs(perm_r) >= abs(observed)).mean())
    return observed, p_value, len(common)


def normalized_mutual_information(part_a: dict, part_b: dict, nodes) -> float:
    """NMI (arithmetic-mean normalisation) between two community partitions,
    restricted to a common node set. No sklearn dependency.
    """

    a = np.array([part_a[n] for n in nodes])
    b = np.array([part_b[n] for n in nodes])
    n = len(nodes)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return float(-(p * np.log(p)).sum())

    contingency = pd.crosstab(a, b).to_numpy()
    pxy = contingency / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    ha, hb = entropy(a), entropy(b)
    if ha + hb == 0:
        return 1.0
    return 2 * mi / (ha + hb)


def compare_layers(paths: Role3Paths, sample: pd.DataFrame, main_similarity: pd.DataFrame,
                    main_membership: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the main network against each Role-2 topic layer, and the topic
    layers against each other. Topic-layer graphs are Role 2's own GEXF
    exports (loaded as-is, not rebuilt); only the *full* per-domain similarity
    matrices (needed for the Mantel test, which Role 2 never computed --
    it only exported the sparsified k-NN version) are derived here from
    Role 1's raw sample.
    """

    topic_graphs = {p: load_role2_topic_layer(paths, p) for p in DOMAINS}
    common = sorted(set.intersection(*[set(G.nodes()) for G in topic_graphs.values()]))

    matrices = topic_layer_similarity_matrices(sample, common)
    matrices["Main"] = main_similarity.loc[common, common]

    partitions = {"Main": main_membership}
    for prefix, G in topic_graphs.items():
        _, membership, _ = detect_communities(G, seed=SEED)
        for node in common:
            membership.setdefault(node, -1)
        partitions[prefix] = membership

    names = ["Main", "T", "E", "S", "V"]
    mantel_rows, nmi_rows = [], []
    for a, b in combinations(names, 2):
        r, p, n = mantel_test(matrices[a], matrices[b])
        mantel_rows.append({"layer_a": a, "layer_b": b, "mantel_r": r, "p_value": p, "n_common": n})
        nmi = normalized_mutual_information(partitions[a], partitions[b], common)
        nmi_rows.append({"layer_a": a, "layer_b": b, "nmi": nmi})
    return pd.DataFrame(mantel_rows), pd.DataFrame(nmi_rows)


# --------------------------------------------- statement network with FDR

def spearman_with_pvalues(prepared: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    codes = list(prepared.columns)
    rows = []
    for a, b in combinations(codes, 2):
        pair = prepared[[a, b]].dropna()
        if len(pair) < min_periods:
            continue
        rho, pval = stats.spearmanr(pair[a], pair[b])
        rows.append({"source": a, "target": b, "rho": float(rho), "p_value": float(pval), "n": len(pair)})
    return pd.DataFrame(rows)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    significant = np.empty(m, dtype=bool)
    significant[order] = q <= alpha
    q_full = np.empty(m)
    q_full[order] = q
    return significant, q_full


def fdr_statement_network(corr_table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    sig, q = benjamini_hochberg(corr_table["p_value"].to_numpy(), alpha=alpha)
    out = corr_table.copy()
    out["q_value"] = q
    out["fdr_significant"] = sig
    return out.sort_values("p_value")


# ------------------------------------------------- global network metrics

def global_metrics(G: nx.Graph, modularity: float, n_communities: int) -> dict:
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "diameter_weighted": nx.diameter(largest_cc, weight="distance") if largest_cc.number_of_nodes() > 1 else 0,
        "avg_shortest_path_weighted": nx.average_shortest_path_length(largest_cc, weight="distance"),
        "transitivity": nx.transitivity(G),
        "avg_clustering_weighted": nx.average_clustering(G, weight="weight"),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        "intensity_assortativity": nx.numeric_assortativity_coefficient(G, "intensity"),
        "modularity": modularity,
        "n_communities": n_communities,
    }


def intensity_position_correlation(centrality: pd.DataFrame) -> pd.DataFrame:
    metrics = ["degree", "strength", "betweenness", "closeness", "eigenvector", "clustering"]
    rows = []
    for metric in metrics:
        for attr in ["intensity", "distance_to_class_mean"]:
            r, p = stats.pearsonr(centrality[metric], centrality[attr])
            rows.append({"centrality_metric": metric, "node_attribute": attr,
                         "pearson_r": float(r), "p_value": float(p)})
    return pd.DataFrame(rows)


def community_summary(centrality: pd.DataFrame, membership: dict) -> pd.DataFrame:
    df = centrality.copy()
    df["community"] = pd.Series(membership)
    agg = df.groupby("community").agg(
        size=("degree", "size"),
        mean_intensity=("intensity", "mean"),
        mean_atypicality=("distance_to_class_mean", "mean"),
        mean_T=("T_mean", "mean"),
        mean_E=("E_mean", "mean"),
        mean_S=("S_mean", "mean"),
        mean_V=("V_mean", "mean"),
        mean_betweenness=("betweenness", "mean"),
        mean_clustering=("clustering", "mean"),
    )
    return agg.sort_values("size", ascending=False)
