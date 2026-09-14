"""Build network objects from Role 1's prepared tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


DOMAINS = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics & Society",
    "V": "Environment",
}

DOMAIN_COLORS = {
    "T": "#3B6FB6",
    "E": "#E08A2E",
    "S": "#8E5EA2",
    "V": "#3E9651",
}


@dataclass(frozen=True)
class Role2Paths:
    """Filesystem layout for a role2 run."""

    root: Path

    @property
    def role1_root(self) -> Path:
        return self.root.parent / "role1-data-preparation"

    @property
    def role1_tables(self) -> Path:
        return self.role1_root / "outputs" / "tables"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"

    @property
    def graphs(self) -> Path:
        return self.outputs / "graphs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"


def default_k(n: int) -> int:
    return int(round(np.sqrt(n)))


def read_role1_tables(paths: Role2Paths) -> dict[str, pd.DataFrame]:
    """Load the role1 handoff tables used by role2."""

    tables = paths.role1_tables
    return {
        "network_sample": pd.read_csv(tables / "network_sample.csv", index_col=0),
        "similarity": pd.read_csv(tables / "similarity_matrix.csv", index_col=0),
        "edge_list": pd.read_csv(tables / "edge_list_knn.csv"),
        "prepared_responses": pd.read_csv(tables / "prepared_responses.csv", index_col=0),
        "codebook": pd.read_csv(tables / "codebook.csv", index_col=0),
        "item_statistics": pd.read_csv(tables / "item_statistics.csv", index_col=0),
    }


def respondent_attributes(sample: pd.DataFrame) -> pd.DataFrame:
    """Create visual node attributes from the role1 node set."""

    class_mean = sample.mean()
    raw_imputed = sample.fillna(class_mean)
    attrs = pd.DataFrame(index=sample.index)
    attrs.index.name = "respondent"
    attrs["intensity"] = sample.mean(axis=1)
    attrs["n_answered"] = sample.notna().sum(axis=1)
    attrs["distance_to_class_mean"] = np.sqrt(((raw_imputed - class_mean) ** 2).sum(axis=1))
    for prefix, name in DOMAINS.items():
        cols = [c for c in sample.columns if c.startswith(prefix)]
        attrs[f"{prefix}_mean"] = sample[cols].mean(axis=1)
        attrs[f"{prefix}_answered"] = sample[cols].notna().sum(axis=1)
        attrs[f"{name.lower().replace(' & ', '_').replace(' ', '_')}_mean"] = attrs[f"{prefix}_mean"]
    return attrs


def build_respondent_graph(edge_list: pd.DataFrame, node_attrs: pd.DataFrame) -> nx.Graph:
    """Build the main undirected weighted respondent graph."""

    G = nx.Graph()
    for respondent, row in node_attrs.iterrows():
        G.add_node(int(respondent), **_clean_attrs(row.to_dict()))

    for row in edge_list.itertuples(index=False):
        source = int(row.source)
        target = int(row.target)
        weight = float(row.weight)
        G.add_edge(source, target, weight=weight, distance=float(row.distance))

    _add_degree_attributes(G)
    return G


def build_topic_layers(sample: pd.DataFrame, k: int) -> dict[str, nx.Graph]:
    """Build one respondent k-NN graph per survey domain."""

    keep = pd.Series(True, index=sample.index)
    for prefix in DOMAINS:
        cols = [c for c in sample.columns if c.startswith(prefix)]
        keep &= sample[cols].notna().sum(axis=1) >= 12

    respondents = sample.index[keep]
    layers: dict[str, nx.Graph] = {}
    for prefix in DOMAINS:
        cols = [c for c in sample.columns if c.startswith(prefix)]
        sub = sample.loc[respondents, cols]
        S = profile_correlation(item_centre(sub))
        G = knn_graph(S, k, labels=respondents)
        nx.set_node_attributes(G, prefix, "domain_layer")
        _add_degree_attributes(G)
        layers[prefix] = G
    return layers


def build_item_network(
    prepared: pd.DataFrame,
    codebook: pd.DataFrame,
    item_stats: pd.DataFrame,
    threshold: float = 0.35,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Build an exploratory statement network using pairwise Spearman rho.

    Role 1 recommends Spearman for statement-to-statement association. This
    visual network uses a transparent absolute-rho cutoff and is labelled as
    exploratory; role3 can apply formal FDR testing before drawing findings.
    """

    rho = prepared.corr(method="spearman", min_periods=20)
    G = nx.Graph()
    for code in rho.index:
        domain = str(codebook.loc[code, "domain"])
        G.add_node(
            code,
            domain=domain,
            domain_name=DOMAINS[domain],
            short=str(codebook.loc[code, "short"]),
            mean=float(item_stats.loc[code, "mean"]),
            consensus=float(item_stats.loc[code, "consensus"]),
        )

    rows = []
    codes = list(rho.index)
    for i, a in enumerate(codes):
        for b in codes[i + 1 :]:
            value = float(rho.loc[a, b])
            if np.isnan(value):
                continue
            rows.append({"source": a, "target": b, "rho": value, "abs_rho": abs(value)})
            if abs(value) >= threshold:
                G.add_edge(a, b, rho=value, weight=abs(value), sign=1 if value > 0 else -1)

    _add_degree_attributes(G)
    return G, pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def item_centre(num: pd.DataFrame) -> pd.DataFrame:
    return (num - num.mean()).fillna(0.0)


def profile_correlation(centred: pd.DataFrame) -> np.ndarray:
    M = np.asarray(centred, dtype=float)
    M = M - M.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = M / norms
    return U @ U.T


def knn_graph(S: np.ndarray, k: int, labels: pd.Index) -> nx.Graph:
    """Symmetric k-nearest-neighbour graph retaining positive weights."""

    labels = [int(x) for x in labels]
    S = np.array(S, dtype=float, copy=True)
    np.fill_diagonal(S, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(labels)
    hubness = np.zeros(len(labels), dtype=int)
    for i in range(len(labels)):
        for j in np.argsort(-S[i], kind="stable")[:k]:
            if S[i, j] > 0:
                hubness[j] += 1
                G.add_edge(labels[i], labels[j], weight=float(S[i, j]), distance=float(1 - S[i, j]))
    nx.set_node_attributes(G, {labels[i]: int(hubness[i]) for i in range(len(labels))}, "hubness")
    return G


def layout_graph(G: nx.Graph, seed: int = 42) -> dict:
    """Create a stable spring layout weighted by opinion similarity."""

    return nx.spring_layout(G, seed=seed, weight="weight", iterations=400, k=0.25)


def attach_layout(G: nx.Graph, pos: dict) -> None:
    for node, (x, y) in pos.items():
        G.nodes[node]["x"] = float(x)
        G.nodes[node]["y"] = float(y)


def node_table(G: nx.Graph) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    df.index.name = "node"
    return df.sort_index()


def edge_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, data in G.edges(data=True):
        rows.append({"source": source, "target": target, **data})
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def graph_summary(G: nx.Graph, *, construction: str, k: int | None = None) -> dict:
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    degrees = [degree for _, degree in G.degree()]
    return {
        "construction": construction,
        "k": k,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "min_degree": int(min(degrees)) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "min_weight": float(min(weights)) if weights else None,
        "max_weight": float(max(weights)) if weights else None,
        "mean_weight": float(np.mean(weights)) if weights else None,
    }


def ensure_output_dirs(paths: Role2Paths) -> None:
    for directory in (paths.tables, paths.graphs, paths.figures):
        directory.mkdir(parents=True, exist_ok=True)


def _add_degree_attributes(G: nx.Graph) -> None:
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    nx.set_node_attributes(G, dict(G.degree(weight="weight")), "strength")


def _clean_attrs(attrs: dict) -> dict:
    out = {}
    for key, value in attrs.items():
        if pd.isna(value):
            continue  # omit missing attributes rather than writing "" into a numeric field
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        else:
            out[key] = value
    return out
