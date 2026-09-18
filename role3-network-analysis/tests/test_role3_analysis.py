from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opinion_network_role3 import analysis as an


def test_graph_loads_directly_from_role2_export():
    paths = an.Role3Paths(ROOT)
    G = an.load_role2_graph(paths)
    assert G.number_of_nodes() == 87
    assert G.number_of_edges() == 459
    assert nx.is_connected(G)
    assert isinstance(next(iter(G.nodes())), int)  # node_type=int round-trip


def test_topic_layer_loads_directly_from_role2_export():
    paths = an.Role3Paths(ROOT)
    G = an.load_role2_topic_layer(paths, "T")
    assert G.number_of_nodes() == 85


def test_communities_and_roles_are_consistent():
    paths = an.Role3Paths(ROOT)
    G = an.load_role2_graph(paths)
    communities, membership, modularity = an.detect_communities(G)
    assert set().union(*communities) == set(G.nodes())
    assert -1 <= modularity <= 1

    roles = an.guimera_amaral_roles(G, communities)
    assert (roles["participation_coefficient"] >= 0).all()
    assert (roles["participation_coefficient"] <= 1.0001).all()


def test_benjamini_hochberg_matches_known_example():
    # p-values with a hand-verified BH(0.05) cutoff: p_(i) <= (i/m)*alpha holds
    # only for i=1,2 (0.001<=0.005, 0.008<=0.01); it fails from i=3 onward
    # (0.039 > 0.015), so exactly the two smallest p-values are significant.
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    significant, q = an.benjamini_hochberg(pvals, alpha=0.05)
    assert significant[:2].all()
    assert not significant[2:].any()
    assert (q >= pvals).all()


def test_nmi_is_one_for_identical_partitions():
    nodes = list(range(20))
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=len(nodes))
    part_a = {n: int(l) for n, l in zip(nodes, labels)}
    part_b = dict(part_a)
    assert np.isclose(an.normalized_mutual_information(part_a, part_b, nodes), 1.0)

    part_c = {n: rng.integers(0, 5) for n in nodes}
    nmi = an.normalized_mutual_information(part_a, part_c, nodes)
    assert 0.0 <= nmi <= 1.0001


def test_mantel_test_recovers_perfect_correlation():
    idx = list(range(15))
    rng = np.random.default_rng(1)
    base = rng.normal(size=(15, 15))
    A = pd.DataFrame((base + base.T) / 2, index=idx, columns=idx)
    B = A.copy()
    r, p, n = an.mantel_test(A, B, n_perm=199)
    assert np.isclose(r, 1.0)
    assert n == 15


def test_knn_graph_matches_role1_role2_construction():
    rng = np.random.default_rng(2)
    S = np.corrcoef(rng.normal(size=(30, 12)))
    G = an.knn_graph(S, k=4, labels=range(30))
    assert min(d for _, d in G.degree()) >= 4
    assert all(d["weight"] > 0 for *_, d in G.edges(data=True))


def test_statement_network_communities_confusion_matrix_sums_to_60():
    paths = an.Role3Paths(ROOT)
    data = an.read_inputs(paths)
    spearman_pvals = an.spearman_with_pvalues(data["prepared_responses"])
    fdr_table = an.fdr_statement_network(spearman_pvals)
    G, membership, modularity, confusion, nmi = an.statement_network_communities(
        fdr_table, data["codebook"])
    assert G.number_of_nodes() == 60
    assert confusion.to_numpy().sum() == 60
    assert 0.0 <= nmi <= 1.0001


def test_sankey_flow_counts_match_node_count():
    nodes = list(range(10))
    part_a = {n: n % 2 for n in nodes}
    part_b = {n: n % 3 for n in nodes}
    flow = an.sankey_flow(part_a, part_b, nodes)
    assert flow["count"].sum() == len(nodes)
    assert set(flow.columns) == {"source", "target", "count"}


def test_compare_layers_returns_partitions_and_common_nodes():
    paths = an.Role3Paths(ROOT)
    data = an.read_inputs(paths)
    G = an.load_role2_graph(paths)
    _, membership, _ = an.detect_communities(G)
    mantel_df, nmi_df, partitions, common = an.compare_layers(
        paths, data["network_sample"], data["similarity"], membership)
    assert len(common) == 85
    assert set(partitions.keys()) == {"Main", "T", "E", "S", "V"}
    assert all(n in partitions["Main"] for n in common)
