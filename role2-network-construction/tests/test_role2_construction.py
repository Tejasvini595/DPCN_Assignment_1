from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opinion_network_role2 import construction as net


def test_main_graph_matches_role1_backbone():
    paths = net.Role2Paths(ROOT)
    tables = net.read_role1_tables(paths)
    attrs = net.respondent_attributes(tables["network_sample"])
    G = net.build_respondent_graph(tables["edge_list"], attrs)

    assert G.number_of_nodes() == 87
    assert G.number_of_edges() == len(tables["edge_list"]) == 459
    assert min(dict(G.degree()).values()) >= 9
    assert nx.is_connected(G)


def test_node_attributes_are_export_ready():
    paths = net.Role2Paths(ROOT)
    tables = net.read_role1_tables(paths)
    attrs = net.respondent_attributes(tables["network_sample"])
    G = net.build_respondent_graph(tables["edge_list"], attrs)
    pos = net.layout_graph(G)
    net.attach_layout(G, pos)
    nodes = net.node_table(G)

    for column in ["intensity", "n_answered", "distance_to_class_mean", "degree", "strength", "x", "y"]:
        assert column in nodes.columns
    assert nodes["intensity"].notna().all()
    assert nodes["x"].between(-2, 2).all()
    assert nodes["y"].between(-2, 2).all()


def test_topic_layers_keep_common_respondents():
    paths = net.Role2Paths(ROOT)
    sample = net.read_role1_tables(paths)["network_sample"]
    layers = net.build_topic_layers(sample, k=net.default_k(len(sample)))

    assert set(layers) == {"T", "E", "S", "V"}
    assert {G.number_of_nodes() for G in layers.values()} == {85}
    assert all(G.number_of_edges() > 0 for G in layers.values())

