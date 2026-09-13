"""Role 2 - network construction and graphical visualisation.

Run from this directory:

    python run_role2.py

The script consumes Role 1's prepared outputs and writes Role 2 tables,
graph exports and figures under outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from opinion_network_role2 import construction as net
from opinion_network_role2 import plots


ROOT = Path(__file__).resolve().parent


def main() -> None:
    paths = net.Role2Paths(ROOT)
    net.ensure_output_dirs(paths)
    tables = net.read_role1_tables(paths)

    sample = tables["network_sample"]
    k = net.default_k(len(sample))

    attrs = net.respondent_attributes(sample)
    G = net.build_respondent_graph(tables["edge_list"], attrs)
    pos = net.layout_graph(G)
    net.attach_layout(G, pos)

    nodes = net.node_table(G)
    edges = net.edge_table(G)
    nodes.to_csv(paths.tables / "respondent_nodes_visual.csv")
    edges.to_csv(paths.tables / "respondent_edges_visual.csv", index=False)
    nx.write_gexf(G, paths.graphs / "respondent_opinion_network.gexf")
    nx.write_graphml(G, paths.graphs / "respondent_opinion_network.graphml")

    layers = net.build_topic_layers(sample, k=k)
    layer_summaries = {}
    for prefix, layer in layers.items():
        nx.write_gexf(layer, paths.graphs / f"topic_layer_{prefix}.gexf")
        net.node_table(layer).to_csv(paths.tables / f"topic_layer_{prefix}_nodes.csv")
        net.edge_table(layer).to_csv(paths.tables / f"topic_layer_{prefix}_edges.csv", index=False)
        layer_summaries[prefix] = net.graph_summary(layer, construction=f"{prefix} topic kNN", k=k)

    item_graph, item_correlations = net.build_item_network(
        tables["prepared_responses"],
        tables["codebook"],
        tables["item_statistics"],
    )
    item_rho = tables["prepared_responses"].corr(method="spearman", min_periods=20)
    item_correlations.to_csv(paths.tables / "item_spearman_correlations.csv", index=False)
    net.node_table(item_graph).to_csv(paths.tables / "item_nodes_visual.csv")
    net.edge_table(item_graph).to_csv(paths.tables / "item_edges_visual.csv", index=False)
    nx.write_gexf(item_graph, paths.graphs / "exploratory_statement_network.gexf")

    plots.fig_respondent_network(G, pos, paths.figures / "fig1_respondent_network.png")
    plots.fig_backbone_diagnostics(G, paths.figures / "fig2_backbone_diagnostics.png")
    plots.fig_topic_layers(layers, paths.figures / "fig3_topic_layers.png")
    plots.fig_item_network(item_graph, paths.figures / "fig4_statement_network.png")
    plots.fig_heatmaps(tables["similarity"], G, item_rho, paths.figures / "fig5_network_heatmaps.png")

    summary = {
        "source": "Role 1 prepared tables",
        "main_network": net.graph_summary(
            G,
            construction="symmetric union k-nearest-neighbour respondent graph, positive centred correlations only",
            k=k,
        ),
        "topic_layers": layer_summaries,
        "statement_network": net.graph_summary(
            item_graph,
            construction="exploratory Spearman statement graph, abs(rho) >= 0.35",
        ),
        "outputs": {
            "tables": sorted(p.name for p in paths.tables.glob("*.csv")),
            "graphs": sorted(p.name for p in paths.graphs.glob("*")),
            "figures": sorted(p.name for p in paths.figures.glob("*.png")),
        },
    }
    (paths.outputs / "role2_network_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[role2] respondent graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, k={k}")
    print(f"[role2] topic layers: {', '.join(f'{k}:{v.number_of_edges()}' for k, v in layers.items())}")
    print(f"[role2] statement graph: {item_graph.number_of_nodes()} nodes, {item_graph.number_of_edges()} edges")
    print(f"[role2] wrote outputs to {paths.outputs}")


if __name__ == "__main__":
    main()
