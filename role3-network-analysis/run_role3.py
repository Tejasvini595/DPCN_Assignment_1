"""Role 3 - detailed network analysis and findings.

Run from this directory:

    python run_role3.py

Consumes Role 1 + Role 2's prepared tables and graphs (never redefines nodes
or edges) and writes centrality, community, validation and comparison
outputs under outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from opinion_network_role3 import analysis as an
from opinion_network_role3 import plots


ROOT = Path(__file__).resolve().parent


def to_builtin(o):
    if isinstance(o, dict):
        return {str(k): to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_builtin(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, bool):
        return o
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def main() -> None:
    paths = an.Role3Paths(ROOT)
    an.ensure_output_dirs(paths)
    data = an.read_inputs(paths)
    M = {}

    # ------------------------------------------- 1. load Role 2's main graph
    G = an.load_role2_graph(paths)
    pos = nx.spring_layout(G, seed=an.SEED, weight="weight", iterations=400, k=0.25)
    print(f"[1] loaded main respondent graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---------------------------------------------- 2. centrality metrics
    centrality = an.compute_centrality(G)
    centrality.to_csv(paths.tables / "respondent_centrality.csv")
    print("[2] centrality metrics computed for", len(centrality), "respondents")

    # -------------------------------------------------- 3. community detection
    communities, membership, modularity = an.detect_communities(G)
    roles = an.guimera_amaral_roles(G, communities)
    roles.to_csv(paths.tables / "node_roles.csv")
    comm_summary = an.community_summary(centrality, membership)
    comm_summary.to_csv(paths.tables / "community_summary.csv")
    M["communities"] = {"n_communities": len(communities), "modularity": modularity,
                        "sizes": [len(c) for c in communities]}
    print(f"[3] {len(communities)} communities, modularity={modularity:.3f}")

    # -------------------------------------------------------- 4. global metrics
    gm = an.global_metrics(G, modularity, len(communities))
    M["global_metrics"] = gm
    print(f"[4] density={gm['density']:.3f} diameter={gm['diameter_weighted']:.2f} "
          f"transitivity={gm['transitivity']:.3f} assort(intensity)={gm['intensity_assortativity']:.3f}")

    # ------------------------------------------------------ 5. null model
    null_df = an.null_model_modularity(data["network_sample"], k=an.default_k(len(data["network_sample"])),
                                        n_rep=200, seed=an.SEED)
    null_df.to_csv(paths.tables / "null_model_modularity.csv", index=False)
    p_value = float((null_df["modularity"] >= modularity).mean())
    M["null_model"] = {
        "n_rep": 200, "observed_modularity": modularity,
        "null_mean_modularity": float(null_df["modularity"].mean()),
        "null_sd_modularity": float(null_df["modularity"].std()),
        "empirical_p_value": p_value,
        "null_mean_clustering": float(null_df["avg_clustering"].mean()),
        "observed_clustering": gm["avg_clustering_weighted"],
    }
    print(f"[5] null model: mean modularity {null_df['modularity'].mean():.3f}, "
          f"observed {modularity:.3f}, p={p_value:.4f}")

    # -------------------------------------------------- 6. k-sensitivity
    k_df = an.k_sensitivity(data["similarity"], range(4, 16))
    k_df.to_csv(paths.tables / "k_sensitivity.csv", index=False)
    M["k_sensitivity"] = {"k_range": [4, 15], "chosen_k": an.default_k(len(data["network_sample"]))}
    print("[6] k-sensitivity scan done for k=4..15")

    # ---------------------------------------------- 7. topic-layer comparison
    mantel_df, nmi_df, layer_partitions, common_respondents = an.compare_layers(
        paths, data["network_sample"], data["similarity"], membership)
    mantel_df.to_csv(paths.tables / "topic_layer_mantel.csv", index=False)
    nmi_df.to_csv(paths.tables / "topic_layer_nmi.csv", index=False)
    M["topic_layers"] = {
        "mean_mantel_r_main_vs_topic": float(
            mantel_df[mantel_df["layer_a"] == "Main"]["mantel_r"].mean()),
        "mean_nmi_main_vs_topic": float(nmi_df[nmi_df["layer_a"] == "Main"]["nmi"].mean()),
    }
    print("[7] topic-layer Mantel/NMI comparison done")

    # ------------------------------------------- 8. statement network + FDR
    spearman_pvals = an.spearman_with_pvalues(data["prepared_responses"])
    fdr_table = an.fdr_statement_network(spearman_pvals)
    fdr_table.to_csv(paths.tables / "statement_fdr_edges.csv", index=False)
    threshold_edges = len(data["item_edges"])
    fdr_sig_edges = fdr_table[fdr_table["fdr_significant"]]
    threshold_pairs = set(zip(data["item_edges"]["source"], data["item_edges"]["target"]))
    fdr_pairs = set(zip(fdr_sig_edges["source"], fdr_sig_edges["target"])) | \
                set(zip(fdr_sig_edges["target"], fdr_sig_edges["source"]))
    overlap = sum(1 for a, b in threshold_pairs if (a, b) in fdr_pairs or (b, a) in fdr_pairs)
    M["statement_network"] = {
        "pairs_tested": len(spearman_pvals),
        "naive_p_below_05": int((spearman_pvals["p_value"] < 0.05).sum()),
        "fdr_significant": int(fdr_sig_edges.shape[0]),
        "role2_threshold_edges": threshold_edges,
        "overlap_threshold_and_fdr": overlap,
    }
    print(f"[8] statement network: {len(spearman_pvals)} pairs tested, "
          f"{int((spearman_pvals['p_value'] < 0.05).sum())} naive p<.05, "
          f"{fdr_sig_edges.shape[0]} FDR-significant, "
          f"role2 threshold network had {threshold_edges}")

    # ----------------------------------- 9. intensity/atypicality vs position
    ipc = an.intensity_position_correlation(centrality)
    ipc.to_csv(paths.tables / "intensity_position_correlations.csv", index=False)
    print("[9] intensity/atypicality vs centrality correlations computed")

    # ------------------------------- 10. Louvain on the statement network
    item_G, item_membership, item_modularity, confusion, item_nmi = an.statement_network_communities(
        fdr_table, data["codebook"])
    confusion.to_csv(paths.tables / "statement_domain_confusion.csv")
    pd.Series(item_membership, name="algorithmic_community").rename_axis("code").to_csv(
        paths.tables / "statement_community_membership.csv")
    M["statement_communities"] = {
        "n_communities": len(set(item_membership.values())),
        "modularity": item_modularity,
        "nmi_vs_survey_domains": item_nmi,
    }
    print(f"[10] statement network: {len(set(item_membership.values()))} algorithmic communities, "
          f"modularity={item_modularity:.3f}, NMI vs. survey domains={item_nmi:.3f}")

    # --------------------------------- 11. cross-domain community flow (Sankey)
    flow_t = an.sankey_flow(layer_partitions["Main"], layer_partitions["T"], common_respondents)
    flow_e = an.sankey_flow(layer_partitions["Main"], layer_partitions["E"], common_respondents)
    flow_t.to_csv(paths.tables / "community_flow_main_to_T.csv", index=False)
    flow_e.to_csv(paths.tables / "community_flow_main_to_E.csv", index=False)
    print("[11] cross-domain community flow tables written")

    # ------------------------------------------------------------ figures
    plots.fig_communities(G, pos, membership, modularity, paths.figures / "fig1_communities.png")
    plots.fig_null_model(null_df, modularity, paths.figures / "fig2_null_model.png")
    plots.fig_k_sensitivity(k_df, an.default_k(len(data["network_sample"])),
                             paths.figures / "fig3_k_sensitivity.png")
    plots.fig_node_roles(roles, paths.figures / "fig4_node_roles.png")
    plots.fig_layer_similarity(mantel_df, nmi_df, paths.figures / "fig5_topic_layer_similarity.png")
    plots.fig_statement_fdr(fdr_table, threshold_edges, paths.figures / "fig6_statement_fdr.png")
    plots.fig_intensity_vs_centrality(centrality, paths.figures / "fig7_intensity_vs_centrality.png")
    plots.fig_statement_communities(item_G, item_membership, confusion, item_modularity,
                                     item_nmi, paths.figures / "fig8_statement_communities.png")
    plots.fig_alluvial(flow_t, flow_e, paths.figures / "fig9_community_flow.png")
    print("[12] figures written")

    (paths.outputs / "role3_analysis_summary.json").write_text(
        json.dumps(to_builtin(M), indent=2), encoding="utf-8")
    print(f"[done] wrote outputs to {paths.outputs}")


if __name__ == "__main__":
    main()
