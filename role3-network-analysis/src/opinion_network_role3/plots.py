"""Role 3 figures: communities, validation, sensitivity, roles, layers, FDR."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COMMUNITY_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B"]


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def fig_communities(G: nx.Graph, pos: dict, membership: dict, modularity: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    nodes = list(G.nodes())
    colors = [COMMUNITY_PALETTE[membership[n] % len(COMMUNITY_PALETTE)] for n in nodes]
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    widths = 0.25 + 1.4 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#999999", alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=90,
                            edgecolors="white", linewidths=0.5, ax=ax)
    ax.set_title(f"Louvain communities on the main respondent network (modularity = {modularity:.3f})")
    ax.axis("off")
    n_com = len(set(membership.values()))
    handles = [Line2D([], [], marker="o", linestyle="", color=COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)],
                       markersize=7, label=f"community {i}") for i in range(n_com)]
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=min(n_com, 4))
    fig.tight_layout()
    _save(fig, out_path)


def fig_null_model(null_df: pd.DataFrame, observed_modularity: float, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].hist(null_df["modularity"], bins=24, color="#BDBDBD", label="null (items shuffled)")
    axes[0].axvline(observed_modularity, color="#C44E52", lw=1.6, ls="--", label="observed")
    axes[0].set_title("Modularity: observed vs. null")
    axes[0].set_xlabel("Louvain modularity")
    axes[0].set_ylabel("null replicates")
    axes[0].legend(frameon=False, fontsize=6.5)

    axes[1].hist(null_df["avg_clustering"], bins=24, color="#BDBDBD", label="null")
    axes[1].set_title("Average clustering under the null")
    axes[1].set_xlabel("average weighted clustering coefficient")
    axes[1].set_ylabel("null replicates")
    fig.tight_layout()
    _save(fig, out_path)


def fig_k_sensitivity(df: pd.DataFrame, chosen_k: int, out_path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    axes[0].plot(df["k"], df["density"], "o-", color="#4C78A8")
    axes[0].axvline(chosen_k, color="#C44E52", ls="--", lw=1)
    axes[0].set_title("Density vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("density")

    axes[1].plot(df["k"], df["modularity"], "o-", color="#54A24B")
    axes[1].axvline(chosen_k, color="#C44E52", ls="--", lw=1)
    axes[1].set_title("Modularity vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Louvain modularity")

    axes[2].plot(df["k"], df["components"], "o-", color="#F58518")
    axes[2].axvline(chosen_k, color="#C44E52", ls="--", lw=1)
    axes[2].set_title("Components vs k")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("connected components")
    fig.tight_layout()
    _save(fig, out_path)


def fig_node_roles(roles: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = [COMMUNITY_PALETTE[c % len(COMMUNITY_PALETTE)] for c in roles["community"]]
    ax.scatter(roles["participation_coefficient"], roles["within_module_z"],
               c=colors, s=45, edgecolors="white", linewidths=0.5)
    ax.axhline(2.5, color="black", lw=0.6, ls=":")
    ax.axvline(0.62, color="black", lw=0.6, ls=":")
    ax.set_xlabel("participation coefficient (ties spread across communities)")
    ax.set_ylabel("within-module strength z-score (local hubness)")
    ax.set_title("Node roles (Guimerà & Amaral, 2005)")
    ax.text(0.02, 2.6, "module hubs", fontsize=6.5, style="italic")
    ax.text(0.64, ax.get_ylim()[0] + 0.1, "connectors", fontsize=6.5, style="italic")
    fig.tight_layout()
    _save(fig, out_path)


def fig_layer_similarity(mantel_df: pd.DataFrame, nmi_df: pd.DataFrame, out_path) -> None:
    names = ["Main", "T", "E", "S", "V"]
    idx = {n: i for i, n in enumerate(names)}
    mat_r = np.eye(len(names))
    mat_nmi = np.eye(len(names))
    for _, row in mantel_df.iterrows():
        i, j = idx[row["layer_a"]], idx[row["layer_b"]]
        mat_r[i, j] = mat_r[j, i] = row["mantel_r"]
    for _, row in nmi_df.iterrows():
        i, j = idx[row["layer_a"]], idx[row["layer_b"]]
        mat_nmi[i, j] = mat_nmi[j, i] = row["nmi"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    im0 = axes[0].imshow(mat_r, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Mantel r between layer\nsimilarity matrices")
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{mat_r[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mat_nmi, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("NMI between layer\ncommunity partitions")
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names)
    axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[1].text(j, i, f"{mat_nmi[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if mat_nmi[i, j] < 0.6 else "black")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out_path)


def fig_statement_fdr(fdr_table, threshold_edges: int, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    tested = len(fdr_table)
    naive05 = int((fdr_table["p_value"] < 0.05).sum())
    fdr_sig = int(fdr_table["fdr_significant"].sum())
    axes[0].bar(["tested\npairs", "naive\np<.05", "FDR\nsignificant", "role2\n|rho|>=0.35"],
                [tested, naive05, fdr_sig, threshold_edges],
                color=["#BDBDBD", "#F58518", "#4C78A8", "#54A24B"])
    axes[0].set_title("Statement-pair edge survival")
    axes[0].set_ylabel("number of item pairs")
    for i, v in enumerate([tested, naive05, fdr_sig, threshold_edges]):
        axes[0].text(i, v + tested * 0.01, str(v), ha="center", fontsize=7)

    axes[1].scatter(fdr_table["rho"], -np.log10(fdr_table["p_value"]),
                     c=np.where(fdr_table["fdr_significant"], "#4C78A8", "#D9D9D9"), s=10, alpha=0.8)
    axes[1].set_xlabel("Spearman rho")
    axes[1].set_ylabel("-log10(p)")
    axes[1].set_title("Volcano: FDR-significant pairs in blue")
    fig.tight_layout()
    _save(fig, out_path)


def fig_intensity_vs_centrality(centrality: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    axes[0].scatter(centrality["intensity"], centrality["degree"], color="#4C78A8", s=30, alpha=0.8)
    axes[0].set_xlabel("intensity (mean answer)")
    axes[0].set_ylabel("degree")
    axes[0].set_title("Enthusiasm vs. connectivity")

    axes[1].scatter(centrality["distance_to_class_mean"], centrality["betweenness"],
                     color="#C44E52", s=30, alpha=0.8)
    axes[1].set_xlabel("distance from class-mean profile (atypicality)")
    axes[1].set_ylabel("betweenness centrality")
    axes[1].set_title("Atypicality vs. brokerage")
    fig.tight_layout()
    _save(fig, out_path)
