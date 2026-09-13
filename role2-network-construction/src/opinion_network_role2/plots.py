"""Graphical visualisations for Role 2."""

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
from matplotlib.patches import Patch

from .construction import DOMAIN_COLORS, DOMAINS


plt.rcParams.update(
    {
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
    }
)


def fig_respondent_network(G: nx.Graph, pos: dict, out_path) -> None:
    """Main respondent graph: structure and agreement intensity."""

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    nodes = list(G.nodes())
    strengths = np.array([G.nodes[n]["strength"] for n in nodes])
    atypicality = np.array([G.nodes[n]["distance_to_class_mean"] for n in nodes])
    sizes = _scale(atypicality, 28, 145)
    widths = _scale(weights, 0.25, 1.7)

    for ax in axes:
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#8F8F8F", alpha=0.34)
        ax.axis("off")

    intensity = np.array([G.nodes[n]["intensity"] for n in nodes])
    sc = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=intensity,
        cmap="viridis",
        node_size=sizes,
        edgecolors="white",
        linewidths=0.45,
        ax=axes[0],
    )
    axes[0].set_title("Main respondent network: coloured by agreement intensity")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("mean response (-2 to +2)")

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=strengths,
        cmap="magma",
        node_size=sizes,
        edgecolors="white",
        linewidths=0.45,
        ax=axes[1],
    )
    axes[1].set_title("Same layout: coloured by weighted degree")
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#666666", markersize=4, label="less atypical"),
        Line2D([], [], marker="o", linestyle="", color="#666666", markersize=9, label="more atypical"),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_backbone_diagnostics(G: nx.Graph, out_path) -> None:
    """Show the construction properties role2 should document."""

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    degrees = np.array([degree for _, degree in G.degree()])
    strengths = np.array([G.nodes[n]["strength"] for n in G.nodes()])

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.55))
    axes[0].hist(weights, bins=24, color="#4C78A8", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Edge weights")
    axes[0].set_xlabel("centred profile correlation")
    axes[0].set_ylabel("edges")
    axes[0].axvline(weights.mean(), color="black", lw=0.8, ls="--")

    axes[1].hist(degrees, bins=np.arange(degrees.min(), degrees.max() + 2) - 0.5, color="#F58518")
    axes[1].set_title("Degree")
    axes[1].set_xlabel("number of neighbours")
    axes[1].set_ylabel("respondents")

    axes[2].scatter(degrees, strengths, s=24, color="#54A24B", alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[2].set_title("Degree vs strength")
    axes[2].set_xlabel("degree")
    axes[2].set_ylabel("weighted degree")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_heatmaps(similarity: pd.DataFrame, G: nx.Graph, item_rho: pd.DataFrame, out_path) -> None:
    """Matrix views of the constructed networks."""

    nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["intensity"])
    similarity = similarity.copy()
    similarity.index = similarity.index.astype(int)
    similarity.columns = similarity.columns.astype(int)
    S = similarity.loc[nodes, nodes].astype(float)
    A = nx.to_pandas_adjacency(G, nodelist=nodes, weight="weight")

    codes = sorted(item_rho.index, key=lambda c: (c[0], int(c[1:])))
    R = item_rho.loc[codes, codes]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.05))
    im0 = axes[0].imshow(S, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
    axes[0].set_title("All respondent similarities")
    axes[0].set_xlabel("respondents sorted by intensity")
    axes[0].set_ylabel("respondents sorted by intensity")
    _thin_matrix_ticks(axes[0], len(nodes))

    im1 = axes[1].imshow(A, cmap="viridis", vmin=0, vmax=max(0.65, A.to_numpy().max()), aspect="auto")
    axes[1].set_title("Retained k-NN backbone")
    axes[1].set_xlabel("same respondent order")
    axes[1].set_ylabel("same respondent order")
    _thin_matrix_ticks(axes[1], len(nodes))

    im2 = axes[2].imshow(R, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    axes[2].set_title("Statement Spearman rho")
    axes[2].set_xlabel("items grouped by domain")
    axes[2].set_ylabel("items grouped by domain")
    axes[2].set_xticks([7, 22, 37, 52])
    axes[2].set_xticklabels(["E", "S", "T", "V"])
    axes[2].set_yticks([7, 22, 37, 52])
    axes[2].set_yticklabels(["E", "S", "T", "V"])

    for ax in axes:
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)

    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    cb0.set_label("Pearson r")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label("edge weight")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    cb2.set_label("Spearman rho")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_topic_layers(layers: dict[str, nx.Graph], out_path) -> None:
    """Small multiples for the four domain-specific respondent layers."""

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    for ax, (prefix, G) in zip(axes.ravel(), layers.items()):
        pos = nx.spring_layout(G, seed=42, weight="weight", iterations=300, k=0.28)
        weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        widths = _scale(weights, 0.2, 1.35) if len(weights) else 0.3
        strengths = np.array([G.nodes[n].get("strength", 0.0) for n in G.nodes()])
        sizes = _scale(strengths, 18, 105) if len(strengths) else 35
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#8A8A8A", alpha=0.3)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=sizes,
            node_color=DOMAIN_COLORS[prefix],
            edgecolors="white",
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title(f"{DOMAINS[prefix]} layer ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_item_network(G: nx.Graph, out_path) -> None:
    """Exploratory statement graph grouped visually by survey domain."""

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    pos = nx.spring_layout(G, seed=42, weight="weight", iterations=600, k=0.7)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    width_lookup = {
        tuple(sorted((u, v))): width
        for (u, v, d), width in zip(G.edges(data=True), _scale(weights, 0.35, 2.2))
    }
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=pos_edges,
        width=[width_lookup[tuple(sorted(e))] for e in pos_edges],
        edge_color="#777777",
        alpha=0.35,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=neg_edges,
        width=[width_lookup[tuple(sorted(e))] for e in neg_edges],
        edge_color="#C44E52",
        style="dashed",
        alpha=0.45,
        ax=ax,
    )
    colors = [DOMAIN_COLORS[G.nodes[n]["domain"]] for n in G.nodes()]
    sizes = _scale(np.array([G.nodes[n]["strength"] for n in G.nodes()]), 45, 170)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors="white", linewidths=0.5, ax=ax)
    labelled = sorted(G.nodes(), key=lambda n: (G.degree(n), G.nodes[n]["strength"]), reverse=True)[:10]
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in labelled},
        font_size=5,
        font_color="#222222",
        ax=ax,
    )
    ax.set_title("Exploratory statement network: Spearman associations")
    ax.axis("off")
    handles = [Patch(facecolor=color, label=name) for color, name in zip(DOMAIN_COLORS.values(), DOMAINS.values())]
    handles.append(Line2D([], [], color="#777777", lw=1.5, label="positive rho"))
    handles.append(Line2D([], [], color="#C44E52", lw=1.5, ls="--", label="negative rho"))
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _scale(values, low: float, high: float):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    span = values.max() - values.min()
    if span == 0:
        return np.full(values.shape, (low + high) / 2)
    return low + (values - values.min()) * (high - low) / span


def _thin_matrix_ticks(ax, n: int) -> None:
    ticks = [0, n // 2, n - 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks])
    ax.set_yticklabels([str(t + 1) for t in ticks])
