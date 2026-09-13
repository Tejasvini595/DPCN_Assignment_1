"""Role 1 figures: the response profile of the class and the case for centring."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

from . import config as cfg                                      # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
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
LIKERT_COLORS = ["#B2182B", "#EF8A62", "#CFCFCF", "#67A9CF", "#2166AC"]


def _save(fig, name):
    fig.savefig(cfg.FIG_DIR / name)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig_responses(item_stats):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax, d in zip(axes.ravel(), cfg.DOMAINS):
        sub = item_stats[item_stats["domain"] == d].sort_values("mean")
        n = sub[["n_SD", "n_D", "n_N", "n_A", "n_SA"]].to_numpy(float)
        pct = 100 * n / n.sum(axis=1, keepdims=True)
        y = np.arange(len(sub))
        left = -(pct[:, 0] + pct[:, 1] + pct[:, 2] / 2)
        for j in range(5):
            ax.barh(y, pct[:, j], left=left, color=LIKERT_COLORS[j], height=0.78,
                    edgecolor="white", linewidth=0.3)
            left = left + pct[:, j]
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  {s}" for c, s in zip(sub.index, sub["short"])], fontsize=6.5)
        ax.set_xlim(-80, 100)
        ax.set_xticks([-75, -50, -25, 0, 25, 50, 75, 100])
        ax.set_xticklabels(["75", "50", "25", "0", "25", "50", "75", "100"])
        ax.set_title(f"{cfg.DOMAINS[d]} ({d})", color=cfg.DOMAIN_COLORS[d])
        ax.set_xlabel("% of respondents  (disagree | agree)")
        ax.tick_params(axis="y", length=0)
    handles = [Patch(color=c, label=l) for c, l in
               zip(LIKERT_COLORS, ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"])]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig1_responses.png")


# ---------------------------------------------------------------- Fig 2
def fig_similarity(S_agree, S_corr, S_null, pa):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    iu = np.triu_indices_from(S_agree, 1)
    ax = axes[0]
    ax.hist(S_agree[iu], bins=40, color="#9E9E9E")
    ax.set_title("(a) Naive agreement similarity")
    ax.set_xlabel("1 - mean |x_i - x_j| / 4")
    ax.set_ylabel("respondent pairs")
    ax.axvline(S_agree[iu].mean(), color="black", ls="--", lw=0.7)
    ax = axes[1]
    bins = np.linspace(-0.6, 0.7, 45)
    ax.hist(S_null[iu], bins=bins, color="#BDBDBD", label="null (items shuffled)", density=True)
    ax.hist(S_corr[iu], bins=bins, histtype="step", color="#2F5F8A", lw=1.3, label="observed", density=True)
    ax.set_title("(b) Centred profile correlation")
    ax.set_xlabel("Pearson r of item-centred profiles")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax = axes[2]
    m = 15
    xs = np.arange(1, m + 1)
    ax.plot(xs, pa["eigenvalues"][:m], "o-", ms=3, color="#2F5F8A", label="observed")
    ax.plot(xs, pa["null_95"][:m], "s--", ms=2.5, color="#9E9E9E", label="null 95th pct")
    ax.set_title("(c) Parallel analysis")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(frameon=False, loc="center right", fontsize=6)
    ax.text(2.8, pa["eigenvalues"][0] * 0.95,
            f"PC1: {100 * pa['pc1_var_explained']:.0f}% of variance\n"
            f"{100 * pa['pc1_same_sign_fraction']:.0f}% loadings same sign\n"
            f"r(PC1, mean score) = {pa['pc1_corr_intensity']:.2f}", fontsize=6.2, va="top")
    fig.tight_layout()
    _save(fig, "fig2_similarity.png")


