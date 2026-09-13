"""Role 1 - data preparation and network design.

    python run_preparation.py

Produces everything Role 1 owns:
  outputs/tables/codebook.csv                 statement code -> domain, text, short label
  outputs/tables/item_statistics.csv          per-statement means, spread, consensus
  outputs/tables/data_audit.csv               Table 1 of the report
  outputs/tables/prepared_responses.csv       encoded -2..+2 matrix, all 91 usable rows
  outputs/tables/network_sample.csv           the 87-respondent sample for the main network
  outputs/tables/similarity_matrix.csv        87 x 87 respondent similarity (the edge weights)
  outputs/tables/edge_list_knn.csv            reference k-NN edge list for Role 2
  outputs/figures/fig1_responses.png          response distribution by domain
  outputs/figures/fig2_similarity.png         why the data has to be centred
  outputs/preparation_metrics.json            every number quoted in the Role 1 sections

Roles 2 and 3 consume network_sample.csv / similarity_matrix.csv and the
node-and-edge definition documented in HANDOFF.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from opinion_network import config as cfg                        # noqa: E402
from opinion_network import plots                                # noqa: E402
from opinion_network import preprocess as pp                     # noqa: E402
from opinion_network.similarity import (agreement_similarity, default_k,  # noqa: E402
                                        knn_graph, profile_correlation)


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
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def permute_items(num, rng):
    """Shuffle each statement's column independently.

    Keeps every statement's answer distribution but destroys the coherence of
    any one person's answers. Used here as the reference distribution for the
    parallel analysis and for the similarity spread; Role 3 reuses the same
    idea as the null model for the network metrics.
    """
    out = num.copy()
    for c in out.columns:
        out[c] = rng.permutation(out[c].to_numpy())
    return out


def parallel_analysis(num, n_rep, seed):
    """Horn (1965): observed eigenvalues vs those of column-permuted data."""
    rng = np.random.default_rng(seed)
    C = pp.item_centre(num).to_numpy()
    ev = np.linalg.eigvalsh(np.corrcoef(C.T))[::-1]
    null = np.array([np.linalg.eigvalsh(np.corrcoef(pp.item_centre(permute_items(num, rng)).T))[::-1]
                     for _ in range(n_rep)])
    null95 = np.percentile(null, 95, axis=0)
    w, v = np.linalg.eigh(np.corrcoef(C.T))
    pc1 = v[:, -1] * np.sign(v[:, -1].sum())
    above = ev > null95          # eigenvalues are descending, so the Trues form a prefix
    return {
        "eigenvalues": ev,
        "null_mean": null.mean(axis=0),
        "null_95": null95,
        "n_significant": int(np.argmin(above)) if not above.all() else int(len(ev)),
        "pc1_var_explained": float(ev[0] / ev.sum()),
        "pc1_same_sign_fraction": float(max((pc1 > 0).mean(), (pc1 < 0).mean())),
        "pc1_corr_intensity": float(np.corrcoef(C @ pc1, num.mean(axis=1))[0, 1]),
        "pc1_loadings": dict(zip(num.columns, pc1)),
    }


def audit_table(q, tech_only, no_env, minor, minor_missing, pct):
    return pd.DataFrame([
        {"check": "Fully blank responses",
         "finding": f"{len(q['blank_respondents'])} (IDs {', '.join(map(str, q['blank_respondents']))})",
         "decision": "Dropped -> 91 usable responses"},
        {"check": "Answered Technology only",
         "finding": f"{len(tech_only)} (IDs {', '.join(map(str, tech_only))})",
         "decision": "Excluded from the respondent network; usable for T-T statement pairs"},
        {"check": "Environment block missing",
         "finding": f"{len(no_env)} (IDs {', '.join(map(str, no_env))})",
         "decision": "Kept in the main network (>= 45/60); excluded from topic layers"},
        {"check": "Scattered item non-response",
         "finding": (f"{len(minor)} respondents, {min(minor_missing)}-{max(minor_missing)} items each"
                     if minor else "0 respondents"),
         "decision": "Kept; missing treated as zero deviation from the item mean"},
        {"check": "Straight-lining (zero variance)",
         "finding": f"{q['straightliners']} respondents (min. within-person SD = {q['min_respondent_sd']:.2f})",
         "decision": "No exclusions needed"},
        {"check": "Duplicate answer profiles",
         "finding": str(q["duplicate_nonblank_profiles"]), "decision": "-"},
        {"check": "Answer distribution (substantive)",
         "finding": ", ".join(f"{k} {pct[k]:.1f}%" for k in
                              ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]),
         "decision": "Strong ceiling effect -> item-centring required"},
    ])


def main():
    for d in (cfg.FIG_DIR, cfg.TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    M = {}

    # ---------------------------------------------------- 1. load & encode
    raw, codebook = pp.load_raw()
    num = pp.encode(raw)
    codebook.to_csv(cfg.TAB_DIR / "codebook.csv")
    q = pp.quality_report(raw, num)
    M["quality"] = q

    nonblank = num.loc[num.notna().any(axis=1)]
    nonblank.to_csv(cfg.TAB_DIR / "prepared_responses.csv")

    partial = {int(k): v for k, v in q["partial_respondents"].items()}
    tech_only = sorted(k for k, v in partial.items() if v["E"] == 0 and v["S"] == 0 and v["V"] == 0)
    no_env = sorted(k for k, v in partial.items()
                    if k not in tech_only and v["V"] <= cfg.MAX_ANSWERED_SKIPPED_BLOCK)
    minor = sorted(k for k in partial if k not in tech_only and k not in no_env)
    minor_missing = [60 - sum(partial[k].values()) for k in minor]

    vc = q["value_counts"]
    substantive = sum(vc[k] for k in cfg.LIKERT)
    pct = {k: 100 * vc[k] / substantive for k in cfg.LIKERT}
    M["answer_distribution_pct"] = pct
    M["agree_pct"] = pct["Agree"] + pct["Strongly Agree"]
    M["disagree_pct"] = pct["Disagree"] + pct["Strongly Disagree"]

    audit = audit_table(q, tech_only, no_env, minor, minor_missing, pct)
    audit.to_csv(cfg.TAB_DIR / "data_audit.csv", index=False)
    print("[1] encoded:", raw.shape, "->", len(nonblank), "usable respondents")

    # ------------------------------------------------- 2. item statistics
    item_stats = pp.item_statistics(nonblank, codebook)
    item_stats.to_csv(cfg.TAB_DIR / "item_statistics.csv")
    by_cons = item_stats.sort_values("consensus")
    M["items"] = {
        "domain_mean_agreement": item_stats.groupby("domain")["pct_agree"].mean().to_dict(),
        "least_consensual": by_cons.head(5)[["short", "mean", "consensus"]].to_dict("index"),
        "most_consensual": by_cons.tail(5)[["short", "mean", "consensus"]].to_dict("index"),
        "negative_mean_items": item_stats[item_stats["mean"] < 0][["short", "mean"]].to_dict("index"),
    }
    print("[2] consensus range:", f"{by_cons['consensus'].min():.2f}-{by_cons['consensus'].max():.2f}")

    # ------------------------------------- 3. sample for the main network
    sample = pp.select(num, cfg.MIN_ANSWERED_TOTAL)
    sample.to_csv(cfg.TAB_DIR / "network_sample.csv")
    layer_ok = pd.Series(True, index=num.index)
    for d in cfg.DOMAINS:
        layer_ok &= num[[c for c in num.columns if c[0] == d]].notna().sum(axis=1) >= cfg.MIN_ANSWERED_LAYER
    M["samples"] = {
        "n_usable": int(len(nonblank)),
        "n_network_sample": int(len(sample)),
        "excluded_from_network": [int(i) for i in nonblank.index.difference(sample.index)],
        "missing_cells_in_sample": int(sample.isna().sum().sum()),
        "missing_pct_in_sample": float(100 * sample.isna().sum().sum() / sample.size),
        "n_topic_layer_sample": int(layer_ok.sum()),
        "rule_network": f">= {cfg.MIN_ANSWERED_TOTAL} of 60 items",
        "rule_layers": f">= {cfg.MIN_ANSWERED_LAYER} of 15 items in every domain",
    }
    print(f"[3] main-network sample: {len(sample)} respondents; topic layers: {int(layer_ok.sum())}")

    # ----------------------------- 4. the case for centring (design step)
    pa = parallel_analysis(sample, cfg.N_PARALLEL, cfg.SEED)
    M["parallel_analysis"] = {k: v for k, v in pa.items() if k != "pc1_loadings"}
    M["parallel_analysis"]["pc1_top_loadings"] = dict(
        sorted(pa["pc1_loadings"].items(), key=lambda kv: -abs(kv[1]))[:8])

    S_agree = agreement_similarity(sample)
    S_corr = profile_correlation(pp.item_centre(sample))
    S_null = profile_correlation(pp.item_centre(permute_items(sample, np.random.default_rng(cfg.SEED))))
    iu = np.triu_indices(len(sample), 1)
    M["similarity"] = {
        "naive_mean": float(S_agree[iu].mean()), "naive_min": float(S_agree[iu].min()),
        "naive_max": float(S_agree[iu].max()),
        "centred_mean": float(S_corr[iu].mean()), "centred_sd": float(S_corr[iu].std()),
        "centred_min": float(S_corr[iu].min()), "centred_max": float(S_corr[iu].max()),
        "null_centred_sd": float(S_null[iu].std()),
        "pairs": int(len(iu[0])),
        "frac_positive": float((S_corr[iu] > 0).mean()),
    }
    pd.DataFrame(S_corr, index=sample.index, columns=sample.index).to_csv(
        cfg.TAB_DIR / "similarity_matrix.csv")
    print(f"[4] PC1 = {100 * pa['pc1_var_explained']:.0f}% of variance, "
          f"r = {pa['pc1_corr_intensity']:.2f} with mean answer -> centring required")

    # ------------------------------ 5. reference backbone for Role 2
    k = cfg.K_NEIGHBOURS or default_k(len(sample))
    G = knn_graph(S_corr, k, labels=sample.index)
    edges = pd.DataFrame([{"source": u, "target": v, "weight": d["weight"],
                           "distance": d["distance"]} for u, v, d in G.edges(data=True)])
    edges.sort_values("weight", ascending=False).to_csv(cfg.TAB_DIR / "edge_list_knn.csv", index=False)
    degrees = np.array([d for _, d in G.degree()])
    M["reference_backbone"] = {
        "k": k, "rule": "k = round(sqrt(n)); symmetric (union) kNN, positive weights only",
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "mean_degree": float(degrees.mean()), "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "note": "Reference construction for Role 2; k and any alternative sparsification are Role 2's call.",
    }
    print(f"[5] reference backbone: k={k}, {G.number_of_edges()} edges, "
          f"degree {degrees.min()}-{degrees.max()}")

    # ------------------------------------------------------- 6. figures
    plots.fig_responses(item_stats)
    plots.fig_similarity(S_agree, S_corr, S_null, pa)

    with open(cfg.PREP_JSON, "w") as f:
        json.dump(to_builtin(M), f, indent=2)
    print("[6] figures and preparation_metrics.json written")


if __name__ == "__main__":
    main()
