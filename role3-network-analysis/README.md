# Role 3 - Network Analysis and Findings

This folder consumes Role 2's constructed graphs and Role 1's prepared tables. It does
not redefine nodes, edges, or the k-NN backbone — it analyses the network exactly as
handed off, and answers the questions Role 2's README explicitly deferred: is the
community structure real, which respondents are central, do opinion camps persist
across topics, and what does the network reveal about class opinions.

## Inputs

- `../role2-network-construction/outputs/graphs/respondent_opinion_network.graphml` -
  the main 87-node graph, loaded directly with `nx.read_graphml` (not re-derived
  from CSV) for every centrality/community calculation.
- `../role2-network-construction/outputs/graphs/topic_layer_{T,E,S,V}.gexf` - the
  four domain layers, loaded directly with `nx.read_gexf` for cross-topic community
  comparison.
- `../role2-network-construction/outputs/tables/item_edges_visual.csv` - Role 2's
  exploratory `|rho| >= 0.35` statement network, for the FDR-correction comparison
  (this one has no graph-file equivalent, so the CSV is the only copy).
- `../role1-data-preparation/outputs/tables/network_sample.csv`,
  `similarity_matrix.csv`, `prepared_responses.csv` - **not** read as a graph, only
  as the raw numbers needed to rebuild the pipeline under conditions Role 1/2 never
  ran: shuffled items for the null model, k=4..15 for the sensitivity scan, and full
  (unsparsified) per-domain similarity matrices for the Mantel test. None of these
  three exist anywhere in Role 1/2's outputs.

## What This Folder Adds

1. **Centrality** - degree, weighted degree (strength), betweenness and closeness
   (weighted by `distance`), eigenvector centrality, and local clustering, for every
   respondent (`respondent_centrality.csv`).
2. **Community detection** - Louvain communities and modularity on the main network,
   plus Guimera-Amaral node roles (within-module z-score, participation coefficient)
   to distinguish local hubs from cross-community connectors.
3. **Null-model validation** - Role 1's own recommendation: shuffle every item's
   column independently, rebuild the *entire* pipeline (centre -> correlate -> k-NN)
   200 times, and compare observed modularity/clustering against that null
   distribution, since k-NN graphs are clustered and modular by construction.
4. **k-sensitivity scan** - the same construction rebuilt for k = 4..15, to show how
   density, modularity, and fragmentation trade off around the chosen k = 9.
5. **Topic-layer comparison** - Mantel tests between each topic layer's full
   similarity matrix and the main network's (and each other's), plus normalised
   mutual information between their community partitions, on the 85 respondents
   common to all four layers.
6. **FDR-corrected statement network** - Role 1 flagged that 1,770 tested item pairs
   need multiple-testing correction; this folder computes Spearman p-values for every
   pair and applies Benjamini-Hochberg correction, then checks which of Role 2's
   visual-threshold edges actually survive.
7. **Intensity vs. position** - correlates each respondent's agreement intensity and
   atypicality (both handed off from Role 1/2 as node attributes) against every
   centrality metric, to test Role 1's open question of whether enthusiasm clusters
   in the network.

See `ROLE3.md` for the full write-up of methods, all key metrics, and findings.

## Outputs

```
outputs/tables/respondent_centrality.csv
outputs/tables/node_roles.csv
outputs/tables/community_summary.csv
outputs/tables/null_model_modularity.csv
outputs/tables/k_sensitivity.csv
outputs/tables/topic_layer_mantel.csv
outputs/tables/topic_layer_nmi.csv
outputs/tables/statement_fdr_edges.csv
outputs/tables/intensity_position_correlations.csv
outputs/figures/fig1_communities.png
outputs/figures/fig2_null_model.png
outputs/figures/fig3_k_sensitivity.png
outputs/figures/fig4_node_roles.png
outputs/figures/fig5_topic_layer_similarity.png
outputs/figures/fig6_statement_fdr.png
outputs/figures/fig7_intensity_vs_centrality.png
outputs/role3_analysis_summary.json
```

## Reproduce

```bash
pip install -r requirements.txt
python3 run_role3.py
python3 -m pytest -q tests
```

All randomness (null-model shuffles, Louvain, spring layout, Mantel permutations)
uses `SEED = 42`, matching Role 1 and Role 2.
