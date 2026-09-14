# Role 2 - Network Construction and Graphical Visualisation

## Purpose

Role 2 takes the prepared Role 1 outputs and turns them into actual network files,
visual tables and figures. Preprocessing decisions are not repeated here: the node
set, similarity definition and reference k-NN backbone come from Role 1.

## Main Respondent Network

The primary network has 87 respondent nodes. Edges are undirected and weighted by
the centred profile correlation supplied in Role 1's `edge_list_knn.csv`. The
construction uses the symmetric k-nearest-neighbour backbone with `k = 9`, because
`round(sqrt(87)) = 9`. Only positive similarities are retained.

Each edge stores:

- `weight`: centred Pearson profile correlation.
- `distance`: `1 - weight`, for later shortest-path calculations.

Each node stores:

- `intensity`: respondent's mean agreement score.
- `n_answered`: number of answered statements.
- `distance_to_class_mean`: how atypical the response profile is.
- domain means for Technology, Education, Ethics & Society and Environment.
- `degree`, `strength`, and layout coordinates `x`, `y`.

## Visualisation Choices

The main layout is a fixed seeded spring layout weighted by similarity, so stronger
edges pull nodes closer. In `fig1_respondent_network.png`, node colour is used in two
ways: first for agreement intensity and then for weighted degree on the same layout.
Node size represents distance from the class mean, and edge width represents
similarity strength.

`fig2_backbone_diagnostics.png` documents the construction visually: the distribution
of retained edge weights, the degree distribution, and the relationship between
degree and weighted degree.

`fig5_network_heatmaps.png` gives a matrix view of the same construction. It shows
the full respondent similarity matrix before sparsification, the retained weighted
k-NN adjacency matrix after sparsification, and the statement-to-statement Spearman
correlation matrix grouped by domain.

## Complementary Graphical Views

Role 2 also constructs two supporting views from the same Role 1 handoff:

- Topic-layer respondent networks for Technology, Education, Ethics & Society and
  Environment. These use the same k-NN construction within each 15-item domain and
  keep the 85 respondents with sufficient answers in every topic.

  Note: within a single 15-item domain (especially the near-consensual Environment
  layer), many respondents end up with exactly-tied similarity scores. Which of several
  tied candidates fills the last k-NN slot can vary slightly by machine/BLAS library, so
  a handful of topic-layer edges may differ between systems even when the code and
  inputs are identical. This is a floating-point tie-break artefact, not a construction
  error: aggregate properties (edge count, degree range, density) are unaffected and
  reproduce exactly across runs.
- An exploratory statement network where nodes are survey statements and edges are
  pairwise Spearman associations with `abs(rho) >= 0.35`. This is visual only; Role 3
  should apply formal FDR correction before treating statement edges as findings.

## Outputs for Role 3

Important files:

- `outputs/graphs/respondent_opinion_network.gexf`
- `outputs/graphs/respondent_opinion_network.graphml`
- `outputs/tables/respondent_nodes_visual.csv`
- `outputs/tables/respondent_edges_visual.csv`
- `outputs/figures/fig1_respondent_network.png`
- `outputs/figures/fig2_backbone_diagnostics.png`
- `outputs/figures/fig3_topic_layers.png`
- `outputs/figures/fig4_statement_network.png`
- `outputs/figures/fig5_network_heatmaps.png`
- `outputs/role2_network_summary.json`

Role 3 can start from the GraphML/GEXF files for analysis software, or from the CSV
node and edge tables for NetworkX/igraph.

## Reproduce

```bash
python3 run_role2.py
python3 -m pytest -q tests
```

Expected main-network result: 87 nodes, 459 edges, one connected component, minimum
degree 9 and maximum degree 16.
