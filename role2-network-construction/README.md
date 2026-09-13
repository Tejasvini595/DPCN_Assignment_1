# Role 2 - Network Construction and Visualisation

This folder consumes the Role 1 handoff and builds the network objects used by the
visualisation stage. Consists the role2 tasks - constructed the network from the defined data given by role1, and performed graphical network visualisations

## Inputs

The script reads these prepared files from `../role1-data-preparation/outputs/tables/`:

- `network_sample.csv` - the 87 respondent nodes.
- `similarity_matrix.csv` - centred respondent profile correlations.
- `edge_list_knn.csv` - the reference positive symmetric k-NN backbone.
- `prepared_responses.csv`, `codebook.csv`, `item_statistics.csv` - for the exploratory statement network.

## Network Construction

The main graph follows Role 1's definition exactly:

- Nodes are respondents in `network_sample.csv`.
- Edges are undirected and weighted by centred profile correlation.
- The backbone is symmetric k-nearest neighbours with `k = round(sqrt(87)) = 9`.
- Only positive similarities are retained.
- `distance = 1 - weight` is stored for later path-based analysis.

Role 2 also exports visual node attributes: agreement intensity, number of items
answered, domain means, distance from the class mean, degree and weighted degree.

## How to Read the Figures

- `fig1_respondent_network.png` is the main network view. Each node is one
  respondent, each edge is a retained similarity link, thicker edges mean stronger
  similarity, and larger nodes are farther from the class average response profile.
- `fig2_backbone_diagnostics.png` checks the graph construction itself: retained
  edge weights, respondent degree counts, and degree versus weighted degree.
- `fig3_topic_layers.png` rebuilds the same respondent-network idea separately
  for Technology, Education, Ethics & Society and Environment.
- `fig4_statement_network.png` is an exploratory statement graph. Nodes are survey
  statements, and edges connect statements whose answers move together by Spearman
  correlation.
- `fig5_network_heatmaps.png` gives matrix views. The first panel shows all
  respondent-to-respondent similarities before sparsification. The second panel
  shows only the retained k-NN edges. The third panel shows statement-to-statement
  Spearman correlations grouped by domain.

## Boundary With Role 3

This folder constructs and visualises networks. It does not interpret communities
or make findings about what the class believes. Community detection, centrality
interpretation, node roles, Mantel correlations between topic layers, NMI between
partitions, null-model comparisons and k-sensitivity plots should be handled in
Role 3.

## Outputs

Run:

```bash
pip install -r requirements.txt
python run_role2.py
python -m pytest -q tests
```

Generated outputs:

- `outputs/graphs/respondent_opinion_network.gexf`
- `outputs/graphs/respondent_opinion_network.graphml`
- `outputs/graphs/topic_layer_T.gexf` through `topic_layer_V.gexf`
- `outputs/graphs/exploratory_statement_network.gexf`
- `outputs/tables/respondent_nodes_visual.csv`
- `outputs/tables/respondent_edges_visual.csv`
- `outputs/tables/topic_layer_*_nodes.csv` and `topic_layer_*_edges.csv`
- `outputs/tables/item_nodes_visual.csv`
- `outputs/tables/item_edges_visual.csv`
- `outputs/figures/fig1_respondent_network.png`
- `outputs/figures/fig2_backbone_diagnostics.png`
- `outputs/figures/fig3_topic_layers.png`
- `outputs/figures/fig4_statement_network.png`
- `outputs/figures/fig5_network_heatmaps.png`
- `outputs/role2_network_summary.json`

## Report Text Starter

Role 2 constructed the network from the prepared Role 1 tables without changing the
node or edge definition. The main respondent graph contains 87 nodes and uses the
positive symmetric k-nearest-neighbour backbone with `k = 9`. Each edge weight is the
Pearson correlation between two respondents' item-centred opinion profiles, so an edge
means that the two respondents deviate from the class consensus in similar directions.
The graph is exported in GEXF and GraphML formats for external visualisation, while the
CSV node and edge tables preserve layout coordinates and visual attributes.

The visualisations use a fixed spring layout weighted by similarity. Node colour shows
agreement intensity or weighted degree, node size shows distance from the class mean,
and edge width shows similarity strength. Matrix heatmaps show the full respondent
similarity matrix, the retained k-NN backbone, and item-to-item Spearman correlations.
Separate topic-layer figures rebuild the same construction within Technology, Education,
Ethics & Society and Environment, allowing Role 3 to compare whether neighbourhoods
persist across topics.
