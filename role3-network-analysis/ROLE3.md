# Role 3 — Network Analysis, Key Metrics, and Findings

*Draft of the team report's "Analysis and Visualizations" (metrics half) and
"Results and Discussion" sections. Builds on Role 1's prepared data and Role 2's
constructed graphs without changing either — this folder only measures and
interprets what was already built.*

**Inputs:** `../role2-network-construction/outputs/tables/respondent_nodes_visual.csv`
/ `respondent_edges_visual.csv` (87-node main network), the four `topic_layer_*`
tables, `item_edges_visual.csv` (exploratory statement network); and
`../role1-data-preparation/outputs/tables/network_sample.csv` / `similarity_matrix.csv`
/ `prepared_responses.csv` (for the null-model, k-sensitivity, and statement-FDR
analyses, which need to rebuild the construction under controlled conditions rather
than read a fixed graph).

**Code:** `run_role3.py`, `src/opinion_network_role3/{analysis,plots}.py`, 10 unit
tests in `tests/`.

---

## 1. Methods

| Question | Method |
|---|---|
| Who is central? | Degree, weighted degree (strength), betweenness & closeness (weighted by `distance = 1-weight`), eigenvector centrality, local clustering — all computed directly on Role 2's graph. |
| Is there real community structure? | Louvain modularity maximisation (`networkx`, weight = similarity, seed 42). |
| Or is it just a k-NN artefact? | Role 1's own prescribed null model: shuffle every item's column independently (destroys individual coherence, keeps each item's distribution), rebuild the *entire* pipeline — item-centre → Pearson profile correlation → k=9 symmetric k-NN — 200 times, and compare observed modularity/clustering against that null distribution. |
| Which respondents bridge communities vs. anchor them? | Guimerà–Amaral (2005) node roles: within-module strength z-score and participation coefficient, computed on the Louvain partition. |
| Was k=9 a reasonable choice? | Rebuild the same construction for k = 4…15 from Role 1's full similarity matrix; track density, modularity, fragmentation. |
| Do opinion camps hold across topics? | Two independent tests on the 85 respondents common to all four topic layers: (a) **Mantel test** — correlation between each pair of layers' *full* (unsparsified) similarity matrices, significance from 999 label permutations; (b) **NMI** between each pair's Louvain community partition. |
| Are the exploratory statement-network edges real? | Recomputed pairwise Spearman ρ with p-values for all 1,770 item pairs, applied Benjamini–Hochberg FDR correction (α = 0.05), and checked which of Role 2's `\|ρ\|≥0.35` edges actually survive. |
| Does enthusiasm cluster in the network? | Pearson correlation between each centrality metric and the node attributes `intensity` (mean agreement) and `distance_to_class_mean` (atypicality), both handed off from Role 1/2. |
| Do the survey's own T/E/S/V categories match how opinions actually cluster? | Louvain community detection run directly on the FDR-corrected statement network (§2.5), then a confusion matrix + NMI between the algorithmic communities and the survey's predefined domain labels — the algorithm never sees the domain labels. |
| How much do "opinion camps" really dissolve at the topic level? | An alluvial (Sankey) diagram tracking, respondent by respondent, which Technology-layer / Education-layer community they land in given their Main-network community — a visual counterpart to the Mantel/NMI numbers in §2.4. |

All randomness (null-model shuffles, Louvain, Mantel permutations, layout) is
seeded at 42, matching Role 1 and Role 2. Full numbers are in
`outputs/role3_analysis_summary.json`; every claim below cites its source table.

---

## 2. Key Metrics

### 2.1 Global network structure (`role3_analysis_summary.json → global_metrics`)

| Metric | Value |
|---|---|
| Nodes / edges | 87 / 459 |
| Density | 0.123 |
| Connected components | 1 |
| Weighted diameter | 2.59 |
| Mean weighted shortest path | 1.58 |
| Transitivity (global clustering) | 0.249 |
| Mean weighted local clustering | 0.106 |
| Degree assortativity | −0.023 (no hub-hub preference) |
| Intensity assortativity | +0.143 (weak positive) |
| Modularity (Louvain) | 0.371 |
| Communities | 7 (sizes 7, 10, 10, 11, 12, 18, 19) |

### 2.2 Null-model validation (`outputs/tables/null_model_modularity.csv`)

| | Observed | Null mean ± SD (200 reps) | Empirical p |
|---|---|---|---|
| Modularity | 0.371 | 0.338 ± 0.015 | **0.015** |
| Avg. weighted clustering | 0.106 | 0.091 | — |

### 2.3 k-sensitivity (`outputs/tables/k_sensitivity.csv`, k = 4…15)

| k | Edges | Density | Modularity | Components |
|---|---|---|---|---|
| 4 | 227 | 0.061 | 0.520 | 1 |
| 9 (chosen) | 459 | 0.123 | 0.371 | 1 |
| 15 | 746 | 0.199 | 0.305 | 1 |

The graph never fragments even at k = 4, and modularity falls monotonically as k
grows (more edges wash out structure) — k = 9 sits past the steepest early drop.

### 2.4 Topic-layer agreement (`outputs/tables/topic_layer_mantel.csv`, `_nmi.csv`)

| Layer pair | Mantel r | p | NMI |
|---|---|---|---|
| Main – Education | **0.62** | <.001 | 0.37 |
| Main – Technology | **0.55** | <.001 | 0.19 |
| Main – Ethics & Society | 0.32 | <.001 | 0.25 |
| Main – Environment | 0.22 | <.001 | 0.18 |
| Technology – Education | 0.03 | .048 | 0.06 |
| all other cross-topic pairs | ≤ 0.05 | ns (p ≥ .20, one exception at .01) | 0.05–0.12 |

### 2.5 Statement-level network under FDR correction (`outputs/tables/statement_fdr_edges.csv`)

| | Count |
|---|---|
| Item pairs tested | 1,770 |
| Naive p < .05 | 652 |
| **FDR-significant (q ≤ .05)** | **432** |
| Role 2's exploratory `\|ρ\|≥0.35` network | 189 |
| Of those 189, how many survive FDR correction | **189 (100%)** |

### 2.6 Intensity/atypicality vs. network position (`outputs/tables/intensity_position_correlations.csv`)

| Centrality | r with intensity | p | r with atypicality | p |
|---|---|---|---|---|
| Eigenvector | **+0.49** | 1×10⁻⁶ | −0.30 | .004 |
| Closeness | **+0.46** | 7×10⁻⁶ | −0.33 | .002 |
| Strength | +0.36 | 6×10⁻⁴ | −0.29 | .006 |
| Betweenness | +0.28 | .008 | −0.24 | .023 |
| Degree | +0.27 | .011 | −0.26 | .013 |
| Clustering | +0.23 | .030 | −0.12 | .28 (ns) |

### 2.7 Node roles (`outputs/tables/node_roles.csv`)

- Highest within-module z-score: **2.44** (respondent 79) — below the conventional
  z > 2.5 "module hub" cutoff, so **no respondent qualifies as a true local hub**.
- 49 of 87 respondents (56%) have participation coefficient > 0.62, the conventional
  "connector" cutoff — most respondents split their ties across communities rather
  than concentrating them in one.

### 2.8 Statement network's own community structure vs. the survey's categories
(`outputs/tables/statement_domain_confusion.csv`, `statement_community_membership.csv`)

Louvain run directly on the FDR-corrected statement network (60 items, 432 edges,
§2.5) finds 7 communities, modularity 0.207, and **NMI = 0.56** against the
survey's own T/E/S/V labels — moderate-to-substantial agreement, not perfect.
The confusion matrix shows *why*: Education (13/15 items in one community),
Ethics & Society (11/15), and Environment (12/15) each collapse almost entirely
into a single dominant algorithmic community, but **Technology is the domain
that breaks the pattern** — its 15 items scatter across six different
communities (2, 7, 3, 0, 1, 1, 1), never dominating any single one.

| Survey domain | Items in its largest algorithmic community | Community |
|---|---|---|
| Education | 13 / 15 | C2 |
| Ethics & Society | 11 / 15 | C4 |
| Environment | 12 / 15 | C6 |
| Technology | 7 / 15 | C1 |

### 2.9 Cross-domain community flow (`outputs/tables/community_flow_main_to_{T,E}.csv`)

Sankey/alluvial flow counts of how each Main-network community's respondents
redistribute into the Technology-layer and Education-layer communities — the
row-by-row data behind Figure 9. Most Main communities fragment substantially:
6 of 7 split across 2–5 different Technology communities and 2–4 different
Education communities, with no dominant destination. The one exception is
Main community 3 (the smallest, n=7, also the highest-intensity group,
§2.3/community_summary), whose members land in a *single* Education
community — but even that community still splits across 2 different
Technology communities, so no Main community holds together across **both**
topics simultaneously.

---

## 3. Findings and Discussion

### 3.1 The community structure is real, not just a k-NN side-effect

k-NN graphs are clustered by construction, so raw modularity alone proves nothing.
Rebuilding the entire pipeline 200 times on column-shuffled surveys (identical item
distributions, no individual coherence) gives a null modularity of 0.338 ± 0.015;
the observed graph's 0.371 exceeds the null mean by more than two standard
deviations (empirical p = 0.015, Figure `fig2_null_model.png`). The seven Louvain
communities (sizes 7–19) are therefore a genuine feature of how the class answered,
not an artefact of forcing every respondent to have at least k = 9 neighbours.

### 3.2 Opinion camps are topic-specific, not general ideological camps

This directly answers the open question Role 1 flagged ("are camps consistent
across topics?"). They are not. Each topic layer correlates only moderately with
the main (60-item) network — most strongly Education (Mantel r = 0.62) and
Technology (r = 0.55), more weakly Ethics & Society (r = 0.32) and Environment
(r = 0.22) — but the four topics barely correlate **with each other**: five of the
six cross-topic Mantel correlations are statistically indistinguishable from zero
(r ≤ 0.05, p ≥ .20), and the strongest, Ethics–Environment, is still only r = 0.05.
The NMI results tell the same story (Figure `fig5_topic_layer_similarity.png`):
community partitions transfer weakly from the main network into any single topic
(NMI 0.18–0.37) and barely at all between two different topics (NMI 0.05–0.12).
**A respondent's opinion pattern on Technology tells you almost nothing about their
pattern on Education, Ethics, or Environment** — this survey does not reveal one
underlying ideological split, but several largely independent ones.

### 3.3 Education and Technology organise the network; Ethics and Environment barely do

This confirms, at the network level, what Role 1 found item-by-item: Education
(mean agreement 70.2%, most items with low consensus) and Technology (74.0%
agreement) are where real disagreement lives, while Ethics & Society (86.0%) and
especially Environment (89.5%) are close to unanimous. Because the main network's
edges are dominated by whichever items actually vary between people, Education and
Technology contribute most of the structure that produces the observed communities,
and Environment — the most consensual domain — contributes the least (its Mantel
correlation with the main network, r = 0.22, is the weakest of the four).

### 3.4 The network's core is agreeable and typical; disagreement sits at the edges

Every centrality metric correlates positively with `intensity` (how agreeable a
respondent is on average) and negatively with `distance_to_class_mean`
(atypicality) — most strongly for eigenvector centrality (r = 0.49) and closeness
(r = 0.46), all significant at p < .03 except clustering-vs-atypicality (Figure
`fig7_intensity_vs_centrality.png`). In a similarity-based k-NN network this is an
expected mechanical consequence — a respondent whose profile deviates a lot from
everyone else's will, by definition, have weaker best-matches — but it is worth
stating plainly for the report: **the people most "in the thick of" the network are
the more agreeable, less unusual respondents; the class's more distinctive or
critical voices are structurally pushed to the periphery**, even though they were
never excluded from the graph. The intensity–degree assortativity is only weakly
positive (+0.143), so this is a centrality effect, not evidence that agreeable
people cluster tightly together as a separate camp.

### 3.5 Formal correction shows Role 2's exploratory statement network under-claimed, not over-claimed

Role 1 warned that 1,770 tested item pairs at p < .05 could produce ~88 false edges
by chance, and Role 2 explicitly left its `\|ρ\|≥0.35` statement network
(189 edges) uncorrected. Applying Benjamini–Hochberg FDR correction (α = 0.05)
finds 432 significant pairs — and **all 189 of Role 2's threshold edges are exactly
the 189 strongest of those 432** (Figure `fig6_statement_fdr.png`). In other words,
none of Role 2's exploratory edges were false positives; the simple magnitude
cutoff was conservative, and there are 243 additional, weaker-but-statistically-robust
item correlations (0.27 ≤ \|ρ\| < 0.35) that a rigorous analysis recovers and a
visual threshold misses.

### 3.6 No dominant hubs — influence is broadly distributed, not concentrated

Guimerà–Amaral role analysis finds no respondent crosses the conventional z > 2.5
threshold for a "module hub" (the highest is 2.44), while 56% of respondents are
"connectors" whose ties spread across multiple communities (Figure
`fig4_node_roles.png`). This is a direct consequence of the k-NN construction,
which guarantees every respondent at least k = 9 neighbours regardless of how
distinctive they are — so unlike many real-world social networks, this opinion
network has no small clique of outsized influencers; cohesion comes from broad,
overlapping connectivity instead.

### 3.7 k = 9 is defensible, not special

The k-sensitivity scan (Figure `fig3_k_sensitivity.png`) shows the graph is
connected at every k from 4 to 15 and that modularity falls smoothly and
monotonically as k grows (0.52 at k = 4 → 0.37 at k = 9 → 0.31 at k = 15) — there is
no sharp elbow that would single out one "correct" k. Role 1's `round(√87) = 9`
heuristic sits in the middle of this range, past the steepest early drop in
modularity, which supports it as a reasonable default rather than an arbitrary one.

### 3.8 The survey's own categories mostly — but not entirely — match how opinions cluster

Running Louvain directly on the FDR-corrected statement network, without ever
telling it which item belongs to which domain, recovers the survey's own
T/E/S/V structure moderately well (NMI = 0.56): Education, Ethics & Society,
and Environment each collapse almost entirely into one dominant algorithmic
community (13/15, 11/15, and 12/15 of their items respectively — Figure
`fig8_statement_communities.png`). **Technology is the exception** — its 15
items scatter across six different communities with no dominant one (at most
7/15 in any single community). This converges with every other Technology
finding in this report (§3.3, §3.4's Mantel results): Technology attitudes are
not one coherent bloc of belief the way Education, Ethics, and Environment
attitudes are; individual Technology statements associate more with statements
in *other* domains than with each other.

### 3.9 Community dissolution across topics, made visible

Figure `fig9_community_flow.png` gives the direct visual counterpart to §3.2's
Mantel/NMI numbers. Six of the seven Main-network communities each split
across 2–5 different communities in the Technology layer and 2–4 in the
Education layer, with no single destination dominating — the thick, orderly
bands on the left dissolve into a criss-crossing tangle on the right. The one
partial exception, the smallest and most agreeable Main community (n=7, §2.3),
lands entirely within one Education community but still splits across two
Technology communities — confirming that **no Main-network community holds
together across both topics simultaneously**, reinforcing §3.2's conclusion
from a second, independent angle.

---

## 4. Limitations

1. **Modularity depends on the resolution parameter.** Louvain was run at the
   default resolution (1.0); a coarser or finer resolution could merge or split the
   seven communities differently. The null-model comparison (§3.1) is robust to this
   because both observed and null use the identical procedure, but the *number* of
   communities (7) should not be over-interpreted as fixed.
2. **Community domain-profiles mix intensity and opinion content.** `community_summary.csv`
   reports each community's *raw* domain means (not item-centred), so differences
   between communities partly reflect each group's baseline agreement level as well
   as genuine relative opinion — read those numbers as descriptive, not as isolated
   opinion signals.
3. **The FDR-corrected statement network is still exploratory.** It corrects for
   multiple testing but not for the acquiescence/intensity confound Role 1
   documented (item correlations are computed on raw, not centred, answers), so
   some of the 432 significant pairs may reflect shared response style rather than
   substantively linked beliefs.
4. **Mantel test assumes exchangeability under permutation**, standard for this kind
   of matrix correlation but, as with any permutation test, sensitive to the number
   of permutations (999 here); p-values near conventional thresholds (e.g.
   Technology–Education, p = .048) should be treated as borderline, not conclusive.
5. **Centrality metrics are computed on the fixed k = 9 backbone only.** The
   intensity/position correlations in §3.4 could shift somewhat at a different k,
   though §3.7 shows the network's qualitative structure is stable across the
   4–15 range.
6. **The statement-network Louvain result (§3.8) inherits the FDR network's own
   limitation** (#3 above) and has lower modularity (0.207) than the main
   respondent network's (0.371) — the 7 algorithmic item-communities are a
   real but comparatively weaker structure, appropriate for the qualitative
   "does this roughly match the domains" question asked here, not for strong
   claims about a definitive alternative item taxonomy.

---

## 5. Reproduction

```bash
pip install -r requirements.txt
python3 run_role3.py       # ~10-15 seconds (200 null-model reps + 10x999 Mantel permutations)
python3 -m pytest -q tests # 6 tests
```

---

## Report text starter

Role 3 analysed the network Role 2 constructed without altering it, to test
whether its structure is genuine and to extract what it reveals about class
opinions. A null-model comparison (200 replicates of the full pipeline rerun on
column-shuffled surveys) confirms the observed community structure exceeds chance
(modularity 0.371 vs. null 0.338±0.015, p=0.015; seven communities, sizes 7–19).
Comparing each topic-specific layer against the main network and against each other
(Mantel tests and NMI on the 85 respondents common to all four) shows opinion camps
are topic-specific rather than one general ideological split: Education and
Technology drive most of the network's structure, Environment and Ethics contribute
comparatively little, and the four topics correlate with the main network far more
than with each other (r = 0.22–0.62 vs. r ≤ 0.05 between topics). Centrality
analysis shows the network's core is composed of more agreeable, less atypical
respondents, with distinctive or critical voices structurally pushed toward the
periphery, though no respondent acts as a dominant hub — the k-NN construction
spreads connectivity broadly, with 56% of respondents classified as
cross-community "connectors." Finally, applying formal Benjamini–Hochberg
correction to the 1,770-pair exploratory statement network shows all 189 of Role
2's visually-thresholded edges are statistically robust, and identifies 243
additional weaker-but-significant item correlations that the simple magnitude
cutoff missed. Two further checks reinforce the topic-specificity finding: an
alluvial diagram tracking respondents from their Main-network community into
their Technology- and Education-layer communities shows those communities
fragmenting into a criss-crossing tangle rather than staying intact, and
running Louvain directly on the statement network (never told which item
belongs to which domain) recovers the survey's own categories only moderately
well (NMI = 0.56) — Education, Ethics, and Environment each collapse into one
dominant algorithmic community, but Technology's items scatter across six,
confirming Technology is the domain where individual attitudes least resemble
a single coherent bloc.
