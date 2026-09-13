# Handoff — what Roles 2 and 3 receive

Role 1 (data preparation and network design) is complete. This note tells you what is ready, what the numbers mean, and what remains open for you to decide. Nothing here constrains how you build or analyse the network; the files are equally usable from Python, R, Gephi or a spreadsheet.

Run `python run_preparation.py` to regenerate everything (about 10 seconds).

---

## The files you need

All in `outputs/tables/`.

| File | Shape | What it is |
|---|---|---|
| **`network_sample.csv`** | 87 × 60 | **The node set.** Encoded answers (−2…+2) for the 87 respondents who answered ≥ 45 of 60 statements. Rows are response IDs, columns are statement codes. Blank cell = missing. |
| **`similarity_matrix.csv`** | 87 × 87 | **The edge weights.** Symmetric Pearson correlation between item-centred profiles. Ranges −0.55 to +0.64. |
| **`edge_list_knn.csv`** | 459 rows | **A reference backbone** at k = 9: `source, target, weight, distance`. Import straight into Gephi, igraph or networkx. |
| `prepared_responses.csv` | 91 × 60 | All usable respondents, for a statement-level network (which can use pairwise-complete data). |
| `item_statistics.csv` | 60 rows | Per statement: n, mean, SD, % agree/neutral/disagree, Tastle–Wierman consensus, counts per option. |
| `codebook.csv` | 60 rows | Statement code → domain, full text, short label for figures. |
| `data_audit.csv` | 7 rows | Table 1 of the report. |
| `preparation_metrics.json` | — | Every number in the Role 1 sections, in machine-readable form. |

**Node attributes you may want.** Compute from `network_sample.csv`:

```python
import pandas as pd
X = pd.read_csv("outputs/tables/network_sample.csv", index_col=0)
intensity = X.mean(axis=1)                                   # overall agreement level
n_answered = X.notna().sum(axis=1)
atypicality = ((X.fillna(X.mean()) - X.mean()) ** 2).sum(axis=1) ** 0.5   # distance from the class
domain_mean = {d: X[[c for c in X if c[0] == d]].mean(axis=1) for d in "TESV"}
```

`intensity` is the one to carry through: it is what centring removed, and whether it clusters in the network is a real question for the analysis stage.

---

## The definition, in one paragraph

**Nodes** are the 87 respondents. **Edges** join respondents whose opinion profiles correlate positively, **weighted** by that correlation and **undirected**. The correlation is computed on *item-centred* answers (each statement minus its class mean), and Pearson additionally removes each person's own average. So an edge means "these two deviate from the class in the same directions," not "these two both agree a lot."

The full rationale is in `ROLE1_REPORT_SECTIONS.md` §3.4 and §4.1. The short version: 80% of answers are agreement, and the first principal component (20% of variance, r = 0.98 with a person's mean answer) is just how strongly someone agrees in general. Without centring, the network sorts people by enthusiasm and the "opinion camps" are an artefact.

**Please keep this definition** even if you change everything else, or the team's sections will describe different networks. If you want to change it, tell me and I will update §4.1 to match.

---

## What is still yours to decide

Role 1 supplies a reference backbone, not a final one. These are genuinely open:

**Role 2 (construction and visualisation)**
* **The value of k.** 9 comes from the `round(√n)` heuristic. Anything in 4–15 is defensible.
* **Whether to use k-NN at all.** A global threshold or a disparity filter would also work; §4.1 explains why we lean k-NN, and that argument should be rewritten if you choose otherwise.
* **Layout, colouring and sizing.** All of it.
* **Whether to build the belief network and the topic layers** (§4.2 sketches both; `prepared_responses.csv` is ready for the first).

**Role 3 (analysis and findings)**
* **Which metrics to report**, and which community detection algorithm to use.
* **How to validate.** Strongly recommended: shuffle each statement's column independently to make null surveys, then push them through the identical construction. k-NN graphs are clustered and modular *by construction*, so raw modularity means little without this comparison. `run_preparation.py → permute_items()` implements the shuffle if useful.
* **Whether communities are stable.** Community detection is randomised; running it many times and comparing partitions is worth the effort here.
* **A sensitivity check over k**, since Role 2 has to pick one value.

---

## Things worth knowing before you start

1. **Three sample sizes exist for good reasons.** 87 for the respondent network (needs whole profiles), 85 for topic layers (needs coverage in every domain), 91 for a statement-level network (needs only pairs). Don't harmonise them; each rule fits its own construction.
2. **Missing answers are already handled.** After centring they become 0, meaning "no deviation," so they neither create nor destroy similarity. You do not need to impute again.
3. **Disagreement is concentrated in Education.** Only E03 (−0.94) and E02 (−0.16) have negative means, and the four least consensual statements are E04, E02, E03 and T08. If the network finds structure, this is where to look first.
4. **Environment and Ethics are near-consensual** (89.5% and 86.0% agreement). Expect them to differentiate people less.
5. **Use Spearman, not Pearson, for statement-to-statement correlations.** Those are raw ordinal answers. Pearson is appropriate for the respondent profiles only because centring has already transformed them.
6. **Correct for multiple testing** in any statement-level network: 1,770 pairs at p < .05 yields about 88 false edges by chance. A false-discovery-rate correction is the standard choice.
7. **Distances are precomputed.** `edge_list_knn.csv` has `distance = 1 − weight`, so betweenness and closeness treat stronger similarity as shorter.

---

## Questions I can answer

Anything about the encoding, inclusion thresholds, the centring argument, or the similarity definition. If you need a different cut of the data (a different threshold, a per-domain matrix, an alternative similarity), ask rather than rebuilding it — that keeps one source of truth for the numbers all three sections quote.
