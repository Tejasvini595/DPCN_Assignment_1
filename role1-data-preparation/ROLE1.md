# Role 1 — Data Preparation and Network Design

This stage converts the raw survey export into an analysis-ready dataset and fixes the
definition of the opinion network: what a node is, what an edge is, and what an edge
weight means. Roles 2 and 3 build and analyse the network from the outputs listed below.
No construction or analysis decisions are made here beyond a reference backbone offered
for comparison.

---

## 1. Source data

`data/raw/Survey_Results_UC.csv` — 96 response rows × 60 attitude statements, collected on
a five-point agreement scale. The statements are divided equally across four domains,
identified by the first character of each item code:

| Prefix | Domain | Items |
|---|---|---|
| `T` | Technology | T01–T15 |
| `E` | Education | E01–E15 |
| `S` | Ethics & Society | S01–S15 |
| `V` | Environment | V01–V15 |

Raw cells contain one of five scale labels, the non-substantive label `No Comments`, or a
blank.

---

## 2. Preparation decisions

**Encoding.** Labels are mapped to a symmetric integer scale so that sign carries the
direction of opinion and zero is the true midpoint:

```
Strongly Disagree -2   Disagree -1   Neutral 0   Agree +1   Strongly Agree +2
```

Any label outside this set raises an error at load time rather than being silently
discarded.

**Non-response.** Blanks and `No Comments` are both treated as **missing**, never as
neutral. An abstention is a refusal to place oneself on the scale, not a middle position;
coding it as zero would manufacture agreement that was never expressed. The dataset
contains 505 blank cells and 37 `No Comments` cells.

**Inclusion.** Three sample sizes are produced because three constructions have different
data requirements. They are deliberately not harmonised:

| Sample | n | Rule | Intended use |
|---|---|---|---|
| Usable | 91 | At least one answer given | Statement-level network (needs only pairwise-complete data) |
| Main network | 87 | ≥ 45 of 60 items answered | Respondent network (needs comparable whole profiles) |
| Topic layers | 85 | ≥ 12 of 15 items in *every* domain | Per-domain layers |

Five respondents (IDs 44, 60, 68, 73, 78) are entirely blank and are dropped. Four
(IDs 30, 39, 77, 87) answered the Technology block only and are excluded from the
respondent network. Two (IDs 79, 123) are missing the Environment block and are retained
in the main network but excluded from topic layers. After selection, only 62 cells
(1.19%) of the main sample remain missing.

**Quality checks.** No straight-lining was detected — the smallest within-person standard
deviation is 0.42, so no respondent answered uniformly. No duplicate response profiles
were found. No exclusions on either ground were necessary. The full audit is recorded in
`outputs/tables/data_audit.csv`.

---

## 3. Why the data must be centred

The class agrees with **79.8%** of statements and disagrees with only **7.0%**. Agreement
is the default response, which means the strongest signal in the raw answers is not what a
respondent believes but how emphatically that respondent endorses statements in general.

Two independent checks confirm this:

- **Parallel analysis** (Horn 1965; 200 column-permuted replicates) finds the first
  principal component holding **19.9%** of the variance, with **95%** of its loadings
  sharing the same sign and a correlation of **r = 0.98** with a respondent's simple mean
  answer. PC1 is agreement intensity and nothing else. Five components exceed the null
  95th percentile.
- **A naive similarity** — one minus mean absolute Likert distance — places every pair
  between 0.50 and 0.93 with a mean of 0.79. Under that measure everyone resembles
  everyone, which yields no usable network structure.

A network built on uncentred answers would therefore sort respondents by enthusiasm, and
any apparent opinion camps would be an artefact of response style.

**The correction applied** is two-stage:

1. **Item-centring** — each statement's class mean is subtracted from its column, removing
   the shared consensus. A missing answer becomes exactly zero deviation, so it neither
   creates nor destroys similarity and requires no imputation.
2. **Pearson correlation between respondents** — the row-centring implicit in Pearson
   additionally removes each respondent's own average, eliminating acquiescence bias.

What remains is relative opinion content: which statements a respondent endorses more or
less than the class does, measured against that respondent's own baseline.

The resulting similarities span **−0.55 to +0.64** with a standard deviation of **0.163**,
against **0.137** for column-shuffled null data. The observed spread exceeds chance, so
genuine structure is present. Of the 3,741 respondent pairs, 46.5% are positive.

---

## 4. Network definition

> **Nodes** are the 87 respondents in the main sample. **Edges** join respondents whose
> item-centred opinion profiles correlate positively, are **undirected**, and are
> **weighted** by that correlation.

An edge means *"these two deviate from the class in the same directions"* — not *"these
two both agree a lot."* Preserving this definition across all three sections is important;
if it is changed, the written report sections must be revised to match, or the three parts
will describe different networks.

---

## 5. Outputs

All tables are in `outputs/tables/`, figures in `outputs/figures/`.

| File | Shape | Contents |
|---|---|---|
| `network_sample.csv` | 87 × 60 | **The node set.** Encoded answers (−2…+2) for respondents meeting the ≥ 45/60 rule. Rows are response IDs, columns are item codes; a blank cell is missing. |
| `similarity_matrix.csv` | 87 × 87 | **The edge weights.** Symmetric Pearson correlation between item-centred profiles, range −0.55 to +0.64. |
| `edge_list_knn.csv` | 459 rows | **Reference backbone** at k = 9: `source, target, weight, distance`. Loads directly into Gephi, igraph or networkx. |
| `prepared_responses.csv` | 91 × 60 | All usable respondents, for a statement-level network. |
| `item_statistics.csv` | 60 rows | Per statement: n, mean, SD, % agree/neutral/disagree, Tastle–Wierman consensus, and counts per option. |
| `codebook.csv` | 60 rows | Item code → domain, full statement text, short label for figures. |
| `data_audit.csv` | 7 rows | Each quality check, its finding, and the decision taken. Table 1 of the report. |
| `preparation_metrics.json` | — | Every quoted figure, machine-readable. |
| `fig1_responses.png` | — | Diverging stacked bars of the response distribution, one panel per domain. |
| `fig2_similarity.png` | — | Naive similarity, centred correlation against null, and the parallel-analysis scree plot. |

The reference backbone uses symmetric (union) k-nearest neighbours at k = round(√87) = 9,
retaining positive weights only: 87 nodes, 459 edges, degree 9–16, density 0.123. Edge
`distance` is precomputed as `1 − weight` so that path-based centralities treat stronger
similarity as shorter.

**Node attributes** can be derived from `network_sample.csv`:

```python
import pandas as pd
X = pd.read_csv("outputs/tables/network_sample.csv", index_col=0)
intensity   = X.mean(axis=1)                                           # overall agreement level
n_answered  = X.notna().sum(axis=1)
atypicality = ((X.fillna(X.mean()) - X.mean()) ** 2).sum(axis=1) ** 0.5  # distance from the class
domain_mean = {d: X[[c for c in X if c[0] == d]].mean(axis=1) for d in "TESV"}
```

`intensity` is the attribute most worth carrying forward: it is precisely what centring
removed, and whether it clusters in the network is a substantive question for the analysis
stage.

---

## 6. Substantive findings from the preparation stage

- Disagreement is concentrated in **Education**. Only E03 (mean −0.94) and E02 (−0.16)
  have negative means, and the four least consensual statements are E04, E02, E03 and T08.
  If the network shows structure, this is where to look first.
- **Environment and Ethics & Society are near-consensual** (89.5% and 86.0% mean agreement
  respectively) and can be expected to differentiate respondents less.

---

## 7. Open decisions

The backbone supplied here is a reference, not a final construction.

**Role 2 — construction and visualisation**
- The value of *k*. The value 9 follows the `round(√n)` heuristic; anything from 4 to 15 is
  defensible.
- Whether to use k-NN at all. A global threshold or a disparity filter would also serve.
- Layout, colouring and node sizing.
- Whether to build the statement-level network and the topic layers.

**Role 3 — analysis and findings**
- Choice of metrics and community-detection algorithm.
- **Validation against a null model.** k-NN graphs are clustered and modular *by
  construction*, so raw modularity carries little meaning without comparison. The
  recommended procedure is to shuffle each statement's column independently, producing
  null surveys with identical item distributions but no individual coherence, and push
  them through the identical construction. `permute_items()` in `run_preparation.py`
  implements the shuffle.
- Stability of communities across repeated randomised runs.
- A sensitivity scan over *k*, since a single value must be fixed for the report.

---

## 8. Notes for downstream use

1. **Missing answers are already handled.** After centring they are zero deviation. No
   further imputation is needed or wanted.
2. **Use Spearman, not Pearson, for statement-to-statement correlations.** Those operate on
   raw ordinal answers. Pearson is appropriate for respondent profiles only because
   centring has already transformed them.
3. **Correct for multiple testing** in any statement-level network: 1,770 item pairs at
   p < .05 produce roughly 88 false edges by chance. A false-discovery-rate correction is
   the standard choice.
4. **Request alternative cuts rather than rebuilding them.** A different threshold, a
   per-domain matrix or an alternative similarity measure can be regenerated from this
   pipeline, which keeps a single source of truth for the numbers all three sections quote.

---

## 9. Code layout and reproduction

| Path | Purpose |
|---|---|
| `run_preparation.py` | Single entry point; regenerates every table, figure and metric |
| `src/opinion_network/config.py` | All parameters: encoding, inclusion thresholds, seed, labels |
| `src/opinion_network/preprocess.py` | Loading, encoding, quality audit, centring, item statistics |
| `src/opinion_network/similarity.py` | Profile correlation, naive similarity, k-NN backbone |
| `src/opinion_network/plots.py` | Figures 1 and 2 |
| `tests/test_preparation.py` | 9 sanity tests covering the encoding and centring guarantees |

```bash
pip install -r requirements.txt
python -m pytest -q tests      # 9 tests, all passing
python run_preparation.py      # roughly 10 seconds
```

All randomness derives from `SEED = 42` in `config.py`, so repeated runs reproduce
identical numbers.
