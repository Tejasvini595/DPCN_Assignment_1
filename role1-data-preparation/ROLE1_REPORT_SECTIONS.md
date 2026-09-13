# Role 1 — Dataset Documentation and Network Design

*Draft of the team report's "Dataset Documentation" section, plus the network definition that opens the "Pipeline Followed" section. Everything after that (construction, visualisation, metrics, findings) belongs to Roles 2 and 3.*

**Owner:** [your name]
**Deliverables:** `run_preparation.py`, the `preprocess.py` / `similarity.py` / `config.py` modules, 9 unit tests, Table 1, Table 2, Figures 1–2, and the prepared datasets that Roles 2 and 3 build on.

---

## 3. Dataset Documentation

### 3.1 What the survey contains

The export has **96 rows** (one per response, identified by an anonymised, non-consecutive response ID) and **60 statements**, 15 in each of four domains coded by prefix: **T** Technology (AI, automation, data, regulation), **E** Education (pedagogy, assessment, research), **S** Ethics & Society (responsibility, inclusion, misinformation) and **V** Environment (climate, resources, sustainability). Every statement was answered on a five-point agreement scale, with an additional *No Comments* option.

There are no demographic fields and no free text. All structure therefore has to come from the pattern of answers itself, which is what makes a network representation the natural tool: we can only learn about the class by comparing responses with each other.

### 3.2 Interpretation, encoding and cleaning

Responses were treated as **ordinal positions on a single agreement axis** and encoded Strongly Disagree = −2, Disagree = −1, Neutral = 0, Agree = +1, Strongly Agree = +2. The sign then carries direction and the magnitude carries strength, with zero at the neutral point.

*No Comments* (37 cells) and blank cells (505 cells) were both coded as **missing** rather than neutral. Declining to comment is an abstention, not a middle opinion; merging it with Neutral would manufacture agreement between two people who simply skipped the same question. This distinction matters for a network, because every spurious similarity becomes a potential edge.

Table 1 documents the audit and the resulting inclusion rules.

**Table 1.** Data audit and inclusion rules. *(generated as `outputs/tables/data_audit.csv`)*

| Check | Finding | Decision |
|---|---|---|
| Fully blank responses | 5 (IDs 44, 60, 68, 73, 78) | Dropped → 91 usable responses |
| Answered Technology only | 4 (IDs 30, 39, 77, 87) | Excluded from the respondent network; usable for T–T statement pairs |
| Environment block missing | 2 (IDs 79, 123) | Kept in the main network (≥ 45/60); excluded from topic layers |
| Scattered item non-response | 17 respondents, 1–4 items each | Kept; missing treated as zero deviation from the item mean |
| Straight-lining (zero variance) | 0 respondents (min. within-person SD = 0.42) | No exclusions needed |
| Duplicate answer profiles | 0 | — |
| Answer distribution (substantive) | SA 43.1%, A 36.7%, N 13.1%, D 4.9%, SD 2.1% | Strong ceiling effect → item-centring required (§3.4) |

Three analysis samples follow from these rules, because each network needs a different kind of completeness:

* **The respondent network** compares whole 60-item profiles, so it uses the **87** people who answered at least 45 of 60 statements (75%). 62 missing cells remain, 1.2% of that matrix.
* **The topic layers** compare profiles within each domain, so they use the **85** people with at least 12 of 15 answers in *every* domain (80%).
* **A statement-level network** correlates two statements at a time and can use anyone who answered both, so it draws on pairwise-complete data from all **91** non-blank responses.

Since these thresholds are judgement calls, they live as named constants in `config.py` (`MIN_ANSWERED_TOTAL = 45`, `MIN_ANSWERED_LAYER = 12`) and can be changed in one place.

### 3.3 What the raw answers show

The class is overwhelmingly agreeable: **79.8%** of substantive answers are Agree or Strongly Agree and only 7.0% disagree (Figure 1). Mean agreement per statement is highest for Environment (89.5%) and Ethics & Society (86.0%), lower for Technology (74.0%) and Education (70.2%).

Because near-universal agreement hides where the real disagreement sits, we also computed the **Tastle–Wierman (2007) consensus index**, which is designed for ordinal scales:

`Cns = 1 + Σᵢ pᵢ · log₂(1 − |Xᵢ − μ| / 4)`

Here pᵢ is the share of respondents choosing option Xᵢ (coded 1–5), μ is the mean answer, and 4 is the width of the scale. The index equals 1 when everyone picks the same option and 0 when the class splits evenly between the two extremes.

* **Least consensual:** E04 *online complements class* (0.55), E02 *exams measure knowledge* (0.58), E03 *compulsory attendance* (0.59), T08 *routine AI diagnosis* (0.62).
* **Most consensual:** E15 *lifelong learning* (0.83), V13 *corporate eco-accountability* (0.81), S05 *control over own data* (0.80).
* Only two statements have a negative mean: E03 *compulsory attendance* (−0.94) and E02 *exams measure knowledge* (−0.16).

The early signal is that disagreement is concentrated in Education and in a few high-stakes Technology items, while Environment and Ethics are close to consensual. Whether that translates into distinct opinion groups is a question for the network itself.

**Figure 1.** Response distribution for all 60 statements, one panel per domain, sorted by mean agreement. Bars are centred on the middle of the Neutral segment, so the portion left of zero is disagreement.

### 3.4 A preparation problem: agreement intensity dominates

Before defining any edge, we checked whether a naive similarity would carry usable information. It would not, for two reasons.

**First, the obvious similarity measure says everyone is alike.** Defining similarity as `1 − (mean absolute answer difference)/4` gives a mean of **0.79** across all 3,741 pairs, with a range of only 0.50 to 0.93 (Figure 2a). Because almost everyone agrees with almost everything, every pair looks similar, and a network built on this measure would be nearly complete and uninformative.

**Second, the largest source of variation is not an opinion at all.** A parallel analysis (Horn, 1965) compares the eigenvalues of the 60×60 item correlation matrix against 200 datasets in which each statement's column has been shuffled independently. Five components exceed the null, but the first dwarfs the rest: it carries **20% of the total variance**, **95% of its loadings share one sign** (so it pushes all 60 statements in the same direction), and respondents' scores on it correlate **r = 0.98** with their overall mean answer (Figure 2c).

In other words, the dominant axis in this survey is *how strongly a person agrees with statements in general*, not what they think about any particular topic. This is the well-documented acquiescence or response-style effect (Paulhus, 1991). Left in the data, it would make the network group people by enthusiasm and label the result "opinion camps."

We therefore prepared the data in two steps:

1. **Item-centring.** Subtract each statement's class mean: `zᵢₐ = xᵢₐ − mₐ`. A value now reads as "how far above or below the class this person sits on this statement," so class-wide consensus contributes nothing.
2. **Pearson correlation between people.** Pearson subtracts each person's own average before comparing, which removes individual intensity.

What remains is **relative opinion content**: which statements a person endorses more or less than the class does, relative to their own baseline.

A four-person toy example makes the difference concrete. Bilal's answers are exactly Asha's minus one, so they hold the same pattern with less enthusiasm:

| | A | B | C | D | Person's mean |
|---|---|---|---|---|---|
| Asha | 2 | 2 | 2 | 1 | 1.75 |
| Bilal | 1 | 1 | 1 | 0 | 0.75 |
| Chen | 2 | 0 | 2 | 2 | 1.50 |
| Divya | 1 | 2 | 0 | 1 | 1.00 |

| Pair | Naive similarity | Centred profile correlation |
|---|---|---|
| Asha–Bilal | 0.75 (Asha's *least* similar pair) | **+1.00** (identical pattern) |
| Asha–Chen | 0.81 | **−0.58** (opposite pattern) |
| Asha–Divya | 0.81 | −0.14 |

The naive measure ranks Chen closer to Asha than Bilal is. The centred correlation correctly identifies Asha and Bilal as holding the same relative views.

Applied to the real data, centred correlations span −0.55 to +0.64 with SD 0.163, against SD 0.137 for the same pipeline run on shuffled surveys (Figure 2b). The wider spread means some pairs are genuinely more alike, and others genuinely more opposed, than chance allows: there is real person-level structure to build a network on.

Intensity is not discarded. Each respondent's mean answer is retained as the node attribute `intensity`, so the later analysis can ask separately whether enthusiasm clusters in the network.

**Figure 2.** (a) Naive agreement similarity is uniformly high. (b) Centred profile correlations are wider than in shuffled surveys. (c) Parallel analysis: five components exceed the null, and the first is general agreement intensity.

---

## 4.1 The network: nodes, edges and design rationale

**Nodes are the 87 respondents.** The assignment asks what the network reveals about *class opinions*, so the primary object should be the people, with edges encoding shared outlook. (A complementary network with statements as nodes is a natural extension; see §4.2.)

**Edges join respondents with similar opinion profiles.** Formally, for respondents *i* and *j*:

`sᵢⱼ = Pearson correlation (zᵢ, zⱼ)` across the 60 item-centred answers

A positive `sᵢⱼ` means the two people deviate from the class consensus in the same directions. Edges are **undirected**, because similarity is symmetric.

**The network is weighted, not binary.** Similarity is a graded quantity: a binary graph would treat `s = 0.15` and `s = 0.60` as the same statement, "these two are alike," discarding exactly the information that distinguishes a core of close peers from a fringe of weak resemblance. We therefore keep `sᵢⱼ` as the edge weight, and store `1 − sᵢⱼ` as a companion `distance` attribute so that path-based measures (betweenness, closeness) treat stronger similarity as shorter distance. Where a metric has both a weighted and an unweighted form, both can be reported.

**Sparsification.** All 3,741 pairs have a similarity value, so the raw graph is complete and must be thinned. We recommend a **symmetric k-nearest-neighbour backbone**: link *i*–*j* if *j* is among *i*'s k most similar peers **or** vice versa, keeping only positive similarities.

The alternative, a global threshold such as "keep all pairs with `s` > 0.3," fails on this data. Only 3% of pairs clear that bar, and because the similarity distribution is narrow, any single cut-off either leaves the most distinctive respondents isolated (their best matches still fall below the line) or admits almost every pair. k-NN adapts to local density and guarantees that every respondent stays connected to their closest peers, which matters when the substantive question is who resembles whom.

For k we suggest the common heuristic `k = round(√n) = round(√87) = 9`, which yields 459 edges, a density of 0.123, and degrees from 9 to 16. Since any single k is somewhat arbitrary, the choice should be confirmed by a sensitivity scan across a range of k values, which belongs to the analysis stage.

**Handling of missing answers.** After centring, a missing answer is set to 0, meaning "no deviation from the class." It therefore neither creates nor destroys similarity, which is the neutral choice. This is equivalent to mean imputation; its main cost is that respondents with many missing items are pulled slightly toward the class centre.

**Table 2.** The network definition, and two complementary networks the same prepared data supports.

| | **A. Respondent network** (main) | **B. Belief network** | **C. Topic layers** |
|---|---|---|---|
| Nodes | 87 respondents | 60 statements | 85 respondents × 4 layers |
| Edge | Pearson *r* of item-centred profiles, k-NN backbone | correlation between two statements' answers | Network A rebuilt on each topic's 15 items |
| Type | undirected, weighted | undirected, weighted, signed | undirected, weighted |
| Question | Which opinion camps exist? | Which beliefs travel together? | Are camps consistent across topics? |

### 4.2 Two complementary networks the prepared data supports

The same encoded matrix supports two further constructions, which we recommend and hand over ready to build:

* **A belief network (B)** with statements as nodes, linked when their answers correlate across respondents. Since the answers are ordinal, Spearman ρ is the appropriate measure, and with 1,770 pairs to test, a false-discovery-rate correction is essential before any edge is called significant. This network asks whether the survey's four topics are how opinions actually organise themselves, or merely how the questionnaire was written.
* **A topic multiplex (C)**: Network A rebuilt four times, once per domain, on the 85 respondents with adequate coverage everywhere. Comparing the layers tests whether someone who thinks like you about technology also thinks like you about the environment.

### 4.3 What is handed over

| Artefact | Content |
|---|---|
| `network_sample.csv` | 87 × 60 encoded matrix, the node set for Network A |
| `similarity_matrix.csv` | 87 × 87 respondent similarity, the edge weights |
| `edge_list_knn.csv` | reference k = 9 backbone as source / target / weight / distance |
| `prepared_responses.csv` | 91 × 60 encoded matrix for the statement-level network |
| `item_statistics.csv`, `codebook.csv` | per-statement means, spread, consensus, short labels |
| `preparation_metrics.json` | every number quoted above |

Node attributes available for the visualisation and analysis stages: `intensity` (mean answer), domain-wise means, number of items answered, and k-NN hubness.

---

## Limitations of the preparation stage

1. **Ceiling effect.** With 80% agreement, most statements carry little variance, which limits every correlation computed downstream.
2. **Likert treated as interval.** Pearson on centred values assumes equal spacing between answer options. This is standard for profile correlations but is an assumption; the statement-level network should use rank-based Spearman instead.
3. **Mean imputation.** Missing answers become "at the class mean," slightly pulling similarities toward zero for the few respondents with several blanks (mainly IDs 79 and 123).
4. **Removing intensity is a choice.** If general enthusiasm is a genuine disposition rather than a response style, centring discards something real. This is why it is preserved as a node attribute rather than deleted.
5. **Inclusion thresholds are judgement calls.** 45/60 and 12/15 are defensible but not unique; both are single constants in `config.py` for easy sensitivity testing.

---

## Reproducing this section

```bash
pip install numpy pandas scipy matplotlib
python -m pytest -q tests      # 9 tests
python run_preparation.py      # ~10 seconds
```

All randomness derives from `SEED = 42` in `config.py`, so reruns reproduce these numbers exactly.

**What the 9 tests check:** the CSV shape and codebook (15 items per domain); that encoding loses nothing but *No Comments*; that *No Comments* is not silently treated as Neutral; the Tastle–Wierman index at both bounds; the 45/60 selection rule; that item-centring zeroes each statement's mean and leaves missing cells at zero; that profile correlation equals a row-wise Pearson; that the toy example above behaves as claimed; and that the k-NN backbone leaves nobody isolated and stores consistent weights and distances.

---

## References for this section

* Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179–185.
* Paulhus, D. L. (1991). Measurement and control of response bias. In *Measures of Personality and Social Psychological Attitudes* (pp. 17–59). Academic Press.
* Tastle, W. J., & Wierman, M. J. (2007). Consensus and dissention: A measure of ordinal dispersion. *International Journal of Approximate Reasoning*, 45(3), 531–545.
