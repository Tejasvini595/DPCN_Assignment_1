# Role 1 — Data preparation and network design

Opinion Network Formation assignment. This part covers how the survey data was
interpreted and prepared, and how the network's nodes and edges are defined.
Roles 2 and 3 build and analyse the network from these outputs.

## Contents

| Path | What it is |
|---|---|
| `ROLE1_REPORT_SECTIONS.md` | The report sections this role owns: Dataset Documentation (§3) and the network definition (§4.1–4.3) |
| `HANDOFF.md` | What Roles 2 and 3 receive, and what is still theirs to decide |
| `run_preparation.py` | Single entry point; regenerates every table, figure and number |
| `src/opinion_network/config.py` | All parameters: encoding, inclusion thresholds, seed |
| `src/opinion_network/preprocess.py` | Loading, encoding, quality audit, item statistics |
| `src/opinion_network/similarity.py` | Profile correlation, naive similarity, k-NN backbone |
| `src/opinion_network/plots.py` | Figures 1 and 2 |
| `tests/test_preparation.py` | 9 sanity tests |
| `outputs/` | Generated tables, figures and `preparation_metrics.json` |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests      # 9 tests
python run_preparation.py      # ~10 seconds
```

All randomness derives from `SEED = 42`, so reruns give identical numbers.

## Summary of what this stage establishes

- 96 rows → **91 usable** (5 blank) → **87** in the main network (≥ 45/60 answered), **85** for topic layers.
- *No Comments* and blanks are **missing**, not neutral: an abstention is not a middle opinion.
- The class agrees with **79.8%** of statements. Disagreement concentrates in Education (E02, E03, E04) and a few Technology items (T08, T09).
- The dominant axis in the data is **general agreement intensity**, not any topic: PC1 holds 20% of the variance and correlates **r = 0.98** with a respondent's mean answer.
- Therefore: item-centre each statement, then use Pearson between respondents. Edges mean *"these two deviate from the class in the same directions."*
- The resulting similarities span −0.55 to +0.64 (SD 0.163 vs 0.137 for shuffled data), so there is real structure to build on.
- Recommended backbone: symmetric k-NN with `k = round(√87) = 9`, weighted by similarity → 459 edges, density 0.123.
