"""Sanity tests for the data-preparation stage.  Run:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opinion_network import config as cfg                        # noqa: E402
from opinion_network import preprocess as pp                     # noqa: E402
from opinion_network.similarity import knn_graph, profile_correlation  # noqa: E402


def test_shape_and_codebook():
    raw, codebook = pp.load_raw()
    assert raw.shape == (96, 60)
    assert list(codebook["domain"].value_counts().sort_index()) == [15, 15, 15, 15]
    assert codebook.index.is_unique


def test_encoding_loses_nothing_but_no_comments():
    raw, _ = pp.load_raw()
    num = pp.encode(raw)
    assert set(np.unique(num.values[~np.isnan(num.values)])) <= {-2, -1, 0, 1, 2}
    assert raw.notna().sum().sum() == num.notna().sum().sum() + (raw == "No Comments").sum().sum()


def test_no_comments_is_not_neutral():
    raw, _ = pp.load_raw()
    num = pp.encode(raw)
    assert num[raw == "No Comments"].isna().all().all()


def test_tastle_wierman_bounds():
    assert np.isclose(pp.tastle_wierman([0, 0, 10, 0, 0]), 1.0)     # unanimous
    assert np.isclose(pp.tastle_wierman([5, 0, 0, 0, 5]), 0.0)      # split at the extremes
    assert 0 < pp.tastle_wierman([1, 2, 3, 2, 1]) < 1


def test_selection_rule():
    raw, _ = pp.load_raw()
    num = pp.encode(raw)
    sample = pp.select(num, cfg.MIN_ANSWERED_TOTAL)
    assert (sample.notna().sum(axis=1) >= cfg.MIN_ANSWERED_TOTAL).all()
    assert len(sample) == 87


def test_item_centring_removes_class_mean():
    raw, _ = pp.load_raw()
    num = pp.encode(raw)
    C = pp.item_centre(num)
    observed = num.notna()
    # among answered cells, each item's deviations average to zero
    assert np.allclose([C[c][observed[c]].mean() for c in C.columns], 0, atol=1e-9)
    assert (C.to_numpy()[~observed.to_numpy()] == 0).all()      # missing -> no deviation


def test_profile_correlation_is_row_pearson():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(10, 30))
    assert np.allclose(profile_correlation(M), np.corrcoef(M))


def test_profile_correlation_ignores_intensity():
    """Two people with the same pattern but different enthusiasm correlate at 1."""
    base = np.array([[2, 2, 2, 1], [1, 1, 1, 0], [2, 0, 2, 2], [1, 2, 0, 1]], float)
    df = pd.DataFrame(base, columns=list("abcd"))
    S = profile_correlation(pp.item_centre(df))
    assert np.isclose(S[0, 1], 1.0)             # Asha and Bilal: identical pattern
    assert S[0, 2] < 0                          # Asha and Chen: opposite pattern


def test_knn_backbone_properties():
    rng = np.random.default_rng(1)
    S = np.corrcoef(rng.normal(size=(30, 12))) + 2       # all positive
    G = knn_graph(S, 4)
    assert min(d for _, d in G.degree()) >= 4            # nobody is isolated
    assert sum(dict(G.nodes(data="hubness")).values()) == 30 * 4
    assert all(d["weight"] > 0 for *_, d in G.edges(data=True))
    assert all(np.isclose(d["distance"], 1 - d["weight"]) for *_, d in G.edges(data=True))
