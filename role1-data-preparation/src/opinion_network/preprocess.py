"""Loading, encoding, quality control and descriptive statistics."""
import numpy as np
import pandas as pd

from . import config as cfg


def load_raw(path=cfg.DATA_RAW):
    """Return (responses, codebook).

    responses: DataFrame indexed by respondent id, columns = item codes (T01...).
    codebook : DataFrame with code, domain, domain_name, statement, short label.
    """
    df = pd.read_csv(path)
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df.index.name = "respondent"
    rows = []
    rename = {}
    for col in df.columns:
        code, statement = col.split(".", 1)
        code = code.strip()
        rename[col] = code
        rows.append({
            "code": code,
            "domain": code[0],
            "domain_name": cfg.DOMAINS[code[0]],
            "statement": statement.strip(),
            "short": cfg.SHORT.get(code, code),
        })
    return df.rename(columns=rename), pd.DataFrame(rows).set_index("code")


def encode(raw):
    """Map Likert labels to -2..+2; blanks and 'No Comments' become NaN."""
    unknown = set(pd.unique(raw.values.ravel())) - set(cfg.LIKERT) \
        - cfg.NON_SUBSTANTIVE - {np.nan}
    unknown = {u for u in unknown if not (isinstance(u, float) and np.isnan(u))}
    if unknown:
        raise ValueError(f"Unexpected response labels: {unknown}")
    return raw.apply(lambda s: s.map(cfg.LIKERT)).astype(float)


def quality_report(raw, num):
    """Summarise missingness, non-substantive answers and response styles."""
    answered = num.notna().sum(axis=1)
    blank = answered[answered == 0].index.tolist()
    partial = answered[(answered > 0) & (answered < num.shape[1])]
    domain_answered = {
        rid: {d: int(num.loc[rid, [c for c in num.columns if c[0] == d]].notna().sum())
              for d in cfg.DOMAINS}
        for rid in partial.index
    }
    nonblank = num.loc[answered > 0]
    values = raw.stack().value_counts()   # stack() drops NaN
    rsd = nonblank.std(axis=1)
    rmean = nonblank.mean(axis=1)
    return {
        "n_rows": int(raw.shape[0]),
        "n_items": int(raw.shape[1]),
        "blank_respondents": [int(b) for b in blank],
        "partial_respondents": {int(k): v for k, v in domain_answered.items()},
        "n_cells": int(raw.size),
        "n_blank_cells": int(raw.isna().sum().sum()),
        "n_no_comments": int((raw == "No Comments").sum().sum()),
        "value_counts": {k: int(v) for k, v in values.items()},
        "straightliners": int((rsd == 0).sum()),
        "min_respondent_sd": float(rsd.min()),
        "respondent_mean": {"mean": float(rmean.mean()), "sd": float(rmean.std()),
                            "min": float(rmean.min()), "max": float(rmean.max())},
        "duplicate_nonblank_profiles": int(nonblank.duplicated().sum()),
    }


def select(num, min_answered, items=None):
    """Keep respondents who answered at least `min_answered` of `items`."""
    sub = num if items is None else num[items]
    return sub.loc[sub.notna().sum(axis=1) >= min_answered]


def item_centre(num):
    """Subtract each item's class mean; missing -> 0 (= no deviation)."""
    return (num - num.mean()).fillna(0.0)


def tastle_wierman(counts):
    """Consensus in [0,1] for a 5-point ordinal distribution (Tastle & Wierman 2007)."""
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    x = np.arange(1, len(p) + 1)
    mu = (p * x).sum()
    width = len(p) - 1
    mask = p > 0
    return float(1 + (p[mask] * np.log2(1 - np.abs(x[mask] - mu) / width)).sum())


def item_statistics(num, codebook):
    rows = []
    for code in num.columns:
        s = num[code].dropna()
        counts = [int((s == v).sum()) for v in (-2, -1, 0, 1, 2)]
        rows.append({
            "code": code,
            "domain": code[0],
            "short": codebook.loc[code, "short"],
            "n": int(len(s)),
            "mean": s.mean(),
            "sd": s.std(),
            "pct_disagree": 100 * (s < 0).mean(),
            "pct_neutral": 100 * (s == 0).mean(),
            "pct_agree": 100 * (s > 0).mean(),
            "consensus": tastle_wierman(counts),
            **{f"n_{v}": c for v, c in zip(("SD", "D", "N", "A", "SA"), counts)},
        })
    return pd.DataFrame(rows).set_index("code")
