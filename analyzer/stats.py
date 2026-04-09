"""
analyzer/stats.py
─────────────────
Column-level statistical profiling.

Numeric  : mean, median, std, variance, CV, skewness, kurtosis, IQR,
           outlier counts (IQR + Z-score), distribution classification,
           normality flag (skew+kurt heuristic), monotonicity, entropy.

Categorical : unique count, cardinality ratio, entropy (Shannon),
              dominance, balance, top-N value table.

Date     : range, gaps, regularity, sorted flag, inferred frequency.

Text     : length stats, blank rate, uniqueness.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    return {col: _col_stats(df[col]) for col in df.columns}


# ══════════════════════════════════════════════════════════════
# TYPE DETECTION
# ══════════════════════════════════════════════════════════════

def detect_type(series: pd.Series) -> str:
    """
    Priority order:
      numeric → date → categorical → text
    """
    clean = series.dropna()
    if len(clean) == 0:
        return "text"

    # 1. Already numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # 2. Already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # 3. Try numeric coercion  (handles "1,234" → strip commas)
    coerced = pd.to_numeric(
        clean.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    if coerced.notna().mean() >= 0.90:
        return "numeric"

    # 4. Try datetime parse on a sample
    sample = clean.head(30).astype(str)
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
        if parsed.notna().mean() >= 0.85:
            return "date"
    except Exception:
        pass

    # 5. Categorical vs text: use cardinality
    n_unique = clean.nunique()
    n_total  = len(clean)
    ratio    = n_unique / n_total

    if ratio < 0.50 and n_unique <= 100:
        return "categorical"

    return "text"


# ══════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════

def _col_stats(series: pd.Series) -> dict:
    col_type = detect_type(series)

    n       = len(series)
    missing = int(series.isna().sum())

    base = {
        "type":         col_type,
        "total":        n,
        "missing":      missing,
        "missing_rate": round(missing / n * 100, 2) if n else 0.0,
    }

    dispatch = {
        "numeric":     _numeric_stats,
        "categorical": _categorical_stats,
        "date":        _date_stats,
        "text":        _text_stats,
    }
    return dispatch.get(col_type, _text_stats)(series, base)


# ══════════════════════════════════════════════════════════════
# NUMERIC
# ══════════════════════════════════════════════════════════════

def _numeric_stats(series: pd.Series, base: dict) -> dict:
    # coerce (handles string-numbers)
    clean = pd.to_numeric(
        series.dropna().astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).dropna()

    if len(clean) < 2:
        return base

    n    = len(clean)
    arr  = clean.to_numpy(dtype=float)

    # ── central tendency ──────────────────────────────────────
    mean   = float(np.mean(arr))
    median = float(np.median(arr))
    mode_r = sp_stats.mode(arr, keepdims=True)
    mode   = float(mode_r.mode[0]) if mode_r.count[0] > 1 else None

    # ── spread ────────────────────────────────────────────────
    std      = float(np.std(arr, ddof=1))
    variance = float(np.var(arr, ddof=1))
    cv       = round(std / abs(mean) * 100, 4) if mean != 0 else None   # coefficient of variation %

    q1, q3   = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr      = q3 - q1
    p05, p95 = float(np.percentile(arr, 5)),  float(np.percentile(arr, 95))
    p10, p90 = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))

    # ── shape ─────────────────────────────────────────────────
    skewness = float(sp_stats.skew(arr))
    kurtosis = float(sp_stats.kurtosis(arr))      # excess kurtosis (normal=0)

    skew_type = (
        "right_skewed" if skewness >  1 else
        "left_skewed"  if skewness < -1 else
        "symmetric"
    )
    distribution = (
        "normal_like"    if abs(skewness) < 0.5 and abs(kurtosis) < 1 else
        "mild_skew"      if abs(skewness) < 1   else
        "highly_skewed"
    )

    # rough normality flag (skew & kurt heuristic — avoids Shapiro on huge n)
    is_normal_like = abs(skewness) < 0.5 and abs(kurtosis) < 1

    # ── outliers ──────────────────────────────────────────────
    # IQR fence
    lo_iqr  = q1 - 1.5 * iqr
    hi_iqr  = q3 + 1.5 * iqr
    out_iqr = int(((arr < lo_iqr) | (arr > hi_iqr)).sum())

    # Z-score fence  (|z| > 3)
    if std > 0:
        z       = np.abs((arr - mean) / std)
        out_z   = int((z > 3).sum())
    else:
        out_z   = 0

    outlier_ratio = round(out_iqr / n * 100, 2)

    # ── monotonicity ─────────────────────────────────────────
    monotonic_inc = bool(np.all(np.diff(arr) >= 0))
    monotonic_dec = bool(np.all(np.diff(arr) <= 0))
    monotonic     = monotonic_inc or monotonic_dec

    # ── entropy (binned) ──────────────────────────────────────
    hist, _ = np.histogram(arr, bins=min(20, n))
    prob     = hist / hist.sum()
    prob     = prob[prob > 0]
    numeric_entropy = float(-np.sum(prob * np.log2(prob)))

    # ── variance score (CV² normalised) ──────────────────────
    variance_score = float((std / abs(mean)) ** 2) if mean != 0 else float("inf")

    # ── zero / negative flags ────────────────────────────────
    zero_count  = int((arr == 0).sum())
    neg_count   = int((arr < 0).sum())

    return {
        **base,
        # central tendency
        "mean":           _r(mean),
        "median":         _r(median),
        "mode":           _r(mode),
        # spread
        "std":            _r(std),
        "variance":       _r(variance),
        "cv_pct":         _r(cv),          # coefficient of variation %
        "min":            _r(float(arr.min())),
        "max":            _r(float(arr.max())),
        "range":          _r(float(arr.max() - arr.min())),
        "q1":             _r(q1),
        "q3":             _r(q3),
        "iqr":            _r(iqr),
        "p05":            _r(p05),
        "p10":            _r(p10),
        "p90":            _r(p90),
        "p95":            _r(p95),
        # shape
        "skewness":       _r(skewness),
        "kurtosis":       _r(kurtosis),
        "skew_type":      skew_type,
        "distribution":   distribution,
        "is_normal_like": is_normal_like,
        # outliers
        "outliers_iqr":   out_iqr,
        "outliers_z":     out_z,
        "outlier_ratio":  outlier_ratio,
        # monotonicity
        "monotonic":      monotonic,
        "monotonic_inc":  monotonic_inc,
        "monotonic_dec":  monotonic_dec,
        # entropy / variance score
        "numeric_entropy":  _r(numeric_entropy),
        "variance_score":   _r(variance_score),
        # misc
        "zero_count":     zero_count,
        "negative_count": neg_count,
        "n_valid":        n,
    }


# ══════════════════════════════════════════════════════════════
# CATEGORICAL
# ══════════════════════════════════════════════════════════════

def _categorical_stats(series: pd.Series, base: dict) -> dict:
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return base

    n            = len(clean)
    vc           = clean.value_counts()
    n_unique     = len(vc)
    probs        = vc.values / n

    # Shannon entropy (bits)
    categorical_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # normalised entropy (0 = one dominant, 1 = perfectly uniform)
    max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0
    normalised_entropy = categorical_entropy / max_entropy

    dominant_val   = vc.index[0]
    dominant_rate  = round(probs[0] * 100, 2)
    balance_score  = round(1.0 - probs[0], 4)         # 0 = skewed, 1 = perfectly spread
    cardinality_ratio = round(n_unique / n, 4)

    # Gini impurity
    gini = float(1.0 - np.sum(probs ** 2))

    return {
        **base,
        "unique_count":         n_unique,
        "cardinality_ratio":    cardinality_ratio,
        "dominant_value":       dominant_val,
        "dominant_rate":        dominant_rate,
        "categorical_entropy":  _r(categorical_entropy),
        "normalised_entropy":   _r(normalised_entropy),
        "balance_score":        _r(balance_score),
        "gini_impurity":        _r(gini),
        "top_values":           vc.head(10).index.tolist(),
        "top_counts":           vc.head(10).values.tolist(),
        "top_pcts":             [round(p * 100, 2) for p in probs[:10]],
        "n_valid":              n,
    }


# ══════════════════════════════════════════════════════════════
# DATE
# ══════════════════════════════════════════════════════════════

def _date_stats(series: pd.Series, base: dict) -> dict:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if len(parsed) == 0:
        return base

    parsed   = parsed.sort_values()
    n        = len(parsed)
    min_d    = parsed.min()
    max_d    = parsed.max()
    span     = (max_d - min_d).days

    # infer frequency from gaps
    gaps_days = parsed.diff().dt.days.dropna()
    median_gap = float(gaps_days.median()) if len(gaps_days) else None
    inferred_freq = _infer_freq(median_gap)

    # regularity: % of gaps within 50% of median
    if median_gap and median_gap > 0:
        regular_pct = round(
            float(((gaps_days - median_gap).abs() / median_gap < 0.5).mean() * 100), 1
        )
    else:
        regular_pct = None

    return {
        **base,
        "min_date":       str(min_d.date()),
        "max_date":       str(max_d.date()),
        "range_days":     span,
        "unique_dates":   int(parsed.nunique()),
        "is_sorted":      bool(parsed.is_monotonic_increasing),
        "median_gap_days":_r(median_gap),
        "inferred_freq":  inferred_freq,
        "regularity_pct": regular_pct,
        "n_valid":        n,
    }


def _infer_freq(median_gap: float | None) -> str | None:
    if median_gap is None:
        return None
    if median_gap <= 1:
        return "daily"
    if median_gap <= 8:
        return "weekly"
    if median_gap <= 35:
        return "monthly"
    if median_gap <= 100:
        return "quarterly"
    return "yearly"


# ══════════════════════════════════════════════════════════════
# TEXT
# ══════════════════════════════════════════════════════════════

def _text_stats(series: pd.Series, base: dict) -> dict:
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return base

    lengths = clean.str.len()
    blanks  = int((clean.str.strip() == "").sum())

    return {
        **base,
        "unique_count": int(clean.nunique()),
        "avg_length":   _r(float(lengths.mean())),
        "median_length":_r(float(lengths.median())),
        "max_length":   int(lengths.max()),
        "min_length":   int(lengths.min()),
        "blank_count":  blanks,
        "blank_rate":   _r(blanks / len(clean) * 100),
        "n_valid":      len(clean),
    }


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _r(val, d: int = 4):
    """Safe round — returns None for NaN/None/inf."""
    try:
        if val is None:
            return None
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, d)
    except Exception:
        return None