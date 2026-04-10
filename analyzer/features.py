"""
analyzer/features.py
─────────────────────
Automatic feature engineering.

Given a parsed DataFrame + stats, suggests (and optionally creates)
derived columns that are likely to increase analytical value.

Rules applied:
  Numeric
    • log1p   — for right-skewed columns  (skew > 1, all values ≥ 0)
    • sqrt    — for moderate right skew   (skew 0.5–1, values ≥ 0)
    • z_score — for high-variance columns (CV% > 50)
    • bin     — creates equal-frequency quartile bins (good for vis)

  Date
    • year / month / day_of_week / quarter extraction
    • is_weekend flag

  Categorical
    • frequency encode  — replace category with its row-count
    • high-cardinality  → frequency encode instead of one-hot

  Cross-column
    • ratio      — if two numeric cols share a name pattern or
                   one is clearly a subset of the other
    • interaction— product of two high-importance numerics

Returns:
  {
    "engineered": [
        {
          "name":        "log1p_price",
          "source_cols": ["price"],
          "transform":   "log1p",
          "reason":      "price is right-skewed (skew=2.3); log transform normalises it",
          "values":      [0.693, 1.099, …],   # list of floats, None for missing
        },
        …
    ]
  }
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

MAX_FEATURES = 20    # cap total engineered columns
SAMPLE_SIZE  = 5_000  # rows to include in preview values


def engineer_features(df: pd.DataFrame, stats: dict) -> dict:
    suggestions: list[dict] = []

    suggestions += _numeric_features(df, stats)
    suggestions += _date_features(df, stats)
    suggestions += _categorical_features(df, stats)
    suggestions += _cross_features(df, stats)

    # deduplicate by name
    seen: set = set()
    unique: list = []
    for f in suggestions:
        if f["name"] not in seen:
            seen.add(f["name"])
            unique.append(f)

    return {"engineered": unique[:MAX_FEATURES]}


# ══════════════════════════════════════════════════════════════
# NUMERIC
# ══════════════════════════════════════════════════════════════

def _numeric_features(df: pd.DataFrame, stats: dict) -> list[dict]:
    out = []

    for col, s in stats.items():
        if s["type"] != "numeric":
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        skew   = s.get("skewness") or 0
        cv     = s.get("cv_pct") or 0
        mn     = s.get("min") or 0

        # log1p  — right-skewed, non-negative
        if skew >= 1.0 and mn >= 0:
            transformed = np.log1p(series)
            out.append(_feat(
                name        = f"log1p_{_slug(col)}",
                source_cols = [col],
                transform   = "log1p",
                reason      = f"'{col}' is right-skewed (skewness={skew:.2f}); log1p compresses the tail.",
                values      = _to_list(transformed, df),
            ))

        # sqrt  — moderate right skew, non-negative
        elif 0.5 <= skew < 1.0 and mn >= 0:
            transformed = np.sqrt(series.clip(lower=0))
            out.append(_feat(
                name        = f"sqrt_{_slug(col)}",
                source_cols = [col],
                transform   = "sqrt",
                reason      = f"'{col}' has moderate right skew ({skew:.2f}); square-root softens it.",
                values      = _to_list(transformed, df),
            ))

        # z-score  — high variance / outlier-heavy
        if cv > 50 and (s.get("n_valid") or 0) > 5:
            mean = series.mean()
            std  = series.std()
            if std > 0:
                z = (series - mean) / std
                out.append(_feat(
                    name        = f"zscore_{_slug(col)}",
                    source_cols = [col],
                    transform   = "z_score",
                    reason      = f"'{col}' has high variability (CV={cv:.0f}%); z-score standardises it for fair comparison.",
                    values      = _to_list(z, df),
                ))

        # quartile bins  — useful for visualisation & grouping
        if (s.get("n_valid") or 0) >= 20 and s.get("unique_count", 999) > 4:
            try:
                binned = pd.qcut(series, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
                out.append(_feat(
                    name        = f"bin_{_slug(col)}",
                    source_cols = [col],
                    transform   = "quartile_bin",
                    reason      = f"'{col}' binned into quartiles — useful for segmentation analysis.",
                    values      = _cat_to_list(binned, df),
                ))
            except Exception:
                pass

    return out


# ══════════════════════════════════════════════════════════════
# DATE
# ══════════════════════════════════════════════════════════════

def _date_features(df: pd.DataFrame, stats: dict) -> list[dict]:
    out = []

    for col, s in stats.items():
        if s["type"] != "date":
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().all():
            continue

        slug = _slug(col)

        out.append(_feat(
            name        = f"{slug}_year",
            source_cols = [col],
            transform   = "extract_year",
            reason      = f"Year extracted from '{col}' — enables year-over-year comparison.",
            values      = _to_list(parsed.dt.year, df),
        ))
        out.append(_feat(
            name        = f"{slug}_month",
            source_cols = [col],
            transform   = "extract_month",
            reason      = f"Month (1–12) extracted from '{col}' — useful for seasonality analysis.",
            values      = _to_list(parsed.dt.month, df),
        ))
        out.append(_feat(
            name        = f"{slug}_dayofweek",
            source_cols = [col],
            transform   = "extract_dayofweek",
            reason      = f"Day of week (0=Mon … 6=Sun) from '{col}' — detects weekly patterns.",
            values      = _to_list(parsed.dt.dayofweek, df),
        ))
        out.append(_feat(
            name        = f"{slug}_quarter",
            source_cols = [col],
            transform   = "extract_quarter",
            reason      = f"Quarter (1–4) from '{col}' — enables quarterly trend analysis.",
            values      = _to_list(parsed.dt.quarter, df),
        ))
        out.append(_feat(
            name        = f"{slug}_is_weekend",
            source_cols = [col],
            transform   = "is_weekend",
            reason      = f"Weekend flag (1/0) from '{col}' — useful for behaviour segmentation.",
            values      = _to_list((parsed.dt.dayofweek >= 5).astype(int), df),
        ))

    return out


# ══════════════════════════════════════════════════════════════
# CATEGORICAL
# ══════════════════════════════════════════════════════════════

def _categorical_features(df: pd.DataFrame, stats: dict) -> list[dict]:
    out = []

    for col, s in stats.items():
        if s["type"] != "categorical":
            continue

        unique = s.get("unique_count") or 0
        card   = s.get("cardinality_ratio") or 0

        # frequency encode — useful for high-cardinality or ML prep
        if unique >= 3:
            freq_map = df[col].value_counts().to_dict()
            encoded  = df[col].map(freq_map)
            out.append(_feat(
                name        = f"freq_{_slug(col)}",
                source_cols = [col],
                transform   = "frequency_encode",
                reason      = (
                    f"Frequency encoding of '{col}' — replaces each category with its row count. "
                    f"{'Recommended for high-cardinality column.' if card > 0.3 else 'Useful as a numeric proxy.'}"
                ),
                values      = _to_list(encoded, df),
            ))

    return out


# ══════════════════════════════════════════════════════════════
# CROSS-COLUMN
# ══════════════════════════════════════════════════════════════

def _cross_features(df: pd.DataFrame, stats: dict) -> list[dict]:
    out = []

    # find numeric pairs that might form meaningful ratios
    num_cols = [
        c for c, s in stats.items()
        if s["type"] == "numeric"
        and (s.get("min") or 0) >= 0   # ratios only make sense for non-negative
        and (s.get("n_valid") or 0) > 5
    ]

    # look for name-based ratio hints  (revenue/cost, sales/visits, etc.)
    ratio_hints = [
        (["revenue","sales","income","profit"], ["cost","expense","spend"]),
        (["clicks","conversions","sales"],      ["visits","views","impressions","sessions"]),
        (["profit","net"],                      ["revenue","gross","sales"]),
    ]

    added_ratios: set = set()

    for numerators, denominators in ratio_hints:
        num_matches = [c for c in num_cols if any(h in c.lower() for h in numerators)]
        den_matches = [c for c in num_cols if any(h in c.lower() for h in denominators)]

        for n_col in num_matches[:2]:
            for d_col in den_matches[:2]:
                if n_col == d_col:
                    continue
                key = tuple(sorted([n_col, d_col]))
                if key in added_ratios:
                    continue
                added_ratios.add(key)

                num_s = pd.to_numeric(df[n_col], errors="coerce")
                den_s = pd.to_numeric(df[d_col], errors="coerce")
                ratio = num_s / den_s.replace(0, float("nan"))

                out.append(_feat(
                    name        = f"ratio_{_slug(n_col)}_per_{_slug(d_col)}",
                    source_cols = [n_col, d_col],
                    transform   = "ratio",
                    reason      = f"Ratio of '{n_col}' ÷ '{d_col}' — captures efficiency or conversion rate.",
                    values      = _to_list(ratio, df),
                ))

    return out


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _feat(name, source_cols, transform, reason, values) -> dict:
    return {
        "name":        name,
        "source_cols": source_cols,
        "transform":   transform,
        "reason":      reason,
        "values":      values,
    }


def _to_list(series: pd.Series, df: pd.DataFrame) -> list:
    """Convert to JSON-safe list, capped at SAMPLE_SIZE rows."""
    s = series.iloc[:SAMPLE_SIZE] if len(series) > SAMPLE_SIZE else series
    result = []
    for v in s:
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                result.append(None)
            else:
                result.append(round(float(v), 6))
        except Exception:
            result.append(None)
    return result


def _cat_to_list(series: pd.Series, df: pd.DataFrame) -> list:
    s = series.iloc[:SAMPLE_SIZE] if len(series) > SAMPLE_SIZE else series
    return [None if pd.isna(v) else str(v) for v in s]


def _slug(col: str) -> str:
    """Collapse column name to a safe identifier fragment."""
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_").lower()[:30]