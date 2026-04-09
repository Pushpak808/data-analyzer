"""
analyzer/charts.py
──────────────────
Generates serialisable chart-data dicts from a DataFrame + stats.

Each chart dict contains everything the frontend needs to render it:
  type, title, priority, labels / values / groups / matrix …

No rendering logic lives here — only data preparation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.chart_selector import (
    choose_univariate_chart,
    choose_bivariate_chart,
)


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def generate_chart_data(df: pd.DataFrame, stats: dict) -> dict:
    charts: dict = {}
    cols = list(df.columns)

    # ── univariate ────────────────────────────────────────────
    for col in cols:
        charts.update(
            _univariate(col, df[col], stats[col])
        )

    # ── bivariate ─────────────────────────────────────────────
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            charts.update(
                _bivariate(df, cols[i], cols[j], stats[cols[i]], stats[cols[j]])
            )

    # ── dataset-level ─────────────────────────────────────────
    charts.update(_correlation_matrix(df, stats))
    charts.update(_missing_overview(df, stats))

    return charts


# ══════════════════════════════════════════════════════════════
# UNIVARIATE
# ══════════════════════════════════════════════════════════════

def _univariate(col: str, series: pd.Series, s: dict) -> dict:
    charts: dict = {}
    clean = series.dropna()
    if len(clean) == 0:
        return charts

    col_type   = s["type"]
    recommended = choose_univariate_chart(s, col_type)

    if col_type == "numeric":
        clean_num = pd.to_numeric(
            clean.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).dropna()

        if len(clean_num) == 0:
            return charts

        # ── SMART: detect categorical numeric (like class 1,2,3) ──
        unique_vals = clean_num.nunique()
        is_integer_like = (clean_num % 1 == 0).all()

        # small integer numeric → treat as categorical
        if is_integer_like and unique_vals <= 15:
            charts[f"bar_{col}"] = _bar(
                clean.astype(str), col, _priority("bar", "bar")
            )

            # pie only if very few categories
            if unique_vals <= 6:
                charts[f"pie_{col}"] = _pie(
                    clean.astype(str), col, _priority("pie", "pie")
                )

            return charts

        # ── real numeric ──
        charts[f"histogram_{col}"] = _histogram(
            clean_num, col, _priority("histogram", recommended)
        )

        charts[f"box_{col}"] = _box(
            clean_num, col, _priority("box", recommended)
        )

    elif col_type == "categorical":
        charts[f"bar_{col}"] = _bar(
            clean, col, _priority("bar", recommended)
        )

        charts[f"pie_{col}"] = _pie(
            clean, col, _priority("pie", recommended)
        )

    elif col_type == "date":
        charts[f"line_{col}"] = _date_line(
            clean, col, _priority("line", recommended)
        )

    return charts


# ══════════════════════════════════════════════════════════════
# BIVARIATE
# ══════════════════════════════════════════════════════════════

def _bivariate(
    df: pd.DataFrame,
    col_x: str, col_y: str,
    sx: dict,   sy: dict,
) -> dict:
    charts: dict = {}
    tx, ty = sx["type"], sy["type"]

    pair = df[[col_x, col_y]].dropna()
    if len(pair) < 3:
        return charts

    recommended = choose_bivariate_chart(tx, ty, sx, sy)

    # numeric × numeric → scatter
    if tx == "numeric" and ty == "numeric":
        nx = pd.to_numeric(pair[col_x], errors="coerce")
        ny = pd.to_numeric(pair[col_y], errors="coerce")
        valid = pd.concat([nx, ny], axis=1).dropna()
        if len(valid) >= 3:
            charts[f"scatter_{col_x}_{col_y}"] = _scatter(
                valid, col_x, col_y, _priority("scatter", recommended)
            )

    # categorical × numeric → grouped bar + box
    elif tx == "categorical" and ty == "numeric":
        ny = pd.to_numeric(pair[col_y], errors="coerce")
        valid = pd.concat([pair[col_x], ny], axis=1).dropna()
        if len(valid) >= 3:
            charts[f"bar_{col_x}_{col_y}"]   = _grouped_bar(valid, col_x, col_y, _priority("bar", recommended))
            charts[f"box_{col_x}_{col_y}"]   = _grouped_box(valid, col_x, col_y, _priority("box", recommended))

    elif ty == "categorical" and tx == "numeric":
        nx = pd.to_numeric(pair[col_x], errors="coerce")
        valid = pd.concat([nx, pair[col_y]], axis=1).dropna()
        if len(valid) >= 3:
            charts[f"bar_{col_y}_{col_x}"]   = _grouped_bar(valid, col_y, col_x, _priority("bar", recommended))
            charts[f"box_{col_y}_{col_x}"]   = _grouped_box(valid, col_y, col_x, _priority("box", recommended))

    # date × numeric → time-series line
    elif tx == "date" and ty == "numeric":
        ny = pd.to_numeric(pair[col_y], errors="coerce")
        valid = pd.concat([pair[col_x], ny], axis=1).dropna()
        if len(valid) >= 2:
            charts[f"line_{col_x}_{col_y}"] = _time_line(valid, col_x, col_y, _priority("line", recommended))

    elif ty == "date" and tx == "numeric":
        nx = pd.to_numeric(pair[col_x], errors="coerce")
        valid = pd.concat([nx, pair[col_y]], axis=1).dropna()
        if len(valid) >= 2:
            charts[f"line_{col_y}_{col_x}"] = _time_line(valid, col_y, col_x, _priority("line", recommended))

    return charts


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _histogram(series: pd.Series, col: str, priority: str) -> dict:
    n_bins = _sturges_bins(len(series))
    counts, edges = np.histogram(series, bins=n_bins)

    labels = [f"{edges[i]:.2g}–{edges[i+1]:.2g}" for i in range(len(counts))]

    return {
        "type":     "histogram",
        "title":    f"{col} — Distribution",
        "labels":   labels,
        "values":   counts.tolist(),
        "col":      col,
        "priority": priority,
    }


def _box(series: pd.Series, col: str, priority: str) -> dict:
    arr = series.to_numpy(dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    outliers = series[
        (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    ].tolist()

    return {
        "type":  "boxplot",
        "title": f"{col} — Boxplot",
        "groups": {
            col: {
                "min":      float(series.min()),
                "q1":       q1,
                "median":   float(series.median()),
                "q3":       q3,
                "max":      float(series.max()),
                "outliers": outliers[:50],
            }
        },
        "priority": priority,
    }


def _bar(series: pd.Series, col: str, priority: str) -> dict:
    counts = series.value_counts().head(20)
    return {
        "type":     "bar",
        "title":    f"{col} — Frequency",
        "labels":   counts.index.astype(str).tolist(),
        "values":   counts.values.tolist(),
        "col":      col,
        "priority": priority,
    }


def _pie(series: pd.Series, col: str, priority: str) -> dict:
    counts = series.value_counts().head(8)
    total  = len(series)
    other  = total - counts.sum()

    labels = counts.index.astype(str).tolist()
    values = counts.values.tolist()

    if other > 0:
        labels.append("Other")
        values.append(int(other))

    return {
        "type":     "pie",
        "title":    f"{col} — Composition",
        "labels":   labels,
        "values":   values,
        "col":      col,
        "priority": priority,
    }


def _date_line(series: pd.Series, col: str, priority: str) -> dict:
    parsed = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    step   = max(1, len(parsed) // 60)
    sampled = parsed.iloc[::step]

    return {
        "type":     "line",
        "title":    f"{col} — Timeline",
        "labels":   sampled.astype(str).tolist(),
        "values":   list(range(len(sampled))),   # frequency proxy
        "priority": priority,
    }


def _scatter(df: pd.DataFrame, x: str, y: str, priority: str) -> dict:
    sample = df.sample(min(600, len(df)), random_state=42) if len(df) > 600 else df
    return {
        "type":     "scatter",
        "title":    f"{x} vs {y}",
        "x":        _safe_list(sample[x]),
        "y":        _safe_list(sample[y]),
        "col_x":    x,
        "col_y":    y,
        "priority": priority,
    }


def _grouped_bar(df: pd.DataFrame, cat: str, num: str, priority: str) -> dict:
    grouped = (
        df.groupby(cat)[num]
          .mean()
          .sort_values(ascending=False)
          .head(15)
    )
    return {
        "type":     "bar",
        "title":    f"Mean {num} by {cat}",
        "labels":   grouped.index.astype(str).tolist(),
        "values":   [round(v, 4) for v in grouped.values.tolist()],
        "col_x":    cat,
        "col_y":    num,
        "priority": priority,
    }


def _grouped_box(df: pd.DataFrame, cat: str, num: str, priority: str) -> dict:
    groups: dict = {}
    # limit to top 12 categories by frequency
    top_cats = df[cat].value_counts().head(12).index

    for name in top_cats:
        clean = pd.to_numeric(
            df.loc[df[cat] == name, num], errors="coerce"
        ).dropna()

        if len(clean) < 4:
            groups[str(name)] = {"insufficient_data": True}
            continue

        q1  = float(clean.quantile(0.25))
        q3  = float(clean.quantile(0.75))
        iqr = q3 - q1
        outliers = clean[
            (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
        ].tolist()

        groups[str(name)] = {
            "min":      float(clean.min()),
            "q1":       q1,
            "median":   float(clean.median()),
            "q3":       q3,
            "max":      float(clean.max()),
            "outliers": outliers[:20],
        }

    return {
        "type":     "boxplot",
        "title":    f"{num} by {cat}",
        "groups":   groups,
        "priority": priority,
    }


def _time_line(df: pd.DataFrame, date: str, num: str, priority: str) -> dict:
    temp = df.copy()
    temp[date] = pd.to_datetime(temp[date], errors="coerce")
    temp = temp.dropna(subset=[date]).sort_values(date)

    # downsample if huge
    step = max(1, len(temp) // 200)
    temp = temp.iloc[::step]

    return {
        "type":     "line",
        "title":    f"{num} over {date}",
        "labels":   temp[date].astype(str).tolist(),
        "values":   _safe_list(temp[num]),
        "col_x":    date,
        "col_y":    num,
        "priority": priority,
    }


# ══════════════════════════════════════════════════════════════
# DATASET-LEVEL
# ══════════════════════════════════════════════════════════════

def _correlation_matrix(df: pd.DataFrame, stats: dict) -> dict:
    numeric_cols = [
        c for c in df.columns
        if stats[c]["type"] == "numeric"
    ]
    if len(numeric_cols) < 2:
        return {}

    num_df = df[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    )
    corr = num_df.corr(method="pearson").round(3)

    return {
        "correlation_matrix": {
            "type":     "correlation_matrix",
            "title":    "Correlation Matrix (Pearson)",
            "labels":   numeric_cols,
            "matrix":   _nan_to_none(corr.values),
            "priority": "high",
        }
    }


def _missing_overview(df: pd.DataFrame, stats: dict) -> dict:
    missing_pcts = {
        col: stats[col].get("missing_rate", 0)
        for col in df.columns
    }

    # row-level missing sample (capped at 100 rows)
    step    = max(1, len(df) // 100)
    sampled = df.isnull().astype(int).iloc[::step]

    return {
        "missing_heatmap": {
            "type":         "missing_heatmap",
            "title":        "Missing Values Overview",
            "columns":      list(df.columns),
            "missing_pcts": list(missing_pcts.values()),
            "rows":         sampled.values.tolist(),
            "priority":     "high",
        }
    }


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _priority(chart_type: str, recommended: str) -> str:
    if chart_type == recommended:
        return "high"
    if chart_type in ("scatter", "bar", "line", "histogram"):
        return "medium"
    return "low"


def _sturges_bins(n: int) -> int:
    """Sturges' rule for histogram bin count."""
    import math
    return max(5, min(50, int(math.ceil(math.log2(n) + 1))))


def _safe_list(series: pd.Series) -> list:
    """Convert to list, replacing NaN/inf with None."""
    import math
    result = []
    for v in series:
        try:
            f = float(v)
            result.append(None if (math.isnan(f) or math.isinf(f)) else f)
        except Exception:
            result.append(None)
    return result


def _nan_to_none(arr) -> list:
    """Recursively replace np.nan with None in nested lists."""
    import math
    result = []
    for row in arr:
        result.append([
            None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            for v in row
        ])
    return result