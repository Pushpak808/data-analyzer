"""
analyzer/charts.py
──────────────────
Generates serialisable chart-data dicts from a DataFrame + stats.
Supports all chart types returned by the updated chart_selector:

  histogram, density, boxplot, violin, ridgeline,
  bar, lollipop, grouped_bar, stacked_bar,
  pie, donut, treemap,
  line, area,
  scatter, heatmap,
  correlation_matrix, missing_heatmap
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from analyzer.chart_selector import (
    choose_univariate_chart,
    choose_bivariate_chart,
    get_alternatives,
)


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def generate_chart_data(df: pd.DataFrame, stats: dict) -> dict:
    charts: dict = {}
    cols = list(df.columns)

    for col in cols:
        charts.update(_univariate(col, df[col], stats[col]))

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            charts.update(
                _bivariate(df, cols[i], cols[j], stats[cols[i]], stats[cols[j]])
            )

    charts.update(_correlation_matrix(df, stats))
    charts.update(_missing_overview(df, stats))

    return charts


# ══════════════════════════════════════════════════════════════
# UNIVARIATE DISPATCH
# ══════════════════════════════════════════════════════════════

def _univariate(col: str, series: pd.Series, s: dict) -> dict:
    charts: dict = {}
    col_type    = s["type"]
    clean       = series.dropna()
    if len(clean) == 0:
        return charts

    recommended = choose_univariate_chart(s, col_type)
    alts        = get_alternatives(recommended)

    # ── numeric ──────────────────────────────────────────────
    if col_type == "numeric":
        num = _to_numeric(clean)
        if num is None or len(num) == 0:
            return charts

        builders = {
            "histogram": lambda: _histogram(num, col),
            "density":   lambda: _density(num, col),
            "boxplot":   lambda: _box_single(num, col),
            "violin":    lambda: _violin_single(num, col),
            "line":      lambda: _line_index(num, col),
        }
        charts.update(_build_set(builders, recommended, alts, col))

    # ── categorical ──────────────────────────────────────────
    elif col_type == "categorical":
        builders = {
            "bar":      lambda: _bar(clean, col),
            "lollipop": lambda: _lollipop(clean, col),
            "pie":      lambda: _pie(clean, col),
            "donut":    lambda: _donut(clean, col),
            "treemap":  lambda: _treemap(clean, col),
        }
        charts.update(_build_set(builders, recommended, alts, col))

    # ── date ─────────────────────────────────────────────────
    elif col_type == "date":
        parsed = pd.to_datetime(clean, errors="coerce").dropna().sort_values()
        if len(parsed) > 0:
            charts[f"line_{col}"] = {
                **_date_line_single(parsed, col),
                "priority": "high",
                "recommended": True,
                "alternatives": get_alternatives("line"),
            }

    return charts


def _build_set(
    builders: dict,
    recommended: str,
    alts: list,
    col: str,
) -> dict:
    """Build recommended chart + relevant alternatives."""
    out = {}
    for chart_type, fn in builders.items():
        if chart_type == recommended or chart_type in alts:
            try:
                data = fn()
                if data:
                    key = f"{chart_type}_{col}"
                    out[key] = {
                        **data,
                        "priority":     "high" if chart_type == recommended else "low",
                        "recommended":  chart_type == recommended,
                        "alternatives": alts if chart_type == recommended else [],
                    }
            except Exception:
                pass
    return out


# ══════════════════════════════════════════════════════════════
# BIVARIATE DISPATCH
# ══════════════════════════════════════════════════════════════

def _bivariate(
    df: pd.DataFrame,
    col_x: str, col_y: str,
    sx: dict, sy: dict,
) -> dict:
    charts: dict = {}
    tx, ty = sx["type"], sy["type"]

    pair = df[[col_x, col_y]].dropna()
    if len(pair) < 3:
        return charts

    recommended = choose_bivariate_chart(tx, ty, sx, sy)
    alts        = get_alternatives(recommended)

    key_prefix = f"{col_x}__{col_y}"

    try:
        # ── date × numeric ───────────────────────────────────
        if tx == "date" or ty == "date":
            date_col, num_col = (col_x, col_y) if tx == "date" else (col_y, col_x)
            pair2 = pair.copy()
            pair2[date_col] = pd.to_datetime(pair2[date_col], errors="coerce")
            pair2 = pair2.dropna().sort_values(date_col)
            if len(pair2) >= 2:
                fn = _area if recommended == "area" else _time_line
                charts[f"{recommended}_{key_prefix}"] = {
                    **fn(pair2, date_col, num_col),
                    "priority": "high", "recommended": True, "alternatives": alts,
                }

        # ── numeric × numeric ────────────────────────────────
        elif tx == "numeric" and ty == "numeric":
            nx = _to_numeric(pair[col_x])
            ny = _to_numeric(pair[col_y])
            if nx is not None and ny is not None:
                valid = pd.DataFrame({"x": nx, "y": ny}).dropna()
                if len(valid) >= 3:
                    data = _line_xy(valid, col_x, col_y) if recommended == "line" \
                           else _scatter(valid, col_x, col_y)
                    charts[f"{recommended}_{key_prefix}"] = {
                        **data, "priority": "high",
                        "recommended": True, "alternatives": alts,
                    }

        # ── categorical × numeric ────────────────────────────
        elif tx == "categorical" and ty == "numeric":
            ny = _to_numeric(pair[col_y])
            if ny is not None:
                valid = pd.DataFrame({col_x: pair[col_x], col_y: ny}).dropna()
                if len(valid) >= 3:
                    data = _cat_num_chart(valid, col_x, col_y, recommended)
                    if data:
                        charts[f"{recommended}_{key_prefix}"] = {
                            **data, "priority": "high",
                            "recommended": True, "alternatives": alts,
                        }

        elif ty == "categorical" and tx == "numeric":
            nx = _to_numeric(pair[col_x])
            if nx is not None:
                valid = pd.DataFrame({col_x: nx, col_y: pair[col_y]}).dropna()
                if len(valid) >= 3:
                    data = _cat_num_chart(valid, col_y, col_x, recommended)
                    if data:
                        charts[f"{recommended}_{key_prefix}"] = {
                            **data, "priority": "high",
                            "recommended": True, "alternatives": alts,
                        }

        # ── categorical × categorical ────────────────────────
        elif tx == "categorical" and ty == "categorical":
            data = _cat_cat_chart(pair, col_x, col_y, recommended)
            if data:
                charts[f"{recommended}_{key_prefix}"] = {
                    **data, "priority": "high",
                    "recommended": True, "alternatives": alts,
                }

    except Exception:
        pass

    return charts


def _cat_num_chart(df: pd.DataFrame, cat: str, num: str, chart_type: str) -> dict | None:
    dispatch = {
        "grouped_bar": lambda: _grouped_bar(df, cat, num),
        "lollipop":    lambda: _grouped_lollipop(df, cat, num),
        "boxplot":     lambda: _grouped_box(df, cat, num),
        "violin":      lambda: _grouped_violin(df, cat, num),
        "stacked_bar": lambda: _grouped_bar(df, cat, num),   # fallback
        "heatmap":     lambda: _cat_num_heatmap(df, cat, num),
        "line":        lambda: _grouped_line(df, cat, num),
        "bar":         lambda: _grouped_bar(df, cat, num),
    }
    fn = dispatch.get(chart_type)
    return fn() if fn else _grouped_bar(df, cat, num)


def _cat_cat_chart(df: pd.DataFrame, col_x: str, col_y: str, chart_type: str) -> dict | None:
    if chart_type in ("stacked_bar", "grouped_bar", "bar"):
        return _stacked_bar(df, col_x, col_y)
    if chart_type == "heatmap":
        return _cat_cat_heatmap(df, col_x, col_y)
    return _stacked_bar(df, col_x, col_y)


# ══════════════════════════════════════════════════════════════
# NUMERIC CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _histogram(series: pd.Series, col: str) -> dict:
    n_bins = _sturges(len(series))
    counts, edges = np.histogram(series, bins=n_bins)
    return {
        "type":   "histogram",
        "title":  f"{col} — Distribution",
        "labels": [f"{edges[i]:.3g}–{edges[i+1]:.3g}" for i in range(len(counts))],
        "values": counts.tolist(),
        "col":    col,
    }


def _density(series: pd.Series, col: str) -> dict:
    from scipy.stats import gaussian_kde
    arr = series.to_numpy(dtype=float)
    kde = gaussian_kde(arr)
    x   = np.linspace(arr.min(), arr.max(), 100)
    y   = kde(x)
    return {
        "type":   "density",
        "title":  f"{col} — Density",
        "labels": _sl(x),
        "values": _sl(y),
        "col":    col,
    }


def _box_single(series: pd.Series, col: str) -> dict:
    return {
        "type":   "boxplot",
        "title":  f"{col} — Boxplot",
        "groups": {col: _box_stats(series)},
        "col":    col,
    }


def _violin_single(series: pd.Series, col: str) -> dict:
    return {
        "type":   "violin",
        "title":  f"{col} — Violin",
        "groups": {col: _violin_stats(series)},
        "col":    col,
    }


def _line_index(series: pd.Series, col: str) -> dict:
    step = max(1, len(series) // 200)
    s    = series.iloc[::step]
    return {
        "type":   "line",
        "title":  f"{col} — Trend",
        "labels": [str(i) for i in s.index],
        "values": _sl(s),
        "col":    col,
    }


def _date_line_single(parsed: pd.Series, col: str) -> dict:
    step = max(1, len(parsed) // 200)
    s    = parsed.iloc[::step]
    return {
        "type":   "line",
        "title":  f"{col} — Timeline",
        "labels": s.astype(str).tolist(),
        "values": list(range(len(s))),
        "col":    col,
    }


# ══════════════════════════════════════════════════════════════
# CATEGORICAL CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _bar(series: pd.Series, col: str) -> dict:
    vc = series.value_counts().head(20)
    return {
        "type":   "bar",
        "title":  f"{col} — Frequency",
        "labels": vc.index.astype(str).tolist(),
        "values": vc.values.tolist(),
        "col":    col,
    }


def _lollipop(series: pd.Series, col: str) -> dict:
    vc = series.value_counts().head(20).sort_values()
    return {
        "type":   "lollipop",
        "title":  f"{col} — Ranking",
        "labels": vc.index.astype(str).tolist(),
        "values": vc.values.tolist(),
        "col":    col,
    }


def _pie(series: pd.Series, col: str) -> dict:
    vc    = series.value_counts().head(8)
    other = len(series) - vc.sum()
    labels = vc.index.astype(str).tolist()
    values = vc.values.tolist()
    if other > 0:
        labels.append("Other"); values.append(int(other))
    return {
        "type":   "pie",
        "title":  f"{col} — Composition",
        "labels": labels,
        "values": values,
        "col":    col,
    }


def _donut(series: pd.Series, col: str) -> dict:
    d = _pie(series, col)
    d["type"]  = "donut"
    d["title"] = f"{col} — Composition"
    return d


def _treemap(series: pd.Series, col: str) -> dict:
    vc = series.value_counts().head(30)
    return {
        "type":   "treemap",
        "title":  f"{col} — Treemap",
        "labels": vc.index.astype(str).tolist(),
        "values": vc.values.tolist(),
        "col":    col,
    }


# ══════════════════════════════════════════════════════════════
# BIVARIATE CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _scatter(df: pd.DataFrame, x: str, y: str) -> dict:
    s = df.sample(min(600, len(df)), random_state=42) if len(df) > 600 else df
    return {
        "type":  "scatter",
        "title": f"{x} vs {y}",
        "x":     _sl(s[x]),
        "y":     _sl(s[y]),
        "col_x": x, "col_y": y,
    }


def _line_xy(df: pd.DataFrame, x: str, y: str) -> dict:
    df2 = df.sort_values(x)
    return {
        "type":  "line",
        "title": f"{y} over {x}",
        "labels": _sl(df2[x]),
        "values": _sl(df2[y]),
        "col_x": x, "col_y": y,
    }


def _time_line(df: pd.DataFrame, date: str, num: str) -> dict:
    step = max(1, len(df) // 200)
    df2  = df.iloc[::step]
    return {
        "type":   "line",
        "title":  f"{num} over {date}",
        "labels": df2[date].astype(str).tolist(),
        "values": _sl(df2[num]),
        "col_x":  date, "col_y": num,
    }


def _area(df: pd.DataFrame, date: str, num: str) -> dict:
    d = _time_line(df, date, num)
    d["type"]  = "area"
    d["title"] = f"{num} over {date} (Area)"
    return d


def _grouped_bar(df: pd.DataFrame, cat: str, num: str) -> dict:
    grouped = (df.groupby(cat)[num].mean()
                 .sort_values(ascending=False).head(15))
    return {
        "type":   "bar",
        "title":  f"Mean {num} by {cat}",
        "labels": grouped.index.astype(str).tolist(),
        "values": [_r4(v) for v in grouped.values],
        "col_x":  cat, "col_y": num,
    }


def _grouped_lollipop(df: pd.DataFrame, cat: str, num: str) -> dict:
    grouped = (df.groupby(cat)[num].mean()
                 .sort_values().head(15))
    return {
        "type":   "lollipop",
        "title":  f"Mean {num} by {cat}",
        "labels": grouped.index.astype(str).tolist(),
        "values": [_r4(v) for v in grouped.values],
        "col_x":  cat, "col_y": num,
    }


def _grouped_box(df: pd.DataFrame, cat: str, num: str) -> dict:
    groups: dict = {}
    top = df[cat].value_counts().head(12).index
    for name in top:
        clean = df.loc[df[cat] == name, num].dropna()
        if len(clean) >= 4:
            groups[str(name)] = _box_stats(clean)
        else:
            groups[str(name)] = {"insufficient_data": True}
    return {
        "type":   "boxplot",
        "title":  f"{num} by {cat}",
        "groups": groups,
        "col_x":  cat, "col_y": num,
    }


def _grouped_violin(df: pd.DataFrame, cat: str, num: str) -> dict:
    groups: dict = {}
    top = df[cat].value_counts().head(8).index
    for name in top:
        clean = df.loc[df[cat] == name, num].dropna()
        if len(clean) >= 10:
            groups[str(name)] = _violin_stats(clean)
        else:
            groups[str(name)] = {"insufficient_data": True}
    return {
        "type":   "violin",
        "title":  f"{num} by {cat}",
        "groups": groups,
        "col_x":  cat, "col_y": num,
    }


def _grouped_line(df: pd.DataFrame, cat: str, num: str) -> dict:
    grouped = (df.groupby(cat)[num].mean()
                 .sort_values(ascending=False).head(10))
    return {
        "type":   "line",
        "title":  f"{num} trend by {cat}",
        "labels": grouped.index.astype(str).tolist(),
        "values": [_r4(v) for v in grouped.values],
        "col_x":  cat, "col_y": num,
    }


def _stacked_bar(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    top_x = df[col_x].value_counts().head(10).index
    top_y = df[col_y].value_counts().head(6).index
    sub   = df[df[col_x].isin(top_x) & df[col_y].isin(top_y)]
    ct    = pd.crosstab(sub[col_x], sub[col_y])
    return {
        "type":     "stacked_bar",
        "title":    f"{col_x} × {col_y}",
        "labels":   ct.index.astype(str).tolist(),
        "series":   {
            str(c): ct[c].tolist() for c in ct.columns
        },
        "col_x":    col_x,
        "col_y":    col_y,
    }


def _cat_num_heatmap(df: pd.DataFrame, cat: str, num: str) -> dict:
    top    = df[cat].value_counts().head(20).index
    sub    = df[df[cat].isin(top)]
    means  = sub.groupby(cat)[num].mean().reindex(top)
    return {
        "type":   "heatmap",
        "title":  f"Mean {num} by {cat}",
        "labels": top.astype(str).tolist(),
        "values": [[_r4(v)] for v in means.values],
        "col_x":  cat, "col_y": num,
    }


def _cat_cat_heatmap(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    top_x = df[col_x].value_counts().head(15).index
    top_y = df[col_y].value_counts().head(15).index
    sub   = df[df[col_x].isin(top_x) & df[col_y].isin(top_y)]
    ct    = pd.crosstab(sub[col_x], sub[col_y]).reindex(
        index=top_x, columns=top_y, fill_value=0
    )
    return {
        "type":   "heatmap",
        "title":  f"{col_x} × {col_y} — Heatmap",
        "x_labels": top_y.astype(str).tolist(),
        "y_labels": top_x.astype(str).tolist(),
        "matrix": _nan_to_none(ct.values),
        "col_x":  col_x, "col_y": col_y,
    }


# ══════════════════════════════════════════════════════════════
# DATASET-LEVEL CHARTS
# ══════════════════════════════════════════════════════════════

def _correlation_matrix(df: pd.DataFrame, stats: dict) -> dict:
    numeric_cols = [c for c in df.columns if stats[c]["type"] == "numeric"]
    if len(numeric_cols) < 2:
        return {}
    num_df = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    corr   = num_df.corr(method="pearson").round(3)
    return {
        "correlation_matrix": {
            "type":     "correlation_matrix",
            "title":    "Correlation Matrix (Pearson)",
            "labels":   numeric_cols,
            "matrix":   _nan_to_none(corr.values),
            "priority": "high",
            "recommended": True,
            "alternatives": [],
        }
    }


def _missing_overview(df: pd.DataFrame, stats: dict) -> dict:
    missing_pcts = [stats[c].get("missing_rate", 0) for c in df.columns]
    step    = max(1, len(df) // 100)
    sampled = df.isnull().astype(int).iloc[::step]
    return {
        "missing_heatmap": {
            "type":         "missing_heatmap",
            "title":        "Missing Values Overview",
            "columns":      list(df.columns),
            "missing_pcts": missing_pcts,
            "rows":         sampled.values.tolist(),
            "priority":     "high",
            "recommended":  True,
            "alternatives": [],
        }
    }


# ══════════════════════════════════════════════════════════════
# STAT HELPERS
# ══════════════════════════════════════════════════════════════

def _box_stats(series: pd.Series) -> dict:
    arr = series.to_numpy(dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr    = q3 - q1
    outliers = series[(series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)].tolist()
    return {
        "min":      float(series.min()),
        "q1":       q1,
        "median":   float(series.median()),
        "q3":       q3,
        "max":      float(series.max()),
        "outliers": outliers[:30],
    }


def _violin_stats(series: pd.Series) -> dict:
    """KDE points + box stats for violin rendering."""
    arr = series.to_numpy(dtype=float)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr)
        xs  = np.linspace(arr.min(), arr.max(), 60)
        ys  = kde(xs)
        kde_x = _sl(xs)
        kde_y = _sl(ys)
    except Exception:
        kde_x, kde_y = [], []
    return {
        **_box_stats(series),
        "kde_x": kde_x,
        "kde_y": kde_y,
    }


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def _to_numeric(series: pd.Series) -> pd.Series | None:
    result = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).dropna()
    return result if len(result) > 0 else None


def _sl(series) -> list:
    """Safe list — NaN/inf → None."""
    result = []
    for v in series:
        try:
            f = float(v)
            result.append(None if (math.isnan(f) or math.isinf(f)) else f)
        except Exception:
            result.append(str(v) if v is not None else None)
    return result


def _r4(v) -> float | None:
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except Exception:
        return None


def _nan_to_none(arr) -> list:
    return [
        [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
         for v in row]
        for row in arr
    ]


def _sturges(n: int) -> int:
    return max(5, min(50, int(math.ceil(math.log2(n) + 1))))