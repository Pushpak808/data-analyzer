import pandas as pd
import numpy as np

from analyzer.chart_selector import (
    choose_univariate_chart,
    choose_bivariate_chart,
)


# =========================================================
# PUBLIC
# =========================================================

def generate_chart_data(df: pd.DataFrame, stats: dict) -> dict:
    charts = {}
    columns = list(df.columns)

    # -----------------------------------------------------
    # UNIVARIATE
    # -----------------------------------------------------
    for col in columns:
        series = df[col]
        col_type = stats[col]["type"]
        col_stats = stats[col]

        charts.update(
            _generate_univariate_charts(
                col,
                series,
                col_type,
                col_stats,
            )
        )

    # -----------------------------------------------------
    # BIVARIATE
    # -----------------------------------------------------
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_x = columns[i]
            col_y = columns[j]

            charts.update(
                _generate_bivariate_charts(
                    df,
                    col_x,
                    col_y,
                    stats[col_x],
                    stats[col_y],
                )
            )

    charts.update(_correlation_matrix(df, stats))
    charts.update(_missing_heatmap(df))

    return charts


# =========================================================
# UNIVARIATE
# =========================================================

def _generate_univariate_charts(col, series, col_type, stats):
    charts = {}
    clean = series.dropna()

    if len(clean) == 0:
        return charts

    recommended = choose_univariate_chart(stats, col_type)

    if col_type == "numeric":
        charts[f"histogram_{col}"] = _histogram_chart(
            clean, col, _priority("histogram", recommended)
        )

        charts[f"box_{col}"] = _box_chart(
            clean, col, _priority("box", recommended)
        )

    elif col_type == "categorical":
        charts[f"bar_{col}"] = _bar_chart(
            clean, col, _priority("bar", recommended)
        )

        charts[f"pie_{col}"] = _pie_chart(
            clean, col, _priority("pie", recommended)
        )

    elif col_type == "date":
        charts[f"line_{col}"] = _line_chart(
            clean, col, _priority("line", recommended)
        )

    return charts


# =========================================================
# BIVARIATE
# =========================================================

def _generate_bivariate_charts(df, col_x, col_y, stats_x, stats_y):
    charts = {}

    type_x = stats_x["type"]
    type_y = stats_y["type"]

    recommended = choose_bivariate_chart(
        type_x,
        type_y,
        stats_x,
        stats_y,
    )

    temp = df[[col_x, col_y]].dropna()
    if len(temp) < 2:
        return charts

    # numeric vs numeric
    if type_x == "numeric" and type_y == "numeric":
        charts[f"scatter_{col_x}_{col_y}"] = _scatter_chart(
            temp,
            col_x,
            col_y,
            _priority("scatter", recommended),
        )

    # categorical vs numeric
    if type_x == "categorical" and type_y == "numeric":
        charts[f"bar_{col_x}_{col_y}"] = _grouped_bar(
            temp,
            col_x,
            col_y,
            _priority("bar", recommended),
        )

        charts[f"box_{col_x}_{col_y}"] = _grouped_box(
            temp,
            col_x,
            col_y,
            _priority("box", recommended),
        )

    if type_y == "categorical" and type_x == "numeric":
        charts[f"bar_{col_y}_{col_x}"] = _grouped_bar(
            temp,
            col_y,
            col_x,
            _priority("bar", recommended),
        )

        charts[f"box_{col_y}_{col_x}"] = _grouped_box(
            temp,
            col_y,
            col_x,
            _priority("box", recommended),
        )

    # date vs numeric
    if type_x == "date" and type_y == "numeric":
        charts[f"line_{col_x}_{col_y}"] = _time_line(
            temp,
            col_x,
            col_y,
            _priority("line", recommended),
        )

    if type_y == "date" and type_x == "numeric":
        charts[f"line_{col_y}_{col_x}"] = _time_line(
            temp,
            col_y,
            col_x,
            _priority("line", recommended),
        )

    return charts


# =========================================================
# CHART BUILDERS
# =========================================================

def _histogram_chart(series, col, priority):
    counts, bins = np.histogram(series, bins=15)

    return {
        "type": "histogram",
        "title": f"{col} Distribution",
        "labels": [
            f"{bins[i]:.2f}-{bins[i+1]:.2f}"
            for i in range(len(counts))
        ],
        "values": counts.tolist(),
        "col": col,
        "priority": priority,
    }


def _box_chart(series, col, priority):
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    median = float(series.median())
    iqr = q3 - q1

    outliers = series[
        (series < q1 - 1.5 * iqr) |
        (series > q3 + 1.5 * iqr)
    ].tolist()

    return {
        "type": "boxplot",
        "title": f"{col} Boxplot",
        "groups": {
            col: {
                "min": float(series.min()),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": float(series.max()),
                "outliers": outliers[:50],
            }
        },
        "priority": priority,
    }


def _bar_chart(series, col, priority):
    counts = series.value_counts().head(15)

    return {
        "type": "bar",
        "title": f"{col} Distribution",
        "labels": counts.index.astype(str).tolist(),
        "values": counts.values.tolist(),
        "col": col,
        "priority": priority,
    }


def _pie_chart(series, col, priority):
    counts = series.value_counts().head(8)

    return {
        "type": "pie",
        "title": f"{col} Distribution",
        "labels": counts.index.astype(str).tolist(),
        "values": counts.values.tolist(),
        "col": col,
        "priority": priority,
    }


def _line_chart(series, col, priority):
    step = max(1, len(series) // 50)
    sampled = series.iloc[::step]

    return {
        "type": "line",
        "title": f"{col} Trend",
        "labels": [str(i) for i in sampled.index],
        "values": sampled.astype(str).tolist(),
        "priority": priority,
    }


def _scatter_chart(df, x, y, priority):
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    return {
        "type": "scatter",
        "title": f"{x} vs {y}",
        "x": df[x].tolist(),
        "y": df[y].tolist(),
        "col_x": x,
        "col_y": y,
        "priority": priority,
    }


def _grouped_bar(df, cat, num, priority):
    grouped = df.groupby(cat)[num].mean().sort_values(ascending=False).head(10)

    return {
        "type": "bar",
        "title": f"{num} by {cat}",
        "labels": grouped.index.astype(str).tolist(),
        "values": grouped.values.tolist(),
        "col_x": cat,
        "col_y": num,
        "priority": priority,
    }


def _grouped_box(df, cat, num, priority):
    groups = {}

    for name, g in df.groupby(cat)[num]:
        clean = g.dropna()

        if len(clean) < 4:
            groups[str(name)] = {
                "insufficient_data": True
            }
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        outliers = clean[
            (clean < q1 - 1.5 * iqr) |
            (clean > q3 + 1.5 * iqr)
        ].tolist()

        groups[str(name)] = {
            "min": float(clean.min()),
            "q1": q1,
            "median": float(clean.median()),
            "q3": q3,
            "max": float(clean.max()),
            "outliers": outliers[:20],
        }

    return {
        "type": "boxplot",
        "title": f"{num} by {cat}",
        "groups": groups,
        "priority": priority,
    }


def _time_line(df, date, num, priority):
    temp = df.copy()
    temp[date] = pd.to_datetime(temp[date])
    temp = temp.sort_values(date)

    return {
        "type": "line",
        "title": f"{num} over {date}",
        "labels": temp[date].astype(str).tolist(),
        "values": temp[num].tolist(),
        "col_x": date,
        "col_y": num,
        "priority": priority,
    }


# =========================================================
# DATASET LEVEL
# =========================================================

def _correlation_matrix(df, stats):
    numeric = [c for c in df.columns if stats[c]["type"] == "numeric"]

    if len(numeric) < 2:
        return {}

    corr = df[numeric].corr().round(3)

    return {
        "correlation_matrix": {
            "type": "correlation_matrix",
            "title": "Correlation Matrix",
            "labels": numeric,
            "matrix": corr.values.tolist(),
            "priority": "high",
        }
    }


def _missing_heatmap(df):
    missing = df.isnull().astype(int)
    step = max(1, len(df) // 100)
    sampled = missing.iloc[::step]

    return {
        "missing_heatmap": {
            "type": "missing_heatmap",
            "title": "Missing Values Heatmap",
            "columns": df.columns.tolist(),
            "rows": sampled.values.tolist(),
            "priority": "high",
        }
    }


# =========================================================
# PRIORITY
# =========================================================

def _priority(chart, recommended):
    if chart == recommended:
        return "high"
    if chart in ("scatter", "bar", "line"):
        return "medium"
    return "low"