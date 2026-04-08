import pandas as pd
import numpy as np
from analyzer.importance import detect_type
from analyzer.chart_selector import choose_univariate_chart, choose_bivariate_chart


def generate_chart_data(df: pd.DataFrame) -> dict:
    charts = {}

    numeric_cols = [c for c in df.columns if detect_type(df[c]) == "numeric"]
    categorical_cols = [c for c in df.columns if detect_type(df[c]) == "categorical"]
    date_cols = [c for c in df.columns if detect_type(df[c]) == "date"]

    # ── 1. HISTOGRAM for each numeric column ──
    for col in df.columns:
        col_type = detect_type(df[col])
        chart_type = choose_univariate_chart(df[col], col_type)

        if chart_type == "histogram":
            clean = df[col].dropna()
            if len(clean) < 2:
                continue

            counts, bin_edges = np.histogram(clean, bins=15)

            charts[f"histogram_{col}"] = {
                "type": "histogram",
                "title": f"{col} — Distribution",
                "labels": [
                    f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                    for i in range(len(counts))
                ],
                "values": counts.tolist(),
                "col": col,
            }

        elif chart_type == "bar":
            counts = df[col].value_counts().head(15)

            charts[f"bar_{col}"] = {
                "type": "bar",
                "title": f"{col} — Distribution",
                "labels": counts.index.astype(str).tolist(),
                "values": counts.values.tolist(),
                "col": col,
            }

        elif chart_type == "pie":
            counts = df[col].value_counts().head(8)

            charts[f"pie_{col}"] = {
                "type": "pie",
                "title": f"{col} — Distribution",
                "labels": counts.index.astype(str).tolist(),
                "values": counts.values.tolist(),
                "col": col,
            }

    # ── 2. BAR CHART — categorical × numeric ──
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        for num in numeric_cols[:3]:
            grouped = df.groupby(cat)[num].mean().dropna().sort_values(ascending=False).head(10)
            charts[f"bar_{cat}_{num}"] = {
                "type": "bar",
                "title": f"Avg {num} by {cat}",
                "labels": grouped.index.tolist(),
                "values": [round(v, 2) for v in grouped.values.tolist()],
                "col_x": cat,
                "col_y": num,
            }

    # ── 3. LINE CHART — numeric over date or index ──
    for num in numeric_cols[:3]:
        if date_cols:
            date_col = date_cols[0]
            temp = df[[date_col, num]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna().sort_values(date_col)
            temp = temp.groupby(date_col)[num].mean().reset_index()
            charts[f"line_{num}_over_{date_col}"] = {
                "type": "line",
                "title": f"{num} over {date_col}",
                "labels": temp[date_col].astype(str).tolist(),
                "values": [round(v, 2) for v in temp[num].tolist()],
                "col_x": date_col,
                "col_y": num,
            }
        else:
            # use row index sampled evenly
            step = max(1, len(df) // 50)
            sampled = df[num].iloc[::step].dropna()
            charts[f"line_{num}_index"] = {
                "type": "line",
                "title": f"{num} — Trend over rows",
                "labels": [str(i) for i in sampled.index.tolist()],
                "values": [round(v, 2) for v in sampled.tolist()],
                "col_x": "index",
                "col_y": num,
            }

    # ── 4. PIE CHART — categorical distribution ──
    for cat in categorical_cols[:2]:
        counts = df[cat].value_counts().head(8)
        charts[f"pie_{cat}"] = {
            "type": "pie",
            "title": f"{cat} — Distribution",
            "labels": counts.index.tolist(),
            "values": counts.values.tolist(),
            "col": cat,
        }

    # ── 5. SCATTER PLOT — numeric pairs ──
    if len(numeric_cols) >= 2:
        for i in range(min(len(numeric_cols), 3)):
            for j in range(i + 1, min(len(numeric_cols), 4)):
                cx, cy = numeric_cols[i], numeric_cols[j]
                temp = df[[cx, cy]].dropna()
                step = max(1, len(temp) // 200)
                sampled = temp.iloc[::step]
                charts[f"scatter_{cx}_{cy}"] = {
                    "type": "scatter",
                    "title": f"{cx} vs {cy}",
                    "x": [round(v, 4) for v in sampled[cx].tolist()],
                    "y": [round(v, 4) for v in sampled[cy].tolist()],
                    "col_x": cx,
                    "col_y": cy,
                }

    # ── 6. CORRELATION MATRIX ──
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        charts["correlation_matrix"] = {
            "type": "correlation_matrix",
            "title": "Correlation Matrix",
            "labels": numeric_cols,
            "matrix": corr.values.tolist(),
        }

    # ── 7. BOX PLOT DATA — numeric per category ──
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        for num in numeric_cols[:3]:
            groups = {}
            for name, group in df.groupby(cat)[num]:
                clean = group.dropna()
                if len(clean) < 4:
                    continue
                q1 = float(clean.quantile(0.25))
                q3 = float(clean.quantile(0.75))
                iqr = q3 - q1
                whisker_low = float(clean[clean >= q1 - 1.5 * iqr].min())
                whisker_high = float(clean[clean <= q3 + 1.5 * iqr].max())
                outliers = clean[(clean < whisker_low) | (clean > whisker_high)].tolist()
                groups[str(name)] = {
                    "min":          whisker_low,
                    "q1":           q1,
                    "median":       float(clean.median()),
                    "q3":           q3,
                    "max":          whisker_high,
                    "outliers":     [round(o, 4) for o in outliers[:20]],
                }
            if groups:
                charts[f"boxplot_{cat}_{num}"] = {
                    "type": "boxplot",
                    "title": f"{num} distribution by {cat}",
                    "col_x": cat,
                    "col_y": num,
                    "groups": groups,
                }

    # ── 8. STACKED BAR — two categoricals ──
    if len(categorical_cols) >= 2 and numeric_cols:
        cat1, cat2 = categorical_cols[0], categorical_cols[1]
        num = numeric_cols[0]
        pivot = df.groupby([cat1, cat2])[num].sum().unstack(fill_value=0)
        pivot = pivot.head(8)
        charts[f"stacked_bar_{cat1}_{cat2}"] = {
            "type": "stacked_bar",
            "title": f"{num} by {cat1} and {cat2}",
            "labels": pivot.index.tolist(),
            "series": {
                str(col): [round(v, 2) for v in pivot[col].tolist()]
                for col in pivot.columns[:6]
            },
            "col_x": cat1,
            "col_stack": cat2,
            "col_y": num,
        }

    # ── 9. TREEMAP DATA ──
    for cat in categorical_cols[:2]:
        if numeric_cols:
            num = numeric_cols[0]
            grouped = df.groupby(cat)[num].sum().dropna().sort_values(ascending=False).head(12)
            charts[f"treemap_{cat}_{num}"] = {
                "type": "treemap",
                "title": f"{num} share by {cat}",
                "labels": grouped.index.tolist(),
                "values": [round(v, 2) for v in grouped.values.tolist()],
            }
        else:
            counts = df[cat].value_counts().head(12)
            charts[f"treemap_{cat}"] = {
                "type": "treemap",
                "title": f"{cat} — Count share",
                "labels": counts.index.tolist(),
                "values": counts.values.tolist(),
            }

    # ── 10. MISSING VALUES HEATMAP ──
    missing_matrix = df.isnull().astype(int)
    step = max(1, len(df) // 100)
    sampled = missing_matrix.iloc[::step]
    charts["missing_heatmap"] = {
        "type": "missing_heatmap",
        "title": "Missing Values Heatmap",
        "columns": df.columns.tolist(),
        "rows": sampled.values.tolist(),
        "total_missing": df.isnull().sum().tolist(),
    }

    return charts